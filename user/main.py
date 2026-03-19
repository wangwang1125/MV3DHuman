"""
MH3D 用户端 — FastAPI 主应用
提供：主页渲染、MJPEG 视频流、WebSocket 实时通信、
      图片上传、3D 生成任务提交/查询、模型下载。
"""

import os
import json
import uuid
import time
import base64
import asyncio
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form
from fastapi.responses import HTMLResponse, StreamingResponse, FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
import uvicorn

from camera import CameraManager, VIEW_ORDER
from api_client import MH3DApiClient

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

CLOUD_API_URL = os.environ.get("MH3D_API_URL", "http://region-42.seetacloud.com:33246")

camera = CameraManager()
api_client = MH3DApiClient(base_url=CLOUD_API_URL)

uploaded_views: dict[str, Optional[bytes]] = {v: None for v in VIEW_ORDER}
uploaded_depths: dict[str, Optional[bytes]] = {v: None for v in VIEW_ORDER}
uploaded_normals: dict[str, Optional[bytes]] = {v: None for v in VIEW_ORDER}
active_ws: list[WebSocket] = []
task_results: dict[str, dict] = {}


@asynccontextmanager
async def lifespan(application: FastAPI):
    yield
    camera.stop()
    await api_client.close()


app = FastAPI(title="MH3D 人体三维重建系统 - 用户端", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
app.mount("/output", StaticFiles(directory=str(OUTPUT_DIR)), name="output_files")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


# ──────────────────────────── 页面 ────────────────────────────

@app.get("/.well-known/appspecific/com.chrome.devtools.json")
async def chrome_devtools_well_known():
    """Chrome DevTools 会请求此路径，返回 200 避免日志里刷 404"""
    return {}

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "cloud_api_url": CLOUD_API_URL,
    })


# ──────────────────────────── 摄像头控制 ────────────────────────────

@app.post("/api/camera/start")
async def camera_start():
    camera.start()
    return {"ok": True}


@app.post("/api/camera/stop")
async def camera_stop():
    camera.stop()
    return {"ok": True}


@app.post("/api/camera/reset")
async def camera_reset():
    camera.reset_capture()
    return {"ok": True}


# ──────────────────────────── MJPEG 视频流 ────────────────────────────

async def _mjpeg_generator():
    while camera.is_running:
        jpeg = await asyncio.to_thread(camera.get_jpeg_bytes)
        if jpeg:
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n")
        else:
            await asyncio.sleep(0.03)


@app.get("/video_feed")
async def video_feed():
    if not camera.is_running:
        return JSONResponse({"error": "摄像头未启动"}, status_code=400)
    return StreamingResponse(
        _mjpeg_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


# ──────────────────────────── WebSocket ────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    active_ws.append(ws)
    sent_views: set[str] = set()
    try:
        while True:
            if camera.is_running:
                status = camera.get_pose_status()
                await ws.send_json({"type": "pose_status", **status})

                voice = camera.pop_voice_message()
                if voice:
                    await ws.send_json({"type": "voice_guide", "text": voice})

                for view in VIEW_ORDER:
                    if view not in sent_views and status["captured"].get(view):
                        jpg = await asyncio.to_thread(camera.get_captured_image_jpeg, view)
                        if jpg:
                            await ws.send_json({
                                "type": "auto_capture",
                                "view": view,
                                "image_base64": base64.b64encode(jpg).decode(),
                            })
                            sent_views.add(view)

            try:
                msg = await asyncio.wait_for(ws.receive_text(), timeout=0.15)
                data = json.loads(msg)
                await _handle_ws_message(ws, data)
            except asyncio.TimeoutError:
                pass

            await asyncio.sleep(0.15)
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        if ws in active_ws:
            active_ws.remove(ws)


async def _handle_ws_message(ws: WebSocket, data: dict):
    action = data.get("action")
    if action == "manual_capture":
        view = camera.manual_capture()
        if view:
            jpg = camera.get_captured_image_jpeg(view)
            if jpg:
                await ws.send_json({
                    "type": "auto_capture",
                    "view": view,
                    "image_base64": base64.b64encode(jpg).decode(),
                })


# ──────────────────────────── 图片上传（导入模式） ────────────────────────────

@app.post("/upload")
async def upload_images(
    view: str = Form(...),
    file: UploadFile = File(...),
):
    if view not in VIEW_ORDER:
        return JSONResponse({"error": f"无效视图: {view}"}, status_code=400)

    content = await file.read()
    uploaded_views[view] = content

    return {"ok": True, "view": view, "size": len(content)}


@app.post("/upload/depth")
async def upload_depth(view: str = Form(...), file: UploadFile = File(...)):
    if view not in VIEW_ORDER:
        return JSONResponse({"error": f"无效视图: {view}"}, status_code=400)
    content = await file.read()
    uploaded_depths[view] = content
    return {"ok": True, "view": view, "size": len(content)}


@app.post("/upload/normal")
async def upload_normal(view: str = Form(...), file: UploadFile = File(...)):
    if view not in VIEW_ORDER:
        return JSONResponse({"error": f"无效视图: {view}"}, status_code=400)
    content = await file.read()
    uploaded_normals[view] = content
    return {"ok": True, "view": view, "size": len(content)}


@app.post("/upload/clear")
async def clear_uploads():
    for v in VIEW_ORDER:
        uploaded_views[v] = None
        uploaded_depths[v] = None
        uploaded_normals[v] = None
    return {"ok": True}


@app.post("/upload/clear_one")
async def clear_upload_one(view: str = Form(...), file_type: str = Form("rgb")):
    """清除单个视图的指定类型上传（file_type: rgb / normal / depth）。"""
    if view not in VIEW_ORDER:
        return JSONResponse({"error": f"无效视图: {view}"}, status_code=400)
    if file_type == "rgb":
        uploaded_views[view] = None
    elif file_type == "normal":
        uploaded_normals[view] = None
    elif file_type == "depth":
        uploaded_depths[view] = None
    else:
        return JSONResponse({"error": "file_type 须为 rgb / normal / depth"}, status_code=400)
    return {"ok": True, "view": view, "file_type": file_type}


# ──────────────────────────── 3D 生成 ────────────────────────────

def _decode_base64_views(b64_map: Optional[dict]) -> dict[str, bytes]:
    out: dict[str, bytes] = {}
    if not b64_map:
        return out
    for v in VIEW_ORDER:
        b64 = b64_map.get(v)
        if not b64:
            continue
        try:
            out[v] = base64.b64decode(b64)
        except Exception:
            pass
    return out


@app.post("/generate")
async def generate(request: Request):
    body = await request.json()
    mode = body.get("mode", "upload")
    params = body.get("params", {})

    images: dict[str, bytes] = {}

    if mode == "capture":
        for v in VIEW_ORDER:
            png = camera.get_captured_image_png(v)
            if png:
                images[v] = png
    else:
        # 优先使用前端传来的 base64（来自预览 data URL），重启后仍可直接生成
        images = _decode_base64_views(body.get("images_base64"))
        if not images:
            for v in VIEW_ORDER:
                if uploaded_views[v] is not None:
                    images[v] = uploaded_views[v]

    if not images:
        return JSONResponse({"error": "至少需要提供一张 RGB 图（任意视图）"}, status_code=400)

    normals = _decode_base64_views(body.get("normals_base64"))
    if not normals:
        normals = {v: uploaded_normals[v] for v in VIEW_ORDER if uploaded_normals.get(v)} or None

    depths = _decode_base64_views(body.get("depths_base64"))
    if not depths:
        depths = {v: uploaded_depths[v] for v in VIEW_ORDER if uploaded_depths.get(v)} or None

    task_id = str(uuid.uuid4())[:8]
    task_results[task_id] = {"status": "submitting", "uid": None, "start_time": time.time()}

    asyncio.create_task(_run_generation(task_id, images, params, normals, depths))

    return {"ok": True, "task_id": task_id}


async def _run_generation(
    task_id: str,
    images: dict[str, bytes],
    params: dict,
    normals: Optional[dict[str, bytes]] = None,
    depths: Optional[dict[str, bytes]] = None,
):
    import traceback

    def _err_msg(e: Exception) -> str:
        """生成可读的错误信息，避免空字符串。"""
        msg = str(e)
        if not msg:
            msg = f"{type(e).__name__}"
        if hasattr(e, "response"):
            resp = e.response
            msg += f" [HTTP {resp.status_code}]"
            try:
                msg += f" {resp.text[:300]}"
            except Exception:
                pass
        return msg

    try:
        task_results[task_id]["status"] = "submitting"
        print(f"[task {task_id}] 提交任务到云端 {api_client.base_url} ...")
        uid = await api_client.submit_task(
            images,
            normals=normals,
            depths=depths,
            remove_background=params.get("remove_background", True),
            seed=params.get("seed", 1234),
            octree_resolution=params.get("octree_resolution", 256),
            num_inference_steps=params.get("num_inference_steps", 30),
            guidance_scale=params.get("guidance_scale", 5.0),
            num_chunks=params.get("num_chunks", 8000),
        )
        task_results[task_id]["uid"] = uid
        task_results[task_id]["status"] = "pending"
        print(f"[task {task_id}] 提交成功, uid={uid}")

        await _broadcast_ws({"type": "task_status", "task_id": task_id, "status": "pending", "uid": uid})

        async def _on_status(d: dict):
            status = d.get("status", "unknown")
            task_results[task_id]["status"] = status
            if d.get("status") == "error" and d.get("message"):
                task_results[task_id]["message"] = d["message"]
            payload = {
                "type": "task_status",
                "task_id": task_id,
                "status": status,
                "uid": uid,
            }
            if d.get("status") == "error" and d.get("message"):
                payload["message"] = d["message"]
            await _broadcast_ws(payload)

        result = await api_client.wait_for_completion(
            uid,
            poll_interval=0.5,
            max_wait=600.0,
            on_status=_on_status,
        )

        if result.get("status") == "completed":
            model_b64 = result.get("model_base64", "")
            model_bytes = base64.b64decode(model_b64)

            out_path = OUTPUT_DIR / f"{task_id}.glb"
            out_path.write_bytes(model_bytes)

            has_matted = any(result.get(k) for k in (
                "image_front_matted_base64", "image_right_matted_base64",
                "image_back_matted_base64", "image_left_matted_base64",
            ))
            task_results[task_id]["status"] = "completed"
            task_results[task_id]["file"] = str(out_path)
            task_results[task_id]["filename"] = f"{task_id}.glb"
            task_results[task_id]["has_matted_views"] = has_matted

            _save_input_images(task_id, images)
            _save_matted_images(task_id, result)
            print(f"[task {task_id}] 生成完成, 文件: {out_path}")

            await _broadcast_ws({
                "type": "task_status",
                "task_id": task_id,
                "status": "completed",
                "uid": uid,
                "filename": f"{task_id}.glb",
                "has_matted_views": has_matted,
            })
        else:
            msg = result.get("message") or f"云端返回非完成状态: {result.get('status')}"
            task_results[task_id]["status"] = "error"
            task_results[task_id]["message"] = msg
            print(f"[task {task_id}] 云端返回错误: {msg}")
            await _broadcast_ws({
                "type": "task_status",
                "task_id": task_id,
                "status": "error",
                "message": msg,
            })

    except Exception as e:
        msg = _err_msg(e)
        task_results[task_id]["status"] = "error"
        task_results[task_id]["message"] = msg
        print(f"[task {task_id}] 异常: {msg}")
        traceback.print_exc()
        await _broadcast_ws({
            "type": "task_status",
            "task_id": task_id,
            "status": "error",
            "message": msg,
        })


def _save_input_images(task_id: str, images: dict[str, bytes]):
    task_dir = OUTPUT_DIR / task_id
    task_dir.mkdir(exist_ok=True)
    for view, data in images.items():
        (task_dir / f"{view}.png").write_bytes(data)


def _save_matted_images(task_id: str, result: dict):
    """将 API 返回的抠图四视图 base64 保存到任务目录，供预览使用。"""
    task_dir = OUTPUT_DIR / task_id
    task_dir.mkdir(exist_ok=True)
    for view in VIEW_ORDER:
        key = f"image_{view}_matted_base64"
        b64 = result.get(key)
        if b64:
            try:
                (task_dir / f"{view}_matted.png").write_bytes(base64.b64decode(b64))
            except Exception:
                pass


async def _broadcast_ws(msg: dict):
    for ws in list(active_ws):
        try:
            await ws.send_json(msg)
        except Exception:
            if ws in active_ws:
                active_ws.remove(ws)


# ──────────────────────────── 任务查询与下载 ────────────────────────────

@app.get("/task/{task_id}")
async def task_status(task_id: str):
    info = task_results.get(task_id)
    if not info:
        return JSONResponse({"error": "任务不存在"}, status_code=404)
    safe = {k: v for k, v in info.items() if k != "file"}
    if info.get("filename"):
        safe["filename"] = info["filename"]
    return safe


# 支持的 mesh 下载格式（与 gradio_app 一致：glb/obj/ply）
DOWNLOAD_MESH_FORMATS = ("glb", "obj", "ply")


@app.get("/download/{task_id}")
async def download_model(task_id: str, format: str = "glb"):
    """
    下载生成好的 mesh。可选格式：glb（默认）、obj、ply。
    服务端存的是 glb；请求 obj/ply 时用 trimesh 实时转换后返回。
    """
    format = (format or "glb").lower().strip()
    if format not in DOWNLOAD_MESH_FORMATS:
        return JSONResponse(
            {"error": f"不支持的格式，仅支持: {', '.join(DOWNLOAD_MESH_FORMATS)}"},
            status_code=400,
        )
    info = task_results.get(task_id)
    if not info or not info.get("file"):
        return JSONResponse({"error": "文件不存在"}, status_code=404)
    fpath = info["file"]
    if not os.path.isfile(fpath):
        return JSONResponse({"error": "文件已被删除"}, status_code=404)

    filename = f"{task_id}.{format}"
    if format == "glb":
        return FileResponse(
            fpath,
            filename=filename,
            media_type="application/octet-stream",
        )

    # obj / ply：用 trimesh 从 glb 转换
    try:
        import trimesh
        loaded = trimesh.load(fpath)
        if isinstance(loaded, trimesh.Scene):
            if not loaded.geometry:
                return JSONResponse({"error": "模型中无几何体"}, status_code=500)
            meshes = list(loaded.geometry.values())
            mesh = meshes[0] if len(meshes) == 1 else trimesh.util.concatenate(meshes)
        else:
            mesh = loaded
        data = mesh.export(file_type=format)
        if isinstance(data, str):
            data = data.encode("utf-8")
        return Response(
            content=data,
            media_type="application/octet-stream",
            headers={"Content-Disposition": f"attachment; filename=\"{filename}\""},
        )
    except Exception as e:
        return JSONResponse(
            {"error": f"格式转换失败: {str(e)}"},
            status_code=500,
        )


@app.get("/preview/{task_id}/{view}")
async def preview_image(task_id: str, view: str):
    task_dir = OUTPUT_DIR / task_id
    fpath = task_dir / f"{view}.png"
    if not fpath.is_file() and view.endswith("_matted"):
        # 抠图未保存时（如云端未返回）回退到原图，避免 404
        fallback = task_dir / f"{view.replace('_matted', '')}.png"
        if fallback.is_file():
            fpath = fallback
    if not fpath.is_file():
        return JSONResponse({"error": "预览图不存在"}, status_code=404)
    return FileResponse(str(fpath), media_type="image/png")


# ──────────────────────────── 云端健康检查 ────────────────────────────

@app.get("/api/health")
async def health_check():
    result = await api_client.check_health()
    return result


# ──────────────────────────── 启动 ────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="MH3D 用户端")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--api-url", default=CLOUD_API_URL, help="云端 API 地址")
    args = parser.parse_args()

    global api_client
    api_client = MH3DApiClient(base_url=args.api_url)

    print(f"MH3D 用户端启动: http://{args.host}:{args.port}")
    print(f"云端 API: {args.api_url}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
