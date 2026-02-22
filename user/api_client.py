"""
云端 MH3D API 异步客户端。
封装 /send、/status、/health 接口，使用 httpx 异步 HTTP。
"""

import base64
import asyncio
from typing import Optional

import httpx

DEFAULT_API_BASE = "http://region-42.seetacloud.com:33246"
REQUEST_TIMEOUT = 600.0
POLL_INTERVAL = 5.0


class MH3DApiClient:
    """异步客户端，与云端 MH3D 推理服务通信。"""

    def __init__(self, base_url: str = DEFAULT_API_BASE):
        self.base_url = base_url.rstrip("/")
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=REQUEST_TIMEOUT)
        return self._client

    async def close(self):
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    async def check_health(self) -> dict:
        client = await self._get_client()
        try:
            resp = await client.get(f"{self.base_url}/health", timeout=10.0)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            return {"status": "error", "message": str(e)}

    async def submit_task(
        self,
        images: dict[str, bytes],
        normals: Optional[dict[str, bytes]] = None,
        depths: Optional[dict[str, bytes]] = None,
        remove_background: bool = True,
        seed: int = 1234,
        octree_resolution: int = 256,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        num_chunks: int = 200000,
    ) -> str:
        """
        提交生成任务。至少一张 RGB 图即可，法线/深度可选。

        Args:
            images: 至少包含一个视图，如 {"front": png_bytes} 或 {"front": ..., "right": ...}
            normals: 可选 {"front": bytes, ...}，参与推理
            depths: 可选 {"front": bytes, ...}，仅接收不参与推理
        Returns:
            任务 uid
        """
        if not images or not any(images.get(v) for v in ("front", "right", "back", "left")):
            raise ValueError("至少需要提供一张 RGB 图（front/right/back/left 至少一个）")
        payload = {
            "remove_background": remove_background,
            "seed": seed,
            "octree_resolution": octree_resolution,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "num_chunks": num_chunks,
        }
        for v in ("front", "right", "back", "left"):
            if v in images and images[v]:
                payload[f"image_{v}"] = base64.b64encode(images[v]).decode()
        if normals:
            for v in ("front", "right", "back", "left"):
                if v in normals and normals[v]:
                    payload[f"normal_{v}"] = base64.b64encode(normals[v]).decode()
        if depths:
            for v in ("front", "right", "back", "left"):
                if v in depths and depths[v]:
                    payload[f"depth_{v}"] = base64.b64encode(depths[v]).decode()
        client = await self._get_client()
        resp = await client.post(f"{self.base_url}/send", json=payload, timeout=30.0)
        resp.raise_for_status()
        data = resp.json()
        uid = data.get("uid")
        if not uid:
            raise ValueError(f"API 未返回 uid: {data}")
        return uid

    async def poll_status(self, uid: str) -> dict:
        """
        查询任务状态。

        Returns:
            status, model_base64（完成时）, message（错误时）,
            以及完成时可能的 image_front_matted_base64 等抠图四视图。
        """
        client = await self._get_client()
        resp = await client.get(f"{self.base_url}/status/{uid}", timeout=15.0)
        resp.raise_for_status()
        return resp.json()

    async def wait_for_completion(
        self,
        uid: str,
        poll_interval: float = POLL_INTERVAL,
        max_wait: float = 600.0,
        on_status=None,
    ) -> dict:
        """
        轮询等待任务完成。

        Args:
            on_status: 可选回调 async def(status_dict)，每次轮询时调用
        Returns:
            最终的状态 dict（含 model_base64 或 error message）
        """
        import time
        start = time.time()
        while time.time() - start < max_wait:
            data = await self.poll_status(uid)
            status = data.get("status", "unknown")
            if on_status:
                await on_status(data)
            if status == "completed":
                return data
            if status == "error":
                return data
            await asyncio.sleep(poll_interval)
        return {"status": "error", "message": f"超时（>{max_wait}秒）"}

    async def generate_sync(
        self,
        images: dict[str, bytes],
        **kwargs,
    ) -> bytes:
        """
        同步生成（POST /generate），返回 OBJ 文件 bytes。
        适用于不需要进度反馈的场景。
        """
        if not images or not any(images.get(v) for v in ("front", "right", "back", "left")):
            raise ValueError("至少需要提供一张 RGB 图（front/right/back/left 至少一个）")
        payload = {
            "remove_background": kwargs.get("remove_background", True),
            "seed": kwargs.get("seed", 1234),
            "octree_resolution": kwargs.get("octree_resolution", 256),
            "num_inference_steps": kwargs.get("num_inference_steps", 50),
            "guidance_scale": kwargs.get("guidance_scale", 5.0),
            "num_chunks": kwargs.get("num_chunks", 200000),
        }
        for v in ("front", "right", "back", "left"):
            if v in images and images[v]:
                payload[f"image_{v}"] = base64.b64encode(images[v]).decode()
        normals = kwargs.get("normals")
        depths = kwargs.get("depths")
        if normals:
            for v in ("front", "right", "back", "left"):
                if v in normals and normals[v]:
                    payload[f"normal_{v}"] = base64.b64encode(normals[v]).decode()
        if depths:
            for v in ("front", "right", "back", "left"):
                if v in depths and depths[v]:
                    payload[f"depth_{v}"] = base64.b64encode(depths[v]).decode()
        client = await self._get_client()
        resp = await client.post(
            f"{self.base_url}/generate", json=payload, timeout=REQUEST_TIMEOUT
        )
        resp.raise_for_status()
        return resp.content
