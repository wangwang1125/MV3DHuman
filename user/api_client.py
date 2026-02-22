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
        remove_background: bool = True,
        seed: int = 1234,
        octree_resolution: int = 256,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        num_chunks: int = 200000,
        height_mm: float = 1750,
    ) -> str:
        """
        提交四视图生成任务。

        Args:
            images: {"front": png_bytes, "right": ..., "back": ..., "left": ...}
        Returns:
            任务 uid
        """
        payload = {
            "image_front": base64.b64encode(images["front"]).decode(),
            "image_right": base64.b64encode(images["right"]).decode(),
            "image_back": base64.b64encode(images["back"]).decode(),
            "image_left": base64.b64encode(images["left"]).decode(),
            "remove_background": remove_background,
            "seed": seed,
            "octree_resolution": octree_resolution,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "num_chunks": num_chunks,
            "height_mm": height_mm,
        }
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
            {"status": "pending|processing|completed|error", "model_base64"?: str, "message"?: str}
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
        payload = {
            "image_front": base64.b64encode(images["front"]).decode(),
            "image_right": base64.b64encode(images["right"]).decode(),
            "image_back": base64.b64encode(images["back"]).decode(),
            "image_left": base64.b64encode(images["left"]).decode(),
            "remove_background": kwargs.get("remove_background", True),
            "seed": kwargs.get("seed", 1234),
            "octree_resolution": kwargs.get("octree_resolution", 256),
            "num_inference_steps": kwargs.get("num_inference_steps", 50),
            "guidance_scale": kwargs.get("guidance_scale", 5.0),
            "num_chunks": kwargs.get("num_chunks", 200000),
            "height_mm": kwargs.get("height_mm", 1750),
        }
        client = await self._get_client()
        resp = await client.post(
            f"{self.base_url}/generate", json=payload, timeout=REQUEST_TIMEOUT
        )
        resp.raise_for_status()
        return resp.content
