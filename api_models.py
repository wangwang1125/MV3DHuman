"""
Pydantic models for Hunyuan3D API server.
支持输入：至少一张 RGB 图（可 1～4 视图），可选法线图、可选深度图（深度图不参与推理）。
响应可包含抠完图的四视图 base64。
"""
from typing import Optional, Literal
from pydantic import BaseModel, Field, model_validator


class GenerationRequest(BaseModel):
    """Request model for multi-view 3D generation API.
    至少一张 RGB 图（image_front/right/back/left 至少一个非空）。可选：法线图、深度图（深度不参与推理）。
    """
    image_front: Optional[str] = Field(
        None,
        description="Base64 encoded front view RGB image (0°) for 3D generation",
        example="iVBORw0KGgoAAAANSUhEUgAAAAQAAAAECAIAAAAmkwkpAAAAEElEQVR4nGP8z4AATAxEcQAz0QEHOoQ+uAAAAABJRU5ErkJggg=="
    )
    image_right: Optional[str] = Field(
        None,
        description="Base64 encoded right view RGB image (90°) for 3D generation",
        example="iVBORw0KGgoAAAANSUhEUgAAAAQAAAAECAIAAAAmkwkpAAAAEElEQVR4nGP8z4AATAxEcQAz0QEHOoQ+uAAAAABJRU5ErkJggg=="
    )
    image_back: Optional[str] = Field(
        None,
        description="Base64 encoded back view RGB image (180°) for 3D generation",
        example="iVBORw0KGgoAAAANSUhEUgAAAAQAAAAECAIAAAAmkwkpAAAAEElEQVR4nGP8z4AATAxEcQAz0QEHOoQ+uAAAAABJRU5ErkJggg=="
    )
    image_left: Optional[str] = Field(
        None,
        description="Base64 encoded left view RGB image (270°) for 3D generation",
        example="iVBORw0KGgoAAAANSUhEUgAAAAQAAAAECAIAAAAmkwkpAAAAEElEQVR4nGP8z4AATAxEcQAz0QEHOoQ+uAAAAABJRU5ErkJggg=="
    )

    @model_validator(mode="after")
    def at_least_one_rgb(self):
        rgb = [self.image_front, self.image_right, self.image_back, self.image_left]
        if not any(r and r.strip() for r in rgb):
            raise ValueError("至少需要提供一张 RGB 图（image_front / image_right / image_back / image_left 至少一个非空）")
        return self
    # 可选：四视图法线图（参与推理）
    normal_front: Optional[str] = Field(None, description="Base64 encoded front view normal map (optional, used in inference)")
    normal_right: Optional[str] = Field(None, description="Base64 encoded right view normal map (optional)")
    normal_back: Optional[str] = Field(None, description="Base64 encoded back view normal map (optional)")
    normal_left: Optional[str] = Field(None, description="Base64 encoded left view normal map (optional)")
    # 可选：四视图深度图（仅接收，不参与推理）
    depth_front: Optional[str] = Field(None, description="Base64 encoded front view depth map (optional, not used in inference)")
    depth_right: Optional[str] = Field(None, description="Base64 encoded right view depth map (optional, not used in inference)")
    depth_back: Optional[str] = Field(None, description="Base64 encoded back view depth map (optional, not used in inference)")
    depth_left: Optional[str] = Field(None, description="Base64 encoded left view depth map (optional, not used in inference)")
    remove_background: bool = Field(
        True,
        description="Whether to automatically remove background from input images"
    )
    texture: bool = Field(
        False,
        description="Whether to generate textures for the 3D model"
    )
    seed: int = Field(
        1234,
        description="Random seed for reproducible generation",
        ge=0,
        le=2**32-1
    )
    octree_resolution: int = Field(
        256,
        description="Resolution of the octree for mesh generation",
        ge=64,
        le=512
    )
    num_inference_steps: int = Field(
        5,
        description="Number of inference steps for generation",
        ge=1,
        le=100
    )
    guidance_scale: float = Field(
        5.0,
        description="Guidance scale for generation",
        ge=0.1,
        le=20.0
    )
    num_chunks: int = Field(
        8000,
        description="Number of chunks for processing",
        ge=1000,
        le=20000
    )
    face_count: int = Field(
        40000,
        description="Maximum number of faces for texture generation",
        ge=1000,
        le=100000
    )


class GenerationResponse(BaseModel):
    """Response model for generation status"""
    uid: str = Field(..., description="Unique identifier for the generation task")


class StatusResponse(BaseModel):
    """Response model for status endpoint. 完成时可选返回抠完图的四视图 base64。"""
    status: str = Field(..., description="Status of the generation task")
    model_base64: Optional[str] = Field(
        None, 
        description="Base64 encoded generated model file (only when status is 'completed')"
    )
    message: Optional[str] = Field(
        None,
        description="Error message (only when status is 'error')"
    )
    # 抠完图的四视图（仅当 status=='completed' 且服务端保存了抠图结果时存在）
    image_front_matted_base64: Optional[str] = Field(None, description="Base64 front view after matting")
    image_right_matted_base64: Optional[str] = Field(None, description="Base64 right view after matting")
    image_back_matted_base64: Optional[str] = Field(None, description="Base64 back view after matting")
    image_left_matted_base64: Optional[str] = Field(None, description="Base64 left view after matting")


class HealthResponse(BaseModel):
    """Response model for health check"""
    status: str = Field(..., description="Health status")
    worker_id: str = Field(..., description="Worker identifier") 