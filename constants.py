"""
Constants and error messages for Hunyuan3D API server.
"""

# Error messages
SERVER_ERROR_MSG = "**NETWORK ERROR DUE TO HIGH TRAFFIC. PLEASE REGENERATE OR REFRESH THIS PAGE.**"
MODERATION_MSG = "YOUR INPUT VIOLATES OUR CONTENT MODERATION GUIDELINES. PLEASE TRY AGAIN."

# Default values
DEFAULT_SAVE_DIR = 'gradio_cache'
DEFAULT_WORKER_ID = None  # Will be generated if None

# API metadata
API_TITLE = "Hunyuan3D Multi-View RGB API Server"
API_DESCRIPTION = """
# Hunyuan3D 2.1 Multi-View RGB API Server

This API server provides endpoints for generating 3D models from multi-view 2D images using the Hunyuan3D model with RGB reconstruction mode.

## Features

- **Multi-View 3D Shape Generation**: Convert 4-view 2D images to high-quality 3D meshes
- **RGB Reconstruction**: Uses only RGB color images without requiring depth or normal maps
- **LoRA Fine-tuning**: Supports custom LoRA weights for enhanced quality
- **Texture Generation**: Generate PBR textures for 3D models
- **Background Removal**: Automatic background removal from input images
- **Multiple Formats**: Support for GLB and OBJ output formats
- **Async Processing**: Background task processing with status tracking
- **Concurrent Inference**: Supports multiple simultaneous generation requests

## Multi-View Input Format

This API requires **4 view images** in the following order:
- **Front View** (0°): The front-facing view of the object
- **Right View** (90°): Right side view (90° clockwise from front)
- **Back View** (180°): Back-facing view (opposite of front)
- **Left View** (270°): Left side view (270° clockwise from front)

All images should be:
- Base64 encoded
- Preferably square aspect ratio
- Consistent lighting and scale across views
- Object centered in each view

## Usage

1. Use `/generate` for immediate 3D model generation from multi-view images
2. Use `/send` for asynchronous processing with status tracking
3. Use `/status/{uid}` to check task progress and retrieve results
4. Use `/health` to verify service status

## Example Request

```json
{
  "image_front": "base64_encoded_front_view...",
  "image_right": "base64_encoded_right_view...",
  "image_back": "base64_encoded_back_view...",
  "image_left": "base64_encoded_left_view...",
  "remove_background": true,
  "texture": false,
  "seed": 1234,
  "num_inference_steps": 5,
  "guidance_scale": 5.0,
  "octree_resolution": 256,
  "num_chunks": 8000
}
```

## Model Information

- **Model**: Hunyuan3D-2.1 Multi-View RGB by Tencent
- **Mode**: 4-View RGB Reconstruction (no depth/normal maps required)
- **License**: TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT
- **Capabilities**: Multi-View Image-to-3D, Texture Generation, LoRA Fine-tuning
"""
API_VERSION = "2.1.0"
API_CONTACT = {
    "name": "Hunyuan3D Team",
    "url": "https://github.com/Tencent/Hunyuan3D",
}
API_LICENSE_INFO = {
    "name": "TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT",
    "url": "https://github.com/Tencent/Hunyuan3D/blob/main/LICENSE",
}

# API tags metadata
API_TAGS_METADATA = [
    {
        "name": "generation",
        "description": "Multi-view 3D model generation endpoints. Generate 3D models from 4-view RGB images with optional textures.",
    },
    {
        "name": "status",
        "description": "Task status and health check endpoints. Monitor generation progress and service health.",
    },
] 