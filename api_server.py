# Hunyuan 3D is licensed under the TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT
# except for the third-party components listed below.
# Hunyuan 3D does not impose any additional limitations beyond what is outlined
# in the repsective licenses of these third-party components.
# Users must comply with all terms and conditions of original licenses of these third-party
# components and must ensure that the usage of the third party components adheres to
# all relevant laws and regulations.

# For avoidance of doubts, Hunyuan 3D means the large language models and
# their software and algorithms, including trained model weights, parameters (including
# optimizer states), machine-learning model code, inference-enabling code, training-enabling code,
# fine-tuning enabling code and other elements of the foregoing made publicly available
# by Tencent in accordance with TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT.

"""
A model worker executes the model.
"""
import argparse
import asyncio
import base64
import logging
import os
import sys
import traceback
import uuid
from typing import Optional

import torch
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse

# Import from root-level modules
from api_models import GenerationRequest, GenerationResponse, StatusResponse, HealthResponse
from logger_utils import build_logger
from constants import (
    SERVER_ERROR_MSG, DEFAULT_SAVE_DIR, API_TITLE, API_DESCRIPTION, 
    API_VERSION, API_CONTACT, API_LICENSE_INFO, API_TAGS_METADATA
)
from model_worker import ModelWorker

# Global variables
SAVE_DIR = DEFAULT_SAVE_DIR
worker_id = str(uuid.uuid4())[:6]

# Ensure SAVE_DIR exists before creating logger
os.makedirs(SAVE_DIR, exist_ok=True)

logger = build_logger("controller", f"{SAVE_DIR}/controller.log")

# Global worker and semaphore instances
workers = []  # List of worker instances
model_semaphore = None

# Task status tracking dictionary
# Structure: {uid: {'status': str, 'message': str, 'file_path': str}}
task_status = {}


async def update_task_status(uid, status, message=None, file_path=None):
    """
    Update task status in the global dictionary.
    
    Args:
        uid: Task unique identifier
        status: Status string (pending/processing/texturing/completed/error)
        message: Optional message (for errors)
        file_path: Optional file path (for completed tasks)
    """
    uid_str = str(uid)
    if uid_str not in task_status:
        task_status[uid_str] = {}
    
    task_status[uid_str]['status'] = status
    if message:
        task_status[uid_str]['message'] = message
    if file_path:
        task_status[uid_str]['file_path'] = file_path
    
    logger.info(f"Task {uid_str} status updated to: {status}")


def _matted_images_base64(save_dir: str, uid: str) -> dict:
    """若存在抠图四视图文件，则返回 base64 字段，否则返回空 dict。"""
    out = {}
    for view in ('front', 'right', 'back', 'left'):
        path = os.path.join(save_dir, f'{uid}_matted_{view}.png')
        if os.path.isfile(path):
            try:
                out[f'image_{view}_matted_base64'] = base64.b64encode(open(path, 'rb').read()).decode()
            except Exception:
                pass
    return out


def select_worker():
    """
    Select a worker using load balancing strategy.
    Chooses the worker with the least current tasks.
    
    Returns:
        ModelWorker: Selected worker instance
    """
    if not workers:
        raise RuntimeError("No workers available")
    
    # Select worker with minimum load
    selected = min(workers, key=lambda w: w.get_load())
    logger.debug(f"Selected worker {selected.worker_id} (load: {selected.get_load()})")
    return selected


app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION,
    contact=API_CONTACT,
    license_info=API_LICENSE_INFO,
    tags_metadata=API_TAGS_METADATA
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/generate", tags=["generation"])
async def generate_3d_model(request: GenerationRequest):
    """
    Generate a 3D model from an input image (synchronous - waits for completion).
    
    This endpoint takes an image and generates a 3D model with optional textures.
    The generation process includes background removal, mesh generation, and optional texture mapping.
    
    Returns:
        FileResponse: The generated 3D model file (GLB or OBJ format)
    """
    logger.info("Worker generating (synchronous)...")
    
    # Convert Pydantic model to dict for compatibility
    params = request.dict()
    
    # Select worker using load balancing
    selected_worker = select_worker()
    logger.info(f"Using worker {selected_worker.worker_id} for synchronous generation")
    
    uid = uuid.uuid4()
    try:
        file_path, uid = await selected_worker.generate(uid, params)
        return FileResponse(file_path)
    except ValueError as e:
        traceback.print_exc()
        logger.error(f"Caught ValueError: {e}")
        ret = {
            "text": SERVER_ERROR_MSG,
            "error_code": 1,
        }
        return JSONResponse(ret, status_code=404)
    except torch.cuda.CudaError as e:
        logger.error(f"Caught torch.cuda.CudaError: {e}")
        ret = {
            "text": SERVER_ERROR_MSG,
            "error_code": 1,
        }
        return JSONResponse(ret, status_code=404)
    except Exception as e:
        logger.error(f"Caught Unknown Error: {e}")
        traceback.print_exc()
        ret = {
            "text": SERVER_ERROR_MSG,
            "error_code": 1,
        }
        return JSONResponse(ret, status_code=404)


@app.post("/send", response_model=GenerationResponse, tags=["generation"])
async def send_generation_task(request: GenerationRequest):
    """
    Send a 3D generation task to be processed asynchronously.
    
    This endpoint starts the generation process in the background and returns a task ID.
    Use the /status/{uid} endpoint to check the progress and retrieve the result.
    
    Returns:
        GenerationResponse: Contains the unique task identifier
    """
    logger.info("Worker send (async task)...")
    
    # Convert Pydantic model to dict for compatibility
    params = request.dict()
    
    uid = uuid.uuid4()
    uid_str = str(uid)
    
    try:
        # Initialize task status as pending
        await update_task_status(uid, 'pending')
        
        # Select worker using load balancing
        selected_worker = select_worker()
        logger.info(f"Using worker {selected_worker.worker_id} for async generation task {uid_str}")
        
        # Create async task instead of thread
        async def run_generation():
            try:
                file_path, _ = await selected_worker.generate(uid, params)
                await update_task_status(uid, 'completed', file_path=file_path)
            except Exception as e:
                logger.error(f"Generation task {uid_str} failed: {e}")
                await update_task_status(uid, 'error', message=str(e))
        
        # Start the task in background
        asyncio.create_task(run_generation())
        
        ret = {"uid": uid_str}
        return JSONResponse(ret, status_code=200)
    except Exception as e:
        logger.error(f"Failed to start generation task: {e}")
        await update_task_status(uid, 'error', message=str(e))
        ret = {"error": "Failed to start generation"}
        return JSONResponse(ret, status_code=500)


@app.get("/health", response_model=HealthResponse, tags=["status"])
async def health_check():
    """
    Health check endpoint to verify the service is running.
    
    Returns:
        HealthResponse: Service health status and worker identifier
    """
    return JSONResponse({"status": "healthy", "worker_id": worker_id}, status_code=200)


@app.get("/workers/status", tags=["status"])
async def workers_status():
    """
    Get the status of all workers including their current load.
    
    Returns:
        dict: Status information for all workers
    """
    worker_statuses = []
    for i, worker in enumerate(workers):
        worker_statuses.append({
            "worker_id": worker.worker_id,
            "current_tasks": worker.get_load(),
            "queue_length": len(worker.task_queue),
            "batch_size": worker.max_batch_size,
            "batch_timeout": worker.batch_timeout
        })
    
    return JSONResponse({
        "total_workers": len(workers),
        "workers": worker_statuses
    }, status_code=200)


@app.get("/status/{uid}", response_model=StatusResponse, tags=["status"])
async def status(uid: str):
    """
    Check the status of a generation task.
    
    Args:
        uid: The unique identifier of the generation task
        
    Returns:
        StatusResponse: Current status of the task and result if completed
    """
    # Check in-memory task status first
    if uid in task_status:
        task_info = task_status[uid]
        status_str = task_info.get('status', 'processing')
        
        if status_str == 'completed':
            # Read and return the generated file，并附带抠完图的四视图（若有）
            file_path = task_info.get('file_path')
            if file_path and os.path.exists(file_path):
                try:
                    base64_str = base64.b64encode(open(file_path, 'rb').read()).decode()
                    response = {'status': 'completed', 'model_base64': base64_str}
                    response.update(_matted_images_base64(SAVE_DIR, uid))
                    return JSONResponse(response, status_code=200)
                except Exception as e:
                    logger.error(f"Error reading file {file_path}: {e}")
                    response = {'status': 'error', 'message': 'Failed to read generated file'}
                    return JSONResponse(response, status_code=500)
            else:
                # Fallback to file system check
                mesh_file_path = os.path.join(SAVE_DIR, f'{uid}.glb')
                
                if os.path.exists(mesh_file_path):
                    try:
                        base64_str = base64.b64encode(open(mesh_file_path, 'rb').read()).decode()
                        response = {'status': 'completed', 'model_base64': base64_str}
                        response.update(_matted_images_base64(SAVE_DIR, uid))
                        return JSONResponse(response, status_code=200)
                    except Exception as e:
                        logger.error(f"Error reading file {mesh_file_path}: {e}")
                        response = {'status': 'error', 'message': 'Failed to read generated file'}
                        return JSONResponse(response, status_code=500)
        
        elif status_str == 'error':
            message = task_info.get('message', 'Unknown error')
            response = {'status': 'error', 'message': message}
            return JSONResponse(response, status_code=500)
        
        else:
            # pending or processing
            response = {'status': status_str}
            return JSONResponse(response, status_code=200)
    
    # Task not found in status dict, fallback to file system check
    else:
        mesh_file_path = os.path.join(SAVE_DIR, f'{uid}.glb')
        
        # If mesh file exists, generation is complete
        if os.path.exists(mesh_file_path):
            try:
                base64_str = base64.b64encode(open(mesh_file_path, 'rb').read()).decode()
                response = {'status': 'completed', 'model_base64': base64_str}
                response.update(_matted_images_base64(SAVE_DIR, uid))
                return JSONResponse(response, status_code=200)
            except Exception as e:
                logger.error(f"Error reading file {mesh_file_path}: {e}")
                response = {'status': 'error', 'message': 'Failed to read generated file'}
                return JSONResponse(response, status_code=500)
        
        # If no file exists, either processing or task doesn't exist
        else:
            response = {'status': 'processing'}
            return JSONResponse(response, status_code=200)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument("--model_path", type=str, default='tencent/Hunyuan3D-2.1')
    parser.add_argument("--subfolder", type=str, default='hunyuan3d-dit-v2-1')
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--limit-model-concurrency", type=int, default=2,
                        help="Maximum number of concurrent model inference (default: 2 for multi-view)")
    parser.add_argument('--low_vram_mode', action='store_true')
    parser.add_argument('--cache-path', type=str, default='./gradio_cache')
    parser.add_argument('--enable_multiview_rgb', action='store_true',
                        help='Enable multi-view RGB reconstruction mode (4 views)')
    parser.add_argument('--rgb_lora_path', type=str, default=None,
                        help='Path to RGB LoRA checkpoint (Lightning .ckpt or PEFT directory)')
    parser.add_argument('--num_views', type=int, default=4,
                        help='Number of views for multi-view reconstruction (fixed to 4)')
    parser.add_argument('--batch-size', type=int, default=2,
                        help='Batch size for parallel processing (default: 2)')
    parser.add_argument('--batch-timeout', type=float, default=1.0,
                        help='Batch timeout in seconds (default: 1.0)')
    parser.add_argument('--num-workers', type=int, default=2,
                        help='Number of worker instances to create (default: 2)')
    args = parser.parse_args()
    logger.info(f"args: {args}")
    
    # Auto-search for RGB LoRA weights if not specified
    if args.enable_multiview_rgb and args.rgb_lora_path is None:
        logger.info("RGB LoRA path not specified, searching in default locations...")
        default_lora_dirs = [
            "./hy3dshape/output_folder/dit/multiview_rgb_lora_finetuning/ckpt",
            "./output_folder/dit/multiview_rgb_lora_finetuning/ckpt",
            "./hy3dshape/output_folder/dit/multiview_rgb_lora_checkpoints",
            "./output_folder/dit/multiview_rgb_lora_checkpoints",
        ]
        
        for lora_dir in default_lora_dirs:
            if os.path.exists(lora_dir):
                # If it's a directory, look for .ckpt files
                if os.path.isdir(lora_dir):
                    ckpt_files = [f for f in os.listdir(lora_dir) if f.endswith('.ckpt')]
                    if ckpt_files:
                        # Sort by filename and choose the latest (largest step number)
                        latest_ckpt = max(ckpt_files, 
                                        key=lambda x: int(x.split('=')[1].split('.')[0]) if '=' in x else 0)
                        args.rgb_lora_path = os.path.join(lora_dir, latest_ckpt)
                        logger.info(f"✅ Auto-discovered RGB LoRA weights: {args.rgb_lora_path}")
                        break
                else:
                    args.rgb_lora_path = lora_dir
                    break
        
        if args.rgb_lora_path is None:
            logger.warning("⚠️ No RGB LoRA weights found, will use base model")
    
    if args.enable_multiview_rgb:
        logger.info("=" * 60)
        logger.info("MULTI-VIEW RGB RECONSTRUCTION MODE")
        logger.info("=" * 60)
        logger.info(f"Number of views: {args.num_views}")
        logger.info(f"RGB LoRA path: {args.rgb_lora_path or 'None (using base model)'}")
        logger.info("=" * 60)

    # Update SAVE_DIR based on cache-path argument
    SAVE_DIR = args.cache_path
    os.makedirs(SAVE_DIR, exist_ok=True)
    

    # Create multiple worker instances
    # Each worker gets its own semaphore to allow true parallel processing
    num_workers = args.num_workers
    logger.info(f"Creating {num_workers} worker instance(s)...")
    
    for i in range(num_workers):
        worker_id_i = f"{worker_id}-{i+1}"
        # Each worker gets its own semaphore (value=1 means one batch per worker at a time)
        # This allows multiple workers to process batches in parallel
        worker_semaphore = asyncio.Semaphore(1)
        
        worker = ModelWorker(
            model_path=args.model_path, 
            subfolder=args.subfolder,
            device=args.device, 
            low_vram_mode=args.low_vram_mode,
            worker_id=worker_id_i,
            model_semaphore=worker_semaphore,
            save_dir=SAVE_DIR,
            status_callback=update_task_status,
            enable_multiview_rgb=args.enable_multiview_rgb,
            rgb_lora_path=args.rgb_lora_path,
            num_views=args.num_views,
            batch_size=args.batch_size,
            batch_timeout=args.batch_timeout
        )
        workers.append(worker)
        logger.info(f"Worker {i+1}/{num_workers} initialized successfully (worker_id: {worker_id_i})")
    
    logger.info(f"All {num_workers} worker(s) initialized successfully")
    logger.info(f"Starting API server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
