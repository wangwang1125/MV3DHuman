"""
Model worker for Hunyuan3D API server.
"""
import asyncio
import os
import time
import uuid
import base64
import trimesh
from io import BytesIO
from PIL import Image
import torch
from collections import deque
from dataclasses import dataclass
from typing import Optional, Dict, Any

# Apply torchvision compatibility fix before other imports
import sys
sys.path.insert(0, './hy3dshape')
sys.path.insert(0, './hy3dpaint')

try:
    from torchvision_fix import apply_fix
    apply_fix()
except ImportError:
    print("Warning: torchvision_fix module not found, proceeding without compatibility fix")
except Exception as e:
    print(f"Warning: Failed to apply torchvision fix: {e}")

from hy3dshape import Hunyuan3DDiTFlowMatchingPipeline
from hy3dshape.rembg import BackgroundRemover
from hy3dshape.utils import logger
from hy3dshape.pipelines import export_to_trimesh


@dataclass
class BatchTask:
    """Batch processing task data structure"""
    uid: str
    params: Dict[str, Any]
    future: asyncio.Future
    submit_time: float


def load_image_from_base64(image):
    """
    Load an image from base64 encoded string.
    
    Args:
        image (str): Base64 encoded image string
        
    Returns:
        PIL.Image: Loaded image
    """
    return Image.open(BytesIO(base64.b64decode(image)))


def build_normal_data_from_params(params, target_size=518):
    """
    从请求 params 中可选的 normal_front/right/back/left (base64) 构建法线张量。
    深度图在 params 中仅接收不参与推理，此处不处理。
    
    Returns:
        dict with 'normal' (1, 4, 3, H, W), 'normal_mask' (1, 4, 1, H, W), or None if no normals provided.
    """
    view_order = ['front', 'right', 'back', 'left']
    keys = [f'normal_{v}' for v in view_order]
    if not any(params.get(k) for k in keys):
        return None
    import numpy as np
    normal_tensors = []
    mask_tensors = []
    for key in keys:
        b64 = params.get(key)
        if b64:
            try:
                img = load_image_from_base64(b64).convert('RGB')
                if img.size[0] != target_size or img.size[1] != target_size:
                    img = img.resize((target_size, target_size), Image.BILINEAR)
                arr = np.array(img).astype(np.float32) / 255.0
                t = torch.FloatTensor(arr).permute(2, 0, 1)
                mask = (t.sum(dim=0) > 0.01).float().unsqueeze(0)
                normal_tensors.append(t)
                mask_tensors.append(mask)
            except Exception as e:
                logger.warning(f"Failed to load normal from {key}: {e}")
                normal_tensors.append(torch.zeros(3, target_size, target_size))
                mask_tensors.append(torch.zeros(1, target_size, target_size))
        else:
            normal_tensors.append(torch.zeros(3, target_size, target_size))
            mask_tensors.append(torch.zeros(1, target_size, target_size))
    multiview_normal = torch.stack(normal_tensors, dim=0).unsqueeze(0)
    multiview_mask = torch.stack(mask_tensors, dim=0).unsqueeze(0)
    return {'normal': multiview_normal, 'normal_mask': multiview_mask}


class ModelWorker:
    """
    Worker class for handling 3D model generation tasks.
    """
    
    def __init__(self,
                 model_path='tencent/Hunyuan3D-2.1',
                 subfolder='hunyuan3d-dit-v2-1',
                 device='cuda',
                 low_vram_mode=False,
                 worker_id=None,
                 model_semaphore=None,
                 save_dir='gradio_cache',
                 status_callback=None,
                 enable_multiview_rgb=False,
                 rgb_lora_path=None,
                 num_views=4,
                 batch_size=2,
                 batch_timeout=1.0):
        """
        Initialize the model worker.
        
        Args:
            model_path (str): Path to the shape generation model
            subfolder (str): Subfolder containing the model files
            device (str): Device to run the model on ('cuda' or 'cpu')
            low_vram_mode (bool): Whether to use low VRAM mode
            worker_id (str): Unique identifier for this worker
            model_semaphore: Semaphore for controlling model concurrency
            save_dir (str): Directory to save generated files
            status_callback: Callback function to update task status
            enable_multiview_rgb (bool): Enable multi-view RGB reconstruction mode
            rgb_lora_path (str): Path to RGB LoRA weights
            num_views (int): Number of views for multi-view mode
            batch_size (int): Maximum batch size for parallel processing
            batch_timeout (float): Timeout in seconds before processing incomplete batch
        """
        self.model_path = model_path
        self.worker_id = worker_id or str(uuid.uuid4())[:6]
        self.device = device
        self.low_vram_mode = low_vram_mode
        self.model_semaphore = model_semaphore
        self.save_dir = save_dir
        self.status_callback = status_callback
        self.enable_multiview_rgb = enable_multiview_rgb
        self.rgb_lora_path = rgb_lora_path
        self.num_views = num_views
        
        # Batch processing configuration
        self.max_batch_size = batch_size
        self.batch_timeout = batch_timeout
        self.task_queue = deque()
        self.queue_lock = asyncio.Lock()
        self.batch_processor_task = None
        
        # Task tracking for load balancing
        self.current_tasks = 0
        self._tasks_lock = asyncio.Lock()
        
        # Add lock for pipeline thread safety (currently unused, enable if needed)
        # If you encounter GPU race conditions, uncomment the lock usage in _generate_internal()
        self.pipeline_lock = asyncio.Lock()
        self.use_pipeline_lock = False  # Set to True if thread safety issues occur
        
        # Create independent CUDA stream for this worker to avoid GPU blocking
        # Each worker uses its own stream, allowing parallel GPU execution
        if device == 'cuda' and torch.cuda.is_available():
            self.cuda_stream = torch.cuda.Stream()
            logger.info(f"[Worker {self.worker_id}] Created independent CUDA stream for GPU isolation")
        else:
            self.cuda_stream = None
        
        logger.info(f"Loading the model {model_path} on worker {self.worker_id} ...")
        logger.info(f"Batch processing enabled: batch_size={batch_size}, timeout={batch_timeout}s")
        
        if enable_multiview_rgb:
            logger.info(f"Enabling multi-view RGB reconstruction mode ({num_views} views)")

        # Initialize background remover
        self.rembg = BackgroundRemover()
        
        # Initialize shape generation pipeline
        # Note: The pipeline configuration should match gradio_app.py
        self.pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            model_path,
            subfolder=subfolder,
            use_safetensors=False,
            device=device
        )
        
        # Load RGB LoRA if specified (for multi-view RGB mode)
        if enable_multiview_rgb and rgb_lora_path:
            logger.info(f"Loading RGB LoRA weights: {rgb_lora_path}")
            try:
                from peft import PeftModel
                
                # Check if Lightning checkpoint format (.ckpt)
                if rgb_lora_path.endswith('.ckpt'):
                    logger.info("Detected Lightning checkpoint format...")
                    ckpt = torch.load(rgb_lora_path, map_location='cpu')
                    
                    if 'state_dict' in ckpt:
                        state_dict = ckpt['state_dict']
                        logger.info(f"Checkpoint contains {len(state_dict)} weights")
                        
                        # Analyze checkpoint content to determine if it's full fine-tuning or LoRA
                        lora_keys = [k for k in state_dict.keys() if 'lora' in k.lower()]
                        model_keys = [k for k in state_dict.keys() if k.startswith('model.')]
                        
                        logger.info(f"  - Model weights: {len(model_keys)}")
                        logger.info(f"  - LoRA parameters: {len(lora_keys)}")
                        
                        success_count = 0
                        
                        # Determine if it's full fine-tuning or LoRA
                        if lora_keys:
                            # LoRA format: contains LoRA parameters
                            logger.info("\nDetected LoRA format checkpoint, loading RGB LoRA weights...")
                            if hasattr(self.pipeline, 'model'):
                                try:
                                    # Extract model-related weights (including LoRA)
                                    model_state_dict = {}
                                    for key, value in state_dict.items():
                                        if key.startswith('model.'):
                                            new_key = key[6:]  # Remove 'model.' prefix
                                            model_state_dict[new_key] = value
                                    
                                    # Apply LoRA config to base model
                                    from peft import LoraConfig, get_peft_model
                                    lora_config = LoraConfig(
                                        r=8,
                                        lora_alpha=8,
                                        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
                                        lora_dropout=0.0,
                                    )
                                    self.pipeline.model = get_peft_model(self.pipeline.model, lora_config)
                                    
                                    # Load weights including LoRA
                                    missing, unexpected = self.pipeline.model.load_state_dict(
                                        model_state_dict, strict=False)
                                    logger.info(f"✅ RGB LoRA weights loaded successfully")
                                    logger.info(f"  - Missing keys: {len(missing)}")
                                    logger.info(f"  - Unexpected keys: {len(unexpected)}")
                                    success_count += 1
                                    
                                except Exception as e:
                                    logger.error(f"❌ Failed to load RGB LoRA weights: {e}")
                                    import traceback
                                    traceback.print_exc()
                        else:
                            # Full fine-tuning format: no LoRA parameters, load full model weights directly
                            logger.info("\nDetected full fine-tuning format checkpoint, loading full model weights...")
                            if hasattr(self.pipeline, 'model'):
                                try:
                                    # Extract model-related weights (full weights, no LoRA)
                                    model_state_dict = {}
                                    for key, value in state_dict.items():
                                        if key.startswith('model.'):
                                            new_key = key[6:]  # Remove 'model.' prefix
                                            model_state_dict[new_key] = value
                                    
                                    # Load full model weights directly (no LoRA)
                                    missing, unexpected = self.pipeline.model.load_state_dict(
                                        model_state_dict, strict=False)
                                    logger.info(f"✅ Full fine-tuning weights loaded successfully")
                                    logger.info(f"  - Missing keys: {len(missing)}")
                                    logger.info(f"  - Unexpected keys: {len(unexpected)}")
                                    success_count += 1
                                    
                                except Exception as e:
                                    logger.error(f"❌ Failed to load full fine-tuning weights: {e}")
                                    import traceback
                                    traceback.print_exc()
                        
                        if success_count > 0:
                            checkpoint_type = "LoRA" if lora_keys else "full fine-tuning"
                            logger.info(f"\n✅ Successfully loaded RGB {checkpoint_type} weights from Lightning checkpoint")
                        else:
                            logger.warning("\n❌ Failed to load RGB weights")
                            
                    else:
                        logger.error("❌ Invalid checkpoint format, missing state_dict")
                
                elif os.path.isdir(rgb_lora_path):
                    # Standard PEFT format directory
                    logger.info("Detected PEFT directory format, using standard LoRA loading...")
                    
                    if hasattr(self.pipeline, 'model'):
                        try:
                            logger.info("Loading RGB LoRA weights to main DiT model...")
                            logger.info(f"  Model type: {type(self.pipeline.model)}")
                            logger.info(f"  LoRA path: {rgb_lora_path}")
                            
                            # Apply LoRA directly to pipeline.model
                            self.pipeline.model = PeftModel.from_pretrained(
                                self.pipeline.model, rgb_lora_path)
                            
                            logger.info("✅ RGB LoRA weights loaded successfully")
                        except Exception as e:
                            logger.error(f"❌ Failed to load RGB LoRA weights: {e}")
                            import traceback
                            traceback.print_exc()
                    else:
                        logger.error("❌ Pipeline does not have model attribute")
                else:
                    logger.error(f"❌ Unsupported RGB LoRA weight format: {rgb_lora_path}")
                    
            except Exception as e:
                import traceback
                logger.error(f"❌ Exception while loading RGB LoRA weights: {e}")
                logger.error("Detailed error:")
                traceback.print_exc()
                logger.warning("Will use base RGB condition model")
        
        # Setup multi-view image processor if needed
        if enable_multiview_rgb:
            logger.info("=" * 60)
            logger.info("CONFIGURING MULTI-VIEW RGB MODE")
            logger.info("=" * 60)
            try:
                from hy3dshape.preprocessors import MVImageProcessorV2
                from hy3dshape.models.conditioner import DinoImageEncoderMV
                
                # Set multi-view image processor
                logger.info(f"Setting MVImageProcessorV2 (size=518)...")
                self.pipeline.image_processor = MVImageProcessorV2(size=518)
                logger.info(f"✅ Image processor type: {type(self.pipeline.image_processor).__name__}")
                
                # Get pipeline dtype
                pipeline_dtype = getattr(self.pipeline, 'dtype', torch.float16)
                if not hasattr(self.pipeline, 'dtype'):
                    # Try to infer from model
                    if hasattr(self.pipeline, 'model') and hasattr(self.pipeline.model, 'dtype'):
                        pipeline_dtype = self.pipeline.model.dtype
                    logger.info(f"Inferred pipeline dtype: {pipeline_dtype}")
                else:
                    logger.info(f"Pipeline dtype: {pipeline_dtype}")
                
                # Check conditioner structure
                if not hasattr(self.pipeline, 'conditioner'):
                    raise RuntimeError("Pipeline does not have 'conditioner' attribute")
                
                if not hasattr(self.pipeline.conditioner, 'main_image_encoder'):
                    raise RuntimeError("Conditioner does not have 'main_image_encoder' attribute")
                
                current_encoder = self.pipeline.conditioner.main_image_encoder
                logger.info(f"Current encoder type: {type(current_encoder).__name__}")
                logger.info(f"Current encoder class: {current_encoder.__class__.__module__}.{current_encoder.__class__.__name__}")
                
                # Check if already DinoImageEncoderMV
                if isinstance(current_encoder, DinoImageEncoderMV):
                    logger.info(f"✅ Encoder is already DinoImageEncoderMV with view_num={current_encoder.view_num}")
                    if current_encoder.view_num != num_views:
                        logger.warning(f"⚠️ Encoder view_num ({current_encoder.view_num}) != requested num_views ({num_views})")
                else:
                    # Replace encoder with multi-view version
                    logger.info(f"Replacing {type(current_encoder).__name__} with DinoImageEncoderMV...")
                    logger.info(f"  Creating DinoImageEncoderMV:")
                    logger.info(f"    - version: facebook/dinov2-large")
                    logger.info(f"    - image_size: 518")
                    logger.info(f"    - use_cls_token: True")
                    logger.info(f"    - view_num: {num_views}")
                    
                    new_encoder = DinoImageEncoderMV(
                        version='facebook/dinov2-large',
                        image_size=518,
                        use_cls_token=True,
                        view_num=num_views
                    )
                    
                    # Copy weights if possible
                    if hasattr(current_encoder, 'model') and hasattr(new_encoder, 'model'):
                        try:
                            logger.info("  Attempting to copy encoder weights...")
                            new_encoder.model.load_state_dict(current_encoder.model.state_dict())
                            logger.info("  ✅ Successfully copied encoder weights")
                        except Exception as copy_error:
                            logger.warning(f"  ⚠️ Could not copy encoder weights: {copy_error}")
                            logger.warning("  Will use default DinoV2 weights")
                    
                    # Move to device
                    logger.info(f"  Moving encoder to {device} with dtype {pipeline_dtype}...")
                    new_encoder = new_encoder.to(device, dtype=pipeline_dtype)
                    
                    # Replace encoder
                    self.pipeline.conditioner.main_image_encoder = new_encoder
                    logger.info(f"  ✅ Encoder replaced")
                    
                    # Verify replacement
                    verify_encoder = self.pipeline.conditioner.main_image_encoder
                    logger.info(f"  Verification - Encoder type: {type(verify_encoder).__name__}")
                    logger.info(f"  Verification - view_num: {getattr(verify_encoder, 'view_num', 'N/A')}")
                
                logger.info("=" * 60)
                logger.info("✅ MULTI-VIEW RGB MODE CONFIGURED")
                logger.info("=" * 60)
                
            except Exception as e:
                logger.error("=" * 60)
                logger.error(f"❌ FAILED TO SETUP MULTI-VIEW COMPONENTS")
                logger.error("=" * 60)
                logger.error(f"Error: {e}")
                import traceback
                traceback.print_exc()
                logger.error("=" * 60)
                raise RuntimeError(f"Multi-view RGB mode setup failed: {e}")
        
        # Clean cache in save_dir
        if os.path.exists(self.save_dir):
            for file in os.listdir(self.save_dir):
                file_path = os.path.join(self.save_dir, file)
                try:
                    os.remove(file_path)
                except Exception as e:
                    logger.warning(f"Failed to remove cache file {file_path}: {e}")
        
        # Batch processor will be started on first generate() call
        self._batch_processor_started = False
    
    async def _batch_processor_loop(self):
        """Batch processor main loop: collect tasks and execute in batches"""
        logger.info(f"[Worker {self.worker_id}] Batch processor loop started")
        
        while True:
            try:
                await asyncio.sleep(0.01)  # Avoid CPU 100%
                
                async with self.queue_lock:
                    if len(self.task_queue) == 0:
                        continue
                    
                    # Check if we should start batch processing
                    should_process = False
                    reason = ""
                    
                    # Condition 1: Queue is full
                    if len(self.task_queue) >= self.max_batch_size:
                        should_process = True
                        reason = f"batch full ({len(self.task_queue)})"
                    
                    # Condition 2: Timeout (oldest task waiting time exceeds timeout)
                    elif len(self.task_queue) > 0:
                        oldest_task = self.task_queue[0]
                        wait_time = time.time() - oldest_task.submit_time
                        if wait_time >= self.batch_timeout:
                            should_process = True
                            reason = f"timeout ({wait_time:.2f}s)"
                    
                    if not should_process:
                        continue
                    
                    # Extract a batch of tasks
                    batch_tasks = []
                    for _ in range(min(self.max_batch_size, len(self.task_queue))):
                        batch_tasks.append(self.task_queue.popleft())
                
                # Process batch outside of lock
                logger.info(f"[Worker {self.worker_id}] Processing batch of {len(batch_tasks)} tasks (reason: {reason})")
                await self._process_batch(batch_tasks)
                
            except Exception as e:
                logger.error(f"[Worker {self.worker_id}] Batch processor error: {e}")
                import traceback
                traceback.print_exc()
    
    async def _process_batch(self, batch_tasks):
        """Process multiple tasks in a batch"""
        if not batch_tasks:
            return
        
        batch_size = len(batch_tasks)
        logger.info(f"[Batch] Processing {batch_size} tasks: {[t.uid for t in batch_tasks]}")
        
        try:
            # Stage 1: Parallel preprocessing (background removal) for all tasks
            async def preprocess_task(task):
                """Preprocess a single task's images"""
                params = task.params
                
                # Update status
                if self.status_callback:
                    await self.status_callback(task.uid, 'processing')
                
                # Multi-view mode（RGB 必选；法线可选参与推理；深度仅接收不参与推理）
                if 'image_front' in params:
                    image = {}
                    view_names = ['image_front', 'image_right', 'image_back', 'image_left']
                    
                    for view_name in view_names:
                        if view_name in params and params[view_name]:
                            img = load_image_from_base64(params[view_name])
                            img = img.convert("RGBA")
                            if params.get("remove_background", True) or img.mode == "RGB":
                                loop = asyncio.get_event_loop()
                                img = await loop.run_in_executor(None, self.rembg, img)
                            simple_name = view_name.replace('image_', '')
                            image[simple_name] = img
                    
                    if not image:
                        raise ValueError("No multi-view images provided")
                    normal_data = build_normal_data_from_params(params)
                    return task.uid, image, params, normal_data
                
                # Single-view mode
                elif 'image' in params:
                    image = params["image"]
                    image = load_image_from_base64(image)
                    image = image.convert("RGBA")
                    if params.get("remove_background", True) or image.mode == "RGB":
                        loop = asyncio.get_event_loop()
                        image = await loop.run_in_executor(None, self.rembg, image)
                    return task.uid, image, params, None
                
                else:
                    raise ValueError("No input image provided")
            
            # Parallel preprocessing of all tasks
            preprocessed = await asyncio.gather(
                *[preprocess_task(task) for task in batch_tasks],
                return_exceptions=True
            )
            
            # Check preprocessing errors
            valid_tasks = []
            for i, result in enumerate(preprocessed):
                if isinstance(result, Exception):
                    logger.error(f"[Batch] Task {batch_tasks[i].uid} preprocessing failed: {result}")
                    batch_tasks[i].future.set_exception(result)
                    if self.status_callback:
                        await self.status_callback(batch_tasks[i].uid, 'error', str(result))
                else:
                    valid_tasks.append((batch_tasks[i], result))
            
            if not valid_tasks:
                logger.warning("[Batch] No valid tasks after preprocessing")
                return
            
            # 保存抠完图的四视图，供 status 接口返回
            for task, result in valid_tasks:
                uid, image, params, normal_data = result[0], result[1], result[2], result[3] if len(result) > 3 else None
                if isinstance(image, dict):
                    for view in ['front', 'right', 'back', 'left']:
                        if view in image:
                            path = os.path.join(self.save_dir, f'{uid}_matted_{view}.png')
                            try:
                                image[view].save(path)
                            except Exception as e:
                                logger.warning(f"Failed to save matted view {view} for {uid}: {e}")
            
            # Build batch inputs
            batch_images = []
            batch_params_list = []
            normal_data_list = []
            for task, result in valid_tasks:
                uid, image, params = result[0], result[1], result[2]
                normal_data = result[3] if len(result) > 3 else None
                batch_images.append(image)
                batch_params_list.append(params)
                normal_data_list.append(normal_data)
            
            # Use first task's parameters as batch parameters
            ref_params = batch_params_list[0]
            
            # 批法线：若任一样本有法线则传入 pipeline（缺失的用零填充）
            batch_normal = None
            batch_normal_mask = None
            if any(nd is not None for nd in normal_data_list):
                target_size = 518
                normals = []
                masks = []
                for nd in normal_data_list:
                    if nd is not None:
                        normals.append(nd['normal'])
                        masks.append(nd['normal_mask'])
                    else:
                        normals.append(torch.zeros(1, 4, 3, target_size, target_size))
                        masks.append(torch.zeros(1, 4, 1, target_size, target_size))
                batch_normal = torch.cat(normals, dim=0)
                batch_normal_mask = torch.cat(masks, dim=0)
            
            # Prepare batch pipeline parameters
            # Use 'trimesh' output_type to get trimesh.Trimesh objects directly
            # Default values match gradio_app.py for consistency
            pipeline_params = {
                'image': batch_images,  # Pass list, pipeline will auto-batch
                'num_inference_steps': ref_params.get('num_inference_steps', 50),  # Match gradio default
                'guidance_scale': ref_params.get('guidance_scale', 7.5),  # Match gradio default
                'octree_resolution': ref_params.get('octree_resolution', 256),
                'num_chunks': ref_params.get('num_chunks', 200000),  # Match gradio default
                'output_type': 'trimesh'  # Changed from 'mesh' to 'trimesh' to get Trimesh objects
            }
            if batch_normal is not None:
                pipeline_params['normal'] = batch_normal
                pipeline_params['normal_mask'] = batch_normal_mask
            
            # Set seeds for each task (can be different)
            generators = []
            for params in batch_params_list:
                if 'seed' in params:
                    generator = torch.Generator()
                    generator = generator.manual_seed(int(params['seed']))
                    generators.append(generator)
                else:
                    generators.append(None)
            
            if all(g is not None for g in generators):
                pipeline_params['generator'] = generators
            
            # Execute batch inference
            try:
                loop = asyncio.get_event_loop()
                
                # Fix view_idxs format for batch processing of multi-view images
                # The prepare_image method returns view_idxs as [[[0,1,2,3]], [[0,1,2,3]], ...]
                # but conditioner expects [[0,1,2,3], [0,1,2,3], ...]
                original_prepare_image = self.pipeline.prepare_image
                
                def fixed_prepare_image(image, mask=None):
                    result = original_prepare_image(image, mask)
                    # Fix view_idxs format if present
                    if 'view_idxs' in result and isinstance(result['view_idxs'], list) and len(result['view_idxs']) > 0:
                        # Check if triple nested: [[[0,1,2,3]], [[0,1,2,3]]]
                        first_item = result['view_idxs'][0]
                        if isinstance(first_item, list) and len(first_item) > 0:
                            if isinstance(first_item[0], list):
                                # Triple nested, flatten to double nested: [[[0,1,2,3]]] -> [[0,1,2,3]]
                                result['view_idxs'] = [item[0] if isinstance(item, list) and len(item) > 0 else item 
                                                       for item in result['view_idxs']]
                    return result
                
                # Temporarily replace prepare_image method
                self.pipeline.prepare_image = fixed_prepare_image
                
                # Define pipeline execution function with CUDA stream support
                def run_pipeline_with_stream():
                    """Execute pipeline in worker's CUDA stream for GPU isolation"""
                    logger.info(f"[Worker {self.worker_id}] Starting pipeline execution in CUDA stream")
                    if self.cuda_stream is not None:
                        # Execute all GPU operations in this worker's stream
                        # The context manager ensures all operations in this block use the stream
                        with torch.cuda.stream(self.cuda_stream):
                            result = self.pipeline(**pipeline_params)
                        # Synchronize this stream to ensure completion before returning
                        self.cuda_stream.synchronize()
                        logger.info(f"[Worker {self.worker_id}] Pipeline execution completed, stream synchronized")
                        return result
                    else:
                        # No CUDA stream (CPU mode), execute normally
                        logger.info(f"[Worker {self.worker_id}] Pipeline execution (CPU mode)")
                        return self.pipeline(**pipeline_params)
                
                try:
                    # Use semaphore to control concurrent batches
                    if self.model_semaphore:
                        async with self.model_semaphore:
                            meshes = await loop.run_in_executor(
                                None, 
                                run_pipeline_with_stream
                            )
                    else:
                        meshes = await loop.run_in_executor(
                            None, 
                            run_pipeline_with_stream
                        )
                finally:
                    # Restore original method
                    self.pipeline.prepare_image = original_prepare_image
                
                logger.info(f"[Worker {self.worker_id}] [Batch] GPU inference completed, got batch results")
                
                # Extract meshes from batch results
                # Pipeline returns List[List[trimesh.Trimesh]] for batch
                # Outer list is batch dimension, inner list is mesh list per sample
                # For each batch element, we typically want the first mesh (index 0)
                extracted_meshes = []
                for i, batch_result in enumerate(meshes):
                    if isinstance(batch_result, list):
                        if len(batch_result) > 0:
                            # Take first mesh from the list
                            mesh = batch_result[0]
                            if hasattr(mesh, 'export'):
                                extracted_meshes.append(mesh)
                            else:
                                # Try to convert Latent2MeshOutput if needed
                                converted = export_to_trimesh(mesh)
                                if isinstance(converted, list) and len(converted) > 0:
                                    extracted_meshes.append(converted[0])
                                else:
                                    extracted_meshes.append(converted)
                        else:
                            raise ValueError(f"Batch result {i} is an empty list")
                    elif hasattr(batch_result, 'export'):
                        # Already a Trimesh object (shouldn't happen with batch, but handle it)
                        extracted_meshes.append(batch_result)
                    else:
                        # Try to convert Latent2MeshOutput if needed
                        converted = export_to_trimesh(batch_result)
                        if isinstance(converted, list) and len(converted) > 0:
                            extracted_meshes.append(converted[0])
                        else:
                            extracted_meshes.append(converted)
                
                meshes = extracted_meshes
                logger.info(f"[Batch] Extracted {len(meshes)} meshes from batch results")
                
            except Exception as e:
                logger.error(f"[Batch] GPU inference failed: {e}")
                import traceback
                traceback.print_exc()
                for task, _ in valid_tasks:
                    task.future.set_exception(e)
                    if self.status_callback:
                        await self.status_callback(task.uid, 'error', str(e))
                return
            
            # Stage 3: Parallel post-processing (export mesh)
            async def postprocess_task(task, mesh):
                """Post-process a single task"""
                uid = task.uid
                
                try:
                    # Ensure mesh is a trimesh.Trimesh object with export method
                    if mesh is None:
                        raise ValueError(f"Mesh is None for task {uid}")
                    
                    if not hasattr(mesh, 'export'):
                        # Try to convert if needed
                        mesh = export_to_trimesh(mesh)
                        if isinstance(mesh, list) and len(mesh) > 0:
                            mesh = mesh[0]
                    
                    # Export mesh
                    loop = asyncio.get_event_loop()
                    final_save_path = os.path.join(self.save_dir, f'{uid}.glb')
                    await loop.run_in_executor(None, mesh.export, final_save_path)
                    
                    # Update status
                    if self.status_callback:
                        await self.status_callback(uid, 'completed', file_path=final_save_path)
                    
                    # Return result
                    task.future.set_result((final_save_path, uid))
                    logger.info(f"[Task {uid}] Completed")
                    
                except Exception as e:
                    logger.error(f"[Task {uid}] Post-processing failed: {e}")
                    import traceback
                    traceback.print_exc()
                    task.future.set_exception(e)
                    if self.status_callback:
                        await self.status_callback(uid, 'error', str(e))
            
            # Parallel post-processing of all tasks
            await asyncio.gather(*[
                postprocess_task(task, meshes[i])
                for i, (task, _) in enumerate(valid_tasks)
            ])
            
            if self.low_vram_mode:
                torch.cuda.empty_cache()
            
            logger.info(f"[Batch] All {len(valid_tasks)} tasks completed")
            
        except Exception as e:
            logger.error(f"[Batch] Batch processing failed: {e}")
            import traceback
            traceback.print_exc()
            for task in batch_tasks:
                if not task.future.done():
                    task.future.set_exception(e)
            
    def get_queue_length(self):
        """
        Get the current queue length for model processing.
        
        Returns:
            int: Number of tasks in the queue
        """
        if self.model_semaphore is None:
            return 0
        else:
            return (self.model_semaphore._value if hasattr(self.model_semaphore, '_value') else 0) + \
                   (len(self.model_semaphore._waiters) if hasattr(self.model_semaphore, '_waiters') and self.model_semaphore._waiters is not None else 0)

    def get_status(self):
        """
        Get the current status of the worker.
        
        Returns:
            dict: Status information including speed and queue length
        """
        return {
            "speed": 1,
            "queue_length": self.get_queue_length(),
        }
    
    def get_load(self):
        """
        Get the current load (number of active tasks) of the worker.
        
        Returns:
            int: Number of currently processing tasks
        """
        return self.current_tasks

    async def generate(self, uid, params):
        """
        Submit a generation task to the batch processing queue.
        
        Args:
            uid: Unique identifier for this generation task
            params (dict): Generation parameters including image and options
            
        Returns:
            tuple: (file_path, uid) - Path to generated file and task ID
        """
        # Start batch processor loop if not already started
        # Note: This method is async, so we're guaranteed to have an event loop
        if not self._batch_processor_started:
            self.batch_processor_task = asyncio.create_task(self._batch_processor_loop())
            self._batch_processor_started = True
            logger.info(f"[Worker {self.worker_id}] Batch processor started")
        
        # Create task and future
        future = asyncio.Future()
        task = BatchTask(
            uid=str(uid),
            params=params,
            future=future,
            submit_time=time.time()
        )
        
        # Add to queue
        async with self.queue_lock:
            self.task_queue.append(task)
            logger.info(f"[Task {uid}] Added to queue (queue length: {len(self.task_queue)})")
        
        # Increment task counter when task is submitted
        async with self._tasks_lock:
            self.current_tasks += 1
            logger.debug(f"[Worker {self.worker_id}] Task count: {self.current_tasks}")
        
        # Initialize status
        if self.status_callback:
            await self.status_callback(uid, 'pending')
        
        # Wait for result
        try:
            result = await future
            return result
        except Exception as e:
            logger.error(f"[Task {uid}] Failed: {e}")
            raise
        finally:
            # Decrement task counter when task completes (success or failure)
            async with self._tasks_lock:
                self.current_tasks = max(0, self.current_tasks - 1)
                logger.debug(f"[Worker {self.worker_id}] Task count: {self.current_tasks}")