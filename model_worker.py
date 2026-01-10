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
from textureGenPipeline import Hunyuan3DPaintPipeline, Hunyuan3DPaintConfig
from hy3dpaint.convert_utils import create_glb_with_pbr_materials


def quick_convert_with_obj2gltf(obj_path: str, glb_path: str):
    textures = {
        'albedo': obj_path.replace('.obj', '.jpg'),
        'metallic': obj_path.replace('.obj', '_metallic.jpg'),
        'roughness': obj_path.replace('.obj', '_roughness.jpg')
        }
    create_glb_with_pbr_materials(obj_path, textures, glb_path)


def load_image_from_base64(image):
    """
    Load an image from base64 encoded string.
    
    Args:
        image (str): Base64 encoded image string
        
    Returns:
        PIL.Image: Loaded image
    """
    return Image.open(BytesIO(base64.b64decode(image)))


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
                 num_views=4):
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
        
        # Add lock for pipeline thread safety (currently unused, enable if needed)
        # If you encounter GPU race conditions, uncomment the lock usage in _generate_internal()
        self.pipeline_lock = asyncio.Lock()
        self.use_pipeline_lock = False  # Set to True if thread safety issues occur
        
        logger.info(f"Loading the model {model_path} on worker {self.worker_id} ...")
        
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
            # LoRA loading logic would go here
            # For now, we assume the pipeline is already configured correctly
        
        # Setup multi-view image processor if needed
        if enable_multiview_rgb:
            try:
                from hy3dshape.preprocessors import MVImageProcessorV2
                from hy3dshape.models.conditioner import DinoImageEncoderMV
                
                self.pipeline.image_processor = MVImageProcessorV2(size=518)
                logger.info("✅ Set MVImageProcessorV2 for multi-view RGB processing")
                
                # Get pipeline dtype
                pipeline_dtype = getattr(self.pipeline, 'dtype', torch.float16)
                if not hasattr(self.pipeline, 'dtype'):
                    # Try to infer from model
                    if hasattr(self.pipeline, 'model') and hasattr(self.pipeline.model, 'dtype'):
                        pipeline_dtype = self.pipeline.model.dtype
                    logger.info(f"Inferred pipeline dtype: {pipeline_dtype}")
                
                # Replace encoder with multi-view version if needed
                if hasattr(self.pipeline, 'conditioner') and hasattr(self.pipeline.conditioner, 'main_image_encoder'):
                    current_encoder = self.pipeline.conditioner.main_image_encoder
                    if not isinstance(current_encoder, DinoImageEncoderMV):
                        logger.info("Replacing encoder with DinoImageEncoderMV...")
                        new_encoder = DinoImageEncoderMV(
                            version='facebook/dinov2-large',
                            image_size=518,
                            use_cls_token=True,
                            view_num=num_views
                        )
                        # Copy weights if possible
                        if hasattr(current_encoder, 'model') and hasattr(new_encoder, 'model'):
                            try:
                                new_encoder.model.load_state_dict(current_encoder.model.state_dict())
                                logger.info("✅ Reused original encoder weights")
                            except Exception as copy_error:
                                logger.warning(f"⚠️ Could not reuse encoder weights: {copy_error}, using defaults")
                        
                        new_encoder = new_encoder.to(device, dtype=pipeline_dtype)
                        self.pipeline.conditioner.main_image_encoder = new_encoder
                        logger.info(f"✅ Successfully replaced with DinoImageEncoderMV (views: {num_views})")
            except Exception as e:
                logger.error(f"Failed to setup multi-view components: {e}")
                import traceback
                traceback.print_exc()
                raise RuntimeError(f"Multi-view RGB mode setup failed: {e}")
        
        # Initialize texture generation pipeline (matching demo.py)
        max_num_view = 6  # can be 6 to 9
        resolution = 512  # can be 768 or 512
        conf = Hunyuan3DPaintConfig(max_num_view, resolution)
        conf.realesrgan_ckpt_path = "hy3dpaint/ckpt/RealESRGAN_x4plus.pth"
        conf.multiview_cfg_path = "hy3dpaint/cfgs/hunyuan-paint-pbr.yaml"
        conf.custom_pipeline = "hy3dpaint/hunyuanpaintpbr"
        self.paint_pipeline = Hunyuan3DPaintPipeline(conf)
        # clean cache in save_dir
        for file in os.listdir(self.save_dir):
            os.remove(os.path.join(self.save_dir, file))
            
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

    @torch.inference_mode()
    async def generate(self, uid, params):
        """
        Generate a 3D model from the given parameters (async version).
        
        Args:
            uid: Unique identifier for this generation task
            params (dict): Generation parameters including image and options
            
        Returns:
            tuple: (file_path, uid) - Path to generated file and task ID
        """
        # Acquire semaphore to control concurrency
        if self.model_semaphore:
            async with self.model_semaphore:
                return await self._generate_internal(uid, params)
        else:
            return await self._generate_internal(uid, params)
    
    async def _generate_internal(self, uid, params):
        """
        Internal generation method that does the actual work.
        
        Args:
            uid: Unique identifier for this generation task
            params (dict): Generation parameters including image and options
            
        Returns:
            tuple: (file_path, uid) - Path to generated file and task ID
        """
        start_time = time.time()
        logger.info(f"Generating 3D model for uid: {uid}")
        
        # Update status to processing
        if self.status_callback:
            await self.status_callback(uid, 'processing')
        
        try:
            # Handle multi-view images (4 views: front, right, back, left)
            if 'image_front' in params:
                # Multi-view mode
                image = {}
                view_names = ['image_front', 'image_right', 'image_back', 'image_left']
                
                for view_name in view_names:
                    if view_name in params and params[view_name]:
                        # Load image from base64
                        img = load_image_from_base64(params[view_name])
                        # Convert to RGBA
                        img = img.convert("RGBA")
                        # Remove background if needed
                        if params.get("remove_background", True) or img.mode == "RGB":
                            loop = asyncio.get_event_loop()
                            img = await loop.run_in_executor(None, self.rembg, img)
                        
                        # Store with simplified key (front, right, back, left)
                        simple_name = view_name.replace('image_', '')
                        image[simple_name] = img
                        logger.info(f"Loaded view: {simple_name}")
                
                if not image:
                    raise ValueError("No multi-view images provided")
                    
            elif 'image' in params:
                # Single-view mode (for backward compatibility)
                image = params["image"]
                image = load_image_from_base64(image)
                # Convert to RGBA and remove background if needed
                image = image.convert("RGBA")
                if params.get("remove_background", True) or image.mode == "RGB":
                    # Run CPU-bound rembg in thread pool
                    loop = asyncio.get_event_loop()
                    image = await loop.run_in_executor(None, self.rembg, image)
            else:
                raise ValueError("No input image provided")

            # Generate mesh (run in thread pool to avoid blocking)
            try:
                loop = asyncio.get_event_loop()
                
                # Prepare pipeline parameters
                pipeline_params = {
                    'image': image,
                    'num_inference_steps': params.get('num_inference_steps', 5),
                    'guidance_scale': params.get('guidance_scale', 5.0),
                    'octree_resolution': params.get('octree_resolution', 256),
                    'num_chunks': params.get('num_chunks', 8000),
                    'output_type': 'mesh'
                }
                
                # Add seed if provided
                if 'seed' in params:
                    generator = torch.Generator()
                    generator = generator.manual_seed(int(params['seed']))
                    pipeline_params['generator'] = generator
                
                # Optional: use lock if thread safety issues occur
                if self.use_pipeline_lock:
                    async with self.pipeline_lock:
                        mesh = await loop.run_in_executor(None, lambda: self.pipeline(**pipeline_params)[0])
                else:
                    mesh = await loop.run_in_executor(None, lambda: self.pipeline(**pipeline_params)[0])
                
                logger.info("---Shape generation takes %s seconds ---" % (time.time() - start_time))
            except Exception as e:
                logger.error(f"Shape generation failed: {e}")
                if self.status_callback:
                    await self.status_callback(uid, 'error', str(e))
                raise ValueError(f"Failed to generate 3D mesh: {str(e)}")

            # Export initial mesh without texture
            initial_save_path = os.path.join(self.save_dir, f'{str(uid)}_initial.glb')
            await loop.run_in_executor(None, mesh.export, initial_save_path)
            
            # Update status to texturing
            if self.status_callback:
                await self.status_callback(uid, 'texturing')
            
            # Generate textured mesh as obj ( as in demo )
            try:
                output_mesh_path_obj = os.path.join(self.save_dir, f'{str(uid)}_texturing.obj')
                
                # For texture generation, use front view or single image
                texture_image = image.get('front', image) if isinstance(image, dict) else image
                
                # Run texture generation in thread pool
                def run_texture_gen():
                    return self.paint_pipeline(
                        mesh_path=initial_save_path,
                        image_path=texture_image,
                        output_mesh_path=output_mesh_path_obj,
                        save_glb=False            
                    )
                
                # Optional: use lock if thread safety issues occur
                if self.use_pipeline_lock:
                    async with self.pipeline_lock:
                        textured_path_obj = await loop.run_in_executor(None, run_texture_gen)
                else:
                    textured_path_obj = await loop.run_in_executor(None, run_texture_gen)
                logger.info("---Texture generation takes %s seconds ---" % (time.time() - start_time))
                logger.info(f"output_mesh_path: {output_mesh_path_obj} textured_path: {textured_path_obj}")

                # Convert textured OBJ to GLB using obj2gltf with PBR support
                logger.info("convert textured OBJ to GLB")
                glb_path_textured = os.path.join(self.save_dir, f'{str(uid)}_texturing.glb')
                await loop.run_in_executor(None, quick_convert_with_obj2gltf, textured_path_obj, glb_path_textured)
                
                # Rename to final path
                logger.info("done.")
                final_save_path = os.path.join(self.save_dir, f'{str(uid)}_textured.glb')
                await loop.run_in_executor(None, os.rename, glb_path_textured, final_save_path)
                logger.info(f"final_save_path: {final_save_path}")

            except Exception as e:
                logger.error(f"Texture generation failed: {e}")
                # Fall back to untextured mesh if texture generation fails
                final_save_path = initial_save_path
                logger.warning(f"Using untextured mesh as fallback: {final_save_path}")

            if self.low_vram_mode:
                torch.cuda.empty_cache()
            
            # Update status to completed
            if self.status_callback:
                await self.status_callback(uid, 'completed')
                
            logger.info("---Total generation takes %s seconds ---" % (time.time() - start_time))
            return final_save_path, uid
            
        except Exception as e:
            logger.error(f"Generation failed for uid {uid}: {e}")
            if self.status_callback:
                await self.status_callback(uid, 'error', str(e))
            raise 