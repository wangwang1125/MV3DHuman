"""
Model worker for Hunyuan3D API server.
"""
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
                 rgb_lora_path=None,
                 num_views=4,
                 device='cuda',
                 low_vram_mode=False,
                 worker_id=None,
                 model_semaphore=None,
                 save_dir='gradio_cache'):
        """
        Initialize the model worker for multi-view RGB reconstruction.
        
        Args:
            model_path (str): Path to the shape generation model
            subfolder (str): Subfolder containing the model files
            rgb_lora_path (str): Path to RGB LoRA checkpoint (optional)
            num_views (int): Number of views for multi-view reconstruction (default: 4)
            device (str): Device to run the model on ('cuda' or 'cpu')
            low_vram_mode (bool): Whether to use low VRAM mode
            worker_id (str): Unique identifier for this worker
            model_semaphore: Semaphore for controlling model concurrency
            save_dir (str): Directory to save generated files
        """
        self.model_path = model_path
        self.subfolder = subfolder
        self.rgb_lora_path = rgb_lora_path
        self.num_views = num_views
        self.worker_id = worker_id or str(uuid.uuid4())[:6]
        self.device = device
        self.low_vram_mode = low_vram_mode
        self.model_semaphore = model_semaphore
        self.save_dir = save_dir
        
        logger.info(f"Loading multi-view RGB model {model_path}/{subfolder} on worker {self.worker_id} ...")
        logger.info(f"Number of views: {num_views}")
        if rgb_lora_path:
            logger.info(f"RGB LoRA path: {rgb_lora_path}")

        # Initialize background remover
        self.rembg = BackgroundRemover()
        
        # Initialize multi-view shape generation pipeline
        self._load_multiview_pipeline()
        
        # Initialize texture generation pipeline (matching demo.py)
        max_num_view = 6  # can be 6 to 9
        resolution = 512  # can be 768 or 512
        conf = Hunyuan3DPaintConfig(max_num_view, resolution)
        conf.realesrgan_ckpt_path = "hy3dpaint/ckpt/RealESRGAN_x4plus.pth"
        conf.multiview_cfg_path = "hy3dpaint/cfgs/hunyuan-paint-pbr.yaml"
        conf.custom_pipeline = "hy3dpaint/hunyuanpaintpbr"
        self.paint_pipeline = Hunyuan3DPaintPipeline(conf)
        # clean cache in save_dir
        if os.path.exists(self.save_dir):
            for file in os.listdir(self.save_dir):
                file_path = os.path.join(self.save_dir, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)
            
    def _load_multiview_pipeline(self):
        """
        Load multi-view RGB reconstruction pipeline with optional LoRA weights.
        Based on gradio_app.py multi-view loading logic.
        """
        logger.info("Loading multi-view RGB reconstruction pipeline...")
        
        # Load base pipeline
        self.pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            self.model_path,
            subfolder=self.subfolder,
            use_safetensors=False,
            device=self.device,
        )
        
        # Load RGB LoRA weights if provided
        if self.rgb_lora_path and os.path.exists(self.rgb_lora_path):
            logger.info(f"Loading RGB LoRA weights from: {self.rgb_lora_path}")
            try:
                from peft import PeftModel, LoraConfig, get_peft_model
                
                # Check if it's Lightning checkpoint format (.ckpt)
                if self.rgb_lora_path.endswith('.ckpt'):
                    logger.info("Detected Lightning checkpoint format, loading LoRA weights...")
                    ckpt = torch.load(self.rgb_lora_path, map_location='cpu')
                    
                    if 'state_dict' in ckpt:
                        state_dict = ckpt['state_dict']
                        logger.info(f"Checkpoint contains {len(state_dict)} weights")
                        
                        # Analyze checkpoint content
                        lora_keys = [k for k in state_dict.keys() if 'lora' in k.lower()]
                        logger.info(f"  - LoRA parameters: {len(lora_keys)}")
                        
                        # Load LoRA weights
                        if lora_keys and hasattr(self.pipeline, 'model'):
                            try:
                                logger.info("Loading LoRA weights to DiT model...")
                                # Extract model weights (including LoRA)
                                model_state_dict = {}
                                for key, value in state_dict.items():
                                    if key.startswith('model.'):
                                        new_key = key[6:]  # Remove 'model.' prefix
                                        model_state_dict[new_key] = value
                                
                                # Apply LoRA config to base model
                                lora_config = LoraConfig(
                                    r=8,
                                    lora_alpha=8,
                                    target_modules=["to_q", "to_k", "to_v", "to_out.0"],
                                    lora_dropout=0.0,
                                )
                                self.pipeline.model = get_peft_model(self.pipeline.model, lora_config)
                                
                                # Load weights with LoRA
                                missing, unexpected = self.pipeline.model.load_state_dict(
                                    model_state_dict, strict=False)
                                logger.info(f"✅ RGB LoRA weights loaded successfully")
                                logger.info(f"  - Missing keys: {len(missing)}")
                                logger.info(f"  - Unexpected keys: {len(unexpected)}")
                                
                            except Exception as e:
                                logger.error(f"❌ RGB LoRA weights loading failed: {e}")
                                import traceback
                                traceback.print_exc()
                        
                    else:
                        logger.error("❌ Invalid checkpoint format, missing state_dict")
                
                elif os.path.isdir(self.rgb_lora_path):
                    # Standard PEFT format directory
                    logger.info("Detected PEFT directory format, using standard LoRA loading...")
                    
                    if hasattr(self.pipeline, 'model'):
                        try:
                            logger.info("Loading RGB LoRA weights to DiT model...")
                            logger.info(f"  Model type: {type(self.pipeline.model)}")
                            logger.info(f"  LoRA path: {self.rgb_lora_path}")
                            
                            # Apply LoRA to pipeline.model
                            self.pipeline.model = PeftModel.from_pretrained(
                                self.pipeline.model, self.rgb_lora_path)
                            
                            logger.info("✅ RGB LoRA weights loaded successfully")
                        except Exception as e:
                            logger.error(f"❌ RGB LoRA weights loading failed: {e}")
                            import traceback
                            traceback.print_exc()
                    else:
                        logger.error("❌ Pipeline has no model attribute")
                else:
                    logger.error(f"❌ Unsupported RGB LoRA weight format: {self.rgb_lora_path}")
                    
            except Exception as e:
                import traceback
                logger.error(f"❌ Exception occurred while loading RGB LoRA weights: {e}")
                logger.error("Detailed error:")
                traceback.print_exc()
                logger.info("Will use base RGB model")
        else:
            if self.rgb_lora_path:
                logger.warning(f"RGB LoRA path does not exist: {self.rgb_lora_path}")
            logger.info("Using base RGB model (no LoRA weights loaded)")
        
        # Replace image processor and encoder for multi-view
        self._setup_multiview_components()

    def _setup_multiview_components(self):
        """
        Replace image processor and encoder with multi-view versions.
        Based on gradio_app.py multi-view setup logic.
        """
        logger.info("Setting up multi-view components...")
        
        try:
            from hy3dshape.preprocessors import MVImageProcessorV2
            from hy3dshape.models.conditioner import DinoImageEncoderMV
            
            # Replace image processor
            self.pipeline.image_processor = MVImageProcessorV2(size=518)
            logger.info("✅ Set MVImageProcessorV2 for multi-view RGB processing")
            
            # Replace encoder in conditioner
            if hasattr(self.pipeline, 'conditioner'):
                if hasattr(self.pipeline.conditioner, 'main_image_encoder'):
                    current_encoder = self.pipeline.conditioner.main_image_encoder
                    
                    # Check if current encoder is not DinoImageEncoderMV
                    if not isinstance(current_encoder, DinoImageEncoderMV):
                        logger.info(f"Detected current encoder type: {type(current_encoder).__name__}")
                        logger.info("Replacing with DinoImageEncoderMV...")
                        
                        # Create new DinoImageEncoderMV encoder
                        new_encoder = DinoImageEncoderMV(
                            version='facebook/dinov2-large',
                            image_size=518,
                            use_cls_token=True,
                            view_num=self.num_views
                        )
                        
                        # Try to reuse weights from original encoder if possible
                        if hasattr(current_encoder, 'model') and hasattr(new_encoder, 'model'):
                            try:
                                new_encoder.model.load_state_dict(current_encoder.model.state_dict())
                                logger.info("✅ Reused original encoder model weights")
                            except Exception as e:
                                logger.warning(f"⚠️ Cannot reuse original encoder weights, using default: {e}")
                        
                        # Move new encoder to same device and dtype
                        new_encoder = new_encoder.to(self.device, dtype=self.pipeline.dtype)
                        
                        # Replace encoder
                        self.pipeline.conditioner.main_image_encoder = new_encoder
                        logger.info(f"✅ Successfully replaced with DinoImageEncoderMV (view count: {self.num_views})")
                    else:
                        logger.info("✅ Encoder is already DinoImageEncoderMV")
                else:
                    logger.warning("⚠️ Conditioner has no main_image_encoder attribute")
            else:
                logger.warning("⚠️ Pipeline has no conditioner attribute")
        
        except Exception as e:
            logger.error(f"❌ Failed to setup multi-view components: {e}")
            import traceback
            traceback.print_exc()
            raise

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
    def generate(self, uid, params):
        """
        Generate a 3D model from multi-view images.
        
        Args:
            uid: Unique identifier for this generation task
            params (dict): Generation parameters including 4 view images and options
            
        Returns:
            tuple: (file_path, uid) - Path to generated file and task ID
        """
        start_time = time.time()
        logger.info(f"Generating 3D model from multi-view images for uid: {uid}")
        
        # Parse 4 view images from base64
        images = {}
        view_names = ['front', 'right', 'back', 'left']
        
        for view in view_names:
            image_key = f'image_{view}'
            if image_key in params:
                try:
                    images[view] = load_image_from_base64(params[image_key])
                    logger.info(f"Loaded {view} view image")
                except Exception as e:
                    logger.error(f"Failed to load {view} view image: {e}")
                    raise ValueError(f"Failed to load {view} view image: {str(e)}")
            else:
                raise ValueError(f"Missing required view: {view}")
        
        if len(images) != 4:
            raise ValueError(f"Expected 4 views, got {len(images)}")
        
        # Remove background from each view if needed
        if params.get('remove_background', True):
            logger.info("Removing background from all views...")
            for view, img in images.items():
                try:
                    images[view] = self.rembg(img.convert('RGB'))
                    logger.info(f"Background removed for {view} view")
                except Exception as e:
                    logger.warning(f"Failed to remove background for {view} view: {e}")
                    # Keep original image if background removal fails
                    images[view] = img.convert('RGBA')
        else:
            # Convert all to RGBA
            for view, img in images.items():
                images[view] = img.convert('RGBA')

        # Generate mesh from multi-view images
        try:
            logger.info("Starting multi-view 3D shape generation...")
            
            # Prepare generator for reproducibility
            generator = torch.Generator()
            if 'seed' in params:
                generator = generator.manual_seed(int(params['seed']))
            
            # Call pipeline with multi-view images (dictionary format)
            mesh_outputs = self.pipeline(
                image=images,  # Dictionary format: {'front': img, 'right': img, 'back': img, 'left': img}
                num_inference_steps=params.get('num_inference_steps', 5),
                guidance_scale=params.get('guidance_scale', 5.0),
                generator=generator,
                octree_resolution=params.get('octree_resolution', 256),
                num_chunks=params.get('num_chunks', 8000),
                output_type='mesh'
            )
            
            logger.info("---Shape generation takes %s seconds ---" % (time.time() - start_time))
            
            # Extract mesh from output (handle Latent2MeshOutput object)
            from hy3dshape.pipelines import export_to_trimesh
            mesh = export_to_trimesh(mesh_outputs)[0]
            logger.info(f"Mesh extracted: {mesh.vertices.shape[0]} vertices, {mesh.faces.shape[0]} faces")
            
        except Exception as e:
            logger.error(f"Multi-view shape generation failed: {e}")
            import traceback
            traceback.print_exc()
            raise ValueError(f"Failed to generate 3D mesh from multi-view images: {str(e)}")

        # Export initial mesh without texture
        initial_save_path = os.path.join(self.save_dir, f'{str(uid)}_initial.glb')
        mesh.export(initial_save_path)
        logger.info(f"Initial mesh exported to: {initial_save_path}")
        
        # Generate textured mesh if requested
        if params.get('texture', False):
            try:
                logger.info("Starting texture generation...")
                output_mesh_path_obj = os.path.join(self.save_dir, f'{str(uid)}_texturing.obj')
                
                # Use front view image for texture generation
                front_image = images['front']
                
                textured_path_obj = self.paint_pipeline(
                    mesh_path=initial_save_path,
                    image_path=front_image,
                    output_mesh_path=output_mesh_path_obj,
                    save_glb=False            
                )
                logger.info("---Texture generation takes %s seconds ---" % (time.time() - start_time))
                logger.info(f"output_mesh_path: {output_mesh_path_obj} textured_path: {textured_path_obj}")

                # Convert textured OBJ to GLB using obj2gltf with PBR support
                logger.info("Converting textured OBJ to GLB...")
                glb_path_textured = os.path.join(self.save_dir, f'{str(uid)}_texturing.glb')
                quick_convert_with_obj2gltf(textured_path_obj, glb_path_textured)
                
                # Rename to final path
                final_save_path = os.path.join(self.save_dir, f'{str(uid)}_textured.glb')
                os.rename(glb_path_textured, final_save_path)
                logger.info(f"Textured mesh saved to: {final_save_path}")
                
            except Exception as e:
                logger.error(f"Texture generation failed: {e}")
                import traceback
                traceback.print_exc()
                # Fall back to untextured mesh if texture generation fails
                final_save_path = initial_save_path
                logger.warning(f"Using untextured mesh as fallback: {final_save_path}")
        else:
            # No texture requested, use initial mesh as final
            final_save_path = initial_save_path
            logger.info("Texture generation skipped (not requested)")

        if self.low_vram_mode:
            torch.cuda.empty_cache()
            
        logger.info("---Total generation takes %s seconds ---" % (time.time() - start_time))
        return final_save_path, uid 