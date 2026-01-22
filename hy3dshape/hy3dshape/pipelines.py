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

import copy
import importlib
import inspect
import os
from typing import List, Optional, Union

import numpy as np
import torch
import torch.serialization
import trimesh
import yaml
from PIL import Image
from diffusers.utils.torch_utils import randn_tensor
from diffusers.utils.import_utils import is_accelerate_version, is_accelerate_available
from tqdm import tqdm

from .models.autoencoders import ShapeVAE
from .models.autoencoders import SurfaceExtractors
from .utils import logger, synchronize_timer, smart_load_model


def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    """
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


@synchronize_timer('Export to trimesh')
def export_to_trimesh(mesh_output):
    if isinstance(mesh_output, list):
        outputs = []
        for mesh in mesh_output:
            if mesh is None:
                outputs.append(None)
            else:
                mesh.mesh_f = mesh.mesh_f[:, ::-1]
                mesh_output = trimesh.Trimesh(mesh.mesh_v, mesh.mesh_f)
                outputs.append(mesh_output)
        return outputs
    else:
        mesh_output.mesh_f = mesh_output.mesh_f[:, ::-1]
        mesh_output = trimesh.Trimesh(mesh_output.mesh_v, mesh_output.mesh_f)
        return mesh_output


def get_obj_from_str(string, reload=False):
    # 修复模块路径：将 hy3dgen 替换为 hy3dshape
    if 'hy3dgen' in string:
        string = string.replace('hy3dgen.shapegen', 'hy3dshape')
        string = string.replace('hy3dgen', 'hy3dshape')
    
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config, **kwargs):
    if "target" not in config:
        raise KeyError("Expected key `target` to instantiate.")
    
    cls = get_obj_from_str(config["target"])
    
    # Support from_pretrained parameter (like in hy3dshape.utils.misc.instantiate_from_config)
    if config.get("from_pretrained", None):
        from_pretrained_kwargs = {
            'use_safetensors': config.get('use_safetensors', False),
            'variant': config.get('variant', 'fp16')
        }
        if 'subfolder' in config:
            from_pretrained_kwargs['subfolder'] = config['subfolder']
        
        # Merge params into from_pretrained_kwargs
        params = config.get("params", dict())
        from_pretrained_kwargs.update(params)
        from_pretrained_kwargs.update(kwargs)
        
        return cls.from_pretrained(
            config["from_pretrained"], 
            **from_pretrained_kwargs
        )
    
    params = config.get("params", dict())
    kwargs.update(params)
    instance = cls(**kwargs)
    return instance


class Hunyuan3DDiTPipeline:
    model_cpu_offload_seq = "conditioner->model->vae"
    _exclude_from_cpu_offload = []

    @classmethod
    @synchronize_timer('Hunyuan3DDiTPipeline Model Loading')
    def from_single_file(
        cls,
        ckpt_path,
        config_path,
        device='cuda',
        dtype=torch.float16,
        use_safetensors=None,
        lora_path=None,
        **kwargs,
    ):
        """
        Load pipeline from checkpoint file.
        
        Args:
            ckpt_path: Path to checkpoint file (Lightning or inference format)
            config_path: Path to config.yaml file
            device: Device to load model on
            dtype: Data type for model
            use_safetensors: Whether to use safetensors format
            lora_path: Optional path to LoRA weights file (separate from main checkpoint)
            **kwargs: Additional arguments
        """
        # load config
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # load ckpt
        if use_safetensors:
            ckpt_path = ckpt_path.replace('.ckpt', '.safetensors')
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Model file {ckpt_path} not found")
        logger.info(f"Loading model from {ckpt_path}")

        if use_safetensors:
            # parse safetensors
            import safetensors.torch
            safetensors_ckpt = safetensors.torch.load_file(ckpt_path, device='cpu')
            ckpt = {}
            for key, value in safetensors_ckpt.items():
                model_name = key.split('.')[0]
                new_key = key[len(model_name) + 1:]
                if model_name not in ckpt:
                    ckpt[model_name] = {}
                ckpt[model_name][new_key] = value
        else:
            # 尝试使用 weights_only=True 加载（更安全）
            try:
                ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=True)
            except Exception as e:
                # 如果 weights_only=True 失败（可能包含 PosixPath 等对象），尝试添加安全全局变量
                error_str = str(e)
                if 'PosixPath' in error_str or 'pathlib' in error_str:
                    logger.warning(f"Checkpoint 包含 pathlib.PosixPath 对象，尝试添加安全全局变量...")
                    try:
                        from pathlib import PosixPath
                        # torch.serialization 已在文件顶部导入
                        torch.serialization.add_safe_globals([PosixPath])
                        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=True)
                        logger.info("✅ 成功使用安全全局变量加载 checkpoint")
                    except Exception as e2:
                        # 如果还是失败，对于 Lightning checkpoint（用户自己的训练结果），使用 weights_only=False
                        logger.warning(f"使用安全全局变量仍失败: {e2}")
                        logger.warning("对于 Lightning checkpoint（训练结果），使用 weights_only=False 加载（这是安全的）")
                        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
                else:
                    # 其他错误，也尝试使用 weights_only=False
                    logger.warning(f"weights_only=True 加载失败: {e}")
                    logger.warning("尝试使用 weights_only=False 加载（仅对可信的 checkpoint 使用）")
                    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        
        # 检测 checkpoint 格式：Lightning 格式包含 'state_dict'，推理格式包含 'model', 'vae', 'conditioner'
        is_lightning_checkpoint = isinstance(ckpt, dict) and 'state_dict' in ckpt
        is_inference_checkpoint = isinstance(ckpt, dict) and 'model' in ckpt and 'vae' in ckpt
        
        if is_lightning_checkpoint:
            logger.info("✅ 检测到 Lightning checkpoint 格式")
            # Lightning checkpoint: 需要先加载预训练模型，然后加载权重
            return cls._load_from_lightning_checkpoint(
                ckpt, config, device, dtype, lora_path, **kwargs
            )
        elif is_inference_checkpoint:
            logger.info("✅ 检测到推理格式 checkpoint")
            # 推理格式 checkpoint: 直接加载
            return cls._load_from_inference_checkpoint(
                ckpt, config, device, dtype, lora_path, **kwargs
            )
        else:
            raise ValueError(f"无法识别 checkpoint 格式: {ckpt_path}\n"
                           f"Lightning checkpoint 应包含 'state_dict' 键\n"
                           f"推理格式 checkpoint 应包含 'model', 'vae', 'conditioner' 键")
    
    @classmethod
    def _load_from_lightning_checkpoint(
        cls,
        ckpt,
        config,
        device='cuda',
        dtype=torch.float16,
        lora_path=None,
        **kwargs,
    ):
        """从 Lightning checkpoint 加载模型"""
        state_dict = ckpt['state_dict']
        
        # 修复配置文件中的模块路径
        def fix_config_module_path(cfg, depth=0):
            if depth > 10:
                return
            if isinstance(cfg, dict):
                if 'target' in cfg:
                    original_target = cfg['target']
                    if 'hy3dgen' in original_target:
                        cfg['target'] = original_target.replace('hy3dgen.shapegen', 'hy3dshape')
                        cfg['target'] = cfg['target'].replace('hy3dgen', 'hy3dshape')
                        logger.info(f"修复模块路径: {original_target} -> {cfg['target']}")
                for key, value in cfg.items():
                    if isinstance(value, (dict, list)):
                        fix_config_module_path(value, depth + 1)
            elif isinstance(cfg, list):
                for item in cfg:
                    if isinstance(item, (dict, list)):
                        fix_config_module_path(item, depth + 1)
        
        fix_config_module_path(config)
        
        # Lightning checkpoint 需要先加载预训练模型
        # 检查 model 配置是否有 from_pretrained
        model_config = config.get('model', {})
        if 'params' in model_config:
            denoiser_cfg = model_config['params'].get('denoiser_cfg', {})
            if denoiser_cfg.get('from_pretrained'):
                logger.info("Lightning checkpoint: 先加载预训练模型，然后加载 checkpoint 权重")
                # 先加载预训练模型
                pipeline = cls.from_pretrained(
                    denoiser_cfg['from_pretrained'],
                    subfolder=denoiser_cfg.get('subfolder', 'hunyuan3d-dit-v2-1'),
                    use_safetensors=denoiser_cfg.get('use_safetensors', False),
                    device=device,
                    dtype=dtype,
                )
            else:
                # 没有 from_pretrained，直接实例化
                pipeline = cls._instantiate_from_config(config, device, dtype, **kwargs)
        else:
            pipeline = cls._instantiate_from_config(config, device, dtype, **kwargs)
        
        # 从 Lightning checkpoint 加载权重
        logger.info("正在从 Lightning checkpoint 加载权重...")
        
        # 加载 model (DiT) 权重
        if hasattr(pipeline, 'model') and pipeline.model is not None:
            model_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith('model.'):
                    new_key = key[6:]  # 去掉 'model.' 前缀
                    model_state_dict[new_key] = value
            if model_state_dict:
                missing, unexpected = pipeline.model.load_state_dict(model_state_dict, strict=False)
                logger.info(f"✅ DiT 模型权重加载完成: {len(model_state_dict)} 个权重")
                if missing:
                    logger.warning(f"  - Missing keys: {len(missing)}")
                if unexpected:
                    logger.warning(f"  - Unexpected keys: {len(unexpected)}")
        
        # 加载 VAE 权重
        if hasattr(pipeline, 'vae') and pipeline.vae is not None:
            vae_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith('first_stage_model.'):
                    new_key = key[19:]  # 去掉 'first_stage_model.' 前缀
                    vae_state_dict[new_key] = value
            if vae_state_dict:
                missing, unexpected = pipeline.vae.load_state_dict(vae_state_dict, strict=False)
                logger.info(f"✅ VAE 权重加载完成: {len(vae_state_dict)} 个权重")
                if missing:
                    logger.warning(f"  - Missing keys: {len(missing)}")
                if unexpected:
                    logger.warning(f"  - Unexpected keys: {len(unexpected)}")
        
        # 加载 Conditioner 权重
        if hasattr(pipeline, 'conditioner') and pipeline.conditioner is not None:
            conditioner_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith('cond_stage_model.'):
                    new_key = key[18:]  # 去掉 'cond_stage_model.' 前缀
                    conditioner_state_dict[new_key] = value
            if conditioner_state_dict:
                missing, unexpected = pipeline.conditioner.load_state_dict(conditioner_state_dict, strict=False)
                logger.info(f"✅ Conditioner 权重加载完成: {len(conditioner_state_dict)} 个权重")
                if missing:
                    logger.warning(f"  - Missing keys: {len(missing)}")
                if unexpected:
                    logger.warning(f"  - Unexpected keys: {len(unexpected)}")
        
        # 加载 ControlNet 权重（如果存在）
        if hasattr(pipeline, 'controlnet') and pipeline.controlnet is not None:
            controlnet_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith('controlnet.'):
                    new_key = key[11:]  # 去掉 'controlnet.' 前缀
                    controlnet_state_dict[new_key] = value
            if controlnet_state_dict:
                missing, unexpected = pipeline.controlnet.load_state_dict(controlnet_state_dict, strict=False)
                logger.info(f"✅ ControlNet 权重加载完成: {len(controlnet_state_dict)} 个权重")
                if missing:
                    logger.warning(f"  - Missing keys: {len(missing)}")
                if unexpected:
                    logger.warning(f"  - Unexpected keys: {len(unexpected)}")
        
        # 如果提供了 LoRA 路径，加载 LoRA 权重
        if lora_path and os.path.exists(lora_path):
            cls._load_lora_weights(pipeline, lora_path, device, dtype)
        
        return pipeline
    
    @classmethod
    def _load_from_inference_checkpoint(
        cls,
        ckpt,
        config,
        device='cuda',
        dtype=torch.float16,
        lora_path=None,
        **kwargs,
    ):
        """从推理格式 checkpoint 加载模型"""
        # 修复配置文件中的模块路径
        def fix_config_module_path(cfg, depth=0):
            if depth > 10:
                return
            if isinstance(cfg, dict):
                if 'target' in cfg:
                    original_target = cfg['target']
                    if 'hy3dgen' in original_target:
                        cfg['target'] = original_target.replace('hy3dgen.shapegen', 'hy3dshape')
                        cfg['target'] = cfg['target'].replace('hy3dgen', 'hy3dshape')
                        logger.info(f"修复模块路径: {original_target} -> {cfg['target']}")
                for key, value in cfg.items():
                    if isinstance(value, (dict, list)):
                        fix_config_module_path(value, depth + 1)
            elif isinstance(cfg, list):
                for item in cfg:
                    if isinstance(item, (dict, list)):
                        fix_config_module_path(item, depth + 1)
        
        fix_config_module_path(config)
        
        # 对于推理格式 checkpoint，移除 from_pretrained，直接使用 checkpoint 中的权重
        def remove_from_pretrained(cfg, depth=0, path=""):
            if depth > 10:
                return 0
            removed_count = 0
            if isinstance(cfg, dict):
                if 'from_pretrained' in cfg:
                    original_from_pretrained = cfg.pop('from_pretrained')
                    logger.info(f"推理格式 checkpoint: 移除 {path}.from_pretrained = {original_from_pretrained}")
                    removed_count += 1
                for key, value in cfg.items():
                    if isinstance(value, (dict, list)):
                        new_path = f"{path}.{key}" if path else key
                        removed_count += remove_from_pretrained(value, depth + 1, new_path)
            elif isinstance(cfg, list):
                for idx, item in enumerate(cfg):
                    if isinstance(item, (dict, list)):
                        new_path = f"{path}[{idx}]" if path else f"[{idx}]"
                        removed_count += remove_from_pretrained(item, depth + 1, new_path)
            return removed_count
        
        removed = remove_from_pretrained(config)
        if removed > 0:
            logger.info(f"✅ 已移除 {removed} 个 from_pretrained 配置，将直接从 checkpoint 加载完整模型权重")
        
        # 实例化 pipeline
        pipeline = cls._instantiate_from_config(config, device, dtype, **kwargs)
        
        # 加载权重
        logger.info("正在从推理格式 checkpoint 加载权重...")
        
        # 加载 model (DiT) 权重
        if 'model' in ckpt and hasattr(pipeline, 'model') and pipeline.model is not None:
            missing, unexpected = pipeline.model.load_state_dict(ckpt['model'], strict=False)
            logger.info(f"✅ DiT 模型权重加载完成: {len(ckpt['model'])} 个权重")
            if missing:
                logger.warning(f"  - Missing keys: {len(missing)}")
            if unexpected:
                logger.warning(f"  - Unexpected keys: {len(unexpected)}")
        
        # 加载 VAE 权重
        if 'vae' in ckpt and hasattr(pipeline, 'vae') and pipeline.vae is not None:
            missing, unexpected = pipeline.vae.load_state_dict(ckpt['vae'], strict=False)
            logger.info(f"✅ VAE 权重加载完成: {len(ckpt['vae'])} 个权重")
            if missing:
                logger.warning(f"  - Missing keys: {len(missing)}")
            if unexpected:
                logger.warning(f"  - Unexpected keys: {len(unexpected)}")
        
        # 加载 Conditioner 权重
        if 'conditioner' in ckpt and hasattr(pipeline, 'conditioner') and pipeline.conditioner is not None:
            missing, unexpected = pipeline.conditioner.load_state_dict(ckpt['conditioner'], strict=False)
            logger.info(f"✅ Conditioner 权重加载完成: {len(ckpt['conditioner'])} 个权重")
            if missing:
                logger.warning(f"  - Missing keys: {len(missing)}")
            if unexpected:
                logger.warning(f"  - Unexpected keys: {len(unexpected)}")
        
        # 加载 ControlNet 权重（如果存在）
        if 'controlnet' in ckpt and hasattr(pipeline, 'controlnet') and pipeline.controlnet is not None:
            missing, unexpected = pipeline.controlnet.load_state_dict(ckpt['controlnet'], strict=False)
            logger.info(f"✅ ControlNet 权重加载完成: {len(ckpt['controlnet'])} 个权重")
            if missing:
                logger.warning(f"  - Missing keys: {len(missing)}")
            if unexpected:
                logger.warning(f"  - Unexpected keys: {len(unexpected)}")
        
        # 如果提供了 LoRA 路径，加载 LoRA 权重
        if lora_path and os.path.exists(lora_path):
            cls._load_lora_weights(pipeline, lora_path, device, dtype)
        
        return pipeline
    
    @classmethod
    def _instantiate_from_config(cls, config, device, dtype, **kwargs):
        """从配置实例化 pipeline"""
        # 修复配置中的模块路径
        def fix_config_module_path(cfg, depth=0):
            if depth > 10:
                return
            if isinstance(cfg, dict):
                if 'target' in cfg:
                    original_target = cfg['target']
                    if 'hy3dgen' in original_target:
                        cfg['target'] = original_target.replace('hy3dgen.shapegen', 'hy3dshape')
                        cfg['target'] = cfg['target'].replace('hy3dgen', 'hy3dshape')
                for key, value in cfg.items():
                    if isinstance(value, (dict, list)):
                        fix_config_module_path(value, depth + 1)
            elif isinstance(cfg, list):
                for item in cfg:
                    if isinstance(item, (dict, list)):
                        fix_config_module_path(item, depth + 1)
        
        fix_config_module_path(config)
        
        # 实例化各个组件
        model = instantiate_from_config(config['model'])
        
        # VAE
        vae_config = config['vae']
        params = vae_config.get("params", dict())
        has_from_pretrained = vae_config.get("from_pretrained", None) is not None
        required_params = ['num_latents', 'embed_dim', 'width', 'heads', 'num_decoder_layers']
        has_complete_params = params and len(params) > 0 and all(p in params for p in required_params)
        
        if has_complete_params:
            vae = instantiate_from_config(vae_config)
        elif has_from_pretrained:
            from_pretrained_kwargs = {
                'use_safetensors': vae_config.get('use_safetensors', False),
                'variant': vae_config.get('variant', 'fp16'),
                'device': device,
                'dtype': dtype,
            }
            if 'subfolder' in vae_config:
                from_pretrained_kwargs['subfolder'] = vae_config['subfolder']
            from_pretrained_kwargs.update(params)
            cls_vae = get_obj_from_str(vae_config["target"])
            vae = cls_vae.from_pretrained(
                vae_config["from_pretrained"], 
                **from_pretrained_kwargs
            )
        else:
            vae = instantiate_from_config(vae_config)
        
        # Conditioner
        conditioner_config = config['conditioner']
        conditioner_params = conditioner_config.get("params", dict())
        if 'main_image_encoder' not in conditioner_params:
            logger.warning("Conditioner config missing 'main_image_encoder', attempting to use default DinoImageEncoderMV...")
            if 'params' not in conditioner_config:
                conditioner_config['params'] = {}
            conditioner_config['params']['main_image_encoder'] = {
                'type': 'DinoImageEncoderMV',
                'kwargs': {
                    'version': 'facebook/dinov2-large',
                    'image_size': 518,
                    'use_cls_token': True,
                    'view_num': 4
                }
            }
        conditioner = instantiate_from_config(conditioner_config)
        
        # Image processor
        image_processor = instantiate_from_config(config['image_processor'])
        
        # Scheduler
        scheduler = instantiate_from_config(config['scheduler'])
        
        model_kwargs = dict(
            vae=vae,
            model=model,
            scheduler=scheduler,
            conditioner=conditioner,
            image_processor=image_processor,
            device=device,
            dtype=dtype,
        )
        model_kwargs.update(kwargs)
        
        return cls(**model_kwargs)
    
    @classmethod
    def _load_lora_weights(cls, pipeline, lora_path, device='cuda', dtype=torch.float16):
        """从单独路径加载 LoRA 权重"""
        import torch
        import os
        
        logger.info(f"正在从单独路径加载 LoRA 权重: {lora_path}")
        
        if not os.path.exists(lora_path):
            logger.warning(f"LoRA 路径不存在: {lora_path}")
            return
        
        try:
            # 检查是否是 Lightning checkpoint 格式
            if lora_path.endswith('.ckpt'):
                ckpt = torch.load(lora_path, map_location='cpu')
                if 'state_dict' in ckpt:
                    state_dict = ckpt['state_dict']
                    # 提取 LoRA 权重
                    lora_state_dict = {}
                    for key, value in state_dict.items():
                        if 'lora' in key.lower():
                            # 如果是 model.lora_xxx 格式，去掉 model. 前缀
                            if key.startswith('model.'):
                                new_key = key[6:]
                            else:
                                new_key = key
                            lora_state_dict[new_key] = value
                    
                    if lora_state_dict:
                        # 检查是否需要应用 LoRA 配置
                        if hasattr(pipeline, 'model') and pipeline.model is not None:
                            # 检查模型是否已经是 PeftModel
                            from peft import PeftModel, LoraConfig, get_peft_model
                            
                            if not isinstance(pipeline.model, PeftModel):
                                # 如果不是 PeftModel，先应用 LoRA 配置
                                logger.info("模型不是 PeftModel，正在应用 LoRA 配置...")
                                lora_config = LoraConfig(
                                    r=8,
                                    lora_alpha=8,
                                    target_modules=["to_q", "to_k", "to_v", "to_out.0"],
                                    lora_dropout=0.0,
                                )
                                pipeline.model = get_peft_model(pipeline.model, lora_config)
                                logger.info("✅ LoRA 配置已应用")
                            
                            # 加载 LoRA 权重
                            missing, unexpected = pipeline.model.load_state_dict(lora_state_dict, strict=False)
                            logger.info(f"✅ LoRA 权重加载完成: {len(lora_state_dict)} 个权重")
                            if missing:
                                logger.warning(f"  - Missing keys: {len(missing)}")
                            if unexpected:
                                logger.warning(f"  - Unexpected keys: {len(unexpected)}")
                        else:
                            logger.warning("Pipeline 中没有 model，无法加载 LoRA 权重")
                    else:
                        logger.warning(f"在 {lora_path} 中未找到 LoRA 权重")
                else:
                    logger.warning(f"{lora_path} 不是有效的 Lightning checkpoint 格式")
            else:
                # 尝试作为标准 PEFT 格式加载
                try:
                    from peft import PeftModel
                    if hasattr(pipeline, 'model') and pipeline.model is not None:
                        pipeline.model = PeftModel.from_pretrained(pipeline.model, lora_path)
                        logger.info(f"✅ 从 PEFT 格式加载 LoRA 权重: {lora_path}")
                    else:
                        logger.warning("Pipeline 中没有 model，无法加载 LoRA 权重")
                except Exception as e:
                    logger.error(f"无法从 {lora_path} 加载 LoRA 权重: {e}")
        except Exception as e:
            logger.error(f"加载 LoRA 权重时出错: {e}")
            import traceback
            traceback.print_exc()

    @classmethod
    def from_pretrained(
        cls,
        model_path,
        device='cuda',
        dtype=torch.float16,
        use_safetensors=False,
        variant='fp16',
        subfolder='hunyuan3d-dit-v2-1',
        **kwargs,
    ):
        kwargs['from_pretrained_kwargs'] = dict(
            model_path=model_path,
            subfolder=subfolder,
            use_safetensors=use_safetensors,
            variant=variant,
            dtype=dtype,
            device=device,
        )
        config_path, ckpt_path = smart_load_model(
            model_path,
            subfolder=subfolder,
            use_safetensors=use_safetensors,
            variant=variant
        )
        return cls.from_single_file(
            ckpt_path,
            config_path,
            device=device,
            dtype=dtype,
            use_safetensors=use_safetensors,
            **kwargs
        )

    def __init__(
        self,
        vae,
        model,
        scheduler,
        conditioner,
        image_processor,
        device='cuda',
        dtype=torch.float16,
        **kwargs
    ):
        self.vae = vae
        self.model = model
        self.scheduler = scheduler
        self.conditioner = conditioner
        self.image_processor = image_processor
        self.kwargs = kwargs
        self.to(device, dtype)

    def compile(self):
        self.vae = torch.compile(self.vae)
        self.model = torch.compile(self.model)
        self.conditioner = torch.compile(self.conditioner)

    def enable_flashvdm(
        self,
        enabled: bool = True,
        adaptive_kv_selection=True,
        topk_mode='mean',
        mc_algo='mc',
        replace_vae=True,
    ):
        if enabled:
            model_path = self.kwargs['from_pretrained_kwargs']['model_path']
            turbo_vae_mapping = {
                'Hunyuan3D-2': ('tencent/Hunyuan3D-2', 'hunyuan3d-vae-v2-0-turbo'),
                'Hunyuan3D-2mv': ('tencent/Hunyuan3D-2', 'hunyuan3d-vae-v2-0-turbo'),
                'Hunyuan3D-2mini': ('tencent/Hunyuan3D-2mini', 'hunyuan3d-vae-v2-mini-turbo'),
            }
            model_name = model_path.split('/')[-1]
            if replace_vae and model_name in turbo_vae_mapping:
                model_path, subfolder = turbo_vae_mapping[model_name]
                self.vae = ShapeVAE.from_pretrained(
                    model_path, subfolder=subfolder,
                    use_safetensors=self.kwargs['from_pretrained_kwargs']['use_safetensors'],
                    device=self.device,
                )
            self.vae.enable_flashvdm_decoder(
                enabled=enabled,
                adaptive_kv_selection=adaptive_kv_selection,
                topk_mode=topk_mode,
                mc_algo=mc_algo
            )
        else:
            model_path = self.kwargs['from_pretrained_kwargs']['model_path']
            vae_mapping = {
                'Hunyuan3D-2': ('tencent/Hunyuan3D-2', 'hunyuan3d-vae-v2-0'),
                'Hunyuan3D-2mv': ('tencent/Hunyuan3D-2', 'hunyuan3d-vae-v2-0'),
                'Hunyuan3D-2mini': ('tencent/Hunyuan3D-2mini', 'hunyuan3d-vae-v2-mini'),
            }
            model_name = model_path.split('/')[-1]
            if model_name in vae_mapping:
                model_path, subfolder = vae_mapping[model_name]
                self.vae = ShapeVAE.from_pretrained(model_path, subfolder=subfolder)
            self.vae.enable_flashvdm_decoder(enabled=False)

    def to(self, device=None, dtype=None):
        if dtype is not None:
            self.dtype = dtype
            self.vae.to(dtype=dtype)
            self.model.to(dtype=dtype)
            self.conditioner.to(dtype=dtype)
        if device is not None:
            self.device = torch.device(device)
            self.vae.to(device)
            self.model.to(device)
            self.conditioner.to(device)

    @property
    def _execution_device(self):
        r"""
        Returns the device on which the pipeline's models will be executed. After calling
        [`~DiffusionPipeline.enable_sequential_cpu_offload`] the execution device can only be inferred from
        Accelerate's module hooks.
        """
        for name, model in self.components.items():
            if not isinstance(model, torch.nn.Module) or name in self._exclude_from_cpu_offload:
                continue

            if not hasattr(model, "_hf_hook"):
                return self.device
            for module in model.modules():
                if (
                    hasattr(module, "_hf_hook")
                    and hasattr(module._hf_hook, "execution_device")
                    and module._hf_hook.execution_device is not None
                ):
                    return torch.device(module._hf_hook.execution_device)
        return self.device

    def enable_model_cpu_offload(self, gpu_id: Optional[int] = None, device: Union[torch.device, str] = "cuda"):
        r"""
        Offloads all models to CPU using accelerate, reducing memory usage with a low impact on performance. Compared
        to `enable_sequential_cpu_offload`, this method moves one whole model at a time to the GPU when its `forward`
        method is called, and the model remains in GPU until the next model runs. Memory savings are lower than with
        `enable_sequential_cpu_offload`, but performance is much better due to the iterative execution of the `unet`.

        Arguments:
            gpu_id (`int`, *optional*):
                The ID of the accelerator that shall be used in inference. If not specified, it will default to 0.
            device (`torch.Device` or `str`, *optional*, defaults to "cuda"):
                The PyTorch device type of the accelerator that shall be used in inference. If not specified, it will
                default to "cuda".
        """
        if self.model_cpu_offload_seq is None:
            raise ValueError(
                "Model CPU offload cannot be enabled because no `model_cpu_offload_seq` class attribute is set."
            )

        if is_accelerate_available() and is_accelerate_version(">=", "0.17.0.dev0"):
            from accelerate import cpu_offload_with_hook
        else:
            raise ImportError("`enable_model_cpu_offload` requires `accelerate v0.17.0` or higher.")

        torch_device = torch.device(device)
        device_index = torch_device.index

        if gpu_id is not None and device_index is not None:
            raise ValueError(
                f"You have passed both `gpu_id`={gpu_id} and an index as part of the passed device `device`={device}"
                f"Cannot pass both. Please make sure to either not define `gpu_id` or not pass the index as part of "
                f"the device: `device`={torch_device.type}"
            )

        # _offload_gpu_id should be set to passed gpu_id (or id in passed `device`)
        # or default to previously set id or default to 0
        self._offload_gpu_id = gpu_id or torch_device.index or getattr(self, "_offload_gpu_id", 0)

        device_type = torch_device.type
        device = torch.device(f"{device_type}:{self._offload_gpu_id}")

        if self.device.type != "cpu":
            self.to("cpu")
            device_mod = getattr(torch, self.device.type, None)
            if hasattr(device_mod, "empty_cache") and device_mod.is_available():
                device_mod.empty_cache()  
                # otherwise we don't see the memory savings (but they probably exist)

        all_model_components = {k: v for k, v in self.components.items() if isinstance(v, torch.nn.Module)}

        self._all_hooks = []
        hook = None
        for model_str in self.model_cpu_offload_seq.split("->"):
            model = all_model_components.pop(model_str, None)
            if not isinstance(model, torch.nn.Module):
                continue

            _, hook = cpu_offload_with_hook(model, device, prev_module_hook=hook)
            self._all_hooks.append(hook)

        # CPU offload models that are not in the seq chain unless they are explicitly excluded
        # these models will stay on CPU until maybe_free_model_hooks is called
        # some models cannot be in the seq chain because they are iteratively called, 
        # such as controlnet
        for name, model in all_model_components.items():
            if not isinstance(model, torch.nn.Module):
                continue

            if name in self._exclude_from_cpu_offload:
                model.to(device)
            else:
                _, hook = cpu_offload_with_hook(model, device)
                self._all_hooks.append(hook)

    def maybe_free_model_hooks(self):
        r"""
        Function that offloads all components, removes all model hooks that were added when using
        `enable_model_cpu_offload` and then applies them again. In case the model has not been offloaded this function
        is a no-op. Make sure to add this function to the end of the `__call__` function of your pipeline so that it
        functions correctly when applying enable_model_cpu_offload.
        """
        if not hasattr(self, "_all_hooks") or len(self._all_hooks) == 0:
            # `enable_model_cpu_offload` has not be called, so silently do nothing
            return

        for hook in self._all_hooks:
            # offload model and remove hook from model
            hook.offload()
            hook.remove()

        # make sure the model is in the same state as before calling it
        self.enable_model_cpu_offload()

    @synchronize_timer('Encode cond')
    def encode_cond(self, image, additional_cond_inputs, do_classifier_free_guidance, dual_guidance):
        bsz = image.shape[0]
        cond = self.conditioner(image=image, **additional_cond_inputs)

        if do_classifier_free_guidance:
            un_cond = self.conditioner.unconditional_embedding(bsz, **additional_cond_inputs)

            if dual_guidance:
                un_cond_drop_main = copy.deepcopy(un_cond)
                un_cond_drop_main['additional'] = cond['additional']

                def cat_recursive(a, b, c):
                    if isinstance(a, torch.Tensor):
                        return torch.cat([a, b, c], dim=0).to(self.dtype)
                    out = {}
                    for k in a.keys():
                        out[k] = cat_recursive(a[k], b[k], c[k])
                    return out

                cond = cat_recursive(cond, un_cond_drop_main, un_cond)
            else:
                def cat_recursive(a, b):
                    if isinstance(a, torch.Tensor):
                        return torch.cat([a, b], dim=0).to(self.dtype)
                    out = {}
                    for k in a.keys():
                        out[k] = cat_recursive(a[k], b[k])
                    return out

                cond = cat_recursive(cond, un_cond)
        return cond

    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def prepare_latents(self, batch_size, dtype, device, generator, latents=None):
        shape = (batch_size, *self.vae.latent_shape)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * getattr(self.scheduler, 'init_noise_sigma', 1.0)
        return latents

    def prepare_image(self, image, mask=None) -> dict:
        if isinstance(image, torch.Tensor) and isinstance(mask, torch.Tensor):
            outputs = {
                'image': image,
                'mask': mask
            }
            return outputs
            
        # Handle dictionary input for multi-view images (MVImageProcessorV2)
        if isinstance(image, dict):
            # For multi-view input, pass the entire dictionary to the image processor
            return self.image_processor(image)
        
        # Handle list of dictionaries for multi-view images
        if isinstance(image, list) and len(image) > 0 and isinstance(image[0], dict):
            outputs = []
            for img_dict in image:
                output = self.image_processor(img_dict)
                outputs.append(output)
            
            cond_input = {k: [] for k in outputs[0].keys()}
            for output in outputs:
                for key, value in output.items():
                    cond_input[key].append(value)
            for key, value in cond_input.items():
                if isinstance(value[0], torch.Tensor):
                    cond_input[key] = torch.cat(value, dim=0)
            return cond_input
            
        if isinstance(image, str) and not os.path.exists(image):
            raise FileNotFoundError(f"Couldn't find image at path {image}")

        if not isinstance(image, list):
            image = [image]

        outputs = []
        for img in image:
            output = self.image_processor(img)
            outputs.append(output)

        cond_input = {k: [] for k in outputs[0].keys()}
        for output in outputs:
            for key, value in output.items():
                cond_input[key].append(value)
        for key, value in cond_input.items():
            if isinstance(value[0], torch.Tensor):
                cond_input[key] = torch.cat(value, dim=0)

        return cond_input

    def get_guidance_scale_embedding(self, w, embedding_dim=512, dtype=torch.float32):
        """
        See https://github.com/google-research/vdm/blob/dc27b98a554f65cdc654b800da5aa1846545d41b/model_vdm.py#L298

        Args:
            timesteps (`torch.Tensor`):
                generate embedding vectors at these timesteps
            embedding_dim (`int`, *optional*, defaults to 512):
                dimension of the embeddings to generate
            dtype:
                data type of the generated embeddings

        Returns:
            `torch.FloatTensor`: Embedding vectors with shape `(len(timesteps), embedding_dim)`
        """
        assert len(w.shape) == 1
        w = w * 1000.0

        half_dim = embedding_dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=dtype) * -emb)
        emb = w.to(dtype)[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if embedding_dim % 2 == 1:  # zero pad
            emb = torch.nn.functional.pad(emb, (0, 1))
        assert emb.shape == (w.shape[0], embedding_dim)
        return emb

    def set_surface_extractor(self, mc_algo):
        if mc_algo is None:
            return
        logger.info('The parameters `mc_algo` is deprecated, and will be removed in future versions.\n'
                    'Please use: \n'
                    'from hy3dshape.models.autoencoders import SurfaceExtractors\n'
                    'pipeline.vae.surface_extractor = SurfaceExtractors[mc_algo]() instead\n')
        if mc_algo not in SurfaceExtractors.keys():
            raise ValueError(f"Unknown mc_algo {mc_algo}")
        self.vae.surface_extractor = SurfaceExtractors[mc_algo]()

    @torch.no_grad()
    def __call__(
        self,
        image: Union[str, List[str], Image.Image] = None,
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        sigmas: List[float] = None,
        eta: float = 0.0,
        guidance_scale: float = 7.5,
        dual_guidance_scale: float = 10.5,
        dual_guidance: bool = True,
        generator=None,
        box_v=1.01,
        octree_resolution=384,
        mc_level=-1 / 512,
        num_chunks=8000,
        mc_algo=None,
        output_type: Optional[str] = "trimesh",
        enable_pbar=True,
        **kwargs,
    ) -> List[List[trimesh.Trimesh]]:
        callback = kwargs.pop("callback", None)
        callback_steps = kwargs.pop("callback_steps", None)

        self.set_surface_extractor(mc_algo)

        device = self.device
        dtype = self.dtype
        do_classifier_free_guidance = guidance_scale >= 0 and \
                                      getattr(self.model, 'guidance_cond_proj_dim', None) is None
        dual_guidance = dual_guidance_scale >= 0 and dual_guidance

        if isinstance(image, torch.Tensor):
            pass
        else:
            cond_inputs = self.prepare_image(image)
            image = cond_inputs.pop('image')
        
        cond = self.encode_cond(
            image=image,
            additional_cond_inputs=cond_inputs,
            do_classifier_free_guidance=do_classifier_free_guidance,
            dual_guidance=False,
        )
        batch_size = image.shape[0]

        t_dtype = torch.long
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, timesteps, sigmas)

        latents = self.prepare_latents(batch_size, dtype, device, generator)
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        guidance_cond = None
        if getattr(self.model, 'guidance_cond_proj_dim', None) is not None:
            logger.info('Using lcm guidance scale')
            guidance_scale_tensor = torch.tensor(guidance_scale - 1).repeat(batch_size)
            guidance_cond = self.get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=self.model.guidance_cond_proj_dim
            ).to(device=device, dtype=latents.dtype)
        with synchronize_timer('Diffusion Sampling'):
            for i, t in enumerate(tqdm(timesteps, disable=not enable_pbar, desc="Diffusion Sampling:", leave=False)):
                # expand the latents if we are doing classifier free guidance
                if do_classifier_free_guidance:
                    latent_model_input = torch.cat([latents] * (3 if dual_guidance else 2))
                else:
                    latent_model_input = latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # predict the noise residual
                timestep_tensor = torch.tensor([t], dtype=t_dtype, device=device)
                timestep_tensor = timestep_tensor.expand(latent_model_input.shape[0])
                noise_pred = self.model(latent_model_input, timestep_tensor, cond, guidance_cond=guidance_cond)

                # no drop, drop clip, all drop
                if do_classifier_free_guidance:
                    if dual_guidance:
                        noise_pred_clip, noise_pred_dino, noise_pred_uncond = noise_pred.chunk(3)
                        noise_pred = (
                            noise_pred_uncond
                            + guidance_scale * (noise_pred_clip - noise_pred_dino)
                            + dual_guidance_scale * (noise_pred_dino - noise_pred_uncond)
                        )
                    else:
                        noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                outputs = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs)
                latents = outputs.prev_sample

                if callback is not None and i % callback_steps == 0:
                    step_idx = i // getattr(self.scheduler, "order", 1)
                    callback(step_idx, t, outputs)

        return self._export(
            latents,
            output_type,
            box_v, mc_level, num_chunks, octree_resolution, mc_algo,
        )

    def _export(
        self,
        latents,
        output_type='trimesh',
        box_v=1.01,
        mc_level=0.0,
        num_chunks=20000,
        octree_resolution=256,
        mc_algo='mc',
        enable_pbar=True
    ):
        if not output_type == "latent":
            latents = 1. / self.vae.scale_factor * latents
            latents = self.vae(latents)
            outputs = self.vae.latents2mesh(
                latents,
                bounds=box_v,
                mc_level=mc_level,
                num_chunks=num_chunks,
                octree_resolution=octree_resolution,
                mc_algo=mc_algo,
                enable_pbar=enable_pbar,
            )
        else:
            outputs = latents

        if output_type == 'trimesh':
            outputs = export_to_trimesh(outputs)

        return outputs


class Hunyuan3DDiTFlowMatchingPipeline(Hunyuan3DDiTPipeline):

    @torch.inference_mode()
    def __call__(
        self,
        image: Union[str, List[str], Image.Image, dict, List[dict], torch.Tensor] = None,
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        sigmas: List[float] = None,
        eta: float = 0.0,
        guidance_scale: float = 5.0,
        generator=None,
        box_v=1.01,
        octree_resolution=384,
        mc_level=0.0,
        mc_algo=None,
        num_chunks=8000,
        output_type: Optional[str] = "trimesh",
        enable_pbar=True,
        mask = None,
        **kwargs,
    ) -> List[List[trimesh.Trimesh]]:
        callback = kwargs.pop("callback", None)
        callback_steps = kwargs.pop("callback_steps", None)
        
        # 获取深度条件相关参数
        controlnet = kwargs.pop("controlnet", None)
        depth = kwargs.pop("depth", None)
        
        # 获取法线图相关参数
        normal = kwargs.pop("normal", None)
        normal_mask = kwargs.pop("normal_mask", None)

        self.set_surface_extractor(mc_algo)

        device = self.device
        dtype = self.dtype
        do_classifier_free_guidance = guidance_scale >= 0 and not (
            hasattr(self.model, 'guidance_embed') and
            self.model.guidance_embed is True
        )

        # print('image', type(image), 'mask', type(mask))
        cond_inputs = self.prepare_image(image, mask)
        image = cond_inputs.pop('image')
        
        cond = self.encode_cond(
            image=image,
            additional_cond_inputs=cond_inputs,
            do_classifier_free_guidance=do_classifier_free_guidance,
            dual_guidance=False,
        )
        
        # 如果有法线图，处理法线图条件并拼接token
        if normal is not None:
            print(f"[Pipeline] 处理法线图，形状: {normal.shape}")
            normal = normal.to(device).to(dtype)
            
            # 法线图需要与RGB图相同的处理流程
            # normal的形状应该是 (B, num_views, 3, H, W)
            # 需要调整mask的格式（如果有）
            normal_mask_tensor = None
            if normal_mask is not None:
                normal_mask_tensor = normal_mask.to(device).to(dtype)
            
            # 通过prepare_image处理normal（与RGB图相同的预处理流程）
            normal_cond_inputs = self.prepare_image(normal, normal_mask_tensor)
            normal_image = normal_cond_inputs.pop('image') if 'image' in normal_cond_inputs else normal
            
            # 编码法线图（使用与RGB图相同的编码器）
            # 注意：如果启用了classifier-free guidance，需要确保normal_cond的batch size与cond一致
            normal_cond = self.encode_cond(
                image=normal_image,
                additional_cond_inputs=normal_cond_inputs,
                do_classifier_free_guidance=do_classifier_free_guidance,
                dual_guidance=False,
            )
            
            # 将法线图的token拼接到RGB图的token后面（与训练时逻辑一致）
            for key in cond:
                if isinstance(cond[key], torch.Tensor):
                    if key in normal_cond and isinstance(normal_cond[key], torch.Tensor):
                        # 检查batch size是否匹配
                        if cond[key].shape[0] != normal_cond[key].shape[0]:
                            print(f"[Pipeline] 警告: batch size不匹配，cond[{key}]: {cond[key].shape[0]}, normal_cond[{key}]: {normal_cond[key].shape[0]}")
                            # 如果cond的batch size更大（classifier-free guidance），需要扩展normal_cond
                            if cond[key].shape[0] == 2 * normal_cond[key].shape[0]:
                                # 重复normal_cond以匹配batch size
                                normal_cond[key] = torch.cat([normal_cond[key], normal_cond[key]], dim=0)
                                print(f"[Pipeline] 已扩展normal_cond[{key}]的batch size到: {normal_cond[key].shape[0]}")
                            else:
                                raise RuntimeError(f"无法匹配batch size: cond[{key}].shape[0]={cond[key].shape[0]}, normal_cond[{key}].shape[0]={normal_cond[key].shape[0]}")
                        
                        # 在序列维度拼接 (dim=1)
                        rgb_shape_before = cond[key].shape[1]
                        cond[key] = torch.cat([cond[key], normal_cond[key]], dim=1)
                        print(f"[Pipeline] 法线图token已拼接，key={key}, RGB tokens={rgb_shape_before}, Normal tokens={normal_cond[key].shape[1]}, 总tokens={cond[key].shape[1]}, batch_size={cond[key].shape[0]}")
        
        # 如果有深度图和controlnet，处理深度条件并注入到cond中
        if controlnet is not None and depth is not None:
            print(f"[Pipeline] 使用ControlNet处理深度图，形状: {depth.shape}")
            depth = depth.to(device).to(dtype)
            depth_features = controlnet(depth)  # (B, 1, hidden_dim) 或 (B, hidden_dim)
            
            # 将深度特征注入到条件中
            if 'additional' not in cond:
                cond['additional'] = {}
            cond['additional']['depth'] = depth_features
            print(f"[Pipeline] 深度特征已注入，形状: {depth_features.shape}")

        batch_size = image.shape[0]

        # 5. Prepare timesteps
        # NOTE: this is slightly different from common usage, we start from 0.
        sigmas = np.linspace(0, 1, num_inference_steps) if sigmas is None else sigmas
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            sigmas=sigmas,
        )
        latents = self.prepare_latents(batch_size, dtype, device, generator)

        guidance = None
        if hasattr(self.model, 'guidance_embed') and \
            self.model.guidance_embed is True:
            guidance = torch.tensor([guidance_scale] * batch_size, device=device, dtype=dtype)
            # logger.info(f'Using guidance embed with scale {guidance_scale}')

        with synchronize_timer('Diffusion Sampling'):
            for i, t in enumerate(tqdm(timesteps, disable=not enable_pbar, desc="Diffusion Sampling:")):
                # expand the latents if we are doing classifier free guidance
                if do_classifier_free_guidance:
                    latent_model_input = torch.cat([latents] * 2)
                else:
                    latent_model_input = latents

                # NOTE: we assume model get timesteps ranged from 0 to 1
                timestep = t.expand(latent_model_input.shape[0]).to(latents.dtype)
                timestep = timestep / self.scheduler.config.num_train_timesteps
                noise_pred = self.model(latent_model_input, timestep, cond, guidance=guidance)

                if do_classifier_free_guidance:
                    noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                outputs = self.scheduler.step(noise_pred, t, latents)
                latents = outputs.prev_sample

                if callback is not None and i % callback_steps == 0:
                    step_idx = i // getattr(self.scheduler, "order", 1)
                    callback(step_idx, t, outputs)

        return self._export(
            latents,
            output_type,
            box_v, mc_level, num_chunks, octree_resolution, mc_algo,
            enable_pbar=enable_pbar,
        )
