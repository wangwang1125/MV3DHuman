import os
from contextlib import contextmanager
from typing import List, Tuple, Optional, Union

import torch
import torch.nn as nn
from torch.optim import lr_scheduler
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_info
from pytorch_lightning.utilities import rank_zero_only

from ...utils.ema import LitEma
from ...utils.misc import instantiate_from_config, instantiate_non_trainable_model, freeze_eval_module
from ...utils import smart_load_model



class Diffuser(pl.LightningModule):
    def __init__(
        self,
        *,
        first_stage_config,
        cond_stage_config,
        denoiser_cfg,
        scheduler_cfg,
        optimizer_cfg,
        pipeline_cfg=None,
        image_processor_cfg=None,
        lora_config=None,
        ema_config=None,
        control_net_config=None,
        control_in_channels: int = None,
        first_stage_key: str = "surface",
        cond_stage_key: str = "image",
        sample_posterior: bool = True,
        scale_by_std: bool = False,
        z_scale_factor: float = 1.0,
        ckpt_path: Optional[str] = None,
        ignore_keys: Union[Tuple[str], List[str]] = (),
        torch_compile: bool = False,
    ):
        super().__init__()
        self.first_stage_key = first_stage_key
        self.cond_stage_key = cond_stage_key
        self.sample_posterior = sample_posterior

        # ========= init optimizer config ========= #
        self.optimizer_cfg = optimizer_cfg

        # ========= init diffusion scheduler ========= #
        self.scheduler_cfg = scheduler_cfg
        self.sampler = None
        if 'transport' in scheduler_cfg:
            self.transport = instantiate_from_config(scheduler_cfg.transport)
            self.sampler = instantiate_from_config(scheduler_cfg.sampler, transport=self.transport)
            self.sample_fn = self.sampler.sample_ode(**scheduler_cfg.sampler.ode_params)

        # ========= init the model ========= #
        self.denoiser_cfg = denoiser_cfg
        self.model = instantiate_from_config(denoiser_cfg, device=None, dtype=None)
        self.cond_stage_model = instantiate_from_config(cond_stage_config)
        if not self.optimizer_cfg.get('train_image_encoder', False):
            self.cond_stage_model = freeze_eval_module(self.cond_stage_model)
            rank_zero_info("Frozen cond_stage_model in eval mode")
        
        # Log conditioner info
        rank_zero_info(f"Initialized cond_stage_model: {type(self.cond_stage_model).__name__}")
        if hasattr(self.cond_stage_model, 'main_image_encoder'):
            rank_zero_info(f"  main_image_encoder: {type(self.cond_stage_model.main_image_encoder).__name__}")
            if hasattr(self.cond_stage_model.main_image_encoder, 'view_num'):
                rank_zero_info(f"  view_num: {self.cond_stage_model.main_image_encoder.view_num}")

        # ========= load conditioner weights from pretrained checkpoint if available ========= #
        # 如果 denoiser 是从预训练模型加载的，从同一个 checkpoint 加载 conditioner 的权重
        # 直接复用预训练模型中的 conditioner，通过键名映射让 load_state_dict 自动处理
        if denoiser_cfg.get("from_pretrained", None):
            try:
                model_path = denoiser_cfg["from_pretrained"]
                subfolder = denoiser_cfg.get("subfolder", "hunyuan3d-dit-v2-mv")
                use_safetensors = denoiser_cfg.get("use_safetensors", True)
                variant = denoiser_cfg.get("variant", "fp16")
                
                _, pretrained_ckpt_path = smart_load_model(
                    model_path,
                    subfolder=subfolder,
                    use_safetensors=use_safetensors,
                    variant=variant
                )
                
                # 加载预训练 checkpoint 并映射键名：conditioner.* -> cond_stage_model.*
                if use_safetensors:
                    import safetensors.torch
                    ckpt = safetensors.torch.load_file(pretrained_ckpt_path, device='cpu')
                    # safetensors 格式：键名是扁平的 conditioner.main_image_encoder.*
                    # 映射到 cond_stage_model.main_image_encoder.*
                    mapped_state_dict = {}
                    conditioner_keys = []
                    for key, value in ckpt.items():
                        if key.startswith('conditioner.'):
                            new_key = key.replace('conditioner.', 'cond_stage_model.', 1)
                            mapped_state_dict[new_key] = value
                            conditioner_keys.append(key)
                    
                    if mapped_state_dict:
                        # 统计预训练 checkpoint 中的 conditioner 权重
                        total_pretrained_params = sum(v.numel() for v in mapped_state_dict.values())
                        rank_zero_info(f"Found {len(conditioner_keys)} conditioner keys in pretrained checkpoint")
                        rank_zero_info(f"  Total conditioner parameters in checkpoint: {total_pretrained_params:,}")
                        rank_zero_info(f"Sample conditioner keys: {conditioner_keys[:5]}")
                        
                        # 检查所有可能的 view_embed 键名变体
                        all_keys = list(ckpt.keys())
                        view_embed_variants = [
                            'view_embed',
                            'view_embedding',
                            'view_pos_embed',
                            'view_position_embedding',
                            'main_image_encoder.view_embed',
                            'conditioner.main_image_encoder.view_embed',
                        ]
                        found_view_embed_keys = []
                        for variant in view_embed_variants:
                            matching_keys = [k for k in all_keys if variant in k.lower()]
                            if matching_keys:
                                found_view_embed_keys.extend(matching_keys)
                        
                        # 检查是否包含 view_embed 权重
                        view_embed_keys = [k for k in mapped_state_dict.keys() if 'view_embed' in k]
                        if view_embed_keys:
                            rank_zero_info(f"  ✓ Found view_embed weights: {view_embed_keys}")
                        elif found_view_embed_keys:
                            rank_zero_info(f"  ⚠ Found potential view_embed keys in checkpoint (but not mapped): {found_view_embed_keys}")
                            rank_zero_info(f"     Attempting to load them...")
                            # 尝试加载这些键
                            for key in found_view_embed_keys:
                                if key.startswith('conditioner.'):
                                    new_key = key.replace('conditioner.', 'cond_stage_model.', 1)
                                    mapped_state_dict[new_key] = ckpt[key]
                                    rank_zero_info(f"     Mapped: {key} -> {new_key}")
                        else:
                            rank_zero_info(f"  ℹ INFO: No view_embed weights found in pretrained checkpoint (this is expected)")
                            rank_zero_info(f"     view_embed is a fixed positional encoding (sincos), not a trainable parameter")
                            rank_zero_info(f"     It will be deterministically initialized using sincos (same as pretrained model)")
                            rank_zero_info(f"     This is consistent with official Hunyuan3D-2 implementation")
                        
                        # 加载权重
                        missing, unexpected = self.load_state_dict(mapped_state_dict, strict=False)
                        loaded_keys = len([k for k in mapped_state_dict.keys() if k.startswith('cond_stage_model.')])
                        
                        # 统计实际加载的 conditioner 参数数量
                        conditioner_params = sum(p.numel() for p in self.cond_stage_model.parameters())
                        conditioner_buffers = sum(b.numel() for b in self.cond_stage_model.buffers())
                        rank_zero_info(f"Successfully loaded {loaded_keys} conditioner weights from pretrained checkpoint")
                        rank_zero_info(f"  Conditioner model total parameters: {conditioner_params:,}")
                        rank_zero_info(f"  Conditioner model total buffers: {conditioner_buffers:,}")
                        rank_zero_info(f"  Conditioner model total: {conditioner_params + conditioner_buffers:,}")
                        
                        # 检查 view_embed（它不参与训练，使用确定性初始化）
                        if hasattr(self.cond_stage_model, 'main_image_encoder') and hasattr(self.cond_stage_model.main_image_encoder, 'view_embed'):
                            view_embed_loaded = any('view_embed' in k for k in mapped_state_dict.keys())
                            if view_embed_loaded:
                                rank_zero_info(f"  ✓ view_embed weights loaded from checkpoint")
                            else:
                                rank_zero_info(f"  ℹ INFO: view_embed not in checkpoint (expected - it's a fixed encoding)")
                                rank_zero_info(f"     Using deterministic sincos initialization (consistent with pretrained model)")
                        
                        if missing:
                            # Missing keys 中的 model.* 是 DiT 模型的权重，这些会在 Hunyuan3DDiT.from_single_file 中加载
                            model_missing = [k for k in missing if k.startswith('model.')]
                            other_missing = [k for k in missing if not k.startswith('model.')]
                            if model_missing:
                                rank_zero_info(f"  Missing keys (DiT model weights, will be loaded separately): {len(model_missing)}")
                            if other_missing:
                                rank_zero_info(f"  Missing keys (other): {len(other_missing)} (first 5: {other_missing[:5]})")
                                # 检查 view_embed（如果 missing 是正常的，因为它是固定编码）
                                view_embed_missing = [k for k in other_missing if 'view_embed' in k]
                                if view_embed_missing:
                                    rank_zero_info(f"  ℹ INFO: view_embed keys in missing list (expected - it's a fixed encoding): {view_embed_missing}")
                        if unexpected:
                            rank_zero_info(f"  Unexpected keys: {len(unexpected)} (first 5: {unexpected[:5]})")
                    else:
                        rank_zero_info("No conditioner weights found in pretrained checkpoint")
                else:
                    # ckpt 格式：conditioner 是一个嵌套字典
                    ckpt = torch.load(pretrained_ckpt_path, map_location='cpu', weights_only=True)
                    if 'conditioner' in ckpt:
                        conditioner_state = ckpt['conditioner']
                        rank_zero_info(f"Found conditioner in pretrained checkpoint with {len(conditioner_state)} top-level keys")
                        # 映射到 cond_stage_model.*
                        mapped_state_dict = {}
                        for key, value in conditioner_state.items():
                            mapped_state_dict[f'cond_stage_model.{key}'] = value
                        
                        if mapped_state_dict:
                            missing, unexpected = self.load_state_dict(mapped_state_dict, strict=False)
                            rank_zero_info(f"Successfully loaded {len(mapped_state_dict)} conditioner weights from pretrained checkpoint")
                            if missing:
                                rank_zero_info(f"  Missing keys: {len(missing)} (first 5: {missing[:5]})")
                            if unexpected:
                                rank_zero_info(f"  Unexpected keys: {len(unexpected)} (first 5: {unexpected[:5]})")
                    else:
                        rank_zero_info("No conditioner found in pretrained checkpoint")
            except Exception as e:
                rank_zero_info(f"Failed to load conditioner weights from pretrained checkpoint: {e}")
                rank_zero_info("Will use randomly initialized conditioner weights")

        self.ckpt_path = ckpt_path
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

        # ========= config controlnet model ========= #
        self.controlnet = None
        self.control_in_channels = control_in_channels
        self.num_views = denoiser_cfg.get('params', {}).get('num_views', 1)  # Get num_views from config
        
        if control_net_config is not None:
            if self.num_views > 1:
                # Use multi-view ControlNet for multiple views
                from ..controlnet_multiview import create_multiview_depth_controlnet
                fusion_strategy = control_net_config.get('fusion_strategy', 'attention')
                out_channels = denoiser_cfg.get('params', {}).get('additional_cond_hidden_state', 768)
                
                self.controlnet = create_multiview_depth_controlnet(
                    in_channels=control_in_channels or 1,
                    num_views=self.num_views,
                    out_channels=out_channels,
                    fusion_strategy=fusion_strategy
                )
                print(f"[INFO] Created MultiViewDepthControlNet with {self.num_views} views, fusion: {fusion_strategy}")
            else:
                # Use single-view ControlNet for single view
                self.controlnet = self._create_depth_controlnet(control_in_channels or 1)
                print(f"[INFO] Created single-view DepthControlNet")
            
        # ========= config lora model ========= #
        if lora_config is not None:
            from peft import LoraConfig, get_peft_model
            loraconfig = LoraConfig(
                r=lora_config.rank,
                lora_alpha=lora_config.rank,
                target_modules=lora_config.get('target_modules')
            )
            # Apply LoRA to the main model
            self.model = get_peft_model(self.model, loraconfig)
            
            # Skip LoRA for ControlNet since it doesn't have matching target modules
            # ControlNet will be trained with full parameters instead
            if self.controlnet is not None:
                print(f"[INFO] Skipping LoRA for ControlNet - target modules {lora_config.get('target_modules')} not found")
                print(f"[INFO] ControlNet will be trained with full parameters")

        # ========= config ema model ========= #
        self.ema_config = ema_config
        if self.ema_config is not None:
            if self.ema_config.ema_model == 'DSEma':
                # from michelangelo.models.modules.ema_deepspeed import DSEma
                from ..utils.ema_deepspeed import DSEma
                self.model_ema = DSEma(self.model, decay=self.ema_config.ema_decay)
            else:
                self.model_ema = LitEma(self.model, decay=self.ema_config.ema_decay)
            #do not initilize EMA weight from ckpt path, since I need to change moe layers
            if ckpt_path is not None:
                self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

        # ========= init vae at last to prevent it is overridden by loaded ckpt ========= #
        self.first_stage_model = instantiate_non_trainable_model(first_stage_config)

        self.scale_by_std = scale_by_std
        if scale_by_std:
            self.register_buffer("z_scale_factor", torch.tensor(z_scale_factor))
        else:
            self.z_scale_factor = z_scale_factor

        # ========= init pipeline for inference ========= #
        self.image_processor_cfg = image_processor_cfg
        self.image_processor = None
        if self.image_processor_cfg is not None:
            self.image_processor = instantiate_from_config(self.image_processor_cfg)
        self.pipeline_cfg = pipeline_cfg
        from ...schedulers import FlowMatchEulerDiscreteScheduler
        scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000)
        self.pipeline = instantiate_from_config(
            pipeline_cfg,
            vae=self.first_stage_model,
            model=self.model,
            scheduler=scheduler, # self.sampler,
            conditioner=self.cond_stage_model,
            image_processor=self.image_processor,
        )

        # ========= torch compile to accelerate ========= #
        self.torch_compile = torch_compile
        if self.torch_compile:
            torch.nn.Module.compile(self.model)
            torch.nn.Module.compile(self.first_stage_model)
            torch.nn.Module.compile(self.cond_stage_model)
            print(f'*' * 100)
            print(f'Compile model for acceleration')
            print(f'*' * 100)

    @contextmanager
    def ema_scope(self, context=None):
        if self.ema_config is not None and self.ema_config.get('ema_inference', False):
            self.model_ema.store(self.model)
            self.model_ema.copy_to(self.model)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.ema_config is not None and self.ema_config.get('ema_inference', False):
                self.model_ema.restore(self.model)
                if context is not None:
                    print(f"{context}: Restored training weights")

    def init_from_ckpt(self, path, ignore_keys=()):
        ckpt = torch.load(path, map_location="cpu")
        if 'state_dict' not in ckpt:
            # deepspeed ckpt
            state_dict = {}
            for k in ckpt.keys():
                new_k = k.replace('_forward_module.', '')
                state_dict[new_k] = ckpt[k]
        else:
            state_dict = ckpt["state_dict"]

        keys = list(state_dict.keys())
        for k in keys:
            for ik in ignore_keys:
                if ik in k:
                    print("Deleting key {} from state_dict.".format(k))
                    del state_dict[k]

        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
            print(f"Unexpected Keys: {unexpected}")

    def on_load_checkpoint(self, checkpoint):
        """
        The pt_model is trained separately, so we already have access to its
        checkpoint and load it separately with `self.set_pt_model`.

        However, the PL Trainer is strict about
        checkpoint loading (not configurable), so it expects the loaded state_dict
        to match exactly the keys in the model state_dict.

        So, when loading the checkpoint, before matching keys, we add all pt_model keys
        from self.state_dict() to the checkpoint state dict, so that they match
        """
        for key in self.state_dict().keys():
            if key.startswith("model_ema") and key not in checkpoint["state_dict"]:
                checkpoint["state_dict"][key] = self.state_dict()[key]

    def configure_optimizers(self) -> Tuple[List, List]:
        lr = self.learning_rate

        params_list = []
        trainable_parameters = list(self.model.parameters())
        
        # Add ControlNet parameters if it exists
        if self.controlnet is not None:
            trainable_parameters.extend(list(self.controlnet.parameters()))
            
        params_list.append({'params': trainable_parameters, 'lr': lr})

        no_decay = ['bias', 'norm.weight', 'norm.bias', 'norm1.weight', 'norm1.bias', 'norm2.weight', 'norm2.bias']


        if self.optimizer_cfg.get('train_image_encoder', False):
            image_encoder_parameters = list(self.cond_stage_model.named_parameters())
            image_encoder_parameters_decay = [param for name, param in image_encoder_parameters if
                                              not any((no_decay_name in name) for no_decay_name in no_decay)]
            image_encoder_parameters_nodecay = [param for name, param in image_encoder_parameters if
                                                any((no_decay_name in name) for no_decay_name in no_decay)]
            # filter trainable params
            image_encoder_parameters_decay = [param for param in image_encoder_parameters_decay if
                                              param.requires_grad]
            image_encoder_parameters_nodecay = [param for param in image_encoder_parameters_nodecay if
                                                param.requires_grad]

            print(f"Image Encoder Params: {len(image_encoder_parameters_decay)} decay, ")
            print(f"Image Encoder Params: {len(image_encoder_parameters_nodecay)} nodecay, ")

            image_encoder_lr = self.optimizer_cfg['image_encoder_lr']
            image_encoder_lr_multiply = self.optimizer_cfg.get('image_encoder_lr_multiply', 1.0)
            image_encoder_lr = image_encoder_lr if image_encoder_lr is not None else lr * image_encoder_lr_multiply
            params_list.append(
                {'params': image_encoder_parameters_decay, 'lr': image_encoder_lr,
                 'weight_decay': 0.05})
            params_list.append(
                {'params': image_encoder_parameters_nodecay, 'lr': image_encoder_lr,
                 'weight_decay': 0.})

        optimizer = instantiate_from_config(self.optimizer_cfg.optimizer, params=params_list, lr=lr)
        if hasattr(self.optimizer_cfg, 'scheduler'):
            scheduler_func = instantiate_from_config(
                self.optimizer_cfg.scheduler,
                max_decay_steps=self.trainer.max_steps,
                lr_max=lr
            )
            scheduler = {
                "scheduler": lr_scheduler.LambdaLR(optimizer, lr_lambda=scheduler_func.schedule),
                "interval": "step",
                "frequency": 1
            }
            schedulers = [scheduler]
        else:
            schedulers = []
        optimizers = [optimizer]

        return optimizers, schedulers

    @rank_zero_only
    @torch.no_grad()
    def on_train_batch_start(self, batch, batch_idx):
        # only for very first batch
        if self.scale_by_std and self.current_epoch == 0 and self.global_step == 0 \
            and batch_idx == 0 and self.ckpt_path is None:
            # set rescale weight to 1./std of encodings
            print("### USING STD-RESCALING ###")

            z_q = self.encode_first_stage(batch[self.first_stage_key])
            z = z_q.detach()

            del self.z_scale_factor
            self.register_buffer("z_scale_factor", 1. / z.flatten().std())
            print(f"setting self.z_scale_factor to {self.z_scale_factor}")

            print("### USING STD-RESCALING ###")

    def on_train_batch_end(self, *args, **kwargs):
        if self.ema_config is not None:
            self.model_ema(self.model)

    def on_train_epoch_start(self) -> None:
        pl.seed_everything(self.trainer.global_rank)

    def forward(self, batch):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16): #float32 for text
            # Print batch info (only once per training run)
            if not hasattr(self, '_printed_batch_info'):
                print(f"\n{'='*70}")
                print(f"[Flow Matching Forward] Batch Information")
                print(f"{'='*70}")
                for key in ['image', 'normal', 'depth']:
                    if key in batch and batch[key] is not None:
                        value = batch[key]
                        if isinstance(value, torch.Tensor):
                            print(f"  {key}: shape={value.shape}")
                        elif isinstance(value, dict):
                            print(f"  {key}: dict with keys={list(value.keys())}")
                        elif isinstance(value, list):
                            if len(value) > 0 and isinstance(value[0], dict):
                                print(f"  {key}: list of {len(value)} dicts, keys={list(value[0].keys())}")
                            else:
                                print(f"  {key}: list of length {len(value)}")
                        else:
                            print(f"  {key}: type={type(value)}")
                print(f"{'='*70}\n")
                self._printed_batch_info = True
            
            # Process RGB images
            # 如果 image 是 dict（多视图）或 list of dicts（batch），通过 pipeline 的 prepare_image 处理（使用 MVImageProcessorV2）
            # 训练时应该与推理时保持一致：都通过 encode_cond 处理
            image_input = batch.get('image')
            mask_input = batch.get('mask')
            
            # 打印调试信息（仅第一次）
            if not hasattr(self, '_printed_mv_debug'):
                print(f"\n{'='*70}")
                print(f"[Multi-View Debug] Image Input Type: {type(image_input)}")
                if isinstance(image_input, dict):
                    print(f"  image_dict keys: {list(image_input.keys())}")
                elif isinstance(image_input, list) and len(image_input) > 0:
                    print(f"  image_list length: {len(image_input)}, first item type: {type(image_input[0])}")
                    if isinstance(image_input[0], dict):
                        print(f"  first dict keys: {list(image_input[0].keys())}")
                print(f"{'='*70}\n")
                self._printed_mv_debug = True
            
            if isinstance(image_input, dict):
                # 单个样本的多视图输入：通过 pipeline 的 prepare_image 处理，得到排序后的 tensor 和 view_idxs
                cond_inputs = self.pipeline.prepare_image(image_input, mask_input)
                
                # 打印 cond_inputs 信息（仅第一次）
                if not hasattr(self, '_printed_cond_inputs_debug'):
                    print(f"\n{'='*70}")
                    print(f"[Multi-View Debug] cond_inputs after prepare_image:")
                    for key, value in cond_inputs.items():
                        if isinstance(value, torch.Tensor):
                            print(f"  {key}: shape={value.shape}, dtype={value.dtype}, min={value.min().item():.4f}, max={value.max().item():.4f}, mean={value.mean().item():.4f}")
                        elif isinstance(value, list):
                            print(f"  {key}: list length={len(value)}, first item={value[0] if len(value) > 0 else 'empty'}")
                        else:
                            print(f"  {key}: type={type(value)}, value={value}")
                    print(f"{'='*70}\n")
                    self._printed_cond_inputs_debug = True
                
                image_tensor = cond_inputs.pop('image')
                mask_tensor = cond_inputs.pop('mask', None)  # 从 cond_inputs 中移除 mask，避免重复传递
                
                # 使用 encode_cond 处理，与推理时保持一致
                # 训练时 disable_drop=True，所以不会进行 classifier-free guidance
                rgb_contexts = self.pipeline.encode_cond(
                    image=image_tensor,
                    additional_cond_inputs=cond_inputs,
                    do_classifier_free_guidance=False,
                    dual_guidance=False,
                )
            elif isinstance(image_input, list) and len(image_input) > 0 and isinstance(image_input[0], dict):
                # batch 的多视图输入：list of dicts，通过 pipeline 的 prepare_image 处理
                cond_inputs = self.pipeline.prepare_image(image_input, mask_input)
                
                # 打印 cond_inputs 信息（仅第一次）
                if not hasattr(self, '_printed_cond_inputs_debug'):
                    print(f"\n{'='*70}")
                    print(f"[Multi-View Debug] cond_inputs after prepare_image (batch):")
                    for key, value in cond_inputs.items():
                        if isinstance(value, torch.Tensor):
                            print(f"  {key}: shape={value.shape}, dtype={value.dtype}, min={value.min().item():.4f}, max={value.max().item():.4f}, mean={value.mean().item():.4f}")
                        elif isinstance(value, list):
                            print(f"  {key}: list length={len(value)}, first item={value[0] if len(value) > 0 else 'empty'}")
                        else:
                            print(f"  {key}: type={type(value)}, value={value}")
                    print(f"{'='*70}\n")
                    self._printed_cond_inputs_debug = True
                
                image_tensor = cond_inputs.pop('image')
                mask_tensor = cond_inputs.pop('mask', None)  # 从 cond_inputs 中移除 mask，避免重复传递
                
                # 使用 encode_cond 处理，与推理时保持一致
                rgb_contexts = self.pipeline.encode_cond(
                    image=image_tensor,
                    additional_cond_inputs=cond_inputs,
                    do_classifier_free_guidance=False,
                    dual_guidance=False,
                )
            else:
                # 单视图输入：直接使用
                rgb_contexts = self.cond_stage_model(image=image_input, text=batch.get('text'), mask=mask_input)
            
            # Process normal maps if available - concatenate with RGB tokens
            if 'normal' in batch and batch['normal'] is not None:
                # Load normal maps and encode them through DinoImageEncoderMV
                normal_input = batch.get('normal')
                normal_mask_input = batch.get('normal_mask')
                if isinstance(normal_input, dict):
                    # 单个样本的多视图 normal：通过 pipeline 的 prepare_image 处理
                    normal_cond_inputs = self.pipeline.prepare_image(normal_input, normal_mask_input)
                    normal_tensor = normal_cond_inputs.pop('image')
                    normal_mask_tensor = normal_cond_inputs.pop('mask', None)  # 从 cond_inputs 中移除 mask，避免重复传递
                    # 使用 encode_cond 处理，与推理时保持一致
                    normal_contexts = self.pipeline.encode_cond(
                        image=normal_tensor,
                        additional_cond_inputs=normal_cond_inputs,
                        do_classifier_free_guidance=False,
                        dual_guidance=False,
                    )
                elif isinstance(normal_input, list) and len(normal_input) > 0 and isinstance(normal_input[0], dict):
                    # batch 的多视图 normal：list of dicts，通过 pipeline 的 prepare_image 处理
                    normal_cond_inputs = self.pipeline.prepare_image(normal_input, normal_mask_input)
                    normal_tensor = normal_cond_inputs.pop('image')
                    normal_mask_tensor = normal_cond_inputs.pop('mask', None)  # 从 cond_inputs 中移除 mask，避免重复传递
                    # 使用 encode_cond 处理，与推理时保持一致
                    normal_contexts = self.pipeline.encode_cond(
                        image=normal_tensor,
                        additional_cond_inputs=normal_cond_inputs,
                        do_classifier_free_guidance=False,
                        dual_guidance=False,
                    )
                else:
                    normal_contexts = self.cond_stage_model(image=normal_input, text=batch.get('text'), mask=normal_mask_input)
                
                # Concatenate RGB and Normal tokens along sequence dimension
                for key in rgb_contexts:
                    if isinstance(rgb_contexts[key], torch.Tensor):
                        if key in normal_contexts and isinstance(normal_contexts[key], torch.Tensor):
                            # Concatenate RGB and Normal tokens
                            rgb_contexts[key] = torch.cat([rgb_contexts[key], normal_contexts[key]], dim=1)
            
            contexts = rgb_contexts
            
            # Print context info (only once per training run)
            if not hasattr(self, '_printed_context_info'):
                print(f"\n{'='*70}")
                print(f"[Flow Matching Forward] Condition Context Summary")
                print(f"{'='*70}")
                for key, value in contexts.items():
                    if isinstance(value, dict):
                        for sub_key, sub_value in value.items():
                            if isinstance(sub_value, torch.Tensor):
                                print(f"  {key}['{sub_key}']: shape={sub_value.shape}, seq_len={sub_value.shape[1]}")
                    elif isinstance(value, torch.Tensor):
                        print(f"  {key}: shape={value.shape}, seq_len={value.shape[1]}")
                print(f"{'='*70}\n")
                self._printed_context_info = True
            
            # Process depth conditioning if available
            if self.controlnet is not None and 'depth' in batch:
                depth = batch['depth'].to(self.device)
                
                if self.num_views > 1:
                    # Multi-view depth processing: expect (B, num_views, 1, H, W)
                    if len(depth.shape) == 5:
                        depth_features = self.controlnet(depth)  # (B, 1, hidden_dim)
                    else:
                        # 单视图 depth：(B, 1, H, W) -> (B, 1, 1, H, W)，不重复凑齐多视图（模型用方向向量）
                        depth = depth.unsqueeze(1)  # (B, 1, 1, H, W)
                        depth_features = self.controlnet(depth)  # (B, 1, hidden_dim)
                else:
                    # Single-view depth processing: expect (B, 1, H, W)
                    if len(depth.shape) == 5:
                        # If multi-view format provided, average across views
                        depth = depth.mean(dim=1)  # (B, 1, H, W)
                    depth_features = self.controlnet(depth)  # (B, 1, hidden_dim)
                
                # Inject depth features into contexts
                if 'additional' not in contexts:
                    contexts['additional'] = {}
                contexts['additional']['depth'] = depth_features

        with torch.autocast(device_type="cuda", dtype=torch.float16):
            with torch.no_grad():
                latents = self.first_stage_model.encode(
                    batch[self.first_stage_key],
                    sample_posterior=self.sample_posterior,
                )
                latents = self.z_scale_factor * latents
                # print(latents.shape)

                # check vae encode and decode is ok? answer is ok!
                # import time
                # from hy3dshape.pipelines import export_to_trimesh
                # latents = 1. / self.z_scale_factor * latents
                # latents = self.first_stage_model(latents)
                # outputs = self.first_stage_model.latents2mesh(
                #     latents,
                #     bounds=1.01,
                #     mc_level=0.0,
                #     num_chunks=20000,
                #     octree_resolution=256,
                #     mc_algo='mc',
                #     enable_pbar=True
                # )
                # mesh = export_to_trimesh(outputs)
                # if isinstance(mesh, list):
                #     for midx, m in enumerate(mesh):
                #         m.export(f"check_{midx}_{time.time()}.glb")
                # else:
                #     mesh.export(f"check_{time.time()}.glb")
                
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = self.transport.training_losses(self.model, latents, dict(contexts=contexts))["loss"].mean()
        return loss

    def on_after_backward(self):
        """在 backward 之后监控梯度（每100步打印一次）"""
        if self.training:
            # 计算梯度范数
            total_norm = 0.0
            param_count = 0
            max_grad_norm = 0.0
            max_grad_param_name = None
            
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    param_norm = param.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
                    param_count += 1
                    if param_norm.item() > max_grad_norm:
                        max_grad_norm = param_norm.item()
                        max_grad_param_name = name
            
            total_norm = total_norm ** (1. / 2)
            
            # 每100步打印一次，或者梯度范数异常时立即打印
            if self.global_step % 100 == 0 or total_norm > 10.0:
                print(f"\n{'='*70}")
                print(f"[Training Debug] Step {self.global_step} (after backward)")
                print(f"  Learning rate: {self.trainer.optimizers[0].param_groups[0]['lr']:.2e}")
                print(f"  Total gradient norm: {total_norm:.6f}")
                print(f"  Max gradient norm: {max_grad_norm:.6f} (param: {max_grad_param_name})")
                print(f"  Parameters with gradients: {param_count}")
                
                # 警告：如果梯度范数太大
                if total_norm > 10.0:
                    print(f"  ⚠ CRITICAL WARNING: Gradient norm is very large ({total_norm:.2f})!")
                    print(f"     This WILL cause pretrained weights to be destroyed!")
                    print(f"     Immediate actions:")
                    print(f"     1) Lower learning rate (current: {self.trainer.optimizers[0].param_groups[0]['lr']:.2e})")
                    print(f"     2) Add gradient clipping (max_norm=1.0)")
                    print(f"     3) Check data preprocessing (normalization, value ranges)")
                elif total_norm > 5.0:
                    print(f"  ⚠ WARNING: Gradient norm is large ({total_norm:.2f})")
                    print(f"     Consider lowering learning rate or adding gradient clipping")
                elif total_norm > 1.0:
                    print(f"  ℹ INFO: Gradient norm is moderate ({total_norm:.2f})")
                else:
                    print(f"  ✓ Gradient norm is normal ({total_norm:.2f})")
                print(f"{'='*70}\n")

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        loss = self.forward(batch)

        # 监控初始权重和 view_embed（仅第一次）
        if batch_idx == 0 and self.global_step == 0:
            print(f"\n{'='*70}")
            print(f"[Training Debug] Step {self.global_step}, Batch {batch_idx} (initialization check)")
            print(f"  Loss: {loss.item():.6f}")
            print(f"  Learning rate: {self.trainer.optimizers[0].param_groups[0]['lr']:.2e}")
            
            # 检查前几个参数的权重（仅第一次）
            print(f"\n[Training Debug] Model weight check (first 3 params):")
            for i, (name, param) in enumerate(self.model.named_parameters()):
                if i < 3:
                    print(f"  {name}: mean={param.data.mean().item():.6f}, std={param.data.std().item():.6f}")
            
            # 检查 view_embed 的值（仅第一次）
            if hasattr(self.cond_stage_model, 'main_image_encoder') and hasattr(self.cond_stage_model.main_image_encoder, 'view_embed'):
                view_embed = self.cond_stage_model.main_image_encoder.view_embed
                print(f"\n[Training Debug] view_embed check:")
                print(f"  view_embed shape: {view_embed.shape}")
                print(f"  view_embed mean: {view_embed.mean().item():.6f}")
                print(f"  view_embed std: {view_embed.std().item():.6f}")
                print(f"  view_embed requires_grad: {view_embed.requires_grad}")
                print(f"  view_embed is buffer: {'view_embed' in dict(self.cond_stage_model.main_image_encoder.named_buffers())}")
                print(f"  ℹ NOTE: view_embed is a FIXED positional encoding (sincos), NOT trainable")
                print(f"     It uses deterministic initialization (same as pretrained model)")
                print(f"     This is consistent with official Hunyuan3D-2 implementation")
            print(f"{'='*70}\n")

        split = 'train'
        loss_dict = {
            f"{split}/simple": loss.detach(),
            f"{split}/total_loss": loss.detach(),
            f"{split}/lr_abs": self.optimizers().param_groups[0]['lr'],
        }
        self.log_dict(loss_dict, prog_bar=True, logger=True, sync_dist=False, rank_zero_only=True)

        return loss

    def validation_step(self, batch, batch_idx, optimizer_idx=0):
        loss = self.forward(batch)
        split = 'val'
        loss_dict = {
            f"{split}/simple": loss.detach(),
            f"{split}/total_loss": loss.detach(),
            f"{split}/lr_abs": self.optimizers().param_groups[0]['lr'],
        }
        self.log_dict(loss_dict, prog_bar=True, logger=True, sync_dist=False, rank_zero_only=True)

        return loss

    @torch.no_grad()
    def sample(self, batch, output_type='trimesh', **kwargs):
        self.cond_stage_model.disable_drop = True

        generator = torch.Generator().manual_seed(0)

        with self.ema_scope("Sample"):
            with torch.amp.autocast(device_type='cuda'):
                try:
                    self.pipeline.device = self.device
                    self.pipeline.dtype = self.dtype
                    print("### USING PIPELINE ###")
                    print(f'device: {self.device} dtype : {self.dtype}')
                    additional_params = {'output_type':output_type}

                    image = batch.get("image", None)
                    mask = batch.get('mask', None)
                    depth = batch.get('depth', None)  # 获取深度图
                    
                    # 如果有controlnet和depth，将controlnet传递给pipeline
                    if self.controlnet is not None and depth is not None:
                        additional_params['controlnet'] = self.controlnet
                        additional_params['depth'] = depth
                    
                    outputs = self.pipeline(image=image, 
                                            mask=mask,
                                            generator=generator,
                                            **additional_params)

                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    print(f"Unexpected {e=}, {type(e)=}")
                    with open("error.txt", "a") as f:
                        f.write(str(e))
                        f.write(traceback.format_exc())
                        f.write("\n")
                    outputs = [None]

        self.cond_stage_model.disable_drop = False
        return [outputs]
    
    def _create_depth_controlnet(self, in_channels=1):
        """Create a simple ControlNet-like adapter for depth conditioning"""
        import torch.nn as nn
        
        class DepthControlNet(nn.Module):
            def __init__(self, in_channels, out_channels=None):
                super().__init__()
                # Simple depth feature extractor
                out_channels = out_channels or 768  # Match additional_cond_hidden_state
                
                self.depth_encoder = nn.Sequential(
                    nn.Conv2d(in_channels, 64, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(64, 128, 3, stride=2, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(128, 256, 3, stride=2, padding=1),
                    nn.ReLU(inplace=True),
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(),
                    nn.Linear(256, out_channels),
                )
                
                # Token expansion: generate diverse tokens similar to MultiViewDepthControlNet
                self.num_depth_tokens = 16
                self.depth_token_queries = nn.Parameter(torch.randn(1, self.num_depth_tokens, out_channels) * 0.02)
                
                self.token_generator = nn.MultiheadAttention(
                    embed_dim=out_channels,
                    num_heads=8,
                    dropout=0.0,
                    batch_first=True
                )
                
                self.token_norm = nn.LayerNorm(out_channels)
                
                # Initialize weights
                for m in self.modules():
                    if isinstance(m, nn.Conv2d):
                        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    elif isinstance(m, nn.Linear):
                        nn.init.normal_(m.weight, 0, 0.01)
                        if m.bias is not None:
                            nn.init.zeros_(m.bias)
                        
            def forward(self, depth):
                """
                Args:
                    depth: (B, 1, H, W) depth maps
                Returns:
                    depth_features: (B, 16, out_channels) depth features as sequence
                """
                B = depth.shape[0]
                features = self.depth_encoder(depth)  # (B, out_channels)
                
                # Generate diverse tokens using cross-attention
                depth_kv = features.unsqueeze(1)  # (B, 1, out_channels)
                queries = self.depth_token_queries.expand(B, -1, -1)  # (B, 16, out_channels)
                
                depth_tokens, _ = self.token_generator(
                    query=queries,
                    key=depth_kv,
                    value=depth_kv,
                    need_weights=False
                )
                
                depth_tokens = self.token_norm(depth_tokens + queries)  # (B, 16, out_channels)
                return depth_tokens
        
        return DepthControlNet(in_channels)
