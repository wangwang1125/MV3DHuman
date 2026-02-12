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
from ...utils.misc import instantiate_from_config, instantiate_non_trainable_model



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
        scale_by_std: bool = False,
        z_scale_factor: float = 1.0,
        ckpt_path: Optional[str] = None,
        ignore_keys: Union[Tuple[str], List[str]] = (),
        torch_compile: bool = False,
    ):
        super().__init__()
        self.first_stage_key = first_stage_key
        self.cond_stage_key = cond_stage_key

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

        self.ckpt_path = ckpt_path
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

        # ========= config controlnet model ========= #
        self.controlnet = None
        self.control_in_channels = control_in_channels
        self.num_views = denoiser_cfg.params.get('num_views', 1)  # Get num_views from config
        
        if control_net_config is not None:
            if self.num_views > 1:
                # Use multi-view ControlNet for multiple views
                from ..controlnet_multiview import create_multiview_depth_controlnet
                fusion_strategy = control_net_config.get('fusion_strategy', 'attention')
                out_channels = denoiser_cfg.params.get('additional_cond_hidden_state', 768)
                
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
                        print(f"  {key}: shape={batch[key].shape}")
                print(f"{'='*70}\n")
                self._printed_batch_info = True
            
            # Process RGB images
            # 如果 image 是 dict（多视图）或 list of dicts（batch），通过 pipeline 的 prepare_image 处理（使用 MVImageProcessorV2）
            image_input = batch.get('image')
            mask_input = batch.get('mask')
            if isinstance(image_input, dict):
                # 单个样本的多视图输入：通过 pipeline 的 prepare_image 处理，得到排序后的 tensor 和 view_idxs
                cond_inputs = self.pipeline.prepare_image(image_input, mask_input)
                image_tensor = cond_inputs.pop('image')
                # view_idxs 会在 cond_inputs 中，传递给 conditioner
                rgb_contexts = self.cond_stage_model(image=image_tensor, text=batch.get('text'), mask=cond_inputs.get('mask'), **cond_inputs)
            elif isinstance(image_input, list) and len(image_input) > 0 and isinstance(image_input[0], dict):
                # batch 的多视图输入：list of dicts，通过 pipeline 的 prepare_image 处理
                cond_inputs = self.pipeline.prepare_image(image_input, mask_input)
                image_tensor = cond_inputs.pop('image')
                rgb_contexts = self.cond_stage_model(image=image_tensor, text=batch.get('text'), mask=cond_inputs.get('mask'), **cond_inputs)
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
                    normal_contexts = self.cond_stage_model(image=normal_tensor, text=batch.get('text'), mask=normal_cond_inputs.get('mask'), **normal_cond_inputs)
                elif isinstance(normal_input, list) and len(normal_input) > 0 and isinstance(normal_input[0], dict):
                    # batch 的多视图 normal：list of dicts，通过 pipeline 的 prepare_image 处理
                    normal_cond_inputs = self.pipeline.prepare_image(normal_input, normal_mask_input)
                    normal_tensor = normal_cond_inputs.pop('image')
                    normal_contexts = self.cond_stage_model(image=normal_tensor, text=batch.get('text'), mask=normal_cond_inputs.get('mask'), **normal_cond_inputs)
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
                        # Fallback: if depth is (B, 1, H, W), treat as single view
                        print(f"[WARNING] Expected multi-view depth (B, {self.num_views}, 1, H, W), got {depth.shape}")
                        depth = depth.unsqueeze(1)  # (B, 1, 1, H, W)
                        # Replicate to match expected number of views
                        depth = depth.repeat(1, self.num_views, 1, 1, 1)  # (B, num_views, 1, H, W)
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
                latents = self.first_stage_model.encode(batch[self.first_stage_key], sample_posterior=True)
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

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        loss = self.forward(batch)
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
