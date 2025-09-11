#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RGBD微调训练脚本
基于Hunyuan3D-2.1架构，采用预训练模型复用策略
固定预训练参数，仅训练深度编码器和跨模态融合模块

使用方法:
python train_rgbd_finetuning.py --config configs/hunyuandit-rgbd-finetuning-flowmatching-dinol518-bf16-lr1e4-4096.yaml
"""

import os
import sys
import argparse
import warnings
from pathlib import Path

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.strategies import DeepSpeedStrategy

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from hy3dshape.utils.misc import instantiate_from_config, load_config
from hy3dshape.utils.trainings.trainer_utils import (
    setup_callbacks, setup_logger, setup_strategy
)

# 忽略一些警告
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


class RGBDFineTuningTrainer:
    """
    RGBD微调训练器
    """
    
    def __init__(self, config_path, resume_path=None, **kwargs):
        self.config = load_config(config_path)
        self.resume_path = resume_path
        
        # 更新配置
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
    
    def setup_model(self):
        """
        设置模型，包括参数冻结策略
        """
        print("正在初始化RGBD微调模型...")
        
        # 实例化模型
        model = instantiate_from_config(self.config.model)
        
        # 应用参数冻结策略
        if hasattr(self.config, 'finetuning') and 'freeze_strategy' in self.config.finetuning:
            self.apply_freeze_strategy(model, self.config.finetuning.freeze_strategy)
        
        # 打印可训练参数统计
        self.print_parameter_stats(model)
        
        return model
    
    def apply_freeze_strategy(self, model, freeze_config):
        """
        应用参数冻结策略
        
        Args:
            model: 模型实例
            freeze_config: 冻结配置
        """
        print("应用参数冻结策略...")
        
        # 冻结VAE参数
        if freeze_config.get('vae', True):
            if hasattr(model, 'first_stage_model'):
                for param in model.first_stage_model.parameters():
                    param.requires_grad = False
                print("✓ VAE参数已冻结")
        
        # 冻结DiT主干网络参数
        if freeze_config.get('dit_backbone', True):
            if hasattr(model, 'denoiser'):
                for param in model.denoiser.parameters():
                    param.requires_grad = False
                print("✓ DiT主干网络参数已冻结")
        
        # RGB编码器冻结策略在RGBDImageEncoder中已处理
        if freeze_config.get('rgb_encoder', True):
            print("✓ RGB编码器参数冻结策略已在编码器中应用")
        
        # 确保深度编码器和融合模块可训练
        if hasattr(model, 'cond_stage_model'):
            cond_model = model.cond_stage_model
            if hasattr(cond_model, 'depth_encoder'):
                for param in cond_model.depth_encoder.parameters():
                    param.requires_grad = True
                print("✓ 深度编码器参数设为可训练")
            
            if hasattr(cond_model, 'fusion_module'):
                for param in cond_model.fusion_module.parameters():
                    param.requires_grad = True
                print("✓ 跨模态融合模块参数设为可训练")
    
    def print_parameter_stats(self, model):
        """
        打印参数统计信息
        """
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        
        print(f"\n参数统计:")
        print(f"  总参数量: {total_params:,}")
        print(f"  可训练参数: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
        print(f"  冻结参数: {frozen_params:,} ({frozen_params/total_params*100:.2f}%)")
        
        # 详细统计各模块参数
        if hasattr(model, 'cond_stage_model'):
            cond_model = model.cond_stage_model
            if hasattr(cond_model, 'depth_encoder'):
                depth_params = sum(p.numel() for p in cond_model.depth_encoder.parameters())
                print(f"  深度编码器参数: {depth_params:,}")
            
            if hasattr(cond_model, 'fusion_module'):
                fusion_params = sum(p.numel() for p in cond_model.fusion_module.parameters())
                print(f"  融合模块参数: {fusion_params:,}")
        
        print()
    
    def setup_data(self):
        """
        设置数据模块
        """
        print("正在初始化RGBD数据模块...")
        
        # 使用RGBD数据模块
        if 'target' in self.config.dataset and 'rgbd' not in self.config.dataset.target.lower():
            # 如果配置中没有使用RGBD数据模块，自动替换
            print("警告: 配置文件中未使用RGBD数据模块，自动替换为RGBDAlignedShapeLatentModule")
            self.config.dataset.target = 'hy3dshape.data.rgbd_dit_asl.RGBDAlignedShapeLatentModule'
        
        data_module = instantiate_from_config(self.config.dataset)
        return data_module
    
    def setup_callbacks(self):
        """
        设置回调函数
        """
        callbacks = []
        
        # 模型检查点
        checkpoint_callback = ModelCheckpoint(
            dirpath=f"checkpoints/rgbd_finetuning_{self.config.name.replace(' ', '_')}",
            filename="{epoch:02d}-{val_loss:.4f}",
            monitor="val_loss",
            mode="min",
            save_top_k=3,
            save_last=True,
            every_n_epochs=1,
        )
        callbacks.append(checkpoint_callback)
        
        # 学习率监控
        lr_monitor = LearningRateMonitor(logging_interval='step')
        callbacks.append(lr_monitor)
        
        # 添加配置中的回调
        if hasattr(self.config, 'callbacks'):
            for callback_name, callback_config in self.config.callbacks.items():
                if callback_name not in ['logger', 'file_loggers']:  # 这些在logger中处理
                    callback = instantiate_from_config(callback_config)
                    callbacks.append(callback)
        
        return callbacks
    
    def setup_logger(self):
        """
        设置日志记录器
        """
        loggers = []
        
        # TensorBoard日志
        tb_logger = TensorBoardLogger(
            save_dir="logs",
            name=f"rgbd_finetuning_{self.config.name.replace(' ', '_')}",
            version=None,
        )
        loggers.append(tb_logger)
        
        # 可选: Weights & Biases日志
        # wandb_logger = WandbLogger(
        #     project="hunyuan3d-rgbd-finetuning",
        #     name=f"rgbd_ft_{self.config.name}",
        #     save_dir="logs"
        # )
        # loggers.append(wandb_logger)
        
        return loggers
    
    def train(self):
        """
        开始训练
        """
        print(f"开始RGBD微调训练: {self.config.name}")
        print(f"配置文件: {self.config}")
        
        # 设置组件
        model = self.setup_model()
        data_module = self.setup_data()
        callbacks = self.setup_callbacks()
        loggers = self.setup_logger()
        
        # 设置训练策略
        strategy = None
        if torch.cuda.device_count() > 1:
            # 多GPU训练
            strategy = "ddp"
            print(f"使用多GPU训练，设备数量: {torch.cuda.device_count()}")
        
        # 创建训练器
        trainer = pl.Trainer(
            max_steps=self.config.training.steps,
            precision=self.config.training.amp_type if self.config.training.use_amp else 32,
            gradient_clip_val=self.config.training.gradient_clip_val,
            gradient_clip_algorithm=self.config.training.gradient_clip_algorithm,
            val_check_interval=self.config.training.val_check_interval,
            limit_val_batches=self.config.training.limit_val_batches,
            callbacks=callbacks,
            logger=loggers,
            strategy=strategy,
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices="auto",
            enable_checkpointing=True,
            enable_progress_bar=True,
            enable_model_summary=True,
        )
        
        # 开始训练
        trainer.fit(
            model=model,
            datamodule=data_module,
            ckpt_path=self.resume_path
        )
        
        print("训练完成!")
        
        # 保存最终模型
        final_model_path = f"checkpoints/rgbd_finetuning_final.ckpt"
        trainer.save_checkpoint(final_model_path)
        print(f"最终模型已保存至: {final_model_path}")


def main():
    parser = argparse.ArgumentParser(description="RGBD微调训练")
    parser.add_argument(
        "--config", 
        type=str, 
        required=True,
        help="训练配置文件路径"
    )
    parser.add_argument(
        "--resume", 
        type=str, 
        default=None,
        help="恢复训练的检查点路径"
    )
    parser.add_argument(
        "--gpus", 
        type=int, 
        default=-1,
        help="使用的GPU数量 (-1表示使用所有可用GPU)"
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=None,
        help="批次大小 (覆盖配置文件中的设置)"
    )
    parser.add_argument(
        "--lr", 
        type=float, 
        default=None,
        help="学习率 (覆盖配置文件中的设置)"
    )
    parser.add_argument(
        "--steps", 
        type=int, 
        default=None,
        help="训练步数 (覆盖配置文件中的设置)"
    )
    
    args = parser.parse_args()
    
    # 检查配置文件是否存在
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"配置文件不存在: {args.config}")
    
    # 设置环境变量
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    # 创建训练器并开始训练
    trainer = RGBDFineTuningTrainer(
        config_path=args.config,
        resume_path=args.resume,
    )
    
    trainer.train()


if __name__ == "__main__":
    main()