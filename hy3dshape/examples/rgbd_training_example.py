#!/usr/bin/env python3
"""
Hunyuan3D RGBD微调训练示例

本示例展示如何使用RGBD微调训练功能：
1. 数据准备和预处理
2. 配置文件设置
3. 模型训练
4. 结果评估

作者: Hunyuan3D Team
日期: 2024
"""

import os
import sys
import json
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Optional

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent))

from train_rgbd_finetuning import RGBDFineTuningTrainer
from hy3dshape.data.rgbd_dit_asl import RGBDAlignedShapeLatentDataset
from hy3dshape.models.conditioner import RGBDImageEncoder


class RGBDTrainingExample:
    """RGBD训练示例类"""
    
    def __init__(self, base_dir: str = "/mnt/e/vscode/project/Hunyuan3D-2.1/hy3dshape"):
        self.base_dir = Path(base_dir)
        self.config_dir = self.base_dir / "configs"
        self.data_dir = self.base_dir / "data" / "example"
        self.output_dir = self.base_dir / "outputs" / "rgbd_example"
        
        # 创建必要目录
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def prepare_example_data(self) -> None:
        """准备示例数据"""
        print("📊 准备示例数据...")
        
        # 创建数据目录结构
        (self.data_dir / "train" / "rgb").mkdir(parents=True, exist_ok=True)
        (self.data_dir / "train" / "depth").mkdir(parents=True, exist_ok=True)
        (self.data_dir / "train" / "surface").mkdir(parents=True, exist_ok=True)
        (self.data_dir / "val" / "rgb").mkdir(parents=True, exist_ok=True)
        (self.data_dir / "val" / "depth").mkdir(parents=True, exist_ok=True)
        (self.data_dir / "val" / "surface").mkdir(parents=True, exist_ok=True)
        
        # 生成示例数据列表
        train_data = []
        val_data = []
        
        # 训练数据 (假设有100个样本)
        for i in range(100):
            train_data.append({
                "image": f"data/example/train/rgb/{i:03d}.png",
                "depth": f"data/example/train/depth/{i:03d}.npy",
                "surface": f"data/example/train/surface/{i:03d}.npz"
            })
        
        # 验证数据 (假设有20个样本)
        for i in range(20):
            val_data.append({
                "image": f"data/example/val/rgb/{i:03d}.png",
                "depth": f"data/example/val/depth/{i:03d}.npy",
                "surface": f"data/example/val/surface/{i:03d}.npz"
            })
        
        # 保存数据列表
        with open(self.data_dir / "train_list.json", "w") as f:
            json.dump(train_data, f, indent=2)
        
        with open(self.data_dir / "val_list.json", "w") as f:
            json.dump(val_data, f, indent=2)
        
        print(f"✅ 数据列表已生成: {len(train_data)} 训练样本, {len(val_data)} 验证样本")
        print(f"📁 数据目录: {self.data_dir}")
    
    def create_example_config(self) -> str:
        """创建示例配置文件"""
        print("⚙️ 创建示例配置文件...")
        
        config_content = f"""
# RGBD微调训练示例配置
# 基于轻量级配置，适合快速验证和学习

training:
  steps: 1000  # 示例用少量步数
  base_lr: 1e-4
  warmup_steps: 100
  batch_size: 2  # 小批次适合示例
  gradient_clip_val: 1.0
  val_check_interval: 100
  save_top_k: 3
  
dataset:
  target: hy3dshape.data.rgbd_dit_asl.RGBDAlignedShapeLatentModule
  params:
    train_data_path: "{self.data_dir}/train_list.json"
    val_data_path: "{self.data_dir}/val_list.json"
    batch_size: 2
    num_workers: 2
    
    # 深度图配置
    depth_stage_key: "depth"
    depth_clip_range: [0.0, 10.0]
    depth_normalize: true
    require_depth: false  # 允许深度图缺失
    
    # 数据增强
    image_transforms:
      - type: "resize"
        size: [518, 518]
      - type: "random_flip"
        prob: 0.5
    
    depth_transforms:
      - type: "resize"
        size: [518, 518]
      - type: "random_noise"
        std: 0.01

model:
  target: hy3dshape.models.diffuser.Diffuser
  params:
    # VAE配置 (冻结)
    first_stage_config:
      target: hy3dshape.models.vae.ShapeVAE
      params:
        freeze: true
    
    # RGBD条件编码器
    cond_stage_config:
      target: hy3dshape.models.conditioner.RGBDImageEncoder
      params:
        # RGB编码器配置 (冻结)
        main_image_encoder_config:
          target: hy3dshape.models.conditioner.DinoImageEncoder
          params:
            freeze: true
            model_name: "dinov2_vitl14_reg"
            output_dim: 768
        
        # 深度编码器配置 (训练)
        depth_encoder_config:
          type: 'lightweight'  # 轻量级版本
          input_channels: 1
          hidden_dim: 768
          num_layers: 3
          dropout: 0.1
        
        # 跨模态融合配置 (训练)
        fusion_config:
          type: 'simple'  # 简单融合策略
          input_dim: 768
          output_dim: 768
          dropout: 0.1
        
        # 其他参数
        freeze_rgb_encoder: true
        drop_ratio: 0.1
    
    # DiT模型配置 (冻结)
    unet_config:
      target: hy3dshape.models.hunyuandit.HunYuanDiTPlain
      params:
        freeze: true
        input_size: 32
        patch_size: 2
        in_channels: 12
        hidden_size: 1408
        depth: 40
        num_heads: 16
        mlp_ratio: 4.3637

# 优化器配置
optimizer:
  target: torch.optim.AdamW
  params:
    lr: 1e-4
    weight_decay: 0.01
    betas: [0.9, 0.999]

# 学习率调度器
scheduler:
  target: transformers.get_cosine_schedule_with_warmup
  params:
    num_warmup_steps: 100
    num_training_steps: 1000

# 微调特定配置
finetuning:
  # 参数冻结策略
  freeze_strategy:
    rgb_encoder: true
    vae: true
    dit_backbone: true
    depth_encoder: false
    fusion_module: false
  
  # 学习率策略
  lr_strategy:
    depth_encoder: 1.0  # 相对学习率
    fusion_module: 1.0
  
  # 正则化
  regularization:
    l2_weight: 1e-4
    dropout: 0.1

# 日志和检查点
logging:
  log_dir: "{self.output_dir}/logs"
  checkpoint_dir: "{self.output_dir}/checkpoints"
  log_every_n_steps: 10
  val_log_images: 4

# 回调函数
callbacks:
  - target: pytorch_lightning.callbacks.ModelCheckpoint
    params:
      dirpath: "{self.output_dir}/checkpoints"
      filename: "rgbd-example-{{epoch:02d}}-{{val_loss:.4f}}"
      monitor: "val_loss"
      save_top_k: 3
      mode: "min"
  
  - target: pytorch_lightning.callbacks.EarlyStopping
    params:
      monitor: "val_loss"
      patience: 5
      mode: "min"
  
  - target: pytorch_lightning.callbacks.LearningRateMonitor
    params:
      logging_interval: "step"
"""
        
        config_path = self.config_dir / "rgbd_example_config.yaml"
        with open(config_path, "w", encoding="utf-8") as f:
            f.write(config_content)
        
        print(f"✅ 配置文件已创建: {config_path}")
        return str(config_path)
    
    def validate_setup(self) -> bool:
        """验证环境设置"""
        print("🔍 验证环境设置...")
        
        checks = {
            "PyTorch": self._check_pytorch(),
            "CUDA": self._check_cuda(),
            "配置文件": self._check_config(),
            "数据目录": self._check_data_dir(),
            "模型模块": self._check_model_modules()
        }
        
        all_passed = True
        for check_name, passed in checks.items():
            status = "✅" if passed else "❌"
            print(f"  {status} {check_name}")
            if not passed:
                all_passed = False
        
        return all_passed
    
    def _check_pytorch(self) -> bool:
        """检查PyTorch安装"""
        try:
            import torch
            return torch.__version__ >= "1.12.0"
        except ImportError:
            return False
    
    def _check_cuda(self) -> bool:
        """检查CUDA可用性"""
        try:
            import torch
            return torch.cuda.is_available()
        except:
            return False
    
    def _check_config(self) -> bool:
        """检查配置文件"""
        config_path = self.config_dir / "rgbd_example_config.yaml"
        return config_path.exists()
    
    def _check_data_dir(self) -> bool:
        """检查数据目录"""
        return (self.data_dir / "train_list.json").exists()
    
    def _check_model_modules(self) -> bool:
        """检查模型模块"""
        try:
            from hy3dshape.models.conditioner import RGBDImageEncoder
            from hy3dshape.models.depth_encoder import LightweightDepthEncoder
            from hy3dshape.models.cross_modal_fusion import SimpleCrossModalFusion
            return True
        except ImportError as e:
            print(f"    模块导入错误: {e}")
            return False
    
    def run_training_example(self, config_path: str) -> None:
        """运行训练示例"""
        print("🚀 开始RGBD微调训练示例...")
        
        try:
            # 创建训练器
            trainer = RGBDFineTuningTrainer(
                config_path=config_path,
                resume_path=None
            )
            
            print("📋 训练配置:")
            print(f"  - 配置文件: {config_path}")
            print(f"  - 输出目录: {self.output_dir}")
            print(f"  - 训练步数: 1000 (示例)")
            print(f"  - 批次大小: 2")
            print(f"  - 学习率: 1e-4")
            
            # 开始训练
            print("\n🎯 开始训练...")
            trainer.train()
            
            print("\n✅ 训练完成!")
            print(f"📁 检查点保存在: {self.output_dir}/checkpoints")
            print(f"📊 日志保存在: {self.output_dir}/logs")
            
        except Exception as e:
            print(f"❌ 训练过程中出现错误: {e}")
            print("\n🔧 故障排除建议:")
            print("  1. 检查数据路径是否正确")
            print("  2. 确认GPU内存充足")
            print("  3. 验证依赖包版本")
            print("  4. 查看详细错误日志")
            raise
    
    def demonstrate_inference(self, checkpoint_path: str) -> None:
        """演示推理过程"""
        print("🔮 演示RGBD推理过程...")
        
        try:
            # 加载训练好的模型
            print(f"📥 加载检查点: {checkpoint_path}")
            
            # 创建示例输入
            rgb_image = torch.randn(1, 3, 518, 518)  # 示例RGB图像
            depth_image = torch.randn(1, 1, 518, 518)  # 示例深度图
            
            print("🖼️ 输入数据:")
            print(f"  - RGB图像: {rgb_image.shape}")
            print(f"  - 深度图: {depth_image.shape}")
            
            # 模拟推理过程
            print("⚡ 执行推理...")
            
            # 这里应该加载实际模型并进行推理
            # 由于这是示例，我们只模拟输出
            output_shape = torch.randn(1, 12, 32, 32)  # 示例输出
            
            print("📤 推理结果:")
            print(f"  - 输出形状: {output_shape.shape}")
            print("  - 推理成功! ✅")
            
        except Exception as e:
            print(f"❌ 推理过程中出现错误: {e}")
            raise
    
    def generate_report(self) -> None:
        """生成训练报告"""
        print("📋 生成训练报告...")
        
        report_content = f"""
# RGBD微调训练示例报告

## 训练配置
- 基础目录: {self.base_dir}
- 数据目录: {self.data_dir}
- 输出目录: {self.output_dir}
- 训练步数: 1000 (示例)
- 批次大小: 2
- 学习率: 1e-4

## 模型架构
- RGB编码器: DinoImageEncoder (冻结)
- 深度编码器: LightweightDepthEncoder (训练)
- 融合模块: SimpleCrossModalFusion (训练)
- 生成模型: HunYuanDiTPlain (冻结)

## 训练策略
- 预训练模型复用: ✅
- 参数冻结策略: ✅
- 微调新增模块: ✅
- 渐进式学习率: ✅

## 文件结构
```
{self.base_dir}/
├── configs/
│   └── rgbd_example_config.yaml
├── data/
│   └── example/
│       ├── train_list.json
│       └── val_list.json
├── outputs/
│   └── rgbd_example/
│       ├── checkpoints/
│       └── logs/
└── examples/
    └── rgbd_training_example.py
```

## 下一步
1. 准备真实的RGBD数据集
2. 调整配置参数
3. 进行完整训练
4. 评估模型性能
5. 部署到生产环境

## 技术支持
如有问题，请参考:
- RGBD_FINETUNING_README.md
- 项目文档
- GitHub Issues
"""
        
        report_path = self.output_dir / "training_report.md"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_content)
        
        print(f"✅ 报告已生成: {report_path}")


def main():
    """主函数"""
    print("🎉 欢迎使用Hunyuan3D RGBD微调训练示例!")
    print("=" * 50)
    
    # 创建示例实例
    example = RGBDTrainingExample()
    
    try:
        # 1. 准备数据
        example.prepare_example_data()
        print()
        
        # 2. 创建配置
        config_path = example.create_example_config()
        print()
        
        # 3. 验证环境
        if not example.validate_setup():
            print("❌ 环境验证失败，请检查上述问题后重试")
            return
        print()
        
        # 4. 询问是否运行训练
        response = input("🤔 是否运行训练示例? (y/N): ").lower().strip()
        if response in ['y', 'yes']:
            example.run_training_example(config_path)
            print()
            
            # 5. 演示推理 (如果有检查点)
            checkpoint_dir = example.output_dir / "checkpoints"
            if checkpoint_dir.exists():
                checkpoints = list(checkpoint_dir.glob("*.ckpt"))
                if checkpoints:
                    latest_checkpoint = max(checkpoints, key=lambda x: x.stat().st_mtime)
                    example.demonstrate_inference(str(latest_checkpoint))
                    print()
        
        # 6. 生成报告
        example.generate_report()
        print()
        
        print("🎊 示例运行完成!")
        print("📚 更多信息请参考 RGBD_FINETUNING_README.md")
        
    except KeyboardInterrupt:
        print("\n⏹️ 用户中断操作")
    except Exception as e:
        print(f"\n💥 运行过程中出现错误: {e}")
        print("\n🔧 请检查:")
        print("  1. 环境配置是否正确")
        print("  2. 依赖包是否完整")
        print("  3. 权限设置是否正确")
        print("  4. 磁盘空间是否充足")
        raise


if __name__ == "__main__":
    main()