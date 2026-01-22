#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hunyuan3D 多视图3D生成简化脚本
使用方法: python MV3D.py --file input_images
"""

import sys
import os
import argparse
import time
from pathlib import Path

# 添加路径
sys.path.insert(0, './hy3dshape')

# 应用torchvision兼容性修复
try:
    from torchvision_fix import apply_fix
    apply_fix()
except ImportError:
    print("Warning: torchvision_fix module not found, proceeding without compatibility fix")
except Exception as e:
    print(f"Warning: Failed to apply torchvision fix: {e}")

import torch
from PIL import Image

from hy3dshape import Hunyuan3DDiTFlowMatchingPipeline
from hy3dshape.pipelines import export_to_trimesh
from hy3dshape.preprocessors import MVImageProcessorV2
from hy3dshape.models.conditioner import DinoImageEncoderMV


def load_multiview_images(image_dir):
    """
    从文件夹加载4张图片：001.jpg, 002.jpg, 003.jpg, 004.jpg
    映射到：front, left, back, right
    
    Args:
        image_dir (str): 图片文件夹路径
        
    Returns:
        dict: {'front': PIL.Image, 'left': PIL.Image, 'back': PIL.Image, 'right': PIL.Image}
    """
    image_dir = Path(image_dir)
    if not image_dir.exists():
        raise FileNotFoundError(f"图片文件夹不存在: {image_dir}")
    
    # 图片文件名映射
    image_files = {
        'front': image_dir / '000.jpg',
        'left': image_dir / '001.jpg',
        'back': image_dir / '002.jpg',
        'right': image_dir / '003.jpg',
    }
    
    images = {}
    for view_name, file_path in image_files.items():
        if not file_path.exists():
            raise FileNotFoundError(f"缺少图片文件: {file_path}")
        
        try:
            img = Image.open(file_path)
            # 转换为RGBA格式
            if img.mode != 'RGBA':
                img = img.convert('RGBA')
            images[view_name] = img
            print(f"✅ 加载 {view_name} 视图: {file_path.name} ({img.size[0]}x{img.size[1]})")
        except Exception as e:
            raise RuntimeError(f"加载图片失败 {file_path}: {e}")
    
    return images



def initialize_model(model_path, subfolder, device, num_views=4, low_vram=False, rgb_lora_path=None):
    """
    初始化多视图RGB模型
    
    Args:
        model_path (str): 模型路径
        subfolder (str): 子文件夹
        device (str): 设备
        num_views (int): 视图数量
        low_vram (bool): 是否使用低显存模式
        rgb_lora_path (str, optional): 全量训练后的checkpoint路径，如果为None则自动搜索
        
    Returns:
        Hunyuan3DDiTFlowMatchingPipeline: 初始化的pipeline
    """
    print(f"正在加载多视图RGB模型...")
    print(f"  模型路径: {model_path}")
    print(f"  子文件夹: {subfolder}")
    print(f"  设备: {device}")
    print(f"  视图数量: {num_views}")
    
    # 检测是否是Hunyuan3D-2mv模型（需要使用safetensors格式）
    is_mv_model = 'Hunyuan3D-2mv' in model_path or 'mv' in model_path.lower() or 'hunyuan3d-dit-v2-mv' in subfolder
    use_safetensors = is_mv_model
    
    if is_mv_model:
        print(f"  检测到Hunyuan3D-2mv模型，使用safetensors格式")
    
    # 加载模型
    try:
        pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            model_path,
            subfolder=subfolder,
            use_safetensors=use_safetensors,
            device=device,
        )
        if use_safetensors:
            print("✅ 使用safetensors格式加载Hunyuan3D-2mv模型成功")
    except Exception as e:
        # 如果safetensors加载失败，尝试ckpt格式
        if use_safetensors:
            print(f"⚠️ 使用safetensors加载失败，尝试ckpt格式: {e}")
            pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
                model_path,
                subfolder=subfolder,
                use_safetensors=False,
                device=device,
            )
            print("✅ 使用ckpt格式加载模型成功")
        else:
            raise
    
    # 自动搜索checkpoint（如果未指定）
    if rgb_lora_path is None:
        # 优先查找mv版本的checkpoint（与训练配置匹配）
        default_lora_dirs = [
            # mv版本的checkpoint路径（优先）
            "./hy3dshape/output_folder/dit/multiview_rgb_finetuning_mv/ckpt",
        ]
        
        for lora_dir in default_lora_dirs:
            if os.path.exists(lora_dir):
                # 如果是目录，查找.ckpt文件
                if os.path.isdir(lora_dir):
                    ckpt_files = [f for f in os.listdir(lora_dir) if f.endswith('.ckpt')]
                    if ckpt_files:
                        # 按文件名排序，选择最新的（步数最大的）
                        try:
                            latest_ckpt = max(ckpt_files, key=lambda x: int(x.split('=')[1].split('.')[0]) if '=' in x else 0)
                            rgb_lora_path = os.path.join(lora_dir, latest_ckpt)
                            print(f"✅ 自动发现RGB checkpoint: {rgb_lora_path}")
                            break
                        except Exception as e:
                            print(f"⚠️ 解析checkpoint文件名失败: {e}")
                            # 如果解析失败，使用第一个文件
                            rgb_lora_path = os.path.join(lora_dir, ckpt_files[0])
                            print(f"✅ 使用找到的第一个checkpoint: {rgb_lora_path}")
                            break
                else:
                    rgb_lora_path = lora_dir
                    print(f"✅ 使用指定的checkpoint路径: {rgb_lora_path}")
                    break
    
    # 加载checkpoint权重（如果提供了路径）
    if rgb_lora_path and os.path.exists(rgb_lora_path):
        print(f"正在加载RGB checkpoint权重: {rgb_lora_path}")
        try:
            from peft import PeftModel
            
            # 检查是否是Lightning checkpoint格式 (.ckpt)
            if rgb_lora_path.endswith('.ckpt'):
                print("检测到Lightning checkpoint格式...")
                ckpt = torch.load(rgb_lora_path, map_location='cpu')
                
                if 'state_dict' in ckpt:
                    state_dict = ckpt['state_dict']
                    print(f"Checkpoint包含 {len(state_dict)} 个权重")
                    
                    # 分析checkpoint内容，判断是全量微调还是LoRA
                    lora_keys = [k for k in state_dict.keys() if 'lora' in k.lower()]
                    model_keys = [k for k in state_dict.keys() if k.startswith('model.')]
                    
                    print(f"  - 模型权重: {len(model_keys)} 个")
                    print(f"  - LoRA参数: {len(lora_keys)} 个")
                    
                    success_count = 0
                    
                    # 判断是全量微调还是LoRA
                    if lora_keys:
                        # LoRA格式：包含LoRA参数
                        print("\n检测到LoRA格式checkpoint，加载RGB LoRA权重...")
                        if hasattr(pipeline, 'model'):
                            try:
                                # 提取model相关的权重（包含LoRA）
                                model_state_dict = {}
                                for key, value in state_dict.items():
                                    if key.startswith('model.'):
                                        new_key = key[6:]  # 去掉'model.'前缀
                                        model_state_dict[new_key] = value
                                
                                # 先应用LoRA配置到基础模型
                                from peft import LoraConfig, get_peft_model
                                lora_config = LoraConfig(
                                    r=8,
                                    lora_alpha=8,
                                    target_modules=["to_q", "to_k", "to_v", "to_out.0"],
                                    lora_dropout=0.0,
                                )
                                pipeline.model = get_peft_model(pipeline.model, lora_config)
                                
                                # 加载包含LoRA的权重
                                missing, unexpected = pipeline.model.load_state_dict(
                                    model_state_dict, strict=False)
                                print(f"✅ RGB LoRA权重加载成功")
                                print(f"  - Missing keys: {len(missing)}")
                                print(f"  - Unexpected keys: {len(unexpected)}")
                                success_count += 1
                                
                            except Exception as e:
                                print(f"❌ RGB LoRA权重加载失败: {e}")
                                import traceback
                                traceback.print_exc()
                    else:
                        # 全量微调格式：不包含LoRA参数，直接加载完整模型权重
                        print("\n检测到全量微调格式checkpoint，加载完整模型权重...")
                        if hasattr(pipeline, 'model'):
                            try:
                                # 检查模型的input_size是否与checkpoint匹配
                                if hasattr(pipeline.model, 'input_size'):
                                    model_input_size = pipeline.model.input_size
                                    print(f"  当前模型 input_size: {model_input_size}")
                                    
                                    # 从checkpoint中推断input_size（通过检查权重形状）
                                    # 通常可以通过检查 x_embedder.pos_embed 的形状来推断
                                    sample_key = None
                                    for key in state_dict.keys():
                                        if key.startswith('model.') and 'pos_embed' in key:
                                            sample_key = key
                                            break
                                    
                                    if sample_key:
                                        ckpt_input_size = state_dict[sample_key].shape[1] if len(state_dict[sample_key].shape) > 1 else None
                                        if ckpt_input_size and ckpt_input_size != model_input_size:
                                            print(f"  ⚠️  警告: checkpoint的input_size ({ckpt_input_size}) 与当前模型的input_size ({model_input_size}) 不匹配")
                                            print(f"  这通常意味着checkpoint是基于不同的预训练模型训练的")
                                            print(f"  如果训练时使用的是 hunyuandit-multiview-rgb-finetuning-flowmatching-dinol518-bf16-lr1e5-4096-mv.yaml")
                                            print(f"  该配置使用 tencent/Hunyuan3D-2mv (input_size=4096)")
                                            print(f"  请确保使用 --model_path tencent/Hunyuan3D-2mv --subfolder hunyuan3d-dit-v2-mv")
                                
                                # 提取model相关的权重（完整权重，不含LoRA）
                                model_state_dict = {}
                                for key, value in state_dict.items():
                                    if key.startswith('model.'):
                                        new_key = key[6:]  # 去掉'model.'前缀
                                        model_state_dict[new_key] = value
                                
                                # 直接加载完整模型权重（不使用LoRA）
                                missing, unexpected = pipeline.model.load_state_dict(
                                    model_state_dict, strict=False)
                                print(f"✅ 全量微调权重加载成功")
                                print(f"  - Missing keys: {len(missing)}")
                                print(f"  - Unexpected keys: {len(unexpected)}")
                                success_count += 1
                                
                            except RuntimeError as e:
                                error_msg = str(e)
                                if "size mismatch" in error_msg.lower():
                                    print(f"❌ 全量微调权重加载失败: 参数形状不匹配")
                                    print(f"  错误信息: {error_msg[:500]}")  # 只显示前500个字符
                                    print(f"  这通常意味着checkpoint是基于不同的预训练模型训练的")
                                    print(f"  如果训练时使用的是 hunyuandit-multiview-rgb-finetuning-flowmatching-dinol518-bf16-lr1e5-4096-mv.yaml")
                                    print(f"  该配置使用 tencent/Hunyuan3D-2mv (input_size=4096)")
                                    print(f"  请确保使用 --model_path tencent/Hunyuan3D-2mv --subfolder hunyuan3d-dit-v2-mv")
                                else:
                                    print(f"❌ 全量微调权重加载失败: {e}")
                                import traceback
                                traceback.print_exc()
                            except Exception as e:
                                print(f"❌ 全量微调权重加载失败: {e}")
                                import traceback
                                traceback.print_exc()
                    
                    if success_count > 0:
                        checkpoint_type = "LoRA" if lora_keys else "全量微调"
                        print(f"\n✅ 从Lightning checkpoint成功加载RGB {checkpoint_type}权重")
                    else:
                        print("\n❌ 没有成功加载RGB权重")
                        
                else:
                    print("❌ Checkpoint格式无效，缺少state_dict")
            
            elif os.path.isdir(rgb_lora_path):
                # 标准的PEFT格式目录
                print("检测到PEFT目录格式，使用标准LoRA加载方式...")
                
                if hasattr(pipeline, 'model'):
                    try:
                        print("正在加载RGB LoRA权重到主DiT模型...")
                        print(f"  模型类型: {type(pipeline.model)}")
                        print(f"  LoRA路径: {rgb_lora_path}")
                        
                        # 直接对 pipeline.model 应用 LoRA
                        pipeline.model = PeftModel.from_pretrained(
                            pipeline.model, rgb_lora_path)
                        
                        print("✅ RGB LoRA权重加载成功")
                    except Exception as e:
                        print(f"❌ RGB LoRA权重加载失败: {e}")
                        import traceback
                        traceback.print_exc()
                else:
                    print("❌ Pipeline没有model属性")
            else:
                print(f"❌ 不支持的RGB checkpoint格式: {rgb_lora_path}")
                
        except Exception as e:
            import traceback
            print(f"❌ 加载RGB checkpoint权重时发生异常: {e}")
            print("详细错误信息:")
            traceback.print_exc()
            print("将使用基础RGB条件模型")
    else:
        if rgb_lora_path:
            print(f"⚠️ RGB checkpoint路径不存在: {rgb_lora_path}")
        print("使用基础RGB条件模型（未加载checkpoint权重）")
    
    # 配置多视图RGB模式
    if hasattr(pipeline, 'image_processor'):
        pipeline.image_processor = MVImageProcessorV2(size=518)
        print("✅ 已设置MVImageProcessorV2用于多视图RGB处理")
        
        # 替换conditioner中的encoder为DinoImageEncoderMV
        if hasattr(pipeline, 'conditioner'):
            if hasattr(pipeline.conditioner, 'main_image_encoder'):
                current_encoder = pipeline.conditioner.main_image_encoder
                
                if not isinstance(current_encoder, DinoImageEncoderMV):
                    print(f"检测到当前encoder类型: {type(current_encoder).__name__}")
                    print("正在替换为DinoImageEncoderMV...")
                    
                    # 创建新的DinoImageEncoderMV encoder
                    new_encoder = DinoImageEncoderMV(
                        version='facebook/dinov2-large',
                        image_size=518,
                        use_cls_token=True,
                        view_num=num_views
                    )
                    
                    # 尝试复用原encoder的权重
                    if hasattr(current_encoder, 'model') and hasattr(new_encoder, 'model'):
                        try:
                            new_encoder.model.load_state_dict(current_encoder.model.state_dict())
                            print("✅ 已复用原encoder的模型权重")
                        except Exception as e:
                            print(f"⚠️ 无法复用原encoder权重，使用默认权重: {e}")
                    
                    # 获取pipeline的dtype
                    pipeline_dtype = getattr(pipeline, 'dtype', torch.float16)
                    if not hasattr(pipeline, 'dtype'):
                        if hasattr(pipeline, 'model') and hasattr(pipeline.model, 'dtype'):
                            pipeline_dtype = pipeline.model.dtype
                    
                    # 将新encoder移到相同设备和数据类型
                    new_encoder = new_encoder.to(device, dtype=pipeline_dtype)
                    
                    # 替换encoder
                    pipeline.conditioner.main_image_encoder = new_encoder
                    print(f"✅ 已成功替换为DinoImageEncoderMV (视图数量: {num_views})")
    
    # 低显存模式
    if low_vram and device.startswith('cuda'):
        print("启用低显存模式: 使用 CPU offload...")
        try:
            if hasattr(pipeline, 'components'):
                pipeline.enable_model_cpu_offload(gpu_id=int(device.split(':')[1]) if ':' in device else 0)
                print("✅ CPU offload 已启用")
            else:
                print("⚠️ Pipeline不支持CPU offload")
        except Exception as e:
            print(f"⚠️ CPU offload启用失败: {e}")
    
    return pipeline


def generate_mesh(pipeline, images, params, device):
    """
    执行推理生成mesh
    
    Args:
        pipeline: Hunyuan3DDiTFlowMatchingPipeline实例
        images (dict): 多视图图片字典
        params (dict): 推理参数
        device (str): 设备
        
    Returns:
        trimesh.Trimesh: 生成的mesh对象
    """
    print("\n开始推理...")
    start_time = time.time()
    
    # 准备生成器
    generator = torch.Generator()
    generator = generator.manual_seed(int(params['seed']))
    
    # 准备模型输入
    model_inputs = {
        'image': images,
        'num_inference_steps': params['steps'],
        'guidance_scale': params['guidance_scale'],
        'generator': generator,
        'octree_resolution': params['octree_resolution'],
        'num_chunks': params['num_chunks'],
        'output_type': 'mesh'
    }
    
    print(f"推理参数:")
    print(f"  - 推理步数: {params['steps']}")
    print(f"  - 引导尺度: {params['guidance_scale']}")
    print(f"  - 八叉树分辨率: {params['octree_resolution']}")
    print(f"  - 随机种子: {params['seed']}")
    
    # 执行推理
    try:
        outputs = pipeline(**model_inputs)
        inference_time = time.time() - start_time
        print(f"✅ 推理完成，耗时: {inference_time:.2f}秒")
        
        # 转换为trimesh
        print("正在转换为trimesh格式...")
        mesh = export_to_trimesh(outputs)[0]
        print(f"✅ Mesh转换完成")
        print(f"  - 顶点数: {mesh.vertices.shape[0]}")
        print(f"  - 面数: {mesh.faces.shape[0]}")
        
        return mesh
        
    except Exception as e:
        print(f"❌ 推理失败: {e}")
        import traceback
        traceback.print_exc()
        raise


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='Hunyuan3D 多视图3D生成简化脚本',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python MV3D.py --file input_images
  python MV3D.py --file input_images --output result.obj --steps 50
  python MV3D.py --file input_images --device cuda --low_vram
        """
    )
    
    parser.add_argument('--file', type=str, required=True,
                       help='输入图片文件夹路径（包含000.jpg, 001.jpg, 002.jpg, 003.jpg）')
    parser.add_argument('--output', type=str, default='output.obj',
                       help='输出OBJ文件路径（默认: output.obj）')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='设备 (默认: cuda)')
    parser.add_argument('--steps', type=int, default=50,
                       help='推理步数 (默认: 50)')
    parser.add_argument('--guidance_scale', type=float, default=7.5,
                       help='引导尺度 (默认: 7.5)')
    parser.add_argument('--seed', type=int, default=1234,
                       help='随机种子 (默认: 1234)')
    parser.add_argument('--octree_resolution', type=int, default=256,
                       help='八叉树分辨率 (默认: 256)')
    parser.add_argument('--num_chunks', type=int, default=200000,
                       help='块数量 (默认: 200000)')
    parser.add_argument('--low_vram', action='store_true',
                       help='启用低显存模式')
    parser.add_argument('--model_path', type=str, default='tencent/Hunyuan3D-2mv',
                       help='模型路径 (默认: tencent/Hunyuan3D-2mv)')
    parser.add_argument('--subfolder', type=str, default='hunyuan3d-dit-v2-mv',
                       help='子文件夹 (默认: hunyuan3d-dit-v2-mv)')
    parser.add_argument('--num_views', type=int, default=4,
                       help='视图数量 (默认: 4)')
    parser.add_argument('--enable_multiview_rgb', action='store_true', default=True,
                       help='启用多视图RGB模式 (默认: True)')
    parser.add_argument('--rgb_lora_path', type=str, default=None,
                       help='全量训练后的checkpoint路径 (可选，如果不指定则自动搜索)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Hunyuan3D 多视图3D生成")
    print("=" * 60)
    print()
    
    # 检查设备
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("⚠️ CUDA不可用，切换到CPU模式")
        args.device = 'cpu'
    
    try:
        # 1. 加载图片
        print("步骤 1/3: 加载多视图图片...")
        images = load_multiview_images(args.file)
        print(f"✅ 成功加载 {len(images)} 张图片\n")
        
        # 3. 初始化模型
        print("步骤 2/3: 初始化模型...")
        pipeline = initialize_model(
            model_path=args.model_path,
            subfolder=args.subfolder,
            device=args.device,
            num_views=args.num_views,
            low_vram=args.low_vram,
            rgb_lora_path=args.rgb_lora_path
        )
        print(f"✅ 模型初始化完成\n")
        
        # 4. 执行推理
        print("步骤 3/3: 执行推理...")
        params = {
            'steps': args.steps,
            'guidance_scale': args.guidance_scale,
            'seed': args.seed,
            'octree_resolution': args.octree_resolution,
            'num_chunks': args.num_chunks,
        }
        
        mesh = generate_mesh(pipeline, images, params, args.device)
        
        # 6. 导出mesh为OBJ格式
        # 确保输出文件扩展名为.obj
        output_path = args.output
        if not output_path.lower().endswith('.obj'):
            # 如果没有扩展名或扩展名不是.obj，添加.obj扩展名
            base_path = os.path.splitext(output_path)[0]
            output_path = base_path + '.obj'
            print(f"⚠️ 输出文件名已自动调整为: {output_path}")
        
        print(f"\n正在导出mesh到: {output_path}")
        mesh.export(output_path)
        print(f"✅ 导出完成: {output_path}")
        
        # 清理显存
        if args.device == 'cuda':
            torch.cuda.empty_cache()
        
        print("\n" + "=" * 60)
        print("✅ 完成！")
        print("=" * 60)
        print(f"输出文件: {os.path.abspath(output_path)}")
        print(f"顶点数: {mesh.vertices.shape[0]}")
        print(f"面数: {mesh.faces.shape[0]}")
        
    except FileNotFoundError as e:
        print(f"\n❌ 错误: {e}")
        sys.exit(1)
    except RuntimeError as e:
        print(f"\n❌ 运行时错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 未知错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
