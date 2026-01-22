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
import glob
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
    从文件夹加载4张图片：001.png, 002.png, 003.png, 004.png
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
        'front': image_dir / '000.png',
        'left': image_dir / '001.png',
        'back': image_dir / '002.png',
        'right': image_dir / '003.png',
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
    
    # 自动搜索checkpoint（如果未指定）
    if rgb_lora_path is None:
        # 优先查找推理格式 checkpoint（推荐）
        default_checkpoint_dirs = [
            # 推理格式 checkpoint 目录（优先）
            "./hy3dshape/output_folder/dit/multiview_rgb_finetuning_mv_inference_checkpoints",
            # Lightning checkpoint 目录（备选）
            "./hy3dshape/output_folder/dit/multiview_rgb_finetuning_mv/ckpt",
        ]
        
        for checkpoint_dir in default_checkpoint_dirs:
            if os.path.exists(checkpoint_dir):
                if os.path.isdir(checkpoint_dir):
                    # 检查是否是推理格式 checkpoint 目录（包含 inference_step_* 子目录）
                    inference_dirs = [d for d in os.listdir(checkpoint_dir) 
                                     if os.path.isdir(os.path.join(checkpoint_dir, d)) 
                                     and 'inference_step' in d]
                    if inference_dirs:
                        # 选择最新的推理格式 checkpoint
                        try:
                            latest_inference_dir = max(inference_dirs, 
                                                      key=lambda x: int(x.split('_')[-1]) if x.split('_')[-1].isdigit() else 0)
                            rgb_lora_path = os.path.join(checkpoint_dir, latest_inference_dir)
                            print(f"✅ 自动发现推理格式 checkpoint: {rgb_lora_path}")
                            break
                        except Exception as e:
                            print(f"⚠️ 解析推理格式 checkpoint 目录名失败: {e}")
                            rgb_lora_path = os.path.join(checkpoint_dir, inference_dirs[0])
                            print(f"✅ 使用找到的第一个推理格式 checkpoint: {rgb_lora_path}")
                            break
                    else:
                        # 检查是否是 Lightning checkpoint 目录（包含 .ckpt 文件）
                        ckpt_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.ckpt')]
                        if ckpt_files:
                            try:
                                latest_ckpt = max(ckpt_files, 
                                               key=lambda x: int(x.split('=')[1].split('.')[0]) if '=' in x else 0)
                                rgb_lora_path = os.path.join(checkpoint_dir, latest_ckpt)
                                print(f"✅ 自动发现 Lightning checkpoint: {rgb_lora_path}")
                                break
                            except Exception as e:
                                print(f"⚠️ 解析 checkpoint 文件名失败: {e}")
                                rgb_lora_path = os.path.join(checkpoint_dir, ckpt_files[0])
                                print(f"✅ 使用找到的第一个 checkpoint: {rgb_lora_path}")
                                break
                else:
                    rgb_lora_path = checkpoint_dir
                    print(f"✅ 使用指定的 checkpoint 路径: {rgb_lora_path}")
                    break
    
    # 如果提供了 checkpoint 路径，直接使用 from_single_file 加载（不加载预训练模型）
    if rgb_lora_path and os.path.exists(rgb_lora_path):
        print(f"正在从 checkpoint 加载模型: {rgb_lora_path}")
        try:
            # 检查是否是推理格式 checkpoint 目录还是 Lightning checkpoint 文件
            if os.path.isdir(rgb_lora_path):
                # 推理格式 checkpoint 目录
                ckpt_file = os.path.join(rgb_lora_path, 'model.ckpt')
                config_file = os.path.join(rgb_lora_path, 'config.yaml')
                
                if not os.path.exists(ckpt_file):
                    raise FileNotFoundError(f"缺少文件: {ckpt_file}")
                if not os.path.exists(config_file):
                    raise FileNotFoundError(f"缺少文件: {config_file}")
                
                print(f"✅ 检测到推理格式 checkpoint 目录: {rgb_lora_path}")
                print("正在从推理格式 checkpoint 加载完整模型（不加载预训练模型）...")
                
                # 直接使用 from_single_file 加载，会自动检测格式
                pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_single_file(
                    ckpt_file,
                    config_file,
                    device=device,
                    dtype=torch.float16,
                    use_safetensors=False,
                )
                print("✅ 成功从推理格式 checkpoint 加载完整模型")
                
            elif os.path.isfile(rgb_lora_path) and rgb_lora_path.endswith('.ckpt'):
                # Lightning checkpoint 文件
                # 需要找到对应的 config.yaml
                ckpt_dir = os.path.dirname(rgb_lora_path)
                config_file = os.path.join(ckpt_dir, 'config.yaml')
                
                # 如果同目录下没有 config.yaml，尝试在多个位置查找
                if not os.path.exists(config_file):
                    possible_config_paths = [
                        os.path.join(os.path.dirname(ckpt_dir), 'config.yaml'),  # 父目录
                        os.path.join(ckpt_dir, '..', 'config.yaml'),  # 父目录（相对路径）
                        os.path.join(os.path.dirname(os.path.dirname(ckpt_dir)), 'config.yaml'),  # 祖父目录
                    ]
                    # 尝试查找训练配置文件（通配符）
                    config_patterns = [
                        os.path.join(os.path.dirname(ckpt_dir), 'hunyuandit-*.yaml'),
                        os.path.join(os.path.dirname(os.path.dirname(ckpt_dir)), 'hunyuandit-*.yaml'),
                    ]
                    for pattern in config_patterns:
                        matches = glob(pattern)
                        if matches:
                            config_file = matches[0]
                            break
                    
                    for possible_path in possible_config_paths:
                        if os.path.exists(possible_path):
                            config_file = possible_path
                            break
                
                if os.path.exists(config_file):
                    print(f"✅ 检测到 Lightning checkpoint 文件: {rgb_lora_path}")
                    print(f"✅ 找到配置文件: {config_file}")
                    print("正在从 Lightning checkpoint 加载模型（自动检测格式）...")
                    
                    # from_single_file 会自动检测格式（Lightning 或推理格式）
                    pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_single_file(
                        rgb_lora_path,
                        config_file,
                        device=device,
                        dtype=torch.float16,
                        use_safetensors=False,
                    )
                    print("✅ 成功从 Lightning checkpoint 加载模型")
                else:
                    # 如果找不到 config.yaml，回退到使用预训练模型 + 加载权重的方式
                    print(f"⚠️  警告: 找不到 config.yaml 文件")
                    print(f"   将回退到使用预训练模型 + 加载 checkpoint 权重的方式")
                    
                    # 先加载预训练模型
                    is_mv_model = 'Hunyuan3D-2mv' in model_path or 'mv' in model_path.lower() or 'hunyuan3d-dit-v2-mv' in subfolder
                    use_safetensors = is_mv_model
                    
                    print("正在加载预训练模型...")
                    pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
                        model_path,
                        subfolder=subfolder,
                        use_safetensors=use_safetensors,
                        device=device,
                    )
                    print("✅ 预训练模型加载完成")
                    
                    # 然后加载 Lightning checkpoint 权重
                    print("正在从 Lightning checkpoint 加载权重...")
                    # 使用安全的加载函数
                    def safe_torch_load(file_path, map_location='cpu', weights_only_preferred=True):
                        try:
                            return torch.load(file_path, map_location=map_location, weights_only=True)
                        except Exception as e:
                            if 'PosixPath' in str(e) or 'pathlib' in str(e):
                                try:
                                    from pathlib import PosixPath
                                    import torch.serialization
                                    torch.serialization.add_safe_globals([PosixPath])
                                    return torch.load(file_path, map_location=map_location, weights_only=True)
                                except:
                                    return torch.load(file_path, map_location=map_location, weights_only=False)
                            else:
                                return torch.load(file_path, map_location=map_location, weights_only=False)
                    
                    ckpt = safe_torch_load(rgb_lora_path, map_location='cpu', weights_only_preferred=True)
                    
                    if 'state_dict' in ckpt:
                        state_dict = ckpt['state_dict']
                        
                        # 加载 model (DiT) 权重
                        if hasattr(pipeline, 'model') and pipeline.model is not None:
                            model_state_dict = {}
                            for key, value in state_dict.items():
                                if key.startswith('model.'):
                                    new_key = key[6:]  # 去掉 'model.' 前缀
                                    model_state_dict[new_key] = value
                            if model_state_dict:
                                missing, unexpected = pipeline.model.load_state_dict(model_state_dict, strict=False)
                                print(f"✅ DiT 模型权重加载完成: {len(model_state_dict)} 个权重")
                                if missing:
                                    print(f"  - Missing keys: {len(missing)}")
                                if unexpected:
                                    print(f"  - Unexpected keys: {len(unexpected)}")
                        
                        # 加载 VAE 权重
                        if hasattr(pipeline, 'vae') and pipeline.vae is not None:
                            vae_state_dict = {}
                            for key, value in state_dict.items():
                                if key.startswith('first_stage_model.'):
                                    new_key = key[19:]  # 去掉 'first_stage_model.' 前缀
                                    vae_state_dict[new_key] = value
                            if vae_state_dict:
                                missing, unexpected = pipeline.vae.load_state_dict(vae_state_dict, strict=False)
                                print(f"✅ VAE 权重加载完成: {len(vae_state_dict)} 个权重")
                                if missing:
                                    print(f"  - Missing keys: {len(missing)}")
                                if unexpected:
                                    print(f"  - Unexpected keys: {len(unexpected)}")
                        
                        # 加载 Conditioner 权重
                        if hasattr(pipeline, 'conditioner') and pipeline.conditioner is not None:
                            conditioner_state_dict = {}
                            for key, value in state_dict.items():
                                if key.startswith('cond_stage_model.'):
                                    new_key = key[18:]  # 去掉 'cond_stage_model.' 前缀
                                    conditioner_state_dict[new_key] = value
                            if conditioner_state_dict:
                                missing, unexpected = pipeline.conditioner.load_state_dict(conditioner_state_dict, strict=False)
                                print(f"✅ Conditioner 权重加载完成: {len(conditioner_state_dict)} 个权重")
                                if missing:
                                    print(f"  - Missing keys: {len(missing)}")
                                if unexpected:
                                    print(f"  - Unexpected keys: {len(unexpected)}")
                        
                        print("✅ 成功从 Lightning checkpoint 加载权重（使用预训练模型作为基础）")
                    else:
                        raise ValueError(f"Lightning checkpoint 格式无效: 缺少 'state_dict' 键")
            else:
                raise ValueError(f"不支持的 checkpoint 格式: {rgb_lora_path}")
                
        except Exception as e:
            import traceback
            print(f"❌ 从 checkpoint 加载失败: {e}")
            print("详细错误信息:")
            traceback.print_exc()
            print("将回退到使用预训练模型")
            # 回退到预训练模型
            is_mv_model = 'Hunyuan3D-2mv' in model_path or 'mv' in model_path.lower() or 'hunyuan3d-dit-v2-mv' in subfolder
            use_safetensors = is_mv_model
            pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
                model_path,
                subfolder=subfolder,
                use_safetensors=use_safetensors,
                device=device,
            )
            print("✅ 使用预训练模型（未加载 checkpoint 权重）")
    else:
        # 如果没有提供 checkpoint 路径，使用预训练模型
        if rgb_lora_path:
            print(f"⚠️ RGB checkpoint 路径不存在: {rgb_lora_path}")
        print("使用预训练模型（未加载 checkpoint 权重）")
        
        is_mv_model = 'Hunyuan3D-2mv' in model_path or 'mv' in model_path.lower() or 'hunyuan3d-dit-v2-mv' in subfolder
        use_safetensors = is_mv_model
        
        pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            model_path,
            subfolder=subfolder,
            use_safetensors=use_safetensors,
            device=device,
        )
        if use_safetensors:
            print("✅ 使用safetensors格式加载Hunyuan3D-2mv模型成功")
    
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
                       help='输入图片文件夹路径（包含000.png, 001.png, 002.png, 003.png）')
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
