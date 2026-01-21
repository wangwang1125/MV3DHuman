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


import os
import random
import shutil
import subprocess
import time
from glob import glob
from pathlib import Path

import gradio as gr
import torch
import trimesh
import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
import uuid
import numpy as np

from hy3dshape.utils import logger
from hy3dpaint.convert_utils import create_glb_with_pbr_materials


MAX_SEED = 1e7
ENV = "Local" # "Huggingface"
if ENV == 'Huggingface':
    """
    Setup environment for running on Huggingface platform.

    This block performs the following:
    - Changes directory to the differentiable renderer folder and runs a shell 
        script to compile the mesh painter.
    - Installs a custom rasterizer wheel package via pip.

    Note:
        This setup assumes the script is running in the Huggingface environment 
        with the specified directory structure.
    """
    import os, spaces, subprocess, sys, shlex
    print("cd /home/user/app/hy3dgen/texgen/differentiable_renderer/ && bash compile_mesh_painter.sh")
    os.system("cd /home/user/app/hy3dgen/texgen/differentiable_renderer/ && bash compile_mesh_painter.sh")
    print('install custom')
    subprocess.run(shlex.split("pip install custom_rasterizer-0.1-cp310-cp310-linux_x86_64.whl"),
                   check=True)
else:
    """
    Define a dummy `spaces` module with a GPU decorator class for local environment.

    The GPU decorator is a no-op that simply returns the decorated function unchanged.
    This allows code that uses the `spaces.GPU` decorator to run without modification locally.
    """
    class spaces:
        class GPU:
            def __init__(self, duration=60):
                self.duration = duration
            def __call__(self, func):
                return func 

def get_example_img_list():
    """
    Load and return a sorted list of example image file paths.

    Searches recursively for PNG images under the './assets/example_images/' directory.

    Returns:
        list[str]: Sorted list of file paths to example PNG images.
    """
    print('Loading example img list ...')
    return sorted(glob('./assets/example_images/**/*.png', recursive=True))


def get_example_txt_list():
    """
    Load and return a list of example text prompts.

    Reads lines from the './assets/example_prompts.txt' file, stripping whitespace.

    Returns:
        list[str]: List of example text prompts.
    """
    print('Loading example txt list ...')
    txt_list = list()
    for line in open('./assets/example_prompts.txt', encoding='utf-8'):
        txt_list.append(line.strip())
    return txt_list


def gen_save_folder(max_size=200):
    """
    Generate a new save folder inside SAVE_DIR, maintaining a maximum number of folders.

    If the number of existing folders in SAVE_DIR exceeds `max_size`, the oldest folder is removed.

    Args:
        max_size (int, optional): Maximum number of folders to keep in SAVE_DIR. Defaults to 200.

    Returns:
        str: Path to the newly created save folder.
    """
    os.makedirs(SAVE_DIR, exist_ok=True)
    dirs = [f for f in Path(SAVE_DIR).iterdir() if f.is_dir()]
    if len(dirs) >= max_size:
        oldest_dir = min(dirs, key=lambda x: x.stat().st_ctime)
        shutil.rmtree(oldest_dir)
        print(f"Removed the oldest folder: {oldest_dir}")
    new_folder = os.path.join(SAVE_DIR, str(uuid.uuid4()))
    os.makedirs(new_folder, exist_ok=True)
    print(f"Created new folder: {new_folder}")
    return new_folder


# Removed complex PBR conversion functions - using simple trimesh-based conversion
def export_mesh(mesh, save_folder, textured=False, type='glb'):
    """
    Export a mesh to a file in the specified folder, optionally including textures.

    Args:
        mesh (trimesh.Trimesh): The mesh object to export.
        save_folder (str): Directory path where the mesh file will be saved.
        textured (bool, optional): Whether to include textures/normals in the export. Defaults to False.
        type (str, optional): File format to export ('glb' or 'obj' supported). Defaults to 'glb'.

    Returns:
        str: The full path to the exported mesh file.
    """
    if textured:
        path = os.path.join(save_folder, f'textured_mesh.{type}')
    else:
        path = os.path.join(save_folder, f'white_mesh.{type}')
    if type not in ['glb', 'obj']:
        mesh.export(path)
    else:
        mesh.export(path, include_normals=textured)
    return path




def quick_convert_with_obj2gltf(obj_path: str, glb_path: str) -> bool:
    # 执行转换
    textures = {
        'albedo': obj_path.replace('.obj', '.jpg'),
        'metallic': obj_path.replace('.obj', '_metallic.jpg'),
        'roughness': obj_path.replace('.obj', '_roughness.jpg')
        }
    create_glb_with_pbr_materials(obj_path, textures, glb_path)
            


def randomize_seed_fn(seed: int, randomize_seed: bool) -> int:
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    return seed


def load_depth_from_16bit_png(depth_file_path, target_size=518):
    """
    加载16位PNG深度图并处理为与训练集兼容的格式
    
    Args:
        depth_file_path (str): 深度图文件路径
        target_size (int): 目标图像尺寸，默认518
    
    Returns:
        torch.Tensor: 处理后的深度图张量，形状为 (1, 1, H, W)
    """
    import cv2
    from PIL import Image
    
    try:
        # 使用OpenCV加载16位深度图
        depth = cv2.imread(depth_file_path, cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYDEPTH)
        
        if depth is None:
            # 尝试使用PIL加载
            depth_pil = Image.open(depth_file_path)
            depth = np.array(depth_pil)
        
        if depth is None:
            raise ValueError(f"无法加载深度图: {depth_file_path}")
        
        # 确保是单通道
        if len(depth.shape) == 3:
            if depth.shape[2] == 1:
                depth = depth[:, :, 0]
            else:
                # 如果是多通道，取第一个通道
                depth = depth[:, :, 0]
        
        print(f"原始深度图形状: {depth.shape}, 数据类型: {depth.dtype}")
        print(f"深度值范围: [{depth.min()}, {depth.max()}]")
        
        # 转换为浮点数
        if depth.dtype == np.uint16:
            # 16位深度图，通常需要除以65535来归一化到[0,1]
            depth = depth.astype(np.float32) / 65535.0
        elif depth.dtype == np.uint8:
            # 8位深度图
            depth = depth.astype(np.float32) / 255.0
        else:
            # 已经是浮点数格式
            depth = depth.astype(np.float32)
        
        # 过滤无效深度值（与训练集处理保持一致）
        depth[depth > 1e9] = 0
        depth[np.isnan(depth)] = 0
        depth[np.isinf(depth)] = 0
        
        # 归一化到[0, 1]范围（与训练集处理保持一致）
        if depth.max() > depth.min():
            depth = (depth - depth.min()) / (depth.max() - depth.min())
        
        print(f"归一化后深度值范围: [{depth.min():.6f}, {depth.max():.6f}]")
        
        # 调整大小到目标尺寸
        if depth.shape[0] != target_size or depth.shape[1] != target_size:
            depth = cv2.resize(depth, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
            print(f"深度图已调整到尺寸: {depth.shape}")
        
        # 转换为torch张量并添加批次和通道维度
        depth_tensor = torch.FloatTensor(depth).unsqueeze(0).unsqueeze(0)  # 形状: (1, 1, H, W)
        
        print(f"最终深度图张量形状: {depth_tensor.shape}")
        
        return depth_tensor
        
    except Exception as e:
        print(f"加载深度图时发生错误: {e}")
        raise gr.Error(f"无法处理深度图文件: {str(e)}")


def load_multiview_depths(depth_files_dict, target_size=518, num_views=None):
    """
    加载多视图深度图（适用于多视图深度训练模式）
    
    Args:
        depth_files_dict: 包含各个视图深度图文件的字典 {'front': file, 'right': file, 'back': file, 'left': file}
        target_size: 目标图像尺寸
        num_views: 期望的视图数量，如果为None则使用实际提供的视图数量
    
    Returns:
        torch.Tensor: 多视图深度图张量，形状为 (1, num_views, 1, H, W)
                      其中 1 是 batch_size，num_views 是视图数量
    """
    # 根据num_views参数确定视图顺序
    if num_views == 2:
        # 双视图模式：front, right
        view_order = ['front', 'right']
    elif num_views == 3:
        # 三视图模式：front, right, back
        view_order = ['front', 'right', 'back']
    else:
        # 默认四视图模式：front, right, back, left
        view_order = ['front', 'right', 'back', 'left']
    
    depth_tensors = []
    
    for view_name in view_order:
        if view_name in depth_files_dict and depth_files_dict[view_name] is not None:
            depth_file = depth_files_dict[view_name]
            file_path = depth_file.name if hasattr(depth_file, 'name') else depth_file
            
            # 加载单个深度图
            depth_tensor = load_depth_from_16bit_png(file_path, target_size)  # (1, 1, H, W)
            depth_tensors.append(depth_tensor.squeeze(0))  # (1, H, W)
        else:
            # 如果某个视图没有深度图，创建零深度图作为占位符
            print(f"警告：视图 {view_name} 没有深度图，使用零深度图")
            depth_tensors.append(torch.zeros(1, target_size, target_size))
    
    # 堆叠所有视图的深度图
    multiview_depth = torch.stack(depth_tensors, dim=0)  # (num_views, 1, H, W)
    # 添加 batch 维度以匹配 ControlNet 的期望输入格式
    multiview_depth = multiview_depth.unsqueeze(0)  # (1, num_views, 1, H, W)
    print(f"多视图深度图张量形状: {multiview_depth.shape} (视图数量: {len(view_order)})")
    
    return multiview_depth


def load_multiview_normals(normal_files_dict, target_size=518, num_views=None):
    """
    加载多视图法线图（适用于多视图法线训练模式）
    
    Args:
        normal_files_dict: 包含各个视图法线图文件的字典 {'front': file, 'right': file, 'back': file, 'left': file}
        target_size: 目标图像尺寸，默认518
        num_views: 期望的视图数量，如果为None则使用实际提供的视图数量
    
    Returns:
        dict: 包含法线图和遮罩的字典
            - 'normal': torch.Tensor, 形状为 (1, num_views, 3, H, W)，RGB三通道法线图
            - 'normal_mask': torch.Tensor, 形状为 (1, num_views, 1, H, W)，法线图遮罩
    """
    from PIL import Image
    import cv2
    
    # 根据num_views参数确定视图顺序
    if num_views == 2:
        # 双视图模式：front, right
        view_order = ['front', 'right']
    elif num_views == 3:
        # 三视图模式：front, right, back
        view_order = ['front', 'right', 'back']
    else:
        # 默认四视图模式：front, right, back, left
        view_order = ['front', 'right', 'back', 'left']
    
    normal_tensors = []
    mask_tensors = []
    
    for view_name in view_order:
        if view_name in normal_files_dict and normal_files_dict[view_name] is not None:
            normal_file = normal_files_dict[view_name]
            file_path = normal_file.name if hasattr(normal_file, 'name') else normal_file
            
            try:
                # 加载法线图（RGB三通道PNG）
                normal_img = Image.open(file_path).convert('RGB')
                
                # 调整大小到目标尺寸
                if normal_img.size[0] != target_size or normal_img.size[1] != target_size:
                    normal_img = normal_img.resize((target_size, target_size), Image.BILINEAR)
                
                # 转换为numpy数组并归一化到[0,1]
                normal_array = np.array(normal_img).astype(np.float32) / 255.0
                
                # 转换为torch张量并调整维度顺序 (H, W, C) -> (C, H, W)
                normal_tensor = torch.FloatTensor(normal_array).permute(2, 0, 1)  # (3, H, W)
                
                # 创建遮罩：检查法线图是否有效（非零像素）
                # 法线图通常是RGB格式，检查是否有非零像素
                valid_mask = (normal_tensor.sum(dim=0) > 0.01).float().unsqueeze(0)  # (1, H, W)
                
                normal_tensors.append(normal_tensor)  # (3, H, W)
                mask_tensors.append(valid_mask)  # (1, H, W)
                
                print(f"✅ 加载视图 {view_name} 法线图: {normal_tensor.shape}, 遮罩: {valid_mask.shape}")
                
            except Exception as e:
                print(f"❌ 加载视图 {view_name} 法线图失败: {e}")
                # 创建零法线图和遮罩作为占位符
                normal_tensors.append(torch.zeros(3, target_size, target_size))
                mask_tensors.append(torch.zeros(1, target_size, target_size))
        else:
            # 如果某个视图没有法线图，创建零法线图和遮罩作为占位符
            print(f"⚠️  警告：视图 {view_name} 没有法线图，使用零法线图")
            normal_tensors.append(torch.zeros(3, target_size, target_size))
            mask_tensors.append(torch.zeros(1, target_size, target_size))
    
    # 堆叠所有视图的法线图和遮罩
    multiview_normal = torch.stack(normal_tensors, dim=0)  # (num_views, 3, H, W)
    multiview_mask = torch.stack(mask_tensors, dim=0)  # (num_views, 1, H, W)
    
    # 添加 batch 维度
    multiview_normal = multiview_normal.unsqueeze(0)  # (1, num_views, 3, H, W)
    multiview_mask = multiview_mask.unsqueeze(0)  # (1, num_views, 1, H, W)
    
    print(f"✅ 多视图法线图处理完成，形状: {multiview_normal.shape} (视图数量: {len(view_order)})")
    
    return {
        'normal': multiview_normal,
        'normal_mask': multiview_mask
    }


def validate_depth_file(file_path):
    """
    验证上传的深度图文件是否有效
    
    Args:
        file_path (str): 文件路径
    
    Returns:
        bool: 文件是否有效
    """
    if not os.path.exists(file_path):
        return False
    
    try:
        # 检查文件扩展名
        ext = os.path.splitext(file_path)[1].lower()
        if ext not in ['.png', '.tiff', '.tif']:
            return False
        
        # 尝试加载文件
        if ext == '.png':
            from PIL import Image
            img = Image.open(file_path)
            # 检查是否为16位图像
            return img.mode in ['I;16', 'L', 'I']
        
        return True
        
    except:
        return False


def enable_cpu_offload_for_pipeline(pipeline, gpu_id=0):
    """
    为 Hunyuan3DDiTPipeline 启用 CPU offload
    由于该 pipeline 没有 components 属性，需要手动处理各个组件
    
    Args:
        pipeline: Hunyuan3DDiTFlowMatchingPipeline 实例
        gpu_id: GPU ID，默认0
    """
    try:
        from accelerate import cpu_offload_with_hook
    except ImportError:
        raise ImportError("`enable_cpu_offload_for_pipeline` requires `accelerate v0.17.0` or higher.")
    
    device = torch.device(f"cuda:{gpu_id}")
    
    # 先将所有模型移到CPU
    if hasattr(pipeline, 'vae') and pipeline.vae is not None:
        pipeline.vae.to("cpu")
    if hasattr(pipeline, 'model') and pipeline.model is not None:
        pipeline.model.to("cpu")
    if hasattr(pipeline, 'conditioner') and pipeline.conditioner is not None:
        pipeline.conditioner.to("cpu")
    if hasattr(pipeline, 'controlnet') and pipeline.controlnet is not None:
        pipeline.controlnet.to("cpu")
    
    # 清理GPU缓存
    torch.cuda.empty_cache()
    
    # 按照 model_cpu_offload_seq 的顺序设置 offload hooks
    # 顺序: conditioner->model->vae
    pipeline._all_hooks = []
    hook = None
    
    # 1. Conditioner
    if hasattr(pipeline, 'conditioner') and pipeline.conditioner is not None:
        if isinstance(pipeline.conditioner, torch.nn.Module):
            _, hook = cpu_offload_with_hook(pipeline.conditioner, device, prev_module_hook=hook)
            pipeline._all_hooks.append(hook)
    
    # 2. Model
    if hasattr(pipeline, 'model') and pipeline.model is not None:
        if isinstance(pipeline.model, torch.nn.Module):
            _, hook = cpu_offload_with_hook(pipeline.model, device, prev_module_hook=hook)
            pipeline._all_hooks.append(hook)
    
    # 3. VAE
    if hasattr(pipeline, 'vae') and pipeline.vae is not None:
        if isinstance(pipeline.vae, torch.nn.Module):
            _, hook = cpu_offload_with_hook(pipeline.vae, device, prev_module_hook=hook)
            pipeline._all_hooks.append(hook)
    
    # ControlNet（如果有）不需要在序列中，因为它会被单独调用
    if hasattr(pipeline, 'controlnet') and pipeline.controlnet is not None:
        if isinstance(pipeline.controlnet, torch.nn.Module):
            _, hook = cpu_offload_with_hook(pipeline.controlnet, device)
            pipeline._all_hooks.append(hook)
    
    pipeline._offload_gpu_id = gpu_id
    print(f"  ✅ CPU offload hooks 已设置，将使用 GPU {gpu_id}")


def setup_multi_gpu_model_parallel(pipeline, gpu_ids):
    """
    将模型的不同组件分配到不同的GPU上，实现模型并行
    注意：不拆分model内部的blocks，而是将整个组件分配到不同GPU
    
    Args:
        pipeline: Hunyuan3DDiTFlowMatchingPipeline 实例
        gpu_ids: GPU ID列表，例如 [0, 1]
    
    Returns:
        pipeline: 配置好的pipeline
    """
    if not isinstance(gpu_ids, list):
        gpu_ids = [int(x.strip()) for x in gpu_ids.split(',') if x.strip()]
    
    num_gpus = len(gpu_ids)
    if num_gpus < 2:
        print(f"⚠️ 需要至少2个GPU才能使用模型并行，当前只有 {num_gpus} 个GPU")
        return pipeline
    
    print(f"🔄 正在配置模型并行，使用GPU: {gpu_ids}")
    
    try:
        # 策略：将大组件分离到不同GPU以最大化显存利用率
        # GPU 0: Conditioner (编码器，相对较小) + ControlNet
        # GPU 1: Model (主模型，占用最多显存) + VAE (解码器)
        
        conditioner_gpu = gpu_ids[0]  # Conditioner放在第一个GPU
        model_gpu = gpu_ids[1] if len(gpu_ids) > 1 else gpu_ids[0]  # Model放在第二个GPU
        vae_gpu = model_gpu  # VAE和Model放在同一个GPU，避免解码时的数据传输
        
        # 将conditioner放在第一个GPU（相对较小）
        if hasattr(pipeline, 'conditioner') and pipeline.conditioner is not None:
            pipeline.conditioner.to(f'cuda:{conditioner_gpu}')
            print(f"  ✅ Conditioner已移至 GPU {conditioner_gpu}")
        
        # 将model放在第二个GPU（最大的组件，需要最多显存）
        if hasattr(pipeline, 'model') and pipeline.model is not None:
            pipeline.model.to(f'cuda:{model_gpu}')
            print(f"  ✅ Model已移至 GPU {model_gpu} (主模型，占用最多显存)")
        
        # 将VAE放在最后一个GPU
        if hasattr(pipeline, 'vae') and pipeline.vae is not None:
            pipeline.vae.to(f'cuda:{vae_gpu}')
            print(f"  ✅ VAE已移至 GPU {vae_gpu}")
        
        # 将controlnet（如果有）放在第一个GPU（和conditioner一起）
        if hasattr(pipeline, 'controlnet') and pipeline.controlnet is not None:
            pipeline.controlnet.to(f'cuda:{conditioner_gpu}')
            print(f"  ✅ ControlNet已移至 GPU {conditioner_gpu}")
        
        # 将scheduler保持在CPU（通常很小）
        # scheduler通常是状态对象，不需要GPU，保持原样即可
        
        pipeline._conditioner_gpu = conditioner_gpu
        pipeline._model_gpu = model_gpu
        pipeline._vae_gpu = vae_gpu
        pipeline._primary_gpu = conditioner_gpu  # 默认主GPU用于输入处理
        
        # 修改pipeline的device属性，确保latents等在正确的GPU上创建
        # 注意：这里设置为model_gpu，因为latents主要用于model
        if hasattr(pipeline, 'device'):
            # 创建一个device对象指向model所在的GPU
            original_device = pipeline.device
            if isinstance(original_device, torch.device):
                pipeline.device = torch.device(f'cuda:{model_gpu}')
            else:
                pipeline.device = f'cuda:{model_gpu}'
            print(f"  ✅ 已将pipeline默认device设置为 GPU {model_gpu} (用于创建latents等tensors)")
        
        # 如果model在GPU 1上，需要包装model来确保所有输入都在正确的GPU上
        if model_gpu != conditioner_gpu and hasattr(pipeline, 'model') and pipeline.model is not None:
            original_model_call = pipeline.model.__call__
            
            def model_wrapper(*args, **kwargs):
                # 将所有输入tensor移动到model所在的GPU
                args_list = []
                for arg in args:
                    if isinstance(arg, torch.Tensor):
                        # 检查设备索引，如果不在model_gpu上就移动
                        if arg.device.index != model_gpu:
                            args_list.append(arg.to(f'cuda:{model_gpu}'))
                        else:
                            args_list.append(arg)
                    else:
                        args_list.append(arg)
                
                # 处理kwargs中的tensors
                new_kwargs = {}
                for k, v in kwargs.items():
                    if isinstance(v, torch.Tensor):
                        if v.device.index != model_gpu:
                            new_kwargs[k] = v.to(f'cuda:{model_gpu}')
                        else:
                            new_kwargs[k] = v
                    elif isinstance(v, dict):
                        # 如果是字典，递归处理其中的tensors
                        processed_dict = {}
                        for dict_k, dict_v in v.items():
                            if isinstance(dict_v, torch.Tensor):
                                if dict_v.device.index != model_gpu:
                                    processed_dict[dict_k] = dict_v.to(f'cuda:{model_gpu}')
                                else:
                                    processed_dict[dict_k] = dict_v
                            else:
                                processed_dict[dict_k] = dict_v
                        new_kwargs[k] = processed_dict
                    else:
                        new_kwargs[k] = v
                
                result = original_model_call(*args_list, **new_kwargs)
                return result
            
            pipeline.model.__call__ = model_wrapper
            print(f"  ✅ 已为Model添加设备自动移动包装（所有输入将自动移动到 GPU {model_gpu}）")
        
        # 如果conditioner和model在不同的GPU上，需要包装conditioner来移动输出
        if conditioner_gpu != model_gpu and hasattr(pipeline, 'conditioner') and pipeline.conditioner is not None:
            original_conditioner_forward = pipeline.conditioner.forward
            original_conditioner_unconditional = pipeline.conditioner.unconditional_embedding
            
            def conditioner_wrapper(*args, **kwargs):
                result = original_conditioner_forward(*args, **kwargs)
                # 将结果移动到model所在的GPU
                if isinstance(result, dict):
                    result = {k: v.to(f'cuda:{model_gpu}') if isinstance(v, torch.Tensor) else v 
                             for k, v in result.items()}
                elif isinstance(result, torch.Tensor):
                    result = result.to(f'cuda:{model_gpu}')
                return result
            
            def unconditional_wrapper(*args, **kwargs):
                result = original_conditioner_unconditional(*args, **kwargs)
                # 将结果移动到model所在的GPU
                if isinstance(result, dict):
                    result = {k: v.to(f'cuda:{model_gpu}') if isinstance(v, torch.Tensor) else v 
                             for k, v in result.items()}
                elif isinstance(result, torch.Tensor):
                    result = result.to(f'cuda:{model_gpu}')
                return result
            
            pipeline.conditioner.forward = conditioner_wrapper
            pipeline.conditioner.unconditional_embedding = unconditional_wrapper
            print(f"  ✅ 已为Conditioner添加设备自动移动包装（输出将自动移动到 GPU {model_gpu}）")
        
        # 如果VAE在不同的GPU上，需要创建一个包装函数来处理设备移动（通常不需要，因为VAE和Model在同一GPU）
        if vae_gpu != model_gpu and hasattr(pipeline, 'vae') and pipeline.vae is not None:
            original_vae_call = pipeline.vae.__call__
            original_latents2mesh = pipeline.vae.latents2mesh
            
            def vae_wrapper(*args, **kwargs):
                # 确保输入在正确的设备上
                args_list = list(args)
                for i, arg in enumerate(args_list):
                    if isinstance(arg, torch.Tensor) and arg.device.index != vae_gpu:
                        args_list[i] = arg.to(f'cuda:{vae_gpu}')
                # 转换kwargs中的tensors
                new_kwargs = {}
                for k, v in kwargs.items():
                    if isinstance(v, torch.Tensor) and v.device.index != vae_gpu:
                        new_kwargs[k] = v.to(f'cuda:{vae_gpu}')
                    else:
                        new_kwargs[k] = v
                result = original_vae_call(*args_list, **new_kwargs)
                return result
            
            def latents2mesh_wrapper(*args, **kwargs):
                # 确保输入在正确的设备上
                args_list = list(args)
                for i, arg in enumerate(args_list):
                    if isinstance(arg, torch.Tensor) and arg.device.index != vae_gpu:
                        args_list[i] = arg.to(f'cuda:{vae_gpu}')
                # 转换kwargs中的tensors
                new_kwargs = {}
                for k, v in kwargs.items():
                    if isinstance(v, torch.Tensor) and v.device.index != vae_gpu:
                        new_kwargs[k] = v.to(f'cuda:{vae_gpu}')
                    else:
                        new_kwargs[k] = v
                result = original_latents2mesh(*args_list, **new_kwargs)
                return result
            
            pipeline.vae.__call__ = vae_wrapper
            pipeline.vae.latents2mesh = latents2mesh_wrapper
            print(f"  ✅ 已为VAE添加设备自动移动包装（自动将数据移动到 GPU {vae_gpu}）")
        
        print(f"✅ 模型并行配置完成")
        print(f"   - GPU {conditioner_gpu}: Conditioner + ControlNet (编码阶段)")
        print(f"   - GPU {model_gpu}: Model + VAE (主计算+解码阶段)")
        
        # 清理所有GPU的缓存
        for gpu_id in gpu_ids:
            torch.cuda.set_device(gpu_id)
            torch.cuda.empty_cache()
        
    except Exception as e:
        print(f"⚠️ 模型并行配置失败: {e}")
        import traceback
        traceback.print_exc()
        print("  回退到单GPU模式")
        # 将所有模型移回主GPU
        if hasattr(pipeline, '_primary_gpu'):
            primary_gpu = pipeline._primary_gpu
            if hasattr(pipeline, 'model') and pipeline.model is not None:
                pipeline.model.to(f'cuda:{primary_gpu}')
            if hasattr(pipeline, 'vae') and pipeline.vae is not None:
                pipeline.vae.to(f'cuda:{primary_gpu}')
            if hasattr(pipeline, 'conditioner') and pipeline.conditioner is not None:
                pipeline.conditioner.to(f'cuda:{primary_gpu}')
    
    return pipeline


def build_model_viewer_html(save_folder, height=660, width=790, textured=False):
    # Remove first folder from path to make relative path
    if textured:
        related_path = f"./textured_mesh.glb"
        template_name = './assets/modelviewer-textured-template.html'
        output_html_path = os.path.join(save_folder, f'textured_mesh.html')
    else:
        related_path = f"./white_mesh.glb"
        template_name = './assets/modelviewer-template.html'
        output_html_path = os.path.join(save_folder, f'white_mesh.html')
    offset = 50 if textured else 10
    with open(os.path.join(CURRENT_DIR, template_name), 'r', encoding='utf-8') as f:
        template_html = f.read()

    with open(output_html_path, 'w', encoding='utf-8') as f:
        template_html = template_html.replace('#height#', f'{height - offset}')
        template_html = template_html.replace('#width#', f'{width}')
        template_html = template_html.replace('#src#', f'{related_path}/')
        f.write(template_html)

    rel_path = os.path.relpath(output_html_path, SAVE_DIR)
    iframe_tag = f'<iframe src="/static/{rel_path}" \
height="{height}" width="100%" frameborder="0"></iframe>'
    print(f'Find html file {output_html_path}, \
{os.path.exists(output_html_path)}, relative HTML path is /static/{rel_path}')

    return f"""
        <div style='height: {height}; width: 100%;'>
        {iframe_tag}
        </div>
    """

@spaces.GPU(duration=60)
def _gen_shape(
    caption=None,
    image=None,
    depth_file=None,  # 单视图深度图文件参数
    mv_image_front=None,
    mv_image_back=None,
    mv_image_left=None,
    mv_image_right=None,
    # 多视图深度图参数
    mv_depth_front=None,
    mv_depth_back=None,
    mv_depth_left=None,
    mv_depth_right=None,
    # 多视图法线图参数
    mv_normal_front=None,
    mv_normal_back=None,
    mv_normal_left=None,
    mv_normal_right=None,
    steps=50,
    guidance_scale=7.5,
    seed=1234,
    octree_resolution=256,
    check_box_rembg=False,
    num_chunks=200000,
    randomize_seed: bool = False,
):
    if not MV_MODE and image is None and caption is None:
        raise gr.Error("请提供图像或文本提示。")
    
    # 深度模式验证
    if DEPTH_MODE and not MV_MODE:
        if image is None:
            raise gr.Error("深度模式下必须提供RGB图像。")
        if depth_file is None:
            raise gr.Error("深度模式下必须上传深度图文件。")
    
    if MV_MODE:
        if mv_image_front is None and mv_image_back is None \
            and mv_image_left is None and mv_image_right is None:
            raise gr.Error("请提供至少一个视图图像。")
        image = {}
        if mv_image_front:
            image['front'] = mv_image_front
        if mv_image_back:
            image['back'] = mv_image_back
        if mv_image_left:
            image['left'] = mv_image_left
        if mv_image_right:
            image['right'] = mv_image_right
        
        # 如果是多视图深度模式，也要验证深度图
        if DEPTH_MODE and MULTIVIEW_DEPTH_MODE:
            depth_provided = any([mv_depth_front, mv_depth_back, mv_depth_left, mv_depth_right])
            if not depth_provided:
                raise gr.Error("多视图深度模式下至少需要提供一个视图的深度图。")
        
        # 如果是多视图法线模式，也要验证法线图
        if MULTIVIEW_NORMAL_MODE:
            normal_provided = any([mv_normal_front, mv_normal_back, mv_normal_left, mv_normal_right])
            if not normal_provided:
                raise gr.Error("多视图法线模式下至少需要提供一个视图的法线图。")

        # 多视图RGB模式不需要验证法线图或深度图，只需要RGB图像

    seed = int(randomize_seed_fn(seed, randomize_seed))

    octree_resolution = int(octree_resolution)
    if caption: print('prompt is', caption)
    save_folder = gen_save_folder()
    stats = {
        'model': {
            'shapegen': f'{args.model_path}/{args.subfolder}',
            'texgen': f'{args.texgen_model_path}',
        },
        'params': {
            'caption': caption,
            'steps': steps,
            'guidance_scale': guidance_scale,
            'seed': seed,
            'octree_resolution': octree_resolution,
            'check_box_rembg': check_box_rembg,
            'num_chunks': num_chunks,
            'mode': 'multiview_rgb' if (MV_MODE and MULTIVIEW_RGB_MODE) else ('multiview_depth' if (MV_MODE and MULTIVIEW_DEPTH_MODE) else ('depth' if DEPTH_MODE else 'rgb')),
        }
    }
    time_meta = {}

    if image is None:
        start_time = time.time()
        try:
            image = t2i_worker(caption)
        except Exception as e:
            raise gr.Error(f"Text to 3D is disable. \
            Please enable it by `python gradio_app.py --enable_t23d`.")
        time_meta['text2image'] = time.time() - start_time

    # remove disk io to make responding faster, uncomment at your will.
    # image.save(os.path.join(save_folder, 'input.png'))
    # 保存 rembg 处理后的图像用于可视化
    rembg_images = {}
    
    if MV_MODE:
        start_time = time.time()
        for k, v in image.items():
            if check_box_rembg or v.mode == "RGB":
                img = rmbg_worker(v.convert('RGB'))
                image[k] = img
                # 保存 rembg 结果用于可视化
                rembg_images[k] = img.copy() if hasattr(img, 'copy') else img
        time_meta['remove background'] = time.time() - start_time
    else:
        if check_box_rembg or image.mode == "RGB":
            start_time = time.time()
            image = rmbg_worker(image.convert('RGB'))
            time_meta['remove background'] = time.time() - start_time
            # 保存 rembg 结果用于可视化
            rembg_images['single'] = image.copy() if hasattr(image, 'copy') else image

    # remove disk io to make responding faster, uncomment at your will.
    # image.save(os.path.join(save_folder, 'rembg.png'))

    # 处理深度图
    depth_tensor = None
    
    # 多视图深度模式
    if MV_MODE and MULTIVIEW_DEPTH_MODE and DEPTH_MODE:
        start_depth_time = time.time()
        try:
            # 收集所有视图的深度图
            depth_files_dict = {}
            if mv_depth_front is not None:
                depth_files_dict['front'] = mv_depth_front
            if mv_depth_right is not None:
                depth_files_dict['right'] = mv_depth_right
            if mv_depth_back is not None:
                depth_files_dict['back'] = mv_depth_back
            if mv_depth_left is not None:
                depth_files_dict['left'] = mv_depth_left
            
            # 加载多视图深度图
            depth_tensor = load_multiview_depths(depth_files_dict, target_size=518, num_views=args.num_views)
            
            if args.device == 'cuda':
                depth_tensor = depth_tensor.cuda()
            
            time_meta['multiview_depth processing'] = time.time() - start_depth_time
            print(f"多视图深度图处理完成，形状: {depth_tensor.shape}")
        except Exception as e:
            print(f"多视图深度图处理失败: {e}")
            raise gr.Error(f"多视图深度图处理失败: {str(e)}")
    
    # 单视图深度模式
    elif DEPTH_MODE and depth_file is not None and not MV_MODE:
        start_depth_time = time.time()
        try:
            depth_tensor = load_depth_from_16bit_png(depth_file.name if hasattr(depth_file, 'name') else depth_file)
            if args.device == 'cuda':
                depth_tensor = depth_tensor.cuda()
            time_meta['depth processing'] = time.time() - start_depth_time
            print(f"单视图深度图处理完成，形状: {depth_tensor.shape}")
        except Exception as e:
            print(f"单视图深度图处理失败: {e}")
            raise gr.Error(f"单视图深度图处理失败: {str(e)}")

    # 处理法线图
    normal_data = None
    
    # 多视图法线模式
    if MV_MODE and MULTIVIEW_NORMAL_MODE:
        start_normal_time = time.time()
        try:
            # 收集所有视图的法线图
            normal_files_dict = {}
            if mv_normal_front is not None:
                normal_files_dict['front'] = mv_normal_front
            if mv_normal_right is not None:
                normal_files_dict['right'] = mv_normal_right
            if mv_normal_back is not None:
                normal_files_dict['back'] = mv_normal_back
            if mv_normal_left is not None:
                normal_files_dict['left'] = mv_normal_left
            
            # 加载多视图法线图
            normal_data = load_multiview_normals(normal_files_dict, target_size=518, num_views=args.num_views)
            
            # 移到GPU
            if args.device == 'cuda':
                normal_data['normal'] = normal_data['normal'].cuda()
                normal_data['normal_mask'] = normal_data['normal_mask'].cuda()
            
            time_meta['multiview_normal processing'] = time.time() - start_normal_time
            print(f"✅ 多视图法线图处理完成，形状: {normal_data['normal'].shape}")
        except Exception as e:
            print(f"❌ 多视图法线图处理失败: {e}")
            raise gr.Error(f"多视图法线图处理失败: {str(e)}")

    # image to white model
    start_time = time.time()

    generator = torch.Generator()
    generator = generator.manual_seed(int(seed))
    
    # 在 low_vram_mode 下清理显存缓存
    if args.low_vram_mode:
        torch.cuda.empty_cache()
    
    # 准备模型输入
    model_inputs = {
        'image': image,
        'num_inference_steps': steps,
        'guidance_scale': guidance_scale,
        'generator': generator,
        'octree_resolution': octree_resolution,
        'num_chunks': num_chunks,
        'output_type': 'mesh'
    }
    
    # 如果有深度图，添加到输入中
    if depth_tensor is not None:
        model_inputs['depth'] = depth_tensor
        mode_str = "多视图" if (MV_MODE and MULTIVIEW_DEPTH_MODE) else "单视图"
        print(f"{mode_str}深度图已添加到模型输入")
        
        # 如果pipeline有controlnet，也传递进去
        if hasattr(i23d_worker, 'controlnet') and i23d_worker.controlnet is not None:
            model_inputs['controlnet'] = i23d_worker.controlnet
            print(f"ControlNet已添加到模型输入")
    
    # 如果有法线图，添加到输入中
    if normal_data is not None:
        model_inputs['normal'] = normal_data['normal']
        model_inputs['normal_mask'] = normal_data['normal_mask']
        print(f"✅ 多视图法线图已添加到模型输入，形状: {normal_data['normal'].shape}")
    
    outputs = i23d_worker(**model_inputs)
    time_meta['shape generation'] = time.time() - start_time
    logger.info("---Shape generation takes %s seconds ---" % (time.time() - start_time))
    
    # 在 low_vram_mode 下推理后清理显存缓存
    if args.low_vram_mode:
        torch.cuda.empty_cache()
    
    # 如果使用了 CPU offload，调用 maybe_free_model_hooks 或手动清理
    if args.low_vram_mode and hasattr(i23d_worker, '_all_hooks') and i23d_worker._all_hooks:
        try:
            # 先尝试使用 pipeline 自带的 maybe_free_model_hooks 方法
            if hasattr(i23d_worker, 'maybe_free_model_hooks'):
                i23d_worker.maybe_free_model_hooks()
            else:
                # 手动清理 hooks（自定义 CPU offload 的情况）
                for hook in i23d_worker._all_hooks:
                    try:
                        hook.offload()
                        hook.remove()
                    except:
                        pass
                # 重新设置 hooks
                if hasattr(i23d_worker, '_offload_gpu_id'):
                    enable_cpu_offload_for_pipeline(i23d_worker, gpu_id=i23d_worker._offload_gpu_id)
        except Exception as e:
            # 如果清理失败，至少清理显存缓存
            torch.cuda.empty_cache()

    tmp_start = time.time()
    mesh = export_to_trimesh(outputs)[0]
    time_meta['export to trimesh'] = time.time() - tmp_start

    stats['number_of_faces'] = mesh.faces.shape[0]
    stats['number_of_vertices'] = mesh.vertices.shape[0]

    stats['time'] = time_meta
    main_image = image if not MV_MODE else image['front']
    return mesh, main_image, save_folder, stats, seed, rembg_images

@spaces.GPU(duration=60)
def generation_all(
    caption=None,
    image=None,
    depth_file=None,  # 单视图深度图文件参数
    mv_image_front=None,
    mv_image_back=None,
    mv_image_left=None,
    mv_image_right=None,
    # 多视图深度图参数
    mv_depth_front=None,
    mv_depth_back=None,
    mv_depth_left=None,
    mv_depth_right=None,
    # 多视图法线图参数
    mv_normal_front=None,
    mv_normal_back=None,
    mv_normal_left=None,
    mv_normal_right=None,
    steps=50,
    guidance_scale=7.5,
    seed=1234,
    octree_resolution=256,
    check_box_rembg=False,
    num_chunks=200000,
    randomize_seed: bool = False,
):
    start_time_0 = time.time()
    mesh, image, save_folder, stats, seed, rembg_images = _gen_shape(
        caption,
        image,
        depth_file=depth_file,  # 传递单视图深度图参数
        mv_image_front=mv_image_front,
        mv_image_back=mv_image_back,
        mv_image_left=mv_image_left,
        mv_image_right=mv_image_right,
        # 传递多视图深度图参数
        mv_depth_front=mv_depth_front,
        mv_depth_back=mv_depth_back,
        mv_depth_left=mv_depth_left,
        mv_depth_right=mv_depth_right,
        # 传递多视图法线图参数
        mv_normal_front=mv_normal_front,
        mv_normal_back=mv_normal_back,
        mv_normal_left=mv_normal_left,
        mv_normal_right=mv_normal_right,
        steps=steps,
        guidance_scale=guidance_scale,
        seed=seed,
        octree_resolution=octree_resolution,
        check_box_rembg=check_box_rembg,
        num_chunks=num_chunks,
        randomize_seed=randomize_seed,
    )
    path = export_mesh(mesh, save_folder, textured=False)
    

    print(path)
    print('='*40)

    # tmp_time = time.time()
    # mesh = floater_remove_worker(mesh)
    # mesh = degenerate_face_remove_worker(mesh)
    # logger.info("---Postprocessing takes %s seconds ---" % (time.time() - tmp_time))
    # stats['time']['postprocessing'] = time.time() - tmp_time

    tmp_time = time.time()
    mesh = face_reduce_worker(mesh)

    # path = export_mesh(mesh, save_folder, textured=False, type='glb')
    path = export_mesh(mesh, save_folder, textured=False, type='obj') # 这样操作也会 core dump

    logger.info("---Face Reduction takes %s seconds ---" % (time.time() - tmp_time))
    stats['time']['face reduction'] = time.time() - tmp_time

    tmp_time = time.time()

    text_path = os.path.join(save_folder, f'textured_mesh.obj')
    path_textured = tex_pipeline(mesh_path=path, image_path=image, output_mesh_path=text_path, save_glb=False)
        
    logger.info("---Texture Generation takes %s seconds ---" % (time.time() - tmp_time))
    stats['time']['texture generation'] = time.time() - tmp_time

    tmp_time = time.time()
    # Convert textured OBJ to GLB using obj2gltf with PBR support
    glb_path_textured = os.path.join(save_folder, 'textured_mesh.glb')
    conversion_success = quick_convert_with_obj2gltf(path_textured, glb_path_textured)

    logger.info("---Convert textured OBJ to GLB takes %s seconds ---" % (time.time() - tmp_time))
    stats['time']['convert textured OBJ to GLB'] = time.time() - tmp_time
    stats['time']['total'] = time.time() - start_time_0
    model_viewer_html_textured = build_model_viewer_html(save_folder, 
                                                         height=HTML_HEIGHT, 
                                                         width=HTML_WIDTH, textured=True)
    if args.low_vram_mode:
        torch.cuda.empty_cache()
    
    # 准备 rembg 图像的返回值
    if MV_MODE:
        rembg_front = rembg_images.get('front', None)
        rembg_right = rembg_images.get('right', None)
        rembg_back = rembg_images.get('back', None)
        rembg_left = rembg_images.get('left', None)
    else:
        # 单视图模式：只返回单个图像，其他为 None
        rembg_front = rembg_images.get('single', None)
        rembg_right = None
        rembg_back = None
        rembg_left = None
    
    return (
        gr.update(value=path),
        gr.update(value=glb_path_textured),
        model_viewer_html_textured,
        stats,
        seed,
        rembg_front,
        rembg_right,
        rembg_back,
        rembg_left,
    )

@spaces.GPU(duration=60)
def shape_generation(
    caption=None,
    image=None,
    depth_file=None,  # 单视图深度图文件参数
    mv_image_front=None,
    mv_image_back=None,
    mv_image_left=None,
    mv_image_right=None,
    # 多视图深度图参数
    mv_depth_front=None,
    mv_depth_back=None,
    mv_depth_left=None,
    mv_depth_right=None,
    # 多视图法线图参数
    mv_normal_front=None,
    mv_normal_back=None,
    mv_normal_left=None,
    mv_normal_right=None,
    steps=50,
    guidance_scale=7.5,
    seed=1234,
    octree_resolution=256,
    check_box_rembg=False,
    num_chunks=200000,
    randomize_seed: bool = False,
):
    start_time_0 = time.time()
    mesh, image, save_folder, stats, seed, rembg_images = _gen_shape(
        caption,
        image,
        depth_file=depth_file,  # 传递单视图深度图参数
        mv_image_front=mv_image_front,
        mv_image_back=mv_image_back,
        mv_image_left=mv_image_left,
        mv_image_right=mv_image_right,
        # 传递多视图深度图参数
        mv_depth_front=mv_depth_front,
        mv_depth_back=mv_depth_back,
        mv_depth_left=mv_depth_left,
        mv_depth_right=mv_depth_right,
        # 传递多视图法线图参数
        mv_normal_front=mv_normal_front,
        mv_normal_back=mv_normal_back,
        mv_normal_left=mv_normal_left,
        mv_normal_right=mv_normal_right,
        steps=steps,
        guidance_scale=guidance_scale,
        seed=seed,
        octree_resolution=octree_resolution,
        check_box_rembg=check_box_rembg,
        num_chunks=num_chunks,
        randomize_seed=randomize_seed,
    )
    stats['time']['total'] = time.time() - start_time_0
    mesh.metadata['extras'] = stats

    path = export_mesh(mesh, save_folder, textured=False)
    print(f"✅ Mesh导出成功: {path}")
    print(f"   文件是否存在: {os.path.exists(path)}")
    print(f"   文件大小: {os.path.getsize(path) if os.path.exists(path) else 'N/A'} bytes")
    
    model_viewer_html = build_model_viewer_html(save_folder, height=HTML_HEIGHT, width=HTML_WIDTH)
    print(f"✅ Model viewer HTML已生成")
    print(f"   HTML内容长度: {len(model_viewer_html)} 字符")
    
    if args.low_vram_mode:
        torch.cuda.empty_cache()
    
    # 准备 rembg 图像的返回值
    if MV_MODE:
        rembg_front = rembg_images.get('front', None)
        rembg_right = rembg_images.get('right', None)
        rembg_back = rembg_images.get('back', None)
        rembg_left = rembg_images.get('left', None)
    else:
        # 单视图模式：只返回单个图像，其他为 None
        rembg_front = rembg_images.get('single', None)
        rembg_right = None
        rembg_back = None
        rembg_left = None
    
    return (
        gr.update(value=path),
        model_viewer_html,
        stats,
        seed,
        rembg_front,
        rembg_right,
        rembg_back,
        rembg_left,
    )


def build_app():
    title = 'Hunyuan3D-2: High Resolution Textured 3D Assets Generation'
    if MV_MODE and not MULTIVIEW_DEPTH_MODE:
        title = 'Hunyuan3D-2mv: Image to 3D Generation with 1-4 Views'
    if 'mini' in args.subfolder:
        title = 'Hunyuan3D-2mini: Strong 0.6B Image to Shape Generator'

    title = 'Hunyuan-3D-2.1'
        
    if TURBO_MODE:
        title = title.replace(':', '-Turbo: Fast ')

    # 根据模式调整标题
    if MULTIVIEW_RGB_MODE:
        title += f" (多视图RGB重建模式 - {args.num_views}视图)"
    elif MULTIVIEW_NORMAL_MODE:
        title += f" (多视图法线条件模式 - {args.num_views}视图)"
    elif MULTIVIEW_DEPTH_MODE:
        title += f" (多视图深度条件模式 - {args.num_views}视图)"
    elif DEPTH_MODE:
        title += " (单视图深度条件模式)"
    elif MV_MODE:
        title += " (多视图模式)"
    
    title_html = f"""
    <div style="font-size: 2em; font-weight: bold; text-align: center; margin-bottom: 5px">

    {title}
    </div>
    <div align="center">
    Tencent Hunyuan3D Team
    </div>
    """
    custom_css = """
    .app.svelte-wpkpf6.svelte-wpkpf6:not(.fill_width) {
        max-width: 1480px;
    }
    .mv-image button .wrap {
        font-size: 10px;
    }

    .mv-image .icon-wrap {
        width: 20px;
    }

    """

    with gr.Blocks(theme=gr.themes.Base(), title='Hunyuan-3D-2.1', analytics_enabled=False, css=custom_css) as demo:
        gr.HTML(title_html)

        with gr.Row():
            with gr.Column(scale=3):
                with gr.Tabs(selected='tab_img_prompt') as tabs_prompt:
                    with gr.Tab('Image Prompt', id='tab_img_prompt', visible=not MV_MODE) as tab_ip:
                        image = gr.Image(label='RGB图像', type='pil', image_mode='RGBA', height=290)
                        
                        # 如果启用深度模式，添加深度图上传组件
                        if DEPTH_MODE:
                            with gr.Row():
                                depth_file = gr.File(
                                    label="深度图文件 (16位PNG)",
                                    file_types=[".png", ".tiff", ".tif"],
                                    type="filepath",
                                    interactive=True
                                )
                            gr.Markdown(
                                "📋 **深度图要求:**\n"
                                "- 格式: 16位PNG、TIFF或TIF文件\n" 
                                "- 尺寸: 建议与RGB图像相同\n"
                                "- 值域: 任意深度值（会自动归一化）\n"
                                "- 注意: Gradio的Image组件会将16位图压缩为8位，因此必须使用File组件上传\n\n"
                                "💡 **使用提示:**\n"
                                "1. 先上传RGB图像\n"
                                "2. 再上传对应的深度图文件\n" 
                                "3. 点击生成按钮开始处理"
                            )
                            
                            # 添加深度图验证状态显示
                            depth_status = gr.HTML("", visible=False)
                            
                            def validate_depth_upload(file):
                                if file is None:
                                    return gr.update(visible=False)
                                
                                try:
                                    if validate_depth_file(file.name if hasattr(file, 'name') else file):
                                        return gr.update(
                                            value="✅ <span style='color: green;'>深度图文件有效</span>",
                                            visible=True
                                        )
                                    else:
                                        return gr.update(
                                            value="❌ <span style='color: red;'>深度图文件格式无效</span>",
                                            visible=True
                                        )
                                except Exception as e:
                                    return gr.update(
                                        value=f"⚠️ <span style='color: orange;'>验证失败: {str(e)}</span>",
                                        visible=True
                                    )
                            
                            depth_file.upload(validate_depth_upload, inputs=[depth_file], outputs=[depth_status])
                        else:
                            depth_file = gr.State(None)
                        
                        caption = gr.State(None)
#                    with gr.Tab('Text Prompt', id='tab_txt_prompt', visible=HAS_T2I and not MV_MODE) as tab_tp:
#                        caption = gr.Textbox(label='Text Prompt',
#                                             placeholder='HunyuanDiT will be used to generate image.',
#                                             info='Example: A 3D model of a cute cat, white background')
                    with gr.Tab('MultiView Prompt', visible=MV_MODE) as tab_mv:
                        # gr.Label('Please upload at least one front image.')
                        gr.Markdown("### RGB 图像")
                        with gr.Row():
                            mv_image_front = gr.Image(label='Front', type='pil', image_mode='RGBA', height=140,
                                                      min_width=100, elem_classes='mv-image')
                            mv_image_back = gr.Image(label='Back', type='pil', image_mode='RGBA', height=140,
                                                     min_width=100, elem_classes='mv-image')
                        with gr.Row():
                            mv_image_left = gr.Image(label='Left', type='pil', image_mode='RGBA', height=140,
                                                     min_width=100, elem_classes='mv-image')
                            mv_image_right = gr.Image(label='Right', type='pil', image_mode='RGBA', height=140,
                                                      min_width=100, elem_classes='mv-image')
                        
                        # 如果启用多视图RGB模式，添加说明信息
                        if MULTIVIEW_RGB_MODE:
                            gr.Markdown(
                                f"📋 **多视图RGB重建模式 ({args.num_views}视图):**\n"
                                f"- 模式: 仅使用RGB彩色图像进行3D重建\n"
                                f"- 不需要: 法线图或深度图\n"
                                f"- 视图顺序: Front(0°) → Right(90°) → Back(180°) → Left(270°)\n"
                                f"- 建议: 上传4个视图的RGB图像以获得最佳效果\n\n"
                                f"💡 **使用提示:**\n"
                                f"1. 按顺序上传各视图的RGB图像（Front、Right、Back、Left）\n"
                                f"2. 图像会自动进行背景移除处理\n"
                                f"3. 点击生成按钮开始3D重建"
                            )
                        
                        # 如果启用多视图深度模式，添加深度图上传组件
                        if MULTIVIEW_DEPTH_MODE and DEPTH_MODE:
                            gr.Markdown(f"### 深度图 (16位PNG) - {args.num_views}视图模式")
                            
                            # 根据视图数量动态创建上传组件
                            if args.num_views == 2:
                                with gr.Row():
                                    mv_depth_front = gr.File(
                                        label="Front Depth",
                                        file_types=[".png", ".tiff", ".tif"],
                                        type="filepath",
                                        interactive=True
                                    )
                                    mv_depth_right = gr.File(
                                        label="Right Depth",
                                        file_types=[".png", ".tiff", ".tif"],
                                        type="filepath",
                                        interactive=True
                                    )
                                mv_depth_back = gr.State(None)
                                mv_depth_left = gr.State(None)
                                view_info = "视图顺序: Front(0°) → Right(90°)\n"
                            elif args.num_views == 3:
                                with gr.Row():
                                    mv_depth_front = gr.File(
                                        label="Front Depth",
                                        file_types=[".png", ".tiff", ".tif"],
                                        type="filepath",
                                        interactive=True
                                    )
                                    mv_depth_right = gr.File(
                                        label="Right Depth",
                                        file_types=[".png", ".tiff", ".tif"],
                                        type="filepath",
                                        interactive=True
                                    )
                                with gr.Row():
                                    mv_depth_back = gr.File(
                                        label="Back Depth",
                                        file_types=[".png", ".tiff", ".tif"],
                                        type="filepath",
                                        interactive=True
                                    )
                                    mv_depth_left = gr.State(None)
                                view_info = "视图顺序: Front(0°) → Right(90°) → Back(180°)\n"
                            else:  # 4 views
                                with gr.Row():
                                    mv_depth_front = gr.File(
                                        label="Front Depth",
                                        file_types=[".png", ".tiff", ".tif"],
                                        type="filepath",
                                        interactive=True
                                    )
                                    mv_depth_back = gr.File(
                                        label="Back Depth",
                                        file_types=[".png", ".tiff", ".tif"],
                                        type="filepath",
                                        interactive=True
                                    )
                                with gr.Row():
                                    mv_depth_left = gr.File(
                                        label="Left Depth",
                                        file_types=[".png", ".tiff", ".tif"],
                                        type="filepath",
                                        interactive=True
                                    )
                                    mv_depth_right = gr.File(
                                        label="Right Depth",
                                        file_types=[".png", ".tiff", ".tif"],
                                        type="filepath",
                                        interactive=True
                                    )
                                view_info = "视图顺序: Front(0°) → Right(90°) → Back(180°) → Left(270°)\n"
                            
                            gr.Markdown(
                                f"📋 **多视图深度图要求 ({args.num_views}视图模式):**\n"
                                f"- 格式: 16位PNG、TIFF或TIF文件\n" 
                                f"- 尺寸: 建议与RGB图像相同\n"
                                f"- {view_info}"
                                f"- 至少提供一个视图的深度图\n\n"
                                f"💡 **使用提示:**\n"
                                f"1. 按顺序上传各视图的RGB图像\n"
                                f"2. 上传对应视图的深度图文件\n" 
                                f"3. 点击生成按钮开始处理"
                            )
                        else:
                            # 创建占位符状态变量
                            mv_depth_front = gr.State(None)
                            mv_depth_back = gr.State(None)
                            mv_depth_left = gr.State(None)
                            mv_depth_right = gr.State(None)
                        
                        # 如果启用多视图法线模式，添加法线图上传组件
                        if MULTIVIEW_NORMAL_MODE:
                            gr.Markdown(f"### 法线图 (RGB PNG) - {args.num_views}视图模式")
                            
                            # 根据视图数量动态创建上传组件
                            if args.num_views == 2:
                                with gr.Row():
                                    mv_normal_front = gr.File(
                                        label="Front Normal",
                                        file_types=[".png"],
                                        type="filepath",
                                        interactive=True
                                    )
                                    mv_normal_right = gr.File(
                                        label="Right Normal",
                                        file_types=[".png"],
                                        type="filepath",
                                        interactive=True
                                    )
                                mv_normal_back = gr.State(None)
                                mv_normal_left = gr.State(None)
                                view_info = "视图顺序: Front(0°) → Right(90°)\n"
                            elif args.num_views == 3:
                                with gr.Row():
                                    mv_normal_front = gr.File(
                                        label="Front Normal",
                                        file_types=[".png"],
                                        type="filepath",
                                        interactive=True
                                    )
                                    mv_normal_right = gr.File(
                                        label="Right Normal",
                                        file_types=[".png"],
                                        type="filepath",
                                        interactive=True
                                    )
                                with gr.Row():
                                    mv_normal_back = gr.File(
                                        label="Back Normal",
                                        file_types=[".png"],
                                        type="filepath",
                                        interactive=True
                                    )
                                    mv_normal_left = gr.State(None)
                                view_info = "视图顺序: Front(0°) → Right(90°) → Back(180°)\n"
                            else:  # 4 views
                                with gr.Row():
                                    mv_normal_front = gr.File(
                                        label="Front Normal",
                                        file_types=[".png"],
                                        type="filepath",
                                        interactive=True
                                    )
                                    mv_normal_back = gr.File(
                                        label="Back Normal",
                                        file_types=[".png"],
                                        type="filepath",
                                        interactive=True
                                    )
                                with gr.Row():
                                    mv_normal_left = gr.File(
                                        label="Left Normal",
                                        file_types=[".png"],
                                        type="filepath",
                                        interactive=True
                                    )
                                    mv_normal_right = gr.File(
                                        label="Right Normal",
                                        file_types=[".png"],
                                        type="filepath",
                                        interactive=True
                                    )
                                view_info = "视图顺序: Front(0°) → Right(90°) → Back(180°) → Left(270°)\n"
                            
                            gr.Markdown(
                                f"📋 **多视图法线图要求 ({args.num_views}视图模式):**\n"
                                f"- 格式: RGB PNG图像文件\n" 
                                f"- 尺寸: 建议与RGB图像相同（推荐518×518）\n"
                                f"- {view_info}"
                                f"- 至少提供一个视图的法线图\n"
                                f"- 法线图应为RGB格式，表示表面法线方向\n\n"
                                f"💡 **使用提示:**\n"
                                f"1. 按顺序上传各视图的RGB图像\n"
                                f"2. 上传对应视图的法线图文件（RGB PNG格式）\n" 
                                f"3. 点击生成按钮开始处理"
                            )
                        else:
                            # 创建占位符状态变量
                            mv_normal_front = gr.State(None)
                            mv_normal_back = gr.State(None)
                            mv_normal_left = gr.State(None)
                            mv_normal_right = gr.State(None)

                with gr.Row():
                    btn = gr.Button(value='Gen Shape', variant='primary', min_width=100)
                    btn_all = gr.Button(value='Gen Textured Shape',
                                        variant='primary',
                                        visible=HAS_TEXTUREGEN,
                                        min_width=100)

                with gr.Group():
                    file_out = gr.File(label="File", visible=False)
                    file_out2 = gr.File(label="File", visible=False)

                with gr.Tabs(selected='tab_options' if TURBO_MODE else 'tab_export'):
                    with gr.Tab("Options", id='tab_options', visible=TURBO_MODE):
                        gen_mode = gr.Radio(
                            label='Generation Mode',
                            info='Recommendation: Turbo for most cases, \
Fast for very complex cases, Standard seldom use.',
                            choices=['Turbo', 'Fast', 'Standard'], 
                            value='Turbo')
                        decode_mode = gr.Radio(
                            label='Decoding Mode',
                            info='The resolution for exporting mesh from generated vectset',
                            choices=['Low', 'Standard', 'High'],
                            value='Standard')
                    with gr.Tab('Advanced Options', id='tab_advanced_options'):
                        with gr.Row():
                            check_box_rembg = gr.Checkbox(
                                value=True, 
                                label='Remove Background', 
                                min_width=100)
                            randomize_seed = gr.Checkbox(
                                label="Randomize seed", 
                                value=True, 
                                min_width=100)
                        seed = gr.Slider(
                            label="Seed",
                            minimum=0,
                            maximum=MAX_SEED,
                            step=1,
                            value=1234,
                            min_width=100,
                        )
                        with gr.Row():
                            num_steps = gr.Slider(maximum=100,
                                                  minimum=1,
                                                  value=5 if 'turbo' in args.subfolder else 30,
                                                  step=1, label='Inference Steps')
                            octree_resolution = gr.Slider(maximum=512, 
                                                          minimum=16, 
                                                          value=256, 
                                                          label='Octree Resolution')
                        with gr.Row():
                            cfg_scale = gr.Number(value=5.0, label='Guidance Scale', min_width=100)
                            num_chunks = gr.Slider(maximum=5000000, minimum=1000, value=8000,
                                                   label='Number of Chunks', min_width=100)
                    with gr.Tab("Export", id='tab_export'):
                        with gr.Row():
                            file_type = gr.Dropdown(label='File Type', 
                                                    choices=SUPPORTED_FORMATS,
                                                    value='glb', min_width=100)
                            reduce_face = gr.Checkbox(label='Simplify Mesh', 
                                                      value=False, min_width=100)
                            export_texture = gr.Checkbox(label='Include Texture', value=False,
                                                         visible=False, min_width=100)
                        target_face_num = gr.Slider(maximum=1000000, minimum=100, value=10000,
                                                    label='Target Face Number')
                        with gr.Row():
                            confirm_export = gr.Button(value="Transform", min_width=100)
                            file_export = gr.DownloadButton(label="Download", variant='primary',
                                                            interactive=False, min_width=100)

            with gr.Column(scale=6):
                with gr.Tabs(selected='gen_mesh_panel') as tabs_output:
                    with gr.Tab('Generated Mesh', id='gen_mesh_panel'):
                        html_gen_mesh = gr.HTML(HTML_OUTPUT_PLACEHOLDER, label='Output')
                    with gr.Tab('Exporting Mesh', id='export_mesh_panel'):
                        html_export_mesh = gr.HTML(HTML_OUTPUT_PLACEHOLDER, label='Output')
                    with gr.Tab('Mesh Statistic', id='stats_panel'):
                        stats = gr.Json({}, label='Mesh Stats')

            with gr.Column(scale=3 if MV_MODE else 2):
                with gr.Tabs(selected='tab_img_gallery') as gallery:
                    with gr.Tab('Image to 3D Gallery', 
                                id='tab_img_gallery', 
                                visible=not MV_MODE) as tab_gi:
                        with gr.Row():
                            gr.Examples(examples=example_is, inputs=[image],
                                        label=None, examples_per_page=18)
            
            # 新增：显示 rembg 处理后的图像
            with gr.Column(scale=3) as rembg_column:
                gr.Markdown("### 背景移除结果预览")
                with gr.Tabs(selected='rembg_preview') as rembg_tabs:
                    with gr.Tab('RemBG Results', id='rembg_preview'):
                        if MV_MODE:
                            with gr.Row():
                                rembg_front = gr.Image(label='Front (RemBG)', type='pil', height=200, visible=True)
                                rembg_right = gr.Image(label='Right (RemBG)', type='pil', height=200, visible=True)
                            with gr.Row():
                                rembg_back = gr.Image(label='Back (RemBG)', type='pil', height=200, visible=True)
                                rembg_left = gr.Image(label='Left (RemBG)', type='pil', height=200, visible=True)
                        else:
                            # 单视图模式下，只显示一个图像，其他视图设为不可见
                            rembg_front = gr.Image(label='RemBG Result', type='pil', height=400, visible=True)
                            rembg_right = gr.Image(label='Right (RemBG)', type='pil', height=200, visible=False)
                            rembg_back = gr.Image(label='Back (RemBG)', type='pil', height=200, visible=False)
                            rembg_left = gr.Image(label='Left (RemBG)', type='pil', height=200, visible=False)

        tab_ip.select(fn=lambda: gr.update(selected='tab_img_gallery'), outputs=gallery)
        #if HAS_T2I:
        #    tab_tp.select(fn=lambda: gr.update(selected='tab_txt_gallery'), outputs=gallery)

        btn.click(
            shape_generation,
            inputs=[
                caption,
                image,
                depth_file,  # 单视图深度图文件参数
                mv_image_front,
                mv_image_back,
                mv_image_left,
                mv_image_right,
                # 多视图深度图参数
                mv_depth_front,
                mv_depth_back,
                mv_depth_left,
                mv_depth_right,
                # 多视图法线图参数
                mv_normal_front,
                mv_normal_back,
                mv_normal_left,
                mv_normal_right,
                num_steps,
                cfg_scale,
                seed,
                octree_resolution,
                check_box_rembg,
                num_chunks,
                randomize_seed,
            ],
            outputs=[file_out, html_gen_mesh, stats, seed, rembg_front, rembg_right, rembg_back, rembg_left]
        ).then(
            lambda: (gr.update(visible=False, value=False), gr.update(interactive=True), gr.update(interactive=True),
                     gr.update(interactive=False)),
            outputs=[export_texture, reduce_face, confirm_export, file_export],
        ).then(
            lambda: gr.update(selected='gen_mesh_panel'),
            outputs=[tabs_output],
        )

        btn_all.click(
            generation_all,
            inputs=[
                caption,
                image,
                depth_file,  # 单视图深度图文件参数
                mv_image_front,
                mv_image_back,
                mv_image_left,
                mv_image_right,
                # 多视图深度图参数
                mv_depth_front,
                mv_depth_back,
                mv_depth_left,
                mv_depth_right,
                # 多视图法线图参数
                mv_normal_front,
                mv_normal_back,
                mv_normal_left,
                mv_normal_right,
                num_steps,
                cfg_scale,
                seed,
                octree_resolution,
                check_box_rembg,
                num_chunks,
                randomize_seed,
            ],
            outputs=[file_out, file_out2, html_gen_mesh, stats, seed, rembg_front, rembg_right, rembg_back, rembg_left]
        ).then(
            lambda: (gr.update(visible=True, value=True), gr.update(interactive=False), gr.update(interactive=True),
                     gr.update(interactive=False)),
            outputs=[export_texture, reduce_face, confirm_export, file_export],
        ).then(
            lambda: gr.update(selected='gen_mesh_panel'),
            outputs=[tabs_output],
        )

        def on_gen_mode_change(value):
            if value == 'Turbo':
                return gr.update(value=5)
            elif value == 'Fast':
                return gr.update(value=10)
            else:
                return gr.update(value=30)

        gen_mode.change(on_gen_mode_change, inputs=[gen_mode], outputs=[num_steps])

        def on_decode_mode_change(value):
            if value == 'Low':
                return gr.update(value=196)
            elif value == 'Standard':
                return gr.update(value=256)
            else:
                return gr.update(value=384)

        decode_mode.change(on_decode_mode_change, inputs=[decode_mode], 
                           outputs=[octree_resolution])

        def on_export_click(file_out, file_out2, file_type, 
                            reduce_face, export_texture, target_face_num):
            if file_out is None:
                raise gr.Error('Please generate a mesh first.')

            print(f'exporting {file_out}')
            print(f'reduce face to {target_face_num}')
            if export_texture:
                mesh = trimesh.load(file_out2)
                save_folder = gen_save_folder()
                path = export_mesh(mesh, save_folder, textured=True, type=file_type)

                # for preview
                save_folder = gen_save_folder()
                _ = export_mesh(mesh, save_folder, textured=True)
                model_viewer_html = build_model_viewer_html(save_folder, 
                                                            height=HTML_HEIGHT, 
                                                            width=HTML_WIDTH,
                                                            textured=True)
            else:
                mesh = trimesh.load(file_out)
                mesh = floater_remove_worker(mesh)
                mesh = degenerate_face_remove_worker(mesh)
                if reduce_face:
                    mesh = face_reduce_worker(mesh, target_face_num)
                save_folder = gen_save_folder()
                path = export_mesh(mesh, save_folder, textured=False, type=file_type)

                # for preview
                save_folder = gen_save_folder()
                _ = export_mesh(mesh, save_folder, textured=False)
                model_viewer_html = build_model_viewer_html(save_folder, 
                                                            height=HTML_HEIGHT, 
                                                            width=HTML_WIDTH,
                                                            textured=False)
            print(f'export to {path}')
            return model_viewer_html, gr.update(value=path, interactive=True)

        confirm_export.click(
            lambda: gr.update(selected='export_mesh_panel'),
            outputs=[tabs_output],
        ).then(
            on_export_click,
            inputs=[file_out, file_out2, file_type, reduce_face, export_texture, target_face_num],
            outputs=[html_export_mesh, file_export]
        )

    return demo


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default='tencent/Hunyuan3D-2.1')
    parser.add_argument("--subfolder", type=str, default='hunyuan3d-dit-v2-1')
    parser.add_argument("--texgen_model_path", type=str, default='tencent/Hunyuan3D-2.1')
    parser.add_argument('--port', type=int, default=6008)
    parser.add_argument('--host', type=str, default='0.0.0.0')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--mc_algo', type=str, default='mc')
    parser.add_argument('--cache-path', type=str, default='./save_dir')
    parser.add_argument('--enable_t23d', action='store_true')
    parser.add_argument('--disable_tex', action='store_true')
    parser.add_argument('--enable_flashvdm', action='store_true')
    parser.add_argument('--compile', action='store_true')
    parser.add_argument('--low_vram_mode', action='store_true', help='Enable low VRAM mode with CPU offload')
    parser.add_argument('--use_multi_gpu', action='store_true', help='Use multiple GPUs for model parallel inference')
    parser.add_argument('--gpu_ids', type=str, default='0', help='Comma-separated GPU IDs to use (e.g., "0,1" for GPU 0 and 1)')
    parser.add_argument('--enable_depth', action='store_true', help='Enable depth-conditioned model (RGBD mode)')
    parser.add_argument('--enable_multiview_depth', action='store_true', help='Enable multi-view depth-conditioned model')
    parser.add_argument('--enable_multiview_normal', action='store_true', help='Enable multi-view normal-conditioned model')
    parser.add_argument('--enable_multiview_rgb', action='store_true', help='Enable multi-view RGB reconstruction model (4 views, no normal maps)')
    parser.add_argument('--depth_lora_path', type=str, default="./hy3dshape/output_folder/dit/depth_lora_checkpoints/ckpt/ckpt-step=00000200.ckpt", help='Path to depth LoRA checkpoint')
    parser.add_argument('--normal_lora_path', type=str, default=None, help='Path to normal LoRA checkpoint')
    parser.add_argument('--rgb_lora_path', type=str, default=None, help='Path to RGB LoRA checkpoint')
    parser.add_argument('--num_views', type=int, default=4, help='Number of views for multi-view model (default: 4)')
    parser.add_argument('--rembg_model', type=str, default='u2net_human_seg', 
                        choices=['u2net', 'u2netp', 'u2net_human_seg', 'silueta', 'isnet-general-use'],
                        help='rembg background removal model (default: u2net). Options: u2net (default, general), u2netp (lightweight, faster), u2net_human_seg (human segmentation), silueta, isnet-general-use')
    args = parser.parse_args()
    args.enable_flashvdm = False

    SAVE_DIR = args.cache_path
    os.makedirs(SAVE_DIR, exist_ok=True)

    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # 检测是否是标准的Hunyuan3D-2mv模型（从Hugging Face加载）
    IS_MV_PRETRAINED = 'Hunyuan3D-2mv' in args.model_path or 'mv' in args.model_path.lower()
    if IS_MV_PRETRAINED and args.subfolder == 'hunyuan3d-dit-v2-1':
        # 自动设置正确的subfolder
        args.subfolder = 'hunyuan3d-dit-v2-mv'
        print(f"✅ 检测到Hunyuan3D-2mv模型，自动设置subfolder为: {args.subfolder}")
    
    MV_MODE = IS_MV_PRETRAINED or args.enable_multiview_depth or args.enable_multiview_normal or args.enable_multiview_rgb
    TURBO_MODE = 'turbo' in args.subfolder
    DEPTH_MODE = args.enable_depth or args.enable_multiview_depth  # 标记是否使用深度图模式
    MULTIVIEW_DEPTH_MODE = args.enable_multiview_depth  # 标记是否使用多视图深度图模式
    MULTIVIEW_NORMAL_MODE = args.enable_multiview_normal  # 标记是否使用多视图法线图模式
    MULTIVIEW_RGB_MODE = args.enable_multiview_rgb  # 标记是否使用多视图RGB重建模式（仅四视图彩色，不使用法线图）

    HTML_HEIGHT = 690 if MV_MODE else 650
    HTML_WIDTH = 500
    HTML_OUTPUT_PLACEHOLDER = f"""
    <div style='height: {650}px; width: 100%; border-radius: 8px; border-color: #e5e7eb; border-style: solid; border-width: 1px; display: flex; justify-content: center; align-items: center;'>
      <div style='text-align: center; font-size: 16px; color: #6b7280;'>
        <p style="color: #8d8d8d;">Welcome to Hunyuan3D!</p>
        <p style="color: #8d8d8d;">No mesh here.</p>
      </div>
    </div>
    """

    INPUT_MESH_HTML = """
    <div style='height: 490px; width: 100%; border-radius: 8px; 
    border-color: #e5e7eb; order-style: solid; border-width: 1px;'>
    </div>
    """
    example_is = get_example_img_list()
    example_ts = get_example_txt_list()

    SUPPORTED_FORMATS = ['glb', 'obj', 'ply', 'stl']

    HAS_TEXTUREGEN = False
    if not args.disable_tex:
        try:
            # Apply torchvision fix before importing basicsr/RealESRGAN
            print("Applying torchvision compatibility fix for texture generation...")
            try:
                from torchvision_fix import apply_fix
                fix_result = apply_fix()
                if not fix_result:
                    print("Warning: Torchvision fix may not have been applied successfully")
            except Exception as fix_error:
                print(f"Warning: Failed to apply torchvision fix: {fix_error}")
            
            # from hy3dgen.texgen import Hunyuan3DPaintPipeline
            # texgen_worker = Hunyuan3DPaintPipeline.from_pretrained(args.texgen_model_path)
            # if args.low_vram_mode:
            #     texgen_worker.enable_model_cpu_offload()

            from hy3dpaint.textureGenPipeline import Hunyuan3DPaintPipeline, Hunyuan3DPaintConfig
            conf = Hunyuan3DPaintConfig(max_num_view=8, resolution=768)
            conf.realesrgan_ckpt_path = "hy3dpaint/ckpt/RealESRGAN_x4plus.pth"
            conf.multiview_cfg_path = "hy3dpaint/cfgs/hunyuan-paint-pbr.yaml"
            conf.custom_pipeline = "hy3dpaint/hunyuanpaintpbr"
            tex_pipeline = Hunyuan3DPaintPipeline(conf)
        
            # Not help much, ignore for now.
            # if args.compile:
            #     texgen_worker.models['delight_model'].pipeline.unet.compile()
            #     texgen_worker.models['delight_model'].pipeline.vae.compile()
            #     texgen_worker.models['multiview_model'].pipeline.unet.compile()
            #     texgen_worker.models['multiview_model'].pipeline.vae.compile()
            
            HAS_TEXTUREGEN = True
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Error loading texture generator: {e}")
            print("Failed to load texture generator.")
            print('Please try to install requirements by following README.md')
            HAS_TEXTUREGEN = False

    HAS_T2I = True
    if args.enable_t23d:
        from hy3dgen.text2image import HunyuanDiTPipeline

        t2i_worker = HunyuanDiTPipeline('Tencent-Hunyuan/HunyuanDiT-v1.1-Diffusers-Distilled')
        HAS_T2I = True

    from hy3dshape import FaceReducer, FloaterRemover, DegenerateFaceRemover, MeshSimplifier, \
        Hunyuan3DDiTFlowMatchingPipeline
    from hy3dshape.pipelines import export_to_trimesh
    from hy3dshape.rembg import BackgroundRemover

    print(f"正在初始化背景移除器，使用模型: {args.rembg_model}")
    rmbg_worker = BackgroundRemover(model_name=args.rembg_model)
    
    # 根据是否启用深度模式、法线模式或RGB模式来加载模型
    if DEPTH_MODE or MULTIVIEW_NORMAL_MODE or MULTIVIEW_RGB_MODE:
        if MULTIVIEW_RGB_MODE:
            print("正在加载多视图RGB重建模型...")
        elif MULTIVIEW_NORMAL_MODE:
            print("正在加载多视图法线条件模型...")
        elif DEPTH_MODE:
            print("正在加载深度条件模型...")
        
        # 自动寻找深度LoRA权重路径（如果未指定）
        if args.depth_lora_path is None:
            # 根据配置文件中的设置，默认的保存路径
            default_lora_dirs = [
                "output_folder/dit/depth_lora_checkpoints",
                "hy3dshape/output_folder/dit/depth_lora_checkpoints",
                "./hy3dshape/output_folder/dit/depth_lora_finetuning/ckpt"
            ]
            
            for lora_dir in default_lora_dirs:
                if os.path.exists(lora_dir):
                    # 查找最新的checkpoint
                    checkpoints = [d for d in os.listdir(lora_dir) 
                                 if d.startswith('step_') and os.path.isdir(os.path.join(lora_dir, d))]
                    if checkpoints:
                        # 按步数排序，选择最大的
                        latest_ckpt = max(checkpoints, key=lambda x: int(x.split('_')[1]))
                        args.depth_lora_path = os.path.join(lora_dir, latest_ckpt)
                        print(f"自动发现深度LoRA权重: {args.depth_lora_path}")
                        break
        
        # 加载支持深度条件、法线条件或RGB重建的模型
        try:
            if DEPTH_MODE:
                # 深度模式需要load_depth参数
                i23d_worker = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
                    args.model_path,
                    subfolder=args.subfolder,
                    use_safetensors=False,
                    device=args.device,
                    # 这些参数可能需要根据实际的pipeline实现调整
                    load_depth=True,  # 启用深度图支持
                    control_in_channels=1,  # 深度图单通道
                )
            elif MULTIVIEW_NORMAL_MODE or MULTIVIEW_RGB_MODE:
                # 法线模式和RGB模式不需要load_depth，直接加载标准模型
                i23d_worker = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
                    args.model_path,
                    subfolder=args.subfolder,
                    use_safetensors=False,
                    device=args.device,
                )
            else:
                # 其他情况加载标准模型
                i23d_worker = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
                    args.model_path,
                    subfolder=args.subfolder,
                    use_safetensors=False,
                    device=args.device,
                )
        except Exception as e:
            print(f"加载模型失败，回退到标准模型: {e}")
            i23d_worker = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
                args.model_path,
                subfolder=args.subfolder,
                use_safetensors=False,
                device=args.device,
            )
        
        # 如果提供了 LoRA 路径，则加载 LoRA 权重
        if args.depth_lora_path and os.path.exists(args.depth_lora_path):
            print(f"正在加载深度 LoRA 权重: {args.depth_lora_path}")
            try:
                from peft import PeftModel
                
                # 输出调试信息
                print(f"Pipeline 类型: {type(i23d_worker)}")
                if hasattr(i23d_worker, 'model'):
                    print(f"Pipeline.model 类型: {type(i23d_worker.model)}")
                    print(f"Pipeline.model 属性: {[attr for attr in dir(i23d_worker.model) if not attr.startswith('_')][:10]}")
                
                # 检查是否是Lightning checkpoint格式 (.ckpt)
                if args.depth_lora_path.endswith('.ckpt'):
                    print("检测到Lightning checkpoint格式，加载LoRA和ControlNet权重...")
                    ckpt = torch.load(args.depth_lora_path, map_location='cpu')
                    
                    if 'state_dict' in ckpt:
                        state_dict = ckpt['state_dict']
                        print(f"Checkpoint包含 {len(state_dict)} 个权重")
                        
                        # 分析checkpoint内容
                        lora_keys = [k for k in state_dict.keys() if 'lora' in k.lower()]
                        controlnet_keys = [k for k in state_dict.keys() if k.startswith('controlnet.')]
                        print(f"  - LoRA参数: {len(lora_keys)} 个")
                        print(f"  - ControlNet参数: {len(controlnet_keys)} 个")
                        
                        success_count = 0
                        
                        # 1. 加载LoRA权重
                        if lora_keys and hasattr(i23d_worker, 'model'):
                            try:
                                print("\n正在加载LoRA权重...")
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
                                i23d_worker.model = get_peft_model(i23d_worker.model, lora_config)
                                
                                # 加载包含LoRA的权重
                                missing, unexpected = i23d_worker.model.load_state_dict(
                                    model_state_dict, strict=False)
                                print(f"✅ LoRA权重加载成功")
                                print(f"  - Missing keys: {len(missing)}")
                                print(f"  - Unexpected keys: {len(unexpected)}")
                                success_count += 1
                                
                            except Exception as e:
                                print(f"❌ LoRA权重加载失败: {e}")
                                import traceback
                                traceback.print_exc()
                        
                        # 2. 加载ControlNet权重
                        if controlnet_keys:
                            try:
                                print("\n正在加载ControlNet权重...")
                                # 从checkpoint中检测视图数量
                                detected_num_views = args.num_views  # 默认值
                                for key in controlnet_keys:
                                    if 'view_embeddings.weight' in key:
                                        # 从view_embeddings.weight的形状推断视图数量
                                        weight_shape = state_dict[key].shape
                                        if len(weight_shape) >= 2:
                                            detected_num_views = weight_shape[0]
                                            print(f"  🔍 从checkpoint检测到视图数量: {detected_num_views}")
                                            # 更新args.num_views以便其他组件使用
                                            args.num_views = detected_num_views
                                        break
                                
                                # 检查pipeline是否有controlnet（可能需要重新创建）
                                if not hasattr(i23d_worker, 'controlnet') or i23d_worker.controlnet is None:
                                    print("  Pipeline中没有controlnet，尝试创建...")
                                    # 创建MultiViewDepthControlNet，使用检测到的视图数量
                                    try:
                                        from hy3dshape.models.controlnet_multiview import create_multiview_depth_controlnet
                                        i23d_worker.controlnet = create_multiview_depth_controlnet(
                                            in_channels=1,
                                            num_views=detected_num_views,
                                            out_channels=768,
                                            fusion_strategy='attention'
                                        )
                                        # 移到正确的设备和数据类型
                                        i23d_worker.controlnet = i23d_worker.controlnet.to(
                                            device=args.device,
                                            dtype=i23d_worker.dtype
                                        )
                                        print(f"  ✅ MultiViewDepthControlNet已创建并移至 {args.device}, 数据类型: {i23d_worker.dtype}, 视图数量: {detected_num_views}")
                                    except Exception as create_error:
                                        print(f"  ❌ 创建ControlNet失败: {create_error}")
                                        import traceback
                                        traceback.print_exc()
                                        i23d_worker.controlnet = None
                                
                                if i23d_worker.controlnet is not None:
                                    # 先将controlnet移到正确的设备和数据类型（在加载权重之前）
                                    target_dtype = i23d_worker.dtype
                                    i23d_worker.controlnet = i23d_worker.controlnet.to(
                                        device=args.device,
                                        dtype=target_dtype
                                    )
                                    print(f"  ControlNet已移至设备: {args.device}, 数据类型: {target_dtype}")
                                    
                                    # 提取controlnet权重（去掉'controlnet.'前缀）
                                    controlnet_state_dict = {}
                                    for key, value in state_dict.items():
                                        if key.startswith('controlnet.'):
                                            new_key = key[11:]  # 去掉'controlnet.'前缀
                                            # 确保权重的数据类型与模型一致
                                            controlnet_state_dict[new_key] = value.to(dtype=target_dtype)
                                    
                                    # 加载权重
                                    missing, unexpected = i23d_worker.controlnet.load_state_dict(
                                        controlnet_state_dict, strict=False)
                                    print(f"✅ ControlNet权重加载成功")
                                    print(f"  - Missing keys: {len(missing)}")
                                    print(f"  - Unexpected keys: {len(unexpected)}")
                                    
                                    success_count += 1
                                    
                            except Exception as e:
                                print(f"❌ ControlNet权重加载失败: {e}")
                                import traceback
                                traceback.print_exc()
                        
                        if success_count > 0:
                            print(f"\n✅ 从Lightning checkpoint成功加载 {success_count} 个组件")
                        else:
                            print("\n❌ 没有成功加载任何组件")
                            
                    else:
                        print("❌ Checkpoint格式无效，缺少state_dict")
                
                elif os.path.isdir(args.depth_lora_path):
                    # 标准的PEFT格式目录
                    print("检测到PEFT目录格式，使用标准LoRA加载方式...")
                    
                    success_count = 0
                    
                    # 加载主DiT模型的LoRA权重
                    # 注意：Pipeline.model 就是 HunYuanDiTPlain 实例，不需要再 .model
                    if hasattr(i23d_worker, 'model'):
                        try:
                            print("正在加载LoRA权重到主DiT模型...")
                            print(f"  模型类型: {type(i23d_worker.model)}")
                            print(f"  LoRA路径: {args.depth_lora_path}")
                            
                            # 直接对 pipeline.model 应用 LoRA
                            i23d_worker.model = PeftModel.from_pretrained(
                                i23d_worker.model, args.depth_lora_path)
                            
                            print("✅ 主DiT模型LoRA权重加载成功")
                            success_count += 1
                        except Exception as e:
                            print(f"❌ 主DiT模型LoRA权重加载失败: {e}")
                            import traceback
                            traceback.print_exc()
                    else:
                        print("❌ Pipeline没有model属性")
                    
                    if success_count > 0:
                        print(f"✅ 总共成功加载 {success_count} 个组件的LoRA权重")
                    else:
                        print("❌ 没有成功加载任何LoRA权重")
                        
                else:
                    print(f"❌ 不支持的LoRA权重格式: {args.depth_lora_path}")
                    
            except Exception as e:
                import traceback
                print(f"❌ 加载深度 LoRA 权重时发生异常: {e}")
                print("详细错误信息:")
                traceback.print_exc()
                print("将使用基础深度条件模型")
        else:
            if args.depth_lora_path:
                print(f"深度 LoRA 路径不存在: {args.depth_lora_path}")
            print("使用基础深度条件模型（未加载LoRA权重）")
        
        # 处理多视图法线模式的LoRA权重加载
        if MULTIVIEW_NORMAL_MODE:
            # 自动寻找法线LoRA权重路径（如果未指定）
            if args.normal_lora_path is None:
                default_lora_dirs = [
                    "./hy3dshape/output_folder/dit/normal_lora_finetuning/ckpt",
                    "./output_folder/dit/normal_lora_finetuning/ckpt",
                    "./hy3dshape/output_folder/dit/multiview_normal_lora_checkpoints",
                    "./output_folder/dit/multiview_normal_lora_checkpoints",
                ]
                
                for lora_dir in default_lora_dirs:
                    if os.path.exists(lora_dir):
                        # 如果是目录，查找.ckpt文件
                        if os.path.isdir(lora_dir):
                            ckpt_files = [f for f in os.listdir(lora_dir) if f.endswith('.ckpt')]
                            if ckpt_files:
                                # 按文件名排序，选择最新的（步数最大的）
                                latest_ckpt = max(ckpt_files, key=lambda x: int(x.split('=')[1].split('.')[0]) if '=' in x else 0)
                                args.normal_lora_path = os.path.join(lora_dir, latest_ckpt)
                                print(f"自动发现法线LoRA权重: {args.normal_lora_path}")
                                break
                        else:
                            args.normal_lora_path = lora_dir
                            break
            
            # 如果提供了法线LoRA路径，则加载LoRA权重
            if args.normal_lora_path and os.path.exists(args.normal_lora_path):
                print(f"正在加载法线 LoRA 权重: {args.normal_lora_path}")
                try:
                    from peft import PeftModel
                    
                    # 检查是否是Lightning checkpoint格式 (.ckpt)
                    if args.normal_lora_path.endswith('.ckpt'):
                        print("检测到Lightning checkpoint格式，加载法线LoRA权重...")
                        ckpt = torch.load(args.normal_lora_path, map_location='cpu')
                        
                        if 'state_dict' in ckpt:
                            state_dict = ckpt['state_dict']
                            print(f"Checkpoint包含 {len(state_dict)} 个权重")
                            
                            # 分析checkpoint内容
                            lora_keys = [k for k in state_dict.keys() if 'lora' in k.lower()]
                            print(f"  - LoRA参数: {len(lora_keys)} 个")
                            
                            success_count = 0
                            
                            # 加载LoRA权重
                            if lora_keys and hasattr(i23d_worker, 'model'):
                                try:
                                    print("\n正在加载法线LoRA权重...")
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
                                    i23d_worker.model = get_peft_model(i23d_worker.model, lora_config)
                                    
                                    # 加载包含LoRA的权重
                                    missing, unexpected = i23d_worker.model.load_state_dict(
                                        model_state_dict, strict=False)
                                    print(f"✅ 法线LoRA权重加载成功")
                                    print(f"  - Missing keys: {len(missing)}")
                                    print(f"  - Unexpected keys: {len(unexpected)}")
                                    success_count += 1
                                    
                                except Exception as e:
                                    print(f"❌ 法线LoRA权重加载失败: {e}")
                                    import traceback
                                    traceback.print_exc()
                            
                            if success_count > 0:
                                print(f"\n✅ 从Lightning checkpoint成功加载法线LoRA权重")
                            else:
                                print("\n❌ 没有成功加载法线LoRA权重")
                                
                        else:
                            print("❌ Checkpoint格式无效，缺少state_dict")
                    
                    elif os.path.isdir(args.normal_lora_path):
                        # 标准的PEFT格式目录
                        print("检测到PEFT目录格式，使用标准LoRA加载方式...")
                        
                        if hasattr(i23d_worker, 'model'):
                            try:
                                print("正在加载法线LoRA权重到主DiT模型...")
                                print(f"  模型类型: {type(i23d_worker.model)}")
                                print(f"  LoRA路径: {args.normal_lora_path}")
                                
                                # 直接对 pipeline.model 应用 LoRA
                                i23d_worker.model = PeftModel.from_pretrained(
                                    i23d_worker.model, args.normal_lora_path)
                                
                                print("✅ 法线LoRA权重加载成功")
                            except Exception as e:
                                print(f"❌ 法线LoRA权重加载失败: {e}")
                                import traceback
                                traceback.print_exc()
                        else:
                            print("❌ Pipeline没有model属性")
                    else:
                        print(f"❌ 不支持的法线LoRA权重格式: {args.normal_lora_path}")
                        
                except Exception as e:
                    import traceback
                    print(f"❌ 加载法线 LoRA 权重时发生异常: {e}")
                    print("详细错误信息:")
                    traceback.print_exc()
                    print("将使用基础法线条件模型")
            else:
                if args.normal_lora_path:
                    print(f"法线 LoRA 路径不存在: {args.normal_lora_path}")
                print("使用基础法线条件模型（未加载LoRA权重）")
        
        # 处理多视图RGB模式的LoRA权重加载
        if MULTIVIEW_RGB_MODE:
            # 自动寻找RGB LoRA权重路径（如果未指定）
            if args.rgb_lora_path is None:
                default_lora_dirs = [
                    "./hy3dshape/output_folder/dit/multiview_rgb_lora_finetuning/ckpt",
                    "./output_folder/dit/multiview_rgb_lora_finetuning/ckpt",
                    "./hy3dshape/output_folder/dit/multiview_rgb_lora_checkpoints",
                    "./output_folder/dit/multiview_rgb_lora_checkpoints",
                ]
                
                for lora_dir in default_lora_dirs:
                    if os.path.exists(lora_dir):
                        # 如果是目录，查找.ckpt文件
                        if os.path.isdir(lora_dir):
                            ckpt_files = [f for f in os.listdir(lora_dir) if f.endswith('.ckpt')]
                            if ckpt_files:
                                # 按文件名排序，选择最新的（步数最大的）
                                latest_ckpt = max(ckpt_files, key=lambda x: int(x.split('=')[1].split('.')[0]) if '=' in x else 0)
                                args.rgb_lora_path = os.path.join(lora_dir, latest_ckpt)
                                print(f"自动发现RGB LoRA权重: {args.rgb_lora_path}")
                                break
                        else:
                            args.rgb_lora_path = lora_dir
                            break
            
            # 如果提供了RGB LoRA路径，则加载LoRA权重
            if args.rgb_lora_path and os.path.exists(args.rgb_lora_path):
                print(f"正在加载RGB LoRA 权重: {args.rgb_lora_path}")
                try:
                    from peft import PeftModel
                    
                    # 检查是否是Lightning checkpoint格式 (.ckpt)
                    if args.rgb_lora_path.endswith('.ckpt'):
                        print("检测到Lightning checkpoint格式...")
                        ckpt = torch.load(args.rgb_lora_path, map_location='cpu')
                        
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
                                if hasattr(i23d_worker, 'model'):
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
                                        i23d_worker.model = get_peft_model(i23d_worker.model, lora_config)
                                        
                                        # 加载包含LoRA的权重
                                        missing, unexpected = i23d_worker.model.load_state_dict(
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
                                if hasattr(i23d_worker, 'model'):
                                    try:
                                        # 提取model相关的权重（完整权重，不含LoRA）
                                        model_state_dict = {}
                                        for key, value in state_dict.items():
                                            if key.startswith('model.'):
                                                new_key = key[6:]  # 去掉'model.'前缀
                                                model_state_dict[new_key] = value
                                        
                                        # 直接加载完整模型权重（不使用LoRA）
                                        missing, unexpected = i23d_worker.model.load_state_dict(
                                            model_state_dict, strict=False)
                                        print(f"✅ 全量微调权重加载成功")
                                        print(f"  - Missing keys: {len(missing)}")
                                        print(f"  - Unexpected keys: {len(unexpected)}")
                                        success_count += 1
                                        
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
                    
                    elif os.path.isdir(args.rgb_lora_path):
                        # 标准的PEFT格式目录
                        print("检测到PEFT目录格式，使用标准LoRA加载方式...")
                        
                        if hasattr(i23d_worker, 'model'):
                            try:
                                print("正在加载RGB LoRA权重到主DiT模型...")
                                print(f"  模型类型: {type(i23d_worker.model)}")
                                print(f"  LoRA路径: {args.rgb_lora_path}")
                                
                                # 直接对 pipeline.model 应用 LoRA
                                i23d_worker.model = PeftModel.from_pretrained(
                                    i23d_worker.model, args.rgb_lora_path)
                                
                                print("✅ RGB LoRA权重加载成功")
                            except Exception as e:
                                print(f"❌ RGB LoRA权重加载失败: {e}")
                                import traceback
                                traceback.print_exc()
                        else:
                            print("❌ Pipeline没有model属性")
                    else:
                        print(f"❌ 不支持的RGB LoRA权重格式: {args.rgb_lora_path}")
                        
                except Exception as e:
                    import traceback
                    print(f"❌ 加载RGB LoRA 权重时发生异常: {e}")
                    print("详细错误信息:")
                    traceback.print_exc()
                    print("将使用基础RGB条件模型")
            else:
                if args.rgb_lora_path:
                    print(f"RGB LoRA 路径不存在: {args.rgb_lora_path}")
                print("使用基础RGB条件模型（未加载LoRA权重）")
        
        # 无论是否加载LoRA，在多视图深度模式下都要设置正确的image processor和encoder
        if MULTIVIEW_DEPTH_MODE and hasattr(i23d_worker, 'image_processor'):
            from hy3dshape.preprocessors import MVImageProcessorV2
            from hy3dshape.models.conditioner import DinoImageEncoderMV
            
            i23d_worker.image_processor = MVImageProcessorV2(size=518)
            print("✅ 已设置MVImageProcessorV2用于多视图深度处理")
            
            # 检查并替换conditioner中的encoder
            if hasattr(i23d_worker, 'conditioner'):
                if hasattr(i23d_worker.conditioner, 'main_image_encoder'):
                    current_encoder = i23d_worker.conditioner.main_image_encoder
                    # 如果当前encoder不是DinoImageEncoderMV，则替换
                    if not isinstance(current_encoder, DinoImageEncoderMV):
                        print(f"检测到当前encoder类型: {type(current_encoder).__name__}")
                        print("正在替换为DinoImageEncoderMV...")
                        
                        # 创建新的DinoImageEncoderMV encoder，使用检测到的视图数量
                        new_encoder = DinoImageEncoderMV(
                            version='facebook/dinov2-large',
                            image_size=518,
                            use_cls_token=True,
                            view_num=args.num_views
                        )
                        
                        # 如果原encoder有已加载的模型权重，尝试复用
                        if hasattr(current_encoder, 'model') and hasattr(new_encoder, 'model'):
                            try:
                                new_encoder.model.load_state_dict(current_encoder.model.state_dict())
                                print("✅ 已复用原encoder的模型权重")
                            except Exception as e:
                                print(f"⚠️ 无法复用原encoder权重，使用默认权重: {e}")
                        
                        # 将新encoder移到相同设备和数据类型
                        new_encoder = new_encoder.to(args.device, dtype=i23d_worker.dtype)
                        
                        # 替换encoder
                        i23d_worker.conditioner.main_image_encoder = new_encoder
                        print(f"✅ 已成功替换为DinoImageEncoderMV (视图数量: {args.num_views})")
        
        # 在多视图法线模式下也要设置正确的image processor和encoder
        if MULTIVIEW_NORMAL_MODE and hasattr(i23d_worker, 'image_processor'):
            from hy3dshape.preprocessors import MVImageProcessorV2
            from hy3dshape.models.conditioner import DinoImageEncoderMV
            
            i23d_worker.image_processor = MVImageProcessorV2(size=518)
            print("✅ 已设置MVImageProcessorV2用于多视图法线处理")
            
            # 检查并替换conditioner中的encoder
            if hasattr(i23d_worker, 'conditioner'):
                if hasattr(i23d_worker.conditioner, 'main_image_encoder'):
                    current_encoder = i23d_worker.conditioner.main_image_encoder
                    # 如果当前encoder不是DinoImageEncoderMV，则替换
                    if not isinstance(current_encoder, DinoImageEncoderMV):
                        print(f"检测到当前encoder类型: {type(current_encoder).__name__}")
                        print("正在替换为DinoImageEncoderMV...")
                        
                        # 创建新的DinoImageEncoderMV encoder，使用检测到的视图数量
                        new_encoder = DinoImageEncoderMV(
                            version='facebook/dinov2-large',
                            image_size=518,
                            use_cls_token=True,
                            view_num=args.num_views
                        )
                        
                        # 如果原encoder有已加载的模型权重，尝试复用
                        if hasattr(current_encoder, 'model') and hasattr(new_encoder, 'model'):
                            try:
                                new_encoder.model.load_state_dict(current_encoder.model.state_dict())
                                print("✅ 已复用原encoder的模型权重")
                            except Exception as e:
                                print(f"⚠️ 无法复用原encoder权重，使用默认权重: {e}")
                        
                        # 将新encoder移到相同设备和数据类型
                        new_encoder = new_encoder.to(args.device, dtype=i23d_worker.dtype)
                        
                        # 替换encoder
                        i23d_worker.conditioner.main_image_encoder = new_encoder
                        print(f"✅ 已成功替换为DinoImageEncoderMV (视图数量: {args.num_views})")
        
        # 在多视图RGB模式下也要设置正确的image processor和encoder
        if MULTIVIEW_RGB_MODE and hasattr(i23d_worker, 'image_processor'):
            from hy3dshape.preprocessors import MVImageProcessorV2
            from hy3dshape.models.conditioner import DinoImageEncoderMV
            
            i23d_worker.image_processor = MVImageProcessorV2(size=518)
            print("✅ 已设置MVImageProcessorV2用于多视图RGB处理")
            
            # 检查并替换conditioner中的encoder
            if hasattr(i23d_worker, 'conditioner'):
                if hasattr(i23d_worker.conditioner, 'main_image_encoder'):
                    current_encoder = i23d_worker.conditioner.main_image_encoder
                    # 如果当前encoder不是DinoImageEncoderMV，则替换
                    if not isinstance(current_encoder, DinoImageEncoderMV):
                        print(f"检测到当前encoder类型: {type(current_encoder).__name__}")
                        print("正在替换为DinoImageEncoderMV...")
                        
                        # 创建新的DinoImageEncoderMV encoder，使用检测到的视图数量
                        new_encoder = DinoImageEncoderMV(
                            version='facebook/dinov2-large',
                            image_size=518,
                            use_cls_token=True,
                            view_num=args.num_views
                        )
                        
                        # 如果原encoder有已加载的模型权重，尝试复用
                        if hasattr(current_encoder, 'model') and hasattr(new_encoder, 'model'):
                            try:
                                new_encoder.model.load_state_dict(current_encoder.model.state_dict())
                                print("✅ 已复用原encoder的模型权重")
                            except Exception as e:
                                print(f"⚠️ 无法复用原encoder权重，使用默认权重: {e}")
                        
                        # 将新encoder移到相同设备和数据类型
                        new_encoder = new_encoder.to(args.device, dtype=i23d_worker.dtype)
                        
                        # 替换encoder
                        i23d_worker.conditioner.main_image_encoder = new_encoder
                        print(f"✅ 已成功替换为DinoImageEncoderMV (视图数量: {args.num_views})")
    elif IS_MV_PRETRAINED:
        # 加载标准的Hunyuan3D-2mv预训练模型（从Hugging Face）
        print("正在加载Hunyuan3D-2mv预训练模型...")
        print(f"  模型路径: {args.model_path}")
        print(f"  Subfolder: {args.subfolder}")
        
        # 检查是否有自定义模型路径
        custom_model_path = os.environ.get('HY3DGEN_MODELS')
        if custom_model_path:
            print(f"  使用自定义模型路径: {custom_model_path}")
        
        # 尝试查找模型文件的实际位置
        possible_paths = [
            os.path.expanduser('~/.cache/hy3dgen'),
            '/root/autodl-tmp/package/hy3dgen',
            os.path.expanduser('~/autodl-tmp/package/hy3dgen'),
            './models',
        ]
        
        model_found = False
        for base_path in possible_paths:
            test_path = os.path.join(base_path, args.model_path, args.subfolder, 'model.fp16.safetensors')
            if os.path.exists(test_path):
                print(f"  ✅ 找到模型文件: {test_path}")
                # 设置环境变量指向找到的路径
                os.environ['HY3DGEN_MODELS'] = base_path
                model_found = True
                break
        
        if not model_found:
            print("  ⚠️ 未找到模型文件，将从 Hugging Face 下载")
        
        try:
            i23d_worker = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
                args.model_path,
                subfolder=args.subfolder,
                use_safetensors=True,  # mv模型使用safetensors格式
                device=args.device,
            )
            print("✅ Hunyuan3D-2mv模型加载成功")
        except Exception as e:
            import traceback
            error_msg = str(e)
            print(f"⚠️ 使用safetensors加载失败: {error_msg}")
            print(f"  详细错误信息:")
            traceback.print_exc()
            
            # 如果是模块导入错误，尝试修复配置文件中的模块路径
            if 'hy3dgen' in error_msg.lower() or 'No module named' in error_msg:
                print("  🔧 检测到模块导入错误，尝试修复...")
                # 这里可以尝试修复配置文件，但更简单的方法是直接使用 ckpt
                print("  💡 建议：检查配置文件中的模块路径是否正确")
            
            try:
                print("  🔄 尝试使用ckpt格式...")
                i23d_worker = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
                    args.model_path,
                    subfolder=args.subfolder,
                    use_safetensors=False,
                    device=args.device,
                )
                print("✅ Hunyuan3D-2mv模型加载成功（使用ckpt格式）")
            except Exception as e2:
                print(f"❌ Hunyuan3D-2mv模型加载失败: {e2}")
                import traceback
                traceback.print_exc()
                print("\n💡 故障排除建议:")
                print("  1. 检查模型文件是否存在:")
                print(f"     ls -lh ~/.cache/hy3dgen/{args.model_path}/{args.subfolder}/")
                print("  2. 设置环境变量指向模型路径:")
                print(f"     export HY3DGEN_MODELS=/root/autodl-tmp/package/hy3dgen")
                print("  3. 检查配置文件中的模块路径是否正确")
                raise
    else:
        print("正在加载标准RGB模型...")
        i23d_worker = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            args.model_path,
            subfolder=args.subfolder,
            use_safetensors=False,
            device=args.device,
        )
    if args.enable_flashvdm:
        mc_algo = 'mc' if args.device in ['cpu', 'mps'] else args.mc_algo
        i23d_worker.enable_flashvdm(mc_algo=mc_algo)
    if args.compile:
        i23d_worker.compile()
    
    # 多GPU模型并行配置
    if args.use_multi_gpu:
        primary_gpu = int(args.gpu_ids.split(',')[0].strip())
        args.device = f'cuda:{primary_gpu}'
        print(f"🚀 启用多GPU模型并行，主GPU: {primary_gpu}")
        i23d_worker = setup_multi_gpu_model_parallel(i23d_worker, args.gpu_ids)
        # 清理所有GPU的显存缓存
        for gpu_id in [int(x.strip()) for x in args.gpu_ids.split(',') if x.strip()]:
            torch.cuda.set_device(gpu_id)
            torch.cuda.empty_cache()
    
    # 在 low_vram_mode 下启用 CPU offload 以减少显存使用
    # 注意：多GPU模式和CPU offload不能同时使用
    if args.low_vram_mode and args.device.startswith('cuda') and not args.use_multi_gpu:
        print("启用 low_vram_mode: 使用 CPU offload 减少显存占用...")
        try:
            # 先尝试使用 pipeline 自带的 enable_model_cpu_offload 方法
            if hasattr(i23d_worker, 'components'):
                i23d_worker.enable_model_cpu_offload(gpu_id=int(args.device.split(':')[1]) if ':' in args.device else 0)
                print("✅ CPU offload 已启用（使用 pipeline 自带方法）")
            else:
                # 如果没有 components 属性，使用自定义的 CPU offload 函数
                gpu_id = int(args.device.split(':')[1]) if ':' in args.device else 0
                enable_cpu_offload_for_pipeline(i23d_worker, gpu_id=gpu_id)
                print("✅ CPU offload 已启用（使用自定义方法）")
        except Exception as e:
            print(f"⚠️ CPU offload 启用失败，将使用常规模式: {e}")
            import traceback
            traceback.print_exc()
        # 清理显存缓存
        torch.cuda.empty_cache()
    elif args.low_vram_mode and args.use_multi_gpu:
        print("⚠️ 多GPU模式和CPU offload不能同时使用，已禁用CPU offload")

    floater_remove_worker = FloaterRemover()
    degenerate_face_remove_worker = DegenerateFaceRemover()
    face_reduce_worker = FaceReducer()

    # https://discuss.huggingface.co/t/how-to-serve-an-html-file/33921/2
    # create a FastAPI app
    app = FastAPI()
    
    # create a static directory to store the static files
    static_dir = Path(SAVE_DIR).absolute()
    static_dir.mkdir(parents=True, exist_ok=True)
    app.mount("/static", StaticFiles(directory=static_dir, html=True), name="static")
    shutil.copytree('./assets/env_maps', os.path.join(static_dir, 'env_maps'), dirs_exist_ok=True)

    if args.low_vram_mode:
        torch.cuda.empty_cache()
    demo = build_app()
    app = gr.mount_gradio_app(app, demo, path="/")
    uvicorn.run(app, host=args.host, port=args.port)
