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


import argparse, sys, os, math, re, glob
from typing import *
import bpy
from mathutils import Vector, Matrix
import numpy as np
import json
import glob
import random
import shutil
import mathutils
import cv2

"""=============== BLENDER ==============="""

IMPORT_FUNCTIONS: Dict[str, Callable] = {
    "obj": bpy.ops.wm.obj_import,
    "glb": bpy.ops.import_scene.gltf,
    "gltf": bpy.ops.import_scene.gltf,
    "usd": bpy.ops.import_scene.usd,
    "fbx": bpy.ops.import_scene.fbx,
    "stl": bpy.ops.import_mesh.stl,
    "usda": bpy.ops.import_scene.usda,
    "dae": bpy.ops.wm.collada_import,
    "ply": bpy.ops.import_mesh.ply,
    "abc": bpy.ops.wm.alembic_import,
    "blend": bpy.ops.wm.append,
}

EXT = {
    'PNG': 'png',
    'JPEG': 'jpg',
    'OPEN_EXR': 'exr',
    'TIFF': 'tiff',
    'BMP': 'bmp',
    'HDR': 'hdr',
    'TARGA': 'tga'
}

PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53]

def radical_inverse(base, n):
    val = 0
    inv_base = 1.0 / base
    inv_base_n = inv_base
    while n > 0:
        digit = n % base
        val += digit * inv_base_n
        n //= base
        inv_base_n *= inv_base
    return val

def halton_sequence(dim, n):
    return [radical_inverse(PRIMES[dim], n) for dim in range(dim)]

def hammersley_sequence(dim, n, num_samples):
    return [n / num_samples] + halton_sequence(dim - 1, n)

def sphere_hammersley_sequence(n, num_samples, offset=(0, 0)):
    u, v = hammersley_sequence(2, n, num_samples)
    u += offset[0] / num_samples
    v += offset[1]
    u = 2 * u if u < 0.25 else 2 / 3 * u + 1 / 3
    theta = np.arccos(1 - 2 * u) - np.pi / 2
    phi = v * 2 * np.pi
    return [phi, theta]

def trellis_cond_camera_sequence(num_cond_views, camera_offset=0.0):
    """
    生成多视图相机序列（Hammersley低差异序列）
    
    Args:
        num_cond_views: 视图数量
        camera_offset: 相机角度偏移（弧度），用于补偿mesh旋转
                       如果mesh从x轴旋转到z轴（绕Y轴-90度），相机角度需要加90度（π/2）
    """
    yaws = []
    pitchs = []
    offset = (np.random.rand(), np.random.rand())
    for i in range(num_cond_views):
        y, p = sphere_hammersley_sequence(i, num_cond_views, offset)
        # 应用相机角度偏移（如果mesh旋转了，相机角度也需要相应调整）
        y = y + camera_offset
        yaws.append(y)
        pitchs.append(p)
    fov_min, fov_max = 10, 70
    radius_min = np.sqrt(3) / 2 / np.sin(fov_max / 360 * np.pi)
    radius_max = np.sqrt(3) / 2 / np.sin(fov_min / 360 * np.pi)
    k_min = 1 / radius_max**2
    k_max = 1 / radius_min**2
    ks = np.random.uniform(k_min, k_max, (1000000,))
    radius = [1 / np.sqrt(k) for k in ks]
    fov = [2 * np.arcsin(np.sqrt(3) / 2 / r) for r in radius]

    views = [{'hangle': y, 'vangle': p, 'cam_dis': r, 'fov': f, 'proj_type': 0} \
             for y, p, r, f in zip(yaws, pitchs, radius, fov)]
    return views

def orthogonal_camera_sequence():
    yaws = [-0.5 * np.pi, 0, 0.5 * np.pi, np.pi, -0.5 * np.pi, -0.5 * np.pi]
    pitchs = [0, 0, 0, 0, 0.5 * np.pi, -0.5 * np.pi]
    radius = [1.5 for i in range(6)]
    fov = [1.5 * np.arcsin(np.sqrt(3) / 2 / r) for r in radius]
    views = [{'hangle': y, 'vangle': p, 'cam_dis': r, 'fov': f, 'proj_type': 1} \
             for y, p, r, f in zip(yaws, pitchs, radius, fov)]
    return views

def four_view_camera_sequence(camera_offset=0.0):
    """
    生成4个视角的相机序列：前、右、后、左
    使用固定的水平角度和垂直角度，确保拍摄前后左右四个方向
    
    Args:
        camera_offset: 相机角度偏移（弧度），用于补偿mesh旋转
                       如果mesh从x轴旋转到z轴（绕Y轴-90度），相机角度需要加90度（π/2）
    """
    # 前、右、后、左四个方向的水平角度 (yaw)
    # 0: 前方, π/2: 右方, π: 后方, 3π/2: 左方
    base_yaws = [0, 0.5 * np.pi, np.pi, 1.5 * np.pi]
    # 应用相机角度偏移
    yaws = [y + camera_offset for y in base_yaws]
    
    # 垂直角度 (pitch) 设为0，保持水平视角
    pitchs = [0, 0, 0, 0]
    
    # 固定相机距离
    radius = [1.4 for i in range(4)]
    
    # 固定FOV，使用透视投影
    fov = [0.8 for i in range(4)]  # 约45度视角
    
    views = [{'hangle': y, 'vangle': p, 'cam_dis': r, 'fov': f, 'proj_type': 0} \
             for y, p, r, f in zip(yaws, pitchs, radius, fov)]
    return views

def switch_to_mr_render(render_base_color, output_nodes):
    bpy.context.scene.view_settings.view_transform = 'Raw'
    bpy.context.scene.use_nodes = True
    tree = bpy.context.scene.node_tree
    links = tree.links

    for i in range(len(output_nodes)):
        if i + 1 != len(output_nodes):
            for l in output_nodes[i][1].links:
                links.remove(l)
        else:
            links.new(output_nodes[i][0], output_nodes[i][1])

    for material in bpy.data.materials:
        if not material.use_nodes:
            continue
        bsdf_node = None
        output_node = None
        node_tree = material.node_tree
        links = material.node_tree.links
        nodes = node_tree.nodes
        for node in node_tree.nodes:
            # Check if the node is a BSDF node
            if node.type == 'BSDF_PRINCIPLED':
                bsdf_node = node
            if node.type == 'OUTPUT_MATERIAL':
                output_node = node
        if bsdf_node is None or output_node is None:
            continue
        #bsdf_node.inputs['Emission'].default_value = 0
        bsdf_node.inputs['Emission Strength'].default_value = 0
        mr_node = None
        bc_node = None
        for node in node_tree.nodes:
            # Check if the node is a BSDF node
            if node.name == 'COMBINE_METALLIC_ROUGHNESS':
                mr_node = node
            if node.name == 'COMBINE_BASE_COLOR':
                bc_node = node
        if mr_node is None:
            combine_rgb_node = nodes.new('ShaderNodeCombineColor')
            #combine_rgb_node.name = 'COMBINE_METALLIC_ROUGHNESS'
            
            # Optionally, set the RGB values
            combine_rgb_node.inputs['Red'].default_value = 1.0
            combine_rgb_node.inputs['Green'].default_value = 0.5
            combine_rgb_node.inputs['Blue'].default_value = 0.0
            metallic_input = bsdf_node.inputs["Metallic"]

            if metallic_input.links:
                source_endpoint = metallic_input.links[0].from_socket
                links.new(source_endpoint, combine_rgb_node.inputs['Blue'])

            roughness_input = bsdf_node.inputs['Roughness']
            if roughness_input.links:
                source_endpoint = roughness_input.links[0].from_socket
                links.new(source_endpoint, combine_rgb_node.inputs['Green'])

            emission_shader = nodes.new("ShaderNodeEmission")
            emission_shader.inputs["Strength"].default_value = 1
            links.new(combine_rgb_node.outputs["Color"], emission_shader.inputs["Color"])

            mix_shader = nodes.new("ShaderNodeMixShader")
            mix_shader.name = 'COMBINE_METALLIC_ROUGHNESS'
            links.new(bsdf_node.outputs["BSDF"], mix_shader.inputs[1])
            links.new(emission_shader.outputs["Emission"], mix_shader.inputs[2])
            mr_node = mix_shader

            mix_shader_bc = nodes.new("ShaderNodeMixShader")
            mix_shader_bc.name = 'COMBINE_BASE_COLOR'
            
            if len(bsdf_node.inputs['Base Color'].links) > 0:
                socket = bsdf_node.inputs['Base Color'].links[0].from_socket
                gamma_node = node_tree.nodes.new(type='ShaderNodeGamma')

                gamma_node.inputs[1].default_value = 0.454
                node_tree.links.new(socket, gamma_node.inputs[0])
                node_tree.links.new(gamma_node.outputs[0], mix_shader_bc.inputs[1])

            links.new(mix_shader.outputs["Shader"], mix_shader_bc.inputs[2])
            bc_node = mix_shader_bc

            for l in output_node.inputs['Surface'].links:
                links.remove(l)
            links.new(mix_shader_bc.outputs["Shader"], output_node.inputs["Surface"])

        mr_node.inputs["Fac"].default_value = 1.0
        if render_base_color:
            bc_node.inputs['Fac'].default_value = 0.0
        else:
            bc_node.inputs['Fac'].default_value = 1.0

def switch_to_color_render(output_nodes):
    bpy.context.scene.view_settings.view_transform = 'Standard'
    bpy.context.scene.use_nodes = True
    tree = bpy.context.scene.node_tree
    links = tree.links

    for i in range(len(output_nodes)):
        if i + 1 == len(output_nodes):
            for l in output_nodes[i][1].links:
                links.remove(l)
        else:
            links.new(output_nodes[i][0], output_nodes[i][1])

    for material in bpy.data.materials:
        if not material.use_nodes:
            continue
        node_tree = material.node_tree
        links = material.node_tree.links
        nodes = node_tree.nodes
        mr_node = None
        bc_node = None
        for node in node_tree.nodes:
            if node.name == 'COMBINE_METALLIC_ROUGHNESS':
                mr_node = node
            if node.name == 'COMBINE_BASE_COLOR':
                bc_node = node
        if mr_node is not None and bc_node is not None:
            mr_node.inputs["Fac"].default_value = 0.0
            if len(bc_node.inputs[1].links) > 0:
                try:
                    node = bc_node.inputs[1].links[0].from_socket.node
                    node.image.colorspace_settings.name = 'sRGB'
                except:
                    pass

# def ConvertNormalMap(input_exr, output_jpg):
#     import OpenEXR
#     import Imath
#     file = OpenEXR.InputFile(input_exr)
#     channels = file.header()['channels'].keys()

#     # Get the image data
#     data_window = file.header()['dataWindow']
#     width = data_window.max.x - data_window.min.x + 1
#     height = data_window.max.y - data_window.min.y + 1

#     # Read the X, Y, and Z channels as 32-bit floats
#     x_channel = np.frombuffer(file.channel('X', Imath.PixelType(Imath.PixelType.FLOAT)), dtype=np.float32)
#     y_channel = np.frombuffer(file.channel('Y', Imath.PixelType(Imath.PixelType.FLOAT)), dtype=np.float32)
#     z_channel = np.frombuffer(file.channel('Z', Imath.PixelType(Imath.PixelType.FLOAT)), dtype=np.float32)

#     # Reshape the channels into 2D arrays
#     x_channel = x_channel.reshape((height, width))
#     y_channel = y_channel.reshape((height, width))
#     z_channel = z_channel.reshape((height, width))

#     # Stack the channels to create a 3D array
#     normal = np.stack((x_channel, y_channel, z_channel), axis=-1)
#     normal = ((normal * 0.5 + 0.5) * 255).astype('uint8')
#     cv2.imwrite(output_jpg, normal)

def ConvertNormalMap_png(input_exr, output_jpg):
    # Read EXR file with OpenCV (returns float32 image)
    exr_img = cv2.imread(input_exr, cv2.IMREAD_UNCHANGED)
    if exr_img is None:
        raise RuntimeError(f"Failed to load EXR file: {input_exr}")
    print(f"EXR shape: {exr_img.shape}, dtype: {exr_img.dtype}")
    
    # Handle different channel counts
    if len(exr_img.shape) == 3 and exr_img.shape[2] == 2:
        # 2-channel normal map: add a third channel (Z component)
        z_channel = np.sqrt(np.maximum(0, 1 - exr_img[:,:,0]**2 - exr_img[:,:,1]**2))
        exr_img = np.dstack([exr_img, z_channel])
    
    normal = ((exr_img * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8)
    cv2.imwrite(output_jpg, normal)
    print(f"Saved normal map to {output_jpg}")
    
    
def ConvertNormalMap(input_exr, output_png, depth_exr=None, normalize_to_front=False, view_index=0):
    """
    转换法线图从EXR到PNG格式，参考visualize_normal.py的处理逻辑
    使用OpenEXR库正确读取X、Y、Z通道的法线数据
    
    Args:
        input_exr: 法线图EXR文件路径
        output_png: 输出PNG文件路径
        depth_exr: 深度图EXR文件路径（用于生成遮罩）
        normalize_to_front: 是否将法线标准化为正视图方向
        view_index: 当前视角索引（0为正视图）
    """
    try:
        import OpenEXR
        import Imath
        HAS_OPENEXR = True
    except ImportError:
        HAS_OPENEXR = False
        print("⚠️  OpenEXR库未安装，将使用OpenCV读取EXR文件")
    
    normal = None
    
    # 优先使用OpenEXR库读取法线数据
    if HAS_OPENEXR:
        try:
            exr_file = OpenEXR.InputFile(input_exr)
            header = exr_file.header()
            
            # 获取图像尺寸
            dw = header['dataWindow']
            width = dw.max.x - dw.min.x + 1
            height = dw.max.y - dw.min.y + 1
            
            # 读取法线通道 (X, Y, Z)
            channels = ['X', 'Y', 'Z']
            channel_data = {}
            
            for channel in channels:
                if channel in header['channels']:
                    channel_str = exr_file.channel(channel, Imath.PixelType(Imath.PixelType.FLOAT))
                    channel_data[channel] = np.frombuffer(channel_str, dtype=np.float32)
                    channel_data[channel] = channel_data[channel].reshape((height, width))
                else:
                    # 如果某个通道不存在，用0填充
                    channel_data[channel] = np.zeros((height, width), dtype=np.float32)
            
            # 组合成3通道图像 (X, Y, Z)
            normal = np.stack([channel_data['X'], channel_data['Y'], channel_data['Z']], axis=2)
            exr_file.close()
            
            print(f"✅ OpenEXR成功读取法线: {normal.shape}, dtype: {normal.dtype}")
            
        except Exception as e:
            print(f"⚠️  OpenEXR读取失败: {e}，尝试OpenCV...")
    
    # 如果OpenEXR失败，使用OpenCV作为备用
    if normal is None:
        exr_img = cv2.imread(input_exr, cv2.IMREAD_UNCHANGED)
        if exr_img is None:
            raise RuntimeError(f"Failed to load EXR file: {input_exr}")
        print(f"EXR shape: {exr_img.shape}, dtype: {exr_img.dtype}")
        
        # Handle different channel counts
        if len(exr_img.shape) == 3 and exr_img.shape[2] == 2:
            # 2-channel normal map: add a third channel (Z component)
            z_channel = np.sqrt(np.maximum(0, 1 - exr_img[:,:,0]**2 - exr_img[:,:,1]**2))
            exr_img = np.dstack([exr_img, z_channel])
        
        normal = exr_img
    
    # 处理无效值
    normal = np.nan_to_num(normal, nan=0.0, posinf=1.0, neginf=-1.0)
    
    # 检查数据范围，判断是否需要从[0,1]转换到[-1,1]
    data_min = normal.min()
    data_max = normal.max()
    
    # 检查是否有有效数据（非零值）
    non_zero_mask = np.any(normal != 0, axis=2)
    non_zero_count = np.sum(non_zero_mask)
    
    print(f"📊 处理前数据范围: [{data_min:.6f}, {data_max:.6f}]")
    print(f"📊 非零像素: {non_zero_count} / {normal.size//3} ({non_zero_count/(normal.size//3)*100:.2f}%)")
    
    if non_zero_count > 0:
        # 如果有非零数据，检查数据范围
        if data_min >= 0.0 and data_max <= 1.0:
            print(f"🔄 检测到[0,1]范围数据，转换到[-1,1]")
            normal = normal * 2.0 - 1.0
        elif data_min >= -1.0 and data_max <= 1.0:
            print(f"✅ 数据已在[-1,1]范围内")
        else:
            print(f"⚠️  数据范围异常: [{data_min:.6f}, {data_max:.6f}]")
            print(f"🔄 尝试归一化到[-1,1]范围")
            # 尝试归一化到[-1,1]范围
            if data_max > data_min:
                normal = 2.0 * (normal - data_min) / (data_max - data_min) - 1.0
    else:
        print(f"⚠️  警告: 没有检测到有效数据，所有像素都是0")
    
    # 最终裁剪到[-1,1]范围
    normal = np.clip(normal, -1.0, 1.0)
    
    # normalize_to_front参数保留用于兼容性，但实际逻辑已在渲染循环中处理
    
    # 创建遮罩：使用深度图或法线图本身
    if depth_exr and os.path.exists(depth_exr):
        # 使用深度图创建遮罩
        print(f"使用深度图遮罩: {depth_exr}")
        depth = cv2.imread(depth_exr, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        if depth is not None:
            # 如果是多通道，取第一个通道
            if len(depth.shape) == 3:
                depth = depth[:, :, 0]
            
            # 处理无效值（与深度图转换逻辑一致）
            depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
            depth[depth > 100.0] = 0.0  # 超过100米视为无效
            depth[depth < 0] = 0.0      # 负值视为无效
            
            # 创建有效区域遮罩
            valid_mask = depth > 0
        else:
            print("⚠️  深度图读取失败，使用法线图自身遮罩")
            normal_magnitude = np.linalg.norm(normal, axis=2)
            valid_mask = normal_magnitude > 0.001
    else:
        # 使用法线图自身创建遮罩
        print("使用法线图自身遮罩")
        normal_magnitude = np.linalg.norm(normal, axis=2)
        valid_mask = normal_magnitude > 0.001
    
    # 将法线从 [-1, 1] 范围转换到 [0, 1] 范围
    normal_01 = normal * 0.5 + 0.5
    
    # 坐标系统转换：Blender -> 标准法线图格式
    # 标准法线图：R=X(右), G=Y(上), B=Z(前)
    # Blender法线：X=右, Y=前, Z=上
    # 需要重新排列：R=Blender_X, G=Blender_Z, B=Blender_Y
    normal_converted = normal_01.copy()
    normal_converted[:, :, 0] = normal_01[:, :, 0]  # Blender X -> R (右)
    normal_converted[:, :, 1] = normal_01[:, :, 2]  # Blender Z -> G (上) 
    normal_converted[:, :, 2] = normal_01[:, :, 1]  # Blender Y -> B (前)
    
    # 转换到 [0, 255] 范围
    normal_uint8 = (normal_converted * 255).clip(0, 255).astype(np.uint8)
    
    # 应用遮罩：无效区域设为0
    normal_uint8[~valid_mask] = [0, 0, 0]
    
    # 调试信息
    valid_pixels = np.sum(valid_mask)
    total_pixels = valid_mask.size
    print(f"有效像素: {valid_pixels}/{total_pixels} ({valid_pixels/total_pixels*100:.1f}%)")
    
    if valid_pixels > 0:
        valid_colors = normal_uint8[valid_mask]
        avg_color = np.mean(valid_colors, axis=0)
        print(f"平均颜色 (R, G, B): {avg_color}")
        
        # 计算法线长度统计
        valid_normal = normal[valid_mask]
        normal_lengths = np.linalg.norm(valid_normal, axis=1)
        print(f"法线长度范围: [{normal_lengths.min():.3f}, {normal_lengths.max():.3f}] (平均: {normal_lengths.mean():.3f})")
    
    cv2.imwrite(output_png, normal_uint8)
    if normalize_to_front:
        print(f"Saved normalized normal map to {output_png} (front-view normalized)")
    else:
        print(f"Saved normal map to {output_png} (with improved processing)")


def ConvertDepthMap(input_exr, output_png, max_valid_depth=100.0, debug=False):
    """
    转换深度图从EXR到PNG格式，参考generate_inference_dataset.py的处理逻辑
    
    Args:
        input_exr: 输入EXR文件路径
        output_png: 输出PNG文件路径
        max_valid_depth: 最大有效深度值（米），超过此值视为无效
        debug: 是否打印调试信息
    """
    try:
        # 读取EXR文件
        depth = cv2.imread(input_exr, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        if depth is None:
            raise RuntimeError(f"Failed to load EXR file: {input_exr}")
        
        # 如果是多通道，取第一个通道
        if len(depth.shape) == 3:
            depth = depth[:, :, 0]
        
        if debug:
            print(f"EXR shape: {depth.shape}, dtype: {depth.dtype}")
        
        # 处理无效值
        # 1. 替换inf和nan为0
        depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 2. 将超过最大有效深度的值设为0（标记为无效/背景）
        depth[depth > max_valid_depth] = 0.0
        
        # 3. 将负值设为0
        depth[depth < 0] = 0.0
        
        if debug:
            valid_mask = depth > 0
            if np.any(valid_mask):
                print(f"   深度范围: [{np.min(depth[valid_mask]):.3f}, {np.max(depth[valid_mask]):.3f}] 米")
                print(f"   有效像素: {np.sum(valid_mask)} / {depth.size} ({np.sum(valid_mask)/depth.size*100:.2f}%)")
            else:
                print(f"   ⚠️  警告: 没有有效深度值！")
        
        # 转换为毫米并保存为16位PNG
        depth_mm = depth * 1000.0  # 转换为毫米
        
        # 限制在16位范围内
        depth_uint16 = np.clip(depth_mm, 0, 65535).astype(np.uint16)
        
        if debug:
            valid_mask = depth_uint16 > 0
            if np.any(valid_mask):
                print(f"   深度(mm)范围: [{np.min(depth_uint16[valid_mask])}, {np.max(depth_uint16[valid_mask])}]")
            clipped = np.sum(depth_mm > 65535)
            if clipped > 0:
                print(f"   ⚠️  警告: {clipped} 个像素深度超过65535mm，已截断")
        
        # 保存为PNG
        cv2.imwrite(output_png, depth_uint16)
        print(f"Saved depth map to {output_png} (16-bit, scaled by 1000)")
        return True
        
    except Exception as e:
        print(f"❌ 转换深度图失败: {e}")
        return False


def init_render(engine='CYCLES', resolution=512, geo_mode=False):
    bpy.context.scene.render.engine = engine
    bpy.context.scene.render.resolution_x = resolution
    bpy.context.scene.render.resolution_y = resolution
    bpy.context.scene.render.resolution_percentage = 100
    bpy.context.scene.render.image_settings.file_format = 'PNG'
    bpy.context.scene.render.image_settings.color_mode = 'RGBA'
    bpy.context.scene.render.film_transparent = True
    
    bpy.context.scene.cycles.device = 'GPU'
    #bpy.context.scene.cycles.samples = 128 if not geo_mode else 1
    bpy.context.scene.cycles.filter_type = 'BOX'
    bpy.context.scene.cycles.filter_width = 1
    bpy.context.scene.cycles.diffuse_bounces = 1
    bpy.context.scene.cycles.glossy_bounces = 1
    # bpy.context.scene.cycles.transparent_max_bounces = 3 if not geo_mode else 0
    # bpy.context.scene.cycles.transmission_bounces = 3 if not geo_mode else 1
    bpy.context.scene.cycles.use_denoising = True
        
    bpy.context.preferences.addons['cycles'].preferences.get_devices()
    # bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'CUDA'
    
def init_nodes(save_depth=False, save_normal=False, save_albedo=False, save_mr = False, save_mist=False):
    if not any([save_depth, save_normal, save_albedo, save_mist]):
        return {}, {}, []
    outputs = {}
    spec_nodes = {}
    composite_nodes = []
    bpy.context.scene.use_nodes = True
    bpy.context.scene.view_layers['ViewLayer'].use_pass_z = save_depth
    bpy.context.scene.view_layers['ViewLayer'].use_pass_normal = save_normal
    bpy.context.scene.view_layers['ViewLayer'].use_pass_diffuse_color = save_albedo
    bpy.context.scene.view_layers['ViewLayer'].use_pass_mist = save_mist
    
    nodes = bpy.context.scene.node_tree.nodes
    links = bpy.context.scene.node_tree.links
    for n in nodes:
        nodes.remove(n)
    
    render_layers = nodes.new('CompositorNodeRLayers')
    
    if save_depth:
        depth_file_output = nodes.new('CompositorNodeOutputFile')
        depth_file_output.base_path = '/'  # 设置为根目录，使绝对路径生效
        depth_file_output.file_slots[0].use_node_format = True
        depth_file_output.format.file_format = "OPEN_EXR"
        links.new(render_layers.outputs['Depth'], depth_file_output.inputs[0])
        
        outputs['depth'] = depth_file_output
        composite_nodes.append((render_layers.outputs['Depth'], depth_file_output.inputs[0]))
    
    if save_normal:
        normal_file_output = nodes.new('CompositorNodeOutputFile')
        normal_file_output.base_path = '/'  # 设置为根目录，使绝对路径生效
        normal_file_output.file_slots[0].use_node_format = True
        normal_file_output.format.file_format = 'OPEN_EXR'
        links.new(render_layers.outputs['Normal'], normal_file_output.inputs[0])
        
        outputs['normal'] = normal_file_output
        composite_nodes.append((render_layers.outputs['Normal'], normal_file_output.inputs[0]))
    
    if save_albedo:
        albedo_file_output = nodes.new('CompositorNodeOutputFile')
        albedo_file_output.base_path = '/'  # 设置为根目录，使绝对路径生效
        albedo_file_output.file_slots[0].use_node_format = True
        albedo_file_output.format.file_format = 'PNG'
        albedo_file_output.format.color_mode = 'RGBA'
        albedo_file_output.format.color_depth = '8'
        
        alpha_albedo = nodes.new('CompositorNodeSetAlpha')
        
        links.new(render_layers.outputs['DiffCol'], alpha_albedo.inputs['Image'])
        links.new(render_layers.outputs['Alpha'], alpha_albedo.inputs['Alpha'])
        links.new(alpha_albedo.outputs['Image'], albedo_file_output.inputs[0])
        
        outputs['albedo'] = albedo_file_output
        #composite_nodes.append((alpha_albedo.outputs['Image'], albedo_file_output.inputs[0]))

    if save_mr:
        mr_file_output = nodes.new('CompositorNodeOutputFile')
        mr_file_output.base_path = '/'  # 设置为根目录，使绝对路径生效
        mr_file_output.file_slots[0].use_node_format = True
        mr_file_output.format.file_format = 'OPEN_EXR'
        
        links.new(render_layers.outputs['Image'], mr_file_output.inputs[0])

        outputs['mr'] = mr_file_output
        composite_nodes.append((render_layers.outputs['Image'], mr_file_output.inputs[0]))
        
    if save_mist:
        bpy.data.worlds['World'].mist_settings.start = 0
        bpy.data.worlds['World'].mist_settings.depth = 10
        
        mist_file_output = nodes.new('CompositorNodeOutputFile')
        mist_file_output.base_path = '/'  # 设置为根目录，使绝对路径生效
        mist_file_output.file_slots[0].use_node_format = True
        mist_file_output.format.file_format = 'PNG'
        mist_file_output.format.color_mode = 'BW'
        mist_file_output.format.color_depth = '16'
        
        links.new(render_layers.outputs['Mist'], mist_file_output.inputs[0])
        
        outputs['mist'] = mist_file_output
        composite_nodes.append((render_layers.outputs['Mist'], mist_file_output.inputs[0]))

    return outputs, spec_nodes, composite_nodes

def init_scene() -> None:
    """Resets the scene to a clean state.

    Returns:
        None
    """
    # delete everything
    for obj in bpy.data.objects:
        bpy.data.objects.remove(obj, do_unlink=True)

    # delete all the materials
    for material in bpy.data.materials:
        bpy.data.materials.remove(material, do_unlink=True)

    # delete all the textures
    for texture in bpy.data.textures:
        bpy.data.textures.remove(texture, do_unlink=True)

    # delete all the images
    for image in bpy.data.images:
        bpy.data.images.remove(image, do_unlink=True)

def init_camera():
    cam = bpy.data.objects.new('Camera', bpy.data.cameras.new('Camera'))
    bpy.context.collection.objects.link(cam)
    bpy.context.scene.camera = cam
    cam.data.sensor_height = cam.data.sensor_width = 32
    cam_constraint = cam.constraints.new(type='TRACK_TO')
    cam_constraint.track_axis = 'TRACK_NEGATIVE_Z'
    cam_constraint.up_axis = 'UP_Y'
    cam_empty = bpy.data.objects.new("Empty", None)
    cam_empty.location = (0, 0, 0)
    bpy.context.scene.collection.objects.link(cam_empty)
    cam_constraint.target = cam_empty
    return cam

def init_lighting():
    # Clear existing lights
    bpy.ops.object.select_all(action="DESELECT")
    bpy.ops.object.select_by_type(type="LIGHT")
    bpy.ops.object.delete()
    
    # Create key light
    default_light = bpy.data.objects.new("Default_Light", bpy.data.lights.new("Default_Light", type="POINT"))
    bpy.context.collection.objects.link(default_light)
    default_light.data.energy = 1000
    default_light.location = (4, 1, 6)
    default_light.rotation_euler = (0, 0, 0)
    
    # create top light
    top_light = bpy.data.objects.new("Top_Light", bpy.data.lights.new("Top_Light", type="AREA"))
    bpy.context.collection.objects.link(top_light)
    top_light.data.energy = 10000
    top_light.location = (0, 0, 10)
    top_light.scale = (100, 100, 100)
    
    # create bottom light
    bottom_light = bpy.data.objects.new("Bottom_Light", bpy.data.lights.new("Bottom_Light", type="AREA"))
    bpy.context.collection.objects.link(bottom_light)
    bottom_light.data.energy = 1000
    bottom_light.location = (0, 0, -10)
    bottom_light.rotation_euler = (0, 0, 0)
    
    return {
        "default_light": default_light,
        "top_light": top_light,
        "bottom_light": bottom_light
    }


def load_object(object_path: str) -> None:
    """Loads a model with a supported file extension into the scene.

    Args:
        object_path (str): Path to the model file.

    Raises:
        ValueError: If the file extension is not supported.

    Returns:
        None
    """
    file_extension = object_path.split(".")[-1].lower()
    if file_extension is None:
        raise ValueError(f"Unsupported file type: {object_path}")

    if file_extension == "usdz":
        # install usdz io package
        dirname = os.path.dirname(os.path.realpath(__file__))
        usdz_package = os.path.join(dirname, "io_scene_usdz.zip")
        bpy.ops.preferences.addon_install(filepath=usdz_package)
        # enable it
        addon_name = "io_scene_usdz"
        bpy.ops.preferences.addon_enable(module=addon_name)
        # import the usdz
        from io_scene_usdz.import_usdz import import_usdz

        import_usdz(context, filepath=object_path, materials=True, animations=True)
        return None

    # load from existing import functions
    import_function = IMPORT_FUNCTIONS[file_extension]

    print(f"Loading object from {object_path}")
    if file_extension == "blend":
        import_function(directory=object_path, link=False)
    elif file_extension in {"glb", "gltf"}:
        import_function(filepath=object_path, merge_vertices=True, import_shading='NORMALS')
    else:
        import_function(filepath=object_path)
        
def delete_invisible_objects() -> None:
    """Deletes all invisible objects in the scene.

    Returns:
        None
    """
    # bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.object.select_all(action="DESELECT")
    for obj in bpy.context.scene.objects:
        if obj.hide_viewport or obj.hide_render:
            obj.hide_viewport = False
            obj.hide_render = False
            obj.hide_select = False
            obj.select_set(True)
    bpy.ops.object.delete()

    # Delete invisible collections
    invisible_collections = [col for col in bpy.data.collections if col.hide_viewport]
    for col in invisible_collections:
        bpy.data.collections.remove(col)
        
def split_mesh_normal():
    bpy.ops.object.select_all(action="DESELECT")
    objs = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    bpy.context.view_layer.objects.active = objs[0]
    for obj in objs:
        obj.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.split_normals()
    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.select_all(action="DESELECT")
            
def delete_custom_normals():
    for this_obj in bpy.data.objects:
        if this_obj.type == "MESH":
            bpy.context.view_layer.objects.active = this_obj
            bpy.ops.mesh.customdata_custom_splitnormals_clear()

def override_material():
    new_mat = bpy.data.materials.new(name="Override0123456789")
    new_mat.use_nodes = True
    new_mat.node_tree.nodes.clear()
    bsdf = new_mat.node_tree.nodes.new('ShaderNodeBsdfDiffuse')
    bsdf.inputs[0].default_value = (0.5, 0.5, 0.5, 1)
    bsdf.inputs[1].default_value = 1
    output = new_mat.node_tree.nodes.new('ShaderNodeOutputMaterial')
    new_mat.node_tree.links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])
    bpy.context.scene.view_layers['ViewLayer'].material_override = new_mat

def unhide_all_objects() -> None:
    """Unhides all objects in the scene.

    Returns:
        None
    """
    for obj in bpy.context.scene.objects:
        obj.hide_set(False)
        
def convert_to_meshes() -> None:
    """Converts all objects in the scene to meshes.

    Returns:
        None
    """
    bpy.ops.object.select_all(action="DESELECT")
    bpy.context.view_layer.objects.active = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"][0]
    for obj in bpy.context.scene.objects:
        obj.select_set(True)
    bpy.ops.object.convert(target="MESH")
        
def triangulate_meshes() -> None:
    """Triangulates all meshes in the scene.

    Returns:
        None
    """
    bpy.ops.object.select_all(action="DESELECT")
    objs = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    bpy.context.view_layer.objects.active = objs[0]
    for obj in objs:
        obj.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.reveal()
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.quads_convert_to_tris(quad_method="BEAUTY", ngon_method="BEAUTY")
    bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.object.select_all(action="DESELECT")

def scene_bbox() -> Tuple[Vector, Vector]:
    """Returns the bounding box of the scene.

    Taken from Shap-E rendering script
    (https://github.com/openai/shap-e/blob/main/shap_e/rendering/blender/blender_script.py#L68-L82)

    Returns:
        Tuple[Vector, Vector]: The minimum and maximum coordinates of the bounding box.
    """
    bbox_min = (math.inf,) * 3
    bbox_max = (-math.inf,) * 3
    found = False
    scene_meshes = [obj for obj in bpy.context.scene.objects.values() if isinstance(obj.data, bpy.types.Mesh)]
    for obj in scene_meshes:
        found = True
        for coord in obj.bound_box:
            coord = Vector(coord)
            coord = obj.matrix_world @ coord
            bbox_min = tuple(min(x, y) for x, y in zip(bbox_min, coord))
            bbox_max = tuple(max(x, y) for x, y in zip(bbox_max, coord))
    if not found:
        raise RuntimeError("no objects in scene to compute bounding box for")
    return Vector(bbox_min), Vector(bbox_max)

def normalize_scene() -> Tuple[float, Vector]:
    """Normalizes the scene by scaling and translating it to fit in a unit cube centered
    at the origin.

    Mostly taken from the Point-E / Shap-E rendering script
    (https://github.com/openai/point-e/blob/main/point_e/evals/scripts/blender_script.py#L97-L112),
    but fix for multiple root objects: (see bug report here:
    https://github.com/openai/shap-e/pull/60).

    Returns:
        Tuple[float, Vector]: The scale factor and the offset applied to the scene.
    """
    scene_root_objects = [obj for obj in bpy.context.scene.objects.values() if not obj.parent]
    if len(scene_root_objects) > 1:
        # create an empty object to be used as a parent for all root objects
        scene = bpy.data.objects.new("ParentEmpty", None)
        bpy.context.scene.collection.objects.link(scene)

        # parent all root objects to the empty object
        for obj in scene_root_objects:
            obj.parent = scene
    else:
        scene = scene_root_objects[0]

    bbox_min, bbox_max = scene_bbox()
    scale = 1 / max(bbox_max - bbox_min)
    scene.scale = scene.scale * scale

    # Apply scale to matrix_world.
    bpy.context.view_layer.update()
    bbox_min, bbox_max = scene_bbox()
    offset = -(bbox_min + bbox_max) / 2
    scene.matrix_world.translation += offset
    bpy.ops.object.select_all(action="DESELECT")
    
    return scale, offset


def get_transform_matrix(obj: bpy.types.Object) -> list:
    pos, rt, _ = obj.matrix_world.decompose()
    rt = rt.to_matrix()
    matrix = []
    for ii in range(3):
        a = []
        for jj in range(3):
            a.append(rt[ii][jj])
        a.append(pos[ii])
        matrix.append(a)
    matrix.append([0, 0, 0, 1])
    return matrix

    
def main(arg):
    os.makedirs(arg.output_folder, exist_ok=True)
    
    camera_offset = - np.pi / 2 
    
    if arg.geo_mode:
        if arg.views!=4:
            views = trellis_cond_camera_sequence(arg.views, camera_offset=camera_offset)
        else:
            views = four_view_camera_sequence(camera_offset=camera_offset)
        arg.save_mesh = True
        arg.save_depth = True
        arg.save_normal = True
    else:
        views = orthogonal_camera_sequence()
        arg.save_albedo = True
        arg.save_mr = True
        arg.save_normal = True
        arg.save_depth = True
        arg.save_mesh = False
    
    # Initialize context
    init_render(engine=arg.engine, resolution=arg.resolution, geo_mode=arg.geo_mode)
    outputs, spec_nodes, composite_nodes = init_nodes(
        save_depth=arg.save_depth,
        save_normal=arg.save_normal,
        save_albedo=arg.save_albedo,
        save_mist=arg.save_mist
    )
    if arg.object.endswith(".blend"):
        delete_invisible_objects()
    else:
        init_scene()
        load_object(arg.object)
        if arg.split_normal:
            split_mesh_normal()
        # delete_custom_normals()
    print('[INFO] Scene initialized.')
    
    # normalize scene
    scale, offset = normalize_scene()
    print('[INFO] Scene normalized.')
    
    
    # Initialize camera and lighting
    cam = init_camera()
    init_lighting()
    print('[INFO] Camera and lighting initialized.')

    # Override material
    #if arg.geo_mode:
    #    override_material()
    
    # Create a list of views
    to_export = {
        "aabb": [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        "scale": scale,
        "offset": [offset.x, offset.y, offset.z],
        "frames": []
    }

    for i, view in enumerate(views):
        cam.location = (
            view['cam_dis'] * np.cos(view['hangle']) * np.cos(view['vangle']),
            view['cam_dis'] * np.sin(view['hangle']) * np.cos(view['vangle']),
            view['cam_dis'] * np.sin(view['vangle'])
        )
        cam.data.lens = 16 / np.tan(view['fov'] / 2)
        
        if view['proj_type'] == 1:
            cam.data.type = "ORTHO"
            cam.data.ortho_scale = 1.2

        bpy.context.scene.render.filepath = os.path.join(arg.output_folder, f'{i:03d}.png')
        for name, output in outputs.items():
            output.file_slots[0].path = os.path.join(arg.output_folder, f'{i:03d}_{name}')
            
        # Render the scene
        if not arg.geo_mode:
            switch_to_mr_render(False, composite_nodes)
            bpy.ops.render.render(write_still=True)
            shutil.copyfile(bpy.context.scene.render.filepath, 
                            bpy.context.scene.render.filepath.replace('.png', '_mr.png'))
            switch_to_color_render(composite_nodes)

        bpy.ops.render.render(write_still=True)
        bpy.context.view_layer.update()
        for name, output in outputs.items():
            ext = EXT[output.format.file_format]
            path = glob.glob(f'{output.file_slots[0].path}*.{ext}')[0]
            os.rename(path, f'{output.file_slots[0].path}.{ext}')
        
        # Convert normal map to PNG format in all modes
        if arg.save_normal:
            # 使用深度图作为遮罩
            depth_exr_path = os.path.join(arg.output_folder, f'{i:03d}_depth.exr')
            ConvertNormalMap(os.path.join(arg.output_folder, f'{i:03d}_normal.exr'), 
                             os.path.join(arg.output_folder, f'{i:03d}_normal.png'),
                             depth_exr=depth_exr_path, view_index=i)
            os.remove(os.path.join(arg.output_folder, f'{i:03d}_normal.exr'))
            
            # 保存标准法线图（_N版本）：其他视图时，通过旋转mesh到正视图位置来重新渲染
            print(f"🔄 为视图{i}重新渲染_N法线图（旋转mesh到正视图位置）...")
            
            # 计算需要旋转的角度（从当前角度到正视图的角度）
            target_angle = 0  # 正视图角度
            current_angle = view['hangle']
            rotation_angle = current_angle  # 需要旋转的角度
            
            # 旋转mesh回到正视图位置
            rotation_matrix = Matrix.Rotation(-rotation_angle, 4, 'Z')
            mesh_objects = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
            for obj in mesh_objects:
                current_matrix = obj.matrix_world.copy()
                new_matrix = rotation_matrix @ current_matrix
                obj.matrix_world = new_matrix
            
            # 更新场景
            bpy.context.view_layer.update()
            
            # 重新设置相机到正视图位置
            cam.location = (
                view['cam_dis'] * np.cos(target_angle) * np.cos(0),
                view['cam_dis'] * np.sin(target_angle) * np.cos(0),
                view['cam_dis'] * np.sin(0)
            )
            
            # 重新渲染法线图
            normal_file_output = outputs['normal']
            normal_file_output.file_slots[0].path = os.path.join(arg.output_folder, f'{i:03d}_normal_N')
            bpy.ops.render.render(write_still=True)
            bpy.context.view_layer.update()
            
            # 获取渲染的法线EXR文件
            path = glob.glob(f'{normal_file_output.file_slots[0].path}*.exr')[0]
            os.rename(path, f'{normal_file_output.file_slots[0].path}.exr')
            
            # 转换并保存PNG
            ConvertNormalMap(os.path.join(arg.output_folder, f'{i:03d}_normal_N.exr'), 
                                os.path.join(arg.output_folder, f'{i:03d}_normal_N.png'),
                                depth_exr=depth_exr_path, view_index=i)
            os.remove(os.path.join(arg.output_folder, f'{i:03d}_normal_N.exr'))
            # 恢复mesh和相机到原始位置
            rotation_matrix = Matrix.Rotation(rotation_angle, 4, 'Z')
            for obj in mesh_objects:
                current_matrix = obj.matrix_world.copy()
                new_matrix = rotation_matrix @ current_matrix
                obj.matrix_world = new_matrix
            
            cam.location = (
                view['cam_dis'] * np.cos(view['hangle']) * np.cos(view['vangle']),
                view['cam_dis'] * np.sin(view['hangle']) * np.cos(view['vangle']),
                view['cam_dis'] * np.sin(view['vangle'])
            )
            bpy.context.view_layer.update()
            
            print(f"✅ 视图{i}的_N法线图渲染完成")
            
        
        # Convert depth map to PNG format in all modes  
        if arg.save_depth:
            success = ConvertDepthMap(os.path.join(arg.output_folder, f'{i:03d}_depth.exr'), 
                                     os.path.join(arg.output_folder, f'{i:03d}_depth.png'),
                                     max_valid_depth=100.0, debug=False)
            if success:
                os.remove(os.path.join(arg.output_folder, f'{i:03d}_depth.exr'))
            else:
                print(f"⚠️  深度图转换失败，保留EXR文件: {i:03d}_depth.exr")

        # Save camera parameters
        metadata = {
            "file_path": f'{i:03d}.png',
            "camera_angle_x": view['fov'],
            'proj_type': view['proj_type'],
            'azimuth': view['hangle'],
            'elevation': view['vangle'],
            'cam_dis': view['cam_dis'],
            "transform_matrix": get_transform_matrix(cam)
        }
        to_export["frames"].append(metadata)

    # Save the camera parameters
    transform_path = os.path.join(arg.output_folder, 'transforms.json')
    with open(transform_path, 'w') as f:
        json.dump(to_export, f, indent=4)
        
    if arg.save_mesh:
        # triangulate meshes
        unhide_all_objects()
        convert_to_meshes()
        triangulate_meshes()
        print('[INFO] Meshes triangulated.')
        
        # export ply mesh
        bpy.ops.wm.ply_export(filepath=os.path.join(arg.output_folder, 'mesh.ply'), 
                              export_triangulated_mesh=True, up_axis='Y', 
                              forward_axis='NEGATIVE_Z')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Renders given obj file by rotation a camera around it.')
    parser.add_argument('--views', type=int, default=24, 
        help='JSON string of views. Contains a list of {yaw, pitch, radius, fov} object.')
    parser.add_argument('--object', type=str, 
        help='Path to the 3D model file to be rendered.')
    parser.add_argument('--output_folder', type=str, default='/tmp', 
        help='The path the output will be dumped to.')
    parser.add_argument('--resolution', type=int, default=512, 
        help='Resolution of the images.')
    parser.add_argument('--engine', type=str, default='CYCLES', 
        help='Blender internal engine for rendering. E.g. CYCLES, BLENDER_EEVEE, ...')
    parser.add_argument('--geo_mode', action='store_true', 
        help='Geometry mode for rendering.')
    parser.add_argument('--save_depth', action='store_true', 
        help='Save the depth maps.')
    parser.add_argument('--save_normal', action='store_true', 
        help='Save the normal maps.')
    parser.add_argument('--save_albedo', action='store_true', 
        help='Save the albedo maps.')
    parser.add_argument('--save_mr', action='store_true', 
        help='Save the MR maps.')
    parser.add_argument('--save_mist', action='store_true', 
        help='Save the mist distance maps.')
    parser.add_argument('--split_normal', action='store_true', 
        help='Split the normals of the mesh.')
    parser.add_argument('--save_mesh', action='store_true', 
        help='Save the mesh as a .ply file.')
    argv = sys.argv[sys.argv.index("--") + 1:]
    args = parser.parse_args(argv)

    main(args)
