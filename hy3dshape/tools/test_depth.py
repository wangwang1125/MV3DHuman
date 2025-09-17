import OpenEXR
import Imath
import numpy as np
import cv2
# 读取EXR深度图
file = OpenEXR.InputFile('mini_depth_trainset/preprocessed/GTP_CTeenW_Ema_09_Stg_Hat_Lsn_Ccs_Mgr/render_cond/000_depth.exr')
dw = file.header()['dataWindow']
size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)

# 读取深度通道
depth_str = file.channel('R', Imath.PixelType(Imath.PixelType.FLOAT))
depth = np.frombuffer(depth_str, dtype=np.float32)
print(depth)
depth = depth.reshape(size[1], size[0])

# 过滤掉超过100的深度值
valid_mask = depth <= 100
filtered_depth = depth.copy()
filtered_depth[~valid_mask] = 0  # 将无效值设为0

print(f"\n过滤后分析:")
print(f"有效深度值范围: {filtered_depth[valid_mask].min()} - {filtered_depth[valid_mask].max()}")
print(f"有效深度值均值: {filtered_depth[valid_mask].mean()}")

# Display images without normalization (original depth values)
cv2.imshow('Filtered Depth (<=100)', filtered_depth.astype(np.uint16)*1000)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 生成点云
def depth_to_pointcloud(depth_map, fx=525.0, fy=525.0, cx=320.0, cy=240.0):
    """
    将深度图转换为点云
    depth_map: 深度图 (H, W)
    fx, fy: 相机内参焦距
    cx, cy: 相机内参主点
    """
    h, w = depth_map.shape
    points = []
    colors = []
    
    for v in range(h):
        for u in range(w):
            z = depth_map[v, u]
            if z > 0 and z <= 100:  # 只处理有效深度值
                # 像素坐标转换为3D坐标
                x = (u - cx) * z / fx
                y = (v - cy) * z / fy
                points.append([x, y, z])
                # 根据深度值设置颜色 (近处红色，远处蓝色)
                color_intensity = min(z / 10.0, 1.0)  # 归一化到0-1
                colors.append([1.0 - color_intensity, 0.0, color_intensity])  # RGB
    
    return np.array(points), np.array(colors)

# 生成点云数据
print("\n生成点云...")
points, colors = depth_to_pointcloud(filtered_depth)
print(f"生成了 {len(points)} 个点")

# 保存点云为PLY格式
def save_ply(filename, points, colors):
    """
    保存点云为PLY格式文件
    """
    with open(filename, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        
        for i in range(len(points)):
            x, y, z = points[i]
            r, g, b = (colors[i] * 255).astype(int)
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {r} {g} {b}\n")

# 保存点云文件
output_file = "depth_pointcloud.ply"
save_ply(output_file, points, colors)
print(f"点云已保存到: {output_file}")
print("可以使用MeshLab、CloudCompare或Blender等软件打开查看点云")

# 显示点云统计信息
if len(points) > 0:
    print(f"\n点云统计:")
    print(f"X范围: {points[:, 0].min():.3f} ~ {points[:, 0].max():.3f}")
    print(f"Y范围: {points[:, 1].min():.3f} ~ {points[:, 1].max():.3f}")
    print(f"Z范围: {points[:, 2].min():.3f} ~ {points[:, 2].max():.3f}")
else:
    print("警告: 没有生成有效的点云数据")