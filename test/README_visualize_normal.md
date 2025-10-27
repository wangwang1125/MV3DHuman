# EXR法线图可视化工具

这个脚本用于可视化EXR格式的法线图，支持多种可视化模式和批量处理。

## 功能特性

- 读取EXR格式的法线图文件（支持X、Y、Z通道）
- 三种可视化模式：RGB、分量显示、长度显示
- 详细的统计信息分析
- 支持单文件和批量处理
- 多种颜色映射选项
- 自动检测和转换数据范围
- 兼容多种EXR读取方式（imageio、OpenEXR、OpenCV）

## 安装依赖

```bash
pip install numpy opencv-python matplotlib imageio OpenEXR
```

注意：如果OpenCV没有启用OpenEXR支持，脚本会自动使用OpenEXR库作为备用方案。

## 使用方法

### 1. 单文件可视化

```bash
# 基本用法 - RGB模式
python visualize_normal.py input_normal.exr -o output.png

# 显示交互式窗口
python visualize_normal.py input_normal.exr --show

# 分量模式 - 分别显示XYZ分量
python visualize_normal.py input_normal.exr -o output.png -m components

# 长度模式 - 显示法线长度
python visualize_normal.py input_normal.exr -o output.png -m length

# 使用不同颜色映射（length模式）
python visualize_normal.py input_normal.exr -o output.png -m length -c plasma

# 保存简单版本（无统计信息）
python visualize_normal.py input_normal.exr -o output.png --simple
```

### 2. 批量处理

```bash
# 批量处理整个目录
python visualize_normal.py --batch /path/to/normal/directory/ -o ./visualizations/

# 自定义文件匹配模式
python visualize_normal.py --batch /path/to/directory/ -o ./output/ -p "**/*_normal_*.exr"

# 批量处理并保存简单版本
python visualize_normal.py --batch /path/to/directory/ -o ./output/ --simple
```

## 可视化模式

### 1. RGB模式 (默认)
- 直接将XYZ分量映射到RGB通道
- X → Red, Y → Green, Z → Blue
- 范围 [-1,1] 映射到 [0,255]

### 2. 分量模式 (components)
- 分别显示XYZ三个分量
- 每个分量使用不同的颜色映射
- 便于分析各个分量的分布

### 3. 长度模式 (length)
- 显示每个像素法线的长度
- 正常法线长度应该接近1.0
- 可用于检测法线数据的质量

## 统计信息

脚本会显示以下统计信息：
- XYZ各分量的最小值、最大值、平均值、标准差
- 法线长度的统计信息
- 有效像素数量和比例
- 图像尺寸信息

## 参数说明

- `input`: 输入EXR文件路径（单文件模式）
- `--batch, -b`: 批量处理模式，指定输入目录
- `--output, -o`: 输出图像路径或目录
- `--mode, -m`: 可视化模式 (rgb/components/length)
- `--colormap, -c`: 颜色映射名称
- `--show, -s`: 显示可视化结果
- `--simple`: 保存简单版本（无统计信息）
- `--pattern, -p`: 批量模式下的文件匹配模式

## 示例输出

脚本会生成包含以下内容的可视化图像：
1. 主法线图（根据选择的模式）
2. XYZ分量分别显示
3. 详细的统计信息文本

## 故障排除

### 常见问题

1. **有效像素为0%**
   - 检查EXR文件是否包含X、Y、Z通道（而不是R、G、B）
   - 使用`--debug`参数查看详细读取信息

2. **OpenCV读取失败**
   - 脚本会自动尝试OpenEXR库作为备用方案
   - 确保安装了OpenEXR库：`pip install OpenEXR`

3. **Qt平台插件错误**
   - 脚本使用非交互式后端，避免Qt问题
   - 如果仍有问题，设置环境变量：`export QT_QPA_PLATFORM=offscreen`

4. **法线数据异常**
   - 检查数据范围是否在[-1,1]内
   - 使用`inspect_exr.py`脚本检查EXR文件通道信息

## 注意事项

- 确保EXR文件包含X、Y、Z通道的法线数据
- 法线数据应该在[-1,1]范围内
- 支持多种EXR读取方式（imageio、OpenEXR、OpenCV）
- 自动处理无效值和异常数据
- 使用`--debug`参数获取详细的处理信息
