#!/bin/bash

# 批量处理obj文件的脚本
# 基于原始pipeline.sh修改

export OPENCV_IO_ENABLE_OPENEXR=1
export OUTPUT_FOLDER=more_depth_trainset/preprocessed
export BLENDER_PATH=/mnt/d/workapp/wsl/blender/4.5/python/bin/python3.11

# 自定义输入文件夹路径 - 可以修改为任意目录
# 默认为当前目录 (.) - 递归搜索所有子目录
# 示例: export INPUT_FOLDER="../test_models" 或 export INPUT_FOLDER="/path/to/your/obj/files"
export INPUT_FOLDER="/mnt/d/down/obj/02Dressed/TeenF"

# 检查Blender路径是否存在
if [ ! -f "$BLENDER_PATH" ]; then
    echo "错误: Blender路径不存在: $BLENDER_PATH"
    echo "请修改BLENDER_PATH变量为正确的Blender可执行文件路径"
    exit 1
fi

# 检查输入目录是否存在
if [ ! -d "$INPUT_FOLDER" ]; then
    echo "错误: 输入目录不存在: $INPUT_FOLDER"
    echo "请修改INPUT_FOLDER变量为正确的目录路径"
    exit 1
fi

# 创建输出目录
mkdir -p "$OUTPUT_FOLDER"

# 函数：处理单个obj文件
process_obj_file() {
    local input_file="$1"
    local name="$2"
    
    echo "正在处理: $input_file"
    echo "输出名称: $name"
    
    # 创建输出目录
    mkdir -p "$OUTPUT_FOLDER/$name/render_cond"
    mkdir -p "$OUTPUT_FOLDER/$name/geo_data"
    
    # 运行Blender渲染
    echo "运行Blender渲染..."
    $BLENDER_PATH -b -P render/render.py -- --object "${input_file}" --output_folder "$OUTPUT_FOLDER/$name/render_cond" --geo_mode --resolution 512 --views 4
    
    if [ $? -ne 0 ]; then
        echo "错误: Blender渲染失败 - $input_file"
        return 1
    fi
    
    # 检查mesh.ply是否生成
    if [ ! -f "$OUTPUT_FOLDER/$name/render_cond/mesh.ply" ]; then
        echo "错误: mesh.ply文件未生成 - $input_file"
        return 1
    fi
    
    # 运行watertight处理
    echo "运行watertight处理..."
    python watertight/watertight_and_sample.py --input_obj "$OUTPUT_FOLDER/$name/render_cond/mesh.ply" --output_prefix "$OUTPUT_FOLDER/$name/geo_data/$name"
    
    if [ $? -ne 0 ]; then
        echo "错误: watertight处理失败 - $input_file"
        return 1
    fi
    
    echo "完成处理: $input_file"
    return 0
}

# 计数器
processed_count=0
success_count=0
failed_count=0

echo "开始批量处理obj文件..."
echo "输出目录: $OUTPUT_FOLDER"
echo "="*50

# 查找并处理所有obj文件
echo "搜索目录: $INPUT_FOLDER"
find "$INPUT_FOLDER" -name "*.obj" -type f | while read -r obj_file; do
    # 跳过输出目录中的文件
    if [[ "$obj_file" == *"$OUTPUT_FOLDER"* ]]; then
        continue
    fi
    
    # 获取文件名（不含扩展名）作为处理名称
    filename=$(basename "$obj_file")
    name="${filename%.obj}"
    
    # 如果文件名是mesh.obj，使用父目录名
    if [ "$name" == "mesh" ]; then
        parent_dir=$(basename "$(dirname "$obj_file")")
        name="${parent_dir}_mesh"
    fi
    
    # 处理文件
    ((processed_count++))
    echo "[$processed_count] 处理文件: $obj_file -> $name"
    
    if process_obj_file "$obj_file" "$name"; then
        ((success_count++))
        echo "✓ 成功处理: $obj_file"
    else
        ((failed_count++))
        echo "✗ 处理失败: $obj_file"
    fi
    
    echo "-"*30
done

echo "="*50
echo "批量处理完成!"
echo "总计处理: $processed_count 个文件"
echo "成功: $success_count 个"
echo "失败: $failed_count 个"
echo "输出目录: $OUTPUT_FOLDER"

if [ $failed_count -gt 0 ]; then
    echo "警告: 有 $failed_count 个文件处理失败，请检查错误信息"
    exit 1
else
    echo "所有文件处理成功!"
    exit 0
fi