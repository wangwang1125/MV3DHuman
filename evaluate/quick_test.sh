#!/bin/bash
# 快速测试评估工具

echo "======================================"
echo "3D Mesh评估工具 - 快速测试"
echo "======================================"
echo ""

# 检查数据集目录
EVAL_BASE_DIR="/mnt/g/evaluate"

if [ ! -d "$EVAL_BASE_DIR" ]; then
    echo "❌ 错误: 评估根目录不存在: $EVAL_BASE_DIR"
    echo "请修改脚本中的 EVAL_BASE_DIR 变量"
    exit 1
fi

echo "✅ 找到评估根目录: $EVAL_BASE_DIR"
echo ""

# 统计方法数和样本数
METHOD_COUNT=$(find "$EVAL_BASE_DIR" -mindepth 1 -maxdepth 1 -type d | wc -l)
echo "📊 发现 $METHOD_COUNT 个方法目录"

# 查找第一个有效样本进行测试
echo "🔍 寻找第一个有效样本进行测试..."
echo ""

FOUND=0
for method_dir in "$EVAL_BASE_DIR"/*; do
    if [ -d "$method_dir" ]; then
        method_name=$(basename "$method_dir")
        
        for obj_dir in "$method_dir"/*; do
            if [ -d "$obj_dir" ]; then
                obj_name=$(basename "$obj_dir")
                gt_mesh="$obj_dir/mesh.ply"
                gen_mesh="$obj_dir/gen_mesh.glb"
                
                if [ -f "$gen_mesh" ] && [ -f "$gt_mesh" ]; then
                    echo "✅ 找到有效样本"
                    echo "   方法名: $method_name"
                    echo "   对象名: $obj_name"
                    echo "   真实mesh: $gt_mesh"
                    echo "   生成mesh: $gen_mesh"
                    echo ""
                    
                    echo "======================================"
                    echo "运行单样本评估测试..."
                    echo "======================================"
                    echo ""
                    
                    python mesh_metrics.py "$gen_mesh" "$gt_mesh" 10000 1
                    
                    FOUND+=1
                fi
            fi
        done
    fi
done

if [ $FOUND -eq 0 ]; then
    echo "❌ 未找到有效样本（需要同时有 mesh.ply 和 gen_mesh.ply）"
    exit 1
fi

echo ""
echo "======================================"
echo "测试完成！"
echo "======================================"
echo ""
echo "接下来可以运行完整评估："
echo ""
echo "  # 评估某个方法的所有样本（快速模式，5000点）"
echo "  python batch_evaluate.py -g $EVAL_BASE_DIR/方法名 -t $EVAL_BASE_DIR/方法名 -n 5000 -o results_quick.json --gen_suffix .ply --gt_suffix .ply"
echo ""
echo "  # 评估某个方法的所有样本（标准模式，10000点）"
echo "  python batch_evaluate.py -g $EVAL_BASE_DIR/方法名 -t $EVAL_BASE_DIR/方法名 -n 10000 -o results_standard.json --gen_suffix .ply --gt_suffix .ply"
echo ""
echo "  # 评估某个方法的所有样本（高精度模式，20000点）"
echo "  python batch_evaluate.py -g $EVAL_BASE_DIR/方法名 -t $EVAL_BASE_DIR/方法名 -n 20000 -o results_high.json --gen_suffix .ply --gt_suffix .ply"
echo ""

