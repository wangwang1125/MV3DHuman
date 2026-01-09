# Hunyuan3D 多视图RGB重建 API 服务器

本文档介绍如何使用 Hunyuan3D 多视图RGB重建模式的 API 服务器。

## 概述

多视图RGB重建 API 服务器支持从4个视角的RGB图像生成高质量3D模型，不需要深度图或法线图。

### 特性

- ✅ **4视图RGB重建**: 从前、右、后、左四个视角的RGB图像生成3D模型
- ✅ **LoRA微调支持**: 自动加载或手动指定RGB LoRA权重
- ✅ **并发处理**: 支持多个请求同时处理（可配置并发数）
- ✅ **异步API**: 提供同步和异步两种API接口
- ✅ **自动背景移除**: 可选的自动背景移除功能
- ✅ **纹理生成**: 可选的PBR纹理生成

## 快速开始

### 1. 启动服务器

使用提供的启动脚本：

```bash
chmod +x start_multiview_rgb_api.sh
./start_multiview_rgb_api.sh
```

或者手动启动：

```bash
python api_server.py \
    --model_path tencent/Hunyuan3D-2.1 \
    --subfolder hunyuan3d-dit-v2-1 \
    --enable_multiview_rgb \
    --rgb_lora_path ./path/to/lora/checkpoint.ckpt \
    --num_views 4 \
    --port 8081 \
    --device cuda \
    --limit-model-concurrency 2
```

### 2. 查看API文档

启动服务器后，访问：
- Swagger UI: http://localhost:8081/docs
- ReDoc: http://localhost:8081/redoc

### 3. 测试API

运行测试脚本：

```bash
python test_multiview_api.py
```

## API 端点

### 1. POST /generate - 同步生成

同步方式生成3D模型，直接返回生成的模型文件。

**请求示例:**

```python
import requests
import base64

# 读取4个视角的图像
with open("front.png", "rb") as f:
    front_b64 = base64.b64encode(f.read()).decode()
with open("right.png", "rb") as f:
    right_b64 = base64.b64encode(f.read()).decode()
with open("back.png", "rb") as f:
    back_b64 = base64.b64encode(f.read()).decode()
with open("left.png", "rb") as f:
    left_b64 = base64.b64encode(f.read()).decode()

response = requests.post("http://localhost:8081/generate", json={
    "image_front": front_b64,
    "image_right": right_b64,
    "image_back": back_b64,
    "image_left": left_b64,
    "remove_background": True,
    "texture": False,
    "seed": 1234,
    "num_inference_steps": 5,
    "guidance_scale": 5.0,
    "octree_resolution": 256,
    "num_chunks": 8000
})

# 保存返回的模型文件
with open("output.glb", "wb") as f:
    f.write(response.content)
```

### 2. POST /send - 异步生成

异步方式提交任务，返回任务ID用于后续查询。

**请求示例:**

```python
# 提交任务
response = requests.post("http://localhost:8081/send", json={
    "image_front": front_b64,
    "image_right": right_b64,
    "image_back": back_b64,
    "image_left": left_b64,
    "remove_background": True,
    "texture": False,
    "seed": 1234
})

uid = response.json()["uid"]
print(f"Task UID: {uid}")
```

### 3. GET /status/{uid} - 查询状态

查询异步任务的处理状态。

**请求示例:**

```python
import time

while True:
    response = requests.get(f"http://localhost:8081/status/{uid}")
    data = response.json()
    
    status = data["status"]
    print(f"Status: {status}")
    
    if status == "completed":
        # 解码base64模型数据
        model_b64 = data["model_base64"]
        model_bytes = base64.b64decode(model_b64)
        
        with open("output.glb", "wb") as f:
            f.write(model_bytes)
        print("Model saved!")
        break
    elif status == "error":
        print(f"Error: {data.get('message')}")
        break
    
    time.sleep(2)  # 等待2秒后再次查询
```

### 4. GET /health - 健康检查

检查服务器运行状态。

```python
response = requests.get("http://localhost:8081/health")
print(response.json())
# {"status": "healthy", "worker_id": "abc123"}
```

## 请求参数说明

### 必需参数

| 参数 | 类型 | 说明 |
|------|------|------|
| `image_front` | string | Base64编码的前视图图像 (0°) |
| `image_right` | string | Base64编码的右视图图像 (90°) |
| `image_back` | string | Base64编码的后视图图像 (180°) |
| `image_left` | string | Base64编码的左视图图像 (270°) |

### 可选参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `remove_background` | bool | true | 是否自动移除背景 |
| `texture` | bool | false | 是否生成纹理 |
| `seed` | int | 1234 | 随机种子 |
| `num_inference_steps` | int | 5 | 推理步数 (1-20) |
| `guidance_scale` | float | 5.0 | 引导比例 (0.1-20.0) |
| `octree_resolution` | int | 256 | 八叉树分辨率 (64-512) |
| `num_chunks` | int | 8000 | 处理块数 (1000-20000) |
| `face_count` | int | 40000 | 纹理生成的最大面数 |

## 图像要求

### 视角顺序

四个视图必须按以下顺序排列：

```
    Front (0°)
       ↑
       |
Left ← ● → Right
(270°) |   (90°)
       ↓
    Back (180°)
```

### 图像质量建议

- **格式**: PNG, JPG (推荐PNG)
- **尺寸**: 建议方形比例，如 512x512, 1024x1024
- **内容**: 
  - 物体应居中
  - 所有视图使用一致的光照
  - 所有视图使用一致的缩放比例
  - 背景尽量简单或使用纯色

## 并发控制

服务器使用信号量(semaphore)控制并发：

- 默认并发数: 2 (多视图模式建议值)
- 可通过 `--limit-model-concurrency` 参数调整
- 超过并发限制的请求会排队等待

**调整并发数示例:**

```bash
python api_server.py \
    --enable_multiview_rgb \
    --limit-model-concurrency 3  # 允许3个并发请求
```

⚠️ **注意**: 多视图模式占用更多显存，建议根据GPU显存大小调整并发数：
- 24GB VRAM: 2-3个并发
- 48GB VRAM: 4-5个并发

## LoRA 权重管理

### 自动发现

如果未指定 `--rgb_lora_path`，服务器会自动搜索以下位置：

1. `./hy3dshape/output_folder/dit/multiview_rgb_lora_finetuning/ckpt/`
2. `./output_folder/dit/multiview_rgb_lora_finetuning/ckpt/`
3. `./hy3dshape/output_folder/dit/multiview_rgb_lora_checkpoints/`
4. `./output_folder/dit/multiview_rgb_lora_checkpoints/`

并自动选择最新的checkpoint。

### 手动指定

```bash
python api_server.py \
    --enable_multiview_rgb \
    --rgb_lora_path /path/to/checkpoint.ckpt  # Lightning格式
```

或

```bash
python api_server.py \
    --enable_multiview_rgb \
    --rgb_lora_path /path/to/peft_dir/  # PEFT格式
```

### 支持的格式

- **Lightning Checkpoint** (`.ckpt`): 训练时保存的checkpoint文件
- **PEFT Directory**: Hugging Face PEFT格式的目录

## 性能优化

### 低显存模式

如果遇到显存不足，可以启用低显存模式：

```bash
python api_server.py \
    --enable_multiview_rgb \
    --low_vram_mode  # 启用CPU卸载
```

### 调整生成参数

快速测试时可以降低质量参数：

```json
{
  "num_inference_steps": 5,      // 降低推理步数
  "octree_resolution": 128,      // 降低分辨率
  "num_chunks": 8000             // 减少处理块数
}
```

高质量生成时增加参数：

```json
{
  "num_inference_steps": 10,     // 增加推理步数
  "octree_resolution": 384,      // 提高分辨率
  "num_chunks": 20000            // 增加处理块数
}
```

## 故障排除

### 问题 1: 服务器启动失败

**可能原因**: 缺少依赖包

**解决方法**:
```bash
pip install -r requirements.txt
```

### 问题 2: CUDA Out of Memory

**可能原因**: 显存不足

**解决方法**:
1. 降低并发数: `--limit-model-concurrency 1`
2. 启用低显存模式: `--low_vram_mode`
3. 降低分辨率: `octree_resolution=128`

### 问题 3: LoRA权重加载失败

**可能原因**: 权重路径不正确或格式不兼容

**解决方法**:
1. 检查路径是否存在
2. 确认是Lightning (.ckpt) 或 PEFT格式
3. 查看服务器日志了解详细错误

### 问题 4: 生成结果质量差

**可能原因**: 输入图像质量或视角不合适

**解决方法**:
1. 确保4个视角正确对应 (front/right/back/left)
2. 确保所有视图光照一致
3. 确保物体在各视图中居中且大小一致
4. 尝试启用背景移除: `"remove_background": true`

## 示例代码

完整的客户端示例代码见 `test_multiview_api.py`。

## 许可证

本项目基于 TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT 协议。

详见: https://github.com/Tencent/Hunyuan3D/blob/main/LICENSE

## 技术支持

如有问题，请访问: https://github.com/Tencent/Hunyuan3D/issues
