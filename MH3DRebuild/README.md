# MH3DRebuild — 多视图 RGB 三维重建 (MV LoRA)

基于 Hunyuan3D-2mv 与 MV LoRA 微调模型，从四视图 RGB 图像重建三维 mesh（GLB）。

**项目根目录即 MH3DRebuild**：所需代码（`hy3dshape`、`torchvision_fix.py`）已拷贝进本目录，无需依赖上级项目。

## 依赖与环境

- 系统：Linux
- Conda 环境：`mh3dr`（需自行创建）
- 在 **MH3DRebuild 根目录** 下安装依赖即可。

### 环境配置

```bash
conda create -n mh3dr python=3.10
conda activate mh3dr
cd /path/to/MH3DRebuild
# 按 CUDA 版本安装 PyTorch，例如：
# pip install torch torchvision
pip install -r requirements.txt
```

## 目录约定

- **confdata/input_data**：四视图图像目录，须包含 `front`、`left`、`back`、`right` 对应图像（如 `front.jpg`、`left.png` 等，支持 `.jpg` / `.png`）。
- **confdata/models**：模型根目录（`HY3DGEN_MODELS` 指向此处）。**基础模型**须按下列结构放置，否则会报 `Model file ... not found`：
  ```
  confdata/models/
  └── tencent/
      └── Hunyuan3D-2mv/
          └── hunyuan3d-dit-v2-mv/
              ├── config.yaml
              └── model.fp16.ckpt
  ```
  可将 HuggingFace 下载的 `tencent/Hunyuan3D-2mv` 整包放到 `confdata/models/` 下，或从已有缓存拷贝对应子目录。  
  **LoRA**：在 `confdata/models/` 下放置 `.ckpt` 或 `step_*` 目录，会优先加载；否则在 `hy3dshape/output_folder` 等路径查找。

## 编译、测试与打包

```bash
./make.sh clear   # 清空上次编译和打包
./make.sh build   # 将 python 脚本合并并编译为可执行程序
./make.sh test    # 测试脚本是否正常（使用 confdata 数据）
./make.sh pack    # 打包节点用于上传部署
```

## 启动方式

```bash
./script/runner.sh main.py <input_data> <models> <output_mesh>
```

例如：

```bash
./script/runner.sh main.py confdata/input_data confdata/models ./out.glb
```

由上层框架调用时，通常通过 `./runner.sh main.py` 启动，由框架追加 `input_data`、`models`、`output_mesh` 参数。

## 目录结构

```
MH3DRebuild/
├── make.sh
├── README.md
├── requirements.txt
├── torchvision_fix.py
├── hy3dshape/          # 已拷贝，推理所需
├── confdata/
│   ├── input_data/     # front, left, back, right 图像
│   └── models/         # 基础模型 tencent/Hunyuan3D-2mv/... + 可选 LoRA
└── script/
    ├── info.py
    ├── main.py
    └── runner.sh
```
