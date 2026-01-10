FROM hunyuan3d21-python-base:cu124
LABEL name="hunyuan3d21"

WORKDIR /workspace

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYOPENGL_PLATFORM=egl
ENV CUDA_HOME=/usr/local/cuda
ENV TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5;8.0;8.6;8.9;9.0"
ENV LD_LIBRARY_PATH=/workspace/miniconda3/envs/hunyuan3d21/lib:/usr/local/cuda/lib64:/usr/lib64

COPY . /workspace/Hunyuan3D-2.1

RUN pip install --no-cache-dir \
    --extra-index-url https://download.blender.org/pypi/ \
    bpy==3.6.0
# RUN cd /workspace/Hunyuan3D-2.1/hy3dpaint/custom_rasterizer && \
#     export CUDA_NVCC_FLAGS="-allow-unsupported-compiler" && \
#     pip install --no-build-isolation -e .

# RUN cd /workspace/Hunyuan3D-2.1/hy3dpaint/DifferentiableRenderer && \
#     bash compile_mesh_painter.sh

# RUN cd /workspace/Hunyuan3D-2.1/hy3dpaint && \
#     mkdir -p ckpt && \
#     wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth -P ckpt

# RUN cd /workspace/Hunyuan3D-2.1/hy3dpaint && \
#     sed -i 's|cfgs/hunyuan-paint-pbr.yaml|hy3dpaint/cfgs/hunyuan-paint-pbr.yaml|' textureGenPipeline.py

# RUN cd /workspace/Hunyuan3D-2.1/hy3dpaint/utils && \
#     sed -i 's|custom_pipeline = config.custom_pipeline|custom_pipeline = os.path.join(os.path.dirname(__file__),"..","hunyuanpaintpbr")|' multiview_utils.py

WORKDIR /workspace/Hunyuan3D-2.1
RUN chmod +x start_multiview_rgb_gradio.sh

RUN apt-get update && apt-get install -y \
    libxi6 libgconf-2-4 libxkbcommon-x11-0 libsm6 libxext6 libxrender-dev && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

EXPOSE 6008

CMD ["/bin/bash", "-lc", "./start_multiview_rgb_gradio.sh"]
