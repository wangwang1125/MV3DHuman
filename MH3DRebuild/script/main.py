'''
MH3DRebuild: Multi-view RGB 3D reconstruction with MV LoRA.
Entry: parse args, load pipeline+LoRA, run inference, export mesh.
'''
import os
import sys
import time
import json
import glob
from typing import Optional

# Project root = MH3DRebuild (contains hy3dshape, torchvision_fix, script, confdata)
def _find_project_root():
    d = os.path.abspath(os.path.dirname(os.path.abspath(__file__)))
    for _ in range(6):
        if os.path.isdir(d):
            try:
                names = os.listdir(d)
            except OSError:
                names = []
            if 'hy3dshape' in names:
                return d
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    return os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

_PROJECT_ROOT = _find_project_root()
_HY3DSHAPE_ROOT = os.path.join(_PROJECT_ROOT, 'hy3dshape')
for p in (_HY3DSHAPE_ROOT, _PROJECT_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from torchvision_fix import apply_fix
    apply_fix()
except ImportError:
    pass
except Exception:
    pass

import torch
from PIL import Image

from info import Info


def parse_option():
    if len(sys.argv) < 2:
        return
    root = {}
    info = Info()
    if sys.argv[1] == '-h':
        root['Name'] = info.SOFTWARE_NAME
        root['NameZh'] = info.SOFTWARE_NAME_ZH
        root['Bref'] = info.SOFTWARE_BRIEF
        root['GPU'] = info.SOFTWARE_WITH_CUDA
        root['Input'] = [{'ID': i.id, 'Name': i.name, 'Type': i.type, 'Bref': i.bref, 'Self': i.sf} for i in info.Input]
        root['Output'] = [{'ID': o.id, 'Name': o.name, 'Type': o.type, 'Bref': o.bref, 'Self': o.sf} for o in info.Output]
        print(json.dumps(root, indent=4, ensure_ascii=False))
        sys.exit(0)
    if sys.argv[1] == '-v':
        root['Name'] = info.SOFTWARE_NAME
        root['Abbr'] = info.SOFTWARE_ABBR
        root['Version'] = info.SOFTWARE_VERSION
        root['BuildTime'] = time.ctime(os.path.getmtime(sys.argv[0]) if os.path.isfile(sys.argv[0]) else 0)
        print(json.dumps(root, indent=4, ensure_ascii=False))
        sys.exit(0)


def resolve_lora_path(models_dir: str) -> Optional[str]:
    """Resolve LoRA path: prefer models_dir, then mv finetuning/lora dirs. Returns None if not found."""
    models_dir = os.path.abspath(models_dir)

    def first_ckpt(d):
        if not os.path.isdir(d):
            return None
        f = glob.glob(os.path.join(d, '*.ckpt'))
        if not f:
            return None
        f.sort(key=os.path.getmtime, reverse=True)
        return f[0]

    def first_peft(d):
        if not os.path.isdir(d):
            return None
        steps = glob.glob(os.path.join(d, 'step_*'))
        if not steps:
            return None
        steps.sort(key=lambda x: int(x.split('_')[-1]) if x.split('_')[-1].isdigit() else 0)
        return steps[-1]

    # 1. models dir
    ckpt = first_ckpt(models_dir)
    if ckpt:
        return ckpt
    peft = first_peft(models_dir)
    if peft:
        return peft

    # 2. mv finetuning / lora (relative to project root)
    root = _PROJECT_ROOT
    finetuning_dirs = [
        os.path.join(root, 'hy3dshape', 'output_folder', 'dit', 'multiview_rgb_finetuning_mv', 'ckpt'),
        os.path.join(root, 'output_folder', 'dit', 'multiview_rgb_finetuning_mv', 'ckpt'),
    ]
    for d in finetuning_dirs:
        ckpt = first_ckpt(d)
        if ckpt:
            return ckpt

    lora_dirs = [
        os.path.join(root, 'hy3dshape', 'output_folder', 'dit', 'multiview_rgb_lora_finetuning_mv', 'ckpt'),
        os.path.join(root, 'output_folder', 'dit', 'multiview_rgb_lora_finetuning_mv', 'ckpt'),
        os.path.join(root, 'hy3dshape', 'output_folder', 'dit', 'multiview_rgb_lora_checkpoints_mv'),
        os.path.join(root, 'output_folder', 'dit', 'multiview_rgb_lora_checkpoints_mv'),
    ]
    for d in lora_dirs:
        ckpt = first_ckpt(d)
        if ckpt:
            return ckpt
        peft = first_peft(d)
        if peft:
            return peft

    # 3. original (non-mv) fallback
    for d in [
        os.path.join(root, 'hy3dshape', 'output_folder', 'dit', 'multiview_rgb_finetuning', 'ckpt'),
        os.path.join(root, 'output_folder', 'dit', 'multiview_rgb_finetuning', 'ckpt'),
    ]:
        ckpt = first_ckpt(d)
        if ckpt:
            return ckpt
    for d in [
        os.path.join(root, 'hy3dshape', 'output_folder', 'dit', 'multiview_rgb_lora_checkpoints'),
        os.path.join(root, 'output_folder', 'dit', 'multiview_rgb_lora_checkpoints'),
    ]:
        peft = first_peft(d)
        if peft:
            return peft

    return None


def load_images(input_data_dir: str) -> dict[str, Image.Image]:
    """Load front/left/back/right from input_data dir. Supports .jpg, .png."""
    out = {}
    for name in ('front', 'left', 'back', 'right'):
        found = None
        for ext in ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'):
            p = os.path.join(input_data_dir, name + ext)
            if os.path.isfile(p):
                found = p
                break
        if not found:
            raise FileNotFoundError(f'Missing view image: {name} (.jpg/.png) in {input_data_dir}')
        out[name] = Image.open(found).convert('RGB')
    return out


def load_pipeline_and_lora(device: str, models_dir: str, rgb_lora_path: Optional[str]):
    from hy3dshape import Hunyuan3DDiTFlowMatchingPipeline
    from hy3dshape.preprocessors import MVImageProcessorV2
    from hy3dshape.models.conditioner import DinoImageEncoderMV

    models_root = os.path.abspath(models_dir)
    os.environ['HY3DGEN_MODELS'] = models_root

    model_path = 'tencent/Hunyuan3D-2mv'
    subfolder = 'hunyuan3d-dit-v2-mv'
    num_views = 4

    pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        model_path,
        subfolder=subfolder,
        use_safetensors=False,
        device=device,
    )

    if rgb_lora_path and os.path.exists(rgb_lora_path):
        try:
            from peft import PeftModel
            if rgb_lora_path.endswith('.ckpt'):
                ckpt = torch.load(rgb_lora_path, map_location='cpu')
                if 'state_dict' not in ckpt:
                    raise ValueError('Checkpoint missing state_dict')
                state_dict = ckpt['state_dict']
                lora_keys = [k for k in state_dict if 'lora' in k.lower()]
                model_keys = [k for k in state_dict if k.startswith('model.')]
                model_state_dict = {k[6:]: v for k, v in state_dict.items() if k.startswith('model.')}

                if lora_keys:
                    from peft import LoraConfig, get_peft_model
                    lora_config = LoraConfig(
                        r=8,
                        lora_alpha=8,
                        target_modules=['qkv', 'proj', 'linear1', 'linear2'],
                        lora_dropout=0.0,
                    )
                    pipeline.model = get_peft_model(pipeline.model, lora_config)
                pipeline.model.load_state_dict(model_state_dict, strict=False)
            elif os.path.isdir(rgb_lora_path):
                pipeline.model = PeftModel.from_pretrained(pipeline.model, rgb_lora_path)
            else:
                raise ValueError(f'Unsupported LoRA format: {rgb_lora_path}')
        except Exception as e:
            import traceback
            print(f'LoRA load failed: {e}')
            traceback.print_exc()

    pipeline.image_processor = MVImageProcessorV2(size=518)
    dtype = getattr(pipeline, 'dtype', torch.float16) or (getattr(pipeline.model, 'dtype', None) or torch.float16)
    if not hasattr(pipeline, 'conditioner') or not hasattr(pipeline.conditioner, 'main_image_encoder'):
        raise RuntimeError('Pipeline missing conditioner.main_image_encoder')
    enc = pipeline.conditioner.main_image_encoder
    if not isinstance(enc, DinoImageEncoderMV):
        new_enc = DinoImageEncoderMV(
            version='facebook/dinov2-large',
            image_size=518,
            use_cls_token=True,
            view_num=num_views,
        )
        if hasattr(enc, 'model') and hasattr(new_enc, 'model'):
            try:
                new_enc.model.load_state_dict(enc.model.state_dict())
            except Exception:
                pass
        new_enc = new_enc.to(device, dtype=dtype)
        pipeline.conditioner.main_image_encoder = new_enc

    return pipeline


def run_inference(
    pipeline,
    image: dict[str, Image.Image],
    device: str,
    seed: int = 1234,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    octree_resolution: int = 256,
    num_chunks: int = 200000,
    rembg_model: str = 'u2net_human_seg',
):
    from hy3dshape.rembg import BackgroundRemover
    from hy3dshape.pipelines import export_to_trimesh

    rembg = BackgroundRemover(model_name=rembg_model)
    for k, v in image.items():
        if v.mode == 'RGB':
            image[k] = rembg(v.convert('RGB'))
        else:
            image[k] = rembg(v)

    gen = torch.Generator()
    if device == 'cuda' and torch.cuda.is_available():
        gen = gen.manual_seed(seed)
    else:
        gen = gen.manual_seed(seed)

    out = pipeline(
        image=image,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=gen,
        octree_resolution=octree_resolution,
        num_chunks=num_chunks,
        output_type='mesh',
    )
    raw = export_to_trimesh(out)
    mesh = raw[0] if isinstance(raw, list) and len(raw) > 0 else raw
    return mesh


if __name__ == '__main__':
    parse_option()

    info = Info()
    n_in = len(info.Input)
    n_out = len(info.Output)
    expected = 1 + n_in + n_out
    if len(sys.argv) != expected:
        usage = ' '.join([f'python {sys.argv[0]}'] + [i.name for i in info.Input] + [o.name for o in info.Output])
        print(f'Usage: {usage}')
        sys.exit(1)

    input_data = os.path.abspath(sys.argv[1])
    models_dir = os.path.abspath(sys.argv[2])
    output_mesh = os.path.abspath(sys.argv[3])

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    rgb_lora_path = resolve_lora_path(models_dir)
    if rgb_lora_path:
        print(f'Using LoRA: {rgb_lora_path}')
    else:
        print('No LoRA found, using base Hunyuan3D-2mv')

    pipeline = load_pipeline_and_lora(device, models_dir, rgb_lora_path)
    image = load_images(input_data)
    mesh = run_inference(pipeline, image, device)
    os.makedirs(os.path.dirname(output_mesh) or '.', exist_ok=True)
    mesh.export(output_mesh)
    print(f'Saved: {output_mesh}')
