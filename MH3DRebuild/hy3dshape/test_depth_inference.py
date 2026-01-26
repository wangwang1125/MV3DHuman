#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test script for depth-conditioned LoRA inference
"""

import os
import sys
import yaml
import torch
import cv2
import numpy as np
from PIL import Image

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hy3dshape.models.diffusion.flow_matching_sit import Diffuser
from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline


def load_depth_map(depth_path):
    """Load and preprocess depth map from EXR file"""
    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise ValueError(f"Failed to load depth map: {depth_path}")
    
    # Extract depth channel
    if len(depth.shape) == 3:
        depth = depth[:, :, 0]
    
    # Filter invalid depth values
    depth[depth > 1e9] = 0
    
    # Normalize depth to [0, 1] range
    if depth.max() > depth.min():
        depth = (depth - depth.min()) / (depth.max() - depth.min())
    
    # Convert to tensor and add batch/channel dimensions
    depth = torch.FloatTensor(depth).unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, H, W)
    
    return depth


def test_depth_inference():
    """Test depth-conditioned inference"""
    
    # Configuration
    config_path = "configs/hunyuandit-depth-lora-flowmatching-dinol518-bf16-lr1e5-4096.yaml"
    lora_checkpoint_path = "output_folder/dit/depth_lora_checkpoints/step_1000"  # Adjust path as needed
    
    # Test data paths
    test_image_path = "tools/mini_depth_trainset/preprocessed/Female_01/render_cond/000.png"
    test_depth_path = "tools/mini_depth_trainset/preprocessed/Female_01/render_cond/000_depth.exr"
    
    print("Loading configuration...")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print("Initializing model...")
    # Initialize model from config
    model = Diffuser(**config['model']['params'])
    
    # Load LoRA weights if available
    if os.path.exists(lora_checkpoint_path):
        print(f"Loading LoRA checkpoint from: {lora_checkpoint_path}")
        from peft import PeftModel
        # 只对主DiT模型应用LoRA，ControlNet使用全参数训练，不需要加载LoRA
        model.model = PeftModel.from_pretrained(model.model, lora_checkpoint_path)
        print("✅ LoRA weights loaded successfully for DiT model")
        
        # ControlNet不使用LoRA，如果有保存的ControlNet权重，需要单独加载
        if model.controlnet is not None:
            controlnet_checkpoint = os.path.join(os.path.dirname(lora_checkpoint_path), 'controlnet.pth')
            if os.path.exists(controlnet_checkpoint):
                model.controlnet.load_state_dict(torch.load(controlnet_checkpoint))
                print("✅ ControlNet weights loaded successfully")
            else:
                print("⚠️  ControlNet checkpoint not found, using initialized weights")
    else:
        print(f"LoRA checkpoint not found at: {lora_checkpoint_path}")
        print("Using base model without LoRA fine-tuning")
    
    # Set model to evaluation mode
    model.eval()
    model = model.cuda()
    
    print("Loading test data...")
    # Load test image
    image = Image.open(test_image_path).convert('RGB')
    
    # Load test depth map
    depth = load_depth_map(test_depth_path)
    depth = depth.cuda()
    
    print("Running inference...")
    with torch.no_grad():
        # Prepare batch
        batch = {
            'image': image,
            'depth': depth,
        }
        
        # Run inference
        outputs = model.sample(batch, output_type='trimesh')
        
        print(f"Generated {len(outputs)} meshes")
        
        # Save results
        output_dir = "depth_inference_results"
        os.makedirs(output_dir, exist_ok=True)
        
        for i, mesh_list in enumerate(outputs):
            if mesh_list and mesh_list[0] is not None:
                output_path = os.path.join(output_dir, f"depth_conditioned_mesh_{i:03d}.glb")
                mesh_list[0].export(output_path)
                print(f"Saved mesh to: {output_path}")
            else:
                print(f"Mesh {i} is None or empty")
    
    print("Inference completed!")


if __name__ == "__main__":
    test_depth_inference()
