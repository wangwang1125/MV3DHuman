# -*- coding: utf-8 -*-

# Hunyuan 3D is licensed under the TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT
# except for the third-party components listed below.
# Hunyuan 3D does not impose any additional limitations beyond what is outlined
# in the repsective licenses of these third-party components.
# Users must comply with all terms and conditions of original licenses of these third-party
# components and must ensure that the usage of the third party components adheres to
# all relevant laws and regulations.

"""
Multi-view ControlNet for processing multiple depth maps from different viewpoints
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class MultiViewDepthControlNet(nn.Module):
    """
    Multi-view depth ControlNet that processes depth maps from multiple viewpoints
    and fuses them into contextual features for 3D generation
    """
    
    def __init__(
        self, 
        in_channels: int = 1,
        num_views: int = 4,
        out_channels: int = 768,
        fusion_strategy: str = "attention",  # "attention", "concat", "average", "weighted"
        hidden_dim: int = 256
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_views = num_views
        self.out_channels = out_channels
        self.fusion_strategy = fusion_strategy
        self.hidden_dim = hidden_dim
        
        # Individual depth encoder for each view
        self.depth_encoder = nn.Sequential(
            # First conv block
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # H/2, W/2
            nn.ReLU(inplace=True),
            
            # Second conv block
            nn.Conv2d(64, 128, 3, stride=2, padding=1),  # H/4, W/4
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),  # H/8, W/8
            nn.ReLU(inplace=True),
            
            # Third conv block
            nn.Conv2d(256, hidden_dim, 3, stride=2, padding=1),  # H/16, W/16
            nn.ReLU(inplace=True),
            
            # Global average pooling
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),  # (B*num_views, hidden_dim)
        )
        
        # View embedding for positional encoding of different viewpoints
        self.view_embeddings = nn.Embedding(num_views, hidden_dim)
        
        # Multi-view fusion layer
        if fusion_strategy == "attention":
            self.fusion_layer = MultiViewAttentionFusion(hidden_dim, num_views, out_channels)
        elif fusion_strategy == "concat":
            self.fusion_layer = nn.Linear(hidden_dim * num_views, out_channels)
        elif fusion_strategy == "weighted":
            self.fusion_layer = WeightedFusion(hidden_dim, num_views, out_channels)
        else:  # average
            self.fusion_layer = nn.Linear(hidden_dim, out_channels)
            
        # Final projection to match DiT expected feature dimension
        self.final_proj = nn.Linear(out_channels, out_channels)
        
        # Token expansion: generate 16 diverse tokens from single depth feature
        # Using learnable queries similar to DETR's object queries
        self.num_depth_tokens = 16  # Match DCA's decoupled_ca_dim
        self.depth_token_queries = nn.Parameter(torch.randn(1, self.num_depth_tokens, out_channels) * 0.02)
        
        # Cross-attention to generate diverse tokens
        self.token_generator = nn.MultiheadAttention(
            embed_dim=out_channels,
            num_heads=8,
            dropout=0.0,
            batch_first=True
        )
        
        # Layer norm for stability
        self.token_norm = nn.LayerNorm(out_channels)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights with appropriate schemes"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, 0, 0.02)
    
    def forward(self, depth_maps: torch.Tensor, enable_padding: bool = True) -> torch.Tensor:
        """
        Forward pass of multi-view depth ControlNet with dynamic view adaptation
        
        Args:
            depth_maps: (B, num_views, 1, H, W) multi-view depth maps
            enable_padding: Whether to enable view padding when num_views < self.num_views
            
        Returns:
            depth_features: (B, 1, out_channels) fused depth features as sequence
        """
        B, actual_num_views, C, H, W = depth_maps.shape
        assert C == self.in_channels, f"Expected {self.in_channels} channels, got {C}"
        
        # Handle variable number of views
        if actual_num_views != self.num_views:
            if actual_num_views > self.num_views:
                # Truncate to expected number of views
                depth_maps = depth_maps[:, :self.num_views, :, :, :]
                actual_num_views = self.num_views
                print(f"Warning: Truncating {actual_num_views} views to {self.num_views}")
            elif enable_padding and actual_num_views < self.num_views:
                # Pad with repeated last view
                depth_maps = self._pad_views(depth_maps, self.num_views)
                actual_num_views = self.num_views
                print(f"Info: Padded {depth_maps.shape[1]} views to {self.num_views} using repetition")
            else:
                # Use dynamic adaptation without padding
                print(f"Info: Using dynamic adaptation for {actual_num_views} views (expected {self.num_views})")
        
        # Reshape for batch processing: (B*actual_num_views, C, H, W)
        depth_flat = depth_maps.view(B * actual_num_views, C, H, W)
        
        # Extract features from each view: (B*actual_num_views, hidden_dim)
        view_features = self.depth_encoder(depth_flat)
        
        # Reshape back to multi-view format: (B, actual_num_views, hidden_dim)
        view_features = view_features.view(B, actual_num_views, self.hidden_dim)
        
        # Add view-specific positional embeddings with dynamic adaptation
        view_indices = torch.arange(actual_num_views, device=depth_maps.device)
        # Ensure view indices don't exceed embedding table size
        view_indices = torch.clamp(view_indices, 0, self.num_views - 1)
        view_embeds = self.view_embeddings(view_indices)  # (actual_num_views, hidden_dim)
        view_embeds = view_embeds.unsqueeze(0).expand(B, -1, -1)  # (B, actual_num_views, hidden_dim)
        
        # Combine view features with positional embeddings
        enhanced_features = view_features + view_embeds  # (B, actual_num_views, hidden_dim)
        
        # Fuse multi-view features with dynamic adaptation
        if self.fusion_strategy == "attention":
            # Attention fusion naturally handles variable sequence lengths
            fused_features = self.fusion_layer(enhanced_features)  # (B, out_channels)
        elif self.fusion_strategy == "concat":
            # For concatenation, we need to handle variable dimensions
            if actual_num_views != self.num_views:
                # Use average pooling to maintain consistent output dimension
                avg_features = enhanced_features.mean(dim=1)  # (B, hidden_dim)
                fused_features = nn.Linear(self.hidden_dim, self.out_channels).to(enhanced_features.device)(avg_features)
                print(f"Info: Using average pooling for concat fusion with {actual_num_views} views")
            else:
                concat_features = enhanced_features.view(B, -1)  # (B, num_views * hidden_dim)
                fused_features = self.fusion_layer(concat_features)  # (B, out_channels)
        elif self.fusion_strategy == "weighted":
            # Weighted fusion can handle variable views
            fused_features = self.fusion_layer(enhanced_features)  # (B, out_channels)
        else:  # average
            avg_features = enhanced_features.mean(dim=1)  # (B, hidden_dim)
            fused_features = self.fusion_layer(avg_features)  # (B, out_channels)
        
        # Final projection
        output_features = self.final_proj(fused_features)  # (B, out_channels)
        
        # Generate 16 diverse depth tokens using learnable queries and cross-attention
        # This is more effective than simple repetition as it allows each token to
        # capture different aspects of the depth information
        B = output_features.shape[0]
        
        # Expand depth feature as key/value
        depth_kv = output_features.unsqueeze(1)  # (B, 1, out_channels)
        
        # Expand learnable queries for this batch
        queries = self.depth_token_queries.expand(B, -1, -1)  # (B, 16, out_channels)
        
        # Use cross-attention: queries attend to depth features
        # This generates 16 diverse tokens, each potentially focusing on different depth aspects
        depth_tokens, _ = self.token_generator(
            query=queries,           # (B, 16, out_channels)
            key=depth_kv,           # (B, 1, out_channels)
            value=depth_kv,         # (B, 1, out_channels)
            need_weights=False
        )
        
        # Add residual connection and normalize
        depth_tokens = self.token_norm(depth_tokens + queries)  # (B, 16, out_channels)
        
        return depth_tokens  # (B, 16, out_channels)
    
    def _pad_views(self, depth_maps: torch.Tensor, target_views: int) -> torch.Tensor:
        """
        Pad depth maps to target number of views using repetition strategy
        
        Args:
            depth_maps: (B, current_views, C, H, W)
            target_views: Target number of views
            
        Returns:
            padded_depth_maps: (B, target_views, C, H, W)
        """
        B, current_views, C, H, W = depth_maps.shape
        
        if current_views >= target_views:
            return depth_maps[:, :target_views, :, :, :]
        
        # Calculate how many views to pad
        views_to_pad = target_views - current_views
        
        # Repeat the last view to fill the gap
        last_view = depth_maps[:, -1:, :, :, :]  # (B, 1, C, H, W)
        repeated_views = last_view.repeat(1, views_to_pad, 1, 1, 1)  # (B, views_to_pad, C, H, W)
        
        # Concatenate original views with repeated views
        padded_depth_maps = torch.cat([depth_maps, repeated_views], dim=1)  # (B, target_views, C, H, W)
        
        return padded_depth_maps


class MultiViewAttentionFusion(nn.Module):
    """Multi-head attention for fusing multi-view depth features"""
    
    def __init__(self, hidden_dim: int, num_views: int, out_channels: int = None, num_heads: int = 8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.out_channels = out_channels or hidden_dim
        
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        # Multi-head attention components
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(0.1)
        
        # Output projection
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Final dimension projection to match expected output channels
        self.final_proj = nn.Linear(hidden_dim, self.out_channels)
        
        # Global token for aggregation
        self.global_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
    
    def forward(self, view_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            view_features: (B, num_views, hidden_dim)
        Returns:
            fused_features: (B, out_channels)
        """
        B, num_views, hidden_dim = view_features.shape
        
        # Add global aggregation token
        global_tokens = self.global_token.expand(B, -1, -1)  # (B, 1, hidden_dim)
        tokens = torch.cat([global_tokens, view_features], dim=1)  # (B, 1+num_views, hidden_dim)
        
        # Multi-head attention
        Q = self.query(tokens)  # (B, 1+num_views, hidden_dim)
        K = self.key(tokens)    # (B, 1+num_views, hidden_dim)
        V = self.value(tokens)  # (B, 1+num_views, hidden_dim)
        
        # Reshape for multi-head attention
        Q = Q.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)  # (B, num_heads, 1+num_views, head_dim)
        K = K.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)  # (B, num_heads, 1+num_views, head_dim)
        V = V.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)  # (B, num_heads, 1+num_views, head_dim)
        
        # Scaled dot-product attention
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention
        attended = torch.matmul(attention_weights, V)  # (B, num_heads, 1+num_views, head_dim)
        
        # Reshape and take only the global token output
        attended = attended.transpose(1, 2).contiguous().view(B, -1, hidden_dim)  # (B, 1+num_views, hidden_dim)
        global_output = attended[:, 0]  # (B, hidden_dim) - take only global token
        
        # Intermediate projection
        intermediate_output = self.out_proj(global_output)  # (B, hidden_dim)
        
        # Final projection to match expected output channels
        return self.final_proj(intermediate_output)  # (B, out_channels)


class WeightedFusion(nn.Module):
    """Learnable weighted fusion of multi-view features"""
    
    def __init__(self, hidden_dim: int, num_views: int, out_channels: int):
        super().__init__()
        self.num_views = num_views
        
        # Attention weights for each view
        self.attention_weights = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 4, 1)
        )
        
        # Final projection
        self.final_proj = nn.Linear(hidden_dim, out_channels)
    
    def forward(self, view_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            view_features: (B, num_views, hidden_dim)
        Returns:
            weighted_features: (B, out_channels)
        """
        B, num_views, hidden_dim = view_features.shape
        
        # Compute attention weights for each view
        attention_logits = self.attention_weights(view_features)  # (B, num_views, 1)
        attention_weights = F.softmax(attention_logits, dim=1)  # (B, num_views, 1)
        
        # Weighted sum of view features
        weighted_features = torch.sum(view_features * attention_weights, dim=1)  # (B, hidden_dim)
        
        # Final projection
        return self.final_proj(weighted_features)


def create_multiview_depth_controlnet(
    in_channels: int = 1,
    num_views: int = 4,
    out_channels: int = 768,
    fusion_strategy: str = "attention"
) -> MultiViewDepthControlNet:
    """Factory function to create multi-view depth ControlNet"""
    return MultiViewDepthControlNet(
        in_channels=in_channels,
        num_views=num_views,
        out_channels=out_channels,
        fusion_strategy=fusion_strategy
    )
