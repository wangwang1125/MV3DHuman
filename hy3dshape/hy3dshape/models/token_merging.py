# Open Source Model Licensed under the Apache License Version 2.0
# Copyright (C) 2024 THL A29 Limited, a Tencent company. All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class TokenMerging(nn.Module):
    """
    Token Merging (ToMe) implementation for reducing token count in multi-view scenarios.
    Based on "Token Merging: Your ViT But Faster" (https://arxiv.org/abs/2210.09461)
    """
    
    def __init__(
        self,
        dim: int,
        r: int = 16,  # Reduction ratio: merge every r tokens into 1
        max_num_merged: Optional[int] = None,
        merge_strategy: str = "attention",  # "attention", "similarity", "random"
        preserve_cls: bool = True,  # Whether to preserve CLS token
    ):
        super().__init__()
        self.dim = dim
        self.r = r
        self.max_num_merged = max_num_merged
        self.merge_strategy = merge_strategy
        self.preserve_cls = preserve_cls
        
        # Learnable merge weights for attention-based merging
        if merge_strategy == "attention":
            self.merge_attention = nn.MultiheadAttention(
                embed_dim=dim,
                num_heads=8,
                dropout=0.1,
                batch_first=True
            )
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tokens of shape (B, N, D) where N is sequence length
            mask: Optional mask for tokens to exclude from merging
            
        Returns:
            merged_tokens: Merged tokens of shape (B, N//r, D)
            merge_weights: Weights used for merging, shape (B, N//r, r)
        """
        B, N, D = x.shape
        
        if N <= self.r or self.r <= 1:
            # No merging needed
            return x, torch.ones(B, N, 1, device=x.device, dtype=x.dtype)
        
        # Calculate target number of tokens after merging
        if self.max_num_merged is not None:
            target_tokens = min(self.max_num_merged, N // self.r)
        else:
            target_tokens = N // self.r
            
        # Handle CLS token preservation
        if self.preserve_cls and N > 1:
            cls_token = x[:, :1, :]  # (B, 1, D)
            patch_tokens = x[:, 1:, :]  # (B, N-1, D)
            num_patches = N - 1
        else:
            cls_token = None
            patch_tokens = x
            num_patches = N
        
        # Merge patch tokens
        if num_patches <= 1:
            merged_patches = patch_tokens
            merge_weights = torch.ones(B, num_patches, 1, device=x.device, dtype=x.dtype)
        else:
            merged_patches, merge_weights = self._merge_tokens(patch_tokens, target_tokens - (1 if cls_token is not None else 0))
        
        # Combine CLS token and merged patches
        if cls_token is not None:
            merged_tokens = torch.cat([cls_token, merged_patches], dim=1)
            # Pad merge weights for CLS token
            cls_weights = torch.ones(B, 1, 1, device=x.device, dtype=x.dtype)
            merge_weights = torch.cat([cls_weights, merge_weights], dim=1)
        else:
            merged_tokens = merged_patches
            
        return merged_tokens, merge_weights
    
    def _merge_tokens(self, tokens: torch.Tensor, target_tokens: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Internal method to merge tokens based on selected strategy"""
        B, N, D = tokens.shape
        
        if N <= target_tokens:
            return tokens, torch.ones(B, N, 1, device=tokens.device, dtype=tokens.dtype)
        
        if self.merge_strategy == "attention":
            return self._attention_merge(tokens, target_tokens)
        elif self.merge_strategy == "similarity":
            return self._similarity_merge(tokens, target_tokens)
        elif self.merge_strategy == "random":
            return self._random_merge(tokens, target_tokens)
        else:
            raise ValueError(f"Unknown merge strategy: {self.merge_strategy}")
    
    def _attention_merge(self, tokens: torch.Tensor, target_tokens: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Merge tokens using attention mechanism"""
        B, N, D = tokens.shape
        
        # Use learned attention to compute merge weights
        # Create queries for target positions
        target_queries = tokens[:, :target_tokens, :]  # (B, target_tokens, D)
        
        # Compute attention between target queries and all tokens
        attn_output, attn_weights = self.merge_attention(
            query=target_queries,
            key=tokens,
            value=tokens
        )
        
        # Normalize attention weights for merging
        merge_weights = F.softmax(attn_weights, dim=-1)  # (B, target_tokens, N)
        
        return attn_output, merge_weights
    
    def _similarity_merge(self, tokens: torch.Tensor, target_tokens: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Merge tokens based on cosine similarity"""
        B, N, D = tokens.shape
        
        # Compute pairwise similarities
        tokens_norm = F.normalize(tokens, dim=-1)  # (B, N, D)
        similarity_matrix = torch.bmm(tokens_norm, tokens_norm.transpose(-2, -1))  # (B, N, N)
        
        # Select most similar tokens to merge
        merged_tokens = []
        merge_weights = []
        used_indices = set()
        
        for i in range(target_tokens):
            # Find unused tokens
            available_indices = [j for j in range(N) if j not in used_indices]
            if not available_indices:
                break
                
            # Select anchor token (first available)
            anchor_idx = available_indices[0]
            anchor_token = tokens[:, anchor_idx:anchor_idx+1, :]  # (B, 1, D)
            
            # Find similar tokens to merge with anchor
            similarities = similarity_matrix[:, anchor_idx, available_indices]  # (B, len(available_indices))
            _, top_indices = torch.topk(similarities, min(self.r, len(available_indices)), dim=-1)
            
            # Get tokens to merge
            tokens_to_merge = []
            weights = []
            for batch_idx in range(B):
                batch_indices = available_indices[top_indices[batch_idx]]
                batch_tokens = tokens[batch_idx:batch_idx+1, batch_indices, :]  # (1, num_merge, D)
                tokens_to_merge.append(batch_tokens)
                weights.append(torch.ones(1, len(batch_indices), 1, device=tokens.device))
                
                # Mark indices as used
                for idx in batch_indices:
                    used_indices.add(idx.item())
            
            # Average merge
            merged_token = torch.mean(torch.cat(tokens_to_merge, dim=0), dim=1, keepdim=True)  # (B, 1, D)
            merged_tokens.append(merged_token)
            merge_weights.append(torch.cat(weights, dim=0))
        
        merged_tokens = torch.cat(merged_tokens, dim=1)  # (B, target_tokens, D)
        merge_weights = torch.cat(merge_weights, dim=1)  # (B, target_tokens, merge_count)
        
        return merged_tokens, merge_weights
    
    def _random_merge(self, tokens: torch.Tensor, target_tokens: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Random token merging (baseline method)"""
        B, N, D = tokens.shape
        
        # Randomly group tokens
        indices = torch.randperm(N, device=tokens.device)
        groups = indices[:target_tokens * self.r].view(target_tokens, self.r)
        
        merged_tokens = []
        merge_weights = []
        
        for group in groups:
            group_tokens = tokens[:, group, :]  # (B, r, D)
            merged_token = torch.mean(group_tokens, dim=1, keepdim=True)  # (B, 1, D)
            weight = torch.ones(B, 1, self.r, device=tokens.device) / self.r
            
            merged_tokens.append(merged_token)
            merge_weights.append(weight)
        
        merged_tokens = torch.cat(merged_tokens, dim=1)  # (B, target_tokens, D)
        merge_weights = torch.cat(merge_weights, dim=1)  # (B, target_tokens, r)
        
        return merged_tokens, merge_weights


class AdaptiveTokenMerging(nn.Module):
    """
    Adaptive Token Merging that adjusts merge ratio based on sequence length
    """
    
    def __init__(
        self,
        dim: int,
        min_r: int = 2,
        max_r: int = 16,
        target_tokens: int = 1369,  # Target token count (original single-view count)
        merge_strategy: str = "attention",
    ):
        super().__init__()
        self.dim = dim
        self.min_r = min_r
        self.max_r = max_r
        self.target_tokens = target_tokens
        self.merge_strategy = merge_strategy
        
        self.token_merger = TokenMerging(
            dim=dim,
            r=1,  # Will be set dynamically
            merge_strategy=merge_strategy,
            preserve_cls=True
        )
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tokens of shape (B, N, D)
            mask: Optional mask
            
        Returns:
            merged_tokens: Merged tokens
            merge_weights: Merge weights
        """
        B, N, D = x.shape
        
        if N <= self.target_tokens:
            # No merging needed
            return x, torch.ones(B, N, 1, device=x.device, dtype=x.dtype)
        
        # Calculate adaptive merge ratio
        r = max(self.min_r, min(self.max_r, N // self.target_tokens))
        
        # Update merger with new ratio
        self.token_merger.r = r
        
        return self.token_merger(x, mask)


class MultiViewTokenMerging(nn.Module):
    """
    Specialized Token Merging for multi-view scenarios
    """
    
    def __init__(
        self,
        dim: int,
        num_views: int = 4,
        view_merge_ratio: float = 0.5,  # Merge ratio within each view
        cross_view_merge_ratio: float = 0.25,  # Additional cross-view merging
        merge_strategy: str = "attention",
    ):
        super().__init__()
        self.dim = dim
        self.num_views = num_views
        self.view_merge_ratio = view_merge_ratio
        self.cross_view_merge_ratio = cross_view_merge_ratio
        self.merge_strategy = merge_strategy
        
        # View-specific token merging
        self.view_merger = TokenMerging(
            dim=dim,
            r=max(1, int(1 / view_merge_ratio)),
            merge_strategy=merge_strategy,
            preserve_cls=True
        )
        
        # Cross-view merging
        self.cross_view_merger = TokenMerging(
            dim=dim,
            r=max(1, int(1 / cross_view_merge_ratio)),
            merge_strategy=merge_strategy,
            preserve_cls=True
        )
    
    def forward(self, x: torch.Tensor, view_splits: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tokens of shape (B, N, D) where N = num_views * tokens_per_view
            view_splits: Optional tensor indicating view boundaries
            
        Returns:
            merged_tokens: Merged tokens
            merge_weights: Merge weights
        """
        B, N, D = x.shape
        
        # Estimate tokens per view if not provided
        if view_splits is None:
            tokens_per_view = N // self.num_views
            view_splits = torch.arange(0, N + 1, tokens_per_view, device=x.device)
        
        # First, merge tokens within each view
        view_merged_tokens = []
        view_merge_weights = []
        
        for i in range(len(view_splits) - 1):
            start_idx = view_splits[i]
            end_idx = view_splits[i + 1]
            view_tokens = x[:, start_idx:end_idx, :]
            
            merged_view, view_weights = self.view_merger(view_tokens)
            view_merged_tokens.append(merged_view)
            view_merge_weights.append(view_weights)
        
        # Concatenate all view tokens
        all_view_tokens = torch.cat(view_merged_tokens, dim=1)  # (B, merged_N, D)
        all_view_weights = torch.cat(view_merge_weights, dim=1)  # (B, merged_N, merge_count)
        
        # Then, apply cross-view merging to reduce redundancy
        final_tokens, cross_view_weights = self.cross_view_merger(all_view_tokens)
        
        # Combine merge weights (approximate)
        final_weights = torch.ones(B, final_tokens.shape[1], 1, device=x.device, dtype=x.dtype)
        
        return final_tokens, final_weights
