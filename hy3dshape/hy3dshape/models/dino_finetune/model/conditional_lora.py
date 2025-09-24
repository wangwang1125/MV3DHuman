import torch
import torch.nn as nn


class ConditionalLoRA(nn.Module):
    """
    条件化LoRA模块，使用深度图条件向量动态调制LoRA权重
    """
    def __init__(
        self,
        qkv: nn.Module,
        rank: int = 4,
        condition_dim: int = 1024,
        alpha: float = 1.0,
    ):
        super().__init__()
        self.qkv = qkv
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # 原始维度
        self.in_features = qkv.in_features
        self.out_features = qkv.out_features
        
        # 为Q和V创建LoRA低秩矩阵
        # 注意：这里我们不直接创建A和B矩阵，而是创建它们的基础
        # 实际的A和B矩阵将由条件向量动态生成
        self.lora_a_q_base = nn.Parameter(torch.zeros(rank, self.in_features))
        self.lora_b_q_base = nn.Parameter(torch.zeros(self.in_features, rank))
        
        self.lora_a_v_base = nn.Parameter(torch.zeros(rank, self.in_features))
        self.lora_b_v_base = nn.Parameter(torch.zeros(self.in_features, rank))
        
        # 条件权重生成器
        self.condition_projector = nn.Sequential(
            nn.Linear(condition_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, 4 * rank)  # 为A_q, B_q, A_v, B_v生成调制系数
        )
        
        # 初始化
        nn.init.kaiming_uniform_(self.lora_a_q_base, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b_q_base)
        nn.init.kaiming_uniform_(self.lora_a_v_base, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b_v_base)

    def forward(self, x, condition=None):
        """
        Args:
            x: 输入特征，形状为 [B, N, C]
            condition: 条件向量，形状为 [B, condition_dim]
        Returns:
            更新后的特征
        """
        # 计算原始qkv
        qkv = self.qkv(x)  # 形状: [B, N, 3 * C]
        
        batch_size = x.shape[0]
        
        if condition is not None:
            # 生成条件权重
            modulation = self.condition_projector(condition)  # [B, 4*rank]
            
            # 分割为各个LoRA矩阵的调制系数
            mod_a_q, mod_b_q, mod_a_v, mod_b_v = torch.chunk(modulation, 4, dim=1)
            
            # 扩展为适当的形状以便于广播
            mod_a_q = mod_a_q.view(batch_size, self.rank, 1)  # [B, rank, 1]
            mod_b_q = mod_b_q.view(batch_size, 1, self.rank)  # [B, 1, rank]
            mod_a_v = mod_a_v.view(batch_size, self.rank, 1)  # [B, rank, 1]
            mod_b_v = mod_b_v.view(batch_size, 1, self.rank)  # [B, 1, rank]
            
            # 为每个样本动态生成LoRA权重
            for i in range(batch_size):
                # 生成当前样本的A和B矩阵
                a_q = self.lora_a_q_base * mod_a_q[i]  # [rank, in_features]
                b_q = self.lora_b_q_base * mod_b_q[i]  # [in_features, rank]
                
                a_v = self.lora_a_v_base * mod_a_v[i]  # [rank, in_features]
                b_v = self.lora_b_v_base * mod_b_v[i]  # [in_features, rank]
                
                # 计算LoRA增量
                delta_q = (x[i] @ b_q) @ a_q * self.scaling  # [N, C]
                delta_v = (x[i] @ b_v) @ a_v * self.scaling  # [N, C]
                
                # 将增量添加到原始qkv张量
                qkv[i, :, :self.in_features] += delta_q  # 更新q部分
                qkv[i, :, -self.in_features:] += delta_v  # 更新v部分
        
        return qkv


import math

def init_conditional_lora(model, rank=4, condition_dim=1024, alpha=1.0, target_modules=["qkv"]):
    """
    将模型中的目标模块替换为条件化LoRA模块
    
    Args:
        model: 要应用LoRA的模型
        rank: LoRA的秩
        condition_dim: 条件向量的维度
        alpha: LoRA缩放因子
        target_modules: 要替换的模块名称列表
    
    Returns:
        更新后的模型
    """
    for name, module in model.named_modules():
        if any(target_name in name for target_name in target_modules):
            if isinstance(module, nn.Linear):
                parent_name = name.rsplit(".", 1)[0]
                parent = model.get_submodule(parent_name)
                target_name = name.rsplit(".", 1)[1]
                
                # 替换为条件化LoRA模块
                setattr(
                    parent, 
                    target_name, 
                    ConditionalLoRA(
                        module,
                        rank=rank,
                        condition_dim=condition_dim,
                        alpha=alpha
                    )
                )
    
    return model