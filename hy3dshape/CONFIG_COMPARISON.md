# 配置文件对比分析

## 深度图 vs 法线图配置对比

### 关键差异

| 配置项 | 深度图 | 法线图 | 原因 |
|--------|--------|--------|------|
| **ControlNet** | ✅ 使用 | ❌ 不使用 | 深度图需要额外的特征提取网络 |
| **with_decoupled_ca** | `true` | `false` | 深度图需要额外的条件处理 |
| **additional_cond_hidden_state** | `768` | - | 深度特征维度 |
| **decoupled_ca_dim** | `16` | - | DCA token数量 |
| **decoupled_ca_weight** | `2.0` | - | 条件权重 |
| **DinoImageEncoderMV** | ✅ | ✅ | 两者都使用 |
| **view_num** | `2` | `2` | 相同 |

## 为什么法线图配置更简单？

### 深度图架构
```
RGB图 → DinoImageEncoderMV → Token (注意力)
深度图 → ControlNet → 额外Token → Decoupled Cross-Attention
```

### 法线图架构  
```
RGB图 → DinoImageEncoderMV → Token (注意力)
法线图 → DinoImageEncoderMV → Token (注意力，和RGB合并)
```

**关键区别**：
- 深度图: ControlNet输出固定16个token用于decoupled cross-attention
- 法线图: 通过DinoImageEncoderMV生成动态token（与RGB图token合并），无需额外条件处理

## 配置完整性检查

### ✅ 深度图配置包含
1. `control_net_config` - ControlNet设置
2. `control_in_channels: 1` - 单通道深度输入
3. `with_decoupled_ca: true` - 启用decoupled cross-attention
4. `additional_cond_hidden_state: 768` - 条件维度
5. `decoupled_ca_dim: 16` - DCA token数量
6. `decoupled_ca_weight: 2.0` - 条件权重

### ✅ 法线图配置包含
1. `load_normal: true` - 启用法线图加载
2. `normal_fusion_strategy: "multiview"` - 法线图融合策略
3. `with_decoupled_ca: false` - 不需要额外条件
4. `DinoImageEncoderMV` - 处理法线图（和RGB一样）
5. `num_views: 2` - 2个视图

### ✅ 两者共同包含
1. `lora_config` - LoRA配置
2. `image_processor_cfg: MVImageProcessorV2` - 多视图处理器
3. `mean/std` - 图像归一化参数
4. `pc_size, pc_sharpedge_size` - 点云采样参数
5. `callbacks` - 训练回调（日志、保存等）

## 总结

**法线图配置参数少于深度图是正常的**，因为：
1. 法线图不需要ControlNet架构
2. 法线图不需要decoupled cross-attention
3. 法线图直接通过DinoImageEncoderMV处理，与RGB图并行
4. 法线图token与RGB token合并，无需额外条件维度

**两种方案的token生成方式不同**：
- **深度图**: RGB token + 16个固定深度token（decoupled CA）
- **法线图**: RGB token + Normal token（直接融合）

因此，法线图配置是**完整且正确的**！

