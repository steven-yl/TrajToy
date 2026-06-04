# src/diffusion_policy/networks.py

"""
Defines the neural network architectures for the conditional diffusion policy.

This file contains the core components:
1. SinusoidalTimeEmbedding: Encodes the diffusion timestep.
2. FiLMConditionedResidualBlock: The main building block for the U-Net,
   which injects conditioning information via FiLM layers.
3. StateEncoder: A powerful, attention-based module that processes the
   structured `state_dict` into a single scene embedding vector.
4. ConditionalUNet: The final, top-level model that assembles all components
   and performs the denoising task.
"""

import torch
import torch.nn as nn
from typing import Any
from typing import Dict
from typing import Sequence
from il.modules.model.utils.conditione_block import (
    FiLMConditionedResidualBlock,
    CrossAttnConditionedResidualBlock,
)
from il.modules.model.utils.state_encoder import StateClsEncoder, StateTokenEncoder
from il.modules.model.utils.embedding_block import SinusoidalTimeEmbedding

class ConditionalUNet1D(nn.Module):
    """
    The main model. It takes a noisy trajectory, a timestep, and the scene context,
    and predicts the noise that was added to the trajectory.
    """
    def __init__(
        self,
        prediction_state_dim: int = 4,
        future_len: int = 25,
        time_embed_dim: int = 128,
        scene_embed_dim: int = 128,
        down_dims: Sequence[int] = (64, 128, 256),
        history_state_dim: int = 7,
        road_feature_dim: int = 2,
        history_len: int = 10,
        road_points: int = 60,
        dropout: float = 0.1,
        **_: Any,
    ):
        super().__init__()
        down_dims = [int(d) for d in down_dims]
        if len(down_dims) < 2:
            raise ValueError("`down_dims` must contain at least two channel sizes.")

        # The total dimension of the conditioning vector
        cond_embed_dim = time_embed_dim + scene_embed_dim
        
        # --- Instantiate all sub-modules ---
        self.time_encoder = nn.Sequential(
            SinusoidalTimeEmbedding(time_embed_dim),
            nn.Linear(time_embed_dim, time_embed_dim), nn.Mish(),
            nn.Linear(time_embed_dim, time_embed_dim)
        )
        self.state_cls_encoder = StateClsEncoder(
            history_state_dim=history_state_dim,
            road_feature_dim=road_feature_dim,
            hidden_dim=scene_embed_dim,
            dropout=dropout,
            history_len=history_len,
            road_points=road_points,
        )
        
        self.initial_conv = nn.Conv1d(prediction_state_dim, down_dims[0], kernel_size=1)
        
        # Downsampling path
        self.down_blocks = nn.ModuleList()
        for i in range(len(down_dims) - 1):
            self.down_blocks.append(nn.ModuleList([
                FiLMConditionedResidualBlock(down_dims[i], down_dims[i], cond_embed_dim),
                FiLMConditionedResidualBlock(down_dims[i], down_dims[i+1], cond_embed_dim),
                nn.Conv1d(down_dims[i+1], down_dims[i+1], kernel_size=3, stride=2, padding=1) # Downsample
            ]))
        
        # Middle block
        self.middle_block1 = FiLMConditionedResidualBlock(down_dims[-1], down_dims[-1], cond_embed_dim)
        self.middle_block2 = FiLMConditionedResidualBlock(down_dims[-1], down_dims[-1], cond_embed_dim)
        
        self.up_blocks = nn.ModuleList()
        up_dims = down_dims[::-1]
        for i in range(len(up_dims) - 1):
            # Note the order of layers in the ModuleList for clarity
            # 与下采样 Conv1d(k=3,s=2,p=1) 配对：对固定 L=25 有 25→13→7→13→25（kernel=4 会得到 7→14 与 skip 错位）
            self.up_blocks.append(nn.ModuleList([
                nn.ConvTranspose1d(up_dims[i], up_dims[i], kernel_size=3, stride=2, padding=1),
                FiLMConditionedResidualBlock(up_dims[i] * 2, up_dims[i+1], cond_embed_dim),
                FiLMConditionedResidualBlock(up_dims[i+1], up_dims[i+1], cond_embed_dim),
            ]))
            
        self.final_conv = nn.Conv1d(down_dims[0], prediction_state_dim, kernel_size=1)
        
    def forward(
        self, 
        noisy_trajectory: torch.Tensor, # (B, L, C_traj)
        timestep: torch.Tensor,         # (B,)
        cond: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        
        # --- 1. Prepare Conditioning Vector ---
        time_embedding = self.time_encoder(timestep)
        scene_embedding = self.state_cls_encoder(cond)
        cond_embedding = torch.cat([time_embedding, scene_embedding], dim=1)
        
        # --- 2. U-Net Forward Pass ---
        # Input shape for Conv1d is (B, C, L), so we need to transpose
        x = noisy_trajectory.permute(0, 2, 1)
        
        x = self.initial_conv(x)
        
        skip_connections = []
        # Downsampling
        for res1, res2, downsample in self.down_blocks:
            x = res1(x, cond_embedding)
            x = res2(x, cond_embedding)
            skip_connections.append(x)
            x = downsample(x)
            
        # Middle
        x = self.middle_block1(x, cond_embedding)
        x = self.middle_block2(x, cond_embedding)
                
        # Upsampling
        for upsample, res1, res2 in self.up_blocks:
            # First, upsample the feature map from the lower level
            x = upsample(x)
            
            # Get the corresponding skip connection (pop from the end)
            skip = skip_connections.pop()
            
            # The defensive crop is now for its intended purpose: handling minor off-by-one errors
            if x.shape[-1] != skip.shape[-1]:
                # This should NOT print with our current config, but is good practice
                print(f"Cropping skip connection from {skip.shape} to {x.shape}")
                diff = skip.shape[-1] - x.shape[-1]
                skip = skip[..., diff//2 : -(diff - diff//2)]

            # Concatenate the upsampled feature map and the skip connection
            x = torch.cat([x, skip], dim=1)
            
            # Process through the residual blocks
            x = res1(x, cond_embedding)
            x = res2(x, cond_embedding)
            
        # Final projection
        x = self.final_conv(x)
        
        # Transpose back to (B, L, C_traj) to match the noise input
        predicted_noise = x.permute(0, 2, 1)
        
        return predicted_noise



# 通过注意力实现条件注入
class AttentionConditionalUNet1D(nn.Module):
    """以 cross-attention 注入场景条件的 1D U-Net 去噪网络。

    与 :class:`ConditionalUNet1D` 的关键区别：
    - 场景不再被压成单个 CLS 向量再走 FiLM，而是由 :class:`StateTokenEncoder` 编码为
      **完整 token 序列**（历史点、道路逐点、限速），U-Net 各残差块通过 cross-attention
      直接 attend 到这些 token，保留道路几何细节，缓解信息瓶颈。
    - FiLM 仅用于注入时间步（diffusion timestep）这类全局标量条件。

    forward 接口与 :class:`ConditionalUNet1D` 保持一致，可直接在 DiffusionPipeline 中替换。
    """

    def __init__(
        self,
        prediction_state_dim: int = 4,
        future_len: int = 25,
        time_embed_dim: int = 128,
        scene_embed_dim: int = 128,
        down_dims: Sequence[int] = (64, 128, 256),
        history_state_dim: int = 7,
        road_feature_dim: int = 2,
        history_len: int = 10,
        road_points: int = 60,
        num_heads: int = 4,
        encoder_layers: int = 2,
        dropout: float = 0.1,
        **_: Any,
    ):
        super().__init__()
        down_dims = [int(d) for d in down_dims]
        if len(down_dims) < 2:
            raise ValueError("`down_dims` must contain at least two channel sizes.")

        # 条件向量仅含时间步 embedding；场景信息走 cross-attention。
        cond_embed_dim = time_embed_dim

        self.time_encoder = nn.Sequential(
            SinusoidalTimeEmbedding(time_embed_dim),
            nn.Linear(time_embed_dim, time_embed_dim), nn.Mish(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        # 场景 token 编码器（输出 token 序列 + padding mask）。
        self.state_token_encoder = StateTokenEncoder(
            history_state_dim=history_state_dim,
            road_feature_dim=road_feature_dim,
            hidden_dim=scene_embed_dim,
            dropout=dropout,
            history_len=history_len,
            road_points=road_points,
            num_heads=num_heads,
            num_layers=encoder_layers,
        )
        context_dim = scene_embed_dim

        def _block(in_c: int, out_c: int) -> CrossAttnConditionedResidualBlock:
            return CrossAttnConditionedResidualBlock(
                in_c, out_c, cond_embed_dim, context_dim,
                num_heads=num_heads, dropout=dropout,
            )

        self.initial_conv = nn.Conv1d(prediction_state_dim, down_dims[0], kernel_size=1)

        # Downsampling path
        self.down_blocks = nn.ModuleList()
        for i in range(len(down_dims) - 1):
            self.down_blocks.append(nn.ModuleList([
                _block(down_dims[i], down_dims[i]),
                _block(down_dims[i], down_dims[i + 1]),
                nn.Conv1d(down_dims[i + 1], down_dims[i + 1], kernel_size=3, stride=2, padding=1),
            ]))

        # Middle block
        self.middle_block1 = _block(down_dims[-1], down_dims[-1])
        self.middle_block2 = _block(down_dims[-1], down_dims[-1])

        # Upsampling path
        self.up_blocks = nn.ModuleList()
        up_dims = down_dims[::-1]
        for i in range(len(up_dims) - 1):
            self.up_blocks.append(nn.ModuleList([
                nn.ConvTranspose1d(up_dims[i], up_dims[i], kernel_size=3, stride=2, padding=1),
                _block(up_dims[i] * 2, up_dims[i + 1]),
                _block(up_dims[i + 1], up_dims[i + 1]),
            ]))

        self.final_conv = nn.Conv1d(down_dims[0], prediction_state_dim, kernel_size=1)

    def forward(
        self,
        noisy_trajectory: torch.Tensor,  # (B, L, C_traj)
        timestep: torch.Tensor,          # (B,)
        cond: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        # --- 1. 条件准备 ---
        cond_embedding = self.time_encoder(timestep)                 # (B, time_embed_dim)
        context, context_mask = self.state_token_encoder(cond)       # (B, S, C), (B, S)

        # --- 2. U-Net ---
        x = noisy_trajectory.permute(0, 2, 1)  # (B, C_traj, L)
        x = self.initial_conv(x)

        skip_connections = []
        for res1, res2, downsample in self.down_blocks:
            x = res1(x, cond_embedding, context, context_mask)
            x = res2(x, cond_embedding, context, context_mask)
            skip_connections.append(x)
            x = downsample(x)

        x = self.middle_block1(x, cond_embedding, context, context_mask)
        x = self.middle_block2(x, cond_embedding, context, context_mask)

        for upsample, res1, res2 in self.up_blocks:
            x = upsample(x)
            skip = skip_connections.pop()
            if x.shape[-1] != skip.shape[-1]:
                diff = skip.shape[-1] - x.shape[-1]
                skip = skip[..., diff // 2: -(diff - diff // 2)]
            x = torch.cat([x, skip], dim=1)
            x = res1(x, cond_embedding, context, context_mask)
            x = res2(x, cond_embedding, context, context_mask)

        x = self.final_conv(x)
        return x.permute(0, 2, 1)  # (B, L, C_traj)
