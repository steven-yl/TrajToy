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
from il.modules.model.utils.conditione_block import FiLMConditionedResidualBlock
from il.modules.model.utils.state_encoder import StateClsEncoder
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


