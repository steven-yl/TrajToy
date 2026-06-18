"""Conditional 1D DiT for trajectory denoising (adaLN + scene cross-attention).

与 :class:`ConditionalUNet1D` 保持相同的 ``forward`` 接口，可在
``TrajDiffusionModelWrapper`` 中直接替换 U-Net 去噪网络。
"""

from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn

from il.modules.model.utils.dit import DiT, Mlp
from il.modules.model.utils.state_encoder import StateTokenEncoder as StateEncoder
# from il.modules.model.utils.state_encoder import StateClsEncoder as StateEncoder
from il.modules.model.utils.time_encoder import TimeEncoder
from il.modules.model.utils.state_encoder import PositionalEncoding

class ConditionalDiT1D(nn.Module):
    """时间步 adaLN + 场景 token cross-attention 的 1D DiT 去噪网络。"""
    def __init__(
        self,
        prediction_state_dim: int = 4,
        future_len: int = 25,
        dit_hidden_dim: int = 128,

        history_state_dim: int = 7,
        road_feature_dim: int = 2,
        history_len: int = 10,
        road_points: int = 60,
        dropout: float = 0.1,

        time_embed_dim: int = 128,

        state_encoder_hidden_dim: int = 128,
        state_encoder_num_heads: int = 4,
        state_encoder_layers: int = 2,

        dit_num_heads: int = 4,
        dit_mlp_ratio: float = 4.0,
        dit_depth: int = 8,
        **_: Any,
    ) -> None:
        super().__init__()

        self.time_encoder = TimeEncoder(time_embed_dim, dit_hidden_dim)
        # self.noise_traj_encoder = nn.Sequential(
        #     nn.Linear(prediction_state_dim, state_encoder_hidden_dim),
        #     nn.Mish(),
        #     nn.Linear(state_encoder_hidden_dim, state_encoder_hidden_dim)
        # )
        # self.noise_traj_pe = PositionalEncoding(hidden_dim=state_encoder_hidden_dim, max_len=int(future_len), dropout=dropout)

        self.state_encoder = StateEncoder(
            history_state_dim=history_state_dim,
            road_feature_dim=road_feature_dim,
            hidden_dim=state_encoder_hidden_dim,
            output_dim=dit_hidden_dim,
            dropout=dropout,
            history_len=int(history_len),
            road_points=int(road_points),
            num_heads=state_encoder_num_heads,
            num_layers=state_encoder_layers,
        )

        # self.state_encoder = StateEncoder(
        #     history_state_dim=history_state_dim,
        #     road_feature_dim=road_feature_dim,
        #     hidden_dim=state_encoder_hidden_dim,
        #     dropout=dropout,
        #     history_len=history_len,
        #     road_points=road_points,
        # )

        self.traj_decoder = DiT(
            future_len=future_len,
            depth=dit_depth,
            output_dim=prediction_state_dim,
            hidden_dim=dit_hidden_dim,
            heads=dit_num_heads,
            dropout=dropout,
            mlp_ratio=dit_mlp_ratio,
        )

    def forward(
        self,
        noisy_trajectory: torch.Tensor,
        timestep: torch.Tensor,
        cond: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        # noisy_trajectory: (B, L, C)
        t_feature = self.time_encoder(timestep)
        # noisy_trajectory_feature = self.noise_traj_pe(self.noise_traj_encoder(noisy_trajectory))
        context_feature, context_mask = self.state_encoder(cond)
        return self.traj_decoder(
            noisy_trajectory,
            t_feature,
            context_feature,
            context_mask,
        )
