"""Transformer 轨迹预测模型。

输入：
  - 历史状态序列 (B, H+1, state_dim)
  - 中心线 / 左边界 / 右边界 各 (B, N, 2) + mask
  - 车道分隔线 (B, D, N, 2) + mask

输出：
  - 预测未来状态 (B, F, 4)  [x, y, heading, v]
"""

from __future__ import annotations

import torch
import torch.nn as nn

from il.modules.model.utils.state_encoder import PositionalEncoding, StateEncoder
from typing import Dict
from il.modules.model.utils.embedding_block import SinusoidalTimeEmbedding
class ConditionalQueryModel(nn.Module):
    """TrajMlp trajectory predictor model."""

    def __init__(
        self,
        history_state_dim: int,
        road_feature_dim: int,
        hidden_dim: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        num_heads: int,
        dropout: float,
        history_len: int,
        future_len: int,
        road_points: int,
    ) -> None:
        super().__init__()
        hidden = int(hidden_dim)
        dropout = float(dropout)
        # 时间编码器
        self.time_encoder = nn.Sequential(
            SinusoidalTimeEmbedding(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim), nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # 噪声轨迹编码器
        self.noise_traj_encoder = nn.Sequential(
            nn.Linear(4, hidden), nn.Mish(),
            nn.Linear(hidden, hidden)
        )
        self.noise_traj_pe = PositionalEncoding(hidden, max_len=int(future_len), dropout=dropout)

        # 状态编码器
        self.state_encoder = StateEncoder(
            history_state_dim=int(history_state_dim),
            road_feature_dim=int(road_feature_dim),
            hidden_dim=hidden,
            dropout=dropout,
            history_len=int(history_len),
            road_points=int(road_points),
        )

        # Encoder
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=int(num_heads),
            dim_feedforward=hidden * 4, dropout=dropout,
            batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=int(num_encoder_layers))

        # 可学习未来查询
        self.future_queries = nn.Parameter(torch.randn(1, int(future_len), hidden) * 0.02)
        self.future_pe = PositionalEncoding(hidden, max_len=int(future_len), dropout=dropout)

        # Decoder
        dec_layer = nn.TransformerDecoderLayer(
            d_model=hidden, nhead=int(num_heads),
            dim_feedforward=hidden * 4, dropout=dropout,
            batch_first=True, norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=int(num_decoder_layers))

        # 输出头：xy (2), heading (1), velocity (1)
        self.xy_head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 2),
        )
        self.heading_head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 1),
        )
        self.velocity_head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 1),
        )
        self._init_weights()

    # ------------------------------------------------------------------

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self, 
        noisy_trajectory: torch.Tensor, # (B, L, C_traj)
        timestep: torch.Tensor,         # (B,)
        cond: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """前向传播。

        Args:
            history: (B, H+1, state_dim)
            history_mask: (B, H+1)
            centerline / left_boundary / right_boundary: (B, N, 2) + mask (B, N)
            lane_dividers: (B, D, N, 2), lane_dividers_mask: (B, D, N)
            max_v: (B,) 可选，样本级最大速度条件
            max_v_mask: (B,) 可选，max_v 有效掩码

        Returns:
            pred: (B, F, 4)  [x, y, heading, v]
        """
        B = noisy_trajectory.size(0)
        time_embedding = self.time_encoder(timestep)
        noise_traj_embedding = self.noise_traj_pe(self.noise_traj_encoder(noisy_trajectory))

        enc_input, enc_pad_mask = self.state_encoder(cond["history"], cond["history_mask"], cond["centerline"], cond["centerline_mask"], cond["left_boundary"], cond["left_boundary_mask"],
                           cond["right_boundary"], cond["right_boundary_mask"], cond["lane_dividers"], cond["lane_dividers_mask"], cond["max_v"], cond["max_v_mask"])
        enc_input = torch.cat([time_embedding.unsqueeze(1), enc_input], dim=1)
        enc_pad_mask = torch.cat([enc_pad_mask.new_ones(B, 1), enc_pad_mask], dim=1)

        memory = self.encoder(enc_input, src_key_padding_mask=enc_pad_mask)

        queries = noise_traj_embedding #self.future_pe(self.future_queries.expand(B, -1, -1))
        decoded = self.decoder(queries, memory, memory_key_padding_mask=enc_pad_mask)

        xy = self.xy_head(decoded)          # (B, F, 2)
        heading = self.heading_head(decoded)  # (B, F, 1)
        velocity = self.velocity_head(decoded)  # (B, F, 1)

        return torch.cat([xy, heading, velocity], dim=-1)  # (B, F, 4)
