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


class TrajMlpModel(nn.Module):
    """TrajMlp trajectory predictor model."""

    def __init__(
        self,
        cfg_history_state_dim: int,
        cfg_road_feature_dim: int,
        cfg_hidden_dim: int,
        cfg_num_encoder_layers: int,
        cfg_num_decoder_layers: int,
        cfg_num_heads: int,
        cfg_dropout: float,
        history_len: int,
        future_len: int,
        road_points: int,
    ) -> None:
        super().__init__()
        hidden = int(cfg_hidden_dim)
        dropout = float(cfg_dropout)

        # 状态编码器
        self.state_encoder = StateEncoder(
            cfg_history_state_dim=int(cfg_history_state_dim),
            cfg_road_feature_dim=int(cfg_road_feature_dim),
            cfg_hidden_dim=hidden,
            cfg_dropout=dropout,
            history_len=int(history_len),
            road_points=int(road_points),
        )

        # Encoder
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=int(cfg_num_heads),
            dim_feedforward=hidden * 4, dropout=dropout,
            batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=int(cfg_num_encoder_layers))

        # 可学习未来查询
        self.future_queries = nn.Parameter(torch.randn(1, int(future_len), hidden) * 0.02)
        self.future_pe = PositionalEncoding(hidden, max_len=int(future_len), dropout=dropout)

        # Decoder
        dec_layer = nn.TransformerDecoderLayer(
            d_model=hidden, nhead=int(cfg_num_heads),
            dim_feedforward=hidden * 4, dropout=dropout,
            batch_first=True, norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=int(cfg_num_decoder_layers))

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
        history: torch.Tensor,
        history_mask: torch.Tensor,
        centerline: torch.Tensor,
        centerline_mask: torch.Tensor,
        left_boundary: torch.Tensor,
        left_boundary_mask: torch.Tensor,
        right_boundary: torch.Tensor,
        right_boundary_mask: torch.Tensor,
        lane_dividers: torch.Tensor,
        lane_dividers_mask: torch.Tensor,
        max_v: torch.Tensor | None = None,
        max_v_mask: torch.Tensor | None = None,
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
        B = history.size(0)
        enc_input, enc_pad_mask = self.state_encoder(history, history_mask, centerline, centerline_mask, left_boundary, left_boundary_mask,
                           right_boundary, right_boundary_mask, lane_dividers, lane_dividers_mask, max_v, max_v_mask)

        memory = self.encoder(enc_input, src_key_padding_mask=enc_pad_mask)

        queries = self.future_pe(self.future_queries.expand(B, -1, -1))
        decoded = self.decoder(queries, memory, memory_key_padding_mask=enc_pad_mask)

        xy = self.xy_head(decoded)          # (B, F, 2)
        heading = self.heading_head(decoded)  # (B, F, 1)
        velocity = self.velocity_head(decoded)  # (B, F, 1)

        return torch.cat([xy, heading, velocity], dim=-1)  # (B, F, 4)
