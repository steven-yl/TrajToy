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

from omegaconf import DictConfig

import math


"""正弦位置编码。"""
class PositionalEncoding(nn.Module):
    """固定正弦位置编码 + Dropout。"""

    def __init__(self, hidden_dim: int, max_len: int = 512, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, hidden_dim, 2).float() * (-math.log(10000.0) / hidden_dim)
        )
        pe = torch.zeros(max_len, hidden_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, D)"""
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TrajectoryPredictor(nn.Module):
    """Transformer 编码器-解码器轨迹预测模型。"""

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        mc = cfg.model
        dc = cfg.data
        hidden = mc.hidden_dim

        # 输入投影
        self.history_proj = nn.Linear(mc.history_state_dim, hidden)
        self.centerline_proj = nn.Linear(mc.road_feature_dim, hidden)
        self.left_boundary_proj = nn.Linear(mc.road_feature_dim, hidden)
        self.right_boundary_proj = nn.Linear(mc.road_feature_dim, hidden)
        self.lane_dividers_proj = nn.Linear(mc.road_feature_dim, hidden)
        self.speed_proj = nn.Linear(2, hidden)

        # 位置编码
        self.history_pe = PositionalEncoding(hidden, max_len=dc.history_len + 1, dropout=mc.dropout)
        self.road_pe = PositionalEncoding(hidden, max_len=dc.road_points, dropout=mc.dropout)

        # 类型嵌入 (0=centerline, 1=left, 2=right, 3=divider, 4=speed_token)
        self.road_type_emb = nn.Embedding(5, hidden)

        # Encoder
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=mc.num_heads,
            dim_feedforward=hidden * 4, dropout=mc.dropout,
            batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=mc.num_encoder_layers)

        # 可学习未来查询
        self.future_queries = nn.Parameter(torch.randn(1, dc.future_len, hidden) * 0.02)
        self.future_pe = PositionalEncoding(hidden, max_len=dc.future_len, dropout=mc.dropout)

        # Decoder
        dec_layer = nn.TransformerDecoderLayer(
            d_model=hidden, nhead=mc.num_heads,
            dim_feedforward=hidden * 4, dropout=mc.dropout,
            batch_first=True, norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=mc.num_decoder_layers)

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
        device = history.device

        history_emb = self.history_pe(self.history_proj(history))

        cl_emb = self.road_pe(self.centerline_proj(centerline)) + self.road_type_emb(
            torch.zeros(1, dtype=torch.long, device=device))
        lb_emb = self.road_pe(self.left_boundary_proj(left_boundary)) + self.road_type_emb(
            torch.ones(1, dtype=torch.long, device=device))
        rb_emb = self.road_pe(self.right_boundary_proj(right_boundary)) + self.road_type_emb(
            torch.full((1,), 2, dtype=torch.long, device=device))

        # 车道分隔线 (B, D, N, 2) → (B, D*N, hidden)
        D = lane_dividers.size(1)
        N = lane_dividers.size(2)
        ld_flat = lane_dividers.reshape(B, D * N, 2)
        ld_emb = self.lane_dividers_proj(ld_flat).reshape(B, D, N, -1)
        for d in range(D):
            ld_emb[:, d] = self.road_pe(ld_emb[:, d])
        ld_emb = ld_emb.reshape(B, D * N, -1) + self.road_type_emb(
            torch.full((1,), 3, dtype=torch.long, device=device))

        if max_v is None:
            max_v_in = torch.zeros(B, 1, device=device, dtype=history.dtype)
        else:
            max_v_in = max_v.to(device=device, dtype=history.dtype).view(B, 1)
        if max_v_mask is None:
            max_v_mask_in = torch.zeros(B, 1, device=device, dtype=history.dtype)
        else:
            max_v_mask_in = max_v_mask.to(device=device, dtype=history.dtype).view(B, 1)
        speed_token = torch.cat([max_v_in, max_v_mask_in], dim=-1).unsqueeze(1)  # (B, 1, 2)
        speed_emb = self.speed_proj(speed_token) + self.road_type_emb(
            torch.full((1,), 4, dtype=torch.long, device=device))

        # 拼接
        enc_input = torch.cat([history_emb, cl_emb, lb_emb, rb_emb, ld_emb, speed_emb], dim=1)
        enc_mask = torch.cat([
            history_mask, centerline_mask, left_boundary_mask,
            right_boundary_mask, lane_dividers_mask.reshape(B, D * N),
            torch.ones(B, 1, device=device, dtype=history_mask.dtype),
        ], dim=1)
        enc_pad = (enc_mask == 0)

        memory = self.encoder(enc_input, src_key_padding_mask=enc_pad)

        queries = self.future_pe(self.future_queries.expand(B, -1, -1))
        decoded = self.decoder(queries, memory, memory_key_padding_mask=enc_pad)

        xy = self.xy_head(decoded)          # (B, F, 2)
        heading = self.heading_head(decoded)  # (B, F, 1)
        velocity = self.velocity_head(decoded)  # (B, F, 1)

        return torch.cat([xy, heading, velocity], dim=-1)  # (B, F, 4)
