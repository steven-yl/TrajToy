"""道路与历史状态的多路投影 + 位置编码 + 类型嵌入，供 Transformer 编码器使用。"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
from typing import Dict

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
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)

class StateEncoder(nn.Module):
    """将历史轨迹与道路几何编码为同维度的 token 序列及 padding mask。"""

    def __init__(
        self,
        history_state_dim: int,
        road_feature_dim: int,
        hidden_dim: int,
        dropout: float,
        history_len: int,
        road_points: int,
    ) -> None:
        super().__init__()
        hidden = int(hidden_dim)

        self.history_proj = nn.Linear(int(history_state_dim), hidden)
        self.centerline_proj = nn.Linear(int(road_feature_dim), hidden)
        self.left_boundary_proj = nn.Linear(int(road_feature_dim), hidden)
        self.right_boundary_proj = nn.Linear(int(road_feature_dim), hidden)
        self.lane_dividers_proj = nn.Linear(int(road_feature_dim), hidden)
        self.speed_proj = nn.Linear(2, hidden)

        self.history_pe = PositionalEncoding(hidden, max_len=int(history_len) + 1, dropout=float(dropout))
        self.road_pe = PositionalEncoding(hidden, max_len=int(road_points), dropout=float(dropout))

        # 类型嵌入 (0=centerline, 1=left, 2=right, 3=divider, 4=speed_token)
        self.road_type_emb = nn.Embedding(5, hidden)

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
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B = history.size(0)
        device = history.device
        dtype = history.dtype
        n_road = centerline.size(1)

        history_emb = self.history_pe(self.history_proj(history))

        cl_emb = self.road_pe(self.centerline_proj(centerline)) + self.road_type_emb(
            torch.zeros(B, n_road, dtype=torch.long, device=device)
        )
        lb_emb = self.road_pe(self.left_boundary_proj(left_boundary)) + self.road_type_emb(
            torch.ones(B, n_road, dtype=torch.long, device=device)
        )
        rb_emb = self.road_pe(self.right_boundary_proj(right_boundary)) + self.road_type_emb(
            torch.full((B, n_road), 2, dtype=torch.long, device=device)
        )

        D = lane_dividers.size(1)
        N = lane_dividers.size(2)
        ld_flat = lane_dividers.reshape(B, D * N, -1)
        ld_emb = self.lane_dividers_proj(ld_flat).reshape(B, D, N, -1)
        for d in range(D):
            ld_emb[:, d] = self.road_pe(ld_emb[:, d])
        ld_emb = ld_emb.reshape(B, D * N, -1) + self.road_type_emb(
            torch.full((B, D * N), 3, dtype=torch.long, device=device)
        )

        speed_token = torch.stack([max_v, max_v_mask], dim=-1)  # (B, n_road, 2)
        speed_emb = self.speed_proj(speed_token) + self.road_type_emb(
            torch.full((B, n_road), 4, dtype=torch.long, device=device)
        )

        enc_input = torch.cat([history_emb, cl_emb, lb_emb, rb_emb, ld_emb, speed_emb], dim=1)
        enc_mask = torch.cat(
            [
                history_mask,
                centerline_mask,
                left_boundary_mask,
                right_boundary_mask,
                lane_dividers_mask.reshape(B, D * N),
                max_v_mask,
            ],
            dim=1,
        )
        enc_pad_mask = enc_mask == 0
        return enc_input, enc_pad_mask



class StateClsEncoder(nn.Module):
    """将历史轨迹与道路几何编码为同维度的 token 序列及 padding mask。"""

    def __init__(
        self,
        history_state_dim: int,
        road_feature_dim: int,
        hidden_dim: int,
        dropout: float,
        history_len: int,
        road_points: int,
    ) -> None:
        super().__init__()
        hidden = int(hidden_dim)

        self.history_proj = nn.Linear(int(history_state_dim), hidden)
        self.centerline_proj = nn.Linear(int(road_feature_dim), hidden)
        self.left_boundary_proj = nn.Linear(int(road_feature_dim), hidden)
        self.right_boundary_proj = nn.Linear(int(road_feature_dim), hidden)
        self.lane_dividers_proj = nn.Linear(int(road_feature_dim), hidden)
        self.speed_proj = nn.Linear(2, hidden)

        self.history_pe = PositionalEncoding(hidden, max_len=int(history_len) + 1, dropout=float(dropout))
        self.road_pe = PositionalEncoding(hidden, max_len=int(road_points), dropout=float(dropout))

        # 类型嵌入 (0=centerline, 1=left, 2=right, 3=divider, 4=speed_token)
        self.road_type_emb = nn.Embedding(5, hidden)

        # --- The [CLS] token, a learnable parameter ---
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden))

         # --- Transformer Encoder for fusing all entity embeddings ---
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=4,
            dim_feedforward=hidden * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(transformer_layer, num_layers=2)

    def forward(
        self,
        state_dict: Dict[str, torch.Tensor]) ->torch.Tensor:
        history = state_dict["history"]
        history_mask = state_dict["history_mask"]
        centerline = state_dict["centerline"]
        centerline_mask = state_dict["centerline_mask"]
        left_boundary = state_dict["left_boundary"]
        left_boundary_mask = state_dict["left_boundary_mask"]
        right_boundary = state_dict["right_boundary"]
        right_boundary_mask = state_dict["right_boundary_mask"]
        lane_dividers = state_dict["lane_dividers"]
        lane_dividers_mask = state_dict["lane_dividers_mask"]
        max_v = state_dict["max_v"]
        max_v_mask = state_dict["max_v_mask"]

        B = history.shape[0]
        device = history.device
        n_road = centerline.shape[1]

        history_emb = self.history_pe(self.history_proj(history))

        cl_emb = self.road_pe(self.centerline_proj(centerline)) + self.road_type_emb(
            torch.zeros(B, n_road, dtype=torch.long, device=device)
        )
        lb_emb = self.road_pe(self.left_boundary_proj(left_boundary)) + self.road_type_emb(
            torch.ones(B, n_road, dtype=torch.long, device=device)
        )
        rb_emb = self.road_pe(self.right_boundary_proj(right_boundary)) + self.road_type_emb(
            torch.full((B, n_road), 2, dtype=torch.long, device=device)
        )

        D = lane_dividers.size(1)
        N = lane_dividers.size(2)
        ld_flat = lane_dividers.reshape(B, D * N, -1)
        ld_emb = self.lane_dividers_proj(ld_flat).reshape(B, D, N, -1)
        for d in range(D):
            ld_emb[:, d] = self.road_pe(ld_emb[:, d])
        ld_emb = ld_emb.reshape(B, D * N, -1) + self.road_type_emb(
            torch.full((B, D * N), 3, dtype=torch.long, device=device)
        )

        speed_token = torch.stack([max_v, max_v_mask], dim=-1)  # (B, n_road, 2)
        speed_emb = self.speed_proj(speed_token) + self.road_type_emb(
            torch.full((B, n_road), 4, dtype=torch.long, device=device)
        )

        # Prepend the [CLS] token
        cls_token = self.cls_token.expand(B, -1, -1)
        cls_mask = torch.ones(B, 1, device=device, dtype=torch.bool)

        full_enc_input = torch.cat([cls_token, history_emb, cl_emb, lb_emb, rb_emb, ld_emb, speed_emb], dim=1)
        enc_mask = torch.cat(
            [   cls_mask,
                history_mask,
                centerline_mask,
                left_boundary_mask,
                right_boundary_mask,
                lane_dividers_mask.reshape(B, D * N),
                max_v_mask,
            ],
            dim=1,
        )
        # True = padding position（与 nn.TransformerEncoder / model_mlp.StateEncoder 约定一致）
        full_mask = enc_mask == 0

        # --- 4. Pass through the Transformer ---
        transformer_output = self.transformer(src=full_enc_input, src_key_padding_mask=full_mask)

        # --- 5. Extract the [CLS] token's output ---
        # The [CLS] token is the first token in the sequence (index 0).
        # Its final hidden state is our holistic scene embedding.
        scene_embedding = transformer_output[:, 0, :]
        
        return scene_embedding


class StateTokenEncoder(nn.Module):
    """将历史轨迹与道路几何编码为 **完整 token 序列**（含 padding mask），供 cross-attention 使用。

    与 :class:`StateClsEncoder` 的区别：后者用一个 [CLS] token 把整个场景压成单个向量
    （信息瓶颈），本类则保留每条 token（历史点、中心线/边界/分隔线逐点、限速）的逐 token 表征，
    让下游 U-Net 通过 cross-attention 直接 attend 到道路几何细节。

    Returns (forward):
        memory: (B, S, hidden) 融合后的场景 token 序列。
        key_padding_mask: (B, S) bool，``True`` 表示该位置为 padding（与
            ``nn.MultiheadAttention`` / ``nn.TransformerEncoder`` 约定一致）。
    """

    def __init__(
        self,
        history_state_dim: int,
        road_feature_dim: int,
        hidden_dim: int,
        dropout: float,
        history_len: int,
        road_points: int,
        num_heads: int = 4,
        num_layers: int = 2,
        output_dim: [int, None] = None,
    ) -> None:
        super().__init__()
        hidden = int(hidden_dim)

        self.history_proj = nn.Linear(int(history_state_dim), hidden)
        self.centerline_proj = nn.Linear(int(road_feature_dim), hidden)
        self.left_boundary_proj = nn.Linear(int(road_feature_dim), hidden)
        self.right_boundary_proj = nn.Linear(int(road_feature_dim), hidden)
        self.lane_dividers_proj = nn.Linear(int(road_feature_dim), hidden)
        self.speed_proj = nn.Linear(2, hidden)

        self.history_pe = PositionalEncoding(hidden, max_len=int(history_len) + 1, dropout=float(dropout))
        self.road_pe = PositionalEncoding(hidden, max_len=int(road_points), dropout=float(dropout))

        # 类型嵌入 (0=centerline, 1=left, 2=right, 3=divider, 4=speed_token, 5=history)
        self.type_emb = nn.Embedding(6, hidden)

        transformer_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=int(num_heads),
            dim_feedforward=hidden * 4,
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(transformer_layer, num_layers=int(num_layers))
        if output_dim is not None:
            self.traj_proj = nn.Linear(int(hidden), output_dim)
        else:
            self.traj_proj = None

    def forward(
        self, state_dict: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        history = state_dict["history"]
        history_mask = state_dict["history_mask"]
        centerline = state_dict["centerline"]
        centerline_mask = state_dict["centerline_mask"]
        left_boundary = state_dict["left_boundary"]
        left_boundary_mask = state_dict["left_boundary_mask"]
        right_boundary = state_dict["right_boundary"]
        right_boundary_mask = state_dict["right_boundary_mask"]
        lane_dividers = state_dict["lane_dividers"]
        lane_dividers_mask = state_dict["lane_dividers_mask"]
        max_v = state_dict["max_v"]
        max_v_mask = state_dict["max_v_mask"]

        B = history.shape[0]
        device = history.device
        n_road = centerline.shape[1]

        history_emb = self.history_pe(self.history_proj(history)) + self.type_emb(
            torch.full((B, history.shape[1]), 5, dtype=torch.long, device=device)
        )

        cl_emb = self.road_pe(self.centerline_proj(centerline)) + self.type_emb(
            torch.zeros(B, n_road, dtype=torch.long, device=device)
        )
        lb_emb = self.road_pe(self.left_boundary_proj(left_boundary)) + self.type_emb(
            torch.ones(B, n_road, dtype=torch.long, device=device)
        )
        rb_emb = self.road_pe(self.right_boundary_proj(right_boundary)) + self.type_emb(
            torch.full((B, n_road), 2, dtype=torch.long, device=device)
        )

        D = lane_dividers.size(1)
        N = lane_dividers.size(2)
        ld_flat = lane_dividers.reshape(B, D * N, -1)
        ld_emb = self.lane_dividers_proj(ld_flat).reshape(B, D, N, -1)
        for d in range(D):
            ld_emb[:, d] = self.road_pe(ld_emb[:, d])
        ld_emb = ld_emb.reshape(B, D * N, -1) + self.type_emb(
            torch.full((B, D * N), 3, dtype=torch.long, device=device)
        )

        speed_token = torch.stack([max_v, max_v_mask], dim=-1)  # (B, n_road, 2)
        speed_emb = self.speed_proj(speed_token) + self.type_emb(
            torch.full((B, n_road), 4, dtype=torch.long, device=device)
        )

        full_enc_input = torch.cat(
            [history_emb, cl_emb, lb_emb, rb_emb, ld_emb, speed_emb], dim=1
        )
        enc_mask = torch.cat(
            [
                history_mask,
                centerline_mask,
                left_boundary_mask,
                right_boundary_mask,
                lane_dividers_mask.reshape(B, D * N),
                max_v_mask,
            ],
            dim=1,
        )
        # True = padding position
        key_padding_mask = enc_mask == 0

        memory = self.transformer(src=full_enc_input, src_key_padding_mask=key_padding_mask)
        
        if self.traj_proj is not None:
            memory = self.traj_proj(memory)
        return memory, key_padding_mask
