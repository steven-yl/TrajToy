"""Conditional DiT (Diffusion Transformer) for 1D trajectory denoising.

与 :class:`ConditionalUNet1D` 保持相同的 ``forward`` 接口，可直接在
``TrajDiffusionModelWrapper`` 中替换 U-Net 去噪网络。

实现参考 ``model_dit.py`` 的 adaLN-Zero DiT block；条件注入方式与
``conditional_UNet1D_model.py`` 对齐：
- :class:`ConditionalDiT1D`：时间步 + 场景 CLS 向量拼接后投影为全局调制向量 ``y``。
- :class:`AttentionConditionalDiT1D`：时间步走 adaLN；场景 token 序列走 cross-attention。
"""

from __future__ import annotations

import math
from typing import Any, Dict

import torch
import torch.nn as nn
from einops import rearrange

from il.modules.model.diffusion.model import Attention
from il.modules.model.diffusion.model_dit import (
    CondSequential,
    DiTBlock,
    ModulatedLayerNorm,
    Modulation,
)
from il.modules.model.utils.embedding_block import SinusoidalTimeEmbedding
from il.modules.model.utils.state_encoder import StateClsEncoder, StateTokenEncoder


class PatchEmbed1D(nn.Module):
    """将 (B, C, L) 轨迹按 patch 切分并投影为 token 序列 (B, N, D)。"""

    def __init__(self, patch_size: int, channels: int, embed_dim: int, bias: bool = True):
        super().__init__()
        self.patch_size = int(patch_size)
        self.proj = nn.Conv1d(
            channels,
            embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=bias,
        )
        w = self.proj.weight.data
        nn.init.xavier_uniform_(w.view(w.shape[0], -1))
        if bias:
            nn.init.constant_(self.proj.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return rearrange(self.proj(x), "b c n -> b n c")


def get_pos_embed_1d(num_patches: int, dim: int) -> nn.Parameter:
    """固定正弦 1D 位置编码，形状 (1, num_patches, dim)。"""
    if dim % 2 != 0:
        raise ValueError("`dim` must be even for sinusoidal position embedding.")
    position = torch.arange(num_patches, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim)
    )
    pe = torch.zeros(num_patches, dim)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return nn.Parameter(pe.unsqueeze(0), requires_grad=False)


def unpatchify_1d(
    x: torch.Tensor,
    patch_size: int,
    channels: int,
    seq_len: int,
) -> torch.Tensor:
    """(B, N, patch_size * C) -> (B, L, C)。"""
    return rearrange(
        x,
        "b n (p c) -> b (n p) c",
        p=patch_size,
        c=channels,
        n=seq_len // patch_size,
    )


class CrossAttnDiTBlock(nn.Module):
    """DiT block + cross-attention 注入场景 token（时间步仍通过 adaLN 调制）。"""

    def __init__(
        self,
        head_dim: int,
        num_heads: int,
        context_dim: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        dim = head_dim * num_heads
        mlp_hidden_dim = int(dim * mlp_ratio)

        self.norm1 = ModulatedLayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.self_attn = Attention(head_dim, num_heads=num_heads, qkv_bias=True)
        self.norm_cross = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.context_proj = nn.Linear(context_dim, dim)
        self.cross_attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True,
        )
        self.norm2 = ModulatedLayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, dim, bias=True),
        )
        self.scale_modulation = Modulation(dim, 3)

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        gate_sa, gate_ca, gate_mlp = self.scale_modulation(y)
        x = x + gate_sa * self.self_attn(self.norm1(x, y))
        ctx = self.context_proj(context)
        x = x + gate_ca * self.cross_attn(
            self.norm_cross(x),
            ctx,
            ctx,
            key_padding_mask=context_mask,
        )[0]
        x = x + gate_mlp * self.mlp(self.norm2(x, y))
        return x


class CrossAttnCondSequential(nn.Sequential):
    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        for module in self._modules.values():
            x = module(x, y, context, context_mask)
        return x


class ConditionalDiT1D(nn.Module):
    """场景 CLS + 时间步全局调制（adaLN）的 1D DiT 去噪网络。"""

    def __init__(
        self,
        prediction_state_dim: int = 4,
        future_len: int = 25,
        time_embed_dim: int = 128,
        scene_embed_dim: int = 128,
        history_state_dim: int = 7,
        road_feature_dim: int = 2,
        history_len: int = 10,
        road_points: int = 60,
        dropout: float = 0.1,
        depth: int = 8,
        head_dim: int = 64,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        patch_size: int = 1,
        **_: Any,
    ):
        super().__init__()
        self.prediction_state_dim = int(prediction_state_dim)
        self.future_len = int(future_len)
        self.patch_size = int(patch_size)
        if self.future_len % self.patch_size != 0:
            raise ValueError(
                f"`future_len` ({self.future_len}) must be divisible by `patch_size` "
                f"({self.patch_size})."
            )

        dim = int(head_dim) * int(num_heads)
        num_patches = self.future_len // self.patch_size

        self.time_encoder = nn.Sequential(
            SinusoidalTimeEmbedding(time_embed_dim),
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.Mish(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        self.state_cls_encoder = StateClsEncoder(
            history_state_dim=history_state_dim,
            road_feature_dim=road_feature_dim,
            hidden_dim=scene_embed_dim,
            dropout=dropout,
            history_len=history_len,
            road_points=road_points,
        )
        self.cond_proj = nn.Sequential(
            nn.Linear(time_embed_dim + scene_embed_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

        self.pos_embed = get_pos_embed_1d(num_patches, dim)
        self.x_embed = PatchEmbed1D(
            self.patch_size, self.prediction_state_dim, dim,
        )
        self.blocks = CondSequential(*[
            DiTBlock(int(head_dim), int(num_heads), mlp_ratio=mlp_ratio)
            for _ in range(int(depth))
        ])
        self.final_norm = ModulatedLayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.final_linear = nn.Linear(dim, self.patch_size * self.prediction_state_dim)
        self._init_weights()

    def _init_weights(self) -> None:
        def _basic_init(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)
        nn.init.constant_(self.final_linear.weight, 0)
        nn.init.constant_(self.final_linear.bias, 0)

    def forward(
        self,
        noisy_trajectory: torch.Tensor,
        timestep: torch.Tensor,
        cond: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        # noisy_trajectory: (B, L, C)
        x = noisy_trajectory.permute(0, 2, 1)  # (B, C, L)
        time_embedding = self.time_encoder(timestep)
        scene_embedding = self.state_cls_encoder(cond)
        y = self.cond_proj(torch.cat([time_embedding, scene_embedding], dim=1))

        x = self.x_embed(x) + self.pos_embed
        x = self.blocks(x, y)
        x = self.final_linear(self.final_norm(x, y))
        x = unpatchify_1d(x, self.patch_size, self.prediction_state_dim, self.future_len)
        return x


class AttentionConditionalDiT1D(nn.Module):
    """时间步 adaLN + 场景 token cross-attention 的 1D DiT 去噪网络。"""

    def __init__(
        self,
        prediction_state_dim: int = 4,
        future_len: int = 25,
        time_embed_dim: int = 128,
        scene_embed_dim: int = 128,
        history_state_dim: int = 7,
        road_feature_dim: int = 2,
        history_len: int = 10,
        road_points: int = 60,
        num_heads: int = 4,
        encoder_layers: int = 2,
        dropout: float = 0.1,
        depth: int = 8,
        head_dim: int = 64,
        mlp_ratio: float = 4.0,
        patch_size: int = 1,
        **_: Any,
    ):
        super().__init__()
        self.prediction_state_dim = int(prediction_state_dim)
        self.future_len = int(future_len)
        self.patch_size = int(patch_size)
        if self.future_len % self.patch_size != 0:
            raise ValueError(
                f"`future_len` ({self.future_len}) must be divisible by `patch_size` "
                f"({self.patch_size})."
            )

        dim = int(head_dim) * int(num_heads)
        num_patches = self.future_len // self.patch_size

        self.time_encoder = nn.Sequential(
            SinusoidalTimeEmbedding(time_embed_dim),
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.Mish(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        self.time_proj = nn.Sequential(
            nn.Linear(time_embed_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
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

        self.pos_embed = get_pos_embed_1d(num_patches, dim)
        self.x_embed = PatchEmbed1D(
            self.patch_size, self.prediction_state_dim, dim,
        )
        self.blocks = CrossAttnCondSequential(*[
            CrossAttnDiTBlock(
                int(head_dim), int(num_heads), context_dim,
                mlp_ratio=mlp_ratio, dropout=dropout,
            )
            for _ in range(int(depth))
        ])
        self.final_norm = ModulatedLayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.final_linear = nn.Linear(dim, self.patch_size * self.prediction_state_dim)
        self._init_weights()

    def _init_weights(self) -> None:
        def _basic_init(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)
        nn.init.constant_(self.final_linear.weight, 0)
        nn.init.constant_(self.final_linear.bias, 0)

    def forward(
        self,
        noisy_trajectory: torch.Tensor,
        timestep: torch.Tensor,
        cond: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        x = noisy_trajectory.permute(0, 2, 1)
        y = self.time_proj(self.time_encoder(timestep))
        context, context_mask = self.state_token_encoder(cond)

        x = self.x_embed(x) + self.pos_embed
        x = self.blocks(x, y, context, context_mask)
        x = self.final_linear(self.final_norm(x, y))
        x = unpatchify_1d(x, self.patch_size, self.prediction_state_dim, self.future_len)
        return x
