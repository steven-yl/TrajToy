import torch
from torch import nn


class FiLMConditionedResidualBlock(nn.Module):
    """
    A residual block for the 1D U-Net that is conditioned via FiLM
    (Feature-wise Linear Modulation).
    """
    def __init__(self, in_channels: int, out_channels: int, cond_embed_dim: int):
        super().__init__()
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        
        # If the number of channels changes, we need a simple 1x1 conv for the residual connection
        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        
        # The "FiLM Generator" is a small MLP that takes the conditioning embedding
        # and predicts the scale (gamma) and shift (beta) parameters.
        self.film_generator = nn.Linear(cond_embed_dim, out_channels * 2)
        
        self.activation = nn.Mish()

    def forward(self, x: torch.Tensor, cond_embedding: torch.Tensor) -> torch.Tensor:
        # x is shape [B, C_in, L]
        # cond_embedding is shape [B, D_cond]
        
        # Generate FiLM parameters
        film_params = self.film_generator(cond_embedding)
        gamma, beta = torch.chunk(film_params, 2, dim=-1) # Split into two tensors
        # Reshape gamma and beta to be broadcastable with x for the FiLM operation
        gamma = gamma.unsqueeze(-1) # -> [B, C_out, 1]
        beta = beta.unsqueeze(-1)   # -> [B, C_out, 1]

        # Main path
        h = self.conv1(x)
        h = self.norm1(h)
        
        # Apply FiLM modulation
        h = h * gamma + beta
        
        h = self.activation(h)
        h = self.conv2(h)
        h = self.norm2(h)
        
        # Add residual connection and return
        return self.activation(h + self.residual_conv(x))


class CrossAttnConditionedResidualBlock(nn.Module):
    """1D U-Net 残差块：FiLM 注入时间步条件 + cross-attention 注入场景 token 序列。

    - FiLM：用 ``cond_embedding``（通常是时间步 embedding）调制每个通道的 scale/shift，
      负责注入「当前去噪到第几步」这类全局标量条件。
    - Cross-attention：把轨迹特征（按时间步展开为 query）与场景 memory（历史点、道路逐点 token）
      做注意力，使每个轨迹位置能直接读取与之相关的道路几何，替代 CLS 单向量的信息瓶颈。
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_embed_dim: int,
        context_dim: int,
        num_heads: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)

        self.residual_conv = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

        self.norm1 = nn.GroupNorm(8, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)

        # FiLM 生成器：由条件 embedding 预测每通道的 gamma/beta。
        self.film_generator = nn.Linear(cond_embed_dim, out_channels * 2)

        # Cross-attention：query 来自轨迹特征，key/value 来自场景 memory。
        self.attn_norm = nn.GroupNorm(8, out_channels)
        self.context_proj = nn.Linear(context_dim, out_channels)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=out_channels,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.activation = nn.Mish()

    def forward(
        self,
        x: torch.Tensor,
        cond_embedding: torch.Tensor,
        context: torch.Tensor,
        context_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # x: [B, C_in, L]; cond_embedding: [B, D_cond];
        # context: [B, S, context_dim]; context_key_padding_mask: [B, S] (True=pad)

        # FiLM 参数
        film_params = self.film_generator(cond_embedding)
        gamma, beta = torch.chunk(film_params, 2, dim=-1)
        gamma = gamma.unsqueeze(-1)  # [B, C_out, 1]
        beta = beta.unsqueeze(-1)

        # 主路径 + FiLM 调制
        h = self.conv1(x)
        h = self.norm1(h)
        h = h * gamma + beta
        h = self.activation(h)
        h = self.conv2(h)
        h = self.norm2(h)

        # Cross-attention：把 [B, C, L] 转成 [B, L, C] 作为 query
        attn_in = self.attn_norm(h)
        q = attn_in.permute(0, 2, 1)  # [B, L, C_out]
        kv = self.context_proj(context)  # [B, S, C_out]
        attn_out, _ = self.cross_attn(
            q, kv, kv, key_padding_mask=context_key_padding_mask, need_weights=False
        )
        h = h + attn_out.permute(0, 2, 1)  # 残差注入，回到 [B, C_out, L]

        return self.activation(h + self.residual_conv(x))
