import math

import torch
from torch import nn


class SinusoidalTimeEmbedding(nn.Module):
    """
    Standard sinusoidal time embedding module from the DDPM paper, followed by
    an MLP to make it more expressive.
    """
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t is shape [B]
        device = t.device
        half_dim = self.embed_dim // 2
        
        # Standard sinusoidal formula
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = t * embeddings
        embeddings = torch.cat([embeddings.sin(), embeddings.cos()], dim=-1)
        
        return embeddings
