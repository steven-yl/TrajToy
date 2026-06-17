import torch
import torch.nn as nn
from il.modules.model.utils.embedding_block import SinusoidalTimeEmbedding

class TimeEncoder(nn.Module):
    def __init__(self, time_embed_dim: int, output_dim: int):
        super().__init__()
        self.time_embed_dim = time_embed_dim
        self.time_encoder = nn.Sequential(
            SinusoidalTimeEmbedding(time_embed_dim),
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.Mish(),
            nn.Linear(time_embed_dim, output_dim),
        )
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.time_encoder(t)