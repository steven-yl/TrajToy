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
