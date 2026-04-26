"""双编码器 Actor-Critic 网络。"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal


class ActorCritic(nn.Module):
    """车辆状态和道路信息分别编码，融合后输出策略和价值。"""

    def __init__(
        self, vehicle_dim: int, road_dim: int, act_dim: int, hidden_sizes: list[int],
    ) -> None:
        super().__init__()
        fusion_dim = hidden_sizes[-1] if hidden_sizes else 64

        def _build_encoder(in_dim: int) -> nn.Sequential:
            layers: list[nn.Module] = []
            d = in_dim
            for h in hidden_sizes:
                layers += [nn.Linear(d, h), nn.Tanh()]
                d = h
            return nn.Sequential(*layers)

        self.vehicle_encoder = _build_encoder(vehicle_dim)
        self.road_encoder = _build_encoder(road_dim)
        self.fusion = nn.Sequential(nn.Linear(fusion_dim * 2, fusion_dim), nn.Tanh())

        self.mean_head = nn.Linear(fusion_dim, act_dim)
        self.log_std = nn.Parameter(torch.zeros(act_dim))
        self.value_head = nn.Linear(fusion_dim, 1)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.mean_head.weight, gain=0.01)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)

    def forward(
        self, vehicle_obs: torch.Tensor, road_obs: torch.Tensor,
    ) -> tuple[Normal, torch.Tensor]:
        veh_feat = self.vehicle_encoder(vehicle_obs)
        road_feat = self.road_encoder(road_obs)
        fused = self.fusion(torch.cat([veh_feat, road_feat], dim=-1))
        mean = self.mean_head(fused)
        std = self.log_std.exp().expand_as(mean)
        return Normal(mean, std), self.value_head(fused).squeeze(-1)

    def get_action_and_value(
        self, vehicle_obs: torch.Tensor, road_obs: torch.Tensor,
        action: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """返回 (action, log_prob, entropy, value)。"""
        dist, value = self(vehicle_obs, road_obs)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)
        return action, log_prob, entropy, value
