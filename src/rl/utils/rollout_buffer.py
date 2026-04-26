"""Rollout Buffer：存储轨迹数据并计算 GAE。"""

from __future__ import annotations

import numpy as np
import torch


class RolloutBuffer:
    """分别存储车辆状态和道路信息两部分观测。"""

    def __init__(
        self, capacity: int, vehicle_dim: int, road_dim: int, act_dim: int, device: str,
    ) -> None:
        self.capacity = capacity
        self.device = device
        self.vehicle_obs = np.zeros((capacity, vehicle_dim), dtype=np.float32)
        self.road_obs = np.zeros((capacity, road_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, act_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.log_probs = np.zeros(capacity, dtype=np.float32)
        self.values = np.zeros(capacity, dtype=np.float32)
        self.advantages = np.zeros(capacity, dtype=np.float32)
        self.returns = np.zeros(capacity, dtype=np.float32)
        self.ptr = 0

    def store(
        self, vehicle_obs: np.ndarray, road_obs: np.ndarray,
        action: np.ndarray, reward: float, done: bool,
        log_prob: float, value: float,
    ) -> None:
        i = self.ptr
        self.vehicle_obs[i] = vehicle_obs
        self.road_obs[i] = road_obs
        self.actions[i] = action
        self.rewards[i] = reward
        self.dones[i] = float(done)
        self.log_probs[i] = log_prob
        self.values[i] = value
        self.ptr += 1

    def compute_gae(self, last_value: float, gamma: float, lam: float) -> None:
        n = self.ptr
        last_gae = 0.0
        for t in reversed(range(n)):
            next_value = last_value if t == n - 1 else self.values[t + 1]
            next_non_terminal = 1.0 - self.dones[t]
            delta = self.rewards[t] + gamma * next_value * next_non_terminal - self.values[t]
            last_gae = delta + gamma * lam * next_non_terminal * last_gae
            self.advantages[t] = last_gae
        self.returns[:n] = self.advantages[:n] + self.values[:n]

    def get_batches(self, batch_size: int):
        n = self.ptr
        indices = np.random.permutation(n)
        for start in range(0, n, batch_size):
            idx = indices[start:min(start + batch_size, n)]
            yield {
                "vehicle_obs": torch.as_tensor(self.vehicle_obs[idx], device=self.device),
                "road_obs": torch.as_tensor(self.road_obs[idx], device=self.device),
                "actions": torch.as_tensor(self.actions[idx], device=self.device),
                "log_probs": torch.as_tensor(self.log_probs[idx], device=self.device),
                "advantages": torch.as_tensor(self.advantages[idx], device=self.device),
                "returns": torch.as_tensor(self.returns[idx], device=self.device),
            }

    def reset(self) -> None:
        self.ptr = 0
