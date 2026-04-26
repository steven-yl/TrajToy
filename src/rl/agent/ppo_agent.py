"""PPO Agent：封装网络、优化器、采集和训练逻辑。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig

from rl.utils.obs_utils import normal_and_flatten_obs, get_obs_dims
from rl.model.actor_critic import ActorCritic
from rl.utils.rollout_buffer import RolloutBuffer

@dataclass
class PPOAgentConfig:
    obs_keys: list[str] = field(
        default_factory=lambda: [
            "vehicle",
            "centerline",
            "left_boundary",
            "right_boundary",
            "lane_dividers",
        ],
    )
    total_timesteps: int = 1_000_000
    steps_per_epoch: int = 2048
    num_epochs: int = 30
    batch_size: int = 256
    lr: float = 1e-4
    gamma: float = 0.95
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    vf_coef: float = 0.0001
    ent_coef: float = 0.001
    max_grad_norm: float = 0.5
    hidden_sizes: list[int] = field(default_factory=lambda: [128, 128])



class PPOAgent:
    """PPO 智能体。"""

    def __init__(self, env, cfg: PPOAgentConfig, device: str='cpu') -> None:
        self.cfg = cfg
        self.env = env
        self.device = torch.device(device)

        obs_keys = list(self.cfg.obs_keys)
        vehicle_dim, road_dim = get_obs_dims(env.observation_space, obs_keys)
        act_dim = int(np.prod(env.action_space.shape))

        self.actor_critic = ActorCritic(
            vehicle_dim, road_dim, act_dim, list(self.cfg.hidden_sizes),
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.actor_critic.parameters(), lr=self.cfg.lr, eps=1e-5)
        self.buffer = RolloutBuffer(
            self.cfg.steps_per_epoch, vehicle_dim, road_dim, act_dim, self.device,
        )

    # ── 采集 ──────────────────────────────────────────────────────
    @property
    def config(self) -> PPOAgentConfig:
        return self.cfg

    @torch.no_grad()
    def collect_rollout(self) -> dict:
        obs_keys = list(self.cfg.obs_keys)

        obs_dict, reset_info = self.env.reset(
            seed=0, random_reset=self.env.config.random_init,
        )
        veh_obs, road_obs = normal_and_flatten_obs(obs_dict, obs_keys, reset_info)

        ep_steps, ep_rewards = [], []
        ep_ret, ep_len = 0.0, 0

        reward_accum = {}
        reward_lists = {}

        self.buffer.reset()

        for _ in range(self.cfg.steps_per_epoch):
            veh_t = torch.as_tensor(veh_obs, device=self.device).unsqueeze(0)
            road_t = torch.as_tensor(road_obs, device=self.device).unsqueeze(0)
            action, log_prob, _, value = self.actor_critic.get_action_and_value(veh_t, road_t)

            act_np = action.squeeze(0).cpu().numpy()
            act_np = np.clip(act_np, self.env.action_space.low, self.env.action_space.high)

            next_obs_dict, reward, terminated, truncated, info = self.env.step(act_np)
            done = terminated or truncated

            self.buffer.store(
                veh_obs, road_obs, act_np, reward, done,
                log_prob.item(), value.item(),
            )
            ep_ret += reward
            ep_len += 1

            # 累积各奖励分量
            for k, v in info.get("reward_components", {}).items():
                reward_accum[k] = reward_accum.get(k, 0.0) + v

            if done:
                ep_steps.append(ep_len)
                ep_rewards.append(ep_ret / ep_len)
                for k, v in reward_accum.items():
                    reward_lists.setdefault(k, []).append(v / ep_len)
                ep_ret, ep_len = 0.0, 0
                reward_accum = {}
                next_obs_dict, reset_info = self.env.reset()
                info = reset_info

            veh_obs, road_obs = normal_and_flatten_obs(next_obs_dict, obs_keys, info)

        veh_t = torch.as_tensor(veh_obs, device=self.device).unsqueeze(0)
        road_t = torch.as_tensor(road_obs, device=self.device).unsqueeze(0)
        _, last_value = self.actor_critic(veh_t, road_t)
        self.buffer.compute_gae(last_value.item(), self.cfg.gamma, self.cfg.gae_lambda)

        stats = {
            "mean_reward": float(np.mean(ep_rewards)) if ep_rewards else 0.0,
            "max_reward": float(np.max(ep_rewards)) if ep_rewards else 0.0,
            "min_reward": float(np.min(ep_rewards)) if ep_rewards else 0.0,
            "num_episodes": len(ep_rewards),
            "mean_steps": float(np.mean(ep_steps)) if ep_steps else 0.0,
        }
        for k, v_list in reward_lists.items():
            stats[f"{k}_mean"] = float(np.mean(v_list))
        return stats

    # ── 优化 ──────────────────────────────────────────────────────

    def update(self) -> dict:
        all_loss, all_pg, all_vf, all_ent = [], [], [], []

        for _ in range(self.cfg.num_epochs):
            for batch in self.buffer.get_batches(self.cfg.batch_size):
                _, new_lp, entropy, new_val = self.actor_critic.get_action_and_value(
                    batch["vehicle_obs"], batch["road_obs"], batch["actions"],
                )
                adv = batch["advantages"]
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)

                ratio = (new_lp - batch["log_probs"]).exp()
                pg1 = adv * ratio
                pg2 = adv * torch.clamp(ratio, 1 - self.cfg.clip_eps, 1 + self.cfg.clip_eps)
                pg_loss = -torch.min(pg1, pg2).mean()

                vf_loss = 0.5 * (new_val - batch["returns"]).pow(2).mean()
                ent_loss = entropy.mean()

                loss = pg_loss + self.cfg.vf_coef * vf_loss - self.cfg.ent_coef * ent_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.cfg.max_grad_norm)
                self.optimizer.step()

                all_loss.append(loss.item())
                all_pg.append(pg_loss.item())
                all_vf.append(vf_loss.item())
                all_ent.append(ent_loss.item())

        return {
            "all_loss": float(np.mean(all_loss)),
            "pg_loss": float(np.mean(all_pg)),
            "vf_loss": float(np.mean(all_vf)),
            "entropy": float(np.mean(all_ent)),
        }

    # ── 推理 ──────────────────────────────────────────────────────

    @torch.no_grad()
    def predict(
        self, obs_dict: dict[str, np.ndarray],
        deterministic: bool = False, info: dict | None = None,
    ) -> np.ndarray:
        obs_keys = list(self.cfg.obs_keys)
        veh_obs, road_obs = normal_and_flatten_obs(obs_dict, obs_keys, info)
        veh_t = torch.as_tensor(veh_obs, device=self.device).unsqueeze(0)
        road_t = torch.as_tensor(road_obs, device=self.device).unsqueeze(0)
        dist, _ = self.actor_critic(veh_t, road_t)
        action = dist.mean if deterministic else dist.sample()
        act_np = action.squeeze(0).cpu().numpy()
        return np.clip(act_np, self.env.action_space.low, self.env.action_space.high)

    # ── 保存 / 加载 ──────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "ac_state_dict": self.actor_critic.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }, path)

    def load(self, path: str | Path) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.actor_critic.load_state_dict(ckpt["ac_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
