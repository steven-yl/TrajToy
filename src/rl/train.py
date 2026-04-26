"""Hydra RL 训练入口。

用法:
    python -m rl.train                              # 默认配置
    python -m rl.train agent.lr=3e-4 device=cuda    # 覆盖参数
    python -m rl.train render_mode=human             # 带渲染
"""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from sim_env import RoadVehicleEnv, EnvConfig, RoadSegmentType
from rl.agent import PPOAgent

OmegaConf.register_new_resolver("now", lambda fmt: datetime.now().strftime(fmt), replace=True)

def train(cfg: DictConfig) -> None:
    """执行 PPO 训练循环。"""
    env = RoadVehicleEnv.bulid_from_config(cfg.env)
    agent = PPOAgent(env, cfg.agent)

    ac = cfg.agent
    lc = cfg.log
    save_dir = Path(lc.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # TensorBoard
    writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=str(save_dir / "tb_logs"))
    except ImportError:
        pass

    total_steps = 0
    iteration = 0

    pbar = tqdm(total=int(ac.total_timesteps), desc="Training", unit="step")
    while total_steps < ac.total_timesteps:
        iteration += 1

        rollout_stats = agent.collect_rollout()
        total_steps += ac.steps_per_epoch

        update_stats = agent.update()

        if writer:
            for k, v in rollout_stats.items():
                writer.add_scalar(f"rollout/{k}", v, total_steps)
            for k, v in update_stats.items():
                writer.add_scalar(f"loss/{k}", v, total_steps)

        pbar.update(ac.steps_per_epoch)
        pbar.set_postfix(
            mean_steps=rollout_stats["mean_steps"],
            eps=rollout_stats["num_episodes"],
            reward=f"{rollout_stats['mean_reward']:.1f}",
            loss=f"{update_stats['all_loss']:.3f}",
        )

        if iteration % lc.save_interval == 0:
            agent.save(save_dir / f"ppo_{total_steps}.pt")

    pbar.close()
    agent.save(save_dir / "ppo_final.pt")
    if writer:
        writer.close()
    env.close()


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    train(cfg)


if __name__ == "__main__":
    main()
