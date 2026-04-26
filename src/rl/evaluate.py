"""RL 模型评估脚本。

用法:
    python -m rl.evaluate log.save_dir=log/rl_train_xxx render_mode=human
"""

from __future__ import annotations

import numpy as np
import hydra
from omegaconf import DictConfig

from rl.agent import PPOAgent
from rl.train import _build_env


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    cfg.render_mode = cfg.get("render_mode", "human") or "human"
    env = _build_env(cfg)
    agent = PPOAgent(env, cfg)

    ckpt_path = f"{cfg.log.save_dir}/ppo_final.pt"
    agent.load(ckpt_path)
    print(f"模型加载: {ckpt_path}")

    obs, info = env.reset(seed=0)
    ep_reward = 0.0
    steps = 0

    for _ in range(cfg.env.max_steps):
        action = agent.predict(obs, deterministic=True, info=info)
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        ep_reward += reward
        steps += 1
        if terminated or truncated:
            break

    print(f"reward={ep_reward:.1f}, steps={steps}, "
          f"progress={info.get('progress', 0):.1%}")
    env.close()


if __name__ == "__main__":
    main()
