import os
import argparse
import csv
import random
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import ProgressBarCallback, CallbackList, BaseCallback
from stable_baselines3.common.utils import set_random_seed
from scenes.ainex_soccer.env import AINexReachEnv


def make_env(rank, seed, action_groups_dir=None, action_group_blend=0.5):
    def _init():
        kwargs = {}
        if action_groups_dir is not None:
            kwargs["action_groups_dir"] = action_groups_dir
            kwargs["action_group_blend"] = action_group_blend
        env = AINexReachEnv(**kwargs)
        env.reset(seed=seed + rank)
        return env
    return _init


class EpisodeCSVCallback(BaseCallback):
    def __init__(self, csv_path: str):
        super().__init__()
        self.csv_path = csv_path
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        if not os.path.exists(csv_path):
            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["episode_reward", "episode_length", "success"])

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])
        for done, info in zip(dones, infos):
            if done and isinstance(info, dict):
                ep = info.get("episode")
                if ep:
                    success = bool(info.get("is_success", False))
                    with open(self.csv_path, "a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow([ep.get("r"), ep.get("l"), int(success)])
        return True


def main():
    repo_root = Path(__file__).resolve().parents[3]
    default_ag_dir = repo_root / "assets" / "action_groups" / "csv"

    parser = argparse.ArgumentParser(description="Train PPO for AINex reach task.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--timesteps", type=int, default=1_500_000, help="Training timesteps.")
    parser.add_argument("--num-envs", type=int, default=max(1, os.cpu_count()), help="Parallel envs.")
    parser.add_argument("--action-groups", type=str, default=str(default_ag_dir),
                        help="Path to action group CSVs. Use '' to disable.")
    parser.add_argument("--action-group-blend", type=float, default=0.5,
                        help="Blend factor: 0=policy only, 1=action group only. Default 0.5.")
    args = parser.parse_args()

    set_random_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    action_groups_dir = Path(args.action_groups) if args.action_groups else None
    if action_groups_dir is not None and action_groups_dir.exists():
        print(f"Using action groups from: {action_groups_dir}, blend={args.action_group_blend}")
    else:
        action_groups_dir = None
        print("Training without action groups (policy-only)")

    num_cpu = max(1, args.num_envs)
    print(f"Launching {num_cpu} parallel environments...")

    env = SubprocVecEnv([
        make_env(i, args.seed, action_groups_dir, args.action_group_blend)
        for i in range(num_cpu)
    ])

    log_dir = os.path.join("logs", "ainex_reach")
    os.makedirs(log_dir, exist_ok=True)
    env = VecMonitor(env, filename=os.path.join(log_dir, "monitor.csv"))

    callback = CallbackList([
        ProgressBarCallback(),
        EpisodeCSVCallback(os.path.join(log_dir, "episode_metrics.csv")),
    ])

    policy_kwargs = dict(net_arch=[256, 256])
    device = "cuda" if torch.cuda.is_available() else "auto"
    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        device=device,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        learning_rate=3e-4,
    )

    print(f"Training PPO for {args.timesteps:,} timesteps...")
    model.learn(total_timesteps=args.timesteps, callback=callback)

    save_path = "scenes/ainex_soccer/policies/ppo_ainex_reach"
    model.save(save_path)
    print(f"Model saved as {save_path}.zip")

    env.close()


if __name__ == "__main__":
    main()
