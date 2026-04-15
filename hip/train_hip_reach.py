#!/usr/bin/env python3
"""Short PPO run: learn xArm controls that pull the welded hand toward the ball.

Run from repo root (needs stable-baselines3 + torch from requirements.txt):

  python3 -m hip.train_hip_reach
  python3 -m hip.train_hip_reach --steps 80000 --success-bonus 4

Then roll out with the viewer (macOS: mjpython):

  .venv/bin/mjpython -m hip.run_policy --model policies/hip_reach_ppo.zip
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, ProgressBarCallback
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv

from hip.env import HipReachEnv


class _TerminationRatioCallback(BaseCallback):
    """Log share of episodes that ended in success (terminated) vs timeout."""

    def __init__(self) -> None:
        super().__init__()
        self._term = 0
        self._total = 0

    def _on_rollout_start(self) -> bool:
        self._term = 0
        self._total = 0
        return True

    def _on_step(self) -> bool:
        dones = self.locals.get("dones")
        infos = self.locals.get("infos")
        if dones is None:
            return True
        if isinstance(dones, (list, np.ndarray)):
            for i, done in enumerate(dones):
                if not done:
                    continue
                self._total += 1
                inf = infos[i] if infos is not None and i < len(infos) else {}
                if isinstance(inf, (list, tuple)) and inf:
                    inf = inf[0]
                if isinstance(inf, dict) and inf.get("terminated"):
                    self._term += 1
        return True

    def _on_rollout_end(self) -> bool:
        if self._total > 0:
            self.logger.record("custom/success_episodes", float(self._term))
            self.logger.record("custom/done_episodes", float(self._total))
        return True


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PPO on HipReachEnv (xArm + limp Ainex + ball).")
    p.add_argument("--steps", type=int, default=50_000, help="Total PPO timesteps (default 50k).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--success-bonus",
        type=float,
        default=4.0,
        help="Extra reward when gripper tip is within success distance (default 4).",
    )
    p.add_argument(
        "--save",
        type=str,
        default="",
        help="Path prefix for policy .zip (default: policies/hip_reach_ppo_<steps>).",
    )
    p.add_argument(
        "--tensorboard",
        type=str,
        default="",
        help="If set, log to this directory (e.g. hip_runs/).",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    cores = os.cpu_count() or 8
    th.set_num_threads(max(1, cores - 1))
    use_mps = os.getenv("USE_MPS", "0") == "1"
    if use_mps and th.backends.mps.is_available():
        device: str = "mps"
    else:
        device = "cpu"
    print(f"device={device}, timesteps={args.steps:,}, success_bonus={args.success_bonus}")

    bonus = float(args.success_bonus)
    set_random_seed(args.seed)

    def make_env() -> HipReachEnv:
        return HipReachEnv(success_bonus=bonus, progress_coef=2.0)

    env = DummyVecEnv([make_env])

    tb = args.tensorboard.strip() or None
    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=dict(net_arch=[256, 256]),
        device=device,
        n_steps=2048,
        batch_size=128,
        n_epochs=10,
        learning_rate=3e-4,
        ent_coef=0.02,
        verbose=1,
        seed=args.seed,
        tensorboard_log=tb,
    )
    callbacks = CallbackList([ProgressBarCallback(), _TerminationRatioCallback()])
    model.learn(total_timesteps=int(args.steps), callback=callbacks)

    repo = Path(__file__).resolve().parent.parent
    policies_dir = repo / "policies"
    policies_dir.mkdir(parents=True, exist_ok=True)
    if args.save.strip():
        save_path = Path(args.save.strip())
        if not save_path.is_absolute():
            save_path = repo / save_path
        save_path = save_path.with_suffix("")
    else:
        k = max(1, int(args.steps) // 1000)
        save_path = policies_dir / f"hip_reach_ppo_{k}k"
    model.save(str(save_path))
    print(f"Saved policy: {save_path}.zip")
    env.close()


if __name__ == "__main__":
    th.set_default_dtype(th.float32)
    main()
