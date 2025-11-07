# arm_train_mac.py
# Fast, Apple-Silicon-friendly PPO training for smaller workloads.
# run with command: .venv/bin/python arm_train_mac.py
# to use with GPU acceleration: USE_MPS=1 .venv/bin/python arm_train_mac.py

import os
import numpy as np
import torch as th

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback, ProgressBarCallback, CallbackList

from scenes.industrial_arm_reaching.env import Z1ReachEnv


# ---------- Small utility callback to track terminated ratio ----------
class TerminationRatioCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.terminated_episodes = 0
        self.total_episodes = 0

    def _on_rollout_start(self) -> None:
        self.terminated_episodes = 0
        self.total_episodes = 0

    def _on_step(self) -> bool:
        dones = self.locals.get("dones", None)
        infos = self.locals.get("infos", None)

        if isinstance(dones, (list, np.ndarray)):
            for done, info in zip(dones, infos or []):
                if done:
                    self.total_episodes += 1
                    if isinstance(info, dict) and info.get("terminated", False):
                        self.terminated_episodes += 1
        elif isinstance(dones, (bool, np.bool_)):
            if dones:
                self.total_episodes += 1
                if isinstance(infos, dict) and infos.get("terminated", False):
                    self.terminated_episodes += 1
        return True

    def _on_rollout_end(self) -> None:
        if self.total_episodes > 0:
            ratio = self.terminated_episodes / self.total_episodes
            self.logger.record(
                "custom/terminated_ratio",
                f"{self.terminated_episodes}/{self.total_episodes} = {(ratio*100):.2f}%"
            )
        self.terminated_episodes = 0
        self.total_episodes = 0


# ---------- Env factory ----------
def make_env():
    def _init():
        env = Z1ReachEnv()
        return env
    return _init


def main():
    # ---- Threads & device tuning for Apple Silicon ----
    cores = os.cpu_count() or 8
    # Leave one core for the OS/UI
    th.set_num_threads(max(1, cores - 1))

    # Default to CPU (often faster for small nets on M-series).
    # Set USE_MPS=1 to try Apple GPU; requires float32 everywhere.
    use_mps = os.getenv("USE_MPS", "0") == "1"
    if use_mps and th.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"Detected cores: {cores}")
    print(f"Using device: {device}")

    # ---- Single-process env is usually fastest on macOS for lightweight models ----
    env = DummyVecEnv([make_env()])  # 1 env, no IPC overhead

    # ---- PPO config tuned for quick iterations on small workloads ----
    # Keep n_steps high (2048) for good on-policy batches even with 1 env.
    # Slightly larger batch can help throughput on CPU.
    policy_kwargs = dict(net_arch=[256, 256])
    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        device=device,
        n_steps=2048,          # per-env rollout length
        batch_size=128,        # divides 2048 cleanly; fewer optimizer steps per update
        n_epochs=10,
        learning_rate=3e-4,
        verbose=1,
        seed=42,
    )

    # ---- Progress bar + custom ratio metric ----
    callbacks = CallbackList([
        ProgressBarCallback(),
        TerminationRatioCallback()
    ])

    # Allow quick override of total steps via env var
    total_timesteps = int(os.getenv("TOTAL_STEPS", "300000"))
    print(f"Training PPO for {total_timesteps:,} timesteps...")

    model.learn(total_timesteps=total_timesteps, callback=callbacks)

    # ---- Save policy ----
    os.makedirs("policies", exist_ok=True)
    save_tag = f"ppo_z1_mac_{total_timesteps//1000}k"
    save_path = f"policies/{save_tag}"
    model.save(save_path)
    print(f"Model saved as {save_path}.zip")

    env.close()


if __name__ == "__main__":
    # Enforce float32 everywhere to keep MPS happy if used
    th.set_default_dtype(th.float32)
    main()