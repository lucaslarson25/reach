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
        elif isinstance(dones, (bool, getattr(np, "bool_", bool))):
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
def make_env(arm_id=None, model_path=None):
    def _init():
        env = Z1ReachEnv(arm_id=arm_id, model_path=model_path)
        return env
    return _init


def main():
    # Arm selection: ARM_ID env var (e.g. z1, arm_2link) or MODEL_PATH for custom scene XML
    arm_id = os.getenv("ARM_ID", "").strip() or None
    model_path = os.getenv("MODEL_PATH", "").strip() or None
    if arm_id:
        print(f"Using arm from registry: {arm_id}")
    if model_path:
        print(f"Using model path: {model_path}")

    # ---- Threads & device tuning for Apple Silicon ----
    cores = os.cpu_count() or 8
    th.set_num_threads(max(1, cores - 1))
    use_mps = os.getenv("USE_MPS", "0") == "1"
    if use_mps and th.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Detected cores: {cores}, device: {device}")

    env = DummyVecEnv([make_env(arm_id=arm_id, model_path=model_path)])

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

    os.makedirs("policies", exist_ok=True)
    tag = arm_id or "custom"
    save_tag = f"ppo_{tag}_mac_{total_timesteps//1000}k"
    save_path = f"policies/{save_tag}"
    model.save(save_path)
    print(f"Model saved as {save_path}.zip")

    env.close()


if __name__ == "__main__":
    # Enforce float32 everywhere to keep MPS happy if used
    th.set_default_dtype(th.float32)
    main()