# Arm-only reach training (parallel envs). Run from project root.
# ARM_ID=arm_2link python -m scenes.arms.training.arm_train

import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback, ProgressBarCallback, CallbackList

from scenes.arms.env import ArmReachEnv


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
            self.logger.record("custom/terminated_ratio",
                f"{self.terminated_episodes}/{self.total_episodes} = {(ratio*100):.2f}%")
        self.terminated_episodes = 0
        self.total_episodes = 0


def make_env(rank, arm_id=None, model_path=None):
    def _init():
        return ArmReachEnv(arm_id=arm_id, model_path=model_path)
    return _init


def main():
    arm_id = os.getenv("ARM_ID", "").strip() or None
    model_path = os.getenv("MODEL_PATH", "").strip() or None
    if arm_id:
        print("Using arm:", arm_id)
    if model_path:
        print("Using model path:", model_path)
    num_cpu = max(1, os.cpu_count())
    print("Parallel envs:", num_cpu)
    env = SubprocVecEnv([make_env(i, arm_id=arm_id, model_path=model_path) for i in range(num_cpu)])
    # PPO defaults aligned with config/arms.yaml (no config file loaded here)
    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=dict(net_arch=[256, 256]),
        device="cuda",
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        learning_rate=3e-4,
        clip_range=0.2,
        gamma=0.99,
        gae_lambda=0.95,
        vf_coef=0.5,
        ent_coef=0.01,
        max_grad_norm=0.5,
        verbose=1,
        seed=42,
    )
    total_timesteps = 300_000
    model.learn(total_timesteps=total_timesteps, callback=CallbackList([ProgressBarCallback(), TerminationRatioCallback()]))
    os.makedirs("policies", exist_ok=True)
    tag = arm_id or "custom"
    save_path = f"policies/ppo_arms_{tag}_parallel"
    model.save(save_path)
    print("Saved:", save_path + ".zip")
    env.close()


if __name__ == "__main__":
    main()
