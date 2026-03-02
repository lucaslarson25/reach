# Arm-only reach training (Mac). Run from project root: python -m scenes.arms.training.arm_train_mac
# ARM_ID=arm_2link python -m scenes.arms.training.arm_train_mac

import os
import numpy as np
import torch as th

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
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
        elif isinstance(dones, (bool, np.bool_)):
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


def make_env(arm_id=None, model_path=None):
    def _init():
        return ArmReachEnv(arm_id=arm_id, model_path=model_path)
    return _init


def main():
    arm_id = os.getenv("ARM_ID", "").strip() or None
    model_path = os.getenv("MODEL_PATH", "").strip() or None
    if arm_id:
        print("Using arm from registry:", arm_id)
    if model_path:
        print("Using model path:", model_path)

    cores = os.cpu_count() or 8
    th.set_num_threads(max(1, cores - 1))
    use_mps = os.getenv("USE_MPS", "0") == "1"
    device = "mps" if (use_mps and th.backends.mps.is_available()) else "cpu"
    print("Cores:", cores, "Device:", device)

    env = DummyVecEnv([make_env(arm_id=arm_id, model_path=model_path)])
    policy_kwargs = dict(net_arch=[256, 256])
    model = PPO(
        "MlpPolicy", env, policy_kwargs=policy_kwargs, device=device,
        n_steps=2048, batch_size=128, n_epochs=10, learning_rate=3e-4,
        verbose=1, seed=42,
    )
    callbacks = CallbackList([ProgressBarCallback(), TerminationRatioCallback()])
    total_timesteps = int(os.getenv("TOTAL_STEPS", "300000"))
    print("Training for", total_timesteps, "timesteps...")
    model.learn(total_timesteps=total_timesteps, callback=callbacks)

    os.makedirs("policies", exist_ok=True)
    tag = arm_id or "custom"
    save_path = f"policies/ppo_arms_{tag}_mac_{total_timesteps//1000}k"
    model.save(save_path)
    print("Saved:", save_path + ".zip")
    env.close()


if __name__ == "__main__":
    th.set_default_dtype(th.float32)
    main()
