import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import ProgressBarCallback, BaseCallback, CallbackList
from scenes.industrial_arm_reaching.env import Z1ReachEnv
import numpy as np
import tqdm


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

    def _on_rollout_end(self):
        if self.total_episodes > 0:
            ratio = self.terminated_episodes / self.total_episodes
            self.logger.record("custom/terminated_ratio", f"{self.terminated_episodes}/{self.total_episodes} = {(ratio*100):.2f}%")

        self.terminated_episodes = 0
        self.total_episodes = 0


def make_env(rank, arm_id=None, model_path=None):
    def _init():
        return Z1ReachEnv(arm_id=arm_id, model_path=model_path)
    return _init


def main():
    arm_id = os.getenv("ARM_ID", "").strip() or None
    model_path = os.getenv("MODEL_PATH", "").strip() or None
    if arm_id:
        print(f"Using arm from registry: {arm_id}")
    if model_path:
        print(f"Using model path: {model_path}")

    num_cpu = max(1, os.cpu_count())
    print(f"Launching {num_cpu} parallel environments...")
    env = SubprocVecEnv([make_env(i, arm_id=arm_id, model_path=model_path) for i in range(num_cpu)])

    callback = CallbackList([
        ProgressBarCallback(),
        TerminationRatioCallback()
    ])

    # === Define PPO model ===
    policy_kwargs = dict(net_arch=[256, 256])
    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        device="cuda",           # GPU acceleration for CUDA systems
        n_steps=2048,            # number of steps per environment
        batch_size=64,
        n_epochs=10,
        learning_rate=3e-4,
    )

    # === Train with progress bar ===
    total_timesteps = 300_000
    print(f"Training PPO for {total_timesteps:,} timesteps...")
    model.learn(total_timesteps=total_timesteps, callback=callback)

    tag = arm_id or "custom"
    save_path = f"policies/ppo_{tag}_parallel"
    model.save(save_path)
    print(f"Model saved as {save_path}.zip")

    env.close()


if __name__ == "__main__":
    main()