import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import ProgressBarCallback, BaseCallback, CallbackList
from envs.mujoco_arm_env import Z1ReachEnv
import numpy as np
import tqdm



class TerminationRatioCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.terminated_episodes = 0
        self.total_episodes = 0

    def _on_rollout_start(self) -> None:
        # reset counters for this rollout
        self.terminated_episodes = 0
        self.total_episodes = 0

    def _on_step(self) -> bool:
        # SB3 locals typically contain 'dones' and 'infos'
        dones = self.locals.get("dones", None)
        infos = self.locals.get("infos", None)

        # Handle both vectorized and non-vectorized cases
        if isinstance(dones, (list, np.ndarray)):
            # Vec env: 'dones' aligns with 'infos' list
            for done, info in zip(dones, infos or []):
                if done:
                    self.total_episodes += 1
                    if isinstance(info, dict) and info.get("terminated", False):
                        self.terminated_episodes += 1
        elif isinstance(dones, (bool, np.bool_)):
            # Non-vec env: single bool and single info dict
            if dones:
                self.total_episodes += 1
                if isinstance(infos, dict) and infos.get("terminated", False):
                    self.terminated_episodes += 1

        return True

    def _on_rollout_end(self):
        if self.total_episodes > 0:
            ratio = self.terminated_episodes / self.total_episodes
            # This adds the value to SB3's logger
            self.logger.record("custom/terminated_ratio", f"{self.terminated_episodes}/{self.total_episodes} = {(ratio*100):.2f}%")

        # reset counts each rollout
        self.terminated_episodes = 0
        self.total_episodes = 0

def make_env(rank):
    """
    Utility function to create a new environment instance.
    Each environment runs in its own process.
    """
    def _init():
        env = Z1ReachEnv()
        return env
    return _init


def main():
    # === Auto-detect CPU cores ===
    num_cpu = max(1, os.cpu_count())  # half of available cores for safety
    print(f"Launching {num_cpu} parallel environments...")

    # === Create vectorized environments ===
    env = SubprocVecEnv([make_env(i) for i in range(num_cpu)])

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
        device="cuda",           # GPU acceleration
        n_steps=2048,            # per environment
        batch_size=64,
        n_epochs=10,
        learning_rate=3e-4,
    )

    # === Train with progress bar ===
    total_timesteps = 1_500_000
    print(f"Training PPO for {total_timesteps:,} timesteps...")
    model.learn(total_timesteps=total_timesteps, callback=callback)

    # === Save model ===
    save_path = "ppo_z1_parallel"
    model.save(save_path)
    print(f"Model saved as {save_path}.zip")

    env.close()


if __name__ == "__main__":
    main()