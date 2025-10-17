import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import ProgressBarCallback
from envs.mujoco_arm_env import Z1ReachEnv
import tqdm


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
    num_cpu = max(1, os.cpu_count() // 2)  # half of available cores for safety
    print(f"Launching {num_cpu} parallel environments...")

    # === Create vectorized environments ===
    env = SubprocVecEnv([make_env(i) for i in range(num_cpu)])

    # === Define PPO model ===
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        device="cuda",           # GPU acceleration
        n_steps=2048,            # per environment
        batch_size=64,
        n_epochs=10,
        learning_rate=3e-4,
    )

    # === Train with progress bar ===
    total_timesteps = 1_000_000
    print(f"Training PPO for {total_timesteps:,} timesteps...")
    model.learn(total_timesteps=total_timesteps, callback=ProgressBarCallback())

    # === Save model ===
    save_path = "ppo_z1_parallel"
    model.save(save_path)
    print(f"Model saved as {save_path}.zip")

    env.close()


if __name__ == "__main__":
    main()