import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import ProgressBarCallback, CallbackList
from scenes.ainex_soccer.env import AINexReachEnv


def make_env(rank):
    def _init():
        return AINexReachEnv()
    return _init


def main():
    num_cpu = max(1, os.cpu_count())
    print(f"Launching {num_cpu} parallel environments...")

    env = SubprocVecEnv([make_env(i) for i in range(num_cpu)])
    callback = CallbackList([ProgressBarCallback()])

    policy_kwargs = dict(net_arch=[256, 256])
    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        device="cuda",
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        learning_rate=3e-4,
    )

    total_timesteps = 1_500_000
    print(f"Training PPO for {total_timesteps:,} timesteps...")
    model.learn(total_timesteps=total_timesteps, callback=callback)

    save_path = "scenes/ainex_soccer/policies/ppo_ainex_reach"
    model.save(save_path)
    print(f"Model saved as {save_path}.zip")

    env.close()


if __name__ == "__main__":
    main()
