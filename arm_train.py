from envs.mujoco_arm_env import Z1ReachEnv
from stable_baselines3 import PPO

env = Z1ReachEnv()
model = PPO("MlpPolicy", env, verbose=1, device="cuda")
model.learn(total_timesteps=200_000)
model.save("ppo_reacher")
env.close()