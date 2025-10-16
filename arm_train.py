from envs.mujoco_arm_env import ReacherEnv
from stable_baselines3 import PPO

env = ReacherEnv()
model = PPO("MlpPolicy", env, verbose=1, device="cuda")
model.learn(total_timesteps=1_000)
model.save("ppo_reacher")
env.close()