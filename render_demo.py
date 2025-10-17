import time
import numpy as np
from envs.mujoco_arm_env import Z1ReachEnv
from stable_baselines3 import PPO

# Create environment with human-render mode
env = Z1ReachEnv(render_mode="human")

# Load trained PPO model
model = PPO.load("ppo_reacher", env=env)

num_episodes = 3

for ep in range(num_episodes):
    obs, _ = env.reset()
    done = False
    ep_reward = 0.0

    while not done:
        # Predict action from model
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        ep_reward += reward

        # Render one frame in the human viewer
        env.render()

        # Limit frame rate to ~60 FPS
        time.sleep(1 / 60)

    print(f"Episode {ep + 1} finished with total reward {ep_reward:.2f}")

# Clean up
env.close()
