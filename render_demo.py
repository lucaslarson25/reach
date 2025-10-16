import time
import mujoco
import numpy as np
from envs.mujoco_arm_env import ReacherEnv
from stable_baselines3 import PPO
import mujoco.viewer

# Create environment without auto-wrappers
env = ReacherEnv(render_mode="human")

# Load trained PPO model
model = PPO.load("ppo_reacher", env=env)

# Launch MuJoCo’s built-in viewer
viewer = mujoco.viewer.launch(env.model, env.data)

# Run a few episodes with rendering
for ep in range(3):
    obs, _ = env.reset()
    done = False
    ep_reward = 0

    while not done:
        # Use trained policy to get action
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        ep_reward += reward

        # Render each frame
        viewer.sync()
        time.sleep(0.01)  # optional slowdown for visibility

    print(f"Episode {ep + 1} finished with total reward {ep_reward:.2f}")

viewer.close()
env.close()
