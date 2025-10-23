import time
import numpy as np
from envs.mujoco_arm_env import Z1ReachEnv
from stable_baselines3 import PPO
import mujoco.viewer

# Create environment with human-render mode
env = Z1ReachEnv(render_mode="human")

# Load trained PPO model
model = PPO.load("ppo_z1_parallel", env=env)

num_episodes = 20
max_episode_duration = 30.0  # seconds

try:
    # Launch a non-blocking viewer
    if env.render_mode == "human" and env.viewer is None:
        env.viewer = mujoco.viewer.launch_passive(env.model, env.data)

    for ep in range(num_episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = 0.0
        start_time = time.time()  # track episode start

        while not done:
            # Predict action from model
            action, _ = model.predict(obs, deterministic=True)

            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward

            # Check for timeout
            elapsed_time = time.time() - start_time
            if elapsed_time >= max_episode_duration:
                truncated = True

            done = terminated or truncated

            # Render frame
            if env.render_mode == "human":
                env.viewer.sync()

            # Limit framerate to ~60 FPS
            time.sleep(1/60)

        print(f"Episode {ep + 1} finished in {elapsed_time:.2f}s with total reward {ep_reward:.2f}")
        print("Episode complete")

finally:
    # Clean up viewer
    env.close()
