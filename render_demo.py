import time
import numpy as np
from envs.mujoco_arm_env import Z1ReachEnv
from stable_baselines3 import PPO
import mujoco.viewer


# Create environment with human-render mode
env = Z1ReachEnv(render_mode="human")

# Load trained PPO model (update path if needed)
model = PPO.load("policies/ppo_z1_parallel_1.5m_best", env=env)

num_episodes = 20
max_episode_duration = 30.0  # seconds

try:
    # Launch a blocking viewer for real-time visualization
    if env.render_mode == "human" and env.viewer is None:
        env.viewer = mujoco.viewer.launch(env.model, env.data)

    for ep in range(num_episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = 0.0
        start_time = time.time()

        while not done:
            # Predict action from trained model
            action, _ = model.predict(obs, deterministic=True)

            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward

            # Check time limits
            elapsed_time = time.time() - start_time
            if elapsed_time >= max_episode_duration:
                truncated = True

            done = terminated or truncated

            # Render the frame (real-time visualization)
            if env.render_mode == "human":
                env.viewer.sync()

            # Limit frame rate to about 120 FPS for smooth playback
            time.sleep(1 / 120)

        print(f"Episode {ep + 1} finished in {elapsed_time:.2f}s with total reward {ep_reward:.2f}")

finally:
    env.close()