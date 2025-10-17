import numpy as np
from envs.mujoco_arm_env import Z1ReachEnv
from stable_baselines3 import PPO

# Path to your trained model
model_path = "ppo_z1_parallel.zip"

# Number of evaluation episodes
num_episodes = 10

# Create environment (no rendering needed)
env = Z1ReachEnv(render_mode=None)

# Load trained PPO model
model = PPO.load(model_path, env=env)

for ep in range(num_episodes):
    obs, _ = env.reset()
    done = False
    steps = 0

    while not done and steps < 500:  # optional max steps per episode
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        steps += 1

    # Compute final distance
    ee_pos = env.data.xpos[-1]
    ball_pos = env.data.body("ball").xpos
    final_dist = np.linalg.norm(ee_pos - ball_pos)
    print(f"Episode {ep+1}: final distance = {final_dist:.4f}")

env.close()
