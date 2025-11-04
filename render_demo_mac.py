import time
import mujoco
import mujoco.viewer
from stable_baselines3 import PPO
from envs.mujoco_arm_env import Z1ReachEnv

# run with: .venv/bin/mjpython render_demo_mac.py

# 1) Create env
env = Z1ReachEnv(render_mode="human")

# 2) Load PPO (use your actual .zip file name)
ppo = PPO.load("policies/ppo_z1_parallel_mac_300k.zip", env=env)

# 3) Launch viewer with the MuJoCo model/data (not the PPO!)
if env.render_mode == "human" and env.viewer is None:
    env.viewer = mujoco.viewer.launch_passive(env.model, env.data)

num_episodes = 10
max_episode_seconds = 30.0

try:
    for ep in range(num_episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = 0.0
        start = time.time()

        while not done:
            action, _ = ppo.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            done = terminated or truncated or (time.time() - start >= max_episode_seconds)

            if env.render_mode == "human":
                env.viewer.sync()
            time.sleep(1/120)

        print(f"Episode {ep+1} reward: {ep_reward:.2f}")

finally:
    env.close()