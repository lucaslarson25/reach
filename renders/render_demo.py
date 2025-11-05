import time
import numpy as np
from stable_baselines3 import PPO
import mujoco.viewer

# updated import path for the env in your new structure
from scenes.industrial_arm_reaching.env import Z1ReachEnv


# Create environment with human-render mode
env = Z1ReachEnv(render_mode="human")
# renders/render_demo.py
# Minimal changes from original:
# - accepts --model and --policy flags
# - imports env from your new scenes path, but falls back to old path if needed
# - passes model_path to the env if supported; otherwise falls back gracefully

import time
import argparse
import os
import sys
import mujoco
import mujoco.viewer
import numpy as np
from stable_baselines3 import PPO

# --- keep imports as close to original as possible, but support new structure ---
try:
    # preferred: new scene-based env
    from scenes.industrial_arm_reaching.env import Z1ReachEnv  # type: ignore
except Exception:
    # fallback: old path (for legacy compatibility)
    from envs.mujoco_arm_env import Z1ReachEnv  # type: ignore


def main():
    # Defaults mirror your current repo layout
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    default_model = os.path.join(
        repo_root, "scenes", "industrial_arm_reaching", "models", "z1scene.xml"
    )
    default_policy = os.path.join(
        repo_root, "scenes", "industrial_arm_reaching", "policies", "ppo_z1_parallel_1.5m_best.zip"
    )

    parser = argparse.ArgumentParser(description="Render trained PPO policy (x86/CUDA friendly).")
    parser.add_argument("--model", type=str, default=default_model, help="Path to MuJoCo XML model.")
    parser.add_argument("--policy", type=str, default=default_policy, help="Path to PPO .zip file.")
    parser.add_argument("--episodes", type=int, default=20, help="Number of episodes.")
    parser.add_argument("--max-seconds", type=float, default=30.0, help="Max seconds per episode.")
    parser.add_argument("--deterministic", action="store_true", help="Deterministic policy actions.")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model XML not found: {args.model}")
    if not os.path.exists(args.policy):
        raise FileNotFoundError(f"Policy file not found: {args.policy}")

    # --- Create environment with human-render mode (try to pass model_path if supported) ---
    try:
        env = Z1ReachEnv(model_path=args.model, render_mode="human")
    except TypeError:
        # older constructor without model_path parameter
        env = Z1ReachEnv(render_mode="human")

    # --- Load trained PPO model (x86/CUDA fine) ---
    model = PPO.load(args.policy, env=env)

    # --- Launch blocking viewer (works on Windows/Linux; use mjpython+launch_passive for macOS) ---
    if env.render_mode == "human" and getattr(env, "viewer", None) is None:
        env.viewer = mujoco.viewer.launch(env.model, env.data)

    try:
        for ep in range(args.episodes):
            obs, _ = env.reset()
            done = False
            ep_reward = 0.0
            start_time = time.time()

            while not done:
                action, _ = model.predict(obs, deterministic=args.deterministic)
                obs, reward, terminated, truncated, info = env.step(action)
                ep_reward += reward

                elapsed_time = time.time() - start_time
                if elapsed_time >= args.max_seconds:
                    truncated = True

                done = terminated or truncated

                if env.render_mode == "human" and env.viewer is not None:
                    env.viewer.sync()

                # ~120 FPS like original
                time.sleep(1 / 120)

            print(f"Episode {ep + 1} finished in {elapsed_time:.2f}s with total reward {ep_reward:.2f}")

    finally:
        env.close()


if __name__ == "__main__":
    # Ensure repo root is on sys.path (keeps imports working when run as a script)
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    main()
# Load trained PPO model (updated path; include .zip)
model = PPO.load(
    "scenes/industrial_arm_reaching/policies/ppo_z1_parallel_1.5m_best.zip",
    env=env
)

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