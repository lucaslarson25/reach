# renders/render_demo_mac.py
#
# Usage (macOS, required):
#   .venv/bin/mjpython renders/render_demo_mac.py --config config/render_run.yaml
#
# This script:
#   - Loads render/config settings from YAML
#   - Dynamically imports the specified env class
#   - Loads the specified PPO policy (.zip)
#   - Uses MuJoCo's passive viewer (required on macOS)
#
# Assumes:
#   - MuJoCo and mjpython are properly installed
#   - You're running from the repo root (so YAML relative paths resolve)
#   - config/render_run.yaml includes:
#       scene.env_class: "package.module:ClassName"
#       scene.model_xml: "path/to/model.xml"
#       policy.path:     "path/to/policy.zip"

import os
import sys
import time
import argparse
import traceback

# Anchor repo root for imports
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import mujoco
import mujoco.viewer
import numpy as np
from stable_baselines3 import PPO

from config.render_loader import load_render_config, import_env


def main():
    parser = argparse.ArgumentParser(description="Scene-agnostic MuJoCo render demo (macOS).")
    parser.add_argument(
        "--config",
        type=str,
        default="config/render_run.yaml",
        help="Path to render YAML config.",
    )
    args = parser.parse_args()

    # Load config
    cfg = load_render_config(args.config)
    scene = cfg["scene"]
    policy_cfg = cfg.get("policy", {})
    run = cfg.get("run", {})

    env_class_path = scene["env_class"]
    model_xml = scene["model_xml"]

    policy_path = policy_cfg.get("path")
    if not policy_path or not os.path.exists(policy_path):
        raise FileNotFoundError(f"Policy file not found at policy.path: {policy_path}")

    episodes = int(run.get("episodes", 10))
    max_seconds = float(run.get("max_seconds_per_ep", 30.0))
    deterministic = bool(run.get("deterministic", True))
    disable_logging = bool(run.get("disable_logging", False))

    # Import environment class
    EnvClass = import_env(env_class_path)

    # Instantiate environment (try model_path with disable_logging; fallback to old signature)
    try:
        env = EnvClass(model_path=model_xml, render_mode="human", disable_logging=disable_logging)
    except TypeError:
        try:
            env = EnvClass(model_path=model_xml, render_mode="human")
        except TypeError:
            # Fallback for envs without model_path arg
            env = EnvClass(render_mode="human")

    # Load PPO model
    model = PPO.load(policy_path, env=env)

    # Launch passive viewer (required on macOS with mjpython)
    if getattr(env, "viewer", None) is None:
        env.viewer = mujoco.viewer.launch_passive(env.model, env.data)

    try:
        for ep in range(episodes):
            obs, _ = env.reset()
            done = False
            ep_reward = 0.0
            start = time.time()

            while not done:
                # Policy action
                action, _ = model.predict(obs, deterministic=deterministic)
                obs, reward, terminated, truncated, info = env.step(action)
                ep_reward += reward

                elapsed = time.time() - start
                if elapsed >= max_seconds:
                    truncated = True

                done = bool(terminated or truncated)

                # Render
                if env.viewer is not None:
                    env.viewer.sync()

                # Throttle to ~120 FPS
                time.sleep(1.0 / 120.0)

            print(f"[Episode {ep + 1}/{episodes}] reward={ep_reward:.2f} time={elapsed:.2f}s")

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    except Exception:
        traceback.print_exc()
    finally:
        env.close()


if __name__ == "__main__":
    main()