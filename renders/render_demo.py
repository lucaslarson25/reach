# renders/render_demo.py
#
# Usage (x86 / CUDA / standard Python):
#   .venv/bin/python renders/render_demo.py --config config/render_run.yaml
#
# This script:
#   - Loads render/config settings from YAML
#   - Dynamically imports the specified env class
#   - Loads the specified PPO policy (.zip)
#   - Uses MuJoCo's standard blocking viewer (works on Linux/Windows/x86)
#
# Assumes:
#   - MuJoCo is installed and configured
#   - You're running from repo root (so YAML relative paths resolve)
#   - config/render_run.yaml has:
#       scene.env_class: "package.module:ClassName"
#       scene.model_xml: "path/to/model.xml"
#       policy.path:     "path/to/policy.zip"

import os
import sys
import time
import argparse
import traceback
import csv
from datetime import datetime

# Make repo root importable
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import mujoco
import mujoco.viewer
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

from config.render_loader import load_render_config, import_env


def get_end_effector_pos(env):
    if hasattr(env, "get_end_effector_pos"):
        try:
            pos = env.get_end_effector_pos()
            return np.array(pos, dtype=np.float32)
        except Exception:
            pass

    model = getattr(env, "model", None)
    data = getattr(env, "data", None)
    if model is None or data is None:
        return None

    for site_name in ("eetip", "r_gripper_tip", "gripper_tip"):
        try:
            site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        except Exception:
            site_id = -1
        if site_id != -1:
            return data.site_xpos[site_id].copy()
    return None


def save_trajectory(trajectory, label):
    if not trajectory:
        return
    log_dir = os.path.join(REPO_ROOT, "logs", "trajectories")
    os.makedirs(log_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(log_dir, f"{label}_{ts}.csv")
    png_path = os.path.join(log_dir, f"{label}_{ts}.png")

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y", "z"])
        writer.writerows(trajectory)

    traj = np.array(trajectory)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], linewidth=1.5)
    ax.set_title("End-effector trajectory")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    fig.tight_layout()
    fig.savefig(png_path, dpi=150)
    plt.close(fig)

    print(f"Saved trajectory: {csv_path}")
    print(f"Saved trajectory plot: {png_path}")


def main():
    parser = argparse.ArgumentParser(description="Scene-agnostic MuJoCo render demo (x86/CUDA).")
    parser.add_argument(
        "--config",
        type=str,
        default="config/render_run.yaml",
        help="Path to render YAML config."
    )
    args = parser.parse_args()

    # Load configuration
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
    trace = bool(run.get("trace", False))

    # Import env class
    EnvClass = import_env(env_class_path)

    # Create environment (try model_path with disable_logging; fallback to old signature)
    try:
        env = EnvClass(model_path=model_xml, render_mode="human", disable_logging=disable_logging)
    except TypeError:
        try:
            env = EnvClass(model_path=model_xml, render_mode="human")
        except TypeError:
            env = EnvClass(render_mode="human")

    # Load PPO model
    model = PPO.load(policy_path, env=env)

    # Launch standard blocking viewer
    if getattr(env, "viewer", None) is None:
        env.viewer = mujoco.viewer.launch_passive(env.model, env.data)

    try:
        for ep in range(episodes):
            obs, _ = env.reset()
            done = False
            ep_reward = 0.0
            start = time.time()
            trajectory = []

            while not done:
                action, _ = model.predict(obs, deterministic=deterministic)
                obs, reward, terminated, truncated, info = env.step(action)
                ep_reward += reward

                elapsed = time.time() - start
                if elapsed >= max_seconds:
                    truncated = True

                done = bool(terminated or truncated)

                if trace:
                    pos = get_end_effector_pos(env)
                    if pos is not None:
                        trajectory.append(pos.tolist())

                if env.viewer is not None:
                    env.viewer.sync()

                # ~120 FPS throttle
                time.sleep(1.0 / 120.0)

            print(f"[Episode {ep + 1}/{episodes}] reward={ep_reward:.2f} time={elapsed:.2f}s")
            if trace:
                save_trajectory(trajectory, f"render_ep{ep + 1}")

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    except Exception:
        traceback.print_exc()
    finally:
        env.close()


if __name__ == "__main__":
    main()