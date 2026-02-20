"""
Evaluate AINex walk-to-ball policy with rendering.
"""
import os
import sys
import time
import argparse
import csv
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import mujoco.viewer
from stable_baselines3 import PPO

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from config.render_loader import load_render_config, import_env


def get_end_effector_pos(env):
    if hasattr(env, "get_end_effector_pos"):
        try:
            return np.array(env.get_end_effector_pos(), dtype=np.float32)
        except Exception:
            pass
    model = getattr(env, "model", None)
    data = getattr(env, "data", None)
    if model is None or data is None:
        return None
    try:
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "r_gripper_tip")
    except Exception:
        return None
    return data.site_xpos[site_id].copy()


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
    ax.set_title("End-effector trajectory (walk-to-ball)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    fig.tight_layout()
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    print(f"Saved trajectory: {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate AINex walk-to-ball policy.")
    parser.add_argument("--config", type=str, default="config/ainex_walk_to_ball.yaml")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max-seconds", type=float, default=25.0)
    parser.add_argument("--deterministic", action="store_true")
    args = parser.parse_args()

    cfg = load_render_config(args.config)
    scene = cfg["scene"]
    policy_cfg = cfg.get("policy", {})

    env_class_path = scene["env_class"]
    model_xml = scene["model_xml"]
    policy_path = policy_cfg.get("path")
    if not policy_path or not os.path.exists(policy_path):
        raise FileNotFoundError(f"Policy not found: {policy_path}")

    EnvClass = import_env(env_class_path)
    try:
        env = EnvClass(model_path=model_xml, render_mode="human")
    except TypeError:
        env = EnvClass(render_mode="human")

    model = PPO.load(policy_path, env=env)

    if getattr(env, "viewer", None) is None:
        env.viewer = mujoco.viewer.launch_passive(env.model, env.data)

    log_dir = os.path.join(REPO_ROOT, "logs", "ainex_walk_to_ball")
    os.makedirs(log_dir, exist_ok=True)
    eval_csv = os.path.join(log_dir, "eval_rollouts.csv")
    if not os.path.exists(eval_csv):
        with open(eval_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["episode", "reward", "length", "success"])

    try:
        for ep in range(args.episodes):
            obs, _ = env.reset()
            done = False
            ep_reward = 0.0
            steps = 0
            start = time.time()
            trajectory = []

            while not done:
                action, _ = model.predict(obs, deterministic=args.deterministic)
                obs, reward, terminated, truncated, info = env.step(action)
                ep_reward += reward
                steps += 1
                if args.max_seconds and (time.time() - start) >= args.max_seconds:
                    truncated = True
                done = bool(terminated or truncated)
                pos = get_end_effector_pos(env)
                if pos is not None:
                    trajectory.append(pos.tolist())
                if env.viewer is not None:
                    env.viewer.sync()
                time.sleep(1.0 / 120.0)

            success = bool(info.get("is_success", False))
            with open(eval_csv, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([ep + 1, ep_reward, steps, int(success)])
            print(f"[Episode {ep + 1}/{args.episodes}] reward={ep_reward:.2f} steps={steps} success={int(success)}")
            save_trajectory(trajectory, f"ainex_walk_to_ball_ep{ep + 1}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
