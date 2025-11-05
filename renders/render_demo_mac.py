# renders/render_demo_mac.py
import argparse
import time
import sys
import mujoco
import mujoco.viewer
from stable_baselines3 import PPO

# Prefer scene-specific env import (new structure)
try:
    from scenes.industrial_arm_reaching.env import Z1ReachEnv  # update if you move the env
except ImportError:
    # Back-compat shim if you still have the old path lying around
    from envs.mujoco_arm_env import Z1ReachEnv  # type: ignore


def parse_args():
    p = argparse.ArgumentParser(description="Render a trained PPO policy on macOS (MuJoCo viewer requires mjpython).")
    p.add_argument(
        "--model",
        default="scenes/industrial_arm_reaching/models/z1scene.xml",
        help="Path to MuJoCo XML scene file.",
    )
    p.add_argument(
        "--policy",
        default="scenes/industrial_arm_reaching/policies/ppo_z1_parallel_1.5m_best.zip",
        help="Path to SB3 .zip policy.",
    )
    p.add_argument("--episodes", type=int, default=10, help="Number of episodes to render.")
    p.add_argument("--max-secs", type=float, default=30.0, help="Max seconds per episode before truncation.")
    return p.parse_args()


def make_env(model_path: str):
    """
    Try to construct the environment with a model_path if supported.
    Fallback to old signature if not.
    """
    try:
        # If your env __init__ supports model_path=...
        env = Z1ReachEnv(render_mode="human", model_path=model_path)  # type: ignore[arg-type]
        return env
    except TypeError:
        # Fallback for older envs that load a fixed internal XML path
        print(
            "[render_demo_mac] Warning: Z1ReachEnv does not accept model_path. "
            "Using its internal default XML instead.",
            file=sys.stderr,
        )
        return Z1ReachEnv(render_mode="human")


def main():
    args = parse_args()

    # 1) Create env (tries model_path, falls back if unsupported)
    env = make_env(args.model)

    # 2) Load PPO policy
    ppo = PPO.load(args.policy, env=env)

    # 3) Launch viewer with the MuJoCo model/data (not the PPO!)
    # On macOS, this requires running under `mjpython`.
    if env.render_mode == "human" and getattr(env, "viewer", None) is None:
        env.viewer = mujoco.viewer.launch_passive(env.model, env.data)

    try:
        for ep in range(args.episodes):
            obs, _ = env.reset()
            done = False
            ep_reward = 0.0
            start = time.time()

            while not done:
                action, _ = ppo.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                ep_reward += reward

                # time limit
                if time.time() - start >= args.max_secs:
                    truncated = True

                done = terminated or truncated

                if env.render_mode == "human" and env.viewer is not None:
                    env.viewer.sync()
                time.sleep(1 / 120)

            print(f"Episode {ep + 1}: total reward = {ep_reward:.2f}")

    finally:
        env.close()


if __name__ == "__main__":
    main()