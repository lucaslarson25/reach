import argparse
import numpy as np
from scenes.industrial_arm_reaching.env import Z1ReachEnv
from stable_baselines3 import PPO


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained PPO policy.")
    parser.add_argument(
        "--model",
        type=str,
        default="ppo_z1_parallel.zip",
        help="Path to the trained PPO .zip file.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Number of evaluation episodes.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=5000,
        help="Max steps per episode (safety cutoff).",
    )
    parser.add_argument("--arm-id", type=str, default=None, help="Arm registry key (e.g. z1, arm_2link).")
    parser.add_argument("--model-path", type=str, default=None, help="Path to scene XML (overrides arm-id if set).")
    args = parser.parse_args()

    env = Z1ReachEnv(render_mode=None, arm_id=args.arm_id, model_path=args.model_path)
    model = PPO.load(args.model, env=env)

    for ep in range(args.episodes):
        obs, _ = env.reset()
        done = False
        steps = 0
        while not done and steps < args.max_steps:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            steps += 1
        ee_pos = env.data.site_xpos[env._ee_site_id]
        ball_pos = env.data.xpos[env._ball_body_id]
        final_dist = np.linalg.norm(ee_pos - ball_pos)
        print(f"Episode {ep + 1}: final distance = {final_dist:.4f}")

    env.close()


if __name__ == "__main__":
    main()
