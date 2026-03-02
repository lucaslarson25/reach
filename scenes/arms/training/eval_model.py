# Eval arm reach policy. Run from project root: python -m scenes.arms.training.eval_model --model policies/ppo_arms_arm_2link_mac_300k.zip --arm-id arm_2link

import argparse
import numpy as np
from stable_baselines3 import PPO
from scenes.arms.env import ArmReachEnv


def main():
    p = argparse.ArgumentParser(description="Evaluate arm reach policy.")
    p.add_argument("--model", type=str, default="policies/ppo_arms_arm_2link_mac_300k.zip", help="Path to .zip policy.")
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--max-steps", type=int, default=5000)
    p.add_argument("--arm-id", type=str, default=None)
    p.add_argument("--model-path", type=str, default=None)
    args = p.parse_args()

    env = ArmReachEnv(render_mode=None, arm_id=args.arm_id, model_path=args.model_path)
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
        print(f"Episode {ep+1}: final distance = {final_dist:.4f}")

    env.close()


if __name__ == "__main__":
    main()
