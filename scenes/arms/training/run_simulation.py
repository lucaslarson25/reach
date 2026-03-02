#!/usr/bin/env python3
"""
Run trained arm policy with MuJoCo viewer. YAML-driven.

Usage (from project root):
  mjpython -m scenes.arms.training.run_simulation --config config/arms.yaml
  mjpython -m scenes.arms.training.run_simulation --config config/arms.yaml --arm-id ur5e
"""
import os
import sys
import argparse
from stable_baselines3 import PPO
from scenes.arms.env import ArmReachEnv

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, _REPO_ROOT)


def main():
    from config.arms_loader import load_arms_config, resolve_policy_path, resolve_policy_paths
    from scenes.arms.arm_registry import get_arm_info

    p = argparse.ArgumentParser(
        description="Run trained arm policy with viewer. Default: panda. CLI overrides YAML.",
    )
    p.add_argument("--config", type=str, default=os.path.join(_REPO_ROOT, "config", "arms.yaml"),
                   help="Path to arms.yaml")
    p.add_argument("--model", type=str, default=None, help="Path to .zip policy (overrides config)")
    p.add_argument("--arm-id", type=str, default=None, help="Arm to run (e.g. panda, aloha). Overrides config.")
    p.add_argument("--steps", type=int, default=None, help="Max simulation steps. Overrides config.")
    p.add_argument("--per-arm-policies", action="store_true", help="Load separate policy per arm. Overrides config.")
    p.add_argument("--debug", action="store_true", help="Print action norm every 100 steps")
    p.add_argument("--stochastic", action="store_true", help="Use stochastic policy")
    args = p.parse_args()

    cfg = load_arms_config(args.config)
    arm_id = args.arm_id if args.arm_id is not None else cfg["scene"].get("arm_id", "panda")
    ball_mode = cfg["scene"].get("ball_mode", "shared")
    per_arm_policies = args.per_arm_policies or cfg["scene"].get("per_arm_policies", False)
    info = get_arm_info(arm_id)
    n_arms = len(info["ee_sites"]) if info and info.get("ee_sites") else 1
    policy_paths = resolve_policy_paths(cfg, arm_id_override=arm_id, n_arms=n_arms)
    if args.model is not None:
        model_path = args.model
        policy_paths = model_path
    steps = args.steps if args.steps is not None else cfg["run"].get("steps", 5000)
    model_path = policy_paths[0] if isinstance(policy_paths, list) else policy_paths
    deterministic = not (args.stochastic or cfg["run"].get("stochastic", False))
    debug = args.debug or cfg["run"].get("debug", False)

    # Create env with viewer; do NOT pass to PPO.load so it stays unwrapped and render() works
    env = ArmReachEnv(arm_id=arm_id, render_mode="human", ball_mode=ball_mode)

    if per_arm_policies and n_arms > 1 and isinstance(policy_paths, list):
        models = [PPO.load(p) for p in policy_paths]
        act_groups = info.get("actuator_groups") or [list(range(env.model.nu))]
        if len(act_groups) != n_arms:
            act_groups = [list(range(env.model.nu))]
    else:
        model_path = policy_paths[0] if isinstance(policy_paths, list) else policy_paths
        models = [PPO.load(model_path)]
        act_groups = [list(range(env.model.nu))]

    # Ensure policy obs shape matches env
    import numpy as np
    obs_shape = env.observation_space.shape
    act_shape = env.action_space.shape
    if models[0].observation_space.shape != obs_shape:
        print("ERROR: Policy expects obs shape", models[0].observation_space.shape, "but env has", obs_shape)
        print("Train a new policy: arm_train_mac or arm_train_per_arm --config", args.config)
        return

    def predict_action(obs, det):
        if len(models) == 1:
            a, _ = models[0].predict(obs, deterministic=det)
            return a
        acts = []
        for i, m in enumerate(models):
            a, _ = m.predict(obs, deterministic=det)
            acts.append((act_groups[i], a))
        full = np.zeros(act_shape[0], dtype=np.float32)
        for (indices, a) in acts:
            for k, idx in enumerate(indices):
                if k < len(a):
                    full[idx] = a[k]
        return full

    obs, _ = env.reset()
    for t in range(steps):
        action = predict_action(obs, deterministic)
        obs, reward, term, trunc, _ = env.step(action)
        env.render()  # opens viewer on first call, then syncs
        if env.viewer is not None and not env.viewer.is_running():
            print("Viewer closed — exiting.")
            break
        if debug:
            if t == 0:
                print("obs shape:", obs.shape, "  action shape:", action.shape)
                print("first step action:", action, "  norm=%.4f" % float(np.linalg.norm(action)))
            elif t % 100 == 0:
                anorm = float(np.linalg.norm(action))
                print(f"step {t}  action norm={anorm:.4f}  reward={reward:.3f}")
        if term or trunc:
            obs, _ = env.reset()
    env.close()


if __name__ == "__main__":
    main()
