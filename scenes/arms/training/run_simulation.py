#!/usr/bin/env python3
"""
Run trained arm policy with MuJoCo viewer. YAML-driven.

Usage (from project root):
  mjpython -m scenes.arms.training.run_simulation --config config/arms.yaml
  mjpython -m scenes.arms.training.run_simulation --config config/arms.yaml --arm-id ur5e

Note: Policy files are pickled with the current environment (e.g. NumPy). Use the same
NumPy major version for training and running (see requirements.txt).
"""
import os
import sys
import argparse
from stable_baselines3 import PPO
from scenes.arms.env import ArmReachEnv

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, _REPO_ROOT)


def main():
    from config.arms_loader import load_arms_config, apply_arm_overrides, resolve_policy_paths
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
    cfg = apply_arm_overrides(cfg, arm_id)
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

    # Create env with viewer; apply overrides for reach, initial_pose, etc.
    scene = cfg["scene"]
    train = cfg["train"]
    env_kw = dict(arm_id=arm_id, render_mode="human", ball_mode=ball_mode)
    if scene.get("initial_pose"):
        env_kw["initial_pose"] = scene["initial_pose"]
    if scene.get("initial_keyframe"):
        env_kw["initial_keyframe"] = scene["initial_keyframe"]
    if train.get("joint_limit_margin_penalty") is not None:
        env_kw["joint_limit_margin_penalty"] = train["joint_limit_margin_penalty"]
    if train.get("reach_max_cap") is not None:
        env_kw["reach_max"] = train["reach_max_cap"]
    if train.get("reach_min_mode"):
        env_kw["reach_min_mode"] = train["reach_min_mode"]
    if train.get("reach_min_fraction") is not None:
        env_kw["reach_min_fraction"] = train["reach_min_fraction"]
    if train.get("reach_min_floor") is not None:
        env_kw["reach_min_floor"] = train["reach_min_floor"]
    env = ArmReachEnv(**env_kw)

    # Resolve and validate policy path(s)
    if args.model is not None:
        load_path = args.model if os.path.isabs(args.model) else os.path.join(_REPO_ROOT, args.model)
    else:
        load_path = policy_paths[0] if isinstance(policy_paths, list) else policy_paths
    if not os.path.isfile(load_path):
        print("ERROR: Policy file not found:", load_path)
        print("Train first: python scripts/train.py --arm-id", arm_id)
        return
    def _load_model(path):
        try:
            return PPO.load(path)
        except ModuleNotFoundError as e:
            if "numpy._core" in str(e) or "numpy.core" in str(e):
                print("ERROR: Policy was saved with a different NumPy version. Use the same NumPy major version for train and run.")
                print("  Current: pip show numpy. Fix: pip install 'numpy>=2.0,<3' (or match the env that trained the policy).")
            raise

    if per_arm_policies and n_arms > 1 and isinstance(policy_paths, list):
        to_load = [p for p in policy_paths if os.path.isfile(p)]
        if len(to_load) != n_arms:
            print("ERROR: Expected", n_arms, "policy files (per_arm_policies). Found:", to_load)
            return
        models = [_load_model(p) for p in to_load]
        act_groups = info.get("actuator_groups") or [list(range(env.model.nu))]
        if len(act_groups) != n_arms:
            act_groups = [list(range(env.model.nu))]
    else:
        models = [_load_model(load_path)]
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
