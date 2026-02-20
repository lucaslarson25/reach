#!/usr/bin/env python3
"""
Smoke test for REACH - runs locally or on Monsoon (headless, no display required).
Use to verify setup before training: python -m tests.smoke_test
"""
from __future__ import annotations

import os
import sys

# Ensure project root is on path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def run_smoke_test() -> tuple[bool, list[str]]:
    """Run quick sanity checks. Returns (success, list of error messages)."""
    errors = []

    # 1. Imports
    try:
        import mujoco
        import torch
        import gymnasium
        from stable_baselines3 import PPO
    except ImportError as e:
        errors.append(f"Import failed: {e}")
        return False, errors

    # 2. Device detection (works with or without GPU)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")

    # 3. AINex env (headless - no render_mode)
    try:
        from scenes.ainex_soccer.env import AINexReachEnv

        env = AINexReachEnv()
        obs, info = env.reset(seed=42)
        assert obs is not None and len(obs) > 0
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        env.close()
        print("  AINexReachEnv: OK")
    except Exception as e:
        errors.append(f"AINexReachEnv failed: {e}")
        return False, errors

    # 4. Quick PPO step (tiny timesteps)
    try:
        from stable_baselines3.common.vec_env import DummyVecEnv

        def make_env():
            def _init():
                e = AINexReachEnv()
                e.reset(seed=42)
                return e

            return _init

        venv = DummyVecEnv([make_env()])
        model = PPO("MlpPolicy", venv, n_steps=32, batch_size=16, device=device, verbose=0)
        model.learn(total_timesteps=64)
        venv.close()
        print("  PPO learn (64 steps): OK")
    except Exception as e:
        errors.append(f"PPO smoke failed: {e}")
        return False, errors

    return True, []


def main() -> int:
    print("REACH smoke test (local or Monsoon)...")
    ok, errors = run_smoke_test()
    if ok:
        print("All smoke tests passed.")
        return 0
    for e in errors:
        print(f"  ERROR: {e}", file=sys.stderr)
    print("Smoke test FAILED.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
