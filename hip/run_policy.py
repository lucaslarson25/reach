#!/usr/bin/env python3
"""Roll out a trained PPO policy with the MuJoCo passive viewer."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np


def _is_mjpython_runtime() -> bool:
    return bool(os.environ.get("MJPYTHON_BIN"))


def _reexec_mjpython_on_darwin_if_needed() -> None:
    if sys.platform != "darwin":
        return
    if _is_mjpython_runtime():
        return
    if os.environ.get("HIP_NO_MJPYTHON_REEXEC") == "1":
        return
    repo_root = Path(__file__).resolve().parent.parent
    candidates = [
        Path(sys.executable).resolve().parent / "mjpython",
        repo_root / ".venv" / "bin" / "mjpython",
        repo_root / "venv" / "bin" / "mjpython",
    ]
    mjpy = next((p for p in candidates if p.is_file()), None)
    if mjpy is None:
        w = shutil.which("mjpython")
        mjpy = Path(w) if w else None
    if mjpy is None or not mjpy.is_file():
        print(
            "hip/run_policy on macOS needs mjpython.\n"
            "  .venv/bin/mjpython -m hip.run_policy --model <path>",
            file=sys.stderr,
        )
        raise SystemExit(1)
    os.execv(str(mjpy), [str(mjpy), "-m", "hip.run_policy", *sys.argv[1:]])


if __name__ == "__main__":
    _reexec_mjpython_on_darwin_if_needed()

import mujoco
import mujoco.viewer
from stable_baselines3 import PPO

from hip.env import HipReachEnv


def _launch_viewer(model: mujoco.MjModel, data: mujoco.MjData):
    if sys.platform == "darwin" and not _is_mjpython_runtime():
        print("Use: .venv/bin/mjpython -m hip.run_policy --model ...", file=sys.stderr)
        raise SystemExit(1)
    return mujoco.viewer.launch_passive(model, data)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True, help="Path to .zip from train_hip_reach")
    ap.add_argument("--deterministic", action="store_true", help="Greedy actions (default).")
    ap.add_argument("--stochastic", action="store_true", help="Sample from policy.")
    args = ap.parse_args()
    det = not args.stochastic

    env = HipReachEnv(success_bonus=0.0)
    model_path = Path(args.model).expanduser()
    if not model_path.is_file():
        print(f"Missing model file: {model_path}", file=sys.stderr)
        raise SystemExit(1)
    # Load without env to avoid SB3 wrapping in Monitor/DummyVecEnv (inference-only).
    model = PPO.load(str(model_path), env=None)

    obs, _ = env.reset(seed=int(time.time()) % 2**31)
    m, d = env.model, env.data

    with _launch_viewer(m, d) as viewer:
        while viewer.is_running():
            action, _ = model.predict(obs, deterministic=det)
            obs, _reward, term, trunc, _info = env.step(np.asarray(action, dtype=np.float32))
            if term or trunc:
                obs, _ = env.reset(seed=int(time.time() * 1000) % 2**31)
            viewer.sync()


if __name__ == "__main__":
    main()
