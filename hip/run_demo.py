#!/usr/bin/env python3
"""Passive viewer: hip-mounted xArm pulls limp Ainex toward a randomly placed ball (IK each frame)."""

from __future__ import annotations

import os
import shutil
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer

from hip.env import HipReachEnv


def _reexec_with_mjpython_on_macos() -> None:
    """MuJoCo ``launch_passive`` only works under ``mjpython`` on macOS (not plain ``python3``)."""
    if sys.platform != "darwin":
        return
    if Path(sys.executable).name == "mjpython":
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
            "On macOS, MuJoCo's passive viewer requires mjpython (not system python3).\n"
            "From the repo root, run:\n"
            "  .venv/bin/mjpython -m hip.run_demo",
            file=sys.stderr,
        )
        raise SystemExit(1)
    os.execv(str(mjpy), [str(mjpy), "-m", "hip.run_demo", *sys.argv[1:]])


def main():
    _reexec_with_mjpython_on_macos()
    env = HipReachEnv()
    m, d = env.model, env.data
    env.reset(seed=int(time.time()) % 2**31)

    print("hip/run_demo: limp Ainex + xArm from right hip, wrist welded to r_el_yaw_link.")
    print("Ball respawns every 4 s. Close window to exit.")

    last_spawn = time.time()
    with mujoco.viewer.launch_passive(m, d) as viewer:
        while viewer.is_running():
            now = time.time()
            if now - last_spawn > 4.0:
                env.reset(seed=int(now * 1000) % 2**31)
                last_spawn = now
            env.apply_ik_to_ball()
            mujoco.mj_step(m, d)
            env.freeze_ball_velocity()
            viewer.sync()


if __name__ == "__main__":
    main()
