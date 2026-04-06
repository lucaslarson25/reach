#!/usr/bin/env python3
"""Passive viewer: hip-mounted xArm pulls limp Ainex toward a randomly placed ball (IK each frame)."""

from __future__ import annotations

import os
import shutil
import sys
import time
from pathlib import Path


def _is_mjpython_runtime() -> bool:
    """True inside MuJoCo's native macOS viewer binary (set by the ``mjpython`` trampoline before execve)."""
    return bool(os.environ.get("MJPYTHON_BIN"))


def _reexec_mjpython_on_darwin_if_needed() -> None:
    """On macOS, ``launch_passive`` only works in the mjpython runtime; plain ``python3`` cannot use the threaded workaround (SIGTRAP / GL)."""
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
            "hip/run_demo on macOS needs MuJoCo's mjpython (passive viewer).\n"
            "Install mujoco in a venv and run:\n"
            "  .venv/bin/mjpython -m hip.run_demo",
            file=sys.stderr,
        )
        raise SystemExit(1)
    os.execv(str(mjpy), [str(mjpy), "-m", "hip.run_demo", *sys.argv[1:]])


if __name__ == "__main__":
    _reexec_mjpython_on_darwin_if_needed()

import mujoco
import mujoco.viewer

from hip.env import HipReachEnv


def _launch_passive_macos(model: mujoco.MjModel, data: mujoco.MjData) -> mujoco.viewer.Handle:
    if sys.platform == "darwin" and not _is_mjpython_runtime():
        print(
            "hip/run_demo: set HIP_NO_MJPYTHON_REEXEC=1 but this viewer needs mjpython on macOS.\n"
            "Run: .venv/bin/mjpython -m hip.run_demo",
            file=sys.stderr,
        )
        raise SystemExit(1)
    return mujoco.viewer.launch_passive(model, data)


def main() -> None:
    env = HipReachEnv()
    m, d = env.model, env.data
    env.reset(seed=int(time.time()) % 2**31)

    if sys.platform == "darwin" and _is_mjpython_runtime():
        print(
            "hip/run_demo: mjpython may log 'Task policy set failed' each frame on macOS; "
            "that QoS message is harmless.",
        )

    print("hip/run_demo: limp Ainex + xArm from right hip, wrist welded to r_el_yaw_link.")
    print("Ball respawns every 4 s. Close window to exit.")

    last_spawn = time.time()
    viewer_cm = _launch_passive_macos if sys.platform == "darwin" else mujoco.viewer.launch_passive
    with viewer_cm(m, d) as viewer:
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
