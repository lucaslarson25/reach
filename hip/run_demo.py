#!/usr/bin/env python3
"""Passive viewer: hip-mounted xArm pulls limp Ainex toward a randomly placed ball (IK each frame)."""

from __future__ import annotations

import os
import queue
import shutil
import sys
import threading
import time
from pathlib import Path


def _maybe_reexec_mjpython() -> None:
    """Optional: ``HIP_REEXEC_MJPYTHON=1`` replaces this process with venv ``mjpython`` (macOS only)."""
    if os.environ.get("HIP_REEXEC_MJPYTHON") != "1":
        return
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
        print("HIP_REEXEC_MJPYTHON=1 but mjpython not found.", file=sys.stderr)
        raise SystemExit(1)
    os.execv(str(mjpy), [str(mjpy), "-m", "hip.run_demo", *sys.argv[1:]])


if __name__ == "__main__":
    _maybe_reexec_mjpython()

import mujoco
import mujoco.viewer

from hip.env import HipReachEnv


def _launch_passive_any_python(model: mujoco.MjModel, data: mujoco.MjData) -> mujoco.viewer.Handle:
    """``launch_passive`` on macOS normally requires ``mjpython``; duplicate the Linux threaded path for plain ``python3``."""
    from mujoco.viewer import _launch_internal

    mujoco.mj_forward(model, data)
    handle_return: queue.Queue = queue.Queue(1)
    on_darwin = sys.platform == "darwin"
    using_mjpython = Path(sys.executable).name == "mjpython"

    if on_darwin and using_mjpython:
        return mujoco.viewer.launch_passive(model, data)

    if on_darwin and not using_mjpython:
        thread = threading.Thread(
            target=_launch_internal,
            args=(model, data),
            kwargs={
                "run_physics_thread": False,
                "handle_return": handle_return,
                "key_callback": None,
                "show_left_ui": True,
                "show_right_ui": True,
            },
        )
        thread.daemon = True
        thread.start()
        return handle_return.get()

    return mujoco.viewer.launch_passive(model, data)


def main() -> None:
    env = HipReachEnv()
    m, d = env.model, env.data
    env.reset(seed=int(time.time()) % 2**31)

    on_darwin = sys.platform == "darwin"
    using_mjpython = Path(sys.executable).name == "mjpython"
    if on_darwin and not using_mjpython:
        print(
            "hip/run_demo: using threaded MuJoCo viewer (plain python3 on macOS). "
            "If the window is blank, run: .venv/bin/mjpython -m hip.run_demo "
            "or HIP_REEXEC_MJPYTHON=1 python3 -m hip.run_demo",
        )
    elif on_darwin and using_mjpython:
        print(
            "hip/run_demo: mjpython may print 'Task policy set failed' once per frame; "
            "that macOS QoS warning is harmless.",
        )

    print("hip/run_demo: limp Ainex + xArm from right hip, wrist welded to r_el_yaw_link.")
    print("Ball respawns every 4 s. Close window to exit.")

    last_spawn = time.time()
    with _launch_passive_any_python(m, d) as viewer:
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
