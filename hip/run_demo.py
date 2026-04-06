#!/usr/bin/env python3
"""Passive viewer: hip-mounted xArm pulls limp Ainex toward a randomly placed ball (IK each frame)."""

import time

import mujoco
import mujoco.viewer
from hip.env import HipReachEnv


def main():
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
