"""
Replay AINex action group CSV files in MuJoCo.

Loads hardware-recorded motions from assets/action_groups/csv/ and replays them
in the viewer. Useful for debugging, visualization, and validating the
Servo→actuator mapping.

Run from repo root (macOS requires mjpython for the viewer):
    .venv/bin/mjpython scenes/ainex_soccer/replay_actiongroup.py [action_name]
    .venv/bin/mjpython scenes/ainex_soccer/replay_actiongroup.py forward_one_step
"""

import argparse
import csv
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np


# Leg actuator indices for balance correction (hip/ankle pitch and roll)
# Right leg: 0-5, Left leg: 6-11
BALANCE_ANKLE_PITCH = [4, 10]   # right, left (same correction for both)
BALANCE_ANKLE_ROLL = [5, 11]     # right, left (opposite sign for lateral)
BALANCE_HIP_PITCH = [2, 8]
BALANCE_HIP_ROLL = [1, 7]       # right, left (opposite sign for lateral)

# Servo1..22 -> ctrl indices (skips head actuators 12-13)
# 0-5 right leg, 6-11 left leg, 14-18 right arm, 19-23 left arm
SERVO_TO_CTRL = [
    0, 1, 2, 3, 4, 5,   # Servo1..6 -> right leg
    6, 7, 8, 9, 10, 11,  # Servo7..12 -> left leg
    14, 15, 16, 17, 18,  # Servo13..17 -> right arm
    19, 20, 21, 22, 23,  # Servo18..22 -> left arm
]

# Ticks (0..1000) -> radians, center 500 = 0
RAD_PER_TICK = 2.09 / 500.0
CTRL_SIGN = np.ones(24, dtype=float)
CTRL_OFFSET = np.zeros(24, dtype=float)


def _model_path() -> Path:
    return Path(__file__).resolve().parent / "models" / "ainex_stable.xml"


def _csv_dir() -> Path:
    return Path(__file__).resolve().parent / "assets" / "action_groups" / "csv"


def load_actiongroup_csv(csv_path: Path) -> list[tuple[int, np.ndarray]]:
    """Load CSV: returns list of (time_ms, servos_22)."""
    frames = []
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ms = int(row["Time"])
            servos = np.array([float(row[f"Servo{i}"]) for i in range(1, 23)], dtype=float)
            frames.append((ms, servos))
    return frames


def ticks_to_ctrl(servos_22: np.ndarray, model: mujoco.MjModel) -> np.ndarray:
    """Convert servo ticks to ctrl targets (radians)."""
    ctrl = np.zeros(model.nu, dtype=float)
    for i, tick in enumerate(servos_22):
        ctrl_idx = SERVO_TO_CTRL[i]
        rad = (tick - 500.0) * RAD_PER_TICK
        rad = CTRL_SIGN[ctrl_idx] * rad + CTRL_OFFSET[ctrl_idx]
        rad = float(np.clip(rad, -2.2, 2.2))
        ctrl[ctrl_idx] = rad
    return ctrl


def _find_foot_geom_ids(model: mujoco.MjModel) -> list[int]:
    """Find geom IDs for foot collision boxes."""
    ids = []
    for gid in range(model.ngeom):
        name = model.geom(gid).name
        if name and "foot_col" in name:
            ids.append(gid)
    return ids


def _has_free_root(model: mujoco.MjModel) -> bool:
    return model.njnt > 0 and model.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE


def _set_root_pose(data: mujoco.MjData, height: float = 0.6) -> None:
    """Place root at (0, 0, height) with upright orientation."""
    data.qpos[0:7] = np.array([0.0, 0.0, height, 1.0, 0.0, 0.0, 0.0], dtype=data.qpos.dtype)
    data.qvel[0:6] = 0.0


def _snap_feet_to_ground(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    foot_ids: list[int],
    floor_z: float = -0.05,
    penetration: float = 0.002,
) -> None:
    """Adjust root height so lowest foot touches ground. Floor plane is at z=-0.05 in model."""
    if not foot_ids:
        return
    mujoco.mj_forward(model, data)
    min_z = min(float(data.geom_xpos[gid][2]) for gid in foot_ids)
    desired_min_z = floor_z - abs(penetration)
    delta = min_z - desired_min_z
    data.qpos[2] -= delta
    data.qvel[0:6] = 0.0
    mujoco.mj_forward(model, data)


def _apply_balance_correction(
    data: mujoco.MjData,
    torso_id: int,
    ctrl: np.ndarray,
    gain: float = 2.0,
) -> None:
    """
    Add IMU-based balance correction to ctrl.
    Uses torso orientation (up vector) to correct forward/back and lateral tilt.
    When leaning left (up_y > 0): add to right leg roll, subtract from left leg roll.
    """
    xmat = data.xmat[torso_id].reshape(3, 3)
    up = xmat[:, 2]  # z-axis = up in world frame
    # Forward tilt: up_x < 0 means leaning forward -> push back via ankle/hip pitch
    forward_tilt = -float(up[0])
    correction_pitch = gain * forward_tilt
    for idx in BALANCE_ANKLE_PITCH + BALANCE_HIP_PITCH:
        ctrl[idx] += correction_pitch
    # Lateral tilt: up_y > 0 means leaning left -> push right
    lateral_tilt = float(up[1])
    correction_roll = gain * lateral_tilt
    for idx in [1, 5]:   # right hip roll, right ankle roll
        ctrl[idx] -= correction_roll
    for idx in [7, 11]:  # left hip roll, left ankle roll
        ctrl[idx] += correction_roll
    # Bias: shift weight right to counteract model asymmetry (robot drifts left)
    ctrl[7] -= 0.08   # left hip roll
    ctrl[11] -= 0.08  # left ankle roll
    np.clip(ctrl, -2.2, 2.2, out=ctrl)


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay AINex action group CSV in MuJoCo")
    parser.add_argument(
        "action",
        nargs="?",
        default="forward_one_step",
        help="Action name (e.g. forward_one_step, clamp_left). Default: forward_one_step",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available action CSV files and exit",
    )
    parser.add_argument(
        "--balance",
        action="store_true",
        default=True,
        help="Enable IMU-based balance correction (default: True)",
    )
    parser.add_argument(
        "--no-balance",
        action="store_false",
        dest="balance",
        help="Disable balance correction",
    )
    parser.add_argument(
        "--balance-gain",
        type=float,
        default=0.6,
        help="Balance correction gain (default: 0.6; reduce if vibrating)",
    )
    args = parser.parse_args()

    csv_dir = _csv_dir()
    if not csv_dir.exists():
        print(f"CSV directory not found: {csv_dir}")
        return

    if args.list:
        files = sorted(csv_dir.glob("*.csv"))
        print("Available actions:")
        for f in files:
            print(f"  {f.stem}")
        return

    csv_path = csv_dir / f"{args.action}.csv"
    if not csv_path.exists():
        print(f"Action not found: {csv_path}")
        print("Use --list to see available actions.")
        return

    model_path = _model_path()
    if not model_path.exists():
        print(f"Model not found: {model_path}")
        return

    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    frames = load_actiongroup_csv(csv_path)
    foot_ids = _find_foot_geom_ids(model)
    free_root = _has_free_root(model)
    torso_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso")

    print(f"Loaded: {model_path}")
    print(f"Action: {csv_path.name} ({len(frames)} frames, {sum(ms for ms, _ in frames)} ms total)")
    print(f"nu={model.nu}, timestep={model.opt.timestep}")
    if args.balance:
        print(f"Balance correction: ON (gain={args.balance_gain})")
    else:
        print("Balance correction: OFF")

    # Initial pose: use symmetric neutral stand (legs mirrored) to avoid left/right bias.
    # Stand.csv has asymmetric values (left hip_roll 650 vs right 500) that shift CoM left.
    # Neutral: both legs 500,500,640,360,500,500; arms/head from stand.
    stand_path = csv_dir / "stand.csv"
    if stand_path.exists():
        stand_frames = load_actiongroup_csv(stand_path)
        s = stand_frames[0][1]
        # Mirror legs: use right leg values for left (500,500,640,360,500,500 each side)
        neutral = np.array([
            s[0], s[1], s[2], s[3], s[4], s[5],   # right leg: keep as-is
            s[0], s[1], s[2], s[3], s[4], s[5],   # left leg: same as right for symmetry
            s[12], s[13], s[14], s[15], s[16], s[17], s[18], s[19], s[20], s[21],  # arms, head
        ], dtype=float)
        init_ctrl = ticks_to_ctrl(neutral, model)
    else:
        init_ctrl = ticks_to_ctrl(frames[0][1], model)

    data.qpos[:] = 0.0
    data.qvel[:] = 0.0
    if free_root:
        _set_root_pose(data, height=0.6)
    mujoco.mj_forward(model, data)
    if free_root and foot_ids:
        _snap_feet_to_ground(model, data, foot_ids, floor_z=-0.05)
    data.ctrl[:] = init_ctrl
    mujoco.mj_forward(model, data)

    # Settling phase: run physics steps before opening viewer so robot stabilizes
    settle_steps = 400
    for _ in range(settle_steps):
        data.ctrl[:] = init_ctrl
        if args.balance:
            _apply_balance_correction(data, torso_id, data.ctrl, gain=args.balance_gain)
        mujoco.mj_step(model, data)

    # Use launch_passive (required on macOS with mjpython; launch() fails)
    with mujoco.viewer.launch_passive(model, data) as viewer:
        time.sleep(0.2)
        while viewer.is_running():
            for ms, servos in frames:
                if not viewer.is_running():
                    break
                ctrl = ticks_to_ctrl(servos, model)
                duration = ms / 1000.0
                steps = max(1, int(duration / model.opt.timestep))
                for _ in range(steps):
                    if not viewer.is_running():
                        break
                    data.ctrl[:] = ctrl
                    if args.balance:
                        _apply_balance_correction(
                            data, torso_id, data.ctrl, gain=args.balance_gain
                        )
                    mujoco.mj_step(model, data)
                    viewer.sync()
            time.sleep(0.2)


if __name__ == "__main__":
    main()
