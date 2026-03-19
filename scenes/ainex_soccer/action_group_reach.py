"""
Action group integration for AINex reach training.

Converts hardware action group CSVs (Servo1..Servo22) into right-arm joint targets
(radians) for use as reference trajectories. The policy learns to blend with or
residual on these references.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np

# Same conversion as ActionGroupEngine (ROS pulse -> rad)
ENCODER_TICKS_PER_RADIAN = (180.0 / math.pi) / 240.0 * 1000.0

# CSV servo ID (1..22) -> (init_pulse, coef). Servos 14-18 = right arm.
_INIT_BY_ID = {
    1: 500, 2: 500, 3: 500, 4: 500, 5: 240, 6: 760, 7: 500, 8: 500,
    9: 500, 10: 500, 11: 500, 12: 500, 13: 875, 14: 125, 15: 500, 16: 500,
    17: 500, 18: 500, 19: 500, 20: 500, 21: 500, 22: 500,
}
_FLIPPED_IDS = {15, 16}
_COEF = {sid: (-ENCODER_TICKS_PER_RADIAN if sid in _FLIPPED_IDS else ENCODER_TICKS_PER_RADIAN)
         for sid in _INIT_BY_ID}


@dataclass
class ActionGroupReach:
    """Right-arm subset of an action group for reach task."""
    name: str
    frames: list[np.ndarray]  # each: 5 floats (r_sho_pitch, r_sho_roll, r_el_pitch, r_el_yaw, r_gripper) in rad


# CSV column index (0-based) -> servo_id. Servos 14-18 = right arm = columns 13-17.
_RIGHT_ARM_CSV_IDX = [13, 14, 15, 16, 17]  # Servo14..Servo18
_RIGHT_ARM_SERVO_IDS = [14, 15, 16, 17, 18]


def _pulse_to_rad(servo_id: int, pulse: float, joint_range: Tuple[float, float]) -> float:
    init = _INIT_BY_ID.get(servo_id, 500)
    coef = _COEF.get(servo_id, ENCODER_TICKS_PER_RADIAN)
    p = float(np.clip(pulse, 0.0, 1000.0))
    ang = (p - init) / coef
    return float(np.clip(ang, joint_range[0], joint_range[1]))


def load_action_group_reach(
    path: Path,
    joint_ranges: list[Tuple[float, float]],
) -> ActionGroupReach:
    """
    Load CSV and convert to right-arm radians.

    joint_ranges: 5 tuples (min, max) for r_sho_pitch, r_sho_roll, r_el_pitch, r_el_yaw, r_gripper.
    """
    frames: list[np.ndarray] = []
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        row0 = next(reader, None)
        if row0 is None:
            return ActionGroupReach(name=path.stem, frames=[])
        prefix = "Servo" if "Servo1" in row0 else "ID"
        # Process first row
        vals = np.array([float(row0[f"{prefix}{i}"]) for i in range(1, 23)], dtype=float)
        arm = np.zeros(5, dtype=np.float32)
        for i, (csv_idx, sid) in enumerate(zip(_RIGHT_ARM_CSV_IDX, _RIGHT_ARM_SERVO_IDS)):
            arm[i] = _pulse_to_rad(sid, vals[csv_idx], joint_ranges[i])
        frames.append(arm)
        for row in reader:
            vals = np.array([float(row[f"{prefix}{i}"]) for i in range(1, 23)], dtype=float)
            arm = np.zeros(5, dtype=np.float32)
            for i, (csv_idx, sid) in enumerate(zip(_RIGHT_ARM_CSV_IDX, _RIGHT_ARM_SERVO_IDS)):
                arm[i] = _pulse_to_rad(sid, vals[csv_idx], joint_ranges[i])
            frames.append(arm)
    return ActionGroupReach(name=path.stem, frames=frames)


def get_interpolated_reference(
    ag: ActionGroupReach,
    phase: float,
) -> np.ndarray:
    """
    Get right-arm reference at phase in [0, 1].
    phase=0 -> first frame, phase=1 -> last frame.
    """
    if not ag.frames:
        return np.zeros(5, dtype=np.float32)
    if len(ag.frames) == 1:
        return ag.frames[0].copy()
    t = np.clip(phase, 0.0, 1.0) * (len(ag.frames) - 1)
    i0 = int(np.floor(t))
    i1 = min(i0 + 1, len(ag.frames) - 1)
    frac = t - i0
    return (1.0 - frac) * ag.frames[i0] + frac * ag.frames[i1]
