"""
Load and convert AINex action groups (CSV) to MuJoCo ctrl targets.
Servo1-22 map to actuators: legs (0-11), right arm (14-18), left arm (19-23).
Head (12-13) is not in action groups.
"""
from pathlib import Path
import csv

import numpy as np

# Servo1..22 -> ctrl indices (skip head 12-13)
SERVO_TO_CTRL = [
    0, 1, 2, 3, 4, 5,       # Servo1-6   -> right leg
    6, 7, 8, 9, 10, 11,     # Servo7-12  -> left leg
    14, 15, 16, 17, 18,     # Servo13-17 -> right arm
    19, 20, 21, 22, 23,     # Servo18-22 -> left arm
]
LEG_CTRL_INDICES = list(range(12))
ARM_CTRL_INDICES = [14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
RIGHT_ARM_CTRL_INDICES = [14, 15, 16, 17, 18]
RAD_PER_TICK = 2.09 / 500.0  # ticks 0-1000 -> radians


def load_actiongroup_csv(csv_path: str | Path) -> list[tuple[int, np.ndarray]]:
    """Load action group CSV. Returns list of (time_ms, servos_22)."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Action group not found: {path}")
    frames = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            t = int(row.get("Time", 200))
            servos = np.array([float(row.get(f"Servo{i}", 500)) for i in range(1, 23)], dtype=np.float64)
            frames.append((t, servos))
    return frames


def ticks_to_ctrl(servos_22: np.ndarray, n_act: int = 24) -> np.ndarray:
    """Convert Servo1-22 ticks (0-1000) to ctrl array. Unmapped actuators stay 0."""
    ctrl = np.zeros(n_act, dtype=np.float64)
    for i, tick in enumerate(servos_22):
        if i < len(SERVO_TO_CTRL):
            ctrl_idx = SERVO_TO_CTRL[i]
            rad = (tick - 500.0) * RAD_PER_TICK
            ctrl[ctrl_idx] = float(np.clip(rad, -2.2, 2.2))
    return ctrl


class ActionGroupPlayer:
    """Cycles through action group frames for walking. Use legs_only=True for walking while policy controls arm."""

    def __init__(self, csv_path: str | Path, legs_only: bool = True):
        self.frames = load_actiongroup_csv(csv_path)
        self.frame_idx = 0
        self.steps_until_next = 0
        self.legs_only = legs_only
        self.dt = 0.002

    def get_leg_ctrl(self, n_act: int = 24) -> np.ndarray:
        """Get ctrl for legs (0-11) from current frame. Head/arm left zero for policy."""
        ctrl = np.zeros(n_act)
        if not self.frames:
            return ctrl
        _, servos = self.frames[self.frame_idx]
        for i in range(12):
            if i < len(SERVO_TO_CTRL):
                ctrl_idx = SERVO_TO_CTRL[i]
                rad = (servos[i] - 500.0) * RAD_PER_TICK
                ctrl[ctrl_idx] = float(np.clip(rad, -2.2, 2.2))
        return ctrl

    def get_ctrl(self, n_act: int = 24) -> np.ndarray:
        """Get full ctrl. Use legs_only=False for full replay."""
        if self.legs_only:
            return self.get_leg_ctrl(n_act)
        if not self.frames:
            return np.zeros(n_act)
        _, servos = self.frames[self.frame_idx]
        return ticks_to_ctrl(servos, n_act)

    def advance(self, dt: float | None = None) -> None:
        dt = dt or self.dt
        if not self.frames:
            return
        self.steps_until_next -= 1
        if self.steps_until_next <= 0:
            self.frame_idx = (self.frame_idx + 1) % len(self.frames)
            t_ms, _ = self.frames[self.frame_idx]
            self.steps_until_next = max(1, int(t_ms / 1000.0 / dt))

    def reset(self) -> None:
        self.frame_idx = 0
        if self.frames:
            t_ms, _ = self.frames[0]
            self.steps_until_next = max(1, int(t_ms / 1000.0 / self.dt))


def get_actiongroup_path(name: str = "forward_one_step") -> Path:
    """Get path to action group CSV."""
    repo = Path(__file__).resolve().parents[2]
    p = repo / "assets" / "action_groups" / "csv" / f"{name}.csv"
    if not p.exists():
        raise FileNotFoundError(f"Action group '{name}' not found at {p}")
    return p
