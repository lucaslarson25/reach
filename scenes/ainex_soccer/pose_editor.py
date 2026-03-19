"""
Interactive pose editor for the AINex MuJoCo model.

Use the viewer's Control sliders to pose the robot, then press Enter in the
terminal to dump qpos/qvel. Paste the output into your XML as a keyframe, or
use it to refine starting poses for the reach task.

Run from repo root (macOS requires mjpython for the viewer):
    .venv/bin/mjpython scenes/ainex_soccer/pose_editor.py

Note: "Press Enter to dump" works on macOS/Linux. On Windows, the viewer
still works; use Ctrl+C to exit.
"""

import sys
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np


def _model_path() -> Path:
    root = Path(__file__).resolve().parent
    return root / "models" / "ainex_stable.xml"


def build_actuator_to_qpos_map(model: mujoco.MjModel) -> list[tuple[str, str, int]]:
    """Return list of (actuator_name, joint_name, qpos_adr) for joint actuators."""
    mapping = []
    for a in range(model.nu):
        act_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a) or f"act{a}"
        j_id = int(model.actuator_trnid[a, 0])
        if j_id < 0:
            continue
        j_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j_id) or f"joint{j_id}"
        qadr = int(model.jnt_qposadr[j_id])
        j_type = int(model.jnt_type[j_id])
        if j_type in (0, 1):  # FREE or BALL
            continue
        mapping.append((act_name, j_name, qadr))
    return mapping


def main() -> None:
    model_path = _model_path()
    if not model_path.exists():
        print(f"Model not found: {model_path}")
        sys.exit(1)

    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)

    print(f"Loaded: {model_path}")
    print(f"nq: {model.nq}  nv: {model.nv}  nu: {model.nu}")
    print("Floating pose mode: NO PHYSICS STEPPING (robot will not fall).")
    print("Use the viewer right panel: 'Control' sliders (these set data.ctrl).")
    print("Press Enter in this terminal anytime to dump qpos/qvel.\n")

    act_map = build_actuator_to_qpos_map(model)
    if not act_map:
        print("No joint actuators found to drive pose.")
        sys.exit(1)

    for i, (_, _, qadr) in enumerate(act_map):
        if i < model.nu:
            data.ctrl[i] = data.qpos[qadr]

    print("Actuator -> Joint mapping (using Control sliders):")
    for i, (act_name, j_name, qadr) in enumerate(act_map):
        print(f"  ctrl[{i:02d}] {act_name} -> {j_name} (qpos[{qadr}])")
    print()

    np.set_printoptions(suppress=True, precision=6, linewidth=200)

    try:
        import select
        has_select = True
    except ImportError:
        has_select = False
        print("(stdin dump disabled on this platform; viewer still works)\n")

    with mujoco.viewer.launch_passive(model, data, show_left_ui=True, show_right_ui=True) as viewer:
        while viewer.is_running():
            viewer.sync()

            for i, (_, _, qadr) in enumerate(act_map):
                if i < model.nu:
                    data.qpos[qadr] = float(data.ctrl[i])
                    data.qvel[qadr] = 0.0

            mujoco.mj_forward(model, data)

            if has_select and select.select([sys.stdin], [], [], 0.0)[0]:
                _ = sys.stdin.readline()
                print("\nqpos length:", model.nq)
                print("qpos:\n", " ".join([f"{x:.6f}" for x in data.qpos]))
                print("\nqvel length:", model.nv)
                print("qvel:\n", " ".join([f"{x:.6f}" for x in data.qvel]))
                print()


if __name__ == "__main__":
    main()
