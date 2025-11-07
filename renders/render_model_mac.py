# renders/render_model_mac.py
#
# Usage (macOS + mjpython):
#   .venv/bin/mjpython renders/render_model_mac.py --config config/render_run.yaml
# or:
#   .venv/bin/mjpython renders/render_model_mac.py --model scenes/industrial_arm_reaching/models/z1scene.xml
#
# This script:
#   - Loads model_xml from YAML by default (scene.model_xml)
#   - Or lets you override with --model
#   - Opens MuJoCo passive viewer (required on macOS)
#   - Optionally prints distance between "eetip" site and "ball" body if present
#
# Assumes:
#   - MuJoCo + mjpython installed
#   - Run from repo root for relative paths to work

import os
import sys
import time
import argparse

# Repo root on path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np
import mujoco
import mujoco.viewer

from config.render_loader import load_render_config


def show_model(xml_path: str) -> None:
    if not os.path.exists(xml_path):
        print(f"Error: model XML not found at: {xml_path}")
        return

    print(f"Loading MuJoCo model: {xml_path}")
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    # Try to resolve ids if present
    def safe_id(obj_type, name):
        try:
            return mujoco.mj_name2id(model, obj_type, name)
        except Exception:
            return -1

    ee_id = safe_id(mujoco.mjtObj.mjOBJ_SITE, "eetip")
    ball_id = safe_id(mujoco.mjtObj.mjOBJ_BODY, "ball")

    print("Opening MuJoCo viewer — press ESC or close the window to exit.")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        try:
            while viewer.is_running():
                mujoco.mj_step(model, data)

                if ee_id != -1 and ball_id != -1:
                    ee_pos = data.site_xpos[ee_id]
                    ball_pos = data.xpos[ball_id]
                    dist = np.linalg.norm(ee_pos - ball_pos)
                    print(f"Distance (EE → Ball): {dist:.4f}", end="\r")

                viewer.sync()
                time.sleep(1.0 / 120.0)
        except KeyboardInterrupt:
            print("\nViewer closed by user.")
        finally:
            print("\nViewer closed.")


def main():
    parser = argparse.ArgumentParser(description="MuJoCo model viewer (macOS).")
    parser.add_argument(
        "--config",
        type=str,
        default="config/render_run.yaml",
        help="YAML config (to read scene.model_xml).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Override: direct path to model XML.",
    )
    args = parser.parse_args()

    if args.model:
        xml_path = args.model
    else:
        cfg = load_render_config(args.config)
        xml_path = cfg["scene"]["model_xml"]

    show_model(xml_path)


if __name__ == "__main__":
    main()