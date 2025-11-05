# renders/render_model.py
# Usage examples:
#   .venv/bin/python -m renders.render_model \
#       --model scenes/industrial_arm_reaching/models/z1scene.xml

import os
import sys
import time
import argparse
import numpy as np
import mujoco
import mujoco.viewer


def show_model(xml_path: str) -> None:
    """Display a MuJoCo model interactively on x86 (CUDA-friendly) using the blocking viewer."""
    if not os.path.exists(xml_path):
        print(f"Error: model XML not found: {xml_path}")
        return

    print(f"Loading MuJoCo model: {xml_path}")
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    # Try to resolve optional IDs for convenience logging
    try:
        ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "eetip")
    except Exception:
        ee_id = None

    try:
        ball_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ball")
    except Exception:
        ball_id = None

    print("Opening MuJoCo viewer — press ESC or close the window to exit.")

    # Blocking, cross-platform viewer (no mjpython requirement)
    with mujoco.viewer.launch(model, data) as viewer:
        try:
            while viewer.is_running():
                # Advance sim
                mujoco.mj_step(model, data)

                # If both objects exist, print EE→ball distance
                if ee_id is not None and ball_id is not None:
                    ee_pos = data.site_xpos[ee_id]
                    ball_pos = data.xpos[ball_id]
                    dist = np.linalg.norm(ee_pos - ball_pos)
                    print(f"Distance (EE -> Ball): {dist:.4f}", end="\r", flush=True)

                # Render frame
                viewer.sync()

                # Gentle throttling (~120 FPS)
                time.sleep(1 / 120)
        except KeyboardInterrupt:
            print("\nViewer closed by user.")
        finally:
            print("\nViewer closed.")


def main():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    default_model = os.path.join(
        repo_root, "scenes", "industrial_arm_reaching", "models", "z1scene.xml"
    )

    parser = argparse.ArgumentParser(description="Render a MuJoCo model (x86/CUDA friendly).")
    parser.add_argument("--model", type=str, default=default_model, help="Path to MuJoCo XML model.")
    args = parser.parse_args()

    show_model(args.model)


if __name__ == "__main__":
    # Ensure repo root is on sys.path (keeps imports flexible if extended later)
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    main()