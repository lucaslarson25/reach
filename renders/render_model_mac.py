import argparse
import os
import time
import numpy as np
import mujoco
import mujoco.viewer


# ============================================================
# Run with:
#   .venv/bin/mjpython -m renders.render_model_mac
# Or specify a custom model:
#   .venv/bin/mjpython -m renders.render_model_mac --model scenes/industrial_arm_reaching/models/z1scene.xml
# ============================================================


def show_model(xml_path: str) -> None:
    """
    Display a MuJoCo model interactively on macOS using mjpython and launch_passive().
    """
    if not os.path.exists(xml_path):
        print(f"Error: Could not find model file at: {xml_path}")
        return

    print(f"📦 Loading MuJoCo model: {xml_path}")
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    # Safe name lookup for optional IDs
    def safe_id(obj_type, name):
        try:
            return mujoco.mj_name2id(model, obj_type, name)
        except Exception:
            return -1

    ee_id = safe_id(mujoco.mjtObj.mjOBJ_SITE, "eetip")
    ball_id = safe_id(mujoco.mjtObj.mjOBJ_BODY, "ball")

    print("Model loaded successfully.")
    print("Opening MuJoCo viewer — press ESC or close the window to exit.")

    # Passive viewer (does not take over simulation control)
    with mujoco.viewer.launch_passive(model, data) as viewer:
        try:
            while viewer.is_running():
                mujoco.mj_step(model, data)

                # If model has both end-effector and ball, compute distance
                if ee_id != -1 and ball_id != -1:
                    ee_pos = data.site_xpos[ee_id]
                    ball_pos = data.xpos[ball_id]
                    dist = np.linalg.norm(ee_pos - ball_pos)
                    print(f"Distance (EE → Ball): {dist:.4f}", end="\r")

                viewer.sync()
                time.sleep(1 / 120)  # smooth refresh without CPU overload

        except KeyboardInterrupt:
            print("\nViewer closed by user.")
        finally:
            print("\nViewer closed.")


def main():
    # Default model: Z1 arm reaching scene
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    default_model = os.path.join(
        repo_root, "scenes", "industrial_arm_reaching", "models", "z1scene.xml"
    )

    parser = argparse.ArgumentParser(description="Render a MuJoCo model (macOS).")
    parser.add_argument(
        "--model",
        type=str,
        default=default_model,
        help="Path to the MuJoCo XML model file.",
    )
    args = parser.parse_args()

    show_model(args.model)


if __name__ == "__main__":
    main()