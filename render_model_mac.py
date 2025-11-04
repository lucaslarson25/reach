import mujoco
import mujoco.viewer
import sys
import time
import numpy as np
import os

# run with: .venv/bin/mjpython render_model_mac.py

def show_model(xml_path):
    """
    Display a MuJoCo model interactively on macOS using mjpython and launch_passive().
    """

    if not os.path.exists(xml_path):
        print(f"Error: Could not find model file at: {xml_path}")
        return

    # Load model and data
    print(f"Loading MuJoCo model: {xml_path}")
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    # Get references for visualization
    try:
        ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "eetip")
        ball_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ball")
    except Exception as e:
        print("Could not find one or more object IDs in model (check XML naming).")
        ee_id, ball_id = None, None

    print("Model loaded successfully.")
    print("Opening MuJoCo viewer — press ESC or close the window to exit.")

    # Launch the interactive passive viewer
    with mujoco.viewer.launch_passive(model, data) as viewer:
        try:
            while viewer.is_running():
                # Step simulation forward
                mujoco.mj_step(model, data)

                # Compute and print distance if possible
                if ee_id is not None and ball_id is not None:
                    ee_pos = data.site_xpos[ee_id]
                    ball_pos = data.xpos[ball_id]
                    dist = np.linalg.norm(ee_pos - ball_pos)
                    print(f"Distance (EE -> Ball): {dist:.4f}", end="\r")

                # Update viewer
                viewer.sync()

                # Limit CPU usage and maintain responsiveness
                time.sleep(1 / 120)

        except KeyboardInterrupt:
            print("\n Viewer closed by user.")
        finally:
            print("\nViewer closed.")


if __name__ == "__main__":
    # Default to z1scene.xml if no argument is passed
    default_xml = os.path.join(os.path.dirname(__file__), "models", "z1scene.xml")
    xml_file = sys.argv[1] if len(sys.argv) > 1 else default_xml
    show_model(xml_file)