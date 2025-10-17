import mujoco
import mujoco.viewer
import sys
import time

def show_model(xml_path):
    # Load the model and data
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    print("Opening model viewer — press ESC or close the window to exit.")

    # Launch interactive viewer (blocking until closed)
    with mujoco.viewer.launch(model, data) as viewer:
        while viewer.is_running():
            # Step the simulation forward a bit (optional)
            mujoco.mj_step(model, data)

            # Render the frame
            viewer.sync()

            # Slow down the loop slightly to reduce CPU usage
            #time.sleep(0.01)

if __name__ == "__main__":
    xml_file = sys.argv[1] if len(sys.argv) > 1 else "model.xml"
    show_model(xml_file)
