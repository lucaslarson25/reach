from pathlib import Path

import mujoco
import mujoco.viewer

_REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = str(_REPO_ROOT / "assets" / "ainex" / "ainex_edited.urdf")


def main():
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()

if __name__ == "__main__":
    main()