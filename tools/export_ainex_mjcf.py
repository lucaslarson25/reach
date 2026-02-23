from pathlib import Path

import mujoco

_REPO_ROOT = Path(__file__).resolve().parents[1]
URDF_PATH = _REPO_ROOT / "assets" / "ainex" / "ainex_edited.urdf"
OUT_PATH = _REPO_ROOT / "assets" / "ainex" / "ainex_exported.xml"

m = mujoco.MjModel.from_xml_path(str(URDF_PATH))
mujoco.mj_saveLastXML(str(OUT_PATH), m)
print(f"saved {OUT_PATH}")