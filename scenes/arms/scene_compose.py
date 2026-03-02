"""
Compose reach scene from arm-only XML.

Writes a single MuJoCo XML that includes floor, ball, and the arm.
User only uploads the arm; floor and ball are fixed.
"""

from __future__ import annotations

from pathlib import Path


def get_models_dir() -> Path:
    return Path(__file__).resolve().parent / "models"


def compose_scene(arm_include_path: str, arm_id: str = "arm") -> str:
    """
    Build scene XML that includes floor.xml, ball.xml, and the arm.

    arm_include_path: path relative to scenes/arms/models/, e.g. "arms/z1/z1_arm.xml"
    arm_id: used for the temp filename so multiple arms don't overwrite.

    Returns absolute path to the composed XML file (so MuJoCo can load it).
    """
    models_dir = get_models_dir()
    # Use a stable name per arm so we don't create many temp files
    out_name = f"_composed_{arm_id}.xml"
    out_path = models_dir / out_name
    content = f'''<mujoco model="reach_scene">
  <include file="floor.xml"/>
  <include file="ball.xml"/>
  <include file="{arm_include_path}"/>
</mujoco>
'''
    out_path.write_text(content)
    return str(out_path.resolve())
