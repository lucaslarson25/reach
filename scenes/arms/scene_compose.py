"""
Compose reach scene from arm-only XML.

Writes a single MuJoCo XML that includes floor, ball(s), and the arm.
User only uploads the arm; floor and ball are fixed.
For multi-arm per_arm mode, ball_count > 1 generates ball_0, ball_1, ...
"""

from __future__ import annotations

from pathlib import Path


def get_models_dir() -> Path:
    return Path(__file__).resolve().parent / "models"


def compose_scene(arm_include_path: str, arm_id: str = "arm", ball_count: int = 1) -> str:
    """
    Build scene XML that includes floor, ball, and the arm. The composed file
    is written *inside* the arm's directory so the arm's meshdir="assets" and
    other relative paths resolve correctly when MuJoCo loads it.

    arm_include_path: path relative to scenes/arms/models/, e.g. "arms/panda/panda_nohand.xml"
    arm_id: used for the output filename.

    Returns absolute path to the composed XML file.
    """
    models_dir = get_models_dir()
    # Write composed scene inside the arm dir so arm's assets/ resolve correctly
    arm_dir = (models_dir / arm_include_path).resolve().parent
    # Relative path from arm_dir back to models_dir for floor/ball
    try:
        rel_to_models = arm_dir.relative_to(models_dir.resolve())
    except ValueError:
        rel_to_models = Path(arm_include_path).parent
    depth = len(rel_to_models.parts)
    prefix = "/".join([".."] * depth) if depth else "."
    arm_basename = Path(arm_include_path).name
    out_path = arm_dir / "_composed.xml"
    if ball_count <= 0:
        ball_count = 1
    if ball_count == 1:
        ball_part = f'  <include file="{prefix}/ball.xml"/>'
    else:
        ball_bodies = "\n".join(
            f'    <body name="ball_{i}" pos="0 0 0.05">\n'
            f'      <geom type="sphere" size="0.03" rgba="1 0 0 1" mass="0.01" contype="1" conaffinity="1"/>\n'
            f'    </body>'
            for i in range(ball_count)
        )
        ball_part = f"  <worldbody>\n{ball_bodies}\n  </worldbody>"
    content = f'''<mujoco model="reach_scene">
  <include file="{prefix}/floor.xml"/>
{ball_part}
  <include file="{arm_basename}"/>
</mujoco>
'''
    out_path.write_text(content)
    return str(out_path.resolve())
