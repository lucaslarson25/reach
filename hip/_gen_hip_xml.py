#!/usr/bin/env python3
"""Generate hip/models/hip_reach.xml (run from repo root or hip/).

Scales the embedded xArm7 so its span is closer to the Ainex humanoid arm.
The xArm ``link_base`` is parented to ``torso`` at the right-hip pocket offset so leg
DOFs do not twist under arm wrenches; ``link7`` is welded to ``r_gripper_link`` with a
stiff equality constraint.

Tweak XARM_LENGTH_SCALE (0–1, vs full-size menagerie arm) if the weld or reach misbehaves.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent

# ~0.35–0.40 matches Ainex upper+lower arm length tier (~0.25–0.30 m vs ~0.8 m full xArm).
XARM_LENGTH_SCALE = 0.37

# Torso world Z so feet rest near the floor (z=0) with hip.env nominal limp pose.
TORSO_Z = 0.245
# Initial ball position (free joint); random resets use env ball box.
BALL_SPAWN_XYZ = "0.38 0 0.42"
# Task site on r_gripper_link frame: between pinch jaws, slightly forward (+X), not on ulnar/pinky side.
GRIPPER_TIP_POS = "0.078 -0.038 0.012"
GRIPPER_TIP_SIZE = "0.006"
# Weld xArm flange to hand base (wrist side) instead of mid-forearm (r_el_yaw_link).
XARM_WELD_BODY2 = "r_gripper_link"
# Stiff equality so link7 and hand do not visibly separate (soft solref looks "detached").
WELD_SOLREF = "0.001 1"
WELD_SOLIMP = "0.9999 0.99999 1e-5 0.9 2"
# Extra damping on right-leg hip joints (arm reactions used to twist r_hip_yaw when base was there).
RIGHT_HIP_JOINT_DAMPING = "1.5"

XARM_MESH = "../../scenes/arms/models/arms/xarm7/assets/{name}.stl"
# r_hip_yaw_link offset from torso in ainex (m); arm pocket offset is added in torso frame.
AINEX_R_HIP_YAW_POS_Y = -0.029


def _f(x: float) -> str:
    s = f"{x:.8g}"
    return s.replace("e-0", "e-").replace("e+0", "e+")


def build_xarm_assets(s: float) -> str:
    sc = _f(s)
    lines = []
    for name in (
        "link_base",
        "link1",
        "link2",
        "link3",
        "link4",
        "link5",
        "link6",
        "link7",
        "end_tool",
    ):
        lines.append(
            f'    <mesh name="{name}" content_type="model/stl" scale="{sc} {sc} {sc}" '
            f'file="{XARM_MESH.format(name=name)}"/>'
        )
    return "\n" + "\n".join(lines)


def build_xarm_defaults(s: float) -> str:
    s2, s3, s4 = s**2, s**3, s**4
    arm = max(0.008, 0.1 * s4)
    fl = max(0.12, 0.5 * (s**0.5))
    d1, d2, d3 = max(1.2, 8 * s2), max(0.9, 5 * s2), max(0.55, 3 * s2)
    g1 = max(45.0, 1500 * s3)
    g2 = max(35.0, 1000 * s3)
    g3 = max(28.0, 800 * s3)
    fr1 = max(6.0, 50 * s3)
    fr2 = max(4.0, 30 * s3)
    fr3 = max(3.0, 20 * s3)
    kd1 = max(12.0, 150 * s3)
    kd2 = max(10.0, 100 * s3)
    kd3 = max(8.0, 80 * s3)
    site = max(0.00035, 0.001 * s)
    return f"""    <default class="xarm7">
      <geom type="mesh" material="white"/>
      <joint axis="0 0 1" armature="{_f(arm)}" range="-6.28319 6.28319" frictionloss="{_f(fl)}"/>
      <general biastype="affine" ctrlrange="-6.28319 6.28319"/>
      <default class="size1">
        <joint damping="{_f(d1)}"/>
        <general gainprm="{_f(g1)}" biasprm="0 -{_f(g1)} -{_f(kd1)}" forcerange="-{_f(fr1)} {_f(fr1)}"/>
      </default>
      <default class="size2">
        <joint damping="{_f(d2)}"/>
        <general gainprm="{_f(g2)}" biasprm="0 -{_f(g2)} -{_f(kd2)}" forcerange="-{_f(fr2)} {_f(fr2)}"/>
      </default>
      <default class="size3">
        <joint damping="{_f(d3)}"/>
        <general gainprm="{_f(g3)}" biasprm="0 -{_f(g3)} -{_f(kd3)}" forcerange="-{_f(fr3)} {_f(fr3)}"/>
      </default>
      <site size="{_f(site)}" rgba="1 0 0 1" group="4"/>
    </default>"""


def build_xarm_subtree(s: float, base_pos: tuple[float, float, float]) -> str:
    s3, s5 = s**3, s**5

    bx, by, bz = base_pos

    def block_inertial(pos, quat, mass, diag):
        px, py, pz = pos
        m = mass * s3
        d0, d1, d2 = (x * s5 for x in diag)
        return (
            f'              <inertial pos="{_f(px * s)} {_f(py * s)} {_f(pz * s)}" quat="{quat}" '
            f'mass="{_f(m)}"\n                diaginertia="{_f(d0)} {_f(d1)} {_f(d2)}"/>'
        )

    site_sz = max(0.0006, 0.002 * s)
    z1 = 0.267 * s
    y3 = -0.293 * s
    x4 = 0.0525 * s
    x5, y5 = 0.0775 * s, -0.3425 * s
    x7, z7 = 0.076 * s, 0.097 * s

    return f"""
            <body name="link_base" pos="{_f(bx)} {_f(by)} {_f(bz)}" quat="0.6532815 0.2705981 -0.6532815 0.2705981" childclass="xarm7">
{block_inertial((-0.021131, -0.0016302, 0.056488), "0.696843 0.20176 0.10388 0.680376", 0.88556, (0.00382023, 0.00335282, 0.00167725))}
              <geom mesh="link_base"/>
              <body name="link1" pos="0 0 {_f(z1)}">
{block_inertial((-0.0002, 0.02905, -0.01233), "0.978953 -0.202769 -0.00441617 -0.0227264", 2.382, (0.00569127, 0.00533384, 0.00293865))}
                <joint name="joint1" class="size1"/>
                <geom mesh="link1"/>
                <body name="link2" quat="1 -1 0 0">
{block_inertial((0.00022, -0.12856, 0.01735), "0.50198 0.86483 -0.00778841 0.00483285", 1.869, (0.00959898, 0.00937717, 0.00201315))}
                  <joint name="joint2" range="-2.059 2.0944" class="size1"/>
                  <geom mesh="link2"/>
                  <body name="link3" pos="0 {_f(y3)} 0" quat="1 1 0 0">
{block_inertial((0.0466, -0.02463, -0.00768), "0.913819 0.289775 0.281481 -0.0416455", 1.6383, (0.00351721, 0.00294089, 0.00195868))}
                    <joint name="joint3" class="size2"/>
                    <geom mesh="link3"/>
                    <body name="link4" pos="{_f(x4)} 0 0" quat="1 1 0 0">
{block_inertial((0.07047, -0.11575, 0.012), "0.422108 0.852026 -0.126025 0.282832", 1.7269, (0.00657137, 0.00647948, 0.00186763))}
                      <joint name="joint4" range="-0.19198 3.927" class="size2"/>
                      <geom mesh="link4"/>
                      <body name="link5" pos="{_f(x5)} {_f(y5)} 0" quat="1 1 0 0">
{block_inertial((-0.00032, 0.01604, -0.026), "0.999311 -0.0304457 0.000577067 0.0212082", 1.3203, (0.00534729, 0.00499076, 0.0013489))}
                        <joint name="joint5" class="size2"/>
                        <geom mesh="link5"/>
                        <body name="link6" quat="1 1 0 0">
{block_inertial((0.06469, 0.03278, 0.02141), "-0.217672 0.772419 0.16258 0.574069", 1.325, (0.00245421, 0.00221646, 0.00107273))}
                          <joint name="joint6" range="-1.69297 3.14159" class="size3"/>
                          <geom mesh="link6"/>
                          <body name="link7" pos="{_f(x7)} 0 {_f(z7)}" quat="1 -1 0 0">
{block_inertial((0.0, -0.00677, -0.01098), "0.487612 0.512088 -0.512088 0.487612", 0.17, (0.000132176, 9.3e-05, 5.85236e-05))}
                            <joint name="joint7" class="size3"/>
                            <geom material="gray" mesh="end_tool"/>
                            <site name="attachment_site" size="{_f(site_sz)}" rgba="0 1 0 1"/>
                          </body>
                        </body>
                      </body>
                    </body>
                  </body>
                </body>
              </body>
            </body>
"""


def main() -> None:
    s = XARM_LENGTH_SCALE
    # Pocket offset was tuned as child of r_hip_yaw; parent on torso = hip offset + same local offset.
    mbx, mby, mbz = 0.0, -0.055 * s, 0.01 * s
    base_torso = (mbx, AINEX_R_HIP_YAW_POS_Y + mby, mbz)
    xarm_subtree = build_xarm_subtree(s, base_torso)
    xarm_meshes = build_xarm_assets(s)
    xarm_defaults_inner = build_xarm_defaults(s)

    ainex_path = REPO / "assets" / "ainex" / "ainex_stable.xml"
    text = ainex_path.read_text()

    old_asset = """  <asset>
    <mesh name="body_link" content_type="model/stl" file="meshes/body_link.STL"/>"""
    new_asset = f"""  <asset>
    <material name="white" rgba="1 1 1 1"/>
    <material name="gray" rgba="0.753 0.753 0.753 1"/>
    <mesh name="body_link" content_type="model/stl" file="../../assets/ainex/meshes/body_link.STL"/>"""
    text = text.replace(old_asset, new_asset)
    text = text.replace('file="meshes/', 'file="../../assets/ainex/meshes/')

    insert_after = '<mesh name="l_gripper_link" content_type="model/stl" file="../../assets/ainex/meshes/l_gripper_link.STL"/>'
    text = text.replace(insert_after, insert_after + xarm_meshes)

    xarm_default_block = (
        "<default>\n    <joint damping=\"2\" armature=\"0.02\" limited=\"true\"/>",
        "<default>\n    <joint damping=\"18\" armature=\"0.02\" limited=\"true\" frictionloss=\"0.8\"/>\n"
        + xarm_defaults_inner
        + "\n",
    )
    text = text.replace(xarm_default_block[0], xarm_default_block[1])

    text = text.replace('<compiler angle="radian"/>', '<compiler angle="radian" autolimits="true"/>')
    text = text.replace(
        '<option timestep="0.002" gravity="0 0 -9.81" iterations="50" tolerance="1e-8" solver="Newton"/>',
        '<option timestep="0.002" gravity="0 0 -9.81" iterations="80" tolerance="1e-8" solver="Newton" integrator="implicitfast"/>',
    )

    text = text.replace(
        "    <!-- floor (currently not being used) -->\n"
        "    <!-- <geom name=\"floor\" type=\"plane\" pos=\"0 0 -0.05\" size=\"5 5 0.1\" rgba=\"0.9 0.9 0.9 1\" friction=\"1.2 0.3 0.3\"/> -->",
        '    <geom name="floor" type="plane" pos="0 0 0" size="5 5 0.1" rgba="0.92 0.92 0.92 1" friction="1.2 0.3 0.3"/>',
    )
    text = text.replace(
        '<body name="torso" pos="0 0 0">',
        f'<body name="torso" pos="0 0 {TORSO_Z}">',
    )

    # Mount xArm on torso (same world pose as old hip-yaw child) so leg/hip DOFs do not carry arm wrench.
    needle_torso = """      <geom type="mesh" contype="0" conaffinity="0" rgba="0.1 0.1 0.1 1" mesh="body_link"/>
      <body name="r_hip_yaw_link\""""
    repl_torso = """      <geom type="mesh" contype="0" conaffinity="0" rgba="0.1 0.1 0.1 1" mesh="body_link"/>
""" + xarm_subtree + """
      <body name="r_hip_yaw_link\""""
    if needle_torso not in text:
        raise SystemExit("torso insertion point not found")
    text = text.replace(needle_torso, repl_torso, 1)

    dhip = RIGHT_HIP_JOINT_DAMPING
    text = text.replace(
        '<joint name="r_hip_yaw" pos="0 0 0" axis="0 0 -1" range="-2.09 2.09" actuatorfrcrange="-6 6" damping="0.02"/>',
        f'<joint name="r_hip_yaw" pos="0 0 0" axis="0 0 -1" range="-2.09 2.09" actuatorfrcrange="-6 6" damping="{dhip}"/>',
        1,
    )
    text = text.replace(
        '<joint name="r_hip_roll" pos="0 0 0" axis="-1 0 0" range="-2.09 2.09" actuatorfrcrange="-6 6" damping="0.02"/>',
        f'<joint name="r_hip_roll" pos="0 0 0" axis="-1 0 0" range="-2.09 2.09" actuatorfrcrange="-6 6" damping="{dhip}"/>',
        1,
    )
    text = text.replace(
        '<joint name="r_hip_pitch" pos="0 0 0" axis="0 -1 0" range="-2.09 2.09" actuatorfrcrange="-6 6" damping="0.02"/>',
        f'<joint name="r_hip_pitch" pos="0 0 0" axis="0 -1 0" range="-2.09 2.09" actuatorfrcrange="-6 6" damping="{dhip}"/>',
        1,
    )

    text = text.replace(
        '<geom type="mesh" contype="0" conaffinity="0" rgba="0.1 0.1 0.1 1" mesh="r_gripper_link"/>\n'
        "              </body>",
        '<geom type="mesh" contype="0" conaffinity="0" rgba="0.1 0.1 0.1 1" mesh="r_gripper_link"/>\n'
        f'                <site name="r_gripper_tip" pos="{GRIPPER_TIP_POS}" size="{GRIPPER_TIP_SIZE}" rgba="1 0.2 0.2 1"/>\n'
        "              </body>",
        1,
    )

    text = text.replace(
        "    </body>\n  </worldbody>",
        f"""    </body>

    <body name="ball" pos="{BALL_SPAWN_XYZ}">
      <freejoint/>
      <geom name="ball_geom" type="sphere" size="0.03" rgba="0.9 0.2 0.2 1" mass="0.08" contype="1" conaffinity="1"/>
    </body>
  </worldbody>""",
    )

    text = text.replace('<mujoco model="ainex">', '<mujoco model="ainex_hip_xarm">')

    act_start = text.find("  <actuator>")
    act_end = text.find("</actuator>", act_start) + len("</actuator>")
    xarm_act = """  <actuator>
    <general name="act1" joint="joint1" class="size1"/>
    <general name="act2" joint="joint2" class="size1" ctrlrange="-2.059 2.0944"/>
    <general name="act3" joint="joint3" class="size2"/>
    <general name="act4" joint="joint4" class="size2" ctrlrange="-0.19198 3.927"/>
    <general name="act5" joint="joint5" class="size2"/>
    <general name="act6" joint="joint6" class="size3" ctrlrange="-1.69297 3.14159"/>
    <general name="act7" joint="joint7" class="size3"/>
  </actuator>"""
    text = text[:act_start] + xarm_act + text[act_end:]

    eq_block = f"""
  <equality>
    <!-- link7 welded to {XARM_WELD_BODY2} (wrist / hand base); identity relpose at compile pose. -->
    <weld name="xarm_wrist" body1="link7" body2="{XARM_WELD_BODY2}" relpose="0 0 0 1 0 0 0"
      solref="{WELD_SOLREF}" solimp="{WELD_SOLIMP}"/>
  </equality>
"""
    text = text.replace("</mujoco>", eq_block + "\n</mujoco>")

    old_key = text[text.find("  <keyframe>") : text.find("</keyframe>") + len("</keyframe>")]
    text = text.replace(old_key, "")

    out = ROOT / "models" / "hip_reach.xml"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    print("Wrote", out.resolve(), f"(xArm length scale={s})")
    print("This script only updates the XML on disk; it does not open a viewer.")
    print("To see the scene: python3 -m hip.run_demo")
    print("  (on macOS use: .venv/bin/mjpython -m hip.run_demo)")
    print("RL (PPO): python3 -m hip.train_hip_reach  then  mjpython -m hip.run_policy --model policies/hip_reach_ppo_50k.zip")


if __name__ == "__main__":
    main()
