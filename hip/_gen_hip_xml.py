#!/usr/bin/env python3
"""Generate hip/models/hip_reach.xml (run from repo root or hip/)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent

XARM_SUBTREE = """
            <body name="link_base" pos="0 -0.055 0.01" quat="0.6532815 0.2705981 -0.6532815 0.2705981" childclass="xarm7">
              <inertial pos="-0.021131 -0.0016302 0.056488" quat="0.696843 0.20176 0.10388 0.680376" mass="0.88556"
                diaginertia="0.00382023 0.00335282 0.00167725"/>
              <geom mesh="link_base"/>
              <body name="link1" pos="0 0 0.267">
                <inertial pos="-0.0002 0.02905 -0.01233" quat="0.978953 -0.202769 -0.00441617 -0.0227264" mass="2.382"
                  diaginertia="0.00569127 0.00533384 0.00293865"/>
                <joint name="joint1" class="size1"/>
                <geom mesh="link1"/>
                <body name="link2" quat="1 -1 0 0">
                  <inertial pos="0.00022 -0.12856 0.01735" quat="0.50198 0.86483 -0.00778841 0.00483285" mass="1.869"
                    diaginertia="0.00959898 0.00937717 0.00201315"/>
                  <joint name="joint2" range="-2.059 2.0944" class="size1"/>
                  <geom mesh="link2"/>
                  <body name="link3" pos="0 -0.293 0" quat="1 1 0 0">
                    <inertial pos="0.0466 -0.02463 -0.00768" quat="0.913819 0.289775 0.281481 -0.0416455" mass="1.6383"
                      diaginertia="0.00351721 0.00294089 0.00195868"/>
                    <joint name="joint3" class="size2"/>
                    <geom mesh="link3"/>
                    <body name="link4" pos="0.0525 0 0" quat="1 1 0 0">
                      <inertial pos="0.07047 -0.11575 0.012" quat="0.422108 0.852026 -0.126025 0.282832" mass="1.7269"
                        diaginertia="0.00657137 0.00647948 0.00186763"/>
                      <joint name="joint4" range="-0.19198 3.927" class="size2"/>
                      <geom mesh="link4"/>
                      <body name="link5" pos="0.0775 -0.3425 0" quat="1 1 0 0">
                        <inertial pos="-0.00032 0.01604 -0.026" quat="0.999311 -0.0304457 0.000577067 0.0212082" mass="1.3203"
                          diaginertia="0.00534729 0.00499076 0.0013489"/>
                        <joint name="joint5" class="size2"/>
                        <geom mesh="link5"/>
                        <body name="link6" quat="1 1 0 0">
                          <inertial pos="0.06469 0.03278 0.02141" quat="-0.217672 0.772419 0.16258 0.574069" mass="1.325"
                            diaginertia="0.00245421 0.00221646 0.00107273"/>
                          <joint name="joint6" range="-1.69297 3.14159" class="size3"/>
                          <geom mesh="link6"/>
                          <body name="link7" pos="0.076 0.097 0" quat="1 -1 0 0">
                            <inertial pos="0 -0.00677 -0.01098" quat="0.487612 0.512088 -0.512088 0.487612" mass="0.17"
                              diaginertia="0.000132176 9.3e-05 5.85236e-05"/>
                            <joint name="joint7" class="size3"/>
                            <geom material="gray" mesh="end_tool"/>
                            <site name="attachment_site" size="0.002" rgba="0 1 0 1"/>
                          </body>
                        </body>
                      </body>
                    </body>
                  </body>
                </body>
              </body>
            </body>
"""

AINEX_TEMPLATE = REPO / "assets" / "ainex" / "ainex_stable.xml"
text = AINEX_TEMPLATE.read_text()

# Mesh paths relative to hip/models/hip_reach.xml
old_asset = """  <asset>
    <mesh name="body_link" content_type="model/stl" file="meshes/body_link.STL"/>"""
new_asset = f"""  <asset>
    <material name="white" rgba="1 1 1 1"/>
    <material name="gray" rgba="0.753 0.753 0.753 1"/>
    <mesh name="body_link" content_type="model/stl" file="../../assets/ainex/meshes/body_link.STL"/>"""
text = text.replace(old_asset, new_asset)
text = text.replace('file="meshes/', 'file="../../assets/ainex/meshes/')

# xarm meshes (append before </asset>, keep all ainex meshes)
insert_after = '<mesh name="l_gripper_link" content_type="model/stl" file="../../assets/ainex/meshes/l_gripper_link.STL"/>'
xarm_meshes = """
    <mesh name="link_base" content_type="model/stl" file="../../scenes/arms/models/arms/xarm7/assets/link_base.stl"/>
    <mesh name="link1" content_type="model/stl" file="../../scenes/arms/models/arms/xarm7/assets/link1.stl"/>
    <mesh name="link2" content_type="model/stl" file="../../scenes/arms/models/arms/xarm7/assets/link2.stl"/>
    <mesh name="link3" content_type="model/stl" file="../../scenes/arms/models/arms/xarm7/assets/link3.stl"/>
    <mesh name="link4" content_type="model/stl" file="../../scenes/arms/models/arms/xarm7/assets/link4.stl"/>
    <mesh name="link5" content_type="model/stl" file="../../scenes/arms/models/arms/xarm7/assets/link5.stl"/>
    <mesh name="link6" content_type="model/stl" file="../../scenes/arms/models/arms/xarm7/assets/link6.stl"/>
    <mesh name="link7" content_type="model/stl" file="../../scenes/arms/models/arms/xarm7/assets/link7.stl"/>
    <mesh name="end_tool" content_type="model/stl" file="../../scenes/arms/models/arms/xarm7/assets/end_tool.stl"/>"""
text = text.replace(insert_after, insert_after + xarm_meshes)

# Defaults: limp humanoid + xarm class
text = text.replace(
    "<default>\n    <joint damping=\"2\" armature=\"0.02\" limited=\"true\"/>",
    "<default>\n    <joint damping=\"18\" armature=\"0.02\" limited=\"true\" frictionloss=\"0.8\"/>\n"
    "    <default class=\"xarm7\">\n"
    "      <geom type=\"mesh\" material=\"white\"/>\n"
    "      <joint axis=\"0 0 1\" armature=\"0.1\" range=\"-6.28319 6.28319\" frictionloss=\"0.5\"/>\n"
    "      <general biastype=\"affine\" ctrlrange=\"-6.28319 6.28319\"/>\n"
    "      <default class=\"size1\">\n"
    "        <joint damping=\"8\"/>\n"
    "        <general gainprm=\"1500\" biasprm=\"0 -1500 -150\" forcerange=\"-50 50\"/>\n"
    "      </default>\n"
    "      <default class=\"size2\">\n"
    "        <joint damping=\"5\"/>\n"
    "        <general gainprm=\"1000\" biasprm=\"0 -1000 -100\" forcerange=\"-30 30\"/>\n"
    "      </default>\n"
    "      <default class=\"size3\">\n"
    "        <joint damping=\"3\"/>\n"
    "        <general gainprm=\"800\" biasprm=\"0 -800 -80\" forcerange=\"-20 20\"/>\n"
    "      </default>\n"
    "      <site size=\"0.001\" rgba=\"1 0 0 1\" group=\"4\"/>\n"
    "    </default>",
)

text = text.replace('<compiler angle="radian"/>', '<compiler angle="radian" autolimits="true"/>')
text = text.replace(
    '<option timestep="0.002" gravity="0 0 -9.81" iterations="50" tolerance="1e-8" solver="Newton"/>',
    '<option timestep="0.002" gravity="0 0 -9.81" iterations="80" tolerance="1e-8" solver="Newton" integrator="implicitfast"/>',
)

# Floor + torso height
text = text.replace(
    "    <!-- floor (currently not being used) -->\n"
    "    <!-- <geom name=\"floor\" type=\"plane\" pos=\"0 0 -0.05\" size=\"5 5 0.1\" rgba=\"0.9 0.9 0.9 1\" friction=\"1.2 0.3 0.3\"/> -->",
    '    <geom name="floor" type="plane" pos="0 0 0" size="5 5 0.1" rgba="0.92 0.92 0.92 1" friction="1.2 0.3 0.3"/>',
)
text = text.replace('<body name="torso" pos="0 0 0">', '<body name="torso" pos="0 0 1.05">')

# Insert xarm on right hip yaw link (hip assembly on torso side)
needle = """        <geom type="mesh" contype="0" conaffinity="0" rgba="0.1 0.1 0.1 1" mesh="r_hip_yaw_link"/>
        <body name="r_hip_roll_link\""""
repl = """        <geom type="mesh" contype="0" conaffinity="0" rgba="0.1 0.1 0.1 1" mesh="r_hip_yaw_link"/>
""" + XARM_SUBTREE + """
        <body name="r_hip_roll_link\""""
if needle not in text:
    raise SystemExit("insertion point not found")
text = text.replace(needle, repl, 1)

# EE site on Ainex hand
text = text.replace(
    '<geom type="mesh" contype="0" conaffinity="0" rgba="0.1 0.1 0.1 1" mesh="r_gripper_link"/>\n'
    "              </body>",
    '<geom type="mesh" contype="0" conaffinity="0" rgba="0.1 0.1 0.1 1" mesh="r_gripper_link"/>\n'
    '                <site name="r_gripper_tip" pos="0.03 0 0.01" size="0.008" rgba="1 0.2 0.2 1"/>\n'
    "              </body>",
    1,
)

# Ball (kinematic target; free joint for random placement)
text = text.replace(
    "    </body>\n  </worldbody>",
    """    </body>

    <body name="ball" pos="0.45 0 1.15">
      <freejoint/>
      <geom name="ball_geom" type="sphere" size="0.03" rgba="0.9 0.2 0.2 1" mass="0.08" contype="1" conaffinity="1"/>
    </body>
  </worldbody>""",
)

# Model name
text = text.replace('<mujoco model="ainex">', '<mujoco model="ainex_hip_xarm">')

# Replace actuators: only xarm
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

# Equality weld (soft-ish for closed chain)
eq_block = """
  <equality>
    <!-- Identity relpose: force link7 and Ainex forearm frames to coincide (not the default-qpos offset). -->
    <weld name="xarm_wrist" body1="link7" body2="r_el_yaw_link" relpose="0 0 0 1 0 0 0"
      solref="0.02 1" solimp="0.99 0.999 0.0001 0.5 2"/>
  </equality>
"""
text = text.replace("</mujoco>", eq_block + "\n</mujoco>")

# Keyframe: squat-ish ainex + xarm home — joint order will be validated by loader script
# Placeholder; tune after mj_load
old_key = text[text.find("  <keyframe>") : text.find("</keyframe>") + len("</keyframe>")]
text = text.replace(old_key, "")

out = ROOT / "models" / "hip_reach.xml"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(text)
print("Wrote", out)
