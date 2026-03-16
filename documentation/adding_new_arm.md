# Adding a New Arm (Upload → Train → Run)

Use this pipeline to add your own arm, train a reach policy, and run the simulation. No code changes are required for most arms—only file placement and optional config.

---

## Pipeline overview

| Step | What you do | Required? |
|------|-------------|-----------|
| 1. Upload | Put the arm MJCF in the right folder | Yes |
| 2. Train | Run the train script with your `arm_id` | Yes |
| 3. Run | Run the simulation script with the same `arm_id` | Yes |
| 4. Registry | Add an entry in `arm_registry.py` (EE site, reach, keyframe) | Only if auto-discovery isn’t enough |
| 5. Overrides | Tweak `config/arm_overrides.yaml` if behavior is off | Only if needed |

---

## What you provide vs what the system adds

| You provide | System adds (do not create these) |
|-------------|-----------------------------------|
| One arm MJCF file (and optional `assets/`) in `scenes/arms/models/arms/<arm_id>/` | **Floor** — checkered plane is included automatically. |
| End-effector `<site>` in your arm XML | **Ball** — red target sphere; position is set each episode within the arm’s reach. |
| Optional: registry entry, overrides | **Scene composition** — your arm is included in a scene that already has floor + ball. |

You **never** need to create `floor.xml` or `ball.xml`; they live under `scenes/arms/models/` and are composed with your arm at load time.

---

## 1. Upload the arm

Put your arm’s MuJoCo MJCF (and any assets it needs) in:

```text
scenes/arms/models/arms/<arm_id>/
```

- **`<arm_id>`** is a short id you choose (e.g. `my_robot`, `custom_ur5`). You’ll use it in `train` and `run`.
- **Main XML** must be named one of: **`arm.xml`**, **`<arm_id>.xml`** (e.g. `my_robot.xml`), or **`scene.xml`**.

**MJCF requirements for training:**

- An **end-effector site** at the arm tip (see example below). The system looks for a `<site>` named (or containing) one of: `eetip`, `hand`, `gripper`, `attachment`, `pin_site`, `tool0`, `ee`, `ee_site`, `end_effector`. If your site has another name, add a [registry entry](#4-optional-registry-entry) and set `ee_site_name`.
- **Actuators** and **joints** defined as usual; the pipeline infers DOF and control from the MJCF.

You can copy an arm from [MuJoCo Menagerie](https://mujoco.readthedocs.io/en/stable/models.html) (e.g. Arms) into `arms/<arm_id>/` and use it as-is.

---

## Examples: Everything you need to upload

### Folder structure

Create a folder named with your arm id. Put **only** the arm model (and its assets) there. Example for arm id **`my_robot`**:

```text
reach/
  scenes/
    arms/
      models/
        arms/
          my_robot/              ← you create this folder (name = arm_id)
            my_robot.xml         ← required: main arm MJCF (or arm.xml / scene.xml)
            assets/              ← optional: meshes, textures (if your XML references them)
              link0.stl
              link1.stl
```

- **`my_robot/`** — Directory name is the **arm_id** you will pass to `train my_robot` and `run my_robot`.
- **`my_robot.xml`** — The main MuJoCo file. Can be named `arm.xml` or `scene.xml` instead.
- **`assets/`** — Optional. If your XML uses `<mesh file="assets/link0.stl"/>`, put the files here; paths are relative to the arm directory.

---

### Arm MJCF template (copy and fill out)

Copy the template below into **`scenes/arms/models/arms/<arm_id>/<arm_id>.xml`** (or `arm.xml`). Replace every **PLACEHOLDER** with your values; the table underneath explains each one. For arms with more than two links, duplicate the link/joint/actuator pattern.

**Template:**

```xml
<mujoco model="ARM_ID">
  <compiler angle="degree" coordinate="local" />
  <option timestep="0.01" gravity="0 0 -9.81"/>

  <worldbody>
    <body name="base" pos="BASE_X BASE_Y BASE_Z">
      <geom type="cylinder" size="BASE_RADIUS BASE_HALFHEIGHT" rgba="0.4 0.4 0.4 1"/>

      <body name="link1" pos="0 0 0">
        <joint name="joint1" type="hinge" axis="AXIS_1_X AXIS_1_Y AXIS_1_Z" range="JOINT1_MIN JOINT1_MAX" damping="0.05"/>
        <geom type="capsule" fromto="0 0 0 LINK1_X LINK1_Y LINK1_Z" size="LINK1_RADIUS" rgba="0.2 0.6 0.9 1"/>

        <body name="link2" pos="LINK1_X LINK1_Y LINK1_Z">
          <joint name="joint2" type="hinge" axis="AXIS_2_X AXIS_2_Y AXIS_2_Z" range="JOINT2_MIN JOINT2_MAX" damping="0.05"/>
          <geom type="capsule" fromto="0 0 0 LINK2_X LINK2_Y LINK2_Z" size="LINK2_RADIUS" rgba="0.2 0.9 0.4 1"/>
          <site name="EE_SITE_NAME" pos="LINK2_X LINK2_Y LINK2_Z" size="0.02" rgba="1 0 0 1"/>
        </body>
      </body>
    </body>
  </worldbody>

  <actuator>
    <motor name="motor1" joint="joint1" ctrlrange="-1 1" gear="1"/>
    <motor name="motor2" joint="joint2" ctrlrange="-1 1" gear="1"/>
  </actuator>

  <keyframe>
    <key name="home" qpos="QPOS1 QPOS2" ctrl="0 0"/>
  </keyframe>
</mujoco>
```

**How to fill out the placeholders:**

| Placeholder | Replace with | Example |
|-------------|--------------|---------|
| **ARM_ID** | Your arm id (same as the folder name). | `my_robot` |
| **BASE_X BASE_Y BASE_Z** | Position of the base in the world (meters). | `0 0 0.02` |
| **BASE_RADIUS BASE_HALFHEIGHT** | Cylinder size for the base (radius, half-height). | `0.03 0.02` |
| **AXIS_1_X AXIS_1_Y AXIS_1_Z** | Joint 1 rotation axis (unit vector). Z = rotate in XY plane. | `0 0 1` |
| **JOINT1_MIN JOINT1_MAX** | Joint 1 limits in **degrees** (because `angle="degree"`). | `-90 90` |
| **LINK1_X LINK1_Y LINK1_Z** | Endpoint of link 1 (start of link 2). | `0.2 0 0` |
| **LINK1_RADIUS** | Capsule radius for link 1 (meters). | `0.02` |
| **AXIS_2_X AXIS_2_Y AXIS_2_Z** | Joint 2 rotation axis. | `0 0 1` |
| **JOINT2_MIN JOINT2_MAX** | Joint 2 limits in degrees. | `-90 90` |
| **LINK2_X LINK2_Y LINK2_Z** | Endpoint of link 2 = EE position. | `0.2 0 0` |
| **LINK2_RADIUS** | Capsule radius for link 2. | `0.02` |
| **EE_SITE_NAME** | Name of the end-effector site. Use one of: `ee_site`, `eetip`, `hand`, `gripper`, `attachment`, `pin_site`, `tool0`, `ee`, `end_effector` (or add your name in the registry). | `ee_site` |
| **QPOS1 QPOS2** | Joint positions at “home” (degrees), one number per joint. | `0 0` |

**Notes:**

- **More joints:** Add more `<body>`, `<joint>`, `<geom>`, and one `<motor>` per joint; extend `<key name="home" qpos="..."/>` with one value per joint.
- **Different geometry:** You can replace `<geom type="capsule" .../>` with `type="box"`, `type="sphere"`, or `type="mesh"` (and add `<asset><mesh .../></asset>`). The **site** must stay at the arm tip.
- **Meshes:** If you use `<geom type="mesh" mesh="..."/>`, add `<compiler meshdir="assets"/>` and put mesh files in **`scenes/arms/models/arms/<arm_id>/assets/`**.

**Filled-out example** (2-link arm, 0.2 m per link, joints ±90°):

```xml
<mujoco model="my_robot">
  <compiler angle="degree" coordinate="local" />
  <option timestep="0.01" gravity="0 0 -9.81"/>
  <worldbody>
    <body name="base" pos="0 0 0.02">
      <geom type="cylinder" size="0.03 0.02" rgba="0.4 0.4 0.4 1"/>
      <body name="link1" pos="0 0 0">
        <joint name="joint1" type="hinge" axis="0 0 1" range="-90 90" damping="0.05"/>
        <geom type="capsule" fromto="0 0 0 0.2 0 0" size="0.02" rgba="0.2 0.6 0.9 1"/>
        <body name="link2" pos="0.2 0 0">
          <joint name="joint2" type="hinge" axis="0 0 1" range="-90 90" damping="0.05"/>
          <geom type="capsule" fromto="0 0 0 0.2 0 0" size="0.02" rgba="0.2 0.9 0.4 1"/>
          <site name="ee_site" pos="0.2 0 0" size="0.02" rgba="1 0 0 1"/>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor name="motor1" joint="joint1" ctrlrange="-1 1" gear="1"/>
    <motor name="motor2" joint="joint2" ctrlrange="-1 1" gear="1"/>
  </actuator>
  <keyframe>
    <key name="home" qpos="0 0" ctrl="0 0"/>
  </keyframe>
</mujoco>
```

---

### Optional: Registry entry example

Use this only if auto-discovery is not enough (e.g. your EE site has a custom name, or you want fixed reach limits).

**File:** `scenes/arms/arm_registry.py` — add an entry to the **`ARM_REGISTRY`** dict:

```python
"my_robot": ArmConfig(
    name="My Robot Arm",                    # Display name (optional, for logs)
    arm_path="arms/my_robot/my_robot.xml",  # Path to your arm XML, relative to scenes/arms/models/
    ee_site_name="ee_site",                 # End-effector site name in your MJCF (if not auto-detected)
    reach_min=0.15,                         # Min ball distance in meters (inner workspace bound)
    reach_max=0.85,                         # Max ball distance in meters (outer workspace bound)
    home_keyframe_name="home",              # Keyframe name used at episode reset
),
```

| Field | Explanation |
|-------|-------------|
| **`name`** | Human-readable label; used in messages and docs. |
| **`arm_path`** | Relative to `scenes/arms/models/`. Must point to your main arm XML. |
| **`ee_site_name`** | The `<site name="...">` at the arm tip. Omit if the name is one of the auto-detected ones (e.g. `ee_site`, `eetip`, `hand`). |
| **`reach_min`** | Minimum horizontal distance (m) for ball placement. Ball is never closer than this. |
| **`reach_max`** | Maximum horizontal distance (m) for ball placement. Omit to let the system infer from the model. |
| **`home_keyframe_name`** | Name of the `<key name="...">` used at reset. Default is `"home"`. |

---

### Optional: Per-arm overrides example

Use this if the arm trains but behaves poorly (erratic, folding, bad start pose).

**File:** `config/arm_overrides.yaml` — add a section keyed by your **arm_id**:

```yaml
# Per-arm overrides. Keys here override config/arms.yaml for this arm only.
my_robot:
  reach_max_cap: 0.5              # Cap max ball distance (m); lower = easier targets
  reach_min_mode: registry        # Use registry reach_min (vs "auto" from model)
  initial_pose: random            # "home" = use keyframe; "random" = sample joint limits
  initial_keyframe: home          # Keyframe name for reset (if not "home")
  ball_mode: shared               # "shared" = one ball; "per_arm" = one ball per arm (bimanual)
  joint_limit_margin_penalty: 0.02 # Penalty when joints are near limits (reduces self-collision)
  model_path: null                # Optional: full path to a pre-composed scene XML
```

| Key | Explanation |
|-----|-------------|
| **`reach_max_cap`** | Upper limit (m) on how far the ball is placed. Lower values make the task easier. |
| **`reach_min_mode`** | `"auto"` = infer from model; `"registry"` = use registry `reach_min` (and fraction/floor if set). |
| **`initial_pose`** | `"home"` = use keyframe at reset; `"random"` = randomize joint positions within limits (useful for bimanual). |
| **`initial_keyframe`** | Name of the keyframe used when `initial_pose` is `"home"`. |
| **`ball_mode`** | `"shared"` = one ball for all arms; `"per_arm"` = one ball per arm (for dual-arm). |
| **`joint_limit_margin_penalty`** | Scalar penalty when joints are within a margin of their limits; helps avoid folding. |
| **`model_path`** | Optional path to a full scene XML; usually leave `null` so the system composes floor + ball + your arm. |

---

**Checklist — minimum to upload:**

1. Create folder: **`scenes/arms/models/arms/<arm_id>/`**
2. Add one XML file named **`arm.xml`**, **`<arm_id>.xml`**, or **`scene.xml`** containing:
   - A **`<worldbody>`** with your arm’s bodies and **`<joint>`**s
   - At least one **`<site>`** at the arm tip with a name like **`ee_site`**, **`eetip`**, **`hand`**, **`tool0`**, etc.
   - **`<actuator>`** with one motor (or similar) per controlled joint
   - Optional but recommended: **`<keyframe>`** with **`<key name="home" .../>`**
3. If your XML references meshes, add an **`assets/`** subfolder and put the files there.
4. Run **`train <arm_id>`** then **`run <arm_id>`** from the project root.

---

## 2. Train

From the **project root** (with venv activated).

**Simple commands** (if you ran `pip install -e .`):

```bash
train my_robot              # default 300k steps
train my_robot 500000       # optional: steps
```

**Script form:**

```bash
python scripts/train.py --arm-id my_robot
python scripts/train.py --arm-id my_robot --steps 500000
```

This saves the policy to `policies/ppo_arms_<arm_id>_mac_300k.zip` (or the step count you used).

---

## 3. Run the simulation

**Simple commands** (if you ran `pip install -e .`):

```bash
run my_robot                # macOS uses mjpython for viewer automatically
run my_robot 10000         # optional: max steps in viewer
```

**Script form:** macOS: `mjpython scripts/run.py --arm-id my_robot` · Windows/Linux: `python scripts/run.py --arm-id my_robot`

The viewer opens with your arm reaching for a ball. Close the viewer to exit.

If you trained with a different step count, either use the same steps in config so the default policy path matches, or pass the policy file: `run my_robot` (or `--model policies/ppo_arms_my_robot_mac_500k.zip`).

---

## 4. Optional: Registry entry

Auto-discovery infers EE site, reach, and DOF from the MJCF. If you need to **set EE site name**, **reach limits**, or **home keyframe** explicitly, add an entry in `scenes/arms/arm_registry.py`:

```python
# In ARM_REGISTRY dict:
"my_robot": ArmConfig(
    name="My Robot Arm",
    arm_path="arms/my_robot/my_robot.xml",
    ee_site_name="tool0",           # optional; discovery will guess if omitted
    reach_min=0.15,                 # optional; inner ball distance (m)
    reach_max=0.85,                 # optional; outer ball distance (m)
    home_keyframe_name="home",      # optional; keyframe used at reset
),
```

- **`arm_path`**: Path to the main arm XML, relative to `scenes/arms/models/`.
- **`ee_site_name`**: Only if your EE site isn’t one of the auto-detected names.
- **`reach_min` / `reach_max`**: Ball sampling range in meters. Omit to use values inferred from the model.
- **`home_keyframe_name`**: Keyframe name in your MJCF used for reset. Omit to use `"home"` or qpos0.

After adding the entry, use the same [train](#2-train) and [run](#3-run-the-simulation) commands with `--arm-id my_robot`.

---

## 5. Optional: Per-arm overrides

If the arm trains but behaves poorly (erratic motion, folding, bad start pose), add a section in **`config/arm_overrides.yaml`** for your `arm_id`:

```yaml
my_robot:
  reach_max_cap: 0.5              # Limit max ball distance (m); lower = easier
  reach_min_mode: registry        # Use registry reach_min
  initial_pose: random             # Randomize start pose each episode
  joint_limit_margin_penalty: 0.02 # Penalize joints near limits (avoid self-collision)
```

**When to use what:**

| Problem | Override | Typical value |
|--------|----------|----------------|
| Erratic motion (circles, close-then-far) | `reach_max_cap` | `0.5` |
| Bimanual arms start too close | `initial_pose` | `random` |
| Two arms fighting for one ball | `ball_mode` | `per_arm` |
| Arm folds into itself / self-collision | `joint_limit_margin_penalty` | `0.015`–`0.03` |
| Ball too far, policy gives up | `reach_max_cap` | Lower than default |
| Different home pose | `initial_keyframe` | Keyframe name in your XML |

Full override keys are listed in the same file and in the “Override Keys Reference” section below.

---

## Cheat sheet (new arm)

```bash
# 1. Put MJCF in scenes/arms/models/arms/<arm_id>/ (arm.xml or <arm_id>.xml or scene.xml)
# 2. Train (after  pip install -e .)
train <arm_id> [steps]

# 3. Run
run <arm_id> [steps]
```

Or without install: `python scripts/train.py --arm-id <arm_id>`, then `mjpython scripts/run.py --arm-id <arm_id>` (macOS) or `python scripts/run.py --arm-id <arm_id>` (Windows/Linux).

Optional: add an entry in `scenes/arms/arm_registry.py` and/or overrides in `config/arm_overrides.yaml`.

---

## Override keys reference

**Scene / reset:**

- `initial_pose`: `"home"` or `"random"`
- `initial_keyframe`: keyframe name for reset
- `ball_mode`: `"shared"` or `"per_arm"`
- `model_path`: full scene XML path (optional)

**Train / ball placement:**

- `reach_min_mode`: `"auto"` or `"registry"`
- `reach_max_cap`: max ball distance (m)
- `joint_limit_margin_penalty`: float or omit to disable

---

## See also

- **Quick start (existing arms):** [QUICKSTART_ARMS.md](../QUICKSTART_ARMS.md)
- **Config:** `config/arms.yaml`, `config/arm_overrides.yaml`
- **Registry:** `scenes/arms/arm_registry.py`
- **Arms README:** `scenes/arms/README.md`
