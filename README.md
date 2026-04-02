# REACH Capstone Project

## Project Overview

The **REACH Capstone Project** is a robotics simulation and reinforcement learning (RL) platform developed at **Northern Arizona University**.
It combines **MuJoCo physics simulation**, **PPO reinforcement learning**, and **Python-based visualization** to train robotic arms and other systems for precision control and interaction tasks.

The system runs cross-platform on **macOS (M-series)** and **Windows/Linux (x86 / CUDA)** with a unified configuration layer driven by a YAML file.

### Table of contents

- [Quick start: Arm reach](#quick-start-arm-reach-setup--train--run)
- [Arm Reach Training](#arm-reach-training-primary-workflow) — supported arms, train/run commands, config, policy paths
- [Adding a new arm (full guide)](#adding-a-new-arm-full-guide) — upload, template, registry, overrides
- [Downloading an arm from GitHub](#downloading-an-arm-from-github) — clone/ZIP, copy subtree, train/run
- [Adding a new leg (biped)](#adding-a-new-leg-biped)
- [YAML Configuration](#yaml-configuration)
- [AINex Humanoid](#ainex-humanoid-reach--stand)
- [Other Scenes](#other-scenes)
- [Training on NAU Monsoon HPC](#training-on-nau-monsoon-hpc)
- [System design and limitations](#system-design-and-limitations)
- [macOS / Windows / Linux](#macos-m-series-instructions)
- [Troubleshooting](#troubleshooting)

---

## Team Members

### Development Team

- **Taylor Davis** ([tjd352@nau.edu](mailto:tjd352@nau.edu))
  **Role:** Team Lead / Coder / Architect
  **Responsibilities:** Integration, simulation architecture, and research coordination

- **Victor Rodriguez** ([vr527@nau.edu](mailto:vr527@nau.edu))
  **Role:** Coder / Recorder / Architect
  **Responsibilities:** Documentation, model development, and codebase maintenance
  **Background:** U.S. Marine Corps veteran with leadership experience

- **Clayton Ramsey** ([car723@nau.edu](mailto:car723@nau.edu))
  **Role:** Coder / Architect
  **Responsibilities:** Environment structure, testing, and simulation support

- **Lucas Larson** ([lwl33@nau.edu](mailto:lwl33@nau.edu))
  **Role:** Coder / Version Control Manager / Architect
  **Responsibilities:** GitHub operations, merge reviews, and repository management

---

## Sponsors

- **Dr. Zach Lerner, Ph.D.**
  Associate Professor, Mechanical Engineering, NAU
  [biomech.nau.edu](https://biomech.nau.edu)

- **Prof. Carlo R. da Cunha, Ph.D.**
  Assistant Professor, Electrical Engineering, NAU
  [ac.nau.edu/~cc3682](https://ac.nau.edu/~cc3682)

---

## Technology Stack

- **Physics Engine:** [MuJoCo 3.x](https://mujoco.readthedocs.io/)
- **RL Framework:** [Stable-Baselines3 (PPO)](https://stable-baselines3.readthedocs.io/)
- **Deep Learning:** PyTorch 2.x
- **Visualization:** MuJoCo Viewer (passive or active)
- **Frontend:** HTML, CSS, Bootstrap 4
- **Supported Environments:** macOS ARM (M-series) and x86 / CUDA
- **Version Control:** Git + GitHub

---

## Quick start: Arm reach (setup → train → run)

Get a robotic arm training and running in the simulator in four steps.

### 1. Clone and go to the project

```bash
git clone <your-repo-url>
cd reach
```

### 2. Install (one time)

**macOS / Linux** (or run `./scripts/setup_arms.sh` from the repo root to do this automatically):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .    # installs the  train  and  run  commands (recommended)
```

**Windows (PowerShell):**

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -e .    # installs  train  and  run  commands
```

If you skip `pip install -e .`, you can still use the script form for everything: `python scripts/train.py --arm-id <arm_id>` and `mjpython scripts/run.py --arm-id <arm_id>` (macOS) or `python scripts/run.py --arm-id <arm_id>` (Windows/Linux). The rest of this section assumes you have the `train` and `run` commands installed.

### 3. Train (default: Panda arm, ~4–5 min)

From the **project root** with your venv activated:

```bash
train panda              # train Panda arm, 300k steps (default)
train ur5e               # train UR5e, 300k steps
train ur5e 500000        # train UR5e for 500k steps (optional second argument)
```

- **First argument:** `arm_id` — which arm to train (e.g. `panda`, `ur5e`, `z1`, `arm_2link`, `aloha`). See the [supported arms table](#supported-arms) below.
- **Second argument (optional):** total training steps. If omitted, the default is 300 000 (from `config/arms.yaml`).

Training saves a policy to `policies/ppo_arms_<arm_id>_mac_<k>k.zip` (e.g. `policies/ppo_arms_panda_mac_300k.zip` or `ppo_arms_ur5e_mac_500k.zip` if you used 500k steps).

**Using the script instead of `train`:**

```bash
python scripts/train.py --arm-id panda
python scripts/train.py --arm-id ur5e --steps 500000
```

### 4. Run the simulation

With a trained policy, open the MuJoCo viewer and run the policy:

```bash
run panda                # run Panda (loads default policy for 300k steps)
run ur5e                 # run UR5e
run ur5e 10000           # run UR5e, limit to 10 000 steps in the viewer (optional second argument)
```

- **First argument:** `arm_id` — must match the arm you trained (and the policy file name).
- **Second argument (optional):** maximum simulation steps in the viewer. If omitted, the default is 5000 (from config). The simulation resets when an episode ends; this cap is the total step count before the process exits.

**Platform behavior:** On **macOS**, the `run` command automatically uses **mjpython** so the MuJoCo passive viewer works with Metal. On **Windows and Linux**, it uses your current **python** (from the venv). No need to type `mjpython` yourself.

**Using the script instead of `run`:**

```bash
mjpython scripts/run.py --arm-id panda        # macOS
python scripts/run.py --arm-id panda          # Windows/Linux
python scripts/run.py --arm-id ur5e --steps 10000
```

The viewer opens with the arm reaching for a ball. Close the viewer window to exit.

---

**Other arms:** Use the same syntax with any supported `arm_id`: `train z1`, `run aloha`, etc. To **add your own arm** (custom MJCF), see [Upload another arm](#upload-another-arm) and the [Adding a new arm (full guide)](#adding-a-new-arm-full-guide) below.

---

## AINex Soccer Assets (from [ainex_soccer](https://github.com/tjdavis51/ainex_soccer))

This repo includes MuJoCo assets and action groups from the AINex Soccer project:

- **assets/ainex/** – Robot meshes (STL), URDF, and MJCF models (`ainex_stable.xml`, `ainex_edited.urdf`)
- **assets/action_groups/raw/** – Original hardware action groups (`.d6a` SQLite databases)
- **assets/action_groups/csv/** – CSV exports of action groups (Servo1–Servo22 + timing)
- **scripts/** – Viewers and replay tools (`view_ainex_stable.py`, `replay_actiongroup.py`, etc.)
- **tools/** – Conversion utilities (`extract_actiongroup_to_csv.py`, `export_ainex_mjcf.py`)

### Quick start (AINex Soccer scripts)

```bash
pip install mujoco numpy
mjpython scripts/view_ainex_stable.py
```

View model (no physics): `mjpython scripts/view_ainex_stable.py`. Replay action groups: `mjpython scripts/replay_actiongroup.py` (default CSV) or `mjpython scripts/replay_actiongroup.py assets/action_groups/csv/wave.csv`; choreographed sequence: `mjpython tools/run_sequence.py`. AINex reach training and eval are under [AINex Humanoid](#ainex-humanoid-reach--stand).

---

## Project Structure

```
reach/
├── README.md
├── requirements.txt
├── pyproject.toml               # Packaging + CLI entry points
│
├── assets/                     # AINex Soccer assets (meshes, action groups)
│   ├── ainex/                  # Robot models, URDF, MJCF, meshes
│   └── action_groups/          # raw (.d6a) and csv exports
├── config/
│   ├── arms.yaml               # Arm reach: arm_id, ball_mode, train/run settings
│   ├── arm_overrides.yaml      # Per-arm overrides (reach, initial pose, penalties)
│   ├── ainex_render.yaml       # AINex stand config
│   ├── ainex_reach.yaml        # AINex reach config
│   └── arms_loader.py          # Loads arms.yaml, resolves policy paths
│
├── scripts/                    # Default entry points
│   ├── train.py                # Train arm reach (delegates to arm_train_mac)
│   └── run.py                  # Run arm policy with viewer (delegates to run_simulation)
├── policies/                   # Trained arm policies (ppo_arms_<arm_id>_mac_<k>k.zip)
│
├── cluster/
│   ├── train_monsoon.sh        # SLURM job for Monsoon HPC (auto-push on success)
│   └── test_monsoon.sh         # Smoke test (local or Monsoon)
├── tests/
│   └── smoke_test.py           # Quick sanity check (imports, env, PPO)
├── documentation/
│   ├── monsoon_setup.md        # Monsoon HPC setup and workflow
│   ├── demos/
│   ├── headshots/
│   ├── logos/
│   └── system_design.md        # Architecture, assumptions, limitations
│
├── envs/                       # Shared environments
│   └── mujoco_arm_env.py
│
├── tools/                      # AINex Soccer conversion tools
├── renders/                    # Rendering and visualization scripts
│   ├── render_demo.py          # x86 / CUDA policy renderer
│   ├── render_demo_mac.py      # macOS policy renderer (uses mjpython)
│   ├── render_model.py         # x86 / CUDA model viewer
│   └── render_model_mac.py     # macOS model viewer (uses mjpython)
│
├── scenes/
│   ├── arms/                   # Arm-only reach (YAML-driven, auto-discovery)
│   │   ├── env.py              # ArmReachEnv
│   │   ├── arm_registry.py     # arm_id → config (ee_site, reach, etc.)
│   │   ├── arm_discovery.py    # Auto-infer from MJCF
│   │   ├── scene_compose.py    # Compose floor + ball + arm
│   │   ├── models/arms/        # MuJoCo Menagerie arms (panda, ur5e, aloha, etc.)
│   │   └── training/           # arm_train_mac, run_simulation, eval_model
│   ├── ainex_soccer/           # AINex humanoid scene + training
│   │   ├── env.py
│   │   ├── models/
│   │   ├── policies/
│   │   └── training/
│   ├── industrial_arm_reaching/
│   │   ├── env.py
│   │   ├── models/
│   │   ├── policies/
│   │   └── training/
│   ├── cartpole/
│   └── industrial_arm_reaching_with_welding/
│
├── logs/                       # Generated training/eval logs
├── website/
│   ├── index.html
│   ├── team.html
│   ├── project.html
│   ├── documents.html
│   └── assets/
│
└── .venv/                      # Virtual environment (ignored by Git)
```

---

## System Design and Limitations

See [System design and limitations](#system-design-and-limitations) in this README for architecture, assumptions, limitations, and sim-to-real risks.

---

## Setting Up the Environment

### 1. Clone the repository

```bash
git clone -b tdev https://github.com/lucaslarson25/reach.git
cd reach
```

### 2. Create and activate a virtual environment

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows (PowerShell)**

```powershell
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
pip install -e .    # installs  train  and  run  commands for arm reach (recommended)
```

If you use `pip install -e .`, you can then use **`train <arm_id> [steps]`** and **`run <arm_id> [steps]`** from the project root (see [Quick start: Arm reach](#quick-start-arm-reach-setup--train--run) and [Train and run commands](#train-and-run-commands-recommended)).

### 4. Verify installation

```bash
python -c "import mujoco; import torch; print('Setup complete.')"
```

### 5. Run smoke test (local or Monsoon)

```bash
python -m tests.smoke_test
```

Works headless (no display). On Monsoon, use `./cluster/test_monsoon.sh` or `sbatch cluster/test_monsoon.sh`.

---

## Arm Reach Training (Primary Workflow)

The **arm reach** scene trains robotic arms to reach a ball. You upload only the arm; the scene (floor + ball) is composed at load time. Training and ball sampling adapt to the arm's length and DOF.

### How It Works

- **YAML-driven:** `config/arms.yaml` defines default arm, ball mode, train/run settings.
- **Auto-discovery:** Arms from [MuJoCo Menagerie](https://mujoco.readthedocs.io/en/stable/models.html) are auto-detected; EE sites and reach are inferred from MJCF.
- **Simple CLI:** After `pip install -e .`, you use **`train <arm_id> [steps]`** and **`run <arm_id> [steps]`** for all arms—no YAML edits required.
- **Single workflow:** Same two commands for every arm (panda, ur5e, aloha, etc.).

### Supported arms {#supported-arms}

| arm_id     | Model                    |
|------------|---------------------------|
| arm_2link  | 2-link demo arm           |
| panda      | Franka Emika Panda        |
| fr3        | Franka FR3                |
| ur5e       | Universal Robots UR5e     |
| ur10e      | Universal Robots UR10e    |
| iiwa14     | KUKA LBR iiwa14           |
| xarm7      | UFACTORY xArm7            |
| sawyer     | Rethink Robotics Sawyer   |
| lite6      | UFactory Lite 6           |
| vx300s     | Trossen ViperX 300 6DOF   |
| wx250s     | Trossen WidowX 250 6DOF   |
| aloha      | ALOHA 2 (dual arm)        |
| unitree_z1, z1 | Unitree Z1            |

### Train and run commands (recommended)

If you ran **`pip install -e .`** in the project root (see [Quick start](#quick-start-arm-reach-setup--train--run)), two commands are installed: **`train`** and **`run`**. Use them from the project root with your venv activated.

**Train:**

```text
train <arm_id> [steps]
```

- **`arm_id`** — Which arm to train (e.g. `panda`, `ur5e`, `z1`, `aloha`). Required.
- **`steps`** — Total PPO timesteps. Optional; default is 300 000 from `config/arms.yaml`.

Examples:

```bash
train panda           # Panda, 300k steps
train ur5e 500000     # UR5e, 500k steps
train aloha           # ALOHA (dual arm), 300k steps
```

**Run (viewer):**

```text
run <arm_id> [steps]
```

- **`arm_id`** — Which arm (and thus which policy file) to load. Must match the arm you trained.
- **`steps`** — Maximum simulation steps in the viewer before exit. Optional; default is 5000.

Examples:

```bash
run panda             # Run Panda policy (default 5k steps in viewer)
run ur5e 10000        # Run UR5e, cap at 10k steps
run aloha             # Run ALOHA policy
```

**Policy file matching:** The run command looks for a policy at `policies/ppo_arms_<arm_id>_mac_<k>k.zip`, where `k` is the training step count in thousands (from config or from the `steps` you used when training). For example, `train ur5e 500000` produces `ppo_arms_ur5e_mac_500k.zip`; `run ur5e` then loads it if the config default is 300k, so for a 500k-trained policy either set `train.total_steps: 500000` in `config/arms.yaml` for that arm or use the script form with `--model policies/ppo_arms_ur5e_mac_500k.zip` (see below).

**macOS:** The `run` command automatically invokes **mjpython** so the MuJoCo passive viewer works. On Windows/Linux it uses **python** from your venv.

**Help:** Run **`train --help`** or **`run --help`** to see all options (including flags that are only available when using the script form).

**Command reference:**

| Command | Meaning |
|--------|--------|
| `train <arm_id>` | Train that arm with default steps (300k). |
| `train <arm_id> <steps>` | Train that arm for that many steps. |
| `run <arm_id>` | Run that arm’s policy in the viewer (default 5k steps). |
| `run <arm_id> <steps>` | Run that arm’s policy, cap viewer at that many steps. |

Always run from the **project root** with your **venv activated**.

### Script form (without `train` / `run` commands)

If you did not run `pip install -e .`, or you need extra flags (e.g. `--per-arm-policies`, `--model`), use the scripts directly from the project root:

```bash
# Train
python scripts/train.py --arm-id panda
python scripts/train.py --arm-id ur5e --steps 500000
python scripts/train.py --arm-id aloha --per-arm-policies

# Run (macOS: mjpython for viewer; Windows/Linux: python)
mjpython scripts/run.py --arm-id panda
mjpython scripts/run.py --arm-id ur5e --steps 10000
mjpython scripts/run.py --arm-id ur5e --model policies/ppo_arms_ur5e_mac_500k.zip
python scripts/run.py --arm-id panda   # Windows/Linux
```

**Observation and reward (arms):** Per arm: `qpos`, `qvel`, `ee_pos`, `ball_pos`, distance to ball, unit direction EE→ball, delta_distance. Reward: dense (closer = more), progress, orientation (EE toward ball), move-away penalty; success bonus within 5 cm. `ee_priority_scale: true` scales actions so joints near the end-effector have higher gain.

### Config (`config/arms.yaml`)

- **scene.arm_id** – which arm (e.g. `panda`, `ur5e`, `z1`, `aloha`)
- **scene.ball_mode** – `shared` (1 ball, all arms) or `per_arm` (N balls for N arms)
- **scene.per_arm_policies** – `true` = train/run separate policy per arm (multi-arm only)
- **train.total_steps** – default 300000
- **train.policy_dir** – where to save policies (default: `policies/`)
- **train.reward_style** – `z1` (industrial-style reward) or `arms`
- **train.reach_min_mode** – `auto` (infer ball range from arm model) or `registry`
- **train.ee_priority_scale** – `true` = scale actions so joints near EE move more (reduces getting stuck)
- **train.reward_time_penalty**, **train.reward_smoothness** – reward shaping
- **run.steps**, **run.deterministic**, **run.debug**, **run.stochastic**

Env vars **ARM_ID**, **TOTAL_STEPS**, **MODEL_PATH**, **USE_MPS** override YAML. CLI flags take precedence.

### Policy paths

Policies are saved under the `policies/` directory with the naming pattern **`ppo_arms_<arm_id>_mac_<k>k.zip`**, where `<k>` is the training step count in thousands (e.g. 300k → `300k`, 500k → `500k`). Examples: `ppo_arms_panda_mac_300k.zip`, `ppo_arms_ur5e_mac_500k.zip`.

When you use **`run <arm_id>`**, the run command infers the policy path from `arm_id` and the default step count in `config/arms.yaml` (`train.total_steps`). If you trained with a non-default step count (e.g. `train ur5e 500000`), either set `train.total_steps: 500000` in the config so `run ur5e` finds the 500k policy, or use the script form and pass the file explicitly: **`python scripts/run.py --arm-id ur5e --model policies/ppo_arms_ur5e_mac_500k.zip`** (or `mjpython` on macOS).

### Eval (metrics only)

```bash
python -m scenes.arms.training.eval_model --model policies/ppo_arms_panda_mac_300k.zip --arm-id panda
```

### Upload another arm

To add **your own** arm (or one from [MuJoCo Menagerie](https://mujoco.readthedocs.io/en/stable/models.html)):

1. **Upload** — Put the arm MJCF in **`scenes/arms/models/arms/<arm_id>/`**. Name the main file **`arm.xml`**, **`<arm_id>.xml`**, or **`scene.xml`**. Include an end-effector **`<site>`** at the arm tip (e.g. named `eetip`, `hand`, `tool0`, or `attachment_site`).
2. **Train** — From the project root: **`train <arm_id>`** or **`train <arm_id> 500000`** (optional steps). With scripts: `python scripts/train.py --arm-id <arm_id> [--steps 500000]`.
3. **Run** — **`run <arm_id>`** or **`run <arm_id> 10000`** (optional max viewer steps). With scripts: `mjpython scripts/run.py --arm-id <arm_id>` (macOS) or `python scripts/run.py --arm-id <arm_id>` (Windows/Linux).

**From GitHub:** If the MJCF lives in another repository, follow [Downloading an arm from GitHub](#downloading-an-arm-from-github) below, then train/run as above.

**Optional:** Add an entry in **`scenes/arms/arm_registry.py`** to set EE site name, reach limits, or home keyframe. If the arm behaves poorly (erratic motion, folding), add a section for your `arm_id` in **`config/arm_overrides.yaml`**. Full pipeline, template, and troubleshooting are in the section below.

---

## Adding a new arm (full guide)

Pipeline: **Upload → Train → Run**. No code changes required for most arms.

| Step | What you do | Required? |
|------|-------------|-----------|
| 1. Upload | Put the arm MJCF in the right folder | Yes |
| 2. Train | Run `train <arm_id> [steps]` | Yes |
| 3. Run | Run `run <arm_id> [steps]` | Yes |
| 4. Registry | Add an entry in `arm_registry.py` (EE site, reach, keyframe) | Only if auto-discovery isn’t enough |
| 5. Overrides | Tweak `config/arm_overrides.yaml` if behavior is off | Only if needed |

**What you provide vs what the system adds:** You provide one arm MJCF (and optional `assets/`) in `scenes/arms/models/arms/<arm_id>/`. The system adds the **floor** and **ball** automatically; you never create `floor.xml` or `ball.xml`.

### Downloading an arm from GitHub

Use this when the model comes from an external repo (e.g. [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) or a lab fork).

1. **Get the files** — Clone or download the repository so you have **everything** the MJCF references: the main `.xml` / `.mjcf`, all `<include>` targets, meshes (`.obj`, `.stl`, …), textures, etc. Missing assets will cause MuJoCo load errors.
   - `git clone https://github.com/OWNER/REPO.git`
   - `git clone --depth 1 https://github.com/OWNER/REPO.git` (shallow clone)
   - Or **Code → Download ZIP** on GitHub and unzip.
2. **Copy into REACH** — Create **`scenes/arms/models/arms/<arm_id>/`** (this folder name is your `--arm-id`). Copy the **entire** arm subtree from the upstream tree, preserving internal paths so `<include>` and `meshdir`/`assetdir` still resolve. Do not copy only the top-level XML unless it has no external assets.
3. **Meet discovery rules** — That folder must contain **`arm.xml`**, **`<arm_id>.xml`**, or **`scene.xml`** (see `scenes/arms/arm_discovery.py`). The model needs an **end-effector `<site>`** at the tip; auto-discovery matches names like `eetip`, `hand`, `gripper`, `attachment_site`, `tool0`, `ee`, `ee_site`, `end_effector`.
4. **Train and run** — From the repo root (venv active): **`train <arm_id>`** or **`train <arm_id> 1000000`**, then **`run <arm_id>`** (or **`run <arm_id> 10000`**). If your policy filename does not match `train.total_steps` in `config/arms.yaml`, pass the zip explicitly, e.g. **`mjpython scripts/run.py --arm-id <arm_id> --model policies/ppo_arms_<arm_id>_mac_1000k.zip`** (macOS).
5. **Licensing** — Respect the upstream license if you commit third-party models in your fork.
6. **If load or discovery fails** — Add or rename an EE `<site>` in the MJCF, add an entry in **`scenes/arms/arm_registry.py`**, or tune **`config/arm_overrides.yaml`** (see [Optional: Registry entry](#optional-registry-entry) and [Optional: Per-arm overrides](#optional-per-arm-overrides) below).

**Upload (naming):** Main XML must be named `arm.xml`, `<arm_id>.xml`, or `scene.xml`. Include an **end-effector `<site>`** at the arm tip; the system looks for names like `eetip`, `hand`, `gripper`, `attachment`, `pin_site`, `tool0`, `ee`, `ee_site`, `end_effector`. **Actuators** and **joints** as usual.

### Arm MJCF template (copy and fill out)

Save as **`scenes/arms/models/arms/<arm_id>/<arm_id>.xml`** (or `arm.xml`). Replace every **PLACEHOLDER**; for more joints, duplicate the link/joint/actuator pattern.

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

| Placeholder | Replace with | Example |
|-------------|--------------|---------|
| **ARM_ID** | Your arm id (folder name). | `my_robot` |
| **BASE_X BASE_Y BASE_Z** | Base position (m). | `0 0 0.02` |
| **BASE_RADIUS BASE_HALFHEIGHT** | Base cylinder size. | `0.03 0.02` |
| **AXIS_1_X AXIS_1_Y AXIS_1_Z** | Joint 1 axis (unit vector). | `0 0 1` |
| **JOINT1_MIN JOINT1_MAX** | Joint 1 limits (degrees). | `-90 90` |
| **LINK1_X LINK1_Y LINK1_Z** | Endpoint of link 1. | `0.2 0 0` |
| **LINK1_RADIUS** | Capsule radius (m). | `0.02` |
| **AXIS_2_X AXIS_2_Y AXIS_2_Z**, **JOINT2_***, **LINK2_*** | Same for joint/link 2. | e.g. `0 0 1`, `-90 90`, `0.2 0 0` |
| **EE_SITE_NAME** | End-effector site name. | `ee_site` |
| **QPOS1 QPOS2** | Home joint positions (degrees). | `0 0` |

**More joints:** Add more `<body>`, `<joint>`, `<geom>`, one `<motor>` per joint, and extend `qpos`. **Meshes:** Use `<compiler meshdir="assets"/>` and put files in `scenes/arms/models/arms/<arm_id>/assets/`.

### Optional: Registry entry

In `scenes/arms/arm_registry.py`, add to **`ARM_REGISTRY`**:

```python
"my_robot": ArmConfig(
    name="My Robot Arm",
    arm_path="arms/my_robot/my_robot.xml",
    ee_site_name="ee_site",
    reach_min=0.15,
    reach_max=0.85,
    home_keyframe_name="home",
),
```

### Optional: Per-arm overrides

In **`config/arm_overrides.yaml`** add a section for your `arm_id`:

```yaml
my_robot:
  reach_max_cap: 0.5
  reach_min_mode: registry
  initial_pose: random
  joint_limit_margin_penalty: 0.02
```

| Problem | Override | Typical value |
|--------|----------|----------------|
| Erratic motion | `reach_max_cap` | `0.5` |
| Bimanual too close | `initial_pose` | `random` |
| Two arms, one ball | `ball_mode` | `per_arm` |
| Arm folds / self-collision | `joint_limit_margin_penalty` | `0.015`–`0.03` |

**Cheat sheet (new arm):** `train <arm_id> [steps]` then `run <arm_id> [steps]`. Optional: registry and/or overrides.

---

## Adding a new leg (biped)

Train any biped to walk toward a ball. Uses a **GNN policy** that adapts to any morphology.

**Quick start:** 1) Drop biped MJCF into `scenes/legs/models/bipeds/<leg_id>/`. 2) Train: `python scripts/train_legs.py --leg-id <leg_id> --steps 500000`. 3) Run: `mjpython scripts/run_legs.py --leg-id <leg_id>`.

**Convention:** Prefer `<leg_id>.xml`; otherwise the first non-`_composed` XML in the folder is used.

**Requirements:** **Floating base** (root body with `freejoint`); the composed scene adds the ball. Set **`torso_body_name`** in the registry (root/pelvis body).

**Registry** (`scenes/legs/leg_registry.py`):

```python
"my_biped": LegConfig(
    name="My Biped",
    leg_path="bipeds/my_biped/robot.xml",
    torso_body_name="pelvis",
    home_keyframe_name="home",
    ball_dist_min=1.0,
    ball_dist_max=5.0,
    ball_side_extent=2.0,
),
```

**Overrides** (`config/leg_overrides.yaml`): `ball_dist_min`, `ball_dist_max`, `initial_pose`, `joint_limit_margin_penalty`, etc.

**Training tips:** `frame_skip` (4–5), `action_scale` (0.5–0.8), `total_steps` (1M+ for legs). **Reference:** Agility Cassie is under `scenes/legs/models/bipeds/agility_cassie/`.

---

## YAML Configuration

### Arm Reach (`config/arms.yaml`)

Used by `scripts/train.py` and `scripts/run.py`. See [Arm Reach Training](#arm-reach-training-primary-workflow) above.

### Legacy Rendering (`config/render_run.yaml`)

Defines what model, policy, and environment are used for rendering (AINex, industrial Z1, etc.):

```yaml
scene:
  env_class: scenes.industrial_arm_reaching.env:Z1ReachEnv
  model_xml: scenes/industrial_arm_reaching/models/z1scene.xml

policy:
  path: scenes/industrial_arm_reaching/policies/ppo_z1_parallel_1.5m_best.zip

run:
  episodes: 10
  max_seconds_per_ep: 30.0
  deterministic: true
```

### To Change What’s Rendered

Edit `scene.env_class`, `scene.model_xml`, `policy.path`, and `run.*` to change what is rendered.

---

## AINex Humanoid (Reach + Stand)

### Train AINex reach policy (metrics + seed)

```bash
.venv/bin/python scenes/ainex_soccer/training/ainex_reach_train.py --seed 42 --timesteps 1500000
```

Logs are written to:
- `logs/ainex_reach/monitor.csv`
- `logs/ainex_reach/episode_metrics.csv`

### Evaluate AINex reach policy (render + trajectories)

```bash
.venv/bin/mjpython scenes/ainex_soccer/training/ainex_reach_eval.py --config config/ainex_reach.yaml --episodes 5 --deterministic
```

Trajectory outputs:
- `logs/trajectories/*.csv`
- `logs/trajectories/*.png`

### Render AINex stand policy

```bash
.venv/bin/mjpython renders/render_demo_mac.py --config config/ainex_render.yaml
```

### Train AINex walk-to-ball (walk around table + reach)

Uses **action groups** for leg walking (from `assets/action_groups/csv/`) and **IMU-like observations** (torso orientation, angular/linear velocity) for balance. Policy controls the right arm for reaching.

```bash
.venv/bin/python scenes/ainex_soccer/training/ainex_walk_to_ball_train.py --seed 42 --timesteps 1000000
```

### Evaluate walk-to-ball policy

```bash
.venv/bin/mjpython scenes/ainex_soccer/training/ainex_walk_to_ball_eval.py --config config/ainex_walk_to_ball.yaml --episodes 5 --deterministic
```

---

## Other Scenes

### Cartpole

Jupyter notebooks for experimentation and analysis (`scenes/cartpole/`): `01_environment_testing.ipynb`, `02_policy_analysis.ipynb`, `03_reward_tuning.ipynb`, `04_data_analysis.ipynb`, `05_visualization.ipynb`. Run `jupyter notebook` or `jupyter lab`.

### Humanoid / Legs

- **scenes/humanoid/** – Placeholder for full-body humanoid scenes. AINex humanoid content is under `scenes/ainex_soccer/`.
- **scenes/legs/** – Placeholder for legs-only scenes (bipedal walking, quadruped, etc.).

### Image Recognition (Gesture Control)

MediaPipe + PyTorch gesture recognition (handshake, fist bump, high five). Integrates with MuJoCo for gesture-controlled robot reaching.

**Collect data:** `python scenes/image_recognition/training/collect_gesture_data.py` (target: ≥100 samples per gesture)  
**Train:** `python scenes/image_recognition/training/train_gesture_classifier.py --data-dir scenes/image_recognition/data --model-type lstm`  
**Inference:** `python scenes/image_recognition/training/infer_gesture.py --model-path scenes/image_recognition/models/gesture_classifier.pth`  
**Gesture-controlled demo:** `python scenes/image_recognition/integration/demo_gesture_control.py --model-path scenes/image_recognition/models/gesture_classifier.pth`

### Industrial Arm Reaching (Legacy)

Legacy multi-arm reach env in `scenes/industrial_arm_reaching/`. Registry arms: z1, arm_2link. Train with `ARM_ID=z1 python -m scenes.industrial_arm_reaching.training.arm_train_mac`. For new arms, prefer `scenes/arms/` with `scripts/train.py`.

---

## Training on NAU Monsoon HPC

For long training (1M–5M+ steps), use [NAU's Monsoon supercomputer](https://in.nau.edu/arc/overview/connecting-to-monsoon/). On success, policies are pushed to `origin monsoon`; pull locally to run them.

**Prerequisites:** NAU credentials, Monsoon account, GitHub SSH key or token for push.

### Connect

- **SSH:** `ssh [YOUR_NAU_ID]@monsoon.hpc.nau.edu`
- **OnDemand:** [ondemand.hpc.nau.edu](https://ondemand.hpc.nau.edu/) → Clusters → Monsoon Cluster Login-Shell

### Setup (one-time)

```bash
cd /scratch/$USER   # or ~/scratch; use /scratch/$USER if needed
git clone https://github.com/lucaslarson25/reach.git
cd reach
git checkout monsoon && git pull origin monsoon

module load cuda
module load mambaforge
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements-hpc.txt
python -c "import mujoco; import torch; print('CUDA:', torch.cuda.is_available())"
```

### Submit training

**Option A — background (no SLURM):** Run in background with `nohup`; on success, script pushes to `monsoon` and can email you.

```bash
cd /scratch/$USER/reach
git checkout monsoon && git pull origin monsoon
# Edit cluster/run_arms_background.sh: ARM_ID, TIMESTEPS, EMAIL
chmod +x cluster/run_arms_background.sh
nohup ./cluster/run_arms_background.sh > logs/arms_train.log 2>&1 &
tail -f logs/arms_train.log
```

**Option B — batch job:** `ARM_ID=panda TIMESTEPS=5000000 sbatch cluster/train_arms_monsoon.sh`

**Monitor:** `squeue -u $USER`; `tail -f logs/monsoon_arms_<JOBID>.out` (arm) or `logs/monsoon_<JOBID>.out` (AINex).

### After training

Script commits and pushes to `origin monsoon` on success. Locally:

```bash
git fetch origin && git checkout monsoon && git pull origin monsoon
run panda   # or mjpython scripts/run.py --arm-id panda
```

### GPU and storage

| GPU | Use | SBATCH |
|-----|-----|--------|
| A100 | Deep learning | `#SBATCH --gpus=a100:1` |
| V100 | Alternative | `#SBATCH --gpus=v100:1` |

**Storage:** Use `/scratch/$USER` (or `~/scratch`) for training; avoid large outputs in home.

**Troubleshooting:** `~/scratch` not found → use `/scratch/$USER`. Push fails → check SSH key or HTTPS token. No email → use `tail -f logs/arms_train.log`. CUDA OOM → reduce batch size or request different GPU.

---

## macOS (M-Series) Instructions

### Arm reach (train + run)

```bash
python scripts/train.py --arm-id panda
mjpython scripts/run.py --arm-id panda
```

### Render a trained PPO policy (legacy)

```bash
.venv/bin/mjpython renders/render_demo_mac.py --config config/render_run.yaml
```

### View a model interactively

```bash
.venv/bin/mjpython renders/render_model_mac.py --config config/render_run.yaml
```

> **Note:** macOS requires **mjpython** to open MuJoCo's passive viewer using Metal graphics. For arm reach, use `scripts/run.py`; for legacy scenes, the YAML file defines which model and policy are rendered.

---

## Windows / Linux (x86 / CUDA) Instructions

### Arm reach (train + run)

```bash
python scripts/train.py --arm-id panda
python scripts/run.py --arm-id panda
```

### Render a trained PPO policy (legacy)

```bash
.venv\Scripts\python.exe renders\render_demo.py --config config\render_run.yaml
```

### View a model interactively

```bash
.venv\Scripts\python.exe renders\render_model.py --config config\render_run.yaml
```

> **Tip:**
> On CUDA systems, PyTorch will automatically use your GPU if available:
>
> ```python
> import torch
> print(torch.cuda.is_available())
> ```

---

## Adding a New Arm

You can train the ball-reaching policy on **any arm** by adding it to the registry or models directory. Per-arm overrides handle geometry-specific issues (erratic motion, bimanual start poses, self-collision).

- **Quick add:** Place MJCF in `scenes/arms/models/arms/<arm_id>/` and run  
  `python scripts/train.py --arm-id <arm_id>`
- **If problems occur:** Edit `config/arm_overrides.yaml` for that arm (tighter ball band, random init, joint-limit penalty).
- **Full guide:** See [Adding a new arm (full guide)](#adding-a-new-arm-full-guide) in this README.

### Legs (Biped Locomotion)

Train bipeds to walk toward a ball. Uses Agility Cassie from [MuJoCo Menagerie](https://mujoco.readthedocs.io/en/stable/models.html).

- **Train:** `python scripts/train_legs.py --leg-id agility_cassie --steps 500000`
- **Run:** `mjpython scripts/run_legs.py --leg-id agility_cassie`
- **Add leg:** Place MJCF in `scenes/legs/models/bipeds/<leg_id>/` and register in `leg_registry.py`
- **Overrides:** `config/leg_overrides.yaml` for per-leg ball placement and penalties
- **Full guide:** See [Adding a new leg (biped)](#adding-a-new-leg-biped) in this README.

**Note:** There is no reference walking gait or motion capture; the policy learns from scratch. **Where rewards go for walking:** (1) move toward ball (forward/distance), (2) stay upright and at target height, (3) stay near the “home” standing pose so it doesn’t retract a leg, (4) avoid lateral slide and torso tilt, (5) encourage quick foot return—bonus when a foot lands (`reward_landing_scale`), penalty for each step a foot is in the air (`penalty_air_time`). Tune these in `config/legs.yaml` if the robot retracts a leg or slides.

---

## System design and limitations

- **Architecture:** MuJoCo 3.x (physics/rendering), Stable-Baselines3 (PPO), Gymnasium envs. YAML selects env, model XML, policy path. Policies trained/evaluated from repo root so relative paths resolve; macOS uses `mjpython` for viewer.
- **Assumptions:** Robot XML is self-contained and valid; action space drives MuJoCo actuators directly.
- **Limitations:** Contact realism (foot/ground, arm/table) is approximate; actuator model is simplified; domain gap vs real (friction, compliance, appearance); evaluation is task-level, not disturbance robustness; no sensor noise/delay by default.
- **Sim-to-real risks:** Joint limits, damping, torque limits can mismatch; friction and contact differ; policies may overfit to deterministic sim.
- **Toward deployment:** Add sensor noise, actuator latency, friction randomization; calibrate with real measurements; add safety limits on velocities/accelerations; validate on simple real tasks; log real rollouts to refine models.

---

## Development Tips

- Always activate your `.venv` before running scripts.
- macOS users **must** use `mjpython` for rendering; Windows/Linux use `python`.
- **Arm reach:** Use `scripts/train.py` and `scripts/run.py` with `--arm-id` to train/run any arm. Policies save to `policies/` at project root.
- **Legacy scenes:** `config/render_run.yaml` controls AINex and industrial arm rendering.
- You can switch to any other scene by updating the YAML paths — no code changes required.
- Large `.zip` policy files: arm reach uses `policies/`; other scenes use `scenes/<name>/policies/`.

---

## Troubleshooting

### `ModuleNotFoundError: No module named 'scenes'`

Set your Python path manually:

```bash
export PYTHONPATH=$(pwd)
```

### `RuntimeError: launch_passive requires mjpython`

Use `mjpython` instead of `python` on macOS:

```bash
.venv/bin/mjpython renders/render_demo_mac.py --config config/render_run.yaml
```

### Policy not found

- **Arm reach:** Policies are in `policies/ppo_arms_<arm_id>_mac_<k>k.zip`. Ensure you pass `--arm-id` matching the trained arm, or use `--model <path>` to specify the policy.
- **Other scenes:** Policies may be in `scenes/<scene_name>/policies/<policy_name>.zip`.

### Viewer closes instantly

Check your model XML path in `config/render_run.yaml`—MuJoCo closes the window immediately if the model fails to load.

### Arm does not move in viewer

Pass `--debug` to print action norms. If norm is ~0, retrain with more steps (`--steps 500000`) or try `--stochastic` when running.

---

## Branching Strategy

- **`dev`** – Integration branch. Feature branches (ainex, vdev, tdev, etc.) merge here to iron out bugs before going to main.
- **`main`** – Production-ready code only. Merge from `dev` when fully tested and functional.

**Flow:** feature branch → `dev` (test, fix) → `main`

---

## Collaboration

- **Mentor Meetings:** Thursdays, 4:30–5:30 PM
- **Sponsor Meetings:** Biweekly Tuesdays, 2:00–3:30 PM
- **Capstone Lectures:** Fridays, 12:45–3:15 PM

**Tools:**

- GitHub Projects & Issues for task tracking
- Google Docs and Markdown for documentation
- Lucidchart for diagrams
- Pull requests and feature branches for version control

---

## Coding & Documentation Standards

- Follows **PEP8** Python style guidelines
- Code is modular, well-commented, and reproducible
- Technical docs use Markdown (`.md`); presentations use PowerPoint or Google Slides
- All diagrams and charts use clear labels and consistent formatting

---

## License

This project was developed for academic purposes as part of the **NAU Computer Science Capstone Program (2024–2025)**.
All rights reserved by the REACH development team and **Northern Arizona University**.

---

**REACH Capstone Project**
_Northern Arizona University – Computer Science Department (2024–2025)_
