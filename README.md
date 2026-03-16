# REACH Capstone Project

## Project Overview

The **REACH Capstone Project** is a robotics simulation and reinforcement learning (RL) platform developed at **Northern Arizona University**.
It combines **MuJoCo physics simulation**, **PPO reinforcement learning**, and **Python-based visualization** to train robotic arms and other systems for precision control and interaction tasks.

The system runs cross-platform on **macOS (M-series)** and **Windows/Linux (x86 / CUDA)** with a unified configuration layer driven by a YAML file.

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

**Other arms:** Use the same syntax with any supported `arm_id`: `train z1`, `run aloha`, etc. To **add your own arm** (custom MJCF), see [Upload another arm](#upload-another-arm) and the full pipeline in [documentation/adding_new_arm.md](documentation/adding_new_arm.md). For a minimal copy-paste cheat sheet, see [QUICKSTART_ARMS.md](QUICKSTART_ARMS.md).

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

See `documentation/system_design.md` for architecture, assumptions, limitations, and sim-to-real risks.

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

**Optional:** Add an entry in **`scenes/arms/arm_registry.py`** to set EE site name, reach limits, or home keyframe. If the arm behaves poorly (erratic motion, folding), add a section for your `arm_id` in **`config/arm_overrides.yaml`** (e.g. `reach_max_cap`, `initial_pose`, `joint_limit_margin_penalty`). **Full pipeline with troubleshooting:** [documentation/adding_new_arm.md](documentation/adding_new_arm.md).

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

For long training (1M–5M+ steps), use [NAU's Monsoon supercomputer](https://in.nau.edu/arc/overview/connecting-to-monsoon/). Jobs run in the background—you can exit the server. On success, policies are pushed to `origin monsoon`; pull locally to get them.

### Arm reach (recommended)

**No batch jobs** (no SLURM failure emails): run in background with `nohup`; on success, auto-pushes and emails you.

```bash
cd /scratch/$USER/reach
git checkout monsoon && git pull origin monsoon
chmod +x cluster/run_arms_background.sh
nohup ./cluster/run_arms_background.sh > logs/arms_train.log 2>&1 &
# Edit the script to set ARM_ID, TIMESTEPS, EMAIL; or: ARM_ID=aloha TIMESTEPS=5000000 nohup ...
```

After training completes (you'll get an email if `mail` works): `git checkout monsoon && git pull` then `mjpython scripts/run.py --arm-id panda`.

**Batch job option:** `sbatch cluster/train_arms_monsoon.sh`

### Full workflow

1. **Connect:** `ssh [NAU_ID]@monsoon.hpc.nau.edu` (or [OnDemand](https://ondemand.hpc.nau.edu/))
2. **Setup:** See `documentation/monsoon_setup.md` (venv, modules, scratch path)
3. **Submit:** `sbatch cluster/train_arms_monsoon.sh`
4. **Monitor:** `squeue -u $USER` and `tail -f logs/monsoon_arms_<JOBID>.out`
5. **Pull locally:** `git checkout monsoon && git pull` when job succeeds

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
- **Full guide:** [documentation/adding_new_arm.md](documentation/adding_new_arm.md)

### Legs (Biped Locomotion)

Train bipeds to walk toward a ball. Uses Agility Cassie from [MuJoCo Menagerie](https://mujoco.readthedocs.io/en/stable/models.html).

- **Train:** `python scripts/train_legs.py --leg-id agility_cassie --steps 500000`
- **Run:** `mjpython scripts/run_legs.py --leg-id agility_cassie`
- **Add leg:** Place MJCF in `scenes/legs/models/bipeds/<leg_id>/` and register in `leg_registry.py`
- **Overrides:** `config/leg_overrides.yaml` for per-leg ball placement and penalties
- **Guide:** [documentation/adding_new_leg.md](documentation/adding_new_leg.md)

**Note:** There is no reference walking gait or motion capture; the policy learns from scratch. **Where rewards go for walking:** (1) move toward ball (forward/distance), (2) stay upright and at target height, (3) stay near the “home” standing pose so it doesn’t retract a leg, (4) avoid lateral slide and torso tilt, (5) encourage quick foot return—bonus when a foot lands (`reward_landing_scale`), penalty for each step a foot is in the air (`penalty_air_time`). Tune these in `config/legs.yaml` if the robot retracts a leg or slides.

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
