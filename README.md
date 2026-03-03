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
│   ├── render_run.yaml         # Legacy: scene, model, policy for rendering
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
```

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

- **YAML-driven:** `config/arms.yaml` defines which arm, ball mode, train/run settings.
- **Auto-discovery:** Arms from [MuJoCo Menagerie](https://mujoco.readthedocs.io/en/stable/models.html) are auto-detected; EE sites and reach are inferred from MJCF.
- **CLI overrides:** No YAML edits needed—use `--arm-id`, `--steps`, etc. to tweak for any arm.
- **Single workflow:** Same commands for all arms (panda, ur5e, aloha, etc.).

### Supported Arms

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

### Train and Run

From project root:

```bash
# Train (default: panda, 300k steps)
python scripts/train.py

# Run in viewer (default: panda). macOS: use mjpython for MuJoCo passive viewer
mjpython scripts/run.py
```

### CLI Overrides

```bash
# Train different arm / steps / ball mode
python scripts/train.py --arm-id aloha
python scripts/train.py --arm-id ur5e --steps 500000
python scripts/train.py --arm-id aloha --per-arm-policies

# Run different arm / policy / steps
mjpython scripts/run.py --arm-id aloha
mjpython scripts/run.py --arm-id ur5e --model policies/ppo_arms_ur5e_mac_500k.zip
mjpython scripts/run.py --arm-id aloha --per-arm-policies
```

### Config (`config/arms.yaml`)

- **scene.arm_id** – which arm (e.g. `panda`, `ur5e`, `aloha`)
- **scene.ball_mode** – `shared` (1 ball, all arms) or `per_arm` (N balls for N arms)
- **scene.per_arm_policies** – `true` = train/run separate policy per arm (multi-arm only)
- **train.total_steps** – default 300000
- **train.policy_dir** – where to save policies (default: `policies/`)
- **train.reward_time_penalty**, **train.reward_smoothness** – reward shaping
- **run.steps**, **run.deterministic**, **run.debug**, **run.stochastic**

Env vars **ARM_ID**, **TOTAL_STEPS**, **MODEL_PATH**, **USE_MPS** override YAML. CLI flags take precedence.

### Policy Paths

Policies are saved to `policies/ppo_arms_<arm_id>_mac_<k>k.zip` (e.g. `policies/ppo_arms_aloha_mac_300k.zip`). The run script derives the path from `--arm-id` and config; use `--model` to override.

### Eval (metrics only)

```bash
python -m scenes.arms.training.eval_model --model policies/ppo_arms_panda_mac_300k.zip --arm-id panda
```

### Adding Another Arm

1. Copy the arm from Menagerie into `scenes/arms/models/arms/<arm_id>/`.
2. Name the main XML `arm.xml`, `<arm_id>.xml`, or `scene.xml`.
3. Use `--arm-id <arm_id>` and train/run. EE sites and reach are inferred from MJCF.

For manual override, add an entry in `scenes/arms/arm_registry.py`. See `scenes/arms/README.md` for details.

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
