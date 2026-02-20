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
│   ├── render_run.yaml          # Defines scene, model, policy, and runtime settings
│   ├── ainex_render.yaml        # AINex stand config
│   ├── ainex_reach.yaml         # AINex reach config
│   └── render_loader.py         # Helper for reading and validating YAML configs
│
├── cluster/
│   ├── train_monsoon.sh         # SLURM job for Monsoon HPC (auto-push on success)
│   └── test_monsoon.sh          # Smoke test (local or Monsoon)
├── tests/
│   └── smoke_test.py            # Quick sanity check (imports, env, PPO)
├── documentation/
│   ├── monsoon_setup.md         # Monsoon HPC setup and workflow
│   ├── demos/
│   ├── headshots/
│   ├── logos/
│   └── *.pdf
│   └── system_design.md         # Architecture, assumptions, limitations
│
├── envs/                        # Shared environments
│   └── mujoco_arm_env.py
│
├── scripts/                    # AINex Soccer viewers & replay (view_ainex_stable.py, etc.)
├── tools/                      # AINex Soccer conversion tools (extract_actiongroup_to_csv.py)
├── renders/                    # Rendering and visualization scripts
│   ├── render_demo.py           # x86 / CUDA policy renderer
│   ├── render_demo_mac.py       # macOS policy renderer (uses mjpython)
│   ├── render_model.py          # x86 / CUDA model viewer
│   └── render_model_mac.py      # macOS model viewer (uses mjpython)
│
├── scenes/
│   ├── ainex_soccer/             # AINex humanoid scene + training
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
├── logs/                        # Generated training/eval logs
│
├── website/
│   ├── index.html
│   ├── team.html
│   ├── project.html
│   ├── documents.html
│   └── assets/
│
└── .venv/                       # Virtual environment (ignored by Git)
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

## YAML Configuration

The **`config/render_run.yaml`** file defines what model, policy, and environment are used for rendering.

Example:

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

Edit the following fields:

- `scene.env_class`: Path to the environment Python class
- `scene.model_xml`: Path to the MuJoCo XML model
- `policy.path`: Path to the trained PPO policy `.zip`
- `run.*`: Adjust runtime parameters like episode count or duration

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

## Training on NAU Monsoon HPC

For long overnight training runs, use [NAU's Monsoon supercomputer](https://in.nau.edu/arc/overview/connecting-to-monsoon/).

### Quick start (submit and leave)

```bash
cd /scratch/YOUR_ID/reach   # or your repo path
sbatch cluster/train_monsoon.sh
```

The job runs in the background; you can disconnect. On success, policies are pushed to `origin monsoon`.

### Full workflow

1. **Connect:** `ssh [NAU_ID]@monsoon.hpc.nau.edu` (or use [OnDemand](https://ondemand.hpc.nau.edu/))
2. **Clone, setup:** See `documentation/monsoon_setup.md` for full instructions
3. **Submit job:** `sbatch cluster/train_monsoon.sh`
4. **Monitor (optional):** `squeue -u $USER` and `tail -f logs/monsoon_<JOBID>.out`
5. **On success:** Policies are pushed to the `monsoon` branch
6. **Team pull:** `git checkout monsoon && git pull` then run simulations locally

---

## macOS (M-Series) Instructions

### Render a trained PPO policy

```bash
.venv/bin/mjpython renders/render_demo_mac.py --config config/render_run.yaml
```

### View a model interactively

```bash
.venv/bin/mjpython renders/render_model_mac.py --config config/render_run.yaml
```

> **Note:**
> macOS requires **mjpython** to open MuJoCo’s passive viewer using Metal graphics.
> The YAML file defines which model and policy are rendered, so no command-line flags are needed beyond the config path.

---

## Windows / Linux (x86 / CUDA) Instructions

### Render a trained PPO policy

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
- All runtime parameters are now controlled in `config/render_run.yaml`.
- You can switch to any other scene by updating the YAML paths — no code changes required.
- Large `.zip` policy files should remain inside their scene’s `/policies` folder.

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

Ensure your trained PPO policy exists at:

```
scenes/<scene_name>/policies/<policy_name>.zip
```

### Viewer closes instantly

Check your model XML path in `config/render_run.yaml` — MuJoCo closes the window immediately if the model fails to load.

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
