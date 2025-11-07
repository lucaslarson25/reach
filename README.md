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

## Project Structure

```
reach/
├── README.md
├── requirements.txt
│
├── config/
│   ├── render_run.yaml          # Defines scene, model, policy, and runtime settings
│   └── render_loader.py         # Helper for reading and validating YAML configs
│
├── documentation/
│   ├── demos/
│   ├── headshots/
│   ├── logos/
│   └── *.pdf
│
├── renders/                     # Rendering and visualization scripts
│   ├── render_demo.py           # x86 / CUDA policy renderer
│   ├── render_demo_mac.py       # macOS policy renderer (uses mjpython)
│   ├── render_model.py          # x86 / CUDA model viewer
│   └── render_model_mac.py      # macOS model viewer (uses mjpython)
│
├── scenes/
│   ├── industrial_arm_reaching/
│   │   ├── env.py
│   │   ├── models/
│   │   ├── policies/
│   │   └── training/
│   ├── cartpole/
│   └── industrial_arm_reaching_with_welding/
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
