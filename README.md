# REACH Capstone Project

## Project Overview

The **REACH Capstone Project** is a robotics simulation and reinforcement learning (RL) system developed at Northern Arizona University.
It combines **MuJoCo physics simulation**, **PPO reinforcement learning**, and **web-based visualization** to train robotic arms to complete goal-driven tasks such as reaching and object interaction.

---

## Team Members

### Development Team

- **Taylor Davis** ([tjd352@nau.edu](tjd352@nau.edu))
  **Role:** Team Lead / Coder / Architect
  **Responsibilities:** Leadership, technical development, integration, and design direction.

- **Victor Rodriguez** ([vr527@nau.edu](vr527@nau.edu))
  **Role:** Coder / Recorder / Architect
  **Responsibilities:** Documentation, training experiments, and model development.
  **Background:** U.S. Marine Corps veteran with leadership experience.

- **Clayton Ramsey** ([car723@nau.edu](car723@nau.edu))
  **Role:** Coder / Architect
  **Responsibilities:** Simulation structure, collaboration, and testing.

- **Lucas Larson** ([lwl33@nau.edu](lwl33@nau.edu))
  **Role:** Coder / Version Control Manager / Architect
  **Responsibilities:** GitHub operations, branch management, and development coordination.

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

- **Physics Engine:** [MuJoCo](https://mujoco.readthedocs.io/) 3.x
- **RL Framework:** [Stable-Baselines3 (PPO)](https://stable-baselines3.readthedocs.io/)
- **Deep Learning:** PyTorch 2.x
- **Frontend:** HTML, CSS, Bootstrap 4
- **Visualization:** MuJoCo Viewer (interactive or headless)
- **Environment:** macOS ARM (M-series) + x86/CUDA support
- **Version Control:** Git / GitHub

---

## 🗂️ Project Structure

```
reach/
├── README.md
├── requirements.txt
│
├── config/
│   ├── default.yaml
│   └── reaching_example.yaml
│
├── documentation/
│   ├── demos/
│   ├── headshots/
│   ├── logos/
│   └── *.pdf
│
├── renders/                      # Render demos and model viewers
│   ├── render_demo.py            # Run PPO policy (x86 / CUDA)
│   ├── render_demo_mac.py        # Run PPO policy (macOS / M-series)
│   ├── render_model.py           # View a model interactively (x86 / CUDA)
│   └── render_model_mac.py       # View a model interactively (macOS)
│
├── scenes/
│   ├── industrial_arm_reaching/
│   │   ├── env.py
│   │   ├── models/
│   │   ├── policies/
│   │   └── training/
│   │       ├── arm_train.py
│   │       ├── arm_train_mac.py
│   │       └── eval_model.py
│   │
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
└── .venv/
```

---

## Getting Started

### 1. Clone the repository

```bash
git clone -b tdev https://github.com/lucaslarson25/reach.git
cd reach
```

### 2. Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate      # macOS/Linux
# OR on Windows:
# .venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## macOS (Apple Silicon) Instructions

### Run the training demo

```bash
.venv/bin/python scenes/industrial_arm_reaching/training/arm_train_mac.py
```

### Render a trained PPO policy

```bash
.venv/bin/mjpython renders/render_demo_mac.py \
  --model scenes/industrial_arm_reaching/models/z1scene.xml \
  --policy scenes/industrial_arm_reaching/policies/ppo_z1_parallel_1.5m_best.zip
```

### View a model interactively

```bash
.venv/bin/mjpython renders/render_model_mac.py \
  --model scenes/industrial_arm_reaching/models/z1scene.xml
```

> **Note:**
> macOS requires `mjpython` instead of `python` to enable MuJoCo’s Metal rendering backend.
> This ensures smooth and hardware-accelerated visualization.

---

## x86 / CUDA (Windows or Linux) Instructions

### Run the training demo

```bash
.venv/bin/python scenes/industrial_arm_reaching/training/arm_train.py
```

### Render a trained PPO policy

```bash
.venv/bin/python renders/render_demo.py \
  --model scenes/industrial_arm_reaching/models/z1scene.xml \
  --policy scenes/industrial_arm_reaching/policies/ppo_z1_parallel_1.5m_best.zip
```

### View a model interactively

```bash
.venv/bin/python renders/render_model.py \
  --model scenes/industrial_arm_reaching/models/z1scene.xml
```

> **Tip:**
> For CUDA acceleration, ensure PyTorch detects your GPU:
>
> ```python
> import torch
> print(torch.cuda.is_available())
> ```
>
> If `True`, PPO training will automatically leverage the GPU.

---

## Configuration Notes

- The YAML files in `/config` are **not actively used** in the current workflow.
  They remain for reference and can be re-enabled for config-driven training later.
- All hyperparameters and model paths are directly defined in:

  - `arm_train.py`
  - `arm_train_mac.py`
  - `env.py`

---

## Development Tips

- Always activate your `.venv` before running scripts.
- On macOS, use `.venv/bin/mjpython` instead of `python`.
- Keep large model `.zip` files inside their scene’s `/policies` folder.
- The `renders` folder contains OS-specific scripts labeled `_mac` or default for CUDA/x86.

---

## Troubleshooting & Common Issues

### `ModuleNotFoundError: No module named 'scenes'`

This occurs when Python cannot find the project root. Fix by setting `PYTHONPATH`:

```bash
export PYTHONPATH=$(pwd)
```

Then rerun your command.

### `RuntimeError: launch_passive requires mjpython`

This happens if you use `python` instead of `mjpython` on macOS.
Always run visualization scripts with:

```bash
.venv/bin/mjpython renders/render_demo_mac.py ...
```

### Slow training performance

If training is slower than expected:

- Reduce parallel environments in `arm_train_mac.py` if memory is limited.
- Verify all CPU cores are being utilized (`os.cpu_count()` output).
- Use smaller timesteps for testing (`total_timesteps = 300000`).

### Viewer crashes or freezes

- Close other MuJoCo windows before opening a new one.
- Lower refresh rate (e.g., `time.sleep(1/60)` instead of `1/120`).

### Policy file not found

Make sure your policy `.zip` file exists at:

```
scenes/industrial_arm_reaching/policies/
```

Otherwise, download or retrain it before running demos.

---

## Team Collaboration

- **Mentor Meetings:** Thursdays, 4:30–5:30 PM
- **Sponsor Meetings:** Biweekly Tuesdays, 2:00–3:30 PM
- **Capstone Lectures:** Fridays, 12:45–3:15 PM

### Communication Tools

- **Task Tracking:** GitHub Projects / Issues
- **Documentation:** Google Docs, Markdown, Microsoft Office
- **Version Control:** Git + GitHub with branching strategy
- **Workflow:** Pull request reviews and feature branches

---

## Project Standards

### Documentation Standards

- Markdown for all technical docs
- Word and PowerPoint for formal deliverables
- Lucidchart / Draw.io for diagrams
- Consistent folder naming and structure

### Coding Standards

- Follow PEP8 Python conventions
- Clean, commented, and modular code
- Responsive and readable HTML/CSS

### Communication Standards

- Professional and prompt communication
- Meeting minutes distributed within 24 hours
- Clear task ownership and follow-ups

---

## License

This project was developed for academic purposes as part of the NAU Computer Science Capstone program.
All rights reserved by the development team and Northern Arizona University.

---

**REACH Capstone Project**
_Northern Arizona University – Computer Science Department (2024–2025)_
