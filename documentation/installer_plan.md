# Installer Plan (Draft)

## Purpose
Define a cross-platform installer scope and requirements for the REACH project
that works on macOS (M-series) and Windows (x86), and prepares for a later
automation phase.

## Supported OS
- macOS: Apple Silicon (M-series), using `mjpython` for MuJoCo viewer.
- Windows: Windows 10/11 (x86), using `python.exe` for MuJoCo viewer.

## Dependencies
System and Python requirements the installer must account for:
- Python 3.x and virtualenv support
- MuJoCo 3.x
- PyTorch 2.x
- Stable-Baselines3 (PPO)
- Project dependencies in `requirements.txt`

## Installer flow: macOS (M-series)
1. Install Python 3.x if missing.
2. Create virtual environment: `python3 -m venv .venv`.
3. Activate: `source .venv/bin/activate`.
4. Install dependencies: `pip install -r requirements.txt`.
5. Verify imports: `python -c "import mujoco; import torch; print('OK')"`.
6. Run render demo using mjpython:
   `.venv/bin/mjpython renders/render_demo_mac.py --config config/render_run.yaml`.

## Installer flow: Windows 10/11
1. Install Python 3.x if missing.
2. Create virtual environment: `python -m venv .venv`.
3. Activate: `.venv\Scripts\activate`.
4. Install dependencies: `pip install -r requirements.txt`.
5. Verify imports: `python -c "import mujoco; import torch; print('OK')"`.
6. Run render demo using python.exe:
   `.venv\Scripts\python.exe renders\render_demo.py --config config\render_run.yaml`.

## User inputs / options
- Install location for the project
- Optional: policy/model download step (if not already present)
- Optional: GPU/CUDA check for Windows

## Validation steps
Minimum checks the installer must pass:
- `python -c "import mujoco; import torch"` succeeds
- Rendering command launches without errors for the OS

## Open questions
- Exact supported Python version(s)?
- Bundle MuJoCo or download during install?
- Default location for policy/model assets?
