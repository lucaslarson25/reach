# Setup

This guide sets up the current REACH repository for local development,
rendering, training, gesture recognition, and NAU Monsoon HPC use.

## Prerequisites

- Python 3.9 or newer.
- Git.
- A display for MuJoCo viewers and OpenCV camera windows.
- A webcam for gesture collection and live gesture inference.
- Optional CUDA GPU for long training on Windows/Linux or Monsoon.
- Optional Apple Silicon MPS for local arm training experiments on macOS.

## Clone

```bash
git clone https://github.com/lucaslarson25/reach.git
cd reach
```

Check out the branch your team is using:

```bash
git checkout <branch-name>
```

## Local Python Environment

Use one virtual environment per checkout. The repo ignores `.venv/`, `venv/`,
and `env/`.

### Windows PowerShell

```powershell
py -3.12 -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If `py -3.12` is not available, use any Python 3.9+ interpreter:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For editable package metadata during development:

```bash
pip install -e .
```

Most workflows still use direct script execution, especially `scripts/train.py`
and `scripts/run.py`.

## Verify Local Setup

Run from the repository root with the virtual environment active:

```bash
python -m tests.smoke_test
```

The smoke test imports the core packages, instantiates `AINexReachEnv`, and runs
a tiny PPO learn step headlessly.

Additional import checks:

```bash
python -c "import mujoco, gymnasium, stable_baselines3, torch; print('core ok')"
python -c "from scenes.arms.env import ArmReachEnv; print('arms ok')"
python -c "from scenes.ainex_soccer.env import AINexEnv, AINexReachEnv, AINexWalkToBallEnv; print('ainex ok')"
python -c "import cv2, mediapipe, sklearn; print('vision ok')"
```

## Platform Notes

### Windows / Linux

Use normal `python` for training and non-mac render scripts:

```powershell
python scripts\train.py --arm-id panda
python scripts\run.py --arm-id panda
python renders\render_demo.py --config config\render_run.yaml
```

Check CUDA availability:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

Some older or legacy training scripts hard-code CUDA or prefer CUDA when
available. The primary `scripts/train.py` path is CPU/MPS-oriented by default
unless `config/arms.yaml` sets a device.

### macOS

Use `mjpython` for MuJoCo viewer workflows:

```bash
mjpython scripts/run.py --arm-id panda
mjpython renders/render_demo_mac.py --config config/render_run.yaml
mjpython tools/view_ainex_stable.py
```

`scripts/train.py` runs with normal `python`. To try Apple MPS for the primary
arm workflow:

```bash
USE_MPS=1 python scripts/train.py --arm-id panda --steps 300000
```

## Repository-Root Execution

Run commands from the repository root unless the command says otherwise. If a
module import fails, set `PYTHONPATH` to the repo root.

Windows PowerShell:

```powershell
$env:PYTHONPATH = (Get-Location).Path
```

macOS / Linux:

```bash
export PYTHONPATH="$(pwd)"
```

## Monsoon / Headless Setup

Use `requirements-hpc.txt` on NAU Monsoon or other headless systems. It swaps
GUI OpenCV for `opencv-python-headless`.

Basic Monsoon setup:

```bash
cd /scratch/$USER
git clone https://github.com/lucaslarson25/reach.git
cd reach
git checkout monsoon

module load cuda
module load mambaforge
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements-hpc.txt
python -m tests.smoke_test
```

The full Monsoon workflow is documented in
`documentation/monsoon_setup.md`.

Useful cluster scripts:

```bash
./cluster/test_monsoon.sh
sbatch cluster/test_monsoon.sh
ARM_ID=panda TIMESTEPS=5000000 sbatch cluster/train_arms_monsoon.sh
nohup ./cluster/run_arms_background.sh > logs/arms_train.log 2>&1 &
```

## Generated Files

Normal runs generate files that are ignored by Git:

- `logs/`
- virtual environments
- generated arm policies in `policies/`
- generated gesture data
- PyTorch `.pth` / `.pt` gesture models
- Python cache and build outputs

Scene-level `.zip` policies already committed under `scenes/*/policies/` are
part of the project assets.

## Common Problems

### `launch_passive requires mjpython`

Use `mjpython` for viewer commands on macOS.

### `ModuleNotFoundError: No module named 'scenes'`

Run from the repository root or set `PYTHONPATH` as shown above.

### Policy Not Found

Primary arm policies default to:

```text
policies/ppo_arms_<arm_id>_mac_<steps_in_k>k.zip
```

Use `--model <path>` with `scripts/run.py` to point at a different policy.

### Webcam Does Not Open

Check OS camera permissions, close other camera apps, and try a different
camera index such as `--camera 1`.

### Monsoon Push Fails

Configure GitHub SSH keys or HTTPS token access on Monsoon before using scripts
that auto-commit and push to `origin monsoon`.
