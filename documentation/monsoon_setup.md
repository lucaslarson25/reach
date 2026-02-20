# Running REACH Training on NAU Monsoon HPC

This guide explains how to run long REACH training jobs on [NAU's Monsoon supercomputer](https://in.nau.edu/arc/overview/connecting-to-monsoon/), automatically push trained policies to a `monsoon` branch on success, and let your team pull and run simulations locally.

## Overview

**Workflow:**
1. Connect to Monsoon via SSH or [OnDemand](https://ondemand.hpc.nau.edu/)
2. Clone the repo, create/checkout `monsoon` branch
3. Submit a SLURM job that runs training overnight
4. On success, the job script commits and pushes policies to `origin/monsoon`
5. Team members pull `monsoon` and run simulations locally

---

## Prerequisites

- **NAU credentials** (same as your NAU login)
- **NAU VPN** (required if not on NAU WiFi)
- **Monsoon account** – [Request one](https://in.nau.edu/arc/overview/) if needed
- **GitHub access** – SSH key or token for pushing

---

## 1. Connect to Monsoon

### Via SSH (terminal)

```bash
ssh [YOUR_NAU_ID]@monsoon.hpc.nau.edu
```

Example: `ssh abc123@monsoon.hpc.nau.edu`

*Classroom users: use `rain` instead of `monsoon` – `ssh abc123@rain.hpc.nau.edu`*

### Via OnDemand (web portal)

1. Go to [https://ondemand.hpc.nau.edu/](https://ondemand.hpc.nau.edu/)
2. Log in with NAU credentials
3. Use **Clusters → Monsoon Cluster Login-Shell** for a terminal
4. Use **Jobs → Job Composer** to submit batch jobs visually

---

## 2. Set Up Git for Push (One-Time)

Monsoon needs to push to GitHub. You have two options:

### Option A: SSH key (recommended)

1. On your **local machine**, generate a key and add it to GitHub:
   ```bash
   ssh-keygen -t ed25519 -C "your_email@nau.edu"
   # Add public key to GitHub: Settings → SSH and GPG keys
   ```

2. On Monsoon, copy your private key (or generate one on Monsoon and add to GitHub):
   ```bash
   # On Monsoon
   mkdir -p ~/.ssh
   # Use scp from your laptop to copy ~/.ssh/id_ed25519 to monsoon:~/.ssh/
   chmod 600 ~/.ssh/id_ed25519
   ```

3. Test:
   ```bash
   ssh -T git@github.com
   ```

### Option B: Personal Access Token (HTTPS)

1. Create a token on GitHub: Settings → Developer settings → Personal access tokens
2. On Monsoon, use HTTPS and configure credential helper:
   ```bash
   git config --global credential.helper store
   git clone https://github.com/lucaslarson25/reach.git
   # When prompted, use your GitHub username and the token as password
   ```

---

## 3. Clone Repo and Create `monsoon` Branch

```bash
cd ~/scratch   # or your preferred work dir
git clone https://github.com/lucaslarson25/reach.git
cd reach

# Create and switch to monsoon branch
git checkout -b monsoon
git push -u origin monsoon
```

---

## 4. Set Up Python Environment on Monsoon

Monsoon uses Linux; you need **Python 3.9+** (system Python may be older). Load Mambaforge first.

```bash
cd ~/reach

# Load CUDA and Python 3.9+ (required for numpy 1.26, mujoco 3, torch 2)
module load cuda
module load mambaforge

# Create venv using the module's Python
deactivate 2>/dev/null || true
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies (use requirements-hpc.txt for headless OpenCV)
pip install --upgrade pip
pip install -r requirements-hpc.txt

# Verify
python -c "import mujoco; import torch; print('CUDA:', torch.cuda.is_available())"
```

If `module load mambaforge` fails, try `module avail python` and load a Python 3.9+ module.

---

## 5. Run Tests (Local or Monsoon)

The same smoke test runs whether you're on your laptop or Monsoon:

```bash
# From repo root, with venv activated
python -m tests.smoke_test
```

On Monsoon, you can also run the test script (loads modules and venv):

```bash
./cluster/test_monsoon.sh
```

Or submit as a short SLURM job:

```bash
sbatch cluster/test_monsoon.sh
```

---

## 6. Submit Training Job

**Run from the repo root** (so the job finds the repo):

```bash
cd ~/scratch/reach

# Edit cluster/train_monsoon.sh if needed (TRAIN_SCRIPT, TIMESTEPS, SEED)
sbatch cluster/train_monsoon.sh
```

Or submit manually:

```bash
sbatch --job-name=reach_train \
       --gpus=a100:1 \
       --time=12:00:00 \
       --output=logs/monsoon_%j.out \
       cluster/train_monsoon.sh
```

Monitor the job:

```bash
squeue -u $USER
```

Check output:

```bash
tail -f logs/monsoon_<JOBID>.out
```

---

## 7. What Happens After Training

The `train_monsoon.sh` script:

1. Runs a smoke test (quick sanity check)
2. Runs the training script
3. If training succeeds (exit 0):
   - Commits new/updated policy files
   - Pushes to `origin monsoon` (exits with error if push fails)
4. If training fails: no push; logs remain for debugging

---

## 8. Team Pull and Run

Once the job has pushed successfully (training completed without error):

```bash
# On your laptop
git fetch origin
git checkout monsoon
git pull origin monsoon
```

Then run the simulation:

```bash
# macOS
.venv/bin/mjpython renders/render_demo_mac.py --config config/ainex_reach.yaml

# Windows/Linux
python renders/render_demo.py --config config/ainex_reach.yaml
```

---

## GPU Resources on Monsoon

| GPU   | Recommended for | Request with |
|-------|-----------------|--------------|
| A100  | Deep learning   | `#SBATCH --gpus=a100:1` |
| V100  | Good alternative | `#SBATCH --gpus=v100:1` |
| P100  | Fallback        | `#SBATCH --gpus=p100:1` |
| K80   | Light workloads | `#SBATCH --gpus=k80:1` |

Check availability:

```bash
gpu_status
```

---

## Storage

- **Home**: `~/` – limited; avoid large outputs
- **Scratch**: `~/scratch` or `/scratch/$USER` – faster, larger; use for training
- **Policies**: `.zip` files are ~2MB; fine to keep in repo and push

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `Permission denied (publickey)` | Set up SSH key or use HTTPS + token |
| `ModuleNotFoundError: mujoco` | Activate venv and ensure `pip install` completed |
| `CUDA out of memory` | Reduce `--num-envs` or batch size in training script |
| Job killed / timeout | Increase `--time` in SBATCH (e.g. 24:00:00) |
| Push fails | Check git config; ensure token/key is valid |

---

## References

- [Connecting to Monsoon](https://in.nau.edu/arc/overview/connecting-to-monsoon/)
- [Submitting Your First Job](https://in.nau.edu/arc/overview/submitting-your-first-job/)
- [GPUs on Monsoon](https://in.nau.edu/arc/gpus/)
- [Using OnDemand](https://in.nau.edu/arc/ondemand/)
- [ARC Support](mailto:ask-arc@nau.edu)
