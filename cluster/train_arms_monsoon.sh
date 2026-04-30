#!/bin/bash
#SBATCH --job-name=reach_arms
#SBATCH --gpus=a100:1
# If your cluster uses different GPU names, try: --gpus=v100:1 or --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --output=logs/monsoon_arms_%j.out
#SBATCH --error=logs/monsoon_arms_%j.err
#
# Arm reach training on Monsoon HPC. Run in background, exit server, job continues.
# On success: pushes policies to origin monsoon. Pull locally to get trained policy.
#
# Usage: sbatch cluster/train_arms_monsoon.sh
# Edit ARM_ID and TIMESTEPS below before submitting.
#
# After job completes successfully:
#   git fetch origin && git checkout monsoon && git pull origin monsoon
#   mjpython scripts/run.py --arm-id panda

# --- Config (edit before submitting, or pass via env) ---
ARM_ID="${ARM_ID:-panda}"
TIMESTEPS="${TIMESTEPS:-1000000}"
# Override: ARM_ID=aloha TIMESTEPS=5000000 sbatch cluster/train_arms_monsoon.sh

# --- Paths ---
# Slurm often runs this script from a *spool copy* (not your repo). BASH_SOURCE then
# resolves outside the project, so .venv / logs / policies are "missing". Use the
# directory you ran sbatch from (always set in batch jobs).
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
  REPO_ROOT="$(cd "${SLURM_SUBMIT_DIR}" && pwd)"
else
  REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "$REPO_ROOT"
mkdir -p logs policies

# --- Environment ---
module purge 2>/dev/null || true
module load cuda 2>/dev/null || true
module load mambaforge 2>/dev/null || true
module load python/3.11.9 2>/dev/null || true
module load python/3.10 2>/dev/null || true

if [ -d ".venv" ]; then
    source .venv/bin/activate
else
    echo "ERROR: .venv not found. Create it first:"
    echo "  module load mambaforge cuda"
    echo "  python3 -m venv .venv && source .venv/bin/activate"
    echo "  pip install -r requirements-hpc.txt"
    exit 1
fi

export PYTHONPATH="$REPO_ROOT:$PYTHONPATH"

# --- Minimal smoke test (arm env only, no AINex) ---
echo "Checking environment..."
if ! python -c "
import mujoco
import torch
from stable_baselines3 import PPO
from scenes.arms.env import ArmReachEnv
e = ArmReachEnv(arm_id='arm_2link')
e.reset(seed=42)
e.close()
print('  OK: mujoco, torch, PPO, ArmReachEnv')
"; then
    echo "ERROR: Environment check failed. Fix dependencies before training."
    exit 1
fi
echo ""

# --- Run training ---
echo "=== Arm Reach Training (Monsoon) ==="
echo "ARM_ID: $ARM_ID"
echo "TIMESTEPS: $TIMESTEPS"
echo "Started: $(date)"
echo ""

TRAIN_EXIT=0
python scripts/train.py --arm-id "$ARM_ID" --steps "$TIMESTEPS" || TRAIN_EXIT=$?

echo ""
echo "Finished: $(date)"

if [ "$TRAIN_EXIT" -ne 0 ]; then
    echo "Training FAILED (exit $TRAIN_EXIT). Not pushing."
    exit "$TRAIN_EXIT"
fi

# --- Push to monsoon branch on success ---
echo ""
echo "Training succeeded. Committing and pushing policies..."

git checkout monsoon 2>/dev/null || git checkout -b monsoon

# Arm policies: policies/ppo_arms_<arm_id>_mac_<k>k.zip (and per-arm: _arm0_, _arm1_, etc.)
HAS_CHANGES=0
for p in policies/ppo_arms_*.zip; do
    [ -f "$p" ] || continue
    if git status --porcelain "$p" 2>/dev/null | grep -q .; then
        git add "$p"
        HAS_CHANGES=1
    fi
done

if [ "$HAS_CHANGES" -eq 1 ]; then
    git commit -m "Monsoon: arm policy $ARM_ID ${TIMESTEPS} steps $(date +%Y-%m-%d)"
    if git push origin monsoon; then
        echo "Pushed to origin monsoon. Pull locally: git checkout monsoon && git pull"
    else
        echo "ERROR: git push failed. Check SSH/HTTPS credentials on Monsoon."
        exit 1
    fi
else
    echo "No new policy files to commit."
fi

echo "Done."
