#!/bin/bash
#SBATCH --job-name=reach_train
#SBATCH --gpus=a100:1
#SBATCH --time=12:00:00
#SBATCH --output=logs/monsoon_%j.out
#SBATCH --error=logs/monsoon_%j.err
#
# REACH training on Monsoon HPC.
# On success: commits policies and pushes to origin monsoon.
# Team can: git checkout monsoon && git pull
#
# Usage: sbatch cluster/train_monsoon.sh
# Customize: TRAIN_SCRIPT, TIMESTEPS, SEED below.

# --- Config (edit as needed) ---
TRAIN_SCRIPT="scenes/ainex_soccer/training/ainex_reach_train.py"
TIMESTEPS=1500000
SEED=42
# Alternatives: ainex_walk_to_ball_train.py, ainex_reach_wholebody_train.py

# --- Paths ---
# Slurm may execute this from a spool copy; prefer submit directory (see train_arms_monsoon.sh).
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
  REPO_ROOT="$(cd "${SLURM_SUBMIT_DIR}" && pwd)"
else
  REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "$REPO_ROOT"
mkdir -p logs

# --- Environment ---
module load cuda 2>/dev/null || true
module load mambaforge 2>/dev/null || true
module load python/3.11.9 2>/dev/null || true
if [ -d ".venv" ]; then
    source .venv/bin/activate
else
    echo "ERROR: .venv not found. Run setup first (see documentation/monsoon_setup.md)"
    exit 1
fi

export PYTHONPATH="$REPO_ROOT:$PYTHONPATH"

# --- Smoke test (quick sanity check before training) ---
echo "Running smoke test..."
if ! python -m tests.smoke_test; then
    echo "ERROR: Smoke test failed. Fix environment before training."
    exit 1
fi
echo ""

# --- Run training ---
echo "=== REACH Monsoon Training ==="
echo "Script: $TRAIN_SCRIPT"
echo "Timesteps: $TIMESTEPS"
echo "Started: $(date)"
echo ""

TRAIN_EXIT=0
python "$TRAIN_SCRIPT" --seed "$SEED" --timesteps "$TIMESTEPS" || TRAIN_EXIT=$?

echo ""
echo "Finished: $(date)"

if [ "$TRAIN_EXIT" -ne 0 ]; then
    echo "Training FAILED (exit $TRAIN_EXIT). Not pushing."
    exit "$TRAIN_EXIT"
fi

# --- Push to monsoon branch on success ---
echo ""
echo "Training succeeded. Committing and pushing policies..."

# Ensure we're on monsoon branch
git checkout monsoon 2>/dev/null || git checkout -b monsoon

# Policy paths (add more if you train other scenes)
POLICY_PATHS=(
    "scenes/ainex_soccer/policies/ppo_ainex_reach.zip"
    "scenes/ainex_soccer/policies/ppo_ainex_walk_to_ball.zip"
)

HAS_CHANGES=0
for p in "${POLICY_PATHS[@]}"; do
    if [ -f "$p" ]; then
        if git status --porcelain "$p" | grep -q .; then
            git add "$p"
            HAS_CHANGES=1
        fi
    fi
done

if [ "$HAS_CHANGES" -eq 1 ]; then
    git commit -m "Monsoon: trained policies $(date +%Y-%m-%d)"
    if ! git push origin monsoon; then
        echo "ERROR: git push failed (check SSH/HTTPS credentials on Monsoon)."
        exit 1
    fi
    echo "Pushed to origin monsoon."
else
    echo "No policy changes to commit."
fi

echo "Done."
