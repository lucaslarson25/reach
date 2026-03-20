#!/bin/bash
#
# Run REACH smoke test - works locally or on Monsoon (interactive or sbatch).
# Usage:
#   Local:        ./cluster/test_monsoon.sh
#   Monsoon:      ./cluster/test_monsoon.sh   (after: cd repo, source .venv)
#   Monsoon sbatch: sbatch cluster/test_monsoon.sh
#
# Optional SLURM directives (ignored when run directly):
#SBATCH --job-name=reach_test
#SBATCH --time=00:05:00
#SBATCH --output=logs/test_%j.out
#SBATCH --error=logs/test_%j.err

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
  REPO_ROOT="$(cd "${SLURM_SUBMIT_DIR}" && pwd)"
else
  REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "$REPO_ROOT"
mkdir -p logs

# Environment (Monsoon: load modules; local: use existing venv)
module load cuda 2>/dev/null || true
module load mambaforge 2>/dev/null || true
module load python/3.11.9 2>/dev/null || true
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

export PYTHONPATH="$REPO_ROOT:$PYTHONPATH"

echo "=== REACH smoke test ==="
python -m tests.smoke_test
exit $?
