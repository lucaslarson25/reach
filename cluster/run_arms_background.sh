#!/bin/bash
#
# Run arm training in background (no SLURM). Survives logout when used with nohup.
# On success: pushes to monsoon branch and emails you.
#
# Usage:
#   nohup ./cluster/run_arms_background.sh > logs/arms_train.log 2>&1 &
#   tail -f logs/arms_train.log   # watch progress
#
# Edit ARM_ID, TIMESTEPS, EMAIL below before running.

ARM_ID="${ARM_ID:-panda}"
TIMESTEPS="${TIMESTEPS:-1000000}"
EMAIL="${EMAIL:-vr527@nau.edu}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
mkdir -p logs policies

# Load env (Monsoon/Wind)
module load cuda 2>/dev/null || true
module load mambaforge 2>/dev/null || true
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi
export PYTHONPATH="$REPO_ROOT:$PYTHONPATH"

send_email() {
    local subject="$1"
    local body="$2"
    if command -v mail &>/dev/null; then
        echo "$body" | mail -s "$subject" "$EMAIL"
    elif command -v mailx &>/dev/null; then
        echo "$body" | mailx -s "$subject" "$EMAIL"
    else
        echo "(mail not available) $subject: $body"
    fi
}

echo "=== Arm training (background) ==="
echo "ARM_ID=$ARM_ID TIMESTEPS=$TIMESTEPS"
echo "Started: $(date)"
echo ""

python scripts/train.py --arm-id "$ARM_ID" --steps "$TIMESTEPS"
TRAIN_EXIT=$?

echo ""
echo "Finished: $(date)"

if [ "$TRAIN_EXIT" -ne 0 ]; then
    send_email "REACH: Training FAILED" "Arm training exited with code $TRAIN_EXIT. Check logs."
    exit $TRAIN_EXIT
fi

# Push to monsoon on success
git checkout monsoon 2>/dev/null || git checkout -b monsoon
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
        send_email "REACH: Training DONE" "Arm policy $ARM_ID (${TIMESTEPS} steps) pushed to monsoon. Pull: git checkout monsoon && git pull"
    else
        send_email "REACH: Push FAILED" "Training succeeded but git push failed. Check credentials on server."
        exit 1
    fi
else
    send_email "REACH: Training DONE (no new policy)" "Training completed but no policy changes to commit."
fi

echo "Done."
