#!/usr/bin/env bash
# One-time setup for arm reach: create venv, install deps, and install train/run commands.
# Run from project root:  ./scripts/setup_arms.sh
# Then:  train panda   and   run panda   (or  train ur5e 500000  /  run ur5e 10000)

set -e
cd "$(dirname "$0")/.."
REPO_ROOT=$(pwd)

if [ ! -f "requirements.txt" ]; then
  echo "Error: Run this script from the reach repo root (or use: ./scripts/setup_arms.sh from repo root)."
  exit 1
fi

if [ ! -d ".venv" ]; then
  echo "Creating .venv..."
  python3 -m venv .venv
fi

echo "Activating .venv and installing dependencies..."
# shellcheck source=/dev/null
source .venv/bin/activate
pip install -q -r requirements.txt
pip install -q -e .

echo ""
echo "Setup complete. In this terminal (or a new one), activate the venv and run:"
echo "  source .venv/bin/activate"
echo "  train panda              # train (or: train ur5e 500000)"
echo "  run panda                # run simulation (or: run ur5e 10000)"
echo ""
echo "On macOS, run uses mjpython automatically. Other arms: train ur5e, run z1, etc. See QUICKSTART_ARMS.md."
