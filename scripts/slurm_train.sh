#!/bin/bash
# ============================================================================
# SLURM Job Script for Training on Monsoon HPC
# ============================================================================
# This script submits a training job to the Monsoon cluster.
#
# Usage:
#   sbatch scripts/slurm_train.sh
#
# To customize:
#   - Adjust time, memory, CPU/GPU requests as needed
#   - Modify config file path
#   - Change job name and output paths
# ============================================================================

# SLURM Job Configuration
# -----------------------
#SBATCH --job-name=reach_train          # Job name
#SBATCH --output=logs/slurm_%j.out      # Standard output log
#SBATCH --error=logs/slurm_%j.err       # Standard error log
#SBATCH --time=24:00:00                 # Time limit (24 hours)
#SBATCH --partition=gpu                 # Partition (gpu, compute, etc.)
#SBATCH --nodes=1                       # Number of nodes
#SBATCH --ntasks=1                      # Number of tasks
#SBATCH --cpus-per-task=8               # CPU cores per task
#SBATCH --mem=32G                       # Memory per node
#SBATCH --gres=gpu:1                    # Number of GPUs (if using GPU)

# TODO: Uncomment and configure email notifications
# #SBATCH --mail-type=BEGIN,END,FAIL
# #SBATCH --mail-user=your_email@nau.edu

# Print job information
# ---------------------
echo "========================================="
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "========================================="

# Load required modules
# ---------------------
# TODO: Adjust module versions as needed for Monsoon
# module load python/3.10
# module load cuda/11.8
# module load cudnn/8.6

# Activate virtual environment
# ----------------------------
# TODO: Create virtual environment first:
#   python -m venv venv
#   source venv/bin/activate
#   pip install -r requirements.txt
# 
# source venv/bin/activate

# Set environment variables
# -------------------------
# export PYTHONPATH="${PYTHONPATH}:$(pwd)"
# export CUDA_VISIBLE_DEVICES=0  # Use first GPU

# Print Python and package versions
# ----------------------------------
# echo "Python version:"
# python --version
# echo ""
# echo "PyTorch version:"
# python -c "import torch; print(torch.__version__)"
# echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
# echo ""

# Run training script
# -------------------
# TODO: Adjust config file path as needed
# python scripts/train.py \
#     --config config/default.yaml \
#     --seed 42

# Print completion information
# ----------------------------
echo "========================================="
echo "End Time: $(date)"
echo "========================================="

