#!/bin/bash
# SLURM wrapper for DINOv2-UNet segmentation training on Oscar GPU.
#
# Usage:
#   sbatch scripts/slurm_train_seg.sbatch                        # default config
#   sbatch scripts/slurm_train_seg.sbatch configs/custom.yaml    # alt config
#
# Environment overrides (export before sbatch):
#   BEEVISION_ROOT    data root (default /oscar/scratch/$USER/beevision)
#   BEEVISION_REPO_DIR override for the repo root if not invoking from it
#
# Logs land in BEEVISION_ROOT/interim/_logs/slurm-<jobid>.{out,err}.
#
#SBATCH --job-name=beevision-train-seg
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/oscar/scratch/%u/beevision/interim/_logs/slurm-%x-%j.out
#SBATCH --error=/oscar/scratch/%u/beevision/interim/_logs/slurm-%x-%j.err

set -euo pipefail

REPO_DIR="${BEEVISION_REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
if [[ ! -d "$REPO_DIR/src/beevision" ]]; then
    echo "ERROR: BEEVISION_REPO_DIR='$REPO_DIR' has no src/beevision/." >&2
    echo "       Run sbatch from the beevision/ directory or export BEEVISION_REPO_DIR." >&2
    exit 2
fi

CONFIG="${1:-$REPO_DIR/configs/train_seg.yaml}"

export BEEVISION_ROOT="${BEEVISION_ROOT:-/oscar/scratch/$USER/beevision}"
export PYTHONPATH="$REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export PATH="$HOME/.local/bin:$PATH"

mkdir -p "$BEEVISION_ROOT/interim/_logs"

echo "=================================================================="
echo "beevision train_seg"
echo "job id        : ${SLURM_JOB_ID:-N/A}"
echo "node          : $(hostname)"
echo "start         : $(date -Iseconds)"
echo "BEEVISION_ROOT: $BEEVISION_ROOT"
echo "CONFIG        : $CONFIG"
echo "repo          : $REPO_DIR"
echo "GPU           : $(nvidia-smi -L 2>/dev/null || echo 'no nvidia-smi')"
echo "=================================================================="

cd "$REPO_DIR"

python3 -u -m beevision.models.segmentation.train --config "$CONFIG"
status=$?

echo "=================================================================="
echo "end           : $(date -Iseconds)"
echo "exit          : $status"
echo "=================================================================="
exit $status
