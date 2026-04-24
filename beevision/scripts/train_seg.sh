#!/bin/bash
# Foreground (interactive) training of the DINOv2-UNet segmentation model.
#
# Usage:
#   bash scripts/train_seg.sh                        # uses configs/train_seg.yaml
#   bash scripts/train_seg.sh configs/custom.yaml    # alt config
#
# Logs stream to stdout and to $BEEVISION_ROOT/interim/_logs/train_seg.log.
# Checkpoints go to $BEEVISION_ROOT/processed/checkpoints/segmentation/.

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${1:-$REPO_DIR/configs/train_seg.yaml}"

export BEEVISION_ROOT="${BEEVISION_ROOT:-/oscar/scratch/$USER/beevision}"
export PYTHONPATH="$REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

LOG_DIR="$BEEVISION_ROOT/interim/_logs"
mkdir -p "$LOG_DIR"

echo "=================================================================="
echo "beevision train_seg"
echo "start         : $(date -Iseconds)"
echo "BEEVISION_ROOT: $BEEVISION_ROOT"
echo "CONFIG        : $CONFIG"
echo "=================================================================="

cd "$REPO_DIR"
python3 -u -m beevision.models.segmentation.train --config "$CONFIG" \
    2>&1 | tee -a "$LOG_DIR/train_seg.log"
