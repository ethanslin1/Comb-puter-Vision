#!/bin/bash
# Foreground (interactive) training of the ResNet-50 7-class cell classifier.
#
# Usage:
#   bash scripts/train_cells.sh                          # uses configs/train_cells.yaml
#   bash scripts/train_cells.sh configs/custom.yaml      # alt config
#
# Logs stream to stdout and to $BEEVISION_ROOT/interim/_logs/train_cells.log.
# Checkpoints go to $BEEVISION_ROOT/processed/checkpoints/cells/.

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${1:-$REPO_DIR/configs/train_cells.yaml}"

export BEEVISION_ROOT="${BEEVISION_ROOT:-/oscar/scratch/$USER/beevision}"
export PYTHONPATH="$REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

LOG_DIR="$BEEVISION_ROOT/interim/_logs"
mkdir -p "$LOG_DIR"

echo "=================================================================="
echo "beevision train_cells"
echo "start         : $(date -Iseconds)"
echo "BEEVISION_ROOT: $BEEVISION_ROOT"
echo "CONFIG        : $CONFIG"
echo "=================================================================="

cd "$REPO_DIR"
python3 -u -m beevision.models.cells.train --config "$CONFIG" \
    2>&1 | tee -a "$LOG_DIR/train_cells.log"
