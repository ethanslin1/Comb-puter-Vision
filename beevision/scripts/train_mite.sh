#!/bin/bash
# Foreground (interactive) training of the ResNet-50 mite classifier.
#
# Usage:
#   bash scripts/train_mite.sh                         # uses configs/train_mite.yaml
#   bash scripts/train_mite.sh configs/custom.yaml     # alt config
#
# Logs stream to stdout and to $BEEVISION_ROOT/interim/_logs/train_mite.log.
# Checkpoints go to $BEEVISION_ROOT/processed/checkpoints/mite/.

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${1:-$REPO_DIR/configs/train_mite.yaml}"

export BEEVISION_ROOT="${BEEVISION_ROOT:-/oscar/scratch/$USER/beevision}"
export PYTHONPATH="$REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

LOG_DIR="$BEEVISION_ROOT/interim/_logs"
mkdir -p "$LOG_DIR"

echo "=================================================================="
echo "beevision train_mite"
echo "start         : $(date -Iseconds)"
echo "BEEVISION_ROOT: $BEEVISION_ROOT"
echo "CONFIG        : $CONFIG"
echo "=================================================================="

cd "$REPO_DIR"
python3 -u -m beevision.models.mite.train --config "$CONFIG" \
    2>&1 | tee -a "$LOG_DIR/train_mite.log"
