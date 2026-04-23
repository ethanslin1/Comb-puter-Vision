#!/bin/bash
# SLURM wrapper for beevision.data.shard_writer.
#
# Packs the interim parquets + PNGs/masks into WebDataset tar shards
# (256 MiB each) under $BEEVISION_ROOT/processed/shards/. Determinism
# (byte-identical tars across reruns) is guaranteed by shard_writer
# itself; this wrapper only supplies the SLURM + environment setup.
#
# Usage:
#   sbatch scripts/slurm_shards.sbatch                 # all four sources
#   sbatch scripts/slurm_shards.sbatch --sources varroa beeimage
#   sbatch scripts/slurm_shards.sbatch --dry-run       # inventory only
#
#SBATCH --job-name=beevision-shards
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/oscar/scratch/%u/beevision/interim/_logs/slurm-%x-%j.out
#SBATCH --error=/oscar/scratch/%u/beevision/interim/_logs/slurm-%x-%j.err

set -euo pipefail

REPO_DIR="${BEEVISION_REPO_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
if [[ ! -d "$REPO_DIR/src/beevision" ]]; then
    echo "ERROR: BEEVISION_REPO_DIR='$REPO_DIR' has no src/beevision/." >&2
    echo "       Run sbatch from the beevision/ directory or export BEEVISION_REPO_DIR." >&2
    exit 2
fi
CONFIG="${BEEVISION_CONFIG:-$REPO_DIR/configs/data.yaml}"

export BEEVISION_ROOT="${BEEVISION_ROOT:-/oscar/scratch/$USER/beevision}"
export PYTHONPATH="$REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export PATH="$HOME/.local/bin:$PATH"

mkdir -p "$BEEVISION_ROOT/interim/_logs" "$BEEVISION_ROOT/processed/shards"

echo "=================================================================="
echo "beevision shards"
echo "job id        : ${SLURM_JOB_ID:-N/A}"
echo "node          : $(hostname)"
echo "start         : $(date -Iseconds)"
echo "BEEVISION_ROOT: $BEEVISION_ROOT"
echo "CONFIG        : $CONFIG"
echo "repo          : $REPO_DIR"
echo "extra args    : $*"
echo "=================================================================="

cd "$REPO_DIR"

python3 -u -m beevision.data.shard_writer --config "$CONFIG" "$@"
status=$?

echo "=================================================================="
echo "end           : $(date -Iseconds)"
echo "exit          : $status"
if [[ $status -eq 0 ]]; then
    echo "shards        :"
    ls -lh "$BEEVISION_ROOT/processed/shards/" | tail -n +2 | awk '{print "  " $NF "  " $5}'
fi
echo "=================================================================="
exit $status
