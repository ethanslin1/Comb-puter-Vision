#!/bin/bash
# SLURM wrapper for any beevision.data.ingest_<source> module.
#
# Usage:
#   sbatch scripts/slurm_ingest.sbatch <source>          # e.g. deepbee_seg
#   sbatch scripts/slurm_ingest.sbatch <source> --limit 50
#
# Environment overrides (export before sbatch, or pass with --export):
#   BEEVISION_ROOT   data root (default /oscar/scratch/$USER/beevision)
#   BEEVISION_CONFIG YAML config (default configs/data.yaml relative to repo)
#   SLURM_ACCOUNT    account for --account (default unset → your default)
#
# Logs land in BEEVISION_ROOT/interim/_logs/slurm-<source>-<jobid>.out.
# The ingest module already logs to interim/_logs/ingest_<source>.log.
#
#SBATCH --job-name=beevision-ingest
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=/oscar/scratch/%u/beevision/interim/_logs/slurm-%x-%j.out
#SBATCH --error=/oscar/scratch/%u/beevision/interim/_logs/slurm-%x-%j.err

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "usage: sbatch $0 <source> [extra args]" >&2
    exit 2
fi

SOURCE="$1"
shift || true

# SLURM copies this script to its spool dir, so $0 resolves there instead
# of the checkout. Use SLURM_SUBMIT_DIR (the cwd at sbatch time) as the
# repo root; callers must invoke sbatch from the beevision/ directory.
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

mkdir -p "$BEEVISION_ROOT/interim/_logs"

echo "=================================================================="
echo "beevision ingest: $SOURCE"
echo "job id        : ${SLURM_JOB_ID:-N/A}"
echo "node          : $(hostname)"
echo "start         : $(date -Iseconds)"
echo "BEEVISION_ROOT: $BEEVISION_ROOT"
echo "CONFIG        : $CONFIG"
echo "repo          : $REPO_DIR"
echo "extra args    : $*"
echo "=================================================================="

cd "$REPO_DIR"

python3 -u -m "beevision.data.ingest_${SOURCE}" --config "$CONFIG" "$@"
status=$?

echo "=================================================================="
echo "end           : $(date -Iseconds)"
echo "exit          : $status"
echo "=================================================================="
exit $status
