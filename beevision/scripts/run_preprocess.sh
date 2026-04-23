#!/usr/bin/env bash
# Run all ingest pipelines in order, then write shards.
# Usage: bash scripts/run_preprocess.sh [configs/data.yaml] [--dry-run]
set -euo pipefail

CONFIG="${1:-configs/data.yaml}"
shift || true
EXTRA_ARGS=("$@")

export BEEVISION_ROOT="${BEEVISION_ROOT:-/oscar/scratch/${USER}/beevision}"

echo "[run_preprocess] BEEVISION_ROOT=$BEEVISION_ROOT"
echo "[run_preprocess] CONFIG=$CONFIG"

python -m beevision.data.ingest_varroa       --config "$CONFIG" "${EXTRA_ARGS[@]}"
python -m beevision.data.ingest_beeimage     --config "$CONFIG" "${EXTRA_ARGS[@]}"
python -m beevision.data.ingest_deepbee_cls  --config "$CONFIG" "${EXTRA_ARGS[@]}"
python -m beevision.data.ingest_deepbee_seg  --config "$CONFIG" "${EXTRA_ARGS[@]}"
python -m beevision.data.shard_writer        --config "$CONFIG" "${EXTRA_ARGS[@]}"
python -m beevision.data.sanity              --config "$CONFIG" "${EXTRA_ARGS[@]}"

echo "[run_preprocess] done."
