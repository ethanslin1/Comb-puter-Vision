#!/bin/bash
# Launch the BeeVision Streamlit dashboard.
#
# Usage:
#   bash scripts/run_app.sh                        # default port 8501
#   bash scripts/run_app.sh --server.port 8080     # pass any flags through
#
# Requires `streamlit` (and a HealthReport JSON to load, or use Demo mode).
# Install once: pip install streamlit

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DASHBOARD="$REPO_DIR/src/beevision/app/dashboard.py"

export BEEVISION_ROOT="${BEEVISION_ROOT:-/oscar/scratch/$USER/beevision}"
export PYTHONPATH="$REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

if ! command -v streamlit >/dev/null 2>&1; then
    echo "ERROR: streamlit is not installed. Install with:" >&2
    echo "       pip install streamlit" >&2
    exit 2
fi

echo "=================================================================="
echo "beevision dashboard"
echo "BEEVISION_ROOT: $BEEVISION_ROOT"
echo "DASHBOARD     : $DASHBOARD"
echo "=================================================================="

cd "$REPO_DIR"
exec streamlit run "$DASHBOARD" "$@"
