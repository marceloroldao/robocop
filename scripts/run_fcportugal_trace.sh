#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FCP_DIR="${FCP_DIR:-$ROOT/.external/FCPCodebase}"
TRACE_OUT="${TRACE_OUT:-$ROOT/results/fcp_walk_trace.jsonl}"
PLAYER_ARGS=("$@")

if [ ! -d "$FCP_DIR" ]; then
  echo "FC Portugal codebase not found at: $FCP_DIR" >&2
  echo "Run: bash scripts/fetch_fcportugal_external.sh" >&2
  exit 2
fi

mkdir -p "$(dirname "$TRACE_OUT")"
export ROBOCOP_FCP_TRACE="$TRACE_OUT"
export PYTHONPATH="$ROOT/scripts/fcp_runtime:$ROOT:$FCP_DIR${PYTHONPATH:+:$PYTHONPATH}"

cd "$FCP_DIR"
echo "[RoboCOP] tracing FC Portugal Walk -> $TRACE_OUT"
exec python Run_Player.py "${PLAYER_ARGS[@]}"
