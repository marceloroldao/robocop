#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FCP_DIR="${FCP_DIR:-$ROOT/.external/FCPCodebase}"
TRACE_OUT="${TRACE_OUT:-$ROOT/results/fcp_walk_trace.jsonl}"
SIMSPARK_HOST="${SIMSPARK_HOST:-127.0.0.1}"
SIMSPARK_PORT="${SIMSPARK_PORT:-3100}"
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
echo "[RoboCOP] connecting agent to ${SIMSPARK_HOST}:${SIMSPARK_PORT} (headless/debug off)"
# FC Portugal defaults to Debug Mode=1. In a headless Docker runtime that also
# enables monitor/drawing facilities. Force -D 0 and pass the actual server
# endpoint explicitly; user arguments may add team/uniform/robot options.
exec python Run_Player.py -i "$SIMSPARK_HOST" -p "$SIMSPARK_PORT" -D 0 "${PLAYER_ARGS[@]}"
