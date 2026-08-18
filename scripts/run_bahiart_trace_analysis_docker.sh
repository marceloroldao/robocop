#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

IMAGE="robocop-bahiart-passive:latest"
TRACE="${1:-$ROOT/results/bahiart_passive_trace.jsonl}"

if [[ ! -s "$TRACE" ]]; then
  echo "[RoboCOP] trace not found or empty: $TRACE" >&2
  exit 2
fi

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "[RoboCOP] building Python 3.13 analysis runtime..."
  docker build -f Dockerfile.bahiart-passive -t "$IMAGE" .
fi

echo "============================================"
echo "RoboCOP — selective trace consolidation"
echo "============================================"
echo "trace: $TRACE"

docker run --rm \
  -v "$ROOT:/workspace" \
  -w /workspace \
  "$IMAGE" \
  python scripts/analyze_bahiart_trace_selective.py \
    "/workspace/${TRACE#$ROOT/}"
