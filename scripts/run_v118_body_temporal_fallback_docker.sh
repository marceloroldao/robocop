#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TRACE="$ROOT/results/v115_multi_episode/combined_trace.jsonl"
OUT="$ROOT/results/v118_body_temporal_fallback.txt"
PACKAGE="$ROOT/results/robocop_v118_latest.tar.gz"
[[ -s "$TRACE" ]] || { echo "Missing V11.5 trace: $TRACE"; exit 1; }

echo "============================================"
echo "RoboCOP — V11.8 BODY-TEMPORAL FALLBACK"
echo "============================================"
echo "trace: $TRACE"

docker run --rm -v "$ROOT:/workspace" -w /workspace robocop-bahiart-passive:latest \
  python scripts/analyze_v118_body_temporal_fallback.py \
    --trace /workspace/results/v115_multi_episode/combined_trace.jsonl \
    --progress-every 500 | tee "$OUT"

rm -f "$PACKAGE"
tar -czf "$PACKAGE" -C "$ROOT/results" v118_body_temporal_fallback.txt

echo
echo "Result:  $OUT"
echo "Package: $PACKAGE"
