#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TRACE="$ROOT/results/v115_multi_episode/combined_trace.jsonl"
OUT="$ROOT/results/original_vs_v117.txt"
PACKAGE="$ROOT/results/robocop_original_vs_v117.tar.gz"
[[ -s "$TRACE" ]] || { echo "Missing V11.5 trace: $TRACE"; exit 1; }

echo "============================================"
echo "RoboCOP — ORIGINAL vs BEST V11.7"
echo "============================================"
echo "trace: $TRACE"

docker run --rm -v "$ROOT:/workspace" -w /workspace robocop-bahiart-passive:latest \
  python scripts/analyze_original_vs_v117.py \
    --trace /workspace/results/v115_multi_episode/combined_trace.jsonl \
  | tee "$OUT"

rm -f "$PACKAGE"
tar -czf "$PACKAGE" -C "$ROOT/results" original_vs_v117.txt

echo
echo "Result:  $OUT"
echo "Package: $PACKAGE"
