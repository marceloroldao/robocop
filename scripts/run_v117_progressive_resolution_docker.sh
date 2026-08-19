#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TRACE="$ROOT/results/v115_multi_episode/combined_trace.jsonl"
OUT="$ROOT/results/v117_progressive_resolution.txt"
[[ -s "$TRACE" ]] || { echo "Missing V11.5 trace: $TRACE"; exit 1; }
echo '============================================'
echo 'RoboCOP — V11.7 PROGRESSIVE RESOLUTION'
echo '============================================'
echo "trace: $TRACE"
docker run --rm -v "$ROOT:/workspace" -w /workspace robocop-bahiart-passive:latest \
 python scripts/analyze_v117_progressive_resolution.py \
 --trace /workspace/results/v115_multi_episode/combined_trace.jsonl --progress-every 500 | tee "$OUT"
echo
echo "Resultado: $OUT"
