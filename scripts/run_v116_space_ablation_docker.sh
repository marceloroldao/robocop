#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TRACE="${1:-$ROOT/results/v115_multi_episode/combined_trace.jsonl}"
OUT="$ROOT/results/v116_space_ablation.txt"
PACKAGE="$ROOT/results/robocop_v116_latest.tar.gz"
[[ -s "$TRACE" ]] || { echo "Missing corrected V11.5 trace: $TRACE"; exit 1; }
echo "============================================"
echo "RoboCOP — V11.6 SPACE / TEMPORAL ABLATION"
echo "============================================"
echo "trace: $TRACE"
# Mandatory guard against the old frozen-body dataset.
docker run --rm -v "$ROOT:/workspace" -w /workspace robocop-bahiart-passive:latest \
  python scripts/validate_v11_full_body_trace.py "${TRACE/$ROOT/\/workspace}" --max-rows 2000

docker run --rm -v "$ROOT:/workspace" -w /workspace robocop-bahiart-passive:latest \
  python scripts/analyze_v116_space_ablation.py --trace "${TRACE/$ROOT/\/workspace}" --progress-every 500 | tee "$OUT"

tar -czf "$PACKAGE" -C "$ROOT/results" "$(basename "$OUT")"
echo;echo "Result: $OUT";echo "Package: $PACKAGE"
