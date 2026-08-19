#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRACE="$ROOT/results/v115_multi_episode/combined_trace.jsonl"
OUT="$ROOT/results/v119_consensus_gate.txt"
PACKAGE="$ROOT/results/robocop_v119_latest.tar.gz"

echo '============================================'
echo 'RoboCOP — V11.9 CONSENSUS GATE'
echo '============================================'
echo "trace: $TRACE"

[[ -s "$TRACE" ]] || { echo "Missing V11.5 trace: $TRACE"; exit 1; }

# Reuse the same analysis image already used successfully by V11.7/V11.8.
# Do not build from a non-existent docker/ directory.
docker image inspect robocop-bahiart-passive:latest >/dev/null 2>&1 || {
  echo 'Missing Docker image: robocop-bahiart-passive:latest'
  echo 'Run one of the existing BahiaRT analysis runners first to create it.'
  exit 2
}

docker run --rm \
  -v "$ROOT:/workspace" \
  -w /workspace \
  robocop-bahiart-passive:latest \
  python scripts/analyze_v119_consensus_gate.py \
    --trace /workspace/results/v115_multi_episode/combined_trace.jsonl \
    --progress-every 500 | tee "$OUT"

rm -f "$PACKAGE"
tar -czf "$PACKAGE" -C "$ROOT/results" v119_consensus_gate.txt

echo
echo "Result:  $OUT"
echo "Package: $PACKAGE"
