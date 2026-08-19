#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRACE="$ROOT/results/v115_multi_episode/combined_trace.jsonl"
echo '============================================'
echo 'RoboCOP — V11.9 CONSENSUS GATE'
echo '============================================'
echo "trace: $TRACE"
[ -s "$TRACE" ] || { echo "missing trace: $TRACE"; exit 2; }
docker build -f "$ROOT/docker/Dockerfile.bahiart-passive" -t robocop-bahiart-passive:latest "$ROOT"
docker run --rm \
  -v "$ROOT:/workspace" \
  -w /workspace \
  robocop-bahiart-passive:latest \
  python scripts/analyze_v119_consensus_gate.py --trace /workspace/results/v115_multi_episode/combined_trace.jsonl
