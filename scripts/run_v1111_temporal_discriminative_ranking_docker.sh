#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRACE="$ROOT/results/v115_multi_episode/combined_trace.jsonl"
echo '============================================'
echo 'RoboCOP — V11.11 TEMPORAL RANKING'
echo '============================================'
[ -s "$TRACE" ] || { echo "missing trace: $TRACE"; exit 2; }
docker run --rm -v "$ROOT:/workspace" -w /workspace robocop-bahiart-passive:latest python scripts/analyze_v1111_temporal_discriminative_ranking.py --trace /workspace/results/v115_multi_episode/combined_trace.jsonl | tee "$ROOT/results/v1111_temporal_discriminative_ranking.txt"
tar -czf "$ROOT/results/robocop_v1111_latest.tar.gz" -C "$ROOT/results" v1111_temporal_discriminative_ranking.txt
echo "result: $ROOT/results/robocop_v1111_latest.tar.gz"
