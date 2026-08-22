#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TRACE="$ROOT/results/v11_multi_episode/combined_trace.jsonl"
OUT="$ROOT/results/v111_indexed_holdout.txt"
[[ -f "$TRACE" ]] || { echo "Missing V11 trace: $TRACE"; exit 1; }
echo "============================================"
echo "RoboCOP — V11.1 INDEXED HOLDOUT ANALYSIS"
echo "============================================"
echo "Reusing existing V11 dataset; no simulation will be repeated."
docker run --rm -v "$ROOT:/workspace" -w /workspace robocop-bahiart-passive:latest python scripts/analyze_full_body_v111_indexed.py --trace /workspace/results/v11_multi_episode/combined_trace.jsonl --progress-every 250 | tee "$OUT"
cd "$ROOT/results"
tar -czf robocop_v111_latest.tar.gz v111_indexed_holdout.txt v11_multi_episode/summary.txt v11_multi_episode/combined_trace.jsonl 2>/dev/null || tar -czf robocop_v111_latest.tar.gz v111_indexed_holdout.txt
if [[ -f http_8081.pid ]]; then kill "$(cat http_8081.pid)" 2>/dev/null || true; fi
nohup timeout 3600 python3 -m http.server 8081 --bind 0.0.0.0 --directory "$ROOT/results" > "$ROOT/results/http_8081.log" 2>&1 & echo $! > "$ROOT/results/http_8081.pid"
IP="$(hostname -I | awk '{print $1}')"
echo;echo "Resultado: $OUT";echo "Pacote: $ROOT/results/robocop_v111_latest.tar.gz";echo "HTTP local: http://$IP:8081/robocop_v111_latest.tar.gz"
