#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TRACE="$ROOT/results/bahiart_multi_episode/combined_trace.jsonl"
OUT="$ROOT/results/v103_adaptive_sensor.txt"
[[ -f "$TRACE" ]] || { echo "Missing trace: $TRACE"; exit 1; }
echo "============================================"
echo "RoboCOP — V10.3 adaptive local resolution"
echo "============================================"
docker run --rm -v "$ROOT:/workspace" -w /workspace robocop-bahiart-passive:latest python scripts/analyze_adaptive_sensor_v103.py --trace /workspace/results/bahiart_multi_episode/combined_trace.jsonl | tee "$OUT"
cd "$ROOT/results"
tar -czf robocop_v103_latest.tar.gz v103_adaptive_sensor.txt bahiart_multi_episode/summary.txt bahiart_multi_episode/combined_trace.jsonl 2>/dev/null || tar -czf robocop_v103_latest.tar.gz v103_adaptive_sensor.txt
if [[ -f http_8081.pid ]]; then kill "$(cat http_8081.pid)" 2>/dev/null || true; fi
nohup timeout 3600 python3 -m http.server 8081 --bind 0.0.0.0 --directory "$ROOT/results" > "$ROOT/results/http_8081.log" 2>&1 &
echo $! > "$ROOT/results/http_8081.pid"
IP="$(hostname -I | awk '{print $1}')"
echo; echo "Resultado: $OUT"; echo "Pacote:    $ROOT/results/robocop_v103_latest.tar.gz"; echo "HTTP local: http://$IP:8081/robocop_v103_latest.tar.gz"
