#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

TRACE="$ROOT/results/bahiart_multi_episode/combined_trace.jsonl"
OUT="$ROOT/results/v9_corrective_trajectory.txt"
HTTP_PORT="${ROBOCOP_HTTP_PORT:-8081}"
HTTP_SECONDS="${ROBOCOP_HTTP_SECONDS:-3600}"

if [[ ! -s "$TRACE" ]]; then
  echo "Missing multi-episode trace: $TRACE" >&2
  echo "Run scripts/run_bahiart_multi_episode_docker.sh first." >&2
  exit 2
fi

mkdir -p results

echo "============================================"
echo "RoboCOP — V9 corrective trajectory replay"
echo "============================================"
echo "trace: $TRACE"
echo

# Reuse the isolated Python 3.13 analysis image if present; rebuild otherwise.
docker build -f Dockerfile.bahiart-passive -t robocop-bahiart-passive:latest . >/dev/null

docker run --rm \
  -v "$ROOT:/workspace" \
  -w /workspace \
  robocop-bahiart-passive:latest \
  python scripts/analyze_corrective_trajectory_v9.py \
    --trace /workspace/results/bahiart_multi_episode/combined_trace.jsonl \
  | tee "$OUT"

if [[ -f "$ROOT/results/http_8081.pid" ]]; then
  OLD_PID="$(cat "$ROOT/results/http_8081.pid" 2>/dev/null || true)"
  if [[ -n "$OLD_PID" ]]; then kill "$OLD_PID" >/dev/null 2>&1 || true; fi
fi

(
  cd "$ROOT/results"
  timeout "$HTTP_SECONDS" python3 -m http.server "$HTTP_PORT" --bind 0.0.0.0 >/tmp/robocop_http_${HTTP_PORT}.log 2>&1
) &
HTTP_PID=$!
echo "$HTTP_PID" > "$ROOT/results/http_8081.pid"

PUBLIC_IP="${ROBOCOP_PUBLIC_IP:-$(hostname -I 2>/dev/null | awk '{print $1}')}"

echo
echo "============================================"
echo "RoboCOP — V9 RESULTADO PRONTO"
echo "============================================"
echo "arquivo: $OUT"
echo "URL:     http://${PUBLIC_IP}:${HTTP_PORT}/v9_corrective_trajectory.txt"
echo "HTTP PID: $HTTP_PID"
echo "tempo:   ${HTTP_SECONDS}s"
