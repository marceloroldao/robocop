#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

TRACE="$ROOT/results/bahiart_multi_episode/combined_trace.jsonl"
OUT="$ROOT/results/v10_interpolated_corrective.txt"
HTTP_PORT="${ROBOCOP_HTTP_PORT:-8081}"
HTTP_SECONDS="${ROBOCOP_HTTP_SECONDS:-3600}"

if [[ ! -s "$TRACE" ]]; then
  echo "Trace not found: $TRACE" >&2
  echo "Run scripts/run_bahiart_multi_episode_docker.sh first." >&2
  exit 2
fi

mkdir -p results

docker build -f Dockerfile.bahiart-passive -t robocop-bahiart-passive:latest .

docker run --rm \
  -v "$ROOT:/workspace" \
  -w /workspace \
  robocop-bahiart-passive:latest \
  python scripts/analyze_interpolated_corrective_v10.py \
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

IP="$(hostname -I 2>/dev/null | awk '{print $1}')"
PUBLIC_IP="${ROBOCOP_PUBLIC_IP:-$IP}"

echo
echo "============================================"
echo "RoboCOP — V10 RESULTADO PRONTO"
echo "============================================"
echo "arquivo: $OUT"
echo "URL:     http://${PUBLIC_IP}:${HTTP_PORT}/v10_interpolated_corrective.txt"
echo "HTTP PID: $HTTP_PID"
echo "tempo:   ${HTTP_SECONDS}s"
