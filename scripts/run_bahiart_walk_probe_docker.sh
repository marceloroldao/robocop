#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

RESULTS="$ROOT/results"
TRACE="$RESULTS/bahiart_walk_probe_trace.jsonl"
SERVER_CONTAINER="robocop-rcssservermj-walk"
MAX_WALK_CYCLES="${ROBOCOP_MAX_WALK_CYCLES:-12000}"
MAX_RECOVERY_CYCLES="${ROBOCOP_MAX_RECOVERY_CYCLES:-2500}"
BLOCK="${ROBOCOP_WALK_BLOCK:-150}"
HTTP_PORT="${ROBOCOP_RESULTS_PORT:-8081}"
HTTP_SECONDS="${ROBOCOP_RESULTS_HTTP_SECONDS:-3600}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_LOG="$RESULTS/bahiart_walk_probe_${STAMP}.log"
SERVER_LOG="$RESULTS/rcssservermj_walk_${STAMP}.log"
SUMMARY="$RESULTS/robocop_walk_summary_${STAMP}.txt"
ARCHIVE="$RESULTS/robocop_walk_results_${STAMP}.tar.gz"
LATEST_ARCHIVE="$RESULTS/robocop_walk_latest.tar.gz"
HTTP_LOG="$RESULTS/http_${HTTP_PORT}.log"
HTTP_PID="$RESULTS/http_${HTTP_PORT}.pid"

cleanup_sim() {
  docker rm -f "$SERVER_CONTAINER" >/dev/null 2>&1 || true
}
trap cleanup_sim EXIT INT TERM

mkdir -p "$RESULTS"
bash scripts/fetch_bahiart_mujoco_external.sh

echo "============================================"
echo "RoboCOP — episodic BahiaRT Walk probe"
echo "============================================"
echo "walk cycles:     $MAX_WALK_CYCLES"
echo "recovery limit:  $MAX_RECOVERY_CYCLES"
echo "command block:   $BLOCK"
echo "trace:           $TRACE"
echo "run log:         $RUN_LOG"

docker build -f Dockerfile.rcssservermj -t robocop-rcssservermj:walk .
docker build -f Dockerfile.bahiart-passive -t robocop-bahiart-passive:latest .

cleanup_sim
docker run -d --name "$SERVER_CONTAINER" --network host robocop-rcssservermj:walk >/dev/null
sleep 4

set +e
docker run --rm \
  --network host \
  -v "$ROOT:/workspace" \
  -w /workspace \
  robocop-bahiart-passive:latest \
  python scripts/run_bahiart_walk_probe.py \
    --host 127.0.0.1 \
    --port 60000 \
    --number 2 \
    --block "$BLOCK" \
    --max-walk-cycles "$MAX_WALK_CYCLES" \
    --max-recovery-cycles "$MAX_RECOVERY_CYCLES" \
    --trace /workspace/results/bahiart_walk_probe_trace.jsonl \
  2>&1 | tee "$RUN_LOG"
AGENT_RC=${PIPESTATUS[0]}
set -e

docker logs "$SERVER_CONTAINER" >"$SERVER_LOG" 2>&1 || true

{
  echo "RoboCOP — BahiaRT Walk probe result bundle"
  echo "timestamp_utc=$STAMP"
  echo "agent_exit_code=$AGENT_RC"
  echo "walk_cycles_target=$MAX_WALK_CYCLES"
  echo "recovery_cycle_limit=$MAX_RECOVERY_CYCLES"
  echo "command_block=$BLOCK"
  echo "trace=$TRACE"
  echo "run_log=$RUN_LOG"
  echo "server_log=$SERVER_LOG"
  if [[ -f "$TRACE" ]]; then
    echo "trace_bytes=$(wc -c < "$TRACE")"
    echo "trace_lines=$(wc -l < "$TRACE")"
  else
    echo "trace_missing=1"
  fi
  echo
  echo "--- last run messages ---"
  tail -n 80 "$RUN_LOG" 2>/dev/null || true
} > "$SUMMARY"

rm -f "$LATEST_ARCHIVE"
TAR_ITEMS=("$(basename "$RUN_LOG")" "$(basename "$SERVER_LOG")" "$(basename "$SUMMARY")")
if [[ -f "$TRACE" ]]; then
  TAR_ITEMS+=("$(basename "$TRACE")")
fi
(
  cd "$RESULTS"
  tar -czf "$(basename "$ARCHIVE")" "${TAR_ITEMS[@]}"
  cp "$(basename "$ARCHIVE")" "$(basename "$LATEST_ARCHIVE")"
)

# Replace an older RoboCOP result server, if one was started by this script.
if [[ -f "$HTTP_PID" ]]; then
  OLD_PID="$(cat "$HTTP_PID" 2>/dev/null || true)"
  if [[ -n "$OLD_PID" ]] && kill -0 "$OLD_PID" 2>/dev/null; then
    kill "$OLD_PID" 2>/dev/null || true
    sleep 1
  fi
  rm -f "$HTTP_PID"
fi

# Serve only the results directory. The timeout closes the public listener
# automatically (default: 1 hour) after the experiment finishes.
nohup timeout "$HTTP_SECONDS" python3 -m http.server "$HTTP_PORT" \
  --bind 0.0.0.0 --directory "$RESULTS" \
  >"$HTTP_LOG" 2>&1 < /dev/null &
HTTP_SERVER_PID=$!
echo "$HTTP_SERVER_PID" > "$HTTP_PID"
sleep 1

HOST_IP="$(hostname -I 2>/dev/null | awk '{print $1}')"
HOST_IP="${HOST_IP:-SEU_IP_DA_VPS}"

echo
echo "============================================"
echo "RoboCOP — RESULTADO PRONTO PARA DOWNLOAD"
echo "============================================"
echo "arquivo: $LATEST_ARCHIVE"
echo "URL:     http://${HOST_IP}:${HTTP_PORT}/robocop_walk_latest.tar.gz"
echo "pasta:   http://${HOST_IP}:${HTTP_PORT}/"
echo "HTTP PID: $HTTP_SERVER_PID"
echo "tempo:   ${HTTP_SECONDS}s (fecha automaticamente)"
echo
echo "Para fechar antes:"
echo "  kill \$(cat '$HTTP_PID')"
echo
echo "Se a URL nao abrir externamente, libere TCP/${HTTP_PORT} no firewall da VPS."
echo "============================================"

exit "$AGENT_RC"
