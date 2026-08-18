#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

EPISODES="${ROBOCOP_V11_EPISODES:-20}"
MAX_WALK="${ROBOCOP_V11_MAX_WALK:-2000}"
BLOCK="${ROBOCOP_WALK_BLOCK:-150}"
HTTP_PORT="${ROBOCOP_HTTP_PORT:-8081}"
HTTP_SECONDS="${ROBOCOP_HTTP_SECONDS:-3600}"
RESULT_DIR="$ROOT/results/v11_multi_episode"
COMBINED="$RESULT_DIR/combined_trace.jsonl"
SUMMARY="$RESULT_DIR/summary.txt"
ANALYSIS="$ROOT/results/v11_full_body_holdout.txt"
PACKAGE="$ROOT/results/robocop_v11_latest.tar.gz"
SERVER_CONTAINER="robocop-rcssservermj-v11"

cleanup_server() { docker rm -f "$SERVER_CONTAINER" >/dev/null 2>&1 || true; }
trap cleanup_server EXIT INT TERM

mkdir -p "$RESULT_DIR"
rm -f "$RESULT_DIR"/episode_*.jsonl "$RESULT_DIR"/episode_*.json "$RESULT_DIR"/server_*.log "$COMBINED" "$SUMMARY" "$ANALYSIS" "$PACKAGE"
: > "$COMBINED"

bash scripts/fetch_bahiart_mujoco_external.sh

echo "============================================"
echo "RoboCOP — V11 FULL BODY DATASET"
echo "============================================"
echo "episodes:         $EPISODES"
echo "max walk/run:     $MAX_WALK"
echo "command block:    $BLOCK"
echo "dataset:          $RESULT_DIR"

docker build -f Dockerfile.rcssservermj -t robocop-rcssservermj:v11 .
docker build -f Dockerfile.bahiart-passive -t robocop-bahiart-passive:latest .

for ((i=1; i<=EPISODES; i++)); do
  printf -v PAD "%03d" "$i"
  TRACE="$RESULT_DIR/episode_${PAD}.jsonl"
  RUN_SUMMARY="$RESULT_DIR/episode_${PAD}.json"
  SERVER_LOG="$RESULT_DIR/server_${PAD}.log"

  echo
  echo "--------------------------------------------"
  echo "RoboCOP V11 — run $i/$EPISODES"
  echo "--------------------------------------------"

  cleanup_server
  docker run -d --name "$SERVER_CONTAINER" --network host robocop-rcssservermj:v11 >/dev/null
  sleep 3

  set +e
  docker run --rm --network host -v "$ROOT:/workspace" -w /workspace \
    robocop-bahiart-passive:latest \
    python scripts/run_bahiart_walk_probe.py \
      --host 127.0.0.1 --port 60000 --number 2 \
      --run-id "$i" --stop-on-fall --block "$BLOCK" \
      --max-walk-cycles "$MAX_WALK" \
      --trace "/workspace/results/v11_multi_episode/episode_${PAD}.jsonl" \
      --summary "/workspace/results/v11_multi_episode/episode_${PAD}.json"
  RC=$?
  set -e

  docker logs "$SERVER_CONTAINER" > "$SERVER_LOG" 2>&1 || true
  cleanup_server

  [[ -s "$TRACE" ]] && cat "$TRACE" >> "$COMBINED"
  if [[ ! -s "$RUN_SUMMARY" ]]; then
    printf '{"run_id":%d,"reason":"PROCESS_ERROR","return_code":%d,"walk_cycles":0,"falls":0}\n' "$i" "$RC" > "$RUN_SUMMARY"
  fi

  python3 - "$RUN_SUMMARY" <<'PY'
import json,sys
r=json.load(open(sys.argv[1],encoding='utf-8'))
print(f"[V11] run={r.get('run_id')} reason={r.get('reason')} walk={r.get('walk_cycles',0)} falls={r.get('falls',0)}")
PY

done

python3 - "$RESULT_DIR" > "$SUMMARY" <<'PY'
import json,statistics,sys
from pathlib import Path
root=Path(sys.argv[1]); runs=[]
for p in sorted(root.glob('episode_*.json')):
    try:runs.append(json.loads(p.read_text(encoding='utf-8')))
    except:pass
lengths=[int(r.get('walk_cycles',0)) for r in runs]
print('='*76)
print('RoboCOP — V11 DATASET SUMMARY')
print('='*76)
print(f'Runs:                    {len(runs)}')
print(f'Total walking cycles:    {sum(lengths)}')
if lengths:
    print(f'Mean episode:            {statistics.mean(lengths):.2f}')
    print(f'Median episode:          {statistics.median(lengths):.2f}')
    print(f'Best episode:            {max(lengths)}')
    print(f'Worst episode:           {min(lengths)}')
    print(f'>=250 cycles:            {sum(x>=250 for x in lengths)}')
    print(f'>=500 cycles:            {sum(x>=500 for x in lengths)}')
print('='*76)
PY
cat "$SUMMARY"

echo
echo "============================================"
echo "RoboCOP — V11 HOLDOUT ANALYSIS"
echo "============================================"
docker run --rm -v "$ROOT:/workspace" -w /workspace robocop-bahiart-passive:latest \
  python scripts/analyze_full_body_v11.py \
    --trace /workspace/results/v11_multi_episode/combined_trace.jsonl | tee "$ANALYSIS"

rm -f "$PACKAGE"
tar -czf "$PACKAGE" -C "$ROOT/results" v11_multi_episode v11_full_body_holdout.txt

if [[ -f "$ROOT/results/http_8081.pid" ]]; then
  OLD="$(cat "$ROOT/results/http_8081.pid" 2>/dev/null || true)"
  [[ -n "$OLD" ]] && kill "$OLD" >/dev/null 2>&1 || true
fi
(
  cd "$ROOT/results"
  timeout "$HTTP_SECONDS" python3 -m http.server "$HTTP_PORT" --bind 0.0.0.0 >/tmp/robocop_http_${HTTP_PORT}.log 2>&1
) &
echo $! > "$ROOT/results/http_8081.pid"
IP="${ROBOCOP_PUBLIC_IP:-$(hostname -I | awk '{print $1}')}"

echo
echo "============================================"
echo "RoboCOP — V11 RESULTADO PRONTO"
echo "============================================"
echo "arquivo: $PACKAGE"
echo "URL:     http://${IP}:${HTTP_PORT}/robocop_v11_latest.tar.gz"
echo "analise: http://${IP}:${HTTP_PORT}/v11_full_body_holdout.txt"
echo "tempo:   ${HTTP_SECONDS}s"
