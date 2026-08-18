#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

EPISODES="${ROBOCOP_EPISODES:-20}"
MAX_WALK="${ROBOCOP_EPISODE_MAX_WALK:-2000}"
BLOCK="${ROBOCOP_WALK_BLOCK:-150}"
HTTP_PORT="${ROBOCOP_HTTP_PORT:-8081}"
HTTP_SECONDS="${ROBOCOP_HTTP_SECONDS:-3600}"
RESUME="${ROBOCOP_RESUME:-0}"
RESULT_DIR="$ROOT/results/bahiart_multi_episode"
MEMORY="$RESULT_DIR/resolutive_memory.json"
COMBINED="$RESULT_DIR/combined_trace.jsonl"
SUMMARY="$RESULT_DIR/summary.txt"
PACKAGE="$ROOT/results/robocop_multi_episode_latest.tar.gz"
SERVER_CONTAINER="robocop-rcssservermj-multi"

cleanup_server() {
  docker rm -f "$SERVER_CONTAINER" >/dev/null 2>&1 || true
}
trap cleanup_server EXIT INT TERM

mkdir -p "$RESULT_DIR"
if [[ "$RESUME" != "1" ]]; then
  rm -f "$MEMORY" "$COMBINED" "$SUMMARY"
  rm -f "$RESULT_DIR"/episode_*.jsonl "$RESULT_DIR"/episode_*.json "$RESULT_DIR"/server_*.log
fi
: > "$COMBINED"

bash scripts/fetch_bahiart_mujoco_external.sh

echo "============================================"
echo "RoboCOP — persistent multi-episode benchmark"
echo "============================================"
echo "episodes:          $EPISODES"
echo "max walk/episode:  $MAX_WALK"
echo "command block:     $BLOCK"
echo "persistent memory: $MEMORY"

docker build -f Dockerfile.rcssservermj -t robocop-rcssservermj:multi .
docker build -f Dockerfile.bahiart-passive -t robocop-bahiart-passive:latest .

for ((i=1; i<=EPISODES; i++)); do
  printf -v PAD "%03d" "$i"
  TRACE="$RESULT_DIR/episode_${PAD}.jsonl"
  RUN_SUMMARY="$RESULT_DIR/episode_${PAD}.json"
  SERVER_LOG="$RESULT_DIR/server_${PAD}.log"

  echo
  echo "--------------------------------------------"
  echo "RoboCOP — run $i/$EPISODES"
  echo "--------------------------------------------"

  cleanup_server
  docker run -d --name "$SERVER_CONTAINER" --network host robocop-rcssservermj:multi >/dev/null
  sleep 3

  MEMORY_ARGS=(--memory-out /workspace/results/bahiart_multi_episode/resolutive_memory.json)
  if [[ -s "$MEMORY" ]]; then
    MEMORY_ARGS+=(--memory-in /workspace/results/bahiart_multi_episode/resolutive_memory.json)
  fi

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
      --run-id "$i" \
      --stop-on-fall \
      --block "$BLOCK" \
      --max-walk-cycles "$MAX_WALK" \
      --trace "/workspace/results/bahiart_multi_episode/episode_${PAD}.jsonl" \
      --summary "/workspace/results/bahiart_multi_episode/episode_${PAD}.json" \
      "${MEMORY_ARGS[@]}"
  RC=$?
  set -e

  docker logs "$SERVER_CONTAINER" > "$SERVER_LOG" 2>&1 || true
  cleanup_server

  if [[ -s "$TRACE" ]]; then
    cat "$TRACE" >> "$COMBINED"
  fi

  if [[ ! -s "$RUN_SUMMARY" ]]; then
    printf '{"run_id":%d,"reason":"PROCESS_ERROR","return_code":%d,"walk_cycles":0,"falls":0}\n' "$i" "$RC" > "$RUN_SUMMARY"
  fi

  python3 - "$RUN_SUMMARY" <<'PY'
import json, sys
p=sys.argv[1]
d=json.load(open(p, encoding='utf-8'))
m=d.get('memory', {})
b=d.get('bridge', {})
print(
    f"[RoboCOP-MULTI] run={d.get('run_id')} reason={d.get('reason')} "
    f"walk={d.get('walk_cycles',0)} falls={d.get('falls',0)} "
    f"records={m.get('records','?')} confirmed={m.get('confirmed_records','?')} "
    f"recalls={b.get('recalls','?')}"
)
PY

done

python3 - "$RESULT_DIR" "$EPISODES" > "$SUMMARY" <<'PY'
import json
import math
import statistics
import sys
from pathlib import Path

root=Path(sys.argv[1])
requested=int(sys.argv[2])
runs=[]
for p in sorted(root.glob('episode_*.json')):
    try:
        runs.append(json.loads(p.read_text(encoding='utf-8')))
    except Exception:
        pass

lengths=[int(r.get('walk_cycles',0)) for r in runs]
falls=sum(int(r.get('falls',0)) for r in runs)
reasons={}
for r in runs:
    reasons[r.get('reason','UNKNOWN')]=reasons.get(r.get('reason','UNKNOWN'),0)+1

recalls=0
completed=0
for r in runs:
    b=r.get('bridge',{})
    recalls += int(b.get('recalls',0) or 0)
    completed += int(b.get('cycles',0) or 0)

layers={'Z1':0,'Z2':0,'Z3':0}
trace=root/'combined_trace.jsonl'
if trace.exists():
    with trace.open(encoding='utf-8') as f:
        for line in f:
            try:
                row=json.loads(line)
            except Exception:
                continue
            recall=row.get('recall')
            if recall:
                layer=recall.get('layer')
                if layer in layers:
                    layers[layer]+=1

memory={}
mem=root/'resolutive_memory.json'
if mem.exists():
    try:
        payload=json.loads(mem.read_text(encoding='utf-8'))
        records=payload.get('records',[])
        conf=[int(x.get('confirmations',1)) for x in records]
        memory={
            'records':len(records),
            'confirmed':sum(x>=3 for x in conf),
            'mean_confirmations':statistics.mean(conf) if conf else 0.0,
            'max_confirmations':max(conf) if conf else 0,
            'observations_admitted':payload.get('counters',{}).get('observations_admitted',0),
            'observations_merged':payload.get('counters',{}).get('observations_merged',0),
        }
    except Exception:
        pass

def pct(n,d): return 100.0*n/d if d else 0.0

print('='*76)
print('RoboCOP — MULTI-EPISODE PERSISTENT MEMORY RESULT')
print('='*76)
print(f'Requested runs:            {requested}')
print(f'Completed summaries:       {len(runs)}')
print(f'Total walking cycles:      {sum(lengths)}')
if lengths:
    print(f'Mean episode length:       {statistics.mean(lengths):.2f}')
    print(f'Median episode length:     {statistics.median(lengths):.2f}')
    print(f'Best episode:              {max(lengths)}')
    print(f'Worst episode:             {min(lengths)}')
    print(f'>=250 cycles:              {sum(x>=250 for x in lengths)} ({pct(sum(x>=250 for x in lengths),len(lengths)):.2f}%)')
    print(f'>=500 cycles:              {sum(x>=500 for x in lengths)} ({pct(sum(x>=500 for x in lengths),len(lengths)):.2f}%)')
    print(f'>=1000 cycles:             {sum(x>=1000 for x in lengths)} ({pct(sum(x>=1000 for x in lengths),len(lengths)):.2f}%)')
    print(f'>=2000 cycles:             {sum(x>=2000 for x in lengths)} ({pct(sum(x>=2000 for x in lengths),len(lengths)):.2f}%)')
print(f'Falls:                     {falls}')
print(f'Stop reasons:              {reasons}')
print()
print('PERSISTENT MEMORY')
print(f"Records:                   {memory.get('records',0)}")
print(f"Confirmed records:         {memory.get('confirmed',0)}")
print(f"Mean confirmations:        {memory.get('mean_confirmations',0):.3f}")
print(f"Max confirmations:         {memory.get('max_confirmations',0)}")
print(f"Observations admitted:     {memory.get('observations_admitted',0)}")
print(f"Observations merged:       {memory.get('observations_merged',0)}")
print()
print('RECALL')
print(f'Recalls total:             {recalls}')
print(f'Recall rate/walk cycle:    {pct(recalls,sum(lengths)):.2f}%')
print(f"Z1 recalls:                {layers['Z1']}")
print(f"Z2 recalls:                {layers['Z2']}")
print(f"Z3 recalls:                {layers['Z3']}")
print()
print('EPISODES')
for r in runs:
    m=r.get('memory',{})
    b=r.get('bridge',{})
    print(
        f"run={int(r.get('run_id',0)):03d} walk={int(r.get('walk_cycles',0)):5d} "
        f"reason={str(r.get('reason','?')):15s} records={int(m.get('records',0)):5d} "
        f"confirmed={int(m.get('confirmed_records',0)):4d} recalls={int(b.get('recalls',0)):5d}"
    )
print('='*76)
PY

cat "$SUMMARY"

rm -f "$PACKAGE"
tar -czf "$PACKAGE" -C "$ROOT/results" bahiart_multi_episode

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
echo "RoboCOP — RESULTADO MULTI-EPISODIO PRONTO"
echo "============================================"
echo "arquivo: $PACKAGE"
echo "URL:     http://${PUBLIC_IP}:${HTTP_PORT}/robocop_multi_episode_latest.tar.gz"
echo "resumo:  http://${PUBLIC_IP}:${HTTP_PORT}/bahiart_multi_episode/summary.txt"
echo "HTTP PID: $HTTP_PID"
echo "tempo:   ${HTTP_SECONDS}s"
echo
echo "Para fechar antes:"
echo "  kill \$(cat '$ROOT/results/http_8081.pid')"
