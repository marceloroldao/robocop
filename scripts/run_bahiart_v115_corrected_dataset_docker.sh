#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
EPISODES="${ROBOCOP_V115_EPISODES:-20}"
MAX_WALK="${ROBOCOP_V115_MAX_WALK:-2000}"
BLOCK="${ROBOCOP_WALK_BLOCK:-150}"
RESULT_DIR="$ROOT/results/v115_multi_episode"
COMBINED="$RESULT_DIR/combined_trace.jsonl"
SUMMARY="$RESULT_DIR/summary.txt"
PACKAGE="$ROOT/results/robocop_v115_dataset_latest.tar.gz"
SERVER="robocop-rcssservermj-v115"
cleanup(){ docker rm -f "$SERVER" >/dev/null 2>&1 || true; }
trap cleanup EXIT INT TERM
mkdir -p "$RESULT_DIR"
rm -f "$RESULT_DIR"/episode_*.jsonl "$RESULT_DIR"/episode_*.json "$RESULT_DIR"/server_*.log "$COMBINED" "$SUMMARY" "$PACKAGE"
: > "$COMBINED"
bash scripts/fetch_bahiart_mujoco_external.sh >/dev/null
docker build -f Dockerfile.rcssservermj -t robocop-rcssservermj:v115 . >/dev/null
docker build -f Dockerfile.bahiart-passive -t robocop-bahiart-passive:latest . >/dev/null

echo "============================================"
echo "RoboCOP — V11.5 CORRECTED FULL-BODY DATASET"
echo "============================================"
echo "episodes:       $EPISODES"
echo "max walk/run:   $MAX_WALK"
echo "dataset:        $RESULT_DIR"

for ((i=1;i<=EPISODES;i++)); do
  printf -v PAD "%03d" "$i"
  TRACE="$RESULT_DIR/episode_${PAD}.jsonl"; JS="$RESULT_DIR/episode_${PAD}.json"; LOG="$RESULT_DIR/server_${PAD}.log"
  echo; echo "--- V11.5 run $i/$EPISODES ---"
  cleanup; docker run -d --name "$SERVER" --network host robocop-rcssservermj:v115 >/dev/null; sleep 3
  set +e
  docker run --rm --network host -v "$ROOT:/workspace" -w /workspace robocop-bahiart-passive:latest \
    python scripts/run_bahiart_walk_probe_v114.py \
      --host 127.0.0.1 --port 60000 --number 2 --run-id "$i" --stop-on-fall --block "$BLOCK" \
      --max-walk-cycles "$MAX_WALK" --trace "/workspace/results/v115_multi_episode/episode_${PAD}.jsonl" \
      --summary "/workspace/results/v115_multi_episode/episode_${PAD}.json"
  RC=$?; set -e
  docker logs "$SERVER" > "$LOG" 2>&1 || true; cleanup
  if [[ ! -s "$TRACE" ]]; then echo "FAIL: run $i produced no trace (rc=$RC)"; exit 2; fi
  if [[ "$i" -eq 1 ]]; then
    echo "--- validating corporal capture on first episode ---"
    docker run --rm -v "$ROOT:/workspace" -w /workspace robocop-bahiart-passive:latest \
      python scripts/validate_v11_full_body_trace.py "/workspace/results/v115_multi_episode/episode_${PAD}.jsonl" --max-rows 1000
  fi
  cat "$TRACE" >> "$COMBINED"
  python3 - "$JS" <<'PY'
import json,sys
r=json.load(open(sys.argv[1],encoding='utf-8'))
print(f"[V11.5] run={r.get('run_id')} reason={r.get('reason')} walk={r.get('walk_cycles',0)} falls={r.get('falls',0)}")
PY
done

echo; echo "--- validating final combined dataset ---"
docker run --rm -v "$ROOT:/workspace" -w /workspace robocop-bahiart-passive:latest \
  python scripts/validate_v11_full_body_trace.py /workspace/results/v115_multi_episode/combined_trace.jsonl --max-rows 3000

python3 - "$RESULT_DIR" > "$SUMMARY" <<'PY'
import json,statistics,sys
from pathlib import Path
root=Path(sys.argv[1]); runs=[]
for p in sorted(root.glob('episode_*.json')):
    try:runs.append(json.loads(p.read_text()))
    except:pass
v=[int(r.get('walk_cycles',0)) for r in runs]
print('='*76);print('RoboCOP — V11.5 CORRECTED DATASET SUMMARY');print('='*76)
print(f'Runs: {len(v)}');print(f'Total walking cycles: {sum(v)}')
if v:
 print(f'Mean episode: {statistics.mean(v):.2f}');print(f'Median episode: {statistics.median(v):.2f}');print(f'Best episode: {max(v)}');print(f'Worst episode: {min(v)}');print(f'>=250 cycles: {sum(x>=250 for x in v)}');print(f'>=500 cycles: {sum(x>=500 for x in v)}');print(f'>=1000 cycles: {sum(x>=1000 for x in v)}')
print('='*76)
PY
cat "$SUMMARY"
tar -czf "$PACKAGE" -C "$ROOT/results" v115_multi_episode
echo;echo "Dataset ready: $PACKAGE";echo "Next: run the S-only vs B-only vs SxB ablation."