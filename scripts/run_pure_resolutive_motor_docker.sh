#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MEM="$ROOT/results/active_abc_memories.pkl"
OUT="$ROOT/results/pure_resolutive_motor"
mkdir -p "$OUT"
[ -s "$MEM" ] || { echo "Missing $MEM — run scripts/run_active_abc_docker.sh preparation first."; exit 2; }
echo '============================================'
echo 'RoboCOP — PURE RESOLUTIVE MOTOR CONTROL'
echo 'BahiaRT Walk skill: DISABLED'
echo 'Resolutive memory -> 23 motor targets'
echo '============================================'
# Reuse the already validated passive image and rcssservermj image.
docker rm -f robocop-pure-server >/dev/null 2>&1 || true
docker run -d --rm --name robocop-pure-server --network host robocop-rcssservermj:walk >/dev/null
cleanup(){ docker rm -f robocop-pure-server >/dev/null 2>&1 || true; }
trap cleanup EXIT
sleep 3
for run in 1 2 3 4 5; do
  echo "--- PURE run $run/5 ---"
  docker run --rm --network host \
    -v "$ROOT:/workspace" -w /workspace \
    robocop-bahiart-passive:latest \
    python scripts/run_pure_resolutive_motor_agent.py \
      --memory /workspace/results/active_abc_memories.pkl \
      --summary "/workspace/results/pure_resolutive_motor/run_${run}.json" \
      --max-cycles 1000 --min-confidence .65 --max-step 2.0 --hold-decay .995 || true
  docker restart robocop-pure-server >/dev/null
  sleep 2
done
python3 - <<'PY'
import json,glob,statistics,pathlib
fs=sorted(glob.glob('results/pure_resolutive_motor/run_*.json'))
r=[json.load(open(x)) for x in fs]
if not r: raise SystemExit('No completed runs')
s={'runs':len(r),'mean_cycles':statistics.mean(x['cycles'] for x in r),'median_cycles':statistics.median(x['cycles'] for x in r),'best_cycles':max(x['cycles'] for x in r),'falls':sum(x['fallen'] for x in r),'mean_displacement':statistics.mean(x['displacement_xy'] for x in r),'mean_stability':statistics.mean(x['mean_stability'] for x in r),'mean_recall_rate':statistics.mean(x['recall_rate'] for x in r),'walk_skill_used':False}
p=pathlib.Path('results/pure_resolutive_motor/summary.json');p.write_text(json.dumps(s,indent=2));print(json.dumps(s,indent=2))
PY
tar -czf "$ROOT/results/robocop_pure_resolutive_motor_latest.tar.gz" -C "$ROOT/results" pure_resolutive_motor
echo "Result: $ROOT/results/robocop_pure_resolutive_motor_latest.tar.gz"
