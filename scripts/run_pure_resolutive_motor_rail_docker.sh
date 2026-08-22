#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)";TRACE="$ROOT/results/v115_multi_episode/combined_trace.jsonl";MEM="$ROOT/results/resolutive_motor_rail.pkl";OUT="$ROOT/results/pure_motor_rail";mkdir -p "$OUT"
[ -s "$TRACE" ] || { echo "missing $TRACE";exit 2; }
if [ ! -s "$MEM" ];then docker run --rm -v "$ROOT:/workspace" -w /workspace robocop-bahiart-passive:latest python scripts/prepare_resolutive_motor_rail.py --trace /workspace/results/v115_multi_episode/combined_trace.jsonl --out /workspace/results/resolutive_motor_rail.pkl;fi
echo '============================================';echo 'RoboCOP — PURE RESOLUTIVE MOTOR RAIL';echo 'BahiaRT Walk skill: DISABLED';echo '============================================'
docker rm -f robocop-rail-server >/dev/null 2>&1||true;docker run -d --rm --name robocop-rail-server --network host robocop-rcssservermj:walk >/dev/null
cleanup(){ docker rm -f robocop-rail-server >/dev/null 2>&1||true; };trap cleanup EXIT;sleep 3
for r in 1 2 3 4 5;do echo "--- RAIL run $r/5 ---";docker run --rm --network host -v "$ROOT:/workspace" -w /workspace robocop-bahiart-passive:latest python scripts/run_pure_resolutive_motor_rail_agent.py --memory /workspace/results/resolutive_motor_rail.pkl --summary "/workspace/results/pure_motor_rail/run_${r}.json" --max-cycles 1000 || true;docker restart robocop-rail-server >/dev/null;sleep 2;done
python3 - <<'PY'
import glob,json,statistics,pathlib
r=[json.load(open(f)) for f in sorted(glob.glob('results/pure_motor_rail/run_*.json'))]
s={'runs':len(r),'mean_cycles':statistics.mean(x['cycles'] for x in r),'median_cycles':statistics.median(x['cycles'] for x in r),'best_cycles':max(x['cycles'] for x in r),'falls':sum(x['fallen'] for x in r),'mean_rail_fraction':statistics.mean(x['rail_fraction'] for x in r),'mean_entries':statistics.mean(x['rail_entries'] for x in r),'mean_displacement':statistics.mean(x['displacement_xy'] for x in r)}
pathlib.Path('results/pure_motor_rail/summary.json').write_text(json.dumps(s,indent=2));print(json.dumps(s,indent=2))
PY
tar -czf "$ROOT/results/robocop_pure_motor_rail_latest.tar.gz" -C "$ROOT/results" pure_motor_rail resolutive_motor_rail.pkl;echo "Result: $ROOT/results/robocop_pure_motor_rail_latest.tar.gz"
