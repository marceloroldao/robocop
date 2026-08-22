#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRACE="$ROOT/results/v115_multi_episode/combined_trace.jsonl"
MEM="$ROOT/results/active_abc_memories.pkl"
OUTDIR="$ROOT/results/active_abc"
RUNS="${ROBOCOP_ABC_RUNS:-10}"
MAXC="${ROBOCOP_ABC_MAX_CYCLES:-2000}"
ALPHA="${ROBOCOP_ABC_ALPHA:-0.15}"
MAXDELTA="${ROBOCOP_ABC_MAX_DELTA:-5.0}"
SERVER="robocop-rcssservermj-abc"
mkdir -p "$OUTDIR"
[ -s "$TRACE" ] || { echo "missing trace: $TRACE"; exit 2; }
bash "$ROOT/scripts/fetch_bahiart_mujoco_external.sh"
docker build -f "$ROOT/Dockerfile.rcssservermj" -t robocop-rcssservermj:walk "$ROOT"
docker build -f "$ROOT/Dockerfile.bahiart-passive" -t robocop-bahiart-passive:latest "$ROOT"
if [ ! -s "$MEM" ]; then
 echo 'Preparing frozen memories (V11.7 stage may take a while)...'
 docker run --rm -v "$ROOT:/workspace" -w /workspace robocop-bahiart-passive:latest python scripts/prepare_active_abc_memories.py --trace /workspace/results/v115_multi_episode/combined_trace.jsonl --out /workspace/results/active_abc_memories.pkl
fi
cleanup(){ docker rm -f "$SERVER" >/dev/null 2>&1 || true; }
trap cleanup EXIT INT TERM
for VAR in baseline original v117; do
 for I in $(seq 1 "$RUNS"); do
  cleanup
  docker run -d --name "$SERVER" --network host robocop-rcssservermj:walk >/dev/null
  sleep 3
  echo "=== ACTIVE ABC variant=$VAR run=$I/$RUNS ==="
  set +e
  docker run --rm --network host -v "$ROOT:/workspace" -w /workspace robocop-bahiart-passive:latest python scripts/run_active_abc_agent.py --variant "$VAR" --memory /workspace/results/active_abc_memories.pkl --summary "/workspace/results/active_abc/${VAR}_${I}.json" --max-cycles "$MAXC" --alpha "$ALPHA" --max-delta "$MAXDELTA"
  set -e
 done
done
python3 - "$OUTDIR" "$RUNS" <<'PY'
import json,sys,statistics,pathlib
p=pathlib.Path(sys.argv[1]);runs=int(sys.argv[2]);out=[]
print('\n'+'='*92);print('RoboCOP — ACTIVE A/B/C SUMMARY');print('='*92)
print(f"{'variant':10s} {'n':>3s} {'mean cyc':>10s} {'median':>8s} {'best':>7s} {'falls':>7s} {'disp':>9s} {'stab':>9s} {'recall':>9s}")
for v in ('baseline','original','v117'):
 xs=[]
 for i in range(1,runs+1):
  f=p/f'{v}_{i}.json'
  if f.exists():xs.append(json.loads(f.read_text()))
 if not xs:continue
 cyc=[x['cycles'] for x in xs];disp=[x['displacement_xy'] for x in xs];stab=[x['mean_stability'] for x in xs];rr=[x['recall_rate'] for x in xs]
 row={'variant':v,'n':len(xs),'mean_cycles':statistics.mean(cyc),'median_cycles':statistics.median(cyc),'best_cycles':max(cyc),'falls':sum(x['fallen'] for x in xs),'mean_displacement':statistics.mean(disp),'mean_stability':statistics.mean(stab),'mean_recall_rate':statistics.mean(rr)};out.append(row)
 print(f"{v:10s} {len(xs):3d} {row['mean_cycles']:10.1f} {row['median_cycles']:8.1f} {row['best_cycles']:7d} {row['falls']:7d} {row['mean_displacement']:9.3f} {row['mean_stability']:9.4f} {100*row['mean_recall_rate']:8.2f}%")
(p/'summary.json').write_text(json.dumps(out,indent=2));
report=p/'summary.txt';report.write_text('\n'.join(json.dumps(x) for x in out)+'\n')
print('='*92);print('summary:',p/'summary.json')
PY
PKG="$ROOT/results/robocop_active_abc_latest.tar.gz"
rm -f "$PKG";tar -czf "$PKG" -C "$ROOT/results" active_abc
echo "Package: $PKG"
