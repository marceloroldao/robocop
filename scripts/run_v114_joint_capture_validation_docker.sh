#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TRACE="$ROOT/results/v114_joint_capture_trace.jsonl"
SERVER="robocop-rcssservermj-v114"

echo "============================================"
echo "RoboCOP — V11.4 JOINT CAPTURE VALIDATION"
echo "============================================"

cd "$ROOT"
bash scripts/fetch_bahiart_mujoco_external.sh >/dev/null

docker build -f Dockerfile.rcssservermj -t robocop-rcssservermj:v114 . >/dev/null
docker build -f Dockerfile.bahiart-passive -t robocop-bahiart-passive:latest . >/dev/null

docker rm -f "$SERVER" >/dev/null 2>&1 || true
docker run -d --name "$SERVER" --network host robocop-rcssservermj:v114 >/dev/null
trap 'docker rm -f "$SERVER" >/dev/null 2>&1 || true' EXIT
sleep 4

rm -f "$TRACE"
set +e
docker run --rm --network host \
  -v "$ROOT:/workspace" -w /workspace \
  robocop-bahiart-passive:latest \
  python scripts/run_bahiart_walk_probe_v114.py \
    --host 127.0.0.1 --port 60000 --number 2 \
    --max-walk-cycles 350 --stop-on-fall \
    --trace /workspace/results/v114_joint_capture_trace.jsonl
RC=$?
set -e

if [[ ! -s "$TRACE" ]]; then
  echo
  echo "FAIL: V11.4 probe did not generate a non-empty trace."
  echo "Agent exit code: $RC"
  exit 1
fi

echo
echo "--- validating captured corporal channels ---"
docker run --rm -v "$ROOT:/workspace" -w /workspace \
  robocop-bahiart-passive:latest \
  python scripts/validate_v11_full_body_trace.py /workspace/results/v114_joint_capture_trace.jsonl --max-rows 350

echo
echo "Trace: $TRACE"
echo "Agent exit code: $RC"
echo "PASS: V11.4 joint capture validation completed"
