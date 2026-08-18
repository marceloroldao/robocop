#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

TRACE="$ROOT/results/bahiart_walk_probe_trace.jsonl"
SERVER_CONTAINER="robocop-rcssservermj-walk"
MAX_WALK_CYCLES="${ROBOCOP_MAX_WALK_CYCLES:-12000}"
MAX_RECOVERY_CYCLES="${ROBOCOP_MAX_RECOVERY_CYCLES:-2500}"
BLOCK="${ROBOCOP_WALK_BLOCK:-150}"

cleanup() {
  docker rm -f "$SERVER_CONTAINER" >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

mkdir -p results
bash scripts/fetch_bahiart_mujoco_external.sh

echo "============================================"
echo "RoboCOP — episodic BahiaRT Walk probe"
echo "============================================"
echo "walk cycles:     $MAX_WALK_CYCLES"
echo "recovery limit:  $MAX_RECOVERY_CYCLES"
echo "command block:   $BLOCK"
echo "trace:           $TRACE"

docker build -f Dockerfile.rcssservermj -t robocop-rcssservermj:walk .
docker build -f Dockerfile.bahiart-passive -t robocop-bahiart-passive:latest .

cleanup
docker run -d --name "$SERVER_CONTAINER" --network host robocop-rcssservermj:walk >/dev/null
sleep 4

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
    --trace /workspace/results/bahiart_walk_probe_trace.jsonl
