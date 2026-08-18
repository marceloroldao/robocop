#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
TRACE="$ROOT/results/bahiart_walk_probe_trace.jsonl"
SERVER_CONTAINER="robocop-rcssservermj-walk"
trap 'docker rm -f "$SERVER_CONTAINER" >/dev/null 2>&1 || true' EXIT
mkdir -p results
bash scripts/fetch_bahiart_mujoco_external.sh
docker build -f Dockerfile.rcssservermj -t robocop-rcssservermj:walk .
docker build -f Dockerfile.bahiart-passive -t robocop-bahiart-passive:latest .
docker rm -f "$SERVER_CONTAINER" >/dev/null 2>&1 || true
docker run -d --name "$SERVER_CONTAINER" --network host robocop-rcssservermj:walk >/dev/null
sleep 4
docker run --rm --network host -v "$ROOT:/workspace" -w /workspace robocop-bahiart-passive:latest python scripts/run_bahiart_walk_probe.py --host 127.0.0.1 --port 60000 --number 2 --max-cycles "${ROBOCOP_MAX_CYCLES:-12000}" --trace /workspace/results/bahiart_walk_probe_trace.jsonl
