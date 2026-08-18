#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
SERVER_CONTAINER="robocop-rcssservermj-v113-audit"
cleanup(){ docker rm -f "$SERVER_CONTAINER" >/dev/null 2>&1 || true; }
trap cleanup EXIT INT TERM
bash scripts/fetch_bahiart_mujoco_external.sh
docker build -f Dockerfile.rcssservermj -t robocop-rcssservermj:walk . >/dev/null
docker build -f Dockerfile.bahiart-passive -t robocop-bahiart-passive:latest . >/dev/null
cleanup
docker run -d --name "$SERVER_CONTAINER" --network host robocop-rcssservermj:walk >/dev/null
sleep 4
echo "============================================"
echo "RoboCOP — V11.3 JOINT SENSOR API AUDIT"
echo "============================================"
docker run --rm --network host -v "$ROOT:/workspace" -w /workspace robocop-bahiart-passive:latest \
  python scripts/probe_bahiart_joint_sensor_api.py --host 127.0.0.1 --port 60000 --number 2
