#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SERVER_IMAGE="robocop-rcssservermj:passive"
AGENT_IMAGE="robocop-bahiart-passive:latest"
SERVER_CONTAINER="robocop-rcssservermj-passive"
TRACE="${ROBOCOP_TRACE:-$ROOT/results/bahiart_passive_trace.jsonl}"

cleanup() {
  docker rm -f "$SERVER_CONTAINER" >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

if [[ ! -d "$ROOT/.external/BahiaRT-MujOCo-base" ]]; then
  echo "[RoboCOP] fetching BahiaRT as external reference..."
  bash scripts/fetch_bahiart_mujoco_external.sh
fi

mkdir -p "$ROOT/results"
rm -f "$TRACE"

echo "========================================"
echo "RoboCOP — BahiaRT passive learning test"
echo "========================================"

echo "[1/5] Building RCSSServerMJ runtime..."
docker build -f Dockerfile.rcssservermj -t "$SERVER_IMAGE" .

echo "[2/5] Starting RCSSServerMJ..."
cleanup
docker run -d --name "$SERVER_CONTAINER" \
  --network host \
  "$SERVER_IMAGE" >/dev/null

echo "[3/5] Waiting for port 60000..."
ready=0
for i in $(seq 1 60); do
  if docker exec "$SERVER_CONTAINER" python - <<'PY'
import sys
port = 60000
for path in ("/proc/net/tcp", "/proc/net/tcp6"):
    try:
        with open(path, "r", encoding="ascii") as f:
            next(f, None)
            for line in f:
                cols = line.split()
                if len(cols) >= 4 and cols[3] == "0A":
                    if int(cols[1].rsplit(":", 1)[1], 16) == port:
                        raise SystemExit(0)
    except OSError:
        pass
raise SystemExit(1)
PY
  then
    ready=1
    echo "[RoboCOP] server ready after ${i}s"
    break
  fi
  sleep 1
done

if [[ "$ready" -ne 1 ]]; then
  echo "[RoboCOP] FAIL: server did not open port 60000"
  docker logs "$SERVER_CONTAINER" || true
  exit 4
fi

echo "[4/5] Building Python 3.13 BahiaRT passive runtime..."
docker build -f Dockerfile.bahiart-passive -t "$AGENT_IMAGE" .

echo "[5/5] Running BahiaRT unchanged + passive resolutive recorder..."
echo "[RoboCOP] trace: $TRACE"
echo "[RoboCOP] press Ctrl+C to stop after enough cycles"

set +e
docker run --rm \
  --network host \
  -v "$ROOT:/workspace" \
  -w /workspace \
  "$AGENT_IMAGE" \
  python scripts/run_bahiart_passive.py \
    --host 127.0.0.1 \
    --port 60000 \
    --trace /workspace/results/$(basename "$TRACE")
agent_rc=$?
set -e

echo
echo "================ SERVER LOG ================"
docker logs --tail 120 "$SERVER_CONTAINER" 2>&1 || true

echo "================ TRACE SUMMARY ============="
if [[ -s "$TRACE" ]]; then
  python3 - "$TRACE" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
rows = []
with path.open("r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            rows.append(json.loads(line))

print(f"cycles={len(rows)}")
if rows:
    last = rows[-1]
    probe = last.get("probe", {})
    mem = last.get("memory", {})
    print(f"completed_transitions={probe.get('completed_transitions', 0)}")
    print(f"admitted_transitions={probe.get('admitted_transitions', 0)}")
    print(f"recalls={probe.get('recalls', 0)}")
    print(f"memory={mem}")
    print(f"last_fallen={last.get('fallen')}")
PY
else
  echo "trace is empty"
fi

echo "============================================"
exit "$agent_rc"
