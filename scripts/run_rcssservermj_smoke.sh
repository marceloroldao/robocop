#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

IMAGE="robocop-rcssservermj:smoke"
CONTAINER="robocop-rcssservermj-smoke"

cleanup() {
  docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
}
trap cleanup EXIT

cleanup

echo "========================================"
echo "RoboCOP — RCSSServerMJ smoke test"
echo "========================================"

echo "[1/4] Building MIT RCSSServerMJ runtime..."
docker build -f Dockerfile.rcssservermj -t "$IMAGE" .

echo "[2/4] Starting server on ports 60000/60001..."
docker run -d --name "$CONTAINER" \
  -p 60000:60000 \
  -p 60001:60001 \
  "$IMAGE" >/dev/null

echo "[3/4] Waiting passively for agent port 60000..."
ready=0
for i in $(seq 1 60); do
  if docker exec "$CONTAINER" python - <<'PY'
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
    echo "Server listener ready after ${i}s."
    break
  fi
  if ! docker ps --format '{{.Names}}' | grep -qx "$CONTAINER"; then
    echo "FAIL: server container exited before opening port 60000"
    docker logs "$CONTAINER" || true
    exit 3
  fi
  sleep 1
done

if [[ "$ready" -ne 1 ]]; then
  echo "FAIL: RCSSServerMJ did not open port 60000 within 60s"
  docker logs "$CONTAINER" || true
  exit 4
fi

echo "[4/4] Performing one real protocol handshake..."
python3 scripts/rcssservermj_smoke_client.py --host 127.0.0.1 --port 60000

echo
echo "Server log tail:"
docker logs --tail 30 "$CONTAINER" || true

echo
echo "PASS: RCSSServerMJ smoke validation completed."
