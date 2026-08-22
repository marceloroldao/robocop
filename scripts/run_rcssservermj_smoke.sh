#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

IMAGE="robocop-rcssservermj:smoke"
CONTAINER="robocop-rcssservermj-smoke"

cleanup() {
  docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
}

show_logs() {
  echo
  echo "================ DOCKER LOG ================"
  docker logs --tail 240 "$CONTAINER" 2>&1 || true
  echo "============= INTERNAL DEBUG LOG ==========="
  docker exec "$CONTAINER" sh -lc '
    echo "cwd=$(readlink /proc/1/cwd 2>/dev/null || true)"
    echo "--- discovered console.log files ---"
    find / -name console.log -type f -print 2>/dev/null | while read -r f; do
      echo "--- $f ---"
      tail -n 300 "$f" || true
    done
  ' 2>&1 || true
  echo "============================================"
}

cleanup
trap cleanup EXIT

echo "========================================"
echo "RoboCOP — RCSSServerMJ smoke test"
echo "========================================"

echo "[1/5] Building MIT RCSSServerMJ runtime..."
docker build -f Dockerfile.rcssservermj -t "$IMAGE" .

echo "[2/5] Preflight: parser + T1 model + MuJoCo compile..."
# -i is mandatory here: without it Docker does not receive this heredoc on stdin.
docker run --rm -i "$IMAGE" python - <<'PY'
from rcsssmj.games.soccer.server.soccer_action_parser import SoccerActionParser
from rcsssmj.resources.spec_provider import ModelSpecProvider

raw = b"(init T1 RoboCOP 1)"
req = SoccerActionParser().parse_init(raw)
if req is None:
    raise SystemExit("FAIL: upstream parser rejected canonical init frame")
print(f"PASS: parser model={req.model_name!r} team={req.team_name!r} player={req.player_no}")

provider = ModelSpecProvider()
spec = provider.load_robot_spec("T1")
if spec is None:
    raise SystemExit("FAIL: ModelSpecProvider could not locate T1")
if spec.body("torso") is None:
    raise SystemExit("FAIL: T1 has no body named 'torso'")
if spec.material("team") is None:
    raise SystemExit("FAIL: T1 has no material named 'team'")

# Compile the standalone robot too; this catches XML/assets/schema failures.
model = spec.compile()
print("PASS: T1 model loaded and compiled")
print(f"nbody={model.nbody} njnt={model.njnt} nu={model.nu}")
PY

echo "[3/5] Starting headless server on ports 60000/60001..."
docker run -d --name "$CONTAINER" \
  -p 60000:60000 \
  -p 60001:60001 \
  "$IMAGE" >/dev/null

echo "[4/5] Waiting passively for agent port 60000..."
ready=0
for i in $(seq 1 60); do
  if docker exec -i "$CONTAINER" python - <<'PY'
port = 60000
for path in ("/proc/net/tcp", "/proc/net/tcp6"):
    try:
        with open(path, "r", encoding="ascii") as f:
            next(f, None)
            for line in f:
                cols = line.split()
                if len(cols) >= 4 and cols[3] == "0A" and int(cols[1].rsplit(":", 1)[1], 16) == port:
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
    show_logs
    exit 3
  fi
  sleep 1
done

if [[ "$ready" -ne 1 ]]; then
  echo "FAIL: RCSSServerMJ did not open port 60000 within 60s"
  show_logs
  exit 4
fi

echo "[5/5] Performing one real protocol handshake..."
set +e
python3 scripts/rcssservermj_smoke_client.py --host 127.0.0.1 --port 60000
client_rc=$?
set -e
sleep 1

if [[ "$client_rc" -ne 0 ]]; then
  echo "FAIL: protocol handshake returned exit code ${client_rc}"
  show_logs
  exit "$client_rc"
fi

show_logs

echo
echo "PASS: RCSSServerMJ smoke validation completed."
