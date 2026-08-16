#!/usr/bin/env bash
set -euo pipefail

HOST="${SIMSPARK_HOST:-127.0.0.1}"
PORT="${SIMSPARK_PORT:-3100}"
TIMEOUT="${SIMSPARK_WAIT_TIMEOUT:-90}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "[RoboCOP] waiting for SimSpark at ${HOST}:${PORT} ..."
echo "[RoboCOP] using passive listener detection (no probe connection to agent port)."

# IMPORTANT: do not use /dev/tcp, nc -z, curl, or another active TCP probe here.
# rcssserver3d treats every connection to port 3100 as a real agent session. A
# connect-and-close readiness probe therefore creates a malformed/aborted agent
# and can interfere with the following FC Portugal handshake.
for ((i=1; i<=TIMEOUT; i++)); do
  if python - "$HOST" "$PORT" <<'PY'
import socket
import sys

host = sys.argv[1]
port = int(sys.argv[2])

# Passive check via /proc/net/tcp{,6}: LISTEN == 0A. This opens no socket.
def listening(path):
    try:
        with open(path, "r", encoding="ascii") as f:
            next(f, None)
            for line in f:
                cols = line.split()
                if len(cols) < 4 or cols[3] != "0A":
                    continue
                try:
                    local_port = int(cols[1].rsplit(":", 1)[1], 16)
                except (ValueError, IndexError):
                    continue
                if local_port == port:
                    return True
    except OSError:
        pass
    return False

sys.exit(0 if listening("/proc/net/tcp") or listening("/proc/net/tcp6") else 1)
PY
  then
    echo "[RoboCOP] SimSpark listener is ready after ${i}s."
    # Give the simulator one extra moment to finish initialization after bind.
    sleep 2
    exec bash scripts/run_fcportugal_trace.sh
  fi
  sleep 1
done

echo "[RoboCOP] ERROR: SimSpark did not listen at ${HOST}:${PORT} within ${TIMEOUT}s." >&2
exit 4
