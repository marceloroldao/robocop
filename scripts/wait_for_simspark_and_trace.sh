#!/usr/bin/env bash
set -euo pipefail

HOST="${SIMSPARK_HOST:-127.0.0.1}"
PORT="${SIMSPARK_PORT:-3100}"
TIMEOUT="${SIMSPARK_WAIT_TIMEOUT:-90}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "[RoboCOP] waiting for SimSpark at ${HOST}:${PORT} ..."

for ((i=1; i<=TIMEOUT; i++)); do
  if (echo > "/dev/tcp/${HOST}/${PORT}") >/dev/null 2>&1; then
    echo "[RoboCOP] SimSpark is ready after ${i}s."
    exec bash scripts/run_fcportugal_trace.sh
  fi
  sleep 1
done

echo "[RoboCOP] ERROR: SimSpark did not become ready at ${HOST}:${PORT} within ${TIMEOUT}s." >&2
exit 4
