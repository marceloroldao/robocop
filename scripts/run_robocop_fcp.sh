#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if ! command -v docker >/dev/null 2>&1; then
  echo "Docker not found." >&2
  exit 2
fi

if ! docker compose version >/dev/null 2>&1; then
  echo "Docker Compose plugin not found." >&2
  exit 3
fi

mkdir -p results

echo "========================================"
echo "RoboCOP + FC Portugal (Docker)"
echo "========================================"
echo "Building isolated runtime..."
docker compose build fcp-agent

echo

echo "Starting FC Portugal agent with RoboCOP tracing."
echo "A SimSpark/rcssserver3d server must be reachable on the host network."
echo "Trace: results/fcp_walk_trace.jsonl"
echo

docker compose run --rm fcp-agent "$@"
