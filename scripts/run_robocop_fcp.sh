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
echo "RoboCOP + FC Portugal + SimSpark"
echo "========================================"
echo "Pulling RoboCup 3D server image..."
docker compose pull rcssserver3d

echo "Building isolated FC Portugal runtime..."
docker compose build fcp-agent

echo
echo "Starting complete stack."
echo "Trace: results/fcp_walk_trace.jsonl"
echo

docker compose up --remove-orphans
