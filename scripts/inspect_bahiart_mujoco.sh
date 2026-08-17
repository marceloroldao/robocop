#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIR="${BAHIART_DIR:-$ROOT/.external/BahiaRT-MujOCo-base}"

if [ ! -d "$DIR" ]; then
  echo "BahiaRT checkout not found at $DIR" >&2
  echo "Run: bash scripts/fetch_bahiart_mujoco_external.sh" >&2
  exit 2
fi

echo "============================================================"
echo "RoboCOP — BahiaRT MuJoCo external inventory"
echo "============================================================"
echo "Commit: $(git -C "$DIR" rev-parse HEAD 2>/dev/null || echo unknown)"
echo

echo "Top-level files/directories"
find "$DIR" -maxdepth 1 -mindepth 1 -printf '%f\n' | sort | sed -n '1,120p'

echo
echo "Potential Python entry points"
find "$DIR" -maxdepth 3 -type f \( -name '*.py' -o -name '*.sh' \) \
  | sed "s#^$DIR/##" \
  | grep -Ei '(^|/)(main|run|start|agent|player|team|train|test|demo|client|server)[^/]*\.(py|sh)$' \
  | sort | sed -n '1,160p' || true

echo
echo "Dependency manifests"
find "$DIR" -maxdepth 3 -type f \( \
  -name 'requirements*.txt' -o \
  -name 'pyproject.toml' -o \
  -name 'setup.py' -o \
  -name 'setup.cfg' -o \
  -name 'environment.yml' -o \
  -name 'Dockerfile*' -o \
  -name 'docker-compose*.yml' -o \
  -name 'docker-compose*.yaml' \
\) -printf '%P\n' | sort

echo
echo "MuJoCo/control-related symbols"
grep -RIlE 'mujoco|mjData|mjModel|data\.ctrl|actuator|observation|action|policy|joint|imu|sensor' "$DIR" \
  --include='*.py' --include='*.xml' --include='*.yml' --include='*.yaml' \
  2>/dev/null | sed "s#^$DIR/##" | sort | sed -n '1,160p' || true

echo
echo "License candidates"
find "$DIR" -maxdepth 2 -type f \( -iname 'LICENSE*' -o -iname 'COPYING*' -o -iname 'NOTICE*' \) -printf '%P\n' | sort || true

echo
echo "============================================================"
echo "Inventory complete. No BahiaRT source was copied into RoboCOP."
echo "============================================================"
