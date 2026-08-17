#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${BAHIART_DIR:-$ROOT/.external/BahiaRT-MujOCo-base}"
REPO="${BAHIART_REPO:-https://gitlab.com/bahiart/BahiaRT-MujOCo-base.git}"
REF="${BAHIART_REF:-}"

mkdir -p "$ROOT/.external"

if [ -d "$DEST/.git" ]; then
  echo "BahiaRT checkout already exists: $DEST"
  git -C "$DEST" fetch --all --tags --prune
else
  echo "Cloning BahiaRT MuJoCo base as an EXTERNAL dependency..."
  git clone "$REPO" "$DEST"
fi

if [ -n "$REF" ]; then
  git -C "$DEST" checkout "$REF"
fi

COMMIT="$(git -C "$DEST" rev-parse HEAD)"
BRANCH="$(git -C "$DEST" rev-parse --abbrev-ref HEAD 2>/dev/null || true)"

echo
echo "BahiaRT external checkout ready"
echo "Path:   $DEST"
echo "Commit: $COMMIT"
echo "Branch: $BRANCH"
echo

echo "License audit"
echo "-------------"
FOUND=0
for name in LICENSE LICENSE.md LICENSE.txt COPYING COPYING.md COPYING.txt NOTICE NOTICE.md; do
  if [ -f "$DEST/$name" ]; then
    FOUND=1
    echo "FOUND: $name"
    sed -n '1,40p' "$DEST/$name"
    echo
  fi
done

if [ "$FOUND" -eq 0 ]; then
  echo "WARNING: no top-level license file was found."
  echo "RoboCOP will treat BahiaRT as external-reference-only until licensing is explicitly verified."
fi

printf '%s\n' "$COMMIT" > "$ROOT/.external/bahiart_mujoco.commit"
