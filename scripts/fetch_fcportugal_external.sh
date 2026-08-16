#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXTERNAL_DIR="${ROOT}/.external"
FCP_DIR="${EXTERNAL_DIR}/FCPCodebase"

mkdir -p "${EXTERNAL_DIR}"

if [ -d "${FCP_DIR}/.git" ]; then
  echo "FC Portugal already present at ${FCP_DIR}"
  git -C "${FCP_DIR}" fetch --all --tags
  git -C "${FCP_DIR}" checkout main
  git -C "${FCP_DIR}" pull --ff-only
else
  git clone https://github.com/m-abr/FCPCodebase.git "${FCP_DIR}"
fi

printf '\nFC Portugal external checkout ready:\n  %s\n' "${FCP_DIR}"
printf 'License: GPL-3.0 (external dependency; not part of RoboCOP core)\n'
printf 'Upstream commit: '
git -C "${FCP_DIR}" rev-parse HEAD
