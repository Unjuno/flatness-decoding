#!/usr/bin/env bash
set -euo pipefail

TARGET="${1:-}"
if [[ -z "${TARGET}" ]]; then
  echo "usage: bash scripts/remote_run_make.sh <make-target> [extra make args...]" >&2
  exit 2
fi
shift || true

mkdir -p logs/remote_runs
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="logs/remote_runs/${STAMP}_${TARGET}.log"

{
  echo "started_at: $(date -Is)"
  echo "host: $(hostname || true)"
  echo "pwd: $(pwd)"
  echo "target: ${TARGET}"
  echo "git_commit: $(git rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "python: $(python --version 2>&1 || true)"
  echo "command: make ${TARGET} $*"
  echo "---"
} | tee "${LOG}"

make "${TARGET}" "$@" 2>&1 | tee -a "${LOG}"
STATUS=${PIPESTATUS[0]}

{
  echo "---"
  echo "finished_at: $(date -Is)"
  echo "status: ${STATUS}"
} | tee -a "${LOG}"

exit "${STATUS}"
