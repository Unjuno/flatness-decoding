#!/usr/bin/env bash
set -euo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

source .venv/bin/activate || true
export PYTHONPATH=src

echo "experiment_start=$(date)"

python scripts/run_morphology_repeat_grid.py \
  --config configs/morphology_repeat_grid.yaml \
  --execute

echo "experiment_finished=$(date)"
