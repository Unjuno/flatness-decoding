#!/usr/bin/env bash
set -euo pipefail

cd "/Users/taka/repos/flatness-decoding"

source .venv/bin/activate || true
export PYTHONPATH=src

echo "experiment_start=$(date)"

python scripts/run_morphology_repeat_grid.py \
  --config configs/morphology_repeat_grid.yaml \
  --execute

echo "experiment_finished=$(date)"
