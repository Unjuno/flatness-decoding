#!/usr/bin/env bash
set -euo pipefail

cd "/Users/taka/repos/flatness-decoding"

source .venv/bin/activate || true
export PYTHONPATH=src

echo "experiment_start=$(date)"

python scripts/run_morphology_repeat_grid.py \
  --conditions qwen_raw qwen_wrapped gpt2 \
  --prompt-limit 20 \
  --repeat-count 20 \
  --seed-start 0 \
  --n-values 1 4 8 16 \
  --max-new-tokens 60 \
  --analysis-preset morphology_20x20_top20 \
  --execute \
  --resume

echo "experiment_finished=$(date)"
