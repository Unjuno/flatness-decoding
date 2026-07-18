import subprocess
import sys
from pathlib import Path

repo = Path.home() / "repos" / "flatness-decoding"
log_dir = repo / "logs"
log_dir.mkdir(exist_ok=True)

delay_hours = float(sys.argv[1]) if len(sys.argv) > 1 else 10.0
delay_seconds = int(delay_hours * 3600)

script = repo / "scripts" / "run_gpt2_medium_micro_safe_job.sh"
log = log_dir / "gpt2_medium_micro_safe_job.log"

script.write_text(f"""#!/usr/bin/env bash
set -euo pipefail

cd "{repo}"

source .venv/bin/activate || true
export PYTHONPATH=src

echo "scheduled_delay_seconds={delay_seconds}"
echo "sleep_start=$(date)"
sleep {delay_seconds}

echo "wake_time=$(date)"
echo "checking whether comparison_gpt2_lite is still running..."

while pgrep -f "comparison_gpt2_lite" >/dev/null; do
  echo "comparison_gpt2_lite still running; waiting 30 minutes... $(date)"
  sleep 1800
done

echo "comparison_gpt2_lite is not running."
echo "experiment_start=$(date)"

cat > configs/comparison_gpt2_medium_micro.yaml <<'YAML'
experiment_name: comparison_gpt2_medium_micro
model_name: gpt2-medium
model_cache_dir: models/hf
device: auto
dtype: auto
seeds: [0]

generation:
  max_new_tokens: 60

prompts:
  path: prompts/main_prompts.jsonl
  limit: 10

baselines:
  methods: [sampling, temperature, top_p]
  temperature: 1.2
  top_p: 0.9
  top_k: 50
  num_beams: 4

n_trial_methods:
  methods: [random, minflat, maxflat]
  n_values: [1, 4, 16]
  proposal_temperature: 1.0
  use_kv_cache_lookahead: false
  candidate_eval_batch_size: 1

runtime:
  inference_mode: false
  resume:
    skip_complete: true
  cleanup:
    device_cache_after_prompt: true
    device_cache_after_group: true
    python_gc_after_group: true

output:
  generations_dir: outputs/generations/comparison_gpt2_medium_micro
  token_log_dir: outputs/token_logs/comparison_gpt2_medium_micro
  summary_csv: results/summary/comparison_gpt2_medium_micro_metrics.csv
  table_csv: results/summary/comparison_gpt2_medium_micro_table.csv
  figures_dir: results/figures/comparison_gpt2_medium_micro
YAML

START=$(date +%s)

nice -n 10 python scripts/run_experiment_parallel.py \\
  --config configs/comparison_gpt2_medium_micro.yaml \\
  --num-shards 2 \\
  --workers 1

python scripts/validate_generation_outputs.py \\
  --input-dir outputs/generations/comparison_gpt2_medium_micro \\
  --token-log-dir outputs/token_logs/comparison_gpt2_medium_micro \\
  --allow-empty-analysis-text \\
  --fail

python - <<'INNERPY'
import json
from pathlib import Path
from collections import defaultdict

rows = []
for p in Path("outputs/token_logs/comparison_gpt2_medium_micro").glob("*.tokenlog.jsonl"):
    for line in p.open():
        rows.append(json.loads(line))

acc = defaultdict(list)
for r in rows:
    key = (r["method"], int(r["n_trials"]))
    acc[key].append(float(r["chosen_flatness"]))

print("method,N,mean_chosen_flatness,count")
for (method, n), vals in sorted(acc.items(), key=lambda x: (x[0][1], x[0][0])):
    print(f"{{method}},{{n}},{{sum(vals)/len(vals):.6f}},{{len(vals)}}")
INNERPY

END=$(date +%s)
echo "experiment_finished=$(date)"
echo "elapsed_seconds=$((END - START))"
echo "elapsed_minutes=$(((END - START) / 60))"
""", encoding="utf-8")

script.chmod(0o755)

with log.open("ab") as f:
    proc = subprocess.Popen(
        ["bash", str(script)],
        stdout=f,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        cwd=repo,
    )

print("scheduled safe gpt2-medium micro job")
print(f"delay_hours={delay_hours}")
print(f"pid={proc.pid}")
print(f"log={log}")
print(f"script={script}")
