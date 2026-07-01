from __future__ import annotations

import subprocess
import sys
from pathlib import Path

repo = Path.home() / "repos" / "flatness-decoding"
log_dir = repo / "logs"
log_dir.mkdir(exist_ok=True)

delay_hours = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0
delay_seconds = int(delay_hours * 3600)

script = repo / "scripts" / "run_morphology_repeat_grid_job.sh"
log = log_dir / "morphology_repeat_grid_job.log"

script.write_text(
    f"""#!/usr/bin/env bash
set -euo pipefail

cd "{repo}"

source .venv/bin/activate || true
export PYTHONPATH=src

echo "scheduled_delay_seconds={delay_seconds}"
echo "sleep_start=$(date)"
sleep {delay_seconds}

echo "wake_time=$(date)"
echo "experiment_start=$(date)"

python scripts/run_morphology_repeat_grid.py \\
  --conditions qwen_raw qwen_wrapped gpt2 \\
  --prompt-limit 20 \\
  --repeat-count 20 \\
  --seed-start 0 \\
  --n-values 1 4 8 16 \\
  --max-new-tokens 60 \\
  --analysis-preset morphology_20x20_top20 \\
  --execute \\
  --resume

echo "experiment_finished=$(date)"
""",
    encoding="utf-8",
)

script.chmod(0o755)

with log.open("ab") as f:
    proc = subprocess.Popen(
        ["bash", str(script)],
        stdout=f,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        cwd=repo,
    )

print("scheduled morphology repeat-grid job")
print(f"delay_hours={delay_hours}")
print(f"pid={proc.pid}")
print(f"log={log}")
print(f"script={script}")
