.PHONY: install check smoke comparison-smoke build-prompts comparison-main eval-smoke table-smoke plot-smoke test dvc-init-local

PYTHONPATH := src

install:
	python -m pip install -r requirements-common.txt

check:
	PYTHONPATH=$(PYTHONPATH) python scripts/check_device.py

smoke:
	PYTHONPATH=$(PYTHONPATH) python scripts/smoke_test.py

comparison-smoke:
	PYTHONPATH=$(PYTHONPATH) python scripts/run_pipeline.py --config configs/comparison_smoke.yaml

build-prompts:
	PYTHONPATH=$(PYTHONPATH) python scripts/build_prompt_subset.py \
		--dataset wikitext \
		--config wikitext-103-raw-v1 \
		--split validation \
		--text-field text \
		--output prompts/main_prompts.jsonl \
		--limit 100 \
		--min-words 40 \
		--prompt-words 32 \
		--prefix main

comparison-main:
	PYTHONPATH=$(PYTHONPATH) python scripts/run_pipeline.py --config configs/comparison_main_template.yaml

eval-smoke:
	PYTHONPATH=$(PYTHONPATH) python scripts/run_evaluate.py \
		--input-dir outputs/generations/comparison_smoke \
		--token-log-dir outputs/token_logs/comparison_smoke \
		--output results/summary/comparison_smoke_metrics.csv

table-smoke:
	PYTHONPATH=$(PYTHONPATH) python scripts/build_results_table.py \
		--metrics results/summary/comparison_smoke_metrics.csv \
		--output results/summary/comparison_smoke_table.csv

plot-smoke:
	PYTHONPATH=$(PYTHONPATH) python scripts/plot_results.py \
		--input results/summary/comparison_smoke_metrics.csv \
		--output-dir results/figures/comparison_smoke

test:
	PYTHONPATH=$(PYTHONPATH) pytest -q

dvc-init-local:
	dvc init
	mkdir -p ../flatness-dvc-storage
	dvc remote add -d localstore ../flatness-dvc-storage
