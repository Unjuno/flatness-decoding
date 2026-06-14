.PHONY: install install-eval download-models check smoke plan-parallel estimate-tiny estimate-smoke estimate-distilgpt2-main estimate-main clean-tiny clean-smoke clean-distilgpt2-main clean-main comparison-tiny comparison-tiny-parallel validate-tiny comparison-smoke comparison-smoke-parallel validate-smoke build-prompts comparison-main comparison-main-parallel validate-main comparison-distilgpt2-main comparison-distilgpt2-main-parallel validate-distilgpt2-main eval-smoke table-smoke extended-smoke ppl-smoke trends-smoke metric-suite-smoke merge-smoke manifest-smoke analyze-smoke token-analyze-smoke plot-smoke paper-check paper-check-final export-smoke-table test dvc-init-local

PYTHONPATH := src

install:
	python -m pip install -r requirements-common.txt

install-eval:
	python -m pip install -r requirements-eval.txt

download-models:
	PYTHONPATH=$(PYTHONPATH) python scripts/download_models.py --manifest configs/model_manifest.yaml --cache-dir models/hf

check:
	PYTHONPATH=$(PYTHONPATH) python scripts/check_device.py

smoke:
	PYTHONPATH=$(PYTHONPATH) python scripts/smoke_test.py

plan-parallel:
	PYTHONPATH=$(PYTHONPATH) python scripts/plan_parallelism.py

estimate-tiny:
	PYTHONPATH=$(PYTHONPATH) python scripts/estimate_experiment_cost.py --config configs/comparison_tiny.yaml

estimate-smoke:
	PYTHONPATH=$(PYTHONPATH) python scripts/estimate_experiment_cost.py --config configs/comparison_smoke.yaml

estimate-distilgpt2-main:
	PYTHONPATH=$(PYTHONPATH) python scripts/estimate_experiment_cost.py --config configs/comparison_distilgpt2_main.yaml

estimate-main:
	PYTHONPATH=$(PYTHONPATH) python scripts/estimate_experiment_cost.py --config configs/comparison_main_template.yaml

clean-tiny:
	PYTHONPATH=$(PYTHONPATH) python scripts/clean_experiment_outputs.py --experiment comparison_tiny --yes

clean-smoke:
	PYTHONPATH=$(PYTHONPATH) python scripts/clean_experiment_outputs.py --experiment comparison_smoke --yes

clean-distilgpt2-main:
	PYTHONPATH=$(PYTHONPATH) python scripts/clean_experiment_outputs.py --experiment comparison_distilgpt2_main --yes

clean-main:
	PYTHONPATH=$(PYTHONPATH) python scripts/clean_experiment_outputs.py --experiment comparison_main --yes

comparison-tiny:
	PYTHONPATH=$(PYTHONPATH) python scripts/run_pipeline.py --config configs/comparison_tiny.yaml

comparison-tiny-parallel:
	PYTHONPATH=$(PYTHONPATH) python scripts/run_experiment_parallel.py --config configs/comparison_tiny.yaml --num-shards auto --workers auto
	PYTHONPATH=$(PYTHONPATH) python scripts/run_evaluate.py \
		--input-dir outputs/generations/comparison_tiny \
		--token-log-dir outputs/token_logs/comparison_tiny \
		--output results/summary/comparison_tiny_metrics.csv
	PYTHONPATH=$(PYTHONPATH) python scripts/build_results_table.py \
		--metrics results/summary/comparison_tiny_metrics.csv \
		--output results/summary/comparison_tiny_table.csv

validate-tiny:
	PYTHONPATH=$(PYTHONPATH) python scripts/validate_generation_outputs.py \
		--input-dir outputs/generations/comparison_tiny \
		--token-log-dir outputs/token_logs/comparison_tiny \
		--require-methods greedy beam sampling top_p random minflat maxflat \
		--fail

comparison-smoke:
	PYTHONPATH=$(PYTHONPATH) python scripts/run_pipeline.py --config configs/comparison_smoke.yaml

comparison-smoke-parallel:
	PYTHONPATH=$(PYTHONPATH) python scripts/run_experiment_parallel.py --config configs/comparison_smoke.yaml --num-shards auto --workers auto
	$(MAKE) validate-smoke eval-smoke table-smoke analyze-smoke token-analyze-smoke plot-smoke

validate-smoke:
	PYTHONPATH=$(PYTHONPATH) python scripts/validate_generation_outputs.py \
		--input-dir outputs/generations/comparison_smoke \
		--token-log-dir outputs/token_logs/comparison_smoke \
		--fail

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

comparison-main-parallel:
	PYTHONPATH=$(PYTHONPATH) python scripts/run_experiment_parallel.py --config configs/comparison_main_template.yaml --num-shards auto --workers auto
	$(MAKE) validate-main
	PYTHONPATH=$(PYTHONPATH) python scripts/run_evaluate.py \
		--input-dir outputs/generations/comparison_main \
		--token-log-dir outputs/token_logs/comparison_main \
		--output results/summary/comparison_main_metrics.csv
	PYTHONPATH=$(PYTHONPATH) python scripts/build_results_table.py \
		--metrics results/summary/comparison_main_metrics.csv \
		--output results/summary/comparison_main_table.csv
	PYTHONPATH=$(PYTHONPATH) python scripts/analyze_results.py \
		--table results/summary/comparison_main_table.csv \
		--output results/summary/comparison_main_analysis.md
	PYTHONPATH=$(PYTHONPATH) python scripts/plot_results.py \
		--input results/summary/comparison_main_metrics.csv \
		--output-dir results/figures/comparison_main

validate-main:
	PYTHONPATH=$(PYTHONPATH) python scripts/validate_generation_outputs.py \
		--input-dir outputs/generations/comparison_main \
		--token-log-dir outputs/token_logs/comparison_main \
		--fail

comparison-distilgpt2-main:
	PYTHONPATH=$(PYTHONPATH) python scripts/run_pipeline.py --config configs/comparison_distilgpt2_main.yaml

comparison-distilgpt2-main-parallel:
	PYTHONPATH=$(PYTHONPATH) python scripts/run_experiment_parallel.py --config configs/comparison_distilgpt2_main.yaml --num-shards auto --workers auto
	$(MAKE) validate-distilgpt2-main
	PYTHONPATH=$(PYTHONPATH) python scripts/run_evaluate.py \
		--input-dir outputs/generations/comparison_distilgpt2_main \
		--token-log-dir outputs/token_logs/comparison_distilgpt2_main \
		--output results/summary/comparison_distilgpt2_main_metrics.csv
	PYTHONPATH=$(PYTHONPATH) python scripts/build_results_table.py \
		--metrics results/summary/comparison_distilgpt2_main_metrics.csv \
		--output results/summary/comparison_distilgpt2_main_table.csv
	PYTHONPATH=$(PYTHONPATH) python scripts/analyze_results.py \
		--table results/summary/comparison_distilgpt2_main_table.csv \
		--output results/summary/comparison_distilgpt2_main_analysis.md
	PYTHONPATH=$(PYTHONPATH) python scripts/plot_results.py \
		--input results/summary/comparison_distilgpt2_main_metrics.csv \
		--output-dir results/figures/comparison_distilgpt2_main

validate-distilgpt2-main:
	PYTHONPATH=$(PYTHONPATH) python scripts/validate_generation_outputs.py \
		--input-dir outputs/generations/comparison_distilgpt2_main \
		--token-log-dir outputs/token_logs/comparison_distilgpt2_main \
		--fail

eval-smoke:
	PYTHONPATH=$(PYTHONPATH) python scripts/run_evaluate.py \
		--input-dir outputs/generations/comparison_smoke \
		--token-log-dir outputs/token_logs/comparison_smoke \
		--output results/summary/comparison_smoke_metrics.csv

table-smoke:
	PYTHONPATH=$(PYTHONPATH) python scripts/build_results_table.py \
		--metrics results/summary/comparison_smoke_metrics.csv \
		--output results/summary/comparison_smoke_table.csv

extended-smoke:
	PYTHONPATH=$(PYTHONPATH) python scripts/evaluate_extended_metrics.py \
		--input-dir outputs/generations/comparison_smoke \
		--output results/summary/comparison_smoke_extended_metrics.csv

ppl-smoke:
	PYTHONPATH=$(PYTHONPATH) python scripts/evaluate_ppl.py \
		--input-dir outputs/generations/comparison_smoke \
		--model distilgpt2 \
		--output results/summary/comparison_smoke_ppl.csv

trends-smoke:
	PYTHONPATH=$(PYTHONPATH) python scripts/fit_n_trends.py \
		--table results/summary/comparison_smoke_table.csv \
		--output results/summary/comparison_smoke_n_trend_models.csv \
		--figure-dir results/figures/comparison_smoke/n_trends

metric-suite-smoke:
	PYTHONPATH=$(PYTHONPATH) python scripts/plot_metric_suite.py \
		--tables results/summary/comparison_smoke_table.csv results/summary/comparison_smoke_extended_metrics.csv results/summary/comparison_smoke_ppl.csv \
		--output-dir results/figures/comparison_smoke/metric_suite

merge-smoke:
	PYTHONPATH=$(PYTHONPATH) python scripts/merge_metric_tables.py \
		--base results/summary/comparison_smoke_table.csv \
		--extras results/summary/comparison_smoke_extended_metrics.csv results/summary/comparison_smoke_ppl.csv \
		--output results/summary/comparison_smoke_merged_table.csv

manifest-smoke:
	PYTHONPATH=$(PYTHONPATH) python scripts/write_run_manifest.py \
		--config configs/comparison_smoke.yaml \
		--output results/summary/comparison_smoke_manifest.json

analyze-smoke:
	PYTHONPATH=$(PYTHONPATH) python scripts/analyze_results.py \
		--table results/summary/comparison_smoke_table.csv \
		--output results/summary/comparison_smoke_analysis.md

token-analyze-smoke:
	PYTHONPATH=$(PYTHONPATH) python scripts/analyze_token_logs.py \
		--token-log-dir outputs/token_logs/comparison_smoke \
		--output-dir results/figures/comparison_smoke/token_analysis \
		--summary-output results/summary/comparison_smoke_token_trace.csv

plot-smoke:
	PYTHONPATH=$(PYTHONPATH) python scripts/plot_results.py \
		--input results/summary/comparison_smoke_metrics.csv \
		--output-dir results/figures/comparison_smoke

paper-check:
	PYTHONPATH=$(PYTHONPATH) python scripts/check_paper_ready.py --manuscript paper/manuscript.md

paper-check-final:
	PYTHONPATH=$(PYTHONPATH) python scripts/check_paper_ready.py --manuscript paper/manuscript.md --fail

export-smoke-table:
	PYTHONPATH=$(PYTHONPATH) python scripts/export_table_markdown.py \
		--input results/summary/comparison_smoke_table.csv \
		--output paper/tables/comparison_smoke_table.md \
		--sort-by method n_trials

test:
	PYTHONPATH=$(PYTHONPATH) pytest -q

dvc-init-local:
	dvc init
	mkdir -p ../flatness-dvc-storage
	dvc remote add -d localstore ../flatness-dvc-storage
