.PHONY: install install-eval download-models check smoke plan-parallel estimate-tiny estimate-smoke estimate-distilgpt2-main estimate-main clean-tiny clean-smoke clean-distilgpt2-main clean-main comparison-tiny comparison-tiny-parallel validate-tiny comparison-smoke comparison-smoke-parallel validate-smoke build-prompts build-writingprompts build-wikitext-pilot comparison-main comparison-main-parallel validate-main comparison-distilgpt2-main comparison-distilgpt2-main-parallel validate-distilgpt2-main eval-smoke table-smoke extended-smoke ppl-smoke trends-smoke metric-suite-smoke merge-smoke manifest-smoke analyze-smoke token-analyze-smoke plot-smoke postprocess-smoke extended-distilgpt2-main ppl-distilgpt2-main trends-distilgpt2-main metric-suite-distilgpt2-main merge-distilgpt2-main manifest-distilgpt2-main token-analyze-distilgpt2-main postprocess-distilgpt2-main extended-main ppl-main trends-main metric-suite-main merge-main manifest-main token-analyze-main postprocess-main postprocess-gpt2-lite postprocess-gpt2-medium-micro postprocess-qwen25-05b-micro paper-check paper-check-final export-smoke-table test dvc-init-local

PYTHONPATH := src
VENV_PYTHON := .venv/bin/python

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
	PYTHONPATH=$(PYTHONPATH) python scripts/clean_experiment_outputs.py --experiment comparison_distilgpt2_writingprompts --yes

clean-main:
	PYTHONPATH=$(PYTHONPATH) python scripts/clean_experiment_outputs.py --experiment comparison_gpt2_writingprompts --yes

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
		--require-methods greedy beam sampling temperature top_p top_k random minflat maxflat \
		--require-token-log-methods random minflat maxflat \
		--fail

build-prompts: build-writingprompts

build-writingprompts:
	PYTHONPATH=$(PYTHONPATH) python scripts/build_prompt_subset.py \
		--dataset euclaise/writingprompts \
		--split validation \
		--prompt-field prompt \
		--reference-field story \
		--output prompts/writingprompts_fixed_100.jsonl \
		--limit 100 \
		--min-prompt-words 3 \
		--prefix wp \
		--shuffle \
		--seed 0

build-wikitext-pilot:
	PYTHONPATH=$(PYTHONPATH) python scripts/build_prompt_subset.py \
		--dataset wikitext \
		--config wikitext-103-raw-v1 \
		--split validation \
		--text-field text \
		--output prompts/wikitext_pilot_prompts.jsonl \
		--limit 100 \
		--min-words 40 \
		--prompt-words 32 \
		--prefix wtpilot \
		--shuffle \
		--seed 0

comparison-main:
	PYTHONPATH=$(PYTHONPATH) python scripts/run_pipeline.py --config configs/comparison_main_template.yaml

comparison-main-parallel:
	PYTHONPATH=$(PYTHONPATH) python scripts/run_experiment_parallel.py --config configs/comparison_main_template.yaml --num-shards auto --workers auto
	$(MAKE) postprocess-main

validate-main:
	PYTHONPATH=$(PYTHONPATH) python scripts/validate_generation_outputs.py \
		--input-dir outputs/generations/comparison_gpt2_writingprompts \
		--token-log-dir outputs/token_logs/comparison_gpt2_writingprompts \
		--fail

comparison-distilgpt2-main:
	PYTHONPATH=$(PYTHONPATH) python scripts/run_pipeline.py --config configs/comparison_distilgpt2_main.yaml

comparison-distilgpt2-main-parallel:
	PYTHONPATH=$(PYTHONPATH) python scripts/run_experiment_parallel.py --config configs/comparison_distilgpt2_main.yaml --num-shards auto --workers auto
	$(MAKE) postprocess-distilgpt2-main

validate-distilgpt2-main:
	PYTHONPATH=$(PYTHONPATH) python scripts/validate_generation_outputs.py \
		--input-dir outputs/generations/comparison_distilgpt2_writingprompts \
		--token-log-dir outputs/token_logs/comparison_distilgpt2_writingprompts \
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

postprocess-smoke: validate-smoke manifest-smoke
	PYTHONPATH=$(PYTHONPATH) python scripts/run_evaluate.py \
		--input-dir outputs/generations/comparison_smoke \
		--token-log-dir outputs/token_logs/comparison_smoke \
		--output results/summary/comparison_smoke_metrics.csv
	PYTHONPATH=$(PYTHONPATH) python scripts/build_results_table.py \
		--metrics results/summary/comparison_smoke_metrics.csv \
		--output results/summary/comparison_smoke_table.csv
	$(MAKE) analyze-smoke token-analyze-smoke plot-smoke extended-smoke ppl-smoke trends-smoke merge-smoke metric-suite-smoke

postprocess-distilgpt2-main: validate-distilgpt2-main manifest-distilgpt2-main
	PYTHONPATH=$(PYTHONPATH) python scripts/run_evaluate.py \
		--input-dir outputs/generations/comparison_distilgpt2_writingprompts \
		--token-log-dir outputs/token_logs/comparison_distilgpt2_writingprompts \
		--output results/summary/comparison_distilgpt2_writingprompts_metrics.csv
	PYTHONPATH=$(PYTHONPATH) python scripts/build_results_table.py \
		--metrics results/summary/comparison_distilgpt2_writingprompts_metrics.csv \
		--output results/summary/comparison_distilgpt2_writingprompts_table.csv
	$(MAKE) analyze-distilgpt2-main token-analyze-distilgpt2-main plot-distilgpt2-main extended-distilgpt2-main ppl-distilgpt2-main trends-distilgpt2-main merge-distilgpt2-main metric-suite-distilgpt2-main

analyze-distilgpt2-main:
	PYTHONPATH=$(PYTHONPATH) python scripts/analyze_results.py \
		--table results/summary/comparison_distilgpt2_writingprompts_table.csv \
		--output results/summary/comparison_distilgpt2_writingprompts_analysis.md

plot-distilgpt2-main:
	PYTHONPATH=$(PYTHONPATH) python scripts/plot_results.py \
		--input results/summary/comparison_distilgpt2_writingprompts_metrics.csv \
		--output-dir results/figures/comparison_distilgpt2_writingprompts

token-analyze-distilgpt2-main:
	PYTHONPATH=$(PYTHONPATH) python scripts/analyze_token_logs.py \
		--token-log-dir outputs/token_logs/comparison_distilgpt2_writingprompts \
		--output-dir results/figures/comparison_distilgpt2_writingprompts/token_analysis \
		--summary-output results/summary/comparison_distilgpt2_writingprompts_token_trace.csv

extended-distilgpt2-main:
	PYTHONPATH=$(PYTHONPATH) python scripts/evaluate_extended_metrics.py \
		--input-dir outputs/generations/comparison_distilgpt2_writingprompts \
		--output results/summary/comparison_distilgpt2_writingprompts_extended_metrics.csv

ppl-distilgpt2-main:
	PYTHONPATH=$(PYTHONPATH) python scripts/evaluate_ppl.py \
		--input-dir outputs/generations/comparison_distilgpt2_writingprompts \
		--model distilgpt2 \
		--output results/summary/comparison_distilgpt2_writingprompts_ppl.csv

trends-distilgpt2-main:
	PYTHONPATH=$(PYTHONPATH) python scripts/fit_n_trends.py \
		--table results/summary/comparison_distilgpt2_writingprompts_table.csv \
		--output results/summary/comparison_distilgpt2_writingprompts_n_trend_models.csv \
		--figure-dir results/figures/comparison_distilgpt2_writingprompts/n_trends

merge-distilgpt2-main:
	PYTHONPATH=$(PYTHONPATH) python scripts/merge_metric_tables.py \
		--base results/summary/comparison_distilgpt2_writingprompts_table.csv \
		--extras results/summary/comparison_distilgpt2_writingprompts_extended_metrics.csv results/summary/comparison_distilgpt2_writingprompts_ppl.csv \
		--output results/summary/comparison_distilgpt2_writingprompts_merged_table.csv

metric-suite-distilgpt2-main:
	PYTHONPATH=$(PYTHONPATH) python scripts/plot_metric_suite.py \
		--tables results/summary/comparison_distilgpt2_writingprompts_table.csv results/summary/comparison_distilgpt2_writingprompts_extended_metrics.csv results/summary/comparison_distilgpt2_writingprompts_ppl.csv \
		--output-dir results/figures/comparison_distilgpt2_writingprompts/metric_suite

manifest-distilgpt2-main:
	PYTHONPATH=$(PYTHONPATH) python scripts/write_run_manifest.py \
		--config configs/comparison_distilgpt2_main.yaml \
		--output results/summary/comparison_distilgpt2_writingprompts_manifest.json

postprocess-main: validate-main manifest-main
	PYTHONPATH=$(PYTHONPATH) python scripts/run_evaluate.py \
		--input-dir outputs/generations/comparison_gpt2_writingprompts \
		--token-log-dir outputs/token_logs/comparison_gpt2_writingprompts \
		--output results/summary/comparison_gpt2_writingprompts_metrics.csv
	PYTHONPATH=$(PYTHONPATH) python scripts/build_results_table.py \
		--metrics results/summary/comparison_gpt2_writingprompts_metrics.csv \
		--output results/summary/comparison_gpt2_writingprompts_table.csv
	$(MAKE) analyze-main token-analyze-main plot-main extended-main ppl-main trends-main merge-main metric-suite-main

postprocess-gpt2-lite:
	PYTHONPATH=$(PYTHONPATH) $(VENV_PYTHON) scripts/run_evaluate.py \
		--input-dir outputs/generations/comparison_gpt2_lite \
		--token-log-dir outputs/token_logs/comparison_gpt2_lite \
		--output results/summary/comparison_gpt2_lite_metrics.csv
	PYTHONPATH=$(PYTHONPATH) $(VENV_PYTHON) scripts/build_results_table.py \
		--metrics results/summary/comparison_gpt2_lite_metrics.csv \
		--output results/summary/comparison_gpt2_lite_table.csv
	PYTHONPATH=$(PYTHONPATH) $(VENV_PYTHON) scripts/analyze_token_logs.py \
		--token-log-dir outputs/token_logs/comparison_gpt2_lite \
		--output-dir results/figures/comparison_gpt2_lite/token_analysis \
		--summary-output results/summary/comparison_gpt2_lite_token_trace.csv
	PYTHONPATH=$(PYTHONPATH) $(VENV_PYTHON) scripts/evaluate_extended_metrics.py \
		--input-dir outputs/generations/comparison_gpt2_lite \
		--output results/summary/comparison_gpt2_lite_extended_metrics.csv
	PYTHONPATH=$(PYTHONPATH) $(VENV_PYTHON) scripts/evaluate_ppl.py \
		--input-dir outputs/generations/comparison_gpt2_lite \
		--model gpt2 \
		--output results/summary/comparison_gpt2_lite_ppl.csv
	PYTHONPATH=$(PYTHONPATH) $(VENV_PYTHON) scripts/merge_metric_tables.py \
		--base results/summary/comparison_gpt2_lite_table.csv \
		--extras results/summary/comparison_gpt2_lite_extended_metrics.csv results/summary/comparison_gpt2_lite_ppl.csv \
		--output results/summary/comparison_gpt2_lite_all_table.csv

postprocess-gpt2-medium-micro:
	PYTHONPATH=$(PYTHONPATH) $(VENV_PYTHON) scripts/run_evaluate.py \
		--input-dir outputs/generations/comparison_gpt2_medium_micro \
		--token-log-dir outputs/token_logs/comparison_gpt2_medium_micro \
		--output results/summary/comparison_gpt2_medium_micro_metrics.csv
	PYTHONPATH=$(PYTHONPATH) $(VENV_PYTHON) scripts/build_results_table.py \
		--metrics results/summary/comparison_gpt2_medium_micro_metrics.csv \
		--output results/summary/comparison_gpt2_medium_micro_table.csv
	PYTHONPATH=$(PYTHONPATH) $(VENV_PYTHON) scripts/analyze_token_logs.py \
		--token-log-dir outputs/token_logs/comparison_gpt2_medium_micro \
		--output-dir results/figures/comparison_gpt2_medium_micro/token_analysis \
		--summary-output results/summary/comparison_gpt2_medium_micro_token_trace.csv
	PYTHONPATH=$(PYTHONPATH) $(VENV_PYTHON) scripts/evaluate_extended_metrics.py \
		--input-dir outputs/generations/comparison_gpt2_medium_micro \
		--output results/summary/comparison_gpt2_medium_micro_extended_metrics.csv
	PYTHONPATH=$(PYTHONPATH) $(VENV_PYTHON) scripts/evaluate_ppl.py \
		--input-dir outputs/generations/comparison_gpt2_medium_micro \
		--model gpt2-medium \
		--output results/summary/comparison_gpt2_medium_micro_ppl.csv
	PYTHONPATH=$(PYTHONPATH) $(VENV_PYTHON) scripts/merge_metric_tables.py \
		--base results/summary/comparison_gpt2_medium_micro_table.csv \
		--extras results/summary/comparison_gpt2_medium_micro_extended_metrics.csv results/summary/comparison_gpt2_medium_micro_ppl.csv \
		--output results/summary/comparison_gpt2_medium_micro_all_table.csv

postprocess-qwen25-05b-micro:
	PYTHONPATH=$(PYTHONPATH) $(VENV_PYTHON) scripts/run_evaluate.py \
		--input-dir outputs/generations/comparison_qwen25_05b_micro \
		--token-log-dir outputs/token_logs/comparison_qwen25_05b_micro \
		--output results/summary/comparison_qwen25_05b_micro_metrics.csv
	PYTHONPATH=$(PYTHONPATH) $(VENV_PYTHON) scripts/build_results_table.py \
		--metrics results/summary/comparison_qwen25_05b_micro_metrics.csv \
		--output results/summary/comparison_qwen25_05b_micro_table.csv
	PYTHONPATH=$(PYTHONPATH) $(VENV_PYTHON) scripts/analyze_token_logs.py \
		--token-log-dir outputs/token_logs/comparison_qwen25_05b_micro \
		--output-dir results/figures/comparison_qwen25_05b_micro/token_analysis \
		--summary-output results/summary/comparison_qwen25_05b_micro_token_trace.csv
	PYTHONPATH=$(PYTHONPATH) $(VENV_PYTHON) scripts/evaluate_extended_metrics.py \
		--input-dir outputs/generations/comparison_qwen25_05b_micro \
		--output results/summary/comparison_qwen25_05b_micro_extended_metrics.csv
	@if [ -d models/hf/Qwen__Qwen2.5-0.5B ]; then \
		PYTHONPATH=$(PYTHONPATH) $(VENV_PYTHON) scripts/evaluate_ppl.py \
			--input-dir outputs/generations/comparison_qwen25_05b_micro \
			--model Qwen__Qwen2.5-0.5B \
			--output results/summary/comparison_qwen25_05b_micro_ppl.csv; \
		PYTHONPATH=$(PYTHONPATH) $(VENV_PYTHON) scripts/merge_metric_tables.py \
			--base results/summary/comparison_qwen25_05b_micro_table.csv \
			--extras results/summary/comparison_qwen25_05b_micro_extended_metrics.csv results/summary/comparison_qwen25_05b_micro_ppl.csv \
			--output results/summary/comparison_qwen25_05b_micro_all_table.csv; \
	else \
		echo "PPL skipped for comparison_qwen25_05b_micro: model unavailable locally"; \
		PYTHONPATH=$(PYTHONPATH) $(VENV_PYTHON) scripts/merge_metric_tables.py \
			--base results/summary/comparison_qwen25_05b_micro_table.csv \
			--extras results/summary/comparison_qwen25_05b_micro_extended_metrics.csv \
			--output results/summary/comparison_qwen25_05b_micro_all_table.csv; \
	fi

analyze-main:
	PYTHONPATH=$(PYTHONPATH) python scripts/analyze_results.py \
		--table results/summary/comparison_gpt2_writingprompts_table.csv \
		--output results/summary/comparison_gpt2_writingprompts_analysis.md

plot-main:
	PYTHONPATH=$(PYTHONPATH) python scripts/plot_results.py \
		--input results/summary/comparison_gpt2_writingprompts_metrics.csv \
		--output-dir results/figures/comparison_gpt2_writingprompts

token-analyze-main:
	PYTHONPATH=$(PYTHONPATH) python scripts/analyze_token_logs.py \
		--token-log-dir outputs/token_logs/comparison_gpt2_writingprompts \
		--output-dir results/figures/comparison_gpt2_writingprompts/token_analysis \
		--summary-output results/summary/comparison_gpt2_writingprompts_token_trace.csv

extended-main:
	PYTHONPATH=$(PYTHONPATH) python scripts/evaluate_extended_metrics.py \
		--input-dir outputs/generations/comparison_gpt2_writingprompts \
		--output results/summary/comparison_gpt2_writingprompts_extended_metrics.csv

ppl-main:
	PYTHONPATH=$(PYTHONPATH) python scripts/evaluate_ppl.py \
		--input-dir outputs/generations/comparison_gpt2_writingprompts \
		--model gpt2 \
		--output results/summary/comparison_gpt2_writingprompts_ppl.csv

trends-main:
	PYTHONPATH=$(PYTHONPATH) python scripts/fit_n_trends.py \
		--table results/summary/comparison_gpt2_writingprompts_table.csv \
		--output results/summary/comparison_gpt2_writingprompts_n_trend_models.csv \
		--figure-dir results/figures/comparison_gpt2_writingprompts/n_trends

merge-main:
	PYTHONPATH=$(PYTHONPATH) python scripts/merge_metric_tables.py \
		--base results/summary/comparison_gpt2_writingprompts_table.csv \
		--extras results/summary/comparison_gpt2_writingprompts_extended_metrics.csv results/summary/comparison_gpt2_writingprompts_ppl.csv \
		--output results/summary/comparison_gpt2_writingprompts_merged_table.csv

metric-suite-main:
	PYTHONPATH=$(PYTHONPATH) python scripts/plot_metric_suite.py \
		--tables results/summary/comparison_gpt2_writingprompts_table.csv results/summary/comparison_gpt2_writingprompts_extended_metrics.csv results/summary/comparison_gpt2_writingprompts_ppl.csv \
		--output-dir results/figures/comparison_gpt2_writingprompts/metric_suite

manifest-main:
	PYTHONPATH=$(PYTHONPATH) python scripts/write_run_manifest.py \
		--config configs/comparison_main_template.yaml \
		--output results/summary/comparison_gpt2_writingprompts_manifest.json

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

writingprompts-auto:
	$${PYTHONPATH:-src} .venv/bin/python scripts/run_writingprompts_auto.py --yes

writingprompts-auto-dry:
	PYTHONPATH=src .venv/bin/python scripts/run_writingprompts_auto.py --dry-run --yes
