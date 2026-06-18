from __future__ import annotations

import argparse
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import yaml
from tqdm import tqdm

from flatness_decoding.baselines import generate_baseline
from flatness_decoding.decoding import generate_flatness_controlled
from flatness_decoding.implementation import IMPLEMENTATION_PROFILE, USE_KV_CACHE_LOOKAHEAD
from flatness_decoding.io_utils import read_jsonl, write_jsonl
from flatness_decoding.memory import cleanup_config, cleanup_device_cache
from flatness_decoding.metrics import continuation_from_text
from flatness_decoding.model_io import load_causal_lm
from flatness_decoding.resources import check_runtime_resources, format_bytes
from flatness_decoding.devices import resolve_device
from flatness_decoding.resume import baseline_output_complete, ntrial_output_complete, resume_config


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--candidate-eval-batch-size", type=int, default=None)
    p.add_argument("--inference-mode", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--allow-swap", action="store_true", help="Allow execution even if system swap is in use.")
    p.add_argument("--min-free-ram-gb", type=float, default=2.0, help="Abort if available RAM is below this threshold.")
    p.add_argument("--min-free-vram-gb", type=float, default=1.0, help="Abort if free CUDA VRAM is below this threshold.")
    return p.parse_args()


def load_config(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def apply_overrides(cfg: dict, args: argparse.Namespace) -> dict:
    if args.candidate_eval_batch_size is not None:
        cfg.setdefault("n_trial_methods", {})["candidate_eval_batch_size"] = int(args.candidate_eval_batch_size)
    if args.inference_mode:
        cfg.setdefault("runtime", {})["inference_mode"] = True
    if args.resume and args.no_resume:
        raise SystemExit("--resume and --no-resume are mutually exclusive")
    if args.resume:
        cfg.setdefault("runtime", {}).setdefault("resume", {})["skip_complete"] = True
    if args.no_resume:
        cfg.setdefault("runtime", {}).setdefault("resume", {})["skip_complete"] = False
    return cfg


def preflight_resource_check(args: argparse.Namespace, device_name: str) -> None:
    resources = check_runtime_resources()
    if resources.swap_used_bytes and resources.swap_used_bytes > 0 and not args.allow_swap:
        raise SystemExit(
            f"swap is in use ({format_bytes(resources.swap_used_bytes)}). Re-run with --allow-swap if intentional."
        )

    if resources.available_ram_bytes is not None:
        min_free_ram_bytes = int(args.min_free_ram_gb * 1024**3)
        if resources.available_ram_bytes < min_free_ram_bytes:
            raise SystemExit(
                f"available RAM too low: {format_bytes(resources.available_ram_bytes)} < {args.min_free_ram_gb:.2f} GiB"
            )

    if resolve_device(device_name).type == "cuda":
        if resources.free_vram_bytes is not None:
            min_free_vram_bytes = int(args.min_free_vram_gb * 1024**3)
            if resources.free_vram_bytes < min_free_vram_bytes:
                raise SystemExit(
                    f"free VRAM too low: {format_bytes(resources.free_vram_bytes)} < {args.min_free_vram_gb:.2f} GiB"
                )
        else:
            print("warning: CUDA VRAM free-space check unavailable; continuing without a VRAM preflight")


def inference_mode_enabled(cfg: dict) -> bool:
    return bool((cfg.get("runtime", {}) or {}).get("inference_mode", False))


def inference_context(cfg: dict):
    return torch.inference_mode() if inference_mode_enabled(cfg) else nullcontext()


def shard_suffix(num_shards: int, shard_index: int) -> str:
    return "" if num_shards <= 1 else f"_shard={shard_index}-of-{num_shards}"


def baseline_stem(exp_name: str, model_name: str, method: str, seed: int, suffix: str) -> str:
    return f"experiment={exp_name}_model={model_name.replace('/', '__')}_method={method}_seed={seed}{suffix}"


def ntrial_stem(exp_name: str, model_name: str, method: str, n_trials: int, proposal_temperature: float, seed: int, suffix: str) -> str:
    return f"experiment={exp_name}_model={model_name.replace('/', '__')}_method={method}_n={n_trials}_temp={proposal_temperature}_seed={seed}{suffix}"


def select_prompt_shard(prompts: list[dict], num_shards: int, shard_index: int) -> list[dict]:
    if num_shards < 1:
        raise ValueError("num_shards must be >= 1")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError("shard_index must satisfy 0 <= shard_index < num_shards")
    return prompts if num_shards == 1 else [row for idx, row in enumerate(prompts) if idx % num_shards == shard_index]


def run_baseline_grid(cfg: dict, model, tokenizer, device, prompts: list[dict], num_shards: int, shard_index: int) -> None:
    model_name = cfg["model_name"]
    exp_name = cfg["experiment_name"]
    gen_cfg = cfg["generation"]
    baseline_cfg = cfg.get("baselines", {})
    cleanup_cfg = cleanup_config(cfg)
    resume_cfg = resume_config(cfg)
    inference_mode = inference_mode_enabled(cfg)
    output_dir = Path(cfg["output"]["generations_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = shard_suffix(num_shards, shard_index)

    for method in baseline_cfg.get("methods", []):
        for seed in cfg.get("seeds", [0]):
            output_path = output_dir / f"{baseline_stem(exp_name, model_name, method, seed, suffix)}.jsonl"
            if resume_cfg["skip_complete"] and baseline_output_complete(
                output_path,
                expected_rows=len(prompts),
                method=method,
                seed=seed,
                shard_index=shard_index,
                num_shards=num_shards,
                inference_mode=inference_mode,
            ):
                print(f"skip complete baseline group: {output_path}")
                continue

            rows = []
            for local_idx, row in enumerate(tqdm(prompts, desc=f"baseline {method} seed={seed}{suffix}")):
                prompt_id = row.get("id", local_idx)
                original_prompt_index = row.get("original_prompt_index", local_idx)
                prompt = str(row["prompt"])
                actual_seed = seed + int(original_prompt_index)
                started = time.perf_counter()
                with inference_context(cfg):
                    result = generate_baseline(
                        model=model,
                        tokenizer=tokenizer,
                        prompt=prompt,
                        method=method,
                        max_new_tokens=int(gen_cfg.get("max_new_tokens", 100)),
                        temperature=float(baseline_cfg.get("temperature", 1.0)),
                        top_p=float(baseline_cfg.get("top_p", 0.9)),
                        top_k=int(baseline_cfg.get("top_k", 50)),
                        num_beams=int(baseline_cfg.get("num_beams", 4)),
                        device=device,
                        seed=actual_seed,
                    )
                elapsed = time.perf_counter() - started
                rows.append({
                    "implementation_profile": IMPLEMENTATION_PROFILE,
                    "experiment_name": exp_name,
                    "prompt_id": prompt_id,
                    "prompt_index": original_prompt_index,
                    "shard_index": shard_index,
                    "num_shards": num_shards,
                    "model": model_name,
                    "method": method,
                    "method_family": "baseline",
                    "n_trials": None,
                    "proposal_temperature": None,
                    "candidate_eval_batch_size": None,
                    "use_kv_cache_lookahead": None,
                    "proposal_sampling_device": None,
                    "selection_device": None,
                    "inference_mode": inference_mode,
                    "resume_skip_complete": resume_cfg["skip_complete"],
                    "cleanup_device_cache_after_prompt": cleanup_cfg["device_cache_after_prompt"],
                    "cleanup_device_cache_after_group": cleanup_cfg["device_cache_after_group"],
                    "baseline_temperature": baseline_cfg.get("temperature", 1.0),
                    "top_p": baseline_cfg.get("top_p"),
                    "top_k": baseline_cfg.get("top_k"),
                    "num_beams": baseline_cfg.get("num_beams"),
                    "seed": seed,
                    "actual_seed": actual_seed,
                    "prompt": prompt,
                    "text": result.text,
                    "continuation_text": continuation_from_text(result.text, prompt),
                    "num_input_tokens": result.num_input_tokens,
                    "num_output_tokens": result.num_output_tokens,
                    "runtime_seconds": elapsed,
                    "tokens_per_second": result.num_output_tokens / elapsed if elapsed > 0 else None,
                })
                if cleanup_cfg["device_cache_after_prompt"]:
                    cleanup_device_cache(device)

            write_jsonl(output_path, rows)
            if cleanup_cfg["device_cache_after_group"]:
                cleanup_device_cache(device, collect_python=cleanup_cfg["python_gc_after_group"])


def run_n_trial_grid(cfg: dict, model, tokenizer, device, prompts: list[dict], num_shards: int, shard_index: int) -> None:
    model_name = cfg["model_name"]
    exp_name = cfg["experiment_name"]
    gen_cfg = cfg["generation"]
    ntrial_cfg = cfg.get("n_trial_methods", {})
    cleanup_cfg = cleanup_config(cfg)
    resume_cfg = resume_config(cfg)
    inference_mode = inference_mode_enabled(cfg)
    output_dir = Path(cfg["output"]["generations_dir"])
    token_log_dir = Path(cfg["output"]["token_log_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    token_log_dir.mkdir(parents=True, exist_ok=True)
    proposal_temperature = float(ntrial_cfg.get("proposal_temperature", 1.0))
    candidate_eval_batch_size = ntrial_cfg.get("candidate_eval_batch_size")
    if candidate_eval_batch_size is not None:
        candidate_eval_batch_size = int(candidate_eval_batch_size)
    use_kv_cache_lookahead = bool(ntrial_cfg.get("use_kv_cache_lookahead", USE_KV_CACHE_LOOKAHEAD))
    highprob_logprob_margin = ntrial_cfg.get("highprob_logprob_margin")
    if highprob_logprob_margin is not None:
        highprob_logprob_margin = float(highprob_logprob_margin)
    highprob_rank_limit = ntrial_cfg.get("highprob_rank_limit")
    if highprob_rank_limit is not None:
        highprob_rank_limit = int(highprob_rank_limit)
    suffix = shard_suffix(num_shards, shard_index)

    for method in ntrial_cfg.get("methods", []):
        for n_trials in ntrial_cfg.get("n_values", [1, 4, 8, 16]):
            for seed in cfg.get("seeds", [0]):
                stem = ntrial_stem(exp_name, model_name, method, int(n_trials), proposal_temperature, seed, suffix)
                output_path = output_dir / f"{stem}.jsonl"
                token_log_path = token_log_dir / f"{stem}.tokenlog.jsonl"
                if resume_cfg["skip_complete"] and ntrial_output_complete(
                    output_path,
                    token_log_path,
                    expected_rows=len(prompts),
                    method=method,
                    n_trials=int(n_trials),
                    seed=seed,
                    shard_index=shard_index,
                    num_shards=num_shards,
                    candidate_eval_batch_size=candidate_eval_batch_size,
                    inference_mode=inference_mode,
                    use_kv_cache_lookahead=use_kv_cache_lookahead,
                ):
                    print(f"skip complete n-trial group: {output_path}")
                    continue

                rows = []
                token_rows = []
                for local_idx, row in enumerate(tqdm(prompts, desc=f"n-trial {method} N={n_trials} seed={seed}{suffix}")):
                    prompt_id = row.get("id", local_idx)
                    original_prompt_index = row.get("original_prompt_index", local_idx)
                    prompt = str(row["prompt"])
                    actual_seed = seed + int(original_prompt_index)
                    started = time.perf_counter()
                    with inference_context(cfg):
                        text, records = generate_flatness_controlled(
                            model=model,
                            tokenizer=tokenizer,
                            prompt=prompt,
                            max_new_tokens=int(gen_cfg.get("max_new_tokens", 100)),
                            n_trials=int(n_trials),
                            proposal_temperature=proposal_temperature,
                            method=method,
                            device=device,
                            seed=actual_seed,
                            candidate_eval_batch_size=candidate_eval_batch_size,
                            use_kv_cache_lookahead=use_kv_cache_lookahead,
                            highprob_logprob_margin=highprob_logprob_margin,
                            highprob_rank_limit=highprob_rank_limit,
                        )
                    elapsed = time.perf_counter() - started
                    num_output_tokens = len(records)
                    common = {
                        "implementation_profile": IMPLEMENTATION_PROFILE,
                        "experiment_name": exp_name,
                        "prompt_id": prompt_id,
                        "prompt_index": original_prompt_index,
                        "shard_index": shard_index,
                        "num_shards": num_shards,
                        "model": model_name,
                        "method": method,
                        "n_trials": int(n_trials),
                        "proposal_temperature": proposal_temperature,
                        "candidate_eval_batch_size": candidate_eval_batch_size,
                        "use_kv_cache_lookahead": use_kv_cache_lookahead,
                        "highprob_logprob_margin": highprob_logprob_margin,
                        "highprob_rank_limit": highprob_rank_limit,
                        "proposal_sampling_device": "cpu",
                        "selection_device": "cpu",
                        "inference_mode": inference_mode,
                        "resume_skip_complete": resume_cfg["skip_complete"],
                        "cleanup_device_cache_after_prompt": cleanup_cfg["device_cache_after_prompt"],
                        "cleanup_device_cache_after_group": cleanup_cfg["device_cache_after_group"],
                        "seed": seed,
                        "actual_seed": actual_seed,
                    }
                    rows.append({
                        **common,
                        "method_family": "n_trial",
                        "baseline_temperature": None,
                        "top_p": None,
                        "top_k": None,
                        "num_beams": None,
                        "prompt": prompt,
                        "text": text,
                        "continuation_text": continuation_from_text(text, prompt),
                        "num_input_tokens": None,
                        "num_output_tokens": num_output_tokens,
                        "runtime_seconds": elapsed,
                        "tokens_per_second": num_output_tokens / elapsed if elapsed > 0 else None,
                    })
                    for record in records:
                        token_rows.append({**record.to_dict(), **common})
                    if cleanup_cfg["device_cache_after_prompt"]:
                        cleanup_device_cache(device)

                write_jsonl(output_path, rows)
                write_jsonl(token_log_path, token_rows)
                if cleanup_cfg["device_cache_after_group"]:
                    cleanup_device_cache(device, collect_python=cleanup_cfg["python_gc_after_group"])


def main() -> None:
    args = parse_args()
    cfg = apply_overrides(load_config(args.config), args)
    prompts = read_jsonl(cfg["prompts"]["path"])
    prompts = [{**row, "original_prompt_index": idx} for idx, row in enumerate(prompts)]
    limit = cfg.get("prompts", {}).get("limit")
    if limit is not None:
        prompts = prompts[: int(limit)]
    prompts = select_prompt_shard(prompts, args.num_shards, args.shard_index)
    print(f"Loaded {len(prompts)} prompts for shard {args.shard_index}/{args.num_shards} from {cfg['prompts']['path']}")
    preflight_resource_check(args, str(cfg.get("device", "auto")))
    model, tokenizer, device = load_causal_lm(
        model_name=cfg["model_name"],
        device_name=cfg.get("device", "auto"),
        dtype_name=cfg.get("dtype", "auto"),
        cache_dir=cfg.get("model_cache_dir", "models/hf"),
    )
    run_baseline_grid(cfg, model, tokenizer, device, prompts, args.num_shards, args.shard_index)
    run_n_trial_grid(cfg, model, tokenizer, device, prompts, args.num_shards, args.shard_index)


if __name__ == "__main__":
    main()
