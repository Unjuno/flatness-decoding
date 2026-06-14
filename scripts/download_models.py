from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=None, help="Model names to download.")
    parser.add_argument("--manifest", default="configs/model_manifest.yaml")
    parser.add_argument("--cache-dir", default="models/hf")
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def load_manifest(path: str | Path) -> list[str]:
    p = Path(path)
    if not p.exists():
        return []
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    models = data.get("models", [])
    if not isinstance(models, list):
        raise ValueError("model manifest must contain a list under key 'models'")
    return [str(item) for item in models]


def download_model(model_name: str, cache_dir: str, trust_remote_code: bool = False) -> None:
    print(f"downloading/checking tokenizer: {model_name}")
    AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        trust_remote_code=trust_remote_code,
    )
    print(f"downloading/checking model: {model_name}")
    AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        trust_remote_code=trust_remote_code,
    )
    print(f"ready: {model_name}")


def main() -> None:
    args = parse_args()
    models = args.models if args.models else load_manifest(args.manifest)
    if not models:
        raise SystemExit("No models specified. Use --models or provide configs/model_manifest.yaml")

    Path(args.cache_dir).mkdir(parents=True, exist_ok=True)
    for model_name in models:
        download_model(model_name, args.cache_dir, args.trust_remote_code)

    print(f"all requested models are cached under {args.cache_dir}")


if __name__ == "__main__":
    main()
