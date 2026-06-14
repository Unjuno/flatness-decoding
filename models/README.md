# Local model cache

Model weights and tokenizer files are downloaded locally and are not committed to Git.

Default cache directory:

```text
models/hf/
```

This directory is ignored by `.gitignore` through the existing `/models/` rule.

## Download models explicitly

```bash
PYTHONPATH=src python scripts/download_models.py
```

or:

```bash
make download-models
```

This reads:

```text
configs/model_manifest.yaml
```

## Download specific models

```bash
PYTHONPATH=src python scripts/download_models.py \
  --models distilgpt2 gpt2 \
  --cache-dir models/hf
```

## Automatic download

Experiment scripts call `AutoTokenizer.from_pretrained` and `AutoModelForCausalLM.from_pretrained` with `cache_dir=models/hf` by default. If the requested model is not cached, `transformers` downloads it on first use.

## Do not commit models

Do not commit:

- model weights
- tokenizer cache files
- safetensors/bin checkpoint files
- Hugging Face cache snapshots

Use DVC or external storage only if model artifacts must be versioned for reproduction.
