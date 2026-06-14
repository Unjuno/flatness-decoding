from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch
import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def file_sha256(path: str | Path) -> str | None:
    p = Path(path)
    if not p.exists():
        return None
    digest = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def package_version(name: str) -> str | None:
    try:
        module = __import__(name)
        return getattr(module, "__version__", None)
    except Exception:
        return None


def torch_device_summary() -> dict[str, Any]:
    cuda_available = torch.cuda.is_available()
    mps_available = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    return {
        "torch_version": torch.__version__,
        "cuda_available": cuda_available,
        "cuda_device_count": torch.cuda.device_count() if cuda_available else 0,
        "cuda_device_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        if cuda_available
        else [],
        "mps_available": mps_available,
    }


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    manifest = {
        "config_path": str(config_path),
        "config_sha256": file_sha256(config_path),
        "config": config,
        "git_commit": git_commit(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "packages": {
            "torch": torch.__version__,
            "transformers": package_version("transformers"),
            "datasets": package_version("datasets"),
            "numpy": package_version("numpy"),
            "pandas": package_version("pandas"),
        },
        "devices": torch_device_summary(),
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
