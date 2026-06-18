from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass
class ResourceCheckResult:
    swap_used_bytes: int | None
    available_ram_bytes: int | None
    free_vram_bytes: int | None
    cuda_device_index: int | None


def _parse_meminfo_kb(key: str) -> int | None:
    meminfo_path = Path("/proc/meminfo")
    if not meminfo_path.exists():
        return None
    for line in meminfo_path.read_text(encoding="utf-8").splitlines():
        if not line.startswith(key + ":"):
            continue
        parts = line.split()
        if len(parts) >= 2:
            return int(parts[1]) * 1024
    return None


def _swap_used_bytes() -> int | None:
    swaps_path = Path("/proc/swaps")
    if not swaps_path.exists():
        return None
    lines = swaps_path.read_text(encoding="utf-8").splitlines()
    if len(lines) <= 1:
        return 0
    total = 0
    for line in lines[1:]:
        parts = line.split()
        if len(parts) >= 4:
            total += int(parts[3]) * 1024
    return total


def check_runtime_resources() -> ResourceCheckResult:
    available_ram_bytes = _parse_meminfo_kb("MemAvailable")
    swap_used_bytes = _swap_used_bytes()

    free_vram_bytes: int | None = None
    cuda_device_index: int | None = None
    if torch.cuda.is_available():
        cuda_device_index = torch.cuda.current_device()
        free_vram_bytes, _ = torch.cuda.mem_get_info()

    return ResourceCheckResult(
        swap_used_bytes=swap_used_bytes,
        available_ram_bytes=available_ram_bytes,
        free_vram_bytes=free_vram_bytes,
        cuda_device_index=cuda_device_index,
    )


def format_bytes(num_bytes: int | None) -> str:
    if num_bytes is None:
        return "UNKNOWN"
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    value = float(num_bytes)
    unit = 0
    while value >= 1024 and unit < len(units) - 1:
        value /= 1024.0
        unit += 1
    return f"{value:.2f} {units[unit]}"
