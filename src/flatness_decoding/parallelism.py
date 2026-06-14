from __future__ import annotations

import os
from dataclasses import asdict, dataclass

import torch


@dataclass
class ParallelPlan:
    backend: str
    cpu_count: int
    cuda_available: bool
    cuda_device_count: int
    mps_available: bool
    recommended_workers: int
    recommended_num_shards: int
    recommended_devices: list[str]
    caution: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def detect_parallel_plan(
    requested_num_shards: int | str = "auto",
    requested_workers: int | str = "auto",
    max_cpu_workers: int = 4,
) -> ParallelPlan:
    """Return a conservative hardware-aware parallel execution plan.

    Policy:
    - Multi-GPU CUDA: one worker per visible GPU by default.
    - Single-GPU CUDA: one worker by default to avoid OOM.
    - Apple MPS: one worker by default because multiple processes can compete for memory.
    - CPU-only: up to half of logical CPUs, capped by max_cpu_workers.
    """
    cpu_count = os.cpu_count() or 1
    cuda_available = torch.cuda.is_available()
    cuda_device_count = torch.cuda.device_count() if cuda_available else 0
    mps_available = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())

    if cuda_available and cuda_device_count > 1:
        backend = "cuda_multi"
        default_workers = cuda_device_count
        default_devices = [str(i) for i in range(cuda_device_count)]
        caution = "Uses one worker per CUDA device by default. Increase shards for more scheduling granularity."
    elif cuda_available and cuda_device_count == 1:
        backend = "cuda_single"
        default_workers = 1
        default_devices = ["0"]
        caution = "Single GPU detected. Keep workers at 1 unless the model is small and memory headroom is known."
    elif mps_available:
        backend = "mps"
        default_workers = 1
        default_devices = []
        caution = "Apple MPS detected. Multiple model processes can contend for memory; start with one worker."
    else:
        backend = "cpu"
        default_workers = max(1, min(max_cpu_workers, cpu_count // 2 or 1))
        default_devices = []
        caution = "CPU-only detected. Increase workers gradually if memory is sufficient."

    workers = default_workers if requested_workers == "auto" else int(requested_workers)
    workers = max(1, workers)

    default_num_shards = max(workers, 4 if workers > 1 else 1)
    num_shards = default_num_shards if requested_num_shards == "auto" else int(requested_num_shards)
    num_shards = max(1, num_shards)

    # If the user asks for more workers than shards, cap workers to shard count.
    workers = min(workers, num_shards)

    return ParallelPlan(
        backend=backend,
        cpu_count=cpu_count,
        cuda_available=cuda_available,
        cuda_device_count=cuda_device_count,
        mps_available=mps_available,
        recommended_workers=workers,
        recommended_num_shards=num_shards,
        recommended_devices=default_devices if backend == "cuda_multi" else [],
        caution=caution,
    )
