from types import SimpleNamespace

import pytest

import scripts.run_experiment_set as run_experiment_set
from flatness_decoding.resources import format_bytes


def test_format_bytes_uses_binary_units():
    assert format_bytes(1024) == "1.00 KiB"
    assert format_bytes(1024**3) == "1.00 GiB"


def test_preflight_rejects_swap_when_disallowed(monkeypatch):
    monkeypatch.setattr(
        run_experiment_set,
        "check_runtime_resources",
        lambda: SimpleNamespace(
            swap_used_bytes=1024,
            available_ram_bytes=8 * 1024**3,
            free_vram_bytes=8 * 1024**3,
            cuda_device_index=0,
        ),
    )
    monkeypatch.setattr(run_experiment_set, "resolve_device", lambda name: SimpleNamespace(type="cpu"))
    with pytest.raises(SystemExit, match="swap is in use"):
        run_experiment_set.preflight_resource_check(
            SimpleNamespace(allow_swap=False, min_free_ram_gb=2.0, min_free_vram_gb=1.0),
            "auto",
        )


def test_preflight_allows_swap_when_explicit(monkeypatch):
    monkeypatch.setattr(
        run_experiment_set,
        "check_runtime_resources",
        lambda: SimpleNamespace(
            swap_used_bytes=1024,
            available_ram_bytes=8 * 1024**3,
            free_vram_bytes=8 * 1024**3,
            cuda_device_index=0,
        ),
    )
    monkeypatch.setattr(run_experiment_set, "resolve_device", lambda name: SimpleNamespace(type="cpu"))
    run_experiment_set.preflight_resource_check(
        SimpleNamespace(allow_swap=True, min_free_ram_gb=2.0, min_free_vram_gb=1.0),
        "auto",
    )

