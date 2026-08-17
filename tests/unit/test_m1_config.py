"""Tests for the frozen M1 initialization validation matrix."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from vllm_tuner.longctx.m1_config import LongContextM1Config, load_longctx_m1_config

REVISION = "a" * 40


def _identity_files(tmp_path: Path) -> tuple[Path, Path, Path]:
    model_dir = tmp_path / "model"
    model_dir.mkdir(parents=True)
    config = {
        "architectures": ["FakeDenseForCausalLM"],
        "num_hidden_layers": 28,
        "hidden_size": 3584,
        "num_attention_heads": 28,
        "num_key_value_heads": 4,
        "max_position_embeddings": 32768,
        "torch_dtype": "bfloat16",
    }
    payloads = {
        "config.json": json.dumps(config).encode(),
        "tokenizer.json": b"{}\n",
        "model.safetensors": b"weights\n",
    }
    for name, payload in payloads.items():
        (model_dir / name).write_bytes(payload)
    model_lock = tmp_path / "model.lock.yaml"
    model_lock.write_text(
        yaml.safe_dump(
            {
                "model": {
                    "repository_id": "Qwen/Test-Model",
                    "revision": REVISION,
                    "parameter_count": 7_615_616_512,
                    "files": {
                        name: {
                            "size_bytes": len(payload),
                            "sha256": hashlib.sha256(payload).hexdigest(),
                        }
                        for name, payload in payloads.items()
                    },
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    runtime_lock = tmp_path / "runtime.lock.yaml"
    runtime_lock.write_text("vllm: {}\n", encoding="utf-8")
    return model_dir, model_lock, runtime_lock


def _valid_data(tmp_path: Path) -> dict[str, object]:
    model_dir, model_lock, runtime_lock = _identity_files(tmp_path)
    return {
        "project_line": "longctx-v5",
        "milestone": "M1",
        "experiment_kind": "planner-initialization-validation",
        "model": {"local_path": str(model_dir), "lock_path": str(model_lock)},
        "runtime": {"lock_path": str(runtime_lock)},
        "artifacts": {"root": str(tmp_path / "artifacts")},
        "gpu": {"device_ids": [0], "count": 1},
        "probes": (
            {
                "probe_id": "cal-75",
                "role": "calibration",
                "gpu_memory_utilization_ppm": 750_000,
                "max_model_len": 32768,
                "max_num_seqs": 512,
                "repeats": 2,
            },
            {
                "probe_id": "cal-80",
                "role": "calibration",
                "gpu_memory_utilization_ppm": 800_000,
                "max_model_len": 32768,
                "max_num_seqs": 512,
                "repeats": 2,
            },
            {
                "probe_id": "cal-85",
                "role": "calibration",
                "gpu_memory_utilization_ppm": 850_000,
                "max_model_len": 32768,
                "max_num_seqs": 512,
                "repeats": 2,
            },
            {
                "probe_id": "heldout-32k",
                "role": "validation",
                "gpu_memory_utilization_ppm": 900_000,
                "max_model_len": 32768,
                "max_num_seqs": 512,
                "repeats": 1,
            },
            {
                "probe_id": "extrapolate-8k",
                "role": "extrapolation",
                "gpu_memory_utilization_ppm": 900_000,
                "max_model_len": 8192,
                "max_num_seqs": 512,
                "repeats": 1,
            },
            {
                "probe_id": "extrapolate-16k",
                "role": "extrapolation",
                "gpu_memory_utilization_ppm": 900_000,
                "max_model_len": 16384,
                "max_num_seqs": 512,
                "repeats": 1,
            },
        ),
        "deployment_distribution": {
            "bins": (
                {
                    "name": "short",
                    "weight_ppm": 500_000,
                    "prompt_tokens": 8064,
                    "reserved_output_tokens": 128,
                },
                {
                    "name": "medium",
                    "weight_ppm": 300_000,
                    "prompt_tokens": 16256,
                    "reserved_output_tokens": 128,
                },
                {
                    "name": "long",
                    "weight_ppm": 200_000,
                    "prompt_tokens": 32640,
                    "reserved_output_tokens": 128,
                },
            ),
            "confidence_ppm": 990_000,
            "iid_assumption": True,
            "assume_no_prefix_reuse": True,
        },
        "safety": {
            "fixed_operational_reserve_bytes": 256 * (1 << 20),
            "kv_reserve_basis_points": 500,
            "calibration_residual_upper_bytes": 0,
            "source": "test",
        },
    }


def test_valid_m1_matrix_and_tuning_conversion(tmp_path: Path) -> None:
    config = LongContextM1Config.model_validate(_valid_data(tmp_path))
    probe = next(probe for probe in config.probes if probe.probe_id == "heldout-32k")
    tuning = config.to_tuning_config(probe)

    assert len(config.probes) == 6
    assert tuning.model_revision == REVISION
    assert tuning.vllm_args == {
        "gpu-memory-utilization": 0.9,
        "max-model-len": 32768,
        "max-num-seqs": 512,
    }
    assert tuning.adaptive_prefill.enabled is False


def test_yaml_loader_normalizes_lists_without_weakening_strict_fields(
    tmp_path: Path,
) -> None:
    data = _valid_data(tmp_path)
    data["probes"] = list(data["probes"])
    distribution = copy.deepcopy(data["deployment_distribution"])
    assert isinstance(distribution, dict)
    distribution["bins"] = list(distribution["bins"])
    data["deployment_distribution"] = distribution
    path = tmp_path / "m1.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    config = load_longctx_m1_config(path)

    assert isinstance(config.probes, tuple)
    assert isinstance(config.deployment_distribution.bins, tuple)


@pytest.mark.parametrize(
    "mutation,error",
    [
        ("insufficient-repeats", "three calibration points with two repeats"),
        ("two-utilizations", "three utilization points"),
        ("validation-profile", "calibration max_model_len"),
        ("validation-not-heldout", "held out from calibration"),
        ("two-contexts", "three validation/extrapolation context points"),
        ("mixed-max-num-seqs", "one max_num_seqs"),
        ("duplicate-id", "probe IDs must be unique"),
    ],
)
def test_invalid_probe_matrix_is_rejected(tmp_path: Path, mutation: str, error: str) -> None:
    data = _valid_data(tmp_path)
    probes = [dict(probe) for probe in data["probes"]]
    if mutation == "insufficient-repeats":
        probes[0]["repeats"] = 1
    elif mutation == "two-utilizations":
        probes[2]["gpu_memory_utilization_ppm"] = 800_000
    elif mutation == "validation-profile":
        probes[3]["max_model_len"] = 16384
    elif mutation == "validation-not-heldout":
        probes[3]["gpu_memory_utilization_ppm"] = 850_000
    elif mutation == "two-contexts":
        probes[-1]["max_model_len"] = 8192
    elif mutation == "mixed-max-num-seqs":
        probes[-1]["max_num_seqs"] = 256
    else:
        probes[-1]["probe_id"] = probes[-2]["probe_id"]
    data["probes"] = tuple(probes)

    with pytest.raises(ValidationError, match=error):
        LongContextM1Config.model_validate(data)


def test_probe_context_cannot_exceed_model_limit(tmp_path: Path) -> None:
    data = _valid_data(tmp_path)
    probes = [dict(probe) for probe in data["probes"]]
    probes[-1]["max_model_len"] = 65536
    data["probes"] = tuple(probes)

    with pytest.raises(ValidationError, match="exceeds model capacity"):
        LongContextM1Config.model_validate(data)


def test_loader_rejects_duplicate_yaml_keys(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.yaml"
    path.write_text('milestone: "M1"\nmilestone: "M1"\n', encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate key"):
        load_longctx_m1_config(path)
