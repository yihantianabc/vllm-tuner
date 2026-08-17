"""Tests for the strict long-context v5 M1 capacity-sweep contract."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from vllm_tuner.longctx.m1_capacity_config import (
    M1CapacityContext,
    M1CapacityLoad,
    LongContextM1CapacityConfig,
    load_longctx_m1_capacity_config,
)

REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"
SOURCE_COMMIT = "b" * 40
INITIALIZATION_ID = "longctx-v5-m1-planner-init-test"


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
    return model_dir.resolve(), model_lock.resolve(), runtime_lock.resolve()


def _seal_initialization(
    artifact_root: Path,
    model_dir: Path,
    model_lock: Path,
    runtime_lock: Path,
    *,
    accepted: bool = True,
) -> Path:
    root = artifact_root / INITIALIZATION_ID
    root.mkdir(parents=True)
    experiment = {
        "project_line": "longctx-v5",
        "milestone": "M1",
        "experiment_kind": "planner-initialization-validation",
        "model": {"local_path": str(model_dir), "lock_path": str(model_lock)},
        "runtime": {"lock_path": str(runtime_lock)},
        "artifacts": {"root": str(artifact_root)},
        "gpu": {"device_ids": [0], "count": 1},
    }
    manifest = {
        "schema_version": "longctx-m1-init.v2",
        "experiment_id": INITIALIZATION_ID,
        "source_commit": SOURCE_COMMIT,
        "model": {
            "lock_path": str(model_lock),
            "model_dir": str(model_dir),
            "expected_repository_id": "Qwen/Test-Model",
            "expected_revision": REVISION,
            "expected_parameter_count": 7_615_616_512,
            "matches_lock": True,
        },
        "runtime": {"lock_path": str(runtime_lock), "matches_lock": True},
    }
    summary = {
        "schema_version": "longctx-m1-init.v2",
        "experiment_id": INITIALIZATION_ID,
        "source_commit": SOURCE_COMMIT,
        "primary_error_passed": accepted,
        "extrapolation_error_passed": accepted,
        "initialization_validation_passed": accepted,
    }
    for name, value in (
        ("experiment.json", experiment),
        ("manifest.json", manifest),
        ("summary.json", summary),
    ):
        (root / name).write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    files = {}
    for path in sorted(root.iterdir()):
        payload = path.read_bytes()
        files[path.name] = {
            "size_bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    (root / "m1-integrity.json").write_text(
        json.dumps(
            {
                "schema": "longctx-m1-init.v2",
                "identity": INITIALIZATION_ID,
                "files": files,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return root.resolve()


def _context(total_kv_tokens: int, rates: tuple[float, ...]) -> dict[str, object]:
    return {
        "context_id": f"context-{total_kv_tokens // 1024}k",
        "total_kv_tokens": total_kv_tokens,
        "output_tokens": 128,
        "loads": tuple(
            {
                "load_id": ("low", "medium", "high")[index],
                "offered_requests_per_second": rate,
            }
            for index, rate in enumerate(rates)
        ),
        "slo": {
            "ttft_ms": float(total_kv_tokens // 4),
            "tpot_ms": 100.0,
            "e2e_ms": float(total_kv_tokens // 4 + 20_000),
            "max_error_rate_ppm": 0,
        },
    }


def _valid_data(tmp_path: Path) -> dict[str, object]:
    model_dir, model_lock, runtime_lock = _identity_files(tmp_path)
    artifact_root = (tmp_path / "longctx-v5-artifacts").resolve()
    initialization_root = _seal_initialization(
        artifact_root,
        model_dir,
        model_lock,
        runtime_lock,
    )
    return {
        "project_line": "longctx-v5",
        "milestone": "M1",
        "experiment_kind": "capacity-sweep",
        "evidence_role": "formal",
        "model": {"local_path": str(model_dir), "lock_path": str(model_lock)},
        "runtime": {"lock_path": str(runtime_lock)},
        "artifacts": {"root": str(artifact_root)},
        "gpu": {"device_ids": [0], "count": 1},
        "initialization_artifact": {
            "experiment_id": INITIALIZATION_ID,
            "root": str(initialization_root),
        },
        "server_profile": {
            "name": "production-default",
            "expected_gpu_memory_utilization_ppm": 900_000,
            "expected_max_model_len": 32_768,
            "expected_max_num_seqs": 256,
            "expected_max_num_batched_tokens": 2_048,
            "inherit_upstream_defaults": True,
        },
        "contexts": (
            _context(8_192, (0.25, 0.50, 0.75)),
            _context(16_384, (0.10, 0.25, 0.50)),
            _context(32_768, (0.05, 0.10, 0.20)),
        ),
        "protocol": {
            "repeats": 3,
            "measurement_seconds": 180,
            "minimum_measured_requests": 100,
            "warmup_requests": 8,
            "warmup_seed": 20_260_817,
            "measurement_seed": 20_260_917,
            "warmup_prompt_index_offset": 1_000_000,
            "client_max_concurrency": 512,
            "request_timeout_seconds": 600.0,
            "burstiness": 1.0,
            "ignore_eos": True,
        },
        "knee_policy": {
            "repeat_aggregation": "median",
            "minimum_valid_repeats": 3,
            "throughput_plateau_max_gain_ppm": 50_000,
            "queue_growth_min_requests_per_second": 0.05,
            "minimum_peak_waiting_requests": 1,
            "minimum_slo_attainment_ppm": 950_000,
            "minimum_achieved_to_offered_ppm": 950_000,
            "minimum_completion_ppm": 950_000,
            "max_p99_dispatch_delay_ms": 50.0,
            "maximum_preemptions_for_stable": 0,
            "maximum_timeouts_for_stable": 0,
            "require_zero_oom_events": True,
            "minimum_joint_signal_repeats": 3,
            "required_prefix_cache_hits_delta": 0,
            "overload_rule": ("throughput-plateau-and-positive-queue-growth-and-slo-failure"),
            "selection_rule": "highest-stable-load-before-first-joint-overload",
            "no_overload_result": "right-censored-above-highest-load",
            "below_lowest_result": "left-censored-below-lowest-load",
        },
    }


def test_formal_matrix_converts_one_point_without_server_overrides(tmp_path: Path) -> None:
    config = LongContextM1CapacityConfig.model_validate(_valid_data(tmp_path))
    context = config.contexts[0]
    load = context.loads[-1]

    tuning = config.to_tuning_config(context, load, repeat_index=1)

    assert config.formal_acceptance_eligible is True
    assert context.input_tokens == 8_064
    assert tuning.model_revision == REVISION
    assert tuning.workload.sample_size == 136
    assert tuning.workload.fixed_input_tokens == 8_064
    assert tuning.workload.fixed_output_tokens == 128
    assert tuning.workload.request_rate == 0.75
    assert tuning.workload.seed == 20_260_917
    assert config.server_profile.expected_max_num_seqs == 256
    assert config.server_profile.expected_max_num_batched_tokens == 2_048
    assert tuning.slo.ttft_ms == context.slo.ttft_ms
    assert tuning.constraints.max_error_rate == 0.0
    assert tuning.telemetry.enabled is True
    assert tuning.adaptive_prefill.enabled is False
    assert tuning.vllm_args == {}


def test_protocol_uses_disjoint_seeds_and_warmup_prompt_indices(tmp_path: Path) -> None:
    config = LongContextM1CapacityConfig.model_validate(_valid_data(tmp_path))

    assert config.protocol.warmup_seed_for_repeat(2) == 20_260_817
    assert config.protocol.measurement_seed_for_repeat(2) == 20_260_917
    assert config.protocol.warmup_prompt_start_for_repeat(2) == 1_000_000
    with pytest.raises(ValueError, match="repeat_index"):
        config.protocol.measurement_seed_for_repeat(3)


def test_loader_normalizes_context_and_load_lists_and_rejects_duplicate_keys(
    tmp_path: Path,
) -> None:
    data = _valid_data(tmp_path)
    contexts = [dict(context) for context in data["contexts"]]
    for context in contexts:
        context["loads"] = list(context["loads"])
    data["contexts"] = contexts
    path = tmp_path / "capacity.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    config = load_longctx_m1_capacity_config(path)

    assert isinstance(config.contexts, tuple)
    assert isinstance(config.contexts[0].loads, tuple)

    duplicate = tmp_path / "duplicate.yaml"
    duplicate.write_text("protocol:\n  repeats: 3\n  repeats: 2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate key"):
        load_longctx_m1_capacity_config(duplicate)


@pytest.mark.parametrize(
    "mutation,error",
    [
        ("contexts", "8K, 16K, and 32K"),
        ("loads", "three offered loads"),
        ("repeats", "exactly three repeats"),
        ("duration", "at least 180"),
        ("requests", "at least 100"),
        ("concurrency", "strict open-loop"),
    ],
)
def test_formal_matrix_gates_are_fail_closed(
    tmp_path: Path,
    mutation: str,
    error: str,
) -> None:
    data = _valid_data(tmp_path)
    protocol = copy.deepcopy(data["protocol"])
    assert isinstance(protocol, dict)
    if mutation == "contexts":
        data["contexts"] = data["contexts"][:-1]
    elif mutation == "loads":
        contexts = [copy.deepcopy(context) for context in data["contexts"]]
        contexts[0]["loads"] = contexts[0]["loads"][:-1]
        data["contexts"] = tuple(contexts)
    elif mutation == "repeats":
        protocol["repeats"] = 2
        knee = copy.deepcopy(data["knee_policy"])
        assert isinstance(knee, dict)
        knee["minimum_valid_repeats"] = 2
        knee["minimum_joint_signal_repeats"] = 2
        data["knee_policy"] = knee
    elif mutation == "duration":
        protocol["measurement_seconds"] = 179
    elif mutation == "requests":
        protocol["minimum_measured_requests"] = 99
    else:
        protocol["client_max_concurrency"] = 134
    data["protocol"] = protocol

    with pytest.raises(ValidationError, match=error):
        LongContextM1CapacityConfig.model_validate(data)


def test_smoke_can_be_small_but_cannot_claim_formal_acceptance(tmp_path: Path) -> None:
    data = _valid_data(tmp_path)
    data["evidence_role"] = "smoke"
    data["contexts"] = (_context(8_192, (0.10,)),)
    protocol = copy.deepcopy(data["protocol"])
    assert isinstance(protocol, dict)
    protocol.update(
        repeats=1,
        measurement_seconds=30,
        minimum_measured_requests=10,
        warmup_requests=1,
        client_max_concurrency=10,
    )
    data["protocol"] = protocol
    knee = copy.deepcopy(data["knee_policy"])
    assert isinstance(knee, dict)
    knee["minimum_valid_repeats"] = 1
    knee["minimum_joint_signal_repeats"] = 1
    data["knee_policy"] = knee

    config = LongContextM1CapacityConfig.model_validate(data)

    assert config.formal_acceptance_eligible is False
    with pytest.raises(ValueError, match="cannot satisfy formal M1 acceptance"):
        config.require_formal_acceptance()


def test_initialization_binding_requires_sealed_success_and_fixed_identity(
    tmp_path: Path,
) -> None:
    unsealed = _valid_data(tmp_path / "unsealed")
    binding = unsealed["initialization_artifact"]
    assert isinstance(binding, dict)
    (Path(binding["root"]) / "m1-integrity.json").unlink()
    with pytest.raises(ValidationError, match="integrity seal"):
        LongContextM1CapacityConfig.model_validate(unsealed)

    failed_root = tmp_path / "failed"
    model_dir, model_lock, runtime_lock = _identity_files(failed_root)
    artifact_root = (failed_root / "longctx-v5-artifacts").resolve()
    initialization = _seal_initialization(
        artifact_root,
        model_dir,
        model_lock,
        runtime_lock,
        accepted=False,
    )
    failed = _valid_data(tmp_path / "valid-template")
    failed.update(
        model={"local_path": str(model_dir), "lock_path": str(model_lock)},
        runtime={"lock_path": str(runtime_lock)},
        artifacts={"root": str(artifact_root)},
        initialization_artifact={
            "experiment_id": INITIALIZATION_ID,
            "root": str(initialization),
        },
    )
    with pytest.raises(ValidationError, match="is not accepted"):
        LongContextM1CapacityConfig.model_validate(failed)

    mismatch = _valid_data(tmp_path / "mismatch")
    other_model = (tmp_path / "mismatch" / "other-model").resolve()
    other_model.mkdir()
    model = copy.deepcopy(mismatch["model"])
    assert isinstance(model, dict)
    model["local_path"] = str(other_model)
    mismatch["model"] = model
    with pytest.raises(ValidationError, match="model path differs"):
        LongContextM1CapacityConfig.model_validate(mismatch)


@pytest.mark.parametrize(
    "field,value,error",
    [
        ("project_line", "slotune", "longctx-v5"),
        ("milestone", "M2", "M1"),
        ("experiment_kind", "planner-initialization-validation", "capacity-sweep"),
    ],
)
def test_v5_m1_capacity_identity_literals_are_fixed(
    tmp_path: Path,
    field: str,
    value: str,
    error: str,
) -> None:
    data = _valid_data(tmp_path)
    data[field] = value
    with pytest.raises(ValidationError, match=error):
        LongContextM1CapacityConfig.model_validate(data)


@pytest.mark.parametrize("field", ["search_space", "scheduler", "kv_cache_dtype"])
def test_legacy_scheduler_and_m2_fields_are_rejected(tmp_path: Path, field: str) -> None:
    data = _valid_data(tmp_path)
    data[field] = {"enabled": True}
    with pytest.raises(ValidationError, match="extra_forbidden"):
        LongContextM1CapacityConfig.model_validate(data)


def test_slo_knee_and_open_loop_parameters_are_strictly_validated(tmp_path: Path) -> None:
    invalid_slo = _valid_data(tmp_path / "slo")
    contexts = [copy.deepcopy(context) for context in invalid_slo["contexts"]]
    contexts[0]["slo"]["ttft_ms"] = float("inf")
    invalid_slo["contexts"] = tuple(contexts)
    with pytest.raises(ValidationError, match="finite"):
        LongContextM1CapacityConfig.model_validate(invalid_slo)

    invalid_knee = _valid_data(tmp_path / "knee")
    knee = copy.deepcopy(invalid_knee["knee_policy"])
    assert isinstance(knee, dict)
    knee["required_prefix_cache_hits_delta"] = 1
    invalid_knee["knee_policy"] = knee
    with pytest.raises(ValidationError, match="Input should be 0"):
        LongContextM1CapacityConfig.model_validate(invalid_knee)

    invalid_joint = _valid_data(tmp_path / "joint")
    knee = copy.deepcopy(invalid_joint["knee_policy"])
    assert isinstance(knee, dict)
    knee["minimum_joint_signal_repeats"] = 2
    invalid_joint["knee_policy"] = knee
    with pytest.raises(ValidationError, match="every valid repeat"):
        LongContextM1CapacityConfig.model_validate(invalid_joint)

    invalid_oom = _valid_data(tmp_path / "oom")
    knee = copy.deepcopy(invalid_oom["knee_policy"])
    assert isinstance(knee, dict)
    knee["require_zero_oom_events"] = False
    invalid_oom["knee_policy"] = knee
    with pytest.raises(ValidationError, match="Input should be True"):
        LongContextM1CapacityConfig.model_validate(invalid_oom)

    invalid_seed = _valid_data(tmp_path / "seed")
    protocol = copy.deepcopy(invalid_seed["protocol"])
    assert isinstance(protocol, dict)
    protocol["warmup_seed"] = protocol["measurement_seed"]
    invalid_seed["protocol"] = protocol
    with pytest.raises(ValidationError, match="seeds must be disjoint"):
        LongContextM1CapacityConfig.model_validate(invalid_seed)


def test_conversion_rejects_unregistered_points_and_repeat_indices(tmp_path: Path) -> None:
    config = LongContextM1CapacityConfig.model_validate(_valid_data(tmp_path))
    foreign_load = M1CapacityLoad(load_id="foreign", offered_requests_per_second=0.6)
    foreign_context = M1CapacityContext(
        context_id="foreign-8k",
        total_kv_tokens=8_192,
        output_tokens=128,
        loads=(foreign_load,),
        slo=config.contexts[0].slo,
    )

    with pytest.raises(ValueError, match="context is not"):
        config.to_tuning_config(foreign_context, foreign_load, 0)
    with pytest.raises(ValueError, match="load is not"):
        config.to_tuning_config(config.contexts[0], foreign_load, 0)
    with pytest.raises(ValueError, match="repeat_index"):
        config.to_tuning_config(config.contexts[0], config.contexts[0].loads[0], 3)
