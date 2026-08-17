"""Tests for the frozen long-context v5 M0 configuration contract."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from vllm_tuner.longctx.m0_config import LongContextM0Config, load_longctx_m0_config

REVISION = "0123456789abcdef0123456789abcdef01234567"


def _write_model_lock(tmp_path: Path, parameter_count: int) -> tuple[Path, Path]:
    model_path = tmp_path / "model"
    model_path.mkdir(parents=True)
    lock_path = tmp_path / "model.lock.yaml"
    lock_path.write_text(
        yaml.safe_dump(
            {
                "model": {
                    "repository_id": "Qwen/Test-Model",
                    "revision": REVISION,
                    "parameter_count": parameter_count,
                    "files": {
                        "config.json": {"size_bytes": 2, "sha256": "a" * 64},
                        "tokenizer.json": {"size_bytes": 2, "sha256": "b" * 64},
                        "model.safetensors": {"size_bytes": 2, "sha256": "c" * 64},
                    },
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return model_path, lock_path


def _valid_config_data(
    tmp_path: Path,
    *,
    evidence_role: str = "formal",
    model_tier: str = "primary-7b-8b",
    parameter_count: int = 7_615_616_512,
    fallback_reason: str | None = None,
) -> dict[str, object]:
    model_path, model_lock = _write_model_lock(tmp_path, parameter_count)
    runtime_lock = tmp_path / "runtime.lock.yaml"
    runtime_lock.write_text("vllm: {}\n", encoding="utf-8")
    payload: dict[str, object] = {
        "project_line": "longctx-v5",
        "milestone": "M0",
        "profile": "production-default",
        "evidence_role": evidence_role,
        "model_tier": model_tier,
        "model": {"local_path": str(model_path), "lock_path": str(model_lock)},
        "artifacts": {"root": str(tmp_path / "longctx-v5-artifacts")},
        "runtime": {"lock_path": str(runtime_lock)},
        "gpu": {"device_ids": [0], "count": 1},
        "workload": {
            "measured_requests": 100,
            "warmup_requests": 5,
            "fixed_input_tokens": 4096,
            "fixed_output_tokens": 128,
            "request_rate": 2.0,
            "max_concurrency": 16,
            "request_timeout_seconds": 300.0,
            "seed": 2026,
            "burstiness": 1.0,
            "ignore_eos": True,
        },
        "vllm_args": {},
    }
    if fallback_reason is not None:
        payload["fallback_reason"] = fallback_reason
    return payload


def test_valid_formal_config_and_yaml_loader_use_model_lock_as_identity(
    tmp_path: Path,
) -> None:
    data = _valid_config_data(tmp_path)
    config_path = tmp_path / "m0.yaml"
    config_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    config = load_longctx_m0_config(config_path)
    identity = config.model.identity()

    assert config.evidence_role == "formal"
    assert config.model_tier == "primary-7b-8b"
    assert identity.repository_id == "Qwen/Test-Model"
    assert identity.revision == REVISION
    assert identity.parameter_count == 7_615_616_512
    assert config.vllm_args == {}


def test_config_rejects_fewer_than_100_measured_requests(tmp_path: Path) -> None:
    data = _valid_config_data(tmp_path)
    workload = copy.deepcopy(data["workload"])
    assert isinstance(workload, dict)
    workload["measured_requests"] = 99
    data["workload"] = workload

    with pytest.raises(ValidationError, match="greater than or equal to 100"):
        LongContextM0Config.model_validate(data)


def test_config_rejects_legacy_artifact_root(tmp_path: Path) -> None:
    data = _valid_config_data(tmp_path)
    data["artifacts"] = {"root": str(tmp_path / "slotune-results" / "m0")}

    with pytest.raises(ValidationError, match="Legacy slotune-results"):
        LongContextM0Config.model_validate(data)


@pytest.mark.parametrize(
    "vllm_args",
    [
        {"scheduler-cls": "legacy.Scheduler"},
        {"kv-cache-dtype": "fp8"},
        {"enable-prefix-caching": False},
        {"max-model-len": 8192},
    ],
)
def test_m0_always_rejects_nonempty_vllm_args(
    tmp_path: Path,
    vllm_args: dict[str, object],
) -> None:
    data = _valid_config_data(tmp_path)
    data["vllm_args"] = vllm_args

    with pytest.raises(ValidationError, match="requires empty vllm_args"):
        LongContextM0Config.model_validate(data)


def test_smoke_role_requires_smoke_tier(tmp_path: Path) -> None:
    valid = _valid_config_data(
        tmp_path / "valid",
        evidence_role="smoke",
        model_tier="smoke",
        parameter_count=751_632_384,
    )
    assert LongContextM0Config.model_validate(valid).model_tier == "smoke"

    invalid = _valid_config_data(
        tmp_path / "invalid",
        evidence_role="smoke",
        model_tier="primary-7b-8b",
        parameter_count=751_632_384,
    )
    with pytest.raises(ValidationError, match="model_tier=smoke"):
        LongContextM0Config.model_validate(invalid)


def test_primary_tier_requires_locked_7b_or_8b_parameter_count(tmp_path: Path) -> None:
    data = _valid_config_data(tmp_path, parameter_count=751_632_384)

    with pytest.raises(ValidationError, match="7B/8B parameters"):
        LongContextM0Config.model_validate(data)


def test_explicit_3b_fallback_requires_locked_3b_and_reason(tmp_path: Path) -> None:
    valid = _valid_config_data(
        tmp_path / "valid",
        model_tier="fallback-3b",
        parameter_count=3_090_000_000,
        fallback_reason="7B compatibility preflight failed with recorded evidence",
    )
    config = LongContextM0Config.model_validate(valid)
    assert config.model_tier == "fallback-3b"
    assert config.fallback_reason is not None

    missing_reason = _valid_config_data(
        tmp_path / "missing",
        model_tier="fallback-3b",
        parameter_count=3_090_000_000,
    )
    with pytest.raises(ValidationError, match="explicit fallback_reason"):
        LongContextM0Config.model_validate(missing_reason)


def test_config_rejects_relative_or_missing_identity_paths(tmp_path: Path) -> None:
    data = _valid_config_data(tmp_path)
    model = copy.deepcopy(data["model"])
    assert isinstance(model, dict)
    model["local_path"] = "relative/model"
    data["model"] = model

    with pytest.raises(ValidationError, match="absolute path"):
        LongContextM0Config.model_validate(data)


def test_config_rejects_non_finite_workload(tmp_path: Path) -> None:
    data = _valid_config_data(tmp_path)
    workload = copy.deepcopy(data["workload"])
    assert isinstance(workload, dict)
    workload["request_rate"] = float("inf")
    data["workload"] = workload

    with pytest.raises(ValidationError, match="must be finite"):
        LongContextM0Config.model_validate(data)


def test_yaml_loader_rejects_duplicate_keys_at_nested_depth(tmp_path: Path) -> None:
    config_path = tmp_path / "duplicate.yaml"
    config_path.write_text(
        "model:\n  local_path: /tmp/a\n  local_path: /tmp/b\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate key"):
        load_longctx_m0_config(config_path)


def test_config_rejects_unknown_legacy_search_fields(tmp_path: Path) -> None:
    data = _valid_config_data(tmp_path)
    data["search_space"] = {"max-num-seqs": [8, 16]}

    with pytest.raises(ValidationError, match="extra_forbidden"):
        LongContextM0Config.model_validate(data)


def test_to_tuning_config_is_one_upstream_default_trial(tmp_path: Path) -> None:
    config = LongContextM0Config.model_validate(_valid_config_data(tmp_path))

    tuning = config.to_tuning_config()

    assert tuning.model == str(config.model.local_path)
    assert tuning.model_revision == REVISION
    assert tuning.tokenizer == str(config.model.local_path)
    assert tuning.gpu.device_ids == [0]
    assert tuning.workload.sample_size == 100
    assert tuning.workload.warmup_requests == 5
    assert tuning.workload.fixed_input_tokens == 4096
    assert tuning.workload.fixed_output_tokens == 128
    assert tuning.workload.request_rate == 2.0
    assert tuning.workload.max_concurrency == 16
    assert tuning.workload.ignore_eos is True
    assert tuning.study.methods == ["default"]
    assert tuning.study.trial_budget == 1
    assert tuning.study.repeat_count == 1
    assert tuning.study.holdout_enabled is False
    assert tuning.adaptive_prefill.enabled is False
    assert tuning.adaptive_prefill.decision_log_enabled is False
    assert tuning.telemetry.enabled is True
    assert tuning.vllm_args == {}
