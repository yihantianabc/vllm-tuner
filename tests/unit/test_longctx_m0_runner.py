"""Integration-style unit tests for the isolated long-context v5 M0 runner."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest
import yaml

from vllm_tuner.experiment.models import TrialResult, TrialStatus, trial_provenance
from vllm_tuner.longctx.integrity import M0_INTEGRITY_FILE, validate_m0_artifacts
from vllm_tuner.longctx.m0_config import LongContextM0Config
from vllm_tuner.longctx.m0_runner import LongContextM0Runner, load_m0_status
from vllm_tuner.longctx.runtime_identity import (
    RuntimeIdentityFacts,
    RuntimeWheelRecordFact,
)

REVISION = "a" * 40
EXECUTION_ENVIRONMENT = {
    "OMP_NUM_THREADS": "8",
    "TOKENIZERS_PARALLELISM": "false",
    "VLLM_NO_USAGE_STATS": "1",
}


class FakeTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[str]:
        del add_special_tokens
        return text.split()

    def decode(self, tokens: list[str]) -> str:
        return " ".join(str(token) for token in tokens)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _clean_cleanup() -> dict[str, object]:
    return {
        "attempted": True,
        "clean": True,
        "process_group_empty": True,
        "port_available": True,
        "gpu_clean": True,
        "errors": [],
    }


def _startup_log(attention_backend: str = "FLASH_ATTN") -> str:
    return (
        "version 0.16.0\n"
        f"Using {attention_backend} attention backend\n"
        "Using max model len 32768\n"
        "config: dtype=torch.bfloat16, kv_cache_dtype=auto, "
        "enable_prefix_caching=True, enable_chunked_prefill=True, seed=0\n"
    )


class FakeTrialController:
    calls: list[str] = []
    attention_backend = "FLASH_ATTN"

    def __init__(self, config, trace, artifacts, **kwargs) -> None:
        del config, kwargs
        self.trace = trace
        self.artifacts = artifacts

    async def run_trial(self, params, trial_id, method) -> TrialResult:
        type(self).calls.append(trial_id)
        base = Path("trials") / trial_id
        requests = [
            {
                "request_id": entry.request_id,
                "status": "success",
                "input_tokens": entry.input_tokens,
                "output_tokens": entry.output_tokens,
            }
            for entry in self.trace.entries
        ]
        aggregate = {
            "num_requests": len(requests),
            "completed": len(requests),
            "failed": 0,
            "total_input_tokens": sum(row["input_tokens"] for row in requests),
            "total_output_tokens": sum(row["output_tokens"] for row in requests),
            "duration": 10.0,
        }
        cleanup = _clean_cleanup()
        self.artifacts.write_json(
            base / "server-command.json",
            {"argv": ["python", "-m", "vllm.entrypoints.openai.api_server"]},
        )
        self.artifacts.write_json(base / "params.json", params)
        self.artifacts.write_json(
            base / "status.json",
            {"status": "COMPLETE", "history": [{"current": "COMPLETE"}]},
        )
        self.artifacts.write_jsonl(base / "request-results.jsonl", requests)
        self.artifacts.write_json(
            base / "benchmark-raw.json",
            {
                "backend": "fake-fixed-trace",
                "request_results": requests,
                "aggregate": aggregate,
            },
        )
        self.artifacts.write_jsonl(base / "prometheus.jsonl", [{"available": True}])
        self.artifacts.write_jsonl(base / "nvml.jsonl", [{"available": True}])
        self.artifacts.write_text(base / "server.log", _startup_log(self.attention_backend))
        self.artifacts.write_json(base / "cleanup.json", cleanup)
        return TrialResult(
            trial_id=trial_id,
            **trial_provenance(trial_id, method),
            status=TrialStatus.COMPLETE,
            params=params,
            measurement_seconds=10.0,
            client={**aggregate, "goodput_requests_per_sec": len(requests) / 10.0},
            constraints={"feasible": True, "violations": []},
            cleanup_status=cleanup,
        )


def _git_repository(path: Path) -> Path:
    path.mkdir()
    (path / "tracked.txt").write_text("clean source\n", encoding="utf-8")
    commands = (
        ["git", "init", "-q"],
        ["git", "config", "user.email", "tests@example.invalid"],
        ["git", "config", "user.name", "SLOTune Tests"],
        ["git", "add", "tracked.txt"],
        ["git", "commit", "-q", "-m", "test source"],
    )
    for command in commands:
        subprocess.run(command, cwd=path, check=True)
    return path


def _model_and_lock(path: Path, parameter_count: int) -> tuple[Path, Path]:
    path.mkdir()
    config_payload = json.dumps(
        {
            "architectures": ["FakeDenseForCausalLM"],
            "model_type": "fake_dense",
            "num_hidden_layers": 28,
            "num_attention_heads": 28,
            "num_key_value_heads": 4,
            "hidden_size": 3584,
            "max_position_embeddings": 32768,
            "torch_dtype": "bfloat16",
        }
    ).encode()
    payloads = {
        "config.json": config_payload,
        "tokenizer.json": b"{}\n",
        "model.safetensors": b"fake weights\n",
    }
    for name, payload in payloads.items():
        (path / name).write_bytes(payload)
    (path / ".slotune-model-revision").write_text(REVISION + "\n", encoding="utf-8")
    lock_path = path.parent / "model.lock.yaml"
    lock_path.write_text(
        yaml.safe_dump(
            {
                "model": {
                    "repository_id": "Qwen/Test-Model",
                    "revision": REVISION,
                    "parameter_count": parameter_count,
                    "files": {
                        name: {"size_bytes": len(payload), "sha256": _sha256(payload)}
                        for name, payload in payloads.items()
                    },
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return path, lock_path


def _runtime_facts(lock_path: Path, package_dir: Path) -> RuntimeIdentityFacts:
    package_dir.mkdir()
    digest = "d" * 64
    return RuntimeIdentityFacts(
        lock_path=lock_path,
        upstream_commit="8" * 40,
        expected_version="0.16.0",
        actual_version="0.16.0",
        package_dir=package_dir,
        wheel_record=RuntimeWheelRecordFact(
            expected_sha256=digest,
            actual_sha256=digest,
            matches=True,
        ),
        environment={},
        source_files=[],
        legacy_patched_scheduler=False,
        matches_lock=True,
        issues=[],
    )


def _config(
    tmp_path: Path,
    *,
    evidence_role: str = "formal",
    model_tier: str = "primary-7b-8b",
    parameter_count: int = 7_615_616_512,
) -> tuple[LongContextM0Config, Path, RuntimeIdentityFacts]:
    model_dir, model_lock = _model_and_lock(tmp_path / "model", parameter_count)
    runtime_lock = tmp_path / "runtime.lock.yaml"
    runtime_lock.write_text("vllm: {}\n", encoding="utf-8")
    repository = _git_repository(tmp_path / "repository")
    config = LongContextM0Config.model_validate(
        {
            "project_line": "longctx-v5",
            "milestone": "M0",
            "profile": "production-default",
            "evidence_role": evidence_role,
            "model_tier": model_tier,
            "model": {"local_path": str(model_dir), "lock_path": str(model_lock)},
            "artifacts": {"root": str(tmp_path / "longctx-v5-artifacts")},
            "runtime": {"lock_path": str(runtime_lock)},
            "gpu": {"device_ids": [0], "count": 1},
            "workload": {
                "measured_requests": 100,
                "warmup_requests": 1,
                "fixed_input_tokens": 8,
                "fixed_output_tokens": 4,
                "request_rate": 10.0,
                "max_concurrency": 8,
                "request_timeout_seconds": 10.0,
                "seed": 2026,
                "ignore_eos": True,
            },
            "vllm_args": {},
        }
    )
    return config, repository, _runtime_facts(runtime_lock, tmp_path / "fake-vllm")


def _runner(
    config: LongContextM0Config,
    repository: Path,
    runtime: RuntimeIdentityFacts,
    experiment_id: str,
    *,
    resume: bool = False,
    controller_factory=FakeTrialController,
) -> LongContextM0Runner:
    return LongContextM0Runner(
        config,
        experiment_id,
        repository=repository,
        resume=resume,
        tokenizer=FakeTokenizer(),
        controller_factory=controller_factory,
        runtime_facts=runtime,
        execution_environment=EXECUTION_ENVIRONMENT,
        capture_environment=False,
    )


@pytest.mark.parametrize("experiment_id", ["", ".", "..", "bad/name", "bad\\name", " bad"])
def test_m0_runner_rejects_unsafe_experiment_id(
    tmp_path: Path,
    experiment_id: str,
) -> None:
    config, repository, runtime = _config(tmp_path)
    with pytest.raises(ValueError, match="portable directory name"):
        _runner(config, repository, runtime, experiment_id)


@pytest.mark.asyncio
async def test_m0_runner_seals_formal_baseline_and_replays_sealed_resume(
    tmp_path: Path,
) -> None:
    config, repository, runtime = _config(tmp_path)
    FakeTrialController.calls = []
    summary = await _runner(config, repository, runtime, "m0-production-canary").run()

    root = config.artifacts.root / "m0-production-canary"
    assert summary["acceptance"]["passed"] is True
    assert summary["acceptance"]["execution_passed"] is True
    assert summary["legacy_results_used"] is False
    assert summary["trial"]["client"]["num_requests"] == 100
    assert FakeTrialController.calls == ["production-default-attempt-0001"]
    validate_m0_artifacts(root)

    resumed = await _runner(
        config,
        repository,
        runtime,
        "m0-production-canary",
        resume=True,
    ).run()
    assert resumed["resume_replayed"] is True
    assert FakeTrialController.calls == ["production-default-attempt-0001"]
    assert (
        load_m0_status(config.artifacts.root, "m0-production-canary")["acceptance"]["passed"]
        is True
    )


@pytest.mark.asyncio
async def test_m0_runner_recovers_unsealed_root_from_cached_trial(
    tmp_path: Path,
) -> None:
    config, repository, runtime = _config(tmp_path)
    FakeTrialController.calls = []
    await _runner(config, repository, runtime, "m0-resume-canary").run()
    root = config.artifacts.root / "m0-resume-canary"
    (root / M0_INTEGRITY_FILE).unlink()

    summary = await _runner(
        config,
        repository,
        runtime,
        "m0-resume-canary",
        resume=True,
    ).run()

    assert summary["resume"]["trial_replayed"] is True
    assert any(
        "summary but no integrity seal" in warning for warning in summary["resume"]["warnings"]
    )
    assert FakeTrialController.calls == ["production-default-attempt-0001"]
    validate_m0_artifacts(root)


@pytest.mark.asyncio
async def test_m0_runner_resume_rejects_tampered_sealed_root(tmp_path: Path) -> None:
    config, repository, runtime = _config(tmp_path)
    await _runner(config, repository, runtime, "m0-tamper-canary").run()
    root = config.artifacts.root / "m0-tamper-canary"
    (root / "m0-summary.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="checksum mismatch"):
        await _runner(
            config,
            repository,
            runtime,
            "m0-tamper-canary",
            resume=True,
        ).run()


@pytest.mark.asyncio
async def test_smoke_execution_passes_without_formal_qualification(
    tmp_path: Path,
) -> None:
    config, repository, runtime = _config(
        tmp_path,
        evidence_role="smoke",
        model_tier="smoke",
        parameter_count=751_632_384,
    )
    summary = await _runner(config, repository, runtime, "m0-smoke-canary").run()

    assert summary["acceptance"]["execution_passed"] is True
    assert summary["acceptance"]["passed"] is False
    assert summary["acceptance"]["checks"]["evidence_role_is_formal"] is False


@pytest.mark.asyncio
async def test_m0_runner_rejects_concurrent_owner(tmp_path: Path) -> None:
    config, repository, runtime = _config(tmp_path)
    first = _runner(config, repository, runtime, "m0-concurrent-canary")
    second = _runner(config, repository, runtime, "m0-concurrent-canary")
    descriptor = first._acquire_run_lock()
    try:
        with pytest.raises(RuntimeError, match="already owned"):
            await second.run()
    finally:
        first._release_run_lock(descriptor)


@pytest.mark.asyncio
async def test_m0_runner_rejects_unlocked_execution_environment(tmp_path: Path) -> None:
    config, repository, runtime = _config(tmp_path)
    runner = LongContextM0Runner(
        config,
        "m0-env-canary",
        repository=repository,
        runtime_facts=runtime,
        execution_environment={
            **EXECUTION_ENVIRONMENT,
            "VLLM_ATTENTION_BACKEND": "FLASHINFER",
        },
    )

    with pytest.raises(ValueError, match="unlocked execution environment variable"):
        await runner.run()


class WrongProvenanceController(FakeTrialController):
    async def run_trial(self, params, trial_id, method) -> TrialResult:
        result = await super().run_trial(params, trial_id, method)
        result.method = "legacy"
        return result


@pytest.mark.asyncio
async def test_m0_runner_rejects_wrong_controller_provenance(tmp_path: Path) -> None:
    config, repository, runtime = _config(tmp_path)
    runner = _runner(
        config,
        repository,
        runtime,
        "m0-provenance-canary",
        controller_factory=WrongProvenanceController,
    )

    with pytest.raises(ValueError, match="provenance mismatch"):
        await runner.run()


class WrongBackendController(FakeTrialController):
    attention_backend = "FLASHINFER"


@pytest.mark.asyncio
async def test_startup_backend_mismatch_cannot_pass_execution(tmp_path: Path) -> None:
    config, repository, runtime = _config(tmp_path)
    summary = await _runner(
        config,
        repository,
        runtime,
        "m0-backend-canary",
        controller_factory=WrongBackendController,
    ).run()

    assert summary["acceptance"]["execution_passed"] is False
    assert summary["acceptance"]["checks"]["startup_profile_verified"] is False
