"""Integration-style tests for M1 initialization probes and Planner validation."""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from pathlib import Path

import pytest
import yaml

from vllm_tuner.longctx.kv_capacity_planner import (
    ContextBin,
    ContextDistributionSpec,
    SafetyPolicy,
)
from vllm_tuner.longctx.m0_config import (
    LongContextM0ArtifactConfig,
    LongContextM0GPUConfig,
    LongContextM0ModelConfig,
    LongContextM0RuntimeConfig,
)
from vllm_tuner.longctx.m1_config import LongContextM1Config, M1InitializationProbe
from vllm_tuner.longctx.m1_runner import (
    M1_INTEGRITY_FILE,
    LongContextM1Runner,
    _validate_tree,
)
from vllm_tuner.longctx.runtime_identity import (
    RuntimeIdentityFacts,
    RuntimeWheelRecordFact,
)
from vllm_tuner.runtime.server import CleanupStatus

TOTAL_MEMORY = 32607 * (1 << 20)
NON_KV_MEMORY = 17_359_500_000
BLOCK_BYTES = 917_504
REVISION = "a" * 40
EXECUTION_ENVIRONMENT = {
    "OMP_NUM_THREADS": "8",
    "TOKENIZERS_PARALLELISM": "false",
    "VLLM_NO_USAGE_STATS": "1",
}


def _git_repository(path: Path) -> Path:
    path.mkdir()
    (path / "tracked.txt").write_text("clean\n", encoding="utf-8")
    for command in (
        ["git", "init", "-q"],
        ["git", "config", "user.email", "tests@example.invalid"],
        ["git", "config", "user.name", "Tests"],
        ["git", "add", "tracked.txt"],
        ["git", "commit", "-q", "-m", "source"],
    ):
        subprocess.run(command, cwd=path, check=True)
    return path


def _model_identity(tmp_path: Path) -> tuple[Path, Path]:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    config = {
        "architectures": ["Qwen2ForCausalLM"],
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
        "model.safetensors": b"fake weights\n",
    }
    for name, payload in payloads.items():
        (model_dir / name).write_bytes(payload)
    lock_path = tmp_path / "model.lock.yaml"
    lock_path.write_text(
        yaml.safe_dump(
            {
                "model": {
                    "repository_id": "Qwen/Test-7B",
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
    return model_dir, lock_path


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


def _config(tmp_path: Path) -> tuple[LongContextM1Config, Path, RuntimeIdentityFacts]:
    model_dir, model_lock = _model_identity(tmp_path)
    runtime_lock_path = tmp_path / "runtime.lock.yaml"
    runtime_lock_path.write_text("vllm: {}\n", encoding="utf-8")
    probes = (
        M1InitializationProbe(
            probe_id="cal-75",
            role="calibration",
            gpu_memory_utilization_ppm=750_000,
            max_model_len=32768,
            max_num_seqs=512,
            repeats=2,
        ),
        M1InitializationProbe(
            probe_id="cal-80",
            role="calibration",
            gpu_memory_utilization_ppm=800_000,
            max_model_len=32768,
            max_num_seqs=512,
            repeats=2,
        ),
        M1InitializationProbe(
            probe_id="cal-85",
            role="calibration",
            gpu_memory_utilization_ppm=850_000,
            max_model_len=32768,
            max_num_seqs=512,
            repeats=2,
        ),
        M1InitializationProbe(
            probe_id="heldout-32k",
            role="validation",
            gpu_memory_utilization_ppm=900_000,
            max_model_len=32768,
            max_num_seqs=512,
            repeats=1,
        ),
        M1InitializationProbe(
            probe_id="extrapolate-8k",
            role="extrapolation",
            gpu_memory_utilization_ppm=900_000,
            max_model_len=8192,
            max_num_seqs=512,
            repeats=1,
        ),
        M1InitializationProbe(
            probe_id="extrapolate-16k",
            role="extrapolation",
            gpu_memory_utilization_ppm=900_000,
            max_model_len=16384,
            max_num_seqs=512,
            repeats=1,
        ),
    )
    config = LongContextM1Config(
        project_line="longctx-v5",
        milestone="M1",
        experiment_kind="planner-initialization-validation",
        model=LongContextM0ModelConfig(local_path=model_dir, lock_path=model_lock),
        runtime=LongContextM0RuntimeConfig(lock_path=runtime_lock_path),
        artifacts=LongContextM0ArtifactConfig(root=tmp_path / "artifacts"),
        gpu=LongContextM0GPUConfig(device_ids=(0,), count=1),
        probes=probes,
        deployment_distribution=ContextDistributionSpec(
            bins=(
                ContextBin(
                    name="short",
                    weight_ppm=500_000,
                    prompt_tokens=8064,
                    reserved_output_tokens=128,
                ),
                ContextBin(
                    name="medium",
                    weight_ppm=300_000,
                    prompt_tokens=16256,
                    reserved_output_tokens=128,
                ),
                ContextBin(
                    name="long",
                    weight_ppm=200_000,
                    prompt_tokens=32640,
                    reserved_output_tokens=128,
                ),
            ),
            confidence_ppm=990_000,
            iid_assumption=True,
            assume_no_prefix_reuse=True,
        ),
        safety=SafetyPolicy(
            fixed_operational_reserve_bytes=256 * (1 << 20),
            kv_reserve_basis_points=500,
            calibration_residual_upper_bytes=0,
            source="test",
        ),
    )
    repository = _git_repository(tmp_path / "repository")
    runtime = _runtime_facts(runtime_lock_path, tmp_path / "fake-vllm")
    return config, repository, runtime


class FakeServer:
    calls: list[str] = []
    current: "FakeServer | None" = None
    cleanup_clean = True

    def __init__(self, config, *, trial_dir, **kwargs) -> None:
        del kwargs
        self.config = config
        self.trial_dir = Path(trial_dir)
        self.log_path = self.trial_dir / "server.log"
        self.command_path = self.trial_dir / "server-command.json"
        self.base_url = f"http://fake/{self.trial_dir.name}"
        self.blocks = 0

    async def start(self, params):
        del params
        type(self).calls.append(self.trial_dir.name)
        type(self).current = self
        self.trial_dir.mkdir(parents=True, exist_ok=True)
        utilization = float(self.config.vllm_args["gpu-memory-utilization"])
        max_model_len = int(self.config.vllm_args["max-model-len"])
        requested = math.ceil(TOTAL_MEMORY * utilization)
        self.blocks = (requested - NON_KV_MEMORY) // BLOCK_BYTES
        tokens = self.blocks * 16
        concurrency = self.blocks / math.ceil(max_model_len / 16)
        self.command_path.write_text(
            json.dumps({"argv": ["fake-vllm"], "environment": {}}),
            encoding="utf-8",
        )
        self.log_path.write_text(
            "Resolved architecture: Qwen2ForCausalLM\n"
            f"Using max model len {max_model_len}\n"
            "Using FLASH_ATTN attention backend\n"
            "config: dtype=torch.bfloat16, kv_cache_dtype=auto, "
            "enable_prefix_caching=True, enable_chunked_prefill=True\n"
            f"Available KV cache memory: {self.blocks * BLOCK_BYTES / (1 << 30):.2f} GiB\n"
            f"GPU KV cache size: {tokens:,} tokens\n"
            f"Maximum concurrency for {max_model_len:,} tokens per request: {concurrency:.2f}x\n",
            encoding="utf-8",
        )
        return self

    async def wait_ready(self) -> bool:
        return True

    async def stop(self) -> CleanupStatus:
        clean = type(self).cleanup_clean
        return CleanupStatus(
            attempted=True,
            clean=clean,
            process_group_empty=clean,
            port_available=clean,
            gpu_clean=clean,
            errors=[] if clean else ["injected cleanup failure"],
            checked_monotonic_ns=1,
        )


async def _metrics_fetcher(base_url: str) -> str:
    del base_url
    server = FakeServer.current
    assert server is not None
    utilization = server.config.vllm_args["gpu-memory-utilization"]
    return (
        "# TYPE vllm:cache_config_info gauge\n"
        "vllm:cache_config_info{"
        'block_size="16",cache_dtype="auto",calculate_kv_scales="False",'
        f'gpu_memory_utilization="{utilization}",num_gpu_blocks="{server.blocks}",'
        'kv_cache_memory_bytes="None",num_gpu_blocks_override="None",'
        'enable_prefix_caching="True",is_attention_free="False",'
        'sliding_window="None",engine="0"} 1.0\n'
    )


def _runner(
    config: LongContextM1Config,
    repository: Path,
    runtime: RuntimeIdentityFacts,
    experiment_id: str,
    *,
    resume: bool = False,
) -> LongContextM1Runner:
    return LongContextM1Runner(
        config,
        experiment_id,
        repository=repository,
        resume=resume,
        server_factory=FakeServer,
        metrics_fetcher=_metrics_fetcher,
        gpu_memory_reader=lambda: (TOTAL_MEMORY, TOTAL_MEMORY),
        runtime_facts=runtime,
        execution_environment=EXECUTION_ENVIRONMENT,
    )


@pytest.mark.asyncio
async def test_m1_runner_calibrates_validates_seals_and_resumes(tmp_path: Path) -> None:
    config, repository, runtime = _config(tmp_path)
    FakeServer.calls = []
    FakeServer.cleanup_clean = True
    summary = await _runner(config, repository, runtime, "m1-init").run()

    root = config.artifacts.root / "m1-init"
    assert summary["primary_error_passed"] is True
    assert len(summary["probe_runs"]) == 9
    assert len(summary["validations"]) == 3
    assert summary["extrapolation_error_passed"] is True
    assert summary["initialization_validation_passed"] is True
    assert {row["evaluation_role"] for row in summary["validations"]} == {
        "validation",
        "extrapolation",
    }
    assert all(abs(row["block_error_percent"]) < 0.1 for row in summary["validations"])
    assert summary["deployment_plan"]["capacity"]["safe_usable_num_blocks"] > 0
    assert FakeServer.calls == [
        "cal-75-r0",
        "cal-75-r1",
        "cal-80-r0",
        "cal-80-r1",
        "cal-85-r0",
        "cal-85-r1",
        "heldout-32k-r0",
        "extrapolate-8k-r0",
        "extrapolate-16k-r0",
    ]
    _validate_tree(root, M1_INTEGRITY_FILE, "longctx-m1-init.v1", "m1-init")

    resumed = await _runner(config, repository, runtime, "m1-init", resume=True).run()
    assert resumed["resume_replayed"] is True
    assert len(FakeServer.calls) == 9


@pytest.mark.asyncio
async def test_m1_resume_rejects_tampered_root(tmp_path: Path) -> None:
    config, repository, runtime = _config(tmp_path)
    FakeServer.cleanup_clean = True
    await _runner(config, repository, runtime, "m1-tamper").run()
    root = config.artifacts.root / "m1-tamper"
    (root / "summary.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="checksum mismatch"):
        await _runner(config, repository, runtime, "m1-tamper", resume=True).run()


@pytest.mark.asyncio
async def test_m1_probe_fails_closed_on_unclean_cleanup(tmp_path: Path) -> None:
    config, repository, runtime = _config(tmp_path)
    FakeServer.calls = []
    FakeServer.cleanup_clean = False
    with pytest.raises(RuntimeError, match="cleanup failed"):
        await _runner(config, repository, runtime, "m1-cleanup").run()

    status = json.loads(
        (config.artifacts.root / "m1-cleanup" / "probes" / "cal-75-r0" / "status.json").read_text(
            encoding="utf-8"
        )
    )
    assert status["status"] == "FAILED"
    failed_dir = config.artifacts.root / "m1-cleanup" / "probes" / "cal-75-r0"
    _validate_tree(
        failed_dir,
        "probe-integrity.json",
        "longctx-m1-probe.v1",
        "cal-75-r0",
    )

    FakeServer.cleanup_clean = True
    summary = await _runner(config, repository, runtime, "m1-cleanup", resume=True).run()
    assert summary["primary_error_passed"] is True
    assert (
        config.artifacts.root
        / "m1-cleanup"
        / "probes"
        / "cal-75-r0-attempt1"
        / "probe-integrity.json"
    ).is_file()
    assert len(FakeServer.calls) == 10
