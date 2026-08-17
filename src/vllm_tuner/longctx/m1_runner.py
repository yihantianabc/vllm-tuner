"""M1 initialization probes, multi-run calibration, and Planner validation."""

from __future__ import annotations

import fcntl
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Awaitable, Callable, Literal, Mapping, Optional, cast

from pydantic import ConfigDict, Field

from vllm_tuner.experiment.artifacts import ArtifactStore
from vllm_tuner.experiment.manifest import (
    git_state,
    sha256_file,
    sha256_json,
    source_tree_sha256,
)
from vllm_tuner.experiment.models import utc_now_iso
from vllm_tuner.runtime.server import ManagedVLLMServer

from .capacity_evidence import (
    CapacityRuntimeEvidence,
    DeviceMemoryEvidence,
    build_capacity_runtime_evidence,
    fetch_metrics_text,
)
from .kv_capacity_planner import (
    ByteEstimate,
    CacheLayoutSpec,
    ContextBin,
    ContextDistributionSpec,
    DeviceBudgetSpec,
    KVDType,
    KVCapacityPlannerInput,
    MultiRunNonKVCalibration,
    NonKVPredictionSpec,
    SafetyPolicy,
    ServingLimits,
    StrictFrozenModel,
    calibrate_non_kv_from_runs,
    model_spec_from_hf_config,
    plan_kv_capacity,
    validate_kv_capacity_plan,
)
from .m0_runner import (
    ALLOWED_EXECUTION_ENVIRONMENT,
    REQUIRED_EXECUTION_ENVIRONMENT,
    SENSITIVE_EXECUTION_PREFIXES,
)
from .m1_config import LongContextM1Config, M1InitializationProbe
from .model_identity import ModelIdentityFacts, require_model_identity
from .runtime_identity import RuntimeIdentityFacts, require_upstream_runtime

M1_SCHEMA_VERSION = "longctx-m1-init.v2"
M1_PROBE_SCHEMA_VERSION = "longctx-m1-probe.v2"
M1_INTEGRITY_FILE = "m1-integrity.json"
PROBE_INTEGRITY_FILE = "probe-integrity.json"
EXPERIMENT_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class M1Manifest(StrictFrozenModel):
    schema_version: str = M1_SCHEMA_VERSION
    experiment_id: str
    created_at: str
    config_sha256: str
    source_commit: str
    source_tree_sha256: str
    runtime: RuntimeIdentityFacts
    model: ModelIdentityFacts
    execution_environment: dict[str, str]

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class ProbeRunRecord(StrictFrozenModel):
    run_id: str
    probe_id: str
    role: str
    repeat: int = Field(ge=0)
    status: str
    evidence_path: str
    cleanup_path: str
    server_log_path: str
    integrity_path: str


class M1Summary(StrictFrozenModel):
    schema_version: str = M1_SCHEMA_VERSION
    experiment_id: str
    finished_at: str
    calibration: dict[str, Any]
    probe_runs: tuple[ProbeRunRecord, ...]
    validations: tuple[dict[str, Any], ...]
    deployment_plan: dict[str, Any]
    primary_error_target_percent: float
    primary_error_passed: bool
    extrapolation_error_passed: bool
    initialization_validation_passed: bool
    source_commit: str
    artifact_root: str


MetricsFetcher = Callable[[str], Awaitable[str]]
GPUMemoryReader = Callable[[int], DeviceMemoryEvidence]

CUDA_MEMORY_PROBE = """\
import json

import torch

torch.cuda.set_device(0)
free_memory_bytes, total_memory_bytes = torch.cuda.mem_get_info(0)
print(json.dumps({
    "cuda_free_memory_bytes": free_memory_bytes,
    "cuda_allocatable_total_memory_bytes": total_memory_bytes,
}, sort_keys=True))
"""


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _tree_files(root: Path, integrity_name: str) -> dict[str, Path]:
    files: dict[str, Path] = {}
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"M1 artifact tree contains symlink: {path}")
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if relative == integrity_name:
            continue
        files[relative] = path
    return files


def _seal_tree(root: Path, integrity_name: str, schema: str, identity: str) -> None:
    integrity_path = root / integrity_name
    if integrity_path.exists():
        raise ValueError(f"artifact tree is already sealed: {integrity_path}")
    files = _tree_files(root, integrity_name)
    if not files:
        raise ValueError("cannot seal empty artifact tree")
    _atomic_json(
        integrity_path,
        {
            "schema": schema,
            "identity": identity,
            "files": {
                name: {"size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
                for name, path in sorted(files.items())
            },
        },
    )


def _validate_tree(root: Path, integrity_name: str, schema: str, identity: str) -> None:
    payload = _read_json(root / integrity_name)
    if payload.get("schema") != schema or payload.get("identity") != identity:
        raise ValueError("artifact integrity identity mismatch")
    expected = payload.get("files")
    if not isinstance(expected, dict):
        raise ValueError("artifact integrity files must be a mapping")
    actual = _tree_files(root, integrity_name)
    if set(expected) != set(actual):
        raise ValueError("artifact integrity file set mismatch")
    for name, path in actual.items():
        record = expected[name]
        if not isinstance(record, dict):
            raise ValueError(f"invalid artifact integrity record: {name}")
        if record.get("size_bytes") != path.stat().st_size or record.get("sha256") != sha256_file(
            path
        ):
            raise ValueError(f"artifact checksum mismatch: {name}")


def _required_positive_probe_int(payload: object, name: str) -> int:
    if not isinstance(payload, dict):
        raise ValueError("CUDA memory probe output must be a JSON object")
    value = payload.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"CUDA memory probe field {name!r} must be a positive integer")
    return value


def _gpu_memory_snapshot(device_index: int) -> DeviceMemoryEvidence:
    physical = subprocess.run(
        [
            "nvidia-smi",
            f"--id={device_index}",
            "--query-gpu=memory.total,memory.free",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    )
    row = next((line for line in physical.stdout.splitlines() if line.strip()), None)
    if row is None:
        raise ValueError("nvidia-smi returned no memory row")
    columns = [column.strip() for column in row.split(",")]
    if len(columns) != 2:
        raise ValueError(f"invalid nvidia-smi memory row: {row!r}")
    try:
        physical_total = int(columns[0]) * (1 << 20)
        physical_free = int(columns[1]) * (1 << 20)
    except ValueError as error:
        raise ValueError(f"invalid nvidia-smi memory row: {row!r}") from error

    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(device_index)
    cuda = subprocess.run(
        [sys.executable, "-c", CUDA_MEMORY_PROBE],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
        env=environment,
    )
    try:
        payload = json.loads(cuda.stdout)
    except json.JSONDecodeError as error:
        raise ValueError("CUDA memory probe returned invalid JSON") from error
    cuda_free = _required_positive_probe_int(payload, "cuda_free_memory_bytes")
    cuda_total = _required_positive_probe_int(payload, "cuda_allocatable_total_memory_bytes")
    return DeviceMemoryEvidence(
        device_index=device_index,
        physical_total_memory_bytes=physical_total,
        physical_free_memory_bytes=physical_free,
        cuda_allocatable_total_memory_bytes=cuda_total,
        cuda_free_memory_bytes=cuda_free,
        physical_minus_cuda_total_bytes=physical_total - cuda_total,
    )


def _clean_execution_environment(environment: Mapping[str, str]) -> dict[str, str]:
    issues: list[str] = []
    for name, expected in REQUIRED_EXECUTION_ENVIRONMENT.items():
        if environment.get(name) != expected:
            issues.append(f"{name} must equal {expected!r}")
    snapshot: dict[str, str] = {}
    for name, value in sorted(environment.items()):
        sensitive = name.startswith(SENSITIVE_EXECUTION_PREFIXES)
        allowed = name in ALLOWED_EXECUTION_ENVIRONMENT or name.startswith("VLLM_TUNER_")
        if sensitive and not allowed:
            issues.append(f"unlocked execution variable is set: {name}")
        if allowed:
            snapshot[name] = value
    if issues:
        raise ValueError("; ".join(issues))
    return snapshot


def _zero_estimate() -> ByteEstimate:
    return ByteEstimate(
        point_bytes=0,
        upper_bytes=0,
        source="unavailable",
        calibration_run_ids=(),
    )


class LongContextM1Runner:
    def __init__(
        self,
        config: LongContextM1Config,
        experiment_id: str,
        *,
        repository: str | Path,
        resume: bool = False,
        server_factory: type[ManagedVLLMServer] = ManagedVLLMServer,
        metrics_fetcher: MetricsFetcher = fetch_metrics_text,
        gpu_memory_reader: GPUMemoryReader = _gpu_memory_snapshot,
        runtime_facts: Optional[RuntimeIdentityFacts] = None,
        model_facts: Optional[ModelIdentityFacts] = None,
        execution_environment: Optional[Mapping[str, str]] = None,
    ) -> None:
        if EXPERIMENT_ID_PATTERN.fullmatch(experiment_id) is None:
            raise ValueError("experiment_id must be one portable path component")
        self.config = config
        self.experiment_id = experiment_id
        self.repository = Path(repository).resolve()
        self.resume = resume
        self.server_factory = server_factory
        self.metrics_fetcher = metrics_fetcher
        self.gpu_memory_reader = gpu_memory_reader
        self.runtime_facts = runtime_facts
        self.model_facts = model_facts
        self.execution_environment = dict(execution_environment or os.environ)
        self.store = ArtifactStore(config.artifacts.root, experiment_id)

    def _acquire_lock(self) -> int:
        self.config.artifacts.root.mkdir(parents=True, exist_ok=True)
        path = self.config.artifacts.root / f".{self.experiment_id}.run.lock"
        descriptor = os.open(
            path,
            os.O_CREAT | os.O_RDWR | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            os.close(descriptor)
            raise RuntimeError(f"M1 experiment is already running: {path}") from error
        return descriptor

    @staticmethod
    def _release_lock(descriptor: int) -> None:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)

    def _identities(
        self,
    ) -> tuple[RuntimeIdentityFacts, ModelIdentityFacts, dict[str, str]]:
        runtime = self.runtime_facts or require_upstream_runtime(self.config.runtime.lock_path)
        locked_model = self.config.model.identity()
        model = self.model_facts or require_model_identity(
            locked_model,
            model_dir=self.config.model.local_path,
            repository_id=locked_model.repository_id,
            revision=locked_model.revision,
            parameter_count=locked_model.parameter_count,
        )
        if not runtime.matches_lock or not model.matches_lock:
            raise ValueError("M1 requires matching runtime and model locks")
        environment = _clean_execution_environment(self.execution_environment)
        return runtime, model, environment

    def _manifest(
        self,
        runtime: RuntimeIdentityFacts,
        model: ModelIdentityFacts,
        environment: dict[str, str],
    ) -> M1Manifest:
        commit, dirty, _ = git_state(self.repository)
        tree = source_tree_sha256(self.repository)
        if commit is None or tree is None or dirty:
            raise ValueError("M1 formal probes require a clean Git source identity")
        return M1Manifest(
            experiment_id=self.experiment_id,
            created_at=utc_now_iso(),
            config_sha256=sha256_json(self.config.model_dump(mode="json")),
            source_commit=commit,
            source_tree_sha256=tree,
            runtime=runtime,
            model=model,
            execution_environment=environment,
        )

    def _initialize_root(self, requested: M1Manifest) -> Optional[dict[str, Any]]:
        root = self.store.root
        if root.exists():
            if not self.resume:
                raise FileExistsError(f"M1 artifact root exists: {root}; use --resume")
            if (root / M1_INTEGRITY_FILE).is_file():
                _validate_tree(root, M1_INTEGRITY_FILE, M1_SCHEMA_VERSION, self.experiment_id)
            existing = M1Manifest.model_validate_json(
                (root / "manifest.json").read_text(encoding="utf-8")
            )
            comparable = (
                "experiment_id",
                "config_sha256",
                "source_commit",
                "source_tree_sha256",
                "runtime",
                "model",
                "execution_environment",
            )
            mismatches = [
                field
                for field in comparable
                if getattr(existing, field) != getattr(requested, field)
            ]
            if mismatches:
                raise ValueError("M1 resume identity mismatch: " + ", ".join(mismatches))
            if (root / M1_INTEGRITY_FILE).is_file():
                return _read_json(root / "summary.json")
            self.store.initialize(exist_ok=True)
            return None
        self.store.initialize()
        self.store.write_json("manifest.json", requested)
        self.store.write_json("experiment.json", self.config.model_dump(mode="json"))
        return None

    def _base_planner_input(
        self,
        *,
        probe: M1InitializationProbe,
        total_memory_bytes: int,
        initial_free_memory_bytes: int,
        non_kv: NonKVPredictionSpec,
        cache: CacheLayoutSpec,
        distribution: Optional[ContextDistributionSpec] = None,
        safety: Optional[SafetyPolicy] = None,
    ) -> KVCapacityPlannerInput:
        model = model_spec_from_hf_config(self.config.model.local_path / "config.json")
        return KVCapacityPlannerInput(
            schema_version="longctx-m1.v1",
            model=model,
            cache=cache,
            device=DeviceBudgetSpec(
                total_memory_bytes=total_memory_bytes,
                initial_free_memory_bytes=initial_free_memory_bytes,
                gpu_memory_utilization_ppm=probe.gpu_memory_utilization_ppm,
            ),
            non_kv=non_kv,
            safety=safety
            or SafetyPolicy(
                fixed_operational_reserve_bytes=0,
                kv_reserve_basis_points=0,
                calibration_residual_upper_bytes=0,
                source="initialization validation raw capacity",
            ),
            serving=ServingLimits(
                max_model_len=probe.max_model_len,
                max_num_seqs=probe.max_num_seqs,
                tensor_parallel_size=1,
                pipeline_parallel_size=1,
            ),
            distribution=distribution
            or ContextDistributionSpec(
                bins=(
                    ContextBin(
                        name=f"full-{probe.max_model_len}",
                        weight_ppm=1_000_000,
                        prompt_tokens=probe.max_model_len - 1,
                        reserved_output_tokens=1,
                    ),
                ),
                confidence_ppm=990_000,
                iid_assumption=True,
                assume_no_prefix_reuse=True,
            ),
        )

    def _select_probe_attempt(self, logical_run_id: str) -> tuple[str, Path, dict[str, Any] | None]:
        probes_root = self.store.root / "probes"
        base = probes_root / logical_run_id
        if not base.exists():
            return logical_run_id, base, None
        if not self.resume:
            raise ValueError(f"probe directory already exists: {base}; use --resume")
        pattern = re.compile(rf"{re.escape(logical_run_id)}(?:-attempt(\d+))?")
        candidates: list[tuple[int, Path]] = []
        for path in probes_root.iterdir():
            match = pattern.fullmatch(path.name)
            if match is None or not path.is_dir():
                continue
            attempt = int(match.group(1) or 0)
            candidates.append((attempt, path))
        for _, path in sorted(candidates):
            integrity = path / PROBE_INTEGRITY_FILE
            if not integrity.is_file():
                raise ValueError(f"unsealed probe attempt cannot be resumed safely: {path}")
            _validate_tree(path, PROBE_INTEGRITY_FILE, M1_PROBE_SCHEMA_VERSION, path.name)
            status = _read_json(path / "status.json")
            if status.get("status") == "COMPLETE":
                return path.name, path, status
            if status.get("status") != "FAILED":
                raise ValueError(f"unknown sealed probe status: {path}")
        next_attempt = max((attempt for attempt, _ in candidates), default=0) + 1
        retry_id = f"{logical_run_id}-attempt{next_attempt}"
        return retry_id, probes_root / retry_id, None

    def _seal_failed_probe_attempt(self, logical_run_id: str, error: BaseException) -> None:
        probes_root = self.store.root / "probes"
        pattern = re.compile(rf"{re.escape(logical_run_id)}(?:-attempt(\d+))?")
        candidates = sorted(
            (
                path
                for path in probes_root.iterdir()
                if path.is_dir() and pattern.fullmatch(path.name)
            ),
            key=lambda path: path.name,
        )
        unsealed = [path for path in candidates if not (path / PROBE_INTEGRITY_FILE).exists()]
        if not unsealed:
            return
        probe_dir = unsealed[-1]
        run_id = probe_dir.name
        if not (probe_dir / "server.log").is_file():
            (probe_dir / "server.log").write_text("server log unavailable\n", encoding="utf-8")
        if not (probe_dir / "server-command.json").is_file():
            _atomic_json(
                probe_dir / "server-command.json",
                {"available": False, "argv": None, "environment": None},
            )
        if not (probe_dir / "cleanup.json").is_file():
            _atomic_json(
                probe_dir / "cleanup.json",
                {
                    "attempted": False,
                    "clean": False,
                    "process_group_empty": False,
                    "port_available": False,
                    "gpu_clean": None,
                    "errors": ["cleanup evidence unavailable"],
                },
            )
        _atomic_json(
            probe_dir / "failure.json",
            {
                "type": type(error).__name__,
                "message": str(error),
                "finished_at": utc_now_iso(),
            },
        )
        _atomic_json(
            probe_dir / "status.json",
            {
                "run_id": run_id,
                "logical_run_id": logical_run_id,
                "status": "FAILED",
                "finished_at": utc_now_iso(),
            },
        )
        _seal_tree(probe_dir, PROBE_INTEGRITY_FILE, M1_PROBE_SCHEMA_VERSION, run_id)

    async def _run_probe(
        self,
        probe: M1InitializationProbe,
        repeat: int,
    ) -> tuple[ProbeRunRecord, CapacityRuntimeEvidence]:
        logical_run_id = f"{probe.probe_id}-r{repeat}"
        run_id, probe_dir, cached_status = self._select_probe_attempt(logical_run_id)
        if cached_status is not None:
            evidence = CapacityRuntimeEvidence.model_validate_json(
                (probe_dir / "capacity-evidence.json").read_text(encoding="utf-8")
            )
            cached_cleanup = _read_json(probe_dir / "cleanup.json")
            if not all(
                cached_cleanup.get(field) is True
                for field in (
                    "clean",
                    "process_group_empty",
                    "port_available",
                    "gpu_clean",
                )
            ):
                raise ValueError(f"cached probe cleanup is not verified: {run_id}")
            return (
                ProbeRunRecord(
                    run_id=run_id,
                    probe_id=probe.probe_id,
                    role=probe.role,
                    repeat=repeat,
                    status="COMPLETE",
                    evidence_path=f"probes/{run_id}/capacity-evidence.json",
                    cleanup_path=f"probes/{run_id}/cleanup.json",
                    server_log_path=f"probes/{run_id}/server.log",
                    integrity_path=f"probes/{run_id}/{PROBE_INTEGRITY_FILE}",
                ),
                evidence,
            )
        probe_dir.mkdir(parents=True)

        device_memory = self.gpu_memory_reader(self.config.gpu.device_ids[0])
        cuda_allocatable_total = device_memory.cuda_allocatable_total_memory_bytes
        cuda_free = device_memory.cuda_free_memory_bytes
        cache = CacheLayoutSpec(
            requested_dtype="auto",
            resolved_dtype=KVDType.BFLOAT16,
            block_size=16,
            reserved_null_blocks=1,
            inline_metadata_bytes_per_layer_block=0,
            page_padding_bytes_per_layer_block=0,
            format_evidence="vllm_full_attention_spec",
        )
        zero = _zero_estimate()
        zero_non_kv = NonKVPredictionSpec(
            weights=zero,
            peak_activations=zero,
            runtime_non_torch=zero,
            post_profile_cuda_graph=zero,
            post_profile_persistent=zero,
            unattributed_runtime_residual=zero,
        )
        preliminary = plan_kv_capacity(
            self._base_planner_input(
                probe=probe,
                total_memory_bytes=cuda_allocatable_total,
                initial_free_memory_bytes=cuda_free,
                non_kv=zero_non_kv,
                cache=cache,
            )
        )

        server = self.server_factory(
            self.config.to_tuning_config(probe),
            trial_dir=probe_dir,
        )
        metrics_text: Optional[str] = None
        cleanup_payload: dict[str, Any]
        try:
            await server.start({})
            if not await server.wait_ready():
                raise RuntimeError("vLLM probe did not become ready")
            metrics_text = await self.metrics_fetcher(server.base_url)
        finally:
            runtime_cleanup = await server.stop()
            cleanup_payload = runtime_cleanup.model_dump(mode="json")
            self.store.write_json(f"probes/{run_id}/cleanup.json", cleanup_payload)
        cleanup_fields = ("clean", "process_group_empty", "port_available", "gpu_clean")
        if not all(cleanup_payload.get(field) is True for field in cleanup_fields):
            self.store.write_json(
                f"probes/{run_id}/status.json",
                {
                    "run_id": run_id,
                    "status": "FAILED",
                    "failure": "cleanup was not positively verified",
                    "finished_at": utc_now_iso(),
                },
            )
            raise RuntimeError(f"M1 probe cleanup failed: {run_id}")
        if metrics_text is None:
            raise RuntimeError("M1 probe produced no metrics exposition")
        log_text = server.log_path.read_text(encoding="utf-8", errors="replace")
        evidence = build_capacity_runtime_evidence(
            run_id=run_id,
            runtime_profile_sha256=preliminary.runtime_profile_sha256,
            server_log_text=log_text,
            metrics_text=metrics_text,
            device_memory=device_memory,
        )
        expected_architecture = model_spec_from_hf_config(
            self.config.model.local_path / "config.json"
        ).architecture
        self._validate_probe_evidence(probe, evidence, expected_architecture)
        self.store.write_json(f"probes/{run_id}/capacity-evidence.json", evidence)
        self.store.write_text(
            f"probes/{run_id}/cache-config-info.prom",
            evidence.cache_config.raw_metric_line + "\n",
        )
        self.store.write_json(
            f"probes/{run_id}/status.json",
            {
                "run_id": run_id,
                "status": "COMPLETE",
                "finished_at": utc_now_iso(),
                "cleanup_verified": all(
                    cleanup_payload.get(field) is True
                    for field in (
                        "clean",
                        "process_group_empty",
                        "port_available",
                        "gpu_clean",
                    )
                ),
            },
        )
        _seal_tree(probe_dir, PROBE_INTEGRITY_FILE, M1_PROBE_SCHEMA_VERSION, run_id)
        return (
            ProbeRunRecord(
                run_id=run_id,
                probe_id=probe.probe_id,
                role=probe.role,
                repeat=repeat,
                status="COMPLETE",
                evidence_path=f"probes/{run_id}/capacity-evidence.json",
                cleanup_path=f"probes/{run_id}/cleanup.json",
                server_log_path=f"probes/{run_id}/server.log",
                integrity_path=f"probes/{run_id}/{PROBE_INTEGRITY_FILE}",
            ),
            evidence,
        )

    @staticmethod
    def _validate_probe_evidence(
        probe: M1InitializationProbe,
        evidence: CapacityRuntimeEvidence,
        expected_architecture: str,
    ) -> None:
        cache = evidence.cache_config
        startup = evidence.startup_format
        checks = {
            "logged_capacity_consistent": evidence.logged_capacity_consistent,
            "utilization_matches_probe": (
                cache.gpu_memory_utilization_ppm == probe.gpu_memory_utilization_ppm
            ),
            "no_explicit_kv_memory": cache.kv_cache_memory_bytes is None,
            "no_block_override": cache.num_gpu_blocks_override is None,
            "requested_cache_dtype_auto": cache.requested_cache_dtype == "auto",
            "resolved_block_size_16": cache.resolved_block_size == 16,
            "prefix_caching_upstream_default": cache.enable_prefix_caching is True,
            "calculate_kv_scales_disabled": cache.calculate_kv_scales is False,
            "not_attention_free": cache.is_attention_free is False,
            "no_sliding_window_cache": cache.sliding_window is None,
            "architecture_matches_lock": startup.architecture == expected_architecture,
            "max_model_len_matches_probe": startup.max_model_len == probe.max_model_len,
            "attention_backend_flash_attn": startup.attention_backend == "FLASH_ATTN",
            "resolved_model_dtype_bfloat16": startup.model_dtype == "torch.bfloat16",
            "requested_kv_dtype_auto_log": startup.requested_kv_cache_dtype == "auto",
            "prefix_caching_log_enabled": startup.enable_prefix_caching is True,
            "chunked_prefill_log_enabled": startup.enable_chunked_prefill is True,
        }
        failed = [name for name, passed in checks.items() if not passed]
        if failed:
            raise ValueError(
                f"M1 probe runtime/cache format mismatch for {probe.probe_id}: " + ", ".join(failed)
            )

    @staticmethod
    def _non_kv_from_calibration(
        calibration: MultiRunNonKVCalibration,
        model_weights_bytes: int,
    ) -> NonKVPredictionSpec:
        if (
            calibration.point_bytes < model_weights_bytes
            or calibration.upper_bytes < model_weights_bytes
        ):
            raise ValueError("calibrated non-KV reserve is smaller than locked model weights")
        zero = _zero_estimate()
        residual = ByteEstimate(
            point_bytes=calibration.point_bytes - model_weights_bytes,
            upper_bytes=calibration.upper_bytes - model_weights_bytes,
            source="multi_run_calibration",
            calibration_run_ids=calibration.run_ids,
        )
        return NonKVPredictionSpec(
            weights=ByteEstimate(
                point_bytes=model_weights_bytes,
                upper_bytes=model_weights_bytes,
                source="structural",
                calibration_run_ids=(),
            ),
            peak_activations=zero,
            runtime_non_torch=zero,
            post_profile_cuda_graph=zero,
            post_profile_persistent=zero,
            unattributed_runtime_residual=residual,
        )

    async def run(self) -> dict[str, Any]:
        descriptor = self._acquire_lock()
        try:
            return await self._run_locked()
        finally:
            self._release_lock(descriptor)

    async def _run_locked(self) -> dict[str, Any]:
        runtime, model, environment = self._identities()
        requested_manifest = self._manifest(runtime, model, environment)
        completed = self._initialize_root(requested_manifest)
        if completed is not None:
            return {**completed, "resume_replayed": True}

        records: list[ProbeRunRecord] = []
        evidences: dict[str, CapacityRuntimeEvidence] = {}
        probes_by_id = {probe.probe_id: probe for probe in self.config.probes}
        for probe in self.config.probes:
            for repeat in range(probe.repeats):
                logical_run_id = f"{probe.probe_id}-r{repeat}"
                try:
                    record, evidence = await self._run_probe(probe, repeat)
                except BaseException as error:
                    self._seal_failed_probe_attempt(logical_run_id, error)
                    raise
                records.append(record)
                evidences[record.run_id] = evidence

        block_sizes = {evidence.cache_config.resolved_block_size for evidence in evidences.values()}
        if len(block_sizes) != 1:
            raise ValueError("M1 probes resolved different block sizes")
        block_size = block_sizes.pop()
        if block_size not in {1, 8, 16, 32}:
            raise ValueError(f"unsupported resolved CUDA block size: {block_size}")
        resolved_block_size = cast(Literal[1, 8, 16, 32], block_size)
        cache = CacheLayoutSpec(
            requested_dtype="auto",
            resolved_dtype=KVDType.BFLOAT16,
            block_size=resolved_block_size,
            reserved_null_blocks=1,
            inline_metadata_bytes_per_layer_block=0,
            page_padding_bytes_per_layer_block=0,
            format_evidence="vllm_full_attention_spec",
        )
        model_spec = model_spec_from_hf_config(self.config.model.local_path / "config.json")
        calibration_run_ids = {record.run_id for record in records if record.role == "calibration"}
        calibration_observations = tuple(
            evidence.observation
            for run_id, evidence in evidences.items()
            if run_id in calibration_run_ids
        )
        calibration = calibrate_non_kv_from_runs(
            observations=calibration_observations,
            model=model_spec,
            cache=cache,
        )
        locked_model = self.config.model.identity()
        model_weights_bytes = sum(
            file.size_bytes
            for path, file in locked_model.files.items()
            if path.endswith(".safetensors")
        )
        non_kv = self._non_kv_from_calibration(calibration, model_weights_bytes)

        validations: list[dict[str, Any]] = []
        for record in records:
            probe = probes_by_id[record.probe_id]
            if probe.role not in {"validation", "extrapolation"}:
                continue
            evidence = evidences[record.run_id]
            planner_input = self._base_planner_input(
                probe=probe,
                total_memory_bytes=evidence.observation.total_memory_bytes,
                initial_free_memory_bytes=evidence.observation.initial_free_memory_bytes,
                non_kv=non_kv,
                cache=cache,
            )
            plan = plan_kv_capacity(planner_input)
            validation = validate_kv_capacity_plan(
                plan=plan,
                observation=evidence.observation,
                block_size=block_size,
                target_error_percent=10.0,
            )
            self.store.write_json(f"planner/inputs/{record.run_id}.json", planner_input)
            self.store.write_json(f"planner/plans/{record.run_id}.json", plan)
            self.store.write_json(f"planner/validations/{record.run_id}.json", validation)
            validation_row = {
                **validation.model_dump(mode="json"),
                "evaluation_role": probe.role,
            }
            validations.append(validation_row)

        deployment_probes = [
            probe for probe in self.config.probes if probe.role in {"validation", "extrapolation"}
        ]
        deployment_probe = max(deployment_probes, key=lambda probe: probe.max_model_len)
        deployment_evidence = next(
            evidence
            for run_id, evidence in evidences.items()
            if run_id.startswith(deployment_probe.probe_id + "-r")
        )
        deployment_input = self._base_planner_input(
            probe=deployment_probe,
            total_memory_bytes=deployment_evidence.observation.total_memory_bytes,
            initial_free_memory_bytes=deployment_evidence.observation.initial_free_memory_bytes,
            non_kv=non_kv,
            cache=cache,
            distribution=self.config.deployment_distribution,
            safety=self.config.safety,
        )
        deployment_plan = plan_kv_capacity(deployment_input)
        self.store.write_json("planner/calibration.json", calibration)
        self.store.write_json("planner/deployment-input.json", deployment_input)
        self.store.write_json("planner/deployment-plan.json", deployment_plan)

        primary_rows = [
            validation
            for validation in validations
            if validation["evaluation_role"] == "validation"
        ]
        extrapolation_rows = [
            validation
            for validation in validations
            if validation["evaluation_role"] == "extrapolation"
        ]
        primary_passed = bool(primary_rows) and all(
            validation["within_target"] is True for validation in primary_rows
        )
        extrapolation_passed = bool(extrapolation_rows) and all(
            validation["within_target"] is True for validation in extrapolation_rows
        )
        commit, dirty, _ = git_state(self.repository)
        tree = source_tree_sha256(self.repository)
        if (
            dirty
            or commit != requested_manifest.source_commit
            or tree != requested_manifest.source_tree_sha256
        ):
            raise ValueError("source identity changed during M1 probes")
        summary = M1Summary(
            experiment_id=self.experiment_id,
            finished_at=utc_now_iso(),
            calibration=calibration.model_dump(mode="json"),
            probe_runs=tuple(records),
            validations=tuple(validations),
            deployment_plan=deployment_plan.model_dump(mode="json"),
            primary_error_target_percent=10.0,
            primary_error_passed=primary_passed,
            extrapolation_error_passed=extrapolation_passed,
            initialization_validation_passed=(primary_passed and extrapolation_passed),
            source_commit=requested_manifest.source_commit,
            artifact_root=str(self.store.root),
        )
        self.store.write_json("summary.json", summary)
        _seal_tree(self.store.root, M1_INTEGRITY_FILE, M1_SCHEMA_VERSION, self.experiment_id)
        _validate_tree(self.store.root, M1_INTEGRITY_FILE, M1_SCHEMA_VERSION, self.experiment_id)
        return summary.model_dump(mode="json")
