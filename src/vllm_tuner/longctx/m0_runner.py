"""Isolated M0 runner for the redesigned long-context v5 project line."""

from __future__ import annotations

import fcntl
import json
import logging
import os
import re
import subprocess
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field

from vllm_tuner.experiment.artifacts import ArtifactStore
from vllm_tuner.experiment.manifest import (
    build_manifest,
    git_state,
    sha256_file,
    sha256_json,
    source_tree_sha256,
    validate_resume_manifest,
)
from vllm_tuner.experiment.models import (
    ExperimentSpec,
    TrialResult,
    TrialStatus,
    utc_now_iso,
)
from vllm_tuner.runtime.controller import TrialController
from vllm_tuner.runtime.failures import UnsafeCleanupError
from vllm_tuner.workloads.generator import generate_trace
from vllm_tuner.workloads.trace import WorkloadTrace

from .integrity import M0_INTEGRITY_FILE, seal_m0_artifacts, validate_m0_artifacts
from .m0_config import LongContextM0Config
from .model_identity import ModelIdentityFacts, require_model_identity
from .runtime_identity import RuntimeIdentityFacts, require_upstream_runtime

logger = logging.getLogger(__name__)

M0_SCHEMA_VERSION = "longctx-v5-m0-v1"
M0_SUMMARY_FILE = "m0-summary.json"
M0_MANIFEST_FILE = "manifest.json"
MODEL_REVISION_MARKER = ".slotune-model-revision"
STARTUP_PROFILE_FILE = "production-default-runtime.json"
PRODUCTION_TRIAL_METHOD = "production-default"
PRODUCTION_TRIAL_PREFIX = "production-default-attempt-"
REQUIRED_CLEANUP_FIELDS = (
    "clean",
    "process_group_empty",
    "port_available",
    "gpu_clean",
)
REQUIRED_EXECUTION_ENVIRONMENT = {
    "OMP_NUM_THREADS": "8",
    "TOKENIZERS_PARALLELISM": "false",
    "VLLM_NO_USAGE_STATS": "1",
}
ALLOWED_EXECUTION_ENVIRONMENT = frozenset(
    {
        *REQUIRED_EXECUTION_ENVIRONMENT,
        "CUDA_CACHE_PATH",
        "CUDA_VISIBLE_DEVICES",
        "FLASHINFER_WORKSPACE_BASE",
        "NUMBA_CACHE_DIR",
        "PYTHONDONTWRITEBYTECODE",
        "PYTHONUNBUFFERED",
        "TORCHINDUCTOR_CACHE_DIR",
        "TORCH_EXTENSIONS_DIR",
        "TORCH_HOME",
        "TRITON_CACHE_DIR",
        "VLLM_CACHE_ROOT",
    }
)
SENSITIVE_EXECUTION_PREFIXES = (
    "CUBLAS_",
    "CUDA_",
    "FLASHINFER_",
    "NCCL_",
    "PYTORCH_",
    "TORCH_",
    "TRITON_",
    "VLLM_",
)
M0_EXECUTION_CHECKS = (
    "project_line_is_v5",
    "profile_is_production_default",
    "trial_identity_is_production_default",
    "no_legacy_results_used",
    "runtime_lock_verified",
    "model_lock_verified",
    "startup_profile_verified",
    "model_revision_marker_verified",
    "model_is_dense",
    "measured_requests_at_least_100",
    "measured_request_count_matches",
    "all_requests_completed",
    "zero_request_failures",
    "trial_complete_and_feasible",
    "cleanup_verified",
    "resume_ready",
)


class ProductionDefaultRuntime(BaseModel):
    """Startup facts proving what upstream production-default resolved to."""

    vllm_version: Optional[str] = None
    attention_backend: Optional[str] = None
    dtype: Optional[str] = None
    kv_cache_dtype: Optional[str] = None
    enable_prefix_caching: Optional[bool] = None
    enable_chunked_prefill: Optional[bool] = None
    max_model_len: Optional[int] = None
    matches_expected: bool
    issues: list[str] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid", frozen=True)


class LongContextM0Manifest(BaseModel):
    """Immutable identity for a v5 M0 production-default canary."""

    schema_version: str = M0_SCHEMA_VERSION
    project_line: str
    milestone: str
    profile: str
    evidence_role: str
    model_tier: str
    fallback_reason: Optional[str] = None
    experiment_id: str
    created_at: str
    config_sha256: str
    runtime: RuntimeIdentityFacts
    model: ModelIdentityFacts
    execution_environment: dict[str, str]
    experiment: ExperimentSpec

    model_config = ConfigDict(extra="forbid", frozen=True)


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON artifact {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def _command_output(command: list[str], cwd: Path, timeout: float = 120.0) -> str:
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError) as error:
        return f"unavailable: {type(error).__name__}: {error}\n"
    output = completed.stdout
    if completed.stderr:
        output += ("\n" if output else "") + completed.stderr
    if completed.returncode != 0:
        output += f"\ncommand_exit_code={completed.returncode}\n"
    return output if output.endswith("\n") else output + "\n"


def _parse_bool(value: Optional[str]) -> Optional[bool]:
    if value == "True":
        return True
    if value == "False":
        return False
    return None


def _production_default_runtime(log_path: Path) -> ProductionDefaultRuntime:
    """Parse stable startup evidence needed to name this profile upstream default."""
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError as error:
        return ProductionDefaultRuntime(
            matches_expected=False,
            issues=[f"unable to read server startup log: {error}"],
        )

    def first(pattern: str) -> Optional[str]:
        match = re.search(pattern, text)
        return match.group(1) if match is not None else None

    version = first(r"\bversion\s+(\d+\.\d+\.\d+)\b")
    attention_backend = first(r"Using\s+([A-Z0-9_]+)\s+attention backend")
    dtype = first(r"\bdtype=([^,\s]+)")
    kv_cache_dtype = first(r"\bkv_cache_dtype=([^,\s]+)")
    prefix_caching = _parse_bool(first(r"\benable_prefix_caching=(True|False)"))
    chunked_prefill = _parse_bool(first(r"\benable_chunked_prefill=(True|False)"))
    max_model_len_text = first(r"Using max model len\s+(\d+)")
    max_model_len = int(max_model_len_text) if max_model_len_text is not None else None

    expected = {
        "vllm_version": (version, "0.16.0"),
        "attention_backend": (attention_backend, "FLASH_ATTN"),
        "dtype": (dtype, "torch.bfloat16"),
        "kv_cache_dtype": (kv_cache_dtype, "auto"),
        "enable_prefix_caching": (prefix_caching, True),
        "enable_chunked_prefill": (chunked_prefill, True),
    }
    issues = [
        f"upstream default {name} mismatch: expected {wanted!r}, found {actual!r}"
        for name, (actual, wanted) in expected.items()
        if actual != wanted
    ]
    return ProductionDefaultRuntime(
        vllm_version=version,
        attention_backend=attention_backend,
        dtype=dtype,
        kv_cache_dtype=kv_cache_dtype,
        enable_prefix_caching=prefix_caching,
        enable_chunked_prefill=chunked_prefill,
        max_model_len=max_model_len,
        matches_expected=not issues,
        issues=issues,
    )


def _model_metadata(config: LongContextM0Config) -> tuple[dict[str, Any], bool]:
    """Read architecture facts and verify the downloader completion marker."""
    model_dir = config.model.local_path
    config_path = model_dir / "config.json"
    model_config = _read_json_object(config_path)
    architectures_value = model_config.get("architectures")
    architectures = (
        [str(value) for value in architectures_value]
        if isinstance(architectures_value, list)
        else []
    )
    model_type = str(model_config.get("model_type", ""))
    if not architectures or not model_type:
        raise ValueError("model config must expose architectures and model_type")
    dense_identity = " ".join([model_type, *architectures]).casefold()
    if "moe" in dense_identity or model_config.get("num_experts") is not None:
        raise ValueError("longctx-v5 M0 requires a dense model, not an MoE model")

    integer_fields = (
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "hidden_size",
        "max_position_embeddings",
    )
    for field in integer_fields:
        value = model_config.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"model config must expose a positive integer {field}")

    locked = config.model.identity()
    marker_path = model_dir / MODEL_REVISION_MARKER
    marker_revision = (
        marker_path.read_text(encoding="utf-8").strip() if marker_path.is_file() else None
    )
    marker_verified = marker_revision == locked.revision
    partial_files = sorted(path.name for path in model_dir.glob("*.part"))
    if partial_files:
        raise ValueError("local model download is incomplete: " + ", ".join(partial_files))
    metadata = {
        "repository_id": locked.repository_id,
        "revision": locked.revision,
        "parameter_count": locked.parameter_count,
        "local_path": str(model_dir),
        "model_type": model_type,
        "architectures": architectures,
        **{field: model_config[field] for field in integer_fields},
        "torch_dtype": model_config.get("torch_dtype"),
        "config_sha256": sha256_file(config_path),
        "revision_marker": marker_revision,
        "revision_marker_verified": marker_verified,
    }
    return metadata, marker_verified


class LongContextM0Runner:
    """Run exactly one upstream-default canary without Legacy tuning machinery."""

    def __init__(
        self,
        config: LongContextM0Config,
        experiment_id: str,
        *,
        repository: str | Path = ".",
        resume: bool = False,
        require_clean_source: bool = True,
        tokenizer: Optional[Any] = None,
        controller_factory: type[TrialController] = TrialController,
        runtime_facts: Optional[RuntimeIdentityFacts] = None,
        model_facts: Optional[ModelIdentityFacts] = None,
        execution_environment: Optional[Mapping[str, str]] = None,
        capture_environment: bool = True,
    ) -> None:
        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", experiment_id) is None:
            raise ValueError("experiment_id must be one portable directory name")
        self.config = config
        self.experiment_id = experiment_id
        self.repository = Path(repository).expanduser().resolve()
        self.resume = resume
        self.require_clean_source = require_clean_source
        self._tokenizer = tokenizer
        self.controller_factory = controller_factory
        self._runtime_facts = runtime_facts
        self._model_facts = model_facts
        self._execution_environment = execution_environment
        self.capture_environment = capture_environment
        self.artifacts = ArtifactStore(config.artifacts.root, experiment_id)
        self._resume_warnings: list[str] = []

    def _acquire_run_lock(self) -> int:
        """Exclusively own one experiment ID for the complete run or resume."""
        lock_root = self.config.artifacts.root
        lock_root.mkdir(parents=True, exist_ok=True)
        lock_path = lock_root / f".{self.experiment_id}.run.lock"
        flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(lock_path, flags, 0o600)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            os.close(descriptor)
            raise RuntimeError(
                f"M0 experiment is already owned by another process: {lock_path}"
            ) from error
        except BaseException:
            os.close(descriptor)
            raise
        os.ftruncate(descriptor, 0)
        os.write(descriptor, f"pid={os.getpid()}\n".encode("ascii"))
        os.fsync(descriptor)
        return descriptor

    @staticmethod
    def _release_run_lock(descriptor: int) -> None:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)

    def _runtime_identity(self) -> RuntimeIdentityFacts:
        if self._runtime_facts is not None:
            if not self._runtime_facts.matches_lock:
                raise ValueError("injected runtime facts do not match the upstream lock")
            return self._runtime_facts
        return require_upstream_runtime(self.config.runtime.lock_path)

    def _model_identity(self) -> ModelIdentityFacts:
        if self._model_facts is not None:
            if not self._model_facts.matches_lock:
                raise ValueError("injected model facts do not match the model lock")
            return self._model_facts
        locked = self.config.model.identity()
        return require_model_identity(
            locked,
            model_dir=self.config.model.local_path,
            repository_id=locked.repository_id,
            revision=locked.revision,
            parameter_count=locked.parameter_count,
        )

    def _execution_environment_identity(self) -> dict[str, str]:
        environment = dict(self._execution_environment or os.environ)
        issues: list[str] = []
        for name, expected in REQUIRED_EXECUTION_ENVIRONMENT.items():
            actual = environment.get(name)
            if actual != expected:
                issues.append(
                    f"execution environment {name} mismatch: expected {expected!r}, found {actual!r}"
                )
        snapshot: dict[str, str] = {}
        for name, value in sorted(environment.items()):
            sensitive = name.startswith(SENSITIVE_EXECUTION_PREFIXES)
            allowed = name in ALLOWED_EXECUTION_ENVIRONMENT or name.startswith("VLLM_TUNER_")
            if sensitive and not allowed:
                issues.append(f"unlocked execution environment variable is set: {name}")
            if allowed or name in REQUIRED_EXECUTION_ENVIRONMENT:
                snapshot[name] = value
        if issues:
            raise ValueError("; ".join(issues))
        return snapshot

    def _load_tokenizer(self) -> Any:
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(
                str(self.config.model.local_path),
                local_files_only=True,
            )
        return self._tokenizer

    def _trace(self) -> WorkloadTrace:
        workload = self.config.workload
        return generate_trace(
            "chat",
            count=workload.measured_requests,
            request_rate=workload.request_rate,
            burstiness=workload.burstiness,
            seed=workload.seed,
            tokenizer=self._load_tokenizer(),
            fixed_input_tokens=workload.fixed_input_tokens,
            fixed_output_tokens=workload.fixed_output_tokens,
        )

    def _requested_manifest(
        self,
        trace_path: Path,
        runtime: RuntimeIdentityFacts,
        model: ModelIdentityFacts,
        execution_environment: dict[str, str],
    ) -> LongContextM0Manifest:
        tuning = self.config.to_tuning_config()
        experiment = build_manifest(
            experiment_id=self.experiment_id,
            model=tuning.model,
            model_revision=model.expected_revision,
            tokenizer=tuning.tokenizer,
            trace_path=trace_path,
            workload=tuning.workload.model_dump(mode="json"),
            slo=tuning.slo.model_dump(mode="json"),
            constraints=tuning.constraints.model_dump(mode="json"),
            gpu_config=tuning.gpu.model_dump(mode="json"),
            telemetry=tuning.telemetry.model_dump(mode="json"),
            study={
                "runner": "longctx-v5-m0",
                "project_line": self.config.project_line,
                "milestone": self.config.milestone,
                "profile": self.config.profile,
                "evidence_role": self.config.evidence_role,
                "model_tier": self.config.model_tier,
                "measured_runs": 1,
                "resume_supported": True,
            },
            vllm_args={},
            search_space={
                "kind": "none",
                "profile": self.config.profile,
                "trial_params": {},
            },
            seed=self.config.workload.seed,
            repository=self.repository,
        )
        if self.require_clean_source:
            if experiment.source_commit is None or experiment.source_tree_sha256 is None:
                raise ValueError("M0 production baseline requires a resolvable Git source identity")
            if experiment.dirty_worktree:
                raise ValueError(
                    "M0 production baseline requires a clean Git worktree; "
                    "--allow-dirty-source is development-only"
                )
        return LongContextM0Manifest(
            project_line=self.config.project_line,
            milestone=self.config.milestone,
            profile=self.config.profile,
            evidence_role=self.config.evidence_role,
            model_tier=self.config.model_tier,
            fallback_reason=self.config.fallback_reason,
            experiment_id=self.experiment_id,
            created_at=utc_now_iso(),
            config_sha256=sha256_json(self.config.model_dump(mode="json")),
            runtime=runtime,
            model=model,
            execution_environment=execution_environment,
            experiment=experiment,
        )

    @staticmethod
    def _validate_resume_identity(
        existing: LongContextM0Manifest,
        requested: LongContextM0Manifest,
    ) -> None:
        fields = (
            "schema_version",
            "project_line",
            "milestone",
            "profile",
            "evidence_role",
            "model_tier",
            "fallback_reason",
            "experiment_id",
            "config_sha256",
            "runtime",
            "model",
            "execution_environment",
        )
        mismatches = [
            field for field in fields if getattr(existing, field) != getattr(requested, field)
        ]
        if mismatches:
            raise ValueError(
                "cannot resume incompatible longctx-v5 M0 artifacts; mismatched fields: "
                + ", ".join(mismatches)
            )
        validate_resume_manifest(existing.experiment, requested.experiment)

    def _write_environment(
        self,
        runtime: RuntimeIdentityFacts,
        execution_environment: dict[str, str],
    ) -> None:
        self.artifacts.write_json("environment/runtime-identity.json", runtime)
        self.artifacts.write_json(
            "environment/execution-environment.json",
            execution_environment,
        )
        _, _, status = git_state(self.repository)
        self.artifacts.write_text(
            "environment/git-state.txt",
            status + "\n" if status else "clean working tree\n",
        )
        if not self.capture_environment:
            unavailable = "unavailable: disabled by test runner\n"
            self.artifacts.write_text("environment/python-packages.txt", unavailable)
            self.artifacts.write_text("environment/nvidia-smi.txt", unavailable)
            self.artifacts.write_text("environment/collect-env.txt", unavailable)
            return
        self.artifacts.write_text(
            "environment/python-packages.txt",
            _command_output([sys.executable, "-m", "pip", "freeze"], self.repository),
        )
        self.artifacts.write_text(
            "environment/nvidia-smi.txt",
            _command_output(["nvidia-smi"], self.repository),
        )
        self.artifacts.write_text(
            "environment/collect-env.txt",
            _command_output([sys.executable, "-m", "torch.utils.collect_env"], self.repository),
        )

    def _initialize_root(
        self,
        requested: LongContextM0Manifest,
        trace_path: Path,
        runtime: RuntimeIdentityFacts,
        model_metadata: dict[str, Any],
        execution_environment: dict[str, str],
    ) -> tuple[LongContextM0Manifest, Optional[dict[str, Any]]]:
        root = self.artifacts.root
        root_preexisting = root.exists()
        if root_preexisting and not self.resume:
            raise FileExistsError(f"M0 artifact directory already exists: {root}; use --resume")

        if root_preexisting:
            manifest_path = root / M0_MANIFEST_FILE
            if not manifest_path.is_file():
                raise ValueError(f"cannot resume M0 root without {M0_MANIFEST_FILE}: {root}")
            sealed = (root / M0_INTEGRITY_FILE).is_file()
            if sealed:
                validate_m0_artifacts(root)
            existing = LongContextM0Manifest.model_validate_json(
                manifest_path.read_text(encoding="utf-8")
            )
            self._validate_resume_identity(existing, requested)
            if sealed:
                return existing, _read_json_object(root / M0_SUMMARY_FILE)
            if (root / M0_SUMMARY_FILE).is_file():
                self._resume_warnings.append(
                    "explicit resume recovered a root with summary but no integrity seal"
                )
            self.artifacts.initialize(exist_ok=True)
            return existing, None

        self.artifacts.initialize()
        self.artifacts.write_json(M0_MANIFEST_FILE, requested)
        self.artifacts.write_yaml("experiment.yaml", self.config.model_dump(mode="json"))
        self.artifacts.save_trace(trace_path)
        self.artifacts.write_json(
            "model-identity.json",
            {
                "lock": requested.model.model_dump(mode="json"),
                "architecture": model_metadata,
            },
        )
        self._write_environment(runtime, execution_environment)
        return requested, None

    def _trial_directories(self) -> list[Path]:
        if not self.artifacts.trials_dir.is_dir():
            return []
        directories: list[Path] = []
        pattern = re.compile(rf"{re.escape(PRODUCTION_TRIAL_PREFIX)}(\d{{4}})")
        for path in self.artifacts.trials_dir.iterdir():
            if not path.is_dir() or pattern.fullmatch(path.name) is None:
                raise ValueError(f"unexpected M0 trial entry: {path}")
            directories.append(path)
        return sorted(directories, key=lambda path: path.name)

    def _next_trial_id(self) -> str:
        numbers = [
            int(path.name.removeprefix(PRODUCTION_TRIAL_PREFIX))
            for path in self._trial_directories()
        ]
        return f"{PRODUCTION_TRIAL_PREFIX}{max(numbers, default=0) + 1:04d}"

    @staticmethod
    def _require_trial_identity(result: TrialResult, expected_trial_id: str) -> None:
        expected = {
            "trial_id": expected_trial_id,
            "method": PRODUCTION_TRIAL_METHOD,
            "phase": "search",
            "source_method": PRODUCTION_TRIAL_METHOD,
            "source_trial_id": None,
        }
        mismatches = [field for field, value in expected.items() if getattr(result, field) != value]
        if mismatches:
            raise ValueError("M0 trial provenance mismatch: " + ", ".join(mismatches))

    def _cached_complete_trial(
        self,
    ) -> Optional[tuple[TrialResult, ProductionDefaultRuntime]]:
        if not self.resume:
            return None
        for trial_dir in reversed(self._trial_directories()):
            try:
                result = self.artifacts.load_trial_result(trial_dir.name)
                if result is None:
                    continue
                self._require_trial_identity(result, trial_dir.name)
                if result.params != {} or not result.selectable:
                    continue
                required_evidence = {
                    "server-command.json",
                    "request-results.jsonl",
                    "benchmark-raw.json",
                    "prometheus.jsonl",
                    "nvml.jsonl",
                    "server.log",
                    "cleanup.json",
                    STARTUP_PROFILE_FILE,
                }
                self.artifacts.validate_trial_artifacts(
                    result.trial_id,
                    require_telemetry=True,
                    require_available=True,
                    required_evidence=required_evidence,
                )
                self.artifacts.validate_cached_trial(result, require_telemetry=True)
                profile = ProductionDefaultRuntime.model_validate_json(
                    (trial_dir / STARTUP_PROFILE_FILE).read_text(encoding="utf-8")
                )
                if not profile.matches_expected:
                    continue
                if result.client.get("num_requests") != self.config.workload.measured_requests:
                    continue
            except (OSError, ValueError) as error:
                self._resume_warnings.append(
                    f"cached trial {trial_dir.name} is not reusable: {error}"
                )
                continue
            logger.info("Replaying complete M0 trial %s", result.trial_id)
            return result, profile
        return None

    def _finalize_trial(self, result: TrialResult) -> TrialResult:
        self.artifacts.ensure_trial_artifacts(result)
        required_evidence = {
            "server-command.json",
            "request-results.jsonl",
            "benchmark-raw.json",
            "prometheus.jsonl",
            "nvml.jsonl",
            "server.log",
            "cleanup.json",
            STARTUP_PROFILE_FILE,
        }
        try:
            self.artifacts.validate_trial_artifacts(
                result.trial_id,
                require_telemetry=True,
                require_available=result.status == TrialStatus.COMPLETE,
                required_evidence=required_evidence,
            )
            self.artifacts.validate_cached_trial(result, require_telemetry=True)
        except ValueError as error:
            if result.status != TrialStatus.COMPLETE:
                raise
            result.status = TrialStatus.FAILED
            violations = list(result.constraints.get("violations", []))
            if "artifact_unavailable" not in violations:
                violations.append("artifact_unavailable")
            result.constraints = {
                **result.constraints,
                "feasible": False,
                "violations": violations,
            }
            result.failure_reason = {
                "type": "ARTIFACT_UNAVAILABLE",
                "message": str(error),
                "phase": "M0_ARTIFACT_FINALIZE",
            }
            self.artifacts.record_artifact_finalizer_failure(result, str(error))
            self.artifacts.seal_trial_artifacts(result)
            self.artifacts.validate_cached_trial(result, require_telemetry=True)
        return result

    def _acceptance(
        self,
        manifest: LongContextM0Manifest,
        result: TrialResult,
        model_metadata: dict[str, Any],
        marker_verified: bool,
        startup_profile: ProductionDefaultRuntime,
    ) -> dict[str, Any]:
        cleanup = result.cleanup_status if isinstance(result.cleanup_status, Mapping) else {}
        current_commit, current_dirty, _ = git_state(self.repository)
        current_tree = source_tree_sha256(self.repository)
        source_is_clean_commit = (
            current_commit == manifest.experiment.source_commit
            and current_tree == manifest.experiment.source_tree_sha256
            and not current_dirty
            and not manifest.experiment.dirty_worktree
        )
        trial_identity = (
            result.method == PRODUCTION_TRIAL_METHOD
            and result.phase == "search"
            and result.source_method == PRODUCTION_TRIAL_METHOD
            and result.source_trial_id is None
            and result.trial_id.startswith(PRODUCTION_TRIAL_PREFIX)
        )
        formal_tier = manifest.model_tier in {"primary-7b-8b", "fallback-3b"}
        fallback_protocol = manifest.model_tier != "fallback-3b" or bool(manifest.fallback_reason)
        checks = {
            "project_line_is_v5": manifest.project_line == "longctx-v5",
            "profile_is_production_default": manifest.profile == "production-default",
            "evidence_role_is_formal": manifest.evidence_role == "formal",
            "formal_model_tier_is_valid": formal_tier,
            "fallback_protocol_satisfied": fallback_protocol,
            "formal_uses_upstream_default_args": self.config.vllm_args == {},
            "trial_identity_is_production_default": trial_identity,
            "no_legacy_results_used": trial_identity,
            "runtime_lock_verified": manifest.runtime.matches_lock,
            "model_lock_verified": manifest.model.matches_lock,
            "startup_profile_verified": startup_profile.matches_expected,
            "source_is_clean_commit": source_is_clean_commit,
            "model_revision_marker_verified": marker_verified,
            "model_is_dense": "moe"
            not in " ".join(model_metadata.get("architectures", [])).casefold(),
            "measured_requests_at_least_100": self.config.workload.measured_requests >= 100,
            "measured_request_count_matches": result.client.get("num_requests")
            == self.config.workload.measured_requests,
            "all_requests_completed": result.client.get("completed")
            == self.config.workload.measured_requests,
            "zero_request_failures": result.client.get("failed") == 0,
            "trial_complete_and_feasible": result.status == TrialStatus.COMPLETE
            and result.constraints.get("feasible") is True,
            "cleanup_verified": all(
                cleanup.get(field) is True for field in REQUIRED_CLEANUP_FIELDS
            ),
            "resume_ready": True,
        }
        execution_passed = all(checks[name] for name in M0_EXECUTION_CHECKS)
        return {
            "passed": all(checks.values()),
            "execution_passed": execution_passed,
            "checks": checks,
        }

    async def run(self) -> dict[str, Any]:
        descriptor = self._acquire_run_lock()
        try:
            return await self._run_locked()
        finally:
            self._release_run_lock(descriptor)

    async def _run_locked(self) -> dict[str, Any]:
        runtime = self._runtime_identity()
        execution_environment = self._execution_environment_identity()
        model = self._model_identity()
        model_metadata, marker_verified = _model_metadata(self.config)
        if not marker_verified:
            raise ValueError("model revision marker is missing or does not match the model lock")
        trace = self._trace()
        with tempfile.TemporaryDirectory(prefix="longctx-v5-m0-") as temporary:
            trace_path = trace.write(Path(temporary) / "trace.jsonl")
            requested = self._requested_manifest(
                trace_path,
                runtime,
                model,
                execution_environment,
            )
            manifest, sealed_summary = self._initialize_root(
                requested,
                trace_path,
                runtime,
                model_metadata,
                execution_environment,
            )

        if sealed_summary is not None:
            return {
                **sealed_summary,
                "resume_replayed": True,
                "artifact_root": str(self.artifacts.root),
            }

        cached = self._cached_complete_trial()
        resumed_trial = cached is not None
        if cached is None:
            tuning = self.config.to_tuning_config()
            controller = self.controller_factory(
                tuning,
                trace,
                self.artifacts,
                tokenizer=self._load_tokenizer(),
            )
            trial_id = self._next_trial_id()
            try:
                result = await controller.run_trial({}, trial_id, PRODUCTION_TRIAL_METHOD)
            except UnsafeCleanupError as error:
                if not isinstance(error.result, TrialResult):
                    raise
                result = error.result
            self._require_trial_identity(result, trial_id)
            startup_profile = _production_default_runtime(
                self.artifacts.trials_dir / trial_id / "server.log"
            )
            self.artifacts.write_json(
                Path("trials") / trial_id / STARTUP_PROFILE_FILE,
                startup_profile,
            )
            result = self._finalize_trial(result)
        else:
            result, startup_profile = cached

        acceptance = self._acceptance(
            manifest,
            result,
            model_metadata,
            marker_verified,
            startup_profile,
        )
        summary = {
            "schema_version": M0_SCHEMA_VERSION,
            "project_line": self.config.project_line,
            "milestone": self.config.milestone,
            "profile": self.config.profile,
            "evidence_role": self.config.evidence_role,
            "model_tier": self.config.model_tier,
            "fallback_reason": self.config.fallback_reason,
            "experiment_id": self.experiment_id,
            "finished_at": utc_now_iso(),
            "manifest": M0_MANIFEST_FILE,
            "trial_id": result.trial_id,
            "trial": result.model_dump(mode="json"),
            "startup_profile": startup_profile.model_dump(mode="json"),
            "acceptance": acceptance,
            "resume": {
                "requested": self.resume,
                "trial_replayed": resumed_trial,
                "warnings": self._resume_warnings,
                "command_flag": "--resume",
            },
            "legacy_results_used": not acceptance["checks"]["no_legacy_results_used"],
            "artifact_root": str(self.artifacts.root),
        }
        self.artifacts.write_json(M0_SUMMARY_FILE, summary)
        seal_m0_artifacts(
            self.artifacts.root,
            experiment_id=self.experiment_id,
            attestation={
                "experiment_id": self.experiment_id,
                "project_line": self.config.project_line,
                "milestone": self.config.milestone,
                "profile": self.config.profile,
                "evidence_role": self.config.evidence_role,
                "model_tier": self.config.model_tier,
                "source_commit": manifest.experiment.source_commit,
                "runtime_upstream_commit": manifest.runtime.upstream_commit,
                "model_revision": manifest.model.expected_revision,
            },
        )
        validate_m0_artifacts(self.artifacts.root)
        return summary


def load_m0_status(root: str | Path, experiment_id: str) -> dict[str, Any]:
    """Validate and load a sealed M0 summary without requiring the current model."""
    experiment_root = Path(root).expanduser().resolve() / experiment_id
    validate_m0_artifacts(experiment_root)
    summary = _read_json_object(experiment_root / M0_SUMMARY_FILE)
    manifest = LongContextM0Manifest.model_validate_json(
        (experiment_root / M0_MANIFEST_FILE).read_text(encoding="utf-8")
    )
    identity = {
        "experiment_id": manifest.experiment_id,
        "project_line": manifest.project_line,
        "milestone": manifest.milestone,
        "profile": manifest.profile,
        "evidence_role": manifest.evidence_role,
        "model_tier": manifest.model_tier,
    }
    mismatches = [name for name, value in identity.items() if summary.get(name) != value]
    if experiment_id != manifest.experiment_id:
        mismatches.append("requested_experiment_id")
    if mismatches:
        raise ValueError("M0 summary/manifest identity mismatch: " + ", ".join(mismatches))
    return summary
