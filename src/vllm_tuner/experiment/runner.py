"""High-level reproducible SLOTune search, repeat, holdout, and report runner."""

from __future__ import annotations

import asyncio
import json
import logging
import math
import random
import statistics
import subprocess
from pathlib import Path
from typing import Any, Mapping, Optional

import pandas as pd  # type: ignore[import-untyped]

from vllm_tuner.benchmarks.alpaca import create_alpaca_workload
from vllm_tuner.config.models import TuningConfig
from vllm_tuner.reporting.plots import summarize_capacity_rows
from vllm_tuner.reporting.report import generate_report
from vllm_tuner.runtime.controller import TrialController
from vllm_tuner.runtime.failures import UnsafeCleanupError
from vllm_tuner.runtime.server import uses_slotune_scheduler
from vllm_tuner.scheduling import (
    DEFAULT_FIXED_BUDGETS,
    SimulationConfig,
    SimulationRequest,
    run_budget_ablation,
)
from vllm_tuner.tuning.optimizer import (
    ConstrainedSearchController,
    SearchMethod,
    SearchRun,
    SearchTrial,
)
from vllm_tuner.tuning.search_space import VLLMSearchSpace
from vllm_tuner.workloads.generator import generate_trace
from vllm_tuner.workloads.profiles import PROFILES
from vllm_tuner.workloads.trace import TraceEntry, WorkloadTrace

from .artifacts import EXPERIMENT_INTEGRITY_FILE, SUMMARY_COMPACT_FILE, ArtifactStore
from .manifest import (
    build_manifest,
    git_state,
    source_tree_sha256,
    validate_resume_manifest,
)
from .models import (
    ExperimentSpec,
    TrialResult,
    TrialStatus,
    trial_provenance,
    utc_now_iso,
)

logger = logging.getLogger(__name__)

VALIDATED_METRICS = (
    "goodput_requests_per_sec",
    "offered_requests_per_sec",
    "achieved_requests_per_sec",
    "p99_ttft_ms",
    "p99_tpot_ms",
    "p99_e2e_ms",
    "peak_memory_mb",
    "mean_gpu_utilization_percent",
)


class SLOTuneExperimentRunner:
    """Own one immutable experiment from trace generation through holdout report."""

    def __init__(
        self,
        config: TuningConfig,
        experiment_id: str,
        *,
        results_root: str | Path = "/root/autodl-tmp/slotune-results",
        repository: str | Path = ".",
        trace_path: Optional[str | Path] = None,
        holdout_trace_path: Optional[str | Path] = None,
        tokenizer: Optional[Any] = None,
        require_clean_source: bool = False,
    ) -> None:
        self.config = config
        self.experiment_id = experiment_id
        self.results_root = Path(results_root).expanduser().resolve()
        self.repository = Path(repository).resolve()
        self.trace_path = Path(trace_path).resolve() if trace_path else None
        self.holdout_trace_path = Path(holdout_trace_path).resolve() if holdout_trace_path else None
        self._tokenizer = tokenizer
        self.require_clean_source = require_clean_source
        self.artifacts = ArtifactStore(self.results_root, experiment_id)
        self.search_space = VLLMSearchSpace(config)
        self.manifest: Optional[ExperimentSpec] = None
        self._artifact_warnings: list[str] = []
        self._root_integrity_validated = False

    def _load_tokenizer(self) -> Any:
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(
                self.config.tokenizer or self.config.model,
                revision=self.config.model_revision,
                local_files_only=Path(self.config.model).exists(),
            )
        return self._tokenizer

    @staticmethod
    def _read_local_prompts(path: Path) -> list[str]:
        if path.suffix.lower() == ".jsonl":
            rows = [
                json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line
            ]
        elif path.suffix.lower() == ".json":
            value = json.loads(path.read_text(encoding="utf-8"))
            rows = value if isinstance(value, list) else [value]
        else:
            raise ValueError("local prompt data must be JSON or JSONL")
        prompts: list[str] = []
        for row in rows:
            instruction = str(row.get("instruction", row.get("prompt", "")))
            extra = str(row.get("input", ""))
            prompt = f"{instruction}\n\n{extra}" if extra else instruction
            if prompt.strip():
                prompts.append(prompt)
        return prompts

    async def _dataset_trace(self, seed: int) -> WorkloadTrace:
        path = Path(self.config.workload.dataset_name).expanduser()
        if path.is_file():
            prompts = self._read_local_prompts(path)
        else:
            workload = create_alpaca_workload(self.config.workload)
            prompts = await workload.get_prompts()
        prompts = prompts[: self.config.workload.sample_size]
        if not prompts:
            raise ValueError("workload did not contain any non-empty prompts")
        tokenizer = self._load_tokenizer()
        timing = generate_trace(
            "chat",
            count=len(prompts),
            request_rate=self.config.workload.request_rate,
            burstiness=self.config.workload.burstiness,
            seed=seed,
        )
        entries = [
            TraceEntry(
                request_id=f"{self.config.workload.name}-{index:06d}",
                scheduled_offset_seconds=timing.entries[index].scheduled_offset_seconds,
                prompt=prompt,
                input_tokens=len(tokenizer.encode(prompt, add_special_tokens=False)),
                output_tokens=self.config.workload.max_tokens,
                profile=self.config.workload.name,
            )
            for index, prompt in enumerate(prompts)
        ]
        return WorkloadTrace(
            seed=seed,
            profile=self.config.workload.name,
            request_rate=self.config.workload.request_rate,
            burstiness=self.config.workload.burstiness,
            entries=entries,
        )

    async def _prepare_trace(self, *, holdout: bool = False) -> WorkloadTrace:
        seed = self.config.workload.seed + (1009 if holdout else 0)
        supplied = self.holdout_trace_path if holdout else self.trace_path
        if supplied is not None:
            return WorkloadTrace.read(
                supplied,
                seed=seed,
                profile=self.config.workload.name,
                request_rate=self.config.workload.request_rate,
                burstiness=self.config.workload.burstiness,
            )
        if self.config.workload.name in PROFILES:
            return generate_trace(
                self.config.workload.name,
                count=self.config.workload.sample_size,
                request_rate=self.config.workload.request_rate,
                burstiness=self.config.workload.burstiness,
                seed=seed,
                tokenizer=self._load_tokenizer(),
                fixed_input_tokens=self.config.workload.fixed_input_tokens,
                fixed_output_tokens=self.config.workload.fixed_output_tokens,
            )
        return await self._dataset_trace(seed)

    def _trace_file(self, trace: WorkloadTrace, label: str) -> Path:
        directory = self.results_root / "_traces"
        directory.mkdir(parents=True, exist_ok=True)
        return trace.write(directory / f"{self.experiment_id}-{label}.jsonl")

    @staticmethod
    def _capacity_rate_slug(rate: float) -> str:
        return format(rate, ".12g").replace("-", "m").replace("+", "").replace(".", "p")

    @staticmethod
    def _empirical_scheduled_rate(trace: WorkloadTrace) -> Optional[float]:
        """Return the realized request rate of a finite persisted schedule."""
        if len(trace.entries) < 2:
            return None
        scheduled_span = (
            trace.entries[-1].scheduled_offset_seconds - trace.entries[0].scheduled_offset_seconds
        )
        if scheduled_span <= 0:
            return None
        return (len(trace.entries) - 1) / scheduled_span

    @staticmethod
    def _capacity_trace(source: WorkloadTrace, request_rate: float) -> WorkloadTrace:
        """Reuse exact requests while generating one deterministic open-loop schedule."""
        if not math.isfinite(request_rate) or request_rate <= 0:
            raise ValueError("capacity request rates must be finite and positive")
        rng = random.Random(source.seed + 17_071)
        shape = 1.0 / (source.burstiness * source.burstiness)
        unit_scale = 1.0 / shape
        offset = 0.0
        entries: list[TraceEntry] = []
        for index, entry in enumerate(source.entries):
            if index:
                unit_interval = rng.gammavariate(shape, unit_scale)
                offset += unit_interval / request_rate
            entries.append(entry.model_copy(update={"scheduled_offset_seconds": round(offset, 9)}))
        return WorkloadTrace(
            seed=source.seed,
            profile=source.profile,
            request_rate=request_rate,
            burstiness=source.burstiness,
            entries=entries,
        )

    def _save_capacity_trace(
        self,
        trace: WorkloadTrace,
        trial_id: str,
        rate_slug: str,
        repeat: int,
    ) -> dict[str, str]:
        content = "\n".join(trace.iter_jsonl()) + "\n"
        trace_relative = Path("trials") / trial_id / "capacity-trace.jsonl"
        checksum_relative = Path("trials") / trial_id / "capacity-trace.sha256"
        point_relative = Path("trials") / trial_id / "capacity-point.json"
        self.artifacts.write_text(trace_relative, content)
        self.artifacts.write_text(
            checksum_relative,
            f"{trace.checksum()}  capacity-trace.jsonl\n",
        )
        self.artifacts.write_json(
            point_relative,
            {
                # Keep the established field as a target-valued compatibility alias.
                "offered_requests_per_sec": trace.request_rate,
                "target_offered_requests_per_sec": trace.request_rate,
                "empirical_scheduled_requests_per_sec": self._empirical_scheduled_rate(trace),
                "repeat": repeat,
                "trace_sha256": trace.checksum(),
                "server_parameters": self.search_space.get_default_params(),
            },
        )
        aggregate_trace = Path("aggregate") / "capacity-traces" / f"rate-{rate_slug}.jsonl"
        aggregate_checksum = Path("aggregate") / "capacity-traces" / f"rate-{rate_slug}.sha256"
        if not (self.artifacts.root / aggregate_trace).exists():
            self.artifacts.write_text(aggregate_trace, content)
            self.artifacts.write_text(
                aggregate_checksum,
                f"{trace.checksum()}  rate-{rate_slug}.jsonl\n",
            )
        return {
            "capacity-trace.jsonl": str(trace_relative),
            "capacity-trace.sha256": str(checksum_relative),
            "capacity-point.json": str(point_relative),
        }

    @staticmethod
    def _summary_stat(namespace: Mapping[str, Any], metric: str, statistic: str) -> Any:
        value = namespace.get(metric)
        return value.get(statistic) if isinstance(value, Mapping) else None

    @staticmethod
    def _capacity_row(result: TrialResult, request_rate: float, repeat: int) -> dict[str, Any]:
        error_types = result.client.get("error_types")
        empirical_scheduled_rate = result.client.get("empirical_scheduled_requests_per_sec")
        return {
            "trial_id": result.trial_id,
            "repeat": repeat,
            # Deprecated target-valued aliases retained for existing JSON/Parquet readers.
            "offered_requests_per_sec": request_rate,
            "measured_offered_requests_per_sec": request_rate,
            "target_offered_requests_per_sec": request_rate,
            "empirical_scheduled_requests_per_sec": empirical_scheduled_rate,
            "achieved_requests_per_sec": result.client.get("achieved_requests_per_sec"),
            "goodput_requests_per_sec": result.client.get("goodput_requests_per_sec"),
            "request_throughput": result.client.get("request_throughput"),
            "output_throughput": result.client.get("output_throughput"),
            "total_token_throughput": result.client.get("total_token_throughput"),
            "p50_ttft_ms": result.client.get("p50_ttft_ms"),
            "p95_ttft_ms": result.client.get("p95_ttft_ms"),
            "p99_ttft_ms": result.client.get("p99_ttft_ms"),
            "p50_tpot_ms": result.client.get("p50_tpot_ms"),
            "p95_tpot_ms": result.client.get("p95_tpot_ms"),
            "p99_tpot_ms": result.client.get("p99_tpot_ms"),
            "p50_e2e_ms": result.client.get("p50_e2e_ms"),
            "p95_e2e_ms": result.client.get("p95_e2e_ms"),
            "p99_e2e_ms": result.client.get("p99_e2e_ms"),
            "completed_requests": result.client.get("completed"),
            "failed_requests": result.client.get("failed"),
            "error_rate": result.client.get("error_rate"),
            "timeout_count": (
                error_types.get("timeout", 0) if isinstance(error_types, Mapping) else None
            ),
            "peak_waiting_requests": SLOTuneExperimentRunner._summary_stat(
                result.engine, "num_requests_waiting", "peak"
            ),
            "peak_kv_cache_usage": SLOTuneExperimentRunner._summary_stat(
                result.engine, "kv_cache_usage_perc", "peak"
            ),
            "preemptions": SLOTuneExperimentRunner._summary_stat(
                result.engine, "num_preemptions_total", "delta"
            ),
            "peak_memory_mb": result.gpu.get("peak_memory_mb"),
            "p95_memory_mb": result.gpu.get("p95_memory_mb"),
            "mean_gpu_utilization_percent": result.gpu.get("mean_gpu_utilization_percent"),
            "energy_joules": result.gpu.get("energy_joules"),
            "energy_per_output_token_joules": result.gpu.get("energy_per_output_token_joules"),
            "status": result.status.value,
            "feasible": result.constraints.get("feasible", False),
            "failure_reason": result.failure_reason,
            "parameters": result.params,
        }

    async def _run_capacity_sweep(self, source: WorkloadTrace) -> list[dict[str, Any]]:
        rates = list(getattr(self.config.workload, "capacity_request_rates", []))
        repeats = int(getattr(self.config.workload, "capacity_repeats", 1))
        if not rates:
            return []
        if repeats < 1:
            raise ValueError("capacity_repeats must be at least 1")

        rows: list[dict[str, Any]] = []
        default_params = self.search_space.get_default_params()
        jobs = [(float(raw_rate), repeat) for raw_rate in rates for repeat in range(repeats)]
        order_rng = random.Random(self.config.study.seed + 41_009)
        order_rng.shuffle(jobs)
        for request_rate, repeat in jobs:
            capacity_trace = self._capacity_trace(source, request_rate)
            rate_slug = self._capacity_rate_slug(request_rate)
            trial_id = f"capacity-rate-{rate_slug}-repeat-{repeat}"
            trace_artifacts = self._save_capacity_trace(capacity_trace, trial_id, rate_slug, repeat)
            result = self._load_cached_trial(dict(default_params), trial_id)
            if result is None:
                try:
                    controller = TrialController(
                        self.config,
                        capacity_trace,
                        self.artifacts,
                        tokenizer=self._load_tokenizer(),
                    )
                    result = await controller.run_trial(dict(default_params), trial_id, "capacity")
                except UnsafeCleanupError:
                    raise
                except Exception as error:
                    result = TrialResult(
                        trial_id=trial_id,
                        **trial_provenance(trial_id, "capacity"),
                        status=TrialStatus.FAILED,
                        params=dict(default_params),
                        finished_at=utc_now_iso(),
                        client={
                            "offered_requests_per_sec": request_rate,
                            "target_offered_requests_per_sec": request_rate,
                            "empirical_scheduled_requests_per_sec": (
                                self._empirical_scheduled_rate(capacity_trace)
                            ),
                        },
                        constraints={"feasible": False, "violations": ["capacity_point_error"]},
                        failure_reason={
                            "type": type(error).__name__,
                            "message": str(error),
                            "phase": "CAPACITY_SWEEP",
                        },
                    )
            result.artifacts.update(trace_artifacts)
            for field, value in trial_provenance(trial_id, result.method).items():
                setattr(result, field, value)
            self._finalize_trial_artifacts(result)
            rows.append(self._capacity_row(result, request_rate, repeat))
        return rows

    @staticmethod
    def _command_output(command: list[str], cwd: Path) -> str:
        try:
            completed = subprocess.run(
                command,
                cwd=cwd,
                check=False,
                capture_output=True,
                text=True,
                timeout=60,
            )
            return completed.stdout + completed.stderr
        except (OSError, subprocess.SubprocessError) as error:
            return f"unavailable: {type(error).__name__}: {error}\n"

    def _save_environment(self) -> None:
        _, _, status = git_state(self.repository)
        self.artifacts.write_text("environment/git-state.txt", status or "clean working tree\n")
        self.artifacts.write_text(
            "environment/python-packages.txt",
            self._command_output(
                [
                    str(Path(self.repository) / ".venv/bin/python"),
                    "-m",
                    "pip",
                    "freeze",
                ],
                self.repository,
            ),
        )
        self.artifacts.write_text(
            "environment/nvidia-smi.txt",
            self._command_output(["nvidia-smi"], self.repository),
        )
        self.artifacts.write_text(
            "environment/collect-env.txt",
            self._command_output(
                [
                    str(Path(self.repository) / ".venv/bin/python"),
                    "-m",
                    "torch.utils.collect_env",
                ],
                self.repository,
            ),
        )

    def _initialize_artifacts(self, trace_path: Path, holdout_trace_path: Path) -> ExperimentSpec:
        root_preexisting = self.artifacts.root.exists()
        requested = build_manifest(
            experiment_id=self.experiment_id,
            model=self.config.model,
            model_revision=self.config.model_revision,
            tokenizer=self.config.tokenizer,
            trace_path=trace_path,
            workload=self.config.workload.model_dump(mode="json"),
            slo=self.config.slo.model_dump(mode="json"),
            constraints=self.config.constraints.model_dump(mode="json"),
            gpu_config=self.config.gpu.model_dump(mode="json"),
            telemetry=self.config.telemetry.model_dump(mode="json"),
            study=self.config.study.model_dump(mode="json", exclude={"resume"}),
            vllm_args=self.config.vllm_args,
            search_space=self.search_space.manifest(),
            seed=self.config.study.seed,
            repository=self.repository,
            holdout_trace_path=holdout_trace_path,
        )
        if self.require_clean_source:
            if requested.source_commit is None or requested.source_tree_sha256 is None:
                raise ValueError(
                    "Formal experiments require a Git repository with a resolvable source identity"
                )
            if requested.dirty_worktree:
                raise ValueError(
                    "Formal experiments require a clean Git worktree; commit or stash all "
                    "tracked and untracked source changes, or use --allow-dirty-source only "
                    "for development runs"
                )
        self.artifacts.initialize(exist_ok=self.config.study.resume)
        manifest_path = self.artifacts.root / "manifest.json"
        if manifest_path.exists():
            existing = ExperimentSpec.model_validate_json(manifest_path.read_text(encoding="utf-8"))
            validate_resume_manifest(existing, requested)
            manifest = existing
            if (self.artifacts.root / EXPERIMENT_INTEGRITY_FILE).is_file():
                self.artifacts.validate_experiment_integrity()
                self._root_integrity_validated = True
        else:
            if root_preexisting:
                raise ValueError(
                    f"Cannot resume {self.experiment_id}: existing artifact directory has no manifest"
                )
            manifest = requested
            self.artifacts.save_manifest(manifest)
            self.artifacts.write_yaml("experiment.yaml", self.config.model_dump(mode="json"))
            self.artifacts.save_trace(trace_path)
            self.artifacts.save_holdout_trace(holdout_trace_path)
            self._save_environment()
        self.manifest = manifest
        return manifest

    def _telemetry_evidence_errors(self, result: TrialResult) -> list[str]:
        """Return semantic telemetry gaps that make a selectable trial unverifiable."""
        if not self.config.telemetry.enabled:
            return []

        errors: list[str] = []
        engine = result.engine
        sample_count = engine.get("sample_count")
        successful_count = engine.get("successful_sample_count")
        if not isinstance(sample_count, int) or sample_count < 2:
            errors.append("engine.sample_count<2")
        if not isinstance(successful_count, int) or successful_count < 2:
            errors.append("engine.successful_sample_count<2")
        if (
            isinstance(sample_count, int)
            and sample_count > 0
            and isinstance(successful_count, int)
            and successful_count / sample_count < 0.8
        ):
            errors.append("engine.sample_coverage<0.8")

        for name in (
            "num_requests_running",
            "num_requests_waiting",
            "kv_cache_usage_perc",
        ):
            summary = engine.get(name)
            if not isinstance(summary, Mapping) or summary.get("available") is not True:
                errors.append(f"engine.{name}")

        for name in (
            "num_preemptions_total",
            "prompt_tokens_total",
            "generation_tokens_total",
        ):
            summary = engine.get(name)
            if (
                not isinstance(summary, Mapping)
                or summary.get("available") is not True
                or not isinstance(summary.get("delta"), (int, float))
            ):
                errors.append(f"engine.{name}.delta")

        if self.config.telemetry.collect_nvml:
            gpu = result.gpu
            gpu_samples = gpu.get("sample_count")
            if not isinstance(gpu_samples, int) or gpu_samples < 2:
                errors.append("gpu.sample_count<2")
            for name in ("memory_used_mb", "gpu_utilization_percent"):
                summary = gpu.get(name)
                if not isinstance(summary, Mapping) or summary.get("available") is not True:
                    errors.append(f"gpu.{name}")
            if not isinstance(gpu.get("peak_memory_mb"), (int, float)):
                errors.append("gpu.peak_memory_mb")
            if not isinstance(gpu.get("mean_gpu_utilization_percent"), (int, float)):
                errors.append("gpu.mean_gpu_utilization_percent")
            if self.config.telemetry.collect_energy:
                power = gpu.get("power_w")
                if not isinstance(power, Mapping) or power.get("available") is not True:
                    errors.append("gpu.power_w")
                if not isinstance(gpu.get("energy_joules"), (int, float)):
                    errors.append("gpu.energy_joules")
                if not isinstance(gpu.get("energy_per_output_token_joules"), (int, float)):
                    errors.append("gpu.energy_per_output_token_joules")
        return errors

    def _finalize_trial_artifacts(self, result: TrialResult) -> None:
        status = self.artifacts.ensure_trial_artifacts(result)
        if status["degraded"]:
            warning = f"Trial {result.trial_id} has unavailable raw evidence: " + ", ".join(
                status["unavailable_data"]
            )
            if warning not in self._artifact_warnings:
                self._artifact_warnings.append(warning)
        self.artifacts.validate_trial_artifacts(
            result.trial_id,
            require_telemetry=self.config.telemetry.enabled,
        )
        semantic_error: Optional[str] = None
        try:
            self.artifacts.validate_cached_trial(
                result,
                require_telemetry=self.config.telemetry.enabled,
            )
        except ValueError as error:
            semantic_error = str(error)

        required_evidence = {
            "server-command.json",
            "request-results.jsonl",
            "benchmark-raw.json",
            "server.log",
            "cleanup.json",
        }
        if self.config.telemetry.enabled:
            required_evidence.add("prometheus.jsonl")
            if self.config.telemetry.collect_nvml:
                required_evidence.add("nvml.jsonl")
        if (
            uses_slotune_scheduler(self.config)
            and self.config.adaptive_prefill.decision_log_enabled
        ):
            required_evidence.add("scheduler-decisions.jsonl")
        unavailable_required = sorted(
            name for name in required_evidence if not status["files"][name]["data_available"]
        )
        unavailable_required.extend(self._telemetry_evidence_errors(result))
        candidate_complete = result.status == TrialStatus.COMPLETE and bool(
            result.constraints.get("feasible", False)
        )
        cleanup = result.cleanup_status
        cleanup_unverified = (
            not isinstance(cleanup, Mapping)
            or any(
                cleanup.get(field) is not True
                for field in (
                    "clean",
                    "process_group_empty",
                    "port_available",
                    "gpu_clean",
                )
            )
            or not status["files"]["cleanup.json"]["data_available"]
        )

        def reseal_failed_result(reason: str) -> None:
            self.artifacts.record_artifact_finalizer_failure(result, reason)
            self.artifacts.seal_trial_artifacts(
                result,
                missing_before=status["missing_before_finalize"],
            )
            self.artifacts.validate_trial_artifacts(
                result.trial_id,
                require_telemetry=self.config.telemetry.enabled,
            )
            self.artifacts.validate_cached_trial(
                result,
                require_telemetry=self.config.telemetry.enabled,
            )

        if candidate_complete and cleanup_unverified:
            violations = list(result.constraints.get("violations", []))
            if "cleanup_error" not in violations:
                violations.append("cleanup_error")
            result.status = TrialStatus.FAILED
            result.constraints = {
                **result.constraints,
                "feasible": False,
                "violations": violations,
            }
            result.failure_reason = {
                "type": "CLEANUP_ERROR",
                "message": "Process-group, port, and GPU cleanup was not fully verified",
                "phase": "ARTIFACT_FINALIZE",
            }
            reseal_failed_result(result.failure_reason["message"])
        elif candidate_complete and unavailable_required:
            violations = list(result.constraints.get("violations", []))
            if "artifact_unavailable" not in violations:
                violations.append("artifact_unavailable")
            result.status = TrialStatus.FAILED
            result.constraints = {
                **result.constraints,
                "feasible": False,
                "violations": violations,
            }
            result.failure_reason = {
                "type": "ARTIFACT_UNAVAILABLE",
                "message": "Required raw evidence is unavailable: "
                + ", ".join(unavailable_required),
                "phase": "ARTIFACT_FINALIZE",
            }
            reseal_failed_result(result.failure_reason["message"])
        elif candidate_complete and semantic_error is not None:
            violations = list(result.constraints.get("violations", []))
            if "artifact_inconsistent" not in violations:
                violations.append("artifact_inconsistent")
            result.status = TrialStatus.FAILED
            result.constraints = {
                **result.constraints,
                "feasible": False,
                "violations": violations,
            }
            result.failure_reason = {
                "type": "ARTIFACT_INCONSISTENT",
                "message": "Raw artifacts disagree with the terminal trial summary",
                "phase": "ARTIFACT_FINALIZE",
                "artifact_inconsistency": semantic_error,
            }
            reseal_failed_result(result.failure_reason["message"])

    def _load_cached_trial(self, params: dict[str, Any], trial_id: str) -> Optional[TrialResult]:
        """Replay one valid terminal result when explicit resume is enabled."""
        if not self.config.study.resume:
            return None
        try:
            cached = self.artifacts.load_trial_result(trial_id)
        except ValueError as error:
            warning = f"Cached trial {trial_id} will be re-run: {error}"
            if warning not in self._artifact_warnings:
                self._artifact_warnings.append(warning)
            return None
        if cached is None:
            return None
        if cached.params != params:
            raise ValueError(
                f"Cached trial {trial_id} parameter mismatch: "
                f"expected {params!r}, found {cached.params!r}"
            )
        try:
            self.artifacts.validate_trial_artifacts(trial_id, require_telemetry=True)
            self.artifacts.validate_cached_trial(
                cached,
                require_telemetry=self.config.telemetry.enabled,
            )
            if cached.selectable:
                required_evidence = {
                    "server-command.json",
                    "request-results.jsonl",
                    "benchmark-raw.json",
                    "server.log",
                }
                if self.config.telemetry.enabled:
                    required_evidence.add("prometheus.jsonl")
                    if self.config.telemetry.collect_nvml:
                        required_evidence.add("nvml.jsonl")
                if (
                    uses_slotune_scheduler(self.config)
                    and self.config.adaptive_prefill.decision_log_enabled
                ):
                    required_evidence.add("scheduler-decisions.jsonl")
                self.artifacts.validate_trial_artifacts(
                    trial_id,
                    require_telemetry=False,
                    require_available=True,
                    required_evidence=required_evidence,
                )
        except ValueError as error:
            warning = f"Cached trial {trial_id} will be re-run: {error}"
            if warning not in self._artifact_warnings:
                self._artifact_warnings.append(warning)
            return None
        logger.info("Replaying cached terminal trial %s", trial_id)
        return cached

    def _read_trial_jsonl(self, trial_id: str, name: str) -> list[dict[str, Any]]:
        path = self.artifacts.trials_dir / trial_id / name
        if not path.exists():
            self._artifact_warnings.append(f"Telemetry source {trial_id}/{name} is unavailable")
            return []
        rows: list[dict[str, Any]] = []
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                self._artifact_warnings.append(
                    f"Ignored invalid JSONL at {trial_id}/{name}:{line_number}: {error}"
                )
                continue
            if isinstance(value, dict):
                rows.append(value)
        return rows

    def _telemetry_series(
        self, best: Optional[SearchTrial]
    ) -> tuple[Optional[str], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
        if best is None or best.result is None:
            self._artifact_warnings.append(
                "No selectable trial was available for the telemetry timeline"
            )
            return None, [], [], []
        trial_id = best.result.trial_id
        return (
            trial_id,
            self._read_trial_jsonl(trial_id, "request-results.jsonl"),
            self._read_trial_jsonl(trial_id, "prometheus.jsonl"),
            self._read_trial_jsonl(trial_id, "nvml.jsonl"),
        )

    @staticmethod
    def _trial_dict(trial: SearchTrial) -> dict[str, Any]:
        result = trial.result
        row: dict[str, Any] = {
            "trial_number": trial.number,
            "method": trial.method.value,
            "parameters": trial.params,
            "status": trial.status.value,
            "objective": trial.objective,
            "repeat_of": trial.repeat_of,
            "holdout": trial.holdout,
            "failure_reason": trial.failure_reason,
        }
        if result is not None:
            row.update(
                {
                    "trial_id": result.trial_id,
                    "goodput_requests_per_sec": result.client.get("goodput_requests_per_sec"),
                    "offered_requests_per_sec": result.client.get("offered_requests_per_sec"),
                    "achieved_requests_per_sec": result.client.get("achieved_requests_per_sec"),
                    "p99_ttft_ms": result.client.get("p99_ttft_ms"),
                    "p99_tpot_ms": result.client.get("p99_tpot_ms"),
                    "p99_e2e_ms": result.client.get("p99_e2e_ms"),
                    "peak_memory_mb": result.gpu.get("peak_memory_mb"),
                    "mean_gpu_utilization_percent": result.gpu.get("mean_gpu_utilization_percent"),
                    "feasible": result.constraints.get("feasible", False),
                }
            )
        return row

    @classmethod
    def _aggregate_validated_metrics(
        cls, trials: list[SearchTrial]
    ) -> dict[str, dict[str, float | int]]:
        """Aggregate only repeated COMPLETE/feasible evidence for one candidate."""
        rows = [cls._trial_dict(trial) for trial in trials if trial.selectable]
        aggregates: dict[str, dict[str, float | int]] = {}
        for metric in VALIDATED_METRICS:
            values = [
                float(row[metric])
                for row in rows
                if isinstance(row.get(metric), (int, float)) and math.isfinite(float(row[metric]))
            ]
            if values:
                aggregates[metric] = {
                    "count": len(values),
                    "median": statistics.median(values),
                    "min": min(values),
                    "max": max(values),
                }
        return aggregates

    def _save_table(self, relative: str, rows: list[dict[str, Any]]) -> Path:
        path = self.artifacts.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        frame_rows = []
        for row in rows:
            normalized = dict(row)
            for key, value in list(normalized.items()):
                if isinstance(value, (dict, list)):
                    normalized[key] = json.dumps(value, sort_keys=True)
            frame_rows.append(normalized)
        pd.DataFrame(frame_rows).to_parquet(path, index=False)
        return path

    @staticmethod
    def _top_candidates(runs: dict[SearchMethod, SearchRun], count: int) -> list[SearchTrial]:
        candidates: list[SearchTrial] = []
        for method in (SearchMethod.DEFAULT, SearchMethod.RANDOM):
            run = runs.get(method)
            if run is not None and run.best is not None:
                candidates.append(run.best)
        tpe = runs.get(SearchMethod.TPE)
        if tpe is not None:
            selectable = sorted(
                (trial for trial in tpe.trials if trial.selectable),
                key=lambda trial: trial.objective_value,
                reverse=True,
            )
            candidates.extend(selectable[:count])
        unique: list[SearchTrial] = []
        seen: set[str] = set()
        for candidate in candidates:
            # Validation is method-specific: the protocol promises independent
            # default, random, and TPE repeat/holdout evidence even when seeded
            # samplers happen to suggest the same parameter mapping. Deduplicate
            # only repeated configurations within one method.
            key = f"{candidate.method.value}:" + json.dumps(candidate.params, sort_keys=True)
            if key not in seen:
                seen.add(key)
                unique.append(candidate)
        return unique

    def _validate_candidates(
        self,
        candidates: list[SearchTrial],
        repeated: list[SearchTrial],
        holdouts: list[SearchTrial],
    ) -> tuple[list[dict[str, Any]], Optional[SearchTrial], Optional[dict[str, Any]]]:
        """Gate final selection on every configured repeat and holdout run."""
        validation_rows: list[dict[str, Any]] = []
        validated: list[tuple[float, float, SearchTrial, dict[str, Any]]] = []
        expected = self.config.study.repeat_count
        minimum_ratio = self.config.study.holdout_min_goodput_ratio

        for candidate in candidates:
            repeat_rows = [
                trial
                for trial in repeated
                if trial.method == candidate.method
                and trial.repeat_of == candidate.number
                and trial.params == candidate.params
                and not trial.holdout
            ]
            holdout_rows = [
                trial
                for trial in holdouts
                if trial.method == candidate.method
                and trial.repeat_of == candidate.number
                and trial.params == candidate.params
                and trial.holdout
            ]
            repeat_values = [trial.objective_value for trial in repeat_rows if trial.selectable]
            holdout_values = [trial.objective_value for trial in holdout_rows if trial.selectable]
            repeats_passed = len(repeat_rows) == expected and len(repeat_values) == expected
            holdout_required = self.config.study.holdout_enabled
            holdout_passed = not holdout_required or (
                len(holdout_rows) == expected and len(holdout_values) == expected
            )
            repeat_median = statistics.median(repeat_values) if repeat_values else None
            holdout_median = statistics.median(holdout_values) if holdout_values else None
            repeat_metrics = self._aggregate_validated_metrics(repeat_rows)
            holdout_metrics = self._aggregate_validated_metrics(holdout_rows)
            ratio: Optional[float] = None
            if repeat_median is not None and holdout_median is not None:
                ratio = holdout_median / repeat_median if repeat_median > 0 else 1.0
            degradation_passed = not holdout_required or (
                ratio is not None and ratio >= minimum_ratio
            )
            reasons: list[str] = []
            if not repeats_passed:
                reasons.append("not_all_repeats_complete_and_feasible")
            if not holdout_passed:
                reasons.append("not_all_holdouts_complete_and_feasible")
            if holdout_passed and not degradation_passed:
                reasons.append("holdout_goodput_degraded")
            is_validated = repeats_passed and holdout_passed and degradation_passed
            row = {
                "candidate": f"{candidate.method.value}-{candidate.number}",
                "method": candidate.method.value,
                "trial_number": candidate.number,
                "parameters": candidate.params,
                "search_goodput_requests_per_sec": candidate.objective,
                "repeat_required": expected,
                "repeat_complete_feasible": len(repeat_values),
                "repeat_median_goodput_requests_per_sec": repeat_median,
                "repeat_min_goodput_requests_per_sec": min(repeat_values, default=None),
                "repeat_max_goodput_requests_per_sec": max(repeat_values, default=None),
                "repeat_metrics": repeat_metrics,
                "holdout_required": holdout_required,
                "holdout_complete_feasible": len(holdout_values),
                "holdout_median_goodput_requests_per_sec": holdout_median,
                "holdout_min_goodput_requests_per_sec": min(holdout_values, default=None),
                "holdout_max_goodput_requests_per_sec": max(holdout_values, default=None),
                "holdout_metrics": holdout_metrics,
                "holdout_to_repeat_goodput_ratio": ratio,
                "minimum_holdout_goodput_ratio": minimum_ratio,
                "validated": is_validated,
                "rejection_reasons": reasons,
            }
            validation_rows.append(row)
            if is_validated and repeat_median is not None:
                validated.append((repeat_median, holdout_median or 0.0, candidate, row))

        if not validated:
            return validation_rows, None, None
        _, _, best_candidate, best_validation = max(
            validated,
            key=lambda item: (item[0], item[1], item[2].objective_value),
        )
        search_observation = self._trial_dict(best_candidate)
        repeat_metrics = best_validation["repeat_metrics"]
        best = {
            "candidate": best_validation["candidate"],
            "method": best_candidate.method.value,
            "parameters": best_candidate.params,
            "status": TrialStatus.COMPLETE.value,
            "feasible": True,
            "validated": True,
            "metric_provenance": "median_of_complete_feasible_repeats",
            "search_observation": search_observation,
            "repeat_required": best_validation["repeat_required"],
            "repeat_complete_feasible": best_validation["repeat_complete_feasible"],
            "repeat_metrics": repeat_metrics,
            "holdout_required": best_validation["holdout_required"],
            "holdout_complete_feasible": best_validation["holdout_complete_feasible"],
            "holdout_metrics": best_validation["holdout_metrics"],
            "holdout_to_repeat_goodput_ratio": best_validation["holdout_to_repeat_goodput_ratio"],
            "minimum_holdout_goodput_ratio": best_validation["minimum_holdout_goodput_ratio"],
        }
        for metric, aggregate in repeat_metrics.items():
            best[metric] = aggregate["median"]
        return validation_rows, best_candidate, best

    def _scheduler_ablation(self, trace: WorkloadTrace, holdout: WorkloadTrace) -> dict[str, Any]:
        def requests(source: WorkloadTrace) -> list[SimulationRequest]:
            return [
                SimulationRequest(
                    request_id=entry.request_id,
                    arrival_time=entry.scheduled_offset_seconds,
                    prompt_tokens=entry.input_tokens,
                    output_tokens=entry.output_tokens,
                    ttft_slo=(self.config.slo.ttft_ms or 1000) / 1000,
                    tpot_slo=(self.config.slo.tpot_ms or 100) / 1000,
                    e2e_slo=(self.config.slo.e2e_ms or 10000) / 1000,
                )
                for entry in source.entries
            ]

        simulation = SimulationConfig(
            seed=self.config.study.seed,
            ttft_slo=(self.config.slo.ttft_ms or 1000) / 1000,
            tpot_slo=(self.config.slo.tpot_ms or 100) / 1000,
            e2e_slo=(self.config.slo.e2e_ms or 10000) / 1000,
        )
        report = run_budget_ablation(
            requests(trace),
            requests(holdout),
            fixed_budgets=DEFAULT_FIXED_BUDGETS,
            simulation_config=simulation,
        )
        value = report.to_dict()
        self.artifacts.write_json("aggregate/scheduler-ablation.json", value)
        return value

    async def run(self) -> dict[str, Any]:
        """Run the complete protocol under the configured experiment timeout."""
        timeout_seconds = self.config.study.timeout_minutes * 60
        try:
            return await asyncio.wait_for(self._run(), timeout=timeout_seconds)
        except TimeoutError as error:
            if self.artifacts.root.exists():
                self.artifacts.write_json(
                    "experiment-failure.json",
                    {
                        "type": "EXPERIMENT_TIMEOUT",
                        "timeout_minutes": self.config.study.timeout_minutes,
                        "finished_at": utc_now_iso(),
                    },
                )
            raise TimeoutError(
                f"Experiment {self.experiment_id} exceeded "
                f"{self.config.study.timeout_minutes} minutes"
            ) from error

    async def _run(self) -> dict[str, Any]:
        """Run search, formal repeats, holdout, scheduler ablation, and report."""
        trace = await self._prepare_trace()
        holdout_trace = await self._prepare_trace(holdout=True)
        if self.config.study.holdout_enabled and trace.checksum() == holdout_trace.checksum():
            raise ValueError(
                "Search and holdout traces are identical; holdout must be excluded from search"
            )
        trace_path = self._trace_file(trace, "search")
        holdout_path = self._trace_file(holdout_trace, "holdout")
        manifest = self._initialize_artifacts(trace_path, holdout_path)

        controller = TrialController(
            self.config,
            trace,
            self.artifacts,
            tokenizer=self._load_tokenizer(),
        )
        search = ConstrainedSearchController(
            self.search_space,
            budget=self.config.study.trial_budget,
            seed=self.config.study.seed,
            n_startup_trials=self.config.study.n_startup_trials,
            prune_enabled=self.config.study.prune_enabled,
        )

        async def evaluator(params: dict[str, Any], trial_id: str):
            cached = self._load_cached_trial(params, trial_id)
            if cached is not None:
                return cached
            result = await controller.run_trial(params, trial_id, trial_id.split("-", 1)[0])
            for field, value in trial_provenance(trial_id, result.method).items():
                setattr(result, field, value)
            self._finalize_trial_artifacts(result)
            return result

        methods = tuple(SearchMethod(method) for method in self.config.study.methods)
        runs = await search.run_all(evaluator, methods)
        all_trials = [trial for run in runs.values() for trial in run.trials]
        trial_rows = [self._trial_dict(trial) for trial in all_trials]
        self._save_table("aggregate/trials.parquet", trial_rows)

        candidates = self._top_candidates(runs, self.config.study.top_candidates)
        repeated = await search.repeat_candidates(
            candidates,
            evaluator,
            repeats=self.config.study.repeat_count,
        )
        repeated_rows = [self._trial_dict(trial) for trial in repeated]
        self._save_table("aggregate/repeated-results.parquet", repeated_rows)

        holdouts: list[SearchTrial] = []
        holdout_rows: list[dict[str, Any]] = []
        if self.config.study.holdout_enabled and candidates:
            holdout_controller = TrialController(
                self.config,
                holdout_trace,
                self.artifacts,
                tokenizer=self._load_tokenizer(),
            )

            async def holdout_evaluator(params: dict[str, Any], trial_id: str):
                cached = self._load_cached_trial(params, trial_id)
                if cached is not None:
                    return cached
                result = await holdout_controller.run_trial(params, trial_id, "holdout")
                for field, value in trial_provenance(trial_id, result.method).items():
                    setattr(result, field, value)
                self._finalize_trial_artifacts(result)
                return result

            holdouts = await search.repeat_candidates(
                candidates,
                holdout_evaluator,
                repeats=self.config.study.repeat_count,
                holdout=True,
            )
            holdout_rows = [self._trial_dict(trial) for trial in holdouts]
        self._save_table("aggregate/holdout-results.parquet", holdout_rows)

        capacity_rows = await self._run_capacity_sweep(trace)
        capacity_summary = summarize_capacity_rows(capacity_rows)
        self._save_table("aggregate/capacity-sweep.parquet", capacity_rows)
        self._save_table("aggregate/capacity-sweep-summary.parquet", capacity_summary)

        scheduler = self._scheduler_ablation(trace, holdout_trace)
        scheduler_path = self.artifacts.aggregate_dir / "scheduler-ablation.json"
        if not scheduler_path.is_file():
            self.artifacts.write_json("aggregate/scheduler-ablation.json", scheduler)
        scheduler_rows: list[dict[str, Any]] = []
        for label, section in (
            ("calibration", scheduler["calibration"]),
            ("held_out", scheduler["held_out"]),
        ):
            scheduler_rows.append(
                {
                    "trace": label,
                    "policy": "adaptive",
                    "budget": None,
                    **section["adaptive"]["metrics"],
                    "goodput_gain_vs_best": section["goodput_gain_vs_best"],
                }
            )
            scheduler_rows.extend(
                {
                    "trace": label,
                    "policy": "fixed",
                    "budget": int(budget),
                    **result["metrics"],
                    "goodput_gain_vs_best": None,
                }
                for budget, result in section.get("fixed_baselines", {}).items()
            )
        search_best = search.best_across(runs)
        validation_rows, validated_candidate, best = self._validate_candidates(
            candidates,
            repeated,
            holdouts,
        )
        self._save_table("aggregate/candidate-validation.parquet", validation_rows)
        telemetry_source, client_series, engine_series, gpu_series = self._telemetry_series(
            validated_candidate
        )
        limitations = [
            "Results apply only to the recorded model, GPU, vLLM version, and traces.",
            "The adaptive policy result is a deterministic simulator result unless a runtime integration artifact is explicitly present.",
        ]
        if best is None:
            limitations.append(
                "No candidate passed every configured repeat and holdout gate; no validated best configuration is claimed."
            )
        report_paths = generate_report(
            self.artifacts.report_dir,
            manifest=manifest.model_dump(mode="json"),
            trials=trial_rows,
            repetitions=repeated_rows,
            holdout=holdout_rows,
            candidate_validation=validation_rows,
            scheduler_results=scheduler_rows,
            capacity_sweep=capacity_rows,
            limitations=limitations,
            client_series=client_series,
            engine_series=engine_series,
            gpu_series=gpu_series,
            telemetry_source=telemetry_source,
            best=best,
            scheduler_negative_conditions=scheduler.get("negative_gain_conditions", []),
        )
        plot_manifest_path = report_paths["plot_manifest"]
        plot_manifest = json.loads(plot_manifest_path.read_text(encoding="utf-8"))
        for plot_name, plot_status in plot_manifest["plots"].items():
            if not plot_status["data_available"]:
                self._artifact_warnings.append(
                    f"Report plot {plot_name} has no measured data: "
                    f"{plot_status['unavailable_reason']}"
                )
            if not plot_status["static_image_available"]:
                self._artifact_warnings.append(
                    f"Report plot {plot_name} uses the HTML fallback: "
                    f"{plot_status['fallback_reason']}"
                )
        manifest.report_artifacts = {
            "files": {
                name: path.relative_to(self.artifacts.root).as_posix()
                for name, path in report_paths.items()
            },
            "plots": plot_manifest["plots"],
        }
        manifest.artifact_warnings = list(self._artifact_warnings)
        self.artifacts.save_manifest(manifest)
        summary = {
            "experiment_id": self.experiment_id,
            "manifest": manifest.model_dump(mode="json"),
            "search": {
                method.value: [self._trial_dict(t) for t in run.trials]
                for method, run in runs.items()
            },
            "search_best": self._trial_dict(search_best) if search_best is not None else None,
            "candidate_validation": validation_rows,
            "best": best,
            "repetitions": repeated_rows,
            "holdout": holdout_rows,
            "capacity_sweep": {
                "points": capacity_rows,
                "by_rate": capacity_summary,
            },
            "scheduler_ablation": scheduler,
            "report": {name: str(path) for name, path in report_paths.items()},
        }
        self.artifacts.write_json("summary.json", summary)
        integrity_exists = (self.artifacts.root / EXPERIMENT_INTEGRITY_FILE).is_file()
        attestation_commit, attestation_dirty, _ = git_state(self.repository)
        self.artifacts.attest_experiment_artifacts(
            attestation={
                "kind": "runner-completion",
                "artifact_schema_version": manifest.artifact_schema_version,
                "attestation_source_commit": attestation_commit,
                "attestation_source_tree_sha256": source_tree_sha256(self.repository),
                "attestation_dirty_worktree": attestation_dirty,
            },
            reseal=integrity_exists,
            validate_existing=not self._root_integrity_validated,
        )
        return json.loads((self.artifacts.root / SUMMARY_COMPACT_FILE).read_text(encoding="utf-8"))
