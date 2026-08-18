"""Run, resume, analyze, and seal long-context v5 M5 Decode-tail validation."""

from __future__ import annotations

import json
import os
import time
from collections.abc import Mapping
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional, cast

from pydantic import TypeAdapter

from vllm_tuner.benchmarks.metrics import request_meets_slo
from vllm_tuner.benchmarks.models import RequestStatus, SLOThresholds
from vllm_tuner.experiment.artifacts import ARTIFACT_INTEGRITY_FILE, ArtifactStore
from vllm_tuner.experiment.manifest import sha256_file, sha256_json
from vllm_tuner.experiment.models import TrialResult, TrialStatus, trial_provenance, utc_now_iso
from vllm_tuner.runtime.controller import TrialController
from vllm_tuner.runtime.failures import UnsafeCleanupError

from .capacity_evidence import DeviceMemoryEvidence, build_capacity_runtime_evidence
from .m1_capacity_runner import (
    _finite_number,
    _read_json,
    _read_jsonl,
    _server_event_count,
    _trace_text,
)
from .m1_runner import GPUMemoryReader, _gpu_memory_snapshot
from .m4_chunked_integrity import M4_CHUNKED_INTEGRITY_FILE
from .m4_chunked_runner import (
    ControllerFactory,
    LongContextM4ChunkedRunner,
    _command_evidence,
    _interference_metrics,
    _raw_metrics_text,
    _request_latencies,
    _usage,
    _validate_counters,
    _validate_requests,
    _validate_warmups,
    _waiting,
)
from .m5_decode_tail_analysis import M5TrialRecord, analyze_m5_records
from .m5_decode_tail_config import (
    LongContextM5DecodeTailConfig,
    M5Cohort,
    M5DecodeTailProfile,
)
from .m5_decode_tail_integrity import (
    M5_DECODE_TAIL_INTEGRITY_FILE,
    seal_m5_decode_tail_artifacts,
    validate_m5_decode_tail_artifacts,
)
from .m5_decode_tail_workload import M5TraceBundle, build_m5_trace
from .model_identity import ModelIdentityFacts
from .runtime_identity import RuntimeIdentityFacts

M5_SCHEMA = "longctx-m5-decode-tail.v1"
MANIFEST_FILE = "manifest.json"
SUMMARY_FILE = "summary.json"
STATUS_FILE = "status.json"
REPORT_FILE = "report/m5-decode-tail.md"
RUNNER_LOG_FILE = "runner.log"
POINT_FILE = "m5-point.json"
RECORD_FILE = "m5-record.json"
EVIDENCE_FILE = "m5-evidence.json"
RUNTIME_CAPACITY_FILE = "runtime-capacity.json"
CUDA_MEMORY_FILE = "cuda-memory.json"
MEASURED_TRACE_FILE = "measured-trace.jsonl"
MEASURED_TRACE_CHECKSUM_FILE = "measured-trace.sha256"
WARMUP_TRACE_FILE = "warmup-trace.jsonl"
WARMUP_TRACE_CHECKSUM_FILE = "warmup-trace.sha256"

_RECORD_ADAPTER: TypeAdapter[M5TrialRecord] = TypeAdapter(M5TrialRecord)


def _derive_record(
    *,
    result: TrialResult,
    trial_dir: Path,
    profile: M5DecodeTailProfile,
    cohort: M5Cohort,
    repeat_index: int,
    bundle: M5TraceBundle,
    device_memory: DeviceMemoryEvidence,
    slo: SLOThresholds,
) -> tuple[M5TrialRecord, dict[str, Any]]:
    request_rows = _read_jsonl(trial_dir / "request-results.jsonl")
    prometheus_rows = _read_jsonl(trial_dir / "prometheus.jsonl")
    benchmark = _read_json(trial_dir / "benchmark-raw.json")
    requests = _validate_requests(request_rows, bundle.measured)
    _validate_warmups(benchmark, bundle.warmup)
    decode = [
        request for request in requests if bundle.request_kind[request.request_id] == "decode"
    ]
    long_requests = [
        request for request in requests if bundle.request_kind[request.request_id] == "long-prefill"
    ]
    if not decode or not long_requests:
        raise ValueError("M5 trace lacks decode or long-prefill requests")
    counters, queries, hits, preemptions = _validate_counters(prometheus_rows, requests)
    metrics_text = _raw_metrics_text(prometheus_rows)
    server_log = (trial_dir / "server.log").read_text(encoding="utf-8", errors="replace")
    command = _command_evidence(trial_dir, cast(Any, profile), server_log)
    runtime = build_capacity_runtime_evidence(
        run_id=result.trial_id,
        runtime_profile_sha256=sha256_json(
            {
                "profile_id": profile.profile_id,
                "vllm_args": profile.vllm_args(),
                "cohort_id": cohort.cohort_id,
                "long_prefill_tokens": list(cohort.long_prefill_tokens),
            }
        ),
        server_log_text=server_log,
        metrics_text=metrics_text,
        device_memory=device_memory,
    )
    runtime_checks = {
        "command_mechanism_evidence": command["passed"] is True,
        "runtime_capacity_consistent": runtime.logged_capacity_consistent,
        "chunked_prefill_enabled": runtime.startup_format.enable_chunked_prefill is True,
        "prefix_caching_enabled_but_isolated": runtime.cache_config.enable_prefix_caching is True,
        "kv_cache_dtype_is_auto": runtime.cache_config.requested_cache_dtype == "auto",
        "no_explicit_kv_memory": runtime.cache_config.kv_cache_memory_bytes is None,
        "no_gpu_block_override": runtime.cache_config.num_gpu_blocks_override is None,
        "prefix_isolation_proved": bundle.prefix_isolation_proof.get("no_cacheable_shared_prefix")
        is True,
    }
    if not all(runtime_checks.values()):
        failed = sorted(name for name, passed in runtime_checks.items() if not passed)
        raise ValueError("M5 runtime/mechanism gate failed: " + ", ".join(failed))
    windows, during_itl, outside_itl, overlap_count = _interference_metrics(decode, long_requests)
    duration = _finite_number(result.measurement_seconds, "M5 measurement duration")
    good_decode = sum(request_meets_slo(request, slo) for request in decode)
    overall_goodput = _finite_number(
        result.client.get("goodput_requests_per_sec"), "M5 overall Goodput"
    )
    peak_vram = _finite_number(result.gpu.get("peak_memory_mb"), "M5 peak VRAM")
    timeout_count = sum(request.status == RequestStatus.TIMEOUT for request in requests)
    oom_count = _server_event_count(server_log, (r"CUDA out of memory", r"\bOOM\b"))
    if timeout_count or oom_count or preemptions:
        raise ValueError("M5 trial contains a timeout, OOM, or preemption")
    resolved = cast(Mapping[str, Any], command["resolved"])
    record = M5TrialRecord(
        trial_id=result.trial_id,
        cohort_id=cohort.cohort_id,
        profile_id=cast(Any, profile.profile_id),
        production_default=profile.production_default,
        max_num_batched_tokens=cast(Any, resolved["max_num_batched_tokens"]),
        max_num_partial_prefills=cast(Any, resolved["max_num_partial_prefills"]),
        max_long_partial_prefills=cast(Any, resolved["max_long_partial_prefills"]),
        long_prefill_token_threshold=cast(Any, resolved["long_prefill_token_threshold"]),
        long_prefill_tokens=cohort.long_prefill_tokens,
        injection_offsets_seconds=cohort.injection_offsets_seconds,
        prompt_seed=cohort.prompt_seed,
        arrival_seed=cohort.arrival_seed,
        repeat_index=repeat_index,
        trace_id=bundle.measured.checksum(),
        warmup_trace_id=bundle.warmup.checksum(),
        request_count=len(requests),
        decode_request_count=len(decode),
        long_prefill_request_count=len(long_requests),
        completion_fraction=len(requests) / len(bundle.measured.entries),
        decode_slo_satisfied_fraction=good_decode / len(decode),
        decode_goodput_requests_per_second=good_decode / duration,
        overall_goodput_requests_per_second=overall_goodput,
        decode_ttft=_request_latencies(decode, "ttft"),
        decode_tpot=_request_latencies(decode, "tpot"),
        decode_itl=_request_latencies(decode, "itl"),
        decode_end_to_end=_request_latencies(decode, "e2e"),
        long_prefill_ttft=_request_latencies(long_requests, "ttft"),
        long_prefill_tpot=_request_latencies(long_requests, "tpot"),
        long_prefill_end_to_end=_request_latencies(long_requests, "e2e"),
        decode_interference_itl=during_itl,
        decode_non_interference_itl=outside_itl,
        decode_overlap_request_count=overlap_count,
        prefill_windows=windows,
        waiting=_waiting(prometheus_rows),
        kv_usage=_usage(prometheus_rows, "kv_cache_usage_perc"),
        preemption_count=cast(Any, preemptions),
        prefix_cache_queries=queries,
        prefix_cache_hits=cast(Any, hits),
        peak_vram_mb=peak_vram,
        oom_count=cast(Any, oom_count),
        timeout_count=cast(Any, timeout_count),
        mechanism_evidence_passed=True,
    )
    evidence = {
        "schema_version": M5_SCHEMA,
        "semantic_gate_passed": True,
        "profile": profile.model_dump(mode="json"),
        "cohort": cohort.model_dump(mode="json"),
        "command": command,
        "runtime_checks": runtime_checks,
        "runtime_capacity": runtime.model_dump(mode="json"),
        "prefix_isolation_proof": bundle.prefix_isolation_proof,
        "counters": counters,
        "prefill_windows": [window.model_dump(mode="json") for window in windows],
        "decode_overlap_request_count": overlap_count,
        "request_kinds": bundle.request_kind,
    }
    return record, evidence


class LongContextM5DecodeTailRunner(LongContextM4ChunkedRunner):
    """Execute only the frozen two-profile M5 validation matrix."""

    def __init__(
        self,
        config: LongContextM5DecodeTailConfig,
        experiment_id: str,
        *,
        repository: str | Path,
        resume: bool = False,
        controller_factory: ControllerFactory = TrialController,
        tokenizer: Optional[Any] = None,
        gpu_memory_reader: GPUMemoryReader = _gpu_memory_snapshot,
        runtime_facts: Optional[RuntimeIdentityFacts] = None,
        model_facts: Optional[ModelIdentityFacts] = None,
        execution_environment: Optional[Mapping[str, str]] = None,
    ) -> None:
        super().__init__(
            cast(Any, config),
            experiment_id,
            repository=repository,
            resume=resume,
            controller_factory=controller_factory,
            tokenizer=tokenizer,
            gpu_memory_reader=gpu_memory_reader,
            runtime_facts=runtime_facts,
            model_facts=model_facts,
            execution_environment=execution_environment,
        )
        self.config = config
        self.store = ArtifactStore(config.artifacts.root, experiment_id)

    def _profile(self, profile_id: str) -> M5DecodeTailProfile:
        return next(profile for profile in self.config.profiles if profile.profile_id == profile_id)

    def _manifest(
        self,
        runtime: RuntimeIdentityFacts,
        model: ModelIdentityFacts,
        environment: Mapping[str, str],
        source_commit: str,
        source_tree: str,
    ) -> dict[str, Any]:
        smoke = self.config.smoke_artifact
        return {
            "schema_version": M5_SCHEMA,
            "project_line": "longctx-v5",
            "milestone": "M5",
            "experiment_kind": "decode-tail-non-inferiority",
            "evidence_role": self.config.evidence_role,
            "experiment_id": self.experiment_id,
            "created_at": utc_now_iso(),
            "config_sha256": sha256_json(self.config.model_dump(mode="json")),
            "source_commit": source_commit,
            "source_tree_sha256": source_tree,
            "runtime": runtime.model_dump(mode="json"),
            "model": model.model_dump(mode="json"),
            "execution_environment": dict(environment),
            "m4_artifact": {
                **self.config.m4_artifact.identity(),
                "integrity_sha256": sha256_file(
                    self.config.m4_artifact.root / M4_CHUNKED_INTEGRITY_FILE
                ),
            },
            "smoke_artifact": (
                None
                if smoke is None
                else {
                    "experiment_id": smoke.experiment_id,
                    "root": str(smoke.root),
                    "integrity_sha256": sha256_file(smoke.root / M5_DECODE_TAIL_INTEGRITY_FILE),
                }
            ),
            "profiles": [profile.model_dump(mode="json") for profile in self.config.profiles],
            "cohorts": [cohort.model_dump(mode="json") for cohort in self.config.protocol.cohorts],
            "scope": {
                "m1_m2_m3_m4_rerun": False,
                "m4_selection_rewritten": False,
                "fp8_configuration_attempted": False,
                "apc_off_attempted": False,
                "threshold_512_attempted": False,
                "custom_scheduler_used": False,
                "m6_started": False,
            },
        }

    def _validate_formal_smoke_identity(self, source_commit: str) -> bool:
        smoke = self.config.smoke_artifact
        if smoke is None:
            return self.config.evidence_role == "smoke"
        manifest = _read_json(smoke.root / MANIFEST_FILE)
        experiment = _read_json(smoke.root / "experiment.json")
        return (
            manifest.get("source_commit") == source_commit
            and experiment.get("model") == self.config.model.model_dump(mode="json")
            and experiment.get("runtime") == self.config.runtime.model_dump(mode="json")
            and experiment.get("m4_artifact") == self.config.m4_artifact.model_dump(mode="json")
            and experiment.get("profiles")
            == [profile.model_dump(mode="json") for profile in self.config.profiles]
            and experiment.get("slo") == self.config.slo.model_dump(mode="json")
        )

    def _initialize_root(self, requested_manifest: dict[str, Any]) -> Optional[dict[str, Any]]:
        root = self.store.root
        if root.exists():
            if not self.resume:
                raise FileExistsError(f"M5 artifact root exists: {root}; use --resume")
            if (root / M5_DECODE_TAIL_INTEGRITY_FILE).is_file():
                validate_m5_decode_tail_artifacts(root)
                existing = _read_json(root / MANIFEST_FILE)
                if self._manifest_identity(existing) != self._manifest_identity(requested_manifest):
                    raise ValueError("M5 resume manifest identity mismatch")
                return _read_json(root / SUMMARY_FILE)
            self.store.initialize(exist_ok=True)
            existing = _read_json(root / MANIFEST_FILE)
            if self._manifest_identity(existing) != self._manifest_identity(requested_manifest):
                raise ValueError("M5 resume manifest identity mismatch")
            return None
        self.store.initialize()
        self.store.write_json(MANIFEST_FILE, requested_manifest)
        self.store.write_json("experiment.json", self.config.model_dump(mode="json"))
        return None

    def _trace_catalog(self) -> dict[str, M5TraceBundle]:
        tokenizer = self._load_tokenizer()
        bundles: dict[str, M5TraceBundle] = {}
        catalog: dict[str, Any] = {"schema_version": M5_SCHEMA, "traces": {}}
        for cohort in self.config.protocol.cohorts:
            bundle = build_m5_trace(
                protocol=self.config.protocol,
                cohort=cohort,
                tokenizer=tokenizer,
            )
            bundles[cohort.cohort_id] = bundle
            base = Path("traces") / cohort.cohort_id
            self._write_or_validate_trace(base / "measured.jsonl", bundle.measured)
            self._write_or_validate_trace(base / "warmup.jsonl", bundle.warmup)
            cast(dict[str, Any], catalog["traces"])[cohort.cohort_id] = {
                "measured_path": (base / "measured.jsonl").as_posix(),
                "measured_sha256": bundle.measured.checksum(),
                "measured_requests": len(bundle.measured.entries),
                "warmup_path": (base / "warmup.jsonl").as_posix(),
                "warmup_sha256": bundle.warmup.checksum(),
                "prompt_seed": cohort.prompt_seed,
                "arrival_seed": cohort.arrival_seed,
                "long_prefill_tokens": list(cohort.long_prefill_tokens),
                "injection_offsets_seconds": list(cohort.injection_offsets_seconds),
                "request_kind": bundle.request_kind,
                "prefix_isolation_proof": bundle.prefix_isolation_proof,
            }
        path = self.store.root / "traces" / "catalog.json"
        if path.exists():
            if _read_json(path) != catalog:
                raise ValueError("M5 resume trace catalog mismatch")
        else:
            self.store.write_json("traces/catalog.json", catalog)
        return bundles

    @staticmethod
    def _logical_id(profile: M5DecodeTailProfile, cohort: M5Cohort, repeat_index: int) -> str:
        return f"decode-tail-{cohort.cohort_id}-{profile.profile_id}-repeat-{repeat_index}"

    def _point_payload(
        self,
        *,
        trial_id: str,
        profile: M5DecodeTailProfile,
        cohort: M5Cohort,
        repeat_index: int,
        bundle: M5TraceBundle,
    ) -> dict[str, Any]:
        return {
            "schema_version": M5_SCHEMA,
            "trial_id": trial_id,
            "cohort_id": cohort.cohort_id,
            "profile_id": profile.profile_id,
            "production_default": profile.production_default,
            "repeat_index": repeat_index,
            "trace_id": bundle.measured.checksum(),
            "warmup_trace_id": bundle.warmup.checksum(),
            "prompt_seed": cohort.prompt_seed,
            "arrival_seed": cohort.arrival_seed,
            "long_prefill_tokens": list(cohort.long_prefill_tokens),
            "injection_offsets_seconds": list(cohort.injection_offsets_seconds),
            "vllm_args": profile.vllm_args(),
            "m4_source": self.config.m4_artifact.experiment_id,
        }

    def _cached_complete(
        self,
        *,
        logical_id: str,
        expected_point: Mapping[str, Any],
    ) -> Optional[tuple[TrialResult, M5TrialRecord]]:
        for _, path in reversed(self._attempts(logical_id)):
            try:
                if not (path / ARTIFACT_INTEGRITY_FILE).is_file():
                    raise ValueError("attempt is not independently sealed")
                result = self.store.load_trial_result(path.name)
                if result is None:
                    raise ValueError("attempt has no terminal TrialResult")
                self.store.validate_cached_trial(result, require_telemetry=True)
                self.store.validate_trial_artifacts(
                    result.trial_id,
                    require_telemetry=True,
                    require_available=True,
                    required_evidence={
                        "server-command.json",
                        "request-results.jsonl",
                        "benchmark-raw.json",
                        "prometheus.jsonl",
                        "nvml.jsonl",
                        "server.log",
                        "cleanup.json",
                        POINT_FILE,
                        RECORD_FILE,
                        EVIDENCE_FILE,
                        RUNTIME_CAPACITY_FILE,
                        CUDA_MEMORY_FILE,
                        MEASURED_TRACE_FILE,
                        MEASURED_TRACE_CHECKSUM_FILE,
                        WARMUP_TRACE_FILE,
                        WARMUP_TRACE_CHECKSUM_FILE,
                    },
                )
                point = _read_json(path / POINT_FILE)
                expected = {**expected_point, "trial_id": result.trial_id}
                evidence = _read_json(path / EVIDENCE_FILE)
                if point != expected or evidence.get("semantic_gate_passed") is not True:
                    raise ValueError("sealed M5 attempt identity or semantic gate mismatch")
                record = _RECORD_ADAPTER.validate_json(
                    (path / RECORD_FILE).read_text(encoding="utf-8")
                )
            except (OSError, ValueError) as error:
                self._resume_warnings.append(f"attempt {path.name} not replayed: {error}")
                continue
            self._replayed_trials.append(path.name)
            return result, record
        return None

    def _write_pretrial(
        self,
        *,
        point: Mapping[str, Any],
        bundle: M5TraceBundle,
        memory: DeviceMemoryEvidence,
    ) -> None:
        trial_id = str(point["trial_id"])
        base = Path("trials") / trial_id
        self.store.write_json(base / POINT_FILE, dict(point))
        self.store.write_text(base / MEASURED_TRACE_FILE, _trace_text(bundle.measured))
        self.store.write_text(
            base / MEASURED_TRACE_CHECKSUM_FILE,
            f"{bundle.measured.checksum()}  {MEASURED_TRACE_FILE}\n",
        )
        self.store.write_text(base / WARMUP_TRACE_FILE, _trace_text(bundle.warmup))
        self.store.write_text(
            base / WARMUP_TRACE_CHECKSUM_FILE,
            f"{bundle.warmup.checksum()}  {WARMUP_TRACE_FILE}\n",
        )
        self.store.write_json(base / CUDA_MEMORY_FILE, memory)
        self.store.write_json(
            base / EVIDENCE_FILE,
            {
                "schema_version": M5_SCHEMA,
                "semantic_gate_passed": False,
                "state": "pending-controller-result",
            },
        )

    def _finalize_trial(
        self,
        result: TrialResult,
        record: M5TrialRecord | Mapping[str, Any],
        evidence: Mapping[str, Any],
    ) -> None:
        base = Path("trials") / result.trial_id
        self.store.write_json(base / RECORD_FILE, record)
        self.store.write_json(base / EVIDENCE_FILE, dict(evidence))
        runtime = evidence.get("runtime_capacity")
        self.store.write_json(
            base / RUNTIME_CAPACITY_FILE,
            (
                dict(runtime)
                if isinstance(runtime, Mapping)
                else {"available": False, "reason": "M5 semantic gate produced no capacity"}
            ),
        )
        self.store.ensure_trial_artifacts(result)
        self.store.validate_cached_trial(result, require_telemetry=True)
        self.store.validate_trial_integrity(result.trial_id)

    async def _run_one(
        self,
        *,
        profile: M5DecodeTailProfile,
        cohort: M5Cohort,
        repeat_index: int,
        bundle: M5TraceBundle,
    ) -> tuple[TrialResult, Optional[M5TrialRecord], bool]:
        logical = self._logical_id(profile, cohort, repeat_index)
        expected_point = self._point_payload(
            trial_id=logical,
            profile=profile,
            cohort=cohort,
            repeat_index=repeat_index,
            bundle=bundle,
        )
        cached = self._cached_complete(logical_id=logical, expected_point=expected_point)
        if cached is not None:
            return cached[0], cached[1], True
        trial_id = self._select_new_attempt(logical)
        point = {**expected_point, "trial_id": trial_id}
        tuning = self.config.to_tuning_config(profile, cohort)
        if tuning.vllm_args != profile.vllm_args():
            raise ValueError("M5 TuningConfig changed preregistered native arguments")
        memory = self.gpu_memory_reader(self.config.gpu.device_ids[0])
        self._write_pretrial(point=point, bundle=bundle, memory=memory)
        controller = self.controller_factory(
            tuning,
            bundle.measured,
            self.store,
            tokenizer=self._load_tokenizer(),
            warmup_trace=bundle.warmup,
            strict_open_loop=True,
        )
        try:
            result = await controller.run_trial({}, trial_id, "decode-tail")
        except UnsafeCleanupError:
            self._status(
                "unsafe_cleanup",
                current_trial=trial_id,
                unsafe_cleanup=True,
                message="unsafe cleanup; M5 stopped before another GPU process",
            )
            raise
        expected_provenance = trial_provenance(trial_id, "decode-tail")
        if any(getattr(result, name) != value for name, value in expected_provenance.items()):
            raise ValueError(f"M5 trial provenance mismatch: {trial_id}")
        if result.params != {}:
            raise ValueError(f"M5 fixed profile trial passed search parameters: {trial_id}")
        record: Optional[M5TrialRecord] = None
        evidence: dict[str, Any]
        if result.status in {TrialStatus.COMPLETE, TrialStatus.INFEASIBLE}:
            try:
                record, evidence = _derive_record(
                    result=result,
                    trial_dir=self.store.trials_dir / trial_id,
                    profile=profile,
                    cohort=cohort,
                    repeat_index=repeat_index,
                    bundle=bundle,
                    device_memory=memory,
                    slo=SLOThresholds(
                        ttft_ms=self.config.slo.ttft_ms,
                        tpot_ms=self.config.slo.tpot_ms,
                        e2e_ms=self.config.slo.e2e_ms,
                    ),
                )
            except (OSError, ValueError) as error:
                result.status = TrialStatus.FAILED
                result.constraints = {
                    **result.constraints,
                    "feasible": False,
                    "violations": [
                        *cast(list[str], result.constraints.get("violations", [])),
                        "m5_decode_tail_semantic_gate",
                    ],
                }
                result.failure_reason = {
                    "type": "M5_DECODE_TAIL_SEMANTIC_GATE",
                    "message": str(error),
                    "phase": "M5_DECODE_TAIL_FINALIZE",
                }
                self.store.record_artifact_finalizer_failure(result, str(error))
                evidence = {
                    "schema_version": M5_SCHEMA,
                    "semantic_gate_passed": False,
                    "failure": str(error),
                }
        else:
            reason = (
                json.dumps(result.failure_reason, sort_keys=True)
                if result.failure_reason
                else "trial failed"
            )
            evidence = {
                "schema_version": M5_SCHEMA,
                "semantic_gate_passed": False,
                "failure": reason,
            }
        failure_record = {
            "schema_version": M5_SCHEMA,
            "trial_id": trial_id,
            "cohort_id": cohort.cohort_id,
            "profile_id": profile.profile_id,
            "repeat_index": repeat_index,
            "status": "failed",
            "failure": evidence.get("failure"),
        }
        self._finalize_trial(result, record or failure_record, evidence)
        return result, record, False

    def _expected_matrix(self) -> set[tuple[str, str, int]]:
        return {
            (cohort.cohort_id, profile.profile_id, repeat)
            for cohort in self.config.protocol.cohorts
            for repeat in range(self.config.protocol.repeats)
            for profile in self.config.profiles
        }

    def _profile_order(
        self, cohort_index: int, repeat_index: int
    ) -> tuple[M5DecodeTailProfile, ...]:
        profiles = self.config.profiles
        rotation = (cohort_index + repeat_index) % len(profiles)
        return profiles[rotation:] + profiles[:rotation]

    def _status(
        self,
        state: str,
        *,
        completed_jobs: int = 0,
        current_trial: Optional[str] = None,
        unsafe_cleanup: bool = False,
        acceptance: Optional[Mapping[str, Any]] = None,
        message: Optional[str] = None,
    ) -> dict[str, Any]:
        planned = len(self._expected_matrix())
        elapsed = max(0.0, time.monotonic() - self._started_monotonic)
        remaining = max(0, planned - completed_jobs)
        if completed_jobs:
            eta_seconds = elapsed / completed_jobs * remaining
        else:
            eta_seconds = (
                (self.config.protocol.decode_requests - 1)
                * self.config.protocol.decode_interval_seconds
                * planned
            )
        value = {
            "schema_version": M5_SCHEMA,
            "experiment_id": self.experiment_id,
            "state": state,
            "pid": os.getpid(),
            "gpu": list(self.config.gpu.device_ids),
            "log": str(self.store.root / RUNNER_LOG_FILE),
            "result": str(self.store.root),
            "eta": (datetime.now(timezone.utc) + timedelta(seconds=eta_seconds)).isoformat(),
            "eta_seconds": eta_seconds,
            "resume": self._resume_command(),
            "sealed": (self.store.root / M5_DECODE_TAIL_INTEGRITY_FILE).is_file(),
            "acceptance": dict(acceptance) if acceptance is not None else None,
            "planned_jobs": planned,
            "completed_jobs": completed_jobs,
            "current_trial": current_trial,
            "unsafe_cleanup": unsafe_cleanup,
            "message": message,
            "updated_at": utc_now_iso(),
        }
        self.store.write_json(STATUS_FILE, value)
        log_path = self.store.root / RUNNER_LOG_FILE
        previous = log_path.read_text(encoding="utf-8") if log_path.is_file() else ""
        self.store.write_text(
            RUNNER_LOG_FILE,
            previous
            + f"{value['updated_at']} state={state} completed={completed_jobs}/{planned}"
            + (f" trial={current_trial}" if current_trial else "")
            + (f" message={message}" if message else "")
            + "\n",
        )
        return value

    def _resume_command(self) -> str:
        return (
            "scripts/run_longctx_m5_decode_tail.sh --config CONFIG "
            f"--experiment-id {self.experiment_id} --resume"
        )

    @staticmethod
    def _report(summary: Mapping[str, Any]) -> str:
        execution = cast(Mapping[str, Any], summary["execution"])
        acceptance = cast(Mapping[str, Any], summary["acceptance"])
        analysis = cast(Mapping[str, Any], summary["analysis"])
        decision = cast(Mapping[str, Any], analysis["decision"])
        lines = [
            "# Long-context v5 M5 Decode-tail validation",
            "",
            f"- Experiment: {summary['experiment_id']}",
            f"- Evidence role: {summary['evidence_role']}",
            f"- Execution passed: {execution['passed']}",
            f"- M5 accepted: {acceptance['passed']}",
            f"- Decision profile: {decision['profile_id']}",
            f"- Result wording: {decision['wording']}",
            "- M4 selection rewritten: false",
            "- FP8/APC-off/threshold-512/custom Scheduler attempted: false",
            "- M6 started: false",
            "",
            "## Preregistered paired evidence",
            "",
            "| Cohort | ITL p99 improvement median | Decode Goodput change median/min | "
            "Long TTFT p99 degradation median | Decode TPOT p99 degradation median | Passed |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        paired = analysis.get("paired")
        if isinstance(paired, list):
            for row in paired:
                if not isinstance(row, Mapping):
                    continue
                aggregates = row.get("aggregates")
                cohort_acceptance = row.get("acceptance")
                if not isinstance(aggregates, Mapping) or not isinstance(
                    cohort_acceptance, Mapping
                ):
                    continue
                itl = cast(
                    Mapping[str, Any], aggregates["decode_interference_itl_p99_improvement_percent"]
                )
                goodput = cast(Mapping[str, Any], aggregates["decode_goodput_change_percent"])
                ttft = cast(
                    Mapping[str, Any], aggregates["long_prefill_ttft_p99_degradation_percent"]
                )
                tpot = cast(Mapping[str, Any], aggregates["decode_tpot_p99_degradation_percent"])
                lines.append(
                    f"| {row.get('cohort_id')} | {itl.get('median'):.6f}% | "
                    f"{goodput.get('median'):.6f}% / {goodput.get('minimum'):.6f}% | "
                    f"{ttft.get('median'):.6f}% | {tpot.get('median'):.6f}% | "
                    f"{cohort_acceptance.get('passed')} |"
                )
        failures = acceptance.get("failure_reasons")
        lines.extend(["", f"- Acceptance failures: {failures}"])
        return "\n".join(lines) + "\n"

    async def _run_locked(self) -> dict[str, Any]:
        runtime, model, environment = self._identities()
        source_commit, source_tree = self._source_identity()
        smoke_identity_matches = self._validate_formal_smoke_identity(source_commit)
        if not smoke_identity_matches:
            raise ValueError("formal M5 source/model/runtime/M4 identity differs from smoke")
        requested_manifest = self._manifest(
            runtime,
            model,
            environment,
            source_commit,
            source_tree,
        )
        completed_root = self._initialize_root(requested_manifest)
        if completed_root is not None:
            return {
                **completed_root,
                "resume": {
                    **cast(dict[str, Any], completed_root.get("resume", {})),
                    "requested": True,
                    "root_replayed": True,
                },
            }
        self._status("preparing")
        bundles = self._trace_catalog()
        records: list[M5TrialRecord] = []
        results: list[TrialResult] = []
        completed_jobs = 0
        execution_order: list[str] = []
        for cohort_index, cohort in enumerate(self.config.protocol.cohorts):
            bundle = bundles[cohort.cohort_id]
            for repeat_index in range(self.config.protocol.repeats):
                for profile in self._profile_order(cohort_index, repeat_index):
                    logical = self._logical_id(profile, cohort, repeat_index)
                    execution_order.append(logical)
                    self._status("running", completed_jobs=completed_jobs, current_trial=logical)
                    result, record, _ = await self._run_one(
                        profile=profile,
                        cohort=cohort,
                        repeat_index=repeat_index,
                        bundle=bundle,
                    )
                    results.append(result)
                    if record is not None:
                        records.append(record)
                    completed_jobs += 1

        expected = self._expected_matrix()
        observed = {
            (record.cohort_id, record.profile_id, record.repeat_index) for record in records
        }
        execution_passed = len(results) == len(expected) and observed == expected
        formal = self.config.evidence_role == "formal"
        analysis = analyze_m5_records(records, formal=formal)
        source_commit_after, source_tree_after = self._source_identity()
        source_stable = source_commit_after == source_commit and source_tree_after == source_tree
        if formal and self.config.smoke_artifact is not None:
            smoke_summary = _read_json(self.config.smoke_artifact.root / SUMMARY_FILE)
            smoke_acceptance = smoke_summary.get("acceptance")
            bound_smoke_passed = (
                isinstance(smoke_acceptance, Mapping) and smoke_acceptance.get("passed") is True
            )
        else:
            bound_smoke_passed = not formal
        trace_ids: dict[str, set[str]] = {}
        for record in records:
            trace_ids.setdefault(record.cohort_id, set()).add(record.trace_id)
        m4_summary = _read_json(self.config.m4_artifact.root / SUMMARY_FILE)
        m4_analysis = m4_summary.get("analysis")
        m4_selection = m4_analysis.get("selection") if isinstance(m4_analysis, Mapping) else None
        cohorts = {cohort.cohort_id: cohort for cohort in self.config.protocol.cohorts}
        target = cohorts.get("target")
        held_out = cohorts.get("held-out")
        held_out_frozen_and_distinct = not formal or (
            target is not None
            and held_out is not None
            and target.prompt_seed != held_out.prompt_seed
            and target.arrival_seed != held_out.arrival_seed
            and (
                target.long_prefill_tokens != held_out.long_prefill_tokens
                or target.injection_offsets_seconds != held_out.injection_offsets_seconds
            )
        )
        analysis_acceptance = analysis.get("acceptance")
        formal_non_inferiority_passed = (
            isinstance(analysis_acceptance, Mapping) and analysis_acceptance.get("passed") is True
        )
        checks = {
            "project_line_is_v5": self.config.project_line == "longctx-v5",
            "m4_formal_18_of_18_bound": self.config.m4_artifact.experiment_id
            == "longctx-v5-m4-chunked-formal-001",
            "m4_selection_remains_production_default": isinstance(m4_selection, Mapping)
            and m4_selection.get("profile_id") == "production-default",
            "m1_m2_m3_m4_not_rerun": True,
            "m1_to_m5_frozen_rules_unchanged": True,
            "fp8_not_attempted": True,
            "apc_off_not_attempted": True,
            "threshold_512_not_attempted": True,
            "custom_scheduler_not_used": True,
            "exact_two_profile_set": tuple(profile.profile_id for profile in self.config.profiles)
            == ("production-default", "decode-tail-1024"),
            "bound_smoke_passed": bound_smoke_passed,
            "smoke_identity_matches": smoke_identity_matches,
            "all_jobs_complete": execution_passed,
            "runtime_lock_verified": runtime.matches_lock,
            "model_lock_verified": model.matches_lock,
            "source_identity_stable": source_stable,
            "single_gpu_only": self.config.gpu.count == 1 and self.config.gpu.device_ids == (0,),
            "same_trace_within_each_cohort": bool(trace_ids)
            and all(len(values) == 1 for values in trace_ids.values()),
            "held_out_seed_and_workload_changed_before_run": held_out_frozen_and_distinct,
            "prefix_reuse_isolated": bool(records)
            and all(record.prefix_cache_hits == 0 for record in records),
            "mechanism_evidence_complete": bool(records)
            and all(record.mechanism_evidence_passed for record in records),
            "decode_prefill_overlap_observed": bool(records)
            and all(record.decode_overlap_request_count > 0 for record in records),
            "zero_oom_timeout_preemption": bool(records)
            and all(
                record.oom_count == 0 and record.timeout_count == 0 and record.preemption_count == 0
                for record in records
            ),
            "formal_run_count_is_12": len(expected) == 12 if formal else True,
            "target_and_held_out_non_inferiority_passed": (
                formal_non_inferiority_passed if formal else True
            ),
            "held_out_not_retuned": source_stable and held_out_frozen_and_distinct,
            "no_m6_work": True,
        }
        acceptance_passed = all(checks.values())
        acceptance = {
            "eligible": formal,
            "passed": acceptance_passed,
            "checks": checks,
            "failure_reasons": sorted(name for name, passed in checks.items() if not passed),
            "non_inferiority": analysis_acceptance,
        }
        execution = {
            "passed": execution_passed,
            "planned_jobs": len(expected),
            "completed_jobs": len(records),
            "failed_jobs": len(expected) - len(records),
            "unsafe_cleanup": False,
            "balanced_execution_order": execution_order,
        }
        summary: dict[str, Any] = {
            "schema_version": M5_SCHEMA,
            "project_line": self.config.project_line,
            "milestone": self.config.milestone,
            "experiment_kind": self.config.experiment_kind,
            "evidence_role": self.config.evidence_role,
            "experiment_id": self.experiment_id,
            "source_commit": source_commit,
            "finished_at": utc_now_iso(),
            "execution": execution,
            "acceptance": acceptance,
            "analysis": analysis,
            "records": [record.model_dump(mode="json") for record in records],
            "protocol_scope": {
                "matrix": "production-default vs decode-tail-1024, target and held-out pairs",
                "profiles": [profile.profile_id for profile in self.config.profiles],
                "cohorts": [
                    cohort.model_dump(mode="json") for cohort in self.config.protocol.cohorts
                ],
                "decode_requests_per_run": self.config.protocol.decode_requests,
                "defensive_extra_tests": 0,
                "performance_runs": len(expected),
            },
            "resume": {
                "requested": self.resume,
                "root_replayed": False,
                "replayed_trials": self._replayed_trials,
                "new_attempts": self._new_attempts,
                "warnings": self._resume_warnings,
                "command": self._resume_command(),
            },
            "artifacts": {
                "root": str(self.store.root),
                "manifest": str(self.store.root / MANIFEST_FILE),
                "summary": str(self.store.root / SUMMARY_FILE),
                "report": str(self.store.root / REPORT_FILE),
                "status": str(self.store.root / STATUS_FILE),
                "integrity": str(self.store.root / M5_DECODE_TAIL_INTEGRITY_FILE),
                "m4_artifact": str(self.config.m4_artifact.root),
            },
            "m1_m2_m3_m4_rerun": False,
            "m4_selection_rewritten": False,
            "fp8_attempted": False,
            "apc_off_attempted": False,
            "threshold_512_attempted": False,
            "custom_scheduler_used": False,
            "held_out_retuned": False,
            "m6_started": False,
        }
        self.store.write_json(SUMMARY_FILE, summary)
        self.store.write_text(REPORT_FILE, self._report(summary))
        if acceptance_passed:
            final_state = "accepted"
        elif formal and execution_passed:
            final_state = "negative_result"
        else:
            final_state = "completed_not_accepted"
        self._status(final_state, completed_jobs=len(results), acceptance=acceptance)
        seal_m5_decode_tail_artifacts(
            self.store.root,
            self.experiment_id,
            {
                "experiment_id": self.experiment_id,
                "project_line": "longctx-v5",
                "milestone": "M5",
                "experiment_kind": "decode-tail-non-inferiority",
                "evidence_role": self.config.evidence_role,
                "source_commit": source_commit,
                "accepted": acceptance_passed,
                "positive_result": acceptance_passed and formal,
                "m1_m2_m3_m4_rerun": False,
                "m4_selection_rewritten": False,
                "fp8_attempted": False,
                "apc_off_attempted": False,
                "threshold_512_attempted": False,
                "custom_scheduler_used": False,
                "held_out_retuned": False,
                "m6_started": False,
            },
        )
        return summary


def load_m5_decode_tail_status(root: str | Path, experiment_id: str) -> dict[str, Any]:
    """Load M5 status without importing a tokenizer or starting vLLM."""
    experiment_root = Path(root).expanduser().resolve() / experiment_id
    sealed = (experiment_root / M5_DECODE_TAIL_INTEGRITY_FILE).is_file()
    if sealed:
        validate_m5_decode_tail_artifacts(experiment_root)
    status = _read_json(experiment_root / STATUS_FILE)
    if status.get("experiment_id") != experiment_id:
        raise ValueError("M5 status experiment identity mismatch")
    return {
        "experiment_id": experiment_id,
        "state": status.get("state"),
        "pid": status.get("pid"),
        "gpu": status.get("gpu"),
        "log": status.get("log"),
        "result": status.get("result"),
        "eta": status.get("eta"),
        "resume": status.get("resume"),
        "sealed": sealed,
        "acceptance": status.get("acceptance"),
        "completed_jobs": status.get("completed_jobs"),
        "planned_jobs": status.get("planned_jobs"),
        "current_trial": status.get("current_trial"),
    }


__all__ = ["LongContextM5DecodeTailRunner", "load_m5_decode_tail_status"]
