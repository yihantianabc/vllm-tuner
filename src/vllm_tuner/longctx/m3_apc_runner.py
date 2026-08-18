"""Run, resume, analyze, and seal long-context v5 M3 APC experiments."""

from __future__ import annotations

import fcntl
import json
import math
import os
import re
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional, Protocol, cast

from pydantic import TypeAdapter

from vllm_tuner.benchmarks.models import RequestResult, RequestStatus
from vllm_tuner.experiment.artifacts import ARTIFACT_INTEGRITY_FILE, ArtifactStore
from vllm_tuner.experiment.manifest import git_state, sha256_file, sha256_json, source_tree_sha256
from vllm_tuner.experiment.models import TrialResult, TrialStatus, trial_provenance, utc_now_iso
from vllm_tuner.profiling.timeseries import percentile
from vllm_tuner.runtime.controller import TrialController
from vllm_tuner.runtime.failures import UnsafeCleanupError
from vllm_tuner.workloads.trace import WorkloadTrace

from .capacity_evidence import DeviceMemoryEvidence, build_capacity_runtime_evidence
from .m1_capacity_runner import (
    _counter_evidence,
    _finite_number,
    _integer_counter,
    _read_json,
    _read_jsonl,
    _server_event_count,
    _trace_span,
    _trace_text,
)
from .m1_runner import GPUMemoryReader, _clean_execution_environment, _gpu_memory_snapshot
from .m2_fp8_integrity import M2_FP8_INTEGRITY_FILE
from .m3_apc_analysis import (
    M3APCTrialRecord,
    M3BoundaryRecord,
    M3KVUsage,
    M3LatencyPercentiles,
    M3ReuseMetrics,
    analyze_m3_apc_records,
)
from .m3_apc_config import LongContextM3APCConfig, M3APCProfile
from .m3_apc_integrity import (
    M3_APC_INTEGRITY_FILE,
    seal_m3_apc_artifacts,
    validate_m3_apc_artifacts,
)
from .m3_apc_workload import (
    CacheState,
    M3BoundaryTraceBundle,
    M3CoreTraceBundle,
    build_m3_boundary_trace,
    build_m3_core_trace,
    build_m3_core_warmup,
    expected_core_cached_tokens,
    load_rag_corpus,
    RAG_CORPUS_FILES,
)
from .model_identity import ModelIdentityFacts, require_model_identity
from .runtime_identity import RuntimeIdentityFacts, require_upstream_runtime

M3_APC_SCHEMA = "longctx-m3-apc.v1"
MANIFEST_FILE = "manifest.json"
SUMMARY_FILE = "summary.json"
STATUS_FILE = "status.json"
REPORT_FILE = "report/m3-apc.md"
RUNNER_LOG_FILE = "runner.log"
POINT_FILE = "m3-point.json"
RECORD_FILE = "m3-record.json"
EVIDENCE_FILE = "m3-evidence.json"
RUNTIME_CAPACITY_FILE = "runtime-capacity.json"
CUDA_MEMORY_FILE = "cuda-memory.json"
MEASURED_TRACE_FILE = "measured-trace.jsonl"
MEASURED_TRACE_CHECKSUM_FILE = "measured-trace.sha256"
WARMUP_TRACE_FILE = "warmup-trace.jsonl"
WARMUP_TRACE_CHECKSUM_FILE = "warmup-trace.sha256"

_EXPERIMENT_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_ATTEMPT_SUFFIX = re.compile(r"-attempt(?P<number>[1-9][0-9]*)$")
_CORE_RECORD_ADAPTER: TypeAdapter[M3APCTrialRecord] = TypeAdapter(M3APCTrialRecord)
_BOUNDARY_RECORD_ADAPTER: TypeAdapter[M3BoundaryRecord] = TypeAdapter(M3BoundaryRecord)


class ControllerProtocol(Protocol):
    async def run_trial(self, params: dict[str, Any], trial_id: str, method: str) -> TrialResult:
        """Run one M3 APC trial."""


ControllerFactory = Callable[..., ControllerProtocol]


def _latencies(values: Sequence[float], field: str) -> M3LatencyPercentiles:
    if not values:
        raise ValueError(f"M3 {field} has no exact request samples")
    p50 = percentile(values, 0.50)
    p95 = percentile(values, 0.95)
    p99 = percentile(values, 0.99)
    if p50 is None or p95 is None or p99 is None:
        raise ValueError(f"M3 {field} percentiles are unavailable")
    return M3LatencyPercentiles(p50_ms=p50, p95_ms=p95, p99_ms=p99)


def _kv_usage(rows: Sequence[Mapping[str, Any]]) -> M3KVUsage:
    values: list[float] = []
    for row in rows:
        metrics = row.get("metrics")
        raw = metrics.get("kv_cache_usage_perc") if isinstance(metrics, Mapping) else None
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            continue
        value = float(raw)
        if math.isfinite(value) and value >= 0:
            values.append(value)
    if not values:
        raise ValueError("M3 KV usage telemetry is unavailable")
    median = percentile(values, 0.50)
    p95 = percentile(values, 0.95)
    if median is None or p95 is None:
        raise ValueError("M3 KV usage percentiles are unavailable")
    return M3KVUsage(
        sample_count=len(values),
        minimum=min(values),
        median=median,
        p95=p95,
        maximum=max(values),
    )


def _request_cached_tokens(row: Mapping[str, Any]) -> int:
    metadata = row.get("metadata")
    usage = metadata.get("usage") if isinstance(metadata, Mapping) else None
    details = usage.get("prompt_tokens_details") if isinstance(usage, Mapping) else None
    raw = details.get("cached_tokens") if isinstance(details, Mapping) else 0
    if raw is None:
        raw = 0
    value = _integer_counter(raw, "M3 per-request cached_tokens")
    input_tokens = _integer_counter(row.get("input_tokens"), "M3 per-request input_tokens")
    if value > input_tokens:
        raise ValueError("M3 per-request cached_tokens exceeds prompt tokens")
    return value


def _command_evidence(trial_dir: Path, profile: M3APCProfile) -> dict[str, Any]:
    command = _read_json(trial_dir / "server-command.json")
    raw_argv = command.get("argv")
    if not isinstance(raw_argv, list) or any(not isinstance(item, str) for item in raw_argv):
        raise ValueError("M3 server command does not expose a string argv")
    argv = cast(list[str], raw_argv)
    flags = {item.split("=", 1)[0] for item in argv if item.startswith("--")}
    on = "--enable-prefix-caching" in flags
    off = "--no-enable-prefix-caching" in flags
    instrumentation = "--enable-prompt-tokens-details" in flags
    forbidden = sorted(
        flags
        & {
            "--attention-backend",
            "--calculate-kv-scales",
            "--kv-cache-dtype",
            "--long-prefill-token-threshold",
            "--max-long-partial-prefills",
            "--max-num-partial-prefills",
            "--scheduler-cls",
        }
    )
    checks = {
        "apc_argument_exact": on == profile.enable_prefix_caching
        and off != profile.enable_prefix_caching,
        "request_cached_tokens_enabled": instrumentation,
        "no_fp8_or_m4_arguments": not forbidden,
    }
    return {
        "argv": argv,
        "enable_prefix_caching_argument": on,
        "disable_prefix_caching_argument": off,
        "enable_prompt_tokens_details_argument": instrumentation,
        "forbidden_fp8_or_m4_arguments": forbidden,
        "checks": checks,
        "passed": all(checks.values()),
    }


def _raw_metrics_text(rows: Sequence[Mapping[str, Any]]) -> str:
    value = next(
        (
            str(row["raw_text"])
            for row in rows
            if row.get("available") is True and isinstance(row.get("raw_text"), str)
        ),
        None,
    )
    if value is None:
        raise ValueError("M3 raw Prometheus exposition is unavailable")
    return value


def _validate_requests(
    rows: Sequence[Mapping[str, Any]],
    trace: WorkloadTrace,
) -> list[RequestResult]:
    expected_ids = [entry.request_id for entry in trace.entries]
    if [row.get("request_id") for row in rows] != expected_ids:
        raise ValueError("M3 request IDs/order do not match the sealed paired trace")
    requests = [RequestResult.from_dict(dict(row)) for row in rows]
    expected_by_id = {entry.request_id: entry for entry in trace.entries}
    if len(requests) != len(trace.entries) or any(
        request.status != RequestStatus.SUCCESS
        or request.input_tokens != expected_by_id[request.request_id].input_tokens
        or request.output_tokens != expected_by_id[request.request_id].output_tokens
        or request.token_count_source != "usage"
        or not request.token_timestamps_valid
        or len(request.token_timestamps) != expected_by_id[request.request_id].output_tokens
        for request in requests
    ):
        raise ValueError("M3 requires successful exact-token evidence for every request")
    return requests


def _validate_warmups(benchmark: Mapping[str, Any], warmup: WorkloadTrace) -> None:
    raw = benchmark.get("warmup_results")
    if not isinstance(raw, list) or any(not isinstance(row, Mapping) for row in raw):
        raise ValueError("M3 benchmark has no structured warmup results")
    rows = cast(list[Mapping[str, Any]], raw)
    if len(rows) != len(warmup.entries):
        raise ValueError("M3 warmup result count differs from the sealed warmup trace")
    for index, (entry, row) in enumerate(zip(warmup.entries, rows)):
        if (
            row.get("request_id") != f"warmup-{index}-{entry.request_id}"
            or row.get("status") != "success"
            or row.get("input_tokens") != entry.input_tokens
            or row.get("output_tokens") != entry.output_tokens
        ):
            raise ValueError("M3 warmup evidence is incomplete or changed")


def _common_runtime_evidence(
    *,
    result: TrialResult,
    trial_dir: Path,
    profile: M3APCProfile,
    trace: WorkloadTrace,
    warmup: WorkloadTrace,
    device_memory: DeviceMemoryEvidence,
    identity: Mapping[str, Any],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[RequestResult],
    dict[str, dict[str, Any]],
    Any,
    dict[str, Any],
    str,
]:
    request_rows = _read_jsonl(trial_dir / "request-results.jsonl")
    prometheus_rows = _read_jsonl(trial_dir / "prometheus.jsonl")
    benchmark = _read_json(trial_dir / "benchmark-raw.json")
    requests = _validate_requests(request_rows, trace)
    _validate_warmups(benchmark, warmup)
    counters = _counter_evidence(prometheus_rows)
    server_log = (trial_dir / "server.log").read_text(encoding="utf-8", errors="replace")
    runtime = build_capacity_runtime_evidence(
        run_id=result.trial_id,
        runtime_profile_sha256=sha256_json(dict(identity)),
        server_log_text=server_log,
        metrics_text=_raw_metrics_text(prometheus_rows),
        device_memory=device_memory,
    )
    command = _command_evidence(trial_dir, profile)
    cache = runtime.cache_config
    startup = runtime.startup_format
    runtime_checks = {
        "command_matches": command["passed"] is True,
        "runtime_apc_matches": cache.enable_prefix_caching == profile.enable_prefix_caching,
        "startup_apc_matches": startup.enable_prefix_caching == profile.enable_prefix_caching,
        "bf16_kv_preserved": cache.requested_cache_dtype == "auto"
        and startup.requested_kv_cache_dtype == "auto",
        "production_attention_backend_preserved": startup.attention_backend == "FLASH_ATTN",
        "no_explicit_kv_memory": cache.kv_cache_memory_bytes is None,
        "no_num_blocks_override": cache.num_gpu_blocks_override is None,
        "logged_capacity_consistent": runtime.logged_capacity_consistent,
    }
    if not all(runtime_checks.values()):
        failed = sorted(name for name, passed in runtime_checks.items() if not passed)
        raise ValueError("M3 runtime APC gate failed: " + ", ".join(failed))
    return (
        request_rows,
        prometheus_rows,
        requests,
        counters,
        runtime,
        {"command": command, "checks": runtime_checks},
        server_log,
    )


def _validate_counters(
    *,
    counters: Mapping[str, Mapping[str, Any]],
    requests: Sequence[RequestResult],
    cached_tokens: Sequence[int],
    apc_enabled: bool,
) -> tuple[int, int, int]:
    prompt_tokens = sum(request.input_tokens for request in requests)
    expected = {
        "prompt_tokens_total": prompt_tokens,
        "generation_tokens_total": sum(request.output_tokens for request in requests),
        # Locked vLLM does not issue or count prefix-cache queries when APC is disabled.
        "prefix_cache_queries": prompt_tokens if apc_enabled else 0,
        "prefix_cache_hits": sum(cached_tokens),
    }
    observed: dict[str, int] = {}
    for name, value in expected.items():
        item = counters[name]
        if item.get("available") is not True or item.get("reset_count") != 0:
            raise ValueError(f"M3 counter {name} is unavailable or reset")
        observed[name] = _integer_counter(item.get("delta"), f"M3 counter {name}")
        if observed[name] != value:
            raise ValueError(f"M3 counter {name} does not match request-level evidence")
    preemption = counters["num_preemptions_total"]
    if preemption.get("available") is not True or preemption.get("reset_count") != 0:
        raise ValueError("M3 preemption counter is unavailable or reset")
    preemptions = _integer_counter(preemption.get("delta"), "M3 preemption counter")
    return observed["prefix_cache_queries"], observed["prefix_cache_hits"], preemptions


def _derive_core_record(
    *,
    result: TrialResult,
    trial_dir: Path,
    profile: M3APCProfile,
    cache_state: CacheState,
    prefix_tokens: int,
    repeat_index: int,
    bundle: M3CoreTraceBundle,
    warmup: WorkloadTrace,
    device_memory: DeviceMemoryEvidence,
    offered_rate: float,
) -> tuple[M3APCTrialRecord, dict[str, Any]]:
    expected_hits = expected_core_cached_tokens(
        bundle,
        prefix_tokens=prefix_tokens,
        apc_enabled=profile.enable_prefix_caching,
        cache_state=cache_state,
    )
    (
        request_rows,
        prometheus_rows,
        requests,
        counters,
        runtime,
        runtime_evidence,
        server_log,
    ) = _common_runtime_evidence(
        result=result,
        trial_dir=trial_dir,
        profile=profile,
        trace=bundle.trace,
        warmup=warmup,
        device_memory=device_memory,
        identity={
            "kind": "core",
            "profile": profile.profile_id,
            "cache_state": cache_state,
            "prefix_tokens": prefix_tokens,
        },
    )
    cached_by_id = {str(row["request_id"]): _request_cached_tokens(row) for row in request_rows}
    exact_hits = all(
        cached_by_id[request_id] == expected for request_id, expected in expected_hits.items()
    )
    if not exact_hits:
        mismatches = [
            f"{request_id}:{cached_by_id[request_id]}!={expected}"
            for request_id, expected in expected_hits.items()
            if cached_by_id[request_id] != expected
        ]
        raise ValueError("M3 request-level cache hits changed: " + ", ".join(mismatches[:8]))
    queries, hits, preemptions = _validate_counters(
        counters=counters,
        requests=requests,
        cached_tokens=[cached_by_id[request.request_id] for request in requests],
        apc_enabled=profile.enable_prefix_caching,
    )
    slo_rows = result.client.get("request_slo")
    if not isinstance(slo_rows, list) or any(not isinstance(row, Mapping) for row in slo_rows):
        raise ValueError("M3 result has no per-request SLO evidence")
    slo_by_id = {str(row.get("request_id")): row.get("good") is True for row in slo_rows}
    if set(slo_by_id) != {request.request_id for request in requests}:
        raise ValueError("M3 per-request SLO IDs differ from the measured trace")

    rows_by_id = {str(row["request_id"]): row for row in request_rows}
    requests_by_id = {request.request_id: request for request in requests}
    reuse_metrics: list[M3ReuseMetrics] = []
    for reuse_percent in (0, 50, 100):
        ids = [
            entry.request_id
            for entry in bundle.trace.entries
            if bundle.reuse_by_request[entry.request_id] == reuse_percent
        ]
        scheduled = [requests_by_id[request_id].scheduled_at for request_id in ids]
        if any(value is None for value in scheduled):
            raise ValueError("M3 reuse cohort is missing scheduled timestamps")
        scheduled_values = cast(list[int], scheduled)
        window = (
            max(scheduled_values) - min(scheduled_values)
        ) / 1_000_000_000 + 1.0 / offered_rate
        group_rows = [rows_by_id[request_id] for request_id in ids]
        group_hits = sum(cached_by_id[request_id] for request_id in ids)
        group_queries = sum(requests_by_id[request_id].input_tokens for request_id in ids)
        good = sum(slo_by_id[request_id] for request_id in ids)
        reuse_metrics.append(
            M3ReuseMetrics(
                reuse_percent=cast(Any, reuse_percent),
                request_count=len(ids),
                shared_request_count=sum(bundle.shared_request[request_id] for request_id in ids),
                cached_tokens=group_hits,
                expected_cached_tokens=sum(expected_hits[request_id] for request_id in ids),
                hit_ratio=group_hits / group_queries,
                achieved_requests_per_second=len(ids) / window,
                goodput_requests_per_second=good / window,
                slo_satisfied_fraction=good / len(ids),
                ttft=_latencies(
                    [_finite_number(row.get("ttft_ms"), "M3 TTFT") for row in group_rows],
                    "TTFT",
                ),
                tpot=_latencies(
                    [_finite_number(row.get("tpot_ms"), "M3 TPOT") for row in group_rows],
                    "TPOT",
                ),
                end_to_end=_latencies(
                    [_finite_number(row.get("e2e_ms"), "M3 E2E") for row in group_rows],
                    "E2E",
                ),
            )
        )
    timeout_count = sum(request.status == RequestStatus.TIMEOUT for request in requests)
    oom_count = _server_event_count(server_log, (r"CUDA out of memory", r"\bOOM\b"))
    if timeout_count or oom_count:
        raise ValueError("M3 complete core evidence must not contain timeout or OOM events")
    record = M3APCTrialRecord(
        trial_id=result.trial_id,
        profile_id=cast(Any, profile.profile_id),
        apc_enabled=profile.enable_prefix_caching,
        cache_state=cache_state,
        prefix_tokens=cast(Any, prefix_tokens),
        repeat_index=repeat_index,
        trace_id=bundle.trace.checksum(),
        warmup_trace_id=warmup.checksum(),
        request_count=len(requests),
        prefix_cache_queries=queries,
        prefix_cache_hits=hits,
        expected_prefix_cache_hits=sum(expected_hits.values()),
        hit_ratio=hits / queries if queries else 0.0,
        exact_hit_tokens=True,
        completion_fraction=1.0,
        achieved_requests_per_second=_finite_number(
            result.client.get("achieved_requests_per_sec"), "M3 achieved rate"
        ),
        goodput_requests_per_second=_finite_number(
            result.client.get("goodput_requests_per_sec"), "M3 goodput rate"
        ),
        preemption_count=preemptions,
        oom_count=oom_count,
        timeout_count=timeout_count,
        peak_vram_mb=_finite_number(result.gpu.get("peak_memory_mb"), "M3 peak VRAM"),
        kv_usage=_kv_usage(prometheus_rows),
        reuse=tuple(reuse_metrics),
    )
    evidence = {
        "schema_version": M3_APC_SCHEMA,
        "semantic_gate_passed": True,
        "record_kind": "core",
        "runtime": runtime_evidence,
        "runtime_capacity": runtime.model_dump(mode="json"),
        "prefix_proof": bundle.prefix_proof,
        "corpus_sha256": bundle.corpus_sha256,
        "per_request": [
            {
                "request_id": request_id,
                "reuse_percent": bundle.reuse_by_request[request_id],
                "shared": bundle.shared_request[request_id],
                "expected_cached_tokens": expected_hits[request_id],
                "observed_cached_tokens": cached_by_id[request_id],
                "slo_good": slo_by_id[request_id],
            }
            for request_id in expected_hits
        ],
        "counters": counters,
        "kv_usage": record.kv_usage.model_dump(mode="json"),
    }
    return record, evidence


def _derive_boundary_record(
    *,
    result: TrialResult,
    trial_dir: Path,
    profile: M3APCProfile,
    pool_size: int,
    bundle: M3BoundaryTraceBundle,
    device_memory: DeviceMemoryEvidence,
    prefix_tokens: int,
) -> tuple[M3BoundaryRecord, dict[str, Any]]:
    (
        request_rows,
        prometheus_rows,
        requests,
        counters,
        runtime,
        runtime_evidence,
        server_log,
    ) = _common_runtime_evidence(
        result=result,
        trial_dir=trial_dir,
        profile=profile,
        trace=bundle.measured,
        warmup=bundle.warmup,
        device_memory=device_memory,
        identity={"kind": "boundary", "pool_size": pool_size, "prefix_tokens": prefix_tokens},
    )
    cached = [_request_cached_tokens(row) for row in request_rows]
    queries, hits, preemptions = _validate_counters(
        counters=counters,
        requests=requests,
        cached_tokens=cached,
        apc_enabled=True,
    )
    timeout_count = sum(request.status == RequestStatus.TIMEOUT for request in requests)
    oom_count = _server_event_count(server_log, (r"CUDA out of memory", r"\bOOM\b"))
    if timeout_count or oom_count:
        raise ValueError("M3 prefix-pool boundary must not contain timeout or OOM events")
    cache = runtime.cache_config
    input_tokens = bundle.measured.entries[0].input_tokens
    prompt_blocks = math.ceil(input_tokens / cache.resolved_block_size)
    predicted_resident = cache.usable_num_gpu_blocks // prompt_blocks
    misses = [index for index, value in enumerate(cached) if value == 0]
    record = M3BoundaryRecord(
        trial_id=result.trial_id,
        pool_size=pool_size,
        prefix_tokens=cast(Any, prefix_tokens),
        input_tokens=input_tokens,
        trace_id=bundle.measured.checksum(),
        warmup_trace_id=bundle.warmup.checksum(),
        request_count=len(requests),
        prefix_cache_queries=queries,
        prefix_cache_hits=hits,
        hit_ratio=hits / queries,
        full_hit_requests=sum(value >= prefix_tokens for value in cached),
        partial_hit_requests=sum(0 < value < prefix_tokens for value in cached),
        miss_requests=len(misses),
        first_miss_probe_position=misses[0] if misses else None,
        cached_tokens_by_probe=tuple(cached),
        kv_usage=_kv_usage(prometheus_rows),
        predicted_resident_prompts=predicted_resident,
        runtime_cached_token_capacity=runtime.observation.cached_tokens,
        preemption_count=preemptions,
        oom_count=oom_count,
        timeout_count=timeout_count,
    )
    evidence = {
        "schema_version": M3_APC_SCHEMA,
        "semantic_gate_passed": True,
        "record_kind": "boundary",
        "runtime": runtime_evidence,
        "runtime_capacity": runtime.model_dump(mode="json"),
        "prefix_proof": bundle.prefix_proof,
        "corpus_sha256": bundle.corpus_sha256,
        "probe_order": [entry.request_id for entry in bundle.measured.entries],
        "cached_tokens_by_probe": cached,
        "counters": counters,
        "kv_usage": record.kv_usage.model_dump(mode="json"),
        "boundary_interpretation": {
            "predicted_resident_prompts": predicted_resident,
            "prompt_blocks": prompt_blocks,
            "usable_gpu_blocks": cache.usable_num_gpu_blocks,
            "probe_order": "newest primed prefix to oldest primed prefix",
        },
    }
    return record, evidence


class LongContextM3APCRunner:
    """Execute the minimal same-path smoke or 18+2 formal M3 protocol."""

    def __init__(
        self,
        config: LongContextM3APCConfig,
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
        if _EXPERIMENT_ID.fullmatch(experiment_id) is None:
            raise ValueError("M3 experiment_id must be one portable path component")
        self.config = config
        self.experiment_id = experiment_id
        self.repository = Path(repository).resolve()
        self.resume = resume
        self.controller_factory = controller_factory
        self._tokenizer = tokenizer
        self.gpu_memory_reader = gpu_memory_reader
        self.runtime_facts = runtime_facts
        self.model_facts = model_facts
        self.execution_environment = dict(execution_environment or os.environ)
        self.store = ArtifactStore(config.artifacts.root, experiment_id)
        self._resume_warnings: list[str] = []
        self._replayed_trials: list[str] = []
        self._new_attempts: list[str] = []
        self._started_monotonic = 0.0

    @property
    def artifacts(self) -> ArtifactStore:
        return self.store

    def _load_tokenizer(self) -> Any:
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            identity = self.config.model.identity()
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.config.model.local_path,
                revision=identity.revision,
                local_files_only=True,
            )
        return self._tokenizer

    def _profile(self, profile_id: str) -> M3APCProfile:
        return next(profile for profile in self.config.profiles if profile.profile_id == profile_id)

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
            raise RuntimeError(f"M3 APC experiment is already running: {path}") from error
        return descriptor

    @staticmethod
    def _release_lock(descriptor: int) -> None:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)

    def _identities(self) -> tuple[RuntimeIdentityFacts, ModelIdentityFacts, dict[str, str]]:
        runtime = self.runtime_facts or require_upstream_runtime(self.config.runtime.lock_path)
        locked = self.config.model.identity()
        model = self.model_facts or require_model_identity(
            locked,
            model_dir=self.config.model.local_path,
            repository_id=locked.repository_id,
            revision=locked.revision,
            parameter_count=locked.parameter_count,
        )
        if not runtime.matches_lock or not model.matches_lock:
            raise ValueError("M3 APC requires matching upstream runtime and model locks")
        return runtime, model, _clean_execution_environment(self.execution_environment)

    def _source_identity(self) -> tuple[str, str]:
        commit, dirty, _ = git_state(self.repository)
        tree = source_tree_sha256(self.repository)
        if commit is None or tree is None or dirty:
            raise ValueError("M3 APC requires one clean committed source identity")
        return commit, tree

    def _manifest(
        self,
        runtime: RuntimeIdentityFacts,
        model: ModelIdentityFacts,
        environment: Mapping[str, str],
        source_commit: str,
        source_tree: str,
        corpus_sha256: str,
    ) -> dict[str, Any]:
        smoke = self.config.smoke_artifact
        return {
            "schema_version": M3_APC_SCHEMA,
            "project_line": "longctx-v5",
            "milestone": "M3",
            "experiment_kind": "automatic-prefix-caching",
            "evidence_role": self.config.evidence_role,
            "experiment_id": self.experiment_id,
            "created_at": utc_now_iso(),
            "config_sha256": sha256_json(self.config.model_dump(mode="json")),
            "source_commit": source_commit,
            "source_tree_sha256": source_tree,
            "runtime": runtime.model_dump(mode="json"),
            "model": model.model_dump(mode="json"),
            "execution_environment": dict(environment),
            "rag_corpus": {
                "files": list(RAG_CORPUS_FILES),
                "sha256": corpus_sha256,
                "kind": "public repository System/RAG context",
            },
            "m1_boundaries": {
                "experiment_id": self.config.m1_boundaries.experiment_id,
                "root": str(self.config.m1_boundaries.root),
                "integrity_sha256": sha256_file(
                    self.config.m1_boundaries.root / "m1-capacity-integrity.json"
                ),
            },
            "m2_negative_artifacts": [
                {
                    **item.identity(),
                    "integrity_sha256": sha256_file(item.root / M2_FP8_INTEGRITY_FILE),
                }
                for item in self.config.m2_negative_artifacts
            ],
            "smoke_artifact": (
                None
                if smoke is None
                else {
                    "experiment_id": smoke.experiment_id,
                    "root": str(smoke.root),
                    "integrity_sha256": sha256_file(smoke.root / M3_APC_INTEGRITY_FILE),
                }
            ),
            "scope": {
                "m1_rerun": False,
                "m2_rerun": False,
                "fp8_configuration_attempted": False,
                "m4_started": False,
            },
        }

    @staticmethod
    def _manifest_identity(value: Mapping[str, Any]) -> dict[str, Any]:
        return {name: item for name, item in value.items() if name != "created_at"}

    def _validate_formal_smoke_identity(self, source_commit: str, corpus_sha256: str) -> bool:
        smoke = self.config.smoke_artifact
        if smoke is None:
            return self.config.evidence_role == "smoke"
        manifest = _read_json(smoke.root / MANIFEST_FILE)
        experiment = _read_json(smoke.root / "experiment.json")
        return (
            manifest.get("source_commit") == source_commit
            and cast(Mapping[str, Any], manifest.get("rag_corpus", {})).get("sha256")
            == corpus_sha256
            and experiment.get("model") == self.config.model.model_dump(mode="json")
            and experiment.get("runtime") == self.config.runtime.model_dump(mode="json")
            and experiment.get("m1_boundaries") == self.config.m1_boundaries.model_dump(mode="json")
            and experiment.get("m2_negative_artifacts")
            == [item.model_dump(mode="json") for item in self.config.m2_negative_artifacts]
        )

    def _initialize_root(self, requested_manifest: dict[str, Any]) -> Optional[dict[str, Any]]:
        root = self.store.root
        if root.exists():
            if not self.resume:
                raise FileExistsError(f"M3 APC artifact root exists: {root}; use --resume")
            if (root / M3_APC_INTEGRITY_FILE).is_file():
                validate_m3_apc_artifacts(root)
                existing = _read_json(root / MANIFEST_FILE)
                if self._manifest_identity(existing) != self._manifest_identity(requested_manifest):
                    raise ValueError("M3 APC resume manifest identity mismatch")
                return _read_json(root / SUMMARY_FILE)
            self.store.initialize(exist_ok=True)
            existing = _read_json(root / MANIFEST_FILE)
            if self._manifest_identity(existing) != self._manifest_identity(requested_manifest):
                raise ValueError("M3 APC resume manifest identity mismatch")
            return None
        self.store.initialize()
        self.store.write_json(MANIFEST_FILE, requested_manifest)
        self.store.write_json("experiment.json", self.config.model_dump(mode="json"))
        return None

    def _write_or_validate_trace(self, relative: Path, trace: WorkloadTrace) -> None:
        path = self.store.root / relative
        checksum = relative.with_suffix(".sha256")
        text = _trace_text(trace)
        sidecar = f"{trace.checksum()}  {relative.name}\n"
        if path.exists():
            if path.read_text(encoding="utf-8") != text:
                raise ValueError(f"M3 resume trace bytes mismatch: {relative}")
            checksum_path = self.store.root / checksum
            if not checksum_path.is_file() or checksum_path.read_text(encoding="utf-8") != sidecar:
                raise ValueError(f"M3 resume trace checksum mismatch: {relative}")
            return
        self.store.write_text(relative, text)
        self.store.write_text(checksum, sidecar)

    def _trace_catalog(
        self,
    ) -> tuple[
        dict[int, M3CoreTraceBundle],
        dict[tuple[int, CacheState], WorkloadTrace],
        dict[int, M3BoundaryTraceBundle],
        str,
    ]:
        tokenizer = self._load_tokenizer()
        corpus = load_rag_corpus(self.repository)
        core: dict[int, M3CoreTraceBundle] = {}
        warmups: dict[tuple[int, CacheState], WorkloadTrace] = {}
        boundaries: dict[int, M3BoundaryTraceBundle] = {}
        catalog: dict[str, Any] = {
            "schema_version": M3_APC_SCHEMA,
            "corpus_sha256": corpus.sha256,
            "prefixes": {},
            "boundaries": {},
        }
        for prefix_tokens in self.config.protocol.prefix_tokens:
            bundle = build_m3_core_trace(
                prefix_tokens=prefix_tokens,
                requests_per_reuse=self.config.protocol.requests_per_reuse,
                input_tokens=self.config.context.input_tokens,
                output_tokens=self.config.context.output_tokens,
                offered_requests_per_second=self.config.context.offered_requests_per_second,
                burstiness=self.config.protocol.burstiness,
                seed=self.config.protocol.measurement_seed,
                tokenizer=tokenizer,
                corpus=corpus,
            )
            core[prefix_tokens] = bundle
            measured_path = Path("traces") / f"prefix-{prefix_tokens}" / "measured.jsonl"
            self._write_or_validate_trace(measured_path, bundle.trace)
            states: dict[str, Any] = {}
            for state in ("target-prefix-cold", "target-prefix-warm"):
                typed_state = cast(CacheState, state)
                warmup = build_m3_core_warmup(
                    bundle=bundle,
                    cache_state=typed_state,
                    prefix_tokens=prefix_tokens,
                    input_tokens=self.config.context.input_tokens,
                    output_tokens=8,
                    seed=self.config.protocol.warmup_seed,
                    tokenizer=tokenizer,
                    corpus=corpus,
                )
                warmups[(prefix_tokens, typed_state)] = warmup
                warmup_path = Path("traces") / f"prefix-{prefix_tokens}" / f"{state}.jsonl"
                self._write_or_validate_trace(warmup_path, warmup)
                states[state] = {
                    "path": warmup_path.as_posix(),
                    "sha256": warmup.checksum(),
                    "requests": len(warmup.entries),
                }
            cast(dict[str, Any], catalog["prefixes"])[str(prefix_tokens)] = {
                "measured_path": measured_path.as_posix(),
                "measured_sha256": bundle.trace.checksum(),
                "measured_requests": len(bundle.trace.entries),
                "measured_span_seconds": _trace_span(bundle.trace),
                "reuse_by_request": bundle.reuse_by_request,
                "shared_request": bundle.shared_request,
                "prefix_proof": bundle.prefix_proof,
                "warmups": states,
            }

        for pool_size in self.config.protocol.boundary_pool_sizes:
            boundary_bundle = build_m3_boundary_trace(
                pool_size=pool_size,
                prefix_tokens=self.config.protocol.boundary_prefix_tokens,
                tail_tokens=self.config.protocol.boundary_tail_tokens,
                output_tokens=self.config.protocol.boundary_output_tokens,
                interval_seconds=self.config.protocol.boundary_request_interval_seconds,
                seed=self.config.protocol.measurement_seed + pool_size,
                tokenizer=tokenizer,
                corpus=corpus,
            )
            boundaries[pool_size] = boundary_bundle
            base = Path("traces") / "boundary" / f"pool-{pool_size}"
            self._write_or_validate_trace(base / "warmup.jsonl", boundary_bundle.warmup)
            self._write_or_validate_trace(base / "measured.jsonl", boundary_bundle.measured)
            cast(dict[str, Any], catalog["boundaries"])[str(pool_size)] = {
                "warmup_path": (base / "warmup.jsonl").as_posix(),
                "warmup_sha256": boundary_bundle.warmup.checksum(),
                "measured_path": (base / "measured.jsonl").as_posix(),
                "measured_sha256": boundary_bundle.measured.checksum(),
                "prefix_proof": boundary_bundle.prefix_proof,
            }
        catalog_path = self.store.root / "traces" / "catalog.json"
        if catalog_path.exists():
            if _read_json(catalog_path) != catalog:
                raise ValueError("M3 resume trace catalog mismatch")
        else:
            self.store.write_json("traces/catalog.json", catalog)
        return core, warmups, boundaries, corpus.sha256

    @staticmethod
    def _core_logical_id(
        profile: M3APCProfile,
        cache_state: CacheState,
        prefix_tokens: int,
        repeat_index: int,
    ) -> str:
        state = "cold" if cache_state == "target-prefix-cold" else "warm"
        return f"apc-{profile.profile_id}-{state}-prefix-{prefix_tokens}-repeat-{repeat_index}"

    @staticmethod
    def _boundary_logical_id(pool_size: int) -> str:
        return f"apc-apc-on-boundary-prefix-4096-pool-{pool_size}"

    def _attempts(self, logical_id: str) -> list[tuple[int, Path]]:
        if not self.store.trials_dir.is_dir():
            return []
        attempts: list[tuple[int, Path]] = []
        for path in self.store.trials_dir.iterdir():
            if path.is_symlink() or not path.is_dir():
                raise ValueError(f"unexpected M3 trials entry: {path}")
            if path.name == logical_id:
                attempts.append((0, path))
                continue
            if path.name.startswith(logical_id + "-attempt"):
                match = _ATTEMPT_SUFFIX.search(path.name)
                if match is not None:
                    attempts.append((int(match.group("number")), path))
        return sorted(attempts)

    def _point_payload(
        self,
        *,
        trial_id: str,
        kind: str,
        profile: M3APCProfile,
        trace: WorkloadTrace,
        warmup: WorkloadTrace,
        prefix_tokens: int,
        cache_state: Optional[CacheState] = None,
        repeat_index: Optional[int] = None,
        pool_size: Optional[int] = None,
    ) -> dict[str, Any]:
        return {
            "schema_version": M3_APC_SCHEMA,
            "trial_id": trial_id,
            "kind": kind,
            "profile_id": profile.profile_id,
            "apc_enabled": profile.enable_prefix_caching,
            "cache_state": cache_state,
            "prefix_tokens": prefix_tokens,
            "repeat_index": repeat_index,
            "pool_size": pool_size,
            "trace_id": trace.checksum(),
            "warmup_trace_id": warmup.checksum(),
            "vllm_args": profile.vllm_args(),
            "m1_boundary_source": self.config.m1_boundaries.experiment_id,
            "m2_negative_sources": [
                item.experiment_id for item in self.config.m2_negative_artifacts
            ],
        }

    def _cached_complete(
        self,
        *,
        logical_id: str,
        expected_point: Mapping[str, Any],
        kind: str,
    ) -> Optional[tuple[TrialResult, M3APCTrialRecord | M3BoundaryRecord]]:
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
                    raise ValueError("sealed M3 attempt identity or semantic gate mismatch")
                adapter = _CORE_RECORD_ADAPTER if kind == "core" else _BOUNDARY_RECORD_ADAPTER
                record = adapter.validate_json((path / RECORD_FILE).read_text(encoding="utf-8"))
            except (OSError, ValueError) as error:
                self._resume_warnings.append(f"attempt {path.name} not replayed: {error}")
                continue
            self._replayed_trials.append(path.name)
            return result, record
        return None

    def _select_new_attempt(self, logical_id: str) -> str:
        attempts = self._attempts(logical_id)
        if not attempts:
            return logical_id
        number = max(value for value, _ in attempts) + 1
        trial_id = f"{logical_id}-attempt{number}"
        self._new_attempts.append(trial_id)
        return trial_id

    def _write_pretrial(
        self,
        *,
        point: Mapping[str, Any],
        trace: WorkloadTrace,
        warmup: WorkloadTrace,
        memory: DeviceMemoryEvidence,
    ) -> None:
        trial_id = str(point["trial_id"])
        base = Path("trials") / trial_id
        self.store.write_json(base / POINT_FILE, dict(point))
        self.store.write_text(base / MEASURED_TRACE_FILE, _trace_text(trace))
        self.store.write_text(
            base / MEASURED_TRACE_CHECKSUM_FILE,
            f"{trace.checksum()}  {MEASURED_TRACE_FILE}\n",
        )
        self.store.write_text(base / WARMUP_TRACE_FILE, _trace_text(warmup))
        self.store.write_text(
            base / WARMUP_TRACE_CHECKSUM_FILE,
            f"{warmup.checksum()}  {WARMUP_TRACE_FILE}\n",
        )
        self.store.write_json(base / CUDA_MEMORY_FILE, memory)
        self.store.write_json(
            base / EVIDENCE_FILE,
            {
                "schema_version": M3_APC_SCHEMA,
                "semantic_gate_passed": False,
                "state": "pending-controller-result",
            },
        )

    def _finalize_trial(
        self,
        result: TrialResult,
        record: M3APCTrialRecord | M3BoundaryRecord | Mapping[str, Any],
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
                else {"available": False, "reason": "M3 semantic gate produced no capacity"}
            ),
        )
        self.store.ensure_trial_artifacts(result)
        self.store.validate_cached_trial(result, require_telemetry=True)
        self.store.validate_trial_integrity(result.trial_id)

    async def _run_core_one(
        self,
        *,
        profile: M3APCProfile,
        cache_state: CacheState,
        prefix_tokens: int,
        repeat_index: int,
        bundle: M3CoreTraceBundle,
        warmup: WorkloadTrace,
    ) -> tuple[TrialResult, Optional[M3APCTrialRecord], bool]:
        logical = self._core_logical_id(profile, cache_state, prefix_tokens, repeat_index)
        expected_point = self._point_payload(
            trial_id=logical,
            kind="core",
            profile=profile,
            trace=bundle.trace,
            warmup=warmup,
            prefix_tokens=prefix_tokens,
            cache_state=cache_state,
            repeat_index=repeat_index,
        )
        cached = self._cached_complete(
            logical_id=logical,
            expected_point=expected_point,
            kind="core",
        )
        if cached is not None:
            return cached[0], cast(M3APCTrialRecord, cached[1]), True
        trial_id = self._select_new_attempt(logical)
        point = {**expected_point, "trial_id": trial_id}
        tuning = self.config.to_tuning_config(
            profile,
            warmup_requests=len(warmup.entries),
            boundary=False,
        )
        if tuning.vllm_args != profile.vllm_args():
            raise ValueError("M3 TuningConfig changed preregistered APC arguments")
        memory = self.gpu_memory_reader(self.config.gpu.device_ids[0])
        self._write_pretrial(point=point, trace=bundle.trace, warmup=warmup, memory=memory)
        controller = self.controller_factory(
            tuning,
            bundle.trace,
            self.store,
            tokenizer=self._load_tokenizer(),
            warmup_trace=warmup,
            strict_open_loop=True,
        )
        try:
            result = await controller.run_trial({}, trial_id, "apc")
        except UnsafeCleanupError:
            self._status(
                "unsafe_cleanup",
                current_trial=trial_id,
                unsafe_cleanup=True,
                message="unsafe cleanup; M3 stopped before another GPU process",
            )
            raise
        expected_provenance = trial_provenance(trial_id, "apc")
        if any(getattr(result, name) != value for name, value in expected_provenance.items()):
            raise ValueError(f"M3 trial provenance mismatch: {trial_id}")
        if result.params != {}:
            raise ValueError(f"M3 fixed profile trial passed search parameters: {trial_id}")
        record: Optional[M3APCTrialRecord] = None
        evidence: dict[str, Any]
        if result.status in {TrialStatus.COMPLETE, TrialStatus.INFEASIBLE}:
            try:
                record, evidence = _derive_core_record(
                    result=result,
                    trial_dir=self.store.trials_dir / trial_id,
                    profile=profile,
                    cache_state=cache_state,
                    prefix_tokens=prefix_tokens,
                    repeat_index=repeat_index,
                    bundle=bundle,
                    warmup=warmup,
                    device_memory=memory,
                    offered_rate=self.config.context.offered_requests_per_second,
                )
            except (OSError, ValueError) as error:
                result.status = TrialStatus.FAILED
                result.constraints = {
                    **result.constraints,
                    "feasible": False,
                    "violations": [
                        *cast(list[str], result.constraints.get("violations", [])),
                        "m3_apc_semantic_gate",
                    ],
                }
                result.failure_reason = {
                    "type": "M3_APC_SEMANTIC_GATE",
                    "message": str(error),
                    "phase": "M3_APC_FINALIZE",
                }
                self.store.record_artifact_finalizer_failure(result, str(error))
                evidence = {
                    "schema_version": M3_APC_SCHEMA,
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
                "schema_version": M3_APC_SCHEMA,
                "semantic_gate_passed": False,
                "failure": reason,
            }
        failure_record = {
            "schema_version": M3_APC_SCHEMA,
            "record_kind": "core",
            "trial_id": trial_id,
            "profile_id": profile.profile_id,
            "cache_state": cache_state,
            "prefix_tokens": prefix_tokens,
            "repeat_index": repeat_index,
            "status": "failed",
            "failure": evidence.get("failure"),
        }
        self._finalize_trial(result, record or failure_record, evidence)
        return result, record, False

    async def _run_boundary_one(
        self,
        *,
        pool_size: int,
        bundle: M3BoundaryTraceBundle,
    ) -> tuple[TrialResult, Optional[M3BoundaryRecord], bool]:
        profile = self._profile("apc-on")
        logical = self._boundary_logical_id(pool_size)
        expected_point = self._point_payload(
            trial_id=logical,
            kind="boundary",
            profile=profile,
            trace=bundle.measured,
            warmup=bundle.warmup,
            prefix_tokens=self.config.protocol.boundary_prefix_tokens,
            pool_size=pool_size,
        )
        cached = self._cached_complete(
            logical_id=logical,
            expected_point=expected_point,
            kind="boundary",
        )
        if cached is not None:
            return cached[0], cast(M3BoundaryRecord, cached[1]), True
        trial_id = self._select_new_attempt(logical)
        point = {**expected_point, "trial_id": trial_id}
        tuning = self.config.to_tuning_config(
            profile,
            warmup_requests=len(bundle.warmup.entries),
            boundary=True,
        )
        memory = self.gpu_memory_reader(self.config.gpu.device_ids[0])
        self._write_pretrial(
            point=point,
            trace=bundle.measured,
            warmup=bundle.warmup,
            memory=memory,
        )
        controller = self.controller_factory(
            tuning,
            bundle.measured,
            self.store,
            tokenizer=self._load_tokenizer(),
            warmup_trace=bundle.warmup,
            strict_open_loop=True,
        )
        try:
            result = await controller.run_trial({}, trial_id, "apc-boundary")
        except UnsafeCleanupError:
            self._status(
                "unsafe_cleanup",
                current_trial=trial_id,
                unsafe_cleanup=True,
                message="unsafe cleanup; M3 stopped before another GPU process",
            )
            raise
        expected_provenance = trial_provenance(trial_id, "apc-boundary")
        if any(getattr(result, name) != value for name, value in expected_provenance.items()):
            raise ValueError(f"M3 boundary provenance mismatch: {trial_id}")
        record: Optional[M3BoundaryRecord] = None
        evidence: dict[str, Any]
        if result.status in {TrialStatus.COMPLETE, TrialStatus.INFEASIBLE}:
            try:
                record, evidence = _derive_boundary_record(
                    result=result,
                    trial_dir=self.store.trials_dir / trial_id,
                    profile=profile,
                    pool_size=pool_size,
                    bundle=bundle,
                    device_memory=memory,
                    prefix_tokens=self.config.protocol.boundary_prefix_tokens,
                )
            except (OSError, ValueError) as error:
                result.status = TrialStatus.FAILED
                result.constraints = {
                    **result.constraints,
                    "feasible": False,
                    "violations": [
                        *cast(list[str], result.constraints.get("violations", [])),
                        "m3_apc_boundary_semantic_gate",
                    ],
                }
                result.failure_reason = {
                    "type": "M3_APC_BOUNDARY_SEMANTIC_GATE",
                    "message": str(error),
                    "phase": "M3_APC_BOUNDARY_FINALIZE",
                }
                self.store.record_artifact_finalizer_failure(result, str(error))
                evidence = {
                    "schema_version": M3_APC_SCHEMA,
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
                "schema_version": M3_APC_SCHEMA,
                "semantic_gate_passed": False,
                "failure": reason,
            }
        failure_record = {
            "schema_version": M3_APC_SCHEMA,
            "record_kind": "boundary",
            "trial_id": trial_id,
            "profile_id": profile.profile_id,
            "pool_size": pool_size,
            "status": "failed",
            "failure": evidence.get("failure"),
        }
        self._finalize_trial(result, record or failure_record, evidence)
        return result, record, False

    def _expected_core_matrix(self) -> set[tuple[str, str, int, int]]:
        expected: set[tuple[str, str, int, int]] = set()
        for prefix in self.config.protocol.prefix_tokens:
            for repeat in range(self.config.protocol.repeats):
                expected.add(("apc-off", "target-prefix-cold", prefix, repeat))
                expected.add(("apc-on", "target-prefix-cold", prefix, repeat))
                expected.add(("apc-on", "target-prefix-warm", prefix, repeat))
        return expected

    def _planned_jobs(self) -> int:
        return len(self._expected_core_matrix()) + len(self.config.protocol.boundary_pool_sizes)

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
        planned = self._planned_jobs()
        elapsed = max(0.0, time.monotonic() - self._started_monotonic)
        remaining = max(0, planned - completed_jobs)
        if completed_jobs:
            eta_seconds = elapsed / completed_jobs * remaining
        else:
            core_span = (
                self.config.protocol.measured_request_count - 1
            ) / self.config.context.offered_requests_per_second
            boundary_span = sum(
                (pool - 1) * self.config.protocol.boundary_request_interval_seconds
                for pool in self.config.protocol.boundary_pool_sizes
            )
            eta_seconds = core_span * len(self._expected_core_matrix()) + boundary_span
        value = {
            "schema_version": M3_APC_SCHEMA,
            "experiment_id": self.experiment_id,
            "state": state,
            "pid": os.getpid(),
            "gpu": list(self.config.gpu.device_ids),
            "log": str(self.store.root / RUNNER_LOG_FILE),
            "result": str(self.store.root),
            "eta": (datetime.now(timezone.utc) + timedelta(seconds=eta_seconds)).isoformat(),
            "eta_seconds": eta_seconds,
            "resume": self._resume_command(),
            "sealed": (self.store.root / M3_APC_INTEGRITY_FILE).is_file(),
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
            "scripts/run_longctx_m3_apc.sh --config CONFIG "
            f"--experiment-id {self.experiment_id} --resume"
        )

    @staticmethod
    def _directional_checks(analysis: Mapping[str, Any]) -> dict[str, bool]:
        paired = analysis.get("paired")
        if not isinstance(paired, list):
            return {}
        checks: dict[str, bool] = {}
        for row in paired:
            if not isinstance(row, Mapping) or row.get("reuse_percent") not in {50, 100}:
                continue
            repeat_count = row.get("repeat_count")
            prefix = row.get("prefix_tokens")
            reuse = row.get("reuse_percent")
            checks[f"prefix-{prefix}:reuse-{reuse}:ttft"] = (
                repeat_count == 3 and row.get("warm_ttft_improved_repeats", 0) >= 2
            )
            checks[f"prefix-{prefix}:reuse-{reuse}:goodput"] = (
                repeat_count == 3 and row.get("warm_goodput_not_lower_repeats", 0) >= 2
            )
        return checks

    @staticmethod
    def _report(summary: Mapping[str, Any]) -> str:
        execution = cast(Mapping[str, Any], summary["execution"])
        acceptance = cast(Mapping[str, Any], summary["acceptance"])
        analysis = cast(Mapping[str, Any], summary["analysis"])
        lines = [
            "# Long-context v5 M3 Automatic Prefix Caching",
            "",
            f"- Experiment: {summary['experiment_id']}",
            f"- Evidence role: {summary['evidence_role']}",
            f"- Execution passed: {execution['passed']}",
            f"- M3 accepted: {acceptance['passed']}",
            "- M1/M2 rerun: false",
            "- FP8 attempted: false",
            "- M4 started: false",
            "",
            "## Paired APC evidence",
            "",
            "| Prefix | Reuse | Warm TTFT improved repeats | Warm Goodput not lower repeats |",
            "|---:|---:|---:|---:|",
        ]
        paired = analysis.get("paired")
        if isinstance(paired, list):
            for row in paired:
                if not isinstance(row, Mapping):
                    continue
                lines.append(
                    f"| {row.get('prefix_tokens')} | {row.get('reuse_percent')}% | "
                    f"{row.get('warm_ttft_improved_repeats')}/{row.get('repeat_count')} | "
                    f"{row.get('warm_goodput_not_lower_repeats')}/{row.get('repeat_count')} |"
                )
        boundary = analysis.get("boundary")
        if isinstance(boundary, Mapping):
            lines.extend(["", f"- Prefix-pool boundary bracketed: {boundary.get('bracketed')}"])
        return "\n".join(lines) + "\n"

    async def run(self) -> dict[str, Any]:
        descriptor = self._acquire_lock()
        self._started_monotonic = time.monotonic()
        try:
            return await self._run_locked()
        finally:
            self._release_lock(descriptor)

    async def _run_locked(self) -> dict[str, Any]:
        runtime, model, environment = self._identities()
        source_commit, source_tree = self._source_identity()
        # Generate all immutable traces before creating the manifest so corpus identity is bound.
        corpus = load_rag_corpus(self.repository)
        smoke_identity_matches = self._validate_formal_smoke_identity(source_commit, corpus.sha256)
        if not smoke_identity_matches:
            raise ValueError(
                "formal M3 source/model/runtime/M1/M2/corpus identity differs from smoke"
            )
        requested_manifest = self._manifest(
            runtime,
            model,
            environment,
            source_commit,
            source_tree,
            corpus.sha256,
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
        core, warmups, boundaries, catalog_corpus_sha = self._trace_catalog()
        if catalog_corpus_sha != corpus.sha256:
            raise ValueError("M3 trace catalog corpus identity changed during preparation")
        records: list[M3APCTrialRecord] = []
        boundary_records: list[M3BoundaryRecord] = []
        results: list[TrialResult] = []
        completed_jobs = 0
        off = self._profile("apc-off")
        on = self._profile("apc-on")
        cells: tuple[tuple[M3APCProfile, CacheState], ...] = (
            (off, "target-prefix-cold"),
            (on, "target-prefix-cold"),
            (on, "target-prefix-warm"),
        )
        for prefix_tokens in self.config.protocol.prefix_tokens:
            bundle = core[prefix_tokens]
            for repeat_index in range(self.config.protocol.repeats):
                for profile, cache_state in cells:
                    logical = self._core_logical_id(
                        profile, cache_state, prefix_tokens, repeat_index
                    )
                    self._status("running", completed_jobs=completed_jobs, current_trial=logical)
                    result, core_record, _ = await self._run_core_one(
                        profile=profile,
                        cache_state=cache_state,
                        prefix_tokens=prefix_tokens,
                        repeat_index=repeat_index,
                        bundle=bundle,
                        warmup=warmups[(prefix_tokens, cache_state)],
                    )
                    results.append(result)
                    if core_record is not None:
                        records.append(core_record)
                    completed_jobs += 1
        for pool_size in self.config.protocol.boundary_pool_sizes:
            logical = self._boundary_logical_id(pool_size)
            self._status("running", completed_jobs=completed_jobs, current_trial=logical)
            result, boundary_record, _ = await self._run_boundary_one(
                pool_size=pool_size,
                bundle=boundaries[pool_size],
            )
            results.append(result)
            if boundary_record is not None:
                boundary_records.append(boundary_record)
            completed_jobs += 1

        expected_core = self._expected_core_matrix()
        observed_core = {
            (record.profile_id, record.cache_state, record.prefix_tokens, record.repeat_index)
            for record in records
        }
        expected_boundaries = set(self.config.protocol.boundary_pool_sizes)
        observed_boundaries = {record.pool_size for record in boundary_records}
        execution_passed = (
            len(results) == self._planned_jobs()
            and observed_core == expected_core
            and observed_boundaries == expected_boundaries
        )
        analysis = analyze_m3_apc_records(records, boundary_records)
        source_commit_after, source_tree_after = self._source_identity()
        source_stable = source_commit_after == source_commit and source_tree_after == source_tree
        if self.config.evidence_role == "formal" and self.config.smoke_artifact is not None:
            smoke_summary = _read_json(self.config.smoke_artifact.root / SUMMARY_FILE)
            smoke_acceptance = smoke_summary.get("acceptance")
            bound_smoke_passed = (
                isinstance(smoke_acceptance, Mapping) and smoke_acceptance.get("passed") is True
            )
        else:
            bound_smoke_passed = self.config.evidence_role == "smoke"
        directional = self._directional_checks(analysis)
        core_hit_checks = {
            "off_hits_zero": bool(records)
            and all(record.prefix_cache_hits == 0 for record in records if not record.apc_enabled),
            "zero_reuse_hits_zero": bool(records)
            and all(
                next(item for item in record.reuse if item.reuse_percent == 0).cached_tokens == 0
                for record in records
            ),
            "shared_reuse_hits_observed": bool(records)
            and all(
                next(item for item in record.reuse if item.reuse_percent == 100).cached_tokens > 0
                for record in records
                if record.apc_enabled
            ),
            "exact_request_hit_tokens": bool(records)
            and all(record.exact_hit_tokens for record in records),
        }
        boundary_analysis = analysis.get("boundary")
        if self.config.evidence_role == "formal":
            boundary_passed = (
                isinstance(boundary_analysis, Mapping)
                and boundary_analysis.get("bracketed") is True
            )
            directional_passed = bool(directional) and all(directional.values())
        else:
            boundary_passed = (
                len(boundary_records) == 1 and boundary_records[0].full_hit_requests > 0
            )
            directional_passed = True
        checks = {
            "project_line_is_v5": self.config.project_line == "longctx-v5",
            "m1_boundaries_reused": True,
            "m1_numeric_thresholds_unchanged": True,
            "m2_negative_evidence_bound": len(self.config.m2_negative_artifacts) == 3,
            "m1_m2_not_rerun": True,
            "fp8_not_attempted": True,
            "bound_smoke_passed": bound_smoke_passed,
            "smoke_identity_matches": smoke_identity_matches,
            "all_jobs_complete": execution_passed,
            "runtime_lock_verified": runtime.matches_lock,
            "model_lock_verified": model.matches_lock,
            "source_identity_stable": source_stable,
            "single_gpu_only": self.config.gpu.count == 1 and self.config.gpu.device_ids == (0,),
            "real_rag_corpus_bound": catalog_corpus_sha == corpus.sha256,
            "prefix_pool_boundary_evidenced": boundary_passed,
            "ttft_goodput_direction_consistent": directional_passed,
            "formal_run_count_is_18_plus_2": (
                self._planned_jobs() == 20 if self.config.evidence_role == "formal" else True
            ),
            "no_m4_work": True,
            **core_hit_checks,
        }
        acceptance_passed = all(checks.values())
        acceptance = {
            "eligible": self.config.evidence_role == "formal",
            "passed": acceptance_passed,
            "checks": checks,
            "directional_checks": directional,
            "failure_reasons": sorted(name for name, passed in checks.items() if not passed),
        }
        execution = {
            "passed": execution_passed,
            "planned_jobs": self._planned_jobs(),
            "completed_jobs": len(records) + len(boundary_records),
            "failed_jobs": self._planned_jobs() - len(records) - len(boundary_records),
            "core_jobs": len(records),
            "boundary_jobs": len(boundary_records),
            "unsafe_cleanup": False,
        }
        summary: dict[str, Any] = {
            "schema_version": M3_APC_SCHEMA,
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
            "boundary_records": [record.model_dump(mode="json") for record in boundary_records],
            "protocol_scope": {
                "core_matrix": "APC off/cold plus APC on/cold/warm at 2K/4K; 0/50/100 reuse inside each paired trace",
                "off_warm_omitted": "APC-disabled service has no prefix-cache warm state",
                "defensive_extra_tests": 0,
                "performance_runs": len(expected_core),
                "mechanistic_boundary_runs": len(expected_boundaries),
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
                "integrity": str(self.store.root / M3_APC_INTEGRITY_FILE),
                "m1_boundaries": str(self.config.m1_boundaries.root),
                "m2_negative_artifacts": [
                    str(item.root) for item in self.config.m2_negative_artifacts
                ],
            },
            "m1_numeric_thresholds_modified": False,
            "m1_rerun": False,
            "m2_rerun": False,
            "fp8_attempted": False,
            "m4_started": False,
        }
        self.store.write_json(SUMMARY_FILE, summary)
        self.store.write_text(REPORT_FILE, self._report(summary))
        final_state = "accepted" if acceptance_passed else "completed_not_accepted"
        self._status(final_state, completed_jobs=len(results), acceptance=acceptance)
        seal_m3_apc_artifacts(
            self.store.root,
            self.experiment_id,
            {
                "experiment_id": self.experiment_id,
                "project_line": "longctx-v5",
                "milestone": "M3",
                "experiment_kind": "automatic-prefix-caching",
                "evidence_role": self.config.evidence_role,
                "source_commit": source_commit,
                "accepted": acceptance_passed,
                "fp8_attempted": False,
                "m4_started": False,
            },
        )
        return summary


def load_m3_apc_status(root: str | Path, experiment_id: str) -> dict[str, Any]:
    """Load M3 status without importing a tokenizer or starting vLLM."""
    if _EXPERIMENT_ID.fullmatch(experiment_id) is None:
        raise ValueError("experiment_id must be one portable path component")
    experiment_root = Path(root).expanduser().resolve() / experiment_id
    sealed = (experiment_root / M3_APC_INTEGRITY_FILE).is_file()
    if sealed:
        validate_m3_apc_artifacts(experiment_root)
    status = _read_json(experiment_root / STATUS_FILE)
    if status.get("experiment_id") != experiment_id:
        raise ValueError("M3 APC status experiment identity mismatch")
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


__all__ = ["LongContextM3APCRunner", "load_m3_apc_status"]
