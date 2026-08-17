"""Formal long-context v5 M1 capacity-sweep execution and evidence sealing.

This runner deliberately keeps initialization geometry and measured capacity
separate.  Initialization is a sealed prerequisite; only the long open-loop
trials created here are passed to the capacity-knee analysis.
"""

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
from vllm_tuner.profiling.timeseries import counter_window_delta, percentile
from vllm_tuner.runtime.controller import TrialController
from vllm_tuner.runtime.failures import UnsafeCleanupError
from vllm_tuner.workloads.generator import generate_trace
from vllm_tuner.workloads.trace import WorkloadTrace

from .capacity_evidence import DeviceMemoryEvidence, build_capacity_runtime_evidence
from .m0_runner import ProductionDefaultRuntime, _production_default_runtime
from .m1_capacity_analysis import (
    CapacityKneePolicy,
    CapacityTrialMetrics,
    CapacityTrialRecord,
    FailedCapacityTrial,
    LatencyPercentiles,
    analyze_capacity_sweep,
)
from .m1_capacity_config import (
    LongContextM1CapacityConfig,
    M1CapacityContext,
    M1CapacityLoad,
)
from .m1_capacity_integrity import (
    M1_CAPACITY_INTEGRITY_FILE,
    seal_m1_capacity_artifacts,
    validate_m1_capacity_artifacts,
)
from .m1_capacity_telemetry import analyze_arrival_window_queue, summarize_dispatch_delay
from .m1_runner import GPUMemoryReader, _clean_execution_environment, _gpu_memory_snapshot
from .model_identity import ModelIdentityFacts, require_model_identity
from .runtime_identity import RuntimeIdentityFacts, require_upstream_runtime

M1_CAPACITY_SCHEMA = "longctx-m1-capacity.v1"
MANIFEST_FILE = "manifest.json"
SUMMARY_FILE = "summary.json"
STATUS_FILE = "status.json"
REPORT_FILE = "report/m1-capacity.md"
RUNNER_LOG_FILE = "runner.log"
CAPACITY_RECORD_FILE = "capacity-record.json"
CAPACITY_EVIDENCE_FILE = "capacity-evidence.json"
POINT_FILE = "capacity-point.json"
CUDA_MEMORY_FILE = "cuda-memory.json"
MEASURED_TRACE_FILE = "measured-trace.jsonl"
MEASURED_TRACE_CHECKSUM_FILE = "measured-trace.sha256"
WARMUP_TRACE_FILE = "warmup-trace.jsonl"
WARMUP_TRACE_CHECKSUM_FILE = "warmup-trace.sha256"
RUNTIME_CAPACITY_FILE = "runtime-capacity.json"
PRODUCTION_PROFILE_FILE = "production-default-runtime.json"

_EXPERIMENT_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_ATTEMPT_SUFFIX = re.compile(r"-attempt(?P<number>[1-9][0-9]*)$")
_COUNTER_NAMES = (
    "prompt_tokens_total",
    "generation_tokens_total",
    "prefix_cache_queries",
    "prefix_cache_hits",
    "num_preemptions_total",
)
_INIT_ONLY_FLAGS = frozenset(
    {
        "--block-size",
        "--gpu-memory-utilization",
        "--kv-cache-dtype",
        "--kv-cache-memory-bytes",
        "--max-model-len",
        "--max-num-batched-tokens",
        "--max-num-seqs",
        "--num-gpu-blocks-override",
        "--scheduler-cls",
    }
)
_CAPACITY_RECORD_ADAPTER: TypeAdapter[CapacityTrialRecord] = TypeAdapter(CapacityTrialRecord)


class ControllerProtocol(Protocol):
    """The TrialController surface used by the capacity runner."""

    async def run_trial(self, params: dict[str, Any], trial_id: str, method: str) -> TrialResult:
        """Run one capacity trial."""


ControllerFactory = Callable[..., ControllerProtocol]


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON object {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        raise ValueError(f"unable to read JSONL {path}: {error}") from error
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSONL {path}:{line_number}: {error}") from error
        if not isinstance(value, dict):
            raise ValueError(f"expected JSON object at {path}:{line_number}")
        rows.append(value)
    return rows


def _trace_text(trace: WorkloadTrace) -> str:
    return "\n".join(trace.iter_jsonl()) + "\n"


def _trace_span(trace: WorkloadTrace) -> float:
    if len(trace.entries) < 2:
        return 0.0
    return trace.entries[-1].scheduled_offset_seconds - trace.entries[0].scheduled_offset_seconds


def _scaled_measured_trace(
    *,
    context: M1CapacityContext,
    load: M1CapacityLoad,
    count: int,
    seed: int,
    tokenizer: Any,
    burstiness: float,
) -> WorkloadTrace:
    """Generate Gamma arrivals, then linearly bind their span to the offered rate."""
    generated = generate_trace(
        "chat",
        count=count,
        request_rate=load.offered_requests_per_second,
        burstiness=burstiness,
        seed=seed,
        tokenizer=tokenizer,
        fixed_input_tokens=context.input_tokens,
        fixed_output_tokens=context.output_tokens,
        request_id_prefix=f"capacity-{context.context_id}-{load.load_id}",
    )
    raw_span = _trace_span(generated)
    if raw_span <= 0:
        raise ValueError("measured Gamma trace does not span positive time")
    target_span = (count - 1) / load.offered_requests_per_second
    factor = target_span / raw_span
    entries = [
        entry.model_copy(
            update={
                "scheduled_offset_seconds": (
                    target_span if index == count - 1 else entry.scheduled_offset_seconds * factor
                )
            }
        )
        for index, entry in enumerate(generated.entries)
    ]
    trace = generated.model_copy(update={"entries": entries})
    span = _trace_span(trace)
    empirical = (len(trace.entries) - 1) / span
    if not math.isclose(
        empirical,
        load.offered_requests_per_second,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise ValueError("scaled measured trace does not reproduce the target offered rate")
    return trace


def _warmup_trace(
    *,
    context: M1CapacityContext,
    count: int,
    seed: int,
    prompt_offset: int,
    tokenizer: Any,
) -> WorkloadTrace:
    return generate_trace(
        "chat",
        count=count,
        request_rate=None,
        burstiness=1.0,
        seed=seed,
        tokenizer=tokenizer,
        fixed_input_tokens=context.input_tokens,
        fixed_output_tokens=context.output_tokens,
        request_index_offset=prompt_offset,
        request_id_prefix=f"warmup-{context.context_id}",
    )


def _finite_number(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a finite number")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{field} must be a finite number")
    return converted


def _integer_counter(value: object, field: str) -> int:
    number = _finite_number(value, field)
    if number < 0 or not number.is_integer():
        raise ValueError(f"{field} must be a non-negative integer counter")
    return int(number)


def _latency_percentiles(values: Sequence[float], field: str) -> LatencyPercentiles:
    if not values:
        raise ValueError(f"{field} has no exact request-level samples")
    p50 = percentile(values, 0.50)
    p95 = percentile(values, 0.95)
    p99 = percentile(values, 0.99)
    if p50 is None or p95 is None or p99 is None:
        raise ValueError(f"{field} percentiles are unavailable")
    return LatencyPercentiles(p50_ms=p50, p95_ms=p95, p99_ms=p99)


def _counter_evidence(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    evidence: dict[str, dict[str, Any]] = {}
    for name in _COUNTER_NAMES:
        values: list[Optional[float]] = []
        for row in rows:
            metrics = row.get("metrics")
            raw = metrics.get(name) if isinstance(metrics, Mapping) else None
            if isinstance(raw, bool) or not isinstance(raw, (int, float)):
                values.append(None)
            else:
                values.append(float(raw))
        evidence[name] = counter_window_delta(values)
    return evidence


def _server_event_count(text: str, patterns: Sequence[str]) -> int:
    return int(any(re.search(pattern, text, re.IGNORECASE) is not None for pattern in patterns))


def _command_profile(command: Mapping[str, Any]) -> dict[str, Any]:
    argv = command.get("argv")
    if not isinstance(argv, list) or not argv or any(not isinstance(item, str) for item in argv):
        raise ValueError("server-command.json does not expose a real argv")
    flags = {item.split("=", 1)[0] for item in cast(list[str], argv) if item.startswith("--")}
    forbidden = sorted(flags & _INIT_ONLY_FLAGS)
    return {
        "argv": argv,
        "init_only_overrides": forbidden,
        "no_init_only_overrides": not forbidden,
    }


def _resolved_production_profile(
    trial_dir: Path,
    expected_gpu_memory_utilization_ppm: int,
    expected_max_model_len: int,
    expected_max_num_seqs: int,
    expected_max_num_batched_tokens: int,
    device_memory: DeviceMemoryEvidence,
) -> dict[str, Any]:
    parsed: ProductionDefaultRuntime = _production_default_runtime(trial_dir / "server.log")
    text = (trial_dir / "server.log").read_text(encoding="utf-8", errors="replace")
    batched = re.search(r"Chunked prefill is enabled with max_num_batched_tokens=(\d+)\.", text)
    max_num_batched_tokens = int(batched.group(1)) if batched is not None else None
    command = _command_profile(_read_json(trial_dir / "server-command.json"))
    # vLLM 0.16 resolves OpenAI-server batch defaults from physical device class.
    # The locked <70 GiB device branch is 2048 batched tokens and 256 sequences;
    # startup explicitly logs 2048 while the no-override command and locked
    # upstream source bind the companion 256 value.
    below_large_gpu_branch = device_memory.physical_total_memory_bytes < 70 * (1 << 30)
    checks = {
        "m0_runtime_profile_matches": parsed.matches_expected,
        "max_model_len_matches": parsed.max_model_len == expected_max_model_len,
        "max_num_batched_tokens_logged": max_num_batched_tokens == expected_max_num_batched_tokens,
        "max_num_seqs_locked_default": below_large_gpu_branch
        and expected_max_num_seqs == 256
        and expected_max_num_batched_tokens == 2_048,
        "gpu_memory_utilization_locked_default": expected_gpu_memory_utilization_ppm == 900_000,
        "no_init_only_command_overrides": command["no_init_only_overrides"],
    }
    return {
        **parsed.model_dump(mode="json"),
        "resolved_max_num_batched_tokens": max_num_batched_tokens,
        "resolved_max_num_seqs": (
            expected_max_num_seqs if checks["max_num_seqs_locked_default"] else None
        ),
        "max_num_seqs_source": "vllm-0.16-openai-device-default+no-command-override",
        "gpu_memory_utilization_ppm": expected_gpu_memory_utilization_ppm,
        "command": command,
        "checks": checks,
        "matches_capacity_profile": all(checks.values()),
    }


def _analysis_policy(config: LongContextM1CapacityConfig) -> CapacityKneePolicy:
    source = config.knee_policy
    return CapacityKneePolicy(
        minimum_trace_duration_seconds=float(config.protocol.measurement_seconds),
        minimum_load_points=3 if config.formal_acceptance_eligible else 2,
        maximum_marginal_achieved_gain_ratio=source.throughput_plateau_max_gain_ppm / 1_000_000,
        minimum_queue_growth_slope_waiting_requests_per_second=(
            source.queue_growth_min_requests_per_second
        ),
        minimum_peak_waiting_requests=source.minimum_peak_waiting_requests,
        minimum_completion_fraction=source.minimum_completion_ppm / 1_000_000,
        minimum_achieved_fraction_of_empirical_offered=(
            source.minimum_achieved_to_offered_ppm / 1_000_000
        ),
        minimum_goodput_fraction_of_empirical_offered=(
            source.minimum_slo_attainment_ppm / 1_000_000
        ),
        minimum_slo_satisfied_fraction=source.minimum_slo_attainment_ppm / 1_000_000,
        maximum_preemptions_for_stable=source.maximum_preemptions_for_stable,
        maximum_timeouts_for_stable=source.maximum_timeouts_for_stable,
        maximum_p99_dispatch_delay_ms=source.max_p99_dispatch_delay_ms,
        require_zero_oom_events=source.require_zero_oom_events,
        minimum_joint_signal_repeats=cast(Any, max(2, source.minimum_joint_signal_repeats)),
    )


def _failure_kind(result: TrialResult) -> str:
    reason = result.failure_reason if isinstance(result.failure_reason, Mapping) else {}
    text = " ".join(str(reason.get(name, "")) for name in ("type", "message", "phase"))
    lowered = text.casefold()
    if "cleanup" in lowered:
        return "cleanup_failure"
    if "out of memory" in lowered or re.search(r"\boom\b", lowered):
        return "oom"
    if "timeout" in lowered:
        return "timeout"
    if "startup" in lowered or "ready" in lowered:
        return "startup_failure"
    if "telemetry" in lowered or "prometheus" in lowered:
        return "telemetry_failure"
    if "server" in lowered:
        return "server_failure"
    return "runner_failure"


def _failed_record(
    *,
    result: TrialResult,
    context: M1CapacityContext,
    load: M1CapacityLoad,
    repeat_index: int,
    trace: WorkloadTrace,
    reason: str,
) -> FailedCapacityTrial:
    kind = _failure_kind(result)
    return FailedCapacityTrial(
        evidence_kind="formal_capacity_sweep",
        trial_id=result.trial_id,
        context_id=context.context_id,
        context_tokens=context.total_kv_tokens,
        load_id=load.load_id,
        repeat_index=cast(Any, repeat_index),
        trace_id=trace.checksum(),
        planned_trace_duration_seconds=_trace_span(trace),
        target_offered_requests_per_second=load.offered_requests_per_second,
        status="failed",
        failure_kind=cast(Any, kind),
        failure_reason=reason.strip(),
        observed_trace_duration_seconds=result.measurement_seconds,
        oom_observed=kind == "oom",
        timeout_observed=kind == "timeout",
    )


def _derive_complete_record(
    *,
    result: TrialResult,
    trial_dir: Path,
    context: M1CapacityContext,
    load: M1CapacityLoad,
    repeat_index: int,
    trace: WorkloadTrace,
    device_memory: DeviceMemoryEvidence,
    config: LongContextM1CapacityConfig,
) -> tuple[CapacityTrialMetrics, dict[str, Any]]:
    request_rows = _read_jsonl(trial_dir / "request-results.jsonl")
    prometheus_rows = _read_jsonl(trial_dir / "prometheus.jsonl")
    benchmark = _read_json(trial_dir / "benchmark-raw.json")
    expected_ids = [entry.request_id for entry in trace.entries]
    observed_ids = [row.get("request_id") for row in request_rows]
    if observed_ids != expected_ids:
        raise ValueError("request result IDs/order do not exactly match the sealed measured trace")
    if len(request_rows) != len(trace.entries):
        raise ValueError("request result count does not match the sealed measured trace")

    requests = [RequestResult.from_dict(row) for row in request_rows]
    successful = [request for request in requests if request.status == RequestStatus.SUCCESS]
    exact_tokens = all(
        request.input_tokens == context.input_tokens
        and request.output_tokens == context.output_tokens
        and request.token_count_source == "usage"
        and request.token_timestamps_valid
        and len(request.token_timestamps) == context.output_tokens
        for request in successful
    )
    if not exact_tokens or len(successful) != len(requests):
        raise ValueError("complete capacity evidence requires exact tokens for every request")

    dispatch = summarize_dispatch_delay(request_rows)
    if (
        not dispatch.available
        or dispatch.sample_count != len(request_rows)
        or dispatch.p99_ms is None
    ):
        raise ValueError("dispatch-delay evidence is unavailable or incomplete")
    started_at = benchmark.get("started_at")
    if isinstance(started_at, bool) or not isinstance(started_at, int):
        raise ValueError("benchmark raw evidence has no monotonic measurement start")
    planned_span = _trace_span(trace)
    scheduled = [request.scheduled_at for request in requests]
    if any(value is None for value in scheduled):
        raise ValueError("request evidence is missing scheduled arrival timestamps")
    scheduled_values = cast(list[int], scheduled)
    observed_span = (max(scheduled_values) - min(scheduled_values)) / 1_000_000_000
    if observed_span <= 0:
        raise ValueError("observed request arrivals do not span positive time")
    empirical = (len(requests) - 1) / observed_span
    if not math.isclose(empirical, load.offered_requests_per_second, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError("observed arrival rate differs from the sealed target trace")

    queue = analyze_arrival_window_queue(
        prometheus_rows,
        measurement_start_monotonic_ns=started_at,
        arrival_window_seconds=planned_span,
        sample_interval_seconds=0.2,
    )
    if not queue.available or queue.tail_slope_requests_per_second is None:
        raise ValueError(f"arrival-window queue evidence unavailable: {queue.reason}")
    if queue.peak_waiting_requests is None:
        raise ValueError("arrival-window peak waiting evidence is unavailable")

    counters = _counter_evidence(prometheus_rows)
    expected_prompt_tokens = sum(request.input_tokens for request in requests)
    expected_generation_tokens = sum(request.output_tokens for request in requests)
    expected_counters = {
        "prompt_tokens_total": expected_prompt_tokens,
        "generation_tokens_total": expected_generation_tokens,
        "prefix_cache_queries": expected_prompt_tokens,
        "prefix_cache_hits": config.knee_policy.required_prefix_cache_hits_delta,
    }
    for name, expected in expected_counters.items():
        evidence = counters[name]
        if evidence.get("available") is not True or evidence.get("reset_count") != 0:
            raise ValueError(f"counter {name} is unavailable or reset inside measurement")
        if _integer_counter(evidence.get("delta"), f"counter {name} delta") != expected:
            raise ValueError(f"counter {name} delta does not match exact request semantics")
    preemption_evidence = counters["num_preemptions_total"]
    if (
        preemption_evidence.get("available") is not True
        or preemption_evidence.get("reset_count") != 0
    ):
        raise ValueError("preemption counter is unavailable or reset inside measurement")
    preemptions = _integer_counter(preemption_evidence.get("delta"), "num_preemptions_total delta")

    metrics_text = next(
        (
            str(row["raw_text"])
            for row in prometheus_rows
            if row.get("available") is True and isinstance(row.get("raw_text"), str)
        ),
        None,
    )
    if metrics_text is None:
        raise ValueError("raw Prometheus exposition is unavailable for KV-block evidence")
    server_log = (trial_dir / "server.log").read_text(encoding="utf-8", errors="replace")
    runtime = build_capacity_runtime_evidence(
        run_id=result.trial_id,
        runtime_profile_sha256=sha256_json(
            {"context": context.context_id, "load": load.load_id, "vllm_args": {}}
        ),
        server_log_text=server_log,
        metrics_text=metrics_text,
        device_memory=device_memory,
    )
    production = _resolved_production_profile(
        trial_dir,
        config.server_profile.expected_gpu_memory_utilization_ppm,
        config.server_profile.expected_max_model_len,
        config.server_profile.expected_max_num_seqs,
        config.server_profile.expected_max_num_batched_tokens,
        device_memory,
    )
    runtime_checks = {
        "runtime_capacity_consistent": runtime.logged_capacity_consistent,
        "production_profile_matches": production["matches_capacity_profile"] is True,
        "cache_utilization_matches": runtime.cache_config.gpu_memory_utilization_ppm
        == config.server_profile.expected_gpu_memory_utilization_ppm,
        "prefix_caching_enabled": runtime.cache_config.enable_prefix_caching is True,
        "cache_dtype_is_auto": runtime.cache_config.requested_cache_dtype == "auto",
        "block_size_matches": runtime.cache_config.resolved_block_size
        == config.server_profile.expected_block_size,
        "no_block_override": runtime.cache_config.num_gpu_blocks_override is None,
        "no_explicit_kv_memory": runtime.cache_config.kv_cache_memory_bytes is None,
    }
    if not all(runtime_checks.values()):
        failed = sorted(name for name, passed in runtime_checks.items() if not passed)
        raise ValueError("runtime capacity/default gate failed: " + ", ".join(failed))

    ttft = [cast(float, row.get("ttft_ms")) for row in request_rows]
    tpot = [cast(float, row.get("tpot_ms")) for row in request_rows]
    e2e = [cast(float, row.get("e2e_ms")) for row in request_rows]
    if any(not isinstance(value, (int, float)) for value in [*ttft, *tpot, *e2e]):
        raise ValueError("request latency evidence is incomplete")
    itl = [
        _finite_number(value, "request ITL")
        for row in request_rows
        for value in cast(list[Any], row.get("itl_ms", []))
    ]
    aggregate = benchmark.get("aggregate")
    if not isinstance(aggregate, Mapping):
        raise ValueError("benchmark aggregate is unavailable")
    duration = _finite_number(aggregate.get("duration"), "benchmark duration")
    completed = len(successful)
    good = _integer_counter(result.client.get("good_requests"), "good request count")
    achieved = _finite_number(
        result.client.get("achieved_requests_per_sec"), "achieved request rate"
    )
    goodput = _finite_number(result.client.get("goodput_requests_per_sec"), "goodput request rate")
    timeout_count = sum(request.status == RequestStatus.TIMEOUT for request in requests)
    oom_count = _server_event_count(server_log, (r"CUDA out of memory", r"\bOOM\b"))
    record = CapacityTrialMetrics(
        evidence_kind="formal_capacity_sweep",
        trial_id=result.trial_id,
        context_id=context.context_id,
        context_tokens=context.total_kv_tokens,
        load_id=load.load_id,
        repeat_index=cast(Any, repeat_index),
        trace_id=trace.checksum(),
        planned_trace_duration_seconds=planned_span,
        target_offered_requests_per_second=load.offered_requests_per_second,
        status="complete",
        observed_trace_duration_seconds=observed_span,
        empirical_offered_requests_per_second=empirical,
        achieved_requests_per_second=achieved,
        p99_dispatch_delay_ms=dispatch.p99_ms,
        completion_fraction=completed / len(requests),
        goodput_requests_per_second=goodput,
        slo_satisfied_fraction=good / len(requests),
        queue_growth_slope_waiting_requests_per_second=(queue.tail_slope_requests_per_second),
        peak_waiting_requests=math.ceil(queue.peak_waiting_requests),
        preemption_count=preemptions,
        oom_count=oom_count,
        timeout_count=timeout_count,
        ttft=_latency_percentiles([float(value) for value in ttft], "TTFT"),
        tpot=_latency_percentiles([float(value) for value in tpot], "TPOT"),
        itl=_latency_percentiles(itl, "ITL"),
        end_to_end=_latency_percentiles([float(value) for value in e2e], "E2E"),
    )
    evidence = {
        "schema_version": M1_CAPACITY_SCHEMA,
        "semantic_gate_passed": True,
        "exact_request_count": len(requests),
        "exact_input_tokens_per_request": context.input_tokens,
        "exact_output_tokens_per_request": context.output_tokens,
        "measurement_duration_seconds": duration,
        "dispatch": dispatch.model_dump(mode="json"),
        "arrival_window_queue": queue.model_dump(mode="json"),
        "counters": counters,
        "runtime_checks": runtime_checks,
        "runtime_capacity": runtime.model_dump(mode="json"),
        "production_profile": production,
        "kv_blocks": {
            "num_gpu_blocks": runtime.cache_config.num_gpu_blocks,
            "usable_num_gpu_blocks": runtime.cache_config.usable_num_gpu_blocks,
            "block_size": runtime.cache_config.resolved_block_size,
            "cached_tokens": runtime.observation.cached_tokens,
        },
    }
    return record, evidence


class LongContextM1CapacityRunner:
    """Run, resume, analyze, and independently seal one v5 M1 E1 matrix."""

    def __init__(
        self,
        config: LongContextM1CapacityConfig,
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
            raise ValueError("experiment_id must be one portable path component")
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
        """Compatibility name used by CLI/report callers."""
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
            raise RuntimeError(f"M1 capacity experiment is already running: {path}") from error
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
            raise ValueError("M1 capacity requires matching upstream runtime and model locks")
        return runtime, model, _clean_execution_environment(self.execution_environment)

    def _source_identity(self) -> tuple[str, str]:
        commit, dirty, _ = git_state(self.repository)
        tree = source_tree_sha256(self.repository)
        if commit is None or tree is None or dirty:
            raise ValueError("M1 capacity requires one clean committed source identity")
        return commit, tree

    def _manifest(
        self,
        runtime: RuntimeIdentityFacts,
        model: ModelIdentityFacts,
        environment: Mapping[str, str],
        source_commit: str,
        source_tree: str,
    ) -> dict[str, Any]:
        return {
            "schema_version": M1_CAPACITY_SCHEMA,
            "project_line": "longctx-v5",
            "milestone": "M1",
            "experiment_kind": "capacity-sweep",
            "experiment_id": self.experiment_id,
            "created_at": utc_now_iso(),
            "config_sha256": sha256_json(self.config.model_dump(mode="json")),
            "source_commit": source_commit,
            "source_tree_sha256": source_tree,
            "runtime": runtime.model_dump(mode="json"),
            "model": model.model_dump(mode="json"),
            "execution_environment": dict(environment),
            "initialization_artifact": {
                "experiment_id": self.config.initialization_artifact.experiment_id,
                "root": str(self.config.initialization_artifact.root),
                "integrity_sha256": sha256_file(
                    self.config.initialization_artifact.root / "m1-integrity.json"
                ),
            },
            "production_vllm_args": {},
            "initialization_concurrency_used_for_capacity_knee": False,
        }

    @staticmethod
    def _manifest_identity(value: Mapping[str, Any]) -> dict[str, Any]:
        return {
            name: value.get(name)
            for name in (
                "schema_version",
                "project_line",
                "milestone",
                "experiment_kind",
                "experiment_id",
                "config_sha256",
                "source_commit",
                "source_tree_sha256",
                "runtime",
                "model",
                "execution_environment",
                "initialization_artifact",
                "production_vllm_args",
                "initialization_concurrency_used_for_capacity_knee",
            )
        }

    def _initialize_root(self, requested_manifest: dict[str, Any]) -> Optional[dict[str, Any]]:
        root = self.store.root
        if root.exists():
            if not self.resume:
                raise FileExistsError(f"M1 capacity artifact root exists: {root}; use --resume")
            if (root / M1_CAPACITY_INTEGRITY_FILE).is_file():
                validate_m1_capacity_artifacts(root)
                summary = _read_json(root / SUMMARY_FILE)
                self._validate_resume_manifest(_read_json(root / MANIFEST_FILE), requested_manifest)
                return summary
            self.store.initialize(exist_ok=True)
            self._validate_resume_manifest(_read_json(root / MANIFEST_FILE), requested_manifest)
            return None
        self.store.initialize()
        self.store.write_json(MANIFEST_FILE, requested_manifest)
        self.store.write_json("experiment.json", self.config.model_dump(mode="json"))
        return None

    def _validate_resume_manifest(
        self, existing: Mapping[str, Any], requested: Mapping[str, Any]
    ) -> None:
        if self._manifest_identity(existing) != self._manifest_identity(requested):
            raise ValueError("M1 capacity resume manifest identity mismatch")

    def _trace_catalog(
        self,
    ) -> tuple[dict[str, WorkloadTrace], dict[tuple[str, str], WorkloadTrace]]:
        tokenizer = self._load_tokenizer()
        warmups: dict[str, WorkloadTrace] = {}
        measured: dict[tuple[str, str], WorkloadTrace] = {}
        catalog: dict[str, Any] = {
            "schema_version": M1_CAPACITY_SCHEMA,
            "warmup": {},
            "measured": {},
        }
        for context in self.config.contexts:
            warmup = _warmup_trace(
                context=context,
                count=self.config.protocol.warmup_requests,
                seed=self.config.protocol.warmup_seed,
                prompt_offset=self.config.protocol.warmup_prompt_index_offset,
                tokenizer=tokenizer,
            )
            warmups[context.context_id] = warmup
            warmup_path = Path("traces") / context.context_id / "warmup.jsonl"
            self._write_or_validate_trace(warmup_path, warmup)
            cast(dict[str, Any], catalog["warmup"])[context.context_id] = self._trace_row(
                warmup_path, warmup
            )
            for load in context.loads:
                trace = _scaled_measured_trace(
                    context=context,
                    load=load,
                    count=self.config.protocol.measured_request_count(load),
                    seed=self.config.protocol.measurement_seed,
                    tokenizer=tokenizer,
                    burstiness=self.config.protocol.burstiness,
                )
                if _trace_span(trace) < self.config.protocol.measurement_seconds:
                    raise ValueError("measured trace span is below configured capacity duration")
                measured[(context.context_id, load.load_id)] = trace
                path = Path("traces") / context.context_id / f"{load.load_id}.jsonl"
                self._write_or_validate_trace(path, trace)
                key = f"{context.context_id}/{load.load_id}"
                cast(dict[str, Any], catalog["measured"])[key] = self._trace_row(path, trace)
        catalog_path = self.store.root / "traces" / "catalog.json"
        if catalog_path.exists():
            if _read_json(catalog_path) != catalog:
                raise ValueError("M1 capacity resume trace catalog mismatch")
        else:
            self.store.write_json("traces/catalog.json", catalog)
        return warmups, measured

    @staticmethod
    def _trace_row(path: Path, trace: WorkloadTrace) -> dict[str, Any]:
        span = _trace_span(trace)
        empirical = (len(trace.entries) - 1) / span if span > 0 else None
        return {
            "path": path.as_posix(),
            "sha256": trace.checksum(),
            "request_count": len(trace.entries),
            "span_seconds": span,
            "target_requests_per_second": trace.request_rate,
            "empirical_requests_per_second": empirical,
        }

    def _write_or_validate_trace(self, relative: Path, trace: WorkloadTrace) -> None:
        path = self.store.root / relative
        checksum_relative = relative.with_suffix(".sha256")
        expected_text = _trace_text(trace)
        expected_sidecar = f"{trace.checksum()}  {relative.name}\n"
        if path.exists():
            if path.read_text(encoding="utf-8") != expected_text:
                raise ValueError(f"M1 capacity resume trace bytes mismatch: {relative}")
            sidecar = self.store.root / checksum_relative
            if not sidecar.is_file() or sidecar.read_text(encoding="utf-8") != expected_sidecar:
                raise ValueError(f"M1 capacity resume trace checksum mismatch: {relative}")
            return
        self.store.write_text(relative, expected_text)
        self.store.write_text(checksum_relative, expected_sidecar)

    def _logical_trial_id(
        self, context: M1CapacityContext, load: M1CapacityLoad, repeat_index: int
    ) -> str:
        return f"capacity-rate-{context.context_id}-{load.load_id}-repeat-{repeat_index}"

    def _attempts(self, logical_id: str) -> list[tuple[int, Path]]:
        if not self.store.trials_dir.is_dir():
            return []
        attempts: list[tuple[int, Path]] = []
        for path in self.store.trials_dir.iterdir():
            if path.is_symlink() or not path.is_dir():
                raise ValueError(f"unexpected M1 capacity trials entry: {path}")
            if path.name == logical_id:
                attempts.append((0, path))
                continue
            if not path.name.startswith(logical_id + "-attempt"):
                continue
            match = _ATTEMPT_SUFFIX.search(path.name)
            if match is not None:
                attempts.append((int(match.group("number")), path))
        return sorted(attempts)

    def _cached_complete(
        self,
        logical_id: str,
        context: M1CapacityContext,
        load: M1CapacityLoad,
        repeat_index: int,
        trace: WorkloadTrace,
    ) -> Optional[tuple[TrialResult, CapacityTrialMetrics]]:
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
                        CAPACITY_RECORD_FILE,
                        CAPACITY_EVIDENCE_FILE,
                        POINT_FILE,
                        CUDA_MEMORY_FILE,
                        MEASURED_TRACE_FILE,
                        MEASURED_TRACE_CHECKSUM_FILE,
                        WARMUP_TRACE_FILE,
                        WARMUP_TRACE_CHECKSUM_FILE,
                        RUNTIME_CAPACITY_FILE,
                        PRODUCTION_PROFILE_FILE,
                    },
                )
                record = _CAPACITY_RECORD_ADAPTER.validate_json(
                    (path / CAPACITY_RECORD_FILE).read_text(encoding="utf-8")
                )
                if not isinstance(record, CapacityTrialMetrics):
                    raise ValueError("sealed attempt is not a complete capacity measurement")
                point = _read_json(path / POINT_FILE)
                expected_point = self._point_payload(
                    context, load, repeat_index, trace, result.trial_id
                )
                if point != expected_point:
                    raise ValueError("sealed attempt capacity point identity mismatch")
                evidence = _read_json(path / CAPACITY_EVIDENCE_FILE)
                if evidence.get("semantic_gate_passed") is not True:
                    raise ValueError("sealed attempt did not pass capacity semantic gates")
                if (
                    record.context_id != context.context_id
                    or record.load_id != load.load_id
                    or record.repeat_index != repeat_index
                    or record.trace_id != trace.checksum()
                ):
                    raise ValueError("sealed capacity record identity mismatch")
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
        next_number = max(number for number, _ in attempts) + 1
        attempt_id = f"{logical_id}-attempt{next_number}"
        self._new_attempts.append(attempt_id)
        return attempt_id

    def _point_payload(
        self,
        context: M1CapacityContext,
        load: M1CapacityLoad,
        repeat_index: int,
        trace: WorkloadTrace,
        trial_id: str,
    ) -> dict[str, Any]:
        return {
            "schema_version": M1_CAPACITY_SCHEMA,
            "evidence_kind": "formal_capacity_sweep",
            "trial_id": trial_id,
            "context_id": context.context_id,
            "context_tokens": context.total_kv_tokens,
            "input_tokens": context.input_tokens,
            "output_tokens": context.output_tokens,
            "load_id": load.load_id,
            "repeat_index": repeat_index,
            "target_offered_requests_per_second": load.offered_requests_per_second,
            "trace_id": trace.checksum(),
            "planned_trace_duration_seconds": _trace_span(trace),
            "request_count": len(trace.entries),
            "vllm_args": {},
            "initialization_concurrency_used_for_capacity_knee": False,
        }

    def _write_pretrial_evidence(
        self,
        *,
        trial_id: str,
        context: M1CapacityContext,
        load: M1CapacityLoad,
        repeat_index: int,
        trace: WorkloadTrace,
        warmup: WorkloadTrace,
        memory: DeviceMemoryEvidence,
    ) -> None:
        base = Path("trials") / trial_id
        self.store.write_json(
            base / POINT_FILE, self._point_payload(context, load, repeat_index, trace, trial_id)
        )
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
            base / CAPACITY_EVIDENCE_FILE,
            {
                "schema_version": M1_CAPACITY_SCHEMA,
                "semantic_gate_passed": False,
                "state": "pending-controller-result",
            },
        )

    def _finalize_trial(
        self,
        result: TrialResult,
        record: CapacityTrialRecord,
        evidence: Mapping[str, Any],
    ) -> None:
        base = Path("trials") / result.trial_id
        self.store.write_json(base / CAPACITY_RECORD_FILE, record)
        self.store.write_json(base / CAPACITY_EVIDENCE_FILE, dict(evidence))
        runtime = evidence.get("runtime_capacity")
        production = evidence.get("production_profile")
        if isinstance(runtime, Mapping):
            self.store.write_json(base / RUNTIME_CAPACITY_FILE, dict(runtime))
        else:
            self.store.write_json(
                base / RUNTIME_CAPACITY_FILE,
                {"available": False, "reason": "trial did not pass runtime-capacity gates"},
            )
        if isinstance(production, Mapping):
            self.store.write_json(base / PRODUCTION_PROFILE_FILE, dict(production))
        else:
            self.store.write_json(
                base / PRODUCTION_PROFILE_FILE,
                {"available": False, "reason": "trial did not pass production-profile gates"},
            )
        self.store.ensure_trial_artifacts(result)
        self.store.validate_cached_trial(result, require_telemetry=True)
        self.store.validate_trial_integrity(result.trial_id)

    async def _run_one(
        self,
        context: M1CapacityContext,
        load: M1CapacityLoad,
        repeat_index: int,
        trace: WorkloadTrace,
        warmup: WorkloadTrace,
    ) -> tuple[TrialResult, CapacityTrialRecord, bool]:
        logical_id = self._logical_trial_id(context, load, repeat_index)
        cached = self._cached_complete(logical_id, context, load, repeat_index, trace)
        if cached is not None:
            return cached[0], cached[1], True
        trial_id = self._select_new_attempt(logical_id)
        tuning = self.config.to_tuning_config(context, load, repeat_index)
        if tuning.vllm_args != {}:
            raise ValueError("M1 E1 production capacity trials must pass vllm_args={}")
        memory = self.gpu_memory_reader(self.config.gpu.device_ids[0])
        self._write_pretrial_evidence(
            trial_id=trial_id,
            context=context,
            load=load,
            repeat_index=repeat_index,
            trace=trace,
            warmup=warmup,
            memory=memory,
        )
        controller = self.controller_factory(
            tuning,
            trace,
            self.store,
            tokenizer=self._load_tokenizer(),
            warmup_trace=warmup,
            strict_open_loop=True,
        )
        try:
            result = await controller.run_trial({}, trial_id, "capacity")
        except UnsafeCleanupError:
            self._status(
                "unsafe_cleanup",
                current_trial=trial_id,
                unsafe_cleanup=True,
                message="unsafe cleanup; capacity sweep stopped before another GPU process",
            )
            raise
        expected_provenance = trial_provenance(trial_id, "capacity")
        if any(getattr(result, name) != value for name, value in expected_provenance.items()):
            raise ValueError(f"capacity trial provenance mismatch: {trial_id}")
        if result.params != {}:
            raise ValueError(f"capacity trial passed non-production parameters: {trial_id}")

        record: CapacityTrialRecord
        evidence: dict[str, Any]
        if result.status in {TrialStatus.COMPLETE, TrialStatus.INFEASIBLE}:
            try:
                record, evidence = _derive_complete_record(
                    result=result,
                    trial_dir=self.store.trials_dir / trial_id,
                    context=context,
                    load=load,
                    repeat_index=repeat_index,
                    trace=trace,
                    device_memory=memory,
                    config=self.config,
                )
            except (OSError, ValueError) as error:
                result.status = TrialStatus.FAILED
                result.constraints = {
                    **result.constraints,
                    "feasible": False,
                    "violations": [
                        *cast(list[str], result.constraints.get("violations", [])),
                        "m1_capacity_semantic_gate",
                    ],
                }
                result.failure_reason = {
                    "type": "M1_CAPACITY_SEMANTIC_GATE",
                    "message": str(error),
                    "phase": "M1_CAPACITY_FINALIZE",
                }
                self.store.record_artifact_finalizer_failure(result, str(error))
                record = _failed_record(
                    result=result,
                    context=context,
                    load=load,
                    repeat_index=repeat_index,
                    trace=trace,
                    reason=str(error),
                )
                evidence = {
                    "schema_version": M1_CAPACITY_SCHEMA,
                    "semantic_gate_passed": False,
                    "failure": str(error),
                }
        else:
            reason = (
                json.dumps(result.failure_reason, sort_keys=True)
                if result.failure_reason
                else "trial failed"
            )
            record = _failed_record(
                result=result,
                context=context,
                load=load,
                repeat_index=repeat_index,
                trace=trace,
                reason=reason,
            )
            evidence = {
                "schema_version": M1_CAPACITY_SCHEMA,
                "semantic_gate_passed": False,
                "failure": reason,
            }
        self._finalize_trial(result, record, evidence)
        return result, record, False

    def _expected_matrix(self) -> set[tuple[str, str, int]]:
        return {
            (context.context_id, load.load_id, repeat)
            for context in self.config.contexts
            for load in context.loads
            for repeat in range(self.config.protocol.repeats)
        }

    def _validate_record_matrix(self, records: Sequence[CapacityTrialRecord]) -> None:
        observed: set[tuple[str, str, int]] = {
            (record.context_id, record.load_id, int(record.repeat_index)) for record in records
        }
        expected = self._expected_matrix()
        if observed != expected or len(observed) != len(records):
            missing = sorted(expected - observed)
            extra = sorted(observed - expected)
            raise ValueError(
                "capacity record/config matrix mismatch: "
                f"missing={missing!r}, extra={extra!r}, records={len(records)}"
            )

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
        eta_seconds = (
            elapsed / completed_jobs * remaining
            if completed_jobs > 0
            else sum(
                _trace_span_for_load(self.config, context, load)
                for context in self.config.contexts
                for load in context.loads
                for _ in range(self.config.protocol.repeats)
            )
        )
        eta = (datetime.now(timezone.utc) + timedelta(seconds=eta_seconds)).isoformat()
        value = {
            "schema_version": M1_CAPACITY_SCHEMA,
            "experiment_id": self.experiment_id,
            "state": state,
            "pid": os.getpid(),
            "gpu": list(self.config.gpu.device_ids),
            "log": str(self.store.root / RUNNER_LOG_FILE),
            "result": str(self.store.root),
            "eta": eta,
            "eta_seconds": eta_seconds,
            "resume": self._resume_command(),
            "sealed": (self.store.root / M1_CAPACITY_INTEGRITY_FILE).is_file(),
            "acceptance": dict(acceptance) if acceptance is not None else None,
            "planned_jobs": planned,
            "completed_jobs": completed_jobs,
            "current_trial": current_trial,
            "unsafe_cleanup": unsafe_cleanup,
            "message": message,
            "updated_at": utc_now_iso(),
        }
        self.store.write_json(STATUS_FILE, value)
        previous = ""
        log_path = self.store.root / RUNNER_LOG_FILE
        if log_path.is_file():
            previous = log_path.read_text(encoding="utf-8")
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
            "scripts/run_longctx_m1_capacity.sh --config CONFIG "
            f"--experiment-id {self.experiment_id} --resume"
        )

    @staticmethod
    def _report(summary: Mapping[str, Any]) -> str:
        acceptance = cast(Mapping[str, Any], summary["acceptance"])
        execution = cast(Mapping[str, Any], summary["execution"])
        analysis = cast(Mapping[str, Any], summary["analysis"])
        lines = [
            "# Long-context v5 M1 capacity sweep",
            "",
            f"- Experiment: {summary['experiment_id']}",
            f"- Evidence role: {summary['evidence_role']}",
            f"- Execution passed: {execution['passed']}",
            f"- Formal acceptance eligible: {acceptance['eligible']}",
            f"- M1 accepted: {acceptance['passed']}",
            "- Initialization concurrency used for knee: false",
            "",
            "## Capacity knees",
            "",
            "| Context | Accepted | Last stable load | First overload load |",
            "|---|---:|---|---|",
        ]
        contexts = analysis.get("contexts", [])
        if isinstance(contexts, list):
            for context in contexts:
                if not isinstance(context, Mapping):
                    continue
                knee = context.get("knee")
                knee_value = knee if isinstance(knee, Mapping) else {}
                lines.append(
                    "| {context} | {passed} | {stable} | {overload} |".format(
                        context=context.get("context_id"),
                        passed=knee_value.get("passed"),
                        stable=knee_value.get("last_stable_load_id"),
                        overload=knee_value.get("first_bracketed_overload_load_id"),
                    )
                )
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
        requested_manifest = self._manifest(runtime, model, environment, source_commit, source_tree)
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
        warmups, measured = self._trace_catalog()
        records: list[CapacityTrialRecord] = []
        results: list[TrialResult] = []
        completed_jobs = 0
        for context in self.config.contexts:
            for load in context.loads:
                trace = measured[(context.context_id, load.load_id)]
                warmup = warmups[context.context_id]
                for repeat_index in range(self.config.protocol.repeats):
                    logical = self._logical_trial_id(context, load, repeat_index)
                    self._status(
                        "running",
                        completed_jobs=completed_jobs,
                        current_trial=logical,
                    )
                    result, record, _ = await self._run_one(
                        context, load, repeat_index, trace, warmup
                    )
                    results.append(result)
                    records.append(record)
                    completed_jobs += 1

        self._validate_record_matrix(records)
        policy = _analysis_policy(self.config)
        analysis = analyze_capacity_sweep(records, policy)
        complete_records = [
            record for record in records if isinstance(record, CapacityTrialMetrics)
        ]
        execution_passed = len(complete_records) == len(records) == len(self._expected_matrix())
        source_commit_after, source_tree_after = self._source_identity()
        source_stable = source_commit_after == source_commit and source_tree_after == source_tree
        init_summary = _read_json(self.config.initialization_artifact.root / "summary.json")
        checks = {
            "project_line_is_v5": self.config.project_line == "longctx-v5",
            "initialization_artifact_accepted": init_summary.get("initialization_validation_passed")
            is True,
            "initialization_primary_error_passed": init_summary.get("primary_error_passed") is True,
            "initialization_extrapolation_error_passed": init_summary.get(
                "extrapolation_error_passed"
            )
            is True,
            "formal_capacity_matrix": self.config.formal_acceptance_eligible,
            "all_capacity_jobs_complete": execution_passed,
            "capacity_knees_accepted": analysis.passed,
            "production_vllm_args_empty": all(result.params == {} for result in results),
            "runtime_lock_verified": runtime.matches_lock,
            "model_lock_verified": model.matches_lock,
            "source_identity_stable": source_stable,
            "no_legacy_results_used": True,
            "capacity_analysis_excludes_initialization": analysis.initialization_evidence_used
            is False,
            "initialization_concurrency_not_used_for_knee": True,
        }
        eligible = self.config.formal_acceptance_eligible
        failure_reasons = [name for name, passed in checks.items() if not passed]
        if not eligible:
            failure_reasons.append("non_formal_evidence_role")
        acceptance = {
            "eligible": eligible,
            "passed": eligible and all(checks.values()),
            "checks": checks,
            "failure_reasons": sorted(set(failure_reasons)),
        }
        execution = {
            "passed": execution_passed,
            "planned_jobs": len(self._expected_matrix()),
            "completed_jobs": len(complete_records),
            "failed_jobs": len(records) - len(complete_records),
            "unsafe_cleanup": False,
        }
        summary: dict[str, Any] = {
            "schema_version": M1_CAPACITY_SCHEMA,
            "project_line": self.config.project_line,
            "milestone": self.config.milestone,
            "experiment_kind": self.config.experiment_kind,
            "evidence_role": self.config.evidence_role,
            "experiment_id": self.experiment_id,
            "finished_at": utc_now_iso(),
            "execution": execution,
            "acceptance": acceptance,
            "analysis": analysis.model_dump(mode="json"),
            "trials": [record.model_dump(mode="json") for record in records],
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
                "integrity": str(self.store.root / M1_CAPACITY_INTEGRITY_FILE),
                "initialization_root": str(self.config.initialization_artifact.root),
            },
            "initialization_concurrency_used_for_capacity_knee": False,
        }
        self.store.write_json(SUMMARY_FILE, summary)
        self.store.write_text(REPORT_FILE, self._report(summary))
        final_state = "accepted" if acceptance["passed"] else "completed_not_accepted"
        self._status(
            final_state,
            completed_jobs=len(records),
            acceptance=acceptance,
        )
        if execution_passed:
            seal_m1_capacity_artifacts(
                self.store.root,
                self.experiment_id,
                {
                    "experiment_id": self.experiment_id,
                    "project_line": "longctx-v5",
                    "milestone": "M1",
                    "experiment_kind": "capacity-sweep",
                    "evidence_role": self.config.evidence_role,
                    "source_commit": source_commit,
                    "initialization_experiment_id": (
                        self.config.initialization_artifact.experiment_id
                    ),
                    "capacity_accepted": acceptance["passed"],
                },
            )
            validate_m1_capacity_artifacts(self.store.root)
        return summary


def _trace_span_for_load(
    config: LongContextM1CapacityConfig,
    context: M1CapacityContext,
    load: M1CapacityLoad,
) -> float:
    del context
    count = config.protocol.measured_request_count(load)
    return (count - 1) / load.offered_requests_per_second


def load_m1_capacity_status(root: str | Path, experiment_id: str) -> dict[str, Any]:
    """Load running or sealed capacity status without importing a model/runtime."""
    if _EXPERIMENT_ID.fullmatch(experiment_id) is None:
        raise ValueError("experiment_id must be one portable path component")
    experiment_root = Path(root).expanduser().resolve() / experiment_id
    sealed = (experiment_root / M1_CAPACITY_INTEGRITY_FILE).is_file()
    if sealed:
        validate_m1_capacity_artifacts(experiment_root)
    status = _read_json(experiment_root / STATUS_FILE)
    if status.get("experiment_id") != experiment_id:
        raise ValueError("M1 capacity status experiment identity mismatch")
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
    }


__all__ = ["LongContextM1CapacityRunner", "load_m1_capacity_status"]
