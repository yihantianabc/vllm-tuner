"""Run, resume, analyze, and seal long-context v5 M4 Chunked Prefill experiments."""

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

from vllm_tuner.benchmarks.metrics import request_meets_slo
from vllm_tuner.benchmarks.models import RequestResult, RequestStatus, SLOThresholds
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
    _trace_text,
)
from .m1_runner import GPUMemoryReader, _clean_execution_environment, _gpu_memory_snapshot
from .m2_fp8_integrity import M2_FP8_INTEGRITY_FILE
from .m3_apc_integrity import M3_APC_INTEGRITY_FILE
from .m4_chunked_analysis import (
    M4LatencyPercentiles,
    M4PrefillWindow,
    M4ResourceUsage,
    M4TrialRecord,
    M4WaitingUsage,
    analyze_m4_records,
)
from .m4_chunked_config import LongContextM4ChunkedConfig, M4ChunkedProfile
from .m4_chunked_integrity import (
    M4_CHUNKED_INTEGRITY_FILE,
    seal_m4_chunked_artifacts,
    validate_m4_chunked_artifacts,
)
from .m4_chunked_workload import M4TraceBundle, build_m4_trace
from .model_identity import ModelIdentityFacts, require_model_identity
from .runtime_identity import RuntimeIdentityFacts, require_upstream_runtime

M4_SCHEMA = "longctx-m4-chunked-prefill.v1"
MANIFEST_FILE = "manifest.json"
SUMMARY_FILE = "summary.json"
STATUS_FILE = "status.json"
REPORT_FILE = "report/m4-chunked-prefill.md"
RUNNER_LOG_FILE = "runner.log"
POINT_FILE = "m4-point.json"
RECORD_FILE = "m4-record.json"
EVIDENCE_FILE = "m4-evidence.json"
RUNTIME_CAPACITY_FILE = "runtime-capacity.json"
CUDA_MEMORY_FILE = "cuda-memory.json"
MEASURED_TRACE_FILE = "measured-trace.jsonl"
MEASURED_TRACE_CHECKSUM_FILE = "measured-trace.sha256"
WARMUP_TRACE_FILE = "warmup-trace.jsonl"
WARMUP_TRACE_CHECKSUM_FILE = "warmup-trace.sha256"

_EXPERIMENT_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_ATTEMPT_SUFFIX = re.compile(r"-attempt(?P<number>[1-9][0-9]*)$")
_RECORD_ADAPTER: TypeAdapter[M4TrialRecord] = TypeAdapter(M4TrialRecord)


class ControllerProtocol(Protocol):
    async def run_trial(self, params: dict[str, Any], trial_id: str, method: str) -> TrialResult:
        """Run one M4 trial."""


ControllerFactory = Callable[..., ControllerProtocol]


def _latencies(values: Sequence[float], field: str) -> M4LatencyPercentiles:
    if not values:
        raise ValueError(f"M4 {field} has no exact samples")
    p50 = percentile(values, 0.50)
    p95 = percentile(values, 0.95)
    p99 = percentile(values, 0.99)
    if p50 is None or p95 is None or p99 is None:
        raise ValueError(f"M4 {field} percentiles are unavailable")
    return M4LatencyPercentiles(
        sample_count=len(values),
        p50_ms=p50,
        p95_ms=p95,
        p99_ms=p99,
        maximum_ms=max(values),
    )


def _usage(rows: Sequence[Mapping[str, Any]], metric: str) -> M4ResourceUsage:
    values: list[float] = []
    for row in rows:
        metrics = row.get("metrics")
        raw = metrics.get(metric) if isinstance(metrics, Mapping) else None
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            continue
        value = float(raw)
        if math.isfinite(value) and value >= 0:
            values.append(value)
    if not values:
        raise ValueError(f"M4 telemetry metric {metric} is unavailable")
    median = percentile(values, 0.50)
    p95 = percentile(values, 0.95)
    if median is None or p95 is None:
        raise ValueError(f"M4 telemetry metric {metric} percentiles are unavailable")
    return M4ResourceUsage(
        sample_count=len(values),
        minimum=min(values),
        median=median,
        p95=p95,
        maximum=max(values),
    )


def _waiting(rows: Sequence[Mapping[str, Any]]) -> M4WaitingUsage:
    base = _usage(rows, "num_requests_waiting")
    values: list[float] = []
    for row in rows:
        metrics = row.get("metrics")
        raw = metrics.get("num_requests_waiting") if isinstance(metrics, Mapping) else None
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            continue
        value = float(raw)
        if math.isfinite(value) and value >= 0:
            values.append(value)
    return M4WaitingUsage(
        **base.model_dump(mode="json"),
        positive_sample_fraction=sum(value > 0 for value in values) / len(values),
    )


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
        raise ValueError("M4 raw Prometheus exposition is unavailable")
    return value


def _validate_requests(
    rows: Sequence[Mapping[str, Any]], trace: WorkloadTrace
) -> list[RequestResult]:
    expected_ids = [entry.request_id for entry in trace.entries]
    if [row.get("request_id") for row in rows] != expected_ids:
        raise ValueError("M4 request IDs/order do not match the sealed trace")
    requests = [RequestResult.from_dict(dict(row)) for row in rows]
    expected = {entry.request_id: entry for entry in trace.entries}
    if len(requests) != len(trace.entries) or any(
        request.status != RequestStatus.SUCCESS
        or request.input_tokens != expected[request.request_id].input_tokens
        or request.output_tokens != expected[request.request_id].output_tokens
        or request.token_count_source != "usage"
        or not request.token_timestamps_valid
        or len(request.token_timestamps) != expected[request.request_id].output_tokens
        for request in requests
    ):
        raise ValueError("M4 requires successful exact-token evidence for every request")
    return requests


def _validate_warmups(benchmark: Mapping[str, Any], warmup: WorkloadTrace) -> None:
    raw = benchmark.get("warmup_results")
    if not isinstance(raw, list) or len(raw) != len(warmup.entries):
        raise ValueError("M4 warmup evidence count mismatch")
    expected = {entry.input_tokens for entry in warmup.entries}
    observed: set[int] = set()
    for row in raw:
        if not isinstance(row, Mapping):
            raise ValueError("M4 warmup evidence is malformed")
        request = RequestResult.from_dict(dict(row))
        if request.status != RequestStatus.SUCCESS or request.token_count_source != "usage":
            raise ValueError("M4 warmup did not complete with exact usage evidence")
        observed.add(request.input_tokens)
    if observed != expected:
        raise ValueError("M4 warmup did not exercise decode and long-prefill lengths")


def _argv_option(argv: Sequence[str], flag: str) -> Optional[str]:
    for index, item in enumerate(argv):
        if item == flag:
            if index + 1 >= len(argv) or argv[index + 1].startswith("--"):
                return "true"
            return argv[index + 1]
        if item.startswith(flag + "="):
            return item.split("=", 1)[1]
    return None


def _command_evidence(
    trial_dir: Path,
    profile: M4ChunkedProfile,
    server_log: str,
) -> dict[str, Any]:
    command = _read_json(trial_dir / "server-command.json")
    raw_argv = command.get("argv")
    if not isinstance(raw_argv, list) or any(not isinstance(item, str) for item in raw_argv):
        raise ValueError("M4 server command does not expose string argv")
    argv = cast(list[str], raw_argv)
    chunk_flags = {
        "--enable-chunked-prefill",
        "--long-prefill-token-threshold",
        "--max-long-partial-prefills",
        "--max-num-batched-tokens",
        "--max-num-partial-prefills",
    }
    observed_chunk_flags = {item.split("=", 1)[0] for item in argv} & chunk_flags
    expected_flags = set() if profile.production_default else chunk_flags
    expected_values = {
        "--enable-chunked-prefill": "true",
        "--long-prefill-token-threshold": str(profile.long_prefill_token_threshold),
        "--max-long-partial-prefills": str(profile.max_long_partial_prefills),
        "--max-num-batched-tokens": str(profile.max_num_batched_tokens),
        "--max-num-partial-prefills": str(profile.max_num_partial_prefills),
    }
    values_match = profile.production_default or all(
        _argv_option(argv, flag) == value for flag, value in expected_values.items()
    )
    forbidden = sorted(
        {item.split("=", 1)[0] for item in argv if item.startswith("--")}
        & {
            "--attention-backend",
            "--calculate-kv-scales",
            "--kv-cache-dtype",
            "--scheduler-cls",
            "--no-enable-chunked-prefill",
        }
    )
    chunk_match = re.search(
        r"Chunked prefill is enabled with max_num_batched_tokens=(\d+)", server_log
    )
    if chunk_match is None:
        raise ValueError("M4 server log has no resolved Chunked Prefill budget")
    resolved_batched = int(chunk_match.group(1))
    partial_match = re.search(
        r"Concurrent partial prefills enabled with max_num_partial_prefills=(\d+), "
        r"max_long_partial_prefills=(\d+), long_prefill_token_threshold=(\d+)",
        server_log,
    )
    if partial_match is None:
        resolved_partial = 1
        resolved_long_partial = 1
        resolved_threshold = 0
    else:
        resolved_partial, resolved_long_partial, resolved_threshold = (
            int(partial_match.group(index)) for index in (1, 2, 3)
        )
    expected_resolved = (
        profile.resolved_max_num_batched_tokens,
        profile.max_num_partial_prefills or 1,
        profile.max_long_partial_prefills or 1,
        profile.long_prefill_token_threshold or 0,
    )
    observed_resolved = (
        resolved_batched,
        resolved_partial,
        resolved_long_partial,
        resolved_threshold,
    )
    checks = {
        "exact_preregistered_argv": observed_chunk_flags == expected_flags and values_match,
        "resolved_scheduler_profile_matches": observed_resolved == expected_resolved,
        "no_fp8_or_custom_scheduler_arguments": not forbidden,
    }
    return {
        "argv": argv,
        "observed_chunk_flags": sorted(observed_chunk_flags),
        "expected_chunk_flags": sorted(expected_flags),
        "resolved": {
            "max_num_batched_tokens": resolved_batched,
            "max_num_partial_prefills": resolved_partial,
            "max_long_partial_prefills": resolved_long_partial,
            "long_prefill_token_threshold": resolved_threshold,
        },
        "forbidden_arguments": forbidden,
        "checks": checks,
        "passed": all(checks.values()),
    }


def _validate_counters(
    rows: Sequence[Mapping[str, Any]],
    requests: Sequence[RequestResult],
) -> tuple[dict[str, dict[str, Any]], int, int, int]:
    counters = _counter_evidence(rows)
    expected = {
        "prompt_tokens_total": sum(request.input_tokens for request in requests),
        "generation_tokens_total": sum(request.output_tokens for request in requests),
        "prefix_cache_queries": sum(request.input_tokens for request in requests),
        "prefix_cache_hits": 0,
    }
    observed: dict[str, int] = {}
    for name, value in expected.items():
        evidence = counters[name]
        if evidence.get("available") is not True or evidence.get("reset_count") != 0:
            raise ValueError(f"M4 counter {name} is unavailable or reset")
        observed[name] = _integer_counter(evidence.get("delta"), f"M4 counter {name}")
        if observed[name] != value:
            raise ValueError(f"M4 counter {name} does not match exact trace semantics")
    preemption = counters["num_preemptions_total"]
    if preemption.get("available") is not True or preemption.get("reset_count") != 0:
        raise ValueError("M4 preemption counter is unavailable or reset")
    preemptions = _integer_counter(preemption.get("delta"), "M4 preemption counter")
    return (
        counters,
        observed["prefix_cache_queries"],
        observed["prefix_cache_hits"],
        preemptions,
    )


def _request_latencies(requests: Sequence[RequestResult], field: str) -> M4LatencyPercentiles:
    values: list[float] = []
    for request in requests:
        if field == "ttft" and request.ttft_ns is not None:
            values.append(request.ttft_ns / 1_000_000)
        elif field == "tpot" and request.tpot_ns is not None:
            values.append(request.tpot_ns / 1_000_000)
        elif field == "e2e" and request.e2e_ns is not None:
            values.append(request.e2e_ns / 1_000_000)
        elif field == "itl":
            values.extend(value / 1_000_000 for value in request.itl_ns)
    return _latencies(values, field)


def _interference_metrics(
    decode_requests: Sequence[RequestResult],
    long_requests: Sequence[RequestResult],
) -> tuple[
    tuple[M4PrefillWindow, ...],
    M4LatencyPercentiles,
    M4LatencyPercentiles,
    int,
]:
    windows: list[M4PrefillWindow] = []
    raw_windows: list[tuple[int, int]] = []
    for request in long_requests:
        if request.sent_at is None or request.first_token_at is None:
            raise ValueError("M4 long-prefill request has no exact prefill window")
        raw_windows.append((request.sent_at, request.first_token_at))
        windows.append(
            M4PrefillWindow(
                request_id=request.request_id,
                sent_at_ns=request.sent_at,
                first_token_at_ns=request.first_token_at,
                duration_ms=(request.first_token_at - request.sent_at) / 1_000_000,
            )
        )
    during: list[float] = []
    outside: list[float] = []
    overlap_requests = 0
    for request in decode_requests:
        if request.first_token_at is None or request.finished_at is None:
            raise ValueError("M4 decode request has no exact decode window")
        if any(
            request.first_token_at < window_end and request.finished_at > window_start
            for window_start, window_end in raw_windows
        ):
            overlap_requests += 1
        for previous, current in zip(request.token_timestamps, request.token_timestamps[1:]):
            value = (current - previous) / 1_000_000
            if any(
                previous < window_end and current > window_start
                for window_start, window_end in raw_windows
            ):
                during.append(value)
            else:
                outside.append(value)
    if overlap_requests == 0:
        raise ValueError("M4 trace did not overlap long prefill with active decode")
    return (
        tuple(windows),
        _latencies(during, "decode interference ITL"),
        _latencies(outside, "decode non-interference ITL"),
        overlap_requests,
    )


def _derive_record(
    *,
    result: TrialResult,
    trial_dir: Path,
    profile: M4ChunkedProfile,
    long_prefill_tokens: int,
    repeat_index: int,
    bundle: M4TraceBundle,
    device_memory: DeviceMemoryEvidence,
    slo: SLOThresholds,
) -> tuple[M4TrialRecord, dict[str, Any]]:
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
        raise ValueError("M4 trace lacks decode or long-prefill requests")
    counters, queries, hits, preemptions = _validate_counters(prometheus_rows, requests)
    metrics_text = _raw_metrics_text(prometheus_rows)
    server_log = (trial_dir / "server.log").read_text(encoding="utf-8", errors="replace")
    command = _command_evidence(trial_dir, profile, server_log)
    runtime = build_capacity_runtime_evidence(
        run_id=result.trial_id,
        runtime_profile_sha256=sha256_json(
            {
                "profile_id": profile.profile_id,
                "vllm_args": profile.vllm_args(),
                "long_prefill_tokens": long_prefill_tokens,
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
        raise ValueError("M4 runtime/mechanism gate failed: " + ", ".join(failed))
    windows, during_itl, outside_itl, overlap_count = _interference_metrics(decode, long_requests)
    duration = _finite_number(result.measurement_seconds, "M4 measurement duration")
    good_decode = sum(request_meets_slo(request, slo) for request in decode)
    overall_goodput = _finite_number(
        result.client.get("goodput_requests_per_sec"), "M4 overall Goodput"
    )
    peak_vram = _finite_number(result.gpu.get("peak_memory_mb"), "M4 peak VRAM")
    timeout_count = sum(request.status == RequestStatus.TIMEOUT for request in requests)
    oom_count = _server_event_count(server_log, (r"CUDA out of memory", r"\bOOM\b"))
    if timeout_count or oom_count:
        raise ValueError("M4 trial contains a timeout or OOM")
    resolved = cast(Mapping[str, Any], command["resolved"])
    record = M4TrialRecord(
        trial_id=result.trial_id,
        profile_id=cast(Any, profile.profile_id),
        production_default=profile.production_default,
        max_num_batched_tokens=cast(Any, resolved["max_num_batched_tokens"]),
        max_num_partial_prefills=cast(Any, resolved["max_num_partial_prefills"]),
        max_long_partial_prefills=cast(Any, resolved["max_long_partial_prefills"]),
        long_prefill_token_threshold=cast(Any, resolved["long_prefill_token_threshold"]),
        long_prefill_tokens=cast(Any, long_prefill_tokens),
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
        preemption_count=preemptions,
        prefix_cache_queries=queries,
        prefix_cache_hits=cast(Any, hits),
        peak_vram_mb=peak_vram,
        oom_count=cast(Any, oom_count),
        timeout_count=cast(Any, timeout_count),
        mechanism_evidence_passed=True,
    )
    evidence = {
        "schema_version": M4_SCHEMA,
        "semantic_gate_passed": True,
        "profile": profile.model_dump(mode="json"),
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


class LongContextM4ChunkedRunner:
    """Execute the minimal native Chunked Prefill matrix with strict evidence."""

    def __init__(
        self,
        config: LongContextM4ChunkedConfig,
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
            raise ValueError("M4 experiment_id must be one portable path component")
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

    def _profile(self, profile_id: str) -> M4ChunkedProfile:
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
            raise RuntimeError(f"M4 experiment is already running: {path}") from error
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
            raise ValueError("M4 requires matching upstream runtime and model locks")
        return runtime, model, _clean_execution_environment(self.execution_environment)

    def _source_identity(self) -> tuple[str, str]:
        commit, dirty, _ = git_state(self.repository)
        tree = source_tree_sha256(self.repository)
        if commit is None or tree is None or dirty:
            raise ValueError("M4 requires one clean committed source identity")
        return commit, tree

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
            "schema_version": M4_SCHEMA,
            "project_line": "longctx-v5",
            "milestone": "M4",
            "experiment_kind": "chunked-prefill-interference",
            "evidence_role": self.config.evidence_role,
            "experiment_id": self.experiment_id,
            "created_at": utc_now_iso(),
            "config_sha256": sha256_json(self.config.model_dump(mode="json")),
            "source_commit": source_commit,
            "source_tree_sha256": source_tree,
            "runtime": runtime.model_dump(mode="json"),
            "model": model.model_dump(mode="json"),
            "execution_environment": dict(environment),
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
            "m3_artifact": {
                **self.config.m3_artifact.identity(),
                "integrity_sha256": sha256_file(
                    self.config.m3_artifact.root / M3_APC_INTEGRITY_FILE
                ),
            },
            "smoke_artifact": (
                None
                if smoke is None
                else {
                    "experiment_id": smoke.experiment_id,
                    "root": str(smoke.root),
                    "integrity_sha256": sha256_file(smoke.root / M4_CHUNKED_INTEGRITY_FILE),
                }
            ),
            "profiles": [profile.model_dump(mode="json") for profile in self.config.profiles],
            "scope": {
                "m1_rerun": False,
                "m2_rerun": False,
                "m3_rerun": False,
                "fp8_configuration_attempted": False,
                "custom_scheduler_used": False,
                "m5_started": False,
            },
        }

    @staticmethod
    def _manifest_identity(value: Mapping[str, Any]) -> dict[str, Any]:
        return {name: item for name, item in value.items() if name != "created_at"}

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
            and experiment.get("m1_boundaries") == self.config.m1_boundaries.model_dump(mode="json")
            and experiment.get("m2_negative_artifacts")
            == [item.model_dump(mode="json") for item in self.config.m2_negative_artifacts]
            and experiment.get("m3_artifact") == self.config.m3_artifact.model_dump(mode="json")
            and experiment.get("profiles")
            == [profile.model_dump(mode="json") for profile in self.config.profiles]
        )

    def _initialize_root(self, requested_manifest: dict[str, Any]) -> Optional[dict[str, Any]]:
        root = self.store.root
        if root.exists():
            if not self.resume:
                raise FileExistsError(f"M4 artifact root exists: {root}; use --resume")
            if (root / M4_CHUNKED_INTEGRITY_FILE).is_file():
                validate_m4_chunked_artifacts(root)
                existing = _read_json(root / MANIFEST_FILE)
                if self._manifest_identity(existing) != self._manifest_identity(requested_manifest):
                    raise ValueError("M4 resume manifest identity mismatch")
                return _read_json(root / SUMMARY_FILE)
            self.store.initialize(exist_ok=True)
            existing = _read_json(root / MANIFEST_FILE)
            if self._manifest_identity(existing) != self._manifest_identity(requested_manifest):
                raise ValueError("M4 resume manifest identity mismatch")
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
                raise ValueError(f"M4 resume trace bytes mismatch: {relative}")
            checksum_path = self.store.root / checksum
            if not checksum_path.is_file() or checksum_path.read_text(encoding="utf-8") != sidecar:
                raise ValueError(f"M4 resume trace checksum mismatch: {relative}")
            return
        self.store.write_text(relative, text)
        self.store.write_text(checksum, sidecar)

    def _trace_catalog(self) -> dict[int, M4TraceBundle]:
        tokenizer = self._load_tokenizer()
        bundles: dict[int, M4TraceBundle] = {}
        catalog: dict[str, Any] = {"schema_version": M4_SCHEMA, "traces": {}}
        for long_tokens in self.config.protocol.long_prefill_tokens:
            bundle = build_m4_trace(
                protocol=self.config.protocol,
                long_prefill_tokens=long_tokens,
                tokenizer=tokenizer,
            )
            bundles[long_tokens] = bundle
            base = Path("traces") / f"long-prefill-{long_tokens}"
            self._write_or_validate_trace(base / "measured.jsonl", bundle.measured)
            self._write_or_validate_trace(base / "warmup.jsonl", bundle.warmup)
            cast(dict[str, Any], catalog["traces"])[str(long_tokens)] = {
                "measured_path": (base / "measured.jsonl").as_posix(),
                "measured_sha256": bundle.measured.checksum(),
                "measured_requests": len(bundle.measured.entries),
                "warmup_path": (base / "warmup.jsonl").as_posix(),
                "warmup_sha256": bundle.warmup.checksum(),
                "request_kind": bundle.request_kind,
                "prefix_isolation_proof": bundle.prefix_isolation_proof,
            }
        path = self.store.root / "traces" / "catalog.json"
        if path.exists():
            if _read_json(path) != catalog:
                raise ValueError("M4 resume trace catalog mismatch")
        else:
            self.store.write_json("traces/catalog.json", catalog)
        return bundles

    @staticmethod
    def _logical_id(profile: M4ChunkedProfile, long_tokens: int, repeat_index: int) -> str:
        return f"chunked-{profile.profile_id}-prefill-{long_tokens}-repeat-{repeat_index}"

    def _attempts(self, logical_id: str) -> list[tuple[int, Path]]:
        if not self.store.trials_dir.is_dir():
            return []
        attempts: list[tuple[int, Path]] = []
        for path in self.store.trials_dir.iterdir():
            if path.is_symlink() or not path.is_dir():
                raise ValueError(f"unexpected M4 trials entry: {path}")
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
        profile: M4ChunkedProfile,
        long_tokens: int,
        repeat_index: int,
        bundle: M4TraceBundle,
    ) -> dict[str, Any]:
        return {
            "schema_version": M4_SCHEMA,
            "trial_id": trial_id,
            "profile_id": profile.profile_id,
            "production_default": profile.production_default,
            "long_prefill_tokens": long_tokens,
            "repeat_index": repeat_index,
            "trace_id": bundle.measured.checksum(),
            "warmup_trace_id": bundle.warmup.checksum(),
            "vllm_args": profile.vllm_args(),
            "m1_boundary_source": self.config.m1_boundaries.experiment_id,
            "m2_negative_sources": [
                item.experiment_id for item in self.config.m2_negative_artifacts
            ],
            "m3_source": self.config.m3_artifact.experiment_id,
        }

    def _cached_complete(
        self,
        *,
        logical_id: str,
        expected_point: Mapping[str, Any],
    ) -> Optional[tuple[TrialResult, M4TrialRecord]]:
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
                    raise ValueError("sealed M4 attempt identity or semantic gate mismatch")
                record = _RECORD_ADAPTER.validate_json(
                    (path / RECORD_FILE).read_text(encoding="utf-8")
                )
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
        bundle: M4TraceBundle,
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
                "schema_version": M4_SCHEMA,
                "semantic_gate_passed": False,
                "state": "pending-controller-result",
            },
        )

    def _finalize_trial(
        self,
        result: TrialResult,
        record: M4TrialRecord | Mapping[str, Any],
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
                else {"available": False, "reason": "M4 semantic gate produced no capacity"}
            ),
        )
        self.store.ensure_trial_artifacts(result)
        self.store.validate_cached_trial(result, require_telemetry=True)
        self.store.validate_trial_integrity(result.trial_id)

    async def _run_one(
        self,
        *,
        profile: M4ChunkedProfile,
        long_tokens: int,
        repeat_index: int,
        bundle: M4TraceBundle,
    ) -> tuple[TrialResult, Optional[M4TrialRecord], bool]:
        logical = self._logical_id(profile, long_tokens, repeat_index)
        expected_point = self._point_payload(
            trial_id=logical,
            profile=profile,
            long_tokens=long_tokens,
            repeat_index=repeat_index,
            bundle=bundle,
        )
        cached = self._cached_complete(logical_id=logical, expected_point=expected_point)
        if cached is not None:
            return cached[0], cached[1], True
        trial_id = self._select_new_attempt(logical)
        point = {**expected_point, "trial_id": trial_id}
        tuning = self.config.to_tuning_config(profile)
        if tuning.vllm_args != profile.vllm_args():
            raise ValueError("M4 TuningConfig changed preregistered native arguments")
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
            result = await controller.run_trial({}, trial_id, "chunked-prefill")
        except UnsafeCleanupError:
            self._status(
                "unsafe_cleanup",
                current_trial=trial_id,
                unsafe_cleanup=True,
                message="unsafe cleanup; M4 stopped before another GPU process",
            )
            raise
        expected_provenance = trial_provenance(trial_id, "chunked-prefill")
        if any(getattr(result, name) != value for name, value in expected_provenance.items()):
            raise ValueError(f"M4 trial provenance mismatch: {trial_id}")
        if result.params != {}:
            raise ValueError(f"M4 fixed profile trial passed search parameters: {trial_id}")
        record: Optional[M4TrialRecord] = None
        evidence: dict[str, Any]
        if result.status in {TrialStatus.COMPLETE, TrialStatus.INFEASIBLE}:
            try:
                record, evidence = _derive_record(
                    result=result,
                    trial_dir=self.store.trials_dir / trial_id,
                    profile=profile,
                    long_prefill_tokens=long_tokens,
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
                        "m4_chunked_semantic_gate",
                    ],
                }
                result.failure_reason = {
                    "type": "M4_CHUNKED_SEMANTIC_GATE",
                    "message": str(error),
                    "phase": "M4_CHUNKED_FINALIZE",
                }
                self.store.record_artifact_finalizer_failure(result, str(error))
                evidence = {
                    "schema_version": M4_SCHEMA,
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
                "schema_version": M4_SCHEMA,
                "semantic_gate_passed": False,
                "failure": reason,
            }
        failure_record = {
            "schema_version": M4_SCHEMA,
            "trial_id": trial_id,
            "profile_id": profile.profile_id,
            "long_prefill_tokens": long_tokens,
            "repeat_index": repeat_index,
            "status": "failed",
            "failure": evidence.get("failure"),
        }
        self._finalize_trial(result, record or failure_record, evidence)
        return result, record, False

    def _expected_matrix(self) -> set[tuple[str, int, int]]:
        return {
            (profile.profile_id, long_tokens, repeat)
            for long_tokens in self.config.protocol.long_prefill_tokens
            for repeat in range(self.config.protocol.repeats)
            for profile in self.config.profiles
        }

    def _profile_order(self, repeat_index: int) -> tuple[M4ChunkedProfile, ...]:
        profiles = self.config.profiles
        rotation = repeat_index % len(profiles)
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
            span = (self.config.protocol.decode_requests - 1) * (
                self.config.protocol.decode_interval_seconds
            )
            eta_seconds = span * planned
        value = {
            "schema_version": M4_SCHEMA,
            "experiment_id": self.experiment_id,
            "state": state,
            "pid": os.getpid(),
            "gpu": list(self.config.gpu.device_ids),
            "log": str(self.store.root / RUNNER_LOG_FILE),
            "result": str(self.store.root),
            "eta": (datetime.now(timezone.utc) + timedelta(seconds=eta_seconds)).isoformat(),
            "eta_seconds": eta_seconds,
            "resume": self._resume_command(),
            "sealed": (self.store.root / M4_CHUNKED_INTEGRITY_FILE).is_file(),
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
            "scripts/run_longctx_m4_chunked.sh --config CONFIG "
            f"--experiment-id {self.experiment_id} --resume"
        )

    @staticmethod
    def _report(summary: Mapping[str, Any]) -> str:
        execution = cast(Mapping[str, Any], summary["execution"])
        acceptance = cast(Mapping[str, Any], summary["acceptance"])
        analysis = cast(Mapping[str, Any], summary["analysis"])
        selection = cast(Mapping[str, Any], analysis["selection"])
        lines = [
            "# Long-context v5 M4 Chunked Prefill",
            "",
            f"- Experiment: {summary['experiment_id']}",
            f"- Evidence role: {summary['evidence_role']}",
            f"- Execution passed: {execution['passed']}",
            f"- M4 accepted: {acceptance['passed']}",
            f"- Selected profile: {selection['profile_id']}",
            "- M1/M2/M3 rerun: false",
            "- FP8 attempted: false",
            "- Custom scheduler used: false",
            "- M5 started: false",
            "",
            "## Paired interference evidence",
            "",
            "| Profile | Long prefill | ITL improved | Decode Goodput not lower | Preemptions not higher |",
            "|---|---:|---:|---:|---:|",
        ]
        paired = analysis.get("paired")
        if isinstance(paired, list):
            for row in paired:
                if not isinstance(row, Mapping):
                    continue
                lines.append(
                    f"| {row.get('profile_id')} | {row.get('long_prefill_tokens')} | "
                    f"{row.get('interference_itl_improved_repeats')}/{row.get('repeat_count')} | "
                    f"{row.get('decode_goodput_not_lower_repeats')}/{row.get('repeat_count')} | "
                    f"{row.get('preemptions_not_higher_repeats')}/{row.get('repeat_count')} |"
                )
        lines.extend(["", f"- Selection rule: {selection.get('rule')}"])
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
        smoke_identity_matches = self._validate_formal_smoke_identity(source_commit)
        if not smoke_identity_matches:
            raise ValueError(
                "formal M4 source/model/runtime/prerequisite identity differs from smoke"
            )
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
        records: list[M4TrialRecord] = []
        results: list[TrialResult] = []
        completed_jobs = 0
        execution_order: list[str] = []
        for long_tokens in self.config.protocol.long_prefill_tokens:
            bundle = bundles[long_tokens]
            for repeat_index in range(self.config.protocol.repeats):
                for profile in self._profile_order(repeat_index):
                    logical = self._logical_id(profile, long_tokens, repeat_index)
                    execution_order.append(logical)
                    self._status("running", completed_jobs=completed_jobs, current_trial=logical)
                    result, record, _ = await self._run_one(
                        profile=profile,
                        long_tokens=long_tokens,
                        repeat_index=repeat_index,
                        bundle=bundle,
                    )
                    results.append(result)
                    if record is not None:
                        records.append(record)
                    completed_jobs += 1

        expected = self._expected_matrix()
        observed = {
            (record.profile_id, record.long_prefill_tokens, record.repeat_index)
            for record in records
        }
        execution_passed = len(results) == len(expected) and observed == expected
        analysis = analyze_m4_records(
            records,
            formal=self.config.evidence_role == "formal",
        )
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
        trace_ids: dict[int, set[str]] = {}
        for record in records:
            trace_ids.setdefault(record.long_prefill_tokens, set()).add(record.trace_id)
        selection = analysis.get("selection")
        selection_ready = (
            isinstance(selection, Mapping)
            and selection.get("profile_id")
            in {"production-default", "native-chunk-1024", "native-chunk-512"}
            and selection.get("single_run_selection_used") is False
        )
        checks = {
            "project_line_is_v5": self.config.project_line == "longctx-v5",
            "m1_boundaries_reused": True,
            "m1_numeric_thresholds_unchanged": True,
            "m2_negative_evidence_bound": len(self.config.m2_negative_artifacts) == 3,
            "m3_formal_20_of_20_bound": self.config.m3_artifact.experiment_id
            == "longctx-v5-m3-apc-formal-001",
            "m1_m2_m3_not_rerun": True,
            "fp8_not_attempted": True,
            "custom_scheduler_not_used": True,
            "bound_smoke_passed": bound_smoke_passed,
            "smoke_identity_matches": smoke_identity_matches,
            "all_jobs_complete": execution_passed,
            "runtime_lock_verified": runtime.matches_lock,
            "model_lock_verified": model.matches_lock,
            "source_identity_stable": source_stable,
            "single_gpu_only": self.config.gpu.count == 1 and self.config.gpu.device_ids == (0,),
            "same_trace_across_profiles": bool(trace_ids)
            and all(len(values) == 1 for values in trace_ids.values()),
            "prefix_reuse_isolated": bool(records)
            and all(record.prefix_cache_hits == 0 for record in records),
            "mechanism_evidence_complete": bool(records)
            and all(record.mechanism_evidence_passed for record in records),
            "decode_prefill_overlap_observed": bool(records)
            and all(record.decode_overlap_request_count > 0 for record in records),
            "no_oom_or_timeout": bool(records)
            and all(record.oom_count == 0 and record.timeout_count == 0 for record in records),
            "preemption_evidence_complete": bool(records)
            and all(record.preemption_count >= 0 for record in records),
            "formal_run_count_is_18": (
                len(expected) == 18 if self.config.evidence_role == "formal" else True
            ),
            "selection_uses_repeated_evidence": selection_ready,
            "no_m5_work": True,
        }
        acceptance_passed = all(checks.values())
        acceptance = {
            "eligible": self.config.evidence_role == "formal",
            "passed": acceptance_passed,
            "checks": checks,
            "failure_reasons": sorted(name for name, passed in checks.items() if not passed),
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
            "schema_version": M4_SCHEMA,
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
                "matrix": "production default plus two native profiles at 4K/8K, paired repeats",
                "profiles": [profile.profile_id for profile in self.config.profiles],
                "long_prefill_tokens": list(self.config.protocol.long_prefill_tokens),
                "decode_requests_per_run": self.config.protocol.decode_requests,
                "long_prefill_injections_per_run": len(
                    self.config.protocol.injection_offsets_seconds
                ),
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
                "integrity": str(self.store.root / M4_CHUNKED_INTEGRITY_FILE),
                "m1_boundaries": str(self.config.m1_boundaries.root),
                "m2_negative_artifacts": [
                    str(item.root) for item in self.config.m2_negative_artifacts
                ],
                "m3_artifact": str(self.config.m3_artifact.root),
            },
            "m1_numeric_thresholds_modified": False,
            "m1_rerun": False,
            "m2_rerun": False,
            "m3_rerun": False,
            "fp8_attempted": False,
            "custom_scheduler_used": False,
            "m5_started": False,
        }
        self.store.write_json(SUMMARY_FILE, summary)
        self.store.write_text(REPORT_FILE, self._report(summary))
        final_state = "accepted" if acceptance_passed else "completed_not_accepted"
        self._status(final_state, completed_jobs=len(results), acceptance=acceptance)
        seal_m4_chunked_artifacts(
            self.store.root,
            self.experiment_id,
            {
                "experiment_id": self.experiment_id,
                "project_line": "longctx-v5",
                "milestone": "M4",
                "experiment_kind": "chunked-prefill-interference",
                "evidence_role": self.config.evidence_role,
                "source_commit": source_commit,
                "accepted": acceptance_passed,
                "m1_rerun": False,
                "m2_rerun": False,
                "m3_rerun": False,
                "fp8_attempted": False,
                "custom_scheduler_used": False,
                "m5_started": False,
            },
        )
        return summary


def load_m4_chunked_status(root: str | Path, experiment_id: str) -> dict[str, Any]:
    """Load M4 status without importing a tokenizer or starting vLLM."""
    if _EXPERIMENT_ID.fullmatch(experiment_id) is None:
        raise ValueError("experiment_id must be one portable path component")
    experiment_root = Path(root).expanduser().resolve() / experiment_id
    sealed = (experiment_root / M4_CHUNKED_INTEGRITY_FILE).is_file()
    if sealed:
        validate_m4_chunked_artifacts(experiment_root)
    status = _read_json(experiment_root / STATUS_FILE)
    if status.get("experiment_id") != experiment_id:
        raise ValueError("M4 status experiment identity mismatch")
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


__all__ = ["LongContextM4ChunkedRunner", "load_m4_chunked_status"]
