"""Run, resume, analyze, and seal long-context v5 M2 FP8 KV experiments."""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import random
import re
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional, Protocol, cast

from pydantic import TypeAdapter

from vllm_tuner.benchmarks.models import RequestResult, RequestStatus
from vllm_tuner.experiment.artifacts import ARTIFACT_INTEGRITY_FILE, ArtifactStore
from vllm_tuner.experiment.manifest import (
    git_state,
    sha256_file,
    sha256_json,
    source_tree_sha256,
)
from vllm_tuner.experiment.models import TrialResult, TrialStatus, trial_provenance, utc_now_iso
from vllm_tuner.profiling.timeseries import percentile
from vllm_tuner.runtime.controller import TrialController
from vllm_tuner.runtime.failures import UnsafeCleanupError
from vllm_tuner.workloads.generator import (
    _fit_exact_token_count,
    _repeat_to_size,
    generate_trace,
)
from vllm_tuner.workloads.trace import TraceEntry, WorkloadTrace

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
from .m2_fp8_analysis import (
    M2FP8TrialRecord,
    M2LatencyPercentiles,
    analyze_m2_fp8_records,
)
from .m2_fp8_config import LongContextM2FP8Config, M2FP8Context, M2FP8Profile
from .m2_fp8_integrity import (
    M2_FP8_INTEGRITY_FILE,
    seal_m2_fp8_artifacts,
    validate_m2_fp8_artifacts,
)
from .model_identity import ModelIdentityFacts, require_model_identity
from .runtime_identity import RuntimeIdentityFacts, require_upstream_runtime

M2_FP8_SCHEMA = "longctx-m2-fp8.v1"
MANIFEST_FILE = "manifest.json"
SUMMARY_FILE = "summary.json"
STATUS_FILE = "status.json"
REPORT_FILE = "report/m2-fp8.md"
RUNNER_LOG_FILE = "runner.log"
POINT_FILE = "m2-point.json"
RECORD_FILE = "m2-record.json"
EVIDENCE_FILE = "m2-evidence.json"
RUNTIME_CAPACITY_FILE = "runtime-capacity.json"
CUDA_MEMORY_FILE = "cuda-memory.json"
MEASURED_TRACE_FILE = "measured-trace.jsonl"
MEASURED_TRACE_CHECKSUM_FILE = "measured-trace.sha256"
QUALITY_TRACE_FILE = "quality-trace.jsonl"
QUALITY_TRACE_CHECKSUM_FILE = "quality-trace.sha256"

_EXPERIMENT_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_ATTEMPT_SUFFIX = re.compile(r"-attempt(?P<number>[1-9][0-9]*)$")
_RECORD_ADAPTER: TypeAdapter[M2FP8TrialRecord] = TypeAdapter(M2FP8TrialRecord)


class ControllerProtocol(Protocol):
    async def run_trial(self, params: dict[str, Any], trial_id: str, method: str) -> TrialResult:
        """Run one M2 FP8 trial."""


ControllerFactory = Callable[..., ControllerProtocol]


def _scaled_trace(
    *,
    context: M2FP8Context,
    count: int,
    seed: int,
    tokenizer: Any,
    burstiness: float,
) -> WorkloadTrace:
    generated = generate_trace(
        "chat",
        count=count,
        request_rate=context.offered_requests_per_second,
        burstiness=burstiness,
        seed=seed,
        tokenizer=tokenizer,
        fixed_input_tokens=context.input_tokens,
        fixed_output_tokens=context.output_tokens,
        request_id_prefix=f"m2-{context.context_id}",
    )
    raw_span = _trace_span(generated)
    if raw_span <= 0:
        raise ValueError("M2 measured trace does not span positive time")
    target_span = (count - 1) / context.offered_requests_per_second
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
    empirical = (len(trace.entries) - 1) / _trace_span(trace)
    if not math.isclose(
        empirical,
        context.offered_requests_per_second,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise ValueError("M2 measured trace does not reproduce its frozen offered rate")
    return trace


def _quality_prompt(
    *,
    target_tokens: int,
    marker: str,
    probe_index: int,
    probe_count: int,
    tokenizer: Any,
) -> tuple[str, int]:
    """Build an exact-length long-context retrieval prompt with one scored marker."""
    filler = (
        "The archived deployment ledger records deterministic request timing, token counts, "
        "cache geometry, and cleanup evidence for an inference service. "
    )
    needle = (
        f"\nVerified ledger field {probe_index + 1}: the retrieval code is {marker}. "
        "This code is authoritative.\n"
    )
    suffix = (
        f"\nQuestion: What is the authoritative retrieval code? Reply with exactly {marker} "
        "and no other words. Ignore padding after this instruction. [END]\n"
    )
    filler_ids = list(tokenizer.encode(filler, add_special_tokens=False))
    needle_ids = list(tokenizer.encode(needle, add_special_tokens=False))
    suffix_ids = list(tokenizer.encode(suffix, add_special_tokens=False))
    remaining = target_tokens - len(needle_ids) - len(suffix_ids)
    if remaining <= 0:
        raise ValueError("quality prompt target is too short for its scored instruction")
    placement = round(remaining * (probe_index + 1) / (probe_count + 1))
    before = _repeat_to_size(filler_ids, placement)
    after = _repeat_to_size(filler_ids, remaining - placement)
    prompt = tokenizer.decode([*before, *needle_ids, *after, *suffix_ids])
    prompt, counted = _fit_exact_token_count(prompt, target_tokens, tokenizer)
    if marker not in prompt or "[END]" not in prompt:
        raise ValueError("quality prompt fitting removed its scored instruction")
    return prompt, counted


def _quality_trace(
    *,
    context: M2FP8Context,
    count: int,
    output_tokens: int,
    seed: int,
    prompt_offset: int,
    tokenizer: Any,
) -> tuple[WorkloadTrace, dict[str, str]]:
    rng = random.Random(seed + context.total_kv_tokens)
    entries: list[TraceEntry] = []
    markers: dict[str, str] = {}
    for index in range(count):
        marker = f"M2-{rng.getrandbits(48):012X}"
        prompt, counted = _quality_prompt(
            target_tokens=context.input_tokens,
            marker=marker,
            probe_index=index,
            probe_count=count,
            tokenizer=tokenizer,
        )
        request_id = f"quality-{context.context_id}-{prompt_offset + index:07d}"
        markers[request_id] = marker
        entries.append(
            TraceEntry(
                request_id=request_id,
                scheduled_offset_seconds=0.0,
                prompt=prompt,
                input_tokens=counted,
                output_tokens=output_tokens,
                profile="m2-quality-retrieval",
                shared_prefix_id=None,
            )
        )
    return (
        WorkloadTrace(
            seed=seed,
            profile="m2-quality-retrieval",
            request_rate=None,
            burstiness=1.0,
            entries=entries,
        ),
        markers,
    )


def _latencies(values: Sequence[float], field: str) -> M2LatencyPercentiles:
    if not values:
        raise ValueError(f"M2 {field} has no exact request samples")
    p50 = percentile(values, 0.50)
    p95 = percentile(values, 0.95)
    p99 = percentile(values, 0.99)
    if p50 is None or p95 is None or p99 is None:
        raise ValueError(f"M2 {field} percentiles are unavailable")
    return M2LatencyPercentiles(p50_ms=p50, p95_ms=p95, p99_ms=p99)


def _checkpoint_scale_keys(model_dir: Path) -> list[str]:
    index = _read_json(model_dir / "model.safetensors.index.json")
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, Mapping):
        raise ValueError("locked model index has no weight_map")
    markers = ("k_scale", "v_scale", "kv_scale", "q_scale", "prob_scale")
    return sorted(
        str(key)
        for key in weight_map
        if isinstance(key, str) and any(marker in key.casefold() for marker in markers)
    )


def _command_evidence(trial_dir: Path, profile: M2FP8Profile) -> dict[str, Any]:
    command = _read_json(trial_dir / "server-command.json")
    raw_argv = command.get("argv")
    if not isinstance(raw_argv, list) or any(not isinstance(item, str) for item in raw_argv):
        raise ValueError("M2 server command does not expose a string argv")
    argv = cast(list[str], raw_argv)

    def option(name: str) -> Optional[str]:
        matches = [index for index, item in enumerate(argv) if item == name]
        if len(matches) > 1:
            raise ValueError(f"M2 server command repeats option {name}")
        if not matches:
            return None
        index = matches[0]
        if index + 1 >= len(argv) or argv[index + 1].startswith("--"):
            raise ValueError(f"M2 server command option {name} has no value")
        return argv[index + 1]

    kv_dtype = option("--kv-cache-dtype")
    calculate = argv.count("--calculate-kv-scales") == 1
    attention_override = option("--attention-backend")
    expected_kv = None if profile.profile_id == "bf16-auto" else "fp8"
    checks = {
        "kv_cache_dtype_argument_matches": kv_dtype == expected_kv,
        "calculate_kv_scales_argument_matches": calculate == profile.calculate_kv_scales,
        "attention_backend_not_forced": attention_override is None,
    }
    return {
        "argv": argv,
        "kv_cache_dtype_argument": kv_dtype,
        "calculate_kv_scales_argument": calculate,
        "attention_backend_argument": attention_override,
        "checks": checks,
        "passed": all(checks.values()),
    }


def _quality_evidence(
    benchmark: Mapping[str, Any],
    quality: WorkloadTrace,
    markers: Mapping[str, str],
) -> dict[str, Any]:
    raw = benchmark.get("warmup_results")
    if not isinstance(raw, list) or any(not isinstance(row, Mapping) for row in raw):
        raise ValueError("M2 benchmark has no structured quality warmup results")
    rows = cast(list[Mapping[str, Any]], raw)
    if len(rows) != len(quality.entries):
        raise ValueError("M2 quality result count differs from its fixed trace")
    probes: list[dict[str, Any]] = []
    passed = 0
    for index, (entry, row) in enumerate(zip(quality.entries, rows)):
        expected_id = f"warmup-{index}-{entry.request_id}"
        marker = markers[entry.request_id]
        output = row.get("output_text")
        success = row.get("status") == "success"
        marker_present = isinstance(output, str) and marker.casefold() in output.casefold()
        exact_input = row.get("input_tokens") == entry.input_tokens
        probe_passed = (
            row.get("request_id") == expected_id and success and marker_present and exact_input
        )
        passed += int(probe_passed)
        probes.append(
            {
                "request_id": row.get("request_id"),
                "marker": marker,
                "marker_present": marker_present,
                "success": success,
                "exact_input_tokens": exact_input,
                "output_tokens": row.get("output_tokens"),
                "output_text_sha256": (
                    hashlib.sha256(output.encode("utf-8")).hexdigest()
                    if isinstance(output, str)
                    else None
                ),
                "passed": probe_passed,
            }
        )
    return {
        "fixed_seed": quality.seed,
        "trace_id": quality.checksum(),
        "probe_count": len(rows),
        "pass_count": passed,
        "all_passed": passed == len(rows),
        "scoring_rule": "successful response contains its exact fixed retrieval marker",
        "probes": probes,
    }


def _derive_record(
    *,
    result: TrialResult,
    trial_dir: Path,
    profile: M2FP8Profile,
    context: M2FP8Context,
    repeat_index: int,
    trace: WorkloadTrace,
    quality: WorkloadTrace,
    quality_markers: Mapping[str, str],
    device_memory: DeviceMemoryEvidence,
    checkpoint_scale_keys: Sequence[str],
) -> tuple[M2FP8TrialRecord, dict[str, Any]]:
    request_rows = _read_jsonl(trial_dir / "request-results.jsonl")
    prometheus_rows = _read_jsonl(trial_dir / "prometheus.jsonl")
    benchmark = _read_json(trial_dir / "benchmark-raw.json")
    expected_ids = [entry.request_id for entry in trace.entries]
    if [row.get("request_id") for row in request_rows] != expected_ids:
        raise ValueError("M2 request IDs/order do not match the fixed paired trace")
    requests = [RequestResult.from_dict(row) for row in request_rows]
    if len(requests) != len(trace.entries) or any(
        request.status != RequestStatus.SUCCESS
        or request.input_tokens != context.input_tokens
        or request.output_tokens != context.output_tokens
        or request.token_count_source != "usage"
        or not request.token_timestamps_valid
        or len(request.token_timestamps) != context.output_tokens
        for request in requests
    ):
        raise ValueError("M2 requires successful exact-token evidence for every measured request")

    observed_offsets = [request.scheduled_at for request in requests]
    if any(value is None for value in observed_offsets):
        raise ValueError("M2 measured requests are missing scheduled timestamps")
    scheduled = cast(list[int], observed_offsets)
    observed_span = (max(scheduled) - min(scheduled)) / 1_000_000_000
    empirical = (len(requests) - 1) / observed_span
    if not math.isclose(
        empirical,
        context.offered_requests_per_second,
        rel_tol=1e-9,
        abs_tol=1e-9,
    ):
        raise ValueError("M2 observed arrivals differ from the fixed paired trace")

    counters = _counter_evidence(prometheus_rows)
    expected_counters = {
        "prompt_tokens_total": sum(request.input_tokens for request in requests),
        "generation_tokens_total": sum(request.output_tokens for request in requests),
        "prefix_cache_queries": sum(request.input_tokens for request in requests),
        "prefix_cache_hits": 0,
    }
    for name, expected in expected_counters.items():
        item = counters[name]
        if (
            item.get("available") is not True
            or item.get("reset_count") != 0
            or _integer_counter(item.get("delta"), f"M2 counter {name}") != expected
        ):
            raise ValueError(f"M2 counter {name} does not match exact request semantics")
    preemption = counters["num_preemptions_total"]
    if preemption.get("available") is not True or preemption.get("reset_count") != 0:
        raise ValueError("M2 preemption counter is unavailable or reset")
    preemptions = _integer_counter(preemption.get("delta"), "M2 preemption counter")

    metrics_text = next(
        (
            str(row["raw_text"])
            for row in prometheus_rows
            if row.get("available") is True and isinstance(row.get("raw_text"), str)
        ),
        None,
    )
    if metrics_text is None:
        raise ValueError("M2 raw Prometheus exposition is unavailable")
    server_log = (trial_dir / "server.log").read_text(encoding="utf-8", errors="replace")
    runtime = build_capacity_runtime_evidence(
        run_id=result.trial_id,
        runtime_profile_sha256=sha256_json(
            {
                "profile_id": profile.profile_id,
                "context_id": context.context_id,
                "vllm_args": profile.vllm_args(),
            }
        ),
        server_log_text=server_log,
        metrics_text=metrics_text,
        device_memory=device_memory,
    )
    command = _command_evidence(trial_dir, profile)
    quality_result = _quality_evidence(benchmark, quality, quality_markers)
    cache = runtime.cache_config
    startup = runtime.startup_format
    runtime_checks = {
        "command_matches": command["passed"] is True,
        "logged_capacity_consistent": runtime.logged_capacity_consistent,
        "requested_cache_dtype_matches": cache.requested_cache_dtype == profile.kv_cache_dtype,
        "startup_cache_dtype_matches": startup.requested_kv_cache_dtype == profile.kv_cache_dtype,
        "calculate_kv_scales_matches": cache.calculate_kv_scales == profile.calculate_kv_scales,
        "attention_backend_matches": startup.attention_backend
        == profile.expected_attention_backend,
        "no_num_blocks_override": cache.num_gpu_blocks_override is None,
        "no_explicit_kv_memory": cache.kv_cache_memory_bytes is None,
        "quality_all_passed": quality_result["all_passed"] is True,
        "checkpoint_has_no_scale_keys": not checkpoint_scale_keys,
    }
    if not all(runtime_checks.values()):
        failed = sorted(name for name, passed in runtime_checks.items() if not passed)
        raise ValueError("M2 runtime/scale/quality gate failed: " + ", ".join(failed))

    ttft = [_finite_number(row.get("ttft_ms"), "M2 TTFT") for row in request_rows]
    tpot = [_finite_number(row.get("tpot_ms"), "M2 TPOT") for row in request_rows]
    e2e = [_finite_number(row.get("e2e_ms"), "M2 E2E") for row in request_rows]
    itl = [
        _finite_number(value, "M2 ITL")
        for row in request_rows
        for value in cast(list[Any], row.get("itl_ms", []))
    ]
    peak_vram = _finite_number(result.gpu.get("peak_memory_mb"), "M2 peak VRAM")
    achieved = _finite_number(result.client.get("achieved_requests_per_sec"), "M2 achieved rate")
    goodput = _finite_number(result.client.get("goodput_requests_per_sec"), "M2 goodput rate")
    good = _integer_counter(result.client.get("good_requests"), "M2 good request count")
    timeout_count = sum(request.status == RequestStatus.TIMEOUT for request in requests)
    oom_count = _server_event_count(server_log, (r"CUDA out of memory", r"\bOOM\b"))
    if timeout_count or oom_count:
        raise ValueError("M2 complete evidence must not contain timeout or OOM events")

    record = M2FP8TrialRecord(
        trial_id=result.trial_id,
        profile_id=cast(Any, profile.profile_id),
        context_id=context.context_id,
        context_tokens=context.total_kv_tokens,
        repeat_index=repeat_index,
        trace_id=trace.checksum(),
        status="complete",
        requested_kv_cache_dtype=profile.kv_cache_dtype,
        calculate_kv_scales=profile.calculate_kv_scales,
        scale_source=profile.scale_source,
        attention_backend=cast(Any, startup.attention_backend),
        backend_resolution=profile.backend_resolution,
        num_gpu_blocks=cache.num_gpu_blocks,
        usable_num_gpu_blocks=cache.usable_num_gpu_blocks,
        block_size=cache.resolved_block_size,
        cached_tokens=runtime.observation.cached_tokens,
        quality_probe_count=int(quality_result["probe_count"]),
        quality_pass_count=int(quality_result["pass_count"]),
        quality_passed=True,
        request_count=len(requests),
        completion_fraction=1.0,
        achieved_requests_per_second=achieved,
        goodput_requests_per_second=goodput,
        slo_satisfied_fraction=good / len(requests),
        preemption_count=preemptions,
        oom_count=oom_count,
        timeout_count=timeout_count,
        peak_vram_mb=peak_vram,
        ttft=_latencies(ttft, "TTFT"),
        tpot=_latencies(tpot, "TPOT"),
        itl=_latencies(itl, "ITL"),
        end_to_end=_latencies(e2e, "E2E"),
    )
    evidence = {
        "schema_version": M2_FP8_SCHEMA,
        "semantic_gate_passed": True,
        "runtime_checks": runtime_checks,
        "command": command,
        "scale": {
            "requested_source": profile.scale_source,
            "calculate_kv_scales": cache.calculate_kv_scales,
            "checkpoint_scale_keys": list(checkpoint_scale_keys),
            "resolution": (
                "dynamic values calculated on first forward by locked vLLM 0.16"
                if profile.calculate_kv_scales
                else (
                    "not applicable: model dtype KV cache"
                    if profile.kv_cache_dtype == "auto"
                    else "1.0 fallback from absent checkpoint scales in locked vLLM 0.16"
                )
            ),
            "silent_scale_fallback": False,
        },
        "backend": {
            "resolved": startup.attention_backend,
            "resolution": profile.backend_resolution,
            "attention_backend_forced": False,
            "fp8_dtype_silently_fell_back": False,
        },
        "quality": quality_result,
        "counters": counters,
        "runtime_capacity": runtime.model_dump(mode="json"),
        "kv_blocks": {
            "num_gpu_blocks": cache.num_gpu_blocks,
            "usable_num_gpu_blocks": cache.usable_num_gpu_blocks,
            "block_size": cache.resolved_block_size,
            "cached_tokens": runtime.observation.cached_tokens,
        },
    }
    return record, evidence


class LongContextM2FP8Runner:
    """Execute the same M2 path for compatibility smoke and the 18-run formal matrix."""

    def __init__(
        self,
        config: LongContextM2FP8Config,
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
            raise ValueError("M2 experiment_id must be one portable path component")
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
        self._checkpoint_scale_keys = _checkpoint_scale_keys(config.model.local_path)

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
            raise RuntimeError(f"M2 FP8 experiment is already running: {path}") from error
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
            raise ValueError("M2 FP8 requires matching upstream runtime and model locks")
        return runtime, model, _clean_execution_environment(self.execution_environment)

    def _source_identity(self) -> tuple[str, str]:
        commit, dirty, _ = git_state(self.repository)
        tree = source_tree_sha256(self.repository)
        if commit is None or tree is None or dirty:
            raise ValueError("M2 FP8 requires one clean committed source identity")
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
            "schema_version": M2_FP8_SCHEMA,
            "project_line": "longctx-v5",
            "milestone": "M2",
            "experiment_kind": "fp8-kv-cache",
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
            "smoke_artifact": (
                None
                if smoke is None
                else {
                    "experiment_id": smoke.experiment_id,
                    "root": str(smoke.root),
                    "integrity_sha256": sha256_file(smoke.root / M2_FP8_INTEGRITY_FILE),
                }
            ),
            "checkpoint_scale_keys": self._checkpoint_scale_keys,
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
            and manifest.get("m1_boundaries", {}).get("experiment_id")
            == self.config.m1_boundaries.experiment_id
            and experiment.get("model") == self.config.model.model_dump(mode="json")
            and experiment.get("runtime") == self.config.runtime.model_dump(mode="json")
        )

    def _initialize_root(self, requested_manifest: dict[str, Any]) -> Optional[dict[str, Any]]:
        root = self.store.root
        if root.exists():
            if not self.resume:
                raise FileExistsError(f"M2 FP8 artifact root exists: {root}; use --resume")
            if (root / M2_FP8_INTEGRITY_FILE).is_file():
                validate_m2_fp8_artifacts(root)
                existing = _read_json(root / MANIFEST_FILE)
                if self._manifest_identity(existing) != self._manifest_identity(requested_manifest):
                    raise ValueError("M2 FP8 resume manifest identity mismatch")
                return _read_json(root / SUMMARY_FILE)
            self.store.initialize(exist_ok=True)
            existing = _read_json(root / MANIFEST_FILE)
            if self._manifest_identity(existing) != self._manifest_identity(requested_manifest):
                raise ValueError("M2 FP8 resume manifest identity mismatch")
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
                raise ValueError(f"M2 resume trace bytes mismatch: {relative}")
            checksum_path = self.store.root / checksum
            if not checksum_path.is_file() or checksum_path.read_text(encoding="utf-8") != sidecar:
                raise ValueError(f"M2 resume trace checksum mismatch: {relative}")
            return
        self.store.write_text(relative, text)
        self.store.write_text(checksum, sidecar)

    def _trace_catalog(
        self,
    ) -> tuple[
        dict[str, WorkloadTrace],
        dict[str, tuple[WorkloadTrace, dict[str, str]]],
    ]:
        tokenizer = self._load_tokenizer()
        measured: dict[str, WorkloadTrace] = {}
        qualities: dict[str, tuple[WorkloadTrace, dict[str, str]]] = {}
        catalog: dict[str, Any] = {"schema_version": M2_FP8_SCHEMA, "contexts": {}}
        for context in self.config.contexts:
            trace = _scaled_trace(
                context=context,
                count=self.config.protocol.measured_request_count(context),
                seed=self.config.protocol.measurement_seed,
                tokenizer=tokenizer,
                burstiness=self.config.protocol.burstiness,
            )
            quality, markers = _quality_trace(
                context=context,
                count=self.config.protocol.quality_probe_count,
                output_tokens=self.config.protocol.quality_output_tokens,
                seed=self.config.protocol.quality_seed,
                prompt_offset=self.config.protocol.quality_prompt_index_offset,
                tokenizer=tokenizer,
            )
            measured[context.context_id] = trace
            qualities[context.context_id] = (quality, markers)
            measured_path = Path("traces") / context.context_id / "measured.jsonl"
            quality_path = Path("traces") / context.context_id / "quality.jsonl"
            self._write_or_validate_trace(measured_path, trace)
            self._write_or_validate_trace(quality_path, quality)
            cast(dict[str, Any], catalog["contexts"])[context.context_id] = {
                "measured_path": measured_path.as_posix(),
                "measured_sha256": trace.checksum(),
                "measured_requests": len(trace.entries),
                "measured_span_seconds": _trace_span(trace),
                "quality_path": quality_path.as_posix(),
                "quality_sha256": quality.checksum(),
                "quality_seed": quality.seed,
                "quality_markers": markers,
            }
        catalog_path = self.store.root / "traces" / "catalog.json"
        if catalog_path.exists():
            if _read_json(catalog_path) != catalog:
                raise ValueError("M2 resume trace catalog mismatch")
        else:
            self.store.write_json("traces/catalog.json", catalog)
        return measured, qualities

    @staticmethod
    def _logical_trial_id(
        profile: M2FP8Profile,
        context: M2FP8Context,
        repeat_index: int,
    ) -> str:
        return f"fp8-{profile.profile_id}-{context.context_id}-repeat-{repeat_index}"

    def _attempts(self, logical_id: str) -> list[tuple[int, Path]]:
        if not self.store.trials_dir.is_dir():
            return []
        attempts: list[tuple[int, Path]] = []
        for path in self.store.trials_dir.iterdir():
            if path.is_symlink() or not path.is_dir():
                raise ValueError(f"unexpected M2 trials entry: {path}")
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
        profile: M2FP8Profile,
        context: M2FP8Context,
        repeat_index: int,
        trace: WorkloadTrace,
        quality: WorkloadTrace,
    ) -> dict[str, Any]:
        return {
            "schema_version": M2_FP8_SCHEMA,
            "trial_id": trial_id,
            "profile_id": profile.profile_id,
            "context_id": context.context_id,
            "context_tokens": context.total_kv_tokens,
            "load_id": context.load_id,
            "offered_requests_per_second": context.offered_requests_per_second,
            "repeat_index": repeat_index,
            "trace_id": trace.checksum(),
            "quality_trace_id": quality.checksum(),
            "vllm_args": profile.vllm_args(),
            "m1_boundary_source": self.config.m1_boundaries.experiment_id,
        }

    def _cached_complete(
        self,
        *,
        logical_id: str,
        profile: M2FP8Profile,
        context: M2FP8Context,
        repeat_index: int,
        trace: WorkloadTrace,
        quality: WorkloadTrace,
    ) -> Optional[tuple[TrialResult, M2FP8TrialRecord]]:
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
                        QUALITY_TRACE_FILE,
                        QUALITY_TRACE_CHECKSUM_FILE,
                    },
                )
                record = _RECORD_ADAPTER.validate_json(
                    (path / RECORD_FILE).read_text(encoding="utf-8")
                )
                point = _read_json(path / POINT_FILE)
                expected = self._point_payload(
                    trial_id=result.trial_id,
                    profile=profile,
                    context=context,
                    repeat_index=repeat_index,
                    trace=trace,
                    quality=quality,
                )
                evidence = _read_json(path / EVIDENCE_FILE)
                if point != expected or evidence.get("semantic_gate_passed") is not True:
                    raise ValueError("sealed M2 attempt identity or semantic gate mismatch")
                if (
                    record.profile_id != profile.profile_id
                    or record.context_id != context.context_id
                    or record.repeat_index != repeat_index
                    or record.trace_id != trace.checksum()
                ):
                    raise ValueError("sealed M2 record identity mismatch")
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
        trial_id: str,
        profile: M2FP8Profile,
        context: M2FP8Context,
        repeat_index: int,
        trace: WorkloadTrace,
        quality: WorkloadTrace,
        memory: DeviceMemoryEvidence,
    ) -> None:
        base = Path("trials") / trial_id
        self.store.write_json(
            base / POINT_FILE,
            self._point_payload(
                trial_id=trial_id,
                profile=profile,
                context=context,
                repeat_index=repeat_index,
                trace=trace,
                quality=quality,
            ),
        )
        self.store.write_text(base / MEASURED_TRACE_FILE, _trace_text(trace))
        self.store.write_text(
            base / MEASURED_TRACE_CHECKSUM_FILE,
            f"{trace.checksum()}  {MEASURED_TRACE_FILE}\n",
        )
        self.store.write_text(base / QUALITY_TRACE_FILE, _trace_text(quality))
        self.store.write_text(
            base / QUALITY_TRACE_CHECKSUM_FILE,
            f"{quality.checksum()}  {QUALITY_TRACE_FILE}\n",
        )
        self.store.write_json(base / CUDA_MEMORY_FILE, memory)
        self.store.write_json(
            base / EVIDENCE_FILE,
            {
                "schema_version": M2_FP8_SCHEMA,
                "semantic_gate_passed": False,
                "state": "pending-controller-result",
            },
        )

    def _finalize_trial(
        self,
        result: TrialResult,
        record: M2FP8TrialRecord | Mapping[str, Any],
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
                else {"available": False, "reason": "M2 semantic gate did not produce capacity"}
            ),
        )
        self.store.ensure_trial_artifacts(result)
        self.store.validate_cached_trial(result, require_telemetry=True)
        self.store.validate_trial_integrity(result.trial_id)

    async def _run_one(
        self,
        *,
        profile: M2FP8Profile,
        context: M2FP8Context,
        repeat_index: int,
        trace: WorkloadTrace,
        quality: WorkloadTrace,
        quality_markers: Mapping[str, str],
    ) -> tuple[TrialResult, Optional[M2FP8TrialRecord], bool]:
        logical_id = self._logical_trial_id(profile, context, repeat_index)
        cached = self._cached_complete(
            logical_id=logical_id,
            profile=profile,
            context=context,
            repeat_index=repeat_index,
            trace=trace,
            quality=quality,
        )
        if cached is not None:
            return cached[0], cached[1], True
        trial_id = self._select_new_attempt(logical_id)
        tuning = self.config.to_tuning_config(profile, context)
        if tuning.vllm_args != profile.vllm_args():
            raise ValueError("M2 TuningConfig changed preregistered profile arguments")
        memory = self.gpu_memory_reader(self.config.gpu.device_ids[0])
        self._write_pretrial(
            trial_id=trial_id,
            profile=profile,
            context=context,
            repeat_index=repeat_index,
            trace=trace,
            quality=quality,
            memory=memory,
        )
        controller = self.controller_factory(
            tuning,
            trace,
            self.store,
            tokenizer=self._load_tokenizer(),
            warmup_trace=quality,
            strict_open_loop=True,
        )
        try:
            result = await controller.run_trial({}, trial_id, "fp8")
        except UnsafeCleanupError:
            self._status(
                "unsafe_cleanup",
                current_trial=trial_id,
                unsafe_cleanup=True,
                message="unsafe cleanup; M2 stopped before another GPU process",
            )
            raise
        expected_provenance = trial_provenance(trial_id, "fp8")
        if any(getattr(result, name) != value for name, value in expected_provenance.items()):
            raise ValueError(f"M2 trial provenance mismatch: {trial_id}")
        if result.params != {}:
            raise ValueError(f"M2 fixed profile trial passed search parameters: {trial_id}")

        record: Optional[M2FP8TrialRecord] = None
        evidence: dict[str, Any]
        if result.status in {TrialStatus.COMPLETE, TrialStatus.INFEASIBLE}:
            try:
                record, evidence = _derive_record(
                    result=result,
                    trial_dir=self.store.trials_dir / trial_id,
                    profile=profile,
                    context=context,
                    repeat_index=repeat_index,
                    trace=trace,
                    quality=quality,
                    quality_markers=quality_markers,
                    device_memory=memory,
                    checkpoint_scale_keys=self._checkpoint_scale_keys,
                )
            except (OSError, ValueError) as error:
                result.status = TrialStatus.FAILED
                result.constraints = {
                    **result.constraints,
                    "feasible": False,
                    "violations": [
                        *cast(list[str], result.constraints.get("violations", [])),
                        "m2_fp8_semantic_gate",
                    ],
                }
                result.failure_reason = {
                    "type": "M2_FP8_SEMANTIC_GATE",
                    "message": str(error),
                    "phase": "M2_FP8_FINALIZE",
                }
                self.store.record_artifact_finalizer_failure(result, str(error))
                evidence = {
                    "schema_version": M2_FP8_SCHEMA,
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
                "schema_version": M2_FP8_SCHEMA,
                "semantic_gate_passed": False,
                "failure": reason,
            }
        failure_record = {
            "schema_version": M2_FP8_SCHEMA,
            "trial_id": trial_id,
            "profile_id": profile.profile_id,
            "context_id": context.context_id,
            "repeat_index": repeat_index,
            "status": "failed",
            "failure": evidence.get("failure"),
        }
        self._finalize_trial(result, record or failure_record, evidence)
        return result, record, False

    def _expected_matrix(self) -> set[tuple[str, str, int]]:
        return {
            (profile.profile_id, context.context_id, repeat)
            for context in self.config.contexts
            for repeat in range(self.config.protocol.repeats)
            for profile in self.config.profiles
        }

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
            eta_seconds = sum(
                _trace_span(
                    _scaled_trace(
                        context=context,
                        count=self.config.protocol.measured_request_count(context),
                        seed=self.config.protocol.measurement_seed,
                        tokenizer=self._load_tokenizer(),
                        burstiness=self.config.protocol.burstiness,
                    )
                )
                for context in self.config.contexts
                for _ in range(self.config.protocol.repeats * len(self.config.profiles))
            )
        value = {
            "schema_version": M2_FP8_SCHEMA,
            "experiment_id": self.experiment_id,
            "state": state,
            "pid": os.getpid(),
            "gpu": list(self.config.gpu.device_ids),
            "log": str(self.store.root / RUNNER_LOG_FILE),
            "result": str(self.store.root),
            "eta": (datetime.now(timezone.utc) + timedelta(seconds=eta_seconds)).isoformat(),
            "eta_seconds": eta_seconds,
            "resume": self._resume_command(),
            "sealed": (self.store.root / M2_FP8_INTEGRITY_FILE).is_file(),
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
            "scripts/run_longctx_m2_fp8.sh --config CONFIG "
            f"--experiment-id {self.experiment_id} --resume"
        )

    @staticmethod
    def _capacity_gain_checks(records: Sequence[M2FP8TrialRecord]) -> dict[str, bool]:
        by_key = {
            (record.profile_id, record.context_id, record.repeat_index): record
            for record in records
        }
        checks: dict[str, bool] = {}
        contexts = sorted({record.context_id for record in records})
        for context_id in contexts:
            baseline = [
                record
                for record in records
                if record.profile_id == "bf16-auto" and record.context_id == context_id
            ]
            for candidate_id in ("fp8-dynamic", "fp8-unit-fallback"):
                candidates = [
                    record
                    for record in records
                    if record.profile_id == candidate_id and record.context_id == context_id
                ]
                if not candidates:
                    continue
                checks[f"{context_id}:{candidate_id}"] = (
                    bool(baseline)
                    and all(
                        by_key[(candidate_id, context_id, candidate.repeat_index)].cached_tokens
                        > by_key[("bf16-auto", context_id, candidate.repeat_index)].cached_tokens
                        for candidate in candidates
                        if ("bf16-auto", context_id, candidate.repeat_index) in by_key
                    )
                    and len(candidates) == len(baseline)
                )
        return checks

    @staticmethod
    def _report(summary: Mapping[str, Any]) -> str:
        execution = cast(Mapping[str, Any], summary["execution"])
        acceptance = cast(Mapping[str, Any], summary["acceptance"])
        analysis = cast(Mapping[str, Any], summary["analysis"])

        def format_delta(row: Mapping[str, Any], name: str) -> str:
            metric = row.get(name)
            value = metric.get("median") if isinstance(metric, Mapping) else None
            return "n/a" if not isinstance(value, (int, float)) else f"{float(value):+.2f}%"

        lines = [
            "# Long-context v5 M2 FP8 KV Cache",
            "",
            f"- Experiment: {summary['experiment_id']}",
            f"- Evidence role: {summary['evidence_role']}",
            f"- Execution passed: {execution['passed']}",
            f"- M2 accepted: {acceptance['passed']}",
            "- M1 thresholds modified: false",
            "- M3 started: false",
            "",
            "## FP8 vs BF16 paired results",
            "",
            "| Context | Cached-token ratio | Goodput delta | TTFT p50 delta | TPOT p50 delta |",
            "|---|---:|---:|---:|---:|",
        ]
        paired = analysis.get("paired_fp8_vs_bf16", [])
        if isinstance(paired, list):
            for row in paired:
                if not isinstance(row, Mapping):
                    continue
                lines.append(
                    "| {context} | {capacity:.3f}x | {goodput} | "
                    "{ttft} | {tpot} |".format(
                        context=row.get("context_id"),
                        capacity=float(
                            cast(Mapping[str, Any], row["cached_tokens_ratio"])["median"]
                        ),
                        goodput=format_delta(
                            cast(Mapping[str, Any], row), "goodput_change_percent"
                        ),
                        ttft=format_delta(cast(Mapping[str, Any], row), "ttft_p50_change_percent"),
                        tpot=format_delta(cast(Mapping[str, Any], row), "tpot_p50_change_percent"),
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
        smoke_identity_matches = self._validate_formal_smoke_identity(source_commit)
        if not smoke_identity_matches:
            raise ValueError("formal M2 source/model/runtime/M1 identity differs from its smoke")
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
        measured, qualities = self._trace_catalog()
        records: list[M2FP8TrialRecord] = []
        results: list[TrialResult] = []
        completed_jobs = 0
        for context in self.config.contexts:
            trace = measured[context.context_id]
            quality, markers = qualities[context.context_id]
            for repeat_index in range(self.config.protocol.repeats):
                for profile in self.config.profiles:
                    logical = self._logical_trial_id(profile, context, repeat_index)
                    self._status("running", completed_jobs=completed_jobs, current_trial=logical)
                    result, record, _ = await self._run_one(
                        profile=profile,
                        context=context,
                        repeat_index=repeat_index,
                        trace=trace,
                        quality=quality,
                        quality_markers=markers,
                    )
                    results.append(result)
                    if record is not None:
                        records.append(record)
                    completed_jobs += 1

        expected = self._expected_matrix()
        observed = {
            (record.profile_id, record.context_id, record.repeat_index) for record in records
        }
        execution_passed = (
            len(results) == len(expected) and len(records) == len(expected) and observed == expected
        )
        analysis = (
            analyze_m2_fp8_records(records)
            if records
            else {
                "schema_version": "longctx-m2-fp8-analysis.v1",
                "record_count": 0,
                "groups": [],
                "paired_fp8_vs_bf16": [],
                "single_run_selection_used": False,
            }
        )
        capacity_checks = self._capacity_gain_checks(records)
        source_commit_after, source_tree_after = self._source_identity()
        source_stable = source_commit_after == source_commit and source_tree_after == source_tree
        profile_ids = {record.profile_id for record in records}
        scale_checks = {
            "dynamic_scale_observed": "fp8-dynamic" in profile_ids
            and all(
                record.calculate_kv_scales and record.scale_source == "dynamic-first-forward"
                for record in records
                if record.profile_id == "fp8-dynamic"
            ),
            "unit_fallback_classified": (
                "fp8-unit-fallback" in profile_ids
                and all(
                    not record.calculate_kv_scales and record.scale_source == "unit-fallback"
                    for record in records
                    if record.profile_id == "fp8-unit-fallback"
                )
            ),
        }
        if self.config.evidence_role == "formal" and self.config.smoke_artifact is not None:
            smoke_summary = self.config.smoke_artifact.summary()
            smoke_checks = _read_json(self.config.smoke_artifact.root / SUMMARY_FILE).get(
                "acceptance", {}
            )
            scale_checks["unit_fallback_classified"] = (
                isinstance(smoke_checks, Mapping)
                and cast(Mapping[str, Any], smoke_checks)
                .get("checks", {})
                .get("unit_fallback_classified")
                is True
            )
            bound_smoke_passed = (
                isinstance(smoke_summary.get("acceptance"), Mapping)
                and cast(Mapping[str, Any], smoke_summary["acceptance"]).get("passed") is True
            )
        else:
            bound_smoke_passed = self.config.evidence_role == "smoke"

        checks = {
            "project_line_is_v5": self.config.project_line == "longctx-v5",
            "m1_boundaries_reused": True,
            "m1_numeric_thresholds_unchanged": True,
            "bound_smoke_passed": bound_smoke_passed,
            "smoke_identity_matches": smoke_identity_matches,
            "all_jobs_complete": execution_passed,
            "quality_all_passed": bool(records)
            and all(record.quality_passed for record in records),
            "checkpoint_has_no_scale_keys": not self._checkpoint_scale_keys,
            "dynamic_scale_observed": scale_checks["dynamic_scale_observed"],
            "unit_fallback_classified": scale_checks["unit_fallback_classified"],
            "fp8_backend_fallback_explicit": bool(records)
            and all(
                record.attention_backend == "FLASHINFER"
                and record.backend_resolution == "automatic-fp8-fallback"
                for record in records
                if record.profile_id.startswith("fp8-")
            ),
            "bf16_production_backend_preserved": bool(records)
            and all(
                record.attention_backend == "FLASH_ATTN"
                and record.backend_resolution == "production-default"
                for record in records
                if record.profile_id == "bf16-auto"
            ),
            "fp8_capacity_increased": bool(capacity_checks) and all(capacity_checks.values()),
            "runtime_lock_verified": runtime.matches_lock,
            "model_lock_verified": model.matches_lock,
            "source_identity_stable": source_stable,
            "single_gpu_only": self.config.gpu.count == 1 and self.config.gpu.device_ids == (0,),
            "formal_matrix_is_18_runs": (
                len(expected) == 18 if self.config.evidence_role == "formal" else True
            ),
            "no_m3_work": True,
        }
        eligible = self.config.evidence_role == "formal"
        acceptance_passed = all(checks.values())
        failure_reasons = sorted(name for name, passed in checks.items() if not passed)
        acceptance = {
            "eligible": eligible,
            "passed": acceptance_passed,
            "checks": checks,
            "capacity_checks": capacity_checks,
            "failure_reasons": failure_reasons,
        }
        execution = {
            "passed": execution_passed,
            "planned_jobs": len(expected),
            "completed_jobs": len(records),
            "failed_jobs": len(expected) - len(records),
            "unsafe_cleanup": False,
        }
        summary: dict[str, Any] = {
            "schema_version": M2_FP8_SCHEMA,
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
            "scale_boundary": {
                "checkpoint_scale_keys": self._checkpoint_scale_keys,
                "formal_scale_source": "dynamic-first-forward",
                "fallback_scale_source": "unit-fallback-smoke-only",
                "unit_fallback_used_in_formal": False,
            },
            "backend_boundary": {
                "bf16": "FLASH_ATTN production default",
                "fp8": "automatic FLASHINFER fallback on RTX 5090",
                "attention_backend_forced": False,
                "latency_comparison_backend_confounded": True,
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
                "integrity": str(self.store.root / M2_FP8_INTEGRITY_FILE),
                "m1_boundaries": str(self.config.m1_boundaries.root),
            },
            "m1_numeric_thresholds_modified": False,
            "m3_started": False,
        }
        self.store.write_json(SUMMARY_FILE, summary)
        self.store.write_text(REPORT_FILE, self._report(summary))
        final_state = "accepted" if acceptance_passed else "completed_not_accepted"
        self._status(final_state, completed_jobs=len(results), acceptance=acceptance)
        seal_m2_fp8_artifacts(
            self.store.root,
            self.experiment_id,
            {
                "experiment_id": self.experiment_id,
                "project_line": "longctx-v5",
                "milestone": "M2",
                "experiment_kind": "fp8-kv-cache",
                "evidence_role": self.config.evidence_role,
                "source_commit": source_commit,
                "accepted": acceptance_passed,
                "m3_started": False,
            },
        )
        return summary


def load_m2_fp8_status(root: str | Path, experiment_id: str) -> dict[str, Any]:
    """Load an M2 smoke/formal status without importing a model or starting vLLM."""
    if _EXPERIMENT_ID.fullmatch(experiment_id) is None:
        raise ValueError("experiment_id must be one portable path component")
    experiment_root = Path(root).expanduser().resolve() / experiment_id
    sealed = (experiment_root / M2_FP8_INTEGRITY_FILE).is_file()
    if sealed:
        validate_m2_fp8_artifacts(experiment_root)
    status = _read_json(experiment_root / STATUS_FILE)
    if status.get("experiment_id") != experiment_id:
        raise ValueError("M2 FP8 status experiment identity mismatch")
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


__all__ = ["LongContextM2FP8Runner", "load_m2_fp8_status"]
