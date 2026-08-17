"""End-to-end trial controller aligned to the documented lifecycle."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Awaitable, Optional

import httpx

from vllm_tuner.benchmarks.metrics import aggregate_request_results
from vllm_tuner.benchmarks.models import (
    BenchmarkResult,
    RequestResult,
    RequestSpec,
    RequestStatus,
    SLOThresholds,
)
from vllm_tuner.benchmarks.sse_client import SSEBenchmarkClient
from vllm_tuner.benchmarks.vllm_bench import VLLMBenchAdapter, VLLMBenchConfig
from vllm_tuner.config.models import TuningConfig
from vllm_tuner.experiment.artifacts import ArtifactStore
from vllm_tuner.experiment.models import (
    TrialResult,
    TrialStatus,
    trial_provenance,
    utc_now_iso,
)
from vllm_tuner.profiling.session import TelemetrySession
from vllm_tuner.tuning.objective import compute_slo_goodput
from vllm_tuner.workloads.trace import WorkloadTrace

from .failures import FailureReason, FailureType, UnsafeCleanupError, classify_failure
from .server import ManagedVLLMServer
from .server import uses_slotune_scheduler
from .state_machine import TrialStateMachine

logger = logging.getLogger(__name__)


class TrialController:
    """Run one server configuration, preserving raw evidence through failures."""

    def __init__(
        self,
        config: TuningConfig,
        trace: WorkloadTrace,
        artifacts: ArtifactStore,
        *,
        tokenizer: Optional[Any] = None,
        server_factory: type[ManagedVLLMServer] = ManagedVLLMServer,
        official_adapter: Optional[VLLMBenchAdapter] = None,
    ) -> None:
        self.config = config
        self.trace = trace
        self.artifacts = artifacts
        self._tokenizer = tokenizer
        self.server_factory = server_factory
        self.official_adapter = official_adapter or VLLMBenchAdapter()

    def _load_tokenizer(self) -> Any:
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(
                self.config.tokenizer or self.config.model,
                revision=self.config.model_revision,
                local_files_only=Path(self.config.model).exists(),
            )
        return self._tokenizer

    def _request_specs(self) -> list[RequestSpec]:
        return [
            RequestSpec(
                request_id=entry.request_id,
                prompt=entry.prompt,
                model=self.config.model,
                max_tokens=entry.output_tokens,
                ignore_eos=self.config.workload.ignore_eos,
                input_tokens=entry.input_tokens,
            )
            for entry in self.trace.entries
        ]

    def _slo(self) -> SLOThresholds:
        return SLOThresholds(
            ttft_ms=self.config.slo.ttft_ms,
            tpot_ms=self.config.slo.tpot_ms,
            e2e_ms=self.config.slo.e2e_ms,
        )

    async def _warmup_sse(
        self, client: SSEBenchmarkClient, specs: list[RequestSpec]
    ) -> list[RequestResult]:
        warmup_results: list[RequestResult] = []
        if not specs:
            return warmup_results
        async with httpx.AsyncClient(
            timeout=self.config.workload.request_timeout_seconds,
            trust_env=False,
        ) as http_client:
            for index in range(self.config.workload.warmup_requests):
                source = specs[index % len(specs)]
                warmup = RequestSpec(
                    request_id=f"warmup-{index}-{source.request_id}",
                    prompt=source.prompt,
                    model=source.model,
                    max_tokens=source.max_tokens,
                    temperature=source.temperature,
                    top_p=source.top_p,
                    ignore_eos=source.ignore_eos,
                    input_tokens=source.input_tokens,
                    endpoint=source.endpoint,
                    extra_body=dict(source.extra_body),
                )
                warmup_results.append(await client.send_request(warmup, http_client, warmup=True))
        return warmup_results

    async def _run_fixed_trace_sse(
        self,
        client: SSEBenchmarkClient,
        specs: list[RequestSpec],
        warmup_results: list[RequestResult],
    ) -> BenchmarkResult:
        """Replay persisted arrival offsets instead of regenerating traffic per trial."""
        started_at = time.perf_counter_ns()
        semaphore = asyncio.Semaphore(self.config.workload.max_concurrency)
        async with httpx.AsyncClient(
            timeout=self.config.workload.request_timeout_seconds,
            trust_env=False,
        ) as http_client:

            async def execute(index: int, spec: RequestSpec) -> RequestResult:
                offset = self.trace.entries[index].scheduled_offset_seconds
                scheduled_at = started_at + int(offset * 1_000_000_000)
                delay = (scheduled_at - time.perf_counter_ns()) / 1_000_000_000
                if delay > 0:
                    await asyncio.sleep(delay)
                async with semaphore:
                    return await client.send_request(
                        spec,
                        http_client,
                        scheduled_at=scheduled_at,
                        warmup=False,
                    )

            outcomes = await asyncio.gather(
                *(execute(index, spec) for index, spec in enumerate(specs)),
                return_exceptions=True,
            )
        finished_at = time.perf_counter_ns()
        requests: list[RequestResult] = []
        for spec, outcome in zip(specs, outcomes):
            if isinstance(outcome, BaseException):
                requests.append(
                    RequestResult(
                        request_id=spec.request_id,
                        finished_at=finished_at,
                        input_tokens=spec.input_tokens or 0,
                        status=RequestStatus.FAILED,
                        error_type="task_exception",
                        error_message=f"{type(outcome).__name__}: {outcome}",
                    )
                )
            else:
                requests.append(outcome)
        aggregate = aggregate_request_results(
            requests,
            started_at=started_at,
            finished_at=finished_at,
            slo=self._slo(),
        )
        return BenchmarkResult(
            backend="sse-fixed-trace",
            started_at=started_at,
            finished_at=finished_at,
            request_results=requests,
            warmup_results=warmup_results,
            aggregate=aggregate,
        )

    def _official_dataset(self, trial_dir: Path) -> Path:
        """Materialize an exact ShareGPT-style prompt set for the official adapter."""
        path = trial_dir / "official-workload.json"
        rows = [
            {
                "id": entry.request_id,
                "conversations": [
                    {"from": "human", "value": entry.prompt},
                    {"from": "gpt", "value": "placeholder"},
                ],
            }
            for entry in self.trace.entries
        ]
        path.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
        return path

    async def _run_official(self, server: ManagedVLLMServer, trial_dir: Path) -> BenchmarkResult:
        dataset = self._official_dataset(trial_dir)
        config = VLLMBenchConfig(
            base_url=server.base_url,
            model=self.config.model,
            output_path=trial_dir / "benchmark-raw.json",
            num_prompts=len(self.trace.entries),
            dataset_name="sharegpt",
            dataset_path=dataset,
            request_rate=self.trace.request_rate or float("inf"),
            burstiness=self.trace.burstiness,
            max_concurrency=self.config.workload.max_concurrency,
            output_len=self.config.workload.fixed_output_tokens,
            ignore_eos=self.config.workload.ignore_eos,
            seed=self.trace.seed,
            warmup_requests=0,
            timeout_s=self.config.workload.request_timeout_seconds
            * max(1, len(self.trace.entries)),
        )
        return await self.official_adapter.run(config, slo=self._slo())

    @staticmethod
    def _gpu_constraint_view(gpu: dict[str, Any]) -> dict[str, Any]:
        result = dict(gpu)
        peak = gpu.get("peak_memory_mb")
        total_summary = gpu.get("memory_total_mb", {})
        total = total_summary.get("peak") if isinstance(total_summary, dict) else None
        if peak is not None and total:
            result["peak_memory_utilization"] = float(peak) / float(total)
        return result

    @staticmethod
    def _apply_energy_policy(gpu: dict[str, Any], enabled: bool) -> dict[str, Any]:
        """Make the collect_energy setting explicit in aggregate and device data."""
        result = dict(gpu)
        result["energy_collection_enabled"] = enabled
        devices = result.get("devices")
        if isinstance(devices, dict):
            result["devices"] = {
                str(device_id): (
                    {
                        **device,
                        "energy_collection_enabled": enabled,
                        **(
                            {}
                            if enabled
                            else {
                                "energy_joules": None,
                                "energy_per_output_token_joules": None,
                            }
                        ),
                    }
                    if isinstance(device, dict)
                    else device
                )
                for device_id, device in devices.items()
            }
        if not enabled:
            result["energy_joules"] = None
            result["energy_per_output_token_joules"] = None
        return result

    @staticmethod
    async def _shield_to_completion(
        awaitable: Awaitable[Any],
    ) -> tuple[Any, Optional[asyncio.CancelledError]]:
        """Finish cleanup despite caller cancellation and report cancellation later."""
        task = asyncio.ensure_future(awaitable)
        pending_cancellation: Optional[asyncio.CancelledError] = None
        while True:
            try:
                return await asyncio.shield(task), pending_cancellation
            except asyncio.CancelledError as error:
                if task.cancelled():
                    raise
                if pending_cancellation is None:
                    pending_cancellation = error

    @staticmethod
    def _server_status(server: Any) -> dict[str, Any]:
        try:
            status = server.status()
            if hasattr(status, "model_dump"):
                return dict(status.model_dump(mode="json"))
            if isinstance(status, dict):
                return dict(status)
        except Exception:
            logger.error("Unable to inspect server status", exc_info=True)
        return {}

    @staticmethod
    def _cleanup_payload(server: Any, value: Any) -> dict[str, Any]:
        candidate = value if value is not None else getattr(server, "cleanup_status", None)
        if candidate is not None and hasattr(candidate, "model_dump"):
            payload = candidate.model_dump(mode="json")
        elif isinstance(candidate, dict):
            payload = dict(candidate)
        else:
            raise TypeError("server.stop() did not return structured cleanup status")
        if not isinstance(payload, dict):
            raise TypeError("server cleanup status must be a JSON object")
        return payload

    @classmethod
    def _failed_cleanup_payload(cls, server: Any, error: BaseException) -> dict[str, Any]:
        status = cls._server_status(server)
        return {
            "attempted": True,
            "clean": False,
            "pid": status.get("pid"),
            "process_group_id": status.get("process_group_id"),
            "term_sent": False,
            "term_sent_monotonic_ns": None,
            "kill_sent": False,
            "kill_sent_monotonic_ns": None,
            "leader_exit_code": status.get("exit_code"),
            "process_group_empty": False,
            "process_group_pids_before": [],
            "process_group_pids_after": [],
            "gpu_check_available": False,
            "compute_pids_baseline": [],
            "compute_pids_before": [],
            "compute_pids_after": [],
            "tracked_compute_pids_after": [],
            "gpu_clean": None,
            "port_available": False,
            "errors": [f"{type(error).__name__}: {error}"],
            "checked_monotonic_ns": time.perf_counter_ns(),
        }

    def _write_raw_artifacts(
        self,
        trial_id: str,
        benchmark: Optional[BenchmarkResult],
        telemetry: Optional[TelemetrySession],
    ) -> None:
        base = Path("trials") / trial_id
        request_results = benchmark.request_results if benchmark is not None else []
        request_rows = [result.to_dict() for result in request_results]
        if not request_rows:
            request_rows = [
                {
                    "record_type": "availability",
                    "available": False,
                    "reason": (
                        "benchmark was not completed"
                        if benchmark is None
                        else "benchmark produced no request results"
                    ),
                }
            ]
        self.artifacts.write_jsonl(
            base / "request-results.jsonl",
            request_rows,
        )
        raw = (
            benchmark.to_dict()
            if benchmark is not None
            else {
                "record_type": "availability",
                "available": False,
                "reason": "benchmark was not completed",
            }
        )
        self.artifacts.write_json(base / "benchmark-raw.json", raw)
        engine_rows = (
            [snapshot.to_dict(include_raw=True) for snapshot in telemetry.engine_snapshots]
            if telemetry is not None
            else []
        )
        gpu_rows = (
            [sample.to_dict() for sample in telemetry.gpu_samples] if telemetry is not None else []
        )
        if not engine_rows:
            engine_rows = [
                {
                    "record_type": "availability",
                    "available": False,
                    "reason": "Prometheus telemetry produced no snapshots",
                }
            ]
        if not gpu_rows:
            gpu_rows = [
                {
                    "record_type": "availability",
                    "available": False,
                    "reason": "NVML telemetry produced no samples",
                }
            ]
        self.artifacts.write_jsonl(base / "prometheus.jsonl", engine_rows)
        self.artifacts.write_jsonl(base / "nvml.jsonl", gpu_rows)
        decision_path = self.artifacts.trial_dir(trial_id) / "scheduler-decisions.jsonl"
        if (
            uses_slotune_scheduler(self.config)
            and self.config.adaptive_prefill.decision_log_enabled
            and not decision_path.exists()
        ):
            self.artifacts.write_jsonl(
                base / "scheduler-decisions.jsonl",
                [
                    {
                        "record_type": "availability",
                        "available": False,
                        "reason": "custom Scheduler produced no decision log",
                    }
                ],
            )

    async def run_trial(self, params: dict[str, Any], trial_id: str, method: str) -> TrialResult:
        """Execute START through terminal status, always cleaning the process group."""
        provenance = trial_provenance(trial_id, method)
        trial_dir = self.artifacts.trial_dir(trial_id)
        self.artifacts.write_json(Path("trials") / trial_id / "params.json", params)

        def save_transition(_: Any) -> None:
            self.artifacts.write_json(Path("trials") / trial_id / "status.json", machine.as_dict())

        machine = TrialStateMachine(on_transition=save_transition)
        server = self.server_factory(self.config, trial_dir=trial_dir)
        telemetry: Optional[TelemetrySession] = None
        benchmark: Optional[BenchmarkResult] = None
        telemetry_result: dict[str, Any] = {
            "client": {"available": False},
            "engine": {"available": False},
            "gpu": {"available": False},
            "window": {"available": False},
        }
        failure: Optional[FailureReason] = None
        started_at = utc_now_iso()
        result: Optional[TrialResult] = None
        cleanup_status: Optional[dict[str, Any]] = None
        cleanup_errors: list[str] = []
        pending_cancellation: Optional[asyncio.CancelledError] = None
        unsafe_cleanup = False

        try:
            machine.transition(TrialStatus.STARTING)
            await server.start(params)
            if not await server.wait_ready():
                reason = server.failure_reason or FailureReason(
                    type=FailureType.STARTUP_TIMEOUT,
                    message="vLLM did not become ready",
                    phase=TrialStatus.STARTING.value,
                )
                raise RuntimeError(reason.message)
            machine.transition(TrialStatus.READY)

            tokenizer = self._load_tokenizer()
            sse_client = SSEBenchmarkClient(
                server.base_url,
                self.config.model,
                timeout=self.config.workload.request_timeout_seconds,
                tokenizer=tokenizer,
                strict_token_count=True,
                require_token_ids=True,
            )
            specs = self._request_specs()
            machine.transition(TrialStatus.WARMING_UP)
            warmups = await self._warmup_sse(sse_client, specs)
            warmup_failures = [item for item in warmups if not item.success]
            if warmup_failures:
                raise RuntimeError(
                    f"{len(warmup_failures)}/{len(warmups)} warmup requests failed: "
                    f"{warmup_failures[0].error_message}"
                )

            machine.transition(TrialStatus.MEASURING)
            telemetry = TelemetrySession(
                prometheus_endpoint=f"{server.base_url}{self.config.telemetry.metrics_path}",
                device_ids=self.config.gpu.device_ids,
                sample_interval=self.config.telemetry.interval_ms / 1000.0,
                output_path=trial_dir / "telemetry.jsonl",
                enable_nvml=self.config.telemetry.collect_nvml,
            )
            if self.config.telemetry.enabled:
                await telemetry.start()
            if self.config.workload.benchmark_backend == "official":
                benchmark = await self._run_official(server, trial_dir)
            else:
                benchmark = await self._run_fixed_trace_sse(sse_client, specs, warmups)
            machine.transition(TrialStatus.COLLECTING)

            output_tokens = sum(item.output_tokens for item in benchmark.request_results)
            if self.config.telemetry.enabled:
                telemetry_result = await telemetry.stop(
                    client_metrics=benchmark.aggregate,
                    output_tokens=output_tokens,
                )
                telemetry_result["gpu"] = self._apply_energy_policy(
                    telemetry_result["gpu"],
                    self.config.telemetry.collect_energy,
                )
            measurement_seconds = benchmark.aggregate.get("duration")
            if not isinstance(measurement_seconds, (int, float)) or measurement_seconds <= 0:
                if benchmark.started_at is None or benchmark.finished_at is None:
                    raise ValueError("benchmark did not expose a valid measurement window")
                measurement_seconds = (benchmark.finished_at - benchmark.started_at) / 1_000_000_000

            server_status = server.status()
            gpu_metrics = self._gpu_constraint_view(telemetry_result["gpu"])
            objective = compute_slo_goodput(
                benchmark.request_results,
                measurement_seconds=float(measurement_seconds),
                offered_requests=len(self.trace.entries),
                offered_requests_per_second=self.trace.request_rate,
                slo=self.config.slo,
                constraints=self.config.constraints,
                engine=telemetry_result["engine"],
                gpu=gpu_metrics,
                server_alive=server_status.running,
            )
            client_metrics = dict(benchmark.aggregate)
            scheduled_span = (
                self.trace.entries[-1].scheduled_offset_seconds
                - self.trace.entries[0].scheduled_offset_seconds
            )
            empirical_scheduled_rate = (
                (len(self.trace.entries) - 1) / scheduled_span
                if len(self.trace.entries) > 1 and scheduled_span > 0
                else None
            )
            client_metrics.update(
                {
                    "goodput_requests_per_sec": objective.goodput_requests_per_sec,
                    "offered_requests_per_sec": objective.offered_requests_per_sec,
                    "target_offered_requests_per_sec": self.trace.request_rate,
                    "empirical_scheduled_requests_per_sec": empirical_scheduled_rate,
                    "achieved_requests_per_sec": objective.achieved_requests_per_sec,
                    "good_requests": objective.good_requests,
                    "total_input_tokens": objective.total_input_tokens,
                    "total_output_tokens": objective.total_output_tokens,
                    "request_slo": [asdict(item) for item in objective.request_slo],
                }
            )
            constraint_metrics = {
                "feasible": objective.constraints.feasible,
                "violations": list(objective.constraints.violations),
                "values": objective.constraints.values,
            }
            machine.transition(TrialStatus.STOPPING)
            last_status = self._server_status(server)
            try:
                cleanup_value, cleanup_cancellation = await self._shield_to_completion(
                    server.stop()
                )
                cleanup_status = self._cleanup_payload(server, cleanup_value)
            except asyncio.CancelledError:
                raise
            except Exception as cleanup_error:
                cleanup_errors.append(f"{type(cleanup_error).__name__}: {cleanup_error}")
                raise RuntimeError(f"Server cleanup failed: {cleanup_error}") from cleanup_error
            if cleanup_cancellation is not None:
                raise cleanup_cancellation
            if cleanup_status.get("clean") is not True:
                raise RuntimeError(
                    "Server cleanup could not be verified: "
                    + ", ".join(str(value) for value in cleanup_status.get("errors", []))
                )
            terminal = (
                TrialStatus.COMPLETE if objective.constraints.feasible else TrialStatus.INFEASIBLE
            )
            machine.transition(terminal, ",".join(objective.constraints.violations) or None)
            result = TrialResult(
                trial_id=trial_id,
                **provenance,
                status=terminal,
                params=params,
                started_at=started_at,
                finished_at=utc_now_iso(),
                measurement_seconds=float(measurement_seconds),
                client=client_metrics,
                engine=telemetry_result["engine"],
                gpu=gpu_metrics,
                constraints=constraint_metrics,
                last_server_status=last_status,
                cleanup_status=cleanup_status,
            )
        except asyncio.CancelledError as error:
            pending_cancellation = error
            failure = FailureReason(
                type=FailureType.CANCELLED,
                message="Trial execution was cancelled",
                phase=machine.status.value,
                retryable=True,
            )
            if not machine.status.terminal:
                machine.transition(TrialStatus.FAILED, failure.message)
            result = TrialResult(
                trial_id=trial_id,
                **provenance,
                status=TrialStatus.FAILED,
                params=params,
                started_at=started_at,
                client=benchmark.aggregate if benchmark is not None else {},
                engine=telemetry_result["engine"],
                gpu=self._gpu_constraint_view(telemetry_result["gpu"]),
                constraints={"feasible": False, "violations": [failure.type.value]},
                failure_reason=failure.model_dump(mode="json"),
                last_server_status=self._server_status(server),
                cleanup_status=cleanup_status,
            )
        except Exception as error:
            failure = server.failure_reason or classify_failure(
                error,
                log_text=server.log_tail(),
                phase=machine.status.value,
                exit_code=self._server_status(server).get("exit_code"),
            )
            if not machine.status.terminal:
                machine.transition(TrialStatus.FAILED, failure.message)
            result = TrialResult(
                trial_id=trial_id,
                **provenance,
                status=TrialStatus.FAILED,
                params=params,
                started_at=started_at,
                finished_at=utc_now_iso(),
                client=benchmark.aggregate if benchmark is not None else {},
                engine=telemetry_result["engine"],
                gpu=self._gpu_constraint_view(telemetry_result["gpu"]),
                constraints={"feasible": False, "violations": [failure.type.value]},
                failure_reason=failure.model_dump(mode="json"),
                last_server_status=self._server_status(server),
                cleanup_status=cleanup_status,
            )
        finally:
            if telemetry is not None and telemetry.running:
                try:
                    telemetry_value, telemetry_cancellation = await self._shield_to_completion(
                        telemetry.stop()
                    )
                    telemetry_result = telemetry_value
                    if telemetry_cancellation is not None and pending_cancellation is None:
                        pending_cancellation = telemetry_cancellation
                except asyncio.CancelledError as cleanup_cancellation:
                    if pending_cancellation is None:
                        pending_cancellation = cleanup_cancellation
                    cleanup_errors.append("Telemetry cleanup task was cancelled")
                except Exception as telemetry_error:
                    logger.error("Telemetry cleanup failed", exc_info=True)
                    cleanup_errors.append(
                        f"Telemetry cleanup failed: "
                        f"{type(telemetry_error).__name__}: {telemetry_error}"
                    )
                    if failure is None:
                        failure = FailureReason(
                            type=FailureType.TELEMETRY_ERROR,
                            message=str(telemetry_error),
                            phase=machine.status.value,
                            retryable=True,
                        )
            telemetry_result["gpu"] = self._apply_energy_policy(
                telemetry_result["gpu"],
                self.config.telemetry.collect_energy,
            )
            if cleanup_status is None or cleanup_status.get("clean") is not True:
                try:
                    (
                        cleanup_value,
                        deferred_cleanup_cancellation,
                    ) = await self._shield_to_completion(server.stop())
                    cleanup_status = self._cleanup_payload(server, cleanup_value)
                    if deferred_cleanup_cancellation is not None and pending_cancellation is None:
                        pending_cancellation = deferred_cleanup_cancellation
                except asyncio.CancelledError as server_cleanup_cancellation:
                    if pending_cancellation is None:
                        pending_cancellation = server_cleanup_cancellation
                    cleanup_errors.append("Server cleanup task was cancelled")
                    cleanup_status = self._failed_cleanup_payload(
                        server, server_cleanup_cancellation
                    )
                except Exception as cleanup_error:
                    logger.error("Server cleanup failed", exc_info=True)
                    cleanup_errors.append(
                        f"Server cleanup failed: {type(cleanup_error).__name__}: {cleanup_error}"
                    )
                    cleanup_status = self._failed_cleanup_payload(server, cleanup_error)

            if cleanup_status is None:
                missing_cleanup = RuntimeError("server cleanup status is unavailable")
                cleanup_errors.append(str(missing_cleanup))
                cleanup_status = self._failed_cleanup_payload(server, missing_cleanup)
            if cleanup_errors:
                recorded_errors = list(cleanup_status.get("errors", []))
                cleanup_status["errors"] = list(dict.fromkeys([*recorded_errors, *cleanup_errors]))

            assert result is not None
            result.engine = telemetry_result["engine"]
            result.gpu = self._gpu_constraint_view(telemetry_result["gpu"])
            result.cleanup_status = cleanup_status
            if cleanup_status.get("clean") is not True:
                unsafe_cleanup = True
                violations = list(result.constraints.get("violations", []))
                if "cleanup_error" not in violations:
                    violations.append("cleanup_error")
                result.constraints = {
                    **result.constraints,
                    "feasible": False,
                    "violations": violations,
                }
                cleanup_detail = {
                    "message": "Server cleanup could not be positively verified",
                    "status": cleanup_status,
                }
                if result.status in {TrialStatus.COMPLETE, TrialStatus.INFEASIBLE}:
                    result.status = TrialStatus.FAILED
                    result.failure_reason = {
                        "type": FailureType.CLEANUP_ERROR.value,
                        "message": cleanup_detail["message"],
                        "phase": TrialStatus.STOPPING.value,
                        "retryable": False,
                        "cleanup": cleanup_status,
                    }
                elif result.failure_reason is None:
                    result.failure_reason = {
                        "type": FailureType.CLEANUP_ERROR.value,
                        "message": cleanup_detail["message"],
                        "phase": machine.status.value,
                        "retryable": False,
                        "cleanup": cleanup_status,
                    }
                else:
                    result.failure_reason = {
                        **result.failure_reason,
                        "cleanup_error": cleanup_detail,
                    }
            result.finished_at = utc_now_iso()
            self.artifacts.write_json(
                Path("trials") / trial_id / "cleanup.json",
                cleanup_status,
            )
            self._write_raw_artifacts(trial_id, benchmark, telemetry)
            if not server.log_path.exists():
                server.log_path.write_text("server was not started\n", encoding="utf-8")
            if not server.command_path.exists():
                server.command_path.write_text(
                    json.dumps({"argv": None, "environment": {}}, indent=2) + "\n",
                    encoding="utf-8",
                )

        self.artifacts.save_trial_result(result)
        if pending_cancellation is not None or unsafe_cleanup:
            # These control-flow signals bypass the runner's normal finalizer.
            # Seal the terminal layout before re-raising so resume/audit sees a
            # complete, checksum-protected failure record.
            self.artifacts.ensure_trial_artifacts(result)
        if pending_cancellation is not None:
            raise pending_cancellation
        if unsafe_cleanup:
            raise UnsafeCleanupError(
                f"Trial {trial_id} cleanup could not be positively verified",
                result=result,
            )
        return result
