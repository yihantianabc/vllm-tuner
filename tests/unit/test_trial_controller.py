"""Integration-style trial lifecycle tests with fake server and benchmark."""

import asyncio
import json
from pathlib import Path

import pytest

from vllm_tuner.benchmarks.models import BenchmarkResult, RequestResult, RequestStatus
from vllm_tuner.config.models import (
    Constraints,
    TelemetryConfig,
    TuningConfig,
    WorkloadConfig,
)
from vllm_tuner.experiment.artifacts import ArtifactStore
from vllm_tuner.experiment.models import TrialStatus
from vllm_tuner.runtime.controller import TrialController
from vllm_tuner.runtime.failures import UnsafeCleanupError
from vllm_tuner.runtime.server import ServerStatus
from vllm_tuner.workloads.trace import TraceEntry, WorkloadTrace


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        return text.split()


class FakeServer:
    def __init__(self, config, *, trial_dir):
        self.base_url = "http://127.0.0.1:8123"
        self.trial_dir = Path(trial_dir)
        self.log_path = self.trial_dir / "server.log"
        self.command_path = self.trial_dir / "server-command.json"
        self.running = False
        self.ready = False
        self.failure_reason = None
        self.cleanup_status = None
        self.stop_calls = 0

    async def start(self, params):
        self.running = True
        self.log_path.write_text("ready\n", encoding="utf-8")
        self.command_path.write_text('{"argv":["fake"]}\n', encoding="utf-8")

    async def wait_ready(self):
        self.ready = True
        return True

    def status(self):
        return ServerStatus(
            pid=1,
            process_group_id=1,
            running=self.running,
            ready=self.ready and self.running,
            port=8123,
            checked_monotonic_ns=1,
        )

    async def stop(self):
        self.stop_calls += 1
        self.running = False
        self.ready = False
        self.cleanup_status = {
            "attempted": True,
            "clean": True,
            "pid": 1,
            "process_group_id": 1,
            "term_sent": True,
            "term_sent_monotonic_ns": 2,
            "kill_sent": False,
            "kill_sent_monotonic_ns": None,
            "leader_exit_code": 0,
            "process_group_empty": True,
            "process_group_pids_before": [1],
            "process_group_pids_after": [],
            "gpu_check_available": True,
            "compute_pids_baseline": [],
            "compute_pids_before": [1],
            "compute_pids_after": [],
            "tracked_compute_pids_after": [],
            "gpu_clean": True,
            "port_available": True,
            "errors": [],
            "checked_monotonic_ns": 3,
        }
        return self.cleanup_status

    def is_running(self):
        return self.running

    def log_tail(self):
        return self.log_path.read_text() if self.log_path.exists() else ""


class FakeOfficialAdapter:
    async def run(self, config, *, slo=None):
        request = RequestResult(
            request_id="chat-000000",
            sent_at=1_000_000_000,
            first_token_at=1_010_000_000,
            finished_at=1_040_000_000,
            input_tokens=10,
            output_tokens=4,
            token_timestamps=[
                1_010_000_000,
                1_020_000_000,
                1_030_000_000,
                1_040_000_000,
            ],
            status=RequestStatus.SUCCESS,
        )
        return BenchmarkResult(
            backend="fake-official",
            started_at=1_000_000_000,
            finished_at=2_000_000_000,
            request_results=[request],
            aggregate={"duration": 1.0, "p99_ttft_ms": 10.0},
        )


class CleanupFailingServer(FakeServer):
    async def stop(self):
        self.stop_calls += 1
        self.running = False
        self.ready = False
        raise RuntimeError("injected cleanup failure")


class BlockingOfficialAdapter:
    def __init__(self) -> None:
        self.entered = asyncio.Event()

    async def run(self, config, *, slo=None):
        self.entered.set()
        await asyncio.Event().wait()


def make_controller(
    tmp_path,
    *,
    server_factory=FakeServer,
    official_adapter=None,
) -> tuple[TrialController, ArtifactStore]:
    config = TuningConfig(
        model="fake-model",
        workload=WorkloadConfig(
            dataset_name="unused",
            sample_size=1,
            warmup_requests=0,
            benchmark_backend="official",
        ),
        telemetry=TelemetryConfig(enabled=False),
        constraints=Constraints(
            max_peak_vram_mb=None,
            max_memory_utilization=None,
        ),
    )
    trace = WorkloadTrace(
        seed=1,
        profile="chat",
        entries=[
            TraceEntry(
                request_id="chat-000000",
                scheduled_offset_seconds=0,
                prompt="hello",
                input_tokens=10,
                output_tokens=4,
                profile="chat",
            )
        ],
    )
    store = ArtifactStore(tmp_path, "exp")
    store.initialize()
    controller = TrialController(
        config,
        trace,
        store,
        tokenizer=FakeTokenizer(),
        server_factory=server_factory,
        official_adapter=official_adapter or FakeOfficialAdapter(),
    )
    return controller, store


@pytest.mark.asyncio
async def test_trial_controller_writes_complete_evidence(tmp_path) -> None:
    controller, store = make_controller(tmp_path)

    result = await controller.run_trial(
        {"tensor_parallel_size": 1, "pipeline_parallel_size": 1},
        "trial-0",
        "default",
    )

    assert result.status is TrialStatus.COMPLETE
    assert result.selectable is True
    assert result.client["goodput_requests_per_sec"] == 1.0
    assert result.cleanup_status is not None
    assert result.cleanup_status["clean"] is True
    cleanup = json.loads((store.trial_dir("trial-0") / "cleanup.json").read_text(encoding="utf-8"))
    assert cleanup == result.cleanup_status
    store.validate_trial_artifacts("trial-0", require_telemetry=True)


@pytest.mark.asyncio
async def test_cleanup_failure_is_persisted_and_aborts_the_search(tmp_path) -> None:
    controller, store = make_controller(tmp_path, server_factory=CleanupFailingServer)

    with pytest.raises(UnsafeCleanupError) as raised:
        await controller.run_trial({}, "cleanup-failure", "default")

    assert raised.value.result is not None
    assert raised.value.result.status is TrialStatus.FAILED
    summary = json.loads(
        (store.trial_dir("cleanup-failure") / "summary.json").read_text(encoding="utf-8")
    )
    status = json.loads(
        (store.trial_dir("cleanup-failure") / "status.json").read_text(encoding="utf-8")
    )
    cleanup = json.loads(
        (store.trial_dir("cleanup-failure") / "cleanup.json").read_text(encoding="utf-8")
    )
    assert summary["status"] == "FAILED"
    assert summary["failure_reason"]["type"] == "CLEANUP_ERROR"
    assert status["status"] == "FAILED"
    assert cleanup["clean"] is False
    assert summary["cleanup_status"] == cleanup
    assert (store.trial_dir("cleanup-failure") / "artifact-status.json").is_file()
    assert (store.trial_dir("cleanup-failure") / "artifact-integrity.json").is_file()
    store.validate_trial_artifacts("cleanup-failure")


@pytest.mark.asyncio
async def test_cancellation_persists_terminal_result_after_cleanup(tmp_path) -> None:
    adapter = BlockingOfficialAdapter()
    controller, store = make_controller(tmp_path, official_adapter=adapter)
    task = asyncio.create_task(controller.run_trial({}, "cancelled", "default"))
    await asyncio.wait_for(adapter.entered.wait(), timeout=1)

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    summary = json.loads(
        (store.trial_dir("cancelled") / "summary.json").read_text(encoding="utf-8")
    )
    status = json.loads((store.trial_dir("cancelled") / "status.json").read_text(encoding="utf-8"))
    cleanup = json.loads(
        (store.trial_dir("cancelled") / "cleanup.json").read_text(encoding="utf-8")
    )
    assert summary["status"] == "FAILED"
    assert summary["failure_reason"]["type"] == "CANCELLED"
    assert summary["finished_at"] is not None
    assert status["status"] == "FAILED"
    assert status["history"][-1]["current"] == "FAILED"
    assert cleanup["clean"] is True
    assert summary["cleanup_status"] == cleanup
    assert (store.trial_dir("cancelled") / "artifact-status.json").is_file()
    assert (store.trial_dir("cancelled") / "artifact-integrity.json").is_file()
    artifact_status = json.loads(
        (store.trial_dir("cancelled") / "artifact-status.json").read_text(encoding="utf-8")
    )
    assert artifact_status["files"]["request-results.jsonl"]["data_available"] is False
    assert artifact_status["files"]["benchmark-raw.json"]["data_available"] is False
    store.validate_trial_artifacts("cancelled")


def test_collect_energy_policy_is_not_a_noop() -> None:
    gpu = {
        "energy_joules": 12.0,
        "energy_per_output_token_joules": 0.5,
        "devices": {
            "0": {
                "energy_joules": 12.0,
                "energy_per_output_token_joules": 0.5,
            }
        },
    }

    disabled = TrialController._apply_energy_policy(gpu, False)
    enabled = TrialController._apply_energy_policy(gpu, True)

    assert disabled["energy_collection_enabled"] is False
    assert disabled["energy_joules"] is None
    assert disabled["energy_per_output_token_joules"] is None
    assert disabled["devices"]["0"]["energy_joules"] is None
    assert disabled["devices"]["0"]["energy_per_output_token_joules"] is None
    assert enabled["energy_collection_enabled"] is True
    assert enabled["energy_joules"] == 12.0
    assert enabled["devices"]["0"]["energy_collection_enabled"] is True
