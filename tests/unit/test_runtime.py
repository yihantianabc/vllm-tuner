"""Trial transitions, failure taxonomy, and port lifecycle tests."""

import json
import signal
import socket

import pytest

import vllm_tuner.runtime.server as server_module
from vllm_tuner.config.models import TuningConfig
from vllm_tuner.experiment.models import TrialStatus
from vllm_tuner.runtime.failures import FailureType, classify_failure
from vllm_tuner.runtime.server import (
    ManagedVLLMServer,
    find_free_port,
    port_is_available,
)
from vllm_tuner.runtime.state_machine import InvalidTransitionError, TrialStateMachine


def test_state_machine_accepts_documented_lifecycle() -> None:
    machine = TrialStateMachine()
    for state in (
        TrialStatus.STARTING,
        TrialStatus.READY,
        TrialStatus.WARMING_UP,
        TrialStatus.MEASURING,
        TrialStatus.COLLECTING,
        TrialStatus.STOPPING,
        TrialStatus.COMPLETE,
    ):
        machine.transition(state)
    assert machine.status is TrialStatus.COMPLETE
    assert len(machine.history) == 7


def test_state_machine_rejects_skipped_phase() -> None:
    with pytest.raises(InvalidTransitionError):
        TrialStateMachine().transition(TrialStatus.MEASURING)


def test_runtime_error_is_not_automatically_oom() -> None:
    reason = classify_failure(RuntimeError("ordinary failure"), phase="MEASURING")
    assert reason.type is FailureType.REQUEST_ERROR


def test_explicit_oom_and_argument_errors_are_classified() -> None:
    assert classify_failure("torch.OutOfMemoryError: CUDA out of memory").type is FailureType.OOM
    assert (
        classify_failure("error: unrecognized arguments: --bad").type
        is FailureType.INVALID_ARGUMENT
    )


def test_port_availability_and_conflict(tmp_path) -> None:
    port = find_free_port()
    assert port_is_available("127.0.0.1", port)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", port))
        listener.listen()
        assert not port_is_available("127.0.0.1", port)


def test_server_command_fixes_parallelism_and_detects_conflict(tmp_path) -> None:
    server = ManagedVLLMServer(TuningConfig(model="test-model"), trial_dir=tmp_path)
    command = server.build_command({"max_num_seqs": 8})
    assert command[command.index("--tensor-parallel-size") + 1] == "1"
    with pytest.raises(ValueError, match="fixed to 1"):
        server.build_command({"tensor_parallel_size": 2})


def test_server_environment_snapshot_keeps_execution_settings_only(tmp_path) -> None:
    server = ManagedVLLMServer(TuningConfig(model="test-model"), trial_dir=tmp_path)
    environment = {
        "OMP_NUM_THREADS": "4",
        "TORCHINDUCTOR_CACHE_DIR": "/cache/inductor",
        "CUDA_VISIBLE_DEVICES": "0",
        "TOKENIZERS_PARALLELISM": "false",
        "VLLM_CACHE_ROOT": "/cache/vllm",
        "HF_HOME": "/cache/hf",
        "HUGGINGFACE_HUB_CACHE": "/cache/hub",
        "TRITON_CACHE_DIR": "/cache/triton",
        "TMPDIR": "/scratch/tmp",
        "TORCH_HOME": "/cache/torch",
        "TRANSFORMERS_CACHE": "/cache/transformers",
        "HF_TOKEN": "hf-secret",
        "HF_TOKEN_PATH": "/secrets/hf-token",
        "HUGGINGFACE_HUB_TOKEN": "hub-secret",
        "VLLM_API_KEY": "vllm-secret",
        "HTTP_PROXY": "http://proxy.invalid",
        "NO_PROXY": "localhost",
        "UNRELATED": "not-recorded",
    }

    safe = server._safe_environment(environment)

    assert safe == {
        "CUDA_VISIBLE_DEVICES": "0",
        "HF_HOME": "/cache/hf",
        "HUGGINGFACE_HUB_CACHE": "/cache/hub",
        "OMP_NUM_THREADS": "4",
        "TMPDIR": "/scratch/tmp",
        "TOKENIZERS_PARALLELISM": "false",
        "TORCHINDUCTOR_CACHE_DIR": "/cache/inductor",
        "TORCH_HOME": "/cache/torch",
        "TRANSFORMERS_CACHE": "/cache/transformers",
        "TRITON_CACHE_DIR": "/cache/triton",
        "VLLM_CACHE_ROOT": "/cache/vllm",
    }


@pytest.mark.asyncio
async def test_start_records_effective_safe_environment(tmp_path, monkeypatch) -> None:
    class FakeProcess:
        pid = 12345

        @staticmethod
        def poll():
            return None

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "7")
    monkeypatch.setenv("OMP_NUM_THREADS", "3")
    monkeypatch.setenv("HF_TOKEN", "must-not-be-recorded")
    monkeypatch.setenv("HTTPS_PROXY", "http://must-not-be-recorded.invalid")
    monkeypatch.setattr(
        server_module.subprocess,
        "Popen",
        lambda *args, **kwargs: FakeProcess(),
    )
    server = ManagedVLLMServer(TuningConfig(model="test-model"), trial_dir=tmp_path)
    monkeypatch.setattr(server, "_compute_pids", lambda: ([55], None))

    await server.start({"max_num_seqs": 8})

    payload = json.loads(server.command_path.read_text(encoding="utf-8"))
    assert payload["environment"]["CUDA_VISIBLE_DEVICES"] == "0"
    assert payload["environment"]["OMP_NUM_THREADS"] == "3"
    assert "HF_TOKEN" not in payload["environment"]
    assert "HTTPS_PROXY" not in payload["environment"]
    assert server._compute_pids_baseline == {55}


class ExitedLeader:
    pid = 4242

    @staticmethod
    def poll():
        return 7


@pytest.mark.asyncio
async def test_stop_uses_saved_group_when_leader_already_exited(tmp_path, monkeypatch) -> None:
    server = ManagedVLLMServer(TuningConfig(model="test-model"), trial_dir=tmp_path)
    server.process = ExitedLeader()
    server.process_group_id = ExitedLeader.pid
    states = iter([(True, [4243]), (False, [])])
    signals: list[tuple[int, signal.Signals]] = []
    monkeypatch.setattr(server, "_process_group_alive", lambda pgid: next(states))
    monkeypatch.setattr(server, "_compute_pids", lambda: ([], None))
    monkeypatch.setattr(server_module, "port_is_available", lambda *_: True)
    monkeypatch.setattr(
        server_module.os,
        "killpg",
        lambda pgid, sent_signal: signals.append((pgid, sent_signal)),
    )

    cleanup = await server.stop(graceful_timeout=0, kill_timeout=0)

    assert signals == [(4242, signal.SIGTERM)]
    assert cleanup.leader_exit_code == 7
    assert cleanup.process_group_pids_before == [4243]
    assert cleanup.process_group_empty is True
    assert cleanup.port_available is True
    assert cleanup.gpu_clean is True
    assert cleanup.clean is True


@pytest.mark.asyncio
async def test_stop_escalates_to_sigkill_when_child_ignores_sigterm(tmp_path, monkeypatch) -> None:
    server = ManagedVLLMServer(TuningConfig(model="test-model"), trial_dir=tmp_path)
    server.process = ExitedLeader()
    server.process_group_id = ExitedLeader.pid
    states = iter([(True, [4243]), (True, [4243]), (False, [])])
    signals: list[signal.Signals] = []
    monkeypatch.setattr(server, "_process_group_alive", lambda pgid: next(states))
    monkeypatch.setattr(server, "_compute_pids", lambda: ([], None))
    monkeypatch.setattr(server_module, "port_is_available", lambda *_: True)
    monkeypatch.setattr(
        server_module.os,
        "killpg",
        lambda pgid, sent_signal: signals.append(sent_signal),
    )

    cleanup = await server.stop(graceful_timeout=0, kill_timeout=0)

    assert signals == [signal.SIGTERM, signal.SIGKILL]
    assert cleanup.term_sent_monotonic_ns is not None
    assert cleanup.kill_sent_monotonic_ns is not None
    assert cleanup.process_group_empty is True
    assert cleanup.clean is True


@pytest.mark.asyncio
async def test_cleanup_gpu_pid_residual_is_not_clean(tmp_path, monkeypatch) -> None:
    server = ManagedVLLMServer(TuningConfig(model="test-model"), trial_dir=tmp_path)
    server.process = ExitedLeader()
    server.process_group_id = ExitedLeader.pid
    states = iter([(True, [4243]), (False, [])])
    compute_pids = iter([([4243], None), ([4243], None)])
    monkeypatch.setattr(server, "_process_group_alive", lambda pgid: next(states))
    monkeypatch.setattr(server, "_compute_pids", lambda: next(compute_pids))
    monkeypatch.setattr(server_module, "port_is_available", lambda *_: True)
    monkeypatch.setattr(server_module.os, "killpg", lambda *_: None)

    cleanup = await server.stop(graceful_timeout=0, kill_timeout=0)

    assert cleanup.tracked_compute_pids_after == [4243]
    assert cleanup.gpu_check_available is True
    assert cleanup.gpu_clean is False
    assert cleanup.clean is False


@pytest.mark.asyncio
async def test_cleanup_tracks_new_gpu_pid_outside_the_process_group(tmp_path, monkeypatch) -> None:
    server = ManagedVLLMServer(TuningConfig(model="test-model"), trial_dir=tmp_path)
    server.process = ExitedLeader()
    server.process_group_id = ExitedLeader.pid
    server._compute_pids_baseline = set()
    server._compute_baseline_error = None
    states = iter([(True, [4243]), (False, [])])
    compute_pids = iter([([777], None), ([777], None)])
    monkeypatch.setattr(server, "_process_group_alive", lambda pgid: next(states))
    monkeypatch.setattr(server, "_compute_pids", lambda: next(compute_pids))
    monkeypatch.setattr(server_module, "port_is_available", lambda *_: True)
    monkeypatch.setattr(server_module.os, "killpg", lambda *_: None)

    cleanup = await server.stop(graceful_timeout=0, kill_timeout=0)

    assert cleanup.compute_pids_baseline == []
    assert cleanup.process_group_pids_before == [4243]
    assert cleanup.compute_pids_before == [777]
    assert cleanup.tracked_compute_pids_after == [777]
    assert cleanup.gpu_clean is False
    assert cleanup.clean is False


@pytest.mark.asyncio
async def test_cleanup_polls_until_tracked_gpu_pid_exits(tmp_path, monkeypatch) -> None:
    server = ManagedVLLMServer(TuningConfig(model="test-model"), trial_dir=tmp_path)
    server.process = ExitedLeader()
    server.process_group_id = ExitedLeader.pid
    states = iter([(True, [4243]), (False, [])])
    compute_pids = iter([([777], None), ([777], None), ([], None)])
    monkeypatch.setattr(server, "_process_group_alive", lambda pgid: next(states))
    monkeypatch.setattr(server, "_compute_pids", lambda: next(compute_pids))
    monkeypatch.setattr(server_module, "port_is_available", lambda *_: True)
    monkeypatch.setattr(server_module.os, "killpg", lambda *_: None)

    cleanup = await server.stop(graceful_timeout=0, kill_timeout=0.2)

    assert cleanup.compute_pids_before == [777]
    assert cleanup.compute_pids_after == []
    assert cleanup.tracked_compute_pids_after == []
    assert cleanup.gpu_clean is True
    assert cleanup.clean is True


@pytest.mark.asyncio
async def test_cleanup_nvml_unavailable_is_unknown_not_clean(tmp_path, monkeypatch) -> None:
    server = ManagedVLLMServer(TuningConfig(model="test-model"), trial_dir=tmp_path)
    monkeypatch.setattr(server, "_compute_pids", lambda: ([], "driver unavailable"))
    monkeypatch.setattr(server_module, "port_is_available", lambda *_: True)

    cleanup = await server.stop(graceful_timeout=0, kill_timeout=0)

    assert cleanup.gpu_check_available is False
    assert cleanup.gpu_clean is None
    assert cleanup.clean is False
    assert any("NVML" in error for error in cleanup.errors)


@pytest.mark.asyncio
async def test_cleanup_occupied_port_is_not_clean(tmp_path, monkeypatch) -> None:
    server = ManagedVLLMServer(TuningConfig(model="test-model"), trial_dir=tmp_path)
    monkeypatch.setattr(server, "_compute_pids", lambda: ([], None))
    monkeypatch.setattr(server_module, "port_is_available", lambda *_: False)

    cleanup = await server.stop(graceful_timeout=0, kill_timeout=0)

    assert cleanup.process_group_empty is True
    assert cleanup.gpu_clean is True
    assert cleanup.port_available is False
    assert cleanup.clean is False
