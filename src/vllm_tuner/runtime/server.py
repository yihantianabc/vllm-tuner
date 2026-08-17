"""Process-group-safe vLLM server lifecycle."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional

import httpx
from pydantic import BaseModel, Field

from vllm_tuner.config.models import TuningConfig
from vllm_tuner.optimization.search_space import canonical_parameter_name

from .failures import FailureReason, FailureType, classify_failure

logger = logging.getLogger(__name__)

EXECUTION_ENV_PREFIXES = (
    "CUBLAS_",
    "CUDA_",
    "FLASHINFER_",
    "HF_",
    "HUGGINGFACE_",
    "NCCL_",
    "NUMBA_",
    "OMP_",
    "SLOTUNE_",
    "TOKENIZERS_",
    "TORCH_",
    "TORCHINDUCTOR_",
    "TRITON_",
    "VLLM_",
    "XDG_",
)
EXECUTION_ENV_KEYS = frozenset(
    {
        "LD_LIBRARY_PATH",
        "PATH",
        "PYTHONDONTWRITEBYTECODE",
        "PYTHONPATH",
        "PYTHONUNBUFFERED",
        "TEMP",
        "TMP",
        "TMPDIR",
        "TORCH_HOME",
        "TRANSFORMERS_CACHE",
    }
)
SECRET_ENV_MARKERS = (
    "ACCESS_KEY",
    "API_KEY",
    "CREDENTIAL",
    "PASSWORD",
    "PRIVATE_KEY",
    "SECRET",
)
PROXY_ENV_KEYS = frozenset({"ALL_PROXY", "HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY"})

SLOTUNE_SCHEDULER_CONFIG_ENV = "SLOTUNE_ADAPTIVE_PREFILL_CONFIG"
SLOTUNE_SCHEDULER_LOG_ENV = "SLOTUNE_SCHEDULER_DECISION_LOG"


def uses_slotune_scheduler(config: TuningConfig) -> bool:
    """Return whether the configured class is implemented by this repository."""
    scheduler_cls = config.vllm_args.get("scheduler-cls", config.vllm_args.get("scheduler_cls"))
    return isinstance(scheduler_cls, str) and scheduler_cls.startswith("vllm_tuner.scheduler.")


def port_is_available(host: str, port: int) -> bool:
    """Check whether a TCP address can be bound now."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind((host, port))
        except OSError:
            return False
    return True


def find_free_port(host: str = "127.0.0.1") -> int:
    """Ask the OS for an ephemeral local port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


class ServerStatus(BaseModel):
    """Last known process and health state saved with every trial."""

    pid: Optional[int] = None
    process_group_id: Optional[int] = None
    running: bool = False
    ready: bool = False
    port: int
    exit_code: Optional[int] = None
    checked_monotonic_ns: int


class CleanupStatus(BaseModel):
    """Auditable evidence that a managed server no longer owns resources."""

    attempted: bool
    clean: bool
    pid: Optional[int] = None
    process_group_id: Optional[int] = None
    term_sent: bool = False
    term_sent_monotonic_ns: Optional[int] = None
    kill_sent: bool = False
    kill_sent_monotonic_ns: Optional[int] = None
    leader_exit_code: Optional[int] = None
    process_group_empty: bool = False
    process_group_pids_before: list[int] = Field(default_factory=list)
    process_group_pids_after: list[int] = Field(default_factory=list)
    gpu_check_available: bool = False
    compute_pids_baseline: list[int] = Field(default_factory=list)
    compute_pids_before: list[int] = Field(default_factory=list)
    compute_pids_after: list[int] = Field(default_factory=list)
    tracked_compute_pids_after: list[int] = Field(default_factory=list)
    gpu_clean: Optional[bool] = None
    port_available: bool = False
    errors: list[str] = Field(default_factory=list)
    checked_monotonic_ns: int


class ManagedVLLMServer:
    """Launch vLLM in its own process group and reliably tear down descendants."""

    def __init__(
        self,
        config: TuningConfig,
        *,
        host: str = "127.0.0.1",
        port: Optional[int] = None,
        trial_dir: str | Path = "logs",
    ) -> None:
        self.config = config
        self.host = host
        self.port = port if port is not None else find_free_port(host)
        self.trial_dir = Path(trial_dir)
        self.log_path = self.trial_dir / "server.log"
        self.command_path = self.trial_dir / "server-command.json"
        self.process: Optional[subprocess.Popen[str]] = None
        self.process_group_id: Optional[int] = None
        self.cleanup_status: Optional[CleanupStatus] = None
        self._compute_pids_baseline: set[int] = set()
        self._compute_baseline_error: Optional[str] = None
        self._tracked_compute_pids: set[int] = set()
        self._cleanup_lock = asyncio.Lock()
        self.ready = False
        self.failure_reason: Optional[FailureReason] = None

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    @staticmethod
    def _append_argument(command: list[str], name: str, value: Any) -> None:
        flag = "--" + name.replace("_", "-").lstrip("-")
        if isinstance(value, bool):
            if value:
                command.append(flag)
            return
        if value is None:
            return
        if isinstance(value, (list, tuple)):
            for item in value:
                command.extend([flag, str(item)])
            return
        command.extend([flag, str(value)])

    def build_command(self, trial_params: dict[str, Any]) -> list[str]:
        """Build a conflict-free command with fixed single-GPU parallelism."""
        params = {key: value for key, value in trial_params.items() if not key.startswith("_")}
        fixed = {"tensor_parallel_size": 1, "pipeline_parallel_size": 1}
        for name, value in fixed.items():
            if params.get(name, value) != value:
                raise ValueError(f"{name} is fixed to 1 for the single-GPU project")
            params[name] = value

        configured = {canonical_parameter_name(name) for name in self.config.vllm_args}
        duplicates = configured.intersection(params)
        if duplicates:
            raise ValueError(
                "trial parameters duplicate vllm_args: " + ", ".join(sorted(duplicates))
            )

        command = [
            sys.executable,
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            self.config.model,
            "--host",
            self.host,
            "--port",
            str(self.port),
        ]
        if self.config.model_revision:
            command.extend(["--revision", self.config.model_revision])
        for name in sorted(params):
            self._append_argument(command, name, params[name])
        for name in sorted(self.config.vllm_args):
            self._append_argument(command, name, self.config.vllm_args[name])
        if "disable_log_requests" not in configured:
            command.append("--disable-log-requests")
        return command

    def _safe_environment(self, environment: dict[str, str]) -> dict[str, str]:
        """Return execution-affecting variables without credentials or proxies."""

        def is_safe_execution_key(key: str) -> bool:
            normalized = key.upper()
            if normalized in PROXY_ENV_KEYS or normalized.endswith("_PROXY"):
                return False
            if normalized.endswith("_TOKEN") or (
                "TOKEN" in normalized and not normalized.startswith("TOKENIZERS_")
            ):
                return False
            if any(marker in normalized for marker in SECRET_ENV_MARKERS):
                return False
            return normalized in EXECUTION_ENV_KEYS or normalized.startswith(EXECUTION_ENV_PREFIXES)

        return {key: environment[key] for key in sorted(environment) if is_safe_execution_key(key)}

    async def start(self, trial_params: dict[str, Any]) -> subprocess.Popen[str]:
        """Start the process after resolving the exact port and command."""
        if self.process is not None and self.process.poll() is None:
            raise RuntimeError("vLLM server is already running")
        if not port_is_available(self.host, self.port):
            self.failure_reason = FailureReason(
                type=FailureType.PORT_IN_USE,
                message=f"Port {self.host}:{self.port} is already in use",
                phase="STARTING",
                retryable=True,
            )
            raise RuntimeError(self.failure_reason.message)

        self.trial_dir.mkdir(parents=True, exist_ok=True)
        command = self.build_command(trial_params)
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = str(self.config.gpu.device_ids[0])
        if uses_slotune_scheduler(self.config):
            environment[SLOTUNE_SCHEDULER_CONFIG_ENV] = (
                self.config.adaptive_prefill.model_dump_json()
            )
            if self.config.adaptive_prefill.decision_log_enabled:
                environment[SLOTUNE_SCHEDULER_LOG_ENV] = str(
                    (self.trial_dir / "scheduler-decisions.jsonl").resolve()
                )
        self.command_path.write_text(
            json.dumps(
                {"argv": command, "environment": self._safe_environment(environment)},
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        compute_pids_baseline, baseline_error = self._compute_pids()
        self._compute_pids_baseline = set(compute_pids_baseline)
        self._compute_baseline_error = baseline_error
        with self.log_path.open("w", encoding="utf-8") as log_handle:
            self.process = subprocess.Popen(
                command,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                env=environment,
                text=True,
                start_new_session=True,
            )
        # start_new_session=True makes the new leader's PID its PGID. Persist it
        # before the leader can exit so descendants remain addressable.
        self.process_group_id = self.process.pid
        self.cleanup_status = None
        self._tracked_compute_pids.clear()
        self.ready = False
        logger.info("Started vLLM process group with PID %s", self.process.pid)
        return self.process

    def log_tail(self, limit_bytes: int = 128 * 1024) -> str:
        """Read bounded failure evidence from the end of the server log."""
        if not self.log_path.exists():
            return ""
        with self.log_path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - limit_bytes))
            return handle.read().decode("utf-8", errors="replace")

    async def wait_ready(
        self,
        timeout: float = 600.0,
        check_interval: float = 1.0,
    ) -> bool:
        """Require both a live process and a successful health endpoint."""
        deadline = time.monotonic() + timeout
        # Local readiness must never be routed through a user/global proxy.  In
        # particular, SOCKS proxy settings commonly present on GPU hosts can
        # make a healthy loopback endpoint look like an empty response.
        async with httpx.AsyncClient(
            timeout=min(10.0, max(1.0, check_interval * 2)), trust_env=False
        ) as client:
            while time.monotonic() < deadline:
                if self.process is None:
                    return False
                exit_code = self.process.poll()
                if exit_code is not None:
                    self.failure_reason = classify_failure(
                        f"vLLM exited before readiness with code {exit_code}",
                        log_text=self.log_tail(),
                        phase="STARTING",
                        exit_code=exit_code,
                    )
                    return False
                for endpoint in ("/health", "/v1/models"):
                    try:
                        response = await client.get(f"{self.base_url}{endpoint}")
                        if response.status_code == 200:
                            self.ready = True
                            return True
                    except httpx.RequestError:
                        continue
                await asyncio.sleep(check_interval)
        self.failure_reason = FailureReason(
            type=FailureType.STARTUP_TIMEOUT,
            message=f"vLLM did not become ready within {timeout:.1f}s",
            phase="STARTING",
            retryable=True,
        )
        return False

    def status(self) -> ServerStatus:
        """Return a fresh process status without guessing missing values."""
        exit_code = self.process.poll() if self.process is not None else None
        running = self.process is not None and exit_code is None
        return ServerStatus(
            pid=self.process.pid if self.process is not None else None,
            process_group_id=self.process_group_id,
            running=running,
            ready=self.ready and running,
            port=self.port,
            exit_code=exit_code,
            checked_monotonic_ns=time.perf_counter_ns(),
        )

    @staticmethod
    def _process_group_pids(process_group_id: int) -> Optional[list[int]]:
        """Return live (non-zombie) group members, or None without procfs."""
        proc = Path("/proc")
        if not proc.is_dir():
            return None
        try:
            entries = list(proc.iterdir())
        except OSError:
            return None
        members: list[int] = []
        for entry in entries:
            if not entry.name.isdigit():
                continue
            try:
                stat = (entry / "stat").read_text(encoding="utf-8")
                after_name = stat[stat.rfind(")") + 2 :].split()
                state = after_name[0]
                group = int(after_name[2])
            except (OSError, ValueError, IndexError):
                continue
            if group == process_group_id and state != "Z":
                members.append(int(entry.name))
        return sorted(members)

    def _process_group_alive(self, process_group_id: int) -> tuple[bool, list[int]]:
        members = self._process_group_pids(process_group_id)
        if members is not None:
            return bool(members), members
        try:
            os.killpg(process_group_id, 0)
        except ProcessLookupError:
            return False, []
        except PermissionError:
            return True, []
        return True, []

    def _compute_pids(self) -> tuple[list[int], Optional[str]]:
        """Return target-GPU compute PIDs, explicitly marking NVML failures."""
        try:
            import pynvml  # type: ignore[import-untyped]

            pynvml.nvmlInit()
            process_ids: set[int] = set()
            try:
                for device_id in self.config.gpu.device_ids:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
                    for process in pynvml.nvmlDeviceGetComputeRunningProcesses(handle):
                        process_ids.add(int(process.pid))
            finally:
                pynvml.nvmlShutdown()
            return sorted(process_ids), None
        except Exception as error:
            return [], f"{type(error).__name__}: {error}"

    async def _wait_until_released(
        self,
        process_group_id: Optional[int],
        timeout: float,
    ) -> tuple[bool, list[int], bool]:
        deadline = time.monotonic() + max(0.0, timeout)
        while True:
            group_pids: list[int]
            if process_group_id is None:
                group_alive, group_pids = False, []
            else:
                group_alive, group_pids = self._process_group_alive(process_group_id)
            port_available = port_is_available(self.host, self.port)
            if not group_alive and port_available:
                return group_alive, group_pids, port_available
            if time.monotonic() >= deadline:
                return group_alive, group_pids, port_available
            await asyncio.sleep(min(0.1, max(0.0, deadline - time.monotonic())))

    async def _wait_for_compute_release(
        self,
        timeout: float,
    ) -> tuple[list[int], list[int], Optional[str]]:
        """Poll NVML until every process created after the launch baseline exits."""
        deadline = time.monotonic() + max(0.0, timeout)
        while True:
            compute_pids, error = self._compute_pids()
            residual = sorted(
                self._tracked_compute_pids.intersection(compute_pids)
                | set(compute_pids).difference(self._compute_pids_baseline)
            )
            if error is not None or not residual or time.monotonic() >= deadline:
                return compute_pids, residual, error
            await asyncio.sleep(min(0.1, max(0.0, deadline - time.monotonic())))

    async def stop(
        self,
        graceful_timeout: float = 30.0,
        kill_timeout: float = 10.0,
    ) -> CleanupStatus:
        """Shield complete process-group cleanup and return structured evidence."""
        cleanup_task = asyncio.create_task(self._stop_impl(graceful_timeout, kill_timeout))
        try:
            return await asyncio.shield(cleanup_task)
        except asyncio.CancelledError:
            with contextlib.suppress(asyncio.CancelledError):
                await cleanup_task
            raise

    async def _stop_impl(self, graceful_timeout: float, kill_timeout: float) -> CleanupStatus:
        async with self._cleanup_lock:
            if graceful_timeout < 0 or kill_timeout < 0:
                raise ValueError("cleanup timeouts must be non-negative")
            if self.cleanup_status is not None and self.cleanup_status.clean:
                return self.cleanup_status

            process = self.process
            pid = process.pid if process is not None else None
            process_group_id = self.process_group_id
            attempted = process is not None or process_group_id is not None
            errors: list[str] = []
            term_sent = False
            term_sent_monotonic_ns: Optional[int] = None
            kill_sent = False
            kill_sent_monotonic_ns: Optional[int] = None
            self.ready = False

            group_pids_before: list[int]
            if process_group_id is None:
                group_alive_before, group_pids_before = False, []
            else:
                group_alive_before, group_pids_before = self._process_group_alive(process_group_id)
            compute_pids_before, gpu_error_before = self._compute_pids()
            self._tracked_compute_pids.update(
                set(compute_pids_before).difference(self._compute_pids_baseline)
            )
            self._tracked_compute_pids.update(
                set(group_pids_before).intersection(compute_pids_before)
            )

            if group_alive_before and process_group_id is not None:
                try:
                    os.killpg(process_group_id, signal.SIGTERM)
                    term_sent = True
                    term_sent_monotonic_ns = time.perf_counter_ns()
                except ProcessLookupError:
                    pass
                except Exception as error:
                    errors.append(f"SIGTERM failed: {type(error).__name__}: {error}")

            group_alive, group_pids_after, port_available = await self._wait_until_released(
                process_group_id, graceful_timeout
            )
            if group_alive and process_group_id is not None:
                try:
                    os.killpg(process_group_id, signal.SIGKILL)
                    kill_sent = True
                    kill_sent_monotonic_ns = time.perf_counter_ns()
                except ProcessLookupError:
                    pass
                except Exception as error:
                    errors.append(f"SIGKILL failed: {type(error).__name__}: {error}")
            if group_alive or not port_available:
                group_alive, group_pids_after, port_available = await self._wait_until_released(
                    process_group_id, kill_timeout
                )

            leader_exit_code: Optional[int] = None
            if process is not None:
                try:
                    leader_exit_code = process.poll()
                except Exception as error:
                    errors.append(f"leader poll failed: {type(error).__name__}: {error}")

            (
                compute_pids_after,
                tracked_compute_pids_after,
                gpu_error_after,
            ) = await self._wait_for_compute_release(kill_timeout)
            gpu_check_available = (
                self._compute_baseline_error is None
                and gpu_error_before is None
                and gpu_error_after is None
            )
            if gpu_check_available:
                tracked_compute_pids_after = sorted(
                    set(tracked_compute_pids_after)
                    | set(group_pids_after).intersection(compute_pids_after)
                )
                gpu_clean: Optional[bool] = not tracked_compute_pids_after
            else:
                tracked_compute_pids_after = []
                gpu_clean = None
                if self._compute_baseline_error is not None:
                    errors.append(f"NVML baseline unavailable: {self._compute_baseline_error}")
                if gpu_error_before is not None:
                    errors.append(f"NVML pre-cleanup unavailable: {gpu_error_before}")
                if gpu_error_after is not None:
                    errors.append(f"NVML post-cleanup unavailable: {gpu_error_after}")

            process_group_empty = not group_alive
            clean = process_group_empty and port_available and gpu_clean is True
            if not process_group_empty:
                errors.append(f"process group {process_group_id} remains alive")
            if not port_available:
                errors.append(f"port {self.host}:{self.port} remains occupied")
            if tracked_compute_pids_after:
                errors.append(
                    "tracked GPU processes remain alive: "
                    + ", ".join(str(value) for value in tracked_compute_pids_after)
                )

            self.cleanup_status = CleanupStatus(
                attempted=attempted,
                clean=clean,
                pid=pid,
                process_group_id=process_group_id,
                term_sent=term_sent,
                term_sent_monotonic_ns=term_sent_monotonic_ns,
                kill_sent=kill_sent,
                kill_sent_monotonic_ns=kill_sent_monotonic_ns,
                leader_exit_code=leader_exit_code,
                process_group_empty=process_group_empty,
                process_group_pids_before=group_pids_before,
                process_group_pids_after=group_pids_after,
                gpu_check_available=gpu_check_available,
                compute_pids_baseline=sorted(self._compute_pids_baseline),
                compute_pids_before=compute_pids_before,
                compute_pids_after=compute_pids_after,
                tracked_compute_pids_after=tracked_compute_pids_after,
                gpu_clean=gpu_clean,
                port_available=port_available,
                errors=errors,
                checked_monotonic_ns=time.perf_counter_ns(),
            )
            # Preserve the cached PGID and structured evidence for postmortem
            # inspection; only discard the leader handle after all checks.
            self.process = None
            return self.cleanup_status

    def is_running(self) -> bool:
        """Return whether the managed leader process is alive."""
        return self.process is not None and self.process.poll() is None
