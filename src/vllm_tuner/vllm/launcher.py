"""Compatibility launcher backed by the process-group-safe runtime server."""

from __future__ import annotations

import asyncio
import logging
import subprocess
from pathlib import Path
from typing import Any, Optional

import httpx

from vllm_tuner.config.models import TuningConfig
from vllm_tuner.runtime.server import ManagedVLLMServer

logger = logging.getLogger(__name__)


class VLLMLauncher:
    """Retain the upstream API while using reliable lifecycle semantics."""

    def __init__(
        self,
        config: TuningConfig,
        host: str = "127.0.0.1",
        port: int = 8000,
        log_dir: Optional[Path] = None,
    ) -> None:
        self.config = config
        self.host = host
        self.port = port
        self.log_dir = log_dir or Path("logs")
        self.base_url = f"http://{host}:{port}"
        self.process: Optional[subprocess.Popen[str]] = None
        self._managed: Optional[ManagedVLLMServer] = None
        self._trial_logs: dict[str, Path] = {}

    def build_command(self, trial_params: dict[str, Any]) -> list[str]:
        """Build the same validated command used by the core runtime."""
        server = ManagedVLLMServer(
            self.config,
            host=self.host,
            port=self.port,
            trial_dir=self.log_dir,
        )
        return server.build_command(trial_params)

    async def start(
        self,
        trial_params: dict[str, Any],
        log_file: Optional[str] = None,
    ) -> subprocess.Popen[str]:
        """Start a trial in an isolated process group."""
        trial_id = str(trial_params.get("_trial_id", "unknown"))
        trial_dir = self.log_dir / f"trial_{trial_id}"
        self._managed = ManagedVLLMServer(
            self.config,
            host=self.host,
            port=self.port,
            trial_dir=trial_dir,
        )
        process = await self._managed.start(trial_params)
        self.process = process
        self._trial_logs[trial_id] = self._managed.log_path
        if log_file is not None:
            logger.warning(
                "The compatibility log_file override is ignored; authoritative log: %s",
                self._managed.log_path,
            )
        return process

    async def wait_ready(self, timeout: int = 300, check_interval: float = 1.0) -> bool:
        """Wait for readiness while also checking process exit."""
        if self._managed is not None:
            ready = await self._managed.wait_ready(timeout, check_interval)
            self.process = self._managed.process
            return ready

        # A small compatibility path keeps dependency-injected process tests useful.
        deadline = asyncio.get_running_loop().time() + timeout
        async with httpx.AsyncClient(timeout=30.0, trust_env=False) as client:
            while asyncio.get_running_loop().time() < deadline:
                if self.process is None or self.process.poll() is not None:
                    return False
                for endpoint in ("/health", "/v1/models"):
                    try:
                        response = await client.get(f"{self.base_url}{endpoint}")
                        if response.status_code == 200:
                            return True
                    except httpx.RequestError:
                        continue
                await asyncio.sleep(check_interval)
        return False

    async def stop(self) -> None:
        """Stop the complete server process group."""
        if self._managed is not None:
            await self._managed.stop()
            self._managed = None
            self.process = None
            return
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                await asyncio.wait_for(asyncio.to_thread(self.process.wait), 30)
            except asyncio.TimeoutError:
                self.process.kill()
                await asyncio.to_thread(self.process.wait)
        self.process = None

    def is_running(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def get_log_file(self, trial_id: str) -> Path:
        return self._trial_logs.get(
            str(trial_id), self.log_dir / f"trial_{trial_id}" / "server.log"
        )


async def test_server_connection(base_url: str, timeout: int = 10) -> bool:
    """Return whether a vLLM health endpoint responds successfully."""
    try:
        async with httpx.AsyncClient(timeout=timeout, trust_env=False) as client:
            response = await client.get(f"{base_url.rstrip('/')}/health")
            return response.status_code == 200
    except httpx.RequestError:
        return False
