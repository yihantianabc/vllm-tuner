"""vLLM launcher for starting server with tuned parameters."""

import asyncio
import logging
import os
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any

import httpx

from vllm_tuner.config.models import TuningConfig

logger = logging.getLogger(__name__)


class VLLMLauncher:
    """Launch and manage vLLM server instance."""

    def __init__(
        self,
        config: TuningConfig,
        host: str = "127.0.0.1",
        port: int = 8000,
        log_dir: Optional[Path] = None,
    ):
        self.config = config
        self.host = host
        self.port = port
        self.log_dir = log_dir or Path("logs")
        self.process: Optional[subprocess.Popen] = None
        self.base_url = f"http://{host}:{port}"

    def build_command(self, trial_params: Dict[str, Any]) -> list[str]:
        """Build vLLM server command with trial parameters."""
        cmd = [
            "python",
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            self.config.model,
            "--host",
            self.host,
            "--port",
            str(self.port),
        ]

        gpu_config = self.config.gpu

        if len(gpu_config.device_ids) > 0:
            visible_devices = ",".join(str(id) for id in gpu_config.device_ids)
            cmd.extend(
                [
                    "--tensor-parallel-size",
                    str(trial_params.get("tensor_parallel_size", 1)),
                    "--pipeline-parallel-size",
                    str(trial_params.get("pipeline_parallel_size", 1)),
                ]
            )

        param_names = {
            "gpu_memory_utilization": "--gpu-memory-utilization",
            "max_num_seqs": "--max-num-seqs",
            "max_num_batched_tokens": "--max-num-batched-tokens",
        }

        for param_name, arg_name in param_names.items():
            if param_name in trial_params:
                cmd.extend([arg_name, str(trial_params[param_name])])

        for key, value in self.config.vllm_args.items():
            cmd.extend([f"--{key}", str(value)])

        cmd.append("--disable-log-requests")

        return cmd

    async def start(
        self, trial_params: Dict[str, Any], log_file: Optional[str] = None
    ) -> subprocess.Popen:
        """Start vLLM server with given parameters."""
        if self.process and self.process.poll() is None:
            raise RuntimeError("vLLM server is already running")

        self.log_dir.mkdir(parents=True, exist_ok=True)

        if log_file is None:
            trial_id = trial_params.get("_trial_id", "unknown")
            log_path = self.log_dir / f"vllm_trial_{trial_id}.log"
        else:
            log_path = Path(log_file)

        cmd = self.build_command(trial_params)
        logger.info(f"Starting vLLM server: {' '.join(cmd)}")

        env = os.environ.copy()
        if len(self.config.gpu.device_ids) > 0:
            visible_devices = ",".join(str(id) for id in self.config.gpu.device_ids)
            env["CUDA_VISIBLE_DEVICES"] = visible_devices

        with open(log_path, "w", encoding="utf-8") as f:
            self.process = subprocess.Popen(
                cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                env=env,
                text=True,
            )

        logger.info(f"vLLM server started with PID {self.process.pid}")
        logger.info(f"Logs written to: {log_path}")

        return self.process

    async def wait_ready(
        self, timeout: int = 300, check_interval: float = 10.0
    ) -> bool:
        """Wait for server to be ready."""
        async with httpx.AsyncClient(timeout=30.0) as client:
            elapsed = 0
            while elapsed < timeout:
                if self.process is None or self.process.poll() is not None:
                    logger.error("vLLM server process is not running")
                    return False

                try:
                    for endpoint in ["/health", "/v1/health", "/v1/models"]:
                        try:
                            response = await client.get(f"{self.base_url}{endpoint}")
                            if response.status_code == 200:
                                logger.info(
                                    f"vLLM server ready at {self.base_url}{endpoint}"
                                )
                                return True
                        except httpx.HTTPStatusError as e:
                            if e.response.status_code != 404:
                                logger.warning(
                                    f"Health check {endpoint} failed: {e.response.status_code}"
                                )
                except (httpx.ConnectError, httpx.ConnectTimeout) as e:
                    logger.debug(f"Connection error: {e}")

                await asyncio.sleep(check_interval)
                elapsed += check_interval

            logger.warning(f"vLLM server did not become ready within {timeout}s")
            return False

    async def stop(self) -> None:
        """Stop vLLM server gracefully."""
        if self.process is None:
            return

        logger.info(f"Stopping vLLM server (PID {self.process.pid})")

        self.process.terminate()

        try:
            self.process.wait(timeout=30)
            logger.info("vLLM server stopped gracefully")
        except subprocess.TimeoutExpired:
            logger.warning("vLLM server did not stop gracefully, killing")
            self.process.kill()
            self.process.wait()

        self.process = None

    def is_running(self) -> bool:
        """Check if server process is running."""
        return self.process is not None and self.process.poll() is None

    def get_log_file(self, trial_id: str) -> Path:
        """Get log file path for a trial."""
        return self.log_dir / f"vllm_trial_{trial_id}.log"


async def test_server_connection(base_url: str, timeout: int = 10) -> bool:
    """Test if vLLM server is responsive."""
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(f"{base_url}/health")
            return response.status_code == 200
    except Exception:
        return False
