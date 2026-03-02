"""Baseline generation runner for vLLM models.

This module provides functionality to generate baseline metrics using
vLLM's default parameters before running optimization.
"""

import asyncio
import json
import logging
import os
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional

import httpx
import yaml

try:
    from datasets import load_dataset
except ImportError:
    raise ImportError(
        "datasets package is required. Install with: pip install datasets"
    )

from vllm_tuner.config.models import TuningConfig
from vllm_tuner.profiling.gpu_collector import GPUCollector
from vllm_tuner.profiling.vllm_metrics import VLLMMetricsTracker

logger = logging.getLogger(__name__)


@dataclass
class BaselineMetrics:
    """Container for baseline metrics."""

    model: str
    timestamp: str
    vllm_params: dict[str, Any]
    benchmark_params: dict[str, Any]

    metrics: dict[str, Any] = field(default_factory=dict)
    gpu_info: dict[str, Any] = field(default_factory=dict)
    memory_samples: List[float] = field(default_factory=list)
    gpu_util_samples: List[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        if self.memory_samples:
            peak_memory = max(self.memory_samples)
            avg_memory = sum(self.memory_samples) / len(self.memory_samples)
        else:
            peak_memory = avg_memory = 0

        if self.gpu_util_samples:
            avg_gpu_util = sum(self.gpu_util_samples) / len(self.gpu_util_samples)
        else:
            avg_gpu_util = 0

        return {
            "model": self.model,
            "timestamp": self.timestamp,
            "configuration": {
                "vllm_params": self.vllm_params,
                "benchmark_params": self.benchmark_params,
            },
            "metrics": {
                **self.metrics,
                "peak_memory_mb": peak_memory,
                "average_memory_mb": avg_memory,
                "average_gpu_utilization": avg_gpu_util,
            },
            "gpu_info": self.gpu_info,
        }


class VLLMBaselineRunner:
    """Runner for generating vLLM baseline metrics."""

    def __init__(self, config: TuningConfig, output_dir: Path):
        self.config = config
        self.output_dir = output_dir
        self.process: Optional[subprocess.Popen] = None
        self.base_url = "http://localhost:8000"
        self.gpu_collector = GPUCollector(device_ids=config.gpu.device_ids)
        self.num_requests = config.workload.sample_size
        self.warmup_requests = self.config.workload.warmup_requests
        self.concurrent_requests = config.workload.concurrent_requests
        self.max_tokens = config.workload.max_tokens
        self.dataset_name = config.workload.dataset_name
        gpu_memory_utilization = config.vllm_args.get("gpu-memory-utilization", "0.6")
        max_num_seqs = config.vllm_args.get("max-num-seqs", "200")
        max_num_batched_tokens = config.vllm_args.get("max-num-batched-tokens", None)
        tensor_parallel_size = (
            str(len(config.gpu.device_ids)) if len(config.gpu.device_ids) > 1 else "1"
        )

        vllm_params_full = {
            **config.vllm_args,
            "gpu_memory_utilization": gpu_memory_utilization,
            "max_num_seqs": int(max_num_seqs),
            "max_num_batched_tokens": int(max_num_batched_tokens)
            if max_num_batched_tokens
            else None,
            "tensor_parallel_size": int(tensor_parallel_size),
        }

        self.metrics = BaselineMetrics(
            model=config.model,
            timestamp=datetime.now().isoformat(),
            vllm_params=vllm_params_full,
            benchmark_params={
                "num_requests": self.num_requests,
                "concurrency": self.concurrent_requests,
                "warmup_requests": self.warmup_requests,
                "max_tokens": self.max_tokens,
                "dataset": self.dataset_name,
            },
        )
        self.benchmark_start_time: Optional[datetime] = None
        self.benchmark_end_time: Optional[datetime] = None
        self.monitoring_task: Optional[asyncio.Task] = None

    def _build_vllm_command(self) -> List[str]:
        """Build vLLM server command with default parameters."""
        config = self.config

        cmd = [
            "python",
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            config.model,
        ]

        if len(config.vllm_args) > 0:
            for key in config.vllm_args.keys():
                cmd.extend(["--" + key, str(config.vllm_args[key])])

        if len(config.gpu.device_ids) > 1:
            cmd.extend(["--tensor-parallel-size", str(len(config.gpu.device_ids))])

        cmd.append("--disable-log-requests")

        return cmd

    async def _start_vllm_server(self) -> subprocess.Popen:
        """Start vLLM server with environment setup."""
        logger.info("Starting vLLM server with default parameters...")

        self.gpu_collector.initialize()

        log_dir = self.output_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / "vllm_baseline.log"

        cmd = self._build_vllm_command()
        logger.info(f"vLLM command: {' '.join(cmd)}")

        env = os.environ.copy()
        visible_devices = ",".join(str(gpu_id) for gpu_id in self.config.gpu.device_ids)
        env["CUDA_VISIBLE_DEVICES"] = visible_devices

        with open(log_path, "w", encoding="utf-8") as log_file:
            self.process = subprocess.Popen(
                cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                env=env,
                text=True,
            )

        logger.info(f"vLLM server started with PID {self.process.pid}")
        logger.info(f"Logs: {log_path}")

        ready = await self._wait_server_ready()
        if not ready:
            raise RuntimeError("vLLM server failed to become ready")

        return self.process

    async def _wait_server_ready(self, timeout: int = 300) -> bool:
        """Wait for vLLM server to be ready."""
        logger.info(f"Waiting for vLLM server at {self.base_url}...")

        start = time.time()
        while time.time() - start < timeout:
            if self.process is None or self.process.poll() is not None:
                logger.error("vLLM server process died")
                return False

            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
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
                                logger.debug(
                                    f"Health check {endpoint}: {e.response.status_code}"
                                )
            except (httpx.ConnectError, httpx.ConnectTimeout):
                pass

            await asyncio.sleep(2.0)

        logger.warning(f"vLLM server did not become ready within {timeout}s")
        return False

    def _load_prompts(self) -> List[str]:
        """Load prompts from dataset."""
        logger.info(f"Loading prompts from {self.dataset_name}...")

        try:
            dataset = load_dataset(self.dataset_name, split="train")
            logger.info(f"Loaded {len(dataset)} examples from dataset")
        except Exception as e:
            raise RuntimeError(f"Failed to load dataset: {e}")

        total_prompts_needed = self.warmup_requests + self.num_requests
        prompts = []
        for example in dataset:
            instruction = example.get("instruction", "")
            input_text = example.get("input", "")

            if input_text:
                prompt = f"{instruction}\\n\\n{input_text}"
            else:
                prompt = instruction

            if prompt.strip():
                prompts.append(prompt)

            if len(prompts) >= total_prompts_needed:
                break

        if len(prompts) < total_prompts_needed:
            raise ValueError(
                f"Not enough prompts in dataset: found {len(prompts)}, need {total_prompts_needed}"
            )

        logger.info(f"Loaded {len(prompts)} prompts")
        return prompts

    async def _monitor_gpu(self, stop_event: asyncio.Event):
        """Continuously monitor GPU metrics during benchmark."""
        logger.info("Starting GPU monitoring...")

        while not stop_event.is_set():
            try:
                all_stats = self.gpu_collector.collect_all()

                for stats in all_stats:
                    self.metrics.memory_samples.append(stats.memory_used_mb)
                    self.metrics.gpu_util_samples.append(stats.gpu_utilization)

            except Exception as e:
                logger.warning(f"GPU monitoring error: {e}")

            await asyncio.sleep(1.0)

        logger.info("GPU monitoring stopped")

    async def _send_request(
        self,
        prompt: str,
        client: httpx.AsyncClient,
        metrics_tracker: VLLMMetricsTracker,
        request_id: str,
    ) -> bool:
        """Send a single completion request to vLLM."""
        url = f"{self.base_url}/v1/completions"

        metrics_tracker.record_request(request_id)
        start_time = time.time()

        output_tokens = 0
        payload = {
            "model": self.config.model,
            "prompt": prompt,
            "max_tokens": self.max_tokens,
            "temperature": 1.0,
            "top_p": 1.0,
        }

        try:
            async with client.stream(
                "POST", url, json=payload, timeout=300
            ) as response:
                response.raise_for_status()

                first_chunk_time = None
                async for chunk in response.aiter_bytes():
                    if first_chunk_time is None:
                        first_chunk_time = time.time()
                        ttft = first_chunk_time - start_time
                        metrics_tracker.record_ttft(request_id, ttft)
                    try:
                        chunk_str = chunk.decode("utf-8")
                        if "choices" in chunk_str:
                            try:
                                data = json.loads(chunk_str)
                                if "choices" in data and data["choices"]:
                                    text = data["choices"][0].get("text", "")
                                    if text:
                                        output_tokens += 1
                            except json.JSONDecodeError:
                                pass
                    except Exception:
                        pass

                metrics_tracker.record_completion(request_id, output_tokens)

        except httpx.TimeoutException:
            logger.error(f"Request {request_id} timed out")
            metrics_tracker.record_error("timeout")
            return False

        except httpx.HTTPStatusError as e:
            logger.error(f"Request {request_id} failed: {e.response.status_code}")
            metrics_tracker.record_error("http")
            return False

        except Exception as e:
            logger.error(f"Request {request_id} error: {e}")
            metrics_tracker.record_error("general")
            return False

        return True

    async def _run_benchmark(
        self,
        prompts: List[str],
        metrics_tracker: VLLMMetricsTracker,
    ) -> dict[str, Any]:
        """Run benchmark with concurrent requests."""
        num_prompts = len(prompts)

        logger.info(
            f"Running benchmark with {num_prompts} prompts, concurrency={self.concurrent_requests}"
        )
        metrics_tracker.start_benchmark()

        async with httpx.AsyncClient() as client:
            semaphore = asyncio.Semaphore(self.concurrent_requests)

            async def execute_request(prompt: str, idx: int):
                request_id = f"req_{idx}"
                async with semaphore:
                    return await self._send_request(
                        prompt, client, metrics_tracker, request_id
                    )

            tasks = [execute_request(prompt, i) for i, prompt in enumerate(prompts)]
            results = await asyncio.gather(*tasks, return_exceptions=False)

        metrics_tracker.end_benchmark()
        failed_count = sum(1 for r in results if not r)
        succeeded_count = sum(1 for r in results if r)
        metrics_summary = metrics_tracker.get_summary()
        logger.info(
            f"Benchmark completed: {succeeded_count}/{num_prompts} requests succeeded ({failed_count} failed)"
        )

        if failed_count > 0:
            raise RuntimeError(f"Aborting: {failed_count} requests failed")

        return metrics_summary

    async def _stop_server(self):
        """Stop vLLM server gracefully."""
        if self.process and self.process.poll() is None:
            logger.info(f"Stopping vLLM server (PID {self.process.pid})...")

            self.process.terminate()

            try:
                self.process.wait(timeout=30)
                logger.info("vLLM server stopped gracefully")
            except subprocess.TimeoutExpired:
                logger.warning("vLLM server did not stop, killing...")
                self.process.kill()
                self.process.wait()

        self.process = None

        if self.gpu_collector.initialized:
            self.gpu_collector.shutdown()

    def _generate_outputs(self):
        """Generate JSON, YAML, and text outputs."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if self.gpu_collector.initialized:
            self.metrics.gpu_info = self.gpu_collector.get_aggregate_stats()
        else:
            logger.warning("GPU collector not initialized, using empty GPU info")
            self.metrics.gpu_info = {}

        json_path = self.output_dir / "baseline_metrics.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self.metrics.to_dict(), f, indent=2)
        logger.info(f"JSON output: {json_path}")

        yaml_data = {
            "model": self.metrics.model,
            "timestamp": self.metrics.timestamp,
            "vllm_params": self.metrics.vllm_params,
            "baseline_metrics": {
                "throughput_requests_per_sec": self.metrics.metrics.get(
                    "throughput_requests_per_sec", 0
                ),
                "throughput_tokens_per_sec": self.metrics.metrics.get(
                    "throughput_tokens_per_sec", 0
                ),
                "avg_latency_ms": self.metrics.metrics.get("avg_latency_ms", 0),
                "p50_latency_ms": self.metrics.metrics.get("p50_latency_ms", 0),
                "p95_latency_ms": self.metrics.metrics.get("p95_latency_ms", 0),
                "p99_latency_ms": self.metrics.metrics.get("p99_latency_ms", 0),
                "peak_memory_mb": max(self.metrics.memory_samples)
                if self.metrics.memory_samples
                else 0,
                "average_memory_mb": (
                    sum(self.metrics.memory_samples) / len(self.metrics.memory_samples)
                    if self.metrics.memory_samples
                    else 0
                ),
                "average_gpu_utilization": (
                    sum(self.metrics.gpu_util_samples)
                    / len(self.metrics.gpu_util_samples)
                    if self.metrics.gpu_util_samples
                    else 0
                ),
                "num_completed": self.metrics.metrics.get("requests_completed", 0),
                "duration_seconds": self.metrics.metrics.get("duration_seconds", 0),
            },
        }

        yaml_path = self.output_dir / "baseline_config.yaml"
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False)
        logger.info(f"YAML output: {yaml_path}")

        self._generate_text_summary()

    def _generate_text_summary(self):
        """Generate human-readable text summary."""
        metrics_dict = self.metrics.to_dict()

        total_memory_mb = self.metrics.gpu_info.get("total_memory_mb", 0)
        peak_memory_mb = metrics_dict["metrics"].get("peak_memory_mb", 0)
        if total_memory_mb > 0:
            utilization_pct = peak_memory_mb / total_memory_mb * 100
        else:
            utilization_pct = 0.0

        summary = f"""
{"=" * 80}
BASELINE METRICS FOR {self.metrics.model.upper()}
{"=" * 80}

Configuration:
  - vLLM Parameters (Defaults):
    ├── gpu_memory_utilization: {self.metrics.vllm_params["gpu_memory_utilization"]}
    ├── max_num_seqs: {self.metrics.vllm_params["max_num_seqs"]}
    ├── max_num_batched_tokens: {self.metrics.vllm_params["max_num_batched_tokens"] or "default"}
    └── tensor_parallel_size: {self.metrics.vllm_params["tensor_parallel_size"]}

  - Benchmark Parameters:
    ├── num_requests: {self.metrics.benchmark_params["num_requests"]}
    ├── concurrency: {self.metrics.benchmark_params["concurrency"]}
    ├── warmup_requests: {self.metrics.benchmark_params["warmup_requests"]}
    └── max_tokens: {self.metrics.benchmark_params["max_tokens"]}

Performance Metrics:
  Throughput:         {metrics_dict["metrics"].get("throughput_requests_per_sec", 0):.2f} requests/sec  ({metrics_dict["metrics"].get("throughput_tokens_per_sec", 0):.0f} tokens/sec)
  Avg Latency:        {metrics_dict["metrics"].get("avg_latency_ms", 0):.2f} ms
  P50 Latency:        {metrics_dict["metrics"].get("p50_latency_ms", 0):.2f} ms
  P95 Latency:        {metrics_dict["metrics"].get("p95_latency_ms", 0):.2f} ms
  P99 Latency:        {metrics_dict["metrics"].get("p99_latency_ms", 0):.2f} ms

GPU Metrics:
  Initial Memory:     {metrics_dict["metrics"].get("initial_memory_mb", 0):.0f} MB
  Peak Memory:        {metrics_dict["metrics"].get("peak_memory_mb", 0):.0f} MB
  Avg Memory:         {metrics_dict["metrics"].get("average_memory_mb", 0):.0f} MB
  Utilization:        {utilization_pct:.1f}%
  Avg GPU Util:       {metrics_dict["metrics"].get("average_gpu_utilization", 0) * 100:.1f}%

Requests:  {metrics_dict["metrics"].get("requests_completed", 0)} completed / {self.metrics.benchmark_params["num_requests"]} total
Duration:  {metrics_dict["metrics"].get("duration_seconds", 0):.2f} seconds

Generated: {self.metrics.timestamp}
{"=" * 80}
"""

        text_path = self.output_dir / "baseline_summary.txt"
        with open(text_path, "w", encoding="utf-8") as f:
            f.write(summary)
        logger.info(f"Text summary: {text_path}")

    async def run(self):
        """Run the full baseline generation workflow."""
        logger.info(f"Starting baseline generation for {self.config.model}")

        stop_event = asyncio.Event()

        def cleanup():
            stop_event.set()
            if self.monitoring_task and not self.monitoring_task.done():
                self.monitoring_task.cancel()

        try:
            await self._start_vllm_server()

            prompts = self._load_prompts()
            warmup_prompts = prompts[: self.warmup_requests]
            main_prompts = prompts[
                self.warmup_requests : self.warmup_requests + self.num_requests
            ]

            self.monitoring_task = asyncio.create_task(self._monitor_gpu(stop_event))

            if self.warmup_requests > 0 and warmup_prompts:
                logger.info(f"Running warmup with {len(warmup_prompts)} requests...")
                warmup_metrics = VLLMMetricsTracker()
                await self._run_benchmark(warmup_prompts, warmup_metrics)
                logger.info("Warmup completed")

            self.benchmark_start_time = datetime.now()
            logger.info(
                f"Starting main benchmark with {len(main_prompts)} requests, concurrency={self.concurrent_requests}"
            )

            main_metrics = VLLMMetricsTracker()
            benchmark_results = await self._run_benchmark(main_prompts, main_metrics)

            self.benchmark_end_time = datetime.now()
            self.metrics.metrics = benchmark_results
            self.metrics.metrics.update(
                {
                    "initial_memory_mb": sum(
                        s.memory_used_mb for s in self.gpu_collector.collect_all()
                    ),
                }
            )

            self._generate_outputs()

            logger.info("Baseline generation completed successfully")

        except Exception as e:
            logger.error(f"Baseline generation failed: {e}", exc_info=True)
            cleanup()
            raise

        finally:
            cleanup()
            await self._stop_server()


async def generate_baseline_from_config(
    config: TuningConfig,
    output_dir: Path,
) -> None:
    """
    Generate baseline metrics from tuning configuration.

    Args:
        config: TuningConfig with model and workload settings
        output_dir: Directory to save baseline results
    """
    runner = VLLMBaselineRunner(config, output_dir)

    await runner.run()
