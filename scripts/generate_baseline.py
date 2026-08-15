#!/usr/bin/env python3
"""
Generate baseline metrics for vLLM models using default parameters.

This script:
- Uses vLLM's default parameters (gpu_memory_utilization=0.9, max_num_seqs=128, max_num_batched_tokens=2048)
- Loads 1000 prompts from the Alpaca dataset
- Runs 5 warmup requests
- Runs 1000 benchmark requests with concurrency=10
- Monitors GPU memory, utilization, and temperature
- Generates JSON, YAML, and text summary outputs
- Aborts on any failures

Usage:
    python scripts/generate_baseline.py --model "Qwen/Qwen2.5-0.5B" --gpu-ids 0
"""

import argparse
import asyncio
import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

import httpx
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src import GPUCollector
from vllm_tuner.profiling import VLLMMetricsTracker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class BaselineConfig:
    """Configuration for baseline generation."""

    model: str = "gpt2"
    gpu_ids: List[int] = field(default_factory=lambda: [0])
    num_requests: int = 1000
    concurrency: int = 10
    warmup: int = 5
    max_tokens: int = 256
    host: str = "127.0.0.1"
    port: int = 8000
    dataset: str = "tatsu-lab/alpaca"
    output_dir: Optional[str] = None
    log_level: str = "INFO"

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.num_requests <= 0:
            raise ValueError("num_requests must be positive")
        if self.concurrency <= 0:
            raise ValueError("concurrency must be positive")
        if self.warmup < 0:
            raise ValueError("warmup must be non-negative")
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")

        if not self.output_dir:
            safe_model_name = self.model.replace("/", "_").replace("\\", "_")
            date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_dir = f"baselines/{safe_model_name}_{date_str}"


@dataclass
class BaselineMetrics:
    """Container for baseline metrics."""

    model: str
    timestamp: str
    vllm_params: Dict[str, Any]
    benchmark_params: Dict[str, Any]

    metrics: Dict[str, Any] = field(default_factory=dict)
    gpu_info: Dict[str, Any] = field(default_factory=dict)
    memory_samples: List[float] = field(default_factory=list)
    gpu_util_samples: List[float] = field(default_factory=list)
    temp_samples: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
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

        if self.temp_samples:
            max_temp = max(self.temp_samples)
            avg_temp = sum(self.temp_samples) / len(self.temp_samples)
        else:
            max_temp = avg_temp = 0

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
                "max_temperature_c": max_temp,
                "average_temperature_c": avg_temp,
            },
            "gpu_info": self.gpu_info,
            "memory_samples_count": len(self.memory_samples),
            "gpu_util_samples_count": len(self.gpu_util_samples),
            "temp_samples_count": len(self.temp_samples),
        }


class VLLMBaselineRunner:
    """Runner for generating vLLM baseline metrics."""

    VLLM_DEFAULT_PARAMS: Dict[str, Any] = {
        "gpu_memory_utilization": 0.9,
        "max_num_seqs": 128,
        "max_num_batched_tokens": 2048,
        "tensor_parallel_size": 1,
        "pipeline_parallel_size": 1,
    }

    def __init__(self, config: BaselineConfig):
        self.config = config
        self.process: Optional[subprocess.Popen] = None
        self.base_url = f"http://{config.host}:{config.port}"
        self.gpu_collector = GPUCollector(device_ids=config.gpu_ids)
        self.metrics = BaselineMetrics(
            model=config.model,
            timestamp=datetime.now().isoformat(),
            vllm_params=self.VLLM_DEFAULT_PARAMS.copy(),
            benchmark_params={
                "num_requests": config.num_requests,
                "concurrency": config.concurrency,
                "warmup_requests": config.warmup,
                "max_tokens": config.max_tokens,
            },
        )
        self.benchmark_start_time: Optional[datetime] = None
        self.benchmark_end_time: Optional[datetime] = None
        self.monitoring_task: Optional[asyncio.Task] = None

    def _build_vllm_command(self) -> List[str]:
        """Build vLLM server command with default parameters."""
        config = self.config
        params = self.VLLM_DEFAULT_PARAMS.copy()
        params["_trial_id"] = "baseline"

        cmd = [
            "python",
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            config.model,
            "--host",
            config.host,
            "--port",
            str(config.port),
            "--gpu-memory-utilization",
            str(params["gpu_memory_utilization"]),
            "--max-num-seqs",
            str(params["max_num_seqs"]),
            "--max-num-batched-tokens",
            str(params["max_num_batched_tokens"]),
        ]

        if len(config.gpu_ids) > 1:
            cmd.extend(["--tensor-parallel-size", str(params["tensor_parallel_size"])])

        cmd.append("--disable-log-requests")

        return cmd

    async def _start_vllm_server(self) -> subprocess.Popen:
        """Start vLLM server with environment setup."""
        logger.info("Starting vLLM server with default parameters...")

        self.gpu_collector.initialize()

        log_dir = Path(self.config.output_dir) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / "vllm_baseline.log"

        cmd = self._build_vllm_command()
        logger.info(f"vLLM command: {' '.join(cmd)}")

        env = os.environ.copy()
        visible_devices = ",".join(str(gpu_id) for gpu_id in self.config.gpu_ids)
        env["CUDA_VISIBLE_DEVICES"] = visible_devices

        with open(log_path, "w") as log_file:
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
                async with httpx.AsyncClient(timeout=5.0) as client:
                    for endpoint in ["/health", "/v1/health", "/v1/models"]:
                        try:
                            response = await client.get(f"{self.base_url}{endpoint}")
                            if response.status_code == 200:
                                logger.info(f"vLLM server ready at {self.base_url}{endpoint}")
                                return True
                        except httpx.HTTPStatusError as e:
                            if e.response.status_code != 404:
                                logger.debug(f"Health check {endpoint}: {e.response.status_code}")
            except (httpx.ConnectError, httpx.ConnectTimeout):
                pass

            await asyncio.sleep(2.0)

        logger.warning(f"vLLM server did not become ready within {timeout}s")
        return False

    def _load_prompts(self) -> List[str]:
        """Load prompts from Alpaca dataset."""
        logger.info(f"Loading prompts from {self.config.dataset}...")

        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("datasets package is required. Install with: pip install datasets")

        try:
            dataset = load_dataset(self.config.dataset, split="train")
            logger.info(f"Loaded {len(dataset)} examples from dataset")
        except Exception as e:
            raise RuntimeError(f"Failed to load dataset: {e}")

        prompts = []
        for example in dataset:
            instruction = example.get("instruction", "")
            input_text = example.get("input", "")

            if input_text:
                prompt = f"{instruction}\n\n{input_text}"
            else:
                prompt = instruction

            if prompt.strip():
                prompts.append(prompt)

            if len(prompts) >= self.config.num_requests:
                break

        if len(prompts) < self.config.num_requests:
            raise ValueError(
                f"Not enough prompts in dataset: found {len(prompts)}, need {self.config.num_requests}"
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
                    self.metrics.temp_samples.append(stats.temperature_c)

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
            "max_tokens": self.config.max_tokens,
            "temperature": 1.0,
            "top_p": 1.0,
        }

        try:
            async with client.stream("POST", url, json=payload, timeout=300) as response:
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
        stop_event: asyncio.Event,
    ) -> Dict[str, Any]:
        """Run benchmark with concurrent requests."""
        num_prompts = len(prompts)
        logger.info(
            f"Running benchmark with {num_prompts} prompts, concurrency={self.config.concurrency}"
        )

        metrics_tracker.start_benchmark()

        async with httpx.AsyncClient() as client:
            semaphore = asyncio.Semaphore(self.config.concurrency)

            async def execute_request(prompt: str, idx: int):
                request_id = f"req_{idx}"
                async with semaphore:
                    return await self._send_request(prompt, client, metrics_tracker, request_id)

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
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        self.metrics.gpu_info = self.gpu_collector.get_aggregate_stats()

        json_path = output_dir / "baseline_metrics.json"
        with open(json_path, "w") as f:
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
                "peak_memory_mb": (
                    max(self.metrics.memory_samples) if self.metrics.memory_samples else 0
                ),
                "average_memory_mb": (
                    sum(self.metrics.memory_samples) / len(self.metrics.memory_samples)
                    if self.metrics.memory_samples
                    else 0
                ),
                "average_gpu_utilization": (
                    sum(self.metrics.gpu_util_samples) / len(self.metrics.gpu_util_samples)
                    if self.metrics.gpu_util_samples
                    else 0
                ),
                "max_temperature_c": (
                    max(self.metrics.temp_samples) if self.metrics.temp_samples else 0
                ),
                "num_completed": self.metrics.metrics.get("requests_completed", 0),
                "duration_seconds": self.metrics.metrics.get("duration_seconds", 0),
            },
        }

        yaml_path = output_dir / "baseline_config.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False)
        logger.info(f"YAML output: {yaml_path}")

        self._generate_text_summary(output_dir)

    def _generate_text_summary(self, output_dir: Path):
        """Generate human-readable text summary."""
        metrics_dict = self.metrics.to_dict()["metrics"]
        summary = f"""
{"=" * 80}
BASELINE METRICS FOR {self.metrics.model.upper()}
{"=" * 80}

Configuration:
  - vLLM Parameters (Defaults):
    ├── gpu_memory_utilization: {self.metrics.vllm_params["gpu_memory_utilization"]}
    ├── max_num_seqs: {self.metrics.vllm_params["max_num_seqs"]}
    ├── max_num_batched_tokens: {self.metrics.vllm_params["max_num_batched_tokens"]}
    └── tensor_parallel_size: {self.metrics.vllm_params["tensor_parallel_size"]}

  - Benchmark Parameters:
    ├── num_requests: {self.metrics.benchmark_params["num_requests"]}
    ├── concurrency: {self.metrics.benchmark_params["concurrency"]}
    ├── warmup_requests: {self.metrics.benchmark_params["warmup_requests"]}
    └── max_tokens: {self.metrics.benchmark_params["max_tokens"]}

Performance Metrics:
  Throughput:         {metrics_dict.get("throughput_requests_per_sec", 0):.2f} requests/sec  ({metrics_dict.get("throughput_tokens_per_sec", 0):.0f} tokens/sec)
  Avg Latency:        {metrics_dict.get("avg_latency_ms", 0):.2f} ms
  P50 Latency:        {metrics_dict.get("p50_latency_ms", 0):.2f} ms
  P95 Latency:        {metrics_dict.get("p95_latency_ms", 0):.2f} ms
  P99 Latency:        {metrics_dict.get("p99_latency_ms", 0):.2f} ms
  TTFT:               {metrics_dict.get("avg_ttft_ms", 0):.2f} ms

GPU Metrics:
  Initial Memory:     {metrics_dict.get("initial_memory_mb", 0):.0f} MB
  Peak Memory:        {metrics_dict.get("peak_memory_mb", 0):.0f} MB
  Avg Memory:         {metrics_dict.get("average_memory_mb", 0):.0f} MB
  Utilization:        {(metrics_dict.get("peak_memory_mb", 0) / self.metrics.gpu_info.get("total_memory_mb", 1) * 100):.1f}%
  Avg GPU Util:       {metrics_dict.get("average_gpu_utilization", 0) * 100:.1f}%
  Max Temperature:    {metrics_dict.get("max_temperature_c", 0):.1f} C

Requests:  {metrics_dict.get("requests_completed", 0)} completed / {self.metrics.benchmark_params["num_requests"]} total
Duration:  {metrics_dict.get("duration_seconds", 0):.2f} seconds

Generated: {self.metrics.timestamp}
{"=" * 80}
"""

        text_path = output_dir / "baseline_summary.txt"
        with open(text_path, "w") as f:
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
            warmup_prompts = prompts[: self.config.warmup]
            main_prompts = prompts[: self.config.num_requests]

            self.monitoring_task = asyncio.create_task(self._monitor_gpu(stop_event))

            if self.config.warmup > 0:
                logger.info(f"Running warmup with {len(warmup_prompts)} requests...")
                warmup_metrics = VLLMMetricsTracker()
                await self._run_benchmark(warmup_prompts, warmup_metrics, stop_event)
                logger.info("Warmup completed")

            self.benchmark_start_time = datetime.now()
            logger.info(
                f"Starting main benchmark with {len(main_prompts)} requests, concurrency={self.config.concurrency}"
            )

            main_metrics = VLLMMetricsTracker()
            benchmark_results = await self._run_benchmark(main_prompts, main_metrics, stop_event)

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

            print_summary(self.metrics.to_dict())

            logger.info("Baseline generation completed successfully")

        except Exception as e:
            logger.error(f"Baseline generation failed: {e}", exc_info=True)
            cleanup()
            raise

        finally:
            cleanup()
            await self._stop_server()
            print(f"\nOutput directory: {self.config.output_dir}")


def print_summary(metrics_dict: Dict[str, Any]):
    """Print a quick summary to console."""
    print("\n" + "=" * 80)
    print("BASELINE METRICS SUMMARY")
    print("=" * 80)

    config = metrics_dict["configuration"]
    vllm_params = config["vllm_params"]
    bench_params = config["benchmark_params"]
    metrics = metrics_dict["metrics"]

    print(f"\nModel: {metrics_dict['model']}")
    print(f"Timestamp: {metrics_dict['timestamp']}")

    print("\nvLLM Parameters (Defaults):")
    for key, value in vllm_params.items():
        print(f"  {key}: {value}")

    print("\nBenchmark Parameters:")
    for key, value in bench_params.items():
        print(f"  {key}: {value}")

    print("\nPerformance:")
    print(f"  Throughput: {metrics.get('throughput_requests_per_sec', 0):.2f} req/s")
    print(f"  Tokens/sec: {metrics.get('throughput_tokens_per_sec', 0):.0f}")
    print(f"  Avg Latency: {metrics.get('avg_latency_ms', 0):.2f} ms")
    print(f"  P95 Latency: {metrics.get('p95_latency_ms', 0):.2f} ms")

    print("\nGPU Metrics:")
    print(f"  Peak Memory: {metrics.get('peak_memory_mb', 0):.0f} MB")
    print(f"  Avg GPU Util: {metrics.get('average_gpu_utilization', 0) * 100:.1f}%")
    print(f"  Max Temperature: {metrics.get('max_temperature_c', 0):.1f} C")

    print(
        f"\nRequests: {metrics.get('requests_completed', 0)} / {bench_params['num_requests']} completed"
    )
    print(f"Duration: {metrics.get('duration_seconds', 0):.2f} s")

    print("\n" + "=" * 80 + "\n")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate baseline metrics for vLLM models using default parameters",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--model",
        type=str,
        default="gpt2",
        help="Model name or HuggingFace path",
    )

    parser.add_argument(
        "--gpu-ids",
        type=str,
        default="0",
        help="Comma-separated list of GPU device IDs",
    )

    parser.add_argument(
        "--num-requests",
        type=int,
        default=1000,
        help="Number of benchmark requests",
    )

    parser.add_argument(
        "--concurrency",
        type=int,
        default=10,
        help="Number of concurrent requests",
    )

    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="Number of warmup requests",
    )

    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Maximum output tokens per request",
    )

    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="vLLM server host",
    )

    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="vLLM server port",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="tatsu-lab/alpaca",
        help="HuggingFace dataset for prompts",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for baseline results",
    )

    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    return parser.parse_args()


async def main():
    """Main entry point."""
    args = parse_args()

    logging.getLogger().setLevel(getattr(logging, args.log_level))

    gpu_ids = [int(gid.strip()) for gid in args.gpu_ids.split(",")]

    config = BaselineConfig(
        model=args.model,
        gpu_ids=gpu_ids,
        num_requests=args.num_requests,
        concurrency=args.concurrency,
        warmup=args.warmup,
        max_tokens=args.max_tokens,
        host=args.host,
        port=args.port,
        dataset=args.dataset,
        output_dir=args.output_dir,
        log_level=args.log_level,
    )

    runner = VLLMBaselineRunner(config)

    try:
        await runner.run()
        return 0
    except Exception as e:
        logger.error(f"Failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
