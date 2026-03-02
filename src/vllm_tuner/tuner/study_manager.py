"""Manage vLLM tuning studies."""

import asyncio
import logging
from pathlib import Path
from typing import Optional, Callable

from vllm_tuner.config.models import TuningConfig
from vllm_tuner.tuner.optimizer import VLLMOptimizer
from vllm_tuner.benchmarks.alpaca import create_alpaca_workload
from vllm_tuner.benchmarks.request_generator import BenchmarkRunner
from vllm_tuner.profiling.gpu_collector import GPUCollector
from vllm_tuner.profiling.vllm_metrics import VLLMMetricsTracker
from vllm_tuner.vllm.launcher import VLLMLauncher
from vllm_tuner.vllm.telemetry import VLLMTelemetryParser

logger = logging.getLogger(__name__)


class StudyManager:
    """Manage a vLLM tuning study from end to end."""

    def __init__(
        self,
        config: TuningConfig,
        study_name: str,
        output_dir: Path,
    ):
        self.config = config
        self.study_name = study_name
        self.output_dir = output_dir
        self.metrics_tracker = VLLMMetricsTracker()

        logs_dir = output_dir / "logs"
        self.launcher = VLLMLauncher(
            config,
            log_dir=logs_dir,
        )

        self.gpu_collector = GPUCollector(
            device_ids=config.gpu.device_ids,
        )

        self.optimizer = VLLMOptimizer(
            config,
            study_name,
            storage_url=f"sqlite:///{output_dir / 'optuna.db'}",
        )

        self.workload = create_alpaca_workload(config.workload)

        self._prompts: Optional[list[str]] = None

    async def run_study(self) -> dict:
        """Run the full tuning study."""
        logger.info(f"Starting tuning study: {self.study_name}")

        try:
            self._prompts = await self.workload.get_prompts()
            logger.info(f"Loaded {len(self._prompts)} prompts from workload")

            async def benchmark_func(params: dict) -> dict:
                return await self._run_benchmark(params)

            result = self.optimizer.optimize(
                benchmark_func,
                n_trials=self.config.study.min_trials,
                timeout_seconds=self.config.study.timeout_minutes * 60,
            )

            return result

        except Exception as e:
            logger.error(f"Study failed: {e}")
            raise

        finally:
            if self.launcher.is_running():
                await self.launcher.stop()

            self.gpu_collector.shutdown()

    async def _run_benchmark(self, params: dict) -> dict:
        """
        Run a single benchmark trial.

        Args:
            params: vLLM parameters for this trial

        Returns:
            Dictionary of benchmark metrics
        """
        trial_id = params.get("_trial_id", "unknown")
        logger.info(f"Running benchmark for trial {trial_id}")

        try:
            await self.launcher.start(params)

            ready = await self.launcher.wait_ready(timeout=600)
            if not ready:
                raise RuntimeError("vLLM server did not become ready")

            gpu_stats = self.gpu_collector.collect_all()
            initial_memory = sum(s.memory_used_mb for s in gpu_stats)

            runner = BenchmarkRunner(
                self.launcher,
                concurrency=self.config.workload.concurrent_requests,
                warmup_requests=self.config.workload.warmup_requests,
            )

            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop and loop.is_running():
                metrics = await runner.run_benchmark(self._prompts)
            else:
                metrics = asyncio.run(runner.run_benchmark(self._prompts))

            final_stats = self.gpu_collector.collect_all()
            final_memory = sum(s.memory_used_mb for s in final_stats)
            aggregate_stats = self.gpu_collector.get_aggregate_stats()

            log_file = self.launcher.get_log_file(trial_id)
            telemetry_parser = VLLMTelemetryParser()
            telemetry = telemetry_parser.parse_log_file(log_file)

            all_memory_samples = []
            for device_id, history in self.gpu_collector.history.items():
                if history:
                    for stats in history:
                        all_memory_samples.append(stats.memory_used_mb)

            average_memory = (
                sum(all_memory_samples) / len(all_memory_samples)
                if all_memory_samples
                else 0
            )

            combined_metrics = {
                **metrics,
                **aggregate_stats,
                **telemetry,
                "initial_memory_mb": initial_memory,
                "peak_memory_mb": final_memory,
                "average_memory_mb": average_memory,
                "memory_delta_mb": final_memory - initial_memory,
                "vllm_params": params,
            }

            return combined_metrics

        except Exception as e:
            logger.error(f"Benchmark failed for trial {trial_id}: {e}")
            return {
                "error": str(e),
                "throughput_requests_per_sec": 0.0,
                "avg_latency_ms": float("inf"),
                "aggregate_memory_utilization": 1.0,
                "oom_detected": True,
            }

        finally:
            if self.launcher.is_running():
                await self.launcher.stop()

            self.gpu_collector.clear_history()

    async def run_single_trial(
        self,
        params: dict,
        result_callback: Optional[Callable] = None,
    ) -> dict:
        """Run a single trial and optionally call callback with results."""
        metrics = await self._run_benchmark(params)

        if result_callback:
            result_callback(params, metrics)

        return metrics

    def get_study_summary(self) -> dict:
        """Get summary of the study."""
        if self.optimizer.study is None:
            return {}

        summary = {
            "study_name": self.study_name,
            "num_trials": len(self.optimizer.study.trials),
            "best_trial": self.optimizer.get_best_result(),
            "top_n": self.optimizer.get_top_n_results(),
        }

        return summary

    def get_best_config(self) -> Optional[dict]:
        """Get the best configuration from the study."""
        best_result = self.optimizer.get_best_result()

        if not best_result:
            return None

        config = {
            "model": self.config.model,
            "vllm_params": best_result["parameters"],
            "metrics": best_result["metrics"],
        }

        return config

    async def benchmark_config(self, config: dict) -> dict:
        """Benchmark a specific configuration without optimization."""
        logger.info(f"Benchmarking specific config: {config.get('model', 'unknown')}")

        return await self._run_benchmark(config)

    async def cleanup(self) -> None:
        """Clean up resources."""
        logger.info("Cleaning up study resources")

        if self.launcher.is_running():
            await self.launcher.stop()

        self.gpu_collector.shutdown()

        if self.workload:
            self.workload.unload()
