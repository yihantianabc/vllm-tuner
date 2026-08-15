"""Backward-compatible facade over the SLOTune experiment runner."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Optional

from vllm_tuner.config.models import TuningConfig
from vllm_tuner.experiment.runner import SLOTuneExperimentRunner
from vllm_tuner.tuner.optimizer import VLLMOptimizer

logger = logging.getLogger(__name__)


class StudyManager:
    """Preserve the upstream entry point while delegating to the correct pipeline."""

    def __init__(self, config: TuningConfig, study_name: str, output_dir: Path):
        self.config = config
        self.study_name = study_name
        self.output_dir = Path(output_dir)
        self.optimizer = VLLMOptimizer(
            config,
            study_name,
            storage_url=f"sqlite:///{self.output_dir / 'optuna.db'}",
        )
        self._prompts: Optional[list[str]] = None
        self._summary: dict[str, Any] = {}
        self._runner: Optional[SLOTuneExperimentRunner] = None

    async def run_study(self) -> dict[str, Any]:
        """Run the complete default/random/TPE, repeat, holdout, and report workflow."""
        self._runner = SLOTuneExperimentRunner(
            self.config,
            self.study_name,
            results_root=self.output_dir / "slotune-results",
            repository=Path.cwd(),
        )
        self._summary = await self._runner.run()
        return self._summary

    async def _run_benchmark(self, params: dict[str, Any]) -> dict[str, Any]:
        """Run one correctly instrumented candidate for compatibility callers."""
        if self._runner is None:
            direct_id = f"{self.study_name}-direct"
            self._runner = SLOTuneExperimentRunner(
                self.config,
                direct_id,
                results_root=self.output_dir / "single-trials",
                repository=Path.cwd(),
            )
            trace = await self._runner._prepare_trace()
            holdout_trace = await self._runner._prepare_trace(holdout=True)
            trace_path = self._runner._trace_file(trace, "search")
            holdout_trace_path = self._runner._trace_file(holdout_trace, "holdout")
            self._runner._initialize_artifacts(trace_path, holdout_trace_path)
            from vllm_tuner.runtime.controller import TrialController

            self._direct_controller = TrialController(
                self.config,
                trace,
                self._runner.artifacts,
                tokenizer=self._runner._load_tokenizer(),
            )
        trial_id = str(params.get("_trial_id", "direct"))
        result = await self._direct_controller.run_trial(params, trial_id, "direct")
        return {
            **result.client,
            "client": result.client,
            "engine": result.engine,
            "gpu": result.gpu,
            "constraints": result.constraints,
            "failure_reason": result.failure_reason,
            "server_alive": bool(
                result.last_server_status and result.last_server_status.get("running")
            ),
        }

    async def run_single_trial(
        self,
        params: dict[str, Any],
        result_callback: Optional[Callable[[dict[str, Any], dict[str, Any]], None]] = None,
    ) -> dict[str, Any]:
        metrics = await self._run_benchmark(params)
        if result_callback is not None:
            result_callback(params, metrics)
        return metrics

    def get_study_summary(self) -> dict[str, Any]:
        """Return the new summary or a compatibility view of a direct Optuna study."""
        if self._summary:
            return self._summary
        if self.optimizer.study is None:
            return {}
        return {
            "study_name": self.study_name,
            "num_trials": len(self.optimizer.study.trials),
            "best_trial": self.optimizer.get_best_result(),
            "top_n": self.optimizer.get_top_n_results(),
        }

    def get_best_config(self) -> Optional[dict[str, Any]]:
        best = self._summary.get("best") if self._summary else self.optimizer.get_best_result()
        if not best:
            return None
        return {
            "model": self.config.model,
            "vllm_params": best.get("parameters", {}),
            "metrics": {
                "goodput_requests_per_sec": best.get(
                    "goodput_requests_per_sec",
                    best.get("metrics", {}).get("goodput_requests_per_sec"),
                )
            },
        }

    async def benchmark_config(self, config: dict[str, Any]) -> dict[str, Any]:
        return await self._run_benchmark(config)

    async def cleanup(self) -> None:
        """The managed trial controller cleans resources after every trial."""
        return None
