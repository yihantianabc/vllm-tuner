"""Compatibility Optuna facade using SLOTune's single feasible-goodput objective."""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
from typing import Any, Callable, Optional

import optuna

from vllm_tuner.config.models import TuningConfig
from vllm_tuner.optimization.search_space import VLLMSearchSpace

logger = logging.getLogger(__name__)


class VLLMOptimizer:
    """Run a single constrained TPE study without sentinel failure values.

    New experiments use :class:`vllm_tuner.tuning.optimizer.ConstrainedSearchController` to
    compare default, random, and TPE with equal budgets. This facade keeps the original public
    API usable for callers that only request one TPE study.
    """

    def __init__(
        self,
        config: TuningConfig,
        study_name: str,
        storage_url: str = "sqlite:///studies/optuna.db",
        directions: Optional[list[str]] = None,
    ) -> None:
        if directions not in (None, ["maximize"]):
            raise ValueError("SLOTune has one objective: maximize feasible SLO goodput")
        self.config = config
        self.study_name = study_name
        self.storage_url = storage_url
        self.search_space = VLLMSearchSpace(config, config.gpu.count)
        self.directions = ["maximize"]
        self.study: Optional[optuna.Study] = None

    @staticmethod
    def _constraints_func(trial: optuna.trial.FrozenTrial) -> tuple[float, ...]:
        return tuple(float(value) for value in trial.user_attrs.get("constraint_values", [0.0]))

    def create_study(self, load_if_exists: Optional[bool] = None) -> optuna.Study:
        """Create a deterministic constrained TPE study; resume is explicit."""
        if load_if_exists is None:
            load_if_exists = self.config.study.resume
        sampler = optuna.samplers.TPESampler(
            multivariate=True,
            seed=self.config.study.seed,
            constraints_func=self._constraints_func,
        )
        pruner: Optional[optuna.pruners.BasePruner] = None
        if self.config.study.prune_enabled:
            pruner = optuna.pruners.MedianPruner(
                n_startup_trials=self.config.study.n_startup_trials,
                n_warmup_steps=5,
                interval_steps=1,
            )
        self.study = optuna.create_study(
            study_name=self.study_name,
            storage=self.storage_url,
            load_if_exists=load_if_exists,
            direction="maximize",
            sampler=sampler,
            pruner=pruner,
        )
        return self.study

    def compute_objective(self, metrics: dict[str, Any]) -> dict[str, float]:
        """Read the sole objective from correctly reduced client metrics."""
        value = metrics.get("goodput_requests_per_sec", metrics.get("request_goodput"))
        if value is None:
            raise ValueError("benchmark metrics do not contain SLO goodput")
        return {"goodput": float(value)}

    def apply_constraints(
        self,
        metrics: dict[str, Any],
        params: Optional[dict[str, Any]] = None,
    ) -> bool:
        """Return explicit feasibility; missing health evidence is not assumed healthy."""
        constraint_block = metrics.get("constraints")
        if isinstance(constraint_block, dict) and "feasible" in constraint_block:
            return bool(constraint_block["feasible"])
        if metrics.get("error") or metrics.get("oom_detected") or metrics.get("oom_errors", 0):
            return False
        if metrics.get("server_alive") is False:
            return False
        error_rate = metrics.get("error_rate")
        if error_rate is not None and float(error_rate) > self.config.constraints.max_error_rate:
            return False
        peak = metrics.get("peak_memory_mb")
        if (
            peak is not None
            and self.config.constraints.max_peak_vram_mb is not None
            and float(peak) > self.config.constraints.max_peak_vram_mb
        ):
            return False
        return True

    def evaluate_trial(
        self,
        trial: optuna.Trial,
        params: dict[str, Any],
        metrics: dict[str, Any],
    ) -> float:
        """Mark infeasible trials as pruned instead of returning negative infinity."""
        trial.set_user_attr("metrics", metrics)
        if not self.apply_constraints(metrics, params):
            violations = metrics.get("constraints", {}).get("violations", ["hard_constraint"])
            trial.set_user_attr("trial_status", "INFEASIBLE")
            trial.set_user_attr("constraint_values", [1.0 for _ in violations] or [1.0])
            raise optuna.TrialPruned("INFEASIBLE: " + ", ".join(map(str, violations)))
        objective = self.compute_objective(metrics)["goodput"]
        trial.set_user_attr("trial_status", "COMPLETE")
        trial.set_user_attr("constraint_values", [0.0])
        return objective

    @staticmethod
    def _resolve_awaitable(value: Any) -> Any:
        if not asyncio.iscoroutine(value):
            return value
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop is None or not loop.is_running():
            return asyncio.run(value)
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            return executor.submit(asyncio.run, value).result()

    def run_trial(
        self, trial: optuna.Trial, benchmark_func: Callable[[dict[str, Any]], Any]
    ) -> float:
        """Run one candidate and let Optuna record real FAIL/PRUNED states."""
        params = self.search_space.apply_params(trial, {})
        params["_trial_id"] = str(trial.number)
        try:
            metrics = self._resolve_awaitable(benchmark_func(params))
            return self.evaluate_trial(trial, params, metrics)
        except optuna.TrialPruned:
            raise
        except Exception as error:
            logger.error("Trial %s failed: %s", trial.number, error, exc_info=True)
            trial.set_user_attr("trial_status", "FAILED")
            trial.set_user_attr(
                "failure_reason", {"type": type(error).__name__, "message": str(error)}
            )
            raise

    def optimize(
        self,
        benchmark_func: Callable[[dict[str, Any]], Any],
        timeout_seconds: Optional[float] = None,
        n_trials: Optional[int] = None,
    ) -> dict[str, Any]:
        """Run TPE while allowing failed trials to be recorded and later trials to continue."""
        if self.study is None:
            self.create_study()
        assert self.study is not None
        budget = n_trials or self.config.study.trial_budget
        timeout = timeout_seconds or self.config.study.timeout_minutes * 60
        self.study.optimize(
            lambda trial: self.run_trial(trial, benchmark_func),
            timeout=timeout,
            n_trials=budget,
            catch=(Exception,),
            show_progress_bar=False,
        )
        return self.get_best_result()

    def _selectable_trials(self) -> list[optuna.trial.FrozenTrial]:
        if self.study is None:
            return []
        return [
            trial
            for trial in self.study.trials
            if trial.state == optuna.trial.TrialState.COMPLETE
            and trial.user_attrs.get("trial_status", "COMPLETE") == "COMPLETE"
            and trial.value is not None
        ]

    def get_best_trial(self) -> Optional[optuna.trial.FrozenTrial]:
        """Select only feasible COMPLETE trials."""
        candidates = self._selectable_trials()
        return max(candidates, key=self._objective_value) if candidates else None

    @staticmethod
    def _objective_value(trial: optuna.trial.FrozenTrial) -> float:
        if trial.value is None:
            raise ValueError("non-selectable Optuna trial has no objective value")
        return float(trial.value)

    @staticmethod
    def _serialize_trial(trial: optuna.trial.FrozenTrial) -> dict[str, Any]:
        return {
            "trial_number": trial.number,
            "value": trial.value,
            "parameters": trial.params,
            "metrics": trial.user_attrs.get("metrics", {}),
            "state": trial.user_attrs.get("trial_status", trial.state.name),
            "failure_reason": trial.user_attrs.get("failure_reason"),
            "datetime_start": (trial.datetime_start.isoformat() if trial.datetime_start else None),
            "datetime_complete": (
                trial.datetime_complete.isoformat() if trial.datetime_complete else None
            ),
        }

    def get_best_result(self) -> dict[str, Any]:
        trial = self.get_best_trial()
        return self._serialize_trial(trial) if trial is not None else {}

    def get_top_n_results(self, n: int = 3) -> list[dict[str, Any]]:
        trials = sorted(
            self._selectable_trials(),
            key=self._objective_value,
            reverse=True,
        )
        return [self._serialize_trial(trial) for trial in trials[:n]]

    def get_all_trials(self) -> list[dict[str, Any]]:
        if self.study is None:
            return []
        return [self._serialize_trial(trial) for trial in self.study.trials]

    def delete_study(self) -> None:
        if self.study is not None:
            optuna.delete_study(study_name=self.study_name, storage=self.storage_url)
            self.study = None
