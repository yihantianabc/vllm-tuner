"""vLLM parameter optimization using Optuna multi-objective optimization."""

import asyncio
import logging
from typing import Dict, Any, Optional, Union

import optuna

from ..config.models import TuningConfig
from ..optimization.search_space import VLLMSearchSpace

logger = logging.getLogger(__name__)


class VLLMOptimizer:
    """Optimize vLLM parameters using Optuna."""

    def __init__(
        self,
        config: TuningConfig,
        study_name: str,
        storage_url: str = "sqlite:///studies/optuna.db",
        directions: Optional[list[str]] = None,
    ):
        self.config = config
        self.study_name = study_name
        self.storage_url = storage_url
        self.search_space = VLLMSearchSpace(config, config.gpu.count)

        if directions is None:
            objectives = config.objectives
            directions = [
                "maximize" if objectives.throughput > 0 else "ignore",
                "minimize" if objectives.latency > 0 else "ignore",
                "minimize" if objectives.memory > 0 else "ignore",
            ]

        self.directions = directions
        self.study: Optional[optuna.Study] = None

    def create_study(
        self,
        load_if_exists: bool = True,
    ) -> optuna.Study:
        """Create or load Optuna study."""
        logger.info(f"Creating/loading Optuna study: {self.study_name}")

        sampler = optuna.samplers.TPESampler(
            multivariate=True,
            seed=2026,
        )

        pruner = None
        directions = [d for d in self.directions if d != "ignore"]

        if self.config.study.prune_enabled:
            pruner = optuna.pruners.MedianPruner(
                n_startup_trials=self.config.study.n_startup_trials,
                n_warmup_steps=5,
                interval_steps=1,
            )

        if len(directions) == 1:
            direction = directions[0]
            self.study = optuna.create_study(
                study_name=self.study_name,
                storage=self.storage_url,
                load_if_exists=load_if_exists,
                direction=direction,
                sampler=sampler,
                pruner=pruner,
            )
        else:
            self.study = optuna.create_study(
                study_name=self.study_name,
                storage=self.storage_url,
                load_if_exists=load_if_exists,
                directions=directions,
                sampler=sampler,
                pruner=pruner,
            )

        logger.info(f"Study created with {len(self.study.trials)} trials")
        return self.study

    def compute_objective(
        self,
        metrics: Dict[str, Any],
    ) -> Dict[str, float]:
        """Compute objective values from metrics."""
        objectives = {}

        throughput = metrics.get("throughput_requests_per_sec", 0.0)
        latency = metrics.get("avg_latency_ms", float("inf"))
        memory_util = metrics.get("aggregate_memory_utilization", 0.0)
        oom = metrics.get("oom_detected", False)
        oom_errors = metrics.get("oom_errors", 0)

        if oom or oom_errors > 0:
            return {
                "throughput": 0.0,
                "latency": float("inf"),
                "memory": 1.0,
            }

        if self.config.objectives.throughput > 0:
            objectives["throughput"] = throughput

        if self.config.objectives.latency > 0:
            objectives["latency"] = latency

        if self.config.objectives.memory > 0:
            objectives["memory"] = memory_util

        return objectives

    def apply_constraints(
        self,
        metrics: Dict[str, Any],
        params: Dict[str, Any],
    ) -> bool:
        """Check if trial satisfies constraints."""
        constraints = self.config.constraints

        if constraints.max_latency_ms is not None:
            latency = metrics.get("avg_latency_ms", 0)
            if latency > constraints.max_latency_ms:
                logger.debug(
                    f"Trial violated latency constraint: {latency} > {constraints.max_latency_ms}"
                )
                return False

        if constraints.max_memory_utilization is not None:
            memory_util = metrics.get("aggregate_memory_utilization", 0)
            if memory_util > constraints.max_memory_utilization:
                logger.debug(
                    f"Trial violated memory constraint: {memory_util} > {constraints.max_memory_utilization}"
                )
                return False

        if constraints.throughput_min is not None:
            throughput = metrics.get("throughput_requests_per_sec", 0)
            if throughput < constraints.throughput_min:
                logger.debug(
                    f"Trial violated throughput constraint: {throughput} < {constraints.throughput_min}"
                )
                return False

        oom = metrics.get("oom_detected", False)
        oom_errors = metrics.get("oom_errors", 0)

        if oom or oom_errors > 0:
            logger.debug("Trial had OOM error")
            return False

        return True

    def evaluate_trial(
        self,
        trial: optuna.Trial,
        params: Dict[str, Any],
        metrics: Dict[str, Any],
    ) -> Union[float, list[float]]:
        """Evaluate a trial and return objective value(s)."""
        trial_id = str(trial.number)
        params["_trial_id"] = trial_id

        if not self.apply_constraints(metrics, params):
            if len(self.directions) == 1:
                return float("-inf")
            else:
                return [float("-inf")] * len(self.directions)

        objectives = self.compute_objective(metrics)

        logger.debug(f"Trial {trial_id} objectives: {objectives}")

        trial.set_user_attr("metrics", metrics)

        if len(self.directions) == 1:
            return list(objectives.values())[0]
        else:
            return list(objectives.values())

    def run_trial(
        self,
        trial: optuna.Trial,
        benchmark_func,
    ) -> Union[float, list[float]]:
        """Run a single optimization trial."""
        trial_id = str(trial.number)
        logger.info(f"Starting trial {trial_id}")

        params = self.search_space.apply_params(trial, {})
        params["_trial_id"] = trial_id

        try:
            coro = benchmark_func(params)
            if asyncio.iscoroutine(coro):
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)

                if loop.is_running():
                    import concurrent.futures

                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(asyncio.run, coro)
                        metrics = future.result()
                else:
                    metrics = loop.run_until_complete(coro)
            else:
                metrics = coro

            return self.evaluate_trial(trial, params, metrics)

        except Exception as e:
            logger.error(f"Trial {trial_id} failed: {e}")

            if len(self.directions) == 1:
                return float("-inf")
            else:
                return [float("-inf")] * len(self.directions)

    def optimize(
        self,
        benchmark_func,
        timeout_seconds: Optional[float] = None,
        n_trials: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Run optimization.

        Args:
            benchmark_func: Function that takes params dict and returns metrics dict
            timeout_seconds: Maximum time to run in seconds
            n_trials: Number of trials to run

        Returns:
            Dictionary with the best parameters and metrics
        """
        if self.study is None:
            self.create_study()

        logger.info(f"Starting optimization with study: {self.study_name}")

        def objective(trial: optuna.Trial) -> Union[float, list[float]]:
            trial_num = trial.number
            logger.info(f"Trial {trial_num} started")

            if (
                len([d for d in self.directions if d != "ignore"]) == 1
                and self.config.study.prune_enabled
            ):
                trial.report(0, 0)

            result = self.run_trial(trial, benchmark_func)

            completed_trials = len(
                [t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE]
            )
            total_trials = n_trials if n_trials else self.config.study.min_trials
            logger.info(
                f"Trial {trial_num} completed ({completed_trials}/{total_trials} trials done)"
            )

            return result

        if timeout_seconds is not None:
            timeout_seconds = min(
                timeout_seconds,
                self.config.study.timeout_minutes * 60,
            )
        else:
            timeout_seconds = self.config.study.timeout_minutes * 60

        if n_trials is None:
            n_trials = self.config.study.min_trials

        def progress_callback(study, trial):
            trial_num = trial.number
            completed = len(
                [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
            )
            failed = len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL])
            total = len(study.trials)
            logger.info(f"Progress: {completed} completed, {failed} failed, {total} total trials")

        self.study.optimize(
            objective,
            timeout=timeout_seconds,
            n_trials=n_trials,
            callbacks=[progress_callback],
            show_progress_bar=False,
        )

        logger.info(f"Optimization completed. Total trials: {len(self.study.trials)}")

        return self.get_best_result()

    def get_best_trial(self) -> Optional[optuna.trial.FrozenTrial]:
        """Get the best trial from the study."""
        if self.study is None or len(self.study.trials) == 0:
            return None

        if len(self.study.directions) == 1:
            return self.study.best_trial
        else:
            return self.study.best_trials[0]

    def get_best_result(
        self,
    ) -> Dict[str, Any]:
        """Get the best parameters and metrics from the study."""
        best_trial = self.get_best_trial()

        if best_trial is None:
            logger.warning("No successful trials found")
            return {}

        params = best_trial.params

        user_attrs = best_trial.user_attrs
        metrics = user_attrs.get("metrics", {})

        result = {
            "trial_number": best_trial.number,
            "value": best_trial.value if len(self.directions) == 1 else best_trial.values,
            "parameters": params,
            "metrics": metrics,
            "state": str(best_trial.state),
            "datetime_start": best_trial.datetime_start.isoformat()
            if best_trial.datetime_start
            else None,
            "datetime_complete": best_trial.datetime_complete.isoformat()
            if best_trial.datetime_complete
            else None,
        }

        value = best_trial.values if len(self.study.directions) > 1 else best_trial.value
        logger.info(f"Best trial: {best_trial.number} with value: {value}")

        return result

    def get_top_n_results(self, n: int = 3) -> list[Dict[str, Any]]:
        """Get top N results from the study."""
        if self.study is None or len(self.study.trials) == 0:
            return []

        top_n = []

        sorted_trials = sorted(
            self.study.trials,
            key=lambda t: (t.state != optuna.trial.TrialState.COMPLETE, t.number),
        )

        for trial in sorted_trials[:n]:
            result = {
                "trial_number": trial.number,
                "value": trial.values if len(self.study.directions) > 1 else trial.value,
                "parameters": trial.params,
                "metrics": trial.user_attrs.get("metrics", {}),
                "state": str(trial.state),
            }
            top_n.append(result)

        return top_n

    def get_all_trials(self) -> list[Dict[str, Any]]:
        """Get all trials from the study."""
        if self.study is None or len(self.study.trials) == 0:
            return []

        all_trials = []

        sorted_trials = sorted(
            self.study.trials,
            key=lambda t: (t.state != optuna.trial.TrialState.COMPLETE, t.number),
        )

        for trial in sorted_trials:
            result = {
                "trial_number": trial.number,
                "value": trial.values if len(self.study.directions) > 1 else trial.value,
                "parameters": trial.params,
                "metrics": trial.user_attrs.get("metrics", {}),
                "state": str(trial.state),
            }
            all_trials.append(result)

        return all_trials

    def delete_study(self) -> None:
        """Delete the study."""
        if self.study is not None:
            optuna.delete_study(
                study_name=self.study_name,
                storage=self.storage_url,
            )
            logger.info(f"Study deleted: {self.study_name}")
