"""Equal-budget default, random, and constrained-TPE search controller."""

from __future__ import annotations

import inspect
import logging
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable, Optional

import optuna

from vllm_tuner.experiment.models import TrialResult, TrialStatus
from vllm_tuner.runtime.failures import UnsafeCleanupError

from .search_space import VLLMSearchSpace

logger = logging.getLogger(__name__)


class SearchMethod(str, Enum):
    DEFAULT = "default"
    RANDOM = "random"
    TPE = "tpe"


@dataclass
class SearchTrial:
    """Optimizer-independent trial record that includes infeasible outcomes."""

    number: int
    method: SearchMethod
    params: dict[str, Any]
    status: TrialStatus
    objective: Optional[float] = None
    result: Optional[TrialResult] = None
    failure_reason: Optional[dict[str, Any]] = None
    repeat_of: Optional[int] = None
    holdout: bool = False

    @property
    def selectable(self) -> bool:
        return self.status == TrialStatus.COMPLETE and self.objective is not None

    @property
    def objective_value(self) -> float:
        if self.objective is None:
            raise ValueError("non-selectable trial has no objective value")
        return float(self.objective)


@dataclass
class SearchRun:
    """All evidence from one search method."""

    method: SearchMethod
    requested_budget: int
    trials: list[SearchTrial] = field(default_factory=list)

    @property
    def evaluated_count(self) -> int:
        return sum(
            trial.status in {TrialStatus.COMPLETE, TrialStatus.INFEASIBLE} for trial in self.trials
        )

    @property
    def best(self) -> Optional[SearchTrial]:
        selectable = [trial for trial in self.trials if trial.selectable]
        return max(selectable, key=lambda trial: trial.objective_value) if selectable else None


Evaluator = Callable[[dict[str, Any], str], TrialResult | Awaitable[TrialResult]]


def _constraints_func(trial: optuna.trial.FrozenTrial) -> tuple[float, ...]:
    values = trial.user_attrs.get("constraint_values", [0.0])
    return tuple(float(value) for value in values)


class ConstrainedSearchController:
    """Run comparable search methods without encoding failures as objective values."""

    def __init__(
        self,
        search_space: VLLMSearchSpace,
        *,
        budget: int,
        seed: int = 2026,
        max_failed_attempt_factor: int = 5,
        n_startup_trials: int = 5,
        prune_enabled: bool = False,
    ) -> None:
        if budget <= 0:
            raise ValueError("budget must be positive")
        if n_startup_trials < 0:
            raise ValueError("n_startup_trials must be non-negative")
        self.search_space = search_space
        self.budget = budget
        self.seed = seed
        self.n_startup_trials = n_startup_trials
        self.prune_enabled = prune_enabled
        self.max_attempts = max(budget, budget * max_failed_attempt_factor)

    def _study(self, method: SearchMethod) -> optuna.Study:
        if method == SearchMethod.RANDOM:
            sampler: optuna.samplers.BaseSampler = optuna.samplers.RandomSampler(seed=self.seed)
        elif method == SearchMethod.TPE:
            sampler = optuna.samplers.TPESampler(
                seed=self.seed,
                n_startup_trials=self.n_startup_trials,
                multivariate=True,
                constraints_func=_constraints_func,
            )
        else:
            sampler = optuna.samplers.RandomSampler(seed=self.seed)
        pruner: optuna.pruners.BasePruner
        if self.prune_enabled:
            pruner = optuna.pruners.MedianPruner(
                n_startup_trials=self.n_startup_trials,
            )
        else:
            pruner = optuna.pruners.NopPruner()
        return optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)

    async def _evaluate(
        self, evaluator: Evaluator, params: dict[str, Any], trial_id: str
    ) -> TrialResult:
        value = evaluator(params, trial_id)
        if inspect.isawaitable(value):
            value = await value
        if not isinstance(value, TrialResult):
            raise TypeError("search evaluator must return TrialResult")
        return value

    async def _run_attempt(
        self,
        method: SearchMethod,
        evaluator: Evaluator,
        run: SearchRun,
        study: optuna.Study,
        attempt: int,
    ) -> None:
        """Execute one method attempt while preserving failures as evidence."""
        optuna_trial = study.ask()
        if method == SearchMethod.DEFAULT:
            params = self.search_space.get_default_params()
        else:
            params = self.search_space.apply_params(optuna_trial, {})
        trial_id = f"{method.value}-{attempt:04d}"
        try:
            result = await self._evaluate(evaluator, params, trial_id)
        except UnsafeCleanupError:
            raise
        except optuna.TrialPruned as error:
            study.tell(optuna_trial, state=optuna.trial.TrialState.PRUNED)
            run.trials.append(
                SearchTrial(
                    attempt,
                    method,
                    params,
                    TrialStatus.PRUNED,
                    failure_reason={"type": "PRUNED", "message": str(error)},
                )
            )
            return
        except Exception as error:
            study.tell(optuna_trial, state=optuna.trial.TrialState.FAIL)
            run.trials.append(
                SearchTrial(
                    attempt,
                    method,
                    params,
                    TrialStatus.FAILED,
                    failure_reason={
                        "type": type(error).__name__,
                        "message": str(error),
                    },
                )
            )
            return

        objective = result.client.get("goodput_requests_per_sec")
        if result.status == TrialStatus.FAILED:
            optuna_trial.set_user_attr("trial_status", TrialStatus.FAILED.value)
            if result.failure_reason is not None:
                optuna_trial.set_user_attr("failure_reason", result.failure_reason)
            study.tell(optuna_trial, state=optuna.trial.TrialState.FAIL)
            value = None
            status = TrialStatus.FAILED
        elif result.status == TrialStatus.PRUNED:
            optuna_trial.set_user_attr("trial_status", TrialStatus.PRUNED.value)
            study.tell(optuna_trial, state=optuna.trial.TrialState.PRUNED)
            value = None
            status = TrialStatus.PRUNED
        elif result.selectable and objective is not None:
            value = float(objective)
            optuna_trial.set_user_attr("constraint_values", [0.0])
            optuna_trial.set_user_attr("trial_status", TrialStatus.COMPLETE.value)
            study.tell(optuna_trial, value)
            status = TrialStatus.COMPLETE
        else:
            violations = result.constraints.get("violations", ["infeasible"])
            optuna_trial.set_user_attr("constraint_values", [1.0 for _ in violations] or [1.0])
            optuna_trial.set_user_attr("trial_status", TrialStatus.INFEASIBLE.value)
            # A finite value lets constrained TPE learn while manual selection excludes it.
            study.tell(optuna_trial, float(objective or 0.0))
            value = None
            status = TrialStatus.INFEASIBLE
        run.trials.append(
            SearchTrial(
                attempt,
                method,
                params,
                status,
                objective=value,
                result=result,
                failure_reason=result.failure_reason,
            )
        )

    async def run_method(self, method: SearchMethod | str, evaluator: Evaluator) -> SearchRun:
        """Collect the same number of measured outcomes for one method."""
        method = SearchMethod(method)
        run = SearchRun(method=method, requested_budget=self.budget)
        study = self._study(method)
        attempt = 0
        while run.evaluated_count < self.budget and attempt < self.max_attempts:
            await self._run_attempt(method, evaluator, run, study, attempt)
            attempt += 1

        if run.evaluated_count != self.budget:
            raise RuntimeError(
                f"{method.value} produced {run.evaluated_count}/{self.budget} evaluated trials "
                f"after {attempt} attempts"
            )
        return run

    async def run_all(
        self,
        evaluator: Evaluator,
        methods: tuple[SearchMethod | str, ...] = (
            SearchMethod.DEFAULT,
            SearchMethod.RANDOM,
            SearchMethod.TPE,
        ),
    ) -> dict[SearchMethod, SearchRun]:
        """Interleave methods in a seeded order while keeping GPU trials sequential."""
        canonical_methods = [SearchMethod(method) for method in methods]
        if len(set(canonical_methods)) != len(canonical_methods):
            raise ValueError("search methods must be unique")
        runs = {
            method: SearchRun(method=method, requested_budget=self.budget)
            for method in canonical_methods
        }
        studies = {method: self._study(method) for method in canonical_methods}
        attempts = {method: 0 for method in canonical_methods}
        order_rng = random.Random(self.seed + 13_337)
        while any(run.evaluated_count < self.budget for run in runs.values()):
            active = [
                method for method in canonical_methods if runs[method].evaluated_count < self.budget
            ]
            order_rng.shuffle(active)
            for method in active:
                attempt = attempts[method]
                if attempt >= self.max_attempts:
                    raise RuntimeError(
                        f"{method.value} produced {runs[method].evaluated_count}/{self.budget} "
                        f"evaluated trials after {attempt} attempts"
                    )
                await self._run_attempt(
                    method,
                    evaluator,
                    runs[method],
                    studies[method],
                    attempt,
                )
                attempts[method] += 1
        counts = {run.evaluated_count for run in runs.values()}
        if counts != {self.budget}:
            raise RuntimeError(f"search methods did not use equal budgets: {sorted(counts)}")
        return runs

    @staticmethod
    def best_across(runs: dict[SearchMethod, SearchRun]) -> Optional[SearchTrial]:
        """Select only feasible COMPLETE trials."""
        candidates = [run.best for run in runs.values() if run.best is not None]
        return max(candidates, key=lambda trial: trial.objective_value) if candidates else None

    async def repeat_candidates(
        self,
        candidates: list[SearchTrial],
        evaluator: Evaluator,
        *,
        repeats: int = 3,
        holdout: bool = False,
    ) -> list[SearchTrial]:
        """Re-run selected configs without changing their parameters."""
        repeated: list[SearchTrial] = []
        jobs = [(candidate, repeat) for candidate in candidates for repeat in range(repeats)]
        order_rng = random.Random(self.seed + (29_003 if holdout else 23_011))
        order_rng.shuffle(jobs)
        for candidate, repeat in jobs:
            phase = "holdout" if holdout else "repeat"
            trial_id = f"{phase}-{candidate.method.value}-{candidate.number}-{repeat}"
            try:
                result = await self._evaluate(evaluator, dict(candidate.params), trial_id)
            except UnsafeCleanupError:
                raise
            except optuna.TrialPruned as error:
                repeated.append(
                    SearchTrial(
                        number=len(repeated),
                        method=candidate.method,
                        params=dict(candidate.params),
                        status=TrialStatus.PRUNED,
                        failure_reason={
                            "type": "PRUNED",
                            "message": str(error),
                        },
                        repeat_of=candidate.number,
                        holdout=holdout,
                    )
                )
                continue
            except Exception as error:
                repeated.append(
                    SearchTrial(
                        number=len(repeated),
                        method=candidate.method,
                        params=dict(candidate.params),
                        status=TrialStatus.FAILED,
                        failure_reason={
                            "type": type(error).__name__,
                            "message": str(error),
                        },
                        repeat_of=candidate.number,
                        holdout=holdout,
                    )
                )
                continue

            objective = result.client.get("goodput_requests_per_sec") if result.selectable else None
            status = result.status
            if status == TrialStatus.COMPLETE and not result.selectable:
                status = TrialStatus.INFEASIBLE
            elif not status.terminal:
                status = TrialStatus.FAILED
            repeated.append(
                SearchTrial(
                    number=len(repeated),
                    method=candidate.method,
                    params=dict(candidate.params),
                    status=status,
                    objective=float(objective) if objective is not None else None,
                    result=result,
                    failure_reason=result.failure_reason,
                    repeat_of=candidate.number,
                    holdout=holdout,
                )
            )
        return repeated
