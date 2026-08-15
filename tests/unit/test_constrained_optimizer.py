"""Equal-budget and failure-safe constrained search tests."""

import pytest

from vllm_tuner.config.models import TuningConfig
from vllm_tuner.experiment.models import TrialResult, TrialStatus
from vllm_tuner.optimization.search_space import VLLMSearchSpace
from vllm_tuner.runtime.failures import UnsafeCleanupError
from vllm_tuner.tuning.optimizer import (
    ConstrainedSearchController,
    SearchMethod,
    SearchTrial,
)


def _result(trial_id: str, params: dict, feasible: bool = True) -> TrialResult:
    return TrialResult(
        trial_id=trial_id,
        method=trial_id.split("-")[0],
        status=TrialStatus.COMPLETE if feasible else TrialStatus.INFEASIBLE,
        params=params,
        client={"goodput_requests_per_sec": float(params.get("max_num_seqs", 1))},
        constraints={"feasible": feasible, "violations": [] if feasible else ["oom"]},
        cleanup_status={"clean": True},
    )


@pytest.mark.asyncio
async def test_default_random_and_tpe_use_equal_evaluation_budget() -> None:
    controller = ConstrainedSearchController(VLLMSearchSpace(TuningConfig()), budget=3, seed=4)

    async def evaluator(params, trial_id):
        return _result(trial_id, params)

    runs = await controller.run_all(evaluator)
    assert {run.evaluated_count for run in runs.values()} == {3}
    assert all(run.best is not None for run in runs.values())


@pytest.mark.asyncio
async def test_search_methods_are_seeded_and_interleaved_to_reduce_drift() -> None:
    async def execution_order() -> list[str]:
        controller = ConstrainedSearchController(VLLMSearchSpace(TuningConfig()), budget=3, seed=17)
        observed: list[str] = []

        async def evaluator(params, trial_id):
            observed.append(trial_id)
            return _result(trial_id, params)

        await controller.run_all(evaluator)
        return observed

    first = await execution_order()
    second = await execution_order()

    assert first == second
    assert len(first) == 9
    for offset in range(0, len(first), 3):
        assert {trial_id.split("-", 1)[0] for trial_id in first[offset : offset + 3]} == {
            "default",
            "random",
            "tpe",
        }


@pytest.mark.asyncio
async def test_infeasible_trial_is_never_selected_best() -> None:
    controller = ConstrainedSearchController(VLLMSearchSpace(TuningConfig()), budget=2, seed=4)
    calls = 0

    async def evaluator(params, trial_id):
        nonlocal calls
        calls += 1
        return _result(trial_id, params, feasible=calls != 1)

    run = await controller.run_method(SearchMethod.RANDOM, evaluator)
    assert run.trials[0].status is TrialStatus.INFEASIBLE
    assert run.best is not None
    assert run.best.status is TrialStatus.COMPLETE


@pytest.mark.asyncio
async def test_returned_failed_trial_is_recorded_and_does_not_consume_budget() -> None:
    controller = ConstrainedSearchController(VLLMSearchSpace(TuningConfig()), budget=1, seed=4)
    calls = 0

    async def evaluator(params, trial_id):
        nonlocal calls
        calls += 1
        if calls == 1:
            return TrialResult(
                trial_id=trial_id,
                method="random",
                status=TrialStatus.FAILED,
                params=params,
                constraints={"feasible": False, "violations": ["server_exit"]},
                failure_reason={"type": "SERVER_EXIT", "message": "worker died"},
            )
        return _result(trial_id, params)

    run = await controller.run_method(SearchMethod.RANDOM, evaluator)

    assert [trial.status for trial in run.trials] == [
        TrialStatus.FAILED,
        TrialStatus.COMPLETE,
    ]
    assert run.evaluated_count == 1


@pytest.mark.asyncio
async def test_repeat_failures_remain_explicit_and_later_repeats_continue() -> None:
    controller = ConstrainedSearchController(VLLMSearchSpace(TuningConfig()), budget=1, seed=4)
    candidate = SearchTrial(
        number=0,
        method=SearchMethod.TPE,
        params={"max_num_seqs": 8},
        status=TrialStatus.COMPLETE,
        objective=1.0,
    )
    calls = 0

    async def evaluator(params, trial_id):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("transport failed")
        if calls == 2:
            return TrialResult(
                trial_id=trial_id,
                method="tpe",
                status=TrialStatus.FAILED,
                params=params,
                constraints={"feasible": False, "violations": ["request_error"]},
                failure_reason={"type": "REQUEST_ERROR", "message": "bad stream"},
            )
        return _result(trial_id, params)

    repeated = await controller.repeat_candidates([candidate], evaluator, repeats=3)

    assert [trial.status for trial in repeated] == [
        TrialStatus.FAILED,
        TrialStatus.FAILED,
        TrialStatus.COMPLETE,
    ]


@pytest.mark.asyncio
async def test_unsafe_cleanup_aborts_before_another_search_attempt() -> None:
    controller = ConstrainedSearchController(VLLMSearchSpace(TuningConfig()), budget=2, seed=4)
    calls = 0

    async def evaluator(params, trial_id):
        nonlocal calls
        calls += 1
        failed = TrialResult(
            trial_id=trial_id,
            method="random",
            status=TrialStatus.FAILED,
            params=params,
            constraints={"feasible": False, "violations": ["cleanup_error"]},
            failure_reason={"type": "CLEANUP_ERROR", "message": "residual worker"},
            cleanup_status={"clean": False},
        )
        raise UnsafeCleanupError("residual worker", result=failed)

    with pytest.raises(UnsafeCleanupError):
        await controller.run_method(SearchMethod.RANDOM, evaluator)

    assert calls == 1
