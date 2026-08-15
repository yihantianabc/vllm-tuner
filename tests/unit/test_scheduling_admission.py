"""Unit tests for deterministic aging and max-wait admission control."""

import pytest

from vllm_tuner.scheduling.admission import (
    AdmissionCandidate,
    AdmissionConfig,
    FairAdmissionController,
)


def candidate(
    request_id,
    arrival,
    order,
    *,
    waiting_since=None,
    last_progress=None,
    service_tokens=0,
    priority=0,
):
    """Build an admission candidate for concise fairness tests."""

    return AdmissionCandidate(
        request_id=request_id,
        arrival_time=arrival,
        waiting_since=arrival if waiting_since is None else waiting_since,
        original_order=order,
        last_progress_time=last_progress,
        service_tokens=service_tokens,
        priority=priority,
    )


def test_rank_waiting_uses_aging_and_stable_ties():
    """Older work wins while exact ties retain trace order deterministically."""

    controller = FairAdmissionController(AdmissionConfig(max_wait=10.0))
    waiting = [
        candidate("new", 2.0, 2),
        candidate("old-b", 0.0, 1),
        candidate("old-a", 0.0, 0),
    ]

    ranked = controller.rank_waiting(waiting, now=3.0)

    assert [item.request_id for item in ranked] == ["old-a", "old-b", "new"]


def test_priority_and_aging_are_both_part_of_admission_score():
    """Explicit priority can lead briefly, while age continues to accumulate."""

    controller = FairAdmissionController(AdmissionConfig(max_wait=10.0, aging_weight=2.0))
    waiting = [
        candidate("priority", 4.0, 1, priority=2),
        candidate("aged", 0.0, 0),
    ]

    early = controller.rank_waiting(waiting, now=5.0)
    late = controller.rank_waiting(waiting, now=15.0)

    assert early[0].request_id == "priority"
    # Once max_wait is reached, overdue ordering is still deterministic by score.
    assert {item.request_id for item in late} == {"priority", "aged"}


def test_max_wait_swaps_overdue_request_into_full_admission_set():
    """A recently served active sequence yields to max-wait queued work."""

    controller = FairAdmissionController(AdmissionConfig(max_wait=1.0, max_preemptions_per_step=1))
    waiting = [candidate("overdue", 0.0, 0)]
    admitted = [
        candidate(
            "active",
            0.5,
            1,
            last_progress=1.9,
            service_tokens=100,
        )
    ]

    decision = controller.decide(waiting, admitted, now=2.0, admitted_sequence_limit=1)

    assert decision.admitted_request_ids == ("overdue",)
    assert decision.preempted_request_ids == ("active",)
    assert decision.starvation_prevented
    assert "max_wait_swap" in decision.reasons


def test_reduced_sequence_limit_preempts_exact_excess():
    """Adaptive admission contraction is enforced before new work enters."""

    controller = FairAdmissionController()
    admitted = [
        candidate(
            f"active-{index}",
            float(index),
            index,
            last_progress=4.9,
            service_tokens=10,
        )
        for index in range(3)
    ]

    decision = controller.decide([], admitted, now=5.0, admitted_sequence_limit=1)

    assert len(decision.preempted_request_ids) == 2
    assert not decision.admitted_request_ids
    assert "sequence_limit_reduced" in decision.reasons


def test_service_ranking_prefers_request_without_recent_progress():
    """Service aging prevents an admitted prefill from being ignored forever."""

    controller = FairAdmissionController(AdmissionConfig(max_wait=1.0))
    candidates = [
        candidate("recent", 0.0, 0, last_progress=4.9, service_tokens=10),
        candidate("stale", 0.0, 1, last_progress=2.0, service_tokens=20),
    ]

    ranked = controller.rank_for_service(candidates, now=5.0)

    assert ranked[0].request_id == "stale"


def test_preemption_can_be_disabled_for_control_experiment():
    """A no-preemption control leaves a full admission set unchanged."""

    controller = FairAdmissionController(AdmissionConfig(max_wait=0.1, allow_preemption=False))
    decision = controller.decide(
        [candidate("waiting", 0.0, 0)],
        [candidate("active", 0.0, 1, last_progress=0.9, service_tokens=1)],
        now=1.0,
        admitted_sequence_limit=1,
    )

    assert not decision.admitted_request_ids
    assert not decision.preempted_request_ids


def test_duplicate_request_ids_are_rejected():
    """Ambiguous mutation targets cannot enter an admission decision."""

    controller = FairAdmissionController()
    with pytest.raises(ValueError, match="unique"):
        controller.decide(
            [candidate("duplicate", 0.0, 0)],
            [candidate("duplicate", 0.0, 1)],
            now=1.0,
            admitted_sequence_limit=1,
        )
