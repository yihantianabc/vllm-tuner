"""Frozen M3 protocol consistency checks."""

import hashlib
from pathlib import Path

import pytest
import yaml

from vllm_tuner.workloads.nonstationary import empirical_request_rate
from vllm_tuner.workloads.trace import WorkloadTrace


def test_frozen_formal_matrix_references_exact_validated_traces() -> None:
    repository = Path(__file__).resolve().parents[2]
    protocol_path = repository / "experiments/adaptive_prefill/m3_formal_protocol.yaml"
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))

    assert set(protocol["policies"]) == {
        "stock",
        "fixed_low",
        "fixed_mid",
        "fixed_high",
        "adaptive",
    }
    assert protocol["formal_execution"]["repeats"] == 3
    assert protocol["formal_execution"]["measured_requests_per_trace"] == 640
    assert protocol["primary_slo_tier"] == "medium"

    for load in protocol["load_points"].values():
        expected_rate = load["empirical_requests_per_second"]
        for trace_kind in ("calibration", "heldout"):
            trace_path = protocol_path.parent / load[f"{trace_kind}_trace"]
            expected_checksum = load[f"{trace_kind}_sha256"]
            assert hashlib.sha256(trace_path.read_bytes()).hexdigest() == expected_checksum
            trace = WorkloadTrace.read(
                trace_path,
                seed=2026,
                profile="nonstationary-formal",
                request_rate=expected_rate,
                burstiness=1.0,
            )
            assert len(trace.entries) == 640
            assert empirical_request_rate(trace) == pytest.approx(expected_rate)


def test_frozen_slo_tiers_are_monotonically_relaxed() -> None:
    repository = Path(__file__).resolve().parents[2]
    protocol = yaml.safe_load(
        (repository / "experiments/adaptive_prefill/m3_formal_protocol.yaml").read_text(
            encoding="utf-8"
        )
    )
    tiers = protocol["slo_tiers_ms"]

    for metric in ("ttft", "tpot", "e2e"):
        assert tiers["strict"][metric] < tiers["medium"][metric] < tiers["loose"][metric]
