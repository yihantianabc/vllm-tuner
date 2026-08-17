"""Determinism and held-out ordering tests for multi-phase traces."""

import pytest

from vllm_tuner.workloads.nonstationary import (
    NonStationaryPhaseSpec,
    empirical_request_rate,
    generate_nonstationary_trace,
    multiply_phase_counts,
    phase_boundaries,
    scale_trace_to_empirical_rate,
)


class CharacterTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [ord(character) for character in text]

    def decode(self, token_ids: list[int]) -> str:
        return "".join(chr(token_id) for token_id in token_ids)


def phases() -> dict[str, NonStationaryPhaseSpec]:
    return {
        "decode": NonStationaryPhaseSpec(
            "decode", "chat", 2, 4.0, 1.0, fixed_input_tokens=32, fixed_output_tokens=16
        ),
        "prefill": NonStationaryPhaseSpec(
            "prefill", "rag", 2, 8.0, 0.5, fixed_input_tokens=64, fixed_output_tokens=4
        ),
        "mixed": NonStationaryPhaseSpec(
            "mixed", "mixed", 2, 4.0, 1.0, fixed_input_tokens=48, fixed_output_tokens=8
        ),
    }


def test_nonstationary_trace_is_deterministic_contiguous_and_exact_length() -> None:
    specs = phases()
    order = [specs[name] for name in ("decode", "prefill", "mixed")]
    tokenizer = CharacterTokenizer()

    first = generate_nonstationary_trace(order, seed=7, tokenizer=tokenizer)
    second = generate_nonstationary_trace(order, seed=7, tokenizer=tokenizer)

    assert first.checksum() == second.checksum()
    assert len(first.entries) == 6
    assert [entry.profile for entry in first.entries] == [
        "decode",
        "decode",
        "prefill",
        "prefill",
        "mixed",
        "mixed",
    ]
    assert [entry.scheduled_offset_seconds for entry in first.entries] == sorted(
        entry.scheduled_offset_seconds for entry in first.entries
    )
    assert len({entry.request_id for entry in first.entries}) == 6
    assert [entry.input_tokens for entry in first.entries] == [32, 32, 64, 64, 48, 48]


def test_heldout_order_reuses_requests_but_changes_phase_sequence() -> None:
    specs = phases()
    tokenizer = CharacterTokenizer()
    calibration = generate_nonstationary_trace(
        [specs[name] for name in ("decode", "prefill", "mixed")],
        seed=7,
        tokenizer=tokenizer,
    )
    heldout = generate_nonstationary_trace(
        [specs[name] for name in ("prefill", "mixed", "decode")],
        seed=7,
        tokenizer=tokenizer,
    )

    calibration_requests = {
        entry.request_id: (entry.prompt, entry.input_tokens, entry.output_tokens)
        for entry in calibration.entries
    }
    heldout_requests = {
        entry.request_id: (entry.prompt, entry.input_tokens, entry.output_tokens)
        for entry in heldout.entries
    }
    assert calibration_requests == heldout_requests
    assert calibration.checksum() != heldout.checksum()
    assert [row["phase"] for row in phase_boundaries(heldout)] == [
        "prefill",
        "mixed",
        "decode",
    ]


def test_formal_count_and_arrival_scaling_preserve_exact_requests() -> None:
    expanded = multiply_phase_counts(list(phases().values()), 3)
    assert [phase.count for phase in expanded] == [6, 6, 6]

    source = generate_nonstationary_trace(
        expanded,
        seed=7,
        tokenizer=CharacterTokenizer(),
    )
    scaled = scale_trace_to_empirical_rate(source, 8.0)

    assert empirical_request_rate(scaled) == pytest.approx(8.0)
    assert [entry.request_id for entry in scaled.entries] == [
        entry.request_id for entry in source.entries
    ]
    assert [entry.prompt for entry in scaled.entries] == [entry.prompt for entry in source.entries]
