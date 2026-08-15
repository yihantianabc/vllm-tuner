"""Fixed trace determinism and profile coverage tests."""

from vllm_tuner.workloads.generator import generate_trace


def test_same_seed_reproduces_exact_trace() -> None:
    first = generate_trace("chat", count=10, request_rate=4, seed=7)
    second = generate_trace("chat", count=10, request_rate=4, seed=7)
    assert first.checksum() == second.checksum()
    assert first.entries == second.entries


def test_holdout_seed_changes_trace() -> None:
    training = generate_trace("mixed", count=10, request_rate=8, seed=7)
    holdout = generate_trace("mixed", count=10, request_rate=8, seed=8)
    assert training.checksum() != holdout.checksum()
    assert any(entry.input_tokens > 2000 for entry in training.entries)


def test_rag_trace_contains_shared_prefix_requests() -> None:
    trace = generate_trace("rag", count=50, request_rate=2, seed=2026)
    assert any(entry.shared_prefix_id is not None for entry in trace.entries)


def test_generated_prompts_are_request_specific() -> None:
    trace = generate_trace("chat", count=10, request_rate=2, seed=2026)
    assert len({entry.prompt for entry in trace.entries}) == len(trace.entries)


def test_fixed_token_lengths_override_profile_samples() -> None:
    trace = generate_trace(
        "rag",
        count=20,
        request_rate=2,
        seed=2026,
        fixed_input_tokens=256,
        fixed_output_tokens=128,
    )
    assert {entry.input_tokens for entry in trace.entries} == {256}
    assert {entry.output_tokens for entry in trace.entries} == {128}


def test_shared_requests_have_a_real_common_prefix_without_collapsing_all_prompts() -> None:
    trace = generate_trace("rag", count=50, request_rate=2, seed=2026, fixed_input_tokens=256)
    shared = [entry.prompt for entry in trace.entries if entry.shared_prefix_id]
    unshared = [entry.prompt for entry in trace.entries if not entry.shared_prefix_id]
    assert len(shared) >= 2
    assert len({prompt[:128] for prompt in shared}) == 1
    assert len({prompt for prompt in shared + unshared}) == len(shared) + len(unshared)
