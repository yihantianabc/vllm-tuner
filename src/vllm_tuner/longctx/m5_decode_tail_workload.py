"""Frozen target and held-out mixed-prefill traces for v5 M5."""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import Any

from vllm_tuner.workloads.generator import _fit_exact_token_count, generate_trace
from vllm_tuner.workloads.trace import TraceEntry, WorkloadTrace

from .m5_decode_tail_config import M5Cohort, M5Protocol


@dataclass(frozen=True)
class M5TraceBundle:
    cohort: M5Cohort
    measured: WorkloadTrace
    warmup: WorkloadTrace
    request_kind: dict[str, str]
    prefix_isolation_proof: dict[str, object]


def _token_ids(text: str, tokenizer: Any) -> list[int]:
    return list(tokenizer.encode(text, add_special_tokens=False))


def _longest_common_prefix(left: list[int], right: list[int]) -> int:
    count = 0
    for first, second in zip(left, right):
        if first != second:
            break
        count += 1
    return count


def _prefix_isolation_proof(
    measured: WorkloadTrace,
    warmup: WorkloadTrace,
    tokenizer: Any,
) -> dict[str, object]:
    entries = [*measured.entries, *warmup.entries]
    tokenized = {entry.request_id: _token_ids(entry.prompt, tokenizer) for entry in entries}
    first_blocks: dict[tuple[int, ...], str] = {}
    maximum_lcp = 0
    pair_count = 0
    for index, entry in enumerate(entries):
        block = tuple(tokenized[entry.request_id][:16])
        if len(block) != 16:
            raise ValueError("M5 prompt is shorter than one cache block")
        collision = first_blocks.get(block)
        if collision is not None:
            raise ValueError(
                f"M5 prompts share a cacheable first block: {collision}, {entry.request_id}"
            )
        first_blocks[block] = entry.request_id
        for other in entries[:index]:
            pair_count += 1
            maximum_lcp = max(
                maximum_lcp,
                _longest_common_prefix(tokenized[entry.request_id], tokenized[other.request_id]),
            )
    if maximum_lcp >= 16:
        raise ValueError("M5 workload contains a cacheable shared prefix")
    return {
        "block_size_tokens": 16,
        "request_count_including_warmup": len(entries),
        "pair_count": pair_count,
        "distinct_first_blocks": len(first_blocks),
        "maximum_pairwise_lcp_tokens": maximum_lcp,
        "no_cacheable_shared_prefix": True,
    }


def _generated_entries(
    *,
    count: int,
    input_tokens: int,
    output_tokens: int,
    seed: int,
    offset: int,
    prefix: str,
    tokenizer: Any,
) -> list[TraceEntry]:
    trace = generate_trace(
        "chat",
        count=count,
        request_rate=None,
        burstiness=1.0,
        seed=seed,
        tokenizer=tokenizer,
        fixed_input_tokens=input_tokens,
        fixed_output_tokens=output_tokens,
        request_index_offset=offset,
        request_id_prefix=prefix,
    )
    entries: list[TraceEntry] = []
    for entry in trace.entries:
        marker = hashlib.sha256(entry.request_id.encode("utf-8")).hexdigest()[:24]
        prompt, counted = _fit_exact_token_count(
            f"[{marker}] {entry.prompt}", input_tokens, tokenizer
        )
        if counted != input_tokens:
            raise ValueError("M5 unique first-block marker changed the exact prompt length")
        entries.append(entry.model_copy(update={"prompt": prompt, "input_tokens": counted}))
    return entries


def _decode_offsets(protocol: M5Protocol, cohort: M5Cohort) -> list[float]:
    rng = random.Random(cohort.arrival_seed)
    offsets = [0.0]
    jitter = protocol.decode_arrival_jitter_fraction
    for _ in range(1, protocol.decode_requests):
        factor = 1.0 + rng.uniform(-jitter, jitter)
        offsets.append(round(offsets[-1] + protocol.decode_interval_seconds * factor, 9))
    return offsets


def build_m5_trace(
    *,
    protocol: M5Protocol,
    cohort: M5Cohort,
    tokenizer: Any,
) -> M5TraceBundle:
    """Build one cohort trace reused byte-for-byte by both profiles and all repeats."""
    if cohort not in protocol.cohorts:
        raise ValueError("M5 trace cohort is not preregistered")
    cohort_offset = 0 if cohort.cohort_id == "target" else 1_000_000
    decode = _generated_entries(
        count=protocol.decode_requests,
        input_tokens=protocol.decode_input_tokens,
        output_tokens=protocol.decode_output_tokens,
        seed=cohort.prompt_seed,
        offset=100_000 + cohort_offset,
        prefix=f"m5-{cohort.cohort_id}-decode",
        tokenizer=tokenizer,
    )
    long_prefills: list[TraceEntry] = []
    for index, tokens in enumerate(cohort.long_prefill_tokens):
        long_prefills.extend(
            _generated_entries(
                count=1,
                input_tokens=tokens,
                output_tokens=protocol.long_output_tokens,
                seed=cohort.prompt_seed + tokens + index * 97,
                offset=300_000 + cohort_offset + index,
                prefix=f"m5-{cohort.cohort_id}-long-{tokens}",
                tokenizer=tokenizer,
            )
        )
    measured_entries = [
        entry.model_copy(
            update={
                "scheduled_offset_seconds": offset,
                "profile": "m5-stable-decode",
                "shared_prefix_id": None,
            }
        )
        for entry, offset in zip(decode, _decode_offsets(protocol, cohort))
    ]
    measured_entries.extend(
        entry.model_copy(
            update={
                "scheduled_offset_seconds": cohort.injection_offsets_seconds[index],
                "profile": "m5-long-prefill-injection",
                "shared_prefix_id": None,
            }
        )
        for index, entry in enumerate(long_prefills)
    )
    measured_entries.sort(key=lambda entry: (entry.scheduled_offset_seconds, entry.request_id))
    span = measured_entries[-1].scheduled_offset_seconds
    measured = WorkloadTrace(
        seed=cohort.prompt_seed,
        profile=f"m5-{cohort.cohort_id}-mixed-prefill",
        request_rate=(len(measured_entries) - 1) / span,
        burstiness=1.0,
        entries=measured_entries,
    )

    warmup_entries = _generated_entries(
        count=1,
        input_tokens=protocol.decode_input_tokens,
        output_tokens=protocol.decode_output_tokens,
        seed=cohort.warmup_seed,
        offset=700_000 + cohort_offset,
        prefix=f"m5-{cohort.cohort_id}-warmup-decode",
        tokenizer=tokenizer,
    )
    for index, tokens in enumerate((4_096, 8_192)):
        warmup_entries.extend(
            _generated_entries(
                count=1,
                input_tokens=tokens,
                output_tokens=protocol.long_output_tokens,
                seed=cohort.warmup_seed + tokens,
                offset=900_000 + cohort_offset + index,
                prefix=f"m5-{cohort.cohort_id}-warmup-long-{tokens}",
                tokenizer=tokenizer,
            )
        )
    warmup = WorkloadTrace(
        seed=cohort.warmup_seed,
        profile=f"m5-{cohort.cohort_id}-warmup",
        request_rate=None,
        burstiness=1.0,
        entries=[
            entry.model_copy(
                update={
                    "scheduled_offset_seconds": 0.0,
                    "profile": (
                        "m5-warmup-decode"
                        if entry.input_tokens == protocol.decode_input_tokens
                        else "m5-warmup-long-prefill"
                    ),
                    "shared_prefix_id": None,
                }
            )
            for entry in warmup_entries
        ],
    )
    kind = {
        entry.request_id: (
            "long-prefill" if entry.profile == "m5-long-prefill-injection" else "decode"
        )
        for entry in measured.entries
    }
    proof = _prefix_isolation_proof(measured, warmup, tokenizer)
    return M5TraceBundle(
        cohort=cohort,
        measured=measured,
        warmup=warmup,
        request_kind=kind,
        prefix_isolation_proof=proof,
    )


__all__ = ["M5TraceBundle", "build_m5_trace"]
