"""Deterministic decode-stream traces with injected 4K--8K long prefills."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

from vllm_tuner.workloads.generator import _fit_exact_token_count, generate_trace
from vllm_tuner.workloads.trace import TraceEntry, WorkloadTrace

from .m4_chunked_config import M4Protocol


@dataclass(frozen=True)
class M4TraceBundle:
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
            raise ValueError("M4 prompt is shorter than one cache block")
        collision = first_blocks.get(block)
        if collision is not None:
            raise ValueError(
                f"M4 prompts share a cacheable first block: {collision}, {entry.request_id}"
            )
        first_blocks[block] = entry.request_id
        for other in entries[:index]:
            pair_count += 1
            maximum_lcp = max(
                maximum_lcp,
                _longest_common_prefix(tokenized[entry.request_id], tokenized[other.request_id]),
            )
    if maximum_lcp >= 16:
        raise ValueError("M4 workload contains a cacheable shared prefix")
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
            raise ValueError("M4 unique first-block marker changed the exact prompt length")
        entries.append(entry.model_copy(update={"prompt": prompt, "input_tokens": counted}))
    return entries


def build_m4_trace(
    *,
    protocol: M4Protocol,
    long_prefill_tokens: int,
    tokenizer: Any,
) -> M4TraceBundle:
    """Build one trace reused byte-for-byte by every preregistered profile."""
    if long_prefill_tokens not in protocol.long_prefill_tokens:
        raise ValueError("M4 trace length is not preregistered")
    decode = _generated_entries(
        count=protocol.decode_requests,
        input_tokens=protocol.decode_input_tokens,
        output_tokens=protocol.decode_output_tokens,
        seed=protocol.measurement_seed,
        offset=100_000 + long_prefill_tokens,
        prefix="m4-decode",
        tokenizer=tokenizer,
    )
    long_prefills = _generated_entries(
        count=len(protocol.injection_offsets_seconds),
        input_tokens=long_prefill_tokens,
        output_tokens=protocol.long_output_tokens,
        seed=protocol.measurement_seed + long_prefill_tokens,
        offset=300_000 + long_prefill_tokens,
        prefix=f"m4-long-{long_prefill_tokens}",
        tokenizer=tokenizer,
    )
    measured_entries = [
        entry.model_copy(
            update={
                "scheduled_offset_seconds": round(index * protocol.decode_interval_seconds, 9),
                "profile": "m4-stable-decode",
                "shared_prefix_id": None,
            }
        )
        for index, entry in enumerate(decode)
    ]
    measured_entries.extend(
        entry.model_copy(
            update={
                "scheduled_offset_seconds": protocol.injection_offsets_seconds[index],
                "profile": "m4-long-prefill-injection",
                "shared_prefix_id": None,
            }
        )
        for index, entry in enumerate(long_prefills)
    )
    measured_entries.sort(key=lambda entry: (entry.scheduled_offset_seconds, entry.request_id))
    span = (protocol.decode_requests - 1) * protocol.decode_interval_seconds
    overall_rate = (len(measured_entries) - 1) / span
    measured = WorkloadTrace(
        seed=protocol.measurement_seed,
        profile=f"m4-decode-with-{long_prefill_tokens}-prefill",
        request_rate=overall_rate,
        burstiness=protocol.burstiness,
        entries=measured_entries,
    )

    warmup_decode = _generated_entries(
        count=1,
        input_tokens=protocol.decode_input_tokens,
        output_tokens=protocol.decode_output_tokens,
        seed=protocol.warmup_seed,
        offset=700_000 + long_prefill_tokens,
        prefix="m4-warmup-decode",
        tokenizer=tokenizer,
    )[0]
    warmup_long = _generated_entries(
        count=1,
        input_tokens=long_prefill_tokens,
        output_tokens=protocol.long_output_tokens,
        seed=protocol.warmup_seed + long_prefill_tokens,
        offset=900_000 + long_prefill_tokens,
        prefix="m4-warmup-long",
        tokenizer=tokenizer,
    )[0]
    warmup = WorkloadTrace(
        seed=protocol.warmup_seed,
        profile=f"m4-warmup-{long_prefill_tokens}",
        request_rate=None,
        burstiness=1.0,
        entries=[
            warmup_decode.model_copy(
                update={
                    "scheduled_offset_seconds": 0.0,
                    "profile": "m4-warmup-decode",
                    "shared_prefix_id": None,
                }
            ),
            warmup_long.model_copy(
                update={
                    "scheduled_offset_seconds": 0.0,
                    "profile": "m4-warmup-long-prefill",
                    "shared_prefix_id": None,
                }
            ),
        ],
    )
    kind = {
        entry.request_id: (
            "long-prefill" if entry.profile == "m4-long-prefill-injection" else "decode"
        )
        for entry in measured.entries
    }
    proof = _prefix_isolation_proof(measured, warmup, tokenizer)
    return M4TraceBundle(
        measured=measured,
        warmup=warmup,
        request_kind=kind,
        prefix_isolation_proof=proof,
    )


__all__ = ["M4TraceBundle", "build_m4_trace"]
