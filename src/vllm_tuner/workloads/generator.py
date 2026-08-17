"""Seeded length and open-loop arrival generation."""

from __future__ import annotations

import math
import random
from typing import Any, Optional

from .profiles import WorkloadProfile, get_profile
from .trace import TraceEntry, WorkloadTrace

SHARED_CONTEXT = (
    "Reference context: performance experiments must preserve request timing, token counts, "
    "and service-level objectives. "
)
USER_TEXT = "Explain the inference-system behavior for this deterministic workload request. "


def _sample_length(
    rng: random.Random,
    profile: WorkloadProfile,
    request_index: int,
) -> tuple[int, int]:
    if profile.name == "mixed":
        if request_index % 3 == 0:
            input_tokens = rng.randint(2048, profile.input_token_range[1])
        else:
            input_tokens = rng.randint(profile.input_token_range[0], 512)
    else:
        input_tokens = rng.randint(*profile.input_token_range)
    return input_tokens, rng.randint(*profile.output_token_range)


def _repeat_to_size(values: list[int], size: int) -> list[int]:
    if size <= 0:
        return []
    if not values:
        raise ValueError("tokenizer returned no tokens for workload text")
    return (values * math.ceil(size / len(values)))[:size]


def _fit_exact_token_count(prompt: str, target: int, tokenizer: Any) -> tuple[str, int]:
    """Repair decode-time BPE merges while preserving an exact input length.

    Decoding ``target`` token IDs is not necessarily an encode/decode fixed point:
    adjacent decoded fragments can merge into fewer BPE tokens.  Truncate any
    overrun first, then append a separator-prefixed token whose repeated form
    closes the remaining gap exactly.  The candidate search keeps this compatible
    with both Hugging Face tokenizers and the minimal tokenizer protocol in tests.
    """
    counted = len(tokenizer.encode(prompt, add_special_tokens=False))
    if counted > target:
        ids = tokenizer.encode(prompt, add_special_tokens=False)[:target]
        prompt = tokenizer.decode(ids)
        counted = len(tokenizer.encode(prompt, add_special_tokens=False))

    padding_candidates = (" x", " a", " z", "\n#", " §")
    while counted < target:
        gap = target - counted
        matched = False
        for padding in padding_candidates:
            candidate = prompt + padding * gap
            candidate_count = len(tokenizer.encode(candidate, add_special_tokens=False))
            if candidate_count == target:
                return candidate, candidate_count
            if counted < candidate_count < target:
                prompt = candidate
                counted = candidate_count
                matched = True
                break
        if not matched:
            break
    if counted != target:
        raise ValueError(
            "tokenizer could not construct the requested fixed prompt length: "
            f"requested={target}, counted={counted}"
        )
    return prompt, counted


def _prompt_for_tokens(
    target: int,
    tokenizer: Optional[Any],
    shared: bool,
    request_index: int,
) -> tuple[str, int]:
    unique_text = (
        f"Request identifier {request_index:08d}; scenario {request_index % 97:02d}. "
        f"{USER_TEXT}"
    )
    if tokenizer is None:
        # Four characters/token is only a generator fallback. The measured client recounts tokens
        # with the model tokenizer or server usage before accepting an artifact.
        target_chars = target * 4
        shared_chars = min(target_chars // 2, 2048) if shared else 0
        prefix = (SHARED_CONTEXT * math.ceil(shared_chars / len(SHARED_CONTEXT)))[:shared_chars]
        body = unique_text * math.ceil((target_chars - len(prefix)) / len(unique_text) + 1)
        return (prefix + body)[:target_chars], target

    shared_tokens = min(target // 2, 512) if shared else 0
    prefix_ids = _repeat_to_size(
        list(tokenizer.encode(SHARED_CONTEXT, add_special_tokens=False)), shared_tokens
    )
    body_ids = list(tokenizer.encode(unique_text, add_special_tokens=False))
    token_ids = prefix_ids + _repeat_to_size(body_ids, target - len(prefix_ids))
    # The Hugging Face default already preserves special tokens. Keeping this
    # call to the minimal tokenizer protocol also supports lightweight test and
    # offline tokenizers that expose only ``decode(token_ids)``.
    prompt = tokenizer.decode(token_ids)
    return _fit_exact_token_count(prompt, target, tokenizer)


def _interarrival(rng: random.Random, request_rate: Optional[float], burstiness: float) -> float:
    if request_rate is None:
        return 0.0
    mean = 1.0 / request_rate
    # Public SLOTune ``burstiness`` is the inter-arrival coefficient of
    # variation (CV); Gamma shape is therefore 1 / CV^2.
    shape = 1.0 / (burstiness * burstiness)
    scale = mean / shape
    return rng.gammavariate(shape, scale)


def generate_trace(
    profile: str | WorkloadProfile,
    *,
    count: int,
    request_rate: Optional[float],
    burstiness: float = 1.0,
    seed: int = 2026,
    tokenizer: Optional[Any] = None,
    fixed_input_tokens: Optional[int] = None,
    fixed_output_tokens: Optional[int] = None,
    request_index_offset: int = 0,
    request_id_prefix: Optional[str] = None,
) -> WorkloadTrace:
    """Create a deterministic trace; callers persist it once and reuse it for all trials."""
    if count <= 0:
        raise ValueError("count must be positive")
    if request_rate is not None and request_rate <= 0:
        raise ValueError("request_rate must be positive or None for an immediate closed-loop trace")
    if burstiness <= 0:
        raise ValueError("burstiness must be positive")
    if fixed_input_tokens is not None and fixed_input_tokens <= 0:
        raise ValueError("fixed_input_tokens must be positive")
    if fixed_output_tokens is not None and fixed_output_tokens <= 0:
        raise ValueError("fixed_output_tokens must be positive")
    if request_index_offset < 0:
        raise ValueError("request_index_offset must be non-negative")
    if request_id_prefix is not None and not request_id_prefix:
        raise ValueError("request_id_prefix must not be empty")
    resolved = get_profile(profile) if isinstance(profile, str) else profile
    rng = random.Random(seed)
    offset = 0.0
    entries: list[TraceEntry] = []
    for index in range(count):
        request_index = request_index_offset + index
        if index:
            offset += _interarrival(rng, request_rate, burstiness)
        sampled_input, sampled_output = _sample_length(rng, resolved, request_index)
        input_tokens = fixed_input_tokens or sampled_input
        output_tokens = fixed_output_tokens or sampled_output
        shared = rng.random() < resolved.shared_prefix_ratio
        prompt, counted_tokens = _prompt_for_tokens(
            input_tokens, tokenizer, shared, request_index=request_index
        )
        id_prefix = request_id_prefix or resolved.name
        entries.append(
            TraceEntry(
                request_id=f"{id_prefix}-{request_index:06d}",
                scheduled_offset_seconds=round(offset, 9),
                prompt=prompt,
                input_tokens=counted_tokens,
                output_tokens=output_tokens,
                profile=resolved.name,
                shared_prefix_id=f"{resolved.name}-shared" if shared else None,
            )
        )
    return WorkloadTrace(
        seed=seed,
        profile=resolved.name,
        request_rate=request_rate,
        burstiness=burstiness,
        entries=entries,
    )
