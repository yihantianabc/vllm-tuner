"""Deterministic real-document System/RAG traces for long-context v5 M3 APC."""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from vllm_tuner.workloads.generator import _fit_exact_token_count
from vllm_tuner.workloads.trace import TraceEntry, WorkloadTrace

CacheState = Literal["target-prefix-cold", "target-prefix-warm"]

RAG_CORPUS_FILES = (
    "README.md",
    "REPRODUCTION.md",
    "TESTING.md",
    "docs/METHODOLOGY.md",
    "docs/FORMAL_EXPERIMENTS.md",
    "docs/user-guide/configuration.md",
    "docs/user-guide/cli-commands.md",
    "docs/architecture/tuning-engine.md",
)

COLD_PREFIX_SETTLE_SECONDS = 2.0


@dataclass(frozen=True)
class RAGCorpus:
    documents: tuple[tuple[str, str], ...]
    sha256: str


@dataclass(frozen=True)
class M3CoreTraceBundle:
    trace: WorkloadTrace
    reuse_by_request: dict[str, int]
    shared_request: dict[str, bool]
    shared_prefix_text: dict[str, str]
    prefix_identity_by_request: dict[str, str]
    prefix_proof: dict[str, object]
    corpus_sha256: str


@dataclass(frozen=True)
class M3BoundaryTraceBundle:
    warmup: WorkloadTrace
    measured: WorkloadTrace
    prefix_proof: dict[str, object]
    corpus_sha256: str


def load_rag_corpus(repository: str | Path) -> RAGCorpus:
    """Load immutable public project text used as meaningful retrieved context."""
    root = Path(repository).resolve()
    documents: list[tuple[str, str]] = []
    digest = hashlib.sha256()
    for relative in RAG_CORPUS_FILES:
        path = root / relative
        text = path.read_text(encoding="utf-8")
        if not text.strip():
            raise ValueError(f"M3 RAG corpus source is empty: {relative}")
        documents.append((relative, text))
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(text.encode("utf-8"))
        digest.update(b"\0")
    return RAGCorpus(tuple(documents), digest.hexdigest())


def _token_ids(text: str, tokenizer: Any) -> list[int]:
    return list(tokenizer.encode(text, add_special_tokens=False))


def _long_document_text(
    *,
    header: str,
    corpus: RAGCorpus,
    target_tokens: int,
    tokenizer: Any,
    rotation: int,
) -> str:
    """Build enough coherent retrieved text without token/string filler loops."""
    documents = corpus.documents
    ordered = documents[rotation % len(documents) :] + documents[: rotation % len(documents)]
    pieces = [header]
    pass_index = 0
    while len(_token_ids("".join(pieces), tokenizer)) < target_tokens + 128:
        for source_index, (name, text) in enumerate(ordered, start=1):
            pieces.append(
                "\n\n### Retrieved source "
                f"{pass_index + 1}.{source_index}: {name}\n"
                f"{text.strip()}\n"
            )
        pass_index += 1
        if pass_index > 2:
            raise ValueError("M3 public RAG corpus is too small for the requested prompt")
    return "".join(pieces)


def _exact_prefix(
    *,
    identity: str,
    prefix_tokens: int,
    corpus: RAGCorpus,
    tokenizer: Any,
    rotation: int,
) -> str:
    identity_digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:12]
    header = (
        f"[{identity_digest} RAG-COLLECTION {identity}]\n"
        "System: You are an evidence-grounded inference operations assistant. "
        "Use only the retrieved project sources below, distinguish measured evidence from "
        "inference, preserve failed runs, and answer the final request concisely.\n"
    )
    raw = _long_document_text(
        header=header,
        corpus=corpus,
        target_tokens=prefix_tokens,
        tokenizer=tokenizer,
        rotation=rotation,
    )
    ids = _token_ids(raw, tokenizer)[:prefix_tokens]
    prefix = tokenizer.decode(ids)
    prefix, counted = _fit_exact_token_count(prefix, prefix_tokens, tokenizer)
    if counted != prefix_tokens:
        raise ValueError("M3 prefix fitting did not preserve the requested token count")
    return prefix


def _prompt_from_prefix(
    *,
    prefix: str,
    request_id: str,
    input_tokens: int,
    corpus: RAGCorpus,
    tokenizer: Any,
    rotation: int,
) -> str:
    # The short common bridge keeps BPE behavior stable at the exact prefix boundary.
    # The digest diverges inside the following 16-token block, so cacheable reuse remains
    # exactly the block-aligned 2K/4K prefix rather than a shared query template.
    digest = hashlib.sha256(request_id.encode("utf-8")).hexdigest()[:16]
    header = (
        f"\nQ{digest} Request: Compare the relevant runtime evidence for request "
        f"{request_id}; cite concrete cache, latency, and service observations.\n"
    )
    raw_tail = _long_document_text(
        header=header,
        corpus=corpus,
        target_tokens=input_tokens,
        tokenizer=tokenizer,
        rotation=rotation,
    )
    full = prefix + raw_tail
    ids = _token_ids(full, tokenizer)[:input_tokens]
    prompt = tokenizer.decode(ids)
    prompt, counted = _fit_exact_token_count(prompt, input_tokens, tokenizer)
    if counted != input_tokens:
        raise ValueError("M3 prompt fitting did not preserve the exact input length")
    return prompt


def _gamma_offsets(count: int, rate: float, burstiness: float, seed: int) -> list[float]:
    rng = random.Random(seed)
    shape = 1.0 / (burstiness * burstiness)
    scale = (1.0 / rate) / shape
    offsets = [0.0]
    for _ in range(1, count):
        offsets.append(offsets[-1] + rng.gammavariate(shape, scale))
    raw_span = offsets[-1]
    if raw_span <= 0:
        raise ValueError("M3 Gamma trace must span positive time")
    target_span = (count - 1) / rate
    factor = target_span / raw_span
    return [
        round(target_span if index == count - 1 else value * factor, 9)
        for index, value in enumerate(offsets)
    ]


def _ensure_first_shared_prefix_settles(offsets: list[float]) -> list[float]:
    """Reserve one deterministic cold-prefill window while preserving total span."""
    if len(offsets) < 2 or offsets[1] >= COLD_PREFIX_SETTLE_SECONDS:
        return offsets
    original_first = offsets[1]
    target_span = offsets[-1]
    if target_span <= COLD_PREFIX_SETTLE_SECONDS or target_span <= original_first:
        raise ValueError("M3 trace is too short for its cold-prefix establishment window")
    factor = (target_span - COLD_PREFIX_SETTLE_SECONDS) / (target_span - original_first)
    adjusted = [0.0]
    adjusted.extend(
        round(
            COLD_PREFIX_SETTLE_SECONDS + (value - original_first) * factor,
            9,
        )
        for value in offsets[1:]
    )
    adjusted[-1] = target_span
    return adjusted


def _longest_common_prefix(left: list[int], right: list[int]) -> int:
    count = 0
    for first, second in zip(left, right):
        if first != second:
            break
        count += 1
    return count


def _prefix_proof(
    *,
    prompts: dict[str, str],
    identities: dict[str, str],
    prefix_tokens: int,
    tokenizer: Any,
) -> dict[str, object]:
    tokenized = {
        request_id: _token_ids(prompt, tokenizer) for request_id, prompt in prompts.items()
    }
    representatives: dict[str, str] = {}
    minimum_shared_lcp: dict[str, int] = {}
    maximum_shared_lcp: dict[str, int] = {}
    identity_counts: dict[str, int] = {}
    for request_id, identity in identities.items():
        representative = representatives.setdefault(identity, request_id)
        identity_counts[identity] = identity_counts.get(identity, 0) + 1
        if representative == request_id:
            continue
        lcp = _longest_common_prefix(tokenized[representative], tokenized[request_id])
        minimum_shared_lcp[identity] = min(minimum_shared_lcp.get(identity, lcp), lcp)
        maximum_shared_lcp[identity] = max(maximum_shared_lcp.get(identity, lcp), lcp)

    first_blocks: dict[tuple[int, ...], str] = {}
    for identity, request_id in representatives.items():
        block = tuple(tokenized[request_id][:16])
        collision = first_blocks.get(block)
        if collision is not None and collision != identity:
            raise ValueError(
                f"M3 distinct prefix identities share their first cache block: {identity}"
            )
        first_blocks[block] = identity

    for identity, count in identity_counts.items():
        if count < 2:
            continue
        minimum = minimum_shared_lcp[identity]
        maximum = maximum_shared_lcp[identity]
        if minimum < prefix_tokens or maximum >= prefix_tokens + 16:
            raise ValueError(
                "M3 shared prompts do not expose exactly one block-aligned prefix length: "
                f"identity={identity}, min_lcp={minimum}, max_lcp={maximum}"
            )
    return {
        "block_size_tokens": 16,
        "nominal_prefix_tokens": prefix_tokens,
        "distinct_prefix_identities": len(representatives),
        "distinct_first_blocks": len(first_blocks),
        "minimum_shared_lcp_tokens": minimum_shared_lcp,
        "maximum_shared_lcp_tokens": maximum_shared_lcp,
        "block_aligned_reuse_proved": True,
    }


def build_m3_core_trace(
    *,
    prefix_tokens: int,
    requests_per_reuse: int,
    input_tokens: int,
    output_tokens: int,
    offered_requests_per_second: float,
    burstiness: float,
    seed: int,
    tokenizer: Any,
    corpus: RAGCorpus,
) -> M3CoreTraceBundle:
    """Build ordered 100%, 50%, and 0% reuse cohorts on one paired trace."""
    if requests_per_reuse < 2 or requests_per_reuse % 2:
        raise ValueError("M3 exact 50% reuse requires a positive even cohort size")
    if input_tokens <= prefix_tokens + 16:
        raise ValueError("M3 input must leave a divergent query tail after the shared prefix")
    count = requests_per_reuse * 3
    offsets = _ensure_first_shared_prefix_settles(
        _gamma_offsets(count, offered_requests_per_second, burstiness, seed)
    )
    entries: list[TraceEntry] = []
    prompts: dict[str, str] = {}
    reuse_by_request: dict[str, int] = {}
    shared_request: dict[str, bool] = {}
    identity_by_request: dict[str, str] = {}
    shared_prefix_text: dict[str, str] = {}

    cohort_specs: list[tuple[int, bool, int]] = []
    cohort_specs.extend((100, True, index) for index in range(requests_per_reuse))
    # Establish the 50% shared prefix once, issue all distinct controls, then reuse it.
    # This keeps cache-hit semantics independent of sub-second open-loop bursts.
    cohort_specs.append((50, True, 0))
    cohort_specs.extend((50, False, index) for index in range(1, requests_per_reuse // 2 + 1))
    cohort_specs.extend(
        (50, True, index) for index in range(requests_per_reuse // 2 + 1, requests_per_reuse)
    )
    cohort_specs.extend((0, False, index) for index in range(requests_per_reuse))

    for global_index, (reuse, is_shared, cohort_index) in enumerate(cohort_specs):
        request_id = f"m3-p{prefix_tokens}-r{reuse}-{cohort_index:03d}"
        identity = (
            f"p{prefix_tokens}-reuse-{reuse}-shared"
            if is_shared
            else f"p{prefix_tokens}-reuse-{reuse}-unique-{cohort_index:03d}"
        )
        prefix = shared_prefix_text.get(identity)
        if prefix is None:
            prefix = _exact_prefix(
                identity=identity,
                prefix_tokens=prefix_tokens,
                corpus=corpus,
                tokenizer=tokenizer,
                rotation=(global_index + prefix_tokens // 16) % len(corpus.documents),
            )
            if is_shared:
                shared_prefix_text[identity] = prefix
        prompt = _prompt_from_prefix(
            prefix=prefix,
            request_id=request_id,
            input_tokens=input_tokens,
            corpus=corpus,
            tokenizer=tokenizer,
            rotation=(global_index * 3 + reuse) % len(corpus.documents),
        )
        prompts[request_id] = prompt
        reuse_by_request[request_id] = reuse
        shared_request[request_id] = is_shared
        identity_by_request[request_id] = identity
        entries.append(
            TraceEntry(
                request_id=request_id,
                scheduled_offset_seconds=offsets[global_index],
                prompt=prompt,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                profile=f"m3-real-rag-reuse-{reuse}",
                shared_prefix_id=identity if is_shared else None,
            )
        )

    proof = _prefix_proof(
        prompts=prompts,
        identities=identity_by_request,
        prefix_tokens=prefix_tokens,
        tokenizer=tokenizer,
    )
    proof["cold_prefix_settle_seconds"] = COLD_PREFIX_SETTLE_SECONDS
    trace = WorkloadTrace(
        seed=seed,
        profile="m3-real-system-rag",
        request_rate=offered_requests_per_second,
        burstiness=burstiness,
        entries=entries,
    )
    return M3CoreTraceBundle(
        trace=trace,
        reuse_by_request=reuse_by_request,
        shared_request=shared_request,
        shared_prefix_text=shared_prefix_text,
        prefix_identity_by_request=identity_by_request,
        prefix_proof=proof,
        corpus_sha256=corpus.sha256,
    )


def expected_core_cached_tokens(
    bundle: M3CoreTraceBundle,
    *,
    prefix_tokens: int,
    apc_enabled: bool,
    cache_state: CacheState,
) -> dict[str, int]:
    """Derive exact per-request cache hits from the persisted trace semantics."""
    seen: set[str] = set()
    expected: dict[str, int] = {}
    warm = cache_state == "target-prefix-warm"
    for entry in bundle.trace.entries:
        identity = entry.shared_prefix_id
        if not apc_enabled or identity is None:
            expected[entry.request_id] = 0
        elif warm or identity in seen:
            expected[entry.request_id] = prefix_tokens
        else:
            expected[entry.request_id] = 0
        if identity is not None:
            seen.add(identity)
    return expected


def build_m3_core_warmup(
    *,
    bundle: M3CoreTraceBundle,
    cache_state: CacheState,
    prefix_tokens: int,
    input_tokens: int,
    output_tokens: int,
    seed: int,
    tokenizer: Any,
    corpus: RAGCorpus,
) -> WorkloadTrace:
    """Warm CUDA equally; only the warm state primes measured target prefixes."""
    entries: list[TraceEntry] = []
    shared_ids = [
        f"p{prefix_tokens}-reuse-100-shared",
        f"p{prefix_tokens}-reuse-50-shared",
    ]
    for index, shared_id in enumerate(shared_ids):
        if cache_state == "target-prefix-warm":
            identity = shared_id
            prefix = bundle.shared_prefix_text[shared_id]
        else:
            identity = f"p{prefix_tokens}-cold-control-{index}"
            prefix = _exact_prefix(
                identity=identity,
                prefix_tokens=prefix_tokens,
                corpus=corpus,
                tokenizer=tokenizer,
                rotation=(seed + index) % len(corpus.documents),
            )
        request_id = f"warmup-{cache_state}-{index}"
        prompt = _prompt_from_prefix(
            prefix=prefix,
            request_id=request_id,
            input_tokens=input_tokens,
            corpus=corpus,
            tokenizer=tokenizer,
            rotation=(seed + index * 3) % len(corpus.documents),
        )
        entries.append(
            TraceEntry(
                request_id=request_id,
                scheduled_offset_seconds=0.0,
                prompt=prompt,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                profile=f"m3-{cache_state}-warmup",
                shared_prefix_id=identity,
            )
        )
    return WorkloadTrace(
        seed=seed,
        profile=f"m3-{cache_state}-warmup",
        request_rate=None,
        burstiness=1.0,
        entries=entries,
    )


def build_m3_boundary_trace(
    *,
    pool_size: int,
    prefix_tokens: int,
    tail_tokens: int,
    output_tokens: int,
    interval_seconds: float,
    seed: int,
    tokenizer: Any,
    corpus: RAGCorpus,
) -> M3BoundaryTraceBundle:
    """Prime a prefix pool once, then probe newest-to-oldest retention exactly once."""
    input_tokens = prefix_tokens + tail_tokens
    warmup_entries: list[TraceEntry] = []
    measured_entries: list[TraceEntry] = []
    prompts: dict[str, str] = {}
    identities: dict[str, str] = {}
    prefixes: dict[str, str] = {}
    for index in range(pool_size):
        identity = f"boundary-p{prefix_tokens}-pool{pool_size}-{index:03d}"
        prefix = _exact_prefix(
            identity=identity,
            prefix_tokens=prefix_tokens,
            corpus=corpus,
            tokenizer=tokenizer,
            rotation=(seed + index) % len(corpus.documents),
        )
        prefixes[identity] = prefix
        request_id = f"boundary-prime-{pool_size}-{index:03d}"
        prompt = _prompt_from_prefix(
            prefix=prefix,
            request_id=request_id,
            input_tokens=input_tokens,
            corpus=corpus,
            tokenizer=tokenizer,
            rotation=(seed + index * 3) % len(corpus.documents),
        )
        warmup_entries.append(
            TraceEntry(
                request_id=request_id,
                scheduled_offset_seconds=0.0,
                prompt=prompt,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                profile="m3-prefix-pool-prime",
                shared_prefix_id=identity,
            )
        )

    for measured_index, original_index in enumerate(reversed(range(pool_size))):
        identity = f"boundary-p{prefix_tokens}-pool{pool_size}-{original_index:03d}"
        request_id = f"boundary-probe-{pool_size}-{original_index:03d}"
        prompt = _prompt_from_prefix(
            prefix=prefixes[identity],
            request_id=request_id,
            input_tokens=input_tokens,
            corpus=corpus,
            tokenizer=tokenizer,
            rotation=(seed + original_index * 5 + 1) % len(corpus.documents),
        )
        prompts[request_id] = prompt
        identities[request_id] = identity
        measured_entries.append(
            TraceEntry(
                request_id=request_id,
                scheduled_offset_seconds=round(measured_index * interval_seconds, 9),
                prompt=prompt,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                profile="m3-prefix-pool-probe",
                shared_prefix_id=identity,
            )
        )
    proof_prompts = dict(prompts)
    proof_identities = dict(identities)
    for entry in warmup_entries:
        proof_prompts[entry.request_id] = entry.prompt
        proof_identities[entry.request_id] = str(entry.shared_prefix_id)
    proof = _prefix_proof(
        prompts=proof_prompts,
        identities=proof_identities,
        prefix_tokens=prefix_tokens,
        tokenizer=tokenizer,
    )
    return M3BoundaryTraceBundle(
        warmup=WorkloadTrace(
            seed=seed,
            profile="m3-prefix-pool-prime",
            request_rate=None,
            burstiness=1.0,
            entries=warmup_entries,
        ),
        measured=WorkloadTrace(
            seed=seed,
            profile="m3-prefix-pool-probe",
            request_rate=1.0 / interval_seconds,
            burstiness=1.0,
            entries=measured_entries,
        ),
        prefix_proof=proof,
        corpus_sha256=corpus.sha256,
    )


__all__ = [
    "CacheState",
    "M3BoundaryTraceBundle",
    "M3CoreTraceBundle",
    "RAGCorpus",
    "build_m3_boundary_trace",
    "build_m3_core_trace",
    "build_m3_core_warmup",
    "expected_core_cached_tokens",
    "load_rag_corpus",
]
