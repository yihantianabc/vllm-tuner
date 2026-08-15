"""Cross-validation against a live vLLM server and its official benchmark CLI."""

import json
import math
import os
from pathlib import Path

import pytest
from transformers import AutoTokenizer

from vllm_tuner.benchmarks.models import RequestSpec
from vllm_tuner.benchmarks.sse_client import SSEBenchmarkClient
from vllm_tuner.benchmarks.vllm_bench import VLLMBenchAdapter, VLLMBenchConfig
from vllm_tuner.experiment.manifest import collect_environment_fingerprint, git_state

BASE_URL = os.getenv("VLLM_TEST_BASE_URL")
MODEL = os.getenv("VLLM_TEST_MODEL")
ARTIFACT_DIR = os.getenv("VLLM_TEST_ARTIFACT_DIR")


@pytest.mark.skipif(
    not BASE_URL or not MODEL,
    reason="Set VLLM_TEST_BASE_URL and VLLM_TEST_MODEL for live cross-validation",
)
@pytest.mark.asyncio
async def test_sse_client_cross_validates_official_vllm_bench(tmp_path) -> None:
    """Both clients should agree on completions and tokenizer-derived totals."""

    assert BASE_URL is not None
    assert MODEL is not None
    prompts = ["Name one primary color.", "Return the number after four."]
    requests = [
        RequestSpec(
            request_id=f"cross-{index}",
            prompt=prompt,
            max_tokens=8,
            ignore_eos=True,
        )
        for index, prompt in enumerate(prompts)
    ]

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL,
        local_files_only=Path(MODEL).exists(),
    )
    sse_result = await SSEBenchmarkClient(
        BASE_URL,
        MODEL,
        tokenizer=tokenizer,
        require_token_ids=True,
    ).run(
        requests,
        warmup_requests=1,
        max_concurrency=1,
        seed=2026,
    )

    artifact_dir = Path(ARTIFACT_DIR) if ARTIFACT_DIR else tmp_path
    artifact_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = artifact_dir / "cross-validation-prompts.jsonl"
    dataset_path.write_text(
        "".join(json.dumps({"prompt": prompt}) + "\n" for prompt in prompts),
        encoding="utf-8",
    )
    official_result = await VLLMBenchAdapter().run(
        VLLMBenchConfig(
            base_url=BASE_URL,
            model=MODEL,
            output_path=artifact_dir / "official-raw.json",
            num_prompts=len(prompts),
            dataset_name="custom",
            dataset_path=dataset_path,
            output_len=8,
            ignore_eos=True,
            max_concurrency=1,
            seed=2026,
            warmup_requests=1,
            extra_args=("--skip-chat-template",),
        )
    )

    assert sse_result.aggregate["completed"] == official_result.aggregate["completed"]
    assert sse_result.aggregate["failed"] == official_result.aggregate["failed"] == 0
    assert (
        sse_result.aggregate["total_input_tokens"]
        == official_result.aggregate["total_input_tokens"]
    )
    assert (
        sse_result.aggregate["total_output_tokens"]
        == official_result.aggregate["total_output_tokens"]
    )
    assert all(
        result.token_timestamps_valid
        and result.token_timestamp_source == "vllm_delta_token_ids"
        and len(result.token_timestamps) == result.output_tokens
        and len(result.itl_ns) == max(0, result.output_tokens - 1)
        for result in sse_result.request_results
    )
    assert sse_result.aggregate["itl_count"] == sum(
        max(0, result.output_tokens - 1) for result in sse_result.request_results
    )

    # The clients run sequentially, so exact latency equality is neither expected
    # nor desirable as an invariant. Both must nevertheless expose real, positive
    # client-side TTFT measurements for comparison in the saved artifacts.
    assert sse_result.aggregate["mean_ttft_ms"] > 0
    assert official_result.aggregate["mean_ttft_ms"] > 0
    assert all(
        result.e2e_ns is not None and result.e2e_ns > 0
        for result in official_result.request_results
    )
    assert all(
        result.tpot_ns is not None and result.tpot_ns > 0
        for result in official_result.request_results
    )
    official_native_itls = official_result.raw_result.get("itls")
    assert isinstance(official_native_itls, list)
    assert len(official_native_itls) == len(official_result.request_results)
    assert all(
        isinstance(intervals, list)
        and intervals
        and all(isinstance(value, (int, float)) and value >= 0 for value in intervals)
        for intervals in official_native_itls
    )
    assert official_result.aggregate["mean_itl_ms"] > 0
    assert all(
        (result.token_timestamps_valid and len(result.itl_ns) == max(0, result.output_tokens - 1))
        or (
            not result.token_timestamps_valid
            and result.itl_ns == []
            and result.token_timestamp_source == "official_vllm_itls_count_mismatch"
        )
        for result in official_result.request_results
    )

    latency_ratios = {}
    for metric in ("mean_ttft_ms", "mean_tpot_ms", "mean_e2e_ms"):
        sse_value = sse_result.aggregate[metric]
        official_value = official_result.aggregate[metric]
        assert isinstance(sse_value, (int, float)) and sse_value > 0
        assert isinstance(official_value, (int, float)) and official_value > 0
        ratio = float(sse_value) / float(official_value)
        assert math.isfinite(ratio) and 0.01 <= ratio <= 100.0
        latency_ratios[metric] = ratio

    if ARTIFACT_DIR:
        (artifact_dir / "sse-result.json").write_text(
            json.dumps(sse_result.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (artifact_dir / "official-result.json").write_text(
            json.dumps(official_result.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        source_commit, dirty_worktree, _ = git_state(Path.cwd())
        comparison = {
            "schema_version": 2,
            "base_url": BASE_URL,
            "model": MODEL,
            "source_commit": source_commit,
            "dirty_worktree": dirty_worktree,
            "environment": collect_environment_fingerprint().model_dump(mode="json"),
            "protocol": {
                "prompts": prompts,
                "max_tokens": 8,
                "ignore_eos": True,
                "max_concurrency": 1,
                "seed": 2026,
                "warmup_requests": 1,
                "sequential_backends": True,
                "latency_ratio_sanity_bounds": [0.01, 100.0],
                "latency_note": (
                    "Backends run sequentially; ratios are a unit/order-of-magnitude "
                    "sanity check, not an equality or performance claim."
                ),
            },
            "completed_equal": (
                sse_result.aggregate["completed"] == official_result.aggregate["completed"]
            ),
            "failed_equal": (sse_result.aggregate["failed"] == official_result.aggregate["failed"]),
            "total_input_tokens_equal": (
                sse_result.aggregate["total_input_tokens"]
                == official_result.aggregate["total_input_tokens"]
            ),
            "total_output_tokens_equal": (
                sse_result.aggregate["total_output_tokens"]
                == official_result.aggregate["total_output_tokens"]
            ),
            "official_per_request_e2e_available": all(
                result.e2e_ns is not None for result in official_result.request_results
            ),
            "official_per_request_tpot_available": all(
                result.tpot_ns is not None for result in official_result.request_results
            ),
            "sse_token_timestamps_valid": all(
                result.token_timestamps_valid for result in sse_result.request_results
            ),
            "sse_itl_count": sse_result.aggregate["itl_count"],
            "official_native_itl_counts": [len(intervals) for intervals in official_native_itls],
            "official_token_timestamp_sources": [
                result.token_timestamp_source for result in official_result.request_results
            ],
            "official_itl_limitation": (
                "Pinned vLLM may emit fewer native ITL intervals than output_tokens - 1; "
                "SLOTune preserves those native values but leaves per-request token ITL "
                "unavailable instead of fabricating a missing interval."
            ),
            "latency_ratios_sse_over_official": latency_ratios,
        }
        (artifact_dir / "comparison.json").write_text(
            json.dumps(comparison, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
