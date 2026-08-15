"""Unit tests for the Alpaca workload loader."""

import pytest

from vllm_tuner.benchmarks.alpaca import AlpacaWorkload
from vllm_tuner.config.models import WorkloadConfig


@pytest.mark.asyncio
async def test_load_local_jsonl_dataset(tmp_path):
    """Load local Alpaca-style JSONL without contacting Hugging Face Hub."""
    dataset_path = tmp_path / "alpaca.jsonl"
    dataset_path.write_text(
        '{"instruction":"Say hello","input":"","output":"Hello"}\n'
        '{"instruction":"Add numbers","input":"2 + 2","output":"4"}\n',
        encoding="utf-8",
    )
    config = WorkloadConfig(
        dataset_name=str(dataset_path),
        sample_size=2,
        prompt_length_distribution="uniform",
    )

    prompts = await AlpacaWorkload(config).load()

    assert prompts == ["Say hello", "Add numbers\n\n2 + 2"]
