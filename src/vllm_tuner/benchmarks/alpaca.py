"""Alpaca workload loader with prompt sampling."""

import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from datasets import load_dataset  # type: ignore[import-untyped]
except ImportError:
    raise ImportError("datasets package is required. Install with: pip install datasets")

from transformers import AutoTokenizer

from vllm_tuner.benchmarks.workload import Workload
from vllm_tuner.config.models import WorkloadConfig

logger = logging.getLogger(__name__)

random.seed(2026)


class AlpacaWorkload(Workload):
    """Workload based on the Alpaca dataset."""

    def __init__(self, config: WorkloadConfig):
        super().__init__(config)
        self._dataset = None
        self.tokenizer: Optional[Any] = None
        self._prompt_lengths: Optional[List[int]] = None

    async def load(self) -> List[str]:
        """Load and sample prompts from Alpaca dataset."""
        logger.info(f"Loading Alpaca dataset: {self.config.dataset_name}")

        try:
            dataset_path = Path(self.config.dataset_name).expanduser()
            if dataset_path.is_file():
                dataset_format = {
                    ".csv": "csv",
                    ".json": "json",
                    ".jsonl": "json",
                }.get(dataset_path.suffix.lower())
                if dataset_format is None:
                    raise ValueError("Local datasets must use a .csv, .json, or .jsonl extension")
                dataset = load_dataset(
                    dataset_format,
                    data_files=str(dataset_path),
                    split="train",
                )
            else:
                dataset = load_dataset(self.config.dataset_name, split="train")
            logger.info(f"Loaded {len(dataset)} examples from Alpaca dataset")
        except Exception as e:
            logger.error(f"Failed to load Alpaca dataset: {e}")
            raise

        prompts = self._extract_prompts(dataset)

        if len(prompts) > self.config.sample_size:
            prompts = self._sample_prompts(prompts, dataset)

        if not prompts:
            raise ValueError("No prompts loaded from dataset")

        logger.info(
            f"Loaded {len(prompts)} prompts with distribution strategy: {self.config.prompt_length_distribution}"
        )

        return prompts

    def _extract_prompts(self, dataset) -> List[str]:
        """Extract prompts from dataset examples."""
        prompts = []

        for example in dataset:
            instruction = example.get("instruction", "")
            input_text = example.get("input", "")

            if input_text:
                prompt = f"{instruction}\n\n{input_text}"
            else:
                prompt = instruction

            if prompt.strip():
                prompts.append(prompt)

        logger.info(f"Extracted {len(prompts)} prompts from dataset")
        return prompts

    def _sample_prompts(self, prompts: List[str], dataset) -> List[str]:
        """Sample prompts based on configured strategy."""
        strategy = self.config.prompt_length_distribution
        sample_size = self.config.sample_size

        if strategy == "uniform":
            return random.sample(prompts, sample_size)

        elif strategy == "weighted":
            return self._weighted_length_sample(prompts)

        else:  # auto or default
            return self._auto_sample(prompts, sample_size)

    def _weighted_length_sample(self, prompts: List[str]) -> List[str]:
        """Sample prompts weighted by length (diverse lengths)."""
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained("gpt2")

        prompt_lengths = [len(self.tokenizer.encode(p)) for p in prompts]

        length_ranges = [
            (0, 50),
            (51, 100),
            (101, 200),
            (201, 500),
            (501, float("inf")),
        ]

        samples_per_range = self.config.sample_size // len(length_ranges)

        sampled = []

        for min_len, max_len in length_ranges:
            range_prompts = [
                prompt
                for prompt, length in zip(prompts, prompt_lengths)
                if min_len <= length <= max_len
            ]

            if len(range_prompts) > samples_per_range:
                sampled.extend(random.sample(range_prompts, samples_per_range))
            else:
                sampled.extend(range_prompts)

        while len(sampled) < self.config.sample_size and len(prompts) > len(sampled):
            remaining_prompts = [p for p in prompts if p not in sampled]
            if remaining_prompts:
                sampled.append(random.choice(remaining_prompts))
            else:
                break

        return sampled[: self.config.sample_size]

    def _auto_sample(self, prompts: List[str], sample_size: int) -> List[str]:
        """Auto sample with mix of strategies based on dataset size."""
        if len(prompts) <= sample_size:
            return prompts

        if len(prompts) < 1000:
            return random.sample(prompts, sample_size)

        return self._weighted_length_sample(prompts)

    def get_prompt_lengths(self, prompts: List[str]) -> List[int]:
        """Get token lengths for prompts."""
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained("gpt2")

        lengths = [len(self.tokenizer.encode(p)) for p in prompts]
        return lengths

    def get_length_stats(self, prompts: List[str]) -> Dict[str, Any]:
        """Get statistics about prompt lengths."""
        lengths = self.get_prompt_lengths(prompts)

        if not lengths:
            return {}

        return {
            "count": len(lengths),
            "min": min(lengths),
            "max": max(lengths),
            "mean": sum(lengths) / len(lengths),
            "median": sorted(lengths)[len(lengths) // 2],
            "p25": sorted(lengths)[int(len(lengths) * 0.25)],
            "p75": sorted(lengths)[int(len(lengths) * 0.75)],
        }

    def get_metadata(self) -> Dict[str, Any]:
        """Get workload metadata."""
        prompts = self._prompts or []

        metadata = {
            "name": self.config.name,
            "dataset_name": self.config.dataset_name,
            "sample_size": self.config.sample_size,
            "num_prompts": len(prompts),
            "prompt_length_distribution": self.config.prompt_length_distribution,
        }

        if prompts:
            metadata["length_stats"] = self.get_length_stats(prompts)

        return metadata


def create_alpaca_workload(config: WorkloadConfig) -> AlpacaWorkload:
    """Create an Alpaca workload from configuration."""
    return AlpacaWorkload(config)
