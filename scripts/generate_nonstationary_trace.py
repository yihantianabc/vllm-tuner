#!/usr/bin/env python3
"""Generate frozen calibration and held-out non-stationary traces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer

from vllm_tuner.workloads.nonstationary import (
    CALIBRATION_PHASE_ORDER,
    DEFAULT_PILOT_PHASES,
    HELDOUT_PHASE_ORDER,
    empirical_request_rate,
    generate_nonstationary_trace,
    multiply_phase_counts,
    phase_boundaries,
    phase_manifest,
    scale_trace_to_empirical_rate,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate phase-labeled traces; labels are for offline analysis only."
    )
    parser.add_argument(
        "--tokenizer",
        default="/root/autodl-tmp/models/Qwen2.5-3B-Instruct",
        help="Local tokenizer or Hugging Face identifier",
    )
    parser.add_argument(
        "--output-dir",
        default="experiments/adaptive_prefill/traces/pilot",
        help="Destination for trace JSONL, checksums, and manifest",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--phase-gap-seconds", type=float, default=0.05)
    parser.add_argument(
        "--count-multiplier",
        type=int,
        default=1,
        help="Multiply every phase count; use 20 for the 640-request Formal traces",
    )
    parser.add_argument(
        "--scaled-load",
        action="append",
        default=[],
        metavar="LABEL=REQUESTS_PER_SECOND",
        help="Also write calibration/heldout traces scaled to an exact empirical rate",
    )
    return parser.parse_args()


def parse_scaled_loads(values: list[str]) -> dict[str, float]:
    loads: dict[str, float] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"scaled load must use LABEL=RATE: {value}")
        label, raw_rate = value.split("=", 1)
        if not label or label in loads:
            raise ValueError(f"scaled load labels must be non-empty and unique: {label}")
        rate = float(raw_rate)
        if rate <= 0:
            raise ValueError("scaled load rates must be positive")
        loads[label] = rate
    return loads


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_path = Path(args.tokenizer).expanduser()
    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_path),
        local_files_only=tokenizer_path.exists(),
    )
    scaled_loads = parse_scaled_loads(args.scaled_load)
    traces = {}
    for label, order in (
        ("calibration", CALIBRATION_PHASE_ORDER),
        ("heldout", HELDOUT_PHASE_ORDER),
    ):
        phases = multiply_phase_counts(
            [DEFAULT_PILOT_PHASES[name] for name in order], args.count_multiplier
        )
        trace = generate_nonstationary_trace(
            phases,
            seed=args.seed,
            tokenizer=tokenizer,
            phase_gap_seconds=args.phase_gap_seconds,
        )
        path = trace.write(output_dir / f"{label}.jsonl")
        checksum = trace.checksum()
        (output_dir / f"{label}.sha256").write_text(f"{checksum}  {path.name}\n", encoding="utf-8")
        traces[label] = {
            "file": path.name,
            "sha256": checksum,
            "phase_order": list(order),
            "phase_config": phase_manifest(phases),
            "boundaries": phase_boundaries(trace),
            "requests": len(trace.entries),
            "empirical_requests_per_second": empirical_request_rate(trace),
        }
        if scaled_loads:
            traces[label]["scaled_loads"] = {}
            for load_label, target_rate in scaled_loads.items():
                scaled = scale_trace_to_empirical_rate(trace, target_rate)
                scaled_path = scaled.write(output_dir / f"{label}-load-{load_label}.jsonl")
                scaled_checksum = scaled.checksum()
                (output_dir / f"{label}-load-{load_label}.sha256").write_text(
                    f"{scaled_checksum}  {scaled_path.name}\n", encoding="utf-8"
                )
                traces[label]["scaled_loads"][load_label] = {
                    "file": scaled_path.name,
                    "sha256": scaled_checksum,
                    "target_requests_per_second": target_rate,
                    "empirical_requests_per_second": empirical_request_rate(scaled),
                }
    manifest = {
        "schema_version": 1,
        "seed": args.seed,
        "tokenizer": str(tokenizer_path.resolve()),
        "phase_gap_seconds": args.phase_gap_seconds,
        "count_multiplier": args.count_multiplier,
        "traces": traces,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
