#!/usr/bin/env python3
"""Run the frozen M4 matrix in deterministic randomized policy order."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from transformers import AutoTokenizer

from vllm_tuner.experiment.formal import (
    AdaptivePrefillFormalMatrix,
    formal_job_order,
    load_formal_protocol,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        default="experiments/adaptive_prefill/m3_formal_protocol.yaml",
    )
    parser.add_argument(
        "--results-root",
        default="/root/autodl-tmp/vllm-tuner-output/slotune-results",
    )
    parser.add_argument("--experiment-prefix", default="slotune-m4-formal-20260817")
    parser.add_argument("--load", action="append", help="Run only a named load point")
    parser.add_argument("--policy", action="append", help="Run only a named policy")
    parser.add_argument("--repeat", action="append", type=int, help="Run only one repeat index")
    parser.add_argument(
        "--trace-kind",
        action="append",
        choices=("calibration", "heldout"),
    )
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


async def main_async(args: argparse.Namespace) -> int:
    protocol = load_formal_protocol(args.protocol)
    tokenizer_path = Path(protocol["pilot_model"]).expanduser()
    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_path), local_files_only=tokenizer_path.exists()
    )
    matrix = AdaptivePrefillFormalMatrix(
        args.protocol,
        results_root=args.results_root,
        experiment_prefix=args.experiment_prefix,
        repository=Path.cwd(),
        tokenizer=tokenizer,
    )
    jobs = formal_job_order(
        protocol,
        seed=args.seed,
        loads=args.load,
        policies=args.policy,
        repeats=args.repeat,
        trace_kinds=args.trace_kind,
    )
    progress_path = (
        Path(args.results_root).expanduser().resolve() / f"{args.experiment_prefix}-progress.jsonl"
    )
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    with progress_path.open("a", encoding="utf-8") as progress:
        for index, job in enumerate(jobs, start=1):
            print(
                f"[{index}/{len(jobs)}] load={job.load} policy={job.policy} "
                f"trace={job.trace_kind} repeat={job.repeat}",
                flush=True,
            )
            result = await matrix.run_job(job)
            row = {
                "index": index,
                "total": len(jobs),
                "load": job.load,
                "policy": job.policy,
                "trace_kind": job.trace_kind,
                "repeat": job.repeat,
                "trial_id": job.trial_id,
                "status": result.status.value,
                "completed": result.client.get("completed"),
                "failed": result.client.get("failed"),
                "achieved_requests_per_sec": result.client.get("achieved_requests_per_sec"),
            }
            progress.write(json.dumps(row, sort_keys=True) + "\n")
            progress.flush()
    sealed = matrix.finalize_ready_contexts()
    print(f"sealed_contexts={len(sealed)}", flush=True)
    for path in sealed:
        print(path, flush=True)
    return 0


def main() -> int:
    return asyncio.run(main_async(parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
