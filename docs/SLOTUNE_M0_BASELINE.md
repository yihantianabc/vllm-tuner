# SLOTune M0 version freeze and bring-up baseline

This is a correctness and reproducibility checkpoint for the adaptive-prefill project. It is
not performance evidence and must not be used to claim a scheduler speedup.

## Frozen source and runtime

- Repository source commit: `6925e1dc2de0e8b94a84c01a0dfc1c0c3db748c6`.
- The run intentionally used `--allow-dirty-source`: the new project plan and the M0 files were
  uncommitted while the checkpoint was created. Formal experiments must use a clean tree.
- vLLM release: `0.16.0`; upstream tag commit:
  `89a77b10846fd96273cce78d86d2556ea582d26e`.
- The wheel and relevant scheduler-source hashes are frozen in
  `patches/vllm-v0.16.0/upstream.lock.yaml`.
- Python 3.12.3, PyTorch 2.9.1+cu130, CUDA 13.0, Transformers 4.57.6,
  FlashInfer 0.6.3, NVIDIA driver 595.71.05.
- GPU: NVIDIA GeForce RTX 5090, 32607 MiB, compute capability 12.0.

## Checks performed on 2026-08-17

| Check | Result | Artifact |
|---|---|---|
| Stock scheduler, Qwen3-0.6B, one observation plus three same-trace repeats | 4/4 trials complete; 8/8 measured requests successful; every request returned 32 output tokens | `/root/autodl-tmp/vllm-tuner-output/slotune-results/slotune-m0-default-20260817` |
| Stock scheduler, Qwen2.5-3B-Instruct bring-up | 2/2 trials complete; 4/4 measured requests successful; output-token counts matched across runs (`32`, `13`) | `/root/autodl-tmp/vllm-tuner-output/slotune-results/slotune-m0-3b-20260817` |
| Custom synchronous passthrough scheduler | 2/2 trials complete; 4/4 measured requests successful; both server commands contain `--scheduler-cls` and `--no-async-scheduling` | `/root/autodl-tmp/vllm-tuner-output/slotune-results/slotune-m0-custom-scheduler-sync-20260817` |

All suites used trace SHA-256
`8fb2cf6503ce874f55bb1f031c49549a7ae9052f7bc4d649f1f471f22f461721`.
The custom-scheduler logs contain vLLM's class-resolution warning with the fully qualified
`vllm_tuner.scheduler.passthrough.PassthroughScheduler` name and confirm that asynchronous
scheduling was disabled. GPU process inspection after each suite found no remaining compute
process.

The three 0.6B repeat goodputs were 9.669, 10.611, and 12.666 requests/s. This spread is
expected for a two-request correctness smoke dominated by startup/runtime jitter. Completion
count, error count, and output-token counts are the M0 repeatability criteria; these latency and
throughput values are explicitly not benchmark results.

One superseded loading-path artifact exists at
`/root/autodl-tmp/vllm-tuner-output/slotune-results/slotune-m0-custom-scheduler-20260817`.
Its YAML used `async-scheduling: false`, which the generic CLI builder correctly treated as an
omitted boolean flag, leaving vLLM's auto-selected async mode enabled. The corrected config uses
`no-async-scheduling: true`; only the corrected `-sync-` artifact is accepted for M0.

## M0 conclusion

M0 passes. The installed vLLM version and relevant sources are frozen, both local models run,
the stock path is repeatable at correctness-smoke granularity, the custom Scheduler can be
loaded in the real V1 EngineCore process, and cleanup is clean. M1 can add step-level
instrumentation while retaining the synchronous stock scheduler as the disabled-controller
reference.
