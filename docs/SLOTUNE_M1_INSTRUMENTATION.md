# SLOTune M1 Scheduler instrumentation

M1 adds read-only step instrumentation around the stock synchronous vLLM V1 Scheduler. The
adaptive controller is disabled in this milestone; `super().schedule()` remains the only code
that makes scheduling decisions.

## Implementation

- `src/vllm_tuner/scheduler/runtime.py` loads the real V1 custom Scheduler and records one
  decision row before returning each stock `SchedulerOutput`.
- `src/vllm_tuner/scheduler/instrumentation.py` contains vLLM-independent signal collection,
  token classification, the decision schema, and the line-buffered JSONL writer.
- `AdaptivePrefillConfig` validates ordered caps, wait thresholds, hysteresis, and logging
  settings. M1 runs it with `enabled: false`.
- `ManagedVLLMServer` passes the validated configuration and a trial-local decision-log path to
  EngineCore. The effective non-secret environment is preserved in `server-command.json`.
- `scheduler-decisions.jsonl` is linked from the trial summary, represented in
  `artifact-status.json`, and checksum-sealed by `artifact-integrity.json`.

Each row contains the three future controller inputs (`decode_backlog`,
`oldest_prefill_wait_ms`, and `kv_cache_usage`), state/cap/reason fields, scheduled Prefill and
Decode tokens, queue sizes, preemption delta, and Scheduler CPU time. Signal collection reads
existing request/KV state and performs no GPU operation or synchronization.

## Verification on 2026-08-17

Config: `experiments/adaptive_prefill/m1_instrumentation_smoke.yaml`.

Artifact root:
`/root/autodl-tmp/vllm-tuner-output/slotune-results/slotune-m1-instrumentation-20260817`.

| Trial | Result | Step rows | Prefill tokens | Decode tokens | Scheduler CPU p50 / max |
|---|---|---:|---:|---:|---:|
| `default-0000` | COMPLETE, 2/2 requests successful, output tokens `32, 32` | 99 | 38 | 93 | 91.744 / 334.232 us |
| `repeat-default-0-0` | COMPLETE, 2/2 requests successful, output tokens `32, 32` | 99 | 38 | 93 | 72.237 / 338.675 us |

For every one of the 198 rows:

- state is `DISABLED` and reason is `controller_disabled`;
- `scheduled_prefill_tokens + scheduled_decode_tokens == total_scheduled_tokens`;
- `step_index` is contiguous from zero;
- KV usage is in `[0, 1]`;
- all required fields are present;
- the decision log is available and checksum-sealed.

The request completion and output-token counts match the M0 synchronous passthrough smoke. The
two-request test is only a behavior/instrumentation smoke, so its latency and goodput are not
used to estimate instrumentation overhead.

Code quality and regression results:

- `ruff check src tests`: passed;
- `mypy src`: passed, 64 source files;
- `pytest -q tests/unit`: 306 passed, 44 pre-existing dependency deprecation warnings;
- M1 GPU smoke: 2 passed, 0 failed, 0 skipped;
- post-run GPU process check: clean.

## M1 conclusion

M1 passes. The three control signals and scheduled-token outcomes are observable on a real
vLLM V1 run, artifacts are auditable, and disabling control preserves stock Scheduler decision
ownership. M2 can now add the pure three-state controller and use its selected Prefill cap in
the custom Scheduler.
