# SLOTune M2 three-state adaptive Prefill controller

M2 implements the project's only vLLM scheduling contribution: an online three-state
controller that selects a per-step Prefill cap from Decode backlog, oldest Prefill wait, and KV
cache usage.

## Design and source changes

- `src/vllm_tuner/scheduler/controller.py` is a deterministic pure state machine with
  `PROTECT_DECODE`, `BALANCED`, and `DRAIN_PREFILL` states.
- State changes require configurable consecutive hysteresis steps and minimum state residency.
  Decode backlog and KV pressure take priority over Prefill drain.
- `max_wait_ms` reuses the oldest-Prefill-wait signal and requests at least
  `min_prefill_progress`; logs distinguish a completed short final chunk from genuine lack of
  progress.
- `src/vllm_tuner/scheduler/runtime.py` reserves stock per-step Decode demand before exposing the
  remaining safe Prefill budget. Reservation is a safety clamp, not a fourth state-selection
  signal.
- The patch `patches/vllm-v0.16.0/0001-add-prefill-token-budget-hook.patch` adds one default-on
  Prefill-budget hook and one independent counter to the stock scheduling loops. Queue order,
  KV allocation/freeing, preemption, request state, and model execution remain owned by vLLM.
- `scripts/apply_vllm_scheduler_patch.sh` checks vLLM 0.16.0 and exact upstream/patched hashes,
  applies the patch idempotently, and refuses unknown Scheduler sources.
- Enabling control without the hook fails at Scheduler construction instead of silently running
  a no-op policy.
- A missing decision log is now an explicit unavailable artifact; a selectable custom-Scheduler
  trial requires that evidence.

The patch applies cleanly to upstream commit
`89a77b10846fd96273cce78d86d2556ea582d26e`. The upstream Scheduler SHA-256 is
`bb36be85a1054cdbfedb35c1f04ee02696d9f94a076f7829e9da0bb4f7987d07`; the patched file is
`ed1b8dc7816a48b69710631e67d929f6d4c3870ce868f422cd820389bd08731c`.

## Tests

- `ruff check src tests`: passed.
- `mypy src`: passed, 65 source files.
- `pytest -q tests/unit`: 314 passed, 44 dependency deprecation warnings.
- Pure controller tests cover all states, Decode/KV priority, hysteresis, minimum residency,
  max-wait progress, and deterministic replay.
- Patch dry-run against a pristine v0.16.0 sparse checkout: passed.
- Adapted upstream CPU checks using the local Qwen3 config: stock token budget, capped Prefill
  budget, preemption/update, and prefix-cache reset passed.
- The unmodified upstream pytest file could not be run directly because its default
  `facebook/opt-125m` config fetch failed through the host's Hugging Face SOCKS/TLS path. This
  was an external test-fixture download failure, not a Scheduler assertion failure; the adapted
  local-model cases were retained instead of hiding the failed attempt.

## Real V1 GPU smoke

Final config: `experiments/adaptive_prefill/m2_adaptive_smoke.yaml`.

Artifact root:
`/root/autodl-tmp/vllm-tuner-output/slotune-results/slotune-m2-adaptive-final-20260817`.

The smoke used eight exact 1024-token prompts, eight exact 64-token outputs, global budget 1024,
and Prefill caps 128/256/512. It is a mechanism test, not a performance comparison.

| Trial | Result | Steps | State counts (Balanced / Drain / Protect) | Prefill / Decode tokens | Scheduler CPU p50 / p99 / max |
|---|---|---:|---:|---:|---:|
| `random-0000` | COMPLETE; 8/8 success; every output 64 tokens | 234 | 4 / 2 / 228 | 7728 / 567 | 79.039 / 201.898 / 415.118 us |
| `repeat-random-0-0` | COMPLETE; 8/8 success; every output 64 tokens | 232 | 4 / 2 / 226 | 7728 / 567 | 78.878 / 171.067 / 528.255 us |

Across both trials:

- all three states and all three configured caps appeared;
- scheduled Prefill tokens never exceeded the selected cap;
- Prefill plus Decode tokens equaled total scheduled tokens, which never exceeded 1024;
- every step with a nonzero Decode backlog scheduled at least one token for every Decode
  request in this non-speculative smoke;
- max-wait rows had progress; one short 32-token final remainder per trial was correctly logged
  as completed rather than starved;
- no preemption, timeout, request error, assertion, leaked process, or GPU residue occurred;
- both decision logs are present and checksum-sealed.

The controller-disabled regression was rerun after applying the patch at
`/root/autodl-tmp/vllm-tuner-output/slotune-results/slotune-m2-disabled-regression-20260817`.
Both trials completed with 2/2 successful requests, output tokens `32, 32`, 99 `DISABLED` rows,
and the same 38 Prefill / 93 Decode scheduled-token totals as M1. This verifies that the hook's
default budget preserves stock behavior for the covered path.

## M2 conclusion and limits

M2 passes its implementation and smoke acceptance criteria. It proves state transitions,
budget conservation, Decode reservation, max-wait handling, artifact integrity, and disabled
fallback on a real V1 EngineCore. It does not prove a performance gain or an overhead percentage:
the smoke is too small and its thresholds intentionally force state changes. M3 must build the
non-stationary workload, run fixed caps and a capacity sweep, then freeze Pilot thresholds before
any Formal claim.
