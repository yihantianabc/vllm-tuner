# Common issues

## Configuration rejects `objectives` or `batch_size`

This is intentional. Replace weighted objectives with latency SLOs and hard constraints:

```yaml
slo:
  ttft_ms: 1000
  tpot_ms: 100
  e2e_ms: null
constraints:
  max_error_rate: 0.01
  require_no_oom: true
search_space:
  gpu_memory_utilization: [0.60, 0.95]
  max_num_seqs: [8, 16, 32, 64]
  max_num_batched_tokens: [1024, 2048, 4096]
  tensor_parallel_size: 1
  pipeline_parallel_size: 1
```

## Multi-GPU configuration is rejected

The SLOTune core is deliberately one GPU. Use exactly one device and fixed TP/PP of one. Extending
the protocol to multi-GPU requires new lifecycle, topology, search, and reporting validation.

## Search-space/vLLM argument collision

Do not put a searched value such as `max-num-seqs` in `vllm_args`. Choose one owner: tune it in
`search_space`, or remove it from search and make it a fixed runtime argument.

## vLLM is missing or the server never becomes ready

```bash
uv pip install vllm --torch-backend=auto
nvidia-smi
```

Inspect `trials/<trial-id>/server.log`, `server-command.json`, and `status.json`. The structured
failure should distinguish port conflict, invalid argument, model load, startup timeout, OOM, and
unexpected exit. Do not relabel an arbitrary exception as OOM.

## Benchmark request fails

Check:

- prompt + output length against `max-model-len`;
- model name/path passed to the OpenAI endpoint;
- raw HTTP/SSE error in `request-results.jsonl`;
- official benchmark stdout/stderr and detailed JSON;
- request timeout and server log.

Asynchronous task exceptions should appear as failed request rows, not disappear from aggregates.

## Token totals are zero

The artifact is not acceptable as formal evidence. Verify tokenizer loading, response usage, and
official detailed-result parsing. The strict SSE client should reject uncountable successful
responses.

## GPU or engine telemetry is unavailable

Missing telemetry must be marked unavailable. Check `/metrics`, NVML permissions, selected device,
and `telemetry.jsonl`. Do not replace missing samples with zero utilization or memory.

## No feasible candidate

Inspect constraint violations before widening the search. Possible causes include an unrealistic
SLO, overload, insufficient VRAM, errors, or server exits. Preserve the no-feasible result. If the
protocol itself is wrong, start a new experiment with a new manifest rather than silently changing
the current study.

## Adaptive scheduler shows no gain

Open `scheduler_ablation.json` or the standalone Markdown report. Light load, homogeneous request
lengths, slow hysteresis, admission contraction, or fairness preemption may make adaptive tie or
regress. This is a valid negative result; confirm it on held-out data.

## Existing output directory

Experiment and demo tools refuse accidental reuse by default. Choose a unique study/output path,
use validated resume only for a matching manifest, or pass the standalone scheduler script's
explicit `--overwrite` when replacement is intentional.
