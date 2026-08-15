# Out-of-memory failures

An OOM is a structured trial failure, not a poor numeric objective. The controller records the
failure phase, log evidence, exit status, and final server state, stops the process group, and
excludes the trial from best selection.

## Common pressure sources

- model weights and runtime compilation/CUDA graph memory;
- high `gpu_memory_utilization` KV allocation;
- many admitted sequences via `max_num_seqs`;
- long contexts or `max-model-len`;
- large `max_num_batched_tokens` prefill work;
- other processes using the GPU.

There is no vLLM server `batch_size` knob in the SLOTune search space.

## Conservative search

```yaml
constraints:
  max_peak_vram_mb: 30000
  max_memory_utilization: 0.92
  require_no_oom: true
search_space:
  gpu_memory_utilization: [0.60, 0.85]
  max_num_seqs: [8, 16, 32]
  max_num_batched_tokens: [1024, 2048, 4096]
  tensor_parallel_size: 1
  pipeline_parallel_size: 1
vllm_args:
  max-model-len: 4096
```

Lower one dimension at a time when diagnosing. Keep the workload trace fixed so the change remains
interpretable.

## Confirm classification and cleanup

1. Inspect `server.log` for CUDA allocation evidence.
2. Check the structured failure type/phase in `status.json` and `summary.json`.
3. Confirm the server process group exited and GPU memory returned before the next trial.
4. Distinguish OOM from invalid arguments, timeouts, request errors, or unrelated runtime failures.
5. Keep the failed trial in the aggregate report.

Peak VRAM comes from the sampled NVML time series. If NVML was unavailable, the report must say so
instead of claiming a zero peak.
