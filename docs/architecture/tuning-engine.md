# SLO-aware tuning engine

## Search space

The core searches only settings that change one-GPU vLLM serving:

```yaml
search_space:
  gpu_memory_utilization: [0.60, 0.95]
  max_num_seqs: [8, 16, 32, 64, 128]
  max_num_batched_tokens: [1024, 2048, 4096, 8192]
  tensor_parallel_size: 1
  pipeline_parallel_size: 1
```

The old `batch_size` example was ineffective and is rejected. Parallel sizes are experiment
constants for the one-card core. Duplicate ownership between search-space keys and `vllm_args`
is rejected.

## Objective

```python
goodput_requests_per_second = requests_meeting_all_slos / measurement_seconds
```

There is one maximize direction. Throughput, tail latency, and memory do not receive arbitrary
weights; they are evidence and/or constraints. Per-request SLO decisions are retained.

## Method budget

For configured budget `N`, default, seeded random, and constrained TPE each collect `N` measured
COMPLETE/INFEASIBLE outcomes. FAILED/PRUNED attempts remain recorded but do not masquerade as
successful evaluations. The controller limits attempts so a broken environment cannot loop
forever.

TPE receives explicit constraint values for infeasible trials. Manual best selection includes
only COMPLETE trials with a real objective. Known failures cannot become best through a sentinel
number.

## Candidate validation

The best default/random candidates and top TPE candidates are deduplicated by parameter set,
repeated according to `repeat_count`, and then evaluated on held-out traffic when enabled. Formal
configs use three repeats. Reports distinguish search, repeat, and held-out rows.

## Reproducibility

Seeded samplers, a persisted trace, fixed software/model identity, and an environment manifest make
the suggestion/evaluation chain inspectable. Resume is opt-in and validates manifest/search-space
compatibility before appending evidence.
