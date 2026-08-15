# Custom fixed workload traces

For comparable search trials, prefer an immutable `WorkloadTrace` JSONL over regenerating traffic
inside each trial. Supply one search trace and a different held-out trace:

```bash
vllm-tuner tune \
  --config config/formal_3b_chat.yaml \
  --study-name custom_trace_001 \
  --trace /root/autodl-tmp/traces/search.jsonl \
  --holdout-trace /root/autodl-tmp/traces/held-out.jsonl \
  --results-root /root/autodl-tmp/slotune-results
```

## `WorkloadTrace` JSONL schema

Each line is one request, ordered by `scheduled_offset_seconds`:

```json
{"request_id":"chat-000001","scheduled_offset_seconds":0.0,"prompt":"Explain KV cache pressure.","input_tokens":7,"output_tokens":128,"profile":"chat","shared_prefix_id":null}
```

Required fields:

- `request_id`: unique stable string;
- `scheduled_offset_seconds`: non-negative offset, monotonically ordered;
- `prompt`: exact text sent to the server;
- `input_tokens`: tokenizer-derived positive count;
- `output_tokens`: positive requested output count;
- `profile`: descriptive workload name;
- `shared_prefix_id`: optional identifier for prefix-reuse analysis.

The CLI records the trace checksum in the experiment manifest. Search methods and repeats use the
same file. Held-out data must not have influenced parameter selection.

## Generated profiles

Without CLI trace paths, these `workload.name` values are deterministic for a fixed seed:

| Profile | Approximate input | Approximate output | Purpose |
|---|---:|---:|---|
| `chat` | 192–320 | 96–160 | decode concurrency |
| `rag` | 1792–2304 | 96–160 | long prefill and shared prefix |
| `mixed` | 256–4096 | 64–256 | head-of-line blocking |
| `codegen` | 384–640 | 384–640 | decode-heavy TPOT |

`request_rate` and `burstiness` control seeded open-loop interarrivals. Use a positive request rate
for capacity experiments. The held-out generator uses a different deterministic seed.

## Local prompt datasets

When `workload.name` is not a named profile, `dataset_name` can point to JSON or JSONL containing
objects with `instruction` or `prompt` and optional `input`:

```json
{"instruction":"Summarize the context.","input":"Context text..."}
```

This prompt-loader path tokenizes the actual text before creating the trace. It is distinct from
the exact `WorkloadTrace` schema above.

## Scheduler-ablation JSONL

The standalone CPU simulator uses a smaller schema documented by its help:

```bash
python scripts/run_scheduler_ablation.py --help
```

A combined file adds `"split":"calibration"` or `"split":"held_out"` to each line and uses
`arrival_time`, `prompt_tokens`, and `output_tokens`. Do not confuse it with the runtime
`WorkloadTrace` format.

## Validation checklist

- IDs are unique within each trace.
- Offsets are sorted and non-negative.
- Token counts were produced by the recorded tokenizer.
- Search and holdout traces have separate checksums.
- The same fixed files are reused for every method and repeat.
- Raw request results retain errors and timeouts rather than dropping rows.
