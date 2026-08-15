# CLI commands

Use `vllm-tuner --help` and `vllm-tuner COMMAND --help` as the authoritative option list.

## `tune`

```bash
vllm-tuner tune \
  --config config/formal_3b_chat.yaml \
  --study-name qwen25_3b_chat_001 \
  --results-root /root/autodl-tmp/slotune-results
```

Important options:

| Option | Meaning |
|---|---|
| `--config PATH` | validated YAML; defaults to `config/default.yaml` |
| `--study-name NAME` | experiment ID; use a unique name |
| `--model PATH` | optional model override |
| `--gpu-count 1` | core supports exactly one GPU |
| `--results-root PATH` | explicit immutable experiment root |
| `--trace PATH` | fixed search `WorkloadTrace` JSONL |
| `--holdout-trace PATH` | fixed unseen validation JSONL |
| `--with-progress` | acknowledge per-trial status progress |

`--baseline/--no-baseline` is a deprecated compatibility option. The equal-budget `default`
method is the comparison baseline.

## Scheduler ablation

```bash
python scripts/run_scheduler_ablation.py \
  --output-dir /root/autodl-tmp/slotune-scheduler/mixed
```

The standalone script needs no GPU. It supports built-in deterministic traces, a combined JSONL
with `split: calibration|held_out`, or separate trace files. Its default fixed budgets are
512/1024/2048/4096/8192, and it writes JSON plus Markdown including negative conditions.

## `report`

```bash
vllm-tuner report --study-name STUDY --format html
```

Supported legacy report formats are HTML, JSON, and Markdown. The complete SLOTune experiment
runner also writes static reports directly inside the explicit results root.

## `export`

```bash
vllm-tuner export --study-name STUDY --format yaml --output /explicit/best.yaml
```

Only a successful selectable candidate can be exported.

## `list-studies`

```bash
vllm-tuner list-studies
```

This lists studies in the configured legacy study directory; explicit SLOTune experiment roots
remain the primary artifact location.
