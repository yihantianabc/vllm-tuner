# CLI Commands

## Available Commands

### `vllm-tuner tune`

Run a tuning study.

```bash
vllm-tuner tune --config <config.yaml> --study-name <name>
```

#### Options

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--config` | `-c` | config/default.yaml | YAML config file |
| `--study-name` | `-n` | Required | Study name |
| `--model` | `-m` | From config | Override model name |
| `--gpu-count` | `-c` | From config | Override GPU count |
| `--no-progress` | | False | Disable progress bar |

#### Examples

```bash
# Basic tuning
vllm-tuner tune --config config/default.yaml --study-name my_study

# Override model
vllm-tuner tune --config config/default.yaml --study-name test --model gpt2

# Multi-GPU with progress disabled
vllm-tuner tune --config docs/user-guide/examples/multi_gpu_tune.yaml --study-name llama --no-progress
```

### `vllm-tuner report`

Generate reports from completed study.

```bash
vllm-tuner report --study-name <name> --format html
```

Formats: html, json, markdown

### `vllm-tuner export`

Export best configuration.

```bash
vllm-tuner export --study-name <name> --format yaml
```

### `vllm-tuner list-studies`

List all available studies.

```bash
vllm-tuner list-studies
```
