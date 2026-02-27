# Custom Workload

This guide explains how to use your own dataset with vLLM-Tuner.

## Quick Start

### Option 1: Hugging Face Dataset

Use any dataset from Hugging Face Hub:

```yaml
workload:
  dataset_name: "your-dataset-name"
  sample_size: 100
  concurrent_requests: 10
```

### Option 2: Custom YAML Format

Create custom workload file:

```yaml
dataset_path: "data/my_prompts.jsonl"

prompts:
  - "What is the capital of France?"
  - "Explain quantum computing in simple terms."
  - "Write a poem about AI."
  - "Translate hello to Spanish."
  - "What is machine learning?"

workload:
  sample_size: 5
  concurrent_requests: 2
  max_tokens: 128
```

## Custom Workload Format

### JSONL Format

```json
[
  {
    "instruction": "Your question here",
    "input": "Optional context",
    "output": "Expected output (optional)"
  }
]
```

### Custom Plain Text File

```
prompt_1
prompt_2
prompt_3
...
```

## Loading from Local Dataset

```yaml
# Local directory
dataset_dir: "data/my_prompts/"

# workloadDirectory options
dataset_dir: "data/my_prompts/"  # Plain text prompts
# dataset_file: "data/prompts.jsonl"  # JSONL file
```

## Prompt Length

### Average Length

Configure for typical input sizes:

```yaml
workload:
  prompt_length_distribution: "auto"  # Auto (default)
  # OR
  prompt_length_distribution: "weighted"  # Weighted distribution
  # OR
  prompt_length_distribution: "uniform"  # Uniform distribution
```

### Custom Length Ranges

For specific requirements:

```yaml
workload:
  min_prompt_length: 50
  max_prompt_length: 500
```

### Prompt Template

Combine instruction and input:

```yaml
workload:
  prompt_template: "{instruction}\\n\\n{input}"

dataset_path: "data/my_prompts.jsonl"
```

## Examples

### Example 1: Code Generation Dataset

QueryList or custom dataset for code tasks.

### Example 2: Comparison/Dataset

```yaml
dataset_path: "data/comparisons.jsonl"
```

Each item:
```json
{
  "instruction": "Compare Python and JavaScript.",
  "input": "What are the main differences?",
  "output": null
}
```

### Example 3: Conversation Dataset

Format with conversational turns:

```yaml
dataset_dir: "data/conversations/"
```

Files:
- conversation_1.json
- conversation_2.json

Each file contains:
```json
{
  "messages": [
    {"role": "user", "content": "Hello!"},
    {"role": "assistant", "content": "Hi! How can I help?"}
  ]
}
```

## Benchmarking Options

Number of requests:

```yaml
workload:
  sample_size: 50       # Number of prompts
```

Concurrent requests:

```yaml
workload:
  concurrent_requests: 10   # Simulated concurrent clients
```

Warmup requests:

```yaml
workload:
  warmup_requests: 5     # Warmup trials to stabilize
```

Max output tokens:

```yaml
workload:
  max_tokens: 256        # Max tokens per response
```

## Dataset Prerequisites

### File Format Requirements

- Must be valid JSON or JSONL format
- Must have at least `sample_size` valid items
- Each item must have required field (instruction/input)

### Dataset Size

- Minimum: 10 prompts
- Recommended: 100-1000 prompts
- Large datasets increase run time but improve reliability

### Data Quality

- Prompts should be meaningful and realistic
- Avoid extremely long prompts unless testing edge cases
- Include variety in prompt lengths

## Troubleshooting

### Dataset Not Found

```bash
# Check file exists
ls -la data/prompts.jsonl

# Check file format
cat data/prompts.jsonl | head -5
```

### Dataset Loading Errors

```bash
# Validate JSONL structure
python3 << 'PYTHON'
import json
with open("data/prompts.jsonl") as f:
    for line in f:
        item = json.loads(line)
        print(item.keys())
PYTHON
```

### Insufficient Samples

Error: `Not enough prompts in dataset: found X, need Y`

**Solution:** Reduce `sample_size` in config
```yaml
workload:
  sample_size: 50  # Reduce if dataset is small
```
