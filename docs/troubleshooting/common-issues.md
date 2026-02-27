# Common Issues

## Installation Issues

### vLLM Not Found

**Problem:**
```
ModuleNotFoundError: No module named 'vllm'
```

**Solution:**
```bash
uv pip install vllm --torch-backend=auto
```

### Pydantic Validation Error

**Problem:**
```
ValidationError: Weights must sum to 100
```

**Solution:**
Ensure objectives sum to 100:
```yaml
objectives:
  throughput: 60
  latency: 30
  memory: 10  # Total: 100
```

### GPU Not Available

**Problem:**
```
NVML initialization failed: No device found
```

**Solution:**
```bash
# Check GPU availability
nvidia-smi

# Set correct CUDA_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES=0
```

## Runtime Issues

### OOM Errors

See [OOM Errors](oom-errors.md).

### vLLM Server Not Starting

**Check logs:**
```bash
tail studies/<study_name>/logs/vllm_baseline.log
studies/<study_name>/logs/vllm_trial_0.log
```

**Common causes:**
- Not enough GPU memory - reduce gpu_memory_utilization
- Model too large for GPU - use smaller model
- Missing dependencies - `uv pip install vllm --torch-backend=auto`

### Benchmark Fails (400 Bad Request)

**Problem:**
```
Request req_X failed: 400
```

**Causes:**
- Model sequences too long (max-model-len issue)
- Invalid prompts
- Authentication issues

**Solution:**
- Increase max-model-len in config
- Set proper max_tokens value
- Check vLLM server logs for details

### Study Not Improving

**Problem:**
Throughput/latency stays constant across trials.

**Solutions:**
- Widen search space ranges
- Relax constraints (increase max_latency_ms)
- Enable pruning to stop bad trials early
- Check for OOM errors in logs
- Increase warmup_requests
