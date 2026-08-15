# SLOTune configurations

| File | Purpose |
|---|---|
| `reproduction_smoke.yaml` | two-request Qwen3-0.6B GPU correctness smoke; never benchmark evidence |
| `default.yaml` | default 3B mixed-profile SLO-goodput experiment |
| `formal_3b_chat.yaml` | 500-request chat protocol, equal default/random/TPE budgets, three repeats, held-out |
| `formal_3b_rag.yaml` | 500-request RAG protocol, equal default/random/TPE budgets, three repeats, held-out |

All core configs are one GPU with TP/PP fixed to one. The formal files assume the local model path
`/root/autodl-tmp/models/Qwen2.5-3B-Instruct`; change the path only by starting a separately named
experiment whose manifest records the new identity.

The formal configs use SSE to replay frozen arrival offsets and declare a 1/2/4/8/16/32
requests/s capacity sweep with three repeats per rate. A final best is exported only after every
configured repeat and holdout is COMPLETE/feasible and the holdout median goodput is at least
80% of the repeat median. Resume replays validated immutable trial artifacts; the legacy
`storage_backend` setting is not the core search-state authority. Official bench remains the live
reference backend for cross-validation.

These files are protocols, not results. The local Qwen2.5-3B preflight used only a two-request
trace and does not satisfy the formal chat/RAG, repeat, capacity, or held-out requirements. Keep
formal result status and immutable artifact paths in the
[`REPRODUCTION.md` register](../REPRODUCTION.md#current-real-results-register), and use the
[`project-plan audit`](../docs/PLAN_AUDIT.md) to distinguish implementation from experimental
completion.

Validate without launching a server:

```bash
python - <<'PY'
from pathlib import Path
from vllm_tuner.config.validation import load_yaml_config

for path in sorted(Path("config").glob("*.yaml")):
    print(path, load_yaml_config(path).workload.name)
PY
```
