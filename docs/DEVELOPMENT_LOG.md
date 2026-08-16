# SLOTune 开发者工作日志（2026-08-15—16）

这是一份项目维护者视角的 engineering log，不是逐字聊天记录。它依据 Git 提交元数据、
命令输出、正式实验日志和保存下来的 artifact 复盘实际开发过程；没有日志时间戳的工作只按
先后顺序记录，不补造精确时刻。除特别注明 `+08:00` 外，实验时间统一写 UTC。

文中证据分三类：

- **真实 GPU 证据**：RTX 5090 上保存了逐请求、Prometheus、NVML、服务日志和完整性记录
  的 smoke、交叉验证或 Qwen2.5-3B formal artifact。
- **CPU simulator 证据**：固定/adaptive token-budget 仿真，只能说明策略机制、公平性和
  负结果，不能说明 vLLM runtime 获得 GPU 加速。
- **审计或推断**：从 sealed raw data 复算出的计数、比例和限制；凡是没有直接测量的容量上限、
  7B/8B 泛化或因果关系，都不写成实验结论。

## 最终证据坐标

| 项目 | 记录 |
|---|---|
| 测量源码 | clean commit `34a25a2e10951bfab1c2a86b4c60aff5bef785df` |
| 核心实现 | `aa9d70a` |
| 可复现环境 | `0d605c3` |
| 方法与协议文档 | `b8f2dc1` |
| smoke 兼容修复/测量提交 | `34a25a2` |
| post-run attestation 工具 | clean commit `ad36ee8e0e15a6d0502a35f9e794b056b9522a82`；tree SHA-256 `8ea95533232bf6b0d45b75513ec4c799f3ab42595fb66abd5e9893142fbfae7a` |
| Chat formal root | `/root/autodl-tmp/slotune-results/qwen25-3b-chat-formal-34a25a2` |
| RAG formal root | `/root/autodl-tmp/slotune-results/qwen25-3b-rag-formal-34a25a2` |
| official/SSE cross-check | `/root/autodl-tmp/slotune-results/cross-validation-34a25a2-20260816` |
| 当前格式 smoke | `/root/autodl-tmp/vllm-tuner-output/slotune-results/smoke-ad36ee8-20260816` |
| formal supervisor log | `/root/autodl-tmp/slotune-results/formal-suite-34a25a2.log` |

模型权重 SHA-256 为
`a70aaede3c09d599f1a632a254012408371a1cd36175062f4eed1343c7fae549`，tokenizer
SHA-256 为 `aa669c9a419bc0ed6a6d34ae06a57772a342e1e7496a63764c1171c9a577dd28`。
所有最终性能数字都只适用于这份本地 Qwen2.5-3B-Instruct、RTX 5090、已记录的软件版本、
SLO 和 seeded traces。

## 时间线

| 时间 | 实际事件 | 可核验证据 |
|---|---|---|
| 2026-08-15 | 冻结 upstream-era baseline；70 个单元测试通过 | [`BASELINE_20260815.md`](BASELINE_20260815.md)，upstream point `7948751` |
| 2026-08-15 | 在 `feat/slotune-core` 上完成 P0 benchmark、telemetry、lifecycle、optimizer、artifact、report 和 scheduler simulator 重构 | 之后汇总为 `aa9d70a` |
| 2026-08-15 15:47:04Z | 停止跟踪生成的 Python bytecode | `e47aa58`（commit 显示为 23:47:04 +08:00） |
| 2026-08-15 17:24:54Z | 核心流水线提交 | `aa9d70a`（01:24:54 +08:00） |
| 2026-08-15 17:25:11Z | 数据盘环境和精确依赖提交 | `0d605c3` |
| 2026-08-15 17:25:35Z | methodology/formal protocol 提交 | `b8f2dc1` |
| 2026-08-15 17:29:49Z | disabled-holdout smoke trace 修复；成为正式测量 commit | `34a25a2` |
| 2026-08-15 17:37:31Z | detached formal suite 启动，先 RAG 后 Chat | log 第一行 `FORMAL_SUITE_START` |
| 2026-08-15 22:30:15Z | RAG 96-trial formal run 完成 | RAG root、log 完成段；22:30:16Z 开始 Chat |
| 2026-08-16 02:07:25Z | Chat formal run 和整个 suite 完成 | `FORMAL_SUITE_COMPLETE` |
| 2026-08-16 | 对 Chat/RAG 做 trial、raw request、telemetry、cleanup、Parquet、report、hash 全量只读审计 | 两个 workload 均 96/96 trial integrity/semantic/cleanup 通过 |
| 2026-08-16 03:35:22Z | 提交 phase/source lineage、root seal、compact sidecar 和 negative-result rendering | clean tool commit `ad36ee8`；不改 sealed trial/raw evidence |
| 2026-08-16 03:37:26Z | fresh 0.6B smoke 自动完成并封存 | `smoke-ad36ee8-20260816`，两个 COMPLETE/selectable trial |
| 2026-08-16 03:39:22Z | Chat post-run attestation 完成 | 共 143 entries（含 96 anchors），seal SHA-256 `7d704bee…e191` |
| 2026-08-16 03:40:07Z | RAG post-run attestation 完成 | 共 143 entries（含 96 anchors），seal SHA-256 `7df0229c…98b7` |

提交时间表示功能被整理进 Git 的时间，并不冒充每个函数开始或完成编写的时间。

## 1. 冻结基线，而不是拿 smoke 当 benchmark

工作开始时，我先保留 upstream-era 运行链，而不是直接覆盖它：

- upstream point：`79487512f37e83cf1e9834783590443c2944a6a0`；
- baseline：70 tests passed；
- Qwen3-0.6B bring-up 只有两个请求、一个 fixed trial；
- 原 artifact 位于
  `/root/autodl-tmp/vllm-tuner-output/studies/reproduction_gpu_20260815_a`；
- 它只能证明 model load、serve、request、Optuna orchestration 和 report export 串起来了。

旧结果中的 TTFT、token throughput 和 peak VRAM 早于 correctness 修复，所以从一开始就标为
“smoke/preflight only”。后来新增的 Qwen3-0.6B current-format smoke 也保持相同边界，没有被
拿来填正式结果表。

常用的基线检查命令是：

```bash
cd /root/autodl-tmp/vllm-tuner
git rev-parse HEAD
git status --short --branch
.venv/bin/python --version
.venv/bin/python -m pip check
nvidia-smi
.venv/bin/pytest -q
```

## 2. 固定可复现环境

最终 GPU 环境是 Python 3.12.3、vLLM 0.16.0、PyTorch 2.9.1+cu130、CUDA runtime 13.0、
driver 595.71.05、Transformers 4.57.6、NumPy 2.2.6、FlashInfer Python 0.6.3，GPU 为
32,607 MiB 的 NVIDIA GeForce RTX 5090。

所有大文件和 cache 都留在 `/root/autodl-tmp`。环境脚本显式导出 `UV_CACHE_DIR`、
`TMPDIR`、`HF_HOME`、`TORCHINDUCTOR_CACHE_DIR`、`TRITON_CACHE_DIR`、`CUDA_CACHE_PATH`、
`VLLM_CACHE_ROOT` 和 `FLASHINFER_WORKSPACE_BASE`。安装采用 frozen/inexact 项目同步，再装
GPU overlay，避免 `uv run` 在每次正式命令前把 host-specific vLLM/PyTorch 组合重置掉。

```bash
cd /root/autodl-tmp/vllm-tuner
./scripts/setup_data_disk_reproduction.sh
./scripts/run_reproduction_command.sh --help
source scripts/data_disk_reproduction_env.sh
source .venv/bin/activate
python -m pip check
```

`0d605c3` 固定了 vLLM 0.16.0、Transformers 4.57.6、NumPy 2.2.6 和 idna 3.18；
`pyproject.toml` 同时约束 `transformers<5`、`numpy<2.3`，让普通开发依赖与 GPU overlay
保持兼容。

## 3. P0 benchmark correctness

### 3.1 SSE 是字节流协议，不是“一 chunk 一 token”

新的 `SSEDecoder` 增量处理 UTF-8、CRLF、跨 chunk 分片、一个 chunk 多个 event、多个
`data:` 行、注释、空行分隔、末尾没有空行以及 `[DONE]`。HTTP error、timeout、invalid
JSON、server error 和 stream 未正常 DONE 都保留为结构化 request failure。

真正关键的区别是：**SSE event timestamp 不等于 token timestamp**。一个 vLLM event 可能
携带多个 delta token IDs。正式 backend 会请求 pinned vLLM 的 `return_token_ids=true`，
然后为该 event 中的每个 token ID 记录同一个到达时刻；同时独立保存 event timestamps。
只有 `len(token_timestamps) == output_tokens` 且没有 text-without-token-IDs 时，ITL 才标为
valid。否则保留 inter-event latency，ITL 留空，绝不伪造“每 event 一个 token”。

对应回归测试覆盖：

```bash
.venv/bin/pytest -q \
  tests/unit/test_benchmark_sse_client.py \
  tests/unit/test_benchmark_metrics.py \
  tests/unit/test_benchmark_result_parser.py \
  tests/unit/test_benchmark_vllm_bench.py
```

### 3.2 指标只从真实边界计算

- TTFT：`sent_at → first non-empty streamed token event`；
- TPOT：`(finished_at - first_token_at) / (output_tokens - 1)`，单 token 为 0；
- ITL：相邻有效 token timestamps；
- E2E：`sent_at → completion boundary`，单独保存，不从其他指标倒推；
- offered、achieved throughput 和 SLO goodput 是三个不同字段；
- warmup 被标记并从 reducer 排除；
- percentile 使用线性插值；异常和 missing evidence 不会变成 0。

`[DONE]` 或 EOF completion timestamp 在退出 `httpx` response context 之前记录，因此
`response.aclose()`/连接池清理延迟不会污染 E2E 和 TPOT。

### 3.3 exact-length trace 和 official cross-check

构造固定长度 prompt 时，`decode(target_ids)` 再 `encode()` 不一定是 BPE fixed point。
生成器会复算长度、截断 overrun，并搜索 separator-prefixed padding 直到重新 encode 后恰好
等于目标；无法精确构造就失败，而不是在 manifest 里写一个虚假的 token count。

official adapter 保留 vLLM 原生 raw output 和原生 ITL。若官方 ITL 个数小于
`output_tokens - 1`，它会明确标为 unavailable/count mismatch，不用插值补齐。

live cross-check 的真实 artifact 使用 Qwen3-0.6B、两个 prompt、每个 8 output tokens：

- completed/failed 数一致；
- input/output token totals 一致；
- SSE token timestamp 完整，ITL count 为 14；
- SSE/official latency ratio：E2E 0.5968、TPOT 0.4455、TTFT 0.7531。

两个 backend 是顺序运行，而且 official arrival process 独立生成，所以这些 latency ratios
只是单位和数量级 sanity check，不是性能等价或 formal 3B 结论。其可复现测试入口是：

```bash
VLLM_TEST_BASE_URL=http://127.0.0.1:8000 \
VLLM_TEST_MODEL=/root/autodl-tmp/models/Qwen3-0.6B \
VLLM_TEST_ARTIFACT_DIR=/root/autodl-tmp/slotune-results/cross-validation-34a25a2-20260816 \
.venv/bin/pytest -q tests/integration/test_sse_client.py
```

## 4. Prometheus、NVML 与 measurement window

每个 trial 在同一个 measurement window 内以 200 ms 间隔连续采样：

- vLLM running/waiting requests、KV cache usage、preemption、prompt/generation token counters、
  prefix-cache 和可用 latency/queue metrics；
- NVML memory、GPU utilization、power、temperature 和 clocks；
- 客户端逐请求 send/first-token/token/finish timestamps。

Prometheus counter 使用 window delta；NVML power 用梯形积分得到 joules，再除以 output tokens。
collector 在取消路径也关闭，并在 server shutdown 前取最后一次 sample。采样失败写
`available=false` 和原因，不用 0 冒充空闲。

正式 artifact 的 JSONL 行数审计结果：

| Workload | Prometheus rows | NVML rows |
|---|---:|---:|
| Chat | 39,826 | 39,838 |
| RAG | 60,752 | 60,758 |

两边的 prompt/generation counter delta 与 request token totals 比例均为 1，GPU/power/energy
为正值。不同 workload 的行数不同，是单个 trial duration 不同，不是缺测。

## 5. Trial lifecycle、失败分类和 cleanup

状态机按 `CREATED → STARTING → READY → WARMING_UP → MEASURING → COLLECTING → STOPPING`
进入 `COMPLETE`、`INFEASIBLE`、`FAILED` 或 `PRUNED`。失败 artifact 保留 phase、exception、
last server state、日志和 cleanup；constraint-INFEASIBLE 与 request FAILED 不混写。

每个 vLLM 在新 session/process group 中启动。cleanup 不是只等 leader PID：

1. 记录启动前 GPU compute PID baseline；
2. 保存 PGID 和进程组成员；
3. 先向整个 process group 发 SIGTERM 并轮询；
4. 仅在超时后向残余组发 SIGKILL；
5. 再检查 PGID、tracked GPU PIDs 和端口；
6. 三者都干净才允许 `cleanup_status.clean=true`，trial 才可 selectable。

正式 192 个 trial 的 cleanup 全部通过，PID-after 为空、端口可重新 bind、没有使用 SIGKILL。
suite 结束后的现场只读 `nvidia-smi` 观察为 2 MiB/0%，且没有 vLLM compute process；这一
精确瞬时读数没有单独封存为 root artifact，不用它替代逐 trial cleanup JSON 证据。

## 6. Artifact、resume 与 optimizer 提交门禁

manifest 记录 config/trace/holdout/search-space SHA-256、environment、model metadata、每个
weight shard、tokenizer、source commit、dirty state 和 source-tree hash。source-tree hash 覆盖
tracked/untracked non-ignored 文件内容、执行位、symlink target 和 tracked deletion；仅比较
“同一个 commit + dirty=true”不足以判断两次 dirty resume 相同。

每个 trial 使用原子写入，并在成为 optimizer evidence 之前执行：

- required-file availability；
- request raw/summary、sample count、token totals 的 semantic replay；
- `artifact-integrity.json` 文件集、size 和 SHA-256 验证；
- terminal state、constraints 和 cleanup selectable gate。

默认拒绝覆盖已有 experiment root；`--resume` 只有在 manifest execution identity 完全一致时
开放。权重、tokenizer、trace、配置、环境关键项或 source tree 改变都会拒绝 silent resume。

## 7. Equal-budget constrained tuning 与 capacity

优化目标只有 SLO goodput。error rate、request/engine OOM、server exit、p99 TTFT/TPOT/E2E、
peak VRAM 和 peak memory utilization 是 hard constraints。失败或 infeasible trial 不会用
sentinel score 伪装成低分可选项。

default、seeded random、constrained TPE 各有 16 个 measured outcomes，并交错运行以减轻
系统漂移。每个 formal workload 的固定 phase 拆分是：

| Phase | 数量 | 含义 |
|---|---:|---|
| search | 48 | default/random/TPE 各 16 |
| repeat | 15 | 五个候选，各三次 exact-parameter repeat |
| holdout | 15 | 同五个候选，各三次独立 holdout trace |
| capacity | 18 | nominal 1/2/4/8/16/32 req/s，各三次，仅 vLLM default |
| total | 96 | 每 workload |

capacity trace 从同一 arrival-shape family 重新缩放。测量 artifact 保存 nominal
`request_rate` 和全部 scheduled offsets，所以 empirical rate 可复算。审计后 schema 显式拆出
target/empirical 列；对 legacy formal evidence 通过 additive audit/view 补充语义，不改旧
sealed rows。图的 x 轴用 target，表格单独显示 empirical、achieved 和 goodput。

## 8. 真实 GPU smoke 与长实验启动

正式 suite 前先运行 clean-tree smoke：

```bash
cd /root/autodl-tmp/vllm-tuner
./scripts/run_data_disk_reproduction.sh smoke-34a25a2-20260816
```

结果是 default + repeat 两个 trial 均 COMPLETE/selectable，trial integrity/cleanup 通过；现场
`nvidia-smi` 观察到 GPU 回到 2 MiB/0%。这个 smoke 只验证 wiring，瞬时 GPU 读数不作为性能
artifact。

### 为什么选 nohup + setsid，而不是依赖 VS Code 连接

长任务需要在关闭 VS Code、SSH 或代理抖动后继续。tmux 也可行，但这次实际选择了
`nohup + setsid --fork --wait + flock`：

- `nohup` 忽略 terminal hangup；
- `setsid` 把 supervisor 放到独立 session，实际观察到 PPID 1；
- `--wait` 保留整个 suite 的最终 exit status；
- `flock` 防止同一 GPU suite 被重复启动；
- stdout/stderr 写固定 log，不依赖 IDE terminal buffer；
- 全程不用 `sudo`，因此不会在无人值守时等待密码。

下面是与实际启动方式等价的安全复现形式；study name 必须换成新的，不能覆盖 reference
roots：

```bash
nohup setsid --fork --wait \
  flock -n /root/autodl-tmp/slotune-results/formal-suite-rerun-001.lock \
  bash -lc '
    set -euo pipefail
    cd /root/autodl-tmp/vllm-tuner
    ./scripts/run_reproduction_command.sh tune \
      --config config/formal_3b_rag.yaml \
      --study-name qwen25-3b-rag-rerun-001 \
      --results-root /root/autodl-tmp/slotune-results
    ./scripts/run_reproduction_command.sh tune \
      --config config/formal_3b_chat.yaml \
      --study-name qwen25-3b-chat-rerun-001 \
      --results-root /root/autodl-tmp/slotune-results
  ' \
  >/root/autodl-tmp/slotune-results/formal-suite-rerun-001.log 2>&1 \
  </dev/null &
```

断开客户端不会停止该 supervisor，但服务器关机/实例回收、宿主机故障或 GPU driver crash
仍然可能终止任务。监控不是任务存活的必要条件；它只缩短发现产品级失败的时间。

实际 log 记录：

```text
FORMAL_SUITE_START 2026-08-15T17:37:31Z commit=34a25a2e10951bfab1c2a86b4c60aff5bef785df
RAG complete       2026-08-15T22:30:15Z
Chat start         2026-08-15T22:30:16Z
FORMAL_SUITE_COMPLETE 2026-08-16T02:07:25Z
```

只读检查可以使用：

```bash
tail -n 80 /root/autodl-tmp/slotune-results/formal-suite-34a25a2.log
nvidia-smi
git status --short --branch
```

## 9. Formal 结果与审计数字

### 9.1 共同完整性

每个 workload 有 96 个 terminal trials：89 COMPLETE、7 INFEASIBLE、0 FAILED、0 PRUNED。
每个 trial 都有 500 个 measured requests，所以每 workload 是 48,000/48,000 请求成功；
每个成功请求固定 128 output tokens，并有 128 个有效 vLLM delta-token timestamps。

因此“7 INFEASIBLE”不是请求失败：它表示一次完整的 500-request 测量触发了 latency 或
memory hard constraint。两个 workload 的 96/96 integrity、required availability、semantic
replay 和 cleanup 均通过，Parquet、root summary、report/plots 与 raw trials 一致。

### 9.2 target rate 与 empirical rate

YAML `request_rate` 是 gamma-arrival generator 的目标均值；500 请求的有限随机 trace 会有
不同的 empirical scheduled rate：

| Workload | target | search empirical | holdout empirical |
|---|---:|---:|---:|
| Chat | 8.000 | 8.029529 | 8.456300 |
| RAG | 4.000 | 4.254534 | 4.331800 |

所以 Chat holdout goodput 8.361 或 RAG holdout 4.311 高于 YAML target，并不意味着完成量
超过真实到达量；它们仍低于对应 empirical rate。

### 9.3 Chat

严格 repeat/holdout 选择得到 `tpe-11`：

```text
gpu_memory_utilization = 0.7857916988402553
max_num_batched_tokens = 4096
max_num_seqs = 16
tensor_parallel_size = pipeline_parallel_size = 1
```

| Phase | Candidate | Goodput req/s | p99 TTFT ms | p99 TPOT ms | p99 E2E ms | Peak MiB | Mean GPU util |
|---|---|---:|---:|---:|---:|---:|---:|
| repeat | default-1 | 7.947565 | 63.636 | 6.882 | 919.104 | 30,565.3 | 99.780% |
| repeat | tpe-11 | 7.948109 | 61.366 | 6.943 | 924.719 | 26,425.3 | 99.684% |
| holdout | default-1 | 8.359356 | 62.023 | 6.878 | 922.764 | 30,565.3 | 98.896% |
| holdout | tpe-11 | 8.360697 | 62.348 | 6.966 | 919.825 | 26,427.3 | 98.711% |

TPE-11 对 default 的 repeat goodput 只高 0.0068%，repeat TTFT 低 3.57%；holdout goodput
高 0.0160%，但 TTFT 反而高 0.52%。这没有达到预设的 15% goodput 或 20% p99 TTFT 门槛。

Chat capacity 的所有 18 个 trial 都 feasible，因此只能给出容量下界，不能声称找到了 knee：

| nominal | empirical scheduled | achieved/goodput median | p99 TTFT | feasible |
|---:|---:|---:|---:|---:|
| 1 | 0.944593 | 0.945 / 0.945 | 52.866 | 3/3 |
| 2 | 1.889186 | 1.888 / 1.888 | 54.084 | 3/3 |
| 4 | 3.778373 | 3.763 / 3.763 | 58.561 | 3/3 |
| 8 | 7.556746 | 7.478 / 7.478 | 60.503 | 3/3 |
| 16 | 15.113492 | 14.770 / 14.770 | 60.391 | 3/3 |
| 32 | 30.226983 | 27.883 / 27.883 | 144.274 | 3/3 |

最高测试点只支持“default capacity ≥27.883 req/s under this trace/SLO”。

Chat 的七个 INFEASIBLE 分类：

- `random-0003/0004/0005`、`tpe-0003/0004`：p99 TTFT >800 ms；
- `random-0010`：peak VRAM 与 memory-utilization constraint；
- `random-0011`：memory-utilization constraint。

七者均为 500/500 request success。

### 9.4 RAG

RAG 最终保留 `default-0` 为 best：

| Phase | Goodput req/s | Range | p99 TTFT ms | p99 TPOT ms | p99 E2E ms | Peak MiB | Mean GPU util |
|---|---:|---:|---:|---:|---:|---:|---:|
| repeat | 4.227418 | 4.225570–4.227453 | 337.028 | 14.028 | 1,913.489 | 30,565.3 | 86.96% |
| holdout | 4.311275 | 4.310998–4.311678 | 431.624 | 14.924 | 2,129.955 | 30,565.3 | 91.85% |

holdout/repeat goodput ratio 为 1.019836。TPE-5 相对 default 的 repeat/holdout goodput
分别为 −0.0013%/+0.0141%，TTFT 分别低 7.54%/8.49%，同样没有达到成功门槛。

RAG capacity 在 nominal 16 req/s 左右出现 knee，之后 achieved plateau 约 11.8 req/s：

| nominal | empirical scheduled | achieved median | goodput median | p99 TTFT | feasible |
|---:|---:|---:|---:|---:|---:|
| 1 | 1.018862 | 1.019 | 1.019 | 221.6 | 3/3 |
| 2 | 2.037723 | 2.034 | 2.034 | 251.6 | 3/3 |
| 4 | 4.075447 | 4.051 | 4.051 | 340.0 | 3/3 |
| 8 | 8.150893 | 8.037 | 8.037 | 525.7 | 3/3 |
| 16 | 16.301787 | 11.423 | 11.412 | 1,179.6 | 3/3 |
| 32 | 32.603574 | 11.797 | 11.561 | 1,728.7 | 1/3 |

RAG 的七个 INFEASIBLE 分类：

- `random-0004/0010/0011`、`tpe-0004`：peak VRAM + memory-utilization；
- `tpe-0008`：p99 TTFT 1,501.335 ms，略高于 1,500 ms SLO；
- nominal-32 capacity repeat 1/2：p99 TTFT 1,883.79/1,728.71 ms。

七者也均为 500/500 request success，没有 OOM 或服务异常退出。

### 9.5 Scheduler simulator 是负结果

fixed 512/1024/2048/4096/8192 与 adaptive 的数据来自 deterministic CPU simulator。
两个 workload 在 calibration 和 held-out 上的 adaptive goodput gain 都是 0%。相对最佳
fixed budget，adaptive p99 TTFT 为：

| Workload | calibration | held-out |
|---|---:|---:|
| Chat | 25.36% worse | 15.58% worse |
| RAG | 167.22% worse | 332.79% worse |

这证明 negative/no-benefit reporting 能保留 downside，不证明 adaptive vLLM runtime 已接入，
也不证明 GPU 性能提升。

## 10. 遇到的问题与诊断

以下每项按“问题 → 证据 → 修复 → 回归”记录。

### 10.1 bwrap permission failure 只属于开发工具环境

- **问题**：默认受限 shell 的 bubblewrap namespace 建立失败，表现为命令还未进入项目就
  permission denied。
- **证据**：同一只读/测试命令在批准的 workspace shell 中正常运行；产品代码、vLLM server
  和 formal child processes 没有对应异常。
- **处理**：把它归类为 agent/tool sandbox 限制，使用有明确范围的 approved exec；没有为了
  绕过工具问题去修改 SLOTune 产品逻辑。
- **回归**：pytest、CLI、GPU smoke 和 formal suite 在实际运行环境通过。这个问题不属于
  SLOTune runtime bug。

### 10.2 Transformers 5 / NumPy 2.4 依赖漂移

- **问题**：宽松解析可能把环境升级到 Transformers 5 或 NumPy 2.4，与 vLLM 0.16 的兼容
  区间不一致，health disconnect 只是 worker 初始化失败的表象。
- **证据**：`pip check`/server log 和 vLLM 依赖约束指向版本不兼容，而不是客户端网络故障。
- **修复**：`transformers==4.57.6`、`numpy==2.2.6`、`vllm==0.16.0`、`idna==3.18`；
  `uv sync --frozen --inexact` 后安装 pinned GPU overlay，所有 cache/TMP 指向数据盘。
- **回归**：`pip check` clean，0.6B smoke、official/SSE live test 和两份 3B formal run 完成。

### 10.3 Tokenizer decode/encode round-trip 改变 token 数

- **问题**：把恰好 N 个 token IDs decode 成文本后，BPE 边界可能合并；重新 encode 不再是 N。
- **证据**：trace 的声明长度与实际 tokenizer count 可不相等，固定长度 workload 因而失真。
- **修复**：`_fit_exact_token_count` 复算、截断，并用 separator-prefixed candidate 逐步补齐；
  构造失败直接报错。
- **回归**：workload generator/trace tests 验证 exact count、seed 和 checksum；formal raw 的
  prompt token counter delta 与 request totals 比例为 1。

### 10.4 SSE event 不等于 token

- **问题**：以 event arrival 代替 token arrival 会在一个 event 携带多个 token IDs 时少算 ITL。
- **证据**：fixture 中 `"text":"ab","token_ids":[11,12]` 只有一个 event，却是两个 token。
- **修复**：强制 pinned vLLM 返回 delta token IDs，分别保存 token/event timestamps；count
  mismatch 时 ITL unavailable。
- **回归**：split/multiple-event、多 token chunk、missing IDs、empty text 和 official native
  ITL tests；live artifact 的 SSE ITL count 为 14。

### 10.5 Response close 时间污染 E2E

- **问题**：若在离开 response context 后取 `finished_at`，HTTP transport `aclose()` 会被算进
  模型完成延迟。
- **证据**：可控 clock/slow-close test 能把 finish 从 `[DONE]` 推迟到 close 完成。
- **修复**：在 `[DONE]` 或允许的 EOF boundary 立即保存 `completion_ns`，再退出 context。
- **回归**：SSE client 的 slow-close/owned-client cleanup tests 同时验证正确 finish 和资源关闭。

### 10.6 OOM gate 不能只看进程退出

- **问题**：allocator OOM 可能出现在 request error 或 engine counters，server 仍暂时存活；只看
  exit code 会把它错当 feasible。
- **证据**：错误字符串、engine `oom_count/oom_detected` 和 request failure 可独立出现。
- **修复**：识别 CUDA/HIP/allocator markers，合并 request/engine evidence，并把
  `require_no_oom` 作为 hard constraint；peak VRAM/memory-utilization 另行判断。
- **回归**：objective/failure taxonomy tests 覆盖 request-only、engine-only、重复证据和普通异常。

### 10.7 NVML `power_w` 命名不一致

- **问题**：采样、序列化和 reducer 曾可能在 `power_w`/`power_usage_w` 之间错位，导致已有
  power sample 却算不出 energy。
- **证据**：NVML raw row 有功率值，而旧聚合路径查找不同字段名时呈 unavailable。
- **修复**：measurement session 统一使用 `power_w`，mW/1000 转 W，梯形积分为 joules；旧
  GPU snapshot collector 的展示字段不混入 formal NVML schema。
- **回归**：NVML scaling、time-series integration、missing sample 和 energy/output-token tests；
  formal artifacts 记录正 power/energy。

### 10.8 Dirty resume 不能只比较 commit + boolean

- **问题**：两个内容不同的 dirty trees 都可能是同一 commit 且 `dirty_worktree=true`。
- **证据**：修改 uncommitted source 后，旧 manifest 身份字段不能区分内容。
- **修复**：增加 `source_tree_sha256`，覆盖 Git execution-relevant path set、bytes、mode、symlink
  和 deletion；resume 比较 model weights/tokenizer/trace/config/environment/source tree。
- **回归**：manifest tests 对 dirty content、tracked deletion、untracked file、weight shard 和
  tokenizer 变化均要求拒绝 resume。

### 10.9 Cleanup 只杀 leader 会留下 worker/GPU PID

- **问题**：vLLM leader 退出后 worker 可能继续占 GPU 或端口，下一 trial 会随机失败。
- **证据**：模拟 exited leader + live PGID，以及 PGID 外 tracked GPU PID，单看 leader 都会
  错报 clean。
- **修复**：new session/PGID，TERM→轮询→必要时 KILL，再检查 process group、baseline-delta
  GPU PIDs 和 port bind。
- **回归**：runtime tests 覆盖残余 group、外部 GPU child、延迟退出、cleanup idempotence；
  formal 192/192 clean 且无 SIGKILL。

### 10.10 Capacity 的 measured/offered 列语义混淆

- **问题**：旧表的 `offered_requests_per_sec` 容易同时被理解为 YAML target、empirical
  scheduled arrival 或 completed throughput。
- **证据**：RAG target 4.0 的有限 trace empirical 为 4.254534，holdout goodput 4.311；只看
  target 会误读为 goodput 超过到达量。
- **修复**：显式新增 `target_offered_requests_per_sec` 和
  `empirical_scheduled_requests_per_sec`，再与 achieved/goodput 分开报告。legacy
  `measured_offered_requests_per_sec` 实际是 target-rate alias，audit 明确标注它不是 empirical
  measurement；现有 raw scheduled offsets 不改。
- **回归**：runner/report/plot/artifact semantic tests 验证列存在、x 轴 target、empirical 复算
  和 legacy fallback；结果快照同时列出 target/empirical。

### 10.11 Repeat/holdout 的 method 与 phase provenance

- **问题**：formal sealed child summary 的 legacy `method` 写成 `repeat`/`holdout`，而 aggregate
  Parquet 使用 source method `default/random/tpe`。数值可追溯，但字段语义不一致。
- **证据**：trial ID 和 `repeat_of` 能还原来源，Parquet 也正确；因此这是 provenance schema
  问题，不是需要重跑 GPU 的测量错误。
- **修复**：新 `TrialResult` 分开 `method`、`phase`、`source_method`、`source_trial_id`；对已封存
  formal trials 只生成 additive `lineage.json`，不篡改 child summary 或 integrity anchor。
- **回归**：fresh lineage、legacy derivation、mismatch rejection、anchor byte-identity tests。

### 10.12 Root 没有 seal，且 summary 重复约 270 MB

- **问题**：每个 trial 已 seal，但 aggregate/report/environment/root 文件没有总封；同时 RAG
  raw scheduler JSON 为 254,310,186 bytes，原 root `summary.json` 为 270,776,544 bytes，raw
  内容约占 summary 的 93.92%。Chat 对应 163,326,753/173,915,114 bytes。
- **证据**：per-trial tamper 会被发现，root report/aggregate tamper 当时没有统一入口；文件
  size/hash 显示重复来自 scheduler raw rows。
- **修复**：新增 `experiment-integrity.json`，seal 所有 non-trial 文件和每个 child integrity
  anchor；新增 `summary.compact-v1.json` sidecar，保留原 summary 的其余字段，把内嵌
  scheduler raw rows 替换为 compact metrics 与 raw path/size/SHA 引用，并附加 attestation
  metadata。原 `summary.json` 和 `aggregate/scheduler-ablation.json` 保持 byte-identical。
- **回归**：root add/delete/modify、child anchor、preflight partial-write、idempotent validate 和
  corrupt-before-reseal tests。

### 10.13 Negative result 没有在报告中显式渲染

- **问题**：raw scheduler JSON 已含 `negative_gain_conditions`，旧 report 只列策略 rows，没有
  明说 0% gain、TTFT regression、default 仍 best；legacy phase/source lineage 也只能从 trial
  ID 和 `repeat_of` 另外还原。
- **证据**：Chat/RAG raw 中各六条 negative conditions；报告读者需要手工打开超大 JSON 才能
  看到 explanation。
- **修复**：fresh report 显式渲染 validated-best/default-remained-best；legacy roots 通过独立
  `lineage.json` 和 additive `aggregate/scheduler-negative-results.json`、
  `report/scheduler-negative-results.md` 提供 provenance 与负结果 view，不改旧 report。
- **回归**：formal raw audit 确认每个 workload 各六条 negative conditions；
  reporting/artifact tests 验证渲染合同、解释、preemption 字段和默认仍最佳的展示机制。

### 10.14 CI 的 Python 版本下限与项目元数据冲突

- **问题**：`.github/workflows/cli.yml` 和 `release.yml` 仍调度 Python 3.9，但
  `pyproject.toml` 已声明 `requires-python = ">=3.10"`。
- **后果**：PR 和 main release job 会在 `pip install -e .[dev]` 阶段被包元数据拒绝；即使
  本机 Python 3.12 的全部测试通过，GitHub Actions 仍会必然红灯。
- **修复**：两份 matrix 统一为 3.10/3.11/3.12，删除仅为 macOS 3.9 设置的 exclude；新增
  契约测试，断言两份 workflow 不含 3.9 且包含三个受支持版本。
- **回归**：重新解析两份 YAML，文档定向测试 11 passed；全量 suite 最终为 299 passed、
  1 skipped。该问题说明发布门禁还必须核对 package metadata 与远端 CI matrix 的交集。

### 10.15 GitHub Actions 帮助文本样式和换行破坏纯字符串断言

- **问题**：直接推送 `72e81c2` 后，GitHub CI run `31925921196` 在 Python 3.10 上得到
  298 passed、1 skipped，唯一失败是 `test_tune_help_exposes_manifest_validated_resume`。
  `tune --help` 的 exit code 为 0，但 Actions 环境中的 Rich/Typer 输出包含 ANSI 样式码并
  受终端宽度换行；测试用原始字符串查找 `--resume` 时失败，本机无颜色的 `CliRunner` 输出
  没有复现。第一次只剥离 ANSI 后，run `31926290503` 已通过前五个断言，但说明短语
  `require clean Git` 仍因跨行空白而失败，证明不能只处理颜色。
- **影响**：业务 CLI 选项实际存在且帮助命令成功，失败属于跨环境测试表示差异；Release run
  `31925921228` 也在测试步骤失败，因此 semantic-release 被正确跳过，没有产生额外 tag 或
  version commit。
- **修复**：测试先用 Click 的 `strip_ansi()` 去除样式，再用 `split()`/`join()` 折叠布局空白，
  最后验证 `--resume`、`--allow-dirty-source` 及其不可变 manifest/clean-source 说明。
  生产 CLI 不改。
- **回归**：定向测试同时验证 exit code 和去样式后的完整选项合同；随后重新运行全量测试和
  GitHub 3.10/3.11/3.12 matrix。该修复不会改变 `34a25a2` 正式测量或 `ad36ee8` attestation。

## 11. Post-run attestation

正式测量 commit 与 attestation tool commit 必须分开记录。attestation 可以新增审计 view 和
root seal，但不能把后来代码冒充成原测量源码，也不能改写 sealed trial/raw evidence。

公开入口：

```bash
./scripts/run_reproduction_command.sh attest \
  --study-name qwen25-3b-chat-formal-34a25a2 \
  --results-root /root/autodl-tmp/slotune-results

./scripts/run_reproduction_command.sh attest \
  --study-name qwen25-3b-rag-formal-34a25a2 \
  --results-root /root/autodl-tmp/slotune-results
```

它先做 root preflight 和全部 trial semantic/integrity validation，再生成：

- `lineage.json`；
- `experiment-audit.json`；
- `summary.compact-v1.json`；
- `aggregate/scheduler-negative-results.json`；
- `report/scheduler-negative-results.md`；
- 最后写 `experiment-integrity.json` 并立即重新验证。

已有有效 root seal 时，普通 `attest` 只验证且不改字节。只有明确 `--reseal` 才允许重建；
它先验证旧 seal，若 evidence 已腐败则拒绝。root seal 的 attestation record 同时保存：

- measurement source commit/tree/dirty state；
- attestation source commit/tree/dirty state；
- tool kind 和时间；
- sealed file set、size、SHA-256 与 trial anchors。

库级公开入口是
`ArtifactStore.attest_experiment_artifacts(attestation={...}, reseal=False)`；调用后或只需验签时
使用 `ArtifactStore.validate_experiment_integrity()`。CLI 和 Python API 走同一个 preflight、
view-generation、seal 和 validation 路径。

实际执行使用 clean tool commit `ad36ee8e0e15a6d0502a35f9e794b056b9522a82`，与 formal
measurement commit `34a25a2e10951bfab1c2a86b4c60aff5bef785df` 分开记录：

| Root | UTC | Total entries (including anchors) | Semantic/status/lineage/negative | Seal SHA-256 |
|---|---|---:|---|---|
| Chat | `2026-08-16T03:39:22.962525+00:00` | 143 total (96 anchors) | 96/96；89 COMPLETE + 7 INFEASIBLE；legacy 96、derived repeat/holdout 30、negative 6 | `7d704beea1890d14f7a411d677b867cdc8a06584a5040dbde2793f6723c8e191` |
| RAG | `2026-08-16T03:40:07.786811+00:00` | 143 total (96 anchors) | 96/96；89 COMPLETE + 7 INFEASIBLE；legacy 96、derived repeat/holdout 30、negative 6 | `7df0229c115ec0ce41cbc3c72624b13597b2a33d8f93a762242dbe723ca498b7` |

Chat 原 `summary.json`/`scheduler-ablation.json` SHA-256 分别保持
`ade1eaa13a4f78c49c498404c100f2e5458c6a194b1d378ccda283d415a04361` /
`c78bdb8d57c5deef51053f41d4e50d8d48f9fe0ee9b5d069220d3a562f138c8b`；RAG 对应
`b9da5621b4f075b387a1e2be93968294367249a205992b0c7cffe6acb5895e2f` /
`3d382db39c7279b27752137567cc7779c510fd6419f6f995aa29608285b5e1e3`。重复普通
`attest` 只验证且 seal 字节不变。封存权威状态以各 root 的 `experiment-integrity.json`
为准。

## 12. 测试和发布门禁

正式 GPU suite 启动前的 clean measurement revision 门禁记录为：

```text
283 passed, 1 skipped
Black: 99 files unchanged
Ruff: passed
mypy: 60 source files passed
uv lock --check: passed
pip check: passed
```

这里的一个 skip 是未提供 live GPU environment variables 时的显式 integration skip；真实
official/SSE cross-check 另行启动 server 后已经通过并保存 artifact。post-run attestation
新增代码完成后的全量门禁实际记录为：

```text
298 passed, 1 skipped, 44 warnings in 29.34s
artifact targeted suite: 31 passed
Black: 96 files clean
Ruff: passed
mypy: Success — 60 source files
git diff --check: passed
```

这些是提交前工作树的实际输出。早先的测试数量估计没有写成结果；测试增加后应记录最终实际
数字，而不是把 collect count 或计划值写成 passed。这组门禁完成后，代码经审阅提交为 clean
`ad36ee8`，再物化两份 formal attestation；它没有被冒充成 `34a25a2` formal measurement
commit 的一部分。

attestation 物化和文档收口后的最终发布门禁为：

```text
299 passed, 1 skipped, 44 warnings in 29.09s
documentation targeted suite: 11 passed
Black: 99 files clean
Ruff: passed
mypy: Success — 60 source files
uv lock --check / pip check / bash -n: passed
uv build: sdist and wheel built successfully
git diff --check: passed
```

提交后的 fresh smoke `smoke-ad36ee8-20260816` 有两个 COMPLETE/selectable trials、共 37 个
sealed entries（其中两个是 anchors），measurement/tool provenance 都是 clean `ad36ee8`；它保存新的
schema-5 recorded lineage，自动 seal 和重复验证均通过。其 `experiment-integrity.json` SHA-256 为
`4b2552d3e375681ab3e1067962c79d50c777635e23034e539294283af946040c`。结束后现场
`nvidia-smi` 为 2 MiB/0%，没有 compute PID；该瞬时观察不替代 cleanup artifact。

标准最终门禁命令：

```bash
cd /root/autodl-tmp/vllm-tuner
.venv/bin/pytest -q --disable-warnings
.venv/bin/black --check src tests
.venv/bin/ruff check .
.venv/bin/mypy src
uv lock --check
.venv/bin/python -m pip check
bash -n scripts/*.sh
.venv/bin/python -m build
git diff --check
git status --short --branch
```

## 13. 最终结论和未完成边界

- 真实 GPU pipeline、两个 3B workloads、三方法 equal budget、repeats、holdout、capacity、
  telemetry、cleanup 和 raw traceability 已完成。
- Chat 的最高点仍 feasible，所以只能报告 ≥27.883 req/s 的 tested lower bound；不能推断
  saturation capacity。
- RAG 的 knee 约在 nominal 16 req/s，32 点两次因 TTFT constraint INFEASIBLE。
- 两个 workload 的 tuning 都没有达到 15% goodput/20% p99 TTFT 预设成功条件；这是正式
  negative result，不做选择性隐藏。
- adaptive scheduler 数据是 CPU simulator，0% goodput gain 且 TTFT regression；runtime
  scheduler 尚未接入 vLLM。
- M6 prefix-caching/APC matrix 是明确 deferred P1，项目计划把它列为 optional，因此不阻塞
  core Definition of Done。
- 没有运行 7B/8B、multi-GPU、线上流量或长期稳定性实验；本日志不把 3B 单机结果外推到这些
  场景。

正式数字的短版见
[`results/qwen25-3b-34a25a2.md`](results/qwen25-3b-34a25a2.md)，复现和 attestation 命令见
[`../REPRODUCTION.md`](../REPRODUCTION.md)，逐项计划验收见 [`PLAN_AUDIT.md`](PLAN_AUDIT.md)。
