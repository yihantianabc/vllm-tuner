# SLOTune 项目实施规划

## 0. 文档信息

- 项目名称：SLOTune
- 完整定位：面向真实负载与 SLO 的 vLLM 调度与 KV Cache 优化实验平台
- 核心场景：单机单卡 LLM 在线推理性能分析、容量规划和离线自动调优
- 当前硬件：NVIDIA RTX 5090 32 GB
- 当前基线：vLLM 0.16.0、PyTorch 2.9.1+cu130、Qwen3-0.6B
- 数据盘根目录：/root/autodl-tmp
- 核心开发周期：10–14 天
- 完整增强周期：3–4 周
- 上游来源：基于 jranaraki/vllm-tuner 二次开发，README 必须明确区分上游能力与个人贡献

## 1. 项目目标

本项目不做聊天应用、Web 管理后台或 Kubernetes 平台。目标是构建一个可信、可解释、可复现的推理性能实验系统，并实现一个有系统深度的调度优化。

项目最终需要回答三个问题：

1. 在指定模型、GPU 和流量模型下，哪些 vLLM 参数组合可以最大化满足 SLO 的有效吞吐？
2. 当性能发生变化时，能否通过 queue、KV Cache、preemption 和 GPU 时间线解释原因？
3. 自适应 Chunked Prefill / Token Budget 策略能否比 FCFS 和固定 budget 获得更好的尾延迟与 goodput？

核心目标函数：

~~~text
SLO Goodput =
  测量窗口内满足 TTFT、TPOT、E2E SLO 的成功请求数
  ───────────────────────────────────────────────
                         测量时间
~~~

约束至少包含：

- 请求错误率不超过设定阈值
- OOM 次数为 0
- vLLM 服务不得异常退出
- p99 TTFT、p99 TPOT 或 p99 E2E 满足实验定义的 SLO
- Peak VRAM 不超过安全阈值
- 自定义调度策略不得造成请求无限饥饿

## 2. 项目价值主线

项目对 AI Infra 面试的价值集中在四点：

1. 测量正确性
   正确区分 TTFT、TPOT、ITL、E2E、吞吐和 goodput，理解 open-loop 与 closed-loop 流量差异。

2. 跨层可观测性
   将客户端延迟与 vLLM waiting queue、KV Cache、preemption、GPU utilization 和显存时间线对齐。

3. 系统机制优化
   实现自适应 Chunked Prefill / Token Budget 策略，解释 prefill、decode、KV Cache 和调度公平性的权衡。

4. 可复现实验
   固定 trace、seed、模型版本、环境指纹和实验协议，保留逐请求原始结果并进行重复实验与 holdout 验证。

## 3. 当前基线与已知问题

当前仓库已经完成：

- RTX 5090 单卡端到端 smoke
- 本地 JSON、JSONL、CSV 数据加载
- max_tokens 配置传递修复
- vLLM 启动阶段健康检查兼容修复
- 数据盘环境、缓存和输出隔离
- 70 项单元测试
- 一键安装与 smoke 脚本

当前结果只能证明链路跑通，不能作为正式性能数据。必须优先解决以下问题：

### P0：Benchmark correctness

- [ ] 请求真正启用 OpenAI SSE streaming
- [ ] 正确处理跨 HTTP chunk 的 SSE event 边界
- [ ] TTFT 从请求发出到首个非空 token 计算
- [ ] ITL 保存相邻 token 的到达间隔
- [ ] TPOT 使用首 token 之后的生成时间和 token 数计算
- [ ] E2E 单独保存，不再使用 TTFT + 平均 TPOT 近似
- [ ] output token 数由 tokenizer、usage 或官方 benchmark 结果获得
- [ ] total input/output tokens 不得为无意义的 0
- [ ] percentile 使用可靠实现，不使用简单下标近似
- [ ] warmup 数据不得进入正式统计窗口
- [ ] asyncio.gather 返回的异常必须被显式记录
- [ ] 所有指标保存逐请求原始值，而不只保存聚合值

### P0：Telemetry correctness

- [ ] 不再依赖易变化的 vLLM 日志正则作为主要指标源
- [ ] 接入 vLLM /metrics
- [ ] 采集 running、waiting、KV usage、preemption、token counters
- [ ] 测量窗口内每 100–200 ms 连续采集 NVML
- [ ] 正确计算 peak、mean、p95 VRAM 和 GPU utilization
- [ ] 可选计算 energy per output token
- [ ] 使用 client、engine、gpu 三个命名空间，禁止字段覆盖
- [ ] Prometheus counter 使用测量窗口前后 delta，不直接使用累计值

### P0：Optimizer correctness

- [ ] 删除无效的 batch_size 搜索参数
- [ ] 单卡固定 tensor_parallel_size=1、pipeline_parallel_size=1
- [ ] 检测 trial 参数与固定 vllm_args 的重复和覆盖
- [ ] 删除名不副实的 60/30/10 weighted objective
- [ ] 使用 SLO goodput 单目标约束优化作为 MVP
- [ ] 失败 trial 标记为 FAIL、PRUNED 或 INFEASIBLE
- [ ] 禁止多目标失败值统一返回负无穷
- [ ] 保存明确的 failure_reason 和最后一次服务状态
- [ ] default、random、TPE 使用相同 trial budget 对照

## 4. 目标架构

~~~text
ExperimentSpec
  model / version / hardware / workload / SLO / search space
                              │
                              ▼
                       Trial Controller
          START → READY → WARMUP → MEASURE → COLLECT → STOP
             │                   │                   │
             ▼                   ▼                   ▼
        vLLM Server       vLLM Bench Adapter    Telemetry Session
                          or SSE Client          /metrics + NVML
             └───────────────────┬───────────────────┘
                                 ▼
                          Result Validator
                request results / engine series / gpu series
                                 ▼
                           Metric Reducer
                   TTFT / TPOT / ITL / E2E / Goodput
                                 ▼
                    Constrained Search Controller
                         default / random / TPE
                                 ▼
                    Holdout Validation + Report
~~~

## 5. 建议代码结构

~~~text
src/vllm_tuner/
├── experiment/
│   ├── models.py              # ExperimentSpec、TrialResult、环境指纹
│   ├── manifest.py            # 保存版本、GPU、模型 hash、trace checksum
│   └── artifacts.py           # 统一管理输出目录和原始数据
├── workloads/
│   ├── trace.py               # 固定 workload trace
│   ├── generator.py           # token 长度分布和到达时间
│   └── profiles.py            # chat、RAG、long-prefill profile
├── benchmarks/
│   ├── vllm_bench.py          # 官方 vllm bench serve 适配器
│   ├── sse_client.py          # 可选的自有流式客户端
│   └── result_parser.py       # 官方 JSON 与逐请求结果解析
├── profiling/
│   ├── prometheus.py          # vLLM /metrics 采集
│   ├── nvml_session.py        # 测量窗口连续采样
│   └── timeseries.py          # 时间对齐与聚合
├── runtime/
│   ├── server.py              # vLLM 进程生命周期
│   ├── state_machine.py       # trial 状态机
│   └── failures.py            # OOM、端口、启动、请求错误分类
├── tuning/
│   ├── objective.py           # SLO、goodput 和约束
│   ├── search_space.py        # 有效参数与条件搜索空间
│   └── optimizer.py           # default、random、TPE
├── scheduling/
│   ├── token_budget.py        # 自适应 Token Budget 策略
│   ├── admission.py           # aging、max-wait、公平性
│   └── simulator.py           # 纯 Python 可确定性调度模拟
└── reporting/
    ├── plots.py               # capacity、Pareto、time series
    └── report.py              # 静态 HTML/Markdown 报告

tests/
├── unit/
├── integration/
└── fixtures/
    ├── sse/
    ├── prometheus/
    └── benchmark_results/
~~~

不要求一次性完成目录迁移。按里程碑逐步抽取，避免大爆炸式重构。

## 6. 里程碑 M0：冻结可复现基线

目标：在开始重构前保留当前已验证状态。

任务：

- [ ] 记录当前 commit、未提交修改和成功 study 路径
- [ ] 保存 Python、vLLM、PyTorch、CUDA、驱动和 FlashInfer 版本
- [ ] 保存 GPU 型号、显存、CPU 和内存信息
- [ ] 保留 reproduction_gpu_20260815_a 作为 bring-up artifact
- [ ] 创建独立开发分支
- [ ] 将已有改动拆成语义清晰的 commits
- [ ] 在 README 标记 0.6B 结果为 smoke，不是 benchmark

验收标准：

- [ ] 全新数据盘环境可通过安装脚本重建
- [ ] smoke 一条命令可运行
- [ ] 70 项测试通过
- [ ] 大文件和缓存仍只写入 /root/autodl-tmp
- [ ] 基线 artifact 可追溯到唯一代码版本

## 7. 里程碑 M1：可信 Benchmark 管线

目标：得到能经受面试追问的逐请求指标。

优先策略：

1. 使用官方 vllm bench serve 作为可信测量后端。
2. 保留自有 SSE client 作为学习、测试和交叉验证工具。
3. 项目贡献放在实验控制、结果验证、约束搜索和跨层遥测，不重复造完整 load generator。

任务：

- [ ] 新增 BenchmarkAdapter 接口
- [ ] 实现 VLLMBenchAdapter
- [ ] 支持 request_rate、burstiness、max_concurrency
- [ ] 支持 fixed input/output token length
- [ ] 支持 ignore_eos、seed、warmup 和结果落盘
- [ ] 解析 completed、failed、input/output tokens
- [ ] 解析 TTFT、TPOT、ITL、E2E 和 goodput
- [ ] 保存官方 benchmark 原始 JSON
- [ ] 自有 SSE client 增加 stream=true
- [ ] 正确解析 data 行、空行、DONE 和拆分 chunk
- [ ] 使用 time.perf_counter_ns 记录客户端时间
- [ ] 建立 RequestSpec 和 RequestResult 类型

RequestResult 至少包含：

~~~text
request_id
scheduled_at
sent_at
first_token_at
finished_at
input_tokens
output_tokens
token_timestamps
status
error_type
~~~

验收标准：

- [ ] fixture 覆盖一个 event 被拆成多个 HTTP chunk
- [ ] fixture 覆盖一个 chunk 包含多个 SSE event
- [ ] fixture 覆盖 DONE、空文本、HTTP 错误和 timeout
- [ ] TTFT、TPOT、ITL、E2E 的单元测试使用手算期望值
- [ ] 与官方 vllm bench serve 对同一 workload 交叉验证
- [ ] completed、token 数和主要延迟指标差异有解释
- [ ] 正式结果中 total tokens 不再为 0

## 8. 里程碑 M2：跨层 Telemetry

目标：不仅知道配置快不快，还能解释为什么。

Prometheus 指标优先采集：

- vllm:num_requests_running
- vllm:num_requests_waiting7
- vllm:kv_cache_usage_perc
- vllm:num_preemptions_total 或当前版本等价指标
- vllm:prompt_tokens_total
- vllm:generation_tokens_total
- vllm:prefix_cache_queries
- vllm:prefix_cache_hits
- vllm:time_to_first_token_seconds
- vllm:inter_token_latency_seconds
- vllm:e2e_request_latency_seconds
- vllm:request_queue_time_seconds

NVML 指标：

- memory used / total
- SM utilization
- power usage
- temperature
- SM clock
- memory clock

任务：

- [ ] 实现异步 TelemetrySession
- [ ] 与 benchmark 测量窗口同时启动和停止
- [ ] 采样间隔配置化，默认 200 ms
- [ ] 保存 Prometheus 原始快照或解析后的 Parquet/JSONL
- [ ] 保存 NVML 时间序列
- [ ] 使用 monotonic timestamp 对齐客户端与服务端数据
- [ ] 计算 peak、mean、p95 和窗口 delta
- [ ] telemetry task 在异常和取消时可靠退出
- [ ] vLLM 停止前完成最后一次采集

验收标准：

- [ ] 压测期间 GPU utilization 不再恒为 0
- [ ] peak_memory_mb 来自真实时间序列最大值
- [ ] token counters 与客户端结果数量级一致
- [ ] waiting queue 与高 TTFT 时间段可以对齐
- [ ] 采集任务不会阻止服务退出
- [ ] telemetry 缺失时 trial 明确降级，不伪造 0 值

## 9. 里程碑 M3：可靠 Trial 生命周期

目标：任何失败都能被正确分类、清理和恢复。

状态机：

~~~text
CREATED
  → STARTING
  → READY
  → WARMING_UP
  → MEASURING
  → COLLECTING
  → STOPPING
  → COMPLETE

任何阶段可进入：
FAILED / INFEASIBLE / PRUNED
~~~

任务：

- [ ] 自动选择或检查空闲端口
- [ ] 启动后同时检查进程状态和健康端点
- [ ] 保存服务启动命令、环境和完整日志
- [ ] 管理整个进程组，而不只管理父进程
- [ ] stop 超时后逐级终止并确认 GPU 进程清理
- [ ] 分类 OOM、端口冲突、参数错误、模型加载错误、请求错误
- [ ] 普通 RuntimeError 不得自动标成 OOM
- [ ] 每个 trial 都输出结构化 failure_reason
- [ ] 已有 study 默认不静默追加不兼容实验
- [ ] resume 时验证 manifest 与 search space 一致

验收标准：

- [ ] 人为端口冲突可得到 PORT_IN_USE
- [ ] 无效 vLLM 参数可得到 INVALID_ARGUMENT
- [ ] 模拟 OOM 可得到 OOM，而普通异常不会被误报
- [ ] 失败后 GPU 无残留进程
- [ ] 失败 trial 不进入 best/Pareto 结果
- [ ] 下一个 trial 可以继续运行

## 10. 里程碑 M4：SLO-aware Autotuner

MVP 搜索空间：

~~~yaml
gpu_memory_utilization:
  low: 0.60
  high: 0.95
max_num_seqs:
  values: [8, 16, 32, 64, 128]
max_num_batched_tokens:
  values: [1024, 2048, 4096, 8192]
tensor_parallel_size:
  fixed: 1
pipeline_parallel_size:
  fixed: 1
~~~

明确区分：

- Server parameters：由 tuner 搜索
- Workload parameters：request rate、burstiness、max concurrency
- Experiment constants：模型、版本、trace、seed、采样设置

任务：

- [ ] 新增 SLOConfig
- [ ] 支持 TTFT、TPOT、E2E 阈值
- [ ] 计算 per-request good/bad
- [ ] 计算 request goodput
- [ ] error、OOM 和服务退出作为硬约束
- [ ] 使用 constrained TPE 或显式 infeasible handling
- [ ] 实现 equal-budget random search baseline
- [ ] 保留 vLLM default baseline
- [ ] 对 search space 做参数关系校验
- [ ] 保存每个候选配置的完整环境和原始结果
- [ ] top candidate 在正式报告前自动重复运行

验收标准：

- [ ] 已知失败 trial 不会被选为 best
- [ ] random 与 TPE 使用相同有效 trial 数
- [ ] 同一 seed 可以复现参数建议顺序
- [ ] 报告明确区分 offered load、achieved throughput 和 goodput
- [ ] best config 必须在 holdout workload 上重新验证
- [ ] 权重字段不再误导用户

## 11. 里程碑 M5：核心深度模块——自适应 Chunked Prefill

这是项目区别于普通 benchmark 工具的核心系统优化。

### 11.1 研究问题

- 固定 token budget 在短交互请求和长 prefill 混合时是否会造成 head-of-line blocking？
- 较小 budget 是否改善 decode/ITL，却牺牲 prefill/TTFT 或吞吐？
- 能否根据 decode backlog、waiting time 和 KV pressure 动态调整 budget？
- 如何避免长请求被持续推迟？
- 策略收益是否能在不同 request rate 和 held-out trace 上保持？

### 11.2 Baseline 策略

- FCFS + vLLM 默认配置
- 固定 Token Budget：512、1024、2048、4096、8192
- 固定 max_num_seqs
- vLLM priority policy，如当前版本支持

### 11.3 自适应策略草案

输入信号：

- waiting decode requests
- waiting prefill requests
- oldest request age
- KV cache usage
- recent p99 TTFT / TPOT
- preemption count
- available token budget

策略示例：

~~~text
if decode_backlog is high or TPOT is near SLO:
    reduce prefill budget
elif oldest_prefill_age exceeds max_wait:
    reserve budget for the oldest prefill
elif KV usage is near pressure threshold:
    reduce admitted sequences
else:
    increase prefill budget toward throughput target
~~~

必须加入：

- aging
- max_wait
- 最小 prefill progress
- budget 上下界
- hysteresis，避免配置频繁抖动
- 策略决策日志

### 11.4 实现路径

低风险路径：

1. 先实现纯 Python deterministic simulator。
2. 使用合成请求验证公平性、等待时间和预算守恒。
3. 再实现入口 admission controller 或 nano-vLLM scheduler 修改。
4. 最后再考虑 vLLM scheduler plugin/内部接口。

不要一开始直接修改 vLLM 深层 scheduler；其内部接口变化快，必须 pin 精确 commit。

### 11.5 验收标准

- [ ] 任一调度 step 不超过总 token budget
- [ ] decode 与 prefill 请求均可取得进展
- [ ] max_wait 测试证明不存在无限饥饿
- [ ] 相同 trace 下策略决策可复现
- [ ] 与至少两个固定 budget baseline 对照
- [ ] 报告 p50/p99 queue time、TTFT、TPOT、goodput
- [ ] 同时报告 fairness、starvation 和 preemption
- [ ] 至少一个实验解释策略无收益或负收益的条件
- [ ] 收益必须在 held-out trace 上复验

## 12. 里程碑 M6：Prefix Caching 扩展

此阶段是 P1，不阻塞核心项目完成。

基础实验维度：

- APC on / off
- cold / warm cache
- shared prefix ratio：0%、25%、50%、75%、100%
- prefix length：512、2048、4096 或硬件允许范围
- arrival order：random、tile、interleave
- short output / long output
- 不同 KV cache pressure

重点指标：

- prefix cache query/hit 或 queried/cached token counters
- p99 TTFT
- output throughput
- SLO goodput
- KV cache usage
- eviction / preemption
- waiting queue

可选差异化：

- [ ] 根据 prefix fingerprint 计算 reuse score
- [ ] 使用 reuse score + age 做 bounded reordering
- [ ] 设置 max_wait 防止低复用请求饥饿
- [ ] 与 FCFS 比较 cache hit、goodput 和公平性

验收标准：

- [ ] 明确区分 cold 与 warm 结果
- [ ] 不声称 APC 会加速长答案 decode
- [ ] 测试共享率为 0 时的额外开销
- [ ] 测试高共享率时的收益
- [ ] 报告 reordering 带来的排队和公平性代价

## 13. 正式实验协议

### 13.1 模型

- Qwen3-0.6B：仅用于 CI、开发和 GPU smoke
- 正式模型：本地可获得的 3B–8B dense model
- 优先 7B/8B，以便在 RTX 5090 上体现 KV Cache 和调度压力
- 每份结果必须记录模型路径、配置 hash 和 tokenizer hash

### 13.2 Workload profiles

| Profile | Input tokens | Output tokens | 特征 | 目的 |
|---|---:|---:|---|---|
| Chat | 约 256 | 约 128 | 短输入、中等输出 | 观察 decode 与并发 |
| RAG | 约 2048 | 约 128 | 长输入、共享前缀 | 观察 prefill/APC |
| Mixed | 256–4096 | 64–256 | 长短混合 | 观察 HOL blocking |
| Codegen，可选 | 约 512 | 约 512 | 长 decode | 观察 TPOT |

所有 trial 必须使用同一份固定 trace，而不是每次重新随机采样。

### 13.3 Capacity sweep

建议 request rate：

~~~text
1 / 2 / 4 / 8 / 16 / 32 / inf
~~~

实际范围需根据正式模型的 baseline 调整。

每个点记录：

- offered request rate
- achieved request rate
- request throughput
- output token throughput
- SLO goodput
- p50/p95/p99 TTFT
- p50/p95/p99 TPOT
- p50/p95/p99 E2E
- errors/timeouts
- waiting queue
- peak KV usage
- preemptions
- peak VRAM
- mean GPU utilization

### 13.4 单个正式 trial

建议流程：

1. 清理或重置相关 cache。
2. 启动服务并等待 READY。
3. 执行 30 秒 warmup。
4. 执行 60–120 秒测量，或至少完成 500 个请求。
5. 停止负载并收集最后快照。
6. 优雅停止服务。
7. 校验进程、GPU 和端口已清理。
8. 校验 artifact 完整性。
9. 验证成功才提交给 optimizer。

### 13.5 重复与验证

- baseline 至少重复 3 次
- random best 至少重复 3 次
- TPE top 3 各重复 3 次
- 配置执行顺序随机化，降低温度和系统漂移影响
- 报告中位数、范围或 bootstrap confidence interval
- 使用未参与搜索的 trace 或 request rate 做 holdout
- 结果只对 model × hardware × workload × vLLM version 有效

### 13.6 预设成功条件

这些是项目验收目标，不是提前承诺的结果：

- 同一 SLO 下 goodput 相比 vLLM default 提升至少 15%；或
- 相同 goodput 下 p99 TTFT 降低至少 20%；或
- 在混合长短请求中显著降低短请求 p99，同时无 starvation
- 三次重复均无 OOM 和服务异常退出
- holdout workload 无明显退化
- 自有指标与官方 benchmark 的主要差异可以解释

即使未达到提升目标，也必须保留负结果并分析原因。

## 14. 测试规划

### Unit tests

- [ ] SSE event 分片与合并
- [ ] TTFT、TPOT、ITL、E2E 数学定义
- [ ] percentile 与 goodput
- [ ] Prometheus counter delta
- [ ] NVML 时间序列聚合
- [ ] trial 状态转换
- [ ] OOM 与普通异常分类
- [ ] 参数冲突和搜索空间校验
- [ ] direction-aware failure handling
- [ ] scheduler budget 守恒
- [ ] scheduler aging/max_wait
- [ ] artifact manifest 和 checksum

### Integration tests

- [ ] 本地假 HTTP/SSE server
- [ ] 假 Prometheus endpoint
- [ ] vLLM 启动、健康检查、请求、停止
- [ ] 失败 trial 后继续下一个 trial
- [ ] Qwen3-0.6B 单卡 GPU smoke
- [ ] official vllm bench adapter smoke
- [ ] telemetry 与 benchmark 同步启动/停止

### Manual performance validation

- [ ] 0.6B 快速验证
- [ ] 7B/8B 正式 baseline
- [ ] capacity sweep
- [ ] fixed token-budget ablation
- [ ] adaptive policy
- [ ] holdout
- [ ] prefix caching，可选

性能测试不应默认进入普通 CI；使用 pytest marker 和显式 GPU 命令。

## 15. Artifact 结构

~~~text
results/<experiment-id>/
├── manifest.json
├── experiment.yaml
├── trace.jsonl
├── trace.sha256
├── environment/
│   ├── python-packages.txt
│   ├── nvidia-smi.txt
│   ├── collect-env.txt
│   └── git-state.txt
├── trials/
│   └── <trial-id>/
│       ├── server-command.json
│       ├── params.json
│       ├── status.json
│       ├── request-results.jsonl
│       ├── benchmark-raw.json
│       ├── prometheus.jsonl
│       ├── nvml.jsonl
│       ├── server.log
│       └── summary.json
├── aggregate/
│   ├── trials.parquet
│   ├── repeated-results.parquet
│   └── holdout-results.parquet
└── report/
    ├── report.html
    ├── capacity-curve.png
    ├── pareto.png
    ├── telemetry-timeline.png
    └── comparison-table.md
~~~

所有大文件继续保存在 /root/autodl-tmp。

## 16. 10–14 天排期

| 时间 | 工作 | 交付物 |
|---|---|---|
| Day 1 | 冻结基线、定义 RequestSpec/RequestResult/TrialResult | 数据模型、设计说明 |
| Day 2 | 官方 vllm bench adapter | 原始 JSON 可解析 |
| Day 3 | SSE correctness 与 metric reducer | fixture tests、正确指标 |
| Day 4 | Prometheus + NVML session | 两类时间序列 |
| Day 5 | Trial 状态机与失败分类 | 可恢复的 trial runner |
| Day 6 | SLO goodput 与 constrained TPE | default/random/TPE |
| Day 7 | GPU integration smoke 与官方交叉验证 | integration artifact |
| Day 8 | 7B/8B baseline capacity sweep | capacity curve |
| Day 9 | 固定 token-budget ablation | TTFT–TPOT–goodput 图 |
| Day 10 | 自适应策略 simulator | 公平性和 budget tests |
| Day 11 | 策略接入运行时 | adaptive trials |
| Day 12 | top candidates 重复实验 | 重复结果 |
| Day 13 | holdout 与负结果分析 | holdout 表格 |
| Day 14 | README、报告、Demo、简历 | 面试交付包 |

若只有 10 天：

- 必须完成 M0–M4
- M5 至少完成 simulator + 固定 budget ablation
- Prefix Caching 和实际 scheduler 集成顺延

## 17. README 包装计划

README 首页顺序：

1. 一句话问题定义
2. 一张真实结果表
3. 一张 capacity/Pareto 图
4. 系统架构
5. Benchmark methodology
6. 三个关键设计决策
7. default vs random vs TPE
8. 自适应 Chunked Prefill 机制
9. Failure cases
10. 一键复现
11. Limitations
12. Upstream vs My Contributions

必须包含：

~~~text
Forked from jranaraki/vllm-tuner.
My work focuses on benchmark correctness, SLO-aware optimization,
cross-layer observability, reproducibility, and scheduling experiments.
~~~

My Contributions 表格应逐项链接到 commit、测试和实验 artifact。

禁止：

- 把上游的 Optuna、HTML 报告说成从零实现
- 用 2 请求 smoke 宣称性能提升
- 未实测却声称多 GPU 性能
- 只给百分比，不给模型、硬件、负载和版本
- 隐藏无收益或负收益的实验

## 18. Demo 计划

三到五分钟 Demo：

1. 展示一条命令执行 0.6B smoke。
2. 打开预先生成的正式报告，不现场等待 7B/8B 冷启动。
3. 展示 default capacity curve。
4. 修改一个 TTFT/TPOT SLO，说明 goodput 目标变化。
5. 展示失败 trial 被标为 INFEASIBLE，而不是成为 best。
6. 对比固定 budget 与自适应 budget。
7. 用 telemetry timeline 解释 queue、KV 和 GPU 的变化。
8. 最后展示 holdout 结果和限制。

准备一个真实 debugging 故事：

- 表面现象：健康检查断连
- 根因定位：vLLM worker 初始化失败
- 深层原因：FlashInfer Python 与 cubin 版本不一致
- 工程修复：版本锁定、数据盘脚本、结构化日志和依赖检查
- 得到的经验：服务端断连只是症状，Infra 调试必须沿进程树和日志定位

## 19. 简历交付模板

完成前可诚实描述：

- 基于开源 vLLM-Tuner 完成 RTX 5090 单卡端到端复现，打通数据加载、服务生命周期、GPU 请求、Optuna 搜索和报告导出，并修复本地数据、生成长度和健康检查等兼容问题。
- 建立数据盘隔离和精确版本锁定的可复现实验环境，为数据加载、配置传递和服务启动补充自动化测试。

完成正式项目后使用占位符替换真实数据：

- 构建 workload-aware vLLM SLO optimizer，实现官方 benchmark 编排、Prometheus/NVML 跨层遥测和 constrained TPE search；在 RTX 5090 + <model> + <workload> 下，将满足 <SLO> 的 goodput 从 <A> 提升至 <B>。
- 实现自适应 Chunked Prefill / Token Budget 策略，根据 decode backlog、KV pressure 和 request age 动态分配预算；相比 FCFS/固定 budget 将 p99 TTFT 降低 <X>% 或 goodput 提升 <Y>%，并通过三次重复和 holdout trace 验证。
- 定位并修复 TTFT、token throughput、peak VRAM、无效搜索参数和失败 Pareto handling 等 benchmark correctness 问题，建立带环境指纹、trace checksum 和逐请求原始数据的可复现流水线。

没有真实数据前不得填写提升百分比。

## 20. 风险与降级方案

| 风险 | 影响 | 降级方案 |
|---|---|---|
| 7B/8B 模型暂时不可获取 | 无法跑正式压力实验 | 先用 0.6B 验证正确性，正式结果延后 |
| vLLM 内部 scheduler API 变化 | 集成成本过高 | 固定 commit，先做 simulator/入口 admission |
| 单次 trial 冷启动过慢 | 搜索成本过高 | 缩小到 12–16 trials、缓存编译产物 |
| 官方与自有指标不一致 | 结果不可信 | 以官方 bench 为 reference，逐项解释差异 |
| Prometheus 指标版本变化 | 解析失败 | 建立 metric alias 和缺失指标显式降级 |
| 自适应策略无收益 | 项目结论不理想 | 保留负结果，分析 workload 条件和 overhead |
| 调度策略造成饥饿 | 结果不可接受 | aging、max_wait、minimum progress |
| 结果过拟合单一 trace | 无法推广 | holdout workload 和多 request rate 验证 |
| GPU 温度/频率漂移 | 重复结果波动 | 随机化顺序、记录 clock/temperature、重复运行 |

## 21. 明确不做的范围

核心版本不做：

- Web 管理后台
- Kubernetes 部署
- 在线无重启调参
- 多机、多 GPU 性能结论
- Tensor Parallel / Pipeline Parallel 优化
- 自研完整 PagedAttention Kernel
- FP8 KV Cache 作为主线
- Speculative Decoding 作为主线
- 模型训练或量化训练
- 生产级多租户鉴权、计费和网关

可以写入 Future Work，但不得包装成已完成能力：

- FP8 KV Cache ablation
- N-gram speculative decoding
- Prefix-aware admission
- vLLM scheduler plugin
- 多 GPU TP/DP/P-D 架构
- 自动扩缩容与容量模型

## 22. Definition of Done

核心项目只有同时满足以下条件才算完成：

- [ ] 正式指标来自可靠 streaming 或官方 benchmark
- [ ] TTFT、TPOT、ITL、E2E、tokens 和 goodput 有单元测试
- [ ] /metrics 和 NVML 在测量窗口连续采样
- [ ] 失败 trial 不会污染最优结果
- [ ] default、random、TPE 使用相同预算
- [ ] 至少一个 3B–8B 正式模型实验
- [ ] 至少两类 workload
- [ ] baseline 和 top candidates 至少重复三次
- [ ] 至少一份 holdout 结果
- [ ] 固定 token-budget ablation 完成
- [ ] 自适应策略至少完成 simulator 和公平性测试
- [ ] 环境、trace、参数、原始请求和日志可追溯
- [ ] README 明确上游来源与个人贡献
- [ ] Demo 可在五分钟内完成
- [ ] 不使用 smoke 数据冒充性能结论
- [ ] 所有大文件继续位于数据盘

## 23. 首批实现任务

建议按以下 PR/commit 顺序推进：

1. benchmark: add typed request results and correct metric reducer
2. benchmark: integrate official vllm bench serve adapter
3. profiling: add Prometheus and continuous NVML telemetry
4. runtime: introduce trial state machine and failure taxonomy
5. tuning: replace weighted objectives with constrained goodput
6. tuning: add equal-budget random baseline and holdout runner
7. scheduling: add deterministic token-budget simulator
8. scheduling: implement adaptive token-budget policy
9. experiments: run capacity and fixed-budget ablations
10. docs: publish results, methodology, limitations and demo

第一步不要直接写调度器。先完成 M1 的指标正确性，否则后续任何优化数字都不可信。

## 24. 官方参考资料

- vLLM Benchmark CLI
  https://docs.vllm.ai/en/latest/benchmarking/cli/
- vLLM Parameter Sweeps
  https://docs.vllm.ai/en/stable/benchmarking/sweeps/
- vLLM Metrics Design
  https://docs.vllm.ai/en/latest/design/metrics/
- vLLM Online Serving /metrics
  https://docs.vllm.ai/en/latest/serving/online_serving/
- vLLM Optimization and Tuning
  https://docs.vllm.ai/en/latest/configuration/optimization/
- vLLM Automatic Prefix Caching
  https://docs.vllm.ai/en/stable/features/automatic_prefix_caching/
- vLLM Prefix Caching Design
  https://docs.vllm.ai/en/v0.21.0/design/prefix_caching/
- vLLM Speculative Decoding，Future Work
  https://docs.vllm.ai/en/latest/features/speculative_decoding/
