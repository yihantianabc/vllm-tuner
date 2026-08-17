# vLLM V1 自适应 Chunked Prefill 调度优化项目计划（中间版）

> 状态：Planning Draft v3。本文只确定后续项目方向；尚未修改 vLLM 源码、学习文档和 README，也没有重新运行正式实验。文中的收益数字均为验收目标，不是已经取得的结果。

## 0. 最终项目定位

项目不再把 SLOTune 自动调参平台作为主线，也不采用上一版同时修改 Scheduler、KVCacheManager 和 BlockPool 的高风险方案。

新的项目题目是：

> **基于真实 vLLM V1 的 Workload-aware Adaptive Chunked Prefill Scheduling**

中文简历可写为：

> **基于 vLLM V1 的自适应 Chunked Prefill 调度优化与 KV Cache 性能分析**

项目只保留一个需要修改 vLLM 源码的核心贡献：

> 在 vLLM V1 Scheduler 中实现三状态自适应 Prefill 控制器，根据 Decode backlog、最老 Prefill 等待时间和 KV Cache 使用率，在运行时动态调整 Prefill 推进强度，以适应非平稳混合流量。

再用两组 vLLM 原生能力实验补足知识面：

1. Automatic Prefix Caching（APC）的冷/热缓存和不同前缀复用率实验；
2. FP8 KV Cache 的显存容量、并发能力、延迟与质量实验；若当前模型、硬件或 backend 不兼容，则降级为 CUDA Graph eager/non-eager 实验。

APC 和 FP8 KV Cache 是实验研究项，不包装成自主实现；简历中的源码贡献只写 Adaptive Scheduler。

### 0.1 一句话面试主线

> 固定 Chunked Prefill 参数只能适合某一类流量。我在真实 vLLM V1 Scheduler 中加入低开销三状态控制器，让系统在 Decode 密集、长 Prefill 突发和混合阶段使用不同 Prefill Budget，并通过固定基线、Oracle 上界和 held-out trace 验证它对尾延迟与 SLO goodput 的影响。

### 0.2 项目边界

本版明确不做：

- 不修改 BlockPool、block hash、ref count 或 KV eviction；
- 不实现 Prefix-aware 请求重排；
- 不重写 PagedAttention、FlashAttention 或 FP8 Decode Kernel；
- 不做 Triton 算子融合；
- 不继续扩展 Optuna、Dashboard、Web 平台或通用调参框架；
- 不把 CPU simulator 结果作为正式结论；
- 不承诺向 vLLM upstream 提交，也不以“可开源贡献”作为验收条件；
- 不在实验完成前预写提升百分比。

这些能力可以在面试中解释，但不作为本项目必须完成的工程范围。

---

## 1. 为什么选择中间版

### 1.1 旧 SLOTune 的问题

| 旧成果 | 真实情况 | 本版处理 |
|---|---|---|
| Qwen2.5-3B Chat/RAG formal | RTX 5090 上的真实 GPU 实验，但 TPE 未显著超过 default | 保留为 Legacy 和负结果分析 |
| Adaptive Token Budget | CPU simulator，未形成正向收益 | 不进入新项目主结果 |
| Optuna/TPE、Trial、Artifact | 工程完整，但离 vLLM 核心机制较远 | 停止扩展 |
| SSE、指标、Prometheus、NVML | 有复用价值 | 收敛成 benchmark/eval 工具 |

旧项目的问题不是“实验全部是假的”，而是主要结果没有证明有价值的 vLLM 优化，平台工作量又掩盖了推理引擎主线。

### 1.2 上一版 SAGE 的问题

上一版同时要求 Adaptive Chunked Prefill、Prefix-aware Admission、Reuse-aware KV Eviction 和可选 Triton Kernel。

其中 Admission/BlockPool 会触及等待队列公平性、block hash、引用计数、淘汰队列和 Hybrid KV Cache 等复杂不变量。即使代码能够运行，也需要大量正确性和压力测试才能证明没有隐藏泄漏或饥饿。

这对当前目标有三个明显风险：

- 学习跨度过大，面试中容易只能背设计而不能解释实现；
- 成熟 vLLM 的默认实现很强，三项修改未必都能得到正收益；
- 项目接近研究或 upstream 贡献的工作量，偏离“可完成、可复现、可答辩”的求职目标。

### 1.3 中间版的平衡

| 维度 | 旧 SLOTune | 中间版 | 上一版 SAGE |
|---|---|---|---|
| 核心贡献 | 外围参数搜索 | 一个真实 Scheduler 改动 | Scheduler + Admission + BlockPool |
| vLLM 源码深度 | 低 | 中等且聚焦 | 高 |
| 实验价值 | 当前结果偏弱 | 非平稳负载下验证动态策略 | 多机制协同研究 |
| 正确性风险 | 低 | 中低 | 高 |
| 面试可解释性 | 一般 | 高 | 当前阶段偏低 |
| 预计时间 | 已有基础 | 约 2～3 周 | 约 3～5 周以上 |

中间版的目标不是证明“比成熟 vLLM 全面更快”，而是证明：

> 在明确的非平稳负载下，单一固定 Prefill Budget 存在阶段性取舍，一个有安全边界的轻量控制器可以提高整体稳定性或 SLO goodput。

---

## 2. 核心研究问题与假设

### 2.1 核心问题

vLLM 的默认 Scheduler 使用固定上限控制单步可调度 Token 数。固定配置通常已经较强，但不同阶段的最优取值可能不同：

- Decode-heavy：需要保护正在生成请求的 TPOT/ITL；
- Long-prefill burst：需要推进长 Prompt，避免等待时间持续累积；
- Mixed：需要在 Decode 延迟和 Prefill 吞吐之间折中；
- KV pressure 较高：继续激进推进 Prefill 可能增加抢占或容量压力。

项目回答：

> 在不改变请求顺序、不修改 KV Cache 正确性、不修改 GPU Kernel 的情况下，能否根据少量在线信号动态调整 Prefill Budget，使同一个配置在变化流量下比默认配置或任一单一固定 Budget 更稳定？

### 2.2 核心假设 H1

> 三状态 Adaptive Prefill Controller 在非平稳混合负载上，可以降低尾部 TTFT/TPOT 或提高 SLO goodput，同时保持吞吐量和长请求公平性。

这也是唯一需要作为“自主优化贡献”证明的假设。

### 2.3 支撑问题，不作为新算法

#### S1：Automatic Prefix Caching

APC 在热缓存和高前缀复用率下应主要减少重复 Prefill 与 TTFT；它不会直接减少每个 Decode Token 的计算量。

#### S2：FP8 KV Cache

FP8 KV Cache 应主要降低 KV 显存占用、提高可容纳 Token 数或并发容量；容量收益不等于吞吐量必然同比增长，且需要检查 scale、backend 与质量影响。

---

## 3. 固定版本和源码范围

### 3.1 初始环境

- GPU：NVIDIA GeForce RTX 5090 32 GB；
- vLLM：固定本机已安装的 0.16.0 对应 commit；
- Python：3.12；
- Smoke 模型：Qwen3-0.6B 或当前可用小模型；
- Pilot 模型：Qwen2.5-3B-Instruct；
- Formal 模型：优先选择兼容的 7B/8B dense instruct 模型；若受显存、下载或 backend 限制，保留 3B 结果并明确降级。

正式结果必须记录：vLLM commit、模型 revision、CUDA、PyTorch、GPU、启动参数和 trace checksum。

### 3.2 允许修改的 vLLM 范围

首选通过固定版本的自定义 `scheduler_cls` 实现；如果接口导致大段复制，再改为最小 fork patch。

允许触及：

| 文件/接口 | 用途 |
|---|---|
| `vllm/config/scheduler.py` | 暴露并加载自定义 Scheduler 配置 |
| `vllm/v1/core/sched/interface.py` | 核对接口契约，原则上不修改 |
| `vllm/v1/core/sched/scheduler.py` | 计算动态 Prefill cap/budget，保留默认调度主语义 |
| `vllm/v1/request.py` | 读取请求进度、到达时间等已有状态，原则上不改结构 |
| Engine metrics/tracing | 输出每步控制器状态和调度统计 |

禁止触及：

- `kv_cache_manager.py` 的分配和命中语义；
- `block_pool.py` 与空闲 block 队列；
- PagedAttention、FlashAttention、ModelRunner Kernel；
- 采样和模型输出语义。

### 3.3 保存方式

不能只修改 `.venv/site-packages` 后运行实验。正式实现必须保存为以下一种形式：

1. 固定 v0.16.0 的小型 vLLM fork；或
2. 当前仓库中的可审查 patch series，加上固定 upstream commit 和应用脚本。

当前 `vllm-tuner` 仓库负责 workload、benchmark、结果聚合和复现，不再发展成平台。

---

## 4. 自适应 Prefill 控制器设计

### 4.1 只使用三个核心信号

每个 Scheduler step 读取：

1. `decode_backlog`：当前需要继续 Decode 的请求数量；
2. `oldest_prefill_wait_ms`：尚未完成 Prefill 的请求中最长等待时间；
3. `kv_cache_usage`：当前 KV Cache 使用率。

preemption、step time、running/waiting 数量可以记录为诊断指标，但第一版不进入控制公式，防止控制器逐渐变成难以解释的启发式集合。

### 4.2 三种状态

#### `PROTECT_DECODE`

触发条件：Decode backlog 较高，或者 KV Cache 使用率接近保护阈值。

行为：

- Prefill cap 使用较小档位；
- Decode Token 不受额外压缩；
- 限制新增长 Prompt 在单步内占用过多 Budget；
- 目标是保护 TPOT/ITL，并减少高压力下的进一步容量冲击。

#### `BALANCED`

触发条件：系统没有明显 Decode 拥塞，Prefill 也未达到最长等待阈值。

行为：

- 使用中等 Prefill cap；
- 维持默认 running-first 和 waiting queue 顺序；
- 作为大部分正常负载的稳态。

#### `DRAIN_PREFILL`

触发条件：最老 Prefill 等待超过阈值，且 Decode backlog 没有处于高位。

行为：

- 使用较大 Prefill cap；
- 保证长 Prompt 得到推进；
- 到达 `max_wait` 的请求获得最低进展保证。

### 4.3 优先级和安全规则

规则优先于状态：

1. 总调度 Token 不得超过 vLLM 原始 `max_num_batched_tokens`；
2. 不改变 waiting queue 的 FCFS/priority 顺序；
3. 不改变 KV block 的分配、释放、命中或淘汰；
4. Decode 每一步保持进展；
5. Prefill 达到 `max_wait` 后必须至少获得 `min_prefill_progress`；
6. 状态切换加入 hysteresis 和最小驻留 step，避免阈值附近来回抖动；
7. 控制器不得触发 GPU 同步；
8. feature disabled 时应退化为默认 Scheduler；
9. 请求输出、错误率和完成数必须与 baseline 对齐。

如果第 4、5 条在极端容量压力下无法同时满足，优先遵守 vLLM 原始容量和正确性约束，并记录该 step 未推进的原因，不能通过越界调度制造结果。

### 4.4 配置草案

~~~yaml
adaptive_prefill:
  enabled: true
  low_prefill_cap: 1024
  balanced_prefill_cap: 4096
  high_prefill_cap: 8192
  decode_backlog_high: TBD_BY_PILOT
  oldest_prefill_wait_ms: TBD_BY_PILOT
  kv_usage_high: TBD_BY_PILOT
  min_prefill_progress: 256
  max_wait_ms: TBD_BY_SLO
  hysteresis_steps: 3
  min_state_residency_steps: 3
~~~

1024/4096/8192 是初始候选档位，不是预设最优值。Pilot 后可以根据模型长度、合法配置和容量拐点调整，但必须在 Formal 前冻结。

### 4.5 决策输出

每个 step 记录：

~~~text
timestamp
controller_state
decode_backlog
oldest_prefill_wait_ms
kv_cache_usage
prefill_cap
scheduled_decode_tokens
scheduled_prefill_tokens
running_requests
waiting_requests
preemption_delta
scheduler_cpu_time_us
reason_code
~~~

这些日志用于回答“控制器为什么切换”和“收益是否真的来自调度变化”，不进入在线复杂优化。

### 4.6 实现规模目标

- Controller：约 150～250 行；
- Scheduler 接入与 metrics：约 150～300 行改动；
- 单元测试：约 300～500 行；
- workload/analysis：复用现有代码并做减法。

代码行数不是验收项；这里的限制是为了避免再次扩展成平台。

---

## 5. 核心实验：非平稳混合负载

### 5.1 为什么必须使用非平稳流量

如果整个实验只有一种固定流量，人工调好的单一 Budget 很可能比自适应控制器更简单、更稳定。Adaptive 的合理使用场景是请求组成随时间变化。

因此正式 trace 至少包含三个连续阶段：

| 阶段 | 请求特征 | 主要压力 |
|---|---|---|
| A：Decode-heavy | 中短 Prompt、较长输出、较高并发 | TPOT/ITL |
| B：Long-prefill burst | 突发 4K～8K Prompt、短输出 | TTFT、等待队列 |
| C：Mixed | 长短 Prompt 和不同输出长度并存 | 延迟与吞吐折中 |

阶段顺序在 held-out trace 中改变，避免控制器只记住固定时间表。控制器不能读取阶段标签或未来请求。

### 5.2 Baseline

- B0：原始 vLLM 默认 Scheduler；
- B1：固定小 Prefill Budget/Cap；
- B2：固定中 Prefill Budget/Cap；
- B3：固定大 Prefill Budget/Cap；
- B4：Adaptive 三状态控制器；
- B5：Offline per-phase Oracle，只作为分析上界。

Oracle 的含义是：实验结束后，为每个阶段选出表现最好的固定档位。它知道阶段边界，不能部署，因此不能作为真实在线方案宣称。

### 5.3 负载点

先对 B0 做 capacity sweep，再冻结三个负载点：

- 约 70% baseline capacity：观察控制器空载开销；
- 约 90% baseline capacity：正式主结果；
- 约 105% baseline capacity：观察拥塞与失效边界，不作为唯一 headline。

如果不同策略的 capacity 不同，正式比较仍使用相同 offered load，不能分别挑各自最有利的请求速率。

### 5.4 主指标

- p50/p95/p99 TTFT；
- p50/p95/p99 TPOT 或 ITL；
- request throughput 和 output tokens/s；
- SLO goodput；
- waiting time；
- preemption/recompute；
- KV Cache usage；
- Scheduler CPU time p50/p99；
- 请求成功率；
- 长请求和短请求分组尾延迟。

SLO 必须在看正式结果前冻结。若没有业务 SLO，可根据 baseline 分布定义一组宽松/中等/严格阈值并全部报告，不能只选择最有利的一条。

### 5.5 实验协议

每个正式配置：

1. 使用完全相同的请求内容、到达时间和随机种子；
2. 固定模型、tokenizer、vLLM commit 和 server flags；
3. 进行固定时长或固定请求数 warmup；
4. 运行 500～1000 个 measured requests，Pilot 可缩小；
5. 至少重复三次；
6. 报告中位数、各次结果和误差范围，不只保留最好一次；
7. 保存逐请求指标、控制器决策和 server log；
8. 使用未参与阈值选择的 seed、阶段顺序或请求比例进行 held-out 复验。

### 5.6 分级验收目标

这些是目标，不是保证。

合格结果：

- Adaptive 在非平稳 held-out trace 上优于原始默认配置；
- 相比最佳单一固定档位，SLO goodput 提高至少 5%，或者关键尾延迟降低至少 10%；
- 总吞吐退化不超过 5%；
- 长请求 p99 等待时间有界；
- Scheduler CPU overhead 小于 3%；
- 三次重复的改善方向一致。

强结果：

- SLO goodput 提高 10%～15%；或
- p99 TTFT/TPOT 降低 15%～20%，且吞吐和公平性不明显退化；
- held-out 的阶段顺序、请求比例变化后仍有收益。

如果 Adaptive 只接近最佳固定档位，或者只在特定阶段有效，仍保留真实数据并解释原因，不把 Pilot 中最好的一次改写为正式结论。此时项目可降级为“调度取舍分析 + 负结果”，但简历 headline 必须相应收缩。

---

## 6. 支撑实验一：Automatic Prefix Caching

本实验只使用 vLLM 原生 APC，不修改缓存算法。

### 6.1 实验矩阵

- APC：off / on；
- Cache 状态：cold / warm；
- Prefix reuse：0% / 50% / 100%；
- Shared prefix length：1K / 2K / 4K tokens；
- Prefix pools：少量热点 / 多前缀对照；
- Offered load：低负载和 capacity knee 附近各一个点。

### 6.2 观察指标

- cached/query tokens 与 hit ratio；
- TTFT、TPOT；
- request throughput；
- KV Cache usage 与 preemption；
- cold 到 warm 的收敛过程。

可以写：

> 系统评估 vLLM APC 在不同前缀长度、复用率和缓存冷热状态下对 TTFT 与容量的影响，并解释 Prefix 命中条件和适用边界。

不能写“实现或优化了 vLLM Prefix Cache”。除非后续确实新增源码改动，否则 APC 只能作为性能分析能力。

---

## 7. 支撑实验二：FP8 KV Cache

本实验只使用 vLLM 已有 FP8 KV Cache 能力，不自行重写 FP8 PagedAttention。

### 7.1 前置兼容性 Gate

先确认：

- RTX 5090、当前 CUDA/PyTorch/vLLM backend 支持目标 KV dtype；
- 模型与 Attention backend 能够稳定运行；
- KV scale 的来源与配置明确；
- 同一模型、请求和 seed 下输出质量可比较；
- 没有 silent fallback 到非 FP8 路径。

Gate 不通过时，记录原因并切换为 CUDA Graph eager/non-eager 支撑实验，不为了凑关键词修改 Kernel。

### 7.2 实验矩阵和指标

- KV Cache dtype：默认 dtype / FP8；
- Context length：短 / 中 / 长；
- 并发：低负载 / capacity knee / 压力点；
- 观察 KV 可用容量、可容纳 Token 数、peak VRAM、最大稳定并发、TTFT、TPOT、吞吐、preemption/OOM 和质量 sanity check。

FP8 KV 的合理预期是容量改善；不能因为单 Token 存储字节减少，就预先宣称端到端吞吐翻倍。实际速度还取决于 Attention backend、scale、内存带宽、调度和其他计算开销。

---

## 8. 正确性与测试

### 8.1 Controller 单元测试

- 三个状态的进入和退出；
- hysteresis 和最小驻留时间；
- KV pressure guard；
- `max_wait` 与 `min_prefill_progress`；
- Budget 上下界；
- 相同输入产生确定决策；
- feature disabled 返回默认 Budget。

### 8.2 Scheduler 回归测试

- 总 Token Budget conservation；
- Decode 保持进展；
- Long Prefill 不发生无界饥饿；
- 不改变 waiting queue 的稳定顺序；
- abort、preemption、cleanup 正常；
- prefix caching on/off 均能运行；
- eager/non-eager smoke；
- 原始 vLLM 相关 Scheduler tests 通过；
- offline greedy 的输出 Token 与默认 Scheduler 对齐。

### 8.3 GPU Smoke 与失败检查

- 先用小模型运行短 trace；
- 检查请求完成数、HTTP/SSE 错误和超时；
- 检查进程和显存清理；
- 检查 decision log 与真实 scheduled tokens 一致；
- 再进入 3B Pilot 和 7B/8B Formal。

性能提升不能来自漏请求、缩短输出、改变请求顺序语义或让长请求超时。

---

## 9. 仓库收敛方向

~~~text
vllm-tuner/
├── patches/                     # 固定 vLLM commit 的 Scheduler patch
├── src/
│   ├── scheduler/               # controller/config/decision schema
│   ├── workloads/               # 非平稳 trace 与 APC trace
│   ├── benchmark/               # vLLM 启动、请求和指标采集
│   └── analysis/                # 聚合、统计和绘图
├── tests/
│   ├── unit/
│   └── integration/
├── experiments/
│   ├── adaptive_prefill/
│   ├── prefix_cache/
│   └── fp8_kv_cache/
├── results/                     # 正式结果和 metadata
├── docs/
└── legacy/                      # 旧 SLOTune/TPE/simulator 说明
~~~

不急于物理删除旧代码。先在 README 和目录入口中降级为 Legacy，等新主线完整后再决定是否移动，避免在转型过程中丢失可复用 benchmark 能力。

最终 README 只需要清楚展示：问题与固定 Budget 的取舍、Scheduler 插入点、三状态控制器、一张非平稳 trace 时间线、一张主结果图、APC/FP8 支撑图、复现命令和限制。

---

## 10. 实施里程碑与时间

### M0：版本冻结与默认基线，主动操作约 1～2 小时

- 固定 vLLM 0.16.0 commit；
- 跑通小模型和 3B baseline；
- 确认自定义 Scheduler 加载路径；
- 验证已有 benchmark 指标是否可信。

验收：默认 vLLM 可重复运行，同一 trace 三次结果波动可解释。

### M1：Scheduler instrumentation，主动操作约 1～3 小时

- 识别 Decode/Prefill Token；
- 记录三个控制信号；
- 记录 scheduled tokens 和 Scheduler CPU time；
- 生成默认 Scheduler 的 step 时间线。

验收：关闭控制器时不改变默认行为。

### M2：三状态控制器，主动开发与调试约 3～8 小时

- 实现纯函数 Controller；
- 接入 Prefill cap/budget；
- 加入 max-wait、min-progress、hysteresis；
- 完成单元测试和小模型 GPU smoke。

验收：无错误、无 starvation，decision log 能解释状态切换。

### M3：核心 workload 与固定基线，主动操作约 2～4 小时，Pilot GPU 约 2～6 小时

- 生成三阶段非平稳 trace；
- 运行 default 和三档固定 Budget；
- 做 capacity sweep；
- 冻结阈值、SLO 和 Formal 配置。

验收：不同阶段确实存在可测的 Budget 取舍；如果不存在，先修正研究假设或 workload，不直接调到“必胜”。

### M4：Formal 与 held-out，后台 GPU 约 5～12 小时，结果审计约 1～3 小时

- default/fixed/adaptive/oracle；
- 三个负载点；
- 三次重复；
- held-out 阶段顺序和请求比例；
- 长短请求分组分析。

验收：形成真实的正结果或明确的适用边界。

### M5：APC 与 FP8 KV 支撑实验，主动操作约 1～3 小时，后台 GPU 约 6～15 小时

- APC cold/warm 与复用率矩阵；
- FP8 compatibility gate；
- FP8 容量、延迟和质量实验；
- 不兼容则执行 CUDA Graph fallback。

验收：每组结论都有机制解释，不把开关实验写成源码贡献。

### M6：仓库减重与面试材料，主动操作约 1～3 小时

- README 主线切换；
- 旧 SLOTune 标记 Legacy；
- 保存 patch、配置、结果和复现命令；
- 整理面试问题与失败边界；
- 学习文档是否补充由后续单独决定。

正常情况下总计约 10～20 小时主动开发/分析，加上约 12～30 小时可脱离 VSCode 的后台 GPU 实验，日历时间约 3～5 天。若遇到 vLLM 构建、FP8 backend 兼容或核心结果不明显，需要增加针对性调试，风险区间约 5～10 天。

### 10.1 长时间后台实验与关闭 VSCode 规则

只有同时满足以下条件的任务，才算“可以关闭 VSCode、不需要中途照看”的后台实验：

1. 预计连续运行时间 **超过 1 小时**；不足 1 小时的实验在当前工作阶段直接完成，不单独安排后台长跑；
2. 同一模型、代码路径、启动方式和 artifact 流程已经通过前台 Smoke/Pilot；
3. 代码、阈值、trace、SLO、实验矩阵和随机种子已经冻结，运行中不需要人工选择下一组参数；
4. 每个 Run 都有独立 ID、超时、状态文件、原始日志、逐请求结果和失败原因；
5. Suite 支持断点续跑，只跳过完整且校验通过的 Run，不因 VSCode 断开而丢失整个矩阵；
6. vLLM 子进程、benchmark、telemetry 和 cleanup 都由统一 runner 管理；
7. 单个 Run 失败会被记录并进入下一个独立 Run；连续出现相同致命错误、GPU 丢失或磁盘空间不足时，watchdog 自动停止整个 Suite；
8. 后台命令脱离 VSCode/SSH 会话运行，使用 `nohup + setsid`，或在环境可用时使用 `tmux`，并保存 PID、启动时间和总日志。

以下任务即使技术上可以放到后台，也**不算**可无人值守长实验：

- 第一次编译或加载修改后的 vLLM；
- 第一次 Scheduler GPU Smoke；
- FP8 KV compatibility gate；
- 仍在选择阈值、调整 workload 或定位错误的 Pilot；
- 预期不足 1 小时的测试；
- 需要看完当前结果才能决定下一步参数的串行调试；
- 没有 timeout、artifact、cleanup 或 resume 保护的临时命令。

### 10.2 实际启动和通知流程

达到后台长跑阶段时，执行顺序固定为：

1. 前台完成等价路径的 Smoke/Pilot，并检查请求成功率、指标和清理；
2. 生成冻结的 Suite manifest 和待运行矩阵；
3. 以脱离终端的方式启动后台 Suite；
4. 检查 PID/进程组存在、日志持续更新、manifest 与输出目录正确、GPU 已开始工作；
5. 确认任务不依赖当前 VSCode 会话后，再通知用户可以断开。

届时必须向用户明确发送类似下面的信息，而不是只说“已经启动”：

> 后台正式实验已经启动并完成脱离会话检查，现在可以关闭 VSCode。请保持 AutoDL 实例开机，不要关机或释放。预计结束时间、PID、日志路径、结果目录和恢复/状态检查命令如下：……

通知中必须包含：

- 正在运行的 Suite 名称和实验范围；
- 预计总时长和大致结束时间；
- PID/进程组；
- 主日志路径；
- artifact/result 根目录；
- 查看状态的只读命令；
- 中断后的 resume 命令；
- “关闭 VSCode 可以，关闭/释放 AutoDL 实例不可以”的提醒。

关闭 VSCode 后，后台 runner 只负责按照冻结矩阵执行、记录失败、清理进程和保存结果，不会自行修改源码、改变阈值或选择性重跑以追求更好数字。用户重新连接后，再对全部结果做审计并决定下一阶段。

### 10.3 里程碑顺序与完成通知

正常执行顺序固定为：

~~~text
M0 版本冻结与默认基线
  ↓
M1 Scheduler instrumentation
  ↓
M2 三状态控制器
  ↓
M3 核心 workload、fixed baselines 与参数冻结
  ↓
M4 Formal、重复实验与 held-out
  ↓
M5 APC、FP8 KV 支撑实验
  ↓
M6 仓库减重、README 与面试材料
~~~

除非出现明确的技术依赖变化，否则不交换 M0～M6 的主顺序。可以提前准备不占 GPU 的脚本或配置，但不能在 M3 尚未冻结阈值、SLO 和 Formal 矩阵时提前宣称 M4 已开始，也不能在 M4 结果尚未审计时完成最终 README。

每完成一个里程碑，都必须单独向用户发送一次完成通知，不将多个阶段静默合并。通知至少包含：

1. 当前完成的里程碑和结论；
2. 实际修改的主要文件或 patch；
3. 执行过的测试/实验及其通过、失败和跳过数量；
4. 是否达到该里程碑的验收条件；
5. 与原计划相比的偏差、未解决风险和真实负结果；
6. 结果或日志的本地路径；
7. 下一里程碑的内容、主动操作时间和预计后台时间；
8. 当前是否已经达到“可以关闭 VSCode”的条件。

建议通知格式：

> **M2 已完成**：三状态控制器和回归测试已通过；修改文件为……；测试结果为……；当前风险为……；下一步进入 M3，预计主动操作……、后台 Pilot……。当前仍需调试/当前可以关闭 VSCode。

只有满足本阶段全部验收条件时才使用“已完成”。如果只完成一部分，必须写“进行中”并列出剩余项；如果发生阻塞、需要修改研究问题或改变 M0～M6 顺序，必须先说明原因和影响，不能自行扩大范围。

对于 M4/M5 这类可能包含长时间后台 Suite 的阶段，需要区分两次通知：

1. **后台启动通知**：Suite 通过脱离会话检查后，告知可以关闭 VSCode，并提供 PID、日志、结果目录和预计结束时间；
2. **里程碑完成通知**：后台运行结束且结果完成完整性审计后，报告该 M 是否真正通过验收。

后台任务完成时如果用户仍处于断开状态，结果先原样保存在 artifact 目录；下一次恢复协作后立即进行审计并发送里程碑完成通知，不在无人交互状态下擅自修改实验方案或进入需要新判断的下一里程碑。

---

## 11. 风险控制

| 风险 | 概率 | 控制方式 |
|---|---:|---|
| vLLM 默认策略已经很好，Adaptive 无明显收益 | 中 | 使用真实非平稳 trace、固定基线和 capacity sweep；不保证必胜 |
| workload 过度人工设计 | 中 | held-out seed、阶段顺序、请求比例；报告低负载和失败场景 |
| 阈值过拟合 | 中 | Pilot 冻结参数，Formal 不再调；与三档固定值和 Oracle 对比 |
| Scheduler 接口不稳定 | 低～中 | 固定 v0.16.0 commit，保存 patch 和回归测试 |
| 控制器增加 CPU 开销 | 低 | 三信号、三状态、O(1) 决策，测 p99 Scheduler time |
| 动态 Batch Shape 影响 CUDA Graph/吞吐 | 中 | 保持有限档位并测 eager/non-eager、GPU gap 和吞吐 |
| 长 Prefill 饥饿 | 低～中 | max-wait、min-progress、分组 p99 wait |
| FP8 KV 不兼容或 silent fallback | 中 | compatibility gate、日志和显存证据；失败则降级 |
| FP8 容量改善但延迟无收益 | 中～高 | 将容量作为主结论，速度结果如实报告 |
| 7B/8B 无法完成 | 中 | 3B 完成功能和实验，正式结论标明模型范围 |
| 为得到好数字反复改 workload | 高风险行为 | 预注册 Formal 配置、保留全部正式结果、只用 Pilot 调参 |

“需要好的结果”不能转化为选择性隐藏负结果。能提高成功概率的正确做法是先构造机制明确、又有 held-out 的非平稳场景，而不是修改统计口径。

---

## 12. 学习文档覆盖与补充范围

现有《vLLM面试教学版-从零重构》对本项目的理论和面试知识覆盖约 85%～90%，对具体实现准备约 65%～75%。

| 项目内容 | 主要学习模块 | 覆盖情况 |
|---|---|---|
| V1 请求生命周期 | M01 | 充分 |
| Scheduler、Continuous Batching | M02 | 充分 |
| Chunked Prefill 与 Token Budget | M02 | 充分 |
| Paged KV Cache、APC | M02 | 充分 |
| FP8、scale、硬件兼容 | M03 | 原理充分，FP8 KV 具体路径需补 |
| CUDA Graph fallback | M04/M08 | 原理覆盖 |
| TTFT、TPOT、Goodput、Profiler | M08 | 充分 |
| 三状态控制器与 hysteresis | 暂无专项 | 需补 |
| v0.16.0 Scheduler 接入和测试 | 暂无专项 | 需结合源码补 |
| 非平稳 trace、Oracle、held-out | 暂无专项 | 需补 |

项目实施前后只需新增约 8～12 小时专项学习：

1. v0.16.0 `Scheduler.schedule()` 的真实执行顺序；
2. Controller 状态机、阈值、hysteresis 和 starvation bound；
3. 动态 Budget 与 CUDA Graph/batch shape 的关系；
4. FP8 KV Cache 的实际 dtype、scale 与 backend 路径；
5. 非平稳负载、Oracle 上界和 held-out 实验设计。

本阶段不修改学习文档；等代码和实验稳定后，再把真实实现补进教学材料，避免先写一套最后没有使用的方案。

---


---

## 14. Definition of Done

### 必须完成

- [ ] 固定 vLLM 0.16.0 commit 和完整实验环境；
- [ ] 自定义 Scheduler 或最小 fork patch 真实运行在 vLLM V1；
- [ ] 实现 `PROTECT_DECODE`、`BALANCED`、`DRAIN_PREFILL` 三状态；
- [ ] 只使用三个核心控制信号；
- [ ] 实现 hysteresis、max-wait 和 min-progress；
- [ ] feature disabled 时退化为默认 Scheduler；
- [ ] 单元测试、Scheduler 回归和 GPU smoke 通过；
- [ ] 完成 default + 三档 fixed + adaptive + offline Oracle；
- [ ] 完成 70%/90%/105% 三个负载点或记录合理降级；
- [ ] 每个 Formal 配置至少重复三次；
- [ ] 完成 held-out trace；
- [ ] 分组报告长短请求的尾延迟和等待；
- [ ] 保存 patch、trace、配置、日志和逐请求结果；
- [ ] APC/FP8 明确标记为原生能力实验；
- [ ] README 不再以 SLOTune 平台为主线；
- [ ] 不使用 CPU simulator 作为 headline result。

### 正结果验收

- [ ] Adaptive 在 held-out 非平稳 trace 上稳定优于 default；
- [ ] 相比最佳单一 fixed 至少满足一项：goodput +5%，或关键 p99 -10%；
- [ ] throughput 退化不超过 5%；
- [ ] Scheduler CPU overhead 小于 3%；
- [ ] 没有隐藏失败、漏请求和无界 starvation。

### 可选加分，不阻塞完成

- [ ] 第二个模型复验；
- [ ] 第二组请求比例 held-out；
- [ ] FP8 KV 在 capacity knee 附近证明容量收益；
- [ ] APC 实验形成清晰的冷/热缓存边界图；
- [ ] patch 代码风格接近 upstream，但不承诺提交 upstream。

---

## 15. 实施前结论

后续执行顺序固定为：

1. 先冻结版本并验证 default；
2. 再做 instrumentation；
3. 只实现 Adaptive Scheduler；
4. 先用 fixed baselines 证明问题存在；
5. 冻结参数后再跑 Formal 和 held-out；
6. 最后补 APC、FP8 KV 支撑实验；
7. 真实结果稳定后再改 README 和学习文档。

除非后续实验明确表明单一 Scheduler 改动完全无法形成项目，本版不自动升级到 Prefix-aware Admission、BlockPool eviction 或 Triton Kernel。
