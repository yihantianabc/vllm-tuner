# vLLM 单卡长上下文推理优化项目计划（面试工程版）

> 状态：Execution v5。M1 v2、M3 formal 和 M4 formal 已完成；M2 已确认当前栈 FP8 KV 不兼容。M4 在原零容忍 Goodput 规则下保留 production default，但两个原生 threshold Profile 均显著降低 Decode 干扰尾延迟。M5 改为独立预注册的 Decode-tail 非劣部署验证，不回写或重解释 sealed M4。

## 0. 项目定位

项目题目：

> **vLLM 单卡长上下文推理优化：KV Cache 容量建模、FP8 KV Cache 与 Prefix 复用**

目标：

1. 实现并验证 `KV Capacity Planner`；
2. 在 RTX 5090 上建立 7B/8B 长上下文容量基线；
3. 测量 FP8 KV 的容量、延迟和质量边界；
4. 测量 APC 的冷/热缓存与不同复用率收益；
5. 调整原生 Chunked Prefill，降低 Long Prefill 对 Decode 的干扰；
6. 形成一个可复现的最终部署 Profile，并与 production default 比较。

项目不是平台或研究算法。Planner 是自主代码贡献；APC、FP8 和 Chunked Prefill 必须写成“使用、分析和组合优化”，不能冒充自主实现。

### 0.1 为什么不是纯消融

纯开关对比含金量有限。新版必须交付：

> 根据模型结构、KV dtype、Block Size、显存预算和上下文分布，预测 vLLM KV blocks、最大缓存 Token 数和安全并发的 Capacity Planner。

实验用于验证 Planner、解释误差并选择部署 Profile，而不是堆配置数量。

### 0.2 不做

- 不修改 BlockPool、KV eviction、PagedAttention 或 CUDA/Triton Kernel；
- 不做 Dashboard、Optuna/TPE、Web 平台或 CPU simulator；
- 不隐藏不兼容、OOM、质量退化和负结果；
- 主结果完成前不启动高风险 Scheduler 研究；
- 不自动改成 Prefix-aware、BlockPool 或 Kernel 项目。

---

## 1. 核心贡献：KV Capacity Planner

基础公式：

~~~text
KV bytes/token =
    num_layers × 2(K+V) × num_kv_heads × head_dim × dtype_bytes
~~~

Planner 额外考虑 block 向上取整、尾块浪费、FP8 scale/metadata、权重/激活/CUDA Graph/workspace 占用和安全余量。

| 输入 | 输出 |
|---|---|
| 模型层数、KV Heads、Head Dim、GQA | KV bytes/token |
| KV dtype、Block Size | 预测 KV blocks/cached tokens |
| GPU 显存、`gpu_memory_utilization` | 非 KV 保留与可用 KV 显存 |
| `max_model_len`、`max_num_seqs` | 安全上下文和并发 Profile |
| 上下文长度分布 | 碎片率与安全余量 |

验收：

- 单元测试覆盖 GQA、dtype、block rounding 和非法配置；
- 与 vLLM 初始化后的 KV block/token 数交叉验证；
- 至少三个上下文/并发点；
- 主要容量预测误差目标 ≤10%；
- 误差必须能分解解释，不能只拟合一个实验点。

---

## 2. 环境与证据边界

- GPU：NVIDIA RTX 5090 32 GB；
- vLLM：固定 0.16.0 upstream commit；
- Smoke 可使用 Qwen3-0.6B；
- M0 固定一个 7B/8B dense instruct 主模型，M1～M5 不换模型；
- 若 7B/8B 不兼容，M0 明确降级到 3B，后续全部重新按该模型执行；
- 固定模型/tokenizer revision、CUDA、PyTorch、driver 和 backend；
- 新性能结论只使用 clean commit 和新 artifact root；
- production default 与手工 Profile 分开命名；
- 旧 SLOTune/TPE、旧三状态 Scheduler 和旧 M4 仅标记 Legacy。

---

## 3. 核心实验

| 实验 | 对比 | 主结果 | 正式规模 |
|---|---|---|---:|
| E1 容量基线 | production default，不同 context/并发 | KV blocks、容量、capacity knee | 每点 3 次 |
| E2 FP8 KV | 默认 KV dtype vs FP8 | 缓存 Token、并发、VRAM、质量 | 约 18 Runs |
| E3 APC | off/on、cold/warm、reuse/prefix length | hit、TTFT、Goodput | 约 18～24 Runs |
| E4 Prefill | default vs 2～3 个原生 Profile | Decode TPOT/ITL、吞吐 | 每点 3 次 |
| E5 Decode-tail 部署 | production default vs `decode-tail-1024` | 干扰期 Decode ITL p99、Goodput 非劣、Long Prefill TTFT 代价 | 12 Runs |

### 3.1 E1：默认容量与 Planner 验证

使用短/中/长三档 context 和三档压力，记录：KV blocks/cached tokens、peak VRAM、最大稳定并发、achieved/offered rate、TTFT、TPOT/ITL、Goodput、waiting、preemption 和 OOM。

Capacity knee 必须由长 Trace 的吞吐平台、队列增长和 SLO 共同确定，不能用几十个请求的短跑推导。

### 3.2 E2：FP8 KV Cache

正式实验前确认：

- backend 真正进入 FP8 KV 路径；
- scale 来源明确且无 silent fallback；
- 固定输入/seed 能完成输出和质量 sanity check。

FP8 以容量为主结论。显存减少不等于吞吐同比增加；延迟或质量代价必须报告。不兼容时保留证据并继续其他主线。

### 3.3 E3：Automatic Prefix Caching

Workload 使用真实长 System Prompt、RAG Context 或公共历史，不使用无业务意义的重复字符串。

精选 0% reuse 控制、50%/100% reuse、2K/4K Prefix、cold/warm。报告 cached/query tokens、hit ratio、TTFT、TPOT、Goodput、KV usage 和 prefix pool 变大后的失效边界。

APC 主要优化 Prefill/TTFT，不能宣称直接加速每个 Decode Token。

### 3.4 E4：Chunked Prefill 校准与候选筛选

先形成稳定 Decode 流，再注入 4K～8K Long Prefill。比较 production default、原生 `long_prefill_token_threshold`/partial-prefill 和 2～3 个预注册 Profile。

M4 formal 已 sealed，结论保持不变：

- `native-threshold-1024` 和 `native-threshold-512` 在 4K/8K 均 3/3 降低干扰期 Decode ITL p99；
- pooled median 改善分别约 41.5% 和 57.1%；
- `threshold-1024` 的 Long Prefill TTFT 代价更小，约 4%～10%；
- M4 的零容忍方向规则要求每个场景至少 2/3 repeat 的 Decode Goodput 不低于 default；1024 在 8K 仅 1/3，512 为 0/3，因此 M4 仍选择 production default；
- 8K Goodput 的实际差值只有约 `+0.0018%` 到 `-0.0213%`，保留为测量结果，但不据此事后修改 M4 selection。

M4 是 M5 的 calibration evidence，不是 M5 held-out。M5 不修改 M4 artifact、selection 或既有阈值。

### 3.5 E5：Decode-tail 非劣部署验证

M5 改为一个明确的 workload-specific deployment objective：在 Goodput 非劣约束内，降低 4K～8K Long Prefill 对稳定 Decode 流的尾延迟干扰。

只比较两个预注册 Profile：

| Profile | 原生 vLLM 参数 | 角色 |
|---|---|---|
| `production-default` | 空参数，使用真实 upstream default | 对照 |
| `decode-tail-1024` | `enable_chunked_prefill=true`、`long_prefill_token_threshold=1024` | M4 calibration 后冻结的 M5 候选 |

不再测试 threshold-512，不重试 FP8，不比较 APC off，不引入 custom Scheduler。APC 与 Chunked Prefill 在 vLLM 0.16.0 production default 中本来已开启；M5 只能把 1024 threshold 写成 workload-specific 原生参数优化，不能宣称自主实现 Chunked Prefill 或“开启了默认关闭的 APC”。

正式矩阵固定为 12 Runs：

1. Target：同一混合 4K/8K Long Prefill + 稳定 Decode Trace，default/1024 各 3 次，共 6 Runs；
2. Held-out：使用新 Prompt/arrival seed，并改变 Long Prefill 注入时刻或长短请求比例，default/1024 各 3 次，共 6 Runs；
3. 每个 repeat 使用完全相同的 Trace、warmup、到达时间和输出长度；Profile 顺序轮转；
4. Held-out 后禁止重新调 threshold、SLO、非劣界或选择规则。

M5 验收在启动前冻结为：

- Target 和 held-out 的干扰期 Decode ITL p99 配对中位改善均 ≥25%；
- Decode Goodput 配对中位变化均 ≥-0.5%，任一 repeat 不低于 -1%；
- Long Prefill TTFT p99 配对中位退化均 ≤15%；
- Decode TPOT p99 配对中位退化均 ≤2%；
- waiting、preemption 和 KV usage 不出现机制相反的恶化，且 0 OOM、0 timeout、0 preemption；
- 报告每次结果、中位数和范围；不使用最好单次作为结论。

通过后，最终 Profile 只能表述为：

> `decode-tail-1024` 在目标长 Prefill 干扰场景中，以预注册的 Goodput/TTFT 非劣边界换取可重复的 Decode tail latency 改善。

若任一 held-out 主验收失败，M5 保留为负结果并停止“最终部署 Profile 优于 default”的措辞；不得改 margin、换 512 或补跑挑结果。

---

## 4. 实验规则与成功条件

### 4.1 公平性

- 固定模型、Trace、seed、到达时间、输出长度和 warmup；
- 明确 cache reset/warm 协议；
- 每组至少三个 paired repeats；
- 报告每次结果、中位数和范围/置信区间；
- 保存逐请求结果、server log、Prometheus/NVML 和 cleanup；
- 不删除失败、OOM 和不兼容结果。

### 4.2 Held-out

最终 Profile 必须使用未参与选择的新 Prompt/arrival seed，并改变长短请求比例或 Long Prefill 注入时刻。Held-out 不再重新调 Profile、non-inferiority margin 或验收阈值；方向消失时收缩结论。

### 4.3 成功条件

至少满足：

1. Planner 主要容量预测误差 ≤10%；
2. FP8 或 APC 至少一个取得可重复的强正向结果；
3. M5 `decode-tail-1024` 同时通过 target/held-out 的 ITL 改善与 Goodput/TTFT/TPOT 非劣验收；
4. 没有隐藏失败、明显质量异常或不可接受 TPOT 退化。

如果 Planner 无法形成可解释预测，且 FP8/APC 都无有效正结果，必须通知用户重新选择主线，不能只靠图表包装。

---

## 5. M0～M6

| 阶段 | 工作 | 验收 |
|---|---|---|
| M0 | 固定版本、主模型、新 artifact root；production baseline 和 100+ 请求 canary | 基线、resume、cleanup 稳定 |
| M1 | 实现 Planner、单测、vLLM block 交叉验证、长 capacity sweep | 主要预测误差目标 ≤10% |
| M2 | FP8 compatibility、质量和约 18 Runs 正式矩阵 | 得到容量与适用边界 |
| M3 | APC 真实前缀 Trace、冷/热与复用矩阵 | hit 证据与 TTFT/Goodput 一致 |
| M4 | Chunked Prefill 干扰实验与 Pareto calibration | sealed selection 保持 default；候选方向有机制依据，不挑最好单次 |
| M5 | default vs `decode-tail-1024`，target/held-out 各三次配对 | ITL p99 改善且 Goodput、TTFT、TPOT 通过预注册非劣界 |
| M6 | README、结果图、复现命令、简历和面试材料 | 所有措辞符合证据边界 |

每个阶段只在验收通过后进入下一阶段；FP8 单项不兼容不阻塞 APC/Planner/Prefill。

---

## 6. Scheduler 可选加分项

Latency-Budgeted Scheduler 不再决定项目成败。只有满足以下条件才讨论：

- M1～M5 已形成正结果；
- Fixed Prefill Profile 在不同阶段出现明确最优值反转；
- 预计动态策略相对最佳全局 fixed 有足够空间；
- 用户明确同意额外时间和失败风险。

条件不满足就省略。条件满足时先提交独立计划和成功率，未经用户确认不修改 Scheduler。

---

## 7. 后台实验与通知

只有预计超过 1 小时、同路径 Smoke 已通过、参数已冻结并具备 timeout/cleanup/resume/status/log 的实验，才启动为无人值守后台任务。首次兼容性检查、调试和短测试不算。

后台启动并确认 PID、GPU、日志、结果目录和 Resume 后，必须通知：

> 后台正式实验已经启动并通过脱离会话检查，现在可以关闭 VSCode。请保持 AutoDL 实例开机，不要关机或释放。以下是预计结束时间、PID、日志、结果目录、状态和 Resume 命令：……

每完成 M0～M6 都单独通知：完成内容、修改文件、测试/实验数量、正负结果、artifact 路径、下一阶段/时间和当前能否关闭 VSCode。

---

## 8. 时间估算

以下原估算包含兼容性排查、失败重试、长 capacity trace 和保守 timeout，不等于每个后期矩阵都必须运行数小时：

| 阶段 | 主动操作 | 后台 GPU |
|---|---:|---:|
| M0～M1 | 4～9 小时 | 4～8 小时 |
| M2～M4 | 6～12 小时 | 12～30 小时 |
| M5～M6 | 4～8 小时 | 6～15 小时 |

总计主动工作约 14～29 小时、后台 GPU 22～53 小时，正常约 4～7 天；兼容性或模型问题可能延长到 7～10 天。

新版 M5 固定 12 Runs，按 M4 实测预计 GPU wall time 约 25～40 分钟，包含实现、同路径 smoke、分析和 seal 的主动工作约 3～5 小时。预计不足 1 小时，不作为无人值守后台任务；状态按当前约定每 5 分钟检查一次。

---

## 9. 风险与主线重选

| 风险 | 处理 |
|---|---|
| 7B/8B 不兼容 | M0 明确降级，后续不迁移阈值 |
| FP8 fallback/质量问题 | 查 backend、scale、容量和质量；不兼容则省略 |
| APC workload 被质疑 | 真实 System/RAG Prefix + 0% reuse 控制 |
| Planner 误差大 | 分解权重、Graph、block、metadata，禁止硬拟合 |
| Decode-tail Profile 不满足非劣界 | 保留 M4 ITL/TTFT trade-off 与 M5 held-out 负结果，不改 margin 或换候选补跑 |
| 项目退化成纯消融 | Planner、预测验证和 Profile 决策必须完成 |

需要重新选择主线的条件：Planner 无法达到可解释预测，并且 FP8/APC 均无可复现正结果；或最终 Profile 相对 production default 没有可展示的容量、TTFT 或 Goodput 改善。

触发后必须通知：

> **主线重选提醒**：当前长上下文优化未形成足够的可展示正结果，继续投入会降低项目含金量。我已停止后续包装；下一步需要与你重新选择主线。

不会自动改成 Scheduler、Prefix-aware、BlockPool 或 Kernel。

---

## 10. 面试价值与 Definition of Done

现有学习文档覆盖 V1 生命周期、Scheduler、Chunked Prefill、Paged KV、GQA、APC、FP8 和性能指标。项目只需补充 KV bytes/token、非 KV 显存、capacity knee、FP8 scale 和 Prefix 冷热边界。

含金量定位：高于旧 SLOTune 和纯开关消融；低于成功的原创 Scheduler/Kernel；完成概率更高，且更容易真正学会和答辩。

完成条件：

- [ ] 固定正式模型、vLLM 和环境；
- [ ] Planner、单测和容量验证完成，主要误差目标 ≤10%；
- [ ] 默认 capacity curve 完成；
- [ ] FP8 容量/质量边界完成；
- [ ] APC cold/warm/reuse 控制完成；
- [x] Chunked Prefill M4 calibration 完成并 sealed；
- [ ] `decode-tail-1024` vs production default 完成 6 个 target + 6 个 held-out Runs；
- [ ] M5 同时通过 ITL 改善与 Goodput/TTFT/TPOT 非劣验收；
- [ ] 至少一个强正向结果可重复；
- [ ] 无隐藏失败、OOM、错误或明显质量退化；
- [ ] 结论只使用 clean、可复查的新 artifacts；
- [ ] 每个里程碑和后台长跑按规则通知；
- [ ] 触发失败条件时先通知并重新选择主线。

执行顺序：

~~~text
M0 版本与正式模型
→ M1 KV Capacity Planner
→ M2 FP8 KV Cache
→ M3 Automatic Prefix Caching
→ M4 Chunked Prefill calibration（sealed）
→ M5 Decode-tail non-inferiority + Held-out
→ M6 README + 面试材料
~~~

先取得可信、可复现的真实 vLLM 优化结果，再决定是否增加 Scheduler 加分项；不让高风险研究假设决定整个项目成败。
