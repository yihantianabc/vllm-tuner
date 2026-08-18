# 长上下文 v5：简历与面试材料

这份材料只使用 sealed artifacts 中能够复查的事实。推荐定位是“推理系统工程、容量规划与
可复现实验决策”，不是“发明了 APC/Chunked Prefill”，也不是“自研 Scheduler/Kernel”。

## 推荐简历成稿

**项目名：vLLM 单卡长上下文推理优化与 KV Capacity Planner**

**技术栈：Python、vLLM 0.16.0、PyTorch、Prometheus、NVML、Qwen2.5-7B、RTX 5090**

- 独立设计并实现 KV Capacity Planner，基于 GQA/KV dtype/block geometry、vLLM null
  block、非 KV 显存校准和安全余量预测 KV blocks、cached tokens 与上下文安全并发；在
  1 个 held-out 32K 点和 2 个未参与校准的 8K/16K profile 上，KV block/并发预测最大绝对
  误差为 **0.0446%**（目标 ≤10%），给出 BF16 KV **192,880 tokens** 及 8K/16K/32K
  **23/11/5** 路安全并发上界。
- 构建可复查的单卡实验链路：冻结 trace、balanced paired repeats、target/held-out、逐请求
  TTFT/TPOT/ITL/Goodput、Prometheus/NVML、cleanup、manifest/checksum/integrity seal；完成
  APC **20/20** 和 Chunked Prefill calibration **18/18**，在真实 RAG 前缀中测得 4K prefix
  50%/100% reuse 的 warm TTFT 配对中位改善 **11.0%/55.1%**，18/18 配对 Goodput 不降低。
- 使用 upstream 原生 Chunked Prefill 冻结并验证 `decode-tail-1024`：12 次运行（6 个
  target/held-out profile pairs）将干扰期 Decode ITL p99 分别降低 **42.8%/42.9%**，Goodput 仅
  **-0.027%/-0.014%**，Long Prefill TTFT p99 代价 **+5.7%/+6.5%**，且 0
  OOM/timeout/preemption；保留 FP8 KV 不兼容和 M4 strict gate 负结果，不用补跑或最好单次
  包装结论。

### 一页简历空间不足时

> 独立实现 vLLM KV Capacity Planner（GQA/block rounding/非 KV 显存校准/安全余量），在
> Qwen2.5-7B + RTX 5090 的 3 个 unseen profile 上将 KV blocks/并发最大预测误差控制在
> 0.0446%；基于 12 runs/6 pairs 的 target/held-out 实验落地 `decode-tail-1024`，将 Long Prefill
> 干扰期 Decode ITL p99 稳定降低 42.8%/42.9%，Goodput 变化仅 -0.027%/-0.014%。

### English version

- Built a KV Capacity Planner for vLLM that models GQA KV geometry, block rounding, the reserved
  null block, calibrated non-KV residency, and safety margins; achieved ≤0.0446% maximum observed
  KV-block/concurrency error across one held-out and two context-extrapolation profiles on
  Qwen2.5-7B/RTX 5090.
- Designed sealed paired/held-out experiments around upstream APC and Chunked Prefill; validated
  `decode-tail-1024` over 12 runs with 42.8%/42.9% median Decode ITL-p99 improvement at
  -0.027%/-0.014% Goodput change and zero OOMs, timeouts, or preemptions.

## 30 秒项目介绍

> 这是一个 vLLM 单卡长上下文推理工程项目。我的核心自主贡献是 KV Capacity Planner：从
> 模型 GQA 结构、KV dtype、block size 和显存预算推导 blocks、cached tokens 与安全并发，
> 再用多点启动数据校准非 KV 占用，并在 unseen profile 上验证。性能侧我没有改 vLLM
> 内核或 Scheduler，而是围绕它原生的 APC 和 Chunked Prefill 建了配对、held-out、可封存的
> 实验链。最终 `decode-tail-1024` 在 12 次运行中把 Decode 干扰 ITL p99 降了约 43%，同时把
> Goodput、TTFT、TPOT、KV 和可靠性都约束在部署边界内。FP8 不兼容和 M4 的负选择也完整
> 保留了。

## 2 分钟主线讲解

1. **业务问题**：单卡长上下文不是只看“能否装下”。长 Prefill 会抢占计算，拖累稳定
   Decode 流的尾延迟；APC 收益又依赖真实可复用前缀，不能用重复字符串证明。
2. **自主工程贡献**：实现 Capacity Planner。结构项先算每 token KV bytes，再处理 block
   rounding、null block、尾块、非 KV 驻留和安全 reserve；输出 cached tokens、安全并发和
   context-distribution 风险。
3. **验证方式**：75/80/85% 显存利用率只用于校准；90% 和 8K/16K profile 留作验证。最坏
   block/并发误差 0.0446%，但明确只对锁定环境成立，换模型/GPU/runtime 要重校准。
4. **原生能力分析**：APC 用 2K/4K 真实 RAG 前缀和 0/50/100% reuse，结果随可复用 token
   增长，并测出 prefix pool 从 48 到 72 的 miss 边界；这说明是 Prefill reuse，不是 Decode
   kernel 加速。
5. **部署选择**：M4 的 strict rule 对任何 Goodput 负方向零容忍，所以仍选 default。M5
   重新预注册工程 non-inferiority，冻结 1024，只跑 target/held-out 各三对，最后在约 43%
   ITL p99 收益下，把 Goodput 损失控制在 0.03% 内。
6. **工程判断**：M5 原 gate 被三个瞬时 KV blocks 触发。我没有改原 artifact 或补跑，而是
   保留负结果，用同一 12-run 数据生成独立工程复分析；结合 p95、waiting、可靠性和
   2.625 MiB 物理量判断其不构成容量风险。

## 面试深挖问答

### 1. Planner 为什么不只是套公式？

基础公式只能得到 payload：

```text
layers × 2(K+V) × KV heads × head_dim × dtype_bytes
```

工程容量还取决于 block 向上取整、每层 page geometry、vLLM 保留的 null block、权重与
runtime residual、CUDA Graph/workspace 等非 KV 占用、固定和比例安全余量，以及请求长度
分布。Planner 把“物理 payload”“运行时校准”“部署 policy”分开，避免把一个实验点硬拟合
成全局常数。

### 2. 0.0446% 为什么这么低，是否过拟合？

需要主动收缩结论。75/80/85% 的六个 probe 用来估计同一锁定环境的非 KV residual；90%
held-out 验证可用 KV 随 utilization 的外推，8K/16K 验证 context geometry。Qwen2.5-7B 的
结构、block size 和运行时没有变，所以 block 数应高度可预测。这个结果证明当前模型/GPU/
runtime 的模型闭环，不证明 residual 可以跨 GPU、跨版本迁移。

### 3. Memory-safe concurrency 和 capacity knee 有什么不同？

Planner 的 23/11/5 是“完整 8K/16K/32K 请求在 KV 上能否安全驻留”的内存上界。线上
capacity knee 由算力、队列、SLO、preemption 和到达过程共同决定，可能远早于 OOM。M1 v2
因此把 SLO service boundary 与 joint saturation boundary 分开报告。

### 4. APC 的 100% reuse 为什么 hit ratio 不是 100%？

Prometheus 记录的是 token hit/query，而不是“请求是否声明复用”。每个请求除共享 prefix
外还有私有后缀，因此 4K/100% reuse 的 token-hit ratio 约 50.8%。0% 控制为零、prefix
越长/reuse 越高 TTFT 收益越大，才和 APC 的 Prefill 机制一致。

### 5. 你是否实现了 APC 或 Chunked Prefill？

没有。它们是 upstream vLLM 0.16.0 的原生能力，而且 production default 已开启。我的贡献
是工作负载设计、观测、配对/held-out 证据、边界分析和 `long_prefill_token_threshold=1024`
的 workload-specific 部署决策。

### 6. M4 既然选 default，为什么 M5 又能选 1024？

两个阶段的问题和规则不同，且 M5 使用新数据。M4 是 calibration，规则对 Goodput 的任何
负方向零容忍；1024 在 8K 只有 1/3 repeat 方向不低于 default，因此必须保留 default。
M5 在候选冻结后，用 target + 未调参 held-out 验证预注册工程边界：Goodput 中位 ≥-0.5%、
单次 ≥-1%，同时限制 TTFT/TPOT/KV/排队/可靠性。M5 通过并不回写 M4。

### 7. 为什么可以修改 M5 的 KV 判据？

不是偷偷修改同一结果。原 formal artifact、失败理由和 production-default decision 保持
byte-identical。后续发现“200 ms 采样中的单点 maximum 必须零变化”不能代表持续容量风险，
于是版本化增加 materiality rule，并从同一 12-run 原始记录生成新 sealed artifact：p95
中位与任一 repeat peak 增量均不得超过 usable KV 的 0.1 percentage point。实际 peak 是
0.02053 point/3 blocks，p95 中位相同、waiting 更低、可靠性事件为零。

### 8. 为什么不用最好的一次结果？

每个比较都是同 trace、同到达时间、同输出长度的 paired repeats，报告每次和中位数；最终
还必须在 prompt/arrival seed 和 workload 组成变化后的 held-out 保持方向。M5 的 headline
42.8%/42.9% 是 paired median，不是最好单次；target 单次范围 36.3%–68.7% 也保留在结果页。

### 9. FP8 失败后为什么停止？

当前锁定栈的 `fp8_e5m2 + TRITON_ATTN` 在 engine profiling 就触发 attention dtype assertion，
checkpoint 又没有校准 scale keys。此时继续正式矩阵只能产生 fallback 或不可比数据。工程上
正确动作是封存启动日志与 BF16 control、明确“无 FP8 性能结论”，继续独立的 Planner/APC/
Prefill 主线，而不是为了正结果换 backend 或反复试错。

### 10. 最终部署时怎么用？

只在锁定的 Qwen2.5-7B、RTX 5090、vLLM 0.16.0 与相近 4K/8K Long Prefill 干扰 workload
使用：保持 upstream 默认 APC/Chunked Prefill，显式设置
`long_prefill_token_threshold=1024`。上线前复核 runtime/model lock、容量 Planner、trace
相似度和 Goodput/TTFT/TPOT guardrails；工作负载或版本改变就重新校准，不沿用“1024 最优”。

## 面试中应主动说出的边界

- 自主实现的是 Planner 和实验/证据系统，不是 vLLM 原生 APC/Chunked Prefill。
- 42.8%/42.9% 是目标干扰场景的 paired median，不是通用吞吐提升。
- Goodput 有微小负变化；Long Prefill TTFT 和 Decode TPOT 有可量化代价。
- M4 的 production-default selection 和 M5 original negative 都没有被删除或改写。
- FP8 是真实负结果，没有容量/质量收益数据。
- 项目未启动 custom Scheduler，也没有 kernel 级贡献。

## 不推荐的表述

- “自研 vLLM Chunked Prefill/APC”——归属错误。
- “吞吐零损失”——实测 Goodput 为轻微负变化，应写约 -0.03%/-0.01%。
- “长上下文性能提升 43%”——缺少指标和场景，应写 Decode interference ITL p99。
- “FP8 将 KV 容量翻倍”——当前栈没有兼容的 paired evidence。
- “Planner 跨模型误差 0.0446%”——只验证了锁定 Qwen2.5-7B/runtime 的 unseen profiles。
- “M4 选出了 1024”——M4 按原规则选择的是 production default；1024 是 M5 工程部署结果。
