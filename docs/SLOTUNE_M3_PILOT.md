# SLOTune M3：非平稳 workload、固定基线与 Formal 冻结

日期：2026-08-17  
状态：完成 Pilot 与 Formal 协议冻结；尚未开始 M4 Formal/held-out 长跑。

## 1. 本阶段结论

M3 已在 Qwen2.5-3B-Instruct 和真实 vLLM V1 Scheduler 上完成 stock、固定
1024/4096/8192 Prefill cap、两版 Adaptive 阈值以及 default capacity sweep。所有
Pilot 请求成功，未发生 OOM、preemption、漏请求或清理失败。

固定 Budget 的取舍确实存在，但当前 Pilot **不支持 Adaptive 已优于 stock** 的结论：

- fixed-low 将 pooled p99 ITL 从 fixed-high 的约 252.5 ms 降到 59.7 ms，但整体
  achieved throughput 从 7.611 降到 7.080 req/s；
- fixed-mid 在 long-prefill 与 mixed 阶段取得最低 p99 TTFT，分别约 905.3 ms 和
  851.8 ms，但 p99 TPOT 高于 fixed-high；
- fixed-high 的整体吞吐最好，并且宽松 SLO 下仍是 fixed-only per-phase Oracle 的
  三阶段选择；因此不能宣称每个阶段的 SLO-goodput 最优固定档不同；
- 冻结候选 Adaptive-tail28 的 achieved throughput 为 7.576 req/s，约比 fixed-high
  低 0.46%，也低于 stock 的 7.649 req/s；它尚未达到项目预设正结果门槛。

这些结果保留为真实 Pilot 证据。M4 将检验更长 trace、三个 offered-load 点和
held-out 顺序，不能用 M3 的 32 请求短跑作为 headline。

## 2. 冻结 trace

Pilot trace 每次 32 个请求，连续包含：

1. `decode_heavy`：12 个 256-token Prompt、256-token Output；
2. `long_prefill_burst`：8 个 4096-token Prompt、32-token Output；
3. `mixed`：12 个确定性混合长度请求。

Calibration 顺序为 decode-heavy → long-prefill → mixed；held-out 改为
long-prefill → mixed → decode-heavy。同名请求的 prompt、输入长度和输出长度完全
一致，只有阶段顺序与到达 offset 改变。控制器从未读取 `profile` 标签；标签仅由
`scripts/analyze_adaptive_prefill_pilot.py` 在运行结束后 join。

Formal trace 将每个 phase 的请求数放大 20 倍，共 640 个 measured requests，并只缩放
arrival offset 形成三个相同请求集合的负载点。校验和与矩阵保存在
`experiments/adaptive_prefill/m3_formal_protocol.yaml` 和
`experiments/adaptive_prefill/traces/formal/manifest.json`。

## 3. Capacity sweep 与负载点

Stock capacity sweep 使用同一 32 请求集合，粗扫结果如下。`empirical offered` 是有限
Gamma arrival schedule 的实际值，不能与配置中的 target alias 混用。

| Target req/s | Empirical offered req/s | Achieved req/s | p99 TTFT ms | p99 TPOT ms | Peak waiting |
|---:|---:|---:|---:|---:|---:|
| 2 | 2.287 | 2.242 | 256.7 | 13.3 | 1 |
| 4 | 4.574 | 4.036 | 377.5 | 24.3 | 1 |
| 8 | 9.147 | 6.564 | 535.9 | 41.0 | 5 |
| 12 | 13.721 | 7.570 | 729.0 | 44.3 | 8 |
| 16 | 18.295 | 7.579 | 869.6 | 44.1 | 12 |
| 24 | 27.442 | 8.628 | 1027.6 | 44.1 | 14 |
| 32 | 36.590 | 8.879 | 1123.7 | 44.1 | 15 |

以约 8.9 achieved req/s 的平台区作为工程 capacity 估计，Formal 冻结为精确 empirical
offered load 6.0、8.0、9.5 req/s，约对应 67%、90%、107%。不同策略必须使用相同
trace 和相同 offered load。

## 4. Stock-only SLO 冻结

两次 stock calibration 的中位聚合分布为：TTFT p50/p95/p99 =
301.0/845.8/956.4 ms，TPOT = 12.2/36.8/42.1 ms，E2E =
1950.5/3163.7/3170.5 ms。Adaptive-tail28 运行前已据此冻结：

| Tier | TTFT ms | TPOT ms | E2E ms | 用途 |
|---|---:|---:|---:|---|
| strict | 300 | 12 | 1950 | 约 stock p50，报告下界 |
| medium | 850 | 37 | 3170 | 约 stock p95，Formal 主口径 |
| loose | 1100 | 47 | 3500 | 高于 stock p99，报告完成稳定性 |

三组口径全部报告，不能只选最有利的一组。ITL 作为 pooled p95/p99/max 单独报告；当前
在线 objective 仍按逐请求 TTFT/TPOT/E2E 计算。

## 5. Fixed 与 Adaptive Pilot

下表是两次 complete/feasible calibration 的逐 trial 指标中位数，SLO good fraction 使用
冻结的 medium SLO。

| Policy | Good fraction | Achieved req/s | p99 TTFT ms | p99 TPOT ms | pooled p99 ITL ms |
|---|---:|---:|---:|---:|---:|
| stock | 0.875 | 7.649 | 956.4 | 42.1 | 249.1 |
| fixed-1024 | 0.344 | 7.080 | 1040.7 | 42.2 | 59.7 |
| fixed-4096 | 0.469 | 7.486 | 896.4 | 44.7 | 138.3 |
| fixed-8192 | 0.594 | 7.611 | 956.8 | 42.4 | 252.5 |
| adaptive-tail28 | 0.562 | 7.576 | 977.8 | 43.7 | 241.5 |

首版 Adaptive 使用 backlog 阈值 8，两次分别有 255 个 `PROTECT_DECODE` step，且没有
进入 `DRAIN_PREFILL`，结果几乎退化为 fixed-low。其 signal 分布为 backlog p95=24、
max=30，因此第二版只把 `decode_backlog_high` 提高到 28，其余保持不变。

冻结候选两次状态计数分别为：

- run 1：BALANCED 628、PROTECT_DECODE 28、DRAIN_PREFILL 6；
- run 2：BALANCED 624、PROTECT_DECODE 28、DRAIN_PREFILL 6。

两次均调度 44,714 Prefill token 与 5,232 Decode token，未出现
`max_wait_progress_not_met`。这证明三状态可解释地工作，但不证明收益。

## 6. 冻结参数与复现入口

- Formal 协议：`experiments/adaptive_prefill/m3_formal_protocol.yaml`；
- Trace 生成：`scripts/generate_nonstationary_trace.py`；
- Phase/Oracle 分析：`scripts/analyze_adaptive_prefill_pilot.py`；
- Pilot 配置：`experiments/adaptive_prefill/m3_*_pilot.yaml`；
- Pilot/分析 artifacts：
  `/root/autodl-tmp/vllm-tuner-output/slotune-results/slotune-m3-*`。

Formal 固定三次重复、640 measured requests、5 个 warmup、三个 offered-load 点、
calibration/held-out 两种阶段顺序。M4 开始后不再根据结果修改 cap、阈值、SLO 或主矩阵；
若结果为负，保留负结果并收缩结论。
