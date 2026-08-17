# Task4 · 多智能体智能路由与科研任务调度

对应《项目能力测评》**第四题**（满分 25 分）：为科研办公平台的异质任务选择性价比最优的大模型。

## 〇、实现口径说明（纯模拟）

> **本实现为纯模拟方案，未接入真实 API 调用**（放弃加分项「成本闭环验证 +3 分」）。
> 所有对比实验、动态路由与降级事件均基于 `data/data.csv` 的 benchmark 数据 + 价格 + 吞吐率经验值进行模拟：
>
> - **任务完成质量** = 任务相关 benchmark 加权分（N/A 按保守口径处理，见下文）；
> - **总成本（RMB）** = 按模型官方单价 × 任务 token 数计算（与真实报价一致，非真实扣费）；
> - **端到端耗时** = 按模型吞吐率经验值估算；
> - **动态路由中间质量** = 理论质量 + 随机噪声（模拟真实波动），seed 固定可复现；
> - **限流/超时降级** = 概率模拟事件（`fail_rate=0.15`）。
>
> 如需接入真实 API：在项目根目录建 `.env`（格式见 `README` 底部），将 `fail_rate` 置 0、
> 以真实响应评测质量即可，代码结构无需改动（`router.py` 的决策接口与真实调用解耦）。

## 一、项目结构

```
task4/
├── data/
│   ├── data.csv                    # 模型能力-成本调研数据（10 款模型 × 23 列）
│   └── data_backup_原始未补全.csv    # 原始 8 款模型数据备份
├── scoring.py                      # 4.3 综合路由评分函数（能力/成本/可用性）
├── router.py                       # 4.3 路由决策 + 降级链 + 异常处理
├── run_demo.py                     # 端到端演示：10 任务路由 + 4 组对比实验 + 可视化
├── experiments/
│   └── results/
│       ├── experiment_summary.json # 4.4 四组策略对比汇总
│       └── experiment_results.csv  # 每个任务/策略的明细
├── logs/
│   ├── routing_log.csv             # 每次路由决策日志
│   └── degradation_events.json     # 降级事件（限流/超时/预算超限/能力不足）
└── figures/                        # 4.5 三张可视化
    ├── routing_heatmap.png         # 路由决策热力图
    ├── pareto_cost_quality.png     # 成本-质量帕累托散点图
    └── degradation_timeline.png    # 降级事件时间线图
```

## 二、快速开始

```bash
python run_demo.py
```

输出：10 个科研任务的路由决策、4 组对比实验结果、3 张可视化图片，并写入 logs/ 与 experiments/results/。

依赖：Python 3.10+，pandas / numpy / matplotlib（均为常见库）。

## 三、核心设计（4.3 路由策略）

### 综合路由评分函数

```
Score(m, t) = α · NormCap(m, t) + β · NormCost(m, t) + γ · NormAvail(m)
```

| 分量 | 含义 | 计算方式 |
|---|---|---|
| NormCap | 能力匹配度 | 任务相关 benchmark 加权分，池内 min-max 归一化到 [0,1] |
| NormCost | 成本效率 | 「满足质量要求下的最低成本」（FrugalGPT 质量约束的软惩罚版本）：CostEff = clamp(质量分/阈值, 0, 1) × max(0, 1 − cost/预算)。质量低于要求时按比例打折（软约束，不“一刀切”，质量略低的模型仍可用但不划算）；质量达标后只看成本；预算外为 0。参照 FrugalGPT(arXiv:2305.05176) 与 Artificial Analysis 性价比口径 |
| NormAvail | 可用性 | 0.6·RPM余量 + 0.4·历史成功率 |

权重随质量要求等级调整（设计依据）：
- 质量「高」：α=0.60, β=0.25, γ=0.15（能力优先）
- 质量「中」：α=0.45, β=0.35, γ=0.20（能力/成本均衡）
- 质量「低」：α=0.30, β=0.50, γ=0.20（成本优先）

任务类型 → benchmark 映射：文献检索/论文精读 → MMLU-Pro + 长文本；代码复现 → HumanEval；
数据分析/公式推导 → MATH；学术写作/审稿回复 → IFEval；图表解读 → MMLU-Pro + 视觉能力。

### 异常处理

1. **能力不足**：任务相关质量分低于阈值（高 75 / 中 60 / 低 45 分）的模型被剔除；
   若全部不足，选择质量最高的模型兜底。
2. **限流/超时**：降级链（仅含能力合格模型）首选→次选→…→兜底，每步记录成本与质量依据。
3. **预算超限**：首选超预算时自动降级到预算内评分最高的合格模型。

### 路由决策伪代码（题目 4.3 要求）

```
Algorithm: route(task, models)
Input:  task   = {task_type, in_tokens, out_tokens, quality_level, budget}
        models = candidate model pool
Output: {chosen_model, est_cost, est_quality, fallback_chain, events}
1  for each model m in models:                              # 对每个候选模型
      score(m) = α·cap(m) + β·cost_eff(m) + γ·avail(m)     # 综合路由评分
      where:
        cap(m)      = Σ w_b · norm(m.bench[b])              # 能力匹配度：任务相关 benchmark 归一化加权
        cost(m)     = price_in(m)·in_tokens + price_out(m)·out_tokens   # 预估成本(RMB)
        cost_eff(m) = normalize(1 / cost(m))                # 成本效率：成本越低得分越高
        avail(m)    = rpm_margin(m) + success_rate(m)       # 可用性：RPM 余量 + 历史成功率
      # 权重 α/β/γ 按 quality_level 调整：高(0.60/0.25/0.15)、中(0.45/0.35/0.20)、低(0.30/0.50/0.20)
2  qualified = { m | quality(m) >= threshold }              # 能力门槛：高75/中60/低45
   if qualified is empty:                                   # 异常1：能力不足
       chosen = argmax quality(m);  log(capability_insufficient)
3  chosen = argmax score(m) over m in qualified             # 首选：合格模型内评分最高
4  if cost(chosen) > budget:                                # 异常3：预算超限
       chosen = argmax score(m) over { m in qualified | cost(m) <= budget }
       log(budget_overrun)
5  fallback_chain = top_5(qualified, key=score)             # 降级链：每项附 cost 与 quality 依据
6  if api_failure(rate_limit or timeout):                   # 异常2：限流/超时
       chosen = next(fallback_chain);  log(api_failure)
       # 预算约束在降级链中始终生效：cost(chosen) <= budget 恒成立
7  return chosen, est_cost, est_quality, fallback_chain, events
```

> 说明：以上为思路级伪代码，使用与 `router.py` 一致的变量命名；可运行实现见 `router.py::route()`。**N/A 处理口径**（避免"一律记 0"失真）：
- 归一化能力分 cap（相对排名）：N/A → 0，表示"无公开数据 = 该维度排名底线"（保守）；
- 绝对质量分 quality（门槛与对比实验）：官方未公布的 benchmark → 用候选池内最低公开分保守估计；
  明确不支持的能力（如纯文本模型做视觉任务）→ 记 0，不能胜任。
- 全流程口径在 scoring.py / router.py 注释与《数据补全说明.md》中一致。

## 四、对比实验结果（4.4）

| 策略 | 说明 | 总成本(元) | 平均质量(分) | 总耗时(s) |
|---|---|---|---|---|
| router | 完整路由系统 | **1.212** | **87.18** | 1416.1 |
| best_quality | 全量能力最强(Claude-3.5-Sonnet) | 3.989 | 85.57 | 2277.7 |
| cheapest | 全量成本最低(Gemini-2.0-Flash) | 0.121 | 85.91 | 1046.7 |
| random | 随机路由 | 2.047 | 83.97 | 1788.4 |

**口径说明**：「综合能力最强」在候选模型有公开数据的 benchmark 上取平均（至少 4/6 项公开数据才参评，
缺失项不计入分母）。当前模型池中 Claude-3.5-Sonnet 综合平均最高，故当选为 best_quality 策略的固定模型。

**结论**：路由策略以 1.212 元（仅为「全量能力最强」Claude 的约 30%）取得最高平均质量（87.18 分）。
全用 Claude 不仅贵 3.3 倍，平均质量反而低约 1.6 分——因为 Claude 在数学/审稿等任务上并非最优
（MATH 71.1 低于 Qwen2.5-72B 的 83.1、IFEval 88 低于 Gemini 的 90.4），"全用最强模型"存在明显浪费。
路由按任务类型分配：代码/精读用 Claude、数学用 Qwen2.5-72B、长文本用 DeepSeek-V3、通用任务用
Gemini，各展所长。比「全量成本最低」质量高约 1.3 分，比随机路由成本降低约 41%。
优势最大的场景：任务质量要求差异大、预算敏感、任务类型混合的科研办公负载。

## 五、数据说明（4.2）

- 10 款模型（闭源 7 / 开源 3），涵盖题目指定的 GPT-4o、Claude 3.5 Sonnet、Gemini 2.0 Flash、Qwen-Max、DeepSeek-V3、GLM-4；其余为 GPT-4o-mini、Llama-3.1-405B、Moonshot-v1-128k、Qwen2.5-72B-Instruct（均为 2024 年及更早发布的同期模型）。
- 每款含定价（USD+RMB）、视觉定价、RPM/TPM、上下文窗口、Structured Output、MMLU/MMLU-Pro/HumanEval/IFEval/MATH/NIAH/PubMedQA 得分。
- 字段口径、来源 URL 与采集日期见 data.csv 的 Benchmark_Notes / Data_Source_URL / Collection_Date，及《数据补全说明.md》。


## 七、附加分：动态路由调整（+3 分）

**题目要求**：第四题的路由在任务执行过程中根据中间结果质量动态调整后续模型选择。

**实现**：`router.py::route_dynamic()`（数据类 `DynamicStep` / `DynamicRoutingDecision`），
主流程演示见 `run_demo.py::run_dynamic()`。

### 机制

```
任务拆成 n_steps=4 步执行：
  每步: 当前模型执行该步 -> 评估中间结果质量（观测 = 理论质量 + 随机噪声，模拟真实波动）
  若 观测质量 < 理论质量 - switch_threshold(5分):
      动态切换到「质量更高 + 成本在预算内 + 非当前」的模型（ranked 按评分降序取首个）
  否则: 保持当前模型
记录每步的模型 / 理论质量 / 观测质量 / 成本 / 是否切换 / 原因
```

### 与静态路由的差异

| 维度 | 静态路由 | 动态路由 |
|---|---|---|
| 模型选择时机 | 任务开始时一次性决定 | 任务执行中按中间质量持续调整 |
| 中途质量下滑 | 仅靠限流/超时被动降级 | 主动切换更优模型 |
| 成本 | 整任务按单一模型计价 | 按各步实际使用模型分块计价 |

### 演示结果（seed=20260812，10 个任务）

- 静态路由：总成本 1.2118 元，平均质量 87.18 分
- 动态路由：总成本 0.7458 元，平均质量（观测）87.45 分，动态切换 1 次
- 典型案例 T05（数据分析）：步骤 1 用 Gemini，步骤 2 观测质量下滑至 71.8（低于预期 75.6−3.0），
  主动切换至质量更高的 DeepSeek-V3（78.1），后续步骤保持 V3——避免整任务质量损失。
  完整逐步日志见 logs/dynamic_routing_log.json，轨迹见 figures/dynamic_trajectory.png（红圈=切换）。

输出：`logs/dynamic_routing_log.json`（每步明细）、`figures/dynamic_trajectory.png`（动态调整轨迹图）。
## 六、可复现性

- 随机种子固定（run_demo.py 内 seed=20260812），每次运行结果一致。
- 限流/超时降级为概率模拟（fail_rate=0.15），真实部署时替换为监控系统触发的事件。