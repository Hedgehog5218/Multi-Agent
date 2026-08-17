# -*- coding: utf-8 -*-
"""
router.py —— 问题四 4.3「路由策略设计」路由决策模块

路由输入：任务描述 + 任务类型 + 输入数据规模(预估 token) + 质量要求等级(高/中/低) + 预算约束(RMB)
路由输出：模型名称 + 预估成本 + 预估质量得分 + 备选方案(降级链)

异常处理（题目要求）：
    1. 能力不足：所有模型能力分均低于质量阈值时，降级选择能力分最高的模型
    2. API 限流/超时：完整降级链（首选 -> 次选 -> ... -> 兜底），每一步记录成本与质量依据
    3. 预算超限：首选成本超预算时，自动降级到满足预算且能力达标、评分最高的模型
"""
from __future__ import annotations
import json
import random
from dataclasses import dataclass, field, asdict
from datetime import datetime

from scoring import Model, load_models, route_scores, QUALITY_CONFIG, TASK_BENCHMARKS

# 降级链长度（含首选）
FALLBACK_CHAIN_LEN = 5
# 单次调用触发限流/超时的概率（演示用，可配置；真实场景由监控系统驱动）
SIMULATED_FAIL_RATE = 0.15
# 触发降级时向前推进的步数
FALLBACK_STEPS = 1


@dataclass
class Task:
    """一个科研路由任务。"""
    task_id: str
    name: str
    task_type: str
    desc: str
    input_tokens: int
    output_tokens: int
    quality: str            # 高 / 中 / 低
    budget_rmb: float
    expected_rpm: float = 10.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class DegradationEvent:
    """降级事件（限流/超时/能力不足/预算超限）。"""
    time: str
    task_id: str
    task_name: str
    model: str
    event_type: str         # api_rate_limit / api_timeout / capability_insufficient / budget_overrun
    reason: str
    fallback_to: str = ""
    fallback_est_cost: float = 0.0
    fallback_est_quality: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RoutingDecision:
    """一次路由决策的完整结果。"""
    task: Task
    chosen_model: str
    score: float
    est_cost: float
    est_quality: float
    fallback_chain: list = field(default_factory=list)   # 降级链（含首选），每项含依据
    events: list = field(default_factory=list)           # 本次任务触发的降级事件
    total_latency_s: float = 0.0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["task"] = self.task.to_dict()
        return d


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _est_quality(models: list[Model], task: Task) -> dict[str, float]:
    """任务相关 benchmark 加权能力分（原始得分加权 -> 0~100 量纲）。
    作为「任务完成质量」的量化标准（4.4 对比实验使用）。

    N/A 的处理（区分两种缺失，避免一律记 0 导致失真）：
      - 「官方未公布」的 benchmark（如 R1 的 IFEval/NIAH）：用候选池内该 benchmark 的
        最低公开分做保守估计，表示“能力未知但按最弱公开水平看待”，而非直接判 0；
      - 「明确不支持」的能力（Vision=None，即纯文本模型）：记 0，且不能胜任视觉任务。
    """
    weights = TASK_BENCHMARKS[task.task_type]
    total_w = sum(weights.values())
    # 计算每个 benchmark 在候选池内的最低公开分（保守估计底）
    floor = {}
    for bench_name in weights:
        vals = [m.bench.get(bench_name) for m in models]
        valid = [v for v in vals if v is not None]
        floor[bench_name] = min(valid) if valid else 0.0
    q = {}
    for m in models:
        total = 0.0
        for bench_name, w in weights.items():
            val = m.bench.get(bench_name)
            if val is None:
                val = 0.0 if bench_name == "Vision" else floor[bench_name]
            total += w * val
        q[m.name] = total / total_w
    return q


def _latency_s(model: Model, input_tokens: int, output_tokens: int) -> float:
    """端到端耗时模拟：固定开销 0.5s + (输入+输出)/吞吐率。
    吞吐率按模型规模/速度经验值设定，用于 4.4 对比实验的耗时维度。"""
    tps = {
        "GPT-4o": 60, "Claude-3.5-Sonnet": 55, "Gemini-2.0-Flash": 120,
        "Qwen2.5-Max": 80, "DeepSeek-V3": 90, "GLM-4": 70,
        "Llama-3.1-405B": 30, "Moonshot-v1-128k": 50,
        "GPT-4o-mini": 90, "Qwen2.5-72B-Instruct": 65,
    }
    speed = tps.get(model.name, 60)
    return round(0.5 + (input_tokens + output_tokens) / speed, 3)


def route(models: list[Model], task: Task, seed: int | None = None,
          fail_rate: float = SIMULATED_FAIL_RATE) -> RoutingDecision:
    """
    对单个任务执行路由决策（完整流程见函数内注释）。

    seed/fail_rate 用于演示限流/超时降级链：每次调用按概率模拟一次 API 失败，
    触发后沿降级链前进一步并记录事件。
    """
    rng = random.Random(seed)
    alpha, beta, gamma = QUALITY_CONFIG[task.quality]["weights"]
    cap_threshold = QUALITY_CONFIG[task.quality]["cap_threshold"]

    # 1. 计算候选池评分与质量分（成本效率仅在“质量达标”集合内比较）
    quality = _est_quality(models, task)
    min_quality = cap_threshold * 100
    scores = route_scores(models, task.task_type, task.input_tokens, task.output_tokens,
                          task.quality, expected_rpm=task.expected_rpm,
                          budget_rmb=task.budget_rmb,
                          quality_scores=quality, min_quality=min_quality)

    # 2. 能力底线筛选（对应「能力不足」异常处理）
    # 质量要求「高/中/低」的阈值(75/60/45)主要作用在成本效率的“质量达标度”软惩罚里；
    # 这里只剔除「明显不达标」（质量 < 阈值×0.6）的模型，其余交由评分函数软性权衡。
    # 图表解读等视觉任务中，不支持视觉输入的模型质量分天然很低，会被底线筛除。
    floor_quality = min_quality * 0.6
    qualified = [m for m in models if quality[m.name] >= floor_quality]

    # 若所有模型均低于能力底线：降级策略 = 选质量分最高的模型（能力不足降级）
    if not qualified:
        best = max(models, key=lambda m: quality[m.name])
        chosen = best
        chosen_score = scores[best.name]["score"]
        chosen_cost = scores[best.name]["cost"]
        chosen_quality = quality[best.name]
        events = [DegradationEvent(
            time=_now(), task_id=task.task_id, task_name=task.name, model=best.name,
            event_type="capability_insufficient",
            reason=f"所有模型质量分均低于能力底线 {floor_quality:.1f} 分，降级选择质量最优模型（{quality[best.name]:.1f} 分）",
            fallback_to=best.name, fallback_est_cost=chosen_cost, fallback_est_quality=chosen_quality)]
    else:
        # 3. 按综合评分排序（首选 = 合格模型内评分最高）
        ranked = sorted(qualified, key=lambda m: scores[m.name]["score"], reverse=True)
        chosen = ranked[0]
        chosen_score = scores[chosen.name]["score"]
        chosen_cost = scores[chosen.name]["cost"]
        chosen_quality = quality[chosen.name]
        events = []

        # 4. 预算检查（对应「预算超限」异常处理）
        if chosen_cost > task.budget_rmb:
            # 在合格模型里找 cost <= 预算 且评分最高者
            within_budget = [m for m in ranked if scores[m.name]["cost"] <= task.budget_rmb]
            if within_budget:
                fallback = within_budget[0]
                events.append(DegradationEvent(
                    time=_now(), task_id=task.task_id, task_name=task.name, model=chosen.name,
                    event_type="budget_overrun",
                    reason=f"首选 {chosen.name} 预估成本 {chosen_cost:.4f} 元超预算 {task.budget_rmb:.2f} 元，降级到预算内评分最高模型",
                    fallback_to=fallback.name,
                    fallback_est_cost=scores[fallback.name]["cost"],
                    fallback_est_quality=quality[fallback.name]))
                chosen = fallback
                chosen_score = scores[fallback.name]["score"]
                chosen_cost = scores[fallback.name]["cost"]
                chosen_quality = quality[fallback.name]

    # 5. 构建降级链（首选 + 次选 + ... + 兜底，每步带成本/质量依据）
    # 关键约束：降级链只包含「能力合格」的模型，避免限流/超时后降级到能力不足的模型；
    # 若合格模型不足 FALLBACK_CHAIN_LEN 个，再用质量最高的不合格模型补齐兜底。
    if qualified:
        ranked_qualified = sorted(qualified, key=lambda m: scores[m.name]["score"], reverse=True)
        chain_models = ranked_qualified[:FALLBACK_CHAIN_LEN]
        if len(chain_models) < FALLBACK_CHAIN_LEN:
            not_qualified = [m for m in models if m not in qualified]
            not_qualified.sort(key=lambda m: quality[m.name], reverse=True)
            chain_models += not_qualified[:FALLBACK_CHAIN_LEN - len(chain_models)]
    else:
        # 所有模型均能力不足：按质量分从高到低作为降级链
        chain_models = sorted(models, key=lambda m: quality[m.name], reverse=True)[:FALLBACK_CHAIN_LEN]

    fallback_chain = []
    for m in chain_models:
        tag = "能力不足兜底" if m not in qualified else "合格"
        fallback_chain.append({
            "model": m.name,
            "score": scores[m.name]["score"],
            "est_cost_rmb": round(scores[m.name]["cost"], 6),
            "est_quality": round(quality[m.name], 2),
            "note": _chain_note(m, scores, quality) + f"（{tag}）",
        })

    # 6. 模拟调用：按概率触发限流/超时 -> 沿降级链前进并记录事件
    latency = _latency_s(chosen, task.input_tokens, task.output_tokens)
    current = chosen
    for step in range(3):   # 最多模拟 3 次连续失败
        if rng.random() >= fail_rate:
            break
        ev_type = rng.choice(["api_rate_limit", "api_timeout"])
        # 在降级链中找当前模型之后的第一个可用模型
        names = [c["model"] for c in fallback_chain]
        try:
            idx = names.index(current.name)
        except ValueError:
            break
        nxt = fallback_chain[idx + 1] if idx + 1 < len(fallback_chain) else None
        if nxt is None:
            break
        events.append(DegradationEvent(
            time=_now(), task_id=task.task_id, task_name=task.name, model=current.name,
            event_type=ev_type,
            reason=("触发 API 限流(RPM 余量不足)" if ev_type == "api_rate_limit"
                    else "触发 API 超时(端到端耗时超过阈值)"),
            fallback_to=nxt["model"], fallback_est_cost=nxt["est_cost_rmb"],
            fallback_est_quality=nxt["est_quality"]))
        current = next(m for m in models if m.name == nxt["model"])
        latency += _latency_s(current, task.input_tokens, task.output_tokens) * 0.3  # 重试开销
        # 若降级后仍超预算，继续沿链走（预算约束始终生效）
        cost_now = scores[current.name]["cost"]
        if cost_now > task.budget_rmb and current.name != chosen.name:
            nxt2 = fallback_chain[names.index(current.name) + 1] if names.index(current.name) + 1 < len(fallback_chain) else None
            if nxt2 and scores[nxt2["model"]]["cost"] <= task.budget_rmb:
                events.append(DegradationEvent(
                    time=_now(), task_id=task.task_id, task_name=task.name, model=current.name,
                    event_type="budget_overrun",
                    reason=f"降级到 {current.name} 后成本仍超预算，继续降级到预算内模型",
                    fallback_to=nxt2["model"], fallback_est_cost=nxt2["est_cost_rmb"],
                    fallback_est_quality=nxt2["est_quality"]))
                current = next(m for m in models if m.name == nxt2["model"])

    # 最终选用：若发生过降级且最后一个降级目标存在，则以其为最终模型
    final_model = current

    return RoutingDecision(
        task=task,
        chosen_model=final_model.name,
        score=round(scores[final_model.name]["score"], 4),
        est_cost=round(scores[final_model.name]["cost"], 6),
        est_quality=round(quality[final_model.name], 2),
        fallback_chain=fallback_chain,
        events=events,
        total_latency_s=round(latency, 3),
    )


def _chain_note(m: Model, scores: dict, quality: dict) -> str:
    """降级链中每个候选的依据说明（成本与质量依据，满足题目要求）。"""
    return (f"评分{scores[m.name]['score']:.3f}，成本{scores[m.name]['cost']:.4f}元，"
            f"质量{quality[m.name]:.1f}分")


def save_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    # 简单自测：加载模型并跑一个示例任务
    models = load_models()
    demo = Task(task_id="demo-1", name="示例：论文精读摘要", task_type="论文精读",
                desc="精读一篇 30 页论文并生成摘要", input_tokens=8000, output_tokens=1500,
                quality="高", budget_rmb=0.5)
    dec = route(models, demo, seed=42)
    print("示例任务路由结果:")
    print("  选用模型:", dec.chosen_model)
    print("  预估成本(元):", dec.est_cost)
    print("  预估质量分:", dec.est_quality)
    print("  降级链:", [c["model"] for c in dec.fallback_chain])
    print("  触发事件数:", len(dec.events))


# ============================================================================
# 附加分：动态路由调整（+3 分）
# 题目：第四题的路由在任务执行过程中根据中间结果质量动态调整后续模型选择。
# 思路：把任务拆成多步执行，每步执行后评估“中间结果质量”（观测质量 = 理论质量 + 随机噪声，
#       模拟真实调用中的质量波动）；若当前模型实际表现明显低于预期，则动态切换到
#       质量更高且成本在预算内的模型继续后续步骤；否则保持当前模型。
# ============================================================================

@dataclass
class DynamicStep:
    """动态路由中的一步：执行哪个模型、观测到多少质量、是否发生切换。"""
    step: int
    model: str
    quality_theory: float   # 该模型的任务理论质量（0~100）
    quality_obs: float      # 该步观测到的中间结果质量（0~100）
    cost: float             # 该步成本（RMB）
    switched: bool          # 本步是否发生动态切换
    reason: str             # 切换原因（未切换为空串）


@dataclass
class DynamicRoutingDecision:
    """一次动态路由决策的完整结果。"""
    task: Task
    steps: list[DynamicStep]
    chosen_model: str       # 最终使用的模型
    total_cost: float
    avg_quality: float      # 各步观测质量的平均（任务完成质量的动态口径）
    final_quality: float    # 最终模型的理论质量
    switch_count: int       # 动态切换次数
    events: list            # 切换事件（供日志与轨迹图）

    def to_dict(self) -> dict:
        return asdict(self)


def route_dynamic(models: list[Model], task: Task, seed: int | None = None,
                  n_steps: int = 4, noise_std: float = 3.0,
                  switch_threshold: float = 5.0, min_improvement: float = 2.0,
                  budget_override: float | None = None) -> DynamicRoutingDecision:
    """
    动态路由调整（附加分 +3）。

    参数：
        n_steps            - 任务被拆分的执行步数（模拟长任务分块处理）
        noise_std          - 中间结果质量观测的随机波动标准差
        switch_threshold   - 切换触发阈值：观测质量 < 理论质量 - 阈值 才考虑切换
        min_improvement    - 目标模型质量至少比当前理论质量高多少才切换（避免无意义切换）
        budget_override    - 预算覆盖（默认用 task.budget_rmb）
    """
    rng = random.Random(seed)
    budget = budget_override if budget_override is not None else task.budget_rmb

    # 复用静态路由的评分与质量计算（成本效率仅在质量达标集合内比较）
    quality = _est_quality(models, task)
    min_quality = QUALITY_CONFIG[task.quality]["cap_threshold"] * 100
    scores = route_scores(models, task.task_type, task.input_tokens, task.output_tokens,
                          task.quality, expected_rpm=task.expected_rpm,
                          budget_rmb=budget,
                          quality_scores=quality, min_quality=min_quality)

    # 候选优先级：合格模型按综合评分降序（能力不足时按质量降序）
    ranked = sorted([m for m in models if quality[m.name] >= min_quality],
                    key=lambda m: scores[m.name]["score"], reverse=True)
    if not ranked:
        ranked = sorted(models, key=lambda m: quality[m.name], reverse=True)

    current = ranked[0]                     # 初始模型 = 静态路由首选
    # 每步 token 量 = 总 token 均分到 n_steps（简化分块假设）
    step_in, step_out = task.input_tokens / n_steps, task.output_tokens / n_steps

    steps: list[DynamicStep] = []
    events = []
    for step in range(1, n_steps + 1):
        q_theory = quality[current.name]
        # 观测质量 = 理论质量 + 随机噪声（模拟真实调用的质量波动），截断到 [0,100]
        q_obs = min(100.0, max(0.0, q_theory + rng.gauss(0, noise_std)))
        cost_step = scores[current.name]["cost"] / n_steps   # 本步成本 = 总成本均分

        switched, reason = False, ""
        # 动态调整判断：实际表现明显低于预期 -> 尝试切换
        if q_obs < q_theory - switch_threshold:
            # 目标：质量更高、成本在预算内、且非当前模型（ranked 已按评分降序）
            candidates = [m for m in ranked
                          if quality[m.name] > q_theory + min_improvement
                          and scores[m.name]["cost"] <= budget
                          and m.name != current.name]
            if candidates:
                nxt = candidates[0]
                reason = (f"中间质量 {q_obs:.1f} 低于预期 {q_theory:.1f}，"
                          f"切换至 {nxt.name}（质量 {quality[nxt.name]:.1f}）")
                events.append({"step": step, "from": current.name, "to": nxt.name,
                               "from_quality": round(q_theory, 2),
                               "obs_quality": round(q_obs, 2),
                               "to_quality": round(quality[nxt.name], 2)})
                current = nxt
                switched = True

        steps.append(DynamicStep(step=step, model=current.name,
                                 quality_theory=round(q_theory, 2),
                                 quality_obs=round(q_obs, 2),
                                 cost=round(cost_step, 6),
                                 switched=switched, reason=reason))

    total_cost = sum(s.cost for s in steps)
    avg_quality = sum(s.quality_obs for s in steps) / len(steps)
    return DynamicRoutingDecision(
        task=task, steps=steps, chosen_model=current.name,
        total_cost=round(total_cost, 6), avg_quality=round(avg_quality, 2),
        final_quality=round(quality[current.name], 2),
        switch_count=sum(1 for s in steps if s.switched), events=events)
