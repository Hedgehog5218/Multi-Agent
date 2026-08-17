# -*- coding: utf-8 -*-
"""
scoring.py —— 问题四 4.3「路由策略设计」评分函数模块

综合路由评分函数（数学表达）：
    Score(m, t) = α · NormCap(m, t) + β · NormCost(m, t) + γ · NormAvail(m)

其中：
    NormCap(m, t)   —— 能力匹配度：模型 m 在任务 t 相关 benchmark 上的加权得分（池内 min-max 归一化到 [0,1]）
    NormCost(m, t)   —— 成本效率：优先按「相对预算」口径 1 - cost/budget（预算内成本越低越高，预算外为 0）
    NormAvail(m)     —— 可用性：0.6 · RPM余量 + 0.4 · 历史成功率

权重设计依据：
    质量要求「高」：能力优先（α=0.60），成本/可用性次要
    质量要求「中」：能力与成本均衡（α=0.45, β=0.35）
    质量要求「低」：成本优先（β=0.50），能力仅作门槛

归一化方式：
    能力分与成本均在候选模型池内做 min-max 归一化；某模型缺少某 benchmark 公开数据时
    该维度记为 0（视为能力未知，路由时天然靠后），符合题目「无公开数据则标注」的口径。
"""
from __future__ import annotations
import math
from pathlib import Path
from dataclasses import dataclass

import pandas as pd

DATA_PATH = Path(__file__).resolve().parent / "data" / "data.csv"

# 任务类型 -> 相关 benchmark 及权重（对应 4.1 场景的 8 类科研任务）
# 说明：文献检索/论文精读依赖长文本与通用推理；代码复现看 HumanEval；
#       数据分析/公式推导看 MATH；学术写作/审稿回复看 IFEval；
#       图表解读需要视觉输入能力（Vision 列），并用 MMLU-Pro 兜底。
TASK_BENCHMARKS: dict[str, dict[str, float]] = {
    "文献检索": {"MMLU_Pro": 0.40, "Long_Context_NIAH": 0.60},
    "论文精读": {"MMLU_Pro": 0.60, "Long_Context_NIAH": 0.40},
    "代码复现": {"HumanEval": 1.00},
    "数据分析": {"MATH": 0.50, "MMLU_Pro": 0.50},
    "公式推导": {"MATH": 1.00},
    "学术写作": {"IFEval": 0.70, "MMLU_Pro": 0.30},
    "图表解读": {"MMLU_Pro": 0.50, "Vision": 0.50},
    "审稿回复": {"IFEval": 1.00},
}

# 质量要求等级 -> (能力门槛阈值, 评分权重 (α, β, γ))
# 权重设计：高质量任务能力优先（α 最大），低质量任务成本优先（β 最大）；
# 相比初版提高 α、降低 β，避免“成本效率”过度主导、路由结果过度集中于最便宜模型，
# 让强模型在高质量任务中真正胜出（对应 4.3 能力匹配度为核心的设计意图）。
QUALITY_CONFIG: dict[str, dict] = {
    "高": {"cap_threshold": 0.75, "weights": (0.70, 0.15, 0.15)},
    "中": {"cap_threshold": 0.60, "weights": (0.55, 0.25, 0.20)},
    "低": {"cap_threshold": 0.45, "weights": (0.35, 0.45, 0.20)},
}

# 可用性默认参数
DEFAULT_EXPECTED_RPM = 10.0      # 单任务预估每分钟请求数（多并发下会更高）
DEFAULT_SUCCESS_RATE = 0.97      # 历史成功率默认值（真实部署可由监控数据回填）


def _to_float(v) -> float | None:
    """将 CSV 单元格转为 float；N/A 或空值返回 None。"""
    if v is None:
        return None
    s = str(v).strip()
    if s == "" or s.upper().startswith("N/A"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _parse_context(v) -> int:
    """从 '131072 (128K)' 这类文本中提取上下文窗口数字。"""
    s = str(v).strip()
    for part in s.replace(",", "").split():
        if part.replace(".", "").isdigit():
            return int(float(part))
    return 128000


def _support_vision(v) -> bool:
    """根据视觉定价规则判断模型是否支持视觉输入。"""
    s = str(v).strip()
    return not s.upper().startswith("N/A")


@dataclass
class Model:
    """单个模型的能力-成本档案（由 data.csv 生成）。"""
    name: str
    model_type: str
    price_in: float            # RMB / 1M input tokens
    price_out: float           # RMB / 1M output tokens
    rpm: float
    tpm: float
    context: int
    vision: bool
    structured: str
    bench: dict                # benchmark 得分（N/A 为 None）
    source: str
    url: str

    def __repr__(self):
        return f"<Model {self.name}>"


def load_models(path: str | Path = DATA_PATH) -> list[Model]:
    """读取 data.csv 并清洗为模型对象列表。"""
    df = pd.read_csv(path, encoding="utf-8-sig")
    models = []
    for _, r in df.iterrows():
        models.append(Model(
            name=str(r["Model_Name"]),
            model_type=str(r["Model_Type"]),
            price_in=float(r["Input_Price_RMB"]),
            price_out=float(r["Output_Price_RMB"]),
            rpm=float(r["RPM_Limit"]),
            tpm=float(r["TPM_Limit"]),
            context=_parse_context(r["Context_Window"]),
            vision=_support_vision(r["Vision_Pricing_Rule"]),
            structured=str(r["Structured_Output"]),
            bench={
                "MMLU": _to_float(r["MMLU"]),
                "MMLU_Pro": _to_float(r["MMLU_Pro"]),
                "HumanEval": _to_float(r["HumanEval"]),
                "IFEval": _to_float(r["IFEval"]),
                "MATH": _to_float(r["MATH"]),
                "Long_Context_NIAH": _to_float(r["Long_Context_NIAH"]),
                "Science_PubMedQA": _to_float(r["Science_PubMedQA"]),
                # Vision 采用 0~100 量纲：支持视觉输入=100，不支持=None（与其它 benchmark 一致）
                "Vision": 100.0 if _support_vision(r["Vision_Pricing_Rule"]) else None,
            },
            source=str(r["Data_Source"]),
            url=str(r["Data_Source_URL"]),
        ))
    return models


def _minmax(values: list[float | None]) -> list[float]:
    """池内 min-max 归一化；N/A 一律记 0（相对排名底线）。
    注意：即便池内同分（如 NIAH 全 100），N/A 也仍记 0，不能与有值项同分。"""
    valid = [v for v in values if v is not None and not math.isnan(v)]
    if not valid:
        return [0.5] * len(values)
    lo, hi = min(valid), max(valid)
    if hi == lo:
        # 池内同分：有值项给 0.5，N/A 项仍记 0
        return [0.5 if (v is not None and not math.isnan(v)) else 0.0 for v in values]
    out = []
    for v in values:
        if v is None or math.isnan(v):
            out.append(0.0)          # 无公开数据 -> 能力未知，记 0（保守底线）
        else:
            out.append((v - lo) / (hi - lo))
    return out


def capability_scores(models: list[Model], task_type: str) -> dict[str, float]:
    """计算候选模型池内，各模型在任务类型 task_type 上的归一化能力匹配度。"""
    bench_weights = TASK_BENCHMARKS[task_type]
    scores: dict[str, float] = {}
    for bench_name, weight in bench_weights.items():
        raw = [m.bench.get(bench_name) for m in models]
        norm = _minmax(raw)
        for m, n in zip(models, norm):
            scores[m.name] = scores.get(m.name, 0.0) + weight * n
    return scores


def estimate_cost(model: Model, input_tokens: int, output_tokens: int) -> float:
    """预估单次调用成本（RMB）：输入/输出分别按每 1M tokens 计价。"""
    return (model.price_in * input_tokens + model.price_out * output_tokens) / 1_000_000


def availability(model: Model, expected_rpm: float = DEFAULT_EXPECTED_RPM,
                 success_rate: float = DEFAULT_SUCCESS_RATE) -> float:
    """可用性 = 0.6·RPM余量 + 0.4·历史成功率（两者均已在 [0,1]）。"""
    rpm_margin = max(0.0, min(1.0, 1.0 - expected_rpm / model.rpm)) if model.rpm > 0 else 0.0
    return 0.6 * rpm_margin + 0.4 * success_rate


def route_scores(models: list[Model], task_type: str, input_tokens: int, output_tokens: int,
                 quality: str, expected_rpm: float = DEFAULT_EXPECTED_RPM,
                 success_rate: float = DEFAULT_SUCCESS_RATE,
                 budget_rmb: float | None = None,
                 quality_scores: dict[str, float] | None = None,
                 min_quality: float = 0.0) -> dict[str, dict]:
    """
    计算候选模型池内每个模型的完整评分（综合路由评分函数）。

    返回 {模型名: {"cap":..., "cost":..., "cost_eff":..., "avail":..., "score":...}}
    """
    alpha, beta, gamma = QUALITY_CONFIG[quality]["weights"]

    cap_map = capability_scores(models, task_type)
    costs = {m.name: estimate_cost(m, input_tokens, output_tokens) for m in models}

    # 成本效率（满足质量要求下的最低成本，FrugalGPT 质量约束的软惩罚版本）：
    #   CostEff(m) = 质量达标度(m) × 成本占用(m)
    #   质量达标度 = clamp(质量分 / 质量阈值, 0, 1)  —— 质量低于要求时按比例打折（软约束，
    #   质量低的模型仍可用但不划算，避免“一刀切”）；质量达标后达标度为 1。
    #   成本占用   = max(0, 1 - cost/预算)          —— 预算内成本越低效率越高，预算外为 0。
    #   未给质量/预算参数时退回简单口径（成本 min-max 归一化）。
    cost_eff = {}
    for name, c in costs.items():
        if quality_scores is not None and min_quality > 0:
            q_fit = min(1.0, max(0.0, quality_scores.get(name, 0.0) / min_quality))
        else:
            q_fit = 1.0
        if budget_rmb is not None and budget_rmb > 0:
            cost_eff[name] = q_fit * max(0.0, 1.0 - c / budget_rmb)
        else:
            cost_vals = [costs[n] for n, _ in costs.items()]
            lo, hi = min(cost_vals), max(cost_vals)
            c_norm = 1.0 if hi == lo else 1.0 - (c - lo) / (hi - lo)
            cost_eff[name] = q_fit * c_norm

    avails = {m.name: availability(m, expected_rpm, success_rate) for m in models}

    result = {}
    for m in models:
        cap = cap_map[m.name]
        ce = cost_eff[m.name]
        av = avails[m.name]
        score = alpha * cap + beta * ce + gamma * av
        result[m.name] = {
            "cap": round(cap, 4),
            "cost": round(costs[m.name], 6),
            "cost_eff": round(ce, 4),
            "avail": round(av, 4),
            "score": round(score, 4),
        }
    return result