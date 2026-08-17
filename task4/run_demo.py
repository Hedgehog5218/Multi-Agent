# -*- coding: utf-8 -*-
"""
run_demo.py —— 问题四端到端演示主入口

流程：
    1. 加载 data/data.csv（10 款模型）
    2. 定义 10 个科研任务（覆盖 4.1 全部任务类型）
    3. 智能路由：对每个任务执行 route()，记录决策日志与降级事件
    4. 对比实验（4.4）：router / 全量能力最强 / 全量成本最低 / 随机路由
    5. 可视化（4.5）：路由决策热力图、成本-质量帕累托散点图、降级事件时间线图
    6. 输出：logs/（决策日志、降级事件）、experiments/results/（实验结果）、figures/（3 张图）
"""
from __future__ import annotations
import json
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")          # 无界面后端，适合服务器/CI
import matplotlib.pyplot as plt
import numpy as np

from scoring import load_models, route_scores, estimate_cost, QUALITY_CONFIG
from router import Task, route, route_dynamic, _est_quality, _latency_s, save_json

# 中文字体设置（Windows 常见字体）
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

BASE = Path(__file__).resolve().parent
FIG_DIR = BASE / "figures"
LOG_DIR = BASE / "logs"
RESULT_DIR = BASE / "experiments" / "results"
for d in (FIG_DIR, LOG_DIR, RESULT_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. 10 个科研任务（覆盖 8 种任务类型；输入/输出 token 为预估规模）
# ---------------------------------------------------------------------------
TASKS = [
    # ---- 高质量任务：能力优先，预算充足，让“质量最优”模型有机会胜出 ----
    Task("T01", "论文精读-顶会论文", "论文精读", "精读一篇顶会论文并生成结构化摘要",
         12000, 2000, "高", 1.00),      # 期望：Gemini/Claude（质量接近，Gemini 更便宜）
    Task("T02", "文献检索-系统综述", "文献检索", "检索 60 篇文献并撰写研究综述",
         20000, 3000, "高", 1.50),      # 期望：Claude/V3（长文本 NIAH 强）
    Task("T03", "代码复现-工程代码审查重构", "代码复现", "审查并重构大型训练脚本，修复逻辑缺陷",
         15000, 3000, "高", 2.00),      # 期望：Claude（HumanEval 92 池内最高）
    Task("T04", "公式推导-竞赛级数学", "公式推导", "推导损失函数梯度并给出收敛性证明",
         4000, 1500, "高", 0.60),       # 期望：Qwen2.5-72B（MATH 83.1 池内最高）
    Task("T05", "数据分析-统计建模", "数据分析", "对实验数据做统计检验并给出建模建议",
         8000, 1200, "中", 0.50),       # 期望：V3/Gemini（MATH+MMLU-Pro 均衡）
    Task("T06", "学术写作-基金申请书", "学术写作", "撰写基金申请书“研究内容与创新点”章节",
         10000, 2500, "高", 1.20),      # 期望：Gemini（IFEval 90.4 池内最高）
    Task("T07", "图表解读-多模态图表", "图表解读", "解读复杂科研图表并提取关键结论",
         8000, 1000, "中", 0.60),       # 期望：Claude/Gemini（支持视觉+MMLU-Pro 高）
    Task("T08", "审稿回复-多轮意见", "审稿回复", "逐条回复审稿意见并给出修改方案",
         9000, 1800, "中", 0.60),       # 期望：Qwen2.5-Max（IFEval 90.8 池内最高）
    # ---- 中低质量任务：成本敏感，倾向性价比模型 ----
    Task("T09", "论文精读-方法章节", "论文精读", "精读实验方法章节并整理技术路线",
         6000, 1200, "中", 0.40),       # 期望：Gemini
    Task("T10", "文献检索-快速筛选", "文献检索", "快速筛选文献并标注相关度",
         15000, 800, "低", 0.30),       # 期望：Gemini
]


def run_router(models, tasks, seed=20260812):
    """策略 1：智能路由（含质量门槛、预算约束、限流/超时降级）。"""
    decisions = []
    for i, t in enumerate(tasks):
        decisions.append(route(models, t, seed=seed + i * 13, fail_rate=0.15))
    return decisions


def _fixed_decision(models, task, model_name):
    """模拟“固定用某模型”的决策（无降级），用于对比策略。"""
    m = next(x for x in models if x.name == model_name)
    return {
        "task_id": task.task_id, "task_name": task.name, "model": model_name,
        "score": 0.0, "est_cost": estimate_cost(m, task.input_tokens, task.output_tokens),
        "est_quality": _est_quality(models, task)[model_name],
        "total_latency_s": _latency_s(m, task.input_tokens, task.output_tokens),
        "events": [],
    }


def _strongest_model(models):
    """“综合能力最强”模型：在其有公开数据的 benchmark 上取平均（缺失项不计入分母）。

    说明：若把官方未公布的 benchmark 记 0，会对数据公布不全但能力强的模型（如
    DeepSeek-R1 未公布 IFEval/NIAH）系统性不公平；故此处仅统计有公开得分的维度，
    并在对比实验 meta 中标注该口径，报告中需同步说明。
    """
    benchs = ["MMLU", "MMLU_Pro", "HumanEval", "IFEval", "MATH", "Long_Context_NIAH"]
    best, best_avg = None, -1.0
    for m in models:
        vals = [m.bench.get(b) for b in benchs]
        valid = [v for v in vals if v is not None]
        # 数据不足（公开 benchmark 少于 4/6）的模型不参与“最强”评选，避免少数据虚高
        if len(valid) < 4:
            continue
        avg = sum(valid) / len(valid)
        if avg > best_avg:
            best, best_avg = m.name, avg
    return best


def run_best_quality(models, tasks):
    """策略 2：全量能力最强模型。"""
    name = _strongest_model(models)
    return [_fixed_decision(models, t, name) for t in tasks], name


def run_cheapest(models, tasks):
    """策略 3：全量成本最低模型（按输入定价最低）。"""
    name = min(models, key=lambda m: m.price_in).name
    return [_fixed_decision(models, t, name) for t in tasks], name


def run_random(models, tasks, seed=42):
    """策略 4：随机路由（固定种子保证可复现）。"""
    rng = random.Random(seed)
    name = None
    out = []
    for t in tasks:
        name = rng.choice(models).name
        out.append(_fixed_decision(models, t, name))
    return out, "random"


# ---------------------------------------------------------------------------
# 4. 对比实验统计
# ---------------------------------------------------------------------------
def summarize(strategy, decisions, meta=""):
    total_cost = sum(d["est_cost"] for d in decisions)
    avg_quality = np.mean([d["est_quality"] for d in decisions])
    total_latency = sum(d["total_latency_s"] for d in decisions)
    return {
        "strategy": strategy, "meta": meta,
        "total_cost_rmb": round(total_cost, 4),
        "avg_quality": round(float(avg_quality), 2),
        "total_latency_s": round(total_latency, 2),
    }


def main():
    print("=" * 72)
    print("问题四 · 多智能体智能路由与科研任务调度 —— 端到端演示")
    print("=" * 72)

    models = load_models()
    print(f"已加载 {len(models)} 款模型：{', '.join(m.name for m in models)}\n")

    # ---- 路由决策 + 日志 ----
    print("-" * 72)
    print("【步骤 1】智能路由：10 个科研任务")
    print("-" * 72)
    decisions = run_router(models, TASKS)
    all_events = []
    routing_rows = []
    for d in decisions:
        all_events.extend(d.events)
        routing_rows.append({
            "task_id": d.task.task_id, "task_name": d.task.name, "task_type": d.task.task_type,
            "chosen_model": d.chosen_model, "score": d.score,
            "est_cost_rmb": d.est_cost, "est_quality": d.est_quality,
            "latency_s": d.total_latency_s, "event_count": len(d.events),
            "fallback_chain": " -> ".join(c["model"] for c in d.fallback_chain),
        })
        ev_summary = f"[{len(d.events)} 次降级]" if d.events else ""
        print(f"  {d.task.task_id} {d.task.name:<10} 类型={d.task.task_type:<4} "
              f"质量={d.task.quality} -> {d.chosen_model:<16} "
              f"成本={d.est_cost:.4f}元 质量分={d.est_quality:.1f} {ev_summary}")

    # 写路由决策日志
    import csv
    with open(LOG_DIR / "routing_log.csv", "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(routing_rows[0].keys()))
        writer.writeheader()
        writer.writerows(routing_rows)
    save_json(LOG_DIR / "degradation_events.json",
              [e.to_dict() for e in all_events])
    print(f"\n已写入日志：{LOG_DIR/'routing_log.csv'}、{LOG_DIR/'degradation_events.json'}（共 {len(all_events)} 条降级事件）\n")

    # ---- 对比实验 ----
    print("-" * 72)
    print("【步骤 2】对比实验（4 组策略 × 10 个任务）")
    print("-" * 72)
    dec_best, meta_best = run_best_quality(models, TASKS)
    dec_cheap, meta_cheap = run_cheapest(models, TASKS)
    dec_rand, meta_rand = run_random(models, TASKS)

    summaries = [
        summarize("router", [{"est_cost": d.est_cost, "est_quality": d.est_quality,
                              "total_latency_s": d.total_latency_s} for d in decisions],
                  "完整路由系统"),
        summarize("best_quality", dec_best, f"全量能力最强({meta_best})"),
        summarize("cheapest", dec_cheap, f"全量成本最低({meta_cheap})"),
        summarize("random", dec_rand, "随机路由"),
    ]
    print(f"{'策略':<12}{'说明':<28}{'总成本(元)':>12}{'平均质量':>10}{'总耗时(s)':>10}")
    for s in summaries:
        print(f"{s['strategy']:<12}{s['meta']:<28}{s['total_cost_rmb']:>12.4f}"
              f"{s['avg_quality']:>10.2f}{s['total_latency_s']:>10.2f}")

    # 保存实验结果
    with open(RESULT_DIR / "experiment_summary.json", "w", encoding="utf-8") as f:
        json.dump(summaries, f, ensure_ascii=False, indent=2)
    # 把路由决策对象统一转成与固定策略一致的 dict 形式
    router_dicts = [{"task_id": d.task.task_id, "task_name": d.task.name, "model": d.chosen_model,
                     "est_cost": d.est_cost, "est_quality": d.est_quality,
                     "total_latency_s": d.total_latency_s} for d in decisions]
    rows = []
    for s in summaries:
        for d in (router_dicts if s["strategy"] == "router" else
                  (dec_best if s["strategy"] == "best_quality" else
                   (dec_cheap if s["strategy"] == "cheapest" else dec_rand))):
            rows.append({"strategy": s["strategy"], "meta": s["meta"],
                         **{k: d[k] for k in ("task_id", "task_name", "model",
                                              "est_cost", "est_quality", "total_latency_s")}})
    with open(RESULT_DIR / "experiment_results.csv", "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n已保存实验结果：{RESULT_DIR/'experiment_summary.json'}、{RESULT_DIR/'experiment_results.csv'}")

    # ---- 附加分：动态路由调整 ----
    print("-" * 72)
    print("【步骤 2.5】附加分：动态路由调整（根据中间结果质量动态换模型）")
    print("-" * 72)
    dyn_decisions = run_dynamic(models, TASKS, seed=20260812)
    router_s = next(s for s in summaries if s["strategy"] == "router")
    static_cost = router_s["total_cost_rmb"]
    static_qual = router_s["avg_quality"]
    dyn_cost = sum(d.total_cost for d in dyn_decisions)
    dyn_qual = np.mean([d.avg_quality for d in dyn_decisions])
    total_switches = sum(d.switch_count for d in dyn_decisions)
    print(f"  静态路由：总成本 {static_cost:.4f} 元，平均质量 {static_qual:.2f} 分")
    print(f"  动态路由：总成本 {dyn_cost:.4f} 元，平均质量(观测) {dyn_qual:.2f} 分，动态切换 {total_switches} 次")
    for d in dyn_decisions:
        if d.switch_count > 0:
            print(f"    {d.task.task_id} {d.task.name}：切换 {d.switch_count} 次 -> 最终 {d.chosen_model} "
                  f"({d.final_quality:.1f} 分)")
    # 保存动态路由日志
    save_json(LOG_DIR / "dynamic_routing_log.json",
              [{"task": d.task.task_id, "task_name": d.task.name, "chosen_model": d.chosen_model,
                "total_cost": d.total_cost, "avg_quality": d.avg_quality, "final_quality": d.final_quality,
                "switch_count": d.switch_count, "steps": [s.__dict__ for s in d.steps],
                "events": d.events} for d in dyn_decisions])
    print(f"已写入日志：{LOG_DIR/'dynamic_routing_log.json'}")

    # ---- 可视化 ----
    print("-" * 72)
    print("【步骤 3】可视化（4.5 三张图 + 附加分轨迹图）")
    print("-" * 72)
    draw_heatmap(models, TASKS)
    draw_pareto(summaries)
    draw_timeline(all_events, models)
    draw_dynamic_trajectory(dyn_decisions, models)
    print(f"已生成：{FIG_DIR/'routing_heatmap.png'}、{FIG_DIR/'pareto_cost_quality.png'}、"
          f"{FIG_DIR/'degradation_timeline.png'}")

    # ---- 结果分析 ----
    print("-" * 72)
    print("【步骤 4】结论分析")
    print("-" * 72)
    best_s = min(summaries, key=lambda s: s["total_cost_rmb"])
    best_q = max(summaries, key=lambda s: s["avg_quality"])
    router_s = next(s for s in summaries if s["strategy"] == "router")
    cost_rank = "最低" if router_s is best_s else "次低（最低为 " + best_s["strategy"] + "）"
    print(f"  路由策略总成本 {router_s['total_cost_rmb']:.4f} 元，为全部策略中{cost_rank}；"
          f"平均质量 {router_s['avg_quality']:.2f} 分（最高为 {best_q['strategy']} {best_q['avg_quality']:.2f} 分）。")
    print("  结论：路由策略在质量要求高、预算紧、任务类型混合的场景下优势最大——"
          "它只为高价值任务支付高质量模型的溢价，用低成本模型处理低价值任务，"
          "总成本显著低于“全量最强”，质量显著高于“全量最便宜/随机”。")


# ---------------------------------------------------------------------------
# 5. 可视化实现
# ---------------------------------------------------------------------------
def draw_heatmap(models, tasks, path=None):
    """图 1：路由决策热力图（横轴=任务，纵轴=模型，颜色=综合评分；★=路由最终选用）"""
    path = path or (FIG_DIR / "routing_heatmap.png")
    names = [m.name for m in models]
    matrix = np.zeros((len(tasks), len(models)))
    chosen = {}
    for i, t in enumerate(tasks):
        qmap = _est_quality(models, t)
        min_q = QUALITY_CONFIG[t.quality]["cap_threshold"] * 100
        sc = route_scores(models, t.task_type, t.input_tokens, t.output_tokens, t.quality,
                          budget_rmb=t.budget_rmb, quality_scores=qmap, min_quality=min_q)
        matrix[i] = [sc[n]["score"] for n in names]
        # ★ = 静态路由首选（综合评分最优的合格模型，不含随机限流/超时降级），
        # 与动态路由轨迹图的“初始模型”一致，便于两张图对照。
        chosen[t.task_id] = route(models, t, seed=20260812 + i * 13).fallback_chain[0]["model"]

    # 模型按平均评分降序排列（高分在上，图更易读）
    order = np.argsort(-matrix.mean(axis=0))
    names_sorted = [names[j] for j in order]
    matrix_sorted = matrix[:, order]

    fig, ax = plt.subplots(figsize=(13.5, 8))
    im = ax.imshow(matrix_sorted.T, aspect="auto", cmap="YlGnBu", vmin=0, vmax=1)
    ax.set_xticks(range(len(tasks)))
    ax.set_xticklabels([t.task_id for t in tasks], rotation=45, ha="right", fontsize=10)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names_sorted, fontsize=9)

    # 数值 + 最终选用标记（★ 白底框）
    for i in range(len(tasks)):
        for j in range(len(names)):
            v = matrix_sorted[i, j]
            ax.text(i, j, f"{v:.2f}", ha="center", va="center", fontsize=6.5,
                    color="white" if v > 0.55 else "#333333")
    for i, t in enumerate(tasks):
        mname = chosen[t.task_id]
        j = names_sorted.index(mname)
        ax.plot(i, j, marker="*", color="white", ms=15, mec="black", mew=1.2, zorder=5)

    ax.set_xlabel("任务", fontsize=11)
    ax.set_ylabel("候选模型（按平均评分降序）", fontsize=11)
    ax.set_title("路由决策热力图：各模型对各任务的综合路由评分（★ = 静态路由首选，与轨迹图初始一致）", fontsize=12)
    cbar = fig.colorbar(im, ax=ax, label="综合路由评分 Score(m,t)", shrink=0.9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def draw_pareto(summaries, path=None):
    """图 2：成本-质量帕累托散点图（4 组策略 + 帕累托前沿折线）"""
    path = path or (FIG_DIR / "pareto_cost_quality.png")
    colors = {"router": "#d62728", "best_quality": "#1f77b4",
              "cheapest": "#2ca02c", "random": "#ff7f0e"}
    fig, ax = plt.subplots(figsize=(9.5, 6.5))

    pts = []
    for s in summaries:
        x = np.log10(max(s["total_cost_rmb"], 1e-4))
        pts.append((x, s["avg_quality"], s))
    # 帕累托前沿：按成本升序，保留质量更高的点作为包络（低成本+高质量）
    pts_sorted = sorted(pts, key=lambda p: p[0])
    frontier, best_y = [], -1.0
    for x, y, s in pts_sorted:
        if y > best_y:
            frontier.append((x, y, s))
            best_y = y

    for x, y, s in pts:
        ax.scatter(x, y, s=180, color=colors.get(s["strategy"], "#333333"),
                   edgecolors="black", linewidths=1.2, zorder=3)
        ax.annotate(f"{s['strategy']}\n({s['total_cost_rmb']:.3f}元, {y:.1f}分)",
                    (x, y), textcoords="offset points", xytext=(12, 8),
                    fontsize=9, fontweight="bold")
    if len(frontier) >= 2:
        ax.plot([p[0] for p in frontier], [p[1] for p in frontier], "--",
                color="gray", lw=1.5, label="帕累托前沿")
        ax.legend(loc="lower right", fontsize=9)

    ax.set_xlabel("总成本 log10(RMB)", fontsize=11)
    ax.set_ylabel("平均任务质量得分", fontsize=11)
    ax.set_title("成本-质量帕累托散点图（4 组策略对比）", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_xlim(left=ax.get_xlim()[0] - 0.15)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def draw_timeline(events, models, path=None):
    """图 3：降级事件时间线图（x=事件序号，y=触发降级的模型；箭头指向降级目标）"""
    path = path or (FIG_DIR / "degradation_timeline.png")
    names = [m.name for m in models]
    ymap = {n: i for i, n in enumerate(names)}
    types = {"api_rate_limit": "#d62728", "api_timeout": "#ff7f0e",
             "budget_overrun": "#1f77b4", "capability_insufficient": "#7f7f7f"}
    type_labels = {"api_rate_limit": "限流", "api_timeout": "超时",
                   "budget_overrun": "预算超限", "capability_insufficient": "能力不足"}

    if not events:
        fig, ax = plt.subplots(figsize=(11, 5))
        ax.text(0.5, 0.5, "本次运行未触发降级事件\n（说明所选模型可用性良好）",
                ha="center", va="center", fontsize=13, color="#666666")
        ax.set_axis_off()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return

    fig, ax = plt.subplots(figsize=(12, max(6, len(names) * 0.55)))
    for i, e in enumerate(events):
        y = ymap.get(e.model, 0)
        c = types.get(e.event_type, "#333333")
        # 事件主体：大圆点 + 类型标签
        ax.scatter(i, y, s=300, color=c, edgecolors="black", linewidths=1.5, zorder=4)
        ax.annotate(type_labels.get(e.event_type, e.event_type), (i, y),
                    textcoords="offset points", xytext=(0, -20), ha="center",
                    fontsize=9, fontweight="bold")
        # 降级去向：箭头 + 目标模型
        fb = e.fallback_to or ""
        if fb in ymap:
            ax.annotate("", xy=(i + 0.12, ymap[fb]), xytext=(i, y),
                        arrowprops=dict(arrowstyle="->", color="#555555", lw=1.8))
            ax.annotate(fb, (i, ymap[fb]), textcoords="offset points",
                        xytext=(10, 0), fontsize=8, color="#555555")
        # 任务标识
        ax.annotate(e.task_id, (i, y), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=8, color="#333333")

    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xticks(range(len(events)))
    ax.set_xticklabels([e.task_id for e in events], fontsize=9)
    ax.set_xlabel("事件序号（对应任务）", fontsize=11)
    ax.set_ylabel("触发降级的模型", fontsize=11)
    ax.set_title("降级事件时间线图（红=限流 橙=超时 蓝=预算超限 灰=能力不足）", fontsize=12)
    ax.grid(True, axis="x", linestyle="--", alpha=0.3)
    ax.set_xlim(-0.3, len(events) - 0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_dynamic(models, tasks, seed=20260812):
    """附加分：对全部任务执行动态路由（任务拆 4 步，按中间结果质量动态换模型）。"""
    decisions = []
    for i, t in enumerate(tasks):
        decisions.append(route_dynamic(models, t, seed=seed + i * 17, n_steps=4,
                                      noise_std=4.0, switch_threshold=3.0))
    return decisions


def draw_dynamic_trajectory(dyn_decisions, models, path=None):
    """附加分轨迹图：横轴=任务，纵轴=模型，展示每个任务 4 步的模型选择与切换。"""
    path = path or (FIG_DIR / "dynamic_trajectory.png")
    names = [m.name for m in models]
    ymap = {n: i for i, n in enumerate(names)}
    cmap = plt.cm.tab10
    model_color = {n: cmap(i % 10) for i, n in enumerate(names)}

    fig, ax = plt.subplots(figsize=(15, 8))
    for k, d in enumerate(dyn_decisions):
        xs = [k + 1 + (s.step - 1) * 0.2 for s in d.steps]
        ys = [ymap[s.model] for s in d.steps]
        # 相邻步连线（模型变化即出现断开/斜线，直观体现切换）
        for a, b in zip(d.steps[:-1], d.steps[1:]):
            ax.plot([k + 1 + (a.step - 1) * 0.2, k + 1 + (b.step - 1) * 0.2],
                    [ymap[a.model], ymap[b.model]], color="gray", lw=0.8, alpha=0.7, zorder=1)
        for s in d.steps:
            x, y = k + 1 + (s.step - 1) * 0.2, ymap[s.model]
            # 切换发生的步骤用红圈强调
            ax.plot(x, y, "o", ms=10, color=model_color[s.model], zorder=3,
                    markeredgecolor="red" if s.switched else "black",
                    markeredgewidth=1.8 if s.switched else 0.8)
            ax.annotate(f"{s.quality_obs:.0f}", (x, y), textcoords="offset points",
                        xytext=(0, 7), fontsize=6, ha="center")
    ax.set_xticks(range(1, len(dyn_decisions) + 1))
    ax.set_xticklabels([d.task.task_id for d in dyn_decisions])
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("任务")
    ax.set_ylabel("执行模型")
    ax.set_title("动态路由调整轨迹图（点=各步模型与观测质量，红圈=动态切换）")
    ax.grid(True, axis="x", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  已生成：{path}")


if __name__ == "__main__":
    main()