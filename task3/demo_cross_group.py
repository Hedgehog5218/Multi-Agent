# -*- coding: utf-8 -*-
# demo_cross_group.py —— 第三题 3.4/3.5 端到端演示
# 运行：python demo_cross_group.py
# 输出：logs/run_*.log 运行日志；logs/cross_group_report.md 跨课题协同分析报告；
#       figures/coupling_network.png 耦合网络图；figures/coupling_type_bar.png 耦合类型柱状图；
#       figures/alignment_process.png 语义对齐过程图。

import argparse
import os
import sys
import time

# 统一控制台输出为 UTF-8，避免 Windows 终端中文乱码
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import matplotlib
matplotlib.use("Agg")  # 无界面后端，适合服务器/脚本环境
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

from agents.base import Logger
from coupling_detector import run as run_detector

# 中文字体配置
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

SEV_WIDTH = {"高": 5.0, "中": 3.0, "低": 1.5}
COLOR_POS = "#2ca02c"
COLOR_NEG = "#d62728"


# --------------------------------------------------------------------------
# 3.5-1 跨课题耦合关系网络图
# --------------------------------------------------------------------------
def plot_coupling_network(candidates, out_path):
    """跨课题耦合关系网络图（优化版）：
    节点 = 课题组（三角布局，颜色区分，节点内标注该组文档）；
    边   = 耦合关系（绿=可协同，红=冲突，线粗=严重程度），边上标签说明 ID/类型/关键实体；
    组内耦合（如专利 vs 论文）用自环表示。"""
    from collections import defaultdict
    import matplotlib.lines as mlines

    fig, ax = plt.subplots(figsize=(13, 9.5))
    ax.set_xlim(-4.4, 4.4)
    ax.set_ylim(-3.5, 3.8)
    ax.axis("off")

    # 课题组节点：三角布局 + 不同颜色 + 组内文档
    groups = {
        "模型压缩组":   {"pos": (0.0, 3.0), "color": "#4c72b0", "docs": "论文手稿 · 技术专利草稿"},
        "分布式训练组": {"pos": (-3.0, -1.1), "color": "#dd8452", "docs": "实验报告 · 结题报告"},
        "联邦学习组":   {"pos": (3.0, -1.1), "color": "#55a868", "docs": "项目申请书"},
    }
    for g, info in groups.items():
        x, y = info["pos"]
        ax.add_patch(plt.Circle((x, y), 0.66, color=info["color"], alpha=0.92, zorder=5))
        ax.text(x, y + 0.12, g, ha="center", va="center", color="white",
                fontsize=12.5, fontweight="bold", zorder=6)
        ax.text(x, y - 0.34, info["docs"], ha="center", va="center", color="white",
                fontsize=8.5, zorder=6, alpha=0.95)

    # 课题组对 -> 耦合列表
    pair_edges = defaultdict(list)
    intra = []  # 组内耦合
    for c in candidates:
        g1, g2 = c["groups"][0], c["groups"][-1]
        if g1 not in groups or g2 not in groups:
            continue
        if g1 == g2:
            intra.append(c)
        else:
            pair_edges[tuple(sorted([g1, g2]))].append(c)

    # 跨组耦合边（同一组对的多条边做垂直偏移）
    for key, clist in pair_edges.items():
        g1, g2 = key
        (x1, y1), (x2, y2) = groups[g1]["pos"], groups[g2]["pos"]
        dx, dy = x2 - x1, y2 - y1
        length = (dx * dx + dy * dy) ** 0.5
        nx, ny = -dy / length, dx / length
        n = len(clist)
        for i, c in enumerate(clist):
            off = (i - (n - 1) / 2.0) * 0.72
            mx, my = (x1 + x2) / 2 + nx * off, (y1 + y2) / 2 + ny * off
            color = COLOR_POS if c["type"] == "positive" else COLOR_NEG
            width = SEV_WIDTH.get(c["severity"], 2.0)
            ax.plot([x1, mx, x2], [y1, my, y2], color=color, linewidth=width,
                    alpha=0.8, zorder=2, solid_capstyle="round")
            _edge_label(ax, mx, my + 0.24, c, color)

    # 组内耦合：自环
    for c in intra:
        g1 = c["groups"][0]
        x1, y1 = groups[g1]["pos"]
        color = COLOR_POS if c["type"] == "positive" else COLOR_NEG
        width = SEV_WIDTH.get(c["severity"], 2.0)
        # 自环放在节点右侧或左侧（避免超出画布）
        side = 1 if x1 <= 0 else -1
        ax.add_patch(plt.Circle((x1 + side * 0.95, y1 - 0.2), 0.42, fill=False,
                                edgecolor=color, linewidth=width, zorder=3))
        _edge_label(ax, x1 + side * 1.95, y1 - 0.45, c, color)

    # 图例
    pos_line = mlines.Line2D([], [], color=COLOR_POS, linewidth=4.5, label="正耦合：可协同")
    neg_line = mlines.Line2D([], [], color=COLOR_NEG, linewidth=4.5, label="负耦合：冲突矛盾")
    sev_line = mlines.Line2D([], [], color="#888888", linewidth=7, label="线越粗 = 严重程度越高")
    ax.legend(handles=[pos_line, neg_line, sev_line], loc="lower left",
              fontsize=11, framealpha=0.95, edgecolor="#cccccc")
    ax.set_title("跨课题耦合关系网络图（节点=课题组，边=耦合关系）",
                 fontsize=14.5, pad=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _edge_label(ax, x, y, c, color):
    """在给定位置绘制耦合边标签：ID + 类型 + 关键实体"""
    tag = "可协同" if c["type"] == "positive" else "冲突"
    short = c["category"].split("-")[-1] if "-" in c["category"] else c["category"]
    ents = "、".join(c["entities"][:2]) or "—"
    label = f"{c['id']} · {tag}\n{short}\n{ents}"
    ax.text(x, y, label, ha="center", va="center", fontsize=9.5, color=color,
            fontweight="bold", zorder=7, linespacing=1.35,
            bbox=dict(boxstyle="round,pad=0.32", facecolor="white",
                      edgecolor=color, linewidth=1.5, alpha=0.96))


def plot_coupling_type_bar(candidates, out_path):
    """按课题组对分组：正耦合 vs 负耦合数量柱状图"""
    pair_names = {
        ("模型压缩组", "分布式训练组"): "压缩组—分布式组",
        ("模型压缩组", "联邦学习组"): "压缩组—联邦组",
        ("分布式训练组", "联邦学习组"): "分布式组—联邦组",
        ("模型压缩组", "模型压缩组"): "压缩组内部",
    }
    pairs = {}
    for c in candidates:
        key = tuple(sorted(c["groups"]))
        pairs.setdefault(key, {"positive": 0, "negative": 0})
        pairs[key][c["type"]] += 1

    labels = [pair_names.get(k, "+".join(k)) for k in pairs.keys()]
    pos = [v["positive"] for v in pairs.values()]
    neg = [v["negative"] for v in pairs.values()]

    x = range(len(labels))
    w = 0.36
    fig, ax = plt.subplots(figsize=(9, 6))
    b1 = ax.bar([i - w / 2 for i in x], pos, w, label="正耦合（可协同）", color=COLOR_POS)
    b2 = ax.bar([i + w / 2 for i in x], neg, w, label="负耦合（冲突矛盾）", color=COLOR_NEG)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("耦合数量", fontsize=12)
    ax.set_title("耦合类型分布（按课题组对分组）", fontsize=14)
    ax.legend(fontsize=10)
    for bars in (b1, b2):
        for b in bars:
            h = b.get_height()
            if h > 0:
                ax.text(b.get_x() + b.get_width() / 2, h + 0.04, str(int(h)),
                        ha="center", va="bottom", fontsize=10)
    maxv = max(list(pos) + list(neg) + [1])
    ax.set_ylim(0, maxv + 0.8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------
# 3.5-3 语义对齐过程图（典型案例）
# --------------------------------------------------------------------------
def plot_alignment_process(result, out_path):
    """选取 V100 资源可用性冲突案例，展示 原始文档->课题解析->语义表示->对齐判定"""
    case = next((c for c in result["candidates"]
                 if c["type"] == "negative" and "V100" in c["entities"]), None)
    if case is None and result["candidates"]:
        case = result["candidates"][0]

    fig, ax = plt.subplots(figsize=(14, 6.5))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 7)
    ax.axis("off")

    def box(x, y, w, h, text, fc="#eaf2fb", ec="#4c72b0", fs=9.5):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                                    facecolor=fc, edgecolor=ec, linewidth=1.2))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, linespacing=1.5)

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                     mutation_scale=16, color="#555555",
                                     linewidth=1.4))

    # ① 原始文档
    ax.text(1.2, 6.5, "① 原始文档", fontsize=12, fontweight="bold")
    box(0.3, 3.9, 2.5, 2.3,
        "《论文手稿》（模型压缩组）\n“全部实验于2026年8月在实验室\n4块V100加速卡上完成”",
        fc="#fdf3e7", ec="#e08a3c")
    box(0.3, 0.9, 2.5, 2.3,
        "《实验报告》（分布式训练组）\n“自2026年7月起，全部老旧V100\n节点已完成下线与报废处置”",
        fc="#fdf3e7", ec="#e08a3c")

    # ② 课题解析
    ax.text(4.3, 6.5, "② 课题解析", fontsize=12, fontweight="bold")
    box(3.4, 3.9, 2.5, 2.3,
        "论文解析智能体\n抽取实体：V100\n事件：use @2026-08",
        fc="#eaf2fb", ec="#4c72b0")
    box(3.4, 0.9, 2.5, 2.3,
        "实验报告解析智能体\n抽取实体：V100\n事件：retire @2026-07",
        fc="#eaf2fb", ec="#4c72b0")

    # ③ 语义表示
    ax.text(7.3, 6.5, "③ 结构化语义表示", fontsize=12, fontweight="bold")
    box(6.4, 2.4, 2.5, 2.3,
        "实体-事件三元组\n(V100, 使用, 2026-08)\n(V100, 退役, 2026-07)",
        fc="#eef6e8", ec="#6aa84f")

    # ④ 对齐判定
    ax.text(10.3, 6.5, "④ 实体对齐与冲突判定", fontsize=12, fontweight="bold")
    box(9.4, 2.4, 3.6, 2.3,
        "实体对齐：V100 ≡ 4块V100 ≡ 老旧V100节点\n"
        "时序推理：使用(2026-08) ＞ 退役(2026-07)\n"
        "判定：负耦合（资源可用性冲突，高）",
        fc="#fdeeee", ec="#d62728", fs=9)

    # 箭头
    arrow(2.85, 5.0, 3.35, 5.0)
    arrow(2.85, 2.0, 3.35, 2.0)
    arrow(5.95, 4.4, 6.35, 3.8)
    arrow(5.95, 2.6, 6.35, 3.2)
    arrow(8.95, 3.5, 9.35, 3.5)

    ax.set_title("语义对齐过程图（典型案例：V100 资源可用性冲突）", fontsize=14, pad=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------
# 主入口
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="task3 跨课题信息耦合端到端演示")
    ap.add_argument("--data", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))
    ap.add_argument("--logs", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs"))
    ap.add_argument("--figures", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures"))
    ap.add_argument("--retry", type=int, default=1,
                    help="检测阶段最多尝试轮数（默认 1，即单轮直接出结果，不做验收重试）")
    args = ap.parse_args()

    os.makedirs(args.logs, exist_ok=True)
    os.makedirs(args.figures, exist_ok=True)

    stamp = time.strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(args.logs, "run_" + stamp + ".log")
    report_path = os.path.join(args.logs, "cross_group_report.md")

    logger = Logger(path=log_path, echo=True)
    print("=" * 62)
    print("task3 · 多智能体语义识别与跨课题信息耦合（DeepSeek 大模型驱动）")
    print("=" * 62)
    logger.log("INFO", "demo", "task3 端到端演示启动，数据目录：" + args.data)

    result = run_detector(args.data, logger=logger, max_detect_tries=args.retry)
    ev = result["evaluation"]

    # 保存运行日志与报告
    logger.dump()
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(result["report_md"])

    # 生成 3 张图
    net_path = os.path.join(args.figures, "coupling_network.png")
    bar_path = os.path.join(args.figures, "coupling_type_bar.png")
    align_path = os.path.join(args.figures, "alignment_process.png")
    plot_coupling_network(result["candidates"], net_path)
    plot_coupling_type_bar(result["candidates"], bar_path)
    plot_alignment_process(result, align_path)
    logger.log("INFO", "demo", "可视化输出：" + net_path + " / " + bar_path + " / " + align_path)
    logger.dump()

    n_pos = sum(1 for c in result["candidates"] if c["type"] == "positive")
    n_neg = sum(1 for c in result["candidates"] if c["type"] == "negative")
    print("")
    print("=" * 62)
    print("✅ task3 端到端演示完成")
    print("  检出耦合：" + str(len(result["candidates"])) + " 条（正 " + str(n_pos) + " / 负 " + str(n_neg) + "）")
    print("  评估：精确率 {:.2%} / 召回率 {:.2%} / F1 {:.2%}".format(ev["precision"], ev["recall"], ev["f1"]))
    print("  报告：" + report_path)
    print("  日志：" + log_path)
    print("  图：" + net_path)
    print("      " + bar_path)
    print("      " + align_path)
    print("=" * 62)


if __name__ == "__main__":
    main()
