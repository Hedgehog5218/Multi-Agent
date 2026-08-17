# -*- coding: utf-8 -*-
"""端到端演示脚本（测评第二题 2.4 / 2.5）。

运行完整流程：
    1. 协调器处理 data/ 下 8 张图片（类型判定 -> 路由 -> 结构化提取）
    2. 生成实验室周报 Markdown（logs/weekly_report.md）
    3. 输出完整处理日志（logs/processing_log.*）
    4. 运行量化评估（logs/evaluation_report.json）
    5. 生成 3 张可视化图（figures/）：
       - swimlane_diagram.png  视觉识别流水线泳道图
       - radar_chart.png       识别准确率对比雷达图
       - sankey_diagram.png    协调器路由决策桑基图
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

# 中文字体配置（Windows 平台）
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun"]
plt.rcParams["axes.unicode_minus"] = False

from coordinator import Coordinator
from evaluation import evaluate, print_report
from agents import TYPE_NAMES

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(BASE_DIR, "figures")
LOG_DIR = os.path.join(BASE_DIR, "logs")


# ----------------------------------------------------------------------
# 可视化 1：视觉识别流水线泳道图
# ----------------------------------------------------------------------
def plot_swimlane(coord: Coordinator, results: list, out: str):
    """泳道图：从图片输入到周报输出的完整流程，每条泳道对应协调器/智能体。

    布局设计（左侧泳道名独立栏，彻底避免文字遮挡）：
      - 最左侧 0~12.5 为泳道名栏（加粗文字独占，不与任何内容重叠）；
      - 右侧 13.5~98 按 5 个阶段分区（图片输入/类型判定/专业识别/一致性检查/周报输出），
        每阶段独立背景色与顶部标签；
      - 处理块圆角矩形，块内两行排版（图片名 / 耗时）；
      - 数据流箭头画在协调器泳道，贯穿 5 阶段。
    """
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 14.2)

    # ---- 泳道定义（上到下：协调器 + 6 个智能体）----
    lanes = ["协调器", "实验装置智能体", "仪器面板智能体", "手写材料智能体",
             "图表材料智能体", "代码结果智能体", "文献页面智能体"]
    lane_h = 1.15
    lane_y = {}
    for i, name in enumerate(lanes):
        lane_y[name] = 12.3 - i * 1.55

    # ---- ① 左侧泳道名栏（独立背景 + 加粗文字，不与内容重叠）----
    ax.add_patch(plt.Rectangle((0.2, 1.9), 12.5, 11.0,
                               facecolor="#e3ebf7", edgecolor="none", zorder=0))
    for name in lanes:
        y = lane_y[name]
        ax.text(0.9, y, name, ha="left", va="center", fontsize=9.5, fontweight="bold",
                color="#111827", zorder=4)

    # ---- 泳道工作区背景（右侧 13.5~98）----
    for name in lanes:
        y = lane_y[name]
        ax.add_patch(plt.Rectangle((13.5, y - lane_h / 2), 84.5, lane_h,
                                   facecolor="#f3f6fc", edgecolor="#c5d5ea",
                                   linewidth=0.7, zorder=1))

    # ---- 5 个阶段分区（背景色 + 顶部标签）----
    stages = [
        ("图片输入", 13.5, 24.5, "#dbeafe"),
        ("类型判定", 26.0, 35.0, "#fef3c7"),
        ("专业识别", 37.0, 80.0, "#dcfce7"),
        ("汇总与一致性检查", 82.0, 91.0, "#fce7f3"),
        ("周报输出", 93.0, 98.0, "#ede9fe"),
    ]
    for st, x0, x1, color in stages:
        ax.add_patch(plt.Rectangle((x0, 1.9), x1 - x0, 11.0,
                                   facecolor=color, alpha=0.35, edgecolor="none", zorder=0))
        ax.text((x0 + x1) / 2, 13.65, st, ha="center", va="center",
                fontsize=10.5, fontweight="bold", color="#1f2937", zorder=4)

    # ---- ② 图片输入区：竖排 8 张图片（x=14.5~22.5，不与泳道名栏重叠）----
    img_y = np.linspace(11.9, 2.3, len(results))
    for r, yi in zip(results, img_y):
        ax.add_patch(FancyBboxPatch((15.0, yi - 0.42), 7.6, 0.84,
                                    boxstyle="round,pad=0.02", facecolor="#93c5fd",
                                    edgecolor="#1d4ed8", linewidth=0.7, zorder=3))
        ax.text(18.8, yi, r.image, ha="center", va="center", fontsize=8.5,
                color="#1e3a8a", zorder=4)

    # ---- ③ 协调器泳道：类型判定 / 一致性检查 / 周报输出 ----
    cy = lane_y["协调器"]
    blocks = [
        (27.0, 8.0, "#fbbf24", "类型判定", f"{len(results)} 张图片 · 文件名+OCR特征"),
        (83.0, 7.0, "#e879f9", "一致性检查", "跨材料交叉验证"),
        (93.5, 4.5, "#a78bfa", "周报输出", "Markdown"),
    ]
    for bx, bw, color, t1, t2 in blocks:
        ax.add_patch(FancyBboxPatch((bx, cy - 0.5), bw, 1.0,
                                    boxstyle="round,pad=0.02", facecolor=color,
                                    edgecolor="black", linewidth=0.8, zorder=3))
        ax.text(bx + bw / 2, cy + 0.18, t1, ha="center", va="center",
                fontsize=8.5, fontweight="bold", zorder=4)
        ax.text(bx + bw / 2, cy - 0.32, t2, ha="center", va="center",
                fontsize=6.6, color="#374151", zorder=4)

    # ---- 数据流箭头（协调器泳道，贯穿 5 阶段）----
    for x1, x2 in [(24.5, 27.0), (35.0, 37.0), (80.5, 83.0), (91.0, 93.5)]:
        ax.annotate("", xy=(x2, cy), xytext=(x1, cy),
                    arrowprops=dict(arrowstyle="-|>", color="#374151", lw=1.8))

    # ---- ④ 专业识别区：各智能体泳道内横向排列处理块 ----
    type_agent_map = {
        1: "实验装置智能体", 2: "仪器面板智能体", 3: "手写材料智能体",
        4: "图表材料智能体", 5: "代码结果智能体", 6: "图表材料智能体",
        7: "手写材料智能体", 8: "文献页面智能体",
    }
    for lane_name in lanes:
        if lane_name == "协调器":
            continue
        members = [r for r in results if type_agent_map.get(r.material_type_id) == lane_name]
        y = lane_y[lane_name]
        xc = 38.0
        for r in members:
            bw = 4.6
            ax.add_patch(FancyBboxPatch((xc, y - 0.47), bw, 0.94,
                                        boxstyle="round,pad=0.02", facecolor="#34d399",
                                        edgecolor="#047857", linewidth=0.7, zorder=3))
            ax.text(xc + bw / 2, y + 0.17, r.image, ha="center", va="center",
                    fontsize=8.5, fontweight="bold", color="#064e3b", zorder=4)
            ax.text(xc + bw / 2, y - 0.25, f"耗时 {max(r.processing_time * 1000, 1):.0f}ms",
                    ha="center", va="center", fontsize=7, color="#065f46", zorder=4)
            xc += bw + 1.4

    # ---- ⑤ 底部说明 ----
    ax.text(50, 0.9, "处理流程（时间轴 →）：图片输入 → 类型判定 → 专业识别 → 汇总与一致性检查 → 周报输出",
            ha="center", va="center", fontsize=9, color="#374151")

    ax.set_xlim(0, 100)
    ax.set_ylim(0, 14.2)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title("视觉识别流水线泳道图（6 个专业视觉智能体 + 协调器）",
                 fontsize=13.5, fontweight="bold", pad=14)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[可视化] 泳道图 -> {out}")


def plot_radar(report: dict, out: str):
    """雷达图：8 张测试图片（按类型）的识别难度差异。

    说明（避免误解）：
      - 蓝色实线 = 识别把握度（智能体自评置信度），体现各类型图片的识别难度差异；
      - 橙色虚线 = 综合识别得分（客观准确率口径，加权合成），作为“满分基准”参考线。
    准确率与置信度是两个不同概念：准确率是事后用 GT 比对的客观结果，
    置信度是识别时智能体对自身把握的自评，二者不矛盾。
    """
    rows = report["rows"]
    labels = []
    scores = []
    confs = []
    for row in rows:
        # 综合识别得分：字段准确率、结构还原率、数值准确率、路由正确 加权
        f = row["字段准确率"] or 0
        s = row["结构还原率"] or 0
        mape = row["数值MAPE"]
        num_acc = (1 - mape) if mape is not None else 1.0
        routing = 1.0 if row["路由正确"] else 0.0
        score = 0.4 * f + 0.25 * num_acc + 0.25 * s + 0.1 * routing
        labels.append(f"P{len(labels)+1}\n{row['材料类型']}")
        scores.append(score)
        confs.append(row["置信度"])

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    scores_c = scores + scores[:1]
    confs_c = confs + confs[:1]
    angles_c = angles + angles[:1]

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
    # 主序列：识别把握度（置信度）——有区分度，体现难度差异
    ax.plot(angles_c, confs_c, color="#2563eb", linewidth=2, label="识别把握度（智能体置信度）")
    ax.fill(angles_c, confs_c, color="#2563eb", alpha=0.2)
    # 参考序列：综合识别得分（客观准确率，全满分时退化为 1.0 基准线）
    ax.plot(angles_c, scores_c, color="#f59e0b", linewidth=1.6, linestyle="--", label="综合识别得分（客观准确率）")
    # 均值辅助线
    mean_conf = sum(confs) / len(confs) if confs else 0
    ax.plot(angles_c, [mean_conf] * len(angles_c), color="#9ca3af", linewidth=0.8,
            linestyle=":", label=f"平均把握度 {mean_conf:.2f}")
    ax.set_xticks(angles)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=8)
    ax.set_title("8 张测试图片识别难度差异（雷达图：把握度 vs 准确率）",
                 fontsize=12.5, fontweight="bold", pad=24)
    ax.legend(loc="upper right", bbox_to_anchor=(1.22, 1.12), fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[可视化] 雷达图 -> {out}")


# ----------------------------------------------------------------------
# 可视化 3：协调器路由决策桑基图
# ----------------------------------------------------------------------
def plot_sankey(results: list, out: str):
    """桑基图：8 张图片从输入到各专业智能体的分配路径（手绘贝塞尔连线）。"""
    fig, ax = plt.subplots(figsize=(13, 8))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)

    # 左侧节点：8 张图片
    left_items = [r.image for r in results]
    n = len(left_items)
    ly = np.linspace(90, 10, n)
    for i, name in enumerate(left_items):
        ax.add_patch(plt.Rectangle((3, ly[i] - 2.2), 18, 4.4, facecolor="#93c5fd",
                                   edgecolor="black", linewidth=0.8))
        short = name[:12] + ".." if len(name) > 12 else name
        ax.text(12, ly[i], short, ha="center", va="center", fontsize=8)

    # 右侧节点：8 个智能体
    agent_short = ["实验装置", "仪器面板", "手写材料", "图表材料",
                   "代码结果", "文献页面"]
    ry = np.linspace(90, 10, len(agent_short))
    for i, an in enumerate(agent_short):
        ax.add_patch(plt.Rectangle((79, ry[i] - 2.2), 18, 4.4, facecolor="#f9a8d4",
                                   edgecolor="black", linewidth=0.8))
        ax.text(88, ry[i], an + "智能体", ha="center", va="center", fontsize=8)

    # 连线：图片 -> 智能体（贝塞尔曲线）；8 类材料映射到 6 个智能体节点
    type_idx = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 3, 7: 2, 8: 5}
    for r in results:
        i = left_items.index(r.image)
        tid = getattr(r, "material_type_id", None)
        j = type_idx.get(tid, 0)
        x0, y0 = 21, ly[i]
        x1, y1 = 79, ry[j]
        mid = (x0 + x1) / 2
        t = np.linspace(0, 1, 50)
        bx = (1 - t) ** 2 * x0 + 2 * (1 - t) * t * mid + t ** 2 * x1
        by = (1 - t) ** 2 * y0 + 2 * (1 - t) * t * ((y0 + y1) / 2) + t ** 2 * y1
        ax.plot(bx, by, color="#6b7280", alpha=0.6, linewidth=1.8)

    ax.set_title("协调器路由决策桑基图（8 张图片 → 专业视觉智能体）", fontsize=13, fontweight="bold")
    ax.text(12, 96, "输入图片", ha="center", fontsize=11, fontweight="bold")
    ax.text(88, 96, "专业视觉智能体", ha="center", fontsize=11, fontweight="bold")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[可视化] 桑基图 -> {out}")


# ----------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser(description="task2 多智能体视觉识别与科研周报生成")
    parser.add_argument("--mode", choices=["auto", "rule", "llm"], default="auto",
                        help="识别模式：auto=已配置大模型则走LLM(失败回退规则版)；rule=规则版；llm=大模型版")
    parser.add_argument("--nofigs", action="store_true", help="跳过可视化生成")
    args = parser.parse_args()
    os.environ["LLM_MODE"] = args.mode

    mode_desc = {
        "auto": "自动（检测到 config.json 大模型配置 → 走 LLM，失败自动回退规则版）",
        "rule": "规则版（本地 OCR + 专业规则提取，可复现、零依赖）",
        "llm": "大模型版（视觉大模型端到端识别，失败返回明确错误）",
    }
    print("=" * 80)
    print("task2 端到端演示：多智能体视觉识别与科研周报生成")
    print("=" * 80)
    print(f"[模式] 识别模式：{mode_desc[args.mode]}")

    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    # 1. 运行协调器流水线
    print("\n[步骤 1/6] 初始化科研周报协调器（Coordinator）...")
    coord = Coordinator(data_dir=os.path.join(BASE_DIR, "data"),
                        logs_dir=LOG_DIR)
    n_images = len([f for f in os.listdir(coord.data_dir)
                    if f.lower().endswith((".png", ".jpg", ".jpeg"))])
    print(f"[步骤 2/6] 逐张识别 {n_images} 张图片：类型判定 → 路由 → 专业提取 ...")
    results = coord.run_pipeline()
    print(f"[协调器] 完成 {len(results)} 张图片的识别，处理日志 -> {LOG_DIR}")

    # 2. 一致性检查 + 周报生成
    print("\n[步骤 3/6] 跨材料一致性检查（7 类规则：传感器/坐标系/指标/时间/数据自洽/待办/主题）...")
    issues = coord.check_consistency()
    n_high = sum(1 for i in issues if i["级别"] == "高")
    n_mid = sum(1 for i in issues if i["级别"] == "中")
    print(f"    → 高风险 {n_high} 项 / 中风险 {n_mid} 项 / 一致确认 {sum(1 for i in issues if i['级别'] in ('低','观察'))} 项")
    print("[步骤 4/6] 生成实验室周报（Markdown，六板块）...")
    report_md = coord.generate_weekly_report(issues)
    report_path = os.path.join(LOG_DIR, "weekly_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_md)
    print(f"[协调器] 实验室周报 -> {report_path}")

    # 3. 量化评估
    print("\n[步骤 5/6] 运行量化评估（字段准确率 / 数值MAPE / 结构还原率 / 路由准确率）...")
    eval_report = evaluate(results, gt_path=os.path.join(BASE_DIR, "data", "ground_truth.json"))
    print_report(eval_report)
    eval_path = os.path.join(LOG_DIR, "evaluation_report.json")
    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(eval_report, f, ensure_ascii=False, indent=2)
    print(f"[评估] 结果 -> {eval_path}")

    # 4. 可视化
    if not args.nofigs:
        print("\n[步骤 6/6] 生成 3 张可视化图（泳道图 / 雷达图 / 桑基图）...")
        plot_swimlane(coord, results, os.path.join(FIG_DIR, "swimlane_diagram.png"))
        plot_radar(eval_report, os.path.join(FIG_DIR, "radar_chart.png"))
        plot_sankey(results, os.path.join(FIG_DIR, "sankey_diagram.png"))
    else:
        print("\n[步骤 6/6] 已跳过可视化生成（--nofigs）")

    print("\n" + "=" * 80)
    print("演示完成。产出清单：")
    print(f"  周报    : {report_path}")
    print(f"  日志    : {LOG_DIR}")
    print(f"  评估    : {eval_path}")
    print(f"  可视化  : {FIG_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
