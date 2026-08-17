# -*- coding: utf-8 -*-
"""1.4 可视化：根据系统运行日志自动生成图表。

- sequence_diagram()         : 通信序列图（UML 时序图风格）
- load_distribution()        : 通信负载分布图（柱状图，收发消息数）
- message_type_distribution(): 消息类型分布饼图
- topology_diagram()         : 通信拓扑图（1.2 拓扑设计辅助图）

所有函数输入为 `MessageBus.log_entries()` 返回的日志条目列表
（每条含 seq/send_time/sender/receiver/message_type/payload 等）。
"""
from __future__ import annotations

import os
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

# 中文字体
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun"]
plt.rcParams["axes.unicode_minus"] = False

PARTICIPANTS = ["coordinator", "literature", "method", "experiment", "verifier", "polish"]
PARTICIPANT_LABELS = {
    "coordinator": "协调器",
    "literature": "文献调研",
    "method": "方法设计",
    "experiment": "实验规划",
    "verifier": "数据核查",
    "polish": "统稿润色",
}
TYPE_COLORS = {
    "TASK_ASSIGN": "#2c7fb8",
    "INFO_QUERY": "#7fcdbb",
    "RESULT_SUBMIT": "#41ab5d",
    "CONFLICT_NOTIFY": "#d95f0e",
    "ACK_RECEIPT": "#969696",
    "NEGOTIATE": "#984ea3",
    "LOCK_REQUEST": "#fed976",
    "LOCK_GRANT": "#fee391",
    "LOCK_RELEASE": "#feb24c",
    "BROADCAST": "#e7298a",
}


def _load_entries(log_path: str) -> List[Dict[str, Any]]:
    import json
    entries = []
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


# ---------------------------------------------------------------------------
# 1) 通信序列图（UML 时序图风格）
# ---------------------------------------------------------------------------
def sequence_diagram(entries: List[Dict[str, Any]], out_path: str,
                     participants: Optional[List[str]] = None) -> str:
    parts = participants or PARTICIPANTS
    xs = {p: i for i, p in enumerate(parts)}
    n = len(parts)
    msgs = sorted(entries, key=lambda e: e.get("seq", 0))

    fig_h = max(10.0, len(msgs) * 0.30 + 2)
    fig, ax = plt.subplots(figsize=(13, fig_h))
    ax.set_xlim(-0.8, n - 0.2)
    ax.set_ylim(len(msgs) + 1.5, -0.5)

    # 生命线
    for p in parts:
        ax.plot([xs[p], xs[p]], [len(msgs) + 0.8, 0.2], ls=":", color="#bbbbbb", lw=1)
    # 参与者头部
    for p in parts:
        ax.text(xs[p], len(msgs) + 1.1, PARTICIPANT_LABELS.get(p, p),
                ha="center", va="center", fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", fc="#e8f1f8", ec="#2c7fb8"))

    # 阶段分隔带
    phase_marks = []
    for e in msgs:
        if e.get("message_type") == "BROADCAST":
            ph = (e.get("payload") or {}).get("phase")
            if ph in ("decompose", "conflict_resolved", "complete"):
                phase_marks.append((e.get("seq", 0), ph))
    phase_names = {"decompose": "阶段1 任务分解", "conflict_resolved": "阶段4 冲突解决",
                   "complete": "阶段5 最终统稿"}
    prev = 0.0
    for seq, ph in phase_marks:
        y = seq + 0.5
        ax.axhspan(prev, y, color="#f7fbff", alpha=0.0)
        ax.text(n - 0.3, y - 0.6, phase_names.get(ph, ph), fontsize=8,
                color="#666666", ha="right", va="center")
        prev = y
    ax.axhspan(prev, len(msgs) + 1.0, color="#f7fbff", alpha=0.0)

    # 消息箭头
    used_labels = 0
    for e in msgs:
        seq = e.get("seq", 0)
        sender, receiver = e.get("sender", ""), e.get("receiver", "")
        mtype = e.get("message_type", "")
        if sender not in xs or receiver not in xs:
            continue
        color = TYPE_COLORS.get(mtype, "#333333")
        x0, x1 = xs[sender], xs[receiver]
        y = seq
        if x0 == x1:
            # 自消息：画成小环
            ax.annotate("", xy=(x0 + 0.22, y - 0.28), xytext=(x0, y),
                        arrowprops=dict(arrowstyle="->", color=color, lw=1.0))
        else:
            ax.annotate("", xy=(x1, y), xytext=(x0, y),
                        arrowprops=dict(arrowstyle="->", color=color, lw=1.2,
                                        shrinkA=2, shrinkB=2))
        label = mtype
        dy = 0.14
        ax.text((x0 + x1) / 2 + 0.05, y + dy, label, fontsize=5.6,
                color=color, ha="center", va="bottom", rotation=0)
        used_labels += 1

    ax.set_xticks([xs[p] for p in parts])
    ax.set_xticklabels([PARTICIPANT_LABELS.get(p, p) for p in parts], fontsize=9)
    ax.set_yticks([])
    ax.set_xlabel("智能体", fontsize=11)
    ax.set_ylabel("消息时序（seq 序号，自上而下）", fontsize=11)
    ax.set_title("通信序列图：一次完整科研协作写作任务的消息交互时序", fontsize=13)
    for spine in ax.spines.values():
        spine.set_visible(False)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# 2) 通信负载分布图（柱状图）
# ---------------------------------------------------------------------------
def load_distribution(entries: List[Dict[str, Any]], out_path: str,
                      participants: Optional[List[str]] = None) -> str:
    parts = participants or PARTICIPANTS
    sent = Counter()
    recv = Counter()
    for e in entries:
        s, r = e.get("sender", ""), e.get("receiver", "")
        if s in parts:
            sent[s] += 1
        if r in parts:
            recv[r] += 1

    labels = [PARTICIPANT_LABELS.get(p, p) for p in parts]
    x = list(range(len(parts)))
    w = 0.38
    fig, ax = plt.subplots(figsize=(9, 5.5))
    b1 = ax.bar([i - w / 2 for i in x], [sent[p] for p in parts], width=w,
                label="发送消息数", color="#2c7fb8")
    b2 = ax.bar([i + w / 2 for i in x], [recv[p] for p in parts], width=w,
                label="接收消息数", color="#feb24c")
    for bar in list(b1) + list(b2):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.4,
                f"{int(bar.get_height())}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("消息数量", fontsize=11)
    ax.set_title("通信负载分布：各智能体收发消息数量（通信热点）", fontsize=12)
    ax.legend()
    ax.grid(axis="y", ls="--", alpha=0.4)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# 3) 消息类型分布饼图
# ---------------------------------------------------------------------------
def message_type_distribution(entries: List[Dict[str, Any]], out_path: str) -> str:
    counter = Counter(e.get("message_type", "UNKNOWN") for e in entries)
    labels_zh = {
        "TASK_ASSIGN": "任务分配", "INFO_QUERY": "信息查询",
        "RESULT_SUBMIT": "结果提交", "CONFLICT_NOTIFY": "冲突通知",
        "ACK_RECEIPT": "确认回执", "NEGOTIATE": "协商",
        "LOCK_REQUEST": "锁请求", "LOCK_GRANT": "锁授予",
        "LOCK_RELEASE": "锁释放", "BROADCAST": "广播",
    }
    labels = [labels_zh.get(k, k) for k in counter.keys()]
    sizes = list(counter.values())
    colors = [TYPE_COLORS.get(k, "#999999") for k in counter.keys()]
    fig, ax = plt.subplots(figsize=(8, 6))
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, autopct="%1.1f%%", startangle=90,
        colors=colors, textprops={"fontsize": 9},
        wedgeprops={"edgecolor": "white", "linewidth": 1})
    for at in autotexts:
        at.set_fontsize(9)
    ax.set_title("消息类型分布：各类消息数量占比", fontsize=12)
    ax.axis("equal")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# 4) 通信拓扑图（1.2 辅助图）
# ---------------------------------------------------------------------------
def topology_diagram(out_path: str,
                     participants: Optional[List[str]] = None) -> str:
    parts = participants or [p for p in PARTICIPANTS if p != "coordinator"]
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 11)
    ax.axis("off")

    cx, cy = 5.0, 6.5
    ax.add_patch(Rectangle((cx - 1.0, cy - 0.55), 2.0, 1.1, fc="#dbe9f8",
                           ec="#2c7fb8", lw=1.5, zorder=5))
    ax.text(cx, cy, "协调器\n(coordinator)", ha="center", va="center",
            fontsize=11, zorder=6)

    import math
    radius = 3.1
    for i, p in enumerate(parts):
        ang = math.pi / 2 - i * (2 * math.pi / len(parts))
        x = cx + radius * math.cos(ang)
        y = cy + radius * math.sin(ang) * 0.78
        ax.plot([cx, x], [cy, y], color="#8c96c6", lw=1.2, zorder=2)
        ax.add_patch(Rectangle((x - 0.95, y - 0.4), 1.9, 0.8, fc="#fde6c4",
                               ec="#d95f0e", lw=1.2, zorder=5))
        ax.text(x, y, PARTICIPANT_LABELS.get(p, p), ha="center", va="center",
                fontsize=9.5, zorder=6)

    # 黑板（共享工件）
    ax.add_patch(Rectangle((2.2, 0.55), 5.6, 1.1, fc="#e5f5e0", ec="#41ab5d",
                           lw=1.5, zorder=5))
    ax.text(5.0, 1.1, "黑板：共享章节工作区 SharedWorkspace\n（按章节写锁 + 版本号，正文全文共享）",
            ha="center", va="center", fontsize=9, zorder=6)
    for p in parts:
        import math as _m
        ang = _m.pi / 2 - parts.index(p) * (2 * _m.pi / len(parts))
        x = cx + radius * _m.cos(ang)
        y = cy + radius * _m.sin(ang) * 0.78
        ax.plot([x, 5.0], [y, 1.1], ls="--", color="#74c476", lw=1.0, zorder=1)

    ax.text(5.0, 10.6, "通信拓扑：星型控制平面 + 黑板数据平面（混合拓扑）",
            ha="center", fontsize=12)
    ax.text(5.0, 10.05, "子智能体经协调器转发控制消息（1~2 跳）；正文经黑板共享（消息只传摘要/增量）",
            ha="center", fontsize=9, color="#555555")
    ax.set_title("1.2 通信拓扑设计图", fontsize=11, pad=50)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def generate_all(log_path: str, figures_dir: str) -> Dict[str, str]:
    """从日志一键生成全部图表，返回 {名称: 路径}。"""
    entries = _load_entries(log_path)
    os.makedirs(figures_dir, exist_ok=True)
    seq = sequence_diagram(entries, os.path.join(figures_dir, "sequence_diagram.png"))
    load = load_distribution(entries, os.path.join(figures_dir, "load_distribution.png"))
    pie = message_type_distribution(entries, os.path.join(figures_dir, "message_type_distribution.png"))
    topo = topology_diagram(os.path.join(figures_dir, "topology_diagram.png"))
    return {
        "sequence_diagram": seq,
        "load_distribution": load,
        "message_type_distribution": pie,
        "topology_diagram": topo,
    }
if __name__ == "__main__":
    # 指定输出图片的名称和路径
    output_filename = "topology_diagram.png"
    
    # 调用函数生成拓扑图
    topology_diagram(output_filename)
