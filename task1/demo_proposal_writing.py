# -*- coding: utf-8 -*-
"""第一题 · 1.3 端到端演示：协作撰写《基于多智能体强化学习的分布式计算资源
调度方法研究》基金申请书。

本版为"纯大模型生成"模式：
- 全部章节正文由 DeepSeek 大模型从零生成（配置见 config.py，无需终端环境变量）；
- 数据/逻辑核查智能体从生成的正文中提取事实并做跨章节一致性检查；
- 发现冲突后由协调器协商仲裁，涉事智能体按决议重新调用大模型修订；
- 全程消息经通信协议传递并落盘日志。

运行方式（在 task1 目录下）：
    python demo_proposal_writing.py

输出：
    logs/messages.jsonl      # 全部消息日志（发送时间/发送方/接收方/类型/摘要）
    logs/demo.log            # 人类可读运行日志
    logs/final_proposal.md   # 最终统稿生成的 Markdown 基金申请书
    figures/*.png            # 可视化图（序列图/负载分布/消息类型饼图/拓扑图）
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if hasattr(sys.stdout, "reconfigure"):  # 统一 UTF-8 输出，避免中文乱码
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from protocol import Message  # noqa: E402
from agents import build_system  # noqa: E402
import visualize  # noqa: E402


def _ts() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def make_listener(log_dir: str):
    """构造消息总线监听器：把每条消息写入 JSONL 与人类可读日志。"""
    os.makedirs(log_dir, exist_ok=True)
    jsonl_path = os.path.join(log_dir, "messages.jsonl")
    txt_path = os.path.join(log_dir, "demo.log")

    def listener(msg: Message) -> None:
        d = msg.to_dict()
        entry = {
            "seq": msg.seq,
            "send_time": msg.timestamp,
            "sender": msg.sender,
            "receiver": msg.receiver,
            "message_type": msg.message_type.value,
            "priority": msg.priority.value,
            "related_message_id": msg.related_message_id,
            "correlation_id": msg.correlation_id,
            "payload_summary": msg.summary(90),
        }
        with open(jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        with open(txt_path, "a", encoding="utf-8") as f:
            f.write(
                f"[{msg.timestamp}] #{msg.seq:>3} {msg.message_type.value:<15} "
                f"{msg.sender:<11} -> {msg.receiver:<11} | {entry['payload_summary']}\n"
            )

    return listener


def main() -> int:
    parser = argparse.ArgumentParser(description="多智能体协同写作端到端演示（DeepSeek 生成）")
    parser.add_argument("--log-dir", default="logs", help="日志输出目录")
    parser.add_argument("--figures-dir", default="figures", help="图表输出目录")
    args = parser.parse_args()

    # 清空旧的日志文件（保留目录）
    for name in ("messages.jsonl", "demo.log"):
        p = os.path.join(args.log_dir, name)
        if os.path.exists(p):
            os.remove(p)

    print("=" * 78)
    print("第一题 · 多智能体通信协议设计与科研协作写作 —— 端到端演示（DeepSeek 生成）")
    print("任务：协作撰写《基于多智能体强化学习的分布式计算资源调度方法研究》基金申请书")
    print("说明：全部章节正文由 DeepSeek 大模型从零生成；冲突由核查智能体自动检测。")
    print("=" * 78)

    listener = make_listener(args.log_dir)
    system = build_system(output_dir=args.log_dir, listener=listener)
    coordinator = system["coordinator"]
    bus = system["bus"]

    t0 = time.time()
    summary: Dict[str, Any] = coordinator.run_demo()
    elapsed = time.time() - t0

    print("\n" + "=" * 78)
    print("演示完成！汇总：")
    print(f"  总耗时          : {elapsed:.2f}s")
    print(f"  消息总数        : {summary['communication']['total_messages']}")
    print(f"  通信总开销 token : {summary['communication']['total_tokens']}")
    print(f"  发现冲突        : {summary['conflicts_found']} 处")
    print(f"  解决冲突        : {summary['conflicts_resolved']} 处")
    final = summary.get("final_proposal", {})
    print(f"  最终申请书      : {final.get('output_path')} "
          f"({final.get('char_count', 0)} 字符, {final.get('section_count', 0)} 章节)")
    print("=" * 78)

    print("\n生成可视化图表 ...")
    fig_paths = visualize.generate_all(
        os.path.join(args.log_dir, "messages.jsonl"), args.figures_dir)
    for k, v in fig_paths.items():
        print(f"  {k:<24}: {v}")

    with open(os.path.join(args.log_dir, "messages.jsonl"), "r", encoding="utf-8") as f:
        lines = [l for l in f if l.strip()]
    print(f"\n日志校验: messages.jsonl 共 {len(lines)} 条消息记录，"
          f"final_proposal.md 存在: {os.path.exists(final.get('output_path', ''))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
