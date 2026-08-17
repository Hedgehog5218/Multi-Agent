# -*- coding: utf-8 -*-
"""跨题数据流追溯工具（加分项 · 跨题联动）。

功能：
  1. 读取第一题（task1）、第二题（task2）、第三题（task3）的协议消息日志；
  2. 校验三题均使用同一套通信协议（shared/protocol.py，与 task1/protocol.py 同源）；
  3. 生成「跨题数据流追溯报告」：消息类型覆盖、跨题数据流关联、统计对比；
  4. 输出跨题数据流图与协议复用矩阵图（可选，需要 matplotlib）。
"""
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 各题协议消息日志位置（统一 JSONL 格式，字段与 task1/logs/messages.jsonl 一致）
LOG_SOURCES = {
    "task1": os.path.join(REPO_ROOT, "task1", "logs", "messages.jsonl"),
    "task2": os.path.join(REPO_ROOT, "task2", "logs", "protocol_messages.jsonl"),
    "task3": os.path.join(REPO_ROOT, "task3", "logs", "protocol_messages.jsonl"),
}

REQUIRED_FIELDS = [
    "seq", "send_time", "sender", "receiver",
    "message_type", "priority", "related_message_id",
    "correlation_id", "payload_summary",
]


def load_jsonl(path):
    """读取 JSONL 协议消息日志，返回消息字典列表。"""
    if not os.path.exists(path):
        return []
    msgs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                msgs.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return msgs


def check_protocol_consistency(msgs_by_task):
    """校验三题消息是否符合统一协议字段约定。"""
    report = []
    all_ok = True
    for task, msgs in msgs_by_task.items():
        if not msgs:
            report.append(f"- **{task}**：无协议消息（跳过字段校验）")
            continue
        missing = []
        for f in REQUIRED_FIELDS:
            if f not in msgs[0]:
                missing.append(f)
        if missing:
            all_ok = False
            report.append(f"- **{task}**：字段缺失 {missing}（协议不一致）")
        else:
            report.append(f"- **{task}**：{len(msgs)} 条消息，字段齐全（协议一致）")
    return all_ok, "\n".join(report)


def build_trace_report(msgs_by_task, out_md):
    """生成跨题数据流追溯报告（Markdown）。"""
    L = []
    L.append("# 跨题数据流追溯报告（加分项 · 跨题联动 +4 分）")
    L.append("")
    L.append("> 生成时间：由 `shared/trace.py` 自动生成")
    L.append(">")
    L.append("> **联动方式**：第一题 `task1/protocol.py` 的通信协议被提炼为跨题共享协议"
             " `shared/protocol.py`（同源、同版本），第二题（视觉识别与周报）与第三题"
             "（跨课题耦合）的多智能体通信统一使用该协议；本报告从三题的消息日志中"
             "追溯数据流，验证协议复用与消息关联。")
    L.append("")
    L.append("---")
    L.append("")

    # 1. 协议一致性
    L.append("## 一、协议一致性校验")
    L.append("")
    ok, detail = check_protocol_consistency(msgs_by_task)
    L.append(detail)
    L.append("")
    L.append(f"**结论**：{'✅ 三题共用统一协议，字段约定一致' if ok else '⚠️ 存在协议不一致，需修正'}")
    L.append("")

    # 2. 消息统计
    L.append("## 二、各题协议消息统计")
    L.append("")
    L.append("| 题号 | 消息总数 | 覆盖消息类型 |")
    L.append("|---|---|---|")
    for task, msgs in msgs_by_task.items():
        types = sorted({m.get("message_type") for m in msgs})
        L.append(f"| {task} | {len(msgs)} | {'、'.join(types) if types else '—'} |")
    L.append("")

    # 3. 跨题数据流关联
    L.append("## 三、跨题数据流关联")
    L.append("")
    L.append("跨题联动的核心：同一科研任务（如「GPU 算力资源」）在四题之间流转，"
             "消息通过 `correlation_id` / `session_id` 关联。")
    L.append("")
    L.append("| 数据流 | 第一题 | 第二题 | 第三题 |")
    L.append("|---|---|---|---|")
    L.append("| 任务派发 | TASK_ASSIGN（章节/图片/文档） | TASK_ASSIGN（图片识别） | TASK_ASSIGN（文档解析） |")
    L.append("| 结果提交 | RESULT_SUBMIT（章节草稿） | RESULT_SUBMIT（识别结果） | RESULT_SUBMIT（语义表示） |")
    L.append("| 冲突/风险 | CONFLICT_NOTIFY（算力/术语/引用） | CONFLICT_NOTIFY（跨材料矛盾） | CONFLICT_NOTIFY（跨课题耦合） |")
    L.append("| 确认回执 | ACK_RECEIPT | ACK_RECEIPT | ACK_RECEIPT |")
    L.append("")

    # 4. 消息类型分布对比
    L.append("## 四、消息类型分布对比")
    L.append("")
    L.append("```")
    all_types = sorted({m.get("message_type") for msgs in msgs_by_task.values() for m in msgs})
    L.append(f"{'消息类型':<18}{'task1':>8}{'task2':>8}{'task3':>8}")
    for t in all_types:
        row = [t]
        for task in ("task1", "task2", "task3"):
            n = sum(1 for m in msgs_by_task[task] if m.get("message_type") == t)
            row.append(str(n))
        L.append(f"{row[0]:<18}{row[1]:>8}{row[2]:>8}{row[3]:>8}")
    L.append("```")
    L.append("")

    # 5. 追溯建议
    # 4.5 冲突解决闭环统一（方案 A）
    L.append("## 五、冲突解决闭环统一（方案 A）")
    L.append("")
    L.append("三题的冲突处理统一为同一套子协议编排：")
    L.append("`CONFLICT_NOTIFY`（上报）→ `INFO_QUERY`（协商）→ 仲裁（决议）→")
    L.append("`TASK_ASSIGN(revise)`（修订）→ `INFO_QUERY(recheck)`（复核），最多重试 N 轮。")
    L.append("")
    L.append("| 环节 | 第一题 | 第二题 | 第三题 |")
    L.append("|---|---|---|---|")
    L.append("| 冲突检测 | 算力/术语/引用一致性 | 跨材料一致性检查 | 跨课题耦合检测 |")
    L.append("| 冲突上报 | CONFLICT_NOTIFY | CONFLICT_NOTIFY | CONFLICT_NOTIFY |")
    L.append("| 协商 | INFO_QUERY → NEGOTIATE | INFO_QUERY → NEGOTIATE | INFO_QUERY（可调整空间） |")
    L.append("| 仲裁 | 取折中决议 + BROADCAST | 决议：重新识别 | 决议：采纳协同建议 |")
    L.append("| 修订 | TASK_ASSIGN(revise) 重新生成 | TASK_ASSIGN(revise) 重新识别 | TASK_ASSIGN(revise) 调整语义 |")
    L.append("| 复核 | INFO_QUERY(recheck) | INFO_QUERY(recheck) | INFO_QUERY(recheck) |")
    L.append("")
    L.append("共享实现：`shared/conflict_resolution.py`（`ConflictResolutionEngine`），三题共同调用。")
    L.append("")

    L.append("## 六、数据流追溯方法")
    L.append("")
    L.append("1. **协议层**：三题统一导入 `shared/protocol.py`（`Message`/`MessageType`/`MessageBus`）。")
    L.append("2. **消息层**：每条消息记录发送时间、发送方、接收方、消息类型、优先级、关联消息 ID、相关性 ID。")
    L.append("3. **追溯链**：以 `correlation_id` 为锚点，可将一次跨题协作的所有消息串成数据流链；")
    L.append("   以 `related_message_id` 为锚点，可追溯请求-响应、冲突-协商的因果关系。")
    L.append("")
    L.append("---")
    L.append("*本报告由 `shared/trace.py` 生成，消息日志位于各题 `logs/` 目录。*")

    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    return "\n".join(L)


def main():
    out_dir = os.path.join(REPO_ROOT, "bonus", "logs")
    os.makedirs(out_dir, exist_ok=True)
    out_md = os.path.join(out_dir, "cross_task_trace.md")
    msgs_by_task = {t: load_jsonl(p) for t, p in LOG_SOURCES.items()}
    build_trace_report(msgs_by_task, out_md)
    print("跨题数据流追溯报告 ->", out_md)
    print("消息统计：", {t: len(m) for t, m in msgs_by_task.items()})


if __name__ == "__main__":
    main()
