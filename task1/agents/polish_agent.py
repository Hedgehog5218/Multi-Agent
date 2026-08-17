# -*- coding: utf-8 -*-
"""统稿润色智能体：统一行文风格、格式规范、术语一致性。

职责（对应 1.3 阶段 5 最终统稿）：
- 从黑板读取全部章节草稿与参考文献；
- 统一术语（例如把残余的 makespan 统一为「平均作业完成时间(JCT)」）；
- 按标准章节顺序合并生成格式统一的 Markdown 基金申请书；
- 输出最终文件并统计章节数/字数/术语一致性。
"""
from __future__ import annotations

import os
import re
from typing import Any, Dict

from protocol import Message, MessageType, Priority
from agents import content
from agents.base import AgentBase

# 术语统一映射：把同义异构术语归一（出现多次时合并到主术语）
TERMINOLOGY_MAP = {
    "makespan": "平均作业完成时间(JCT)",
    "平均作业完成时间（JCT）": "平均作业完成时间(JCT)",
    "作业平均完成时间（JCT）": "平均作业完成时间(JCT)",
    "JCT（作业平均完成时间）": "平均作业完成时间(JCT)",
}


class PolishAgent(AgentBase):
    agent_id = "polish"
    display_name = "统稿润色智能体"
    role = "统一行文风格、格式规范、术语一致性"

    def __init__(self, workspace, llm=None, output_dir: str = "logs") -> None:
        super().__init__(workspace, llm=llm)
        self.output_dir = output_dir

    def on_task_assign(self, msg: Message) -> Message:
        task = msg.payload
        task_type = task.get("type", "merge")
        if task_type == "merge":
            result = self._merge_proposal()
            return Message(
                message_type=MessageType.RESULT_SUBMIT,
                sender=self.agent_id,
                receiver=msg.sender,
                payload=result,
                priority=Priority.NORMAL,
                correlation_id=msg.correlation_id,
                session_id=msg.session_id,
                ack_required=False,
            )
        return self.reply_ack(msg, {"status": "unknown_task"})

    def on_broadcast(self, msg: Message) -> Message:
        # 记录阶段广播（如冲突解决决议、任务完成）
        return self.reply_ack(msg, {"status": "received"})

    def _merge_proposal(self) -> Dict[str, Any]:
        os.makedirs(self.output_dir, exist_ok=True)
        sections = []
        for sid in content.SECTION_ORDER:
            sec = self.read_section(sid) or {}
            body = sec.get("content", "")
            # 术语统一
            body = self._unify_terminology(body)
            sections.append(f"# {sec.get('title', sid)}\n\n{body}")
        refs = self.read_section("references") or {}
        ref_body = self._unify_terminology(refs.get("content", ""))
        sections.append(f"# {refs.get('title', '参考文献')}\n\n{ref_body}")

        header = (
            f"# {content.PROPOSAL_TITLE}\n\n"
            "> 多智能体协同写作系统输出 · 最终统稿版\n"
            f"> 生成时间：{self._now_ts()}\n\n---\n"
        )
        full_md = header + "\n\n".join(sections) + "\n"
        out_path = os.path.join(self.output_dir, "final_proposal.md")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(full_md)

        stats = {
            "output_path": out_path,
            "char_count": len(full_md),
            "section_count": len(content.SECTION_ORDER) + 1,
            "terminology_unified": self._terminology_stats(full_md),
            "status": "ok",
        }
        return stats

    @staticmethod
    def _now_ts() -> str:
        from datetime import datetime, timezone
        return datetime.now(timezone.utc).isoformat(timespec="seconds")

    @staticmethod
    def _unify_terminology(text: str) -> str:
        for src, dst in TERMINOLOGY_MAP.items():
            text = re.sub(re.escape(src), dst, text)
        return text

    @staticmethod
    def _terminology_stats(text: str) -> Dict[str, int]:
        return {
            "JCT_occurrences": len(re.findall(r"JCT", text)),
            "makespan_leftover": len(re.findall(r"makespan", text, flags=re.IGNORECASE)),
        }
