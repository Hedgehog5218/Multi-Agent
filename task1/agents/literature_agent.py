# -*- coding: utf-8 -*-
"""文献调研智能体：检索相关文献，提炼研究背景与 related work。

- 《立项依据》正文由 DeepSeek 从零生成（知识库提供写作任务与文献口径）；
- 《参考文献》表基于调研资料库整理（初始调研范围缺 [7]，用于引用一致性冲突演示）。
"""
from __future__ import annotations

from typing import Any, Dict

from protocol import Message, MessageType, Priority
from agents import content
from agents.base import AgentBase


class LiteratureAgent(AgentBase):
    agent_id = "literature"
    display_name = "文献调研智能体"
    role = "检索相关文献，提炼研究背景与 related work"

    def on_task_assign(self, msg: Message) -> Message:
        task_type = msg.payload.get("type", "draft")
        if task_type == "draft":
            self._draft()
        elif task_type == "revise":
            self._revise(msg.payload)
        return Message(
            message_type=MessageType.RESULT_SUBMIT,
            sender=self.agent_id,
            receiver=msg.sender,
            payload={"task": task_type, "status": "ok"},
            priority=Priority.NORMAL,
            correlation_id=msg.correlation_id,
            session_id=msg.session_id,
            ack_required=False,
        )

    def on_info_query(self, msg: Message) -> Message:
        """响应协调器关于文献/引用的查询。"""
        section = self.read_section("references") or {}
        return Message(
            message_type=MessageType.RESULT_SUBMIT,
            sender=self.agent_id,
            receiver=msg.sender,
            payload={
                "query": msg.payload.get("query", ""),
                "reference_keys": section.get("claims", {}).get("reference_keys", []),
            },
            priority=Priority.NORMAL,
            correlation_id=msg.correlation_id,
            session_id=msg.session_id,
            ack_required=False,
        )

    def _draft(self) -> None:
        # 立项依据正文：由 DeepSeek 生成
        body = self.llm.generate_section(self.agent_id, "project_basis")
        self._write_with_lock("project_basis", body, {},
                              title=content.SECTION_TASKS["project_basis"]["title"])
        # 参考文献表：基于当前调研资料（初始缺 [7]）
        keys = content.AGENT_FACTS["literature"]["available_reference_keys"]
        ref_body = content.build_references(keys)
        self._write_with_lock("references", ref_body, {"reference_keys": keys},
                              title="参考文献")

    def _revise(self, task: Dict[str, Any]) -> None:
        """引用一致性冲突解决：补齐调研资料中缺失的文献 [7]。"""
        if task.get("revise_references"):
            keys = content.all_reference_keys()
            body = content.build_references(keys)
            self._write_with_lock("references", body, {"reference_keys": keys},
                                  title="参考文献")

    def _write_with_lock(self, section_id: str, body: str,
                         claims: Dict[str, Any], title: str) -> None:
        """带写锁协议的章节写入。"""
        grant = self.send(MessageType.LOCK_REQUEST, "coordinator",
                          {"section_id": section_id, "intent": "write"},
                          priority=Priority.NORMAL)
        if grant is None or grant.message_type != MessageType.LOCK_GRANT:
            raise RuntimeError(f"获取章节 {section_id} 写锁失败")
        try:
            base = self.workspace.version(section_id)
            ok = self.workspace.write(section_id, self.agent_id, body,
                                      claims=claims, base_version=base, title=title)
            if not ok:
                raise RuntimeError(f"写入章节 {section_id} 失败（版本冲突）")
        finally:
            self.send(MessageType.LOCK_RELEASE, "coordinator",
                      {"section_id": section_id}, priority=Priority.LOW)
