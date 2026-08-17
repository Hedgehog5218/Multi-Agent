# -*- coding: utf-8 -*-
"""方法设计智能体：撰写研究方法与技术路线（正文由 DeepSeek 生成）。

部门私有口径：训练算力需求 3000 GPU·小时；作业完成时间指标称 makespan。
"""
from __future__ import annotations

from typing import Any, Dict

from protocol import Message, MessageType, Priority
from agents import content
from agents.base import AgentBase


class MethodAgent(AgentBase):
    agent_id = "method"
    display_name = "方法设计智能体"
    role = "撰写研究方法与技术路线"

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
        """响应协调器协商查询：给出本部门主张与可调整空间。"""
        q = msg.payload.get("query", "")
        if q == "gpu_hours":
            proposal = {
                "current": content.AGENT_FACTS["method"]["gpu_hours_claim"],
                "can_reduce_to": 2200,
                "optimization": "共享 Critic + 混合精度 + 精简消融实验",
            }
        elif q == "terminology":
            proposal = {"unify_to": "平均作业完成时间(JCT)"}
        else:
            proposal = {}
        return Message(
            message_type=MessageType.RESULT_SUBMIT,
            sender=self.agent_id,
            receiver=msg.sender,
            payload={"query": q, "proposal": proposal},
            priority=Priority.NORMAL,
            correlation_id=msg.correlation_id,
            session_id=msg.session_id,
            ack_required=False,
        )

    def _draft(self) -> None:
        # 方法侧算力需求按部门口径申报，并持久化到章节 claims（供后续修订读取）
        expected = content.AGENT_FACTS["method"]["gpu_hours_claim"]
        body = self.llm.generate_section_checked(self.agent_id, "research_content",
                                                 expected_gpu=expected)
        self._write_with_lock("research_content", body,
                              {"gpu_hours_claim": expected},
                              title=content.SECTION_TASKS["research_content"]["title"])
        body = self.llm.generate_section_checked(self.agent_id, "technical_route",
                                                 no_gpu=True)
        self._write_with_lock("technical_route", body, {},
                              title=content.SECTION_TASKS["technical_route"]["title"])

    def _revise(self, task: Dict[str, Any]) -> None:
        """按协调器决议重新生成（累积式：算力值取"当前 claims 与决议"合并结果）。"""
        decisions = task.get("decisions", {})
        current = (self.read_section("research_content") or {}).get("claims", {})
        gpu = decisions.get("gpu_hours_claim", current.get("gpu_hours_claim"))
        extra: list = []
        if "gpu_hours_claim" in decisions:
            extra.append(f"本章节中的训练算力需求统一改写为 {gpu} GPU·小时。")
        if "metric_terms" in decisions:
            extra.append(f"作业完成时间指标统一写『{decisions['metric_terms'][0]}』，"
                         f"不得再出现 makespan。")
        # 只重写受影响的研究内容章节（技术路线不含算力/术语约束，无需重写）
        body = self.llm.generate_section_checked(self.agent_id, "research_content",
                                                 expected_gpu=gpu,
                                                 extra_constraints=extra)
        self._write_with_lock("research_content", body,
                              {"gpu_hours_claim": gpu},
                              title=content.SECTION_TASKS["research_content"]["title"])

    def _write_with_lock(self, section_id: str, body: str,
                         claims: Dict[str, Any], title: str) -> None:
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
