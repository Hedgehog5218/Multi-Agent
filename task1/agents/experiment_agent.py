# -*- coding: utf-8 -*-
"""实验规划智能体：设计实验方案、评估指标与预期结果（正文由 DeepSeek 生成）。

部门私有口径：可用算力预算 2000 GPU·小时；作业完成时间指标称 JCT。
"""
from __future__ import annotations

from typing import Any, Dict

from protocol import Message, MessageType, Priority
from agents import content
from agents.base import AgentBase


class ExperimentAgent(AgentBase):
    agent_id = "experiment"
    display_name = "实验规划智能体"
    role = "设计实验方案、评估指标与预期结果"

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
        q = msg.payload.get("query", "")
        if q == "gpu_budget":
            proposal = {
                "current": content.AGENT_FACTS["experiment"]["gpu_budget"],
                "can_increase_to": 2200,
                "optimization": "降低重复实验次数、提前终止、混合精度训练",
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
        # 实验侧算力预算按部门口径申报，并持久化到章节 claims
        expected = content.AGENT_FACTS["experiment"]["gpu_budget"]
        body = self.llm.generate_section_checked(self.agent_id, "experiment_plan",
                                                 expected_gpu=expected)
        self._write_with_lock("experiment_plan", body,
                              {"gpu_budget": expected},
                              title=content.SECTION_TASKS["experiment_plan"]["title"])
        body = self.llm.generate_section_checked(self.agent_id, "expected_outcomes",
                                                 no_gpu=True)
        self._write_with_lock("expected_outcomes", body, {},
                              title=content.SECTION_TASKS["expected_outcomes"]["title"])

    def _revise(self, task: Dict[str, Any]) -> None:
        """按协调器决议重新生成。

        只处理算力决议：术语决议（C2）下实验方案本来就统一用 JCT，无需重写；
        只重写实验方案（预期成果不含算力数字，无需重写）。
        """
        decisions = task.get("decisions", {})
        if "gpu_budget" not in decisions:
            return
        gpu = decisions["gpu_budget"]
        extra = [f"本章节中的训练算力预算统一改写为 {gpu} GPU·小时。"]
        body = self.llm.generate_section_checked(self.agent_id, "experiment_plan",
                                                 expected_gpu=gpu,
                                                 extra_constraints=extra)
        self._write_with_lock("experiment_plan", body,
                              {"gpu_budget": gpu},
                              title=content.SECTION_TASKS["experiment_plan"]["title"])

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
