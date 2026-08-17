# -*- coding: utf-8 -*-
"""数据/逻辑核查智能体：核实数据一致性、公式正确性、引用准确性。

与旧版的关键区别：**不再依赖任何预埋的结构化声明(claims)**，而是直接从
各智能体生成的正文中提取"事实"（算力数字、指标术语、文献引用），再做
跨章节一致性检查 —— 冲突是各章节由不同智能体独立生成而自然涌现的。

检测规则：
  R1 算力一致性：研究内容中的算力需求 == 实验方案中的算力预算（单位 GPU·小时）
  R2 术语一致性：作业完成时间指标在全文使用同一叫法（JCT / makespan 不混用）
  R3 引用一致性：正文引用的每条文献编号都出现在参考文献表中
"""
from __future__ import annotations

import re
from typing import Any, Dict, List

from protocol import Message, MessageType, Priority
from agents.base import AgentBase

# 涉及术语/算力检查的章节
_GPU_RE = re.compile(r"(\d+)\s*GPU\s*[·.]?\s*小时")
_REF_RE = re.compile(r"\[(\d+)\]")


def extract_gpu_hours(text: str) -> List[int]:
    """从正文提取形如 '3000 GPU·小时' 的算力数字（按出现顺序）。"""
    return [int(m) for m in _GPU_RE.findall(text)]


def extract_ref_keys(text: str) -> List[str]:
    """从正文提取文献引用编号，如 '[7]'。"""
    return [f"[{m}]" for m in _REF_RE.findall(text)]


class VerifierAgent(AgentBase):
    agent_id = "verifier"
    display_name = "数据/逻辑核查智能体"
    role = "核实数据一致性、公式正确性、引用准确性"

    def __init__(self, workspace, llm=None) -> None:
        super().__init__(workspace, llm=llm)
        self._last_conflicts: List[Dict[str, Any]] = []

    def on_task_assign(self, msg: Message) -> Message:
        task_type = msg.payload.get("type", "review")
        if task_type == "review":
            conflicts = self._review_all()
            for c in conflicts:
                self.send(MessageType.CONFLICT_NOTIFY, "coordinator",
                          {"conflict": c}, priority=Priority.HIGH)
        return Message(
            message_type=MessageType.RESULT_SUBMIT,
            sender=self.agent_id,
            receiver=msg.sender,
            payload={"task": task_type, "status": "ok",
                     "conflicts_found": len(self._last_conflicts)},
            priority=Priority.NORMAL,
            correlation_id=msg.correlation_id,
            session_id=msg.session_id,
            ack_required=False,
        )

    def on_info_query(self, msg: Message) -> Message:
        q = msg.payload.get("query", "")
        if q == "recheck":
            remaining = self._review_all()
            return Message(
                message_type=MessageType.RESULT_SUBMIT,
                sender=self.agent_id,
                receiver=msg.sender,
                payload={"query": "recheck", "remaining_conflicts": remaining,
                         "resolved": len(remaining) == 0},
                priority=Priority.HIGH,
                correlation_id=msg.correlation_id,
                session_id=msg.session_id,
                ack_required=False,
            )
        return self.reply_ack(msg)

    # ------------------------------------------------------------------
    def _review_all(self) -> List[Dict[str, Any]]:
        conflicts: List[Dict[str, Any]] = []
        text = {sid: (self.read_section(sid) or {}).get("content", "")
                for sid in ("project_basis", "research_content", "technical_route",
                            "experiment_plan", "expected_outcomes", "references")}
        conflicts += self._check_gpu_consistency(text)
        conflicts += self._check_terminology(text)
        conflicts += self._check_citations(text)
        self._last_conflicts = conflicts
        return conflicts

    def _check_gpu_consistency(self, text: Dict[str, str]) -> List[Dict[str, Any]]:
        """R1：方法侧（研究内容+技术路线）申报的算力总量应与实验侧
        （实验方案+预期成果）的算力预算一致。

        每侧取所有 GPU·小时 数字的最大值作为"总量申报"
        （总量是最大的那个，分解子项较小）。
        """
        method_nums = (extract_gpu_hours(text["research_content"])
                       + extract_gpu_hours(text["technical_route"]))
        exp_nums = (extract_gpu_hours(text["experiment_plan"])
                    + extract_gpu_hours(text["expected_outcomes"]))
        if method_nums and exp_nums:
            claim, budget = max(method_nums), max(exp_nums)
            if claim != budget:
                return [{
                    "conflict_id": "C1-gpu-budget",
                    "rule": "R1 算力一致性",
                    "severity": "CRITICAL",
                    "description": (f"方法侧申报算力需求为 {claim} GPU·小时，"
                                    f"而实验侧预算为 {budget} GPU·小时，两者不一致"),
                    "evidence": {"claim": claim, "budget": budget},
                    "involved": ["method", "experiment"],
                }]
        # 单侧缺失：方法侧未声明算力 或 实验侧未声明预算
        if not method_nums and exp_nums:
            return [{
                "conflict_id": "C1-gpu-budget",
                "rule": "R1 算力一致性",
                "severity": "CRITICAL",
                "description": "方法侧（研究内容/技术路线）未声明训练算力需求，"
                               "无法与实验侧预算核对",
                "evidence": {"claim": None, "budget": max(exp_nums)},
                "involved": ["method", "experiment"],
            }]
        if method_nums and not exp_nums:
            return [{
                "conflict_id": "C1-gpu-budget",
                "rule": "R1 算力一致性",
                "severity": "CRITICAL",
                "description": "实验侧（实验方案/预期成果）未声明算力预算，"
                               "无法与方法侧算力需求核对",
                "evidence": {"claim": max(method_nums), "budget": None},
                "involved": ["method", "experiment"],
            }]
        return []

    def _check_terminology(self, text: Dict[str, str]) -> List[Dict[str, Any]]:
        """R2：作业完成时间指标不得混用 makespan 与 JCT。"""
        rc_ep = (text["research_content"] + text["experiment_plan"]).lower()
        has_makespan = "makespan" in rc_ep
        has_jct = "jct" in rc_ep
        if has_makespan and has_jct:
            return [{
                "conflict_id": "C2-terminology",
                "rule": "R2 术语一致性",
                "severity": "HIGH",
                "description": "同一指标（作业完成时间）在研究内容与实验方案中"
                               "混用了 makespan 与 JCT 两种叫法",
                "evidence": {"has_makespan": has_makespan, "has_jct": has_jct},
                "involved": ["method", "experiment"],
            }]
        return []

    def _check_citations(self, text: Dict[str, str]) -> List[Dict[str, Any]]:
        """R3：正文引用的每条文献编号都应在参考文献表中。"""
        cited = set(extract_ref_keys(text["project_basis"]))
        refs = set(extract_ref_keys(text["references"]))
        missing = sorted(cited - refs)
        if missing:
            return [{
                "conflict_id": "C3-citation",
                "rule": "R3 引用一致性",
                "severity": "HIGH",
                "description": f"立项依据正文引用 {missing}，但参考文献表中缺失",
                "evidence": {"missing": missing},
                "involved": ["literature"],
            }]
        return []
