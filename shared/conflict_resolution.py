# -*- coding: utf-8 -*-
"""跨题共享的冲突解决子协议（加分项 · 跨题联动 · 方案 A）。

设计目标：
  把第一题「协商 → 仲裁 → 修订 → 复核」四步闭环从 task1/coordinator.py 中
  抽象为**跨题可复用**的冲突解决引擎，让第二题（视觉一致性）与第三题
  （跨课题耦合）也走同一套闭环，从而做到三题在冲突解决层面真正统一。

子协议消息编排（与 shared/protocol.py 的消息类型一致）：
  1. CONFLICT_NOTIFY  检测方 → 协调器：上报冲突
  2. INFO_QUERY       协调器 → 涉事方：协商（问可调整空间）
  3. NEGOTIATE / RESULT_SUBMIT  涉事方 → 协调器：提交可调整空间
  4. BROADCAST        协调器 → 全体：广播仲裁决议
  5. TASK_ASSIGN(type=revise)   协调器 → 涉事方：按决议修订（重新生成/识别/解析）
  6. INFO_QUERY(recheck)        协调器 → 检测方：复核冲突是否消失
  7. 若未消失 → 回到 2，最多 max_rounds 轮

用法（每题一个适配器，注入四个回调）：
  engine = ConflictResolutionEngine(
      negotiate=...,      # (conflict) -> {agent_id: opinion}
      arbitrate=...,      # (conflict, opinions) -> decision
      revise=...,         # (conflict, decision) -> None
      recheck=...,        # (conflict) -> remaining conflicts 列表
  )
  result = engine.resolve(conflict)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from shared.protocol import Message, MessageType, Priority


@dataclass
class Conflict:
    """跨题统一的冲突对象。

    三题上报的冲突都归一化为该结构，便于复用同一套解决闭环：
      conflict_id : 冲突编号（如 C1-gpu-budget / V2-lidar / D01-v100）
      rule        : 触发规则名
      severity    : CRITICAL / HIGH / MEDIUM / LOW
      description : 冲突描述
      evidence    : 证据（数值/原文等）
      involved    : 涉事智能体 id 列表
      category    : 可选，冲突类别（如 资源可用性/新颖性/技术路线）
    """
    conflict_id: str
    rule: str
    severity: str
    description: str
    involved: List[str]
    evidence: Dict[str, Any] = field(default_factory=dict)
    category: str = ""


class ConflictResolutionEngine:
    """通用冲突解决引擎：协商 → 仲裁 → 修订 → 复核，最多 max_rounds 轮。

    通过注入四个回调与具体系统解耦：
      negotiate : (conflict) -> {agent_id: opinion}          # 协商
      arbitrate : (conflict, opinions) -> decision           # 仲裁
      revise    : (conflict, decision) -> None               # 修订
      recheck   : (conflict) -> remaining_conflicts 列表      # 复核
    """

    def __init__(
        self,
        negotiate: Callable[[Conflict], Dict[str, Any]],
        arbitrate: Callable[[Conflict, Dict[str, Any]], Dict[str, Any]],
        revise: Callable[[Conflict, Dict[str, Any]], None],
        recheck: Callable[[Conflict], List[Any]],
        max_rounds: int = 2,
        on_event: Optional[Callable[[str, str], None]] = None,
    ):
        self.negotiate = negotiate
        self.arbitrate = arbitrate
        self.revise = revise
        self.recheck = recheck
        self.max_rounds = max(1, max_rounds)
        self.on_event = on_event  # (阶段名, 描述) -> None，用于日志/追溯

    def _log(self, phase: str, text: str) -> None:
        if self.on_event:
            self.on_event(phase, text)

    def resolve(self, conflict: Conflict) -> Dict[str, Any]:
        """对单个冲突执行完整闭环，返回解决结果。"""
        self._log("冲突解决", f"开始解决 {conflict.conflict_id}（{conflict.severity}，涉及 {conflict.involved}）")
        last_remaining: List[Any] = []
        for round_no in range(1, self.max_rounds + 1):
            self._log("协商", f"第 {round_no} 轮：向 {conflict.involved} 协商可调整空间")
            opinions = self.negotiate(conflict) or {}

            self._log("仲裁", f"第 {round_no} 轮：协调器仲裁，意见={opinions}")
            decision = self.arbitrate(conflict, opinions) or {}

            self._log("修订", f"第 {round_no} 轮：按决议 {decision} 修订")
            self.revise(conflict, decision)

            self._log("复核", f"第 {round_no} 轮：复核 {conflict.conflict_id} 是否消失")
            last_remaining = self.recheck(conflict) or []
            if len(last_remaining) == 0:
                self._log("冲突解决", f"{conflict.conflict_id} 已解决（第 {round_no} 轮），决议={decision}")
                return {
                    "conflict_id": conflict.conflict_id,
                    "resolved": True,
                    "rounds": round_no,
                    "decision": decision,
                    "remaining": [],
                }
            self._log("冲突解决", f"{conflict.conflict_id} 第 {round_no} 轮仍未解决，剩余 {len(last_remaining)} 条，重试")
        self._log("冲突解决", f"{conflict.conflict_id} 达到最大轮数仍未解决，标记为未解决")
        return {
            "conflict_id": conflict.conflict_id,
            "resolved": False,
            "rounds": self.max_rounds,
            "decision": None,
            "remaining": last_remaining,
        }


def broadcast_decision(bus, decision: Dict[str, Any], conflict: Conflict,
                       session_id: str = "default") -> None:
    """通过协议 BROADCAST 消息广播仲裁决议（三题通用）。"""
    if bus is None:
        return
    bus.send(Message(
        message_type=MessageType.BROADCAST,
        sender="coordinator",
        receiver="all",
        payload={
            "phase": "conflict_resolved",
            "conflict_id": conflict.conflict_id,
            "decision": decision,
            "summary": f"冲突 {conflict.conflict_id} 仲裁决议：{decision}",
        },
        priority=Priority.HIGH,
        session_id=session_id,
    ))


def send_notify(bus, conflict: Conflict, sender: str = "verifier",
                session_id: str = "default") -> None:
    """通过协议 CONFLICT_NOTIFY 消息上报冲突（三题通用）。"""
    if bus is None:
        return
    bus.send(Message(
        message_type=MessageType.CONFLICT_NOTIFY,
        sender=sender,
        receiver="coordinator",
        payload={
            "conflict": {
                "conflict_id": conflict.conflict_id,
                "rule": conflict.rule,
                "severity": conflict.severity,
                "description": conflict.description,
                "evidence": conflict.evidence,
                "involved": conflict.involved,
            },
            "summary": conflict.description[:80],
        },
        priority=Priority.HIGH,
        session_id=session_id,
    ))


def resolution_pseudocode() -> str:
    """冲突解决闭环伪代码（对应三题共同流程，供报告/文档使用）。"""
    return """输入：冲突 conflict（含涉事方、严重程度、证据）
输出：解决结果 resolved

流程：
1.  检测方 发送 CONFLICT_NOTIFY -> 协调器（上报冲突）
2.  协调器 发送 INFO_QUERY -> 每个涉事方（协商：问可调整空间）
3.  各涉事方 回复 NEGOTIATE / RESULT_SUBMIT（可调整空间，如 方法可压到2200、实验可提到2200）
4.  协调器 仲裁：取各方可接受的折中值 -> decision
5.  协调器 发送 BROADCAST -> 全体（广播仲裁决议）
6.  协调器 发送 TASK_ASSIGN(type=revise, decisions) -> 涉事方（修订：重新生成/识别/解析）
7.  协调器 发送 INFO_QUERY(recheck) -> 检测方（复核冲突是否消失）
8.  if 复核结果为空: return {resolved: True, decision}
9.  else: 回到第 2 步，最多重试 max_rounds 轮
10. return {resolved: False, remaining}
"""
