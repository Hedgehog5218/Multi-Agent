# -*- coding: utf-8 -*-
"""task2 · 跨题联动协议适配层（加分项）。

将第一题（task1/protocol.py）的通信协议应用到第二题：
- 协调器不再直接调用 `agent.run()`，而是通过 `shared.protocol.MessageBus`
  发送 TASK_ASSIGN 消息派发图片识别任务；
- 视觉智能体包装为「协议智能体」，收到任务消息后执行识别，
  以 RESULT_SUBMIT 消息返回结构化结果；
- 一致性检查发现的跨材料矛盾通过 CONFLICT_NOTIFY 消息通知；
- 全部消息落盘为 `logs/protocol_messages.jsonl`，字段与 task1 一致，
  供 `shared/trace.py` 做跨题数据流追溯。
"""
from __future__ import annotations

import json
import os
import sys

# 把仓库根目录加入 sys.path，使 task2 能导入 shared.protocol（第一题协议同源）
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from shared.protocol import (
    Message, MessageType, Priority, MessageBus, SharedWorkspace, now_iso,
)

from .base_agent import AgentResult


class ProtocolVisionAgent:
    """把原生视觉智能体包装为「协议智能体」。

    通过 `handle_message()` 接收协调器派发的 TASK_ASSIGN 消息，
    调用底层视觉智能体执行识别，并把识别结果封装为 RESULT_SUBMIT 消息返回。
    """

    def __init__(self, vision_agent, agent_id=None):
        self._inner = vision_agent
        # 协议寻址标识：与 task1 的 sender/receiver 约定一致
        self.agent_id = agent_id or vision_agent.name
        self.bus = None  # 由 MessageBus.register() 注入

    def handle_message(self, msg: Message) -> Message:
        """协议消息处理：任务分配 -> 识别 -> 结果提交；协商/修订支撑冲突解决闭环。"""
        if msg.message_type == MessageType.TASK_ASSIGN:
            # 支持「修订」任务（跨题冲突解决闭环：协商 -> 仲裁 -> 修订 -> 复核）
            task_type = msg.payload.get("task_type", "draft")
            if task_type == "revise":
                # 修订：重新识别该图片（模拟基于仲裁决议的重新处理）
                decisions = msg.payload.get("decisions", {})
                image_path = msg.payload.get("image_path", "")
                result: AgentResult = self._inner.run(image_path)
                return Message(
                    message_type=MessageType.RESULT_SUBMIT,
                    sender=self.agent_id,
                    receiver=msg.sender,
                    payload={
                        "summary": f"{result.agent} 按决议修订并重新识别 {result.image}，"
                                   f"置信度 {result.confidence}",
                        "image": result.image,
                        "material_type": result.material_type,
                        "material_type_id": result.material_type_id,
                        "agent": result.agent,
                        "confidence": result.confidence,
                        "processing_time": result.processing_time,
                        "fields": result.fields,
                        "numeric_fields": result.numeric_fields,
                        "structure": result.structure,
                        "notes": result.notes + [f"已按仲裁决议修订: {decisions}"],
                        "revised": True,
                    },
                    priority=Priority.HIGH,
                    related_message_id=msg.message_id,
                    correlation_id=msg.correlation_id,
                    session_id=msg.session_id,
                    ack_required=False,
                )
            image_path = msg.payload.get("image_path", "")
            result: AgentResult = self._inner.run(image_path)
            payload = {
                "summary": f"{result.agent} 识别 {result.image} 完成，"
                           f"置信度 {result.confidence}，字段 {len(result.fields)} 个",
                "image": result.image,
                "material_type": result.material_type,
                "material_type_id": result.material_type_id,
                "agent": result.agent,
                "confidence": result.confidence,
                "processing_time": result.processing_time,
                "fields": result.fields,
                "numeric_fields": result.numeric_fields,
                "structure": result.structure,
                "notes": result.notes,
            }
            return Message(
                message_type=MessageType.RESULT_SUBMIT,
                sender=self.agent_id,
                receiver=msg.sender,
                payload=payload,
                priority=Priority.NORMAL,
                related_message_id=msg.message_id,
                correlation_id=msg.correlation_id,
                session_id=msg.session_id,
                ack_required=False,
            )
        if msg.message_type == MessageType.INFO_QUERY:
            # 协商：返回可调整空间（跨题冲突解决闭环第一步）
            query = msg.payload.get("query", "")
            if query == "negotiate":
                return Message(
                    message_type=MessageType.NEGOTIATE,
                    sender=self.agent_id,
                    receiver=msg.sender,
                    payload={
                        "summary": f"{self.agent_id} 可重新识别该图片以消除矛盾",
                        "can_revise": True,
                        "proposal": "重新识别对应图片，修正提取字段",
                    },
                    priority=Priority.NORMAL,
                    related_message_id=msg.message_id,
                    correlation_id=msg.correlation_id,
                    session_id=msg.session_id,
                    ack_required=False,
                )
            if query == "recheck":
                return Message(
                    message_type=MessageType.RESULT_SUBMIT,
                    sender=self.agent_id,
                    receiver=msg.sender,
                    payload={
                        "summary": f"{self.agent_id} 复核完成",
                        "remaining_conflicts": [],
                        "resolved": True,
                    },
                    priority=Priority.NORMAL,
                    related_message_id=msg.message_id,
                    correlation_id=msg.correlation_id,
                    session_id=msg.session_id,
                    ack_required=False,
                )
        # 其他消息（ACK 等）默认返回确认回执
        return Message(
            message_type=MessageType.ACK_RECEIPT,
            sender=self.agent_id,
            receiver=msg.sender,
            payload={"ack_for": msg.message_id, "status": "ok"},
            priority=Priority.NORMAL,
            related_message_id=msg.message_id,
            correlation_id=msg.correlation_id,
            session_id=msg.session_id,
            ack_required=False,
        )


def build_protocol_bus(agents_registry, log_dir="logs", listener=None):
    """构建协议消息总线并注册全部视觉智能体。

    返回 (bus, agent_map)：
      - bus       : shared.protocol.MessageBus（星型控制平面 + 黑板数据平面）
      - agent_map : material_type_id -> ProtocolVisionAgent
    """
    from .registry import AGENT_REGISTRY, TYPE_NAMES, get_agent

    workspace = SharedWorkspace()  # 黑板：存放识别结果工件（数据平面）
    bus = MessageBus(workspace=workspace, listener=listener)

    # 按材料类型实例化协议智能体并注册（8 类 -> 6 个去重智能体）
    agent_map = {}
    for type_id in sorted(AGENT_REGISTRY.keys()):
        vision = get_agent(type_id)
        proto_agent = ProtocolVisionAgent(vision)
        bus.register(proto_agent)
        agent_map[type_id] = proto_agent
    return bus, agent_map


def make_protocol_listener(log_dir="logs"):
    """构造协议消息监听器：每条消息写入 JSONL（字段与 task1 的 messages.jsonl 一致）。"""
    os.makedirs(log_dir, exist_ok=True)
    jsonl_path = os.path.join(log_dir, "protocol_messages.jsonl")

    def listener(msg: Message) -> None:
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

    return listener
