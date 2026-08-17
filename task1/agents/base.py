# -*- coding: utf-8 -*-
"""智能体基类：统一的收发消息、回执与任务处理接口。

所有子智能体（文献调研/方法设计/实验规划/数据核查/统稿润色）与协调器
都继承自 `AgentBase`，从而保证：
1. 消息经 `self.bus.send()` 统一发送（星型拓扑、统一日志）；
2. 收到消息后统一返回回执（ACK_RECEIPT），满足协议可靠投递；
3. 智能体之间不直接引用彼此对象，仅通过消息与共享黑板交互，
   符合"子智能体经协调器转发、不直接通信"的拓扑设计。
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from protocol import Message, MessageType, Priority, now_iso

logger = logging.getLogger("agents")


class AgentBase:
    agent_id: str = "agent"
    display_name: str = "智能体"

    def __init__(self, workspace, llm=None, runtime=None) -> None:
        self.workspace = workspace
        self.llm = llm
        self.runtime = runtime  # 可选：运行时上下文（协调器使用）
        self.bus: Any = None
        self._log = []

    # ---- 消息收发 ------------------------------------------------------
    def send(self, msg_type: MessageType, receiver: str, payload: Dict[str, Any],
             priority: Priority = Priority.NORMAL,
             related_message_id: Optional[str] = None,
             correlation_id: Optional[str] = None,
             session_id: str = "proposal-writing",
             ack_required: bool = True) -> Optional[Message]:
        msg = Message(
            message_type=msg_type,
            sender=self.agent_id,
            receiver=receiver,
            payload=payload,
            priority=priority,
            related_message_id=related_message_id,
            correlation_id=correlation_id,
            session_id=session_id,
            ack_required=ack_required,
        )
        return self.bus.send(msg)

    def reply_ack(self, msg: Message, extra: Optional[Dict[str, Any]] = None,
                  priority: Priority = Priority.HIGH) -> Message:
        """对收到的消息返回确认回执。"""
        payload = {"ack_for": msg.message_id, "status": "ok"}
        if extra:
            payload.update(extra)
        return Message(
            message_type=MessageType.ACK_RECEIPT,
            sender=self.agent_id,
            receiver=msg.sender,
            payload=payload,
            priority=priority,
            related_message_id=msg.message_id,
            correlation_id=msg.correlation_id,
            session_id=msg.session_id,
            ack_required=False,
        )

    def on_broadcast(self, msg: Message) -> Message:
        """接收广播：默认记录回执，子类可覆盖以响应阶段广播。"""
        return self.reply_ack(msg, {"status": "received"})

    def handle_message(self, msg: Message) -> Optional[Message]:
        """消息分派：子类重写 `on_*` 方法即可。"""
        handler = getattr(self, f"on_{msg.message_type.value.lower()}", None)
        if handler is not None:
            return handler(msg)
        # 未知类型：返回回执并记录
        logger.warning("%s 收到未处理消息 %s", self.agent_id, msg.message_type)
        return self.reply_ack(msg)

    # ---- 工具 ----------------------------------------------------------
    def read_section(self, section_id: str) -> Optional[Dict[str, Any]]:
        return self.workspace.read(section_id)

    def write_section(self, section_id: str, content: str,
                      claims: Optional[Dict[str, Any]] = None) -> bool:
        """加锁 -> 校验版本 -> 写入 -> 释放锁（同步与并发控制演示）。"""
        acquired = self.workspace.acquire(section_id, self.agent_id)
        if not acquired:
            return False
        try:
            base = self.workspace.version(section_id)
            ok = self.workspace.write(section_id, self.agent_id, content,
                                      claims=claims, base_version=base)
            return ok
        finally:
            self.workspace.release(section_id, self.agent_id)
