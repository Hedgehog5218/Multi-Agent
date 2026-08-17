# -*- coding: utf-8 -*-
# 多智能体基础组件：结构化消息、日志器、智能体基类。
# 跨题联动（加分项）：本模块直接复用第一题的通信协议实现 shared/protocol.py
# （与 task1/protocol.py 同源同版本），保证第二/三题的智能体通信与第一题
# 使用同一套消息格式与消息类型，便于跨题数据流追溯。

import os
import sys
import time
import json
import uuid

# 把仓库根目录加入 sys.path，使 task3 能导入 shared.protocol（第一题协议同源）
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from shared.protocol import Message as ProtocolMessage
from shared.protocol import MessageType as _ProtocolMessageType
from shared.protocol import Priority, now_iso


class MessageType:
    """消息类型枚举（与第一题协议枚举值完全一致；ACK 为 ACK_RECEIPT 的向后兼容别名）。

    注意：这里的成员值是 shared.protocol.MessageType 的枚举成员，
    因此可无缝传入第一题协议的 Message 构造器，保证跨题消息格式统一。
    """
    TASK_ASSIGN = _ProtocolMessageType.TASK_ASSIGN
    INFO_QUERY = _ProtocolMessageType.INFO_QUERY
    RESULT_SUBMIT = _ProtocolMessageType.RESULT_SUBMIT
    CONFLICT_NOTIFY = _ProtocolMessageType.CONFLICT_NOTIFY
    ACK = _ProtocolMessageType.ACK_RECEIPT          # 兼容旧命名：ACK == ACK_RECEIPT
    ACK_RECEIPT = _ProtocolMessageType.ACK_RECEIPT
    NEGOTIATE = _ProtocolMessageType.NEGOTIATE
    BROADCAST = _ProtocolMessageType.BROADCAST


class Message:
    """结构化消息：字段与第一题通信协议完全一致（message_id/类型/发送方/接收方/
    载荷/时间戳/优先级/关联消息ID/相关性ID/会话ID/协议版本/TTL/是否需要回执）。

    底层直接复用 shared.protocol.Message（第一题协议同源），保证三题消息格式统一。
    """
    def __init__(self, msg_type, sender, receiver, payload, priority=1,
                 ref_msg_id="", msg_id=None, timestamp=None, session_id="task3-cross-group"):
        # 统一消息类型：兼容字符串与枚举
        if isinstance(msg_type, str):
            msg_type = _ProtocolMessageType(msg_type)
        # 统一优先级：1/2/3 -> LOW/NORMAL/HIGH（与第一题协议一致）
        if isinstance(priority, int):
            priority = {1: Priority.LOW, 2: Priority.NORMAL, 3: Priority.HIGH}.get(priority, Priority.NORMAL)
        self._proto = ProtocolMessage(
            message_type=msg_type,
            sender=sender,
            receiver=receiver,
            payload=payload,
            message_id=msg_id or None,
            timestamp=timestamp or now_iso(),
            priority=priority,
            related_message_id=ref_msg_id or None,
            session_id=session_id,
        )
        self.msg_id = self._proto.message_id          # 兼容旧字段名
        self.msg_type = self._proto.message_type      # 兼容旧字段名（枚举）
        self.sender = sender
        self.receiver = receiver
        self.payload = payload
        self.priority = self._proto.priority
        self.ref_msg_id = ref_msg_id
        self.timestamp = self._proto.timestamp
        self.correlation_id = self._proto.correlation_id
        self.session_id = self._proto.session_id

    def summary(self):
        """消息体摘要（用于日志）"""
        return self.payload.get("summary", "")

    def to_dict(self):
        """输出协议消息字典（字段与第一题 task1/logs/messages.jsonl 一致）"""
        return self._proto.to_dict()

    def to_json(self):
        return self._proto.to_json()


class Logger:
    """运行日志：完整日志写入文件与内存；终端只显示简洁进度（console）。

    跨题联动：每条协议消息额外落盘为 logs/protocol_messages.jsonl，
    字段与 task1 的 messages.jsonl 一致，供 shared/trace.py 做跨题数据流追溯。
    """
    def __init__(self, path=None, echo=False):
        self.path = path
        self.echo = echo          # echo=True 时 console() 会打印到终端
        self.entries = []
        self.protocol_messages = []
        # 协议消息 JSONL 路径：与运行日志同目录
        self.protocol_jsonl_path = None
        if path:
            self.protocol_jsonl_path = os.path.join(os.path.dirname(path), "protocol_messages.jsonl")

    def log(self, level, source, text):
        """完整日志（写入文件，不打印终端）"""
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] [{level}] [{source}] {text}"
        self.entries.append(line)

    def console(self, text):
        """简洁的终端进度提示（不写入日志文件）"""
        if self.echo:
            print(text, flush=True)

    def message(self, msg, direction="send"):
        """记录一条协议消息：文本日志 + 协议 JSONL（跨题追溯用）。

        与第一题一致：每条逻辑消息只在发送方落盘一次（JSONL 一条记录），
        receive 方向仅写文本日志，避免同一消息重复计数。
        """
        if direction == "send":
            self.log("INFO", msg.sender,
                     f"发送消息 {msg.msg_id} [{msg.msg_type.value}] -> {msg.receiver} | 优先级={msg.priority.value} | {msg.summary()}")
        else:
            self.log("INFO", msg.receiver,
                     f"收到消息 {msg.msg_id} [{msg.msg_type.value}] 来自 {msg.sender} | {msg.summary()}")
            return  # 接收方向不写协议 JSONL（避免重复）
        # 协议消息 JSONL（字段与第一题 task1/logs/messages.jsonl 一致）
        self.protocol_messages.append(msg)
        if self.protocol_jsonl_path:
            os.makedirs(os.path.dirname(self.protocol_jsonl_path), exist_ok=True)
            d = msg.to_dict()
            entry = {
                "seq": len(self.protocol_messages),
                "send_time": d.get("timestamp"),
                "sender": d.get("sender"),
                "receiver": d.get("receiver"),
                "message_type": d.get("message_type", "").value if hasattr(d.get("message_type"), "value") else d.get("message_type"),
                "priority": d.get("priority", "").value if hasattr(d.get("priority"), "value") else d.get("priority"),
                "related_message_id": d.get("related_message_id"),
                "correlation_id": d.get("correlation_id"),
                "payload_summary": msg.summary()[:90],
            }
            with open(self.protocol_jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def dump(self):
        text = "\n".join(self.entries) + "\n"
        if self.path:
            with open(self.path, "w", encoding="utf-8") as f:
                f.write(text)
        return text


class Agent:
    """智能体基类：负责消息的发送/接收与日志记录（协议消息统一落盘）"""
    def __init__(self, name, group, logger):
        self.name = name
        self.group = group
        self.logger = logger
        self.inbox = []
        self.outbox = []

    def send(self, receiver, msg_type, payload, priority=1, ref_msg_id="", session_id="task3-cross-group"):
        msg = Message(msg_type=msg_type, sender=self.name, receiver=receiver,
                      payload=payload, priority=priority, ref_msg_id=ref_msg_id,
                      session_id=session_id)
        self.outbox.append(msg)
        self.logger.message(msg, "send")
        return msg

    def receive(self, msg):
        self.inbox.append(msg)
        self.logger.message(msg, "receive")

    def reply_ack(self, msg):
        self.send(msg.sender, MessageType.ACK,
                  {"summary": f"确认收到 {msg.msg_id}"}, priority=1,
                  ref_msg_id=msg.msg_id)
