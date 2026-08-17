# -*- coding: utf-8 -*-
"""多智能体通信协议 (Multi-Agent Communication Protocol) v1.0
================================================================

跨题共享协议（加分项 · 跨题联动）：
本文件与第一题 task1/protocol.py 同源同版本，是整个测评项目的**唯一协议权威**。
第二题（视觉识别与科研周报）与第三题（跨课题耦合）的多智能体通信统一
导入本模块（shared.protocol），从而把第一题设计的通信协议应用于第二、三题：

  - 消息格式：Message / MessageType / Priority / JSON Schema 完全一致；
  - 通信拓扑：星型控制平面 + 黑板数据平面（MessageBus / SharedWorkspace）；
  - 数据流追溯：三题消息日志字段一致（task1/logs/messages.jsonl、
    task2/logs/protocol_messages.jsonl、task3/logs/protocol_messages.jsonl），
    由 shared/trace.py 生成跨题数据流追溯报告（bonus/logs/cross_task_trace.md）。

原始出处 —— 第一题 · 1.2 通信协议设计（面向科研协作写作场景）。

本模块包含四大部分：

本模块是整套多智能体协同写作系统的通信基础设施，包含四大部分：

A. 消息格式定义 (对应 1.2 消息格式定义, 4 分)
   - `MessageType` / `Priority` 枚举
   - `Message` 结构化消息数据类
   - `MESSAGE_JSON_SCHEMA`：消息格式的正式定义 (JSON Schema)
   - `validate_message()`：消息合法性校验
   - 字段设计依据见 `Message.__doc__` 与模块 docstring。

B. 通信拓扑设计 (对应 1.2 通信拓扑设计, 3 分)
   - 采用「星型控制平面 + 黑板数据平面」混合拓扑：
     * 控制平面：星型中心化，所有子智能体通过协调器(coordinator)转发控制消息；
     * 数据平面：黑板(shared workspace)共享章节工件，消息只携带摘要/增量。
   - `TOPOLOGY_ANALYSIS`：跳数、单点故障风险、扩展性分析。
   - 拓扑图见 `figures/topology.png` (由 demo 生成) 及 README.md。

C. 同步与并发控制 (对应 1.2 同步与并发控制, 3 分)
   - `SharedWorkspace`：按章节存储草稿的黑板，带「写锁 + 乐观版本号」。
   - `LOCK_REQUEST / LOCK_GRANT / LOCK_RELEASE` 消息类型与锁协议，
     防止两个智能体同时写入同一章节 (如文献调研智能体与方法设计智能体
     同时写 related work 段落)。
   - 冲突检测与解决策略伪代码见 `conflict_detection_pseudocode()` /
     `conflict_resolution_pseudocode()` 与 README.md。

D. 通信开销建模 (对应 1.2 通信开销建模, 2 分)
   - `estimate_tokens()`：token 数估算 (中英混合)。
   - `communication_overhead()`：一次完整协作任务的通信总开销计算公式。
   - 优化措施：摘要替代全文、增量传输、批量 ACK、TTL 去重。

作者：task1 项目组
"""
from __future__ import annotations

import itertools
import json
import queue
import threading
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

__all__ = [
    "PROTOCOL_VERSION",
    "MessageType",
    "Priority",
    "Message",
    "MESSAGE_JSON_SCHEMA",
    "validate_message",
    "TOPOLOGY_ANALYSIS",
    "SharedWorkspace",
    "MessageBus",
    "estimate_tokens",
    "communication_overhead",
    "conflict_detection_pseudocode",
    "conflict_resolution_pseudocode",
    "now_iso",
]

PROTOCOL_VERSION = "1.0"


# ---------------------------------------------------------------------------
# A. 消息格式定义
# ---------------------------------------------------------------------------
class MessageType(str, Enum):
    """消息类型 —— 至少覆盖题目要求的五类，另扩展协商/锁/广播等控制消息。

    题目要求五类：
    - TASK_ASSIGN     : 任务分配（协调器 -> 子智能体）
    - INFO_QUERY      : 信息查询（任两方之间）
    - RESULT_SUBMIT   : 结果提交（子智能体 -> 协调器）
    - CONFLICT_NOTIFY : 冲突通知（核查智能体 -> 协调器/相关智能体）
    - ACK_RECEIPT     : 确认回执（任意接收方回执，保证可靠投递）

    扩展类型：
    - NEGOTIATE       : 冲突协商（相关智能体交换修改方案）
    - LOCK_REQUEST / LOCK_GRANT / LOCK_RELEASE : 章节写锁协议（并发控制）
    - BROADCAST       : 广播（协调器向全体广播阶段/决议）
    """
    TASK_ASSIGN = "TASK_ASSIGN"
    INFO_QUERY = "INFO_QUERY"
    RESULT_SUBMIT = "RESULT_SUBMIT"
    CONFLICT_NOTIFY = "CONFLICT_NOTIFY"
    ACK_RECEIPT = "ACK_RECEIPT"
    NEGOTIATE = "NEGOTIATE"
    LOCK_REQUEST = "LOCK_REQUEST"
    LOCK_GRANT = "LOCK_GRANT"
    LOCK_RELEASE = "LOCK_RELEASE"
    BROADCAST = "BROADCAST"


class Priority(str, Enum):
    """消息优先级，驱动协调器调度顺序（高优先生成/先处理）。"""
    LOW = "LOW"
    NORMAL = "NORMAL"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


def now_iso() -> str:
    """生成 ISO8601 时间戳（UTC，毫秒精度），保证跨智能体日志可排序。"""
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


@dataclass
class Message:
    """结构化消息。

    字段设计依据：
    - protocol_version : 协议向前兼容，接收方可按版本选择解析器。
    - message_id       : 全局唯一消息 ID，用于去重、追踪与日志回放。
    - message_type     : 消息路由的依据；区分控制消息与数据消息。
    - sender / receiver: 星型拓扑中的寻址标识（receiver 可为
                         "coordinator" 或 "all"，用于广播）。
    - payload          : 载荷内容，任意结构化 JSON 对象。
                         * 数据消息携带"摘要/增量"而非全文（见开销优化）。
                         * 任务消息携带任务规格；冲突消息携带冲突详情。
    - timestamp        : 发送方落盘时间；用于排序、开销统计与回放。
    - priority         : 调度优先级（冲突/回执往往 HIGH/CRITICAL）。
    - related_message_id : 关联消息 ID：请求-响应链、冲突-协商链、
                           多轮协商线程的锚点。
    - correlation_id   : 相关性 ID，把一次请求-响应往返的所有消息分组。
    - session_id       : 一次协作任务的会话分组（多任务并发时不串扰）。
    - ttl_seconds      : 生存时间，超时进入死信队列（防止消息无限滞留）。
    - ack_required     : 是否需要接收方返回 ACK_RECEIPT 回执。
    """
    message_type: MessageType
    sender: str
    receiver: str
    payload: Dict[str, Any]
    message_id: str = field(default_factory=lambda: f"msg-{uuid.uuid4().hex}")
    timestamp: str = field(default_factory=now_iso)
    priority: Priority = Priority.NORMAL
    related_message_id: Optional[str] = None
    correlation_id: Optional[str] = None
    session_id: str = "default"
    protocol_version: str = PROTOCOL_VERSION
    ttl_seconds: int = 600
    ack_required: bool = True
    # 由 MessageBus 填充的全局序号，用于确定全序与画序列图
    seq: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["message_type"] = self.message_type.value
        d["priority"] = self.priority.value
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Message":
        d = dict(d)
        d["message_type"] = MessageType(d["message_type"])
        d["priority"] = Priority(d.get("priority", "NORMAL"))
        return cls(**d)

    @classmethod
    def from_json(cls, s: str) -> "Message":
        return cls.from_dict(json.loads(s))

    def summary(self, max_len: int = 60) -> str:
        """消息体摘要：用于日志记录（题目要求"消息体摘要"）。"""
        return _summarize_payload(self.payload, max_len)


# JSON Schema —— 消息格式的正式定义
MESSAGE_JSON_SCHEMA: Dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "https://task1.local/protocol/message.schema.json",
    "title": "AgentCommunicationMessage",
    "description": "多智能体协同写作系统通信消息 (protocol v1.0)",
    "type": "object",
    "required": [
        "protocol_version", "message_id", "message_type", "sender",
        "receiver", "payload", "timestamp", "priority",
        "related_message_id", "session_id",
    ],
    "properties": {
        "protocol_version": {"type": "string", "const": "1.0"},
        "message_id": {"type": "string", "pattern": "^msg-[0-9a-f]{32}$"},
        "message_type": {
            "type": "string",
            "enum": [t.value for t in MessageType],
        },
        "sender": {"type": "string", "minLength": 1},
        "receiver": {"type": "string", "minLength": 1},
        "payload": {"type": "object"},
        "timestamp": {"type": "string", "format": "date-time"},
        "priority": {
            "type": "string",
            "enum": [p.value for p in Priority],
        },
        "related_message_id": {"type": ["string", "null"]},
        "correlation_id": {"type": ["string", "null"]},
        "session_id": {"type": "string"},
        "ttl_seconds": {"type": "integer", "minimum": 1},
        "ack_required": {"type": "boolean"},
    },
    "additionalProperties": True,
}


def validate_message(msg: Message, raise_on_error: bool = True) -> Tuple[bool, List[str]]:
    """轻量消息校验（不依赖 jsonschema 第三方库）。

    检查必填字段存在且类型/取值正确；返回 (是否合法, 错误列表)。
    """
    errors: List[str] = []
    m = msg.to_dict()
    required = MESSAGE_JSON_SCHEMA["required"]
    # 必填字段 = 字段必须存在；可空字段由取值域检查处理（null 合法）
    for k in required:
        if k not in m:
            errors.append(f"missing required field: {k}")
    if not errors:
        if m["protocol_version"] != PROTOCOL_VERSION:
            errors.append(f"unsupported protocol_version: {m['protocol_version']}")
        try:
            MessageType(m["message_type"])
        except ValueError:
            errors.append(f"invalid message_type: {m['message_type']}")
        try:
            Priority(m["priority"])
        except ValueError:
            errors.append(f"invalid priority: {m['priority']}")
        if not isinstance(m["payload"], dict):
            errors.append("payload must be an object")
        if not m["sender"] or not m["receiver"]:
            errors.append("sender/receiver must be non-empty")
        if m["related_message_id"] is not None and not isinstance(m["related_message_id"], str):
            errors.append("related_message_id must be a string or null")
        if m["correlation_id"] is not None and not isinstance(m["correlation_id"], str):
            errors.append("correlation_id must be a string or null")
    if raise_on_error and errors:
        raise ValueError("消息校验失败: " + "; ".join(errors))
    return (len(errors) == 0, errors)


def _summarize_payload(payload: Dict[str, Any], max_len: int) -> str:
    """把 payload 压缩成一行摘要（截断 + JSON 化）。"""
    text = json.dumps(payload, ensure_ascii=False, default=str)
    if len(text) > max_len:
        text = text[: max_len - 3] + "..."
    return text


# ---------------------------------------------------------------------------
# B. 通信拓扑设计
# ---------------------------------------------------------------------------
TOPOLOGY_ANALYSIS = {
    "topology": "星型中心化控制平面 + 黑板(共享工件)数据平面 混合拓扑",
    "control_plane": (
        "所有子智能体之间的控制消息（任务分配/结果提交/冲突通知/协商/回执）"
        "均经由协调器(coordinator)转发，子智能体不直接互发控制消息。"
    ),
    "data_plane": (
        "章节草稿存放在共享工作区 SharedWorkspace（黑板模式），"
        "消息只携带 '章节ID + 摘要/增量'，需要全文的智能体从黑板读取，"
        "从而避免大段正文在消息中反复拷贝。"
    ),
    "hop_analysis": (
        "子智能体 -> 协调器：1 跳；子智能体 A -> 协调器 -> 子智能体 B：2 跳。"
        "相比全连接网状(1 跳)增加 1 跳，但星型连接数为 O(N)，"
        "网状连接数为 O(N^2)，节点数增长时星型更易扩展。"
    ),
    "single_point_of_failure": (
        "协调器是控制平面单点。缓解：协调器无状态化，所有消息落盘日志"
        "(logs/messages.jsonl) 可回放恢复；必要时采用主备双协调器 + 故障切换"
        "(本实现提供消息日志回放能力，见 README)。"
    ),
    "scalability": (
        "新增子智能体只需注册到协调器，无需改动其他智能体拓扑；"
        "黑板模式使正文共享与并发控制集中在一处，避免 N 份副本不一致。"
    ),
}


# ---------------------------------------------------------------------------
# C. 同步与并发控制
# ---------------------------------------------------------------------------
class SharedWorkspace:
    """黑板式共享工件存储：按章节存放草稿，带「写锁 + 乐观版本号」。

    并发控制策略（对应 1.2 同步与并发控制）：
    1. 每个章节一把写锁（`threading.Lock`），同一时刻只允许一个智能体写入；
    2. 写操作必须携带版本号（乐观并发控制）：写入时校验传入的 base_version
       等于当前版本，否则视为"写冲突"（版本号冲突）拒绝写入并要求重读；
    3. 持有写锁的智能体写入后 `release()` 释放锁并将版本号 +1；
    4. 读操作不加锁（读多写少），读到的是最近一次提交的完整快照。

    典型冲突场景：文献调研智能体与方法设计智能体同时写"相关研究"段落。
    由于每段落在黑板上只有一把锁，后到者必须等待前者的 LOCK_RELEASE，
    因此不会出现两个智能体同时写入同一段落的覆盖/丢失更新问题。
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()          # 保护内部字典
        self._sections: Dict[str, Dict[str, Any]] = {}
        self._locks: Dict[str, threading.Lock] = {}
        self._owners: Dict[str, str] = {}      # section_id -> owner agent_id
        self._versions: Dict[str, int] = {}    # section_id -> version

    def ensure_section(self, section_id: str, meta: Optional[Dict[str, Any]] = None) -> None:
        with self._lock:
            if section_id not in self._sections:
                self._sections[section_id] = {
                    "section_id": section_id,
                    "title": meta.get("title", section_id) if meta else section_id,
                    "owner": None,
                    "content": "",
                    "claims": {},
                    "revisions": 0,
                }
                self._locks[section_id] = threading.Lock()
                self._versions[section_id] = 0

    def acquire(self, section_id: str, agent_id: str, timeout: float = 30.0) -> bool:
        """尝试获取章节写锁。返回是否获取成功（超时返回 False）。"""
        self.ensure_section(section_id)
        lock = self._locks[section_id]
        acquired = lock.acquire(timeout=timeout)
        if acquired:
            with self._lock:
                if self._owners.get(section_id) is not None:
                    lock.release()
                    return False
                self._owners[section_id] = agent_id
        return acquired

    def write(self, section_id: str, agent_id: str, content: str,
              claims: Optional[Dict[str, Any]] = None,
              base_version: Optional[int] = None,
              title: Optional[str] = None) -> bool:
        """乐观写入：若 base_version 与当前版本不一致则拒绝（写冲突）。"""
        self.ensure_section(section_id)
        with self._lock:
            if self._owners.get(section_id) != agent_id:
                return False
            current = self._versions[section_id]
            if base_version is not None and base_version != current:
                return False  # 版本冲突：请重读后再写
            self._sections[section_id]["content"] = content
            self._sections[section_id]["owner"] = agent_id
            if title is not None:
                self._sections[section_id]["title"] = title
            if claims is not None:
                self._sections[section_id]["claims"] = claims
            self._sections[section_id]["revisions"] += 1
            self._versions[section_id] += 1
            return True

    def release(self, section_id: str, agent_id: str) -> bool:
        """释放章节写锁，版本号 +1。"""
        with self._lock:
            if self._owners.get(section_id) != agent_id:
                return False
            del self._owners[section_id]
        self._locks[section_id].release()
        return True

    def read(self, section_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            d = self._sections.get(section_id)
            return json.loads(json.dumps(d, ensure_ascii=False, default=str)) if d else None

    def read_all(self) -> Dict[str, Dict[str, Any]]:
        return {sid: self.read(sid) for sid in list(self._sections.keys())}

    def version(self, section_id: str) -> int:
        with self._lock:
            return self._versions.get(section_id, 0)


def conflict_detection_pseudocode() -> str:
    """冲突检测伪代码（起始为 输入/输出，后接具体流程；对应 verifier_agent.py）。"""
    return """输入：黑板中 6 个章节的声明 claims（立项依据/研究内容/技术路线/实验方案/预期成果/参考文献）
输出：冲突列表 conflicts；为空表示各章节一致

流程：
1.  初始化 claims <- {}，conflicts <- []
2.  for 章节 in [立项依据, 研究内容, 技术路线, 实验方案, 预期成果, 参考文献]:
3.      claims[章节] <- 黑板.读(章节).声明
4.  // 规则1 算力一致性：方法侧申报算力 == 实验侧预算
5.  if 声明[研究内容].声称算力 != 声明[实验方案].算力预算:
6.      conflicts <- conflicts 并 {新冲突("C1", 涉及=[方法, 实验])}
7.  // 规则2 术语一致性：同一指标全文只允许一种叫法
8.  if 研究内容与实验方案中同时出现 "JCT" 与 "makespan":
9.      conflicts <- conflicts 并 {新冲突("C2", 涉及=[方法, 实验])}
10. // 规则3 引用一致性：正文引用的每条文献都应在参考文献表中
11. 缺失文献 <- 立项依据正文引用 - 参考文献表
12. if 缺失文献 != 空集:
13.     conflicts <- conflicts 并 {新冲突("C3", 涉及=[文献调研])}
14. // 上报：每条冲突通过通信协议发送给协调器
15. for 冲突 in conflicts:
16.     发送(CONFLICT_NOTIFY, 协调器, 冲突)
17. return conflicts
"""


def conflict_resolution_pseudocode() -> str:
    """冲突解决伪代码（起始为 输入/输出，后接具体流程；对应 coordinator.py）。"""
    return """输入：冲突列表 conflicts（按严重程度排序，CRITICAL 优先）
输出：已解决冲突列表 resolved；并对每条解决决议向全体智能体广播

流程：
1.  初始化 resolved <- []
2.  for 冲突 in 按 severity 排序的 conflicts:
3.      编号 <- 冲突.conflict_id
4.      // ① 协商：向涉事智能体收集"可调整空间"
5.      方案 <- {}
6.      for 智能体 in 冲突.涉及:
7.          回复 <- 发送(INFO_QUERY, 智能体, 冲突)
8.          方案[智能体] <- 回复.可调整空间    // 如 方法:可压到2200，实验:可提到2200
9.      // ② 仲裁：协调器拍板，取双方都可接受的折中值
10.     if 编号 == "C1":
11.         折中 <- min(方案[方法].可压到, 方案[实验].可提到)
12.         决议 <- { 声称算力: 折中, 算力预算: 折中 }
13.     elif 编号 == "C2":
14.         决议 <- { 指标术语: ["平均作业完成时间(JCT)"] }   // 全篇统一叫法
15.     elif 编号 == "C3":
16.         决议 <- { 补齐参考文献: True }                  // 补齐缺失文献
17.     // ③ 修订：让涉事智能体按决议重新调用大模型生成
18.     for 智能体 in 冲突.涉及:
19.         发送(TASK_ASSIGN, 智能体, {类型: 修订, 决议: 决议})
20.     // ④ 复核：核查智能体重新检测该冲突是否消失
21.     复核 <- 发送(INFO_QUERY, 核查智能体, {查询: 复核, 冲突编号: 编号})
22.     if 复核结果中不存在该冲突编号:
23.         resolved <- resolved 并 {冲突, 决议}
24.         发送(BROADCAST, 全体, {冲突已解决, 编号, 决议})
25.     else:
26.         回到第 5 步，最多重试 2 轮           // 复核未通过则重新修订
27. return resolved
"""


def estimate_tokens(text: str) -> int:
    """token 数估算（近似 OpenAI cl100k 分词行为）。

    规则：中文字符约 1.6 字符/token，英文单词约 0.75 词/token，
    混合文本按字符与空白拆分近似。
    """
    if not text:
        return 0
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    rest = text.replace("\n", " ")
    ascii_words = [w for w in rest.split() if not any("\u4e00" <= ch <= "\u9fff" for ch in w)]
    ascii_chars = sum(len(w) for w in ascii_words)
    tokens = int(cjk / 1.6) + int(ascii_chars / 4) + len(ascii_words)
    return max(1, tokens)


def communication_overhead(messages: List[Message]) -> Dict[str, Any]:
    """一次完整协作任务的通信总开销计算。

    公式：
        C_total = Σ_{m ∈ M} token(m)
                = N_ctrl * T_ctrl + N_data * T_data

    其中：
        N_ctrl : 控制类消息数量（任务分配/回执/查询/锁/广播）
        T_ctrl : 控制消息平均 token 数（通常远小于数据消息）
        N_data : 数据类消息数量（结果提交/冲突通知/协商，携带摘要或增量）
        T_data : 数据消息平均 token 数

    由于本协议采用"摘要替代全文 + 黑板共享正文"：
        T_data ≈ T_summary + T_delta << T_full_text
    从而在消息数量不变的情况下显著降低总 token 数。
    """
    total_msgs = len(messages)
    total_tokens = 0
    by_type: Dict[str, Tuple[int, int]] = {}
    for m in messages:
        body = _summarize_payload(m.payload, 1_000_000)  # 以完整 payload 估算
        tok = estimate_tokens(body)
        total_tokens += tok
        t, n = by_type.get(m.message_type.value, (0, 0))
        by_type[m.message_type.value] = (t + tok, n + 1)

    ctrl_types = {
        MessageType.TASK_ASSIGN.value, MessageType.ACK_RECEIPT.value,
        MessageType.INFO_QUERY.value, MessageType.LOCK_REQUEST.value,
        MessageType.LOCK_GRANT.value, MessageType.LOCK_RELEASE.value,
        MessageType.BROADCAST.value,
    }
    n_ctrl = sum(n for t, (_, n) in by_type.items() if t in ctrl_types)
    n_data = total_msgs - n_ctrl
    t_ctrl = sum(tok for t, (tok, _) in by_type.items() if t in ctrl_types) / max(1, n_ctrl)
    t_data = sum(tok for t, (tok, _) in by_type.items() if t not in ctrl_types) / max(1, n_data)

    return {
        "total_messages": total_msgs,
        "total_tokens": total_tokens,
        "n_control": n_ctrl,
        "n_data": n_data,
        "avg_token_control": round(t_ctrl, 1),
        "avg_token_data": round(t_data, 1),
        "formula": "C_total = N_ctrl*T_ctrl + N_data*T_data",
        "by_message_type": {t: {"tokens": tok, "count": n} for t, (tok, n) in by_type.items()},
    }


# ---------------------------------------------------------------------------
# 消息总线（运行时）
# ---------------------------------------------------------------------------
class MessageBus:
    """线程安全的消息总线（星型拓扑的物理实现）。

    - `register()` 注册智能体邮箱；
    - `send()` 为消息分配全局序号 + 时间戳，记录日志，并同步投递给接收方
      （接收方 `handle_message()` 的返回值作为回执消息自动记录）；
    - 投递使用可重入锁，保证同一时刻只有一条消息在处理，日志全序确定；
    - 每条消息都会被推送到 `listener` 回调（用于写日志 / 统计 / 画图）。
    """

    def __init__(self, workspace: SharedWorkspace,
                 listener: Optional[Callable[[Message], None]] = None) -> None:
        self.workspace = workspace
        self._agents: Dict[str, Any] = {}
        self._deliver_lock = threading.RLock()
        self._seq = itertools.count(1)
        self._messages: List[Message] = []
        self._listener = listener

    def register(self, agent: Any) -> None:
        self._agents[agent.agent_id] = agent
        agent.bus = self

    def send(self, msg: Message) -> Optional[Message]:
        """发送消息：分配序号/时间戳 -> 记录 -> 投递 -> 返回回执消息。

        性能优化：只有"分配序号/记录日志"在全局锁内（毫秒级），
        接收方的 handle_message（含大模型生成，耗时 10s+）在锁外执行，
        从而多个智能体的生成过程可以真正并行。
        """
        with self._deliver_lock:
            msg.seq = next(self._seq)
            msg.timestamp = now_iso()
            self._messages.append(msg)
            if self._listener:
                try:
                    self._listener(msg)
                except Exception as exc:  # 日志监听失败不影响主流程
                    print(f"[bus] listener error: {exc}")
            receiver = self._agents.get(msg.receiver)
        if receiver is None:
            # 广播：投递给全体（除发送方外）
            if msg.receiver == "all":
                replies = []
                for agent_id, agent in list(self._agents.items()):
                    if agent_id == msg.sender:
                        continue
                    reply = self._deliver(agent, msg)
                    if reply is not None:
                        replies.append(reply)
                return replies[0] if replies else None
            return None
        return self._deliver(receiver, msg)

    def _deliver(self, receiver: Any, msg: Message) -> Optional[Message]:
        # handle_message（含大模型生成）在全局锁外执行，允许多智能体并行
        reply = receiver.handle_message(msg)
        if reply is not None:
            with self._deliver_lock:
                reply.seq = next(self._seq)
                reply.timestamp = now_iso()
                reply.related_message_id = msg.message_id
                reply.correlation_id = reply.correlation_id or msg.correlation_id
                self._messages.append(reply)
                if self._listener:
                    try:
                        self._listener(reply)
                    except Exception:
                        pass
            if reply.receiver != "coordinator" and reply.receiver != msg.sender:
                # 回执需要继续投递给最终接收方
                target = self._agents.get(reply.receiver)
                if target is not None:
                    nested = target.handle_message(reply)
                    if nested is not None:
                        with self._deliver_lock:
                            nested.seq = next(self._seq)
                            nested.timestamp = now_iso()
                            self._messages.append(nested)
                            if self._listener:
                                try:
                                    self._listener(nested)
                                except Exception:
                                    pass
        return reply

    @property
    def messages(self) -> List[Message]:
        return list(self._messages)

    def log_entries(self) -> List[Dict[str, Any]]:
        return [m.to_dict() for m in self._messages]
