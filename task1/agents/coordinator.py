# -*- coding: utf-8 -*-
"""协调器智能体：任务分解、调度、冲突仲裁与结果汇总。

对应 1.3 端到端演示的五阶段流程：
  阶段1 任务分解  : 把申请书拆分为章节，广播分配方案；
  阶段2 并行起草  : 三个起草智能体在独立线程并行撰写初稿；
  阶段3 交叉审查  : 数据核查智能体审阅各章节，通过协议发送冲突通知；
  阶段4 冲突解决  : 协调器仲裁，调度涉事智能体协商并修订，核查智能体复核；
  阶段5 最终统稿  : 统稿润色智能体合并生成统一格式的 Markdown 申请书。

协调器同时扮演星型拓扑的中心节点：
- 处理子智能体发来的 LOCK_REQUEST / LOCK_RELEASE（章节写锁协议）；
- 收集子智能体发来的 CONFLICT_NOTIFY（冲突通知）。
"""
from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

from protocol import (Message, MessageType, Priority, SharedWorkspace,
                      communication_overhead)
from agents.base import AgentBase
from agents import content


class CoordinatorAgent(AgentBase):
    agent_id = "coordinator"
    display_name = "协调器"
    role = "任务分解、调度、冲突仲裁与结果汇总"

    # 章节 -> 负责智能体 的映射（任务分解结果）
    SECTION_PLAN = {
        "project_basis": "literature",
        "references": "literature",
        "research_content": "method",
        "technical_route": "method",
        "experiment_plan": "experiment",
        "expected_outcomes": "experiment",
    }
    DRAFT_GROUPS = {
        "literature": ["project_basis", "references"],
        "method": ["research_content", "technical_route"],
        "experiment": ["experiment_plan", "expected_outcomes"],
    }

    def __init__(self, workspace: SharedWorkspace, llm=None) -> None:
        super().__init__(workspace, llm=llm)
        self._pending_conflicts: List[Dict[str, Any]] = []
        self._conflicts_lock = threading.Lock()
        self._phase_log: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # 控制消息处理（被其他智能体调用）
    # ------------------------------------------------------------------
    def on_lock_request(self, msg: Message) -> Message:
        section_id = msg.payload.get("section_id")
        acquired = self.workspace.acquire(section_id, msg.sender, timeout=0)
        if acquired:
            return Message(
                message_type=MessageType.LOCK_GRANT,
                sender=self.agent_id,
                receiver=msg.sender,
                payload={"section_id": section_id, "status": "granted"},
                priority=Priority.NORMAL,
                correlation_id=msg.correlation_id,
                session_id=msg.session_id,
                ack_required=False,
            )
        return Message(
            message_type=MessageType.ACK_RECEIPT,
            sender=self.agent_id,
            receiver=msg.sender,
            payload={"section_id": section_id, "status": "busy"},
            priority=Priority.HIGH,
            correlation_id=msg.correlation_id,
            session_id=msg.session_id,
            ack_required=False,
        )

    def on_lock_release(self, msg: Message) -> Message:
        self.workspace.release(msg.payload.get("section_id"), msg.sender)
        return self.reply_ack(msg)

    def on_conflict_notify(self, msg: Message) -> Message:
        conflict = msg.payload.get("conflict", {})
        with self._conflicts_lock:
            self._pending_conflicts.append(conflict)
        return self.reply_ack(msg, {"received": conflict.get("conflict_id")})

    def on_ack_receipt(self, msg: Message) -> Message:
        return None  # 回执无需再回执

    # ------------------------------------------------------------------
    # 五阶段主流程
    # ------------------------------------------------------------------
    def run_demo(self, session_id: str = "proposal-writing") -> Dict[str, Any]:
        t0 = time.time()
        self._phase("1 任务分解", f"将申请书拆分为 {len(self.SECTION_PLAN)} 个章节并分配")

        plan = {
            "proposal": content.PROPOSAL_TITLE,
            "sections": self.SECTION_PLAN,
            "draft_groups": self.DRAFT_GROUPS,
        }
        self.send(MessageType.BROADCAST, "all",
                  {"phase": "decompose", "plan": plan},
                  priority=Priority.HIGH, session_id=session_id)

        # ---- 阶段2：并行起草 ----
        self._phase("2 并行起草", "文献/方法/实验三个智能体并行撰写初稿")
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = []
            for agent_id, sections in self.DRAFT_GROUPS.items():
                futures.append(pool.submit(
                    self._draft_worker, agent_id, sections, session_id))
            for f in futures:
                f.result(timeout=180)
        self._phase("2 并行起草", "三份初稿全部提交完成")

        # ---- 阶段3：交叉审查 ----
        self._phase("3 交叉审查", "数据核查智能体审阅全部章节并检查一致性")
        with self._conflicts_lock:
            self._pending_conflicts = []
        self.send(MessageType.TASK_ASSIGN, "verifier",
                  {"type": "review", "scope": "all_sections"},
                  priority=Priority.HIGH, session_id=session_id)
        with self._conflicts_lock:
            conflicts = list(self._pending_conflicts)
        self._phase("3 交叉审查", f"发现 {len(conflicts)} 处不一致")
        for c in conflicts:
            self._phase("3 交叉审查",
                        f"冲突 {c['conflict_id']}: {c['description']}")

        # ---- 阶段4：冲突解决 ----
        resolved = self._resolve_conflicts(conflicts, session_id)

        # ---- 阶段5：最终统稿 ----
        self._phase("5 最终统稿", "统稿润色智能体合并章节并生成 Markdown 申请书")
        polish_result = self.send(MessageType.TASK_ASSIGN, "polish",
                                  {"type": "merge"},
                                  priority=Priority.HIGH, session_id=session_id)
        self.send(MessageType.BROADCAST, "all",
                  {"phase": "complete",
                   "final_doc": (polish_result or {}).payload.get("output_path")},
                  priority=Priority.HIGH, session_id=session_id)

        # ---- 汇总 ----
        overhead = communication_overhead(self.bus.messages)
        summary = {
            "session_id": session_id,
            "elapsed_seconds": round(time.time() - t0, 2),
            "phases": list(self._phase_log),
            "conflicts_found": len(conflicts),
            "conflicts_resolved": len(resolved),
            "final_proposal": (polish_result or {}).payload,
            "communication": overhead,
        }
        self._phase("完成", f"端到端演示结束，共 {overhead['total_messages']} 条消息，"
                            f"{overhead['total_tokens']} tokens")
        return summary

    def _draft_worker(self, agent_id: str, sections: List[str], session_id: str) -> None:
        """单智能体起草任务（在线程池中执行，体现并行）。"""
        sec_specs = [{"section_id": sid} for sid in sections]
        self.send(MessageType.TASK_ASSIGN, agent_id,
                  {"type": "draft", "sections": sec_specs},
                  priority=Priority.NORMAL, session_id=session_id)

    def _send_revise(self, agent_id: str, task: Dict[str, Any], session_id: str) -> None:
        """向单个智能体下发修订任务（供线程池并行调用）。"""
        self.send(MessageType.TASK_ASSIGN, agent_id, task,
                  priority=Priority.HIGH, session_id=session_id)

    def _resolve_conflicts(self, conflicts: List[Dict[str, Any]],
                           session_id: str) -> List[Dict[str, Any]]:
        resolved: List[Dict[str, Any]] = []
        for conflict in sorted(conflicts, key=lambda c: c.get("severity", "")):
            cid = conflict["conflict_id"]
            self._phase("4 冲突解决", f"处理冲突 {cid}，涉事智能体 {conflict['involved']}")
            decisions = self._negotiate(conflict, session_id)
            self._phase("4 冲突解决", f"冲突 {cid} 仲裁决议: {decisions}")

            ok = False
            for attempt in (1, 2):  # 复核不通过则最多重试 2 轮修订
                # 并行下发修订任务（涉事智能体同时调用大模型重写，缩短耗时）
                tasks = []
                for agent_id in conflict["involved"]:
                    task = {"type": "revise", "decisions": decisions}
                    if agent_id == "literature" and cid == "C3-citation":
                        task = {"type": "revise", "revise_references": True}
                    tasks.append((agent_id, task))
                with ThreadPoolExecutor(max_workers=len(tasks)) as pool:
                    futures = [pool.submit(self._send_revise, aid, tk, session_id)
                               for aid, tk in tasks]
                    for fut in futures:
                        fut.result(timeout=300)

                # 核查智能体复核
                recheck = self.send(MessageType.INFO_QUERY, "verifier",
                                    {"query": "recheck", "conflict_id": cid},
                                    priority=Priority.HIGH, session_id=session_id)
                remaining = (recheck or {}).payload.get("remaining_conflicts", [])
                ok = len([c for c in remaining if c.get("conflict_id") == cid]) == 0
                self._phase("4 冲突解决",
                            f"冲突 {cid} 第 {attempt} 轮复核{'通过' if ok else '未通过'}")
                if ok:
                    break

            if ok:
                resolved.append({**conflict, "decision": decisions})
                self.send(MessageType.BROADCAST, "all",
                          {"phase": "conflict_resolved", "conflict_id": cid,
                           "decision": decisions},
                          priority=Priority.HIGH, session_id=session_id)
            else:
                self._phase("4 冲突解决",
                            f"冲突 {cid} 多轮修订后仍未解决（保留待人工处理）")
        return resolved

    def _negotiate(self, conflict: Dict[str, Any], session_id: str) -> Dict[str, Any]:
        """协商：向涉事智能体查询主张与可调整空间，协调器仲裁。"""
        cid = conflict["conflict_id"]
        opinions: Dict[str, Any] = {}
        for agent_id in conflict["involved"]:
            query = {"query": {
                "C1-gpu-budget": "gpu_hours" if agent_id == "method" else "gpu_budget",
                "C2-terminology": "terminology",
                "C3-citation": "citations",
                "C4-reward-metric": "metrics",
            }.get(cid, ""), "section_id": {
                "C1-gpu-budget": "research_content" if agent_id == "method" else "experiment_plan",
                "C2-terminology": "research_content" if agent_id == "method" else "experiment_plan",
                "C3-citation": "references",
                "C4-reward-metric": "technical_route",
            }.get(cid, ""), "conflict_id": cid}
            reply = self.send(MessageType.INFO_QUERY, agent_id, query,
                              priority=Priority.HIGH, session_id=session_id)
            opinions[agent_id] = (reply or {}).payload or {}

        # 仲裁规则
        if cid == "C1-gpu-budget":
            method_prop = opinions.get("method", {}).get("proposal", {})
            exp_prop = opinions.get("experiment", {}).get("proposal", {})
            gpu = min(method_prop.get("can_reduce_to", 3000),
                      exp_prop.get("can_increase_to", 2000))
            return {
                "gpu_hours_claim": gpu,
                "gpu_budget": gpu,
                "reason": f"双方协商后统一算力为 {gpu} GPU·小时（共享Critic+混合精度+精简消融）",
            }
        if cid == "C2-terminology":
            term = "平均作业完成时间(JCT)"
            return {
                "metric_terms": [term],
                "makespan_note": "",
                "reason": "统一术语为「平均作业完成时间(JCT)」，消除 makespan 混用",
            }
        if cid == "C3-citation":
            return {"revise_references": True, "reason": "补齐正文已引用的文献 [7]"}
        if cid == "C4-reward-metric":
            return {"metrics": ["平均作业完成时间(JCT)", "资源利用率", "能耗", "Jain公平性指数"],
                    "reason": "实验方案评估指标补充 JCT 口径"}
        return {"reason": "默认仲裁"}

    # ---- 工具 ----
    def _phase(self, name: str, detail: str) -> None:
        entry = {"phase": name, "detail": detail, "time": time.strftime("%H:%M:%S")}
        self._phase_log.append(entry)
        print(f"[coordinator] 阶段 {name}: {detail}")
