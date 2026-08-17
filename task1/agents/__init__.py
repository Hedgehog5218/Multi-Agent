# -*- coding: utf-8 -*-
"""task1 多智能体协同写作系统 —— 智能体包。

包含 5 个子智能体 + 1 个协调器：
- literature  : 文献调研智能体
- method      : 方法设计智能体
- experiment  : 实验规划智能体
- verifier    : 数据/逻辑核查智能体
- polish      : 统稿润色智能体
- coordinator : 协调器（星型拓扑中心，负责分解/调度/仲裁）
"""
from __future__ import annotations

from typing import Dict, List, Optional

from protocol import MessageBus, SharedWorkspace
from agents.base import AgentBase
from agents.coordinator import CoordinatorAgent
from agents.literature_agent import LiteratureAgent
from agents.method_agent import MethodAgent
from agents.experiment_agent import ExperimentAgent
from agents.verifier_agent import VerifierAgent
from agents.polish_agent import PolishAgent
from agents.llm import LLMClient

__all__ = [
    "AgentBase", "CoordinatorAgent", "LiteratureAgent", "MethodAgent",
    "ExperimentAgent", "VerifierAgent", "PolishAgent", "LLMClient",
    "build_system", "SUB_AGENT_IDS",
]

SUB_AGENT_IDS = ["literature", "method", "experiment", "verifier", "polish"]


def build_system(output_dir: str = "logs", listener=None) -> Dict[str, AgentBase]:
    """构建并注册完整系统（协调器 + 5 子智能体 + 消息总线 + 黑板）。

    返回 {"coordinator": ..., "agents": {...}, "bus": ..., "workspace": ...}
    """
    workspace = SharedWorkspace()
    bus = MessageBus(workspace, listener=listener)
    llm = LLMClient()

    coordinator = CoordinatorAgent(workspace, llm=llm)
    agents = {
        "literature": LiteratureAgent(workspace, llm=llm),
        "method": MethodAgent(workspace, llm=llm),
        "experiment": ExperimentAgent(workspace, llm=llm),
        "verifier": VerifierAgent(workspace, llm=llm),
        "polish": PolishAgent(workspace, llm=llm, output_dir=output_dir),
    }
    bus.register(coordinator)
    for a in agents.values():
        bus.register(a)
    return {
        "coordinator": coordinator,
        "agents": agents,
        "bus": bus,
        "workspace": workspace,
    }
