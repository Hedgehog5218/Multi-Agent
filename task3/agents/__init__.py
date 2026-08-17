# -*- coding: utf-8 -*-
# agents 包：第三题多智能体语义识别与跨课题信息耦合（LLM 驱动）
from .parser_agents import build_parser_agents
from .coupling_agent import CouplingDetectorAgent

__all__ = ["build_parser_agents", "CouplingDetectorAgent"]
