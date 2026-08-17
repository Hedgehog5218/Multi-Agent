# -*- coding: utf-8 -*-
"""智能体注册表：维护材料类型 -> 专业智能体的路由映射。"""

from __future__ import annotations

from typing import Dict, Type

from .base_agent import VisionAgent
from .experiment_agent import ExperimentPhotoAgent
from .instrument_agent import InstrumentPanelAgent
from .handwritten_agent import HandwrittenMaterialAgent
from .chart_material_agent import ChartMaterialAgent
from .code_agent import CodeResultAgent
from .paper_agent import PaperPageAgent

# 材料类型 id -> 智能体类（8 类材料 -> 6 个专业智能体，每个专注 1-2 类）
#   手写材料智能体：类型 3 组会白板 + 类型 7 实验记录本
#   图表材料智能体：类型 4 论文图表 + 类型 6 学术会议 PPT
AGENT_REGISTRY: Dict[int, Type[VisionAgent]] = {
    1: ExperimentPhotoAgent,        # 实验装置照片
    2: InstrumentPanelAgent,        # 仪器/数据面板截图
    3: HandwrittenMaterialAgent,    # 组会白板拍照（手写材料）
    4: ChartMaterialAgent,          # 论文图表截图（图表材料）
    5: CodeResultAgent,             # 代码运行结果截图
    6: ChartMaterialAgent,          # 学术会议 PPT 截图（图表材料）
    7: HandwrittenMaterialAgent,    # 实验记录本扫描件（手写材料）
    8: PaperPageAgent,              # 文献 PDF 页面截图
}

# 材料类型 id -> 中文名称
TYPE_NAMES: Dict[int, str] = {
    1: "实验装置照片",
    2: "仪器数据面板截图",
    3: "组会白板拍照",
    4: "论文图表截图",
    5: "代码运行结果截图",
    6: "学术会议PPT截图",
    7: "实验记录本扫描件",
    8: "文献PDF页面截图",
}


def get_agent(material_type_id: int) -> VisionAgent:
    """按材料类型 id 实例化对应智能体（指定实例实际负责的类型）。"""
    cls = AGENT_REGISTRY[material_type_id]
    return cls(material_type_id=material_type_id)


def all_agents() -> list:
    """返回 6 个去重后的智能体实例列表（用于文档与泳道图）。"""
    seen = []
    for tid in sorted(set(AGENT_REGISTRY.keys())):
        a = get_agent(tid)
        if a.name not in [x.name for x in seen]:
            seen.append(a)
    return seen
