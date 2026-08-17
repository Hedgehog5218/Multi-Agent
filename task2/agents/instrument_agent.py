# -*- coding: utf-8 -*-
"""仪器/数据面板截图智能体（材料类型 2）。

职责：从仪器软件面板（如 RViz、示波器、数据采集界面）截图中提取
软件名称、应用场景、订阅话题、坐标系、数据通道及关键数值参数。
适用场景：RViz/Matlab/示波器等数据面板截图。
"""

from __future__ import annotations

import re
from typing import List

from .base_agent import VisionAgent


class InstrumentPanelAgent(VisionAgent):
    """专注“仪器/数据面板截图”的视觉智能体。"""

    name = "InstrumentPanelAgent"
    material_type = "仪器数据面板截图"
    material_type_id = 2

    input_spec = {
        "输入": "仪器/软件数据面板截图，含数值、曲线、状态指示",
        "示例": "RViz 点云显示、示波器、状态监控面板",
    }
    output_schema = {
        "软件": "str，面板所属软件",
        "应用": "str，应用/算法场景",
        "话题": "str，订阅的 ROS 话题",
        "固定坐标系": "str，TF 固定坐标系",
        "点云通道": "str，点云字段通道",
        "numeric_fields": "dict，面板关键数值（Yaw/Pitch/距离/强度等）",
    }
    prompt_hint = """【角色】你是仪器/数据面板识别专家，擅长读取软件面板中的状态与读数。
【任务】从仪器或软件数据面板截图中提取软件名称、应用场景、订阅话题、固定坐标系、点云通道及关键数值读数，输出结构化 JSON。
【输入】仪器/软件数据面板截图（RViz、示波器、状态监控面板等），含数值、曲线、状态指示。
【输出要求】字段：软件、应用、话题、固定坐标系、点云通道；数值字段单独输出 numeric_fields（Yaw/Pitch/距离/强度上下限等）。
【处理要点】话题路径需完整保留（如 /sensor/scan）；数值取标签右侧同行读数；坐标系统一识别。
【示例输出】{"软件":"Matlab","应用":"电池健康状态监测","话题":"/bms/voltage","固定坐标系":"base","点云通道":"temperature","numeric_fields":{"视图Yaw":0.0,"视图Pitch":0.0,"视图距离":2.5,"强度最小值":10.0,"强度最大值":80.0}}"""
    numeric_schema = {'视图Yaw': 0.0, '视图Pitch': 0.0, '视图距离': 0.0, '强度最小值': 0.0, '强度最大值': 0.0}
    structure_schema = {'面板标题': '面板名称(Displays)', '显示类型': '点云显示类型(PointCloud2/Grid/Map 之一)', '坐标系数量': '固定坐标系个数'}
