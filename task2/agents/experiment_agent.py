# -*- coding: utf-8 -*-
"""实验装置照片智能体（材料类型 1）。

职责：从实验装置实拍照片中提取装置名称、底盘类型、传感器清单、
通信模块、计算设备等结构化字段。
适用场景：实验室实验平台/小车/设备实物照片。
"""

from __future__ import annotations

from typing import List

from .base_agent import VisionAgent


class ExperimentPhotoAgent(VisionAgent):
    """专注“实验装置照片”的视觉智能体。"""

    name = "ExperimentPhotoAgent"
    material_type = "实验装置照片"
    material_type_id = 1

    input_spec = {
        "输入": "实验装置实物照片（JPEG/PNG），画面含器材、连线、标注文字",
        "建议分辨率": "不低于 500x500",
    }
    output_schema = {
        "装置名称": "str，装置/平台名称",
        "底盘类型": "str，如阿克曼/麦克纳姆",
        "传感器": "list[str]，传感器清单",
        "通信模块": "str，通信方式",
        "计算设备": "str，车载计算设备",
    }
    prompt_hint = """【角色】你是实验室实验装置识别专家，擅长从装置实物照片中识别器材组成。
【任务】识别图中的实验装置/实验平台，提取装置名称、底盘类型、传感器清单、通信模块、计算设备，输出结构化 JSON。
【输入】实验装置实物照片（JPEG/PNG），画面含实验器材、连线、标注文字；建议分辨率不低于 500x500。
【输出要求】字段：装置名称(str)、底盘类型(str)、传感器(list[str])、通信模块(str)、计算设备(str)；无法识别时填"未识别"或空列表。
【处理要点】优先读取画面中的标注文字；传感器名称按标准术语归一化（如 激光雷达/LiDAR 统一）；多个传感器按出现顺序列出。
【示例输出】{"装置名称":"AGV-X1 移动平台","底盘类型":"麦克纳姆","传感器":["激光雷达","超声波","编码器"],"通信模块":"WiFi","计算设备":"工控机"}"""
    numeric_schema = {}
    structure_schema = {'标注数': '图中标注文字个数', '标注文字': '图中标注文字列表(如 GPS、激光雷达、IMU 等)'}


