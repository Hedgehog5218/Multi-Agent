# -*- coding: utf-8 -*-
"""代码运行结果截图智能体（材料类型 5）。

职责：从终端输出/训练曲线截图（accuracy、loss 曲线）中提取命令、
话题、消息类型、数据尺寸参数与时间戳。
适用场景：终端 rostopic echo、训练日志、性能曲线。
"""

from __future__ import annotations

import re
from typing import List

from .base_agent import VisionAgent


class CodeResultAgent(VisionAgent):
    """专注“代码运行结果截图”的视觉智能体。"""

    name = "CodeResultAgent"
    material_type = "代码运行结果截图"
    material_type_id = 5

    input_spec = {
        "输入": "代码运行结果截图，含终端输出或 accuracy/loss 曲线",
        "示例": "终端命令输出、训练曲线",
    }
    output_schema = {
        "命令": "str，执行的命令",
        "话题": "str，ROS 话题",
        "消息类型": "str，消息/数据类型",
        "帧ID": "str，frame_id",
        "传感器": "str，传感器型号",
        "numeric_fields": "dict，尺寸/时间戳等数值",
    }
    prompt_hint = """【角色】你是代码运行结果识别专家，擅长解析终端输出与训练日志。
【任务】从代码运行结果截图（终端输出、accuracy/loss 曲线）中提取命令、话题、消息类型、帧ID、传感器型号及关键数值参数，输出结构化 JSON。
【输入】代码运行结果截图，含终端输出或训练曲线。
【输出要求】字段：命令、话题、消息类型、帧ID、传感器；数值字段输出 numeric_fields（height/width/point_step/row_step/数据长度/时间戳等）。
【处理要点】命令与话题容忍 OCR 粘连（如 rostopicecho）；消息类型优先按结构字段推断（height/width/point_step → PointCloud2）；数据长度取 data 数组值。
【示例输出】{"命令":"rosbag record /camera/image_raw","话题":"/camera/image_raw","消息类型":"sensor_msgs/Image","帧ID":"camera_link","传感器":"Realsense D435","numeric_fields":{"height":720.0,"width":1280.0,"point_step":1.0,"row_step":1280.0,"数据长度":921600.0,"时间戳secs":1700000000.0}}"""
    numeric_schema = {'height': 0.0, 'width': 0.0, 'point_step': 0.0, 'row_step': 0.0, '数据长度': 0.0, '时间戳secs': 0.0}
    structure_schema = {'点云字段数': '点云字段个数', '消息帧数': '消息帧数', '小端序': 'is_bigendian 取反(False=>True)'}
