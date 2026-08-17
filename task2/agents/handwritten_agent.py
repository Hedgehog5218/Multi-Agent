# -*- coding: utf-8 -*-
"""手写材料智能体（聚合 2 类手写材料，纯 LLM 驱动）。

负责 2 类同为“手写”的科研材料：
    类型 3：组会白板拍照   —— 提取讨论/设计语义（主题、模块、公式、变量）
    类型 7：实验记录本扫描件 —— 提取事实/记录语义（标题、坐标系、结构类型）

设计说明：本智能体为纯 LLM 智能体——只定义给大模型的“契约”
（角色 Prompt、输入规范、输出键名 schema），识别完全由 DeepSeek 完成，
代码中不包含任何“预写识别结果”的规则逻辑。
"""

from __future__ import annotations

from .base_agent import VisionAgent

# 按实际材料类型提供输出契约（键名与 ground truth 评估对齐）
_META = {
    3: {  # 组会白板
        "material_type": "组会白板拍照",
        "input_spec": {
            "输入": "组会白板拍照照片，含手写公式推导与流程图",
            "建议": "画面文字应尽量清晰、对比度适中",
        },
        "output_schema": {
            "主题": "str，讨论主题",
            "模块": "list[str]，流程/系统模块",
            "公式": "list[str]，识别出的公式",
            "关键变量": "dict[str,str]，变量符号及含义",
        },
        "numeric_schema": {},
        "structure_schema": {"流程图模块数": "系统流程主要模块个数", "公式数": "公式个数"},
    },
    7: {  # 实验记录本
        "material_type": "实验记录本扫描件",
        "input_spec": {
            "输入": "实验记录本扫描件，含手写文字、数据表格、手绘结构图",
        },
        "output_schema": {
            "记录标题": "str，记录条目标题",
            "坐标系": "list[str]，涉及坐标系/TF 结构",
            "结构类型": "str，结构图类型",
        },
        "numeric_schema": {},
        "structure_schema": {"坐标系数量": "坐标系个数", "绘制方式": "手绘/印刷 之一"},
    },
}


class HandwrittenMaterialAgent(VisionAgent):
    """手写材料智能体：组会白板 + 实验记录本（纯 LLM）。"""

    name = "HandwrittenMaterialAgent"
    material_type = "手写材料（白板/记录本）"
    material_type_id = 3

    prompt_hint = """【角色】你是科研手写材料识别专家，擅长解读组会白板与实验记录本两类手写材料。
【任务】根据图片内容提取结构化 JSON：
  - 组会白板（讨论/设计语义）：主题、模块(list)、公式(list)、关键变量(dict)
  - 实验记录本（事实/记录语义）：记录标题、坐标系(list)、结构类型
【输入】手写材料图片：白板拍照（含系统架构/公式/待办）或实验记录本扫描件（含结构图/数据/配置）。
【输出要求】输出字段与图片子类型对应；字段无法识别时置空或 null，不要编造。
【处理要点】手写 OCR 易错：结合上下文纠错；公式按符号语义重组；坐标系名按编辑相似度匹配标准 TF 名。
【示例输出-白板】{"主题":"多传感器融合定位方案讨论","模块":["状态估计","地图构建","回环检测","定位输出"],"公式":["x_k=A·x_{k-1}+B·u_k"],"关键变量":{"x":"状态向量","u":"控制输入"}}
【示例输出-记录本】{"记录标题":"陀螺仪标定记录","坐标系":["world","body","camera"],"结构类型":"标定参数记录表"}"""

    def __init__(self, material_type_id: int = 3):
        super().__init__(material_type_id=material_type_id)
        meta = _META.get(material_type_id, _META[3])
        self.material_type = meta["material_type"]
        self.input_spec = meta["input_spec"]
        self.output_schema = meta["output_schema"]
        self.numeric_schema = meta["numeric_schema"]
        self.structure_schema = meta["structure_schema"]
