# -*- coding: utf-8 -*-
"""图表材料智能体（聚合 2 类展示型图表材料，纯 LLM 驱动）。

负责 2 类同为“图表 + 文字说明”的科研材料：
    类型 4：论文图表截图   —— 提取图标题、子图、图例、对比对象
    类型 6：学术会议 PPT 截图 —— 提取图表编号/标题、对比对象、结果表格与数值

设计说明：本智能体为纯 LLM 智能体——只定义给大模型的“契约”
（角色 Prompt、输入规范、输出键名 schema），识别完全由 DeepSeek 完成，
代码中不包含任何“预写识别结果”的规则逻辑。
"""

from __future__ import annotations

from .base_agent import VisionAgent

# 按实际材料类型提供输出契约（键名与 ground truth 评估对齐）
_META = {
    4: {  # 论文图表
        "material_type": "论文图表截图",
        "input_spec": {
            "输入": "论文图表截图，含标题、坐标轴、图例、子图",
            "示例": "算法路径对比图、误差曲线、柱状图",
        },
        "output_schema": {
            "图标题": "str，图表标题",
            "子图a": "str，子图(a)内容",
            "子图b": "str，子图(b)内容",
            "对比对象": "list[str]，参与对比的算法/方法",
        },
        "numeric_schema": {},
        "structure_schema": {"子图数": "子图个数", "图例数": "图例/对比对象个数"},
    },
    6: {  # 学术会议 PPT
        "material_type": "学术会议PPT截图",
        "input_spec": {
            "输入": "学术会议 PPT 截图，含方法框架图、结果表格",
            "示例": "对比曲线 + 结果指标表",
        },
        "output_schema": {
            "图编号": "str，如 图3-7",
            "图表标题": "str，图标题",
            "对比对象": "list[str]，对比方法",
            "误差类型": "list[str]，误差指标",
            "横轴": "str，横轴含义",
            "表编号": "str，如 表3-2",
            "表标题": "str，表格标题",
            "numeric_fields": "dict，表格数值（MAE/欧氏距离/皮尔逊系数，区分 LQR 与本文、x/y 方向）",
        },
        "numeric_schema": {
            "MAE_x_LQR": 0.0, "MAE_x_本文": 0.0, "MAE_y_LQR": 0.0, "MAE_y_本文": 0.0,
            "ED_x_LQR": 0.0, "ED_x_本文": 0.0, "ED_y_LQR": 0.0, "ED_y_本文": 0.0,
            "PCC_x_LQR": 0.0, "PCC_x_本文": 0.0, "PCC_y_LQR": 0.0, "PCC_y_本文": 0.0,
        },
        "structure_schema": {
            "曲线子图数": "曲线子图个数",
            "表格行数": "表格行数",
            "表格列数": "表格列数",
            "性能提升标注数": "表中 ↑xx% 提升标注个数",
        },
    },
}


class ChartMaterialAgent(VisionAgent):
    """图表材料智能体：论文图表 + 学术会议 PPT（纯 LLM）。"""

    name = "ChartMaterialAgent"
    material_type = "图表材料（论文图表/PPT）"
    material_type_id = 4

    prompt_hint = """【角色】你是科研图表材料识别专家，擅长解读论文图表与学术汇报 PPT 两类展示型图表材料。
【任务】根据图片内容自动识别其子类型并提取结构化 JSON：
  - 论文图表：图标题、子图a/b、对比对象(list)
  - 学术会议 PPT：图编号、图表标题、对比对象(list)、误差类型(list)、横轴、表编号、表标题，以及表格数值 numeric_fields
【输入】图表材料图片：论文图表截图或 PPT 页面截图（含对比曲线、结果表格等）。
【输出要求】输出字段与图片子类型对应；表格数值放入 numeric_fields；字段缺失置空，不要编造。
【处理要点】子图按 (a)(b) 标注识别，容忍 OCR 拆字；竖排轴标签做字符级重组；表格数值取指标行内 token 开头数值。
【示例输出-论文图表】{"图标题":"不同光照条件下的目标检测精度对比","子图a":"YOLOv8 检测结果","子图b":"RT-DETR 检测结果","对比对象":["YOLOv8","RT-DETR"]}
【示例输出-PPT】{"图编号":"图3-7","图表标题":"三种优化器的收敛曲线对比","对比对象":["SGD","Adam","LAMB"],"误差类型":["训练损失","验证损失"],"横轴":"迭代轮数","表编号":"表3-2","表标题":"不同优化器性能指标","numeric_fields":{"MAE_x_LQR":0.112,"MAE_x_本文":0.094,"MAE_y_LQR":0.121,"MAE_y_本文":0.105,"ED_x_LQR":1.2,"ED_x_本文":1.0,"ED_y_LQR":1.4,"ED_y_本文":1.1,"PCC_x_LQR":0.91,"PCC_x_本文":0.93,"PCC_y_LQR":0.89,"PCC_y_本文":0.92}}"""

    def __init__(self, material_type_id: int = 4):
        super().__init__(material_type_id=material_type_id)
        meta = _META.get(material_type_id, _META[4])
        self.material_type = meta["material_type"]
        self.input_spec = meta["input_spec"]
        self.output_schema = meta["output_schema"]
        self.numeric_schema = meta["numeric_schema"]
        self.structure_schema = meta["structure_schema"]
