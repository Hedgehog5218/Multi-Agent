# -*- coding: utf-8 -*-
"""文献 PDF 页面截图智能体（材料类型 8）。

职责：从文献 PDF 页面截图中提取论文标题、摘要内容、研究主题关键词
与参考文献条目。
适用场景：文献 PDF 首页截图、摘要页、参考文献页。
"""

from __future__ import annotations

import re
from typing import List

from .base_agent import VisionAgent


class PaperPageAgent(VisionAgent):
    """专注“文献 PDF 页面截图”的视觉智能体。"""

    name = "PaperPageAgent"
    material_type = "文献PDF页面截图"
    material_type_id = 8

    input_spec = {
        "输入": "文献 PDF 页面截图，含摘要、公式、参考文献",
    }
    output_schema = {
        "论文标题": "str，论文标题",
        "研究主题": "list[str]，研究主题关键词",
        "摘要关键词": "list[str]，摘要中出现的关键词",
        "numeric_fields": {"参考文献条数": "int"},
    }
    prompt_hint = """【角色】你是文献信息识别专家，擅长提取论文元数据。
【任务】从文献 PDF 页面截图中提取论文标题、研究主题关键词、摘要关键词与参考文献信息，输出结构化 JSON。
【输入】文献 PDF 页面截图，含摘要、公式、参考文献。
【输出要求】字段：论文标题、研究主题(list[str])、摘要关键词(list[str])；数值字段：参考文献条数。
【处理要点】标题取页面顶部最长文本；参考文献编号连续，按最大编号统计条数（容忍个别条目 OCR 漏检）。
【示例输出】{"论文标题":"基于对比学习的跨模态表示学习方法研究","研究主题":["对比学习","跨模态","表示学习"],"摘要关键词":["预训练","多模态对齐","下游任务迁移"],"参考文献条数":42}"""
    numeric_schema = {'参考文献条数': 0}
    structure_schema = {'含摘要': '是否含摘要', '含参考文献': '是否含参考文献', '参考文献格式': 'GB/T 7714'}


