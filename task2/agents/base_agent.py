# -*- coding: utf-8 -*-
"""视觉智能体基类模块。

定义统一的结果结构 AgentResult 与抽象基类 VisionAgent。
每个专业视觉智能体继承 VisionAgent，只需实现自己的字段/结构提取逻辑；
OCR、计时、置信度估计等通用流程由基类统一完成。
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

from .ocr_utils import ocr_image


@dataclass
class AgentResult:
    """单个视觉智能体的结构化识别结果。"""

    image: str                       # 图片文件名
    material_type: str               # 材料类型（1~8 类之一）
    agent: str                       # 智能体名称
    material_type_id: int = 0        # 材料类型编号（1~8）
    fields: Dict[str, Any] = field(default_factory=dict)      # 提取的语义字段
    numeric_fields: Dict[str, float] = field(default_factory=dict)  # 提取的数值字段
    structure: Dict[str, Any] = field(default_factory=dict)    # 图表/版面结构信息
    confidence: float = 0.0          # 整体置信度 0~1
    processing_time: float = 0.0     # 处理耗时（秒）
    raw_text: str = ""               # OCR 原始文本（用于日志追溯）
    notes: List[str] = field(default_factory=list)  # 备注

    def to_dict(self) -> dict:
        """转为可 JSON 序列化的字典。"""
        return asdict(self)


class VisionAgent:
    """视觉智能体抽象基类。

    子类需提供:
        name           智能体名称
        material_type  负责的材料类型
        input_spec     输入规范说明
        output_schema  输出字段的 JSON Schema 描述
    并实现:
        extract(image_path, text_items) -> (fields, numeric_fields, structure, confidence)
    """

    # ---- 描述性元信息（供文档/报告使用）----
    name: str = "base"
    material_type: str = "未知"
    material_type_id: int = 0
    input_spec: Dict[str, Any] = {}
    output_schema: Dict[str, Any] = {}
    # 输出契约：数值字段键名模板 / 版面结构字段模板（Prompt 要求 LLM 严格按此输出）
    numeric_schema: Dict[str, Any] = {}
    structure_schema: Dict[str, Any] = {}
    prompt_hint: str = ""

    def __init__(self, material_type_id: Optional[int] = None):
        """构造智能体；协调器可指定该实例实际负责的材料类型编号。"""
        if material_type_id is not None:
            self.material_type_id = material_type_id

    def run(self, image_path: str) -> AgentResult:
        """执行完整识别流程（双模式）。

        模式由环境变量 LLM_MODE 控制：
          - rule：规则版（本地 OCR + 专业规则，可复现、零依赖）
          - llm ：大模型版（按 Prompt 蓝图调用视觉大模型，失败返回明确失败结果）
          - auto：默认。已配置大模型（config.json / LLM_API_KEY）则走 LLM，失败自动回退规则版
        """
        t0 = time.time()
        mode = os.environ.get("LLM_MODE", "auto").lower()
        llm_on = (mode == "llm") or (mode == "auto" and self._llm_enabled())

        if llm_on:
            llm_result = self._run_with_llm(image_path, t0)
            if llm_result is not None:
                return llm_result
            if mode == "llm":
                # 显式 LLM 模式：调用失败给出明确失败结果（不静默）
                elapsed = round(time.time() - t0, 3)
                return AgentResult(
                    image=os.path.basename(image_path),
                    material_type=self.material_type,
                    material_type_id=self.material_type_id,
                    agent=self.name,
                    fields={}, numeric_fields={}, structure={},
                    confidence=0.0, processing_time=elapsed, raw_text="",
                    notes=["LLM 调用失败：请检查 config.json 的 api_key/base_url/网络"],
                )
            # auto 模式：LLM 失败自动回退规则版
            print(f"    [提示] {os.path.basename(image_path)}：LLM 调用失败，已自动回退规则版")
        return self._run_rule(image_path, t0)

    def _run_rule(self, image_path: str, t0: float) -> AgentResult:
        """规则版识别：OCR 感知 -> 专业规则提取。"""
        text_items = ocr_image(image_path)
        raw_text = "\n".join(
            it["text"] for it in sorted(text_items, key=lambda t: (t["y"], t["x"]))
        )
        fields, numeric_fields, structure, confidence = self.extract(image_path, text_items)
        elapsed = round(time.time() - t0, 3)
        return AgentResult(
            image=os.path.basename(image_path),
            material_type=self.material_type,
            material_type_id=self.material_type_id,
            agent=self.name,
            fields=fields,
            numeric_fields=numeric_fields,
            structure=structure,
            confidence=round(confidence, 3),
            processing_time=elapsed,
            raw_text=raw_text[:2000],
        )

    # ---- 大模型模式 ----
    @staticmethod
    def _llm_enabled() -> bool:
        """是否启用大模型后端。"""
        try:
            from .llm_backend import is_enabled
            return is_enabled()
        except Exception:
            return False

    def _build_llm_prompt(self) -> str:
        """构造大模型提示词：角色/任务 + 输出键名契约（字段/数值/结构键名与评估对齐）。"""
        lines = [f"{self.prompt_hint}"]
        lines.append("请识别这张图片，输出严格 JSON（不要输出任何其他文字），结构为：")
        lines.append('{"fields": {...}, "numeric_fields": {...}, "structure": {...}, "confidence": 0.0}')
        # fields 键名契约
        fkeys = list((self.output_schema or {}).keys())
        if fkeys:
            lines.append(f"- fields 键名（必须使用这些键，不要自创）：{json.dumps(fkeys, ensure_ascii=False)}")
        # numeric_fields 键名契约
        if self.numeric_schema:
            lines.append(f"- numeric_fields 键名（必须使用这些键并填上从图中读取的数值）：{json.dumps(self.numeric_schema, ensure_ascii=False)}")
        else:
            lines.append("- numeric_fields：提取图中出现的所有关键数值，键名用语义化名称，无法确定时留空对象")
        # structure 键名契约
        if self.structure_schema:
            lines.append("- structure：这是【版面/结构统计】部分，只输出数量、个数、布尔标志等版面统计值，"
                         "键名必须严格使用以下契约，不要自创键名，也不要把 fields（内容字段）重复填进这里：")
            lines.append(f"  {json.dumps(self.structure_schema, ensure_ascii=False)}")
            lines.append("  注意：键名含“数/个数/数量”时值必须是整数；无法确定时置 null，不要填字符串或数组。")
            # 动态示例：按 structure 键名生成一个虚构占位示例，示范“键名 + 值的类型”
            demo = {}
            for k in self.structure_schema:
                if any(t in k for t in ("数", "个数", "数量", "数目")):
                    demo[k] = 2
                elif any(t in k for t in ("文字", "列表", "清单")):
                    demo[k] = ["示例A", "示例B"]
                else:
                    demo[k] = "示例值"
            lines.append(f"  示例 structure（内容为虚构占位，仅示范键名与值的类型，请勿照抄值）：{json.dumps(demo, ensure_ascii=False)}")
        else:
            lines.append("- structure：版面结构信息（数量/布尔标志等），无法确定时留空对象")
        lines.append("- confidence：0~1 的识别置信度")
        lines.append("无法识别的字段置 null 或空对象，不要编造，不要自创键名。")
        return "\n".join(lines)

    def _run_with_llm(self, image_path: str, t0: float) -> Optional[AgentResult]:
        """调用视觉大模型识别，返回 AgentResult；失败返回 None。"""
        try:
            from . import llm_backend
            prompt = self._build_llm_prompt()
            # 优先使用缓存（保证评估可复现，避免大模型输出波动）
            raw = llm_backend.get_cached(image_path)
            if raw is None:
                # 评估路径：只用主视觉后端（use_fallback=False），失败回退规则版，
                # 避免文本 fallback（DeepSeek 基于 OCR 重构）产生与视觉口径不一致的结果
                raw = llm_backend.llm_extract(image_path, prompt, use_fallback=False)
                if raw is not None:
                    llm_backend.put_cache(image_path, raw)
            if not raw:
                return None
            fields = dict(raw.get("fields") or {})
            # 兼容：部分模型会把 numeric_fields 嵌套在 fields 内部，需提取到顶层
            if isinstance(fields.get("numeric_fields"), dict):
                inner = fields.pop("numeric_fields")
                if "numeric_fields" not in raw:
                    raw["numeric_fields"] = inner
            numeric = raw.get("numeric_fields") or {}
            if isinstance(numeric, dict):
                numeric = {k: v for k, v in numeric.items() if isinstance(v, (int, float))}
            structure = raw.get("structure") or {}
            try:
                conf = float(raw.get("confidence", 0.8))
            except (TypeError, ValueError):
                conf = 0.8
            elapsed = round(time.time() - t0, 3)
            return AgentResult(
                image=os.path.basename(image_path),
                material_type=self.material_type,
                material_type_id=self.material_type_id,
                agent=self.name,
                fields=fields,
                numeric_fields=numeric,
                structure=structure,
                confidence=round(min(max(conf, 0.0), 1.0), 3),
                processing_time=elapsed,
                raw_text="[LLM] " + json.dumps(raw, ensure_ascii=False)[:2000],
            )
        except Exception:
            return None

    # ---- 子类需实现 ----
    def extract(self, image_path: str, text_items: List[dict]):
        """提取字段/数值/结构与置信度。"""
        raise NotImplementedError

    # ---- 通用工具方法 ----
    @staticmethod
    def norm(s: str) -> str:
        """规范化文本：去除空白、全半角统一、小写。"""
        if s is None:
            return ""
        s = str(s).replace(" ", "").replace("\u3000", "").replace("\ufeff", "")
        s = s.replace("（", "(").replace("）", ")")
        return s.lower()

    @staticmethod
    def find_text(text_items: List[dict], keywords: List[str]) -> List[str]:
        """在 OCR 文本项中查找包含任一关键词的文本。"""
        hits = []
        for it in text_items:
            t = it["text"]
            if any(k.lower() in t.lower() for k in keywords):
                hits.append(t)
        return hits

    @staticmethod
    def find_numbers(text: str) -> List[float]:
        """从字符串中提取所有浮点数。"""
        return [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", text)]

    @staticmethod
    def match_near(items: List[dict], anchor: str, tol_x: int = 200, tol_y: int = 40) -> Optional[str]:
        """在 OCR 文本项中查找位于 anchor 右侧/下方附近的文本（表格值识别）。"""
        for it in items:
            t = it["text"]
            if VisionAgent.norm(t) == VisionAgent.norm(anchor):
                # 取同一行右侧最近的文本
                same = [x for x in items if abs(x["y"] - it["y"]) <= tol_y and x["x"] >= it["x"] - 5]
                same.sort(key=lambda x: x["x"])
                if len(same) > 1:
                    return same[1]["text"]
        return None
