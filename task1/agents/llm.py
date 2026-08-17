# -*- coding: utf-8 -*-
"""DeepSeek LLM 客户端：多智能体内容生成的统一入口。

设计要点（本次大改）：
- 配置来自 config.py，代码内配置，无需终端环境变量；
- 每次调用在终端输出提示语，便于观察每个智能体正在做什么；
- 正文完全由大模型从零生成，本模块不包含任何"预留答案/模板正文"；
- 调用失败自动重试 1 次，仍失败则抛出异常终止运行（不做本地模板回退）；
- 各章节写作任务描述与部门私有口径来自 agents/content.py（知识库）。
"""
from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

from config import API_KEY, BASE_URL, MODEL
from agents import content


class LLMClient:
    """面向 DeepSeek 的 OpenAI 兼容客户端。"""

    def __init__(self) -> None:
        self.api_key = API_KEY
        self.base_url = BASE_URL.rstrip("/")
        self.model = MODEL

    # ------------------------------------------------------------------
    def chat(self, messages: List[Dict[str, Any]], temperature: float = 0.7,
             max_tokens: int = 2500, timeout: int = 180) -> str:
        """调用一次 DeepSeek chat/completions；失败重试一次，仍失败抛异常。"""
        body = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        last_err: Optional[Exception] = None
        for attempt in (1, 2):
            try:
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                return data["choices"][0]["message"]["content"].strip()
            except (urllib.error.URLError, KeyError, IndexError,
                    json.JSONDecodeError) as exc:
                last_err = exc
                print(f"  [DeepSeek] 第 {attempt} 次调用失败: {exc}，"
                      f"{'正在重试…' if attempt == 1 else '重试仍失败，终止'}")
                if attempt == 1:
                    time.sleep(2)
        raise RuntimeError(f"DeepSeek API 调用失败: {last_err}")

    # ------------------------------------------------------------------
    def generate_section(self, agent_id: str, section_id: str,
                         extra_constraints: Optional[List[str]] = None) -> str:
        """按知识库中的写作任务描述 + 私有事实，让 DeepSeek 从零撰写整章。

        extra_constraints：冲突解决后的额外硬约束（如"算力统一为 2200"、
        "术语统一为 JCT"），用于修订阶段按决议重写。
        返回整章 Markdown 正文。
        """
        task = content.SECTION_TASKS.get(section_id)
        if task is None:
            raise ValueError(f"未知章节: {section_id}")

        facts = content.AGENT_FACTS.get(agent_id, {})
        agent_name = content.AGENT_NAMES.get(agent_id, agent_id)

        fact_lines = []
        if "gpu_hours_claim" in facts:
            fact_lines.append(f"- 本部门（方法设计）口径：本课题训练算力需求为 "
                              f"{facts['gpu_hours_claim']} GPU·小时。")
        if "gpu_budget" in facts:
            fact_lines.append(f"- 本部门（实验规划）口径：本课题可用算力预算为 "
                              f"{facts['gpu_budget']} GPU·小时。")
        if facts.get("metric_terms"):
            fact_lines.append(f"- 本部门对作业完成时间指标的口径写法：{facts['metric_terms']}。")
        if facts.get("citation_range"):
            fact_lines.append(f"- 引用文献只能用编号 {facts['citation_range']} 标注（如 [1]）。")
        if facts.get("reference_hint"):
            fact_lines.append(f"- {facts['reference_hint']}")

        NL = "\n"
        prompt = (
            f"你是科研基金申请书写作专家。请撰写《{content.PROPOSAL_TITLE}》"
            f"基金申请书的【{task['title']}】章节。{NL}{NL}"
            f"【本章节任务要求】{NL}{task['spec']}{NL}{NL}"
            f"【你所在的部门及口径（必须遵守）】{NL}"
            + (NL.join(fact_lines) if fact_lines else "（无特殊口径）")
            + f"{NL}{NL}【输出要求】Markdown 格式；小节标题用 ## x.x；学术化中文表达；"
              f"总长度 500~900 字；不要输出与本章节无关的内容；不要输出章节大标题。"
        )
        if extra_constraints:
            prompt += f"{NL}{NL}【本次修订的额外硬约束（必须遵守）】{NL}" + NL.join(extra_constraints)

        print(f"  [DeepSeek] {agent_name} 正在生成《{task['title']}》…")
        return self.chat(
            [
                {"role": "system", "content": "你是一名严谨的科研基金申请书写作专家。"},
                {"role": "user", "content": prompt},
            ]
        )

    # ------------------------------------------------------------------
    _GPU_NUM_RE = re.compile(r"(\d+)\s*GPU\s*[·.]?\s*小时")

    def generate_section_checked(self, agent_id: str, section_id: str,
                                 expected_gpu: Optional[int] = None,
                                 no_gpu: bool = False,
                                 extra_constraints: Optional[List[str]] = None) -> str:
        """生成并校验算力规则：不满足时用更强约束重试一次。

        - expected_gpu：该章节第一个 GPU·小时 数字必须等于该值（算力总量申报）；
        - no_gpu       ：该章节不应出现任何 GPU·小时 数字。
        重试后仍不满足则返回最后一次结果，由核查智能体兜底检测。
        """
        body = ""
        for attempt in (1, 2):
            body = self.generate_section(agent_id, section_id,
                                         extra_constraints=extra_constraints)
            nums = [int(m) for m in self._GPU_NUM_RE.findall(body)]
            if expected_gpu is not None:
                if nums and nums[0] == expected_gpu:
                    return body
                # 无数字或第一个数字不符，都视为不满足 -> 重试
                if attempt == 1:
                    extra = list(extra_constraints or [])
                    extra.append(f"本章节必须明确写出训练算力总量「累计 "
                                 f"{expected_gpu} GPU·小时」，并作为第一个出现的算力数字。")
                    extra_constraints = extra
                    print("  [DeepSeek] 算力数字校验未通过，正在重试…")
                    continue
                return body  # 重试后仍不满足：返回最后一次，交由核查智能体兜底
            if no_gpu:
                if not nums:
                    return body
                if attempt == 1:
                    extra = list(extra_constraints or [])
                    extra.append("本章节不要出现任何具体算力数字（如 N GPU·小时）。")
                    extra_constraints = extra
                    print("  [DeepSeek] 检测到多余算力数字，正在重试…")
                    continue
            return body  # 重试后仍未满足：返回最后一次，交由核查智能体兜底
        return body
