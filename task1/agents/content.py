# -*- coding: utf-8 -*-
"""知识库：为各智能体提供"写作任务描述 + 部门私有口径 + 调研文献资料"。

本模块【不含任何整章正文答案】。所有章节正文均由 DeepSeek 大模型从零生成；
本模块只提供三类信息：
1. SECTION_TASKS：每个章节"写什么、什么结构"的任务要求（不是答案）；
2. AGENT_FACTS   ：各智能体所在部门的私有口径（如方法部门认为算力需 3000、
                   实验部门认为预算只有 2000）——信息不对称使跨章节冲突自然涌现；
3. REFERENCES    ：文献调研智能体检索到的真实文献资料（初始调研范围缺 [7]，
                   用于演示"正文引用但文献表缺失"的引用一致性冲突）。
"""
from __future__ import annotations

from typing import Any, Dict, List

PROPOSAL_TITLE = "基于多智能体强化学习的分布式计算资源调度方法研究"

# 智能体中文名
AGENT_NAMES: Dict[str, str] = {
    "literature": "文献调研智能体",
    "method": "方法设计智能体",
    "experiment": "实验规划智能体",
    "verifier": "数据/逻辑核查智能体",
    "polish": "统稿润色智能体",
    "coordinator": "协调器",
}

# ---------------------------------------------------------------------------
# 章节写作任务（只描述"写什么 + 结构"，正文由大模型生成）
# ---------------------------------------------------------------------------
SECTION_TASKS: Dict[str, Dict[str, Any]] = {
    "project_basis": {
        "title": "一、立项依据",
        "spec": (
            "包含 4 个小节（用 ## 1.1~1.4 编号）：\n"
            "1.1 研究背景与意义（分布式集群规模扩大、节点异构、负载动态，"
            "传统启发式调度的局限，单智能体强化学习调度的不足，引出多智能体"
            "强化学习 MARL 的价值）；\n"
            "1.2 国内外研究现状（集群资源调度、深度强化学习调度、多智能体"
            "强化学习三条线，必须引用文献，正文用 [1]~[9] 标注）；\n"
            "1.3 拟解决的关键科学问题（3 条）；\n"
            "1.4 研究目标。"
        ),
    },
    "research_content": {
        "title": "二、研究内容",
        "spec": (
            "包含 4 个小节（用 ## 2.1~2.4 编号）：\n"
            "2.1 面向资源调度的多智能体协同决策框架（节点级智能体、集中训练-"
            "分布执行、状态/动作/奖励设计）；\n"
            "2.2 基于注意力值分解的可扩展调度策略学习算法（值分解单调约束、"
            "注意力 Mixing 网络）；\n"
            "2.3 任务依赖约束下的调度-迁移联合优化；\n"
            "2.4 多目标联合优化与在线自适应。"
        ),
    },
    "technical_route": {
        "title": "三、技术路线",
        "spec": (
            "包含 2 个小节（用 ## 3.1~3.2 编号）：\n"
            "3.1 总体技术路线（环境建模→智能体设计→训练算法→系统集成与评估"
            "四层递进）；\n"
            "3.2 关键方法与公式（奖励函数、值分解、策略优化目标）。\n"
            "注意：本章节不要提及任何具体算力数字（如 N GPU·小时），算力细节"
            "由其它章节负责。"
        ),
    },
    "experiment_plan": {
        "title": "四、实验方案",
        "spec": (
            "包含 5 个小节（用 ## 4.1~4.5 编号）：\n"
            "4.1 实验数据与仿真环境（Alibaba Cluster Trace、Google Borg 合成"
            "负载、离散事件模拟器）；\n"
            "4.2 对比基线（FIFO、DRF、Tetris、DeepRM、Decima、单智能体 PPO）；\n"
            "4.3 评估指标；\n"
            "4.4 实验设置（16/64/256 节点三组规模，每组 5 次重复）；\n"
            "4.5 预期实验结果。"
        ),
    },
    "expected_outcomes": {
        "title": "五、预期成果",
        "spec": (
            "包含 3 个小节（用 ## 5.1~5.3 编号）：5.1 学术成果（论文、专利、"
            "软著）；5.2 数据与平台成果（开源基准）；5.3 人才培养。\n"
            "注意：本章节不要提及任何具体算力数字（如 N GPU·小时），算力细节"
            "由其它章节负责。"
        ),
    },
}

# ---------------------------------------------------------------------------
# 各智能体的部门私有口径（信息不对称 -> 冲突自然涌现）
# ---------------------------------------------------------------------------
AGENT_FACTS: Dict[str, Dict[str, Any]] = {
    "literature": {
        "citation_range": "[1]~[9]",
        # 文献调研智能体初始只"检索到"部分文献（缺 [7]）
        "available_reference_keys": ["[1]", "[2]", "[3]", "[4]", "[5]", "[6]", "[8]", "[9]"],
        "reference_hint": "你的调研资料库当前收录的文献为："
                          + ", ".join(["[1]", "[2]", "[3]", "[4]", "[5]", "[6]", "[8]", "[9]"]),
    },
    "method": {
        "gpu_hours_claim": 3000,          # 方法部门口径：训练算力需求
        "metric_terms": "makespan（本部门统一称作业完成时间为 makespan）",
        "citation_range": "[1]~[9]",
    },
    "experiment": {
        "gpu_budget": 2000,               # 实验部门口径：可用算力预算
        "metric_terms": "平均作业完成时间(JCT)（本部门统一称作业完成时间为 JCT）",
        "citation_range": "[1]~[9]",
    },
}

# ---------------------------------------------------------------------------
# 文献调研资料库（真实文献；初始收录范围缺 [7]，用于引用一致性冲突演示）
# ---------------------------------------------------------------------------
REFERENCES: Dict[str, str] = {
    "[1]": "Mao H, Alizadeh M, Menache I, et al. Resource management with deep reinforcement learning[C]//HotNets 2016.",
    "[2]": "Mao H, Schwarzkopf M, Venkatakrishnan S B, et al. Learning scheduling algorithms for data processing clusters[C]//SIGCOMM 2019.",
    "[3]": "Ghodsi A, Zaharia M, Hindman B, et al. Dominant resource fairness: fair allocation of multiple resource types[C]//NSDI 2011.",
    "[4]": "Grandl R, Ananthanarayanan G, Kandula S, et al. Multi-resource packing for cluster schedulers[J]. IEEE/ACM TON, 2016.",
    "[5]": "Rashid T, Samvelyan M, de Witt C S, et al. QMIX: monotonic value function factorisation for deep multi-agent reinforcement learning[C]//ICML 2018.",
    "[6]": "Yu C, Velu A, Vinitsky E, et al. The surprising effectiveness of PPO in cooperative multi-agent games[C]//NeurIPS 2022.",
    "[7]": "Lowe R, Wu Y, Tamar A, et al. Multi-agent actor-critic for mixed cooperative-competitive environments[C]//NeurIPS 2017.",
    "[8]": "Alibaba. Alibaba cluster trace program 2018[EB/OL]. https://github.com/alibaba/clusterdata.",
    "[9]": "Verma A, Pedrosa L, Korupolu M, et al. Large-scale cluster management at Google with Borg[C]//EuroSys 2015.",
}


# 章节顺序（最终申请书合并顺序）
SECTION_ORDER: List[str] = [
    "project_basis", "research_content", "technical_route",
    "experiment_plan", "expected_outcomes",
]


def build_references(keys: List[str]) -> str:
    """按给定文献键列表格式化参考文献表（资料整理，非正文答案）。"""
    lines = []
    for k in keys:
        if k in REFERENCES:
            lines.append(f"{k} {REFERENCES[k]}")
    return "\n".join(lines)


def all_reference_keys() -> List[str]:
    """知识库中已收录的全部文献键（冲突解决后用于补齐缺失引用）。"""
    return list(REFERENCES.keys())
