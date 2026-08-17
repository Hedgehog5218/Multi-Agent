# -*- coding: utf-8 -*-
# 5 个课题解析智能体：调用 DeepSeek LLM 把各课题组文档解析为结构化语义表示
# （实体、带时间的事件、属性断言、技术要点），供后续跨课题耦合检测使用。
# 本版本不再使用任何硬编码实体/规则，抽取结果完全由大模型产出。

from .base import Agent
from .llm import chat_json, normalize_time, LLMError


class DocSemantics:
    """一份文档的结构化语义表示（由 LLM 解析产出）"""
    def __init__(self, doc_name, group):
        self.doc_name = doc_name
        self.group = group
        self.entities = {}        # 实体名 -> {"type": 类型, "aliases": 别名列表}
        self.events = []          # {"kind", "time", "time_text", "entity", "sentence", "desc"}
        self.claims = []          # {"entity", "attr", "value", "sentence"}
        self.tech_points = []     # {"tech", "desc", "sentence"}

    def to_dict(self):
        return {
            "doc_name": self.doc_name,
            "group": self.group,
            "entities": list(self.entities.keys()),
            "events": self.events,
            "claims": self.claims,
            "tech_points": self.tech_points,
        }


PARSER_SYSTEM = """你是一名资深的科研文档解析专家。你的任务是把课题组产出的科研文档解析为结构化语义表示，
供下游的多智能体系统做跨课题组耦合检测。你需要忠实于原文，不要编造原文中没有的信息。

实体类型（type）取值：resource（算力/设备/资源）、data（数据集/工具/词表）、tech（技术/算法/方法）、
doc（文档/专利/知识产权）、group（课题组）、other（其他）。
实体还需要给出别名列表（aliases），即原文中同一实体的不同说法（如“4块V100”“老旧V100节点”都指向 V100）。

事件类型（kind）建议取值：use（使用/训练）、retire（退役/报废）、delete（删除/停用）、apply（申请/提交）、
novelty（首创/首次声明）、promise（承诺/共享）、purchase（采购/购置）、plan（计划/拟采用）、
adopt（采用/作为baseline）、advise（建议/迁移/推荐）、cooperate（合作）、boundary（划清边界/知识产权）、other（其他）。
每个事件请给出：kind、time（原文中的时间描述，如“2026年8月”“2026年第四季度”，没有则留空）、
entity（事件涉及的主要实体名，没有则留空）、description（用原文原句说明，不超过80字）。
注意：不要将文档头部/落款的日期、作者、编号、报告名称等元信息识别为事件；只提取与研究活动、资源状态、
时间线相关的事件。

属性断言（claims）用于记录关键数值/事实，如“训练算力：4块V100”“压缩率：90%”“预算：460万元”。

技术要点（tech_points）用于记录该文档涉及的核心技术/方法及一句话说明。"""


def _parse_llm_json_to_semantics(doc_name, group, data):
    """把 LLM 返回的 JSON 转换为 DocSemantics"""
    sem = DocSemantics(doc_name, group)
    # 实体
    for e in data.get("entities", []) or []:
        name = (e.get("name") or "").strip()
        if not name:
            continue
        sem.entities[name] = {
            "type": e.get("type", "other"),
            "aliases": e.get("aliases", []) or [],
        }
    # 事件
    for ev in data.get("events", []) or []:
        desc = (ev.get("description") or "").strip()
        time_text = (ev.get("time") or "").strip()
        sem.events.append({
            "kind": (ev.get("kind") or "other").strip(),
            "time": normalize_time(time_text),
            "time_text": time_text,
            "entity": (ev.get("entity") or "").strip(),
            "sentence": desc,
            "desc": desc[:80],
        })
    # 断言
    for c in data.get("claims", []) or []:
        sem.claims.append({
            "entity": (c.get("entity") or "").strip(),
            "attr": (c.get("attr") or "").strip(),
            "value": (c.get("value") or "").strip(),
            "sentence": (c.get("description") or c.get("sentence") or "").strip(),
        })
    # 技术要点
    for tp in data.get("tech_points", []) or []:
        sem.tech_points.append({
            "tech": (tp.get("tech") or tp.get("name") or "").strip(),
            "desc": (tp.get("description") or "").strip(),
            "sentence": (tp.get("sentence") or "").strip(),
        })
    return sem


PARSER_OUTPUT_SCHEMA = """请严格按以下 JSON 结构输出（不要输出 JSON 以外的任何文字）：
{
  "entities": [{"name": "实体名", "type": "类型", "aliases": ["别名1", "别名2"]}],
  "events": [{"kind": "事件类型", "time": "时间描述或空", "entity": "涉及实体或空", "description": "原文原句（≤80字）"}],
  "claims": [{"entity": "实体或空", "attr": "属性名", "value": "数值/事实", "description": "原文原句"}],
  "tech_points": [{"tech": "技术名", "description": "一句话说明"}]
}"""


class DocParserAgent(Agent):
    """课题解析智能体（LLM 驱动）"""
    DOC_KEY = ""
    GROUP = ""

    def __init__(self, logger):
        super().__init__(name=self.DOC_KEY.replace(".txt", "") + "解析智能体",
                         group=self.GROUP, logger=logger)
        self.doc_text = None
        self.semantics = None
        self.llm_usage = None

    def parse(self, doc_text):
        """调用 LLM 解析文档，返回 DocSemantics"""
        self.doc_text = doc_text
        self.logger.console(f"正在解析《{self.DOC_KEY}》（{self.GROUP}）…")
        self.logger.log("INFO", self.name,
                        f"开始解析文档《{self.DOC_KEY}》（所属课题组：{self.GROUP}），调用大模型……")
        user = (
            f"请解析以下由「{self.GROUP}」产出的科研文档《{self.DOC_KEY}》。\n\n"
            f"【文档全文】\n{doc_text}\n\n{PARSER_OUTPUT_SCHEMA}"
        )
        data, usage = chat_json(
            [{"role": "system", "content": PARSER_SYSTEM},
             {"role": "user", "content": user}],
            json_mode=True, logger=self.logger, label=self.name,
        )
        self.llm_usage = usage
        sem = _parse_llm_json_to_semantics(self.DOC_KEY, self.GROUP, data)
        self.semantics = sem
        self.logger.log("INFO", self.name,
                        f"LLM 解析完成：实体 {len(sem.entities)} 个，事件 {len(sem.events)} 个，"
                        f"断言 {len(sem.claims)} 条，技术要点 {len(sem.tech_points)} 个")
        self.logger.console(f"《{self.DOC_KEY}》解析完成：实体 {len(sem.entities)}、事件 {len(sem.events)}、"
                            f"技术要点 {len(sem.tech_points)}")
        return sem


class PaperAgent(DocParserAgent):
    """论文手稿解析智能体（模型压缩组）"""
    DOC_KEY = "论文手稿.txt"
    GROUP = "模型压缩组"


class PatentAgent(DocParserAgent):
    """技术专利草稿解析智能体（模型压缩组）"""
    DOC_KEY = "技术专利草稿.txt"
    GROUP = "模型压缩组"


class ExperimentAgent(DocParserAgent):
    """实验报告解析智能体（分布式训练组）"""
    DOC_KEY = "实验报告.txt"
    GROUP = "分布式训练组"


class ProposalAgent(DocParserAgent):
    """项目申请书解析智能体（联邦学习组）"""
    DOC_KEY = "项目申请书.txt"
    GROUP = "联邦学习组"


class FinalReportAgent(DocParserAgent):
    """结题报告解析智能体（分布式训练组）"""
    DOC_KEY = "结题报告.txt"
    GROUP = "分布式训练组"


# 文档 -> 解析智能体 的映射（顺序固定，保证演示可复现）
PARSER_CLASSES = [
    PaperAgent,
    PatentAgent,
    ExperimentAgent,
    ProposalAgent,
    FinalReportAgent,
]


def build_parser_agents(logger):
    """构建 5 个课题解析智能体（协调器统一创建）"""
    return [cls(logger) for cls in PARSER_CLASSES]
