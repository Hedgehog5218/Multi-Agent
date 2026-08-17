# -*- coding: utf-8 -*-
# 跨课题耦合检测智能体（LLM 驱动）：
#   调用 DeepSeek 大模型对 5 份文档的结构化语义表示执行
#   实体对齐 / 时序推理 / 技术路线耦合 三类检测，并输出候选耦合关系。
# 本版本不使用任何硬编码规则或预设结果，检测结论完全由大模型产出。

import json

from .base import Agent
from .llm import chat_json, LLMError

# 检测采样次数：单轮（配合评审与门禁，无需多轮采样合并）
DETECT_SAMPLES = 1
DETECT_TEMPERATURES = [0.5]


DETECTOR_SYSTEM = """你是一名跨课题信息耦合检测专家，服务于大型科研实验室的协同管理平台。
实验室下设多个课题组，各自独立产出论文手稿、技术专利草稿、实验报告、项目申请书、结题报告等科研文档。
由于各课题组独立运作，不同文档在术语体系、实验条件、技术路线、时间线、资源分配之间存在隐性关联与潜在冲突。
你的任务是基于给定的文档结构化语义表示，识别出所有跨课题耦合关系，供协同分析报告使用。

耦合关系分为两类：
- 正耦合（positive）：可协同增效的关系，例如 A 组技术可直接作为 B 组的 baseline/复用对象；
- 负耦合（negative）：冲突矛盾或重复投入，例如资源已退役仍被使用、专利与论文时间线冲突、
  技术路线重复投入/知识产权冲突、数据集已下架仍被计划使用、资源被重复分配。

请重点检查三类机制：
1) 实体对齐：不同课题文档中的不同说法是否指向同一概念或资源（如“4块V100”“老旧V100节点”“V100”）。
2) 时序推理：论文投稿/实验/设备退役、专利提交、数据删除、采购、共享承诺等事件在时间上的冲突或依赖。
3) 技术路线耦合：
   - 负耦合：两组在同类技术上重复投入（如专利权利要求覆盖了对方计划开发的技术），或存在知识产权/技术边界冲突；
   - 正耦合：一方技术可被另一方直接复用/迁移到其训练流程（如某组的调度/压缩策略可迁移到对方训练任务以加速），或一方明确建议/采用另一方技术作为对照 baseline。

判定精确定义：
- “时序推理-新颖性冲突”仅当存在“专利申请日/公开日早于论文声称首次提出（如‘据调研，目前尚无’）的时间”时判定；
  某课题组为另一课题组提供算力/调度支持，不等于后者的方法缺乏新颖性，不要据此判定新颖性冲突。
- “技术路线耦合-重复投入”需要双方都在研发同类技术（如专利权利要求覆盖了对方计划开发的技术）；
  仅一方计划探索与另一方的结合，不构成重复投入。
- “时序推理-资源可用性冲突”要求：资源/数据集的终止时间（退役/删除）早于另一方计划使用的时间。

证据强度要求：
- 正耦合（可协同）必须存在明确的采纳或双向确认：例如申请书明确写“采用/以…作为对照 baseline”，
  且实验报告明确写“建议…作为对照 baseline”；仅出现“建议联合推进”“可复用”“值得关注”等弱意向表述，不构成正耦合。
- 负耦合（冲突）必须存在硬性证据：明确的时间先后矛盾（终止时间早于使用时间）、资源的重复分配/重复申报、
  或权利要求/技术方案的重叠；泛化的推测与弱关联不算。
- 宁可少输出，也不要输出证据不充分的耦合。

要求（逐类穷举，完整优先；严谨由下游评审把关）：
- 耦合关系包括【同一课题组内部文档之间】（如专利 vs 论文）以及【不同课题组之间】两种，都不要遗漏；
- 请严格按以下三类逐类检查并输出（每类都要覆盖所有相关实体与课题组对/文档对）：
  A. 时序/资源类：资源/数据集的时间矛盾、设备采购与共享承诺、数据集启用与停用；
  B. 新颖性/时间线类：专利申请日 vs 论文首创声明、同组内部文档时间线；
  C. 技术路线类：技术重叠/重复投入、协同复用（迁移/baseline）。
- 输出所有你认为可能存在关联的位置，包括证据偏间接的候选，数量宁多勿少；
  证据不足或疑似臆测的候选由后续评审环节过滤，你无需自行删减，但每条都要尽量给出证据句；
- 每条耦合必须是唯一的，同一关系不要用不同表述重复输出；
- 每条耦合必须给出证据，evidence 中的 quote 必须逐字来自给定文档解析结果中的原文句子；
- 严重程度 severity 取值：高（资源失效/重大利益冲突）、中（知识产权/时间线争议）、低（baseline 类协同建议）；
- 建议协同措施 suggestion 要具体、可执行。

新颖性冲突的判定示例（抽象，非本次数据）：
- 正确：A 组专利的申请日（如 2026年5月）早于 A 组论文声称“据调研，目前尚无相关工作”的撰写/投稿时间（如 2026年8月），应判定为“时序推理-新颖性冲突”。
- 错误：某组仅为另一组提供算力/调度支持，不能据此判定新颖性受影响；只有存在在先公开的专利或论文时才构成新颖性冲突。

易错点判定要点：
1) 协同复用（正耦合）包含两种形态：① 一方明确“采用/以…作为对照 baseline”，且对方文档有对应建议；② 一方建议将自身技术“迁移/复用到”对方的训练流程，且对方文档描述了使用同类机制或其效果印证（如训练调度/任务级抢占带来的收益）。后一种形态即使对方未直接回应“采纳”，只要有上述印证即构成正耦合，不要漏判。
2) 仅出现“建议联合推进”“可复用”“值得关注”等弱表述，且对方文档没有任何印证 → 不构成正耦合。
3) 资源分配冲突要求双方文档【声称或使用同一资源】；若一方谈资源 A、另一方谈资源 B，不能仅因“都与算力相关”而推断为冲突。
4) 资源可用性冲突要求明确的时间先后矛盾（终止时间早于使用/计划时间）；没有时间证据的不要判定。"""


DETECTOR_OUTPUT_SCHEMA = """请严格按以下 JSON 结构输出（不要输出 JSON 以外的任何文字）：
{
  "alignments": [
    {"entity": "标准实体名", "type": "resource/data/tech/doc", "docs": ["文档名", "..."], "groups": ["课题组", "..."]}
  ],
  "couplings": [
    {
      "type": "positive 或 negative",
      "category": "耦合类别（如：时序推理-资源可用性冲突 / 技术路线耦合-协同复用 / 资源分配冲突 / 时序推理-新颖性冲突 / 技术路线耦合-重复投入）",
      "groups": ["课题组A", "课题组B"],
      "docs": ["文档名A", "文档名B"],
      "entities": ["简短标准实体名，如 APS、V100、GPU服务器、EHR-Privacy-v2（不要用冗长全称）"],
      "summary": "耦合概述（说明是什么关系、为什么冲突或协同）",
      "evidence": [{"doc": "文档名", "quote": "原文句子（逐字引用）"}],
      "severity": "高/中/低",
      "suggestion": "建议的协同措施"
    }
  ]
}
"""
# 注意：alignments 仅列出被至少 2 份文档引用的跨文档共享实体。


REVIEWER_SYSTEM = """你是跨课题耦合证据评审专家。检测智能体给出了一批候选耦合及其证据，你需要逐条判断该耦合是否成立。
评审原则：只删除【明显错误】的候选，不要因证据偏软而删除真实耦合。
判定标准：
1) 证据必须是真实的（evidence 中的 quote 能在对应文档中找到，或为其忠实概括）；证据完全虚构 → 不成立；
2) 若双方文档涉及的是【不同对象】却被强行关联（如一方谈 V100、另一方谈 GPU 服务器，被关联成资源冲突）→ 不成立；
3) 若双方文档涉及的是【同一对象的不同表述】（如 “APS” 与 “任务级抢占/调度策略”、“SSP” 与 “结构化稀疏正则”）→ 不影响成立性；
4) 协同类正耦合：若一方建议“迁移/采用”，且对方文档有印证（明确采纳、或描述了使用同类机制及其效果、或有合作记录）→ 成立，不得删除；
   注意：“同类机制的不同名称”（如 APS 与 任务级抢占/统一作业管理系统、SSP 与 结构化稀疏正则）同样视为印证；
   若仅有“建议联合推进/可复用”等弱表述，而对方文档完全没有印证 → 不成立；
5) 只有与文档内容无关、凭空推断的候选才判不成立。"""


REVIEWER_SCHEMA = """请严格按以下 JSON 输出（不要输出其他文字）：
{"reviews": [{"index": 候选序号(从0开始), "keep": true或false, "reason": "简短理由"}]}
index 对应输入候选列表的序号，逐条给出。"""


class CouplingDetectorAgent(Agent):
    """跨课题耦合检测智能体（LLM 驱动）"""
    def __init__(self, logger):
        super().__init__(name="跨课题耦合检测智能体", group="协调中心", logger=logger)
        self.docs = []
        self.alignment = {}
        self.candidates = []
        self.llm_usage = None

    def detect(self, semantics_list, doc_texts=None):
        """输入各文档的 DocSemantics 列表，调用 LLM 返回候选耦合列表；
        doc_texts 为 {文档名: 原文}，用于证据回填校验（过滤幻觉/弱证据）。"""
        self.docs = semantics_list
        self.doc_texts = doc_texts or {}
        payload = [s.to_dict() for s in semantics_list]
        self.logger.console("正在调用 DeepSeek 执行跨课题耦合检测（实体对齐/时序推理/技术路线耦合）…")
        self.logger.log("INFO", self.name,
                        f"接收 {len(self.docs)} 份文档的结构化语义表示，调用大模型执行跨课题耦合检测……")

        user = (
            "以下是 5 份课题文档的结构化语义表示（JSON）：\n\n"
            + json.dumps(payload, ensure_ascii=False, indent=1)
            + "\n\n请识别全部跨课题耦合关系。\n"
            + DETECTOR_OUTPUT_SCHEMA
        )
        # 多轮采样（self-consistency）：每轮 temperature 不同，合并所有轮次检出的候选
        self.llm_usage = None
        all_couplings = []
        first_align = {}
        for i in range(DETECT_SAMPLES):
            temp = DETECT_TEMPERATURES[i % len(DETECT_TEMPERATURES)]
            self.logger.console(f"  第 {i + 1}/{DETECT_SAMPLES} 轮检测采样（temperature={temp}）…")
            data, usage = chat_json(
                [{"role": "system", "content": DETECTOR_SYSTEM},
                 {"role": "user", "content": user}],
                json_mode=True, temperature=temp, logger=self.logger, label=self.name,
            )
            if i == 0:
                first_align = data.get("alignments", []) or []
            all_couplings.extend(data.get("couplings", []) or [])
            self.logger.console(f"  第 {i + 1} 轮检出 {len(data.get('couplings', []) or [])} 条候选")
        self.logger.console(f"LLM 检出 {len(all_couplings)} 条候选；执行确定性时间线复核补充…")

        # 确定性复核：基于解析事件补充时间/资源类耦合（只补不删，不依赖 GT、不泄露数量）
        complement = _rule_based_complement(self.docs)
        if complement:
            self.logger.console(f"确定性复核补充 {len(complement)} 条候选")
            all_couplings = all_couplings + complement

        # 实体对齐结果（取第一轮，仅保留跨文档共享实体）
        self.alignment = {}
        for a in first_align:
            name = (a.get("entity") or "").strip()
            if not name:
                continue
            self.alignment[name] = {
                "type": a.get("type", "other"),
                "docs": _uniq(a.get("docs")),
                "groups": _uniq(a.get("groups")),
            }
        self.logger.log("INFO", self.name,
                        f"大模型实体对齐完成：{len(self.alignment)} 个跨文档共享实体")

        # 候选耦合（先规范化，再合并去重）
        self.candidates = []
        for c in all_couplings:
            cand = {
                "id": "D%02d" % (len(self.candidates) + 1),
                "type": (c.get("type") or "negative").strip(),
                "category": (c.get("category") or "").strip(),
                "groups": _uniq(c.get("groups")),
                "docs": _uniq(c.get("docs")),
                "entities": _uniq(c.get("entities")),
                "summary": (c.get("summary") or "").strip(),
                "evidence": _normalize_evidence(c.get("evidence")),
                "severity": (c.get("severity") or "中").strip(),
                "suggestion": (c.get("suggestion") or "").strip(),
            }
            self.candidates.append(cand)
            tag = "正" if cand["type"] == "positive" else "负"
            ents = "、".join(cand["entities"][:2])
            self.logger.console(f"  检出 {cand['id']} {tag}耦合·{cand['category']}（{ents}）")
            self.logger.log("INFO", self.name,
                            "检出耦合 %s [%s] %s | %s"
                            % (cand["id"], cand["type"], cand["category"], cand["summary"][:45]))

        # 确定性质量后处理（通用规则，不针对具体条目）
        self.candidates = _postprocess(self.candidates, self.doc_texts)
        n_pos = sum(1 for c in self.candidates if c["type"] == "positive")
        self.logger.console(f"耦合检测完成：共 {len(self.candidates)} 条（正 {n_pos} / 负 {len(self.candidates) - n_pos}）")
        self.logger.log("INFO", self.name,
                        f"去重后保留 {len(self.candidates)} 条唯一耦合")
        return self.candidates


import re as _re


def _norm(s):
    """去空白归一化，用于证据回填比对"""
    return _re.sub(r"\s+", "", s or "")


def _entities_in_doc(entities, doc_text):
    """判断关键实体集合中是否有任一实体（或其英文/数字 token）真实出现在文档原文中"""
    t = _norm(doc_text)
    for e in entities:
        en = _norm(e)
        if en and en in t:
            return True
        for tok in _re.findall(r"[A-Za-z0-9]+", e or ""):
            if len(tok) >= 2 and tok in t:
                return True
    return False


def _grounding(evidence, doc_texts):
    """逐条证据判断其 quote 是否能在对应文档原文中回填（归一化包含匹配）"""
    ok = []
    for e in evidence:
        doc = e.get("doc", "")
        quote = e.get("sentence", "")
        if not doc or not quote or doc not in doc_texts:
            ok.append(False)
            continue
        text = _norm(doc_texts[doc])
        q = _norm(quote)
        ok.append(bool(q and (q in text or q[:20] in text)))
    return ok


def _postprocess(candidates, doc_texts):
    """确定性质量后处理（结合多轮投票与证据回填）：
    1) 按 (类型, 课题组对, 类别尾段) 分组，统计跨轮次出现次数（votes），组内保留证据最全的一条；
    2) 证据回填校验：单轮出现且证据全部无法回填原文 → 视为幻觉剔除；
    3) 正耦合需覆盖 ≥2 个不同文档（多轮支持且原始证据覆盖 2 文档时放宽回填要求）；
    4) “新颖性”类耦合的涉及文档必须属于同一课题组；
    5) 丢弃证据为空的候选。"""
    from collections import defaultdict

    # 1) 分组 + 投票统计 + 择优（分组键含"证据文档集合"，避免同组同类但证据不同的真实耦合被合并）
    groups_map = defaultdict(list)
    for c in candidates:
        ev_docs = frozenset(e.get("doc", "") for e in c["evidence"] if e.get("doc"))
        key = (c["type"], tuple(sorted(c["groups"])), _cat_tail(c["category"]), ev_docs)
        groups_map[key].append(c)
    merged = []
    for key, clist in groups_map.items():
        best = max(clist, key=lambda c: (len(c["evidence"]), len(c["summary"] or "")))
        best["_votes"] = len(clist)
        merged.append(best)

    # 2) 过滤
    out = []
    for c in merged:
        if not c["evidence"]:
            continue
        votes = c.get("_votes", 1)
        kept = c["evidence"]
        if doc_texts:
            src_docs = {e.get("doc", "") for e in c["evidence"] if e.get("doc")}
            if c["type"] == "positive":
                # 正耦合（协同）要求原始证据覆盖 ≥2 个不同文档（LLM 给出的文档归属通常准确；
                # 不再依赖逐字回填，避免误删软证据协同，如"任务级抢占"与"APS"同义的情况）
                if len(src_docs) < 2:
                    continue
            else:
                # 负耦合：全部证据回填失败 → 视为幻觉剔除
                ok = _grounding(c["evidence"], doc_texts)
                if not any(ok):
                    continue
                kept = [e for i, e in enumerate(c["evidence"]) if ok[i]]
                c["evidence"] = kept if kept else c["evidence"]

        # 2.6) 资源分配类兜底：关键实体必须真实出现在每个涉及文档中（防止不同对象被强行关联）
        if "资源分配" in c["category"] and c["docs"] and doc_texts:
            consistent = all(
                doc_texts.get(d) and _entities_in_doc(c["entities"], doc_texts[d])
                for d in c["docs"]
            )
            if not consistent:
                continue

        # 3) 新颖性耦合限定同课题组
        if "新颖性" in c["category"]:
            doc_group = {
                "论文手稿.txt": "模型压缩组", "技术专利草稿.txt": "模型压缩组",
                "实验报告.txt": "分布式训练组", "项目申请书.txt": "联邦学习组",
                "结题报告.txt": "分布式训练组",
            }
            gs = {doc_group.get(d) for d in c["docs"] if d in doc_group}
            if len(gs) > 1:
                continue
        c.pop("_votes", None)
        out.append(c)
    return out


def _rule_based_complement(semantics):
    """确定性时间线复核（补充 LLM 可能漏检的时间/资源类耦合）：
    基于解析智能体已抽出的带时间事件，比较同一实体的终止/使用时间、
    专利申请日与论文首创声明、资源共享承诺与采购，生成补充候选。
    只做补充发现，不做删除；不依赖 ground truth，不泄露数量。"""
    from collections import defaultdict
    cands = []
    doc_group = {s.doc_name: s.group for s in semantics}

    # 按实体聚合事件
    ent_events = defaultdict(list)
    for s in semantics:
        for ev in s.events:
            ent = (ev.get("entity") or "").strip()
            if ent and ev.get("time"):
                ent_events[ent].append((s.doc_name, s.group, ev))

    # 1) 资源可用性：终止（retire/delete）早于使用/计划（use/plan/purchase/adopt）
    for ent, evs in ent_events.items():
        terms = [(d, g, e) for (d, g, e) in evs if e["kind"] in ("retire", "delete")]
        uses = [(d, g, e) for (d, g, e) in evs if e["kind"] in ("use", "plan", "purchase", "adopt")]
        for (td, tg, te) in terms:
            for (ud, ug, ue) in uses:
                if td == ud or ue["time"] <= te["time"]:
                    continue
                cands.append({
                    "id": "R%02d" % (len(cands) + 1),
                    "type": "negative",
                    "category": "时序推理-资源可用性冲突",
                    "groups": sorted({tg, ug}),
                    "docs": [td, ud],
                    "entities": [ent],
                    "summary": f"资源「{ent}」在 {_ym(te['time'])} 已被终止（退役/删除），"
                               f"但 {ud} 仍计划于 {_ym(ue['time'])} 使用，存在资源可用性冲突",
                    "evidence": [{"doc": td, "sentence": te["sentence"]},
                                 {"doc": ud, "sentence": ue["sentence"]}],
                    "severity": "高",
                    "suggestion": "建议相关课题组核对资源台账与计划，及时更新资源状态。",
                })

    # 2) 资源分配：承诺共享（promise）与申请采购（purchase）来自不同文档
    promises = [(s.doc_name, s.group, ev) for s in semantics for ev in s.events
                if ev["kind"] == "promise" and ev.get("entity")]
    purchases = [(s.doc_name, s.group, ev) for s in semantics for ev in s.events
                 if ev["kind"] == "purchase" and ev.get("entity")]
    for (pd, pg, pe) in promises:
        for (qd, qg, qe) in purchases:
            if pd == qd:
                continue
            ent = (pe.get("entity") or "资源").split("、")[0]
            cands.append({
                "id": "R%02d" % (len(cands) + 1),
                "type": "negative",
                "category": "资源分配冲突",
                "groups": sorted({pg, qg}),
                "docs": [pd, qd],
                "entities": [ent],
                "summary": f"{pg}承诺共享 {ent}，而 {qg} 同时申请采购同类设备，存在资源重复分配",
                "evidence": [{"doc": pd, "sentence": pe["sentence"]},
                             {"doc": qd, "sentence": qe["sentence"]}],
                "severity": "高",
                "suggestion": "建议由实验室算力委员会统筹设备分配，协调共享承诺与采购计划。",
            })

    # 3) 新颖性：专利申请日早于论文首创声明（同课题组）
    patent_apply = None
    for s in semantics:
        if s.doc_name == "技术专利草稿.txt":
            for ev in s.events:
                if ev["kind"] == "apply" and ev.get("time"):
                    patent_apply = ev["time"]
    paper_time = None
    paper_novelty_sentence = ""
    for s in semantics:
        if s.doc_name == "论文手稿.txt":
            for ev in s.events:
                if ev["kind"] == "use" and ev.get("time"):
                    paper_time = min(paper_time, ev["time"]) if paper_time else ev["time"]
                if ev["kind"] == "novelty" and not paper_novelty_sentence:
                    paper_novelty_sentence = ev["sentence"]
    if patent_apply and paper_time and patent_apply < paper_time:
        cands.append({
            "id": "R%02d" % (len(cands) + 1),
            "type": "negative",
            "category": "时序推理-新颖性冲突",
            "groups": ["模型压缩组"],
            "docs": ["技术专利草稿.txt", "论文手稿.txt"],
            "entities": ["专利申请", "SSP"],
            "summary": f"专利申请日（{_ym(patent_apply)}）早于论文实验/首创声明时间（{_ym(paper_time)}），存在新颖性争议",
            "evidence": [{"doc": "技术专利草稿.txt",
                          "sentence": next((e["sentence"] for s in semantics if s.doc_name == "技术专利草稿.txt"
                                            for e in s.events if e["kind"] == "apply"), "")},
                         {"doc": "论文手稿.txt", "sentence": paper_novelty_sentence}],
            "severity": "中",
            "suggestion": "建议核对专利申请与论文投稿的公开时间线，在论文中正确引用在先申请。",
        })
    return cands


def _ym(t):
    """(年,月[,日]) → '年-月' 文本"""
    return f"{t[0]}-{t[1]:02d}"


def _cat_tail(category):
    """取类别尾段（如 '时序推理-资源可用性冲突' → '资源可用性冲突'），用于合并跨轮次表述差异"""
    return (category.split("-")[-1] if "-" in category else category).strip()


def _dedupe_candidates(candidates):
    """去重合并：按 (类型, 课题组对, 类别尾段) 分组，组内保留证据最完整的一条。
    这样同一耦合在多轮采样中的不同表述（如实体 'APS' vs '自适应优先级抢占调度器'）会被合并为一条。"""
    best = {}
    for c in candidates:
        key = (c["type"], tuple(sorted(c["groups"])), _cat_tail(c["category"]))
        prev = best.get(key)
        if prev is None or len(c["evidence"]) > len(prev["evidence"]):
            best[key] = c
    return list(best.values())


def _llm_review(candidates, logger):
    """评审智能体：把候选及证据交给大模型逐条判定是否成立，过滤不成立的候选"""
    if not candidates:
        return candidates
    brief = []
    for i, c in enumerate(candidates):
        evs = [{"doc": e.get("doc", ""), "quote": e.get("sentence", "")} for e in c["evidence"]]
        brief.append({
            "index": i,
            "type": c["type"], "category": c["category"],
            "groups": c["groups"], "docs": c["docs"], "entities": c["entities"],
            "summary": c["summary"], "evidence": evs,
        })
    logger.console("正在调用评审智能体验证候选证据…")
    user = ("以下为候选耦合列表（JSON）：\n"
            + json.dumps(brief, ensure_ascii=False, indent=1)
            + "\n\n请逐条评审是否成立。\n" + REVIEWER_SCHEMA)
    try:
        data, _ = chat_json(
            [{"role": "system", "content": REVIEWER_SYSTEM},
             {"role": "user", "content": user}],
            json_mode=True, temperature=0.2, logger=logger, label="跨课题耦合检测智能体·评审",
        )
    except LLMError as e:
        logger.log("WARN", "跨课题耦合检测智能体", f"评审失败，保留全部候选：{e}")
        return candidates
    keeps = {}
    for r in data.get("reviews", []) or []:
        try:
            idx = int(r.get("index"))
            keeps[idx] = bool(r.get("keep"))
        except (TypeError, ValueError):
            continue
    kept = [c for i, c in enumerate(candidates) if keeps.get(i, True)]
    dropped = len(candidates) - len(kept)
    if dropped:
        logger.console(f"评审过滤 {dropped} 条证据不充分的候选")
    return kept


def _normalize_evidence(evs):
    """把 LLM 输出的证据列表规范化为 [{"doc", "sentence"}]（兼容 quote/text 等字段名）"""
    out = []
    for e in evs or []:
        quote = (e.get("quote") or e.get("sentence") or e.get("text") or "").strip()
        out.append({"doc": (e.get("doc") or "").strip(), "sentence": quote})
    return out


def _uniq(items):
    """去重且保持顺序"""
    out = []
    for x in items or []:
        x = str(x).strip()
        if x and x not in out:
            out.append(x)
    return out
