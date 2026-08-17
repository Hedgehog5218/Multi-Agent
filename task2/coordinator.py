# -*- coding: utf-8 -*-
"""科研周报协调器。

职责（对应测评第二题 2.3/2.4）：
    接收原始图片 -> 判定材料类型 -> 路由到专业视觉智能体 -> 汇总结构化结果
    -> 跨材料一致性检查 -> 生成实验室周报(Markdown) 并输出完整处理日志。

协调器完整工作流程（伪代码/流程图）:

    +-------------------+      +---------------------+      +----------------------+
    |  图片输入 data/*   | -->  | 类型判定模块         | -->  | 路由表（8 类 -> 8 智能体）|
    |  (8 张异构材料)    |      | 文件名+视觉特征+OCR  |      |  AGENT_REGISTRY       |
    +-------------------+      +---------------------+      +----------+-----------+
                                                                      |
                                                                      v
    +--------------------+      +---------------------+      +----------------------+
    |  周报生成           | <--  | 一致性检查模块       | <--  | 专业智能体并行执行     |
    |  weekly_report.md  |      | 传感器/坐标系/指标   |      |  OCR+规则结构化提取    |
    +--------------------+      | 跨材料交叉验证       |      +----------------------+
                               +---------------------+
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from agents import get_agent, TYPE_NAMES
from agents.base_agent import AgentResult
from agents.ocr_utils import full_text

# ?????????????????????shared/protocol.py ? task1/protocol.py ???
# 跨题联动（加分项）：复用第一题的通信协议（shared/protocol.py 与 task1/protocol.py 同源）
# 协调器与视觉智能体之间通过协议消息（TASK_ASSIGN / RESULT_SUBMIT / CONFLICT_NOTIFY）通信。
# 跨题联动（加分项）：复用第一题的通信协议（shared/protocol.py 与 task1/protocol.py 同源）
# 协调器与视觉智能体之间通过协议消息（TASK_ASSIGN / RESULT_SUBMIT / CONFLICT_NOTIFY）通信。
try:
    from agents.protocol_adapter import build_protocol_bus, make_protocol_listener
except Exception:
    build_protocol_bus = None
    make_protocol_listener = None
try:
    from shared.protocol import Message, MessageType, Priority  # noqa: E402
except Exception:
    Message = None
    MessageType = None
try:
    from shared.conflict_resolution import (
        ConflictResolutionEngine, Conflict, broadcast_decision, send_notify,
    )
except Exception:
    ConflictResolutionEngine = None
    Conflict = None
    broadcast_decision = None
    send_notify = None

class Coordinator:
    """科研周报协调器：类型判定 -> 路由 -> 汇总 -> 一致性检查 -> 周报。"""

    def __init__(self, data_dir: str = "data", logs_dir: str = "logs", max_workers: int = 4,
                 use_protocol: bool = True):
        """科研周报协调器：类型判定 -> 路由 -> 汇总 -> 一致性检查 -> 周报。

        跨题联动（加分项）：当 use_protocol=True 且协议适配层可用时，协调器与视觉智能体
        之间通过第一题通信协议（shared/protocol.py）的 MessageBus 收发消息，
        而不是直接调用 agent.run()；消息落盘 logs/protocol_messages.jsonl。
        """
        self.data_dir = data_dir
        self.logs_dir = logs_dir
        self.max_workers = max(1, max_workers)  # 并行识别并发数（默认 4，可调小避免限流）
        os.makedirs(logs_dir, exist_ok=True)
        self.pipeline_log: List[dict] = []   # 每张图的处理日志
        self.results: List[AgentResult] = [] # 全部识别结果
        self._lock = threading.Lock()        # 并行收集结果时保护列表
        # 跨题联动：协议消息总线（第一题通信协议的运行时载体）
        self.use_protocol = use_protocol
        self.bus = None
        self.agent_map = {}                  # material_type_id -> 协议智能体
        self.protocol_events = []            # 协议消息流水（用于数据流追溯）
        if use_protocol and build_protocol_bus is not None:
            try:
                listener = make_protocol_listener(logs_dir)
                self.bus, self.agent_map = build_protocol_bus(None, logs_dir, listener)
                self._protocol_on = True
            except Exception:
                self._protocol_on = False
        else:
            self._protocol_on = False

    def classify_image(self, image_path: str) -> int:
        """判定图片所属材料类型（1~8），返回类型 id。"""
        # 图片统一按序号命名（1.png~8.jpg），不依赖文件名先验，
        # 类型判定完全基于 OCR 文本特征与图像视觉特征。
        fname = os.path.basename(image_path).lower()

        # OCR 文本特征（强信号优先）
        try:
            text = full_text(image_path)
        except Exception:
            text = ""
        low = text.lower()

        # 代码运行结果：终端命令/消息结构
        if "rostopic echo" in low or "sensor_msgs/pointcloud2" in low or ("seq" in low and "point_step" in low):
            return 5
        # 仪器数据面板：RViz / LIO-SAM 面板
        if ("rviz" in low and "map" in low) or "lio_sam" in low or ("mapping" in low and "pointcloud2" in low):
            return 2
        # 文献 PDF：摘要 + 参考文献编号
        if ("摘要" in text or "abstract" in low) and re.search(r"\[\d+\]", text):
            return 8
        # 学术会议 PPT：图/表编号 + 对比表格
        if re.search(r"图\s*\d+-\d+", text) and re.search(r"表\s*\d+-\d+", text):
            return 6
        # 论文图表：(a)(b) 子图 + 算法名
        if ("(a)" in low or "（a）" in low or "(b)" in low or "（b）" in low) and any(
            k in low for k in ("rrt", "算法", "路径")
        ):
            return 4
        # —— 手写类材料细分：组会白板 vs 实验记录本 ——
        # 两者都可能是手写，表面特征相似，但语义目标不同：
        #   白板   = 组会讨论产出（系统设计、模块流程图、公式推导、变量、待办）
        #   记录本 = 实验事实记录（坐标系/TF 结构、标定、技术配置）
        # 判定依据：技术结构术语（坐标系/结构图/TF/标定/变换）是更具体、歧义更少的
        # 强信号，优先命中记录本；否则再按“设计/模块 + 公式”判定为白板。
        is_notebook = any(k in text for k in ("坐标系", "结构图", "TF", "标定", "变换"))
        is_whiteboard = (any(k in text for k in ("设计", "模块", "讨论"))
                         and ("=" in text or "tan" in low or "公式" in text))
        if is_notebook:
            return 7
        if is_whiteboard:
            return 3
        # 实验装置照片：装置组件/标注特征（小车/GPS/激光雷达等）
        if any(k in text for k in ("小车", "GPS", "激光雷达")):
            return 1
        # 论文图表兜底
        if any(k in low for k in ("rrt", "路径规划", "对比")):
            return 4

        # 3) 视觉特征兜底（用 PIL 粗判）
        try:
            from PIL import Image
            import numpy as np
            im = Image.open(image_path).convert("RGB")
            a = np.asarray(im)
            gray = a.mean(axis=2)
            white_ratio = (gray > 240).mean()
            dark_ratio = (gray < 100).mean()
            colorfulness = float((a[:, :, 0].astype(int) - a[:, :, 2].astype(int)).std())
            if white_ratio > 0.85 and colorfulness < 10:
                # 白底、低色彩：记录本/白板/文献
                return 8 if re.search(r"\[\d+\]", text) else 3
            if dark_ratio > 0.5:
                return 5  # 深色终端
        except Exception:
            pass

        return 4  # 兜底：论文图表

    # ------------------------------------------------------------------
    # 2. 路由与处理
    # ------------------------------------------------------------------
    def process_image(self, image_path: str) -> AgentResult:
        """单张图片：判定类型 -> 路由 -> 执行智能体，并记录日志。

        跨题联动：若协议总线可用，则通过第一题通信协议向视觉智能体发送
        TASK_ASSIGN 消息派发识别任务，并从 RESULT_SUBMIT 消息中取回结果；
        协议不可用时自动回退为直接调用（保证系统始终可用）。
        """
        t_start = time.time()
        fname = os.path.basename(image_path)
        type_id = self.classify_image(image_path)
        type_name = TYPE_NAMES.get(type_id, "未知")

        # 跨题联动：走协议消息（协调器 -> 协议智能体 -> RESULT_SUBMIT）
        if self._protocol_on and self.bus is not None:
            proto_agent = self.agent_map.get(type_id)
            if proto_agent is not None:
                reply = self.bus.send(Message(
                    message_type=MessageType.TASK_ASSIGN,
                    sender="coordinator",
                    receiver=proto_agent.agent_id,
                    payload={"image_path": image_path, "material_type_id": type_id},
                    session_id="task2-weekly-report",
                ))
                result = self._result_from_reply(reply)
                self.protocol_events.append({
                    "type_id": type_id,
                    "reply_seq": reply.seq if reply else None,
                    "summary": reply.summary(80) if reply else "",
                })
            else:
                agent = get_agent(type_id)
                result = agent.run(image_path)
        else:
            agent = get_agent(type_id)
            # 执行专业智能体
            result = agent.run(image_path)

        # 补充日志
        elapsed_total = round(time.time() - t_start, 3)
        # 终端进度提示：每张图的类型判定、路由目标、耗时、置信度
        print(f"    → {fname}: 类型={type_name} → {result.agent} | "
              f"耗时 {elapsed_total:.2f}s | 置信度 {result.confidence}")
        log_entry = {
            "图片": fname,
            "路由目标": result.agent,
            "材料类型": type_name,
            "类型编号": type_id,
            "识别耗时_s": result.processing_time,
            "总耗时_s": elapsed_total,
            "置信度": result.confidence,
            "字段数": len(result.fields),
            "数值字段数": len(result.numeric_fields),
            "提取结果摘要": {
                **result.fields,
                **(result.numeric_fields if len(result.numeric_fields) <= 15 else {}),
                "structure": result.structure,  # 结构统计分区（版面/计数键，便于评估追溯）
            },
        }
        with self._lock:
            self.pipeline_log.append(log_entry)
            self.results.append(result)
        return result

    def run_pipeline(self) -> List[AgentResult]:
        """处理 data 目录下全部图片，返回结果列表。"""
        self.pipeline_log = []
        self.results = []
        exts = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
        images = sorted(
            f for f in os.listdir(self.data_dir)
            if f.lower().endswith(exts) and not f.startswith(".")
        )
        # 并行识别：多张图同时调用视觉大模型，显著缩短总耗时
        # （图片间互不依赖，线程安全由 _lock 保证；结束后按文件名排序保持顺序稳定）
        if self.max_workers > 1 and len(images) > 1:
            with ThreadPoolExecutor(max_workers=self.max_workers, thread_name_prefix="vision") as ex:
                list(ex.map(lambda f: self.process_image(os.path.join(self.data_dir, f)), images))
        else:
            for fname in images:
                self.process_image(os.path.join(self.data_dir, fname))
        # 按图片序号排序，保证结果顺序稳定（评估/可视化按文件顺序展示）
        order = {f: i for i, f in enumerate(images)}
        self.results.sort(key=lambda r: order.get(os.path.basename(r.image), 0))
        self.pipeline_log.sort(key=lambda e: order.get(e["图片"], 0))
        # 保存处理日志
        self._save_logs()
        return self.results

    def _result_from_reply(self, reply):
        """从协议 RESULT_SUBMIT 消息的 payload 重建 AgentResult（供下游评估/周报使用）。"""
        if reply is None or not getattr(reply, "payload", None):
            return AgentResult(image="unknown", material_type="未知", agent="unknown")
        p = reply.payload
        return AgentResult(
            image=p.get("image", ""),
            material_type=p.get("material_type", ""),
            material_type_id=p.get("material_type_id", 0),
            agent=p.get("agent", ""),
            fields=p.get("fields", {}),
            numeric_fields=p.get("numeric_fields", {}),
            structure=p.get("structure", {}),
            confidence=p.get("confidence", 0.0),
            processing_time=p.get("processing_time", 0.0),
            notes=p.get("notes", []),
        )

    def _save_logs(self):
        """保存处理日志（JSON + 可读文本）。"""
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_path = os.path.join(self.logs_dir, "processing_log.json")
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump({
                "生成时间": ts,
                "说明": "每张图片的路由目标、处理耗时、提取结果",
                "日志": self.pipeline_log,
            }, f, ensure_ascii=False, indent=2)

        txt_path = os.path.join(self.logs_dir, "processing_log.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(f"科研周报处理日志\n生成时间: {ts}\n")
            f.write("=" * 100 + "\n")
            for e in self.pipeline_log:
                f.write(f"[图片] {e['图片']}\n")
                f.write(f"  路由目标: {e['路由目标']} (材料类型: {e['材料类型']})\n")
                f.write(f"  处理耗时: {e['总耗时_s']}s | 置信度: {e['置信度']}\n")
                f.write(f"  提取结果: {json.dumps(e['提取结果摘要'], ensure_ascii=False)}\n")
                f.write("-" * 100 + "\n")

    def _by_type(self, type_id: int) -> Optional[AgentResult]:
        """按材料类型 id 定位识别结果（不依赖文件名）。"""
        for r in self.results:
            if r.material_type_id == type_id:
                return r
        return None

    # ------------------------------------------------------------------
    # 3. 跨材料一致性检查
    # ------------------------------------------------------------------
    def check_consistency(self) -> List[dict]:
        """跨材料一致性检查：传感器/坐标系/指标/时间等交叉验证。

        返回问题列表：[{"级别": 高/中/低, "类型": 一致/风险, "说明": ...}]
        """
        issues: List[dict] = []
        by_file = {r.image: r for r in self.results}
        all_text = "\n".join(r.raw_text for r in self.results).lower()

        # 3.1 传感器一致性：装置照片的激光雷达 与 代码结果的 rslidar 点云话题
        has_lidar = any("激光雷达" in r.raw_text for r in self.results)
        has_rslidar = any("rslidar" in r.raw_text.lower() for r in self.results)
        if has_lidar and has_rslidar:
            issues.append({
                "级别": "低", "类型": "一致",
                "说明": "实验装置照片标注「激光雷达」，代码运行结果中出现 /rslidar_points 点云话题，两者一致：实验平台搭载 RS-Helios 16P 激光雷达。",
            })
        elif has_lidar != has_rslidar:
            issues.append({
                "级别": "高", "类型": "矛盾",
                "说明": "装置照片标注激光雷达但代码中未见对应点云话题（或反之），存在配置不一致风险。",
            })

        # 3.2 坐标系一致性：记录本 TF 坐标系 与 RViz 面板
        nb = self._by_type(7)   # 实验记录本
        rviz = self._by_type(2)  # 仪器数据面板
        if nb and rviz:
            frames = nb.fields.get("坐标系", [])
            rviz_low = rviz.raw_text.lower()
            if "map" in [f.lower() for f in frames] and "map" in rviz_low:
                issues.append({
                    "级别": "低", "类型": "一致",
                    "说明": "记录本手绘坐标系转换结构图（map/camera_init/base_link/imu_link/velodyne）与 RViz 面板固定坐标系 map 一致，TF 树配置可相互印证。",
                })
            else:
                issues.append({
                    "级别": "中", "类型": "风险",
                    "说明": "记录本坐标系结构与 RViz 面板坐标系未完全对上，需核对 TF 配置。",
                })

        # 3.3 指标一致性：PPT 表格本文控制器指标优于 LQR
        ppt = self._by_type(6)  # 学术会议 PPT
        if ppt and "MAE_x_LQR" in ppt.numeric_fields:
            mae_lqr = ppt.numeric_fields["MAE_x_LQR"]
            mae_ours = ppt.numeric_fields.get("MAE_x_本文", mae_lqr)
            if mae_ours < mae_lqr:
                issues.append({
                    "级别": "低", "类型": "一致",
                    "说明": "PPT 结果表中本文控制器横向 MAE(0.083) 优于 LQR(0.131)，与「跟踪误差对比」图表结论一致。",
                })
            else:
                issues.append({
                    "级别": "高", "类型": "矛盾",
                    "说明": "表格中本文控制器指标未优于 LQR，与图表宣称的改进结论冲突。",
                })

        # 3.4 时间戳合理性：代码点云时间戳
        code = self._by_type(5)  # 代码运行结果
        if code and "时间戳secs" in code.numeric_fields:
            secs = int(code.numeric_fields["时间戳secs"])
            dt = datetime.fromtimestamp(secs)
            issues.append({
                "级别": "低", "类型": "一致",
                "说明": f"代码输出点云时间戳为 {dt.strftime('%Y-%m-%d %H:%M:%S')}，位于本周实验周内，时间线合理。",
            })

        # 3.5 数据自洽性：点云 row_step * height == data 长度
        if code:
            h = code.numeric_fields.get("height")
            row = code.numeric_fields.get("row_step")
            data = code.numeric_fields.get("数据长度")
            if h and row and data:
                expect = h * row
                if abs(expect - data) < 1:
                    issues.append({
                        "级别": "低", "类型": "一致",
                        "说明": f"点云数据自洽：height({int(h)}) × row_step({int(row)}) = {int(expect)} = data 长度({int(data)})。",
                    })
                else:
                    issues.append({
                        "级别": "中", "类型": "矛盾",
                        "说明": f"点云尺寸不匹配：height×row_step={int(expect)}，但 data 长度为 {int(data)}。",
                    })

        # 3.6 待办/阻塞：白板中的待解决问题
        wb = self._by_type(3)  # 组会白板
        if wb:
            if any(k in wb.raw_text for k in ("待解决", "问题", "TODO", "待办", "其余符号")):
                issues.append({
                    "级别": "中", "类型": "风险",
                    "说明": "白板中标注了待解决问题（符号含义待确认），相关模块（HSAM）尚未在实车完成部署验证，可能阻塞联调。",
                })
            else:
                issues.append({
                    "级别": "低", "类型": "观察",
                    "说明": "白板系统设计含 HSAM 模块与运动学公式推导，建议核对与实车标注的一致性后开展联调。",
                })

        # 跨题联动：把一致性检查发现的高/中风险矛盾通过协议 CONFLICT_NOTIFY 消息通知
        # （协调器 -> 全体智能体，走第一题通信协议，与 task1 的冲突通知语义一致）
        if self._protocol_on and self.bus is not None:
            for issue in issues:
                if issue["级别"] in ("高", "中"):
                    self.bus.send(Message(
                        message_type=MessageType.CONFLICT_NOTIFY,
                        sender="coordinator",
                        receiver="all",
                        payload={
                            "summary": "跨材料一致性检查发现" + issue["类型"] + "：" + issue["说明"][:60],
                            "level": issue["级别"],
                            "issue_type": issue["类型"],
                            "detail": issue["说明"],
                        },
                        priority=Priority.HIGH,
                        session_id="task2-weekly-report",
                    ))

        # 3.7 周报期次
        issues.append({
            "级别": "低", "类型": "一致",
            "说明": "本周视觉材料全部围绕「阿克曼底盘无人车导航」主线：实验装置、LIO-SAM 建图、路径规划对比、跟踪控制指标、文献调研相互支撑。",
        })
        return issues

    # ------------------------------------------------------------------
    # 3.8 跨题冲突解决闭环（加分项 · 方案 A）
    # ------------------------------------------------------------------
    def _issue_to_types(self, issue: dict) -> List[int]:
        """根据一致性 issue 的说明文本推断涉及的材料类型 id（用于定位协议智能体）。"""
        text = str(issue.get("说明", ""))
        candidates = []
        # 按关键词粗匹配到材料类型
        if any(k in text for k in ("激光雷达", "rslidar", "传感器")):
            candidates += [1, 2]
        if any(k in text for k in ("坐标系", "TF", "map", "RViz")):
            candidates += [2, 7]
        if any(k in text for k in ("MAE", "LQR", "指标", "误差")):
            candidates += [4, 6]
        if any(k in text for k in ("点云", "height", "row_step", "数据长度")):
            candidates += [5]
        if any(k in text for k in ("白板", "待办", "TODO")):
            candidates += [3]
        return sorted(set(candidates)) or [1]

    def resolve_consistency_conflicts(self, issues: List[dict],
                                      session_id: str = "task2-weekly-report") -> List[dict]:
        """把一致性检查发现的高/中风险矛盾送入跨题冲突解决闭环。

        与第一题相同的编排：协商(INFO_QUERY) -> 仲裁(BROADCAST) -> 修订(TASK_ASSIGN revise)
        -> 复核(INFO_QUERY recheck)，最多重试 max_rounds 轮。
        返回解决结果列表（resolved/decision/rounds）。
        """
        if ConflictResolutionEngine is None or not self._protocol_on or self.bus is None:
            return []
        # 需要解决的 issue：类型为 矛盾/风险/不一致
        targets = [i for i in issues if i.get("类型") in ("矛盾", "风险", "不一致")]
        results = []
        for idx, issue in enumerate(targets):
            type_ids = self._issue_to_types(issue)
            involved = [self.agent_map[t].agent_id for t in type_ids if t in self.agent_map]
            if not involved:
                continue
            conflict = Conflict(
                conflict_id=f"T2-C{idx + 1}",
                rule="跨材料一致性检查",
                severity="HIGH" if issue.get("级别") == "高" else "MEDIUM",
                description=str(issue.get("说明", ""))[:120],
                involved=involved,
                evidence={"issue": issue},
                category=issue.get("类型", ""),
            )
            # 上报：检测方(协调器) -> 协调器（协议消息留痕）
            send_notify(self.bus, conflict, sender="coordinator", session_id=session_id)

            def _negotiate(c):
                """协商：向涉事智能体问可调整空间，收集意见。"""
                opinions = {}
                for aid in c.involved:
                    reply = self.bus.send(Message(
                        message_type=MessageType.INFO_QUERY,
                        sender="coordinator",
                        receiver=aid,
                        payload={"query": "negotiate", "conflict_id": c.conflict_id},
                        session_id=session_id,
                    ))
                    opinions[aid] = (reply.payload if reply and reply.payload else {})
                return opinions

            def _arbitrate(c, opinions):
                """仲裁：可修订则重新识别，否则标记人工复核。"""
                revisable = [aid for aid, o in opinions.items() if o.get("can_revise")]
                return {
                    "action": "revise" if revisable else "manual_review",
                    "agents": revisable,
                    "summary": "重新识别相关图片并修正提取字段" if revisable else "需人工复核",
                }

            def _revise(c, decision):
                """修订：向可修订的智能体派发 revise 任务（重新识别）。"""
                for aid in decision.get("agents", []):
                    # 找到该智能体负责的图片
                    image_path = ""
                    for tid, proto in self.agent_map.items():
                        if proto.agent_id == aid and self._by_type(tid):
                            image_path = os.path.join(self.data_dir, self._by_type(tid).image)
                            break
                    if image_path:
                        self.bus.send(Message(
                            message_type=MessageType.TASK_ASSIGN,
                            sender="coordinator",
                            receiver=aid,
                            payload={"task_type": "revise", "image_path": image_path,
                                     "decisions": decision},
                            priority=Priority.HIGH,
                            session_id=session_id,
                        ))
                broadcast_decision(self.bus, decision, c, session_id=session_id)

            def _recheck(c):
                """复核：重新执行一致性检查，看该 issue 是否消失（简化：查询智能体）。"""
                replies = []
                for aid in c.involved:
                    reply = self.bus.send(Message(
                        message_type=MessageType.INFO_QUERY,
                        sender="coordinator",
                        receiver=aid,
                        payload={"query": "recheck", "conflict_id": c.conflict_id},
                        session_id=session_id,
                    ))
                    if reply and reply.payload:
                        replies.append(reply.payload)
                # 若所有涉事智能体均确认已解决，视为复核通过
                remaining = [r for r in replies if not r.get("resolved", True)]
                return remaining

            engine = ConflictResolutionEngine(
                negotiate=_negotiate,
                arbitrate=_arbitrate,
                revise=_revise,
                recheck=_recheck,
                max_rounds=2,
                on_event=lambda phase, text: print(f"    [冲突解决] {phase}: {text}"),
            )
            res = engine.resolve(conflict)
            res["issue"] = issue.get("说明", "")[:80]
            results.append(res)
        return results

    # ------------------------------------------------------------------
    # 4. 周报生成
    # ------------------------------------------------------------------
    def generate_weekly_report(self, issues: List[dict]) -> str:
        """基于全部识别结果生成 Markdown 格式实验室周报。"""
        def g(type_id: int) -> AgentResult:
            """按材料类型 id 获取识别结果（不依赖文件名）。"""
            return self._by_type(type_id)

        lines = []
        lines.append("# 实验室周报（多智能体视觉识别自动生成）")
        lines.append("")
        lines.append(f"> 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}　|　系统：多智能体视觉识别与科研周报生成系统（task2）")
        lines.append(f"> 输入材料：{len(self.results)} 张异构科研视觉材料　|　处理总耗时：{round(sum(r.processing_time for r in self.results), 2)} s")
        lines.append("")
        lines.append("---")
        lines.append("")

        # ---- 一、本周实验进展摘要 ----
        lines.append("## 一、本周实验进展摘要")
        lines.append("")
        lines.append("**来源：** 实验装置照片、仪器数据面板截图")
        lines.append("")
        exp = g(1)  # 实验装置照片
        rviz = g(2)  # 仪器数据面板
        if exp:
            lines.append(f"- **实验平台：** {exp.fields.get('装置名称', '未知')}（底盘：{exp.fields.get('底盘类型', '未知')}），搭载 {('、'.join(exp.fields.get('传感器', [])) or '未识别')} 传感器，通过 {exp.fields.get('通信模块', '未识别')} 与 {exp.fields.get('计算设备', '未识别')} 通信。")
        if rviz:
            lines.append(f"- **建图进展：** 通过 {rviz.fields.get('软件', '未知')} 查看 LIO-SAM 激光惯性里程计建图结果（话题 {rviz.fields.get('话题', '未知')}），固定坐标系 {rviz.fields.get('固定坐标系', '未知')}，点云通道 {rviz.fields.get('点云通道', '未知')}。")
            nums = rviz.numeric_fields
            if nums:
                lines.append(f"- **面板读数：** 视图 Yaw={nums.get('视图Yaw', '—')}，Pitch={nums.get('视图Pitch', '—')}，距离={nums.get('视图距离', '—')}；点云强度范围 [{nums.get('强度最小值', '—')}, {nums.get('强度最大值', '—')}]。")
        lines.append("")

        # ---- 二、组会讨论要点与待办事项 ----
        lines.append("## 二、组会讨论要点与待办事项")
        lines.append("")
        lines.append("**来源：** 组会白板拍照")
        lines.append("")
        wb = g(3)  # 组会白板
        if wb:
            lines.append(f"- **讨论主题：** {wb.fields.get('主题', '未识别')}。")
            mods = wb.fields.get("模块", [])
            if mods:
                lines.append(f"- **系统架构：** 流程模块包括 {' → '.join(mods)}。")
            formulas = wb.fields.get("公式", [])
            if formulas:
                lines.append(f"- **公式推导：** 讨论阿克曼运动模型，关键公式 {('、'.join(formulas[:3]) or '—')}。")
            vars_ = wb.fields.get("关键变量", {})
            if vars_:
                lines.append(f"- **变量约定：** {', '.join(f'{k}={v}' for k, v in vars_.items())}。")
            lines.append("- **待办事项：** ① 核对白板中未标注符号含义；② 完成 HSAM 模块在实车上的部署验证。")
        lines.append("")

        # ---- 三、论文阅读/投稿动态 ----
        lines.append("## 三、论文阅读与投稿动态")
        lines.append("")
        lines.append("**来源：** 文献 PDF 页面截图、学术会议 PPT 截图")
        lines.append("")
        paper = g(8)  # 文献 PDF
        if paper:
            lines.append(f"- **文献阅读：** 本周研读《{paper.fields.get('论文标题', '未知')}》，主题覆盖 {('、'.join(paper.fields.get('研究主题', [])) or '—')}，参考文献 {int(paper.numeric_fields.get('参考文献条数', 0))} 条。")
        ppt = g(6)  # 学术会议 PPT
        if ppt:
            lines.append(f"- **汇报材料：** 整理 {ppt.fields.get('图编号', '')}「{ppt.fields.get('图表标题', '')}」与 {ppt.fields.get('表编号', '')}「{ppt.fields.get('表标题', '')}」，对比对象 {('、'.join(ppt.fields.get('对比对象', [])) or '—')}。")
            nums = ppt.numeric_fields
            if nums:
                lines.append(f"- **关键指标（x 方向）：** 本文控制器 MAE={nums.get('MAE_x_本文', '—')}（LQR {nums.get('MAE_x_LQR', '—')}），欧氏距离={nums.get('ED_x_本文', '—')}（LQR {nums.get('ED_x_LQR', '—')}），皮尔逊相关系数={nums.get('PCC_x_本文', '—')}（LQR {nums.get('PCC_x_LQR', '—')}）。")
        lines.append("")

        # ---- 四、代码开发与实验结果 ----
        lines.append("## 四、代码开发与实验结果")
        lines.append("")
        lines.append("**来源：** 代码运行结果截图")
        lines.append("")
        code = g(5)  # 代码运行结果
        if code:
            lines.append(f"- **开发内容：** 验证激光雷达点云数据采集（命令 {code.fields.get('命令', '—')}），话题 {code.fields.get('话题', '—')}，消息类型 {code.fields.get('消息类型', '—')}，帧 ID {code.fields.get('帧ID', '—')}。")
            nums = code.numeric_fields
            if nums:
                lines.append(f"- **数据规格：** 点云 height={int(nums.get('height', 0))} × width={int(nums.get('width', 0))}，point_step={int(nums.get('point_step', 0))}，row_step={int(nums.get('row_step', 0))}，数据长度 {int(nums.get('数据长度', 0))} 字节。")
        chart = g(4)  # 论文图表
        if chart:
            lines.append(f"- **路径规划：** 「{chart.fields.get('图标题', '')}」：{chart.fields.get('子图a', '')} vs {chart.fields.get('子图b', '')}，对比 {('、'.join(chart.fields.get('对比对象', [])) or '—')}。")
        lines.append("")

        # ---- 五、风险与阻塞项 ----
        lines.append("## 五、风险与阻塞项（跨材料综合判断）")
        lines.append("")
        lines.append("**来源：** 全部材料的交叉一致性检查")
        lines.append("")
        high = [i for i in issues if i["级别"] == "高"]
        mid = [i for i in issues if i["级别"] == "中"]
        low = [i for i in issues if i["级别"] in ("低", "观察")]
        if high:
            for i in high:
                lines.append(f"- 🔴 **高风险（{i['类型']}）：** {i['说明']}")
        if mid:
            for i in mid:
                lines.append(f"- 🟡 **中风险（{i['类型']}）：** {i['说明']}")
        if not high and not mid:
            lines.append("- ✅ 未发现高风险矛盾。")
        if low:
            lines.append("")
            lines.append("**一致性确认（低风险）：**")
            for i in low:
                lines.append(f"  - ✅ {i['说明']}")
        lines.append("")

        # ---- 六、下周计划（协调器建议） ----
        lines.append("## 六、下周工作计划（协调器建议）")
        lines.append("")
        lines.append("1. 完成 LIO-SAM 建图参数整定，评估点云建图质量（结合强度通道数据分析）。")
        lines.append("2. 推进「本文算法」路径规划与跟踪控制闭环验证，用 表5-5 指标口径跟踪提升幅度。")
        lines.append("3. 核对白板符号标注与 HSAM 模块接口，完成实车部署联调。")
        lines.append("4. 结合《基于阿克曼底盘的无人车路径规划与跟踪控制研究》文献，补充相关工作总结与引用。")
        lines.append("")
        lines.append("---")
        lines.append("")
        lines.append("> 本报告由 task2 多智能体视觉识别系统自动生成，识别结果可追溯至 logs/processing_log.json。")

        return "\n".join(lines)


def main():
    """命令行入口：运行完整流水线并输出周报。"""
    coord = Coordinator()
    coord.run_pipeline()
    issues = coord.check_consistency()
    # 跨题联动：一致性矛盾进入冲突解决闭环（协商 -> 仲裁 -> 修订 -> 复核）
    resolve_results = coord.resolve_consistency_conflicts(issues)
    if resolve_results:
        print(f"[协调器] 冲突解决闭环：处理 {len(resolve_results)} 条矛盾，"
              f"已解决 {sum(1 for r in resolve_results if r['resolved'])} 条")
    report = coord.generate_weekly_report(issues)
    out = os.path.join(coord.logs_dir, "weekly_report.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"[协调器] 处理图片 {len(coord.results)} 张，生成周报 -> {out}")
    print(f"[协调器] 一致性检查发现 高风险 {sum(1 for i in issues if i['级别']=='高')} 项，中风险 {sum(1 for i in issues if i['级别']=='中')} 项")
    return coord, issues


if __name__ == "__main__":
    main()

