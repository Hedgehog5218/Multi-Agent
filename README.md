# 多智能体系统能力测评（跨题联动版）

本仓库对应《多智能体系统能力测评：面向科研办公场景的智能协作》四道题 + 加分项。

## 题目结构

| 题目 | 目录 | 说明 |
|---|---|---|
| 第一题 · 多智能体通信协议设计与科研协作写作 | `task1/` | 5 子智能体协作撰写基金申请书 |
| 第二题 · 多智能体视觉识别与科研周报生成 | `task2/` | 8 类视觉材料 -> 实验室周报 |
| 第三题 · 多智能体语义识别与跨课题信息耦合 | `task3/` | 5 份课题文档 -> 耦合检测报告 |
| 第四题 · 多智能体智能路由与科研任务调度 | `task4/` | 科研任务 -> 模型路由 |
| **加分项 · 跨题联动** | `shared/` + `bonus/` | **第一题协议应用于第二、三题，数据流可追溯** |

## 加分项 · 跨题联动（+4 分）

- **协议复用**：`shared/protocol.py` 与第一题 `task1/protocol.py` 同源同版本，
  第二题（`task2/agents/protocol_adapter.py`、`task2/coordinator.py`）与
  第三题（`task3/agents/base.py`）的智能体通信统一使用该协议。
- **冲突解决闭环统一（方案 A）**：三题共用 `shared/conflict_resolution.py` 的
  `ConflictResolutionEngine`，冲突处理统一走「CONFLICT_NOTIFY → INFO_QUERY 协商 →
  仲裁 → TASK_ASSIGN(revise) 修订 → INFO_QUERY 复核」闭环。
- **数据流追溯**：三题协议消息日志字段一致
  （`task1/logs/messages.jsonl`、`task2/logs/protocol_messages.jsonl`、
  `task3/logs/protocol_messages.jsonl`），由 `shared/trace.py` 生成
  跨题数据流追溯报告 `bonus/logs/cross_task_trace.md` 与数据流图
  `bonus/figures/cross_task_dataflow.svg`。
- 详见 [`bonus/README.md`](bonus/README.md)。

## 运行

```bash
# 四题各自端到端演示
cd task1 && python demo_proposal_writing.py
cd task2 && python demo_lab_weekly_report.py
cd task3 && python demo_cross_group.py
cd task4 && python run_demo.py

# 跨题数据流追溯报告
python shared/trace.py
```
