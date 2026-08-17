# 第一题 · 多智能体通信协议设计与科研协作写作（DeepSeek 生成版）

本项目是一套**多智能体协同写作系统**：1 个协调器 + 5 个子智能体，通过自研通信
协议协作撰写《基于多智能体强化学习的分布式计算资源调度方法研究》基金申请书。
**全部章节正文由 DeepSeek 大模型从零生成**，不依赖任何本地预置答案。

- **1.2 通信协议设计（12 分）**：消息格式、拓扑、同步并发控制、通信开销建模
- **1.3 端到端演示（8 分）**：任务分解 → 并行起草 → 交叉审查 → 冲突解决 → 最终统稿
- **1.4 可视化（5 分）**：通信序列图、负载分布柱状图、消息类型饼图

---

## 一、目录结构

```
task1/
├── config.py              # DeepSeek API 配置（代码内配置，无需终端环境变量）
├── protocol.py            # 通信协议：消息格式(JSON Schema)/拓扑/并发控制/开销模型/消息总线
├── agents/
│   ├── __init__.py        #   系统装配 build_system()
│   ├── base.py            #   智能体基类（收发消息、回执、写锁协议）
│   ├── coordinator.py     #   协调器：任务分解/调度/冲突仲裁
│   ├── literature_agent.py#   文献调研智能体（正文由 DeepSeek 生成）
│   ├── method_agent.py    #   方法设计智能体（正文由 DeepSeek 生成）
│   ├── experiment_agent.py#   实验规划智能体（正文由 DeepSeek 生成）
│   ├── verifier_agent.py  #   数据/逻辑核查智能体（从正文提取事实并检测冲突）
│   ├── polish_agent.py    #   统稿润色智能体（合并生成 Markdown 申请书）
│   ├── llm.py             #   DeepSeek 客户端（生成 + 算力校验重试）
│   └── content.py         #   知识库：章节任务描述 + 部门私有口径 + 文献资料（无整章答案）
├── demo_proposal_writing.py # 端到端演示入口
├── visualize.py           # 可视化生成
├── logs/                  # 运行日志 + 最终申请书（运行时生成）
└── figures/               # 可视化图（运行时生成）
```

## 二、快速开始

```bash
python demo_proposal_writing.py
```

无需任何终端配置：DeepSeek 的 API Key / 服务地址 / 模型名已配置在 `config.py` 中。
运行后自动生成 `logs/`（消息日志 + 最终申请书）与 `figures/`（4 张图）。

### API 配置（config.py）

```python
API_KEY  = "sk-..."                      # DeepSeek API Key
BASE_URL = "https://api.deepseek.com/v1" # DeepSeek 兼容 OpenAI 协议
MODEL    = "deepseek-chat"               # DeepSeek 模型名
```

> ⚠️ **安全提醒**：`config.py` 包含真实 API Key，请勿提交到公开仓库；
> 如必须提交请先替换为占位符或在 `.gitignore` 中忽略该文件。

## 三、系统如何工作（无预留答案）

**关键设计：本项目没有"事先写好的正文答案"。**

1. `agents/content.py` 只提供三类信息：
   - 每个章节"写什么、什么结构"的**任务描述**（不是答案）；
   - 各智能体的**部门私有口径**（如方法部门认为算力需 3000 GPU·小时、
     实验部门认为预算只有 2000 GPU·小时）——信息不对称使冲突**自然涌现**；
   - 文献调研智能体检索到的**文献资料**（初始调研范围缺 [7]）。
2. 三个起草智能体把"任务描述 + 部门口径"交给 **DeepSeek 从零生成**整章正文，
   生成后做**算力数字校验**（不满足自动以更强约束重试一次）。
3. 数据核查智能体**从正文中提取事实**（正则提取 GPU·小时 数字、JCT/makespan
   术语、[n] 文献引用），做跨章节一致性检查；发现冲突经通信协议上报协调器。
4. 协调器走"**协商 → 仲裁 → 修订 → 复核**"四步闭环：问涉事智能体可调整空间、
   取折中决议、让涉事智能体按决议**重新调用 DeepSeek 生成**修订版、核查智能体复核
   （复核不通过自动重试最多 2 轮）。
5. 统稿润色智能体合并全部章节，统一术语，输出 Markdown 申请书。

**自然涌现的冲突示例**（每次运行由大模型生成，可能不完全相同）：
- C1 算力：方法侧申报 3000 vs 实验侧预算 2000 → 仲裁统一为 2200；
- C2 术语：研究内容用 makespan、实验方案用 JCT → 统一为「平均作业完成时间(JCT)」；
- C3 引用：若正文引用了文献表中缺失的 [7] → 由文献调研智能体补齐。

## 四、1.2 通信协议设计

### 4.1 消息格式（JSON Schema + 字段依据）

`Message` 覆盖题目要求的字段：消息类型（`TASK_ASSIGN/INFO_QUERY/RESULT_SUBMIT/
CONFLICT_NOTIFY/ACK_RECEIPT`）、发送方/接收方（`sender/receiver`）、载荷（`payload`）、
时间戳（`timestamp`）、优先级（`priority`）、关联消息 ID（`related_message_id`）；
扩展 `message_id/correlation_id/session_id/ttl_seconds/ack_required` 等。
正式定义见 `protocol.py` 的 `MESSAGE_JSON_SCHEMA`（JSON Schema 2020-12）与
`validate_message()`。

### 4.2 拓扑（星型控制平面 + 黑板数据平面）

- 控制平面：所有控制消息**一律经协调器转发**，子智能体之间不直接通信；
- 数据平面：章节正文存黑板（`SharedWorkspace`），消息只传摘要/增量；
- 跳数：子→协调器 1 跳，子→子 2 跳；单点故障靠"无状态+日志回放+主备"缓解；
- 扩展性：连接数 O(N)，新增智能体只需注册。

### 4.3 同步与并发控制

- 写冲突防护：**章节级写锁（悲观）+ 乐观版本号**双保险，见 `SharedWorkspace`；
- 跨章节语义冲突：核查智能体从正文提取声明 → 检测 → 上报 → 协调器仲裁 →
  涉事智能体重新生成 → 复核。伪代码见 `protocol.py` 的
  `conflict_detection_pseudocode()` / `conflict_resolution_pseudocode()`。

### 4.4 通信开销建模

`C_total = Σ token(m) = N_ctrl × T_ctrl + N_data × T_data`（`communication_overhead()`）。
优化：摘要替代全文、增量传输、批量回执、TTL 去重。

## 五、日志与可视化

- `logs/messages.jsonl`：每条消息的发送时间/发送方/接收方/类型/摘要；
- `logs/demo.log`：人类可读运行日志；
- `logs/final_proposal.md`：最终申请书；
- `figures/`：通信序列图、负载分布柱状图、消息类型饼图、通信拓扑图。


## 六、跨题联动（加分项）

本项目的通信协议已被提炼为**跨题共享协议** `shared/protocol.py`（与本目录
`protocol.py` 同源同版本），第二题（视觉识别与周报）与第三题（跨课题耦合）
的多智能体通信统一复用该协议；三题消息日志字段一致，由 `shared/trace.py`
生成跨题数据流追溯报告 `bonus/logs/cross_task_trace.md`。详见仓库根目录 `README.md`。
