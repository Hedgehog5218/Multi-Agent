# task2：多智能体视觉识别与科研周报生成系统

本项目对应《多智能体系统能力测评》**第二题**：构建多智能体系统，从异构科研视觉材料中自动提取结构化科研信息并生成实验室周报。

## 一、项目结构

```
task2/
├── data/                        # 8 张科研视觉材料（按 1~8 序号命名，覆盖 8 种类型）
│   ├── 1.png                     # ① 实验装置照片：HunterSE 无人小车
│   ├── 2.png                     # ② 仪器数据面板截图：RViz + LIO-SAM 建图
│   ├── 3.png                     # ③ 论文图表截图：路径规划算法对比
│   ├── 4.png                     # ④ 学术会议 PPT 截图：LQR 与本文控制器对比 + 结果表
│   ├── 5.png                     # ⑤ 文献 PDF 页面截图：论文摘要 + 参考文献
│   ├── 6.png                     # ⑥ 代码运行结果截图：rostopic echo 点云消息
│   ├── 7.jpg                     # ⑦ 组会白板拍照：无人车导航系统设计 + 阿克曼运动模型
│   ├── 8.jpg                     # ⑧ 实验记录本扫描件：坐标系转换结构图
│   └── ground_truth.json         # 每张图片的结构化标注（字段/数值/结构期望值）
├── agents/                       # 6 个专业视觉智能体（8 类材料按特征归并为 6 组）
│   ├── __init__.py               # 包入口（注册表）
│   ├── registry.py               # 8 类材料 -> 6 个智能体 路由表
│   ├── base_agent.py             # 智能体基类（统一结果结构 + 纯 LLM 模式）
│   ├── ocr_utils.py              # OCR 工具（RapidOCR + 缓存 + 内存降级）
│   ├── llm_backend.py            # 大模型后端（数眼智能 qwen3.7-flash 视觉 / OpenAI 兼容）
│   ├── experiment_agent.py       # ① 实验装置照片智能体
│   ├── instrument_agent.py       # ② 仪器数据面板智能体
│   ├── handwritten_agent.py      # ③ 手写材料智能体（组会白板 + 实验记录本）
│   ├── chart_material_agent.py   # ④ 图表材料智能体（论文图表 + 学术会议 PPT）
│   ├── code_agent.py             # ⑤ 代码运行结果智能体
│   └── paper_agent.py            # ⑥ 文献 PDF 页面智能体
├── coordinator.py                # 科研周报协调器（类型判定/路由/汇总/一致性检查/周报）
├── demo_lab_weekly_report.py     # 端到端演示入口（运行流水线 + 评估 + 3 张可视化）
├── prompts.md                     # 8 个智能体的完整 Prompt 设计文档
├── docs/
│   └── task2_答辩材料.md           # 答辩材料：GT 标注清单、评分点对照、FAQ
├── evaluation.py                 # 量化评估（4 个指标）
├── logs/                         # 运行产物（周报、处理日志、评估报告、OCR 缓存）
│   ├── weekly_report.md          # 实验室周报（Markdown）
│   ├── processing_log.json/txt   # 每张图片的路由目标、耗时、提取结果
│   ├── evaluation_report.json    # 评估结果
│   └── ocr_cache.json            # OCR 结果缓存（首次识别后自动生成）
└── figures/                      # 3 张可视化图
    ├── swimlane_diagram.png      # 视觉识别流水线泳道图
    ├── radar_chart.png           # 识别准确率对比雷达图
    └── sankey_diagram.png        # 协调器路由决策桑基图
```

## 二、运行方法

### 环境依赖

- Python 3.9+
- 依赖库：`rapidocr_onnxruntime`（本地 OCR，内置中英文模型）、`matplotlib`、`numpy`、`Pillow`

安装：

```bash
pip install rapidocr_onnxruntime matplotlib numpy pillow
```

### 一键运行（推荐）

**Windows 下直接双击 `run.bat`**，自动完成：检查/安装依赖 → 读取 `config.json` 配置 → 调用数眼智能 qwen3.7-flash 视觉大模型完成识别 → 生成周报/日志/评估/可视化，全程无需在终端输入任何命令。\n\n> 性能优化：8 张图**并行识别**（默认 4 路并发）+ **大图自动压缩**（超过 1600px 先缩放再上传），全程约 **1.5~2 分钟**（串行版约 6 分钟）。

```bash
# 等价于双击 run.bat（Windows）
run.bat
# 或直接运行
python demo_lab_weekly_report.py
```

运行后自动完成：
1. 协调器处理 `data/` 下 8 张图片（类型判定 → 路由 → 结构化提取）
2. 生成实验室周报 `logs/weekly_report.md`
3. 输出处理日志 `logs/processing_log.json`
4. 运行量化评估并输出 `logs/evaluation_report.json`
5. 生成 3 张可视化图到 `figures/`

> 说明：首次运行会执行完整 OCR 并将结果缓存到 `logs/ocr_cache.json`，后续运行直接复用缓存。若图片文件变化，缓存自动失效。

### LLM 配置（config.json，纯 LLM 版，已配置数眼智能 qwen3.7-flash 视觉后端）

所有 LLM 配置集中在项目根目录 `config.json`（**无需手动设置环境变量**），当前已配置：

- **主后端：数眼智能 qwen3.7-flash（多模态视觉）**——`mode = "vision"`，端到端直接看图，能识别照片/手写/手绘/终端截图等各类材料，实测 8 张图全部识别成功、置信度 0.85+；
- **备后端（fallback）：DeepSeek（文本 + OCR）**——主后端调用失败/配额不足时自动降级到 `llm.fallback`（`OCR 提取文字层 + LLM 语义理解`），保证一键运行始终可用。

```json
{
  "llm": {
    "api_key": "sk-数眼智能Key（已配置）",
    "base_url": "https://platform.shuyanai.com/v1",
    "model": "qwen3.7-flash",
    "mode": "vision",
    "fallback": {
      "api_key": "sk-DeepSeekKey（已配置）",
      "base_url": "https://api.deepseek.com/v1",
      "model": "deepseek-chat",
      "mode": "text"
    }
  }
}
```

- **纯 LLM 模式**：识别完全由大模型完成，不再使用本地规则解析兜底
- `base_url` 支持任意 OpenAI 兼容服务（DeepSeek / Qwen / 数眼智能中转等），`proxy` 可配置代理；更换后端只需修改 `config.json`，无需改代码

> 安全提示：`config.json` 含 API Key，已加入 `.gitignore`，提交 Git 前请确认不会上传。

### 单独运行模块

```bash
python coordinator.py    # 仅运行协调器流水线并生成周报
python evaluation.py     # 运行协调器 + 评估
```

## 三、系统设计（对应 2.3）

### 3.1 智能体设计

系统设计 **6 种专业视觉智能体**（符合题目"至少 5 种、各专注 1-2 类"要求），8 类材料按内在视觉/语义特征归并为 6 组：

| 智能体 | 负责材料类型 | 归并依据 | 输出 schema（关键字段） |
| --- | --- | --- | --- |
| ExperimentPhotoAgent | 实验装置照片 | —（专一） | 装置名称、传感器、通信模块、计算设备 |
| InstrumentPanelAgent | 仪器数据面板截图 | —（专一） | 软件、应用、话题、固定坐标系、点云通道、关键读数 |
| HandwrittenMaterialAgent | 组会白板拍照 + 实验记录本扫描件 | 同为手写材料：共享 OCR 纠错、模糊匹配与语义提取 | 白板：主题/模块/公式/变量；记录本：标题/坐标系/结构类型 |
| ChartMaterialAgent | 论文图表截图 + 学术会议 PPT 截图 | 同为展示型图表材料：共享图表标题/子图/对比对象/结果表格提取 | 图表：图标题/子图/对比对象；PPT：图编号/表编号/误差类型/表格数值 |
| CodeResultAgent | 代码运行结果截图 | —（专一） | 命令、话题、消息类型、帧ID、传感器、尺寸参数 |
| PaperPageAgent | 文献 PDF 页面截图 | —（专一） | 论文标题、研究主题、摘要关键词、参考文献条数 |

> **归并理由**：论文图表与 PPT 本质都是"图表 + 文字说明"的展示型材料，白板与实验记录本本质都是"手写材料"，各自提取逻辑高度同源。归并后智能体职责边界更清晰，协调器只需判定 6 类，无需再区分"论文图表 vs PPT""白板 vs 记录本"这两对易混边界。

每个智能体的处理管线（**纯 LLM 驱动**）：**视觉大模型端到端识别（数眼智能 qwen3.7-flash 直接看图，按 `prompts.md` 的 Prompt 蓝图，遵循输出键名契约）→ 结构化 JSON 输出**，并附带置信度与处理耗时；RapidOCR 作为视觉感知层提取图片文字层，供低文字密度材料兜底与日志追溯。智能体代码中**不含任何"预写识别结果"的规则逻辑**，识别完全由大模型完成；每个智能体通过 `numeric_schema`/`structure_schema` 定义输出键名契约，保证大模型输出与评估口径一致。

每个智能体的完整 Prompt 设计（角色/任务/输入/输出要求/处理要点/示例）见 [`prompts.md`](prompts.md)。

### 3.2 识别模式与后端设计

- **主通道（视觉大模型）**：数眼智能 qwen3.7-flash（OpenAI 兼容接口，`mode = "vision"`）端到端看图，按 `prompts.md` 的 Prompt 蓝图与输出键名契约（`fields`/`numeric_fields`/`structure`）输出结构化 JSON，实测 8 张图全部识别成功；
- **备用通道（fallback）**：DeepSeek 文本模型，`OCR 提取图片文字层（含空间位置）→ LLM 语义理解`，主后端失败时自动降级，保证一键运行始终可用；
- 两个通道输出统一 `AgentResult`，评估与可视化完全一致。

### 3.3 协调器工作流程

```
图片输入(data/*.png|jpg)
        │
        ▼
① 类型判定（文件名 + 图像视觉特征 + OCR 文本特征）
        │
        ▼
② 路由（材料类型 → 对应专业智能体）
        │
        ▼
③ 专业智能体识别（视觉大模型端到端看图 + 结构化输出）
        │
        ▼
④ 汇总输出（统一 AgentResult）
        │
        ▼
⑤ 跨材料一致性检查（传感器/坐标系/指标/时间戳/数据自洽）
        │
        ▼
⑥ 实验室周报生成（Markdown）
```

协调器内置 7 类一致性检查规则，例如：
- 装置照片的「激光雷达」与代码结果中的 `/rslidar_points` 话题一致性；
- 记录本坐标系结构与 RViz 面板固定坐标系的 TF 一致性；
- PPT 结果表中「本文控制器指标优于 LQR」与图表结论的一致性；
- 点云 `height × row_step == data 长度` 的数据自洽性。

### 3.4 评估指标（对应 2.3 评估部分）

定义 4 个量化指标：

1. **字段提取准确率**：语义字段值提取正确数 / 期望字段总数（字符串含相似度容忍 OCR 噪声，列表按元素命中率）。
2. **数值识别误差率（MAPE）**：数值字段提取值与 ground truth 的平均绝对百分比误差，缺失按 100% 误差计。
3. **图表结构还原正确率**：图表/版面结构元素（标题、子图数、表格行列、坐标系数等）还原命中率。
4. **类型路由准确率**：协调器判定材料类型与 ground truth 一致的比例（系统级指标）。

## 四、数据准备说明（对应 2.2）

8 张图片围绕「阿克曼底盘无人车导航」科研主线，完整覆盖题目要求的 8 种材料类型，来源均为实验过程实际产出（实拍/截图/白板拍照）。每张图片的 ground truth（结构化字段与期望值）见 `data/ground_truth.json`，字段/数值/结构三层标注。

## 五、评估结果

| 指标 | 结果（数眼智能 qwen3.7-flash 实测） |
| --- | --- |
| 字段提取准确率 | 0.914 |
| 数值识别误差率（MAPE） | 0.0 |
| 图表结构还原正确率 | 0.762 |
| 类型路由准确率 | 1.0 |

8 张图片全部路由正确、数值零误差、置信度 0.85~0.98；字段准确率 0.914（个别字段差异为表述口径：如“LTE 通信模块未识别”“LIO-SAM 应用表述”等，多次运行在 0.886~0.914 间波动）；结构还原率 0.762 的差异集中在 3 张图的**计数口径**（表头行/指标列是否计入、模块粒度划分），并非识别错误，详见 [`docs/task2_答辩材料.md`](docs/task2_答辩材料.md) FAQ Q4.5。逐项结果见 `logs/evaluation_report.json` 与运行终端输出。

## 六、可视化（对应 2.5）

| 图 | 说明 |
| --- | --- |
| `figures/swimlane_diagram.png` | 视觉识别流水线泳道图：从图片输入到周报输出，每条泳道对应协调器/智能体，标注处理时间与数据流向 |
| `figures/radar_chart.png` | 识别准确率对比雷达图：8 张测试图片（按类型）的综合识别得分与智能体置信度对比，展示不同类型识别难度差异 |
| `figures/sankey_diagram.png` | 协调器路由决策桑基图：8 张图片从输入到各专业智能体的分配路径 |

## 七、答辩材料

面向报告 PPT 与答辩的完整材料（ground truth 标注清单、评分点逐条对照、答辩 FAQ）见 [`docs/task2_答辩材料.md`](docs/task2_答辩材料.md)。

## 八、跨题联动（加分项）

- **协议复用**：协调器与视觉智能体之间通过第一题通信协议（`shared/protocol.py`，
  与 `task1/protocol.py` 同源同版本）收发消息：协调器发送 `TASK_ASSIGN` 派发
  图片识别任务，视觉智能体以 `RESULT_SUBMIT` 返回结构化结果，跨材料一致性
  检查以 `CONFLICT_NOTIFY` 通知矛盾。实现见 `agents/protocol_adapter.py` 与
  `coordinator.py`（`use_protocol=True` 时自动启用）。
- **数据流追溯**：协议消息落盘 `logs/protocol_messages.jsonl`（字段与第一题
  `task1/logs/messages.jsonl` 一致），由 `shared/trace.py` 做跨题数据流追溯。
- 传统处理日志仍保留在 `logs/processing_log.json`（每张图片的路由/耗时/提取结果）。
