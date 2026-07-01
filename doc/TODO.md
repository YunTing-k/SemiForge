# DG-TFT Agent 深度集成 — 架构设计与开发 TODO

> 本文档不进仓库（`.gitignore`），仅用于开发过程跟踪。
> 目标：将 Agent 与 Workflow 深度融合，实现智能化的数据准备、训练调参、结果解读。

---

## 开发原则
1. **本文档是唯一权威** — 每次 compact 后，agent 必须先完整阅读本文档再开始工作。AGENTS.md 已强制执行此规则。
2. **不硬编码** — 路径、常量、文件夹名等不可硬编码：固定不变的值放 `constants.py`，需用户可调的放 `agent_configs.json`。目录名用下划线
3. **代码全英文，带有必要注释**
4. **代码规范，带有unittest测试**
5. **用户可见性** — 除了日志系统外，必要的流程，进度，信息需要通过console print展示，相关配色和格式参考已有代码（rich开发相关注意事项见`C:\Users\admin\Desktop\PythonFile\TECoSimAgent\doc\rich_pitfalls.md`）
6. **系统管理意识** — 开发中及时对照架构文档，及时更新。如果架构和实现有冲突或者有实现的抉择，需要及时和用户讨论
7. **`minimal_run` 优先** — 任何新增的管线功能（agent 介入、诊断、重试、质量门、知识反哺等）必须在实现时检查 `minimal_run` 开关：当 INI `[runtime] minimal_run = yes` 时，该功能应被跳过，仅执行最简脚本流程。新增 INI 级行为开关统一放在 `[runtime]` section 下
8. **版本号规则** — 当前主版本锁定 1.x.x，本次大阶段彻底结束后才升 2.0.0。小功能/修复改末位（1.0.x），中规模功能/新模块改第二位（1.x.0）
9. **测试零残留** — 所有测试不得向真实 `session/`、`log/` 等持久化目录写入文件。测试必须使用 `tempfile.TemporaryDirectory` 或 mock `SESSION_PATH` 指向临时目录。tearDown 中必须清理所有临时文件。测试结束后运行 `Get-ChildItem session` 确认无新增残留。

## 架构原则

1. **Agent 只在失败/不达标时介入** — 正常路径不跑 Agent，保证效率
2. **Agent 只决策，不执行** — 调参方案由 Agent 产出，参数写入和脚本执行是固定代码
3. **结构化输出，不走自由文本** — 所有 Agent 关键动作通过 tool function call / JSON schema
4. **参数隔离** — 每个 Agent 只能改自己 INI section，不能越界
5. **回退单向线性链** — 每次 1 跳，方向只能向上游，由上一级 Agent 决定是否继续跳
6. **可观测性** — 每次 Agent 决策、每次回退、每次重试都有结构化日志和 TUI 反馈
7. **可配置** — 超参、重试次数、模型 tier 全部走 `agent_configs.json`
8. **测试先行** — 每个子任务有对应单元/集成测试
9. **全量记录，不预选 best** — 所有试次保留不删，Agent E 结合偏好和专家知识判断最优路径
10. **偏好驱动** — 用户偏好注入全链路 Agent，影响调参决策、回退判断和最终解读

## 零、当前代码基线

| 模块 | 文件 | 当前状态 |
|------|------|----------|
| 数据准备 | `data_preparer.py` | worker SubAgent → verifier LLM → 程序检查，**单次执行无重试**，worker/verifier 之间无交接 |
| 阶段定义 | `stages.py` | 5 个 StageDef，线性顺序，`depends_on` 仅声明依赖关系 |
| 阶段执行 | `stage_runner.py` | `run_stage()` 调 Python 子进程执行建模脚本，流式输出 |
| 编排器 | `orchestrator.py` | 顺序执行 + fail-fast + resume + 失败诊断 |
| 质量门 | `quality_guard.py` | **纯规则驱动**：静态阈值 + 逐步参数调整策略表 + 安全边界 + 白名单 |
| 指标解析 | `metrics_parser.py` | 从 JSON 产物中提取 RMS 等数值指标 |
| 报表生成 | `report_builder.py` | 纯规则驱动，生成 MD/JSON/Manifest 三份报告 |
| 错误诊断 | `diagnostics.py` + `subagent_hooks.py` | 规则分类 + 可选 Explorer SubAgent 诊断 |

### 当前 Agent 介入点

1. **数据准备 worker** — SubAgent，处理原始数据（`data_preparer.py:307`）
2. **数据准备 verifier** — 轻量 LLM，验证输出结构（`data_preparer.py:356`）
3. **失败诊断 explorer** — SubAgent，阶段失败后只读诊断（`orchestrator.py:468`）

**训练阶段（fit-physical / train-static / train-hysteresis）完全没有 Agent 参与。**

### 当前产出路径（问题）

当前工作流产出分散在三个平级目录中，管理混乱：

```
<project_root>/
├── session/<uuid>/         # 会话元数据（context.json, tasks.json, workflow_runs.json, agents/）
├── outputs/dg_tft_runs/<uuid>/run_NNNN/   # 工作流运行产物
├── prepared_data/<uuid>/1/                 # 数据准备产物（prep_id 固定为 "1"）
```

**现存问题**：
1. `outputs/` 和 `prepared_data/` 是 `session/` 的平级目录，不在 `session/<uuid>/` 下，清理会话时无法一并删除（`session_remove_cli` 只删 `session/<uuid>/`）
2. 三个目录各自嵌 `session_uuid` 子目录，路径冗长且重复
3. `prep_id` 硬编码为 `"1"`（`tool_def.py:2631`），`PreparationTracker._next_prep_id` 计数器增长但实际不使用
4. `manager.py:260` 使用硬编码 `"session"` 字符串而非 `SESSION_PATH` 常量
5. `DG_TFT_OUTPUT_ROOT` 配置键不存在（`tool_def.py:2359` 引用了不存在的 key，永远 fallback）
6. `PreparationTracker` 文档注释写 `session/<uuid>/prepared_data/` 但实际路径是 `prepared_data/<uuid>/`
7. 无法通过单一目录观察某个 session 下的全部产物

---

## 零改一、产出路径统一到 session/<uuid>/

### 目标结构

所有工作流相关的模型产出、数据准备产物、运行记录、报告全部放入 `session/<uuid>/`：

```
session/<uuid>/
├── user_history                 # 现有 — shell 历史
├── context.json                 # 现有 — AgentContext 序列化
├── cron_configs.json            # 现有 — cron 配置
├── tasks.json                   # 现有 — Scoreboard 任务
├── messages.json                # 现有 — LLM 消息历史
├── workflow_runs.json           # 现有 — WorkflowManager + PreparationTracker 持久化
├── agents/                      # 现有 — SubAgent dump 文件
└── runs/                        # 新建 — 所有工作流运行产物（取代 outputs/ + prepared_data/）
    └── run_NNNN/
        ├── model_config.ini     # 运行时 INI 配置
        ├── dataset_profile.json
        ├── config_resolution.json
        ├── data_prep/           # 数据准备产物（原 prepared_data/<uuid>/1/）
        │   ├── attempt_1/
        │   │   ├── *_train.xlsx
        │   │   ├── CV_train.xlsx
        │   │   ├── it/
        │   │   ├── worker_report.json
        │   │   ├── verifier_result.json
        │   │   └── manifest.json
        │   └── attempt_2/
        ├── check-data/
        │   └── attempt_1/
        ├── 01_physical_kernel/
        │   ├── attempt_1/
        │   └── attempt_2/
        ├── 03_full_model/
        │   └── attempt_1/
        ├── 04_hysteresis/
        │   └── attempt_1/
        ├── operation_log.json   # 全量操作记录（TODO 第 3.6 节）
        ├── RUN_SUMMARY.md
        ├── RUN_SUMMARY.json
        └── ARTIFACT_MANIFEST.json
```

> 注：专家知识文件（`knowledge/*.md`）和 Agent 反哺的发现知识（`knowledge/_discovered/*.md`）是**公共目录**（见 3.5 和 4.4），不属于 `session/<uuid>/`，不归入本文档的 session 路径统一范围。

### 改动项

#### 删除旧路径常量 / 配置

| 旧项 | 位置 | 操作 |
|------|------|------|
| `DG_TFT_DEFAULT_OUTPUT_ROOT` = `"./outputs/dg_tft_runs"` | `agent_configs.json:64` | **删除** |
| `_PREP_OUTPUT_BASE_DIR` = `"prepared_data"` | `data_preparer.py:43` | **删除** |
| 硬编码 `"session"` 字符串 | `manager.py:260` | 改用 `SESSION_PATH` 常量 |

#### 新增 / 修改配置

| 新配置项 | 默认值 | 用途 |
|----------|--------|------|
| `DG_TFT_RUNS_SUBDIR` | `"runs"` | 运行目录名，相对于 `session/<uuid>/` |

#### 代码改动

| # | 文件 | 改动 |
|---|------|------|
| 0 | `manager.py:260` | `_runs_file()`: `os.path.join("session", ...)` → `os.path.join(SESSION_PATH, ...)` |
| 1 | `config_builder.py:31-32,71-79,124` | `_session_id()` 新增 `session_dir` 输出；`build_runtime_config()` 构造 `session/<uuid>/runs/run_NNNN/` 替代 `outputs/dg_tft_runs/<uuid>/run_NNNN/`；删除 `DG_TFT_DEFAULT_OUTPUT_ROOT` 引用 |
| 2 | `config_builder.py:32` | `_session_id()` 新增返回 `runs_base_dir`：`Path(session_path) / session_uuid / runs_subdir` |
| 3 | `tool_def.py:2358-2365` | 删除不存在 key `DG_TFT_OUTPUT_ROOT` 的读取和使用；`start_dg_tft_workflow` 的 `output_root` 参数语义更新 |
| 4 | `tool_def.py:2630-2631` | `prepare_dg_tft_dataset` tool handler：输出目录从 `prepared_data/<uuid>/1` 改为 `session/<uuid>/runs/run_NNNN/data_prep/attempt_1/`（在数据准备重试循环中改为 `attempt_N/`） |
| 5 | `data_preparer.py:43` | 删除 `_PREP_OUTPUT_BASE_DIR`，输出目录由调用方传入 |
| 6 | `data_preparer.py:262` | `prep_id` 派生逻辑更新（不再从 `"1"` 目录名推断） |
| 7 | `manager.py:38` | 修正 `PreparationTracker` 文档注释（`session/<uuid>/prepared_data/` → 实际路径） |
| 8 | `orchestrator.py:135` | `_read_output_dir()` 仍读 INI 中的 `paths.output_dir`，无需改动（INI 由 `config_builder` 写入正确路径） |
| 9 | `orchestrator.py:755` | `_retry_N/` 重试路径：保留逻辑，待 TODO 3.6.5 改为 `attempt_N/` 且不删除 |
| 10 | `artifacts.py:79` | 修正文档注释示例路径 |
| 11 | `.gitignore` | 删除 `/prepared_data/` 和 `/outputs/`（产出已归入 `/session/` 下） |
| 12 | `session.py:300-341` | `session_remove_cli()`：`shutil.rmtree(session/<uuid>/)` 一次性清理全部内容（无需额外清理 `outputs/` 和 `prepared_data/`） |
| 13 | `agent_configs.json` | 删除 `DG_TFT_DEFAULT_OUTPUT_ROOT`；新增 `DG_TFT_RUNS_SUBDIR`：`"runs"` |
| 14 | `agent_context.py` | 无需改动 — Session 路径只读保护范围扩展确认（`system_read_only_paths` 包含 `./session/`，runs 子目录自然受保护） |
| 15 | `test/dg_tft_integration_test.py:50` | mock 配置中 `DG_TFT_DEFAULT_OUTPUT_ROOT` → `DG_TFT_RUNS_SUBDIR` |
| 16 | `test/dg_tft_builder_test.py:157` | 测试中 `output_root=str(...)` 语义不变（测试用 temp dir，不依赖真实 session 路径） |

### TODO

- [x] **0.0.1** `manager.py`：`_runs_file()` 使用 `SESSION_PATH` + `WORKFLOW_RUNS_NAME` 常量替代硬编码
- [x] **0.0.2** `agent_configs.json`：删除 `DG_TFT_DEFAULT_OUTPUT_ROOT`，新增 `DG_TFT_RUNS_SUBDIR` = `"runs"`
- [x] **0.0.3** `config_builder.py`：路径构造改为 `session/<uuid>/<DG_TFT_RUNS_SUBDIR>/run_NNNN/`
- [x] **0.0.4** `tool_def.py`：删除不存在 key `DG_TFT_OUTPUT_ROOT`；`prepare_dg_tft_dataset` 输出目录改用新路径
- [x] **0.0.5** `data_preparer.py`：删除 `_PREP_OUTPUT_BASE_DIR`，输出目录改为调用方传入参数
- [x] **0.0.6** `manager.py`：修正 `PreparationTracker` 文档注释
- [x] **0.0.7** `artifacts.py`：修正文档注释示例路径
- [x] **0.0.8** `.gitignore`：删除 `/prepared_data/` 和 `/outputs/` 行
- [x] **0.0.9** `session.py`：确认 `session_remove_cli` 可一次性清理 session 目录（无需额外改动）
- [x] **0.0.10** 测试：更新 `dg_tft_integration_test.py` 等 mock 配置键名
- [x] **0.0.11** 测试：验证 `session_remove_cli` 清理后无残留目录
- [x] **0.0.12** 全量测试通过验证
- [x] **0.0.13** `constants.py`：新增 `WORKFLOW_RUNS_NAME`、`PREP_SUBDIR_NAME`
- [x] **0.0.14** `run_model.py` + INI 模板：`quick_run`/`debug_run` → `minimal_run` 合并
- [x] **0.0.15** `prompt.py`：添加 `minimal_run` override 指引
- [x] **0.0.16** `model_config.ini`：过期 `quick_run`/`debug_run` → `minimal_run`

---

## 零改二、基础设施改造 — SubAgent 增强

> 以下改造是 Phase 2/3 的前置依赖，不依赖业务逻辑，可独立提前完成。

### SubAgent 增强：本地 Tool 注入 + 终止模式

#### 问题

当前 `SubAgent` (`src/agent/subagent.py`) 有三个限制：

1. **无法注入自定义 tool** — `build_tools()` 只能从 `tool_def.py` 固定函数列表 + MCP 工具取，建模 Agent 需要的 `adjust_params` / `fallback_to_previous` / `inspect_xlsx` 等无注入口
2. **接受纯文本作为有效终止** — `run()` 循环中 LLM 返回 `content`（纯文本）就直接 `DONE`，无法强制要求结构化输出
3. **终止模式单一** — 没有区分"干活 agent 文本输出就行"和"决策 agent 必须走 schema"的场景

#### 本地运行时 Tool

`subagent_tools.py` 中的 tool 是独立于 `tool_def.py` / `tool_dispatch.py` 的完整工具实现。不走全局 dispatch，由 `execute_tool_calls()` 本地执行：

```
ctx.tools 构成:
  1. tool_def.py 的固定 tool → tool_dispatch.py 分发
  2. MCP tool → tool_dispatch.py 分发
  3. build_tools() SUPPORTED_TYPES 过滤
  4. 追加 subagent_tools (本地 tool)
  5. 发送给 LLM

execute_tool_calls() 分发:
  · tool 在 subagent_tools 中 →
      有 handler → 本地执行，结果返回 LLM（调查 tool）
      无 handler → 捕获 arguments（输出 tool）
  · 否则 → call_tools() → tool_dispatch.py
```

#### 终止模式（stop_mode）

```python
agent = SubAgent(
    stop_mode="plain_text",  # 默认：纯文本终止（内容→reasoning 兜底→提醒重试）
    # 或
    stop_mode="schema",      # 必须调用 stop_schemas 中的 tool 才能终止
    stop_schemas=["adjust_params", "fallback_to_previous"],
    # 或
    stop_mode="any",         # 不强制，任何时候终止都行
)
```

三种模式：

| stop_mode | 何时 DONE | 适用场景 |
|-----------|-----------|----------|
| `plain_text` | LLM 返回 content（文本）。content 为空→fallback 到 reasoning。都没有→提醒 LLM 输出，继续循环 | Preparer（干活 agent，给个总结就行） |
| `schema` | LLM 调用了 `stop_schemas` 列表中的任意一个 tool | 所有建模 Agent（必须结构化输出决策） |
| `any` | 任何情况（文本、tool call、reasoning）都行 | Agent E（解读，输出文本报告） |

`schema` 模式下：
- LLM 调用 `inspect_xlsx`（调查 tool，在 subagent_tools 中有 handler）→ 执行并返回结果 → 不终止，继续
- LLM 调用 `adjust_params`（输出 tool，在 stop_schemas 中）→ 捕获 arguments → DONE
- LLM 返回纯文本 → reminder "you must call [tool_name] to submit your decision" → 继续

#### 预期接口

```python
# 干活 Agent
agent = SubAgent(subagent_type="preparer")

# 决策 Agent
agent = SubAgent(
    subagent_type="explorer",
    subagent_tools=subagent_tools_for("fit-physical"),   # adjust_params + switch_kernel + fallback
    stop_mode="schema",
    stop_schemas=["adjust_params", "switch_kernel", "fallback_to_previous"],
)

# 解读 Agent
agent = SubAgent(
    subagent_type="explorer",
    stop_mode="any",
)
```

#### 改动点

| # | 位置 | 改动 |
|---|------|------|
| 1 | `__init__` | `extra_tools` → `subagent_tools: list[dict]`，新增 `tool_handlers: dict`、`stop_mode: str`、`stop_schemas: list[str]` |
| 2 | `build_tools()` | 在过滤后的 `ctx.tools` 末尾追加 `self.subagent_tools` |
| 3 | `build_messages()` | `stop_mode` 不为 `plain_text` 时 system prompt 措辞变化 |
| 4 | `execute_tool_calls()` | subagent_tool 拦截 → 查 `tool_handlers`（有则本地执行返回，无则捕获 arguments）；`stop_mode="schema"` 且匹配 `stop_schemas` → 捕获 → DONE |
| 5 | `run()` 循环 | `stop_mode="plain_text"` → content/reasoning 终止逻辑；`stop_mode="schema"` → 纯文本不终止，追加 reminder；`stop_mode="any"` → 不检查模式，随时 DONE |
| 6 | 空响应兜底 | `stop_mode="schema"` 且 max_steps 耗尽未调 stop_schemas 中 tool → ERROR |

#### TODO

- [x] **0.1.1** `SubAgent.__init__`：`extra_tools` → `subagent_tools`，新增 `tool_handlers`、`stop_mode`、`stop_schemas`
- [x] **0.1.2** 创建 `src/tool/subagent_tools.py` 空壳
- [x] **0.1.3** `build_tools()` 追加 `subagent_tools` 到 `ctx.tools`
- [x] **0.1.4** `build_messages()` 根据 `stop_mode` 调整 system prompt 措辞
- [x] **0.1.5** `execute_tool_calls()` 本地 tool 分发：有 handler → 执行返回；无 handler → 捕获；`stop_mode="schema"` 且匹配 `stop_schemas` → DONE
- [x] **0.1.6** `run()` 循环：三种 stop_mode 终止逻辑
- [x] **0.1.7** 单元测试：`stop_mode="plain_text"` → 纯文本/内容空→reasoning 兜底
- [x] **0.1.8** 单元测试：`stop_mode="schema"` → 调查 tool 执行不终止，输出 tool 捕获终止，纯文本不终止
- [x] **0.1.9** 单元测试：`stop_mode="any"` → 任何响应都 DONE
- [x] **0.1.10** 单元测试：`stop_mode="schema"` + max_steps 耗尽 → ERROR

### 涉及文件

| 文件 | 改动类型 |
|------|----------|
| `src/agent/subagent.py` | **增强** — `subagent_tools` + `tool_handlers` + `stop_mode` + `stop_schemas` |
| `src/tool/subagent_tools.py` | **新建** — 所有 subagent 专用 tool 的 schema 定义 + 调查类 tool 的 handler 实现 |
| `src/agent/progress.py` | **无需改动** — `SUPPORTED_TYPES` 维持现有过滤逻辑 |

---

## 一、总体架构

### 1.1 流水线与回退链

```
Raw Data                        Standard Data
   │                                  │
   ▼                                  │
┌──────────┐                          │
│ Preparer │ (Agent, 仅原始数据入口)     │
│  重新整理  │                          │
└────┬─────┘                          │
     │                                │
     ▼                                ▼
┌──────────┐
│ Verifier │ (Agent, 统一入口)
│  验证数据  │
└────┬─────┘
     │  handoff
     ▼
┌──────────────┐
│  check-data  │ ←── Agent A（失败时介入）
│  (脚本执行)   │
└──────┬───────┘
       │  handoff
       ▼
┌──────────────┐
│ fit-physical │ ←── Agent B（失败/质量不达标时介入）
│  (脚本执行)   │
└──────┬───────┘
       │  handoff
       ▼
┌──────────────┐
│ train-static │ ←── Agent C（失败/质量不达标时介入）
│  (脚本执行)   │
└──────┬───────┘
       │  handoff
       ▼
┌──────────────────┐
│ train-hysteresis │ ←── Agent D（失败/质量不达标时介入）
│   (脚本执行)      │
└──────┬───────────┘
       │
       ▼
┌──────────────────────────┐
│  Expert Agent E          │
│  解读结果 + 生成诊断报告    │
└──────────────────────────┘
```

### 1.2 回退拓扑（线性链，逐级上跳）

```
train-hysteresis → train-static → fit-physical → check-data → Verifier → Preparer
      (Agent D)  →   (Agent C)  →   (Agent B)  →  (Agent A) → (Verifier) → (Preparer)

规则：
· 每个 Agent 只能回退到紧邻的上一级（1 跳）
· 上一级 Agent 收到 fallback 后，自行决定是否需要继续向上回退
· 理论上最底层的 D 可以一路回退到 Preparer（D→C→B→A→Verifier→Preparer）
· 实现层保证回退方向正确：只能向上游，不能向下游或平级
```

### 1.3 Agent 触发时机

Agent **不持续运行**，仅在以下两种情况下被触发：

1. **硬失败**：脚本崩溃、退出码非 0、产物缺失
2. **软失败**：脚本跑通但质量不达标（RMS 超阈值等）

达标则直接进入下一阶段，不触发 Agent。

### 1.4 参数试次管理

无论 Agent 是否触发，**每次阶段执行都全量记录**。每个阶段可能经历多次执行（初始 + Agent 调参后重试 + fallback 回来的重新执行）：

```
Per-Stage Attempt Log (每次 run_stage 调用都记录):
  · 第 N 次执行的参数配置（INI snapshot）
  · 第 N 次执行的多个指标结果（metrics，至少 1 个，如 train-static 有 IV + CV）
  · 第 N 次执行的触发原因（initial / agent_adjust / quality_fail / fallback_sent / fallback_received / upstream_changed / verifier_fail / prep_retry）
  · 上游依赖标记（如 train-static.attempt_1 基于 fit-physical.attempt_2）

产物存储：
  · outputs/{run_id}/{stage}/attempt_N/ — 保留不删
  · operation_log.json 记录完整依赖树
  · 最终输出：Agent E 结合偏好 + 专家知识判断最优路径
```

### 1.5 外部 Tool 统一

当前 DG-TFT 对外暴露 4 个独立 tool，对应新旧混杂的流程：

| 当前 Tool | 问题 |
|-----------|------|
| `prepare_dg_tft_dataset` | 数据准备被视为独立步骤，和后续建模割裂 |
| `start_dg_tft_workflow` | 仅启动建模，不管数据准备 |
| `query_dg_tft_workflow` | 状态查询 |
| `read_dg_tft_workflow_report` | 读报告 |

新架构下数据准备—建模—解读是**一个有机整体**，Verifier 桥接两端，回退链贯通全程。收敛为 3 个 tool：

```jsonc
// ── 1. dg_tft_run ── 一键跑完整 Pipeline
{
  "name": "dg_tft_run",
  "description": "Run the complete DG-TFT modeling pipeline: data preparation → verification → modeling → interpretation.",
  "parameters": {
    "dataset_dir": {"type": "string", "description": "Path to raw or standard dataset directory"},
    "standard_data": {"type": "boolean", "default": false, "description": "If true, skip Preparer and go straight to Verifier"},
    "hysteresis_dir": {"type": "string", "description": "Optional path to I-t hysteresis data"},
    "device_name": {"type": "string", "description": "Optional device identifier for output naming"},
    "preferences": {"type": "string", "description": "User-specified modeling priorities, e.g. 'hysteresis accuracy first, CV secondary, IV can be loose'. Injected into all agents and final diagnosis."}
  },
  "returns": "{ run_id, status }"
}

// ── 2. dg_tft_status ── 阶段级进度查询
{
  "name": "dg_tft_status",
  "description": "Query the progress of a DG-TFT pipeline run, returned at stage granularity.",
  "parameters": {
    "run_id": {"type": "integer", "description": "Run ID returned by dg_tft_run"}
  },
  "returns": "{
    run_id, status, current_stage,
    stages: [
      { name: 'data-prep', status: 'DONE' },
      { name: 'check-data', status: 'RUNNING' },
      ...
    ],
    elapsed_s, diagnostics (if failed)
  }"
}

// ── 3. dg_tft_report ── 读最终诊断报告
{
  "name": "dg_tft_report",
  "description": "Read the final modeling report and AI diagnostic interpretation for a completed run.",
  "parameters": {
    "run_id": {"type": "integer", "description": "Run ID"}
  },
  "returns": "Markdown report text (RUN_SUMMARY.md) with interpretation appended"
}
```

**内部映射**：`dg_tft_run` 内部根据 `standard_data` 标志走不同的入口（2.1 两入口设计），串联所有阶段到 Agent E 解读。`preferences` 注入所有 Agent 的 system prompt + 最终解读。`dg_tft_status` 返回阶段级粒度，TUI 层渲染为进度条。

#### 涉及文件

| 文件 | 改动 |
|------|------|
| `constants.py` | 删除 `TOOL_NAME_PREPARE_DATASET`，`TOOL_NAME_START_WORKFLOW` → `TOOL_NAME_DG_TFT_RUN`，`TOOL_NAME_QUERY_WORKFLOW` → `TOOL_NAME_DG_TFT_STATUS`，`TOOL_NAME_READ_WORKFLOW_REPORT` → `TOOL_NAME_DG_TFT_REPORT` |
| `tool_def.py` | 删除 4 个旧 tool def，新增 3 个统一 tool def |
| `tool_dispatch.py` | 删除 4 个旧 dispatch，新增 3 个分支 |
| `command.py` | 删除 `cmd_workflow_list`/`cmd_workflow_get`/`cmd_workflow_report`，新增 `cmd_dg_tft_run`/`cmd_dg_tft_status`/`cmd_dg_tft_report`/`cmd_dg_tft_list` |
| `orchestrator.py` / `data_preparer.py` | tool handler 适配新入口 |

#### TODO

- [ ] **1.5.1** `constants.py`：重命名 4 个 tool 常量
- [ ] **1.5.2** `tool_def.py` + `tool_dispatch.py`：替换 tool 定义和 dispatch
- [ ] **1.5.3** `command.py`：重写 CLI command 函数，对齐 `dg_tft_run`/`status`/`report`/`list`
- [ ] **1.5.4** 删除旧的 `cmd_workflow_*` 函数和注册

### 1.6 用户偏好传递

`preferences` 由主 Agent（SemiForge）根据用户请求中的侧重描述生成，作为 `dg_tft_run` 的参数传入。如果用户请求中没有偏好倾向，传入空字符串，全链路 Agent 按默认均衡策略运作。

`dg_tft_run` 的 `preferences` 参数影响全链路决策。不是硬编码规则，而是作为自然语言注入各 Agent 的 system prompt，由 Agent 自行权衡。

#### 注入点

```
dg_tft_run(preferences="hysteresis 精度优先, IV 可宽松")
         │
         ├→ Agent A～D system prompt 注入
         │   "用户偏好：hysteresis accuracy first, IV can be loose.
         │    如果 IV 相关的质量不达标反复出现，适当调低其优先级，
         │    聚焦用户关心的 hysteresis 指标。"
         │
         ├→ Agent E system prompt 注入
         │   "用户要求迟滞精度优先"
         │   → 多指标加权时 hysteresis 权重最高
         │
         └→ quality gate 阈值：默认不动
             偏好不修改阈值——阈值是程序化硬闸门（快，不跑 LLM）
             所有超标指标仍会触发 Agent，Agent 自行基于偏好决定优先级
```

#### 偏好如何影响 Agent

- **决策 Agent (B/C/D)**：system prompt 中的偏好使 Agent 在 `adjust_params` 决策时倾斜——hysteresis 相关参数激进，IV 相关参数如果已经接近阈值则快速收敛。Agent 可以明确拒绝继续为低优先级指标重试（"IV rms 0.002 已接近阈值 0.001，用户标记为 low priority，建议接受"）
- **回退 Agent (A)**：仅负责 `fallback_to_previous`，收到偏好后据此判断是否值得回退（如果上游问题影响的是用户不在意的指标，可能选择继续推进）
- **回退判断**：Agent 决定是否 `fallback` 时会参考偏好——如果当前阶段指标不达标但对用户偏好目标影响小，可能选择继续推进而非回退
- **解读 Agent (E)**：多指标加权分析时 hysteresis 权重最高。`recommended_path` 优先选择 hysteresis 表现最好的路径，即使 IV/CV 不是全局最优
- **quality gate**：阈值 = 硬闸门（程序化，不跑 Agent），偏好 = Agent 决策权重（Agent 被触发后）。两者不冲突：阈值决定 Agent 是否被唤醒，偏好决定被唤醒后怎么做

#### 三层兜底与冲突处理

```
程序化硬兜底 (_PARAMETER_BOUNDS + tool schema)  ← 不可破
        ↓
quality gate 阈值                             ← 不可破，决定是否唤醒 Agent
        ↓
专家知识 (knowledge/*.md)                      ← Agent 参考
        ↓
用户偏好 (preferences)                         ← Agent 倾斜方向
```

前两层是程序化的，不可绕过。后两层是文本化的，Agent 在 system prompt 中收到后自行权衡。

**冲突不在程序层，在 Agent 推理层**：Agent 同时收到知识和偏好，自行判断。`adjust_params` 的值永远在 tool schema 定义的合法范围内——schema 本身就杜绝了越界参数。Agent 的 `reasoning` 字段记录权衡过程，Agent E 可回溯。

#### 实现方式

Agent system prompt 模板新增 `{user_preferences}` 占位符，与 `{expert_knowledge}` 并列注入：

```
==== USER PREFERENCES ====
{user_preferences}
==== END USER PREFERENCES ====

==== EXPERT KNOWLEDGE (DO NOT OUTPUT THIS SECTION) ====
{expert_knowledge}
==== END EXPERT KNOWLEDGE ====
```

---

## 二、数据输入阶段：Preparer-Verifier 架构

### 2.1 两入口设计

| 入口 | 场景 | 流程 |
|------|------|------|
| 原始数据 | 用户提供 txt/csv/xlsx 等测量文件 | Preparer（整理）→ Verifier（验证）→ 进入 check-data |
| 标准数据集 | 用户已有 `*_train.xlsx` 格式的标准数据 | Verifier（验证）→ 进入 check-data |

Verifier 是整个 Pipeline **第一个必经关卡**，无论哪种入口都要过。

### 2.2 Preparer-Verifier 重试循环（原始数据入口）

```
┌─────────────────────────────────────────────────────────────┐
│                    数据准备主流程                             │
│                                                             │
│  ┌──────────┐     handoff      ┌──────────┐                │
│  │ Preparer │ ────────────────→ │ Verifier │                │
│  │ (Worker  │  · 观察到了什么    │ (Checker │                │
│  │  SubAgent│  · 做了什么        │  Agent)  │                │
│  │  实例)   │  · 可能的遗漏      │          │                │
│  └──────────┘                   └────┬─────┘                │
│       ↑                              │                      │
│       │         FAIL + reason        │                      │
│       └──────────────────────────────┘                      │
│                                                             │
│  re-spawn 每一轮，handoff 历史通过外部上下文保留              │
│  最多 N 轮（N 可配置，默认 3）                                │
└─────────────────────────────────────────────────────────────┘
```

#### Preparer Handoff 结构

Preparer 通过 `submit_handoff` tool（`stop_mode="schema"`，定义在 `subagent_tools.py`）输出结构化 handoff：

```json
{
  "observations": ["发现 3 个 IV CSV 文件", "CSV 列名有前导空格，已 strip"],
  "actions": ["处理了 IDVBG_VDS01V.csv → IDVBG_VDS0.1V_train.xlsx"],
  "potential_issues": ["IDVD 系列只有 CSV，可能缺少部分 VDS 扫描"],
  "output_manifest": { "iv_files": 12, "cv_files": 1, "it_files": 0 }
}
```

#### Verifier → Preparer 反馈

```json
{
  "overall_ok": false,
  "critical_issues": [
    {"file": "IDVTG_VDS5V_train.xlsx", "issue": "缺少 'VD_V' 列"}
  ],
  "suggestion": "检查原始列名，可能存在命名不一致（'VD (V)' vs 'VD_V'）",
  "worker_notes": "上一轮处理了 12 个文件，IDVD 系列的 VDS 值可能用错了"
}
```

#### 上下文保留

- **每轮 re-spawn**：Preparer 和 Verifier 创建**全新实例**（清空 context window）
- **持久化**：所有 handoff 和 Verifier 反馈写入 `operation_log.json`（不另建 `prep_handoff_history.json`）
- **注入方式**：Preparer 重启时 orchestrator 从 `operation_log.json` 读历史 handoff + Verifier 反馈，注入 system prompt；Verifier 启动时注入本轮 Preparer 的 handoff（从 `agent.result` 拿）

### 2.3 标准数据集入口（仅 Verifier）

用户声明已准备好标准数据时，跳过 Preparer，Verifier 直接验证：

- 检查目录结构（`*_train.xlsx`、`CV_train.xlsx`、`it/`）
- 检查每个 IV 文件是否有 `long` sheet 和必需列
- **通过**：进入 check-data
- **失败**：fallback 到 Preparer 重新整理（保持程序逻辑一致），同时给用户一个 warning 提示标准数据可能不标准

### 2.4 Verifier 迁移到 SubAgent

当前 Verifier 是手动 LLM 循环（`_run_verifier` 自己构造 messages + tool loop + handler）。迁移到 SubAgent：

**subagent_tools**：
| Tool | 用途 |
|------|------|
| `inspect_xlsx` | 读取 XLSX 文件结构：输入 `file_path`，返回 `{sheet_name: [columns]}` |
| `report_verification` | 提交结构化验证结论（已有，`build_report_verification_tool()`） |

**SubAgent 配置**：

**执行流程**：
```
Verifier SubAgent:
  1. glob_file → 列出 output_dir 下所有文件
  2. inspect_xlsx → 逐个检查 IV/CV XLSX 的 sheet 和列
  3. report_verification → DONE (stop_mode="schema", stop_schemas=["report_verification"])
```

**实现位置**：
- `inspect_xlsx` tool schema 定义：`data_preparer.py`（新建 `_build_inspect_xlsx_tool()`）
- `inspect_xlsx` handler：复用现有 `_execute_verifier_tool` 中 `read_file` 的 `openpyxl` 逻辑
- 不进 `tool_def.py` / `tool_dispatch.py` / `SUPPORTED_TYPES` / MCP Router — 走 `subagent_tools` 本地拦截
- 旧函数 `_build_verifier_tools()`、`_execute_verifier_tool()`、`_run_verifier()` 的手动循环：全部删除

### 2.5 TODO

- [x] **2.5.1** 定义 handoff 数据结构（`PrepHandoff` / `VerifierFeedback`）
- [x] **2.5.2** 操作记录读写：handoff + Verifier 反馈写入 `operation_log.json`，orchestrator 读取注入 prompt
- [x] **2.5.3** 改造 `_run_worker()`：注入历史 handoff + feedback 到 prompt
- [x] **2.5.4** Verifier 迁移到 SubAgent：用 `explorer` type + `stop_mode="schema"` + `stop_schemas=["report_verification"]` 替代手动 LLM 循环；删除 `_build_verifier_tools()`、`_execute_verifier_tool()`、`_run_verifier()` 旧代码
- [x] **💬 2.5.X** ~~实现时讨论：Verifier 用 `explorer` type 会暴露 `bash`/`web_search` 等不需要的能力，是否新开独立的 `verifier` SubAgent type？~~ → **决策：使用 `explorer` type，通过 tool registration 控制可用工具，不新增 SubAgent type**
- [x] **2.5.5** 实现 `inspect_xlsx` tool 到 `subagent_tools.py`（schema + openpyxl handler）
- [x] **2.5.6** 实现重试循环主函数 `run_data_preparation_with_retry()`
- [x] **2.5.7** 实现标准数据集入口（skip Preparer，Verifier 直接验证；失败 → fallback 到 Preparer + warning）
- [x] **2.5.8** 配置项 `DG_TFT_PREP_MAX_RETRIES` 添加到 `agent_configs.json`
- [x] **2.5.9** 数据准备产出目录迁移：`prepared_data/<uuid>/1/` → `session/<uuid>/runs/run_NNNN/data_prep/attempt_N/`（由零改一完成后自然实现）
- [x] **2.5.10** 单元测试：handoff 序列化/反序列化
- [x] **2.5.11** 集成测试：模拟多次失败后成功

### 2.6 涉及文件

| 文件 | 改动类型 |
|------|----------|
| `data_preparer.py` | **重度改造** — worker/verifier re-spawn + handoff 注入 + 重试循环 + 标准数据入口；Verifier 从手动 LLM 迁移到 SubAgent；产出目录迁移 |
| `schema.py` | **新增** — `PrepHandoff`, `VerifierFeedback` dataclass |
| `agent_configs.json` | **新增** — `DG_TFT_PREP_MAX_RETRIES` |

---

## 三、建模阶段：嵌入专家知识的 Agent

### 3.1 角色定义

| Agent | 对应阶段 | 触发条件 | 可操作 INI section | 可回退到 |
|-------|----------|----------|--------------------|----------|
| A | check-data | 脚本硬失败 | 无（本阶段无调参） | Verifier |
| B | fit-physical | 硬失败 或 质量不达标 | `[physical_fit]` | check-data (Agent A) |
| C | train-static | 硬失败 或 质量不达标 | `[full_model]` | fit-physical (Agent B) |
| D | train-hysteresis | 硬失败 或 质量不达标 | `[hysteresis]` | train-static (Agent C) |
| E | 最终解读 | 所有阶段完成（成功或失败） | 无 | 不参与回退 |

### 3.2 INI 参数暴露与安全兜底

> 以下 adjust_params 示例中的具体参数值（如 `max_nfev_scale=3.0`）**仅为示意**。实际参数暴露需要专门的工程阶段来完成。

#### 目标

将各建模阶段的可调超参数**尽可能多地暴露在 INI 中**，配以两层安全兜底：程序化硬兜底 + Agent 软兜底。

#### 两层兜底

```
Agent adjust_params 输出
      │
      ▼
┌─ 程序化硬兜底（fast，不跑 LLM）──┐
│ · 参数白名单：是否在允许列表中？    │
│ · 安全边界：值是否在 [lo, hi]？    │
│ · 类型校验：int / float / enum？  │
│ · 阶段隔离：是否属于当前 INI section？│
└────────────────────────────────┘
      │ 通过
      ▼
┌─ Agent 软兜底（LLM，调参时附带）──┐
│ · 合理性检查：epochs=50000 合理吗？ │
│ · 交互检查：lr=0.01 & cv_weight=1？ │
│ Agent 本身在输出 adjust_params 前   │
│ 结合专家知识做合理性自检              │
└────────────────────────────────┘
      │
      ▼
   写入 INI → 跑 engine
```

#### 参数暴露方式

每个阶段维护一张参数定义表（Python 数据结构 → 自动生成 tool JSON schema），而不是手写 `adjust_params` tool def。生成的 schema 放入 `subagent_tools.py`

```python
# subagent_tools.py 示例 — fit-physical 阶段参数定义
PHYSICAL_FIT_PARAMS = {
    "max_nfev_scale": {"type": float, "lo": 0.5, "hi": 5.0, "default": 1.0, "desc": "NFEV scale factor"},
    "max_iv_points":  {"type": int,   "lo": 5000, "hi": 200000, "default": 50000, "desc": "IV subsample count"},
    "initial_guess_vto": {"type": float, "lo": -5.0, "hi": 5.0, "default": 0.0, "desc": "Initial VTO guess"},
    # ... 后续随 engine 演进持续扩充
}
```

`suggest_adjustments` / `apply_adjustments` 的静态策略表废弃。Agent 通过 tool schema 知道所有可调参数及其合法范围。

#### TODO

- [x] **3.2.1** 审计 `[physical_fit]` INI section，将所有可调参数定义到 `PHYSICAL_FIT_PARAMS` 表
- [x] **3.2.2** 审计 `[full_model]` INI section（含 KAN 超参），定义 `FULL_MODEL_PARAMS` 表
- [x] **3.2.3** 审计 `[hysteresis]` INI section，定义 `HYSTERESIS_PARAMS` 表
- [x] **3.2.4** 实现工具：`param_def_table → tool JSON schema` 自动生成函数（`params_to_tool_schema()` + `get_stage_param_schema()`）
- [x] **3.2.5** 程序化硬兜底：`validate_agent_adjustments()` — 白名单校验 + 安全边界 + 类型 + 阶段隔离（新增于 `quality_guard.py`，从 `param_defs` 读取参数表）
- [x] **3.2.6** Agent 软兜底：`get_soft_guard_prompt()` 返回自查指引，注入 Agent system prompt
- [x] **3.2.7** 后续维护：engine 新增超参 → 仅更新 `param_defs.py` 对应表，tool schema 和 hard guard 自动同步

### 3.3 Agent 运行模式

每个建模 Agent（A/B/C/D）的通用运行流程：

```
1. 触发（硬失败 或 质量不达标）
2. 加载阶段专家知识（knowledge/*.md）+ 用户偏好（preferences）
3. 分析当前状态（错误信息 / 不达标的多项指标 + 上一阶段 handoff）
4. 决策：
   a. 可本阶段修复 → 调用 adjust_params tool（结构化输出）→ 重试
   b. 怀疑上游问题 → 调用 fallback_to_previous tool（结构化输出）
   c. 无法判断 → 标记 exhausted，记录当前信息，orchestrator 继续推进后续阶段
5. 重试后达标 → 产出 handoff，传给下一阶段
6. 重试耗尽仍未达标 → 标记 exhausted，所有试次记录保留，传给解读 Agent E
```

**关键**：Agent 通过 tool function call 输出结构化 JSON，不输出自由文本。调参写入和脚本执行仍是固定代码。

**Warnings 字段**：`adjust_params` 和 `fallback_to_previous` 的输出中 `warnings` 为可选字段。仅当阈值和偏好持续冲突导致 Agent 被反复唤醒时发出（如"用户标记 IV 低优先级但阈值仍触发"），在**第 max_retries-1 次**调用时提醒一次。TUI 同步输出，operation_log 记录。

### 3.4 各 Agent 详解

#### 3.4.1 Agent A — check-data

**触发**：check-data 脚本硬失败（退出码非 0 / 产物缺失）

**工具**：
| Tool | 用途 |
|------|------|
| `fallback_to_previous` | 回退到上一阶段（Verifier），通知数据可能有问题 |

**不需要 `adjust_params`** — check-data 阶段没有可调的建模参数。

**Fallback 输出**：
```json
{
  "target": "verifier",
  "reason": "check-data 报告 'column VD_V not found'",
  "evidence": "Verifier 声称该文件有 long 表且列完整，但实际列名为 ['VD (V)', 'VBG (V)', ...]",
  "suggestion": "检查原始 CSV 列名，可能有单位括号未处理",
  "warnings": []
}
```

**回退链**：A 回退到 Verifier → Verifier 判断是否需要进一步回退到 Preparer。

- [ ] **3.4.1.1** 定义 Agent A system prompt（数据检查领域知识）
- [ ] **3.4.1.2** 定义 `fallback_to_previous` tool（用于 Agent A 回退到 Verifier）
- [ ] **3.4.1.3** 实现 Agent A 运行循环
- [ ] **💬 3.4.1.X** 实现时讨论：Agent A 用 `explorer` type 是否够？需 `read_file`/`glob_file` 分析错误日志，当前有 `web_search`/`web_fetch` 冗余但无害

#### 3.4.2 Agent B — fit-physical

**触发**：硬失败 或 物理拟合质量不达标（`iv_rms_log10_error_dec` 超阈值）

**工具**：
| Tool | 用途 |
|------|------|
| `adjust_params` | 调整 `[physical_fit]` 参数（仅此 section） |
| `switch_kernel` | 切换物理内核 |
| `fallback_to_previous` | 回退到 check-data (Agent A) |

> ⚠ 以下 `adjust_params` 中的参数值仅为示意。实际可调参数由 3.2 的 `PHYSICAL_FIT_PARAMS` 定义表驱动，tool schema 自动生成。

**`adjust_params` 输出**：
```json
{
  "section": "physical_fit",
  "adjustments": {
    "max_nfev_scale": 3.0
  },
  "reasoning": "IV RMS 2.3×10⁻² 远超阈值。先增大 max_nfev。",
  "warnings": [
    "用户偏好标记 IV 为低优先级，但 IV RMS 阈值将持续唤醒本 Agent。建议接受当前 IV 质量或放宽阈值"
  ]
}
```

`warnings` 字段可选，用于结构化提醒。触发条件：

| 场景 | 内容 |
|------|------|
| 指标反复超标但用户偏好标为低优先级 | 仅在**第 max_retries-1 次**发出（避免每轮重复），建议接受或调阈值 |
| 专家知识指示参数异常但用户偏好倾向更宽 | 在 reasoning 中记录，不发独立 warning |

**`fallback_to_previous` 输出**：
```json
{
  "target": "check-data",
  "reason": "提取的 VTO 为 -2.5V，预期 -0.5~0.5V，怀疑 IV 数据未做 mA→A 转换",
  "evidence": "低 VDS 区域 IV 误差 8%，高 VDS 区域仅 1.2%",
  "suggestion": "请 check-data Agent 验证 ID 列的数量级",
  "warnings": []
}
```

**`switch_kernel` 输出**（仅 Agent B）：
```json
{
  "kernel": "ekv_level2",
  "reasoning": "当前 UOTFT kernel 在亚阈值区产生数值振荡，换回 EKV Level 2 物理内核",
  "warnings": []
}
```

- [ ] **3.4.2.1** 定义 Agent B system prompt（物理建模专家知识）
- [ ] **3.4.2.2** 定义 `adjust_params` tool（限 `[physical_fit]`）
- [ ] **3.4.2.3** 定义 `switch_kernel` tool
- [ ] **3.4.2.4** 定义 `fallback_to_previous` tool（回退到 check-data）
- [ ] **3.4.2.5** 实现 Agent B 运行循环
- [ ] **3.4.2.6** 接入参数校验 pipe（→ Step 8 已建，此处为 Agent B 集成调用）
- [ ] **💬 3.4.2.X** 实现时讨论：Agent B 用 `explorer` type，需 `read_file` 读指标文件 + 物理拟合日志，现有工具集基本够用

#### 3.4.3 Agent C — train-static

**触发**：硬失败 或 静态模型质量不达标

**工具**：
| Tool | 用途 |
|------|------|
| `adjust_params` | 调整 `[full_model]` 参数（epochs, lr, cv_weight, KAN 超参等） |
| `fallback_to_previous` | 回退到 fit-physical (Agent B) |

> ⚠ 以下 `adjust_params` 中的参数值仅为示意。实际可调参数由 3.2 的 `FULL_MODEL_PARAMS` 定义表驱动。

**`adjust_params` 输出**：
```json
{
  "section": "full_model",
  "adjustments": {
    "epochs": 2000,
    "learning_rate": 0.0005,
    "cv_weight": 15.0
  },
  "reasoning": "loss 曲线在 500 epoch 后平台期，CV 误差远大于 IV，需提高 cv_weight",
  "warnings": []
}
```

- [ ] **3.4.3.1** 定义 Agent C system prompt（NN 建模专家知识）
- [ ] **3.4.3.2** 定义 `adjust_params` tool（限 `[full_model]`，含 KAN 超参）
- [ ] **3.4.3.3** 定义 `fallback_to_previous` tool（回退到 fit-physical）
- [ ] **3.4.3.4** 实现 Agent C 运行循环
- [ ] **💬 3.4.3.X** 实现时讨论：Agent C 用 `explorer` type，可能需要 `bash` 运行 quick Python 分析 loss 曲线，当前工具集够用

#### 3.4.4 Agent D — train-hysteresis

**触发**：硬失败 或 迟滞模型质量不达标

**工具**：
| Tool | 用途 |
|------|------|
| `adjust_params` | 调整 `[hysteresis]` 参数 |
| `fallback_to_previous` | 回退到 train-static (Agent C) |

> ⚠ adjust_params 参数为示意，实际由 3.2 的 `HYSTERESIS_PARAMS` 定义表驱动。

- [ ] **3.4.4.1** 定义 Agent D system prompt（迟滞建模专家知识）
- [ ] **3.4.4.2** 定义 `adjust_params` tool（限 `[hysteresis]`）
- [ ] **3.4.4.3** 定义 `fallback_to_previous` tool（回退到 train-static）
- [ ] **3.4.4.4** 实现 Agent D 运行循环
- [ ] **💬 3.4.4.X** 实现时讨论：Agent D 用 `explorer` type，同 Agent C

### 3.5 专家知识体系

> 专家知识的**来源**在后续提供（原始记录 → 提取 → 结构化），本节只定义知识的**落点**和**流向**。

#### 知识载体

```
src/workflow/dg_tft/knowledge/    # 预先编写的专家知识（随代码提交，loader.py 读取注入 Agent）
├── __init__.py
├── loader.py                     # 统一入口：load_knowledge(stage) -> str
├── physical_kernel.md            # 物理内核知识
├── static_model.md               # 静态模型训练知识
├── hysteresis.md                 # 迟滞模型知识
├── data_inspection.md            # 数据检查知识
└── final_interpret.md            # 最终解读知识

knowledge/_discovered/            # Agent 反哺的发现知识（项目根目录，公共可见，不按 session 隔离）
└── run_{run_id}.md               # Agent E 自动生成的单次运行事实性发现
```

> 两个目录物理隔离：前者是人工编写的领域专家知识（`loader.py` 加载后注入 Agent system prompt），后者是 Agent E 运行后自动反哺的事实性发现（`contribute_knowledge` tool 写入，见 4.4）。两者都是公共目录、不按 session 隔离。

#### 预编写知识（`src/workflow/dg_tft/knowledge/`）与反哺知识（`knowledge/_discovered/`）的注入方式不同

| 类型 | 来源 | 内容规模 | 注入方式 |
|------|------|----------|----------|
| 预编写知识 | 人工编写，随代码提交 | 小（每阶段一篇 .md） | **全量注入** system prompt（`EXPERT KNOWLEDGE` block） |
| 反哺发现知识 | Agent E 运行后自动生成 | 大（随运行次数增长） | **渐进式披露** — 摘要注入 prompt，详细报告按需 read_file |

**反哺知识的渐进式披露机制**：

```
Agent E 调用 contribute_knowledge tool
  ├→ knowledge/_discovered/run_{run_id}.md          # 详细报告（完整数据）
  └→ knowledge/_discovered/run_{run_id}_summary.md   # 简洁摘要（1-2 段：conditions + observation + key result）

新一轮 Agent 启动时：
  loader.load_discovered_summaries(stage) → 收集所有 _summary.md，按阶段归并
  → 注入 system prompt 的 EXPERT KNOWLEDGE block（与预编写知识并列）
  → Agent 如需查看某个 run 的完整细节 → 用 read_file 读取 run_{run_id}.md
```

**渐进式披露的理由**：
- 反哺知识随运行次数线性增长，全量注入会迅速撑爆 context window
- 摘要（每个 100-200 字）提供足够线索让 Agent 判断是否需要深入查阅
- 详细报告留磁盘，Agent 自行按需读取（与当前诊断 subagent 的 `read_file` 模式一致）

#### 知识流向（更新）

```
预编写：原始记录 → 提取 → knowledge/*.md → 全量注入 Agent system prompt

反哺：  Agent E 产出 → contribute_knowledge tool
         ├→ _summary.md → loader.load_discovered_summaries(stage) → 摘要注入 system prompt
         └→ .md (详细) → Agent 按需 read_file 查阅
```

Agent system prompt = 角色定义 + 任务说明 + 用户偏好（`USER PREFERENCES` block） + 专家知识（`EXPERT KNOWLEDGE` block，含预编写全量 + 反哺摘要，内部参考不可输出） + 上一阶段 handoff。

- [x] **3.5.1** 创建 `src/workflow/dg_tft/knowledge/` 目录
- [x] **3.5.2** 实现 `loader.py`
- [x] **3.5.3** 定义 system prompt 模板
- [x] **3.5.4** 建知识文件空壳
- [ ] **3.5.5** 后续：原始记录 → 提取 → 填充（→ Step 12: Agent E 知识反哺 4.4）

### 3.6 参数试次记录

**每次阶段执行都记录**，不论 Agent 是否触发。orchestrator 不选 best——全量存储，由 Agent E 结合用户偏好和专家知识自行判断最优路径。

```json
{
  "stage": "fit-physical",
  "attempts": [
    {
      "n": 1,
      "trigger": "initial",
      "agent_adjustments": null,
      "metrics": {"iv_rms_log10_error_dec": 0.0021},
      "ini_snapshot": "...",
      "duration_s": 67.0
    },
    {
      "n": 2,
      "trigger": "agent_adjust",
      "agent_adjustments": {"physical_fit.max_nfev_scale": 2.0},
      "agent_reasoning": "IV RMS 超标，怀疑收敛不足",
      "metrics": {"iv_rms_log10_error_dec": 0.0015},
      "ini_snapshot": "...",
      "duration_s": 89.0
    },
    {
      "n": 3,
      "trigger": "fallback_received",
      "fallback_from": "train-static",
      "fallback_reason": "train-static Agent C 认为 fit 结果有问题",
      "agent_adjustments": {"physical_fit.max_nfev_scale": 3.0},
      "agent_reasoning": "加大 nfev 确保收敛",
      "metrics": {"iv_rms_log10_error_dec": 0.0010},
      "ini_snapshot": "...",
      "duration_s": 95.0
    }
  ]
}
```

**下游阶段记录上游依赖**：

```json
{
  "stage": "train-static",
  "attempts": [
    {
      "n": 1,
      "trigger": "initial",
      "upstream": "fit-physical.2",
      "metrics": {"residual_iv_rms": 0.003, "residual_cv_rms": 0.015},
      "agent_adjustments": null
    },
    {
      "n": 2,
      "trigger": "agent_adjust",
      "upstream": "fit-physical.2",
      "metrics": {"residual_iv_rms": 0.002, "residual_cv_rms": 0.012},
      "agent_adjustments": {"full_model.learning_rate": 0.0005}
    }
  ]
}
```

Agent E 解读时基于完整依赖树 + 用户偏好（如"hysteresis 优先"） + 专家知识，综合判断：
- 哪条路径在多指标权衡下最优
- 如果用户偏好迟滞精度，即使 IV rms 略高也可接受
- 诊断报告给出推荐路径及其理由

#### operation_log.json 完整结构

所有 handoff、指标、Agent 决策、fallback 都记录在此文件。Agent E 唯一数据源。

```jsonc
{
  "run_id": 42,
  "created_at": "2026-07-01 10:00:00",
  "preferences": "hysteresis 精度优先",
  "final_status": "done",
  "stages": {
    "data-prep": {
      "attempts": [
        {"n": 1, "trigger": "initial",
         "handoff": {"observations": [...], "actions": [...], "potential_issues": [...]},
         "duration_s": 45},
        {"n": 2, "trigger": "verifier_fail",
         "handoff": {...},
         "verifier_feedback": {"overall_ok": false, "critical_issues": [...]},
         "duration_s": 52}
      ]
    },
    "check-data": { "attempts": [...] },
    "fit-physical": {
      "attempts": [
        {"n": 1, "trigger": "initial", "upstream": "check-data.1",
         "handoff": {"status": "quality_warn", "metrics": {"iv_rms_log10_error_dec": 0.0021}},
         "ini_snapshot": "...", "duration_s": 67},
        {"n": 2, "trigger": "agent_adjust", "upstream": "check-data.1",
         "agent_decision": {"tool": "adjust_params", "adjustments": {...}, "reasoning": "..."},
         "handoff": {"status": "ok", "metrics": {"iv_rms_log10_error_dec": 0.0015}},
         "ini_snapshot": "...", "duration_s": 89},
        {"n": 3, "trigger": "fallback_received", "fallback_from": "train-static",
         "fallback_info": {"reason": "VTO 初值异常", "evidence": "低 VDS 误差 8%"},
         "agent_decision": {"tool": "adjust_params", ...},
         "metrics": {"iv_rms_log10_error_dec": 0.0010}, "ini_snapshot": "...", "duration_s": 95}
      ]
    },
    "train-static": {
      "attempts": [
        {"n": 1, "trigger": "initial", "upstream": "fit-physical.2", ...},
        {"n": 2, "trigger": "agent_adjust", "upstream": "fit-physical.2", ...},
        {"n": 3, "trigger": "fallback_sent",
         "fallback": {"target": "fit-physical", "reason": "VTO 初值异常", "evidence": "..."}}
      ]
    }
    // train-hysteresis, final-report ...
  },
  "artifacts": {"fitted_params.json": "...", "dg_tft_static_core.va": "..."}
}
```

#### Agent 视角隔离

Re-spawn / 新 Agent 启动时，orchestrator 从 `operation_log.json` 过滤后注入 prompt。每个 Agent 只看到：

| 注入内容 | 来源 | 说明 |
|----------|------|------|
| 自己阶段的历史 attempts | `operation_log` | 之前试过什么参数、得到什么指标 |
| 紧邻上游的最终 handoff | orchestrator 程序化生成（非 Agent 产出） | 一条，含上游 attempt 号、状态、指标、warnings |
| 如果是回退触发 | 下游 Agent 的 `fallback_to_previous` 结果 | reason + evidence + suggestion |

看不到：其他阶段内部重试历史、更上游的原始数据、下游其他阶段的详细信息。防止上下文过载。

#### 前向 Handoff（程序化，非 Agent）

阶段成功后，orchestrator 在 `run_stage()` 返回后自动生成，不进 Agent 不跑 LLM：

```jsonc
{
  "stage": "fit-physical",
  "attempt": 3,
  "upstream": "check-data.1",
  "status": "ok",                    // ok | exhausted | quality_warn
  "metrics": {"iv_rms_log10_error_dec": 0.0010},
  "warnings": [],
  "key_outputs": ["01_physical_kernel/fitted_params.json"]
}
```

用途：
- 下游 Agent 触发时注入 system prompt，提供上游上下文
- Agent E 解读时还原完整链路
- 成功时不消耗 LLM 调用

#### 产物目录结构

```
session/<uuid>/runs/run_NNNN/    （由零改一统一管理）
├── data_prep/
│   ├── attempt_1/
│   └── attempt_2/
├── check-data/
│   └── attempt_1/
├── 01_physical_kernel/
│   ├── attempt_1/
│   └── attempt_2/
├── 03_full_model/
│   └── attempt_1/   ← 基于 fit-physical.2
├── 04_hysteresis/
│   └── attempt_1/
├── operation_log.json
├── RUN_SUMMARY.md
└── ARTIFACT_MANIFEST.json
```

**规则**：
- `attempt_N/` 目录**不删除**，Agent E 解读需要完整历史
- 旧 `_retry_N` 命名和 `_cleanup_retry_dirs()` 逻辑废弃
- `prepared_data/{prep_id}/` 废弃，改用 `session/<uuid>/runs/run_NNNN/data_prep/`（零改一完成后自然切换）
- 建模脚本的 `paths.output_dir` 指向当前 `attempt_N/`，脚本正常写入

- [x] **3.6.1** 定义 `StageAttemptLog` dataclass（含 `n`, `trigger`[8 种：initial/agent_adjust/quality_fail/fallback_sent/fallback_received/upstream_changed/verifier_fail/prep_retry], `upstream`, `handoff`, `agent_decision`, `ini_snapshot`）
- [x] **3.6.2** 试次记录写入 `operation_log.json`，每 attempt 追加
- [x] **3.6.3** `run_id` 统一：废除独立 `prep_id`，数据准备产物归入 `session/<uuid>/runs/run_NNNN/data_prep/`
- [x] **3.6.4** `attempt_N/` 目录创建逻辑，建模脚本 INI 指向当前 attempt 目录
- [x] **3.6.5** 清理废弃的 `_retry_N` 命名和 `_cleanup_retry_dirs()` 代码

### 3.7 回退链实现：编排器状态机

#### 问题

当前 `run_workflow()` 是线性 `for sd in stages:` 循环（`orchestrator.py:164`），不支持回退。一旦阶段失败就 `return False` 终止。改造后需要支持：

- 当前阶段失败 → 触发 Agent → Agent 决定 `adjust_params`（重新跑当前阶段）或 `fallback`（回到上一阶段）
- 回到上一阶段 → 触发上一阶段 Agent → 决定继续回退还是修复当前阶段
- 可能发生多跳回退（D→C→B→A→Verifier→Preparer）

#### 状态机设计

用**阶段指针 + while 循环**替代 for 循环。阶段分两类：

| kind | 执行方式 | 例子 |
|------|----------|------|
| `"agent"` | `spawn_agent()` → `agent.run()` | Preparer, Verifier |
| `"script"` | `run_stage()` 子进程 | check-data, fit-physical, train-static, train-hysteresis |

```
pointer = 当前阶段在 stages 列表中的索引
fallback_ctx = None  # 下游 Agent 传上来的回退上下文

while pointer < len(stages):
    sd = stages[pointer]
    
    # ── 0. 检测上游是否变更 ──
    last_upstream = get_last_upstream(sd.name, operation_log)
    current_upstream = get_current_upstream(pointer - 1, operation_log)
    if last_upstream and last_upstream != current_upstream:
        trigger = "upstream_changed"
        # 用上次最优参数重跑脚本（Agent 阶段则 re-spawn）
    else:
        trigger = "initial" if no_previous_attempt else last_trigger
    
    # ── 1. 执行阶段 ──
    if sd.kind == "script":
        result = run_stage(sd, params, ...)
        metrics = parse_metrics(output_dir, sd.name)
    elif sd.kind == "agent":
        agent = spawn_agent(sd, fallback_ctx)
        agent_result = agent.run()
        # agent 阶段的"成功"由 stop_mode schema 判定
    
    # ── 2. 生成前向 handoff（程序化）──
    handoff = build_handoff(sd, result/metrics, attempt_n, upstream)
    operation_log.append(sd.name, {n, trigger, handoff, ...})
    
    # ── 3. 判断 ──
    if result.ok and quality_ok(metrics):
        pointer += 1; fallback_ctx = None; continue
    
    # ── 4. 失败/不达标 → 检查 retry 次数 ──
    if retries_exhausted:
        # 有指标的 attempt → 选 primary_metric 最优的
        # 全是硬错误 → 标记 failed，级联 skip 下游
        if has_any_metric:
            best = pick_best_by_primary_metric(attempts)
            best["handoff"]["status"] = "exhausted"
            pointer += 1; continue
        else:
            mark_failed_and_skip_dependents(sd, stages); break
    
    # ── 5. 触发 Agent ──
    agent = spawn_agent(sd, result, metrics, fallback_ctx)
    agent_result = agent.run()  # stop_mode="schema"
    
    # ── 6. 解析 Agent 决策 ──
    if agent_result.tool == "adjust_params":
        # 记录 attempt
        apply_params(agent_result.adjustments); continue
    
    if agent_result.tool == "fallback_to_previous":
        if pointer == 0: mark_failed(sd); break
        pointer -= 1
        fallback_ctx = agent_result  # {reason, evidence, suggestion}
        continue  # 回到步骤 1，不先跑脚本，直接触发上游 Agent
    
    # switch_kernel（仅 Agent B）：应用内核切换，continue 同 adjust_params
```

#### 关键边界

| 场景 | 行为 |
|------|------|
| Agent 调用 `adjust_params` | 记录 attempt，应用参数，重跑当前阶段 |
| retry 次数耗尽且有指标 | 选 primary_metric 最优 attempt，标记 `status="exhausted"`，推进下游 |
| retry 次数耗尽且全硬错误 | 标记当前阶段 failed，级联 skip 所有 `depends_on` 下游阶段 |
| 下游收到上游 `status="exhausted"` | Agent system prompt 告知"上游质量不达标但已尽力"，Agent 自行决定是否放宽要求 |
| 阶段重入时上游已变更 | `trigger="upstream_changed"`，用上次最优参数跑脚本（不先跑 Agent） |
| Agent 调用 `fallback` 且 `pointer == 0` | 触发最终失败，进入 Agent E |
| 回退后上一阶段 Agent 也 `fallback` | 指针再次 -1，级联继续 |
| 回退后上一阶段 `adjust_params` → 重跑 → 成功 | 指针 +1，下游阶段全部因上游变更被重跑 |
| 超时 | 任意时刻 `elapsed > timeout` → 直接终止 |

#### TUI 同步

- `pointer` 移动时：当前阶段 task 改为 `in_progress`，离开的阶段 `completed`
- 回退时：被回退的阶段 task 从 `completed` 改回 `in_progress`
- Agent 运行时：Console 流式输出 Agent 分析（但不阻塞脚本流式输出）

#### TODO

- [ ] **3.7.1** 用 while + pointer 替代 for 循环
- [ ] **3.7.2** `StageDef` 增加 `kind: str`（`"agent"` / `"script"`），`stages.py` 头部增加 Preparer + Verifier StageDef
- [ ] **3.7.3** Agent 决策路由：`adjust_params` / `fallback_to_previous` / `switch_kernel`
- [ ] **3.7.4** exhaustion 分支：选 primary_metric 最优 → 标记 `exhausted` → 推进；全硬错误 → 级联 skip
- [ ] **3.7.5** `upstream_changed` 检测：进入阶段时比较上游 attempt 号，变更则用上次最优参数重跑
- [ ] **3.7.6** 前向 handoff 生成：`run_stage()` 成功后程序化产出
- [ ] **3.7.7** `depends_on` 级联 skip：全硬错误时 skip 下游
- [ ] **3.7.8** 回退上下文传递（`fallback_ctx` 携带 reason + evidence + suggestion）
- [ ] **3.7.9** TUI task board 随 pointer 移动同步
- [ ] **3.7.10** Console 输出：回退原因、Agent 决策、重试计数、exhaustion 状态
- [ ] **3.7.11** 集成测试：模拟 D→C→B 三级回退
- [ ] **3.7.12** 集成测试：模拟 exhaustion → 级联 skip

---

### 3.8 涉及文件

| 文件 | 改动类型 |
|------|----------|
| `quality_guard.py` | **更名/重构** — 保留安全边界和阈值逻辑；静态调参策略表废弃，替换为 Agent tool；扩展 `_PARAMETER_BOUNDS` 和 `_PARAMETER_WHITELIST` 覆盖所有 INI 参数 |
| `subagent_tools.py` | **新建/扩展** — 各阶段 tool schema（`adjust_params`、`fallback_to_previous`、`switch_kernel`）+ `inspect_xlsx` handler |
| `orchestrator.py` | **重度改造** — Agent 触发、运行循环、回退链、TUI 同步、试次全量记录；删除 `_cleanup_retry_dirs()` 调用，`_retry_N` → `attempt_N/` 且不删除 |
| `stages.py` | **扩展** — StageDef 增加 agent 相关字段 |
| `data_preparer.py` | **扩展** — Verifier 上移为 Pipeline 节点，支持来自 check-data 的 fallback |
| `agent_configs.json` | **新增** — 各阶段 Agent 配置 |
| `subagent_hooks.py` | **扩展** — 建模阶段 Agent 触发 hook |
| `knowledge/` | **新增目录** — 专家知识文件 |

---

## 四、结果阶段：LLM 解读

### 4.1 定位

> Agent E **只解读，不操作工作流**。输入详细（全量操作记录 + 用户偏好 + 多指标数据），输出简洁（诊断报告含推荐路径和加权分析）。

### 4.2 输入

- 完整操作记录（`operation_log.json`，含所有阶段所有试次的依赖树和全量指标）
- 用户偏好（`preferences`，如"hysteresis 优先"）
- 产物清单（Verilog-A 文件、pth 文件等）
- 最终 INI 参数快照

### 4.3 输出

Agent E 基于完整操作记录 + 用户偏好 + 专家知识，产出多指标加权的诊断报告：

```json
{
  "user_preferences": "hysteresis 精度优先, IV 可宽松",
  "recommended_path": {
    "chain": "fit-physical.2 → train-static.2 → train-hysteresis.1",
    "rationale": "fit.2 提供了足够的物理基线（IV rms 0.0015），train-static.2 平衡了 IV/CV，hysteresis 在此基础上表现最优"
  },
  "modeling_status": {
    "overall": "success",
    "quality_grade": "A",
    "summary": "结合用户偏好（迟滞优先），迟滞模型达到 rms 0.005 优秀水平。IV rms 0.002 略高于阈值但用户接受。"
  },
  "metric_breakdown": [
    {"stage": "fit-physical", "attempt": 2, "iv_rms_log10_error_dec": 0.0015, "weight": 0.2},
    {"stage": "train-static", "attempt": 2, "residual_iv_rms": 0.002, "weight": 0.1},
    {"stage": "train-static", "attempt": 2, "residual_cv_rms": 0.012, "weight": 0.3},
    {"stage": "train-hysteresis", "attempt": 1, "step_log_rmse_median": 0.005, "weight": 0.4}
  ],
  "failure_diagnosis": {
    "root_cause": "fit-physical.1 VTO 初值不当（-2.5V），Agent B 调大 max_nfev 后修复",
    "confidence": 0.85
  },
  "input_output_interpretation": {
    "inputs": { "data_source": "IDVBG/IDVTG/IDVD family 61404 行 IV 数据" },
    "outputs": {
      "verilog_a_static": "dg_tft_static_core.va",
      "verilog_a_dynamic": "dg_tft_dynamic_srh.va"
    },
    "trustworthiness": "高 — 基于用户偏好，迟滞模型精度优秀。全 VDS 范围误差 < 1%。"
  }
}
```

#### 全链路失败时的输出

> ⚠ 失败态的 schema 需要后续进一步详细讨论，此处为初步定义。

Pipeline 无任何 usable 产物时，Agent E 基于全量错误记录定位 root cause：

```jsonc
{
  "user_preferences": "hysteresis 精度优先",
  "modeling_status": {
    "overall": "failed",
    "summary": "数据准备失败，未进入建模阶段。Verifier 连续 3 次驳回 Preparer 的输出。"
  },
  "root_cause": {
    "stage": "data-prep",
    "issue": "Preparer 未正确处理 IDVTG 系列的 column mapping（'VD (V)' → 'VD_V'）",
    "evidence": "Verifier feedback run_42: 'IDVTG_VDS5V_train.xlsx missing VD_V column'",
    "first_failure_at": "data-prep.attempt_1"
  },
  "cascade_failures": [
    {"stage": "check-data", "status": "skipped", "reason": "upstream data-prep failed"},
    {"stage": "fit-physical", "status": "skipped", "reason": "dependent on check-data"}
  ],
  "input_output_interpretation": {
    "inputs": { "data_source": "IDVBG/IDVTG/IDVD raw CSV/XLSX files" },
    "outputs": {},
    "trustworthiness": "N/A — 未生成任何模型"
  },
  "next_steps": [
    "检查原始数据中 IDVTG 系列文件的列名格式",
    "考虑手动调整 column mapping 后重试",
    "如为标准数据集入口，检查数据是否符合 *_train.xlsx 格式要求"
  ]
}
```

根因分析原则：在所有失败阶段中找到**最上游的首次失败**，下游标记为 `skipped (cascade)`，避免误导用户认为下游也有独立问题。

### 4.4 知识反哺

> ⚠ 此功能依赖专家知识体系（3.5）已搭建且主知识文件有一定内容积累后才有意义。格式需要专门讨论优化后再落定，当前文档仅定义方向。

Agent E 产出诊断后，调用 `contribute_knowledge` tool（`stop_schemas` 中可选），将本次运行的**事实性发现**写入公共知识库。

#### 产出位置（双文件）

```
knowledge/_discovered/run_{run_id}.md         ← 详细报告（完整数据，供 Agent 按需 read_file 查阅）
knowledge/_discovered/run_{run_id}_summary.md ← 简洁摘要（1-2 段，注入新一轮 Agent 的 system prompt）
```

后续人工或知识管理员 Agent 审查后，有价值的提升到 `knowledge/physical_kernel.md` 等主知识文件。
> 注：此处 `knowledge/` 为项目根下的公共知识目录，与 `src/workflow/dg_tft/knowledge/`（预先编写的专家知识、随代码提交）物理隔离——前者是 Agent 运行中自动反哺的事实性发现，后者是人工编写的领域指导。

#### 格式要求

**报告性，不推测**。只描述观察到的现象、条件和结果。不写"可能是因为"。

```markdown
## Run #42 | 2026-07-03

### 数据特征
- IV family: 12 个文件 (IDVBG/IDVTG/IDVD), 61404 行
- VDS 范围: 0.1V ~ 5V
- CV: 有 (4 sheets: cgg/cbg/ctg/vds)
- I-t: 无

### fit-physical — max_nfev_scale 与收敛
在数据量 61404 行的条件下：
- max_nfev_scale=1.0 → IV RMS 0.0021（不达标）
- max_nfev_scale=2.0 → IV RMS 0.0015（不达标）
- max_nfev_scale=3.0 → IV RMS 0.0010（达标）

```
### 条件
数据量 ≥ 60000 行 / VDS range ≥ 5V

### 观察
max_nfev_scale ≥ 3.0 时物理拟合达标
```

### train-static — cv_weight 与 IV/CV 平衡
在 fit-physical.3 的基础上：
- cv_weight=1.0 → IV rms 0.003, CV rms 0.015
- cv_weight=15.0 → IV rms 0.002, CV rms 0.012

```
### 条件
上游 fit-physical IV rms ≈ 0.001

### 观察
cv_weight=15.0 时 IV/CV 综合指标优于 cv_weight=1.0

`contribute_knowledge` tool 输出示例：
```json
{
  "summary": "在数据量 ≥ 60000 行且 VDS ≥ 5V 条件下，max_nfev_scale 需 ≥ 3.0 才能使物理拟合 IV RMS 达标。cv_weight=15.0 时 IV/CV 综合指标优于 1.0。",
  "findings": [
    {
      "stage": "fit-physical",
      "observation": "max_nfev_scale=3.0 时 IV RMS 达标 (0.0010)，1.0 和 2.0 不达标",
      "conditions": "数据量 ≥ 60000 行, VDS ≥ 5V",
      "evidence": "run_42, fit-physical attempts 1-3"
    }
  ]
}
```

`summary` 字段（100-200 字）：注入新一轮 Agent system prompt 的 `EXPERT KNOWLEDGE` block。
`findings` 字段：写入 `run_{run_id}.md` 详细报告，含每个发现的条件、观察和证据。

### 4.5 TODO

- [ ] **4.5.1** 创建 `knowledge/_discovered/` 公共目录（项目根目录，不按 session 隔离）
- [ ] **4.5.2** 定义 `contribute_knowledge` tool，写入 `knowledge/_discovered/run_{run_id}.md`
- [ ] **4.5.3** Agent E 的 `stop_schemas` 包含 `contribute_knowledge`（可选，有发现时调用）
- [ ] **4.5.4** 定义 `OperationRecord` dataclass（`schema.py`）
- [ ] **4.5.5** 操作记录持久化（`operation_log.json`）
- [ ] **4.5.6** 设计 Agent E 的 tool 集（读取操作记录、指标、产物）
- [ ] **4.5.7** 定义 Agent E system prompt（含 `{user_preferences}` + 多指标加权策略 + 知识反哺指导）
- [ ] **💬 4.5.X** 实现时讨论：Agent E 用 `explorer` type，`stop_mode="any"`
- [ ] **4.5.8** 在流程末尾调用 Agent E，结果追加到 RUN_SUMMARY.md
- [ ] **4.5.9** Console 输出解读摘要

### 4.6 涉及文件

| 文件 | 改动类型 |
|------|----------|
| `schema.py` | **新增** — `OperationRecord`, `InterpretationResult` |
| `orchestrator.py` | **扩展** — 操作记录收集 + Agent E 调用 |
| `report_builder.py` | **扩展** — 解读结果嵌入 MD report |

---

## 五、开发顺序

全部串行，每一步依赖前一步完成。
  0. ✅ 零改一 (0.0.1-0.0.12) — 产出路径统一到 session/<uuid>/
  1. ✅ 零改二 (0.1.1-0.1.10) — SubAgent 增强：subagent_tools + tool_handlers + stop_mode + stop_schemas

  2. ✅ 专家知识体系 (3.5)
      目录 + loader + 空壳文件 + system prompt 模板（含 {user_preferences} 和 {expert_knowledge} 占位符）
      → 依赖：1

   3. ✅ 参数试次记录 (3.6)
      StageAttemptLog 数据结构 + operation_log 写入 + attempt_N 目录 + run_id 统一
      → 依赖：1

   4. ✅ 数据输入 (2.5.1-2.5.11)
      Preparer-Verifier 重试循环 + 标准数据入口（失败→fallback到Preparer） + Verifier 迁移到 SubAgent + 目录迁移
      → 依赖：1
      → 可交付节点

    5. ✅ INI 参数审计与暴露 (3.2)
      审计所有阶段 INI section → 定义参数表 → 自动生成 tool schema → 两层兜底
      → 依赖：4（需确认 engine 的 INI 结构稳定）

   6. ⚠ Tool 统一 (1.5)
      将现有的 4 个散 tool（prepare_dg_tft_dataset、start_dg_tft_workflow、
      query_dg_tft_workflow、read_dg_tft_workflow_report）收敛为 3 个统一接口：
      dg_tft_run / dg_tft_status / dg_tft_report。
      删除旧命令行 cmd_workflow_*，新增 cmd_dg_tft_*。
      更新 constants.py / tool_def.py / tool_dispatch.py / command.py。
      → 依赖：4, 5（workflow + 参数审计均稳定）
      → 重要：提前执行此步，确保后续开发始终面向统一的 3-tool 接口。

   7. ⚠ while+pointer + 数据纳入 Pipeline + Agent A (3.7 + 3.4.1)
      结构性重构 + 第一个 Agent 一次性完成：
      - StageDef 增加 `kind` 字段（`"agent"` / `"script"`）
      - Preparer + Verifier StageDef 补充到 stages 列表头部
      - while+pointer 替代 for 循环（无 fallback 也先替换，为后续铺路）
      - Agent A check-data 实现：仅 `fallback_to_previous`
      - 回退链首次走通：A → Verifier → Preparer
      - TUI 同步（进度条、回退指示）
      → 依赖：2, 3, 4, 5, 6
      → 重要：与此步同时修复 `subagent_tools.py` 中 check-data 注册使用
        `fallback_to_previous` 而非旧 `fallback_to_verifier`；
        移除 `SUBAGENT_TOOL_FALLBACK_TO_VERIFIER` 常量。

   8. Agent 基础设施
      adjust_params / fallback_to_previous / switch_kernel tool defs + handler
      参数校验 pipe（Agent 输出 → 安全边界 → 合理性检查 → INI 写入）
      通用 Agent 运行循环（spawn → run → 解析 tool 结果）
      → 依赖：1, 5, 6, 7
      → while+pointer 已就位，Agent 基础设施可直接对接

   9. Agent B — fit-physical (3.4.2)
      adjust_params + switch_kernel + fallback_to_previous (→ Agent A)
      → 依赖：8

  10. Agent C — train-static (3.4.3)
      adjust_params + fallback_to_previous (→ Agent B)
      → 依赖：9

  11. Agent D — train-hysteresis (3.4.4)
      adjust_params + fallback_to_previous (→ Agent C)
      → 依赖：10

  12. Agent E — 解读 (4.1-4.4)
      stop_mode="any"，纯文本输出
      读取 operation_log + 用户偏好 → 多指标加权分析 → 推荐最优路径
      → 依赖：操作记录完整
      └── 知识反哺 (4.4) 为 Agent E 的可选后续阶段

---

## 六、配置项汇总

```json
{
  // 零改一 — 路径统一
  "DG_TFT_RUNS_SUBDIR": "runs",            // 运行子目录名，相对于 session/<uuid>/
  // 以下旧 key 删除：
  //   "DG_TFT_DEFAULT_OUTPUT_ROOT": "./outputs/dg_tft_runs"  (由零改一删除)

  // 零改二 — SubAgent 增强
  "DG_TFT_PREP_MAX_RETRIES": 3,
  "DG_TFT_PREP_HANDOFF_HISTORY_SIZE": 5,

  // Phase 2
  "DG_TFT_MODELING_AGENT_MODEL_TIER": "medium",
  "DG_TFT_MODELING_AGENT_MAX_STEPS": 20,
  "DG_TFT_MODELING_AGENT_TIMEOUT_S": 600,
  "DG_TFT_FALLBACK_CHAIN_MAX": 6,
  "DG_TFT_KNOWLEDGE_DIR": "src/workflow/dg_tft/knowledge",    // 预先编写的专家知识目录（loader.py 读取，注入 Agent）
  "DG_TFT_DISCOVERED_KNOWLEDGE_DIR": "knowledge/_discovered",  // Agent 反哺发现知识目录（项目根，公共，不按 session 隔离）
  "DG_TFT_QUALITY_PHYSICAL_IV_RMS_MAX": 0.001,
  "DG_TFT_QUALITY_STATIC_IV_RMS_MAX": 0.005,
  "DG_TFT_QUALITY_STATIC_CV_RMS_MAX": 0.01,
  "DG_TFT_QUALITY_HYSTERESIS_RMS_MAX": 0.01,
  "DG_TFT_QUALITY_MAX_RETRIES": 3,

  // Phase 3
  "DG_TFT_INTERPRET_AGENT_MODEL_TIER": "medium",
  "DG_TFT_INTERPRET_AGENT_MAX_STEPS": 10
}
```

> ⚠ 品质阈值为示意值，实际以合理建模指标确定。当前代码中的值见 `config/agent_configs.json`。
