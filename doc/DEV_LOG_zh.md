# SemiForge 开发日志

## 开发规范

以下规范适用于本项目的所有代码贡献，旨在保证工程一致性和可维护性。

### 1. 测试优先
- 每个新模块必须附带对应的测试文件（`*_test.py`），使用 `unittest` 框架。
- 运行全部测试：`python -m unittest discover -s test -p "*_test.py"`。
- 提交前确保全部测试通过。CI 引入前至少本地全量通过。

### 2. 文档同步更新
- 行为变更（新增工具、修改 prompt 路由、调整 agent_configs.json 键值含义、SubAgent 生命周期变化）必须在本文档中记录。
- `src/context/prompt.py` 中的路由指引、`config/agent_configs.json` 中的注释说明需与代码行为保持同步。

### 3. 注释与输出全英文
- 代码注释、docstring、`console.print()`、`sys_log` 一律使用英文。
- 文件头 docstring 中的作者/机构信息除外。
- 示例：`"Stage check-data failed: division by zero"` ✅，`"check-data阶段失败"` ❌。

### 4. 用户可见性与控制台风格
- 每个关键路径节点必须有 `console.print()` 输出，silent failure 不可接受。
- 颜色使用规范（使用 `src/constants.py` 中定义的常量）：
  - 错误：`WF_STATUS_ERROR`（bold red）
  - 警告：`WF_STATUS_WARN`（bold yellow）
  - 成功：`WF_STATUS_SUCCESS`（#8CDCA0）
  - 次要信息/详情：`bright_black`
  - **禁止滥用 `MAJOR_COLOR1` / `MAJOR_COLOR2`**，仅用于模块名、配置项名称等真正需要强调的关键标识，不整句使用。
- 使用 `Text.append()` 拼接富文本时，不同颜色的片段应分别 append 并指定 style，避免嵌套 Rich markup 被外层 style 覆盖。
- `sys_log` 保留完整审计日志；`console.print()` 面向实时用户可见进度。两者互补，不互相替代。

### 5. 配置驱动，禁止硬编码
- 路径、超时、阈值、开关等可调参数一律从 `config/agent_configs.json` 读取，不写死在代码中。
- 时间相关配置键统一使用 `_S` 后缀表示秒（如 `DG_TFT_TIMEOUT_S_CHECK_DATA`、`SUBAGENT_TIMEOUT_S`）。
- 字符串常量（工具名、标签名）统一定义在 `src/constants.py`（命名规范：`TOOL_NAME_*`、`*_LABEL`）。

### 6. 线程安全
- 任何跨线程共享的可变状态（如 `WorkflowManager._runs`、`PreparationTracker._preps`）必须加 `threading.Lock`。
- 所有 mutation 操作（create/update/mark_done/mark_failed/save/load）必须在锁内执行。

### 7. 权限体系
- 任何影响文件系统或执行外部命令的操作须通过 `pause_for_permission` / `ask_permission_tui` 请求用户授权。
- 工具权限映射使用 `PERMISSION_PRESETS` 常量，不手写权限列表。

### 8. 导入规范
- 包内模块互引用统一使用相对导入：`from .schema import WorkflowRun`。
- 外部调用方（`main.py`、`agent_context.py` 等）使用绝对导入：`from src.workflow.dg_tft.manager import WorkflowManager`。
- 禁止混用两种路径引用同一模块（会导致 Python 注册两个独立的 module 对象，`isinstance` 检查失败）。

### 9. Session 持久化
- 需跨会话恢复的数据结构（`WorkflowManager`、`Scoreboard` 等）必须实现 `save_to_file()` / `load_from_file()`。
- 持久化文件统一落在 `session/<session_uuid>/` 下，JSON 格式。
- `to_dict()` 和 `from_dict()` 不依赖 `dataclass.asdict`（其内部 `deepcopy` 可能导致序列化失败）。

### 10. SubAgent 生命周期
- 注册：将 `SubAgentProgress` 写入 `parent_ctx.agent_list`。
- 执行：`agent.run()` 同步阻塞。
- 注销：`finally` 块中从 `agent_list` 移除。
- 静默模式：通过 `getattr(ctx, "subagent_mute", False)` 检查，为 True 时禁止 console 输出和权限弹窗。
- SubAgent 创建使用 `clone_context()` 获取独立上下文副本。

### 11. 错误处理
- 所有异常必须同时 `sys_log.error(...)` + `console.print(..., style="bold red")`。
- 非关键路径（如诊断 subagent 失败）的异常不应向上传播，使用 try/except 兜底。

### 12. Agent 与固定代码的边界
- **Agent/LLM 介入范围**：数据格式理解与转换、产出物格式校验、失败原因归纳解读。
- **纯规则代码负责**：训练执行、配置生成、质量门禁阈值判定、参数调优、报告文件生成。
- 不要让 LLM/SubAgent 决定训练算法参数或直接操作训练代码 — "把开放性留给 Agent，把执行权留给固定代码"。

### 13. Git 提交规范
- Commit message 使用英文，简明祈使语气。
- 单次提交尽量聚焦一个主题（避免"改了一堆东西"式的提交）。
- 提交前检查：`git status` → `git diff` → 确认无调试代码、无 secrets、无意外修改。

---

## 开发时间线

### 2026-06-25 | Phase 0–2 — 基础设施 + 数据审计 + 配置生成

**Phase 0**：DG-TFT 建模代码迁移至 `src/device_modeling/dg_tft/`。锁定双解释器架构：Agent（`AGI`）与建模（`CUDA`）严格分离，通过 `agent_configs.json` 中的 `DG_TFT_MODELING_PYTHON_EXE` 显式指定。

**Phase 1**：工作流引擎骨架 `src/workflow/dg_tft/` — `schema.py`（WorkflowRun、StageResult、DatasetProfile）、`manager.py`（线程安全 CRUD + session 持久化）、`stage_runner.py`（统一解释器解析）。接入 `AgentContext` 和 `main.py`。

**Phase 2**：`dataset_inspector.py`（subprocess 调 CUDA `validate_dataset.py` → DatasetProfile，I-t 自动发现，温度/HSPICE 检测）+ `config_builder.py`（模板 .ini → per-run 运行时配置）。累计测试：103。

---

### 2026-06-25 | Phase 3 — Subprocess 版全自动流程

`stages.py`（StageDef dataclass，7 个阶段）+ `orchestrator.py`（`run_workflow()` 顺序执行、fail-fast、hysteresis 条件跳过、断点续传）。通过 `ctx.console` 输出终端进度。累计测试：135。

---

### 2026-06-25 | Phase 4 + 4.5 — Agent 工具接入 + 数据整理

**Phase 4**：三个 workflow 工具（`start_dg_tft_workflow`、`query_dg_tft_workflow`、`read_dg_tft_workflow_report`）+ dispatch + 内建命令。System prompt 从 TECoSim/simulator 重写为 DeviceModeling/DG-TFT。

**Phase 4.5**：`data_preparer.py` — 三步数据整理流程：
1. **Preparer subagent**（type=`preparer`）：将原始 CSV/XLSX 转换为 `*_train.xlsx`
2. **Verifier LLM**（fast model，mini tool-loop）：用结构化 `report_verification` schema 检查产出物
3. **`_check_output()`**：纯 Python CRITICAL/WARNING 判定（零 LLM）

同日 bugfix：worker subagent 使用 CUDA python 以引用 pandas，verifier 进度展示，已整理数据跳过，前台同步执行，train-static metadata 兼容。累计测试：234。

---

### 2026-06-25 | Phase 5 — 报告、产物与任务面板

`artifacts.py` + `report_builder.py`（`build_and_write_reports()` → ARTIFACT_MANIFEST.json、RUN_SUMMARY.json、RUN_SUMMARY.md）。Orchestrator 集成：成功/失败均自动生成报告。任务面板同步：阶段任务在 Scoreboard 上创建/完成。控制台摘要展示路径、指标和失败建议。累计测试：237。

---

### 2026-06-26 | Phase 6 — 失败诊断 + SubAgent 协作

`diagnostics.py`：6 大类关键字匹配错误分类（DATA_ERROR / CONFIG_ERROR / TRAINING_NUMERICAL_ERROR / MISSING_ARTIFACT_ERROR / HSPICE_ENV_ERROR / UNKNOWN_ERROR），含置信度评分。

`subagent_hooks.py`：6 个 explorer subagent hook 定义，含参数化 prompt 模板。`orchestrator.py` 集成：`_run_stage_diagnosis()` 自动分类 → 可选 spawn explorer subagent（由 `DG_TFT_ENABLE_DIAGNOSIS` 控制）。诊断结果存入 `run.extra["diagnosis"]`；report builder 包含 Diagnosis 段落。累计测试：314。

---

### 2026-06-26 | Phase 7 — 质量守卫 + 自动调参重试

`metrics_parser.py`：从产物 JSON 提取指标（fitted_params.json、residual metrics、hysteresis analysis）。

`quality_guard.py`：`assess_quality()` 阈值判定 → `suggest_adjustments()` 渐进式调参策略引擎 → `apply_adjustments()` 原地 INI 改写。参数白名单 + 边界保护。Retry 循环（最多 3 次，可配），`_retry_N/` 子目录隔离，保留最优 attempt 产物。

Orchestrator 集成：`_enforce_quality_gate()` 在每阶段成功后执行。Report builder："Quality" 章节含 retry log 表格。阈值基于实际 debug run 数据校准。累计测试：370。

---

### 2026-06-26 | Phase 8 — 脚本函数化重构

7 个 engine/dynamic 脚本均新增 `run_*()` 可 import 入口函数。`run_model.py` 重写：内层 `subprocess.run()` 消除，改为 `from engine.xxx import run_yyy; run_yyy()` 直接调用。

AGI→CUDA 外层 subprocess 保留（解释器隔离不变），仅去除 CUDA 内部的双层 subprocess。无功能开关 — 函数化为唯一路径，对 orchestrator 完全透明。累计测试：393（12 skip = CUDA 依赖不在 AGI 环境）。

---

### 2026-06-27 | Phase 9 — 测试、稳定性与收尾

**P9-T1**：补充单元测试（+26）：report quality gate 渲染、diagnostics（truncate_list、stdout-only 分类）、quality_guard（INI 读取错误、遗漏调整）。

**P9-T2**：Agent 集成测试（+64，2 个新文件）：`dg_tft_workflow_tool_test.py`（38 tests — 任务面板联动、序列化、dispatch、prompt 路由）+ `dg_tft_workflow_background_test.py`（26 tests — 并发 WorkflowManager、状态转换、DatasetProfile、持久化）。

**P9-T3**：三套测试数据集（`test_data/{static_only,full,corrupted}/`）+ 可复现生成脚本。

**P9-T5**：收尾清理 — 无调试代码，无硬编码路径，session 持久化链验证，`.gitignore` 更新。

---

### 2026-06-27 | 阶段超时 + 整体超时

超时值从硬编码迁移至 `agent_configs`（`DG_TFT_TIMEOUT_S_CHECK_DATA=600`、`DG_TFT_TIMEOUT_S_FIT_PHYSICAL=1800` 等）。命名使用 `_S` 后缀，与现有 `SUBAGENT_TIMEOUT_S` 保持一致。

`_resolve_stage_timeout()`：agent_configs 优先 → StageDef.timeout 兜底。`_compute_run_timeout()`：Σ(已解析各阶段) × 2。`_elapsed_exceeds()` 在 orchestrator 每阶段前守卫。测试：+9。

---

### 2026-06-28 | TECoSim 剔除 + SemiForge 品牌统一

删除 `simulator_param.py` 和 `simulator_support.py`（DesignManager、RunManager、launch_sim、init_design、read_log — 约 1900 行）。移除 6 个 simulator 工具定义/handler/dispatch 分支。清除 13 个死常量、2 个配置项。TECoSim 相关命名全部改为 SemiForge。新增 Apache 2.0 LICENSE.txt；版权声明更新。

---

### 2026-06-29 | 零改一 — 产出路径统一到 session/\<uuid\>/

所有工作流产出目录从分散的三个平级路径（`outputs/dg_tft_runs/`、`prepared_data/`、`session/`）统一聚合到 `session/<uuid>/runs/` 下。15 个子任务覆盖 8 个源文件：

**路径重组**：`config_builder.py` 新增 `_runs_base_dir()` 构造 `session/<uuid>/runs/`。`tool_def.py` 中 `prepare_dg_tft_dataset` 处理器输出目录改用新路径。删除 `agent_configs.json` 中不存在的 `DG_TFT_OUTPUT_ROOT` key 引用。

**代码清理**：`data_preparer.py` 删除 `_PREP_OUTPUT_BASE_DIR`，输出目录改为调用方传入。`manager.py` 使用 `SESSION_PATH` 常量替代硬编码 `"session"`。修正 `PreparationTracker`、`artifacts.py` 中的文档注释路径。`.gitignore` 删除 `/prepared_data/`、`/outputs/`、`/test_data/` 条目。

**minimal_run 合并**：`quick_run` 和 `debug_run` 合并为 `minimal_run`（INI `[runtime]` section）。`run_model.py`、`prompt.py`、INI 模板同步更新。新增 2026-06-29 日 bugfix：`config_builder.py` 中 `output_dir` 使用 `Path.resolve()` 写入绝对路径，防止引擎 `resolve_path()` 对相对路径二次嵌套。

**测试修复**：`dg_tft_diagnostics_test.py` 补全 `_make_mock_ctx` 缺失属性 + `@patch _launch_diagnosis_subagent` 消除 MagicMock 序列化 traceback。`dg_tft_workflow_background_test.py` tearDown 添加 `test_persist` 清理。`test_subagent_unit.py` 自带 `session/test-unit-001/` 清理。

---

### 2026-06-29 | 零改二 — SubAgent 增强

**动机**：SubAgent 此前无法注入自定义 tool、强制结构化输出，或区分"干活 agent 文本即可"与"决策 agent 必须走 schema"两种场景。

**三层改造**：

1. **本地 Tool 注入**（`subagent.py` + 新建 `subagent_tools.py`）：
   - `SubAgent.__init__` 新增 `subagent_tools`、`tool_handlers`、`stop_mode`、`stop_schemas` 四个可选参数
   - `build_tools()`：`SUPPORTED_TYPES` 过滤后追加本地 tool schema
   - `execute_tool_calls()`：新增 `_handle_subagent_tool()` 本地分发 — handler 有 → 执行返回；无 handler + stop_schema → 捕获 args → DONE；无 handler 非 stop → FAIL（防配置错误静默）

2. **三种终止模式**：
   - `plain_text`（默认）：LLM 纯文本响应 → DONE，空 content ⇒ reasoning 兜底 → 提醒继续
   - `schema`：必须调用 `stop_schemas` 中列出的 tool 才能终止；纯文本不终止，提醒 LLM 调用指定 tool
   - `any`：任何情况（文本/tool/reasoning）都可以终止

3. **Tool 统一入口**（`subagent_tools.py`）：
   - 7 个 tool name 常量（`SUBAGENT_TOOL_*`）
   - `get_tools(stage)` 统一入口返回 `(schemas, handlers)`
   - `_STAGE_TOOLS` 注册表：`fit-physical`（3 schemas + 1 handler）、`check-data`（1 schema + 0 handler）
   - call site 无需分别管理 schemas 和 handlers 两个 dict

**单元测试**：15 项（`test_subagent_unit.py`），覆盖 plain_text/text fallback/schema 调查→输出/schema exhaustion/any/FAIL fallback/get_tools 集成。

---

### 2026-06-30 | 专家知识体系 (3.5)

新建 `src/workflow/dg_tft/knowledge/` 目录：
- `loader.py`：`load_knowledge(stage)` 按阶段名读取对应 `.md` 内容；`build_agent_prompt(stage, preferences)` 拼接 **USER PREFERENCES** + **EXPERT KNOWLEDGE** 双 block 模板，支持 `discovered_dir` 参数聚合反哺摘要
- 5 个知识文件空壳：`data_inspection.md` / `physical_kernel.md` / `static_model.md` / `hysteresis.md` / `final_interpret.md`
- 渐进式披露设计：预编写知识全量注入 prompt；反哺知识生成双文件（详细 `.md` + 摘要 `_summary.md`），摘要聚合注入 prompt，详情按需 `read_file`

**路径常量迁移**：`DG_TFT_KNOWLEDGE_DIR`、`DG_TFT_RUNS_SUBDIR`、`DG_TFT_DEFAULT_CONFIG_TEMPLATE` 从 `agent_configs.json` 移入 `constants.py`。源代码中 `ctx.agent_configs.get(...)` 全部改用常量作为 fallback 默认值。7 个源文件 + 5 个测试文件同步更新。`agent_configs.json` 保留的路径 key 仅限于机器/系统相关的可执行文件路径（`DG_TFT_AGENT_PYTHON_EXE`、`DG_TFT_MODELING_PYTHON_EXE`、`BASH_PATH`、`RIPGREP_PATH`）。

**测试**：14 项（`dg_tft_knowledge_test.py`），覆盖有效阶段、未映射阶段、文件缺失、偏好/知识组合、空白偏好、空输入。

---

#### 2026-07-01 | 参数试次记录 (3.6)

**`operation_log.py`** — `StageAttemptLog` + `OperationLog` 基础组件：

- `AttemptTrigger` 枚举：`INITIAL` / `QUALITY_FAIL` / `AGENT_ADJUST` / `FALLBACK_SENT` / `FALLBACK_RECEIVED` / `UPSTREAM_CHANGED`
- `StageAttemptLog` dataclass：`n` / `trigger` / `upstream` / `handoff` / `agent_decision` / `ini_snapshot` / `duration_s` / `metrics` / `adjustments` / `ok` / `error` / `quality_pass` — 完整 `to_dict()` → `from_dict()` 序列化
- `OperationLog`：`initialize(run_id, preferences)` → `append_attempt(stage, attempt)` → `set_final_status(status)` + `load()` / `get_stage_attempts()` / `get_last_attempt()` 查询接口

**编排器集成**：
- `_log_initial_attempt()` — 无条件记录 attempt_1（含 metrics），在 `_verify_artifacts` 之后、`_enforce_quality_gate` 之前调用。minimal_run / 无 quality threshold 的阶段也会被记录
- `_enforce_quality_gate()` 仅负责 retry（n≥2），不再重复记录初始 attempt
- 流水线完成/超时/失败均写入 `final_status`
- 重试目录 `_retry_N` → `attempt_{N+1}`（attempt_1 为初始执行，attempt_2 为第一次重试）
- 移除 `_cleanup_retry_dirs()` 函数 — 所有 attempt 目录保留不删，供 Agent E 解读

**`run_id` 统一**：
- `PreparationTracker.register_prep()` 新增可选 `run_id` 参数，传入时以 run_id 作为 key
- `prepare_dg_tft_dataset` handler 先创建 pending run 获取 run_id，prep 产物存入 `session/<uuid>/runs/run_{run_id}/data_prep/attempt_1/`
- `_get_prep_info()` 支持 run_id 回退查找

**测试**：26 项（`dg_tft_operation_log_test.py`），覆盖枚举值、dataclass 序列化/反序列化/缺字段容错、OperationLog 初始化/追加/多阶段/持久化/损坏文件/嵌套目录自动创建/顺序保持。

---

### 2026-07-01 | 数据输入阶段 (TODO 2.5.1–2.5.11) — Version 1.2.0

### 2.5.1 handoff 数据结构 (`schema.py`)
- `PrepHandoff` — Preparer 到 Verifier 的结构化交接信息，含 `observations` / `actions` / `potential_issues` / `output_manifest`
- `VerifierFeedback` — Verifier 的结构化反馈，含 `overall_ok` / `critical_issues` / `suggestion` / `worker_notes` / `summary`
- `PrepHandoff.from_worker_report()` — 从现有 `worker_report.json` 转换 handoff 的工厂方法
- `VerifierFeedback.from_dict()` — 支持 `critical_issues` 为字符串列表的兼容格式

### 2.5.5 `subagent_tools.py` 增强
- `inspect_xlsx` handler 从占位实现替换为真实 `openpyxl` 读表逻辑（sheet 名 + 列头），返回 `{"sheets": {...}, "sheet_count": N}`
- 新增 `_REPORT_VERIFICATION_SCHEMA` — Verifier 输出工具 schema，与旧 `build_report_verification_tool()` 等价
- `_STAGE_TOOLS` 注册 `"data-prep"` 阶段：`inspect_xlsx`（调查工具）+ `report_verification`（输出工具）
- 导入 `FAIL_LABEL` 供 handler 错误返回值使用

### 2.5.4 Verifier 迁移到 SubAgent
- **删除**：`_build_verifier_tools()` / `_execute_verifier_tool()` / `_run_verifier()` 手动 LLM 循环（~150 行旧代码）
- **删除**：`build_report_verification_tool()` — 工具 schema 已迁移到 `subagent_tools.py`
- **新增**：`_run_verifier_subagent(ctx, output_dir, report_verification_tool_name)` — 使用 `explorer` type + `stop_mode="schema"` + `stop_schemas=["report_verification"]`
  - Verifier SubAgent 通过 `glob_file`（explorer type 自带）列出文件
  - 通过 `inspect_xlsx`（subagent tool，有 handler）检查 XLSX 结构
  - 通过 `report_verification`（subagent tool，无 handler，output tool）输出结构化结果
  - 失败时有 fallback：返回 `overall_ok=False` 兜底 dict
- Verifier prompt 更新：`read_file` → `inspect_xlsx`

### 2.5.2-2.5.3 worker handoff 注入
- `_run_worker()` 新增可选参数 `handoff_history_text` — 非空时注入 prompt 末尾作为 "Previous attempt feedback"
- `_get_handoff_history(op_log)` — 从 `operation_log.json` 读取历史 data-prep attempts，提取 handoff + verifier feedback 拼接为文本

### 2.5.6 重试循环
- `_run_worker_and_verifier()` — 单次 worker→verifier→check 周期，返回 `(handoff, verifier_feedback, ok, issues)`
- `run_data_preparation_with_retry()` — Preparer-Verifier 主循环：
  - 每次 attempt 创建独立目录 `data_prep/attempt_N/`
  - 初始化 `OperationLog` 于 `run_output_dir/operation_log.json`
  - 初试 attempt 触发原因 = `INITIAL`，retry = `VERIFIER_FAIL`
  - worker 异常不中断循环，继续下一 attempt
  - 所有 attempts 记录到 operation_log，含 handoff + verifier_feedback
  - 成功时 `final_status="done"`，耗尽时 `final_status="exhausted"`
- `run_data_preparation()` 保留为兼容性包装，内部委托 `_run_worker_and_verifier`

### 2.5.7 标准数据集入口
- `run_verification_only()` — 跳过 Preparer，Verifier 直接验证已有数据
  - 验证通过：使用现有数据进入后续阶段
  - 验证失败 + 有 raw_dataset_dir：fallback 到 `run_data_preparation_with_retry()`
  - 验证失败 + 无 raw_dataset_dir：直接返回失败

### Operation Log 增强
- `AttemptTrigger` 枚举新增 `VERIFIER_FAIL` 和 `PREP_RETRY`
- `OperationLog` 现在在数据准备阶段初始化（早于 orchestrator），orchestrator 的 `run_workflow()` 后续可复用同一文件

### 其他改动
- `tool_def.py`：`prepare_dg_tft_dataset` handler 路径中间变量重命名为 `run_output_dir` / `attempt_1_dir`，语义更清晰
- `data_preparer.py`：删除未使用常量 `_PREP_VERIFIER_MAX_TURNS` / `_PREP_VERIFIER_MODEL_TIER`；删除多余 import `client` / `deepseek_support`
- `agent_configs.json`：`DG_TFT_PREP_MAX_RETRIES` / `DG_TFT_PREP_HANDOFF_HISTORY_SIZE` 已存在
- `DG_TFT_RUNS_SUBDIR` / `PREP_SUBDIR_NAME` 常量由零改一提供，无需新增配置

### 测试
- 更新 `test/dg_tft_preparer_test.py`：删除 `TestBuildReportVerificationTool`，mock patch 从 `_run_worker` / `_run_verifier` 改为 `_run_worker_and_verifier`
- 新增 `test/dg_tft_prep_retry_test.py`：31 项 — `PrepHandoff`/`VerifierFeedback` 序列化往返、`inspect_xlsx` handler、data-prep stage tools 注册、`AttemptTrigger` 枚举值、历史 handoff 构建、worker handoff 读取、重试循环（成功/retry/耗尽/异常继续/目录创建/operation_log 写入/retry trigger）、标准数据入口（通过/失败无 fallback/fallback 到 Preparer）
- 全量 864 tests pass (0 failures, 0 errors)

---

### 2026-07-01 | TECoSimAgent v0.2.6 同步升级 + 数据准备修复

### TECoSimAgent 同步 (commit `72c8038`)

**bash 风险检测 v1.5 → v1.10** (`bash_support.py`)：
- 新增输出重定向检测 `_check_output_redirect()`
- 新增 `$()` / backtick 命令替换 + `<()` / `>()` 进程替换评估
- 引号内容掩码消误报 `_mask_quoted()`
- here-doc 检测、eval 链替换分析
- `BASH_INLINE_SCRIPT` 风险等级 4 → 3
- +85 命令（包管理、构建、容器、网络等），扩展 docker/kubectl/git 子命令分类

**工具结果截断**：
- `truncate_tool_result()` 字段级迭代截断，保留 JSON 结构
- `SUBAGENT_EARLY_WARN_STEPS` 键名统一

**Markdown 渲染主题**：
- `get_console()` + Theme（H1-H6/代码/列表/表格/引用/链接/分隔线样式）
- `ContentMD` / `ReasonMD` 自定义 MD 元素（圆角表格、图标分隔线）

**prompt.py**：流收集器 None sentinel 修复、reasoning-only 补丁、图标化 resume

**SubAgent**：`start_time`/`elapsed_s` 持久化、GNU bash 提示、truncate_tool_result 集成

**常量/配置**：+25 常量、限值刷新（20000/60/3000）

### 数据准备修复 + 增强

- **operation_log 贯通**：handler 改用 `run_data_preparation_with_retry`，orchestrator 加载已有 log
- **handoff 修复**：不再覆盖 worker 写的 `worker_report.json`
- **manifest 写入**：retry loop 结束写 `manifest.json` + `prep_file_manifest.json`
- **Console 输出**：Preparer/Verifier 标记完成状态 + 统计
- **agent_list 归档**：worker/verifier/diagnosis SubAgent `pop` → `if_archived = True`
- **get_console() 启用**：`main.py` 使用带 Theme 的 Console
- **SubAgent TUI 展示**：`_TOOL_DISPLAY_KEYS` 新增 subagent tool 摘要

### 测试
- `bash_risk_test.py`：适应重新分类
- `dg_tft_preparer_test.py`：mock 目标更新 + SESSION_PATH 临时目录 + MagicMock 泄漏修复
- TODO.md：新增开发原则第 8 条「测试零残留」
- 864 tests pass，session/ 目录无残留
---

## 经验总结

### Python 模块双重导入陷阱
`from src.workflow.dg_tft.schema import X` 与 `from workflow.dg_tft.schema import X` 会被 Python 注册为两个独立模块。包内统一使用相对导入（`from .schema import`）。

### `dataclass.asdict` 与 deepcopy
`asdict()` 内部使用 `deepcopy()`。含锁/mock 的复杂嵌套结构会导致序列化失败。手动 `to_dict()` 更安全。

### Orchestrator 设计要点
- 阶段定义（静态元数据）与执行逻辑（动态编排）分离
- 断点续传：通过 `_stage_done()` 检查 `run.stage_results` 跳过已完成阶段
- 必经阶段 fail-fast；可选阶段 log warning 后继续
- Quality gate 在每个可训练阶段后执行，而非只在流水线末尾

### 控制台输出风格
- 错误：`WF_STATUS_ERROR`（bold red）| 警告：`WF_STATUS_WARN`（bold yellow）| 成功：`WF_STATUS_SUCCESS`（#8CDCA0）| 细节：`bright_black`
- 代码输出中不出现中文；`sys_log` 与 `console.print()` 互补

### 测试策略
当前 911 tests 全部通过（11 skipped）。运行命令：`python -m unittest discover -s test -p "*_test.py"`。

### Agent 介入边界
LLM/SubAgent 仅介入三处：数据整理、格式校验、失败诊断。训练执行、配置生成、质量判定、报告生成为纯规则代码。

---

### 2026-07-01 | INI 参数审计与暴露 (3.2)

### 参数定义表 (`param_defs.py`)
- `PHYSICAL_FIT_PARAMS`：3 个参数（max_iv_points / max_cv_points / max_nfev_scale）
- `FULL_MODEL_PARAMS`：11 个参数（epochs / learning_rate / batch_iv / batch_cv / cv_weight / cv_residual_level / enable_cv_residual / finetune_base_pkan / early_stop_patience / early_stop_min_epochs / symbolic_training_weight）
- `HYSTERESIS_PARAMS`：6 个参数（enabled / regime / epochs / hidden_capture / hidden_emission / amp_parameterization）
- 每参数含 type / lo / hi / default / desc；string 类型额外含 enum 约束
- `_STAGE_PARAM_MAP`：stage → (INI section, param_table) 映射

### Tool Schema 自动生成
- `params_to_tool_schema(param_table, tool_name, stage_label)` → OpenAI 兼容 function call schema
- `get_stage_param_schema(stage_name)` → 一键获取阶段 schema
- int → "integer"，float → "number"，bool → "boolean"，string → "string"（带 enum）
- 自动注入 bounds（minimum/maximum）+ description
- reasoning 字段置为 required

### 程序化硬兜底 (`quality_guard.py`)
- `validate_agent_adjustments(params, param_table, stage_section, stage_name)` — 四步校验：
  1. 白名单：param key 必须在 param_table 中
  2. 类型：int/float 支持 string→numeric 转换；bool 支持 yes/no/true/false 转换
  3. Enum：string 值必须在允许列表中
  4. 边界：超出 [lo, hi] 自动 clamp 并生成 warning
- 返回 (sanitized_params, warnings) — 不抛异常

### Agent 软兜底
- `get_soft_guard_prompt()` — 5 条自查规则（bounds / 物理合理性 / 参数交互 / 渐进性 / 专家知识）

### 文件
- `src/workflow/dg_tft/param_defs.py` — 新建：参数表 + schema 生成 + 软兜底
- `src/workflow/dg_tft/quality_guard.py` — 新增 `validate_agent_adjustments()` + 类型转换助手
- `test/dg_tft_param_defs_test.py` — 新建：45 tests
- TODO.md — 3.2.1–3.2.7 全部完成

---

### 2026-07-01 | DG-TFT 引擎 v90→v95 合并

### 引擎升级
- `dg_tft_full_model.py`：13.6K→19K 行。核心新增：基础 IV PKAN 训练管线（原独立 `train_base_pkan.py`）、6 个硬精度 gate、~25 新 CLI 参数、CV charge PKAN 拆分、`CompactKAN` 重命名、符号引擎扩展、safe scale search
- `fit_physical_kernel.py`：新增 IV 测量阶段对齐、CV gate 自动定向、CV quality 筛选
- `validate_dataset.py`：新增 CV 布局检测（`looks_like_shared_bias_wide_table`）

### 新增引擎模块
- `cv_preprocessing.py` — CV 预处理工具（过渡元数据、居中坐标、布局检测）
- `measurement_preprocessing.py` — IV 测量阶段漂移对齐
- `tensor_utils.py` — 从旧 `train_base_pkan.py` 提取的共享张量/数据工具

### 移除
- `train_base_pkan.py`（功能迁入 `dg_tft_full_model.py`）
- `dg_tft_ekv_level2.va`（替换为 `templates/physical_kernel_template.va`）

### 新增目录
- `templates/`（2 VA 模板：physical_kernel、portable_glue）
- `tools/`（7 脚本：acceptance bundle、hysteresis banks、ablation、delivery validation）

### 新增 actions
- `doctor`：环境诊断
- `smoke-static`：轻量静态模型验证
- `quick-static`：快速 IV/CV 训练 + 架构验证
- `package-release`：完整发布组装
- `package-portable`：可移植归档
- `verify-package`：包验证

### run_model.py 重构
- 架构从**直接 import** 切换为 **subprocess 调用**（匹配 v95 原始架构，所有入口改为 CLI-only）
- 保留迁移适配：`_normalize_cv_format()`、`_prepare_fallback_model_dir()`、`minimal_run`、example data guard、`train-base-pkan` action
- v95 helper：`run()`/`run_gate()`/`run_artifact_stage()`、`resolve_hspice()`、`dynamic_env()`、`doctor()`、release 组装

### 参数表扩展（3.2 维护 — 设计见 3.2.7）
- `PHYSICAL_FIT_PARAMS`：3→6（+3 `[preprocessing]` params）
- `FULL_MODEL_PARAMS`：23→44（+21：base IV 训练、IV residual、symbolic、LR multipliers、训练调度、cv_train_subset 等）
- `HYSTERESIS_PARAMS`：7→13（+6：fast/slow regime 拆分）
- 总计 32→73 参数
- `quality_guard.py` `_PARAMETER_BOUNDS` 新增 `[preprocessing]` section，`full_model` +24 key，`hysteresis` +2 key
- `_PARAMETER_WHITELIST`：30→56 entries

### 测试
- `dg_tft_functionized_api_test.py`：重写，匹配 v95 CLI-only 架构
- `dg_tft_param_defs_test.py`：预期参数计数 22→44；新增 `preprocessing` section
- `dg_tft_quality_guard_test.py`：string/bool 参数跳过列表扩展到 7 项
- 911 tests pass（11 skipped）

### 文件
- `src/device_modeling/dg_tft/engine/` — 替换 4 个、新增 3 个、删除 2 个
- `src/device_modeling/dg_tft/{templates,tools}/` — 新建
- `src/device_modeling/dg_tft/{dynamic,reporting}/` — 全覆盖更新
- `src/device_modeling/dg_tft/{config,run_model.py}` — 全覆盖更新

---

### 2026-07-01 | 3.2 维护 — KAN 架构参数暴露 + minimal_run 精度门跳过 + 产物路径修正

**KAN 架构参数轻量扩展**：
- 温度 KAN 隐藏层/网格/阶数参数化：`temperature_kan_hidden`（8）、`temperature_kan_grid`（3）、`temperature_kan_k`（3）→ `[temperature]` section
- 引擎 `dg_tft_full_model.py`：新增 `--temperature-kan-hidden/grid/k` argparse；`ResidualDiagnostic.__init__` 接受并用于 `CompactKAN` 构建

**训练正则化三件套**：
- `weight_decay`（1e-6）、`grad_clip`（5.0）、`lr_eta_min_frac`（0.02）→ `[full_model]` section
- 引擎已有 argparse，仅补全 INI → CLI 传参管线

**`--skip-precision-gates` 与 minimal_run 对齐**：
- 引擎新增 `--skip-precision-gates` flag，门控 3 个精度门（Base IV-PKAN / IV residual held-out / Deployment HSPICE）→ 超标时打印 warning 而非 RuntimeError
- `run_model.py` 中 `quick=True` 时同时传 `--quick-run` + `--skip-precision-gates`，保证 minimal_run 流程不因训练质量中断
- `apply_quick_run_preset` 中补加 `iv_residual_stage_epochs` 压缩（≤140）

**产物与 gitignore 修正**：
- `report_builder.py`：ARTIFACT_MANIFEST 写入顺序移到最后，消除 JSON/MD 误报 missing
- `.gitignore`：忽略 `src/device_modeling/.runtime_cache/`、`trained_model/`、`validation/`

**参数统计**：`FULL_MODEL_PARAMS` 44→50，总参数 79。

**受影响文件**：`dg_tft_full_model.py`、`run_model.py`、`model_config_new_device_template.ini`、`param_defs.py`、`quality_guard.py`、`report_builder.py`、`.gitignore`、测试 2 个。

**测试**：911 pass（11 skipped）。

---

### 2026-07-01 | 开发顺序调整 — Tool 统一 + while+pointer 提前

**变更**：
1. 将 1.5 Tool 统一从原 Step 13 提至 Step 6 — 确保后续 Agent 开发面向 3 个统一 tool
2. 将 while+pointer (3.7) 与 Agent A (3.4.1) 合并为 Step 7 — Agent A 依赖 fallback，fallback 依赖 while+pointer，三者互相锁死，一次性解决
3. 语义统一：移除 `fallback_to_verifier`，全链统一用 `fallback_to_previous` — 线性 pipeline 上每个 Agent 都向上一跳

**新的开发顺序**：
0-5 ✅ → 6. Tool 统一 → 7. while+pointer + Agent A → 8. Agent 基础设施 → 9. Agent B → 10. Agent C → 11. Agent D → 12. Agent E

---

### 2026-07-01 | Step 6 Tool 统一 + 质量门多指标 + Agent 决策设计

### Step 6: Tool 统一 (1.5)

**4→3 tool 收敛**：
- `prepare_dg_tft_dataset` + `start_dg_tft_workflow` → **`dg_tft_run`**
- `query_dg_tft_workflow` → **`dg_tft_status`**
- `read_dg_tft_workflow_report` → **`dg_tft_report`**

**入口统一为 `entry` enum**（替代 `skip_prep`/`direct` 两个 bool）：
- `entry="full"` (默认)：Preparer→Verifier→pipeline（原始数据）
- `entry="verifier"`：Verifier→pipeline（train-ready 数据，需验证）
- `entry="check-data"`：直跑 pipeline（用户明确跳过检查）

**受影响文件**：
- `constants.py`：旧 4 常量 → 新 3 常量（TOOL_NAME_DG_TFT_RUN/STATUS/REPORT）
- `tool_def.py`：旧 4 def + 4 handler → 新 3 def + 3 handler，`dg_tft_run` 统一 merge 预处理+建模
- `tool_dispatch.py`、`command.py`、`prompt.py`、`agent_context.py`：同步更新
- 测试 6 个文件：全部迁移到新 tool/entry 命名

### Bug 修复

1. **`run_id` padding 不一致**：`dg_tft_run` 用 `f"run_{run_id}"`（→ `run_1`），`config_builder` 用 `f"run_{run_id:04d}"`（→ `run_0001`）→ 导致 verifier 产物与 workflow 产物分离。统一为 `:04d`。

2. **RUN_SUMMARY 状态显示 RUNNING**：`_finalize_run` 在 `mark_done` 之前调了 `build_and_write_reports`，导致 report 快照到旧状态。改为先 mark_done 再 build。

3. **TUI 只显示 4/5**：final-report no-op 被静默跳过 → 加占位打印 `✓ (report generation handled by orchestrator)`。

4. **Artifact 文件名过期**：v90→v93 engine 升级后 checkpoint 名变了（`v90_safe_residual_state.pt`→`v93_safe_residual_state.pt`），且不再单独产出 `.va` 文件。同步 `stages.py` `key_artifacts`、`artifacts.py` `ARTIFACT_FILENAMES` + `STAGE_ARTIFACTS`、hysteresis 嵌套路径。

### Verifier 增强

- Prompt 从"抽查代表性文件"→"逐个检查所有 `*_train.xlsx` 文件"
- `operation_log.json` 记录 verifier 的 `agent_decision`（完整 IV families + CV sheets + it/ 文件审计）

### 质量门升级

**从单指标 → 多指标**：`_STAGE_THRESHOLD_KEYS` 类型从 `dict[str, str]` → `dict[str, dict[str, str]]`。
- Train-static 同时检查 `residual_iv_rms` (0.35) + `residual_cv_rms` (0.06)
- `DG_TFT_QUALITY_STATIC_CV_RMS_MAX` 从死代码 → 实际连线
- `get_quality_thresholds()` 返回所有已定义 metric 的阈值
- 触发规则：**任意一个 metric 超标即触发**（不是全部超标才触发）

### Agent 决策设计更新

- `AttemptTrigger` 新增 `AGENT_ACCEPT`：Agent 主动接受当前结果继续的 trigger
- TODO 3.3 更新：四选一决策流程（adjust_params / fallback / accept / 耗尽）+ trigger→handoff 映射表
- TODO 3.6 更新：operation_log schema 增加 `agent_accept` 示例

### Session 清理规则

- TODO.md 开发原则 #10：只允许删除非 UUID mock 目录（MagicMock/test_tools/test_integration），严禁删 UUID v4 session
- AGENTS.md 一行浓缩版规则

### 测试

- 全量 911 tests pass (0 failures, 0 errors, 11 skipped)

### 2026-07-01 | Step 7 while+pointer + Agent A + 数据纳入 Pipeline (3.7 + 3.4.1)

**结构性重构**：`orchestrator.py` 中 for 循环 → while+pointer 状态机。

**StageDef.kind**：`"agent"`（Preparer, Verifier）vs `"script"`（check-data 到 final-report）。Agent 阶段 spawn SubAgent；脚本阶段调 `run_stage()`。

**Pipeline 整合**：Preparer + Verifier 现在是 `_CORE_STAGES` 中的 StageDef 条目（用 `get_stages(entry=)` 过滤）。数据准备从 `tool_def.py:dg_tft_run` 移入 orchestrator 的 while+pointer 循环。

**Agent A（check-data）**：硬失败 → `_spawn_agent_a()` SubAgent（`stop_mode="schema"`）→ `fallback_to_previous` tool。Agent 判断失败是否需要回退到 Verifier 还是最终失败。

**回退链**：A → Verifier → Preparer。`_revert_tasks()` 将回退指针处的 TUI task board 条目重新标为 in_progress。`_find_stage_pointer()` 按名称找阶段索引。

**懒配置构建**：`config_path` 在第一个脚本阶段前才 build（Verifier 通过后），不在 `tool_def.py` 中。

**Subagent tools**：移除 `SUBAGENT_TOOL_FALLBACK_TO_VERIFIER` — 所有回退统一用 `fallback_to_previous`。

**data_preparer.py 新增**：`run_preparer_stage()`、`run_verifier_stage()`、`copy_dataset_to_data_prep()`。

**测试**：911 pass (0 failures, 0 errors, 11 skipped)，session 目录零残留。

### 2026-07-01 | Engine v95→v96 合并 — 冻结 KAN Grid 参数

**根因**：`list(kan.parameters())` 将所有参数（含 `act_fun.*.grid` B-spline 节点位置）传给 Adam。Adam 移动节点导致非单调间距 → IV 曲线 IDVG/IDVD 出现可见跳变。

**CompactKAN 变更**：
- `_optimizer_parameter_names`：记录 pykan 构造时的可训练参数合约
- `optimizer_parameters(enabled)`：仅返回 pykan 标记的可训练权重，过滤 grid/mask/affine 元数据
- `grid_audit()`：校验 B-spline 节点严格单调性
- `assert_valid_grids(context)`：grid 非单调时抛 RuntimeError

**训练调点**：11 处 `list(x.parameters())` → `x.optimizer_parameters(enabled=True/False)`

**校验**：15 处 `assert_valid_grids()` 调用（checkpoint 加载后 + 每 epoch）

**Metrics**：2 处 `kan_grid_audit` 字典（各 4 个 KAN block）

### 2026-07-01 | Pipeline 稳定化 — Agent 阶段检查、Verifier 入口、报表修复

**Agent 阶段成败检查**：`_run_agent_stage` 现在返回 `(attempt_dir, ok)`。while+pointer 仅在 ok 时前进；失败时重试或回退到 Preparer。

**Verifier 入口修复**：标准数据集在 Verifier 检查前先拷贝到 `data_prep/attempt_N/`（之前检查空目录）。

**Verifier→Preparer 回退**：Verifier 失败 → pointer 跳回 DATA_PREP，worker 带着 handoff history 重处理。

**报表修复**：
- RUN_SUMMARY 0KB：manifest 在报告文件写入后构建
- operation_log duration_s：`run_stage` 环绕 `time.monotonic()` 计时
- ARTIFACT_MANIFEST 自指：排除自身条目

**Metrics 统一**：`metrics_parser` v90→v93。引擎只写 `v93_safe_residual_metrics.json`（删除 v92 双写）。

**Agent 阶段记入 stage_results**：verifier/data-prep 现在通过 `wm.update_stage()` 记入 `workflow_runs.stage_results`，与脚本阶段统一。

**Stage display names**：补了 `data-prep`/`verifier` 条目。

**v96 bug 修复**：`kan_grid_audit` 循环变量 `audit` 遮蔽了 DataFrame `audit` — 改名为 `block_audit`。

**测试**：911 pass (0 failures, 0 errors, 11 skipped)

### 2026-07-02 | Pipeline 稳定化 — 第二轮：TUI 状态、路径修复、Prompt 增强

**TUI Task 状态修复**：
- `_revert_tasks` 原来把 fallback pointer 之后**所有** task 标为 IN_PROGRESS，包括从未开始的 PENDING task。修复为只 revert `TaskStatus.COMPLETED` 的 task。
- 进度显示：`display_idx` 从单调递增值改为 `pointer + 1`，回退时正确显示位置。
- `_finalize_run` 新增 `board`/`task_map` 参数；失败时遍历剩余 task 标 DELETED；caller `tool_def.py` except handler 也做 task 清理。
- Task 格式改为 `[DG-TFT, ID: 1] [Step-1] Prepare dataset`。

**Rich MarkupError 崩溃修复**：`_console_summary` / `_print_diagnosis` / `error_exit` / `tool_def.py` 异常的打印全部 escape 或 `markup=False`。引擎报错文本中的 `[...]` 不再被误判为 Rich markup。

**Quality Gate 重试移除**：`_enforce_quality_gate`（~180 行）替换为 `_check_quality_once`（只检查不重试），删除相关 helper 函数、未使用 import（`shutil`、`suggest_adjustments` 等）。重试留给未来 Agent B。相关 4 个测试更新为新签名。

**Nested dir 路径修复**：
- orchestrator `output_dir` 改用 `os.path.abspath`（原来相对路径 `./session/uuid/runs/run_0001`）
- `run_model.py:resolve_path` 加最后的 `.resolve()` 防御
- 路径翻倍链：orchestrator 相对路径 → quality gate retry 写入 INI → `resolve_path` 基于 config 父目录解析 → 双倍嵌套

**目录扁平化**：`DG_TFT_RUNS_SUBDIR` 从 `"runs"` → `""`，路径 `session/<uuid>/runs/run_NNNN` → `session/<uuid>/run_NNNN`。

**Preparer Prompt 增强**：Worker prompt 的 XLSX 处理部分新增 wide-format 识别和 reshape 指令（`pd.melt`/`df.stack` → 标准四列 `VD_V VBG_V VTG_V ID_A`）。

**测试**：915 passed, 11 skipped

### 2026-07-02 | Engine 完整同步 v96 + 预训练 PKAN + Pipeline 首次全通过

**Engine v96 完整同步**：
- dg_tft_full_model.py 整体替换为 v96 基线（之前只合并了 CompactKAN grid freeze）
- v96 新增：plateau jump 诊断 + 3 gate checks、KAN grid monotonicity audit、VDS≈0 排除、6 阈值 quick-run 宽松
- Port 3 处自定义：skip-precision-gates（3 个 gate）、Temperature KAN 可配（ResidualDiagnostic + 3 CLI 参数）
- 其余 5 个 engine 文件 + 8 个 dynamic 文件 MD5 与 v96 完全一致

**Bug 修复**：
- dg_tft_full_model.py: 修复 udit DataFrame 被 or block, audit 循环覆盖的变量名冲突 bug（v96 原有 bug，sync 后重新引入）
- param_defs.py: 9 个默认值对齐 v96 INI（learning_rate 0.002→0.0012、cv_weight 8.0→10.0、symbolic_training_weight 0.015→0.05 等）
- 
un_model.py: fallback 默认值对齐 v96（epochs、lr、early_stop 等）
- Verifier 入口模式下 CV_原始_train.xlsx 备份文件干扰 verifier 检查：排查后确认为纯格式差异，删除冗余文件

**预训练 PKAN 模型**：
- v96 自带的 	rained_model/base_physical_pkan/ 包含预训练 KAN 权重（pkan_torch_model.pt 53KB）
- 路径：src/device_modeling/trained_model/base_physical_pkan/（已加入 .gitignore）
- 之前 RELEASE_ROOT = CODE_ROOT.parent 解析为 src/device_modeling/，我方拷贝到了错误的 src/trained_model/
- fallback metadata（1315 字节空壳）被引擎读取，导致 base IV-PKAN 从随机初始化起步，收敛到 on=0.074 > 0.06
- 修正路径后 base IV-PKAN 收敛到 on=0.049（与 v96 单独测试一致），通过所有质量门

**Pipeline 首次全通过**：
- 全部 5 个阶段 DONE：verifier ✅ check-data ✅ fit-physical ✅ train-static ✅ train-hysteresis ✅
- train-static ~1h52m，0 错误，10/10 产物齐全
- 指标：iv_rms=0.248, residual_iv_rms=0.229, residual_cv_rms=0.021, hysteresis=0.048

**测试**：916 passed, 11 skipped

### 2026-07-02 | Step 8: Agent 基础设施

**Tool 定义** (subagent_tools.py):
- 新增 3 个 tool schema：~~switch_kernel（Agent B 切物理内核 — 2026-07-02 删除，当前仅 EKV Level-2）~~、ccept_quality（接受当前质量）、contribute_knowledge（Agent E 反哺知识）
- 新增 _contribute_knowledge_handler：写入 knowledge/_discovered/run_{id}_{timestamp}.md
- _STAGE_TOOLS 注册表扩展覆盖全部 6 阶段：train-static/train-hysteresis 各 4 tools（inspect_xlsx + adjust_params + fallback_to_previous + accept_quality），final-interpret 含 contribute_knowledge
- get_tools() 新增 djust_params_override 参数：orchestrator 注入 auto-generated 阶段专属 schema

**参数校验 Pipe** (quality_guard.py):
- pply_agent_adjustments(config_path, adjustments, ini_section) → snapshot：写 INI + 返回操作快照
- snapshot_ini_section(config_path, ini_section) → str：读 INI section 快照

**通用 Agent 运行循环** (orchestrator.py):
- _spawn_modeling_agent() 通用 Agent B/C/D spawn：
  - auto-generated djust_params schema（param_defs → params_to_tool_schema）
  - 每阶段专属 tool set
  - stop_mode="schema"，4 种决策：adjust_params / fallback_to_previous / accept_quality / error
  - 注入用户偏好 + 软护盾（get_soft_guard_prompt()）
  - try/except 包裹应对 mock/test 上下文
- _process_agent_decision() 决策处理 pipe：
  - adjust_params → alidate_agent_adjustments() → pply_agent_adjustments() → retry
    - fallback_to_previous → 记录 FALLBACK_SENT → pointer 跳转上游
  - accept_quality → 记录 AGENT_ACCEPT → 推进
- _spawn_agent_e() Agent E 最终解读：stop_mode="any" + contribute_knowledge
- _assess_stage_quality() 纯规则质量检查（无 LLM 调用）
- while+pointer 脚本阶段重试循环：hard failure + quality fail 均触发 agent 决策
- 4 个映射表：_STAGE_HAS_MODELING_AGENT、_UPSTREAM_STAGE、_AGENT_LABELS
- 旧占位符 _check_quality_once / _agent_stage_placeholder / _agent_e_placeholder 替换为实际实现

**测试**：
- 	est_subagent_unit.py：8 新测试（train-static/hysteresis/final-interpret 工具注册、adjust_params_override、contribute_knowledge handler）
- dg_tft_quality_guard_test.py：4 新测试（apply_agent_adjustments 写入/创建 section、snapshot_ini_section 正常/缺失）
- dg_tft_orchestrator_test.py：9 新测试（_stage_has_modeling_agent、_normalize_tool_name、_process_agent_decision adjust/accept/fallback/empty）
- **936 passed, 11 skipped**（从 916 增至 936，+20 新测试）

**Agent 决策矩阵**：
| Agent | 触发 | 可用决策 |
|-------|------|----------|
| A | check-data 硬失败 | fallback_to_previous → Verifier |
| B | fit-physical 硬失败/质量不达标 | adjust_params / fallback_to_previous / accept_quality |
| C | train-static 硬失败/质量不达标 | adjust_params / fallback_to_previous / accept_quality |
| D | train-hysteresis 硬失败/质量不达标 | adjust_params / fallback_to_previous / accept_quality |
| E | Pipeline 结束 | contribute_knowledge + 自由文本 |

### 2026-07-02 | Agent E 知识反哺修复

- **contribute_knowledge → stop_schema**：从 handler（LLM 反复调用，输出多个碎片文件）改为 stop_schema tool（只能调用一次，凝聚成完整知识贡献）
- **文件命名**：从 
un_{run_id:04d}_{timestamp}.md（跨 session 碰撞、无意义）改为 {title_slug}_{timestamp}.md（从 agent 的英文 title slugify，人类可读）
- **minimal_run 跳过 Agent E**：新增 _is_minimal_run(config_path) 检查
- **必须时才贡献**：system prompt 明确标注 contribute_knowledge 为 OPTIONAL，仅真正新发现时才调用
- **英文强制**：Agent E prompt + contribute_knowledge tool description 标注 English only
- **gitignore**：新增 /knowledge/_discovered/（不进仓库）

### 2026-07-02 | 专家知识文件 + INI 三方一致性审计（三轮修复）

**switch_kernel 删除**：
- 引擎仅存在 EKV Level-2 一种物理内核，无 kernel 抽象层。switch_kernel 写入 INI 但引擎不消费
- 全量删除：subagent_tools.py、orchestrator.py、测试、全部文档（6 files, -100 lines）

**5 个专家知识文件 Tier 1 创建**：
- 知识四维分类：原理/行为/当前能力/经验（JSONL 提取）
- 模板：6 section（Global Context → Upstream/Downstream → INI Parameters → Quality Metrics → Engine Behavior → Failure Patterns）
- 写-审-检工作流：subagent 调研代码 → 写 MD → review subagent 查事实/遗漏 → 本人 INI 跨检
- 5 文件：data_inspection.md（~240行）、physical_kernel.md（~400行）、static_model.md（~550行）、hysteresis.md（~430行）、final_interpret.md（~270行）
- 目录统一：knowledge/（项目根），loader `_default_knowledge_dir()` 更新，constants.py 更新
- Section 6（Failure Patterns）留空，标注从 Zikang 的 Codex JSONL 日志填充

**第一轮 review（subagent 交叉验证）**：
- 5 个 explore subagent 并行审查，发现 40+ 事实错误/遗漏/过简
- 关键错误：physical_kernel.md 声称 CUDA（实为纯 CPU scipy）、static_model.md IV-PKAN 输入特征全错、hysteresis.md 所有 enum 值错误、final_interpret.md 级联逻辑完全错误（pipeline 不立即停止）

**第二轮审查（INI/代码/MD 三方一致性）**：
- 68 个参数全面审计（6 PHY + 50 FULL + 12 HYS）
- 发现 8 个死参数（[base_pkan] 全部不传子进程）
- 发现 5 个默认值不匹配（base_iv_epochs 700→540、hysteresis epochs 1800→4000 等）
- 发现 critical bug：cv_train_subset enum 含 "all_gate_cv" 但引擎只接受 "all"（agent 调整会 argparse 崩溃）
- 发现 cv_residual_level hi=3 但引擎 choices=[0,1,2]
- 发现温度参数 4/6 key 名不匹配导致 agent 调整 SILENTLY IGNORED
- param_defs.py 修复：regime/amp_parameterization enum 全部对齐引擎 argparse
- 删除死参数：hysteresis print_every（从不传子进程）
- quality_guard.py 同步修复：自动调整策略 "aggressive"→"independent"，移除 print_every bounds/whitelist

**第三轮（补全缺失参数）**：
- 21 个 run_model.py 传入子进程但未在 param_defs 中暴露的参数全部补全：
  - [full_model] 9 个：reset_base_from_physical_fit、train_base_iv_pkan、train_iv_residual_stage、freeze_iv_residual_after_stage、iv_residual_holdout_stride、iv_residual_off_anchor_weight、disabled_iv_pkan/cv_pkan/cv_charge_pkan
  - [symbolic] 10 个：edge_steps、edge_topk、iv_edge_topk、iv_adaptive_top2_mse、iv_calibration_max_terms、edge_finetune_steps、network_finetune_steps、device_iv_finetune_steps、device_iv_measured_steps、device_cv_finetune_steps
  - [validation] 2 个：safe_scale_steps、joint_cv_safe_scale_search
- 温度 key 命名修复：param_defs 新增 `ini_key` 字段映射，`apply_agent_adjustments` 改造为接受 param_table 并按 section+ini_key 写入正确的 INI
- _STAGE_PARAM_MAP 扩展为包含 [symbolic] 和 [validation]
- quality_guard bounds + whitelist 同步新增
- 最终：68 → 71 参数，agent 可控全部流水线传入的建模超参数

**知识文件 MD 同步**：
- physical_kernel.md：CUDA→纯 CPU scipy、引擎路径修正、IV 残差 6 因子权重表、CV trap fit 独立系数、正则化精确值、f_scale=0.55、CGG 10kHz 保留、auto-orientation 阈值 1.55、iv_stage_alignment 修改数据
- static_model.md：finetune_base_pkan 默认值、cv_residual_level 语义、IV-PKAN 6 特征、两套调度机制、CV residual 21 slots/7 active、final-report kind="script"、训练阶段架构
- hysteresis.md：全部 enum 值修正、epochs/hidden 运行时默认值、移除 print_every、knowledge 未注入标注
- final_interpret.md：级联逻辑全面重写、Agent E 仅收 last attempt、upstream 字段永为空、contribute_knowledge 机制、minimal_run 跳过、pre-computed classification

**测试**：935 passed, 11 skipped
- **gitignore 补充**：/knowledge/_discovered/ 不进仓库

---

## 2026-07-02 — 专家知识文件三轮审计 + 修复

**背景**：5 个知识文件（data_inspection/physical_kernel/static_model/hysteresis/final_interpret.md）已经过三轮 subagent 审查，但 agent 不完全可靠，需要额外验证。

**审计架构**：分三轮运行 5 个 explore subagent，每轮等待完成后审查结果再进下一轮：

- **第一轮 — 正向提取清单**：穷举 MD 中每条可验证的事实声明（参数默认值、枚举值、行为描述、算法步骤），输出清单列出每条声明的待核对源文件。产出：187+83+82+68+49 = 469 条声明。
- **第二轮 — 逐条核对**：对每条声明读源文件验证，输出 ✅/❌/⚠️/📋。对于参数声明，三值交叉比对 param_defs → run_model.py 传入值 → 引擎 argparse 默认值。产出：18 个错误 + 27 个过度简化 + 17 个 TODO 项。
- **第三轮 — 反向扫描**：不再看 MD，只看源代码，找出影响 agent 决策但 MD 完全未提的关键信息。产出：48 个缺失项（10+10+7+8+13）。

**关键发现（18 个错误）**：

| 文件 | 错误 | 严重度 |
|------|------|--------|
| data_inspection.md | `_run_stage_diagnosis` 声称对 check-data 运行但代码用 `break` 跳过 | critical |
| data_inspection.md | "所有 check-data 错误都分类为 DATA_ERROR" — 实际 keyword matching 是 stage-agnostic，CONFIG_ERROR/TRAINING_NUMERICAL_ERROR 关键词可匹配 | high |
| data_inspection.md | `run_artifact_stage` 检查的 artifacts 是 `.pt` checkpoints（非 `dataset_validation.json`） | medium |
| physical_kernel.md | IV 采样 subthreshold 权重 MD 写 ×3，代码实际 base 1.0 + 3.0 = 有效 4.0× | medium |
| physical_kernel.md | IV 采样 on-region 权重 MD 写 ×1.5，代码实际 base 1.0 + 1.5 = 有效 2.5× | medium |
| physical_kernel.md | 缺失 MISSING_ARTIFACT_ERROR 完整关键词分类（6 个关键词，confidence 0.9） | medium |
| static_model.md | `cv_residual_level` MD 写 "2 or 3" 但引擎 choices=[0,1,2]，最大是 2 | high |
| static_model.md | CV training phase 4 边界参数标为 INI-writable，实际不在 param_defs 中 | high |
| static_model.md | "50 parameters total" 实际数为 51/63/71（不同计数维度） | low |
| static_model.md | IV-PKAN 输出名称与引擎 CORR_NAMES 不匹配（dlog_rs_fwd/reverse vs dpsi_V） | medium |
| static_model.md | `base_iv_epochs` MD 默认 700，引擎 argparse 默认 540 | medium |
| hysteresis.md | `fast_amp_parameterization` 默认写 "None→fallback"，实际 param_defs 默认 "conservative" | high |
| hysteresis.md | `slow_amp_parameterization` 默认写 "None→fallback"，实际 param_defs 默认 "independent" | high |
| final_interpret.md | attempt_n > 1 的成功重跑 NEVER 记录到 operation_log（仅记录 agent decision entry） | critical |
| final_interpret.md | PREP_RETRY trigger 在 enum 中存在但生产代码 NEVER 发射 | medium |
| final_interpret.md | Agents B/C/D quality_fail trigger 仅用于 prompt 上下文，log 实际写 AGENT_ADJUST | medium |
| final_interpret.md | 级联受害者描述假设 operation_log 有记录 — 实际首次尝试硬失败不记 op_log | high |

**关键缺失项（48 项）**：

- **data_inspection.md**（10 项）：Agent A prompt 不含 stdout summary（只含截断 stderr 2K 字符）、CV normalization 在 Exception 时静默返回、诊断分类 concatenation 边界可能产生假阳性、CV 验证要求 ≥3 行否则 layout="unknown"、I-t 自动发现关键词 "it" 是子串匹配可触发误报
- **physical_kernel.md**（10 项）：**Critical：** Agent B prompt 完全不注入专家知识（zero engine context）、9 个引擎内部阈值硬编码不可调（RS 求解器阈值 / VDS 过滤器 0.55V / Stage3 transfer 比率 / smooth_pos ε 等）、silent data drops（NaN/非正电容/无"long"sheet 全部静默丢弃无警告）
- **static_model.md**（7 项）：**Critical：** 5 个额外精度门未文档化（base IV region 7-check gate / IV residual heldout gain gate @ 8e-4 / region degradation gate @ 0.008 / KAN grid audit / charge passivity gate）、30+ 个诊断文件仅列出 12 个、引擎错误消息未分类
- **hysteresis.md**（8 项）：**Critical：** Acceptance gate RuntimeError 作为独立硬失败路径未文档化、7 个 acceptance gate 阈值硬编码不可调、数据过滤可能静默丢弃所有段引发 "no SRH tau labels" RuntimeError
- **final_interpret.md**（13 项）：**Critical：** inline prompt 混淆 ok/quality_pass（仅看子进程退出码，标 PASS 可能 quality_pass=False）、错误文本完全不注入 prompt、Agent E 未经 output_dir/config_path、15 行截断输出不存磁盘、minimal_run 跳过在失败路径失效

**知识文件修复**（共 ~40 处修改）：

| 文件 | 修改项 |
|------|--------|
| data_inspection.md | `_run_stage_diagnosis` 段落替换为正确说明、错误分类补丁（stage-agnostic keyword matching）、新增 Agent A prompt 内容章节（stderr 仅 2K chars、stdout 摘要不在 prompt 中）、CV normalization 静默异常说明、artifact stage artifacts 澄清（.pt 非 dataset_validation.json）、I-t 子串匹配警告、缺失关键词补充 |
| physical_kernel.md | IV 采样权重修正 ×3→×4, ×1.5→×2.5、MISSING_ARTIFACT_ERROR 分类及其它缺失关键词添加、knowledge 未注入标注（critical box）、引擎内部硬编码阈值节（9 个不可调参数）、adjust_params schema 无诊断上下文标注、CV auto-orientation 硬编码阈值警告、静默数据过滤节 |
| static_model.md | cv_residual_level "2 or 3"→"2" 全部修正、CV phase params "INI-writable"→"engine-internal only"、"50 parameters"→"63/71"、IV-PKAN 输出名称修正为引擎 CORR_NAMES（dpsi_V 替代 dlog_rs_*）、新增 5 个引擎精度门节、base_iv_epochs 默认差异脚注、引擎错误消息分类表 |
| hysteresis.md | fast_amp_parameterization 默认 "None→fallback"→"conservative"、slow_amp_parameterization 默认 "None→fallback"→"independent"、acceptance gate RuntimeError 作为独立失败路径说明、7 个硬编码 acceptance gate 阈值表、静默数据过滤说明（RuntimeError 触发条件）、Hardcoded training 超参数列表（lr/wd/gradient_clip 不可调）、HSPICE 失败非阻塞标注 |
| final_interpret.md | operation_log 覆盖缺口（尝试硬失败/成功重跑不记录）、PREP_RETRY trigger 从未发射说明、quality_fail trigger 实际使用澄清（仅 Agent A 用）、inline prompt ok 字段与 quality_pass 混淆警告、error text 不注入标注、Agent E 工具完整列表（bash/grep_file/web_search/web_fetch）、15 行输出截断警告、minimal_run 跳过在失败路径已损坏、新增 agent_reasoning/ini_snapshot 字段文档、质量阈值来源说明（agent_configs 可配置非固定常量） |

**审计报告存档**：`doc/audit_r1/` `doc/audit_r2/`（Round 1 清单 + Round 2 验证 + Round 3 反向扫描结果可于后续 cleanup 删除）

---

## 2026-07-03 — 专家经验提取（Codex 会话 JSONL → knowledge/experience/）

### Pass 0-3 Pipeline

**背景**：`ref/` 下有 2 个 Codex 会话 JSONL（42MB 06-29 建模诊断 + 566MB 05-22 全生命周期开发）。需从中提取泛化专家经验，补充到 `knowledge/experience/*.md` 尾部。

**Pass 0 — 密度比较**：Python 脚本对两文件做关键词密度抽样。File 2（06-29, 42MB）建模诊断密度高（127 meaningful/11232 行），File 1（05-22, 566MB）虽数值密度高但内容为 Origin 画图/CSV 转换。先从 File 2 实验。

**Pass 1 — JSONL → Compressed Dialog**：Python streaming 脚本逐行处理，保留 `[USER]` + `[AGENT:commentary/final_answer]` 全文，`[TOOL:name/call_id]` + `[OUTPUT:call_id]` 用 call_id tag 链接。File 2：42MB → 601KB（1.4%），28 turns, 207 agent msgs。File 1：566MB → 3.6MB（0.6%），328 turns, 2840 agent msgs。

**Pass 2 — CD → Diagnostic Extracts**：关键词过滤（root cause/fix/根因/定位/参数 etc.），tool 序列 >5 对压缩为 `[... N tool calls ...]`。File 2：601KB → 98KB，28/28 turns 全有诊断内容。File 1：3.6MB → 1.0MB，327/328 turns。

**Pass 3 — Subagent 提取 → experience/*.md**：5 个 parallel subagent 各管一个 stage（避免单 agent 聚焦 richest stage 忽略其他）。两轮提取：
- **Round 1（06-29 session）**：11 条目 177 行（单 agent 提取）
- **Round 2（06-29 session, 5-agent parallel 重新提取）**：21 条目 488 行，质量显著提升（+91% 条目，+176% 行数）。关键提升：hysteresis 从 3 条→6 条（新增 amp_blend 缺失、4/8-state 接口不兼容），final_interpret 从 3 条→6 条（新增 quality gate 收敛陷阱、硬编码路径）
- **Round 3（05-22 session, 5-agent parallel 增量提取）**：新增 principle insights 到各文件，仅追加 `Additional Evidence` 子条目防重复

### 交叉验证 vs 代码库

对 5 个 experience/*.md 逐条核实（4 个 explore agent parallel）：
- ❌ **删除 1 条**：qsolve(y) 在 y=80 硬截断——代码中从未存在（`-80` 是弱反型下界 clip，非上界）
- ❌ **删除 4 条过时**：`--skip-post-train-cv-polish` 已实现、v90 三个默认值已恢复、I-t 硬编码文件名（glob 回退已存在）、列 T 不识别（已修复）
- ✏️ **修正数字**：CV passivity scan 48/96→80/160/1200 点，无双阶段粗/细架构
- 📝 **标注 7 条**：HSPICE 参考模型（已切到符号化 KAN）、硬编码路径（源码零匹配）、amp_blend（已恢复）、4/8-state（自动检测存在）、amp 元数据 dict（部分缓解）、残差安全扫描 ext_bad（主扫描已修 CV 子扫描仍是绝对零）、checkpoint-diagnostic-only（读端已级联回退 写端仍 v92）

### 可操作性重构

**核心原则**：Agent B/C/D 的经验必须是他们能用工具影响的。

- **Agent 能力声明**：每个 experience 文件开头添加 "Agent X tools"、"Can influence"、"Cannot change" 声明
- **非可操作条目精简**：引擎代码级内容压缩为"诊断信号 + 正确工具选择"（如 contact resistance jitter → `accept_quality` 或 `fallback_to_previous`）
- **引擎架构知识搬家**：static_model 的 Derivative Observables/Physical Kernel Completeness/CV Charge Model + hysteresis 的 KAN Additive/Multiplicative → 移到 `final_interpret_experience.md` § Engine Architecture Diagnostics（共 8 条）
- **Agent E 详实**：最终 14 条目（6 个 pipeline 级 + 8 个引擎结构诊断），上下文余量最大，可包含从 B/C/D 角度看到的"引擎层面结构性缺陷"
- **交叉引用清理**：批量删除 diagnostic.md 行号（中间文件 gitignored 不存在）、跨 agent "See Agent E" 指针（agent 隔离）、代码文件行号（版本漂移）。全量文件自包含化

### Loader 升级

`src/workflow/dg_tft/knowledge/loader.py` v1.1：
- 新增 `_STAGE_EXPERIENCE_MAP`：5 stage → `knowledge/experience/*_experience.md`
- 新增 `load_experience(stage)`：从 `knowledge/experience/` 子目录读取
- `build_agent_prompt()` 三 block 顺序：`USER PREFERENCES` → `EXPERT KNOWLEDGE（code-derived）` → `EXPERIENCE（practical patterns）`
- 测试 7 条新增（T15-T21）：load_experience 全阶段 + experience 注入 + 顺序验证。21/21 pass，963 total pass（无回归）

### Git Commits

```
caaa11a feat: extract expert knowledge from 06-29 Codex session into knowledge/experience/*.md (5 agents parallel, 21 entries)
85b9e08 fix: cross-verify experience/*.md vs codebase — remove 1 wrong claim, remove 4 outdated, fix numbers, annotate 7
2125d38 refactor: restructure experience/*.md for agent actionability — add capability prefaces, trim non-actionable
5374f4d refactor: move engine architecture diagnostics from B/C/D to Agent E — keep B/C/D actionability-focused
4977526 fix: remove all fragile cross-references — diagnostic line numbers, cross-agent pointers, code file:line
280ff5e feat: add load_experience() + _STAGE_EXPERIENCE_MAP to loader — EXPERIENCE block in build_agent_prompt()
```

### 维护约定

- 改 `param_defs` 或引擎 argparse → 同步更新 `knowledge/*.md` + 检查 `knowledge/experience/*.md` 有无过时条目
- 新 Codex 会话 JSONL → 用 Pass 1-3 pipeline 提取，5-agent parallel 写入
- experience 中过时条目（问题已被引擎修复）**直接删除**，不保留。`[REMOVED]`/`[OUTDATED]`/`[FIXED]` 仅作过渡标记。

---

## 2026-07-07 — Agent A-E 全线完成 + 知识注入统合

### Step 15: Agent preparer + verifier

- 新增 `VERIFIER_AGENT_LABEL`，精炼 `preparer` 类型
- Preparer: `stop_mode="schema"` + `submit_handoff` (结构化交接)
- Verifier: `stop_mode="schema"` + `report_verification` (结构化检查)
- Handoff 传递链: Preparer → bidirection → op_log → prompt injection
- Fallback 动态插入: `verifier`/`check-data` 入口下 Verifier 失败→插入 DATA_PREP
- Agent A: 专用的 `diagnostic` 类型，工具 `inspect_xlsx` + `fallback_to_previous`
- Agent A → Verifier fallback: 接收 Prep handoff + Verifier report + Agent B fallback 信息

### Step 16-18: Agent B/C/D — 建模 Agent 统合

- 共用 `modeling` 类型，差异化由 per-stage schema + prompt 驱动
- `adjust_params` 内联 handler: tool-call 层验证，空参数直接返回失败 → LLM 同会话重试
- Agent A-E 全线注入专家知识（`build_agent_prompt` 调 `knowledge/*.md` + `experience/*.md`）
- 注入顺序: EXPERT KNOWLEDGE → EXPERIENCE → 任务上下文
- Fallback_ctx 传播: 上游指标、下游回退信息、本阶段历史 attempt 全部注入
- Bugfix: `_stage_done` 清 `fallback_ctx`（上游重跑成功后需消费回退上下文）
- Bugfix: retry 循环退出后 `fallback_ctx` 丢失 → 提前检查 + 链耗尽时 `_finalize_run`

### Step 19: Agent E — 最终解读

- 专用 `interpreter` 类型，`stop_mode="any"`
- 全量 attempt 历史注入 prompt（非仅最后一条）
- 输出 `FINAL_INTERPRETATION.md` 落盘（不再 15 行控制台截断）
- `contribute_knowledge` 可选：结构化模板 + YAML frontmatter (`title`/`summary`)
- `load_discovered_summaries()`: 启动扫描 `_discovered/*.md` YAML header → 摘要注入 prompt
- `dg_tft_report` 返回 `FINAL_INTERPRETATION.md` 路径
- `_spawn_agent_e` 后置到 `_finalize_run` 之后（报告先就绪）

### 全量改动统计

- `constants.py`: +3 常量 (`VERIFIER_AGENT_LABEL`, `DIAGNOSTIC_AGENT_LABEL`, `MODELING_AGENT_LABEL`, `INTERPRETER_AGENT_LABEL`, `FINAL_INTERPRETATION_FILENAME`)
- `progress.py`: +80 行（verifier/diagnostic/modeling/interpreter 四个类型的完整定义）
- `subagent_tools.py`: `submit_handoff` schema + `_AGENT_TYPE_CONFIG` (5 agent types) + handler 改动
- `subagent.py`: `__init__` 中 type 自动解析 + handler 成功时 stop_schema 合并
- `orchestrator.py`: ~250 行重写/新增（Agent A-E spawn 全链路、fallback_ctx 传播、handoff 注入）
- `data_preparer.py`: ~60 行改动（Preparer handoff 结构化、Verifier type 切换）
- `loader.py`: `load_discovered_summaries()` 实现（YAML header 扫描）
- `tool_def.py`: `dg_tft_report` 增加 `FINAL_INTERPRETATION.md` 返回
- `data_inspection.md`: 工具清单 + 三种入口说明 + Agent A prompt 内容更新
- `final_interpret.md`: +185 行 Stage Principles for Interpretation

### Git Commits

```
6efed53 feat: Agent A-E全线完成 — 知识注入统一, fallback_ctx传播, Agent E全量解读, 动态阶段插入, contribute_knowledge YAML结构
```

---

## 2026-07-08 — Agent 流程清理、minimal_run 合理化、Agent 流测试

### Phase 6 遗留清理

确认 Diagnosis subagent（`diagnostics.py` + `subagent_hooks.py` + `_run_stage_diagnosis` + `_launch_diagnosis_subagent`）是 Phase 6 (2026-06-26) 的遗留产物，已被 Agent A+E 完全取代。`agent_configs.json` 中 `DG_TFT_ENABLE_DIAGNOSIS = false` 默认关闭但代码仍在。

**执行全量删除**：
- 删除 `src/workflow/dg_tft/subagent_hooks.py`（315 行，6 个 hook 定义 + prompt 模板）
- 删除 `src/workflow/dg_tft/diagnostics.py`（294 行，ErrorCategory 关键字分类引擎 + DiagnosticResult）
- 删除 `test/dg_tft_diagnostics_test.py`（~994 行，85+ 测试）
- `orchestrator.py`：移除 `_run_stage_diagnosis`、`_launch_diagnosis_subagent`、`_collect_log_snippet`、`_print_diagnosis` 四个函数（~180 行），移除 import、常量、注释
- `report_builder.py`：移除 Diagnosis section（~25 行）及 `has_diag` 条件分支
- `agent_configs.json`：移除 `DG_TFT_ENABLE_DIAGNOSIS`、`DG_TFT_DIAGNOSIS_MAX_STEPS`、`DG_TFT_DIAGNOSIS_TIMEOUT_S`
- `TODO.md`：基线描述更新，Agent 介入点从 3 个改为 Agent A-E 全链路
- `dg_tft_report_test.py`：1 个 diagnosis 测试改写为 quality+warnings 测试

**Agent 矩阵精简为 7 个（完整对齐设计）**：Preparer / Verifier / Agent A / B / C / D / E

### minimal_run 行为合理化

**原行为**：Agent E 显式跳过，Agent A/B/C/D 仅在 quality fail 时跳过（旁路 quality gate），hard failure 时仍触发。

**新行为**（用户要求）：
- Agent A/B/C/D：**hard failure + quality fail 全部跳过**。L405（Agent A）+ L436（Agent B/C/D）加 `not _is_minimal_run` guard
- Agent E：**永不跳过**。删除 `_is_minimal_run` return 语句
- Agent E prompt 条件注入 minimal_run 标记：`"minimal_run mode: quick diagnostics only — ... do NOT critique model quality as if this were a full production run."`
- 注入方式：`_spawn_agent_e_inner` 签名 +`config_path`，L1698-1703 条件追加，非 minimal_run 下零 token 开销

**最终 minimal_run Agent 矩阵**：

| Agent | 运行？ | 原因 |
|-------|--------|------|
| Preparer | ✅ | 无 minimal_run 检查 |
| Verifier | ✅ | 同上 |
| Agent A | ❌ | hard failure 分支 + not _is_minimal_run |
| Agent B/C/D (hard failure) | ❌ | 同上 |
| Agent B/C/D (quality fail) | ❌ | quality gate 旁路 |
| Agent E | ✅ | 永不被跳过，prompt 含 minimal_run 标记 |

### Bug 修复

**Agent A prompt 缺失 `output_dir`**：L775 新增 `f"Output directory: {output_dir}"`，之前 `output_dir` 在函数签名但从未注入 prompt。

**Agent E prompt 构建语法错误**：`prompt_lines.extend([...]` 后缺 `])` 关闭列表和调用，L1737 修复。

### 知识文件同步

`knowledge/final_interpret.md` 全面对齐当前代码：
- §1 "minimal_run always runs Agent E" 更新：移除 "no longer skipped" 历史参照，改为纯陈述 "When minimal_run is active, prompt includes a flag..."
- §3 删除：移除引用已删除的 `_run_stage_diagnosis` / `DiagnosticResult` 的整段
- prompt summary 描述更新：从 "LAST attempt only" + "quality_pass 缺失" → "ALL attempts with quality_pass"
- 所有过时声明清除，无 "previously"/"used to"/"no longer" 等历史参照

### Agent 流测试套件

**新建** `test/dg_tft_agent_flow_test.py`（36 tests，908 行）：

| 测试类 | Tests | 覆盖 |
|--------|-------|------|
| TestNormalizeToolName | 7 | adjust_params / fallback / accept / 启发式 / error 解析 |
| TestProcessAgentDecision | 4 | 四种决策返回 + op_log 记录 |
| TestAgentA | 5 | fallback 决策、拒绝、错误降级、Prep handoff 接收、fallback_ctx 注入 |
| TestAgentE | 3 | 成功/失败 FINAL_INTERPRETATION.md 落盘、minimal_run hint |
| TestOrchestratorAgentAFlow | 2 | run_workflow 全路径：fallback→Verifier 终止、拒绝 fallback |
| TestOrchestratorModelingAgentFlow | 3 | Agent B hard fail→adjust→retry、quality fail→accept、B fallback→A 拒绝 |
| TestOrchestratorTwoStageFallback | 1 | D fallback→C adjust→恢复 成功 |
| TestOrchestratorMinimalRun | 2 | ABCD 跳过+E 运行、Agent A 不 spawn |
| TestRunAgentStage | 4 | Prep→Verifier 成功、Verifier fail 返回 False、fallback_ctx 注入 handoff、has_modeling_agent |
| TestOperationLogAgentEntries | 2 | adjust→agent_adjust 记录、fallback→fallback_sent 记录 |

**测试方法**：三层 mock 隔离 — SubAgent patch（mock SubAgent 构造，无 LLM 调用）→ run_stage patch（mock StageResult，无子进程）→ _assess_stage_quality patch（控制 quality gate）。全部使用 `tempfile.mkdtemp()` + `tearDown rmtree()`，`_save_runs = lambda: None` 阻止 WorkflowManager 落盘。911 全量测试通过，session/ 无残留。

### Git Commits

```
9821d3c feat: Agent flow test suite (36 tests), Phase 6 legacy cleanup, minimal_run rationalization, Agent E awareness, knowledge sync
```

---

## 2026-07-08 — Preparer/Verifier Prompt 重构 + Pipeline Bug 修复

### 三个生产 Bug 修复

**Bug 1 — Preparer 空转三次**：`WORKER_PROMPT_TEMPLATE` 含 `{report_name}` 占位符但 `.format()` 未传该参数 → `KeyError: 'report_name'` → SubAgent 从未启动。operation_log 中 `error: "'report_name'"` + `duration_s: 0.0` 三次完全相同。修复：`.format()` 加 `report_name=_PREP_WORKER_REPORT_NAME`。

**Bug 2 — `'dict' object has no attribute 'status'` 崩溃**：`_revert_tasks` 中 `task.status` 点号访问 TypedDict（运行时擦除为 plain dict）。Scoreboard `get_task()` 返回 dict，但 code 用了 attribute access。四个 fallback 路径全部受影响：Agent A fallback、Agent B/C/D hard-failure fallback、quality-fail fallback、Verifier→Preparer 重置。修复：`task.status` → `task.get("status")`。

**Bug 3 — Preparer 耗尽后无效跌落**：旧逻辑 `pointer += 1` 跌落到 Verifier → Verifier 在空目录上跑无意义 LLM 调用 → 回来看到 Preparer 重试计数器已耗尽 → 来回震荡直到 Verifier 也耗尽。修复：Preparer 耗尽直接 `_finalize_run` + `return False` 终止管线。

### 两个设计缺陷修复

**缺陷 1 — 4 条 early-return 路径不执行 Agent E**：超时 (`L236`)、Preparer 耗尽 (`L311`)、Verifier 耗尽 (`L321`)、fallback 链耗尽 (`L564`) 四条路径直接 return，不调 `_spawn_agent_e`。修复：每条 return 前追 `_spawn_agent_e(...)`。

**缺陷 2 — Verifier 失败重置到 DATA_PREP 时 retry_counts 不清零**：`retry_counts["data-prep_retries"]` 保留旧值导致 Preparer 无法获得新鲜重试机会。修复：`retry_counts.pop("data-prep_retries", None)`。

### Preparer Prompt 重构 — 引擎格式契约驱动

**核心原则**：Preparer 对流进来的数据格式不做假设（文件名、布局均无硬编码），但输出必须严格满足引擎硬要求。

**新增 `Engine requirements your output MUST meet` 区块**：明确列出引擎对三类数据的精确格式契约（从引擎源码直接提取）：

| 数据 | 契约来源 | 关键要求 |
|------|---------|---------|
| IV | `fit_physical_kernel.py:248-286` | 文件名 `IDVBG*/IDVTG*/IDVD*_train.xlsx`、sheet=`long`（大小写敏感）、4 列不可缺 |
| CV | `fit_physical_kernel.py:322-382` | 固定名 `CV_train.xlsx`、Row0=描述 Row1=列头 Row2+=数据（`header=None, iloc[1], iloc[2:]` 硬编码） |
| I-t | `analyze_it_hysteresis.py:32-41` | 8 个精确文件名，VBG/VTG family 分组 + temperature 标记 |

**CV 处理重写**：不再猜源文件叫 `CV_wide_tables.xlsx`。`glob_file` 按关键词找 CV 文件 → `inspect_xlsx` 查 Row 0/1 布局 → 如 Row 0 是列头（非描述文本）→ `openpyxl.insert_rows(1)` 插入空白行 → `wb.save` 到 `CV_train.xlsx`。不改数据值，只调行位。

**I-t 处理重写**：盲 `copytree` → `glob_file` 列出文件 → `inspect_xlsx` 读列头 → 从 `vbg`/`vtg`/`temp` 关键词推断 family 和 role → 按 VBG1/2/3, VTG1/2/3, *_temperature 分组排序编号 → `shutil.copy` 重命名为引擎 8 个期望名。无法分类时 default 到 VTG + 记录 potential_issues。

**精度规则强化**：新增"CV/I-t 布局修复时用 `openpyxl.insert_rows()` — 禁止 pandas 重序列化"、"CV/I-t 数据值必须原样通过"。

### Verifier Prompt 重构 — 与 Preparer 共享引擎契约

新增 `Engine requirements this data MUST satisfy` 区块，与 Preparer 保持一致：

- IV：新增文件名前缀检查（`IDVBG*` vs 任意 `*_train.xlsx`）
- CV：**新增 Row 0/1 布局检查** — 检测 Row 0 是否是列头而非描述，如是则标为"无声损坏"（引擎偏移一行读取）
- I-t：**从"数文件数"变为"8 个精确文件名对照检查"** — 列出文件名，报告缺失和命名错误

### test_subagent_unit.py 残留目录修复

`session/test-unit-001` 在测试运行后残留。根因：清理代码在 `if __name__ == "__main__"` 块内，pytest 不执行该块。修复：新增 `atexit.register(_cleanup_test_unit)` 在模块级注册，进程退出时无论何种 runner 均清理。

### Git Commits

```
(待提交)
```
