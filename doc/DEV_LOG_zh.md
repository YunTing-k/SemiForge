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

### 2026-07-01 | 参数试次记录 (3.6)

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

## 2026-07-01 | 数据输入阶段 (TODO 2.5.1–2.5.11) — Version 1.2.0

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

## 2026-07-01 | TECoSimAgent v0.2.6 同步升级 + 数据准备修复

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
当前 864 tests 全部通过（12 skipped）。运行命令：`python -m unittest discover -s test -p "*_test.py"`。

### Agent 介入边界
LLM/SubAgent 仅介入三处：数据整理、格式校验、失败诊断。训练执行、配置生成、质量判定、报告生成为纯规则代码。
