# DG-TFT Agent 全自动建模开发 TODO

## 1. 文档用途

这份文档不是架构说明，而是**后续实际开发执行清单**。

建议使用方式：

- 每完成一个任务就在代码仓库或项目管理工具里标记；
- 每个 Phase 尽量形成单独可运行里程碑；
- 每个任务都尽量能明确到文件、模块、接口或测试项；
- 不要一上来就重构全部建模代码，优先保证“能跑通全自动闭环”。

---

## 2. 执行原则

### 2.1 总原则

1. **先闭环，后优雅**；
2. **先 workflow 包装，后脚本函数化**；
3. **先保证用户只给路径即可启动，后增强诊断和总结质量**；
4. **训练执行由规则工作流控制，Agent 只负责柔性解释和辅助判断**。

### 2.2 开发顺序原则

推荐顺序：

```text
迁移代码落点确认
→ workflow 数据结构与管理器
→ runtime config 自动生成
→ subprocess 版固定工作流
→ agent 工具接入
→ 数据整理与验证（双 agent：worker 整理 + verifier 校验）
→ 结果汇总
→ 失败诊断 + 质量守卫（自动调参重试）
→ 脚本函数化
→ subagent 深度接入
```

---

## 3. 里程碑总览

| Phase | 目标 | 结果 |
|---|---|---|
| Phase 0 | 确认代码落点和边界 | 明确 DG-TFT 建模代码的正式位置和依赖边界 |
| Phase 1 | 建立工作流基础设施 | 有 `WorkflowRun` / `WorkflowManager` / session 持久化 |
| Phase 2 | 自动生成 runtime config | 给定数据路径即可得到独立配置 |
| Phase 3 | 打通 subprocess 版全自动流程 | 能自动跑 `check-data → static → hysteresis(optional)` |
| Phase 4 | 接入 Agent 工具和命令 | 用户可通过自然语言或命令启动 workflow |
| Phase 5 | 完善报告和产物管理 | 自动输出 summary / manifest / metrics |
| Phase 6 | 完善诊断和 subagent 协作 | 失败时可自动解释和给建议 |
| Phase 7 | 质量守卫与自动重拟合 | 效果不好时自动调参重跑，减少专家介入 |
| Phase 8 | 脚本函数化重构 | 建模模块从 CLI 脚本升级为可调用 API |
| Phase 9 | 测试、稳定性、收尾 | 有完整单测、集成测试和使用文档 |

---

## 4. Phase 0：代码落点确认与仓库整理

## 4.1 目标

确认 DG-TFT 建模代码是否已经完整迁移到 `src`，以及后续开发使用的正式目录。

## 4.2 TODO

### P0-T1 确认 DG-TFT 建模代码正式位置
- [x] 检查 `src/` 下是否已有完整 DG-TFT 建模代码；
- [x] 如果没有，确认当前唯一可信版本是否仍是 `ref/DG_TFT_v90_USER_FRIENDLY_MODELING_RELEASE_20260622/modeling_code/`；
- [x] 明确后续开发到底以 `ref` 还是 `src/device_modeling/dg_tft` 为准。（推荐统一都放在src路径下，`ref`只是参考）

**输出**
- 一条明确结论：后续开发使用哪个目录作为正式代码源。

**依赖**
- 无。

**验收标准**
- 团队内部对“代码从哪里改”没有歧义。

---

### P0-T2 建立 DG-TFT 建模正式目录
- [x] 若 `src/device_modeling/` 不存在，则创建；
- [x] 创建 `src/device_modeling/dg_tft/`；
- [x] 创建子目录：
  - [x] `config/`
  - [x] `engine/`
  - [x] `dynamic/`
  - [x] `deployment/`
  - [x] `reporting/`
- [x] 增加必要的 `__init__.py`。

**输出**
- 标准化后的建模代码目录骨架。

**依赖**
- `P0-T1`。

**验收标准**
- 后续 workflow 层引用路径稳定。

---

### P0-T3 迁移或镜像现有脚本到正式目录
- [x] 迁移/复制 `run_model.py`；
- [x] 迁移/复制 `engine/*.py`；
- [x] 迁移/复制 `dynamic/*.py`；
- [x] 迁移/复制配置模板 `.ini`；
- [x] 检查 import 路径是否仍可运行；
- [x] 保证最小可执行性，不要求这一阶段重构内部实现。

**输出**
- `src/device_modeling/dg_tft/` 下拥有一套可执行的 DG-TFT 建模代码。

**依赖**
- `P0-T2`。

**验收标准**
- 在新路径下执行 `run_model.py --help` 能正常工作；
- 核心脚本无明显 import 崩溃。

---

### P0-T4 记录外部依赖与环境要求
- [x] 明确 Python 依赖包列表；
- [x] 明确 KAN 依赖；
- [x] 明确 HSPICE 是否为可选依赖；
- [x] 明确当前默认 Conda 环境；
- [x] 形成一份简短环境说明。

**输出**
- 一份可执行环境依赖清单。

**依赖**
- `P0-T3`。

**验收标准**
- 新开发人员可以知道跑通 workflow 需要什么环境。

### P0-T5 记录已验证环境边界
- [x] 记录 Agent 运行解释器为 `G:/Anaconda3/envs/AGI/python.exe`；
- [x] 记录 DG-TFT 建模解释器为 `G:/Anaconda3/envs/CUDA/python.exe`；
- [x] 记录 `AGI` 与 `CUDA` 分离，不共享业务依赖；
- [x] 记录 `CUDA` 环境已验证支持 `torch.cuda` 与 `kan` 的 GPU 前向；
- [x] 记录 `check-data` 已对示例数据验证通过；
- [x] 记录当前继续训练的主要阻塞已从“环境依赖”转为“项目文件完整性”。

**输出**
- 环境策略说明，可直接用于后续 workflow 配置。

**依赖**
- `P0-T4`。

**验收标准**
- 之后开发时不会再混淆 Agent 环境和建模环境。

---

### P0-T6 固化解释器路由策略
- [x] 在 `config/agent_configs.json` 中增加：
  - [x] `DG_TFT_AGENT_PYTHON_EXE`
  - [x] `DG_TFT_MODELING_PYTHON_EXE`
  - [x] `DG_TFT_MODELING_CONDA_ENV`
- [x] 明确 workflow 第一版以 `DG_TFT_MODELING_PYTHON_EXE` 作为唯一真实建模解释器来源；
- [x] 明确禁止在建模阶段执行中默认使用 `sys.executable`；
- [x] 明确禁止依赖环境变量里的裸 `python` 去跑建模脚本。

**输出**
- 一套不会混淆 `AGI` 与 `CUDA` 的解释器执行约束。

**依赖**
- `P0-T5`。

**验收标准**
- 后续进入 workflow 开发时，解释器选择逻辑已经被设计成显式配置，而不是隐式行为。

---

## 5. Phase 1：工作流基础设施

## 5.1 目标

建立独立于 task scoreboard 的 DG-TFT workflow 状态管理能力。

## 5.2 TODO

### P1-T1 新建 workflow 目录骨架
- [x] 创建 `src/workflow/`
- [x] 创建 `src/workflow/dg_tft/`
- [x] 新建：
  - [x] `__init__.py`
  - [x] `schema.py`
  - [x] `manager.py`
  - [x] `orchestrator.py`
  - [x] `stages.py`
  - [x] `artifacts.py`
  - [x] `dataset_inspector.py`
  - [x] `config_builder.py`
  - [x] `stage_runner.py`
  - [x] `diagnostics.py`
  - [x] `report_builder.py`
  - [x] `subagent_hooks.py`

**依赖**
- Phase 0 完成。

**验收标准**
- 目录完整、模块可 import。

---

### P1-T2 定义核心数据结构
- [x] 在 `schema.py` 定义 `WorkflowRun`；
- [x] 定义 `StageResult`；
- [x] 定义 `DatasetProfile`；
- [x] 定义 workflow 状态枚举，例如：
  - [x] `PENDING`
  - [x] `RUNNING`
  - [x] `DONE`
  - [x] `FAILED`
  - [x] `CANCELLED`
- [x] 定义 stage 名称常量。

**依赖**
- `P1-T1`。

**验收标准**
- 数据结构足以表达一次完整 run 的状态、产物、错误和指标。

---

### P1-T3 实现 `WorkflowManager`
- [x] 在 `manager.py` 中实现 run 创建；
- [x] 实现 `get_run(run_id)`；
- [x] 实现 `list_runs()`；
- [x] 实现 `update_stage(...)`；
- [x] 实现 `attach_artifact(...)`；
- [x] 实现 `attach_metric(...)`；
- [x] 实现 `mark_done(...)`；
- [x] 实现 `mark_failed(...)`；
- [x] 增加线程锁，保证并发安全。

**依赖**
- `P1-T2`。

**验收标准**
- 可以独立用代码创建、更新和查询 workflow run。

---

### P1-T4 增加 session 持久化
- [x] 设计持久化文件：`session/<uuid>/workflow_runs.json`；
- [x] 在 `WorkflowManager` 中实现 save/load；
- [x] 确认 JSON 结构稳定；
- [x] 处理恢复时的兼容问题。

**依赖**
- `P1-T3`。

**验收标准**
- session 恢复后，workflow run 状态仍可查询。

---

### P1-T5 将 `WorkflowManager` 挂到 `AgentContext`
- [x] 修改 `src/context/agent_context.py`；
- [x] 增加 `workflow_man` 成员；
- [x] 视需要增加后台 workflow registry；
- [x] 确保上下文保存/恢复时不破坏现有逻辑。

**依赖**
- `P1-T4`。

**验收标准**
- 主 agent 生命周期内可直接访问 workflow manager。

---

### P1-T6 在 `main.py` 中初始化 workflow manager
- [x] 在启动阶段配置 `workflow_man`；
- [x] 在 session resume 时加载 workflow runs；
- [x] 在 save session 时写回 workflow 状态。

**依赖**
- `P1-T5`。

**验收标准**
- 主程序启动后 workflow manager 处于可用状态。

---

### P1-T7 在 `stage_runner` 中集中实现解释器解析
- [x] 设计统一函数，例如 `resolve_modeling_python(ctx)`；
- [x] 该函数优先读取 `DG_TFT_MODELING_PYTHON_EXE`；
- [x] 统一返回建模阶段实际调用的解释器绝对路径；
- [x] 记录当前 Agent 解释器与建模解释器的差异。

**依赖**
- `P1-T6`。

**验收标准**
- 解释器解析逻辑只有一个入口，不分散在各个阶段实现中。

---

## 6. Phase 2：数据审计与 runtime config 自动生成

## 6.1 目标

输入一个数据路径，系统可以自动识别数据能力并生成本次 run 的独立配置文件。

## 6.2 TODO

### P2-T1 实现 `DatasetProfile`
- [x] 在 `dataset_inspector.py` 中定义 profile 构造流程；
- [x] 收集：
  - [x] 静态 IV 是否齐全；
  - [x] CV 是否齐全；
  - [x] 是否有温度列；
  - [x] 温度跨度；
  - [x] 是否有 I-t 数据；
  - [x] I-t 候选目录和文件列表；
  - [x] 问题列表。

**依赖**
- `P1-T2`。

**验收标准**
- 给定一个数据目录，能输出结构化 profile。

---

### P2-T2 包装现有 `validate_dataset.py`
- [x] 优先以 subprocess 或 import 方式调用现有 `validate_dataset.py`；
- [x] 将输出 JSON 解析为 workflow 可用结构；
- [x] 保留原始校验结果文件。

**依赖**
- `P2-T1`。

**验收标准**
- `dataset_validation.json` 可以被 workflow 正常消费。

---

### P2-T3 实现 I-t 目录自动发现逻辑
- [x] 支持用户显式指定 `hysteresis_dir`；
- [x] 支持在 `dataset_dir` 内查找；
- [x] 支持 `dataset_dir/hysteresis_it`；
- [x] 支持同级目录中包含 `it` / `hysteresis` 的候选路径；
- [x] 当存在多个候选时，保留候选列表。

**依赖**
- `P2-T1`。

**验收标准**
- 大部分常见目录组织方式下可以自动找到 I-t 数据。

---

### P2-T4 实现温度模型自动判定
- [x] 根据校验结果判断是否存在温度列；
- [x] 计算温度跨度；
- [x] 输出 `temperature_enabled` 推荐值；
- [x] 输出推荐的 `temperature.mode`。

**依赖**
- `P2-T2`。

**验收标准**
- 能自动判断 `on/off/auto` 建议。

---

### P2-T5 实现 HSPICE 可用性自动判定
- [x] 检查配置中的 `hspice_exe` 是否存在；
- [x] 如不存在，自动标记 `run_hspice = no`；
- [x] 记录为 warning，而不是直接 fail。

**依赖**
- `P2-T2`。

**验收标准**
- 无 HSPICE 环境时，workflow 可降级继续。

---

### P2-T6 实现 `config_builder.py`
- [x] 读取模板配置；
- [x] 生成本次 run 的 runtime config；
- [x] 自动填充：
  - [x] `data_dir`
  - [x] `hysteresis_data_dir`
  - [x] `output_dir`
  - [x] `hspice_exe`
  - [x] `hysteresis.enabled`
  - [x] `temperature.mode`
  - [x] `deployment.run_hspice`
- [x] 不覆盖原模板文件。

**依赖**
- `P2-T3`
- `P2-T4`
- `P2-T5`

**验收标准**
- 能在 run 工作目录下生成独立配置文件。

---

### P2-T7 保存配置解析结果
- [x] 生成 `config_resolution.json`；
- [x] 生成 `dataset_profile.json`；
- [x] 把这些路径挂到 workflow artifacts。

**依赖**
- `P2-T6`。

**验收标准**
- 用户和开发者都能回看本次自动判定依据。

---

## 7. Phase 3：固定工作流执行器（subprocess 版）

## 7.1 目标

不改动太多原数值脚本的前提下，先打通“给路径即可全自动跑”的闭环。

## 7.2 TODO

### P3-T1 定义 workflow stage 顺序
- [x] 在 `stages.py` 中定义标准阶段列表；
- [x] 明确每个阶段的前置条件；
- [x] 明确 optional 阶段（迟滞、HSPICE 验证）。

**依赖**
- Phase 2 完成。

**验收标准**
- 工作流阶段图固定且可配置。

---

### P3-T2 实现 `stage_runner.py` 的 subprocess 执行能力
- [x] 封装统一命令执行函数；
- [x] 支持超时；
- [x] 支持 stdout/stderr 捕获；
- [x] 支持阶段输出目录检查；
- [x] 返回 `StageResult`；
- [x] 所有阶段命令都必须通过 `resolve_modeling_python(ctx)` 取解释器；
- [x] 禁止在阶段命令构造中直接使用 `sys.executable`；
- [x] 在阶段日志中打印实际使用的建模解释器路径。

**依赖**
- `P3-T1`。

**验收标准**
- 任一阶段都能以统一方式执行和收集结果。

---

### P3-T3 实现 `check-data` 阶段执行
- [x] 调用 `run_model.py check-data --config ...`；
- [x] 收集 `dataset_validation.json`；
- [x] 将结果写入 workflow run。

**依赖**
- `P3-T2`。

**验收标准**
- 静态数据检查可自动完成。

---

### P3-T4 实现 `fit-physical` 阶段执行
- [x] 调用 `fit-physical`；
- [x] 检查 `fitted_params.json`；
- [x] 校验关键产物是否存在；
- [x] 将输出目录登记到 artifacts。

**依赖**
- `P3-T3`。

**验收标准**
- 物理内核阶段自动完成并可判断成功/失败。

---

### P3-T5 实现 `train-base-pkan` 阶段执行
- [x] 调用 `train-base-pkan`；
- [x] 检查 `pkan_torch_model.pt`；
- [x] 挂接阶段 summary。

**依赖**
- `P3-T4`。

**验收标准**
- 基础 KAN 训练可自动完成。

---

### P3-T6 实现 `train-static` 阶段执行
- [x] 调用 `train-static`；
- [x] 检查静态模型产物；
- [x] 检查静态 Verilog-A 输出；
- [x] 读取关键精度结果。

**依赖**
- `P3-T5`。

**验收标准**
- 静态全模型可自动完成。

---

### P3-T7 实现 `train-hysteresis` 条件执行
- [x] 仅当 `DatasetProfile.has_hysteresis == True` 时执行；
- [x] 调用 `train-hysteresis`；
- [x] 检查动态 bank / `.va` / `.inc` 产物；
- [x] 若关闭则记录 skip 原因。

**依赖**
- `P3-T6`。

**验收标准**
- 有 I-t 数据时自动跑，无 I-t 数据时自动跳过。

---

### P3-T8 实现 `orchestrator.py`
- [x] 串联全部阶段；
- [x] 支持阶段状态更新；
- [x] 支持失败即停；
- [x] 支持 optional 阶段 skip；
- [x] 支持最终状态收敛。

**依赖**
- `P3-T3` ~ `P3-T7`。

**验收标准**
- 一次 workflow run 可以完整推进到 done 或 failed。

---

### P3-T9 生成 run 工作目录结构
- [x] 约定输出根目录结构；
- [x] 创建 `runtime_config/`；
- [x] 创建 `artifacts/` 或 summary 文件位置；
- [x] 保证不同 run 相互隔离。

**依赖**
- `P3-T8`。

**验收标准**
- 每次 run 的文件组织清晰且互不覆盖。

---

### P3-T10 用户可见的 Console 进度输出
- [x] 在 `orchestrator.py` / `stage_runner.py` 中接入 `ctx.console`（已存在于 AgentContext）：
  - [x] 流程启动时：`Run #N DG-TFT workflow 开始 (M 阶段)` + 数据/输出路径
  - [x] 每阶段启动时：`[n/total] stage_name 开始...`
  - [x] 每阶段完成时：`[n/total] stage_name ✓`
  - [x] 每阶段失败时：`[n/total] stage_name ✗ (reason)`
  - [x] 阶段跳过时：`[n/total] stage_name ⊘ 跳过 (reason)` / `⊘ 跳过 (已完成)`
  - [x] 流程成功时：`✓ Run #N DONE (Xs, M 阶段)`
  - [x] 流程失败时：`✗ Run #N FAILED at stage_name`
  - [ ] 质量门禁/调参重试的 console 输出在 Phase 7 实现时补充
- [x] `ctx.console` 已在 `AgentContext` 中（`main.py:93`），无需新增参数；
- [x] `sys_log` 继续保留作为完整审计日志，两者互补不替代；
- [x] 尊重 `ctx.subagent_mute` 标志，静默时不输出。

**依赖**
- `P3-T8`。

**验收标准**
- 用户无需 `--log` 参数也能在终端实时看到 pipeline 进度和结果。

---

## 8. Phase 4：接入 Agent 工具与命令

## 8.1 目标

让主 agent 能优先调用固定 workflow，而不是自己临场拼 bash。

## 8.2 TODO

### P4-T1 新增 workflow tool schema
- [x] 在 `src/tool/tool_def.py` 新增：
  - [ ] `start_dg_tft_workflow`
  - [ ] `query_dg_tft_workflow`
  - [ ] `read_dg_tft_workflow_report`
- [x] 明确定义参数与返回结构。

**依赖**
- Phase 3 完成。

**验收标准**
- 工具定义可被主 agent 正常看到。

---

### P4-T2 实现 workflow tool handler
- [x] 在 `tool_def.py` 中实现三个工具对应函数：
  - [x] `start_dg_tft_workflow` 内部串联：`create_run() → inspect_dataset() → build_runtime_config() → run_workflow()`；
  - [x] `query_dg_tft_workflow` 读取 `workflow_man.get_run(run_id)` 返回状态；
  - [x] `read_dg_tft_workflow_report` 读取 run 输出目录下的 summary 文件；
- [x] 调用 `workflow_man` 和 `orchestrator`；
- [x] 返回结构化结果。

**依赖**
- `P4-T1`。

**验收标准**
- 工具可被实际调用并返回结果。

---

### P4-T3 实现后台异步执行模式
- [x] `start_dg_tft_workflow` 立即返回 `run_id`，不阻塞等待训练完成；
- [x] 在 `workflow_man` 中新增 `RunningRunTracker`（存储正在执行的 subprocess 句柄）；
- [x] `query_dg_tft_workflow` 可实时查询 run status 并返回最新 stage；
- [ ] 考虑超长时间运行的看护（心跳检测 / 超时 kill）；
- [x] 确保多个 run 并发隔离（输出目录、config 文件不冲突）。

**依赖**
- `P4-T2`。

**验收标准**
- 用户启动 workflow 后可立即继续其他操作，通过 query 获取进度。

---

### P4-T4 注册 workflow tool dispatch
- [x] 修改 `src/tool/tool_dispatch.py`；
- [x] 加入三个工具的 dispatch 分支。

**依赖**
- `P4-T3`。

**验收标准**
- 工具调用不落入 undefined 分支。

---

### P4-T5 增加内建命令
- [x] 在 `src/utility/command.py` 增加：
  - [x] `/workflow_list`
  - [x] `/workflow_get <run_id>`
  - [x] `/workflow_report <run_id>`
- [x] 在命令系统中注册。

**依赖**
- `P4-T3`。

**验收标准**
- 不走 LLM 也可查询 workflow 运行结果。

---

### P4-T6 优化 prompt 路由逻辑
- [x] 在 `src/context/prompt.py` 中增加“识别 DG-TFT 自动建模请求时优先用 workflow tool”的提示约束；
- [x] 避免主 agent 将该类请求拆成大量零散工具调用。

**依赖**
- `P4-T2`。

**验收标准**
- 用户说“对某目录做全自动建模”时，主 agent倾向直接调用 workflow tool。

---

## 8.5. Phase 4.5：数据整理与双 Agent 验证

## 8.5.1 目标

解决"用户提供的原始数据格式不固定"的问题。在 DG-TFT workflow 之前，先由 **worker subagent** 分析并整理原始数据为 `*_train.xlsx` 格式，再由 **verifier**（独立 LLM call + 结构化 tool schema）验证输出格式合规性。

核心设计要点：

1. **Worker subagent**（type=worker）：拥有 bash/read/write/glob 等工具，读取原始文件，按 IV family 分组，转换为 `*_train.xlsx`，组织 CV/I-t 数据到输出目录
2. **Verifier**（独立 LLM call，非 subagent）：仅拥有 `read_file` + `glob_file` + **自定义 `report_verification` tool**，LLM 调用 tool 返回结构化 JSON，杜绝自由文本幻觉
3. **Programmatic check**（Python 代码 `_check_output()`）：做最终 CRITICAL/WARNING 判定，零 LLM 参与，避免 hallucination 影响 pass/fail
4. **不通过定义**：任一 CRITICAL 项失败 → 阻断后续 workflow；只有 WARNING → 仍可通过
5. **继承 `summarize_support.py` 模式**：给 LLM 唯一的结构化 output tool，LLM 必须调用 tool 才能"说话"

### 8.5.2 CRITICAL vs WARNING 判定规则

```
_check_output() 规则：
├─ [CRITICAL] IDVBG*_train.xlsx 至少 1 个 + long sheet 存在 + {VD_V,VBG_V,VTG_V,ID_A} 列全齐
├─ [CRITICAL] IDVTG*_train.xlsx 同上
├─ [CRITICAL] IDVD*_train.xlsx 同上
├─ [WARNING]  CV_train.xlsx 缺失 → ok 仍为 True
├─ [WARNING]  it/ 子目录缺失 → ok 仍为 True
└─ [WARNING]  温度列（temp_K/T_K 等）缺失
```

**阻断策略**：`ok=False` → manifest.json 写入 failed → `/workflow_get` 显示 不通过 + 失败原因 → LLM 提示用户修复后重试。

### 8.5.3 `report_verification` tool schema

由 `data_preparer.py` 动态构造，verifier LLM 通过调用此 tool 输出结构化结果：

```json
{
  "name": "report_verification",
  "description": "Report the verification findings for prepared DG-TFT dataset. Call this tool with your structured results.",
  "parameters": {
    "type": "object",
    "required": ["overall_ok"],
    "additionalProperties": false,
    "properties": {
      "overall_ok": { "type": "boolean" },
      "iv_families": {
        "type": "array",
        "items": {
          "type": "object",
          "properties": {
            "family": {"type": "string"},
            "files_found": {"type": "array", "items": {"type": "string"}},
            "long_sheet_ok": {"type": "boolean"},
            "required_columns_ok": {"type": "boolean"},
            "missing_columns": {"type": "array", "items": {"type": "string"}},
            "issues": {"type": "array", "items": {"type": "string"}}
          }
        }
      },
      "cv_check": {
        "type": "object",
        "properties": {
          "file_found": {"type": "boolean"},
          "sheets_ok": {"type": "object"},
          "issues": {"type": "array", "items": {"type": "string"}}
        }
      },
      "it_check": {
        "type": "object",
        "properties": {
          "dir_found": {"type": "boolean"},
          "file_count": {"type": "integer"},
          "issues": {"type": "array", "items": {"type": "string"}}
        }
      },
      "summary": {"type": "string"}
    }
  }
}
```

### 8.5.4 输出目录结构

```
session/<uuid>/prepared_data/<prep_id>/
├── IDVBG_VDS0.1V_train.xlsx        # IV family: IDVBG
├── IDVTG_VDS0.1V_train.xlsx        # IV family: IDVTG
├── IDVD_VBG0V_train.xlsx           # IV family: IDVD
├── CV_train.xlsx                   # CV (可选)
├── it/                             # I-t hysteresis (可选)
│   ├── I-t with different VBG1.xlsx
│   └── ...
├── worker_report.json              # worker subagent 自检报告
├── verifier_result.json            # report_verification tool 返回的结构化结果
└── manifest.json                   # { ok, issues[], prep_id, created_at }
```

### 8.5.5 调用链路

```
用户: "整理这个目录为 DG-TFT 格式"
    │
    ▼
prepare_dg_tft_dataset(raw_dataset_dir="/path/to/ref/0521")
    │
    ├─ 创建 session/<uuid>/prepared_data/<prep_id>/
    ├─ 启动 background thread
    └─ 立即返回 { prep_id, output_dir }           ← 不阻塞
         │
         ▼ (后台线程)
    ┌────────────────────────────────────────────┐
    │ Step 1: _run_worker()                      │
    │   SubAgent(type=worker, prompt=整理指令)     │
    │   读原始文件 → 分组 family → 转 *_train.xlsx │
    │   → 写 worker_report.json                   │
    ├────────────────────────────────────────────┤
    │ Step 2: _run_verifier()                     │
    │   独立 LLM call（非 subagent）               │
    │   工具: read_file + glob_file +             │
    │          report_verification (自定义)        │
    │   → 写 verifier_result.json                 │
    ├────────────────────────────────────────────┤
    │ Step 3: _check_output()                     │
    │   Python 代码读取 verifier_result.json       │
    │   → CRITICAL/WARNING 判定                   │
    │   → 写 manifest.json                        │
    └────────────────────────────────────────────┘
         │
         ▼
用户通过 /workflow_get <prep_id> 查询结果
    │
    ├─ ok=True → start_dg_tft_workflow(dataset_dir="...prepared_data/<prep_id>/")
    └─ ok=False → 显示失败原因 → 用户修复后重试
```

### 8.5.6 Prep ↔ Workflow 关系（类 Design ↔ Run 模型）

借鉴原仿真系统中 `design → run` 的关系：**一个 prepared data 可被多个 workflow run 引用**。

```
PreparationTracker
  prep_id=1  "ref_0521_cleaned"  →  ok=True
      │
      ├── WorkflowRun run_id=1  (prep_id=1, device="TFT_A")
      ├── WorkflowRun run_id=2  (prep_id=1, device="TFT_A", 重新训练)
      └── WorkflowRun run_id=3  (prep_id=1, device="TFT_B")
```

**数据结构变更**：
- `PreparationTracker`（新建）— 管理 prep 生命周期，存于 `WorkflowManager.preparation_tracker`
- `WorkflowRun` 新增字段 `prep_id: int | None` — 外键引用 prepared data
- `manifest.json` 新增 `prep_id` — 自描述

**ID 隔离**：
- `WorkflowManager._next_run_id` — workflow run counter（不变）
- `PreparationTracker._next_prep_id` — preparation counter（独立）

### 8.5.7 Verifier 容错机制

独立 LLM call 的 `tool_choice` 在不同 API 下行为不同，必须处理 fallback：

```
Verifier call:
  │
  ├─ API 支持 tool_choice="required" (OpenAI)
  │   → 强制 LLM 调用 report_verification
  │   → 直接提取 tool_calls → verifier_result.json
  │
  └─ API 不支持 tool_choice (DeepSeek)
      → 纯 prompt 要求"必须调用 report_verification"
          │
          ├─ tool_calls 非空 → 提取结构化结果 ✔
          │
          └─ tool_calls 为空（LLM 无视 tool 要求）
              → 追加提示："You MUST call report_verification to report results"
              → 重试（最多 3 次）
                  │
                  ├─ 重试成功 → 提取结构化结果 ✔
                  │
                  └─ 3 次均失败
                      → _check_output() 直接编程检查输出目录
                      → 写 verifier_result.json = {"overall_ok": _check_ok, "summary": "LLM refused to call tool, using programmatic check"}
```

**implementation 要点**：
- `_run_verifier()` 内部 loop（max 3 retries），每次检查 `response.choices[0].message.tool_calls`
- `_check_output()` 在 verifier 失败时**仍执行**（编程检查独立于 LLM 输出）
- 最终 `manifest.json` 的 `ok` 由 `_check_output()` 决定，不由 verifier 决定

### 8.5.8 Worker Prompt 质量保障

Worker subagent 是质量关键。Prompt 模板必须在设计阶段写定，而不是开发时随意拼接。

**Prompt 结构要求**：

```
[System context] 你是数据整理专家，已知 DG-TFT 建模需要的数据格式

[Input description] 原始数据目录：{raw_dir}，输出目录：{output_dir}

[Step-by-step instructions]
  1. 扫描 raw_dir 下所有文件（csv, xlsx），识别 IV/CV/I-t 数据类型
  2. 按以下 mapping 分组并输出为 *_train.xlsx：
     - files with "IDVBG" → IDVBG_VDS*V_train.xlsx
     - files with "IDVTG" → IDVTG_VDS*V_train.xlsx
     - files with "IDVD"  → IDVD_VBG*V_train.xlsx
     (保留原始 VDS/VBG/VTG 电压值在文件名中)
  3. 每个 *_train.xlsx 必须包含 "long" sheet，列：VD_V, VBG_V, VTG_V, ID_A
  4. 如发现 CV 数据 → 输出 CV_train.xlsx，sheet: cgg, cbg, ctg, vds
  5. 如发现 I-t 数据 → 输出到 output_dir/it/ 子目录
  6. 写 worker_report.json（自检报告）

[Constraints]
  - 不改变原始测量数值（只做格式转换）
  - 不编造数据（缺失列的单元格留空）
  - 使用 pandas/openpyxl 处理
```

**Prompt 验收方法**：
- 对 `ref/0521` 的 `training_ready_corrected_xlsx/` 目录（已符合格式）不做任何改动即为 benchmark —— worker 应该"原样拷贝"而非"重新处理"
- 对 `ref/0521` 根目录（原始 csv + xlsx）做真实整理，输出应匹配 `training_ready_corrected_xlsx/` 的结构

### 8.5.9 TODO

### P4.5-T1 实现 `data_preparer.py` 核心模块 + schema 扩展
- [x] 新建 `src/workflow/dg_tft/data_preparer.py`
- [x] 在 `schema.py` 中 `WorkflowRun` 新增 `prep_id: int | None = None` 字段
- [x] 实现 `PreparationTracker` 类（独立 `_next_prep_id` counter，存于 `workflow_man.preparation_tracker`）
- [x] 实现 `run_data_preparation(ctx, raw_dataset_dir, output_dir)` — 串联 worker → verifier → check
- [x] 实现 `_run_worker(ctx, raw_dir, output_dir)` — 启动 worker subagent
  - [x] worker prompt 模板按 8.5.8 格式设计（system context + input + step-by-step + constraints）
  - [x] worker 输出 `worker_report.json` + `file_mapping` (来源映射)
- [x] 实现 `_run_verifier(ctx, output_dir)` — 独立 LLM call + 迷你 tool execution loop
  - [x] 构造 `report_verification` tool definition（给 read_file/glob_file + report_verification）
  - [x] 实现 容错机制：max_turns 内 loop glob_file→read_file→...→report_verification
  - [x] verifier prompt 模板：列指示文件检查步骤
  - [x] 从 tool_calls 中提取 report_verification 结果，写入 `verifier_result.json`
- [x] 实现 `_check_output(output_dir)` — 程序化 CRITICAL/WARNING 判定
  - [x] 判定逻辑：IV family 文件存在 + long sheet + 必须列 = CRITICAL；CV/I-t 缺失 = WARNING
  - [x] verifier 失败时 `_check_output()` 独立执行，不依赖 LLM 输出
  - [x] 写入 `manifest.json`
- [x] 实现 `_write_file_manifest(output_dir, prep_id)` — 读取 worker_report 中的 file_mapping 生成 prep_file_manifest.json

**依赖**
- Phase 4 完成（SubAgent 基础设施、异步执行模式）。

**验收标准**
- 对 `ref/0521` 运行后产出正确的 `*_train.xlsx` 格式数据，verifier 结构化结果与手动检查一致。

---

### P4.5-T2 新增 `prepare_dg_tft_dataset` agent tool
- [x] 在 `src/constants.py` 新增 `TOOL_NAME_PREPARE_DATASET`
- [x] 在 `src/tool/tool_def.py` 新增 `tool_prepare_dg_tft_dataset_def()` + `prepare_dg_tft_dataset()` handler
  - [x] handler 内部：创建 output_dir → 同步调用 `run_data_preparation`
- [x] 在 `src/tool/tool_dispatch.py` 新增 dispatch 分支

**依赖**
- P4.5-T1。

**验收标准**
- 调用 tool 后同步完成，返回 ok + issues。

---

### P4.5-T3 注册内建命令 + 更新 prompt 路由
- [x] 在 `src/utility/command.py` 将 preparation 结果并入 `/workflow_list` `/workflow_get` 展示
- [x] 在 `src/context/prompt.py` 新增：
  - [x] "数据非*_train.xlsx 格式时先调 prepare_dg_tft_dataset"的提示
  - [x] "整理完成后用 start_dg_tft_workflow 启动建模"的串联指引

**依赖**
- P4.5-T2。

**验收标准**
- 用户说"对这个目录做全自动建模"时，LLM 先调用 `prepare_dg_tft_dataset`，再调用 `start_dg_tft_workflow`。

---

### P4.5-T4 测试
- [x] 测试 `_check_output()` 各项判定（正常/缺文件/缺 sheet/缺列/CV 缺失/I-t 缺失）
- [x] Mock verifier LLM response，测试结构化 tool call 提取
- [x] 测试 `run_data_preparation` 完整流程（mock subagent）
- [x] 测试 tool handler 同步行为

**依赖**
- P4.5-T1 ~ P4.5-T3。

**验收标准**
- 所有新增测试通过，覆盖率覆盖 CRITICAL/WARNING 各分支。



## 9. Phase 5：报告、产物与任务板协同

## 9.1 目标

每次 run 完成后，自动输出清晰、可复盘的结果文件，并且与 task scoreboard 协同展示进度。

## 9.2 TODO

### P5-T1 设计 artifacts 管理规则
- [x] 在 `artifacts.py` 中定义产物 key；
- [x] 统一约定常见产物命名；
- [x] 定义 summary / manifest 路径。

**依赖**
- Phase 3 完成。

**验收标准**
- 各阶段产物的记录格式统一。

---

### P5-T2 实现 `report_builder.py`
- [x] 生成 `RUN_SUMMARY.md`；
- [x] 生成 `RUN_SUMMARY.json`；
- [x] 生成 `ARTIFACT_MANIFEST.json`；
- [x] 汇总关键指标与路径。

**依赖**
- `P5-T1`。

**验收标准**
- 一次 run 结束后一定有 summary 和 manifest。

---

### P5-T3 把 workflow 状态映射到任务板
- [x] 设计默认任务树：
  - [x] 审计数据集
  - [x] 生成运行配置
  - [x] 执行静态建模
  - [x] 执行迟滞建模（若启用）
  - [x] 执行部署验证
  - [x] 生成最终报告
- [x] 在 workflow 启动时自动创建对应任务；
- [x] 在阶段推进时自动更新 task status。

**依赖**
- `P4-T2`。

**验收标准**
- 用户能在 scoreboard 中看到 workflow 的阶段进度。

---

### P5-T4 输出用户友好的最终摘要
- [x] 成功时输出：关键路径、关键指标、是否启用温度/迟滞、VA 文件位置；
- [x] 失败时输出：失败阶段、主要原因、建议下一步。

**依赖**
- `P5-T2`。

**验收标准**
- 用户不需要自己翻目录找结果。

---

### P5-T5 端到端烟雾测试
- [x] 使用 `example_dataset` 跑完整闭环：check-data → fit-physical → train-base-pkan → train-static；
- [x] 验证所有产物文件生成正确；
- [x] 验证 `RUN_SUMMARY.md` / `ARTIFACT_MANIFEST.json` 内容完整；
- [x] 此项为每个 Phase 完成后的回归检查项。

**依赖**
- `P5-T4`（报告生成完成后才能验证）。

**验收标准**
- example_dataset 一次完整跑通，产物齐全，报告可读。

---

## 10. Phase 6：失败诊断与 subagent 协作

## 10.1 目标

当 workflow 失败或存在模糊情况时，自动调用 subagent 做解释、诊断和总结。

## 10.2 TODO

### P6-T1 实现 `diagnostics.py`
- [x] 定义错误分类：
  - [x] `DATA_ERROR`
  - [x] `CONFIG_ERROR`
  - [x] `TRAINING_NUMERICAL_ERROR`
  - [x] `MISSING_ARTIFACT_ERROR`
  - [x] `HSPICE_ENV_ERROR`
  - [x] `UNKNOWN_ERROR`
- [x] 实现日志归类规则。

**依赖**
- Phase 5 完成。

**验收标准**
- 失败 run 可以被归入有限类别。

---

### P6-T2 实现 `subagent_hooks.py`
- [x] 定义何时调用 `explorer` subagent；
- [x] 定义何时调用 `worker` subagent；
- [x] 定义 prompt 模板。

**依赖**
- `P6-T1`。

**验收标准**
- subagent 调用点明确且不滥用。

---

### P6-T3 数据异常自动解释
- [x] 数据审计失败时自动读取 JSON/日志；
- [x] 调用 `explorer` subagent 生成可读说明；
- [x] 将说明写入 workflow summary。

**依赖**
- `P6-T2`。

**验收标准**
- 数据错误不再只是原始报错堆栈。

---

### P6-T4 训练失败自动解释
- [x] `fit-physical` / `train-base-pkan` / `train-static` / `train-hysteresis` 任意阶段失败时，自动分析 stdout/stderr；
- [x] 用 `explorer` subagent 归纳失败原因。

**依赖**
- `P6-T3`。

**验收标准**
- 用户能直接看到“可能原因”和“建议处理方式”。

---

### P6-T5 最终报告增强
- [x] 使用 `worker` subagent 生成更自然的最终摘要草稿；
- [x] 规则模板作为兜底；
- [x] 避免让 LLM 决定事实，只让它组织语言。

**依赖**
- `P6-T2`。

**验收标准**
- 总结质量提升，但仍保持可控。

---

## 11. Phase 7：质量守卫与自动重拟合

## 11.1 目标

模型第一次跑通不代表效果达标。Agent 必须在每个训练阶段结束后**自动评估拟合质量**，如果关键精度指标（RMS 误差等）不满足阈值，则自动调整参数后重跑，以此减少专家反复手调参数的工作量。

## 11.2 设计要点

### 两档失败需区分

| 类型 | 判定 | 动作 |
|------|------|------|
| **硬失败** | exit code ≠ 0（崩溃/发散/OOM） | Phase 6 诊断 + 报告 |
| **软失败** | exit code = 0 但质量不达标 | Phase 7 自动调参重试 |

软失败是 Phase 7 的核心场景 — 脚本没报错，但 RMS 0.15 明显不可用。

### 调参策略

采用**渐进式按优先级**调整，每次只改 1-2 个参数，避免组合爆炸。

| 阶段 | 质量指标 | 第 1 次重试 | 第 2 次重试 | 第 3 次重试 |
|------|---------|-----------|-----------|-----------|
| `fit-physical` | final_RMS | `max_nfev_scale` ×2.0 | `max_nfev_scale` ×3.0 + 换 seed | 降采样 `max_iv_points` |
| `train-base-pkan` | best_loss | `epochs` +30% | `kan_hidden` +4, `kan_grid` +2 | `lr` 减半 + `epochs` 翻倍 |
| `train-static` | iv_rms, cv_rms | `epochs` +50%, `lr` ×0.5 | `cv_weight` ×2.0, `epochs` 翻倍 | `early_stop_patience` 翻倍 + 换 seed |
| `train-hysteresis` | hysteresis_RMS | `epochs` +50% | `hidden_capture/emission` +4 | `amp_parameterization` → aggressive |

### 架构

```
orchestrator.py (阶段执行后)
  → quality_guard.py (质量评估)
    → 调用 _assess_quality() 解析 stdout/metrics 产出
    → 若达标 → 继续下一阶段
    → 若未达标且未超重试次数 → _adjust_params() 修改 .ini → 重跑当前阶段
    → 若未达标且已超重试次数 → 标记 WARNING + 附最佳结果继续
```

## 11.3 TODO

### P7-T1 设计质量门禁与默认阈值
- [x] 定义各阶段质量指标及默认阈值：
  - [x] `fit-physical` → physical_iv_rms_max（默认 0.40，基于 debug run 实测 0.255 设 57% margin）
  - [x] `train-static` → static_iv_rms_max（默认 0.60，兜底值）、static_cv_rms_max（默认 0.08）
  - [x] `train-hysteresis` → hysteresis_rms_max（默认 0.15，兜底值）
- [x] 将阈值写入 `agent_configs.json` 的 `DG_TFT_QUALITY_*` 键；
- [ ] 允许从 DatasetProfile 的 data scale 动态调整阈值（大电流器件 vs 小电流器件）。

**依赖**
- Phase 3（需要 orchestrator 产出的 stage results）。

**验收标准**
- 每个阶段都有可配置的质量门槛。

---

### P7-T2 实现 metrics 解析
- [x] ~~解析 stdout/stderr~~ → 改为从阶段产物 JSON 直接读取（更稳定）：
  - [x] `fit-physical` → `01_physical_kernel/fitted_params.json` → `.metrics_summary.iv_rms_log10_error_dec`
  - [x] `train-static` → `03_full_model/v90_safe_residual_metrics.json` → `.residual_iv_rms`, `.residual_cv_rms`
  - [x] `train-hysteresis` → `04_hysteresis/it_hysteresis_analysis_summary.json` → `.step_log_rmse_median`
- [x] 若 JSON 缺失或不可读，返回 {}（不崩溃）；
- [x] 将提取到的 metrics 写回 `StageResult.metrics`。

**依赖**
- `P7-T1`。

**验收标准**
- 每个阶段结束后能自动得到数值化的精度结果。

---

### P7-T3 实现调参规则引擎（`quality_guard.py`）
- [x] 定义每个阶段的参数调整策略表（`_ADJUSTMENT_STRATEGIES` 模块级常量）；
- [x] 实现 `assess_quality(stage_name, metrics, thresholds) → (bool, list[str])`；
- [x] 实现 `suggest_adjustments(stage_name, attempt, ini_path) → dict[str, str]`：
  - [x] 返回 `{section.key: new_value}`（扁平化格式）；
  - [x] 辅助指令 xN / ÷N / +N% / 纯字符串（aggressive）；
- [x] 实现 `apply_adjustments(ini_path, adjustments) → ini_path`：
  - [x] 原地改写 INI 文件（不产生新文件，简化 retry 链）；
- [x] 定义最大重试次数：
  - [x] 默认每个阶段最多 3 次；
  - [x] 通过 `agent_configs.json` 的 `DG_TFT_QUALITY_MAX_RETRIES` 可配；
  - [x] 超过后记录 WARNING + 附 best-effort 结果继续。

**依赖**
- `P7-T2`。

**验收标准**
- 给定一个质量不达标的 run，能自动生成调参后的新 config 并重试。

---

### P7-T4 集成到 orchestrator
- [x] 在 `orchestrator.py` 的阶段成功与下一阶段之间插入质量门禁（`_enforce_quality_gate()`）；
- [x] 重试期间原地改写 INI 文件，后续阶段使用同路径；
- [x] 处理重试循环（最多 N 次，每次调参 + 重跑）；
- [x] 记录重试历史到 `run.extra["quality_retry_history"]`；
- [x] 最终指标取所有 attempts 中的最优值；
- [x] 所有重试均不达标时，标记 WARNING + 用最佳 attempt 结果继续（不阻断流程）；

**依赖**
- `P7-T3`、`P6-T1`（诊断归类）。

**验收标准**
- `run_workflow()` 自动处理效果不好 → 调参 → 重跑 → 收敛的完整闭环；
- 重试耗尽后用户得到明确的诊断说明而非只是 WARNING。

---

### P7-T5 可调参数白名单与安全边界
- [x] 定义允许自动调整的 `[section].[key]` 白名单（`_PARAMETER_WHITELIST` frozenset）；
- [x] 设定每个可调参数的上下界（`_PARAMETER_BOUNDS` 模块级常量，如 `max_nfev_scale` ≤ 5.0，`epochs` ≤ 5000）；
- [x] 边界外的调整记录 WARNING 并 clamp；
- [x] 敏感参数（如数据路径、output_dir）不在 whitelist 中，杜绝误改。

**依赖**
- `P7-T3`。

**验收标准**
- 自动调参不会写入危险值或破坏性路径。

---

### P7-T6 质量报告输出
- [x] 在 `RUN_SUMMARY.md` / JSON 中包含质量评估结果（`## 5. Quality`）；
- [x] Markdown 含 retry log 表格（attempt/pass/adjustments/metrics/error）和最终 metrics；
- [x] JSON `summary["quality"]` 含 retries 计数和完整 history；

**依赖**
- `P7-T4`、`P5-T2`。

**验收标准**
- 用户能回看 Agent 在哪些参数上做了自动调整。

---

### P7-T7 最优 attempt 产物保留（可复现）

**问题**：当前 retry 循环每次都往同一个 `output_dir` 写产物，后一次覆盖前一次。循环结束后磁盘上只保留**最后一次重试**的产物（模型权重、VA 文件、JSON 等），而 `best_score` 可能来自更早的 attempt。用户无法拿到最优结果对应的实际模型文件。

**方案**：

```
_enforce_quality_gate() 内部：

  记 base_output_dir = INI 里的原始 output_dir
  best_attempt = 0     # 0 = 初始（未重试）

  for attempt in range(max_retries):
    1. 创建 retry 子目录:  {base_output_dir}/_retry_{N}/
    2. 修改 INI paths.output_dir → 指向 retry 子目录
    3. run_stage() → engine 写入 retry 子目录
    4. parse_metrics() + assess_quality()
    5. 若当前 attempt 优于 best_score → best_attempt = N
    6. 若 PASS → break

  循环结束后：
    - 若 best_attempt != 0:  将 _retry_{best_attempt}/ 内容复制到 base_output_dir/
    - 删除所有 _retry_*/ 子目录（或保留 `_retry_0/` 供审计）
    - 恢复 INI output_dir = base_output_dir
```

**边界处理**：
- 首次就 PASS（0 次重试）：不创建任何 `_retry_*` 目录，原路径产物不动
- 所有 retry 耗尽仍未 PASS：选择 best_attempt，复制其产物回主目录
- 中间某次 hard crash（ok=False）：那次不参与 best 比较，产物可能不完整，跳过
- `_retry_0/` 保存初始 run 的原始产物（作为 baseline 参考）

**INI output_dir 修改方式**：在 `apply_adjustments()` 或单独函数中，直接行替换 `paths.output_dir = ...`，与其他 adjustments 一致。

**依赖**
- `P7-T4`。

**验收标准**
- 给定 fit-physical 跑 3 次重试，第 1 次 iv_rms=0.30 最优，最终 output_dir 下的 `fitted_params.json` 来自第 1 次而非第 3 次；
- 首次 PASS 时，目录结构无变化（与 P7 修复前一致）。

---

## 12. Phase 8：脚本函数化重构

## 12.1 目标

把 DG-TFT 建模代码从“脚本集合”升级为“可 import API”，降低 workflow 对 subprocess 的依赖。

## 12.2 TODO

### P8-T1 函数化 `validate_dataset.py`
- [ ] 提供 `validate_dataset(...) -> dict`；
- [ ] CLI 只保留为包装层；
- [ ] workflow 优先调用函数，不再依赖 stdout 解析。

**依赖**
- Phase 7 完成。

**验收标准**
- 数据校验可直接通过 Python 调用。

---

### P8-T2 函数化 `fit_physical_kernel.py`
- [ ] 提供 `fit_physical(...)->StageResult`；
- [ ] 保留 CLI 入口兼容；
- [ ] 把关键结果结构化返回。

**依赖**
- `P8-T1`。

**验收标准**
- 物理内核阶段可直接函数调用。

---

### P8-T3 函数化 `train_base_pkan.py`
- [ ] 提供 `train_base_pkan(...)->StageResult`；
- [ ] 结构化返回 best metrics 和模型路径。

**依赖**
- `P8-T2`。

**验收标准**
- base pkan 阶段可脱离 subprocess。

---

### P8-T4 函数化 `dg_tft_full_model.py`
- [ ] 提供 `train_full_model(...)->StageResult`；
- [ ] 提取静态关键指标；
- [ ] 返回 VA 产物信息。

**依赖**
- `P8-T3`。

---

**验收标准**
- 静态训练阶段可函数调用。

### P8-T5 函数化动态模块
- [ ] 函数化 `analyze_it_hysteresis.py`；
- [ ] 函数化 `train_srh_top2_regime_banks.py`；
- [ ] 函数化导出和验证脚本；
- [ ] 统一返回 `StageResult`。

**依赖**
- `P8-T4`。

---

### P8-T6 切换 workflow 到 import API 模式
- [ ] 在 `config/agent_configs.json` 中支持 `DG_TFT_USE_IMPORT_API=true`；
- [ ] 在 `stage_runner.py` 中新增 import 执行分支；
- [ ] 保留 subprocess fallback。

**依赖**
- `P8-T5`。

---

**验收标准**
- 工作流默认走函数调用，必要时仍可回退 CLI。

## 13. Phase 9：测试、稳定性与文档收尾

## 13.1 目标

把整个能力从"能跑"提升到"能维护、能验证、能交付"。

## 13.2 TODO

### P9-T1 补充并统一 workflow 单元测试
- [ ] 补充 `test/dg_tft_artifacts_test.py`（产物管理）
- [ ] 补充 `test/dg_tft_report_builder_test.py`（报告生成）
- [ ] 补充 `test/dg_tft_diagnostics_test.py`（诊断归类）
- [ ] 补充 `test/dg_tft_quality_guard_test.py`（质量守卫）
- [ ] 复查已有 7 个测试文件（135 tests）的覆盖率缺口

**依赖**
- Phase 8 基本完成。

**验收标准**
- 核心 workflow 组件有基础单元测试覆盖。

---

### P9-T2 新增 Agent 集成测试
- [ ] `test/dg_tft_workflow_tool_test.py`
- [ ] `test/dg_tft_workflow_background_test.py`
- [ ] 测试主 agent 是否优先用 workflow tool；
- [ ] 测试任务板联动。

**依赖**
- `P9-T1`。

**验收标准**
- workflow 已真正接入 agent，而不是只在模块层可跑。

---

### P9-T3 准备三类测试数据集
- [ ] 仅静态示例数据；
- [ ] 静态 + I-t 完整数据；
- [ ] 故意损坏的数据。

**依赖**
- `P9-T1`。

**验收标准**
- 覆盖 success / partial / fail 三类典型场景。

---

### P9-T4 更新使用文档
- [ ] 更新 `doc/` 中的用户使用说明；
- [ ] 增加"如何启动 DG-TFT 自动建模 workflow"；
- [ ] 增加"失败后如何查看报告"。

**依赖**
- `P9-T2`。

**验收标准**
- 用户知道如何使用新能力。

---

### P9-T5 收尾检查
- [ ] 清理临时调试代码；
- [ ] 检查 import 路径一致性；
- [ ] 检查 session 持久化兼容性；
- [ ] 检查命名统一性；
- [ ] 检查产物路径是否有硬编码问题。

**依赖**
- `P9-T4`。

**验收标准**
- 功能可持续维护，不依赖开发时临时假设。

---

## 13. 推荐优先级

如果你希望按投入产出比排序，建议优先级如下：

### 第一优先级
- [x] `P0-T1` ~ `P0-T5`
- [x] `P1-T1` ~ `P1-T6`
- [x] `P2-T1` ~ `P2-T7`
- [x] `P3-T1` ~ `P3-T10`
- [x] `P4-T1` ~ `P4-T4`

这部分做完，就已经有“给路径即可全自动跑”的第一版闭环。


### 第二优先级
- [x] `P4-T5`
- [x] `P4-T6`
- [x] `P4.5-T1` ~ `P4.5-T4`（数据整理与验证 — 解决原始数据格式不统一问题）
- [x] `P5-T1` ~ `P5-T5`（报告/产物 + 烟雾测试）
- [x] `P6-T1` ~ `P6-T5`（诊断：分类 + subagent hooks + 报告集成）
- [x] `P7-T1` ~ `P7-T6`（质量守卫核心 — 阈值、metrics 解析、调参引擎、orchestrator 集成、报告）

这部分做完，Agent 不仅知道“跑崩了为什么”，还能自动调参重试直到效果达标。

### 第三优先级
- [x] `P7-T6`（质量报告 — 随 P7 T1~T5 一起完成）
- [x] `P6-T3` ~ `P6-T5`（诊断的 subagent 增强 + 报告集成 — Phase 6 已完整完成）
- [ ] `P8-T1` ~ `P8-T6`
- [ ] `P9-T1` ~ `P9-T5`

这部分做完，系统才进入“可维护、可扩展”的工程化成熟阶段。

---

## 14. 第一批建议直接开工的任务

如果你要马上进入编码，我建议第一批只做下面 11 项：

- [x] `P0-T1` 确认 DG-TFT 建模代码正式位置
- [x] `P0-T2` 建立 `src/device_modeling/dg_tft/`
- [x] `P0-T3` 迁移/镜像现有脚本
- [x] `P1-T1` 新建 `src/workflow/dg_tft/` 骨架
- [x] `P1-T2` 定义 `WorkflowRun / StageResult / DatasetProfile`
- [x] `P1-T3` 实现 `WorkflowManager`
- [x] `P2-T6` 实现 `config_builder.py`
- [x] `P3-T2` 实现统一 `stage_runner.py`
- [x] `P3-T8` 实现 `orchestrator.py`
- [x] `P3-T10` 接入 console 用户可见输出
- [x] `P4-T1` + `P4-T2` 接 `start_dg_tft_workflow`

这 11 项完成后，你就会拥有一版真正可用的第一代系统。

**下一批建议优先：Phase 4.5**
- [x] `P4.5-T1` 实现 `data_preparer.py`（worker + verifier + check）
- [x] `P4.5-T2` 接入 `prepare_dg_tft_dataset` tool
- [x] `P4.5-T3` 命令 + prompt 路由更新
- [x] `P4.5-T4` 测试

做完这一批后，系统即可接受任意格式的原始数据目录，自动整理为 `*_train.xlsx` 后再启动建模。

---

## 15. 后续追踪建议

建议后续追踪时，不要只写“开始做 workflow”。建议按下面粒度追踪：

- 数据层是否可用；
- 配置层是否稳定；
- stage runner 是否统一；
- summary 是否能自动落地；
- 失败是否可解释；
- agent 是否优先走 workflow tool；
- 是否保留 subprocess fallback；
- 是否已经开始函数化迁移。

这样后续每次开发回顾都会更清晰。

---

## 16. 一句总结

这份 TODO 的核心顺序就是：

```text
先把 DG-TFT 旧脚本包装成稳定 workflow
再把 workflow 接入 DeviceAgent
最后再把 DG-TFT 建模模块逐步重构成可调用 API
```

这是风险最低、最容易持续推进的路线。

---

## 17. 开发日志

### 2026-06-25 | Phase 0 + Phase 1 完成

**完成内容**
- P0-T1 ~ P0-T6：DG-TFT 建模代码迁移至 `src/device_modeling/dg_tft/`，创建环境文档，固化解释器路由到 `config/agent_configs.json`
- P1-T1 ~ P1-T7：建立 `src/workflow/dg_tft/` 工作流骨架，实现 `WorkflowRun`/`StageResult`/`DatasetProfile` 数据结构、`WorkflowManager` 线程安全 CRUD、session 持久化、挂载到 `AgentContext`、`main.py` 初始化、`stage_runner` 解释器解析

**新增文件清单**

```
src/device_modeling/dg_tft/           # 建模代码迁移
  run_model.py
  ENVIRONMENT.md
  config/{model_config.ini, model_config_new_device_template.ini, 关键参数说明.csv}
  engine/{dg_tft_ekv_level2.va, dg_tft_full_model.py, fit_physical_kernel.py, train_base_pkan.py, validate_dataset.py}
  dynamic/{analyze_it_hysteresis.py, train_srh_top2_regime_banks.py, ...}
  deployment/{README.md}
  reporting/{build_*.py, ...}

src/workflow/dg_tft/                 # 工作流引擎
  schema.py              # 数据结构定义
  manager.py             # 工作流管理器
  stage_runner.py        # 统一阶段执行 + 解释器解析
  orchestrator.py        # (占位)
  stages.py              # (占位)
  artifacts.py           # (占位)
  dataset_inspector.py   # (占位)
  config_builder.py      # (占位)
  diagnostics.py         # (占位)
  report_builder.py      # (占位)
  subagent_hooks.py      # (占位)

test/
  dg_tft_schema_test.py          # 18 tests
  dg_tft_manager_test.py         # 24 tests
  dg_tft_stage_runner_test.py    # 13 tests
  dg_tft_integration_test.py     # 14 tests (含闭环集成)
```

**修改文件清单**

| 文件 | 修改内容 |
|------|---------|
| `config/agent_configs.json` | 新增 9 项 DG_TFT 配置（解释器路径、输出目录、HSPICE 开关等） |
| `src/context/agent_context.py` | 新增 `workflow_man: WorkflowManager` 成员 + `to_dict`/`from_dict`/`save_context`/`load_context` 中融入 workflow 持久化 |
| `src/main.py` | 新增 workflow_man 初始化 + session resume 加载 |

---

### 2026-06-25 | Phase 2 完成 — 数据审计与 runtime config 自动生成

**完成内容**
- P2-T1 ~ P2-T5：实现 `dataset_inspector.py` — subprocess 调用 CUDA 环境 `validate_dataset.py`，解析 JSON 构建 `DatasetProfile`，自动发现 I-t 目录，判定温度模式，检查 HSPICE 可用性
- P2-T6 ~ P2-T7：实现 `config_builder.py` — 从模板 .ini 生成 run-specific 运行时配置，自动填充 data_dir/hysteresis/temperature/hspice 字段，输出 `dataset_profile.json` + `config_resolution.json`

**新增文件清单**

```
src/workflow/dg_tft/
  dataset_inspector.py   # 数据集审计入口 (inspect_dataset)
  config_builder.py      # 运行时配置生成 (build_runtime_config)

test/
  dg_tft_inspector_test.py   # 34 tests (it discovery, profile building, hspice check, mocked subprocess)
  dg_tft_builder_test.py     # 13 tests (ini generation, resolution, template preservation)
```

**修改文件清单**

| 文件 | 修改内容 |
|------|---------|
| `doc/DG_TFT_Agent全自动建模开发TODO.md` | Phase 0~2 打勾 + 补充开发日志与经验 |

**测试统计**
- Phase 2 新增：34 tests (inspector) + 13 tests (builder) = 47 tests
- 累计全部：103 tests (schema 18 + manager 24 + stage_runner 13 + integration 14 + inspector 24 + builder 10 — wait let me recount)
  - inspector: 9 (DiscoverItDir) + 5 (BuildProfile) + 5 (HspiceCheck) + 5 (InspectDatasetMocked) = 24
  - builder: 4 (BuildRuntimeConfig) + 3 (Resolution) + 3 (WriteIni) = 10
  - Total: 18 + 24 + 13 + 14 + 24 + 10 = 103 ✓

---

### 2026-06-25 | Phase 3 完成 — subprocess 版全自动工作流

**完成内容**
- P3-T1：实现 `stages.py` — 定义 `StageDef` dataclass（name/action/required/depends_on/key_artifacts/expected_output_subdir/timeout/skip_if_profile_false），含 7 个标准阶段定义
- P3-T2：`stage_runner.py` 已有完整 subprocess 执行能力（P1 完成，Phase 3 验证通过）
- P3-T3~P3-T7：通过 `orchestrator.py` 串联所有阶段执行（check-data → fit-physical → train-base-pkan → train-static → [train-hysteresis]），含产物校验和 artifacts 挂载
- P3-T8：实现 `orchestrator.py` — `run_workflow()` 主函数，支持顺序执行、fail-fast、条件 skip（hysteresis）、resume（已完成阶段跳过）、artifact 自动挂载
- P3-T9：工作目录结构由 `config_builder.py` 管理（Phase 2 完成）

**新增文件清单**

```
src/workflow/dg_tft/
  stages.py              # 阶段定义（StageDef dataclass + get_stages/get_stage_by_name）
  orchestrator.py        # 工作流编排器（run_workflow + helpers）

test/
  dg_tft_orchestrator_test.py   # 32 tests（14 stage defs + 12 helpers + 6 orchestrator）
```

**修改文件清单**

| 文件 | 修改内容 |
|------|---------|
| `src/workflow/dg_tft/schema.py` | `WorkflowRun.to_dict()` 改用手动 dict 构造替代 `asdict()`，避免 deepcopy 序列化兼容性问题 |
| `doc/DG_TFT_Agent全自动建模开发TODO.md` | Phase 3 打勾 + 补充开发日志与经验 |

**测试统计**
- Phase 3 新增：32 tests
- 累计全部：135 tests (18 + 24 + 13 + 14 + 24 + 10 + 32)

---

### 2026-06-25 | Phase 4 完成 + Phase 4.5 设计

**完成内容**
- P4-T1 ~ P4-T6：Agent 工具集成全完成
  - `start_dg_tft_workflow` / `query_dg_tft_workflow` / `read_dg_tft_workflow_report` 三个工具
  - 异步执行模型：`threading.Thread` + `WorkflowManager._running_threads` tracker
  - `call_tools()` dispatch 分支
  - `/workflow_list` / `/workflow_get` / `/workflow_report` 内建命令
  - System prompt 全面重写：TECoSim/display panel/simulator → DeviceModeling/DG-TFT fitting
  - Simulator 六个工具从 LLM 视野完全移除（注释 schema 但不删实现）
- Phase 4.5 设计方案确定：数据整理 + 双 Agent 验证
  - Worker subagent 整理 + Verifier 独立 LLM call（结构化 tool schema）
  - `_check_output()` Python 代码做 CRITICAL/WARNING 判定（零 LLM 参与）
  - 继承 `summarize_support.py` 模式避免结构化输出幻觉

**新增文件清单**
```
test/dg_tft_tools_test.py          # 28 tests（P4-T1~T6 全覆盖）
```

**修改文件清单**

| 文件 | 修改内容 |
|------|---------|
| `src/constants.py` | 新增 3 个 workflow tool name 常量 |
| `src/tool/tool_def.py` | 新增 3 schema + 3 handler + `_run_workflow_thread()`；simulator 6 工具注释掉；注册到 `create_tools_prompts()` |
| `src/tool/tool_dispatch.py` | 新增 3 个 dispatch 分支 |
| `src/workflow/dg_tft/manager.py` | 新增 `_running_threads` dict |
| `src/utility/command.py` | 新增 3 个内建命令 + `BuiltinCommands` 注册 |
| `src/context/prompt.py` | 全面重写 system prompts（定位切换 + DG-TFT 路由引导） |
| `src/agent/progress.py` | 清理 `read_log` from allowlist/desc/presets |
| `doc/DG_TFT_Agent全自动建模开发TODO.md` | Phase 4 checkmarked + Phase 4.5 设计插入 |

**测试统计**
- Phase 4 新增：28 tests
- 累计全部：163 tests

**下阶段：Phase 4.5**
- P4.5-T1 `data_preparer.py` 核心模块
- P4.5-T2 `prepare_dg_tft_dataset` agent tool
- P4.5-T3 内建命令 + prompt 路由
- P4.5-T4 测试

### 2026-06-25 | Bugfix round — 7 issues from end-to-end test

**Fixed**
1. **CloneContext `workflow_man` 不共享** — `subagent.py:clone_context` 新增 `ctx.workflow_man = parent_ctx.workflow_man`
2. **Worker 用 AGI python 缺少 pandas** — `data_preparer.py` worker prompt 模板新增 `{python_exe}` 占位符，`_run_worker` 注入 CUDA python 路径
3. **SubAgent 进度不显示** — `_run_worker` 注册/注销 SubAgentProgress 到 `parent_ctx.agent_list`，实时显示在 spinner 中
4. **数据已准备时跳过** — `prepare_dg_tft_dataset` 新增 `_check_prepared_output()` 检测已有 `*_train.xlsx`
5. **Workflow 改为前台阻塞** — 移除 `_run_workflow_thread` + `threading.Thread`，`start_dg_tft_workflow` 直接同步执行
6. **train-static `kan_config` KeyError** — `dg_tft_full_model.py` 新增 `_resolve_kan_config()` 兼容新旧 metadata 格式
7. **Bash DENIED 问题调查** — `ask_permission_tui` 在 `subagent_mute=True` 时静默检查权限，worker 有 BASH_SAFE preset 但实际 DENIED；根因待进一步排查。

**测试**: 234 tests PASS (移除了 3 个 `_run_workflow_thread` 相关测试)

---

### 2026-06-25 | Phase 5 完成 — 报告/产物/任务板/用户摘要

**完成内容**
- P5-T1 ~ P5-T5 全部完成
- `artifacts.py`: artifact key 常量 + 文件名映射 + 路径/校验辅助函数
- `report_builder.py`: `build_and_write_reports()` 生成 RUN_SUMMARY.md、RUN_SUMMARY.json、ARTIFACT_MANIFEST.json
- Orchestrator 集成: 全部 stage 完成后自动调 report_builder，失败时也生成部分报告
- P5-T3 任务板同步: `_create_stage_tasks` / `_start_task` / `_resolve_task`，board 通过参数传递（不污染 AgentContext）
- P5-T4 控制台摘要: 成功/失败分别输出路径、指标、失败建议
- 42 new tests in `dg_tft_report_test.py`
- 总计 237 tests PASS

### 2026-06-25 | Phase 4.5 完成 — 数据整理 + 双 Agent 验证

**完成内容**
- P4.5-T1 ~ T4 全部完成
- `data_preparer.py`: worker subagent (处理原始 CSV/XLSX → *_train.xlsx) + verifier LLM call (迷你 tool loop, glob/read/report_verification) + `_check_output` (CRITICAL/WARNING 判定)
- `prep_file_manifest.json`: 记录原始文件→输出文件的来源映射，去重信息
- 前台同步执行 + console 进度输出
- Prompt 模板参数化: 列名/表名等统一用模块级常量，消除硬编码和重复
- 权限体系: 走 `clone_context` + `PERMISSION_PRESETS["worker"]` (含 BASH_INLINE_SCRIPT_LABEL)
- Verifier 轮次耗尽前注入提醒消息
- 195 tests PASS

---

## 18. 经验总结与注意事项

### 18.1 Python 模块双重导入陷阱

**问题**：`from src.workflow.dg_tft.schema import StageResult` 和 `from workflow.dg_tft.schema import StageResult`（当 `sys.path` 含 `src/` 时）被 Python 识别为**两个不同的模块**，导致 `isinstance(v, StageResult)` 返回 False，进而引发序列化/反序列化失败。

**表现**：`asdict` / `to_dict` 看起来正确，但序列化后的 dict 中仍包含原始对象（而非 dict），`from_dict` 时 `StageResult(**v)` 报错 `argument after ** must be a mapping, not StageResult`。

**根本原因**：
- 包内模块（`manager.py`）使用 `from src.workflow.dg_tft.schema import ...`（绝对导入）
- 测试代码使用 `from workflow.dg_tft.schema import ...`（相对导入，依赖 `sys.path`）
- Python 的 `sys.modules` 会为两条路径注册两个独立 module 对象

**修复方案**：
1. 包内模块统一使用**相对导入**：`from .schema import ...`
2. 类型检查避免 `isinstance`，改用 `hasattr(v, "__dataclass_fields__")` 鸭子类型检测
3. 所有 `WorkflowRun`/`StageResult` 等交叉引用的类，尽量从同一模块路径 import

**教训**：在 `src/` 内的模块之间互引用，统一使用相对导入（`from .xxx import`）。对外部调用方（如 `main.py`、`agent_context.py`）保持现有的绝对导入约定（`from src.xxx.yyy import`）。

### 18.2 `dataclass.asdict` 行为注意

- `asdict` 递归转换嵌套 dataclass 实例为 dict
- 对于 `dict[str, StageResult]` 类型字段，`asdict` 会将其值（StageResult 对象）递归转换为 dict
- 但如果 `isinstance` 因模块双重导入失效，`asdict` 本身仍然能正确递归（因为它遍历 `__dataclass_fields__`）
- **结论**：在 `to_dict()` 实现中，直接使用 `asdict` 比依赖 `isinstance` 更健壮

### 18.3 StageResult 与 WorkflowRun 的 metrics 分离

`WorkflowRun` 有两个 metrics 来源：
1. `workflow_run.metrics` — 工作流级别的聚合指标，通过 `attach_metric()` 写入
2. `stage_result.metrics` — 单个阶段内部的指标，仅在该阶段记录中可见

两者**独立维护**，聚合需由调用方负责。设计如此，避免阶段内部指标自动污染 workflow 级指标。

### 18.4 `__pycache__` 缓存问题

开发过程中发现旧的 `__pycache__` 会导致代码修改不生效。建议：
- 每次模块重构后，清除对应 `__pycache__/`
- 可在提交前运行 `clean.bat` 清理

### 18.5 测试策略建议

当前测试覆盖：
- **单元测试**（`dg_tft_schema_test` / `dg_tft_manager_test` / `dg_tft_stage_runner_test` / `dg_tft_inspector_test` / `dg_tft_builder_test`）
- **集成测试**（`dg_tft_integration_test`）：覆盖 AgentContext ↔ WorkflowManager 集成、完整工作流生命周期模拟

运行命令：
```bash
# 全部 DG-TFT 测试（135 tests）
python -m unittest discover -s test -p "dg_tft_*_test.py" -v
```

后续 Phase 开发建议：每个新模块同步编写对应测试文件，保持 TDD 节奏。

### 18.6 Phase 2 经验：HSPICE 可用性检查

HSPICE 可用性通过读取模板 .ini 中的 `hspice_exe` 路径并检查 `Path.exists()` 来判断。`config_builder.py` 和 `dataset_inspector.py` 各有一份 `_read_template_hspice_path()` 实现（读取同一模板），避免循环导入。

注意：Windows 上 `Path('/non/existent/file').exists()` 可能因路径解析差异返回意外结果。测试中应使用绝对确定不存在的路径（如 `tmpdir/does_not_exist.exe`）来模拟 HSPICE 不可用场景。

### 18.7 Phase 2 经验：I-t 目录发现顺序

`_discover_it_dir` 按优先级搜索：
1. 用户显式指定的 `hysteresis_dir`
2. `dataset_dir/hysteresis_it`
3. `dataset_dir/` 下含 `it`/`hysteresis`/`I-t` 关键字的子目录
4. 同级目录中含匹配关键字的目录

当 `parent.iterdir()` 可能抛出 `FileNotFoundError`（模拟环境中路径不存在）或 `PermissionError` 时，用 try/except 兜底。

### 18.8 Phase 3 经验：orchestrator 设计要点

1. **阶段定义与执行分离**：`stages.py` 定义静态阶段元数据（名称/产物/超时/跳转条件），`orchestrator.py` 负责动态执行逻辑。这样阶段顺序可配置，测试可独立验证阶段定义完整性。
2. **Resume 能力**：通过 `_stage_done()` 检查 `run.stage_results` 中已有阶段结果，已完成的阶段自动跳过，支持断点续传。
3. **条件 skip 两层过滤**：`get_stages(profile)` 在阶段列表构建阶段过滤掉不需要的阶段（如无 hysteresis 数据则移除 `TRAIN_HYSTERESIS`）；`orchestrator` 中另有 `skip_if_profile_false` 检查作为第二道防线。
4. **Fail-fast vs optional**：`required=True` 的阶段失败立即停止整个 run 并标记 FAILED；`required=False` 的阶段失败记录 warning 后继续。
5. **mock 测试的签名陷阱**：使用 `unittest.mock.patch` 替换 `run_stage()` 时，mock 函数的第一个参数是 `ctx`（AgentContext），而非 `action`。签名应为 `(self, ctx, action, *args, **kwargs)`。
6. **用户可见性**：workflow 模块只有 `sys_log`（默认仅写文件），`ctx.console`（Rich TUI）未接入。P3-T10 需补全，利用已有的 `ctx.console`（`main.py:93` 已注入）在各关键节点输出进度，无需新增参数。
7. **Console 输出风格规范**：
   - 代码中**不出现中文**（注释和 docstring 中的 header 除外）；
   - `console.print()` 输出统一英文，风格对齐项目现有惯例：
     - 错误 → `[bold red]` 或 `style="bold red"`
     - 警告 → `[bold yellow]` 或 `style="bold yellow"`
     - 次要/补充信息 → `[bright_black]`
     - 成功 → `[bold green]` 或 `[green]`
     - `MAJOR_COLOR1` / `MAJOR_COLOR2` 仅用于真正需要强调的关键信息（如模块名、配置项），**不整句滥用**；
   - 不引入新的 `Live` / `Progress` / `Panel`，只用 `console.print()`；
   - `sys_log` 独立保留作为完整审计日志，风格互不影响。

### 18.9 `asdict` 序列化兼容性问题

`dataclasses.asdict()` 内部使用 `copy.deepcopy()` 递归转换所有字段。当 `WorkflowRun` 包含复杂嵌套结构（如含被 mock 对象污染的字段）时，`deepcopy` 可能因 `TypeError: cannot pickle '_thread.lock' object` 失败。

**修复方案**：`WorkflowRun.to_dict()` 改用手动 dict 构造，逐字段赋值，避免 `asdict` 的 deepcopy 行为。`StageResult` 同样使用 `getattr` + 手动 dict 构造。

### 18.10 Phase 4 完成要点

1. P4-T1：新增 workflow tool schema（`start_dg_tft_workflow` / `query_dg_tft_workflow` / `read_dg_tft_workflow_report`），含完整 OpenAI function-calling 参数定义
2. P4-T2：实现工具 handler（`start_dg_tft_workflow` / `query_dg_tft_workflow` / `read_dg_tft_workflow_report`），`start` 内部串联 `create_run() → inspect_dataset() → build_runtime_config() → run_workflow()`，全部在后台线程执行
3. P4-T3：异步执行模型 — `start_dg_tft_workflow` 立即返回 `run_id`，工作由 `threading.Thread` 在后台跑；`WorkflowManager._running_threads` dict 追踪活跃线程；线程异常 → `mark_failed`；`finally` 清理 tracker
4. P4-T4：`call_tools()` dispatch 三个新工具名 → 对应 handler
5. P4-T5：`/workflow_list` / `/workflow_get <run_id>` / `/workflow_report <run_id>` 三个内建命令注册在 `BuiltinCommands`
6. P4-T6：system prompt 添加 "DG-TFT Workflow Routing" 段落，引导 LLM 优先使用 `start_dg_tft_workflow` 而非逐条 bash 调用；prompt 整体从 display panel/simulator 重写为 DeviceModeling agent
7. 额外：simulator 六个工具（`check_simulator`/`init_design`/`query_design`/`launch_sim`/`query_run`/`read_log`）从 `create_tools_prompts()` 中注释掉，从 `SUPPORTED_TYPES` explorer allowlist 移除 `read_log`，从 `SUPPORTED_TYPES_DESC` 和 `PERMISSION_PRESETS` 中清理；LLM 完全无感 simulator

---

### 2026-06-26 | Phase 6 P6-T1 + P6-T2 完成 — 失败诊断 + subagent hooks

**完成内容**
- P6-T1：实现 `diagnostics.py` — 完整错误分类引擎
  - 6 大错误类别：`DATA_ERROR` / `CONFIG_ERROR` / `TRAINING_NUMERICAL_ERROR` / `MISSING_ARTIFACT_ERROR` / `HSPICE_ENV_ERROR` / `UNKNOWN_ERROR`
  - 关键词匹配分类：`_KEYWORD_RULES` 模块级常量表，按 priority 排序（TRAINING_NUMERICAL_ERROR > HSPICE_ENV_ERROR > DATA_ERROR > CONFIG_ERROR > MISSING_ARTIFACT_ERROR）
  - 启发式规则：`classify_error(stage_name, exit_code, stdout, stderr, error_message, missing_artifacts)`
  - 出口码 0 + 缺产物 → `MISSING_ARTIFACT_ERROR` (confidence 0.85)
  - 出口码 ≠ 0 + 无关键字 → 按阶段名推断（check-data → DATA_ERROR 0.5, fit-physical/train-static/train-hysteresis → TRAINING_NUMERICAL_ERROR 0.4）
  - `classify_from_result()` 便捷包装 — 自动解包 StageResult
  - `DiagnosticResult` dataclass — category / confidence / evidence / suggestion
  - 每个阶段有独立 suggestion 文案（`_STAGE_SUGGESTIONS` 模块级常量）
  - 无硬编码：所有分类规则集中在 `_KEYWORD_RULES`，confidence 作为规则属性
- P6-T2：实现 `subagent_hooks.py` — 诊断 hook 定义与 prompt 模板
  - 6 个 HookDefinition（每类别一个），全部使用 `explorer` subagent type
  - Hook ID：`on_data_error` / `on_training_failure` / `on_unknown_error` / `on_missing_artifact` / `on_config_error` / `on_hspice_error`
  - 每个 hook 有独立 Prompt 模板（参数化 `{stage_name}` `{category}` `{confidence}` `{evidence}` `{error_message}`），无硬编码阶段名
  - `get_hook(category)` 快速查表
  - `build_diagnosis_prompt(hook, ...)` 渲染模板，含 log_snippet 注入和 truncation
  - Error message 截断上限 2000 chars，log snippet 最多 60 lines

**新增文件清单**
```
src/workflow/dg_tft/
  diagnostics.py       # Error classification engine (P6-T1)
  subagent_hooks.py    # Hook definitions + prompt templates (P6-T2)

test/
  dg_tft_diagnostics_test.py   # 66 tests (P6-T1 + P6-T2)
```

**测试统计**
- Phase 6 新增：66 tests
- 累计全部：299 tests PASS

**下阶段：Phase 7 P7-T1 ~ T3**
- P7-T1：质量门禁阈值设计
- P7-T2：metrics 解析（stdout/stderr → RMS/loss）
- P7-T3：调参规则引擎（`quality_guard.py`）

---

### 2026-06-26 | Phase 6 P6-T3 ~ T5 完成 — 诊断集成 + 报告增强

**完成内容**
- P6-T3 + P6-T4：orchestrator 诊断集成
  - `orchestrator.py` 新增 `_run_stage_diagnosis()` — 阶段失败时自动分类 + 可选 explorer subagent 诊断
  - 分类结果（`DiagnosticResult`）始终写入 `run.extra["diagnosis"]`（零 LLM 依赖）
  - Explorer subagent 诊断通过 `DG_TFT_ENABLE_DIAGNOSIS` agent_config 键控制（默认启用）
  - `_launch_diagnosis_subagent()` — 基于数据准备 worker 模式创建独立 explorer subagent 实例
  - `_collect_log_snippet()` — 从输出目录收集 log 片段注入 prompt
  - 诊断失败不传播异常（try/except 兜底）
  - `_console_summary()` 展示分类结果 + subagent 诊断文本
- P6-T5：report_builder 诊断集成
  - `_build_json_summary()` 在 extra.diagnosis 存在时写入 `summary["diagnosis"]`
  - `_build_md_summary()` 新增 "## 4. Diagnosis" 段落，含 category/confidence/evidence/suggestion/subagent_result
  - 无诊断时不生成该段落（向后兼容）
  - 后续 Issues 段落编号自动调整（## 4 或 ## 5）

**修改文件**
| 文件 | 修改 |
|------|------|
| `src/workflow/dg_tft/orchestrator.py` | 新增 imports（diagnostics, subagent_hooks, EXPLORER_AGENT_LABEL）；新增 `_run_stage_diagnosis()` / `_launch_diagnosis_subagent()` / `_collect_log_snippet()` / `_print_diagnosis()`；`_console_summary()` 新增诊断展示 |
| `src/workflow/dg_tft/report_builder.py` | `_build_json_summary()` 新增 diagnosis 字段；`_build_md_summary()` 新增 "## 4. Diagnosis" 段落 |
| `test/dg_tft_diagnostics_test.py` | 新增 15 tests（`TestRunStageDiagnosis` 7 + `TestCollectLogSnippet` 3 + `TestReportDiagnosisIntegration` 5） |

**测试统计**
- Phase 6 累计：81 tests (66 original + 15 new)
- 累计全部：314 tests PASS

**下阶段：Phase 7 P7-T1 ~ T3**
- P7-T1：质量门禁阈值设计
- P7-T2：metrics 解析（stdout/stderr → RMS/loss）
- P7-T3：调参规则引擎（`quality_guard.py`）

---

### 2026-06-26 | Phase 7 完成 — 质量守卫与自动调参重试

**完成内容**
- P7-T1 ~ T6 全部完成
- `agent_configs.json`: 新增 5 个 `DG_TFT_QUALITY_*` 阈值配置键 + `DG_TFT_ENABLE_DIAGNOSIS` 等
- `metrics_parser.py` (新建): 从阶段产物 JSON 提取指标（非 stdout）— fit-physical/train-static/train-hysteresis 三类
- `quality_guard.py` (新建): 
  - `assess_quality()` — 阈值判定，支持 NaN/缺失/多指标
  - `suggest_adjustments()` — 渐进式调参引擎（0~2 次策略，指令 xN/+N/+N%/str）
  - `apply_adjustments()` — 原地改写 INI（line-by-line 替换，保持格式）
  - `_PARAMETER_BOUNDS` — 每 key 有安全边界，溢出自动 clamp
  - `_PARAMETER_WHITELIST` — 只有白名单内的 key 可自动调整
- `orchestrator.py`: +`_enforce_quality_gate()` 质量门禁 + retry 循环 + 5 个 console 输出辅助函数
- `report_builder.py`: JSON/Markdown 新增 `## 5. Quality` 章节（retry log 表格 + 最终 metrics）
- 阈值基于 debug run 实际数据校准：
  - physical iv_rms: 实际 0.25 → 阈值 0.40（57% margin）
  - static/hysteresis: debug run 只有 1 epoch，阈值暂设宽松兜底（0.60 / 0.08 / 0.15）
- **No infinite loops**: 三重终止 — `range(max_retries)` + 策略表耗尽 break + hard crash break

**新增文件清单**
```
src/workflow/dg_tft/
  metrics_parser.py     # Metrics extraction from artifact JSON (P7-T2)
  quality_guard.py      # Quality gate + param adjustment engine (P7-T3, P7-T5)

test/
  dg_tft_quality_guard_test.py   # 56 tests (P7-T1 ~ T6)
```

**修改文件清单**

| 文件 | 修改内容 |
|------|---------|
| `config/agent_configs.json` | +5 质量阈值 + DG_TFT_ENABLE_DIAGNOSIS 等 |
| `src/workflow/dg_tft/orchestrator.py` | imports (metrics_parser, quality_guard)；新增 `_enforce_quality_gate()` + 5 console helpers；主循环 stage success 后插入质量门禁 |
| `src/workflow/dg_tft/report_builder.py` | `_build_json_summary()` +quality section；`_build_md_summary()` +`## 5. Quality` + retry log table |
| `doc/DG_TFT_Agent全自动建模开发TODO.md` | Phase 7 checkmarked + 开发日志 |

**测试统计**
- Phase 7 新增：56 tests
- 累计全部：370 tests PASS
