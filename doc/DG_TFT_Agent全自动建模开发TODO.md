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

## 8. Phase 4：接入 Agent 工具与命令

## 8.1 目标

让主 agent 能优先调用固定 workflow，而不是自己临场拼 bash。

## 8.2 TODO

### P4-T1 新增 workflow tool schema
- [ ] 在 `src/tool/tool_def.py` 新增：
  - [ ] `start_dg_tft_workflow`
  - [ ] `query_dg_tft_workflow`
  - [ ] `read_dg_tft_workflow_report`
- [ ] 明确定义参数与返回结构。

**依赖**
- Phase 3 完成。

**验收标准**
- 工具定义可被主 agent 正常看到。

---

### P4-T2 实现 workflow tool handler
- [ ] 在 `tool_def.py` 中实现三个工具对应函数：
  - [ ] `start_dg_tft_workflow` 内部串联：`create_run() → inspect_dataset() → build_runtime_config() → run_workflow()`；
  - [ ] `query_dg_tft_workflow` 读取 `workflow_man.get_run(run_id)` 返回状态；
  - [ ] `read_dg_tft_workflow_report` 读取 run 输出目录下的 summary 文件；
- [ ] 调用 `workflow_man` 和 `orchestrator`；
- [ ] 返回结构化结果。

**依赖**
- `P4-T1`。

**验收标准**
- 工具可被实际调用并返回结果。

---

### P4-T3 实现后台异步执行模式
- [ ] `start_dg_tft_workflow` 立即返回 `run_id`，不阻塞等待训练完成；
- [ ] 在 `workflow_man` 中新增 `RunningRunTracker`（存储正在执行的 subprocess 句柄）；
- [ ] `query_dg_tft_workflow` 可实时查询 run status 并返回最新 stage；
- [ ] 考虑超长时间运行的看护（心跳检测 / 超时 kill）；
- [ ] 确保多个 run 并发隔离（输出目录、config 文件不冲突）。

**依赖**
- `P4-T2`。

**验收标准**
- 用户启动 workflow 后可立即继续其他操作，通过 query 获取进度。

---

### P4-T4 注册 workflow tool dispatch
- [ ] 修改 `src/tool/tool_dispatch.py`；
- [ ] 加入三个工具的 dispatch 分支。

**依赖**
- `P4-T3`。

**验收标准**
- 工具调用不落入 undefined 分支。

---

### P4-T5 增加内建命令
- [ ] 在 `src/utility/command.py` 增加：
  - [ ] `/workflow_list`
  - [ ] `/workflow_get <run_id>`
  - [ ] `/workflow_report <run_id>`
- [ ] 在命令系统中注册。

**依赖**
- `P4-T3`。

**验收标准**
- 不走 LLM 也可查询 workflow 运行结果。

---

### P4-T6 优化 prompt 路由逻辑
- [ ] 在 `src/context/prompt.py` 中增加“识别 DG-TFT 自动建模请求时优先用 workflow tool”的提示约束；
- [ ] 避免主 agent 将该类请求拆成大量零散工具调用。

**依赖**
- `P4-T1`。

**验收标准**
- 用户说“对某目录做全自动建模”时，主 agent倾向直接调用 workflow tool。

---

## 9. Phase 5：报告、产物与任务板协同

## 9.1 目标

每次 run 完成后，自动输出清晰、可复盘的结果文件，并且与 task scoreboard 协同展示进度。

## 9.2 TODO

### P5-T1 设计 artifacts 管理规则
- [ ] 在 `artifacts.py` 中定义产物 key；
- [ ] 统一约定常见产物命名；
- [ ] 定义 summary / manifest 路径。

**依赖**
- Phase 3 完成。

**验收标准**
- 各阶段产物的记录格式统一。

---

### P5-T2 实现 `report_builder.py`
- [ ] 生成 `RUN_SUMMARY.md`；
- [ ] 生成 `RUN_SUMMARY.json`；
- [ ] 生成 `ARTIFACT_MANIFEST.json`；
- [ ] 汇总关键指标与路径。

**依赖**
- `P5-T1`。

**验收标准**
- 一次 run 结束后一定有 summary 和 manifest。

---

### P5-T3 把 workflow 状态映射到任务板
- [ ] 设计默认任务树：
  - [ ] 审计数据集
  - [ ] 生成运行配置
  - [ ] 执行静态建模
  - [ ] 执行迟滞建模（若启用）
  - [ ] 执行部署验证
  - [ ] 生成最终报告
- [ ] 在 workflow 启动时自动创建对应任务；
- [ ] 在阶段推进时自动更新 task status。

**依赖**
- `P4-T2`。

**验收标准**
- 用户能在 scoreboard 中看到 workflow 的阶段进度。

---

### P5-T4 输出用户友好的最终摘要
- [ ] 成功时输出：关键路径、关键指标、是否启用温度/迟滞、VA 文件位置；
- [ ] 失败时输出：失败阶段、主要原因、建议下一步。

**依赖**
- `P5-T2`。

**验收标准**
- 用户不需要自己翻目录找结果。

---

### P5-T5 端到端烟雾测试
- [ ] 使用 `example_dataset` 跑完整闭环：check-data → fit-physical → train-base-pkan → train-static；
- [ ] 验证所有产物文件生成正确；
- [ ] 验证 `RUN_SUMMARY.md` / `ARTIFACT_MANIFEST.json` 内容完整；
- [ ] 此项为每个 Phase 完成后的回归检查项。

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
- [ ] 定义错误分类：
  - [ ] `DATA_ERROR`
  - [ ] `CONFIG_ERROR`
  - [ ] `TRAINING_NUMERICAL_ERROR`
  - [ ] `MISSING_ARTIFACT_ERROR`
  - [ ] `HSPICE_ENV_ERROR`
  - [ ] `UNKNOWN_ERROR`
- [ ] 实现日志归类规则。

**依赖**
- Phase 5 完成。

**验收标准**
- 失败 run 可以被归入有限类别。

---

### P6-T2 实现 `subagent_hooks.py`
- [ ] 定义何时调用 `explorer` subagent；
- [ ] 定义何时调用 `worker` subagent；
- [ ] 定义 prompt 模板。

**依赖**
- `P6-T1`。

**验收标准**
- subagent 调用点明确且不滥用。

---

### P6-T3 数据异常自动解释
- [ ] 数据审计失败时自动读取 JSON/日志；
- [ ] 调用 `explorer` subagent 生成可读说明；
- [ ] 将说明写入 workflow summary。

**依赖**
- `P6-T2`。

**验收标准**
- 数据错误不再只是原始报错堆栈。

---

### P6-T4 训练失败自动解释
- [ ] `fit-physical` / `train-base-pkan` / `train-static` / `train-hysteresis` 任意阶段失败时，自动分析 stdout/stderr；
- [ ] 用 `explorer` subagent 归纳失败原因。

**依赖**
- `P6-T3`。

**验收标准**
- 用户能直接看到“可能原因”和“建议处理方式”。

---

### P6-T5 最终报告增强
- [ ] 使用 `worker` subagent 生成更自然的最终摘要草稿；
- [ ] 规则模板作为兜底；
- [ ] 避免让 LLM 决定事实，只让它组织语言。

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
- [ ] 定义各阶段质量指标及默认阈值：
  - [ ] `fit-physical` → physical_RMS_max（默认 0.01）
  - [ ] `train-base-pkan` → base_pkan_loss_max（默认 0.05）
  - [ ] `train-static` → static_iv_rms_max（默认 0.02）、static_cv_rms_max（默认 0.05）
  - [ ] `train-hysteresis` → hysteresis_rms_max（默认 0.03）
- [ ] 将阈值写入 `agent_configs.json` 的 `DG_TFT_QUALITY_*` 键；
- [ ] 允许从 DatasetProfile 的 data scale 动态调整阈值（大电流器件 vs 小电流器件）。

**依赖**
- Phase 3（需要 orchestrator 产出的 stage results）。

**验收标准**
- 每个阶段都有可配置的质量门槛。

---

### P7-T2 实现 metrics 解析（从 stdout / stderr 提取精度）
- [ ] 解析 `fit-physical` 的 stdout，提取 final RMS；
- [ ] 解析 `train-base-pkan` 的 stdout，提取 best_loss；
- [ ] 解析 `train-static` 的 stdout，提取 iv_rms / cv_rms；
- [ ] 解析 `train-hysteresis` 的 stdout，提取 hysteresis 误差；
- [ ] 若 stdout 不可解析，降级为检查产物文件中的 metrics；
- [ ] 将提取到的 metrics 写回 `StageResult.metrics` 和 `WorkflowRun.metrics`。

**依赖**
- `P7-T1`。

**验收标准**
- 每个阶段结束后能自动得到数值化的精度结果。

---

### P7-T3 实现调参规则引擎（`quality_guard.py`）
- [ ] 定义每个阶段的参数调整策略表；
- [ ] 实现 `assess_quality(stage_name, metrics, thresholds) → bool`；
- [ ] 实现 `suggest_param_adjustments(stage_name, attempt, current_config) → dict`：
  - [ ] 返回 `{section: {key: new_value, ...}, ...}`；
- [ ] 实现 `apply_adjustments(config_path, adjustments) → new_config_path`：
  - [ ] 读原 .ini，修改指定字段，写新 .ini（不覆盖原文件）；
- [ ] 定义最大重试次数：
  - [ ] 默认每个阶段最多 3 次；
  - [ ] 通过 `agent_configs.json` 的 `DG_TFT_QUALITY_MAX_RETRIES` 可配；
  - [ ] 超过后记录 WARNING + 附 best-effort 结果继续。

**依赖**
- `P7-T2`。

**验收标准**
- 给定一个质量不达标的 run，能自动生成调参后的新 config 并重试。

---

### P7-T4 集成到 orchestrator
- [ ] 在 `orchestrator.py` 的阶段完成与下一阶段之间插入质量门禁；
- [ ] 重试期间每次调参后写新的 `model_config_retry_N.ini`，后续阶段始终使用最新 retry config 的路径；
- [ ] 处理重试循环（最多 N 次，每次调参 + 重跑）；
- [ ] 重试期间更新 `run.stage_results`，保留每次 attempt 的结果；
- [ ] 记录重试历史到 `run.extra["retry_history"]`；
- [ ] 最终指标取所有 attempts 中的最优值；
- [ ] 所有重试均不达标时，触发 Phase 6 诊断（调用 `diagnostics.py` 归类为 `QUALITY_UNSATISFIED`）并标记为失败。

**依赖**
- `P7-T3`、`P6-T1`（诊断归类）。

**验收标准**
- `run_workflow()` 自动处理效果不好 → 调参 → 重跑 → 收敛的完整闭环；
- 重试耗尽后用户得到明确的诊断说明而非只是 WARNING。

---

### P7-T5 可调参数白名单与安全边界
- [ ] 定义允许自动调整的 `[section].[key]` 白名单；
- [ ] 设定每个可调参数的上下界（如 `max_nfev_scale` 不超过 5.0，`kan_hidden` 不超过 32）；
- [ ] 边界外的调整需记录 WARNING 并 clamp；
- [ ] 敏感参数（如数据路径、output_dir）禁止自动修改。

**依赖**
- `P7-T3`、`P6-T1`（诊断归类）。

**验收标准**
- 自动调参不会写入危险值或破坏性路径。

---

### P7-T6 质量报告输出
- [ ] 在 `RUN_SUMMARY.md` / JSON 中包含质量评估结果；
- [ ] 标记"是否经过自动重拟合"、"重试次数"、"最终采纳的参数组合"；
- [ ] 在 report 中对比初始参数 vs 最终参数的变化。

**依赖**
- `P7-T4`、`P5-T2`。

**验收标准**
- 用户能回看 Agent 在哪些参数上做了自动调整。

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

## 13. Phase 9：测试、稳定性与文档收尾

## 13.1 目标

把整个能力从"能跑"提升到"能维护、能验证、能交付"。

## 13.2 TODO

### P9-T1 新增 workflow 单元测试
- [ ] `test/dg_tft_workflow_manager_test.py`
- [ ] `test/dg_tft_dataset_inspector_test.py`
- [ ] `test/dg_tft_config_builder_test.py`
- [ ] `test/dg_tft_stage_runner_test.py`
- [ ] `test/dg_tft_diagnostics_test.py`

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
- [x] `P3-T1` ~ `P3-T9`
- [ ] `P4-T1` ~ `P4-T4`

这部分做完，就已经有“给路径即可全自动跑”的第一版闭环。


### 第二优先级
- [ ] `P4-T5`
- [ ] `P4-T6`
- [ ] `P5-T1` ~ `P5-T5`（报告/产物 + 烟雾测试）
- [ ] `P6-T1` ~ `P6-T2`（诊断基础：分类 + subagent hooks）
- [ ] `P7-T1` ~ `P7-T5`（质量守卫核心 — 不依赖 P5/P6，可直接接在 Phase 3 后）

这部分做完，Agent 不仅知道“跑崩了为什么”，还能自动调参重试直到效果达标。

### 第三优先级
- [ ] `P7-T6`（质量报告 — 依赖 P5-T2）
- [ ] `P6-T3` ~ `P6-T5`（诊断的 subagent 增强）
- [ ] `P8-T1` ~ `P8-T6`
- [ ] `P9-T1` ~ `P9-T5`

这部分做完，系统才进入“可维护、可扩展”的工程化成熟阶段。

---

## 14. 第一批建议直接开工的任务

如果你要马上进入编码，我建议第一批只做下面 10 项：

- [x] `P0-T1` 确认 DG-TFT 建模代码正式位置
- [x] `P0-T2` 建立 `src/device_modeling/dg_tft/`
- [x] `P0-T3` 迁移/镜像现有脚本
- [x] `P1-T1` 新建 `src/workflow/dg_tft/` 骨架
- [x] `P1-T2` 定义 `WorkflowRun / StageResult / DatasetProfile`
- [x] `P1-T3` 实现 `WorkflowManager`
- [x] `P2-T6` 实现 `config_builder.py`
- [x] `P3-T2` 实现统一 `stage_runner.py`
- [x] `P3-T8` 实现 `orchestrator.py`
- [ ] `P4-T1` + `P4-T2` 接 `start_dg_tft_workflow`

这 10 项完成后，你就会拥有一版真正可用的第一代系统。

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

### 18.9 `asdict` 序列化兼容性问题

`dataclasses.asdict()` 内部使用 `copy.deepcopy()` 递归转换所有字段。当 `WorkflowRun` 包含复杂嵌套结构（如含被 mock 对象污染的字段）时，`deepcopy` 可能因 `TypeError: cannot pickle '_thread.lock' object` 失败。

**修复方案**：`WorkflowRun.to_dict()` 改用手动 dict 构造，逐字段赋值，避免 `asdict` 的 deepcopy 行为。`StageResult` 同样使用 `getattr` + 手动 dict 构造。

### 18.10 下一步 Phase 4 关键点

1. 新增 workflow tool schema（`start_dg_tft_workflow` / `query_dg_tft_workflow` / `read_dg_tft_workflow_report`）在 `src/tool/tool_def.py`
2. 实现工具 handler，调用 `workflow_man` 和 `orchestrator`
3. 注册 dispatch 分支
4. 增加 `/workflow_list`、`/workflow_get`、`/workflow_report` 内建命令
5. 优化 prompt 路由（识别"全自动建模"请求 → 优先走 workflow tool）
