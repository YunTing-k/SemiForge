# DeviceAgent 中 DG-TFT 全自动建模工作流改造方案

> **状态：Phase 0–8 已完成** | 393 tests PASS
> 
> 本文档为原始改造设计（2026-06-25）。方案中描述的"短期 subprocess 包装 → 中期函数化"两阶段已全部落地。
> 当前进展详见 `doc/DG_TFT_Agent全自动建模开发TODO.md`。
> 
> ---

## 1. 目标

把当前 DG-TFT 建模流程改造成“**工作流优先、Agent 辅助**”的自动化系统，使用户只需要提供数据目录，Agent 就可以自动完成：

1. 数据检查与能力判定；
2. 配置生成与输出目录准备；
3. 静态 IV/CV 建模；
4. 温度建模判定与训练；
5. I-t 迟滞建模判定与训练；
6. Verilog-A 导出与部署验证；
7. 结果整理、失败诊断和最终汇报。

这里的核心思想不是继续走传统的“开放式 agent loop 自由探索”，而是：

- **流程顺序固定**；
- **每个阶段的输入/输出固定**；
- **是否调用 subagent、调用哪个 subagent、何时回退到规则逻辑**由主工作流控制器决定；
- LLM/Agent 主要负责“柔性决策、诊断、总结、异常解释”，不负责替代数值训练本身。

---

## 2. 先说一个关键事实

基于当前仓库扫描结果：

- `src/` 下目前能明确看到的是 **DeviceAgent 的通用 agent 框架**；
- DG-TFT 建模代码当前仍能明确找到于：
  - `ref/DG_TFT_v90_USER_FRIENDLY_MODELING_RELEASE_20260622/modeling_code/`
- 没有在当前 `src/` 扫描结果中看到完整的 `validate_dataset.py` / `fit_physical_kernel.py` / `train_base_pkan.py` / `dg_tft_full_model.py` / `analyze_it_hysteresis.py` 这些迁移后的正式位置。

因此，本方案分为两层（✅ 两阶段均已完成）：

### 2.1 短期可落地路径 ✅
先**包装现有脚本式代码**，尽快实现"给路径即可全自动跑"。

### 2.2 中期正确路径 ✅（已通过 Phase 8 实现）
再把 DG-TFT 建模代码从"脚本集合"重构为"可调用库 + 工作流引擎"。

代码已迁移至正式目录，详见`doc/DG_TFT_Agent全自动建模开发TODO.md`。

---

## 3. 当前流程中的人为干预点

结合 `ref/.../modeling_code/run_model.py` 和各阶段脚本，当前主要人工点有：

1. **人工判断数据是否齐全**
   - 需要自己看 IV/CV/I-t 文件是否在；
   - 需要自己判断温度跨度是否够；
   - 需要自己判断迟滞数据是否可用。

2. **人工编辑配置文件**
   - `model_config.ini` 需要手改 `data_dir` / `hysteresis_data_dir` / `output_dir` / `hspice_exe`；
   - 对新器件需要复制模板配置。

3. **人工决定跑哪些阶段**
   - 是只跑静态还是跑全流程；
   - 迟滞数据不足时要人工关掉 `hysteresis.enabled`；
   - HSPICE 是否可用也要人工判断。

4. **人工处理异常**
   - 数据格式不合规时，需要人打开日志判断是缺 sheet、缺列还是命名不对；
   - 某阶段失败时，要人工看 stdout/stderr 或 JSON 输出再重跑。

5. **人工整理交付物**
   - 结果图、JSON、VA 文件、验证记录、最终说明需要人工汇总；
   - 用户最终看到的结果不够“Agent 化”。

6. **Agent 框架与建模代码未真正耦合**
   - 当前 `DeviceAgent/src` 已经有主 agent、subagent、task scoreboard、background agents；
   - 但 DG-TFT 建模还没有变成 `DeviceAgent` 的一类“正式工作流对象”。

---

## 4. 目标形态

### 4.1 用户视角
用户只需要说一句：

```text
对 C:/xxx/yyy 数据目录做 DG-TFT 全自动建模
```

系统自动完成：

- 检查数据；
- 自动判断是否有温度数据、是否有迟滞数据；
- 自动构造本次 run 的配置；
- 自动执行静态/动态/验证阶段；
- 自动输出最终结论：
  - 成功/失败；
  - 每阶段产物；
  - 最终模型路径；
  - Verilog-A 路径；
  - 关键精度指标；
  - 失败原因和下一步建议。

### 4.2 系统视角
系统不再依赖“LLM 临场决定下一步”，而是由一个**固定状态机**驱动：

```text
接收请求
→ 建立 workflow run
→ 数据审计
→ 生成配置
→ 运行静态阶段
→ 判定并运行动态阶段
→ 部署验证
→ 打包结果
→ 自动总结
```

LLM / subagent 只在以下节点介入：

- 数据异常解释；
- 含糊情况判定；
- 失败日志归因；
- 结果摘要撰写；
- 对用户的自然语言反馈。

---

## 5. 推荐总体架构

建议采用 **Workflow Engine + Agent Hooks** 架构。

### 5.1 三层结构

#### A. 数值建模层
负责真正的训练和导出。

建议目录：

```text
src/device_modeling/dg_tft/
    config/
    engine/
    dynamic/
    deployment/
    reporting/
```

来源：把当前 `ref/.../modeling_code/` 里的内容迁移或镜像到这里。

#### B. 工作流编排层
负责固定阶段顺序、输入输出、失败回退和状态持久化。

建议目录：

```text
src/workflow/dg_tft/
    schema.py
    manager.py
    orchestrator.py
    stages.py
    artifacts.py
    dataset_inspector.py
    config_builder.py
    stage_runner.py
    diagnostics.py
    report_builder.py
    subagent_hooks.py
```

#### C. Agent 接入层
负责把工作流暴露给主 agent / subagent / 命令行 / Tool schema。

涉及：

- `src/tool/tool_def.py`
- `src/tool/tool_dispatch.py`
- `src/context/agent_context.py`
- `src/utility/command.py`
- `src/main.py`

---

## 6. Python 解释器架构

建议明确采用**双解释器分离架构**，不要默认 Agent 和建模脚本共用一个 Python。

### 6.1 当前解释器分工

#### Agent 解释器
- 环境名：`AGI`
- 路径：`G:/Anaconda3/envs/AGI/python.exe`
- 角色：运行 `DeviceAgent` 主程序、工具系统、subagent、task scoreboard、会话管理等
- 备注：这也是当前环境变量里的默认 `python`

#### 建模解释器
- 环境名：`CUDA`
- 路径：`G:/Anaconda3/envs/CUDA/python.exe`
- 角色：运行 DG-TFT 建模脚本，包括静态训练、KAN 训练、动态训练、数据校验等

### 6.2 为什么必须分离

1. Agent 依赖和建模依赖不同：
   - Agent 更偏交互、LLM、终端工具链；
   - 建模更偏 `torch`、`pykan`、科学计算和 Excel 处理。

2. 建模环境会涉及 CUDA、GPU、Torch 版本，稳定性要求更高。

3. 即使未来两个环境偶然一致，架构上也不应该依赖这种偶然一致。

### 6.3 工作流执行原则

工作流引擎调用建模阶段时：

- **不要使用 `sys.executable` 假定与 Agent 同环境**；
- **始终显式指定建模解释器路径**；
- **优先使用绝对路径 `python.exe`**，不要依赖 `conda activate`。

推荐策略：

1. 一级方案：直接调用

```text
G:/Anaconda3/envs/CUDA/python.exe <script> ...
```

2. 二级方案：如后续需要兼容多机，可支持配置项切换为 `conda run -n CUDA python ...`

### 6.4 建议新增配置项

建议在 `config/agent_configs.json` 中增加：

```json
{
  "DG_TFT_AGENT_PYTHON_EXE": "G:/Anaconda3/envs/AGI/python.exe",
  "DG_TFT_MODELING_PYTHON_EXE": "G:/Anaconda3/envs/CUDA/python.exe",
  "DG_TFT_MODELING_CONDA_ENV": "CUDA"
}
```

其中：

- `DG_TFT_AGENT_PYTHON_EXE`：记录 Agent 自身解释器，主要用于文档化和排查；
- `DG_TFT_MODELING_PYTHON_EXE`：工作流真正执行 DG-TFT 建模时使用；
- `DG_TFT_MODELING_CONDA_ENV`：保留作辅助信息，不建议作为首选执行方式。

### 6.5 为什么这件事优先级很高

显式绑定 `DG_TFT_MODELING_PYTHON_EXE` 是当前最值得优先落地的基础设施之一，原因不是它复杂，而是它会影响**所有后续阶段**：

1. 如果 workflow 默认使用 `sys.executable`，那么由于 Agent 当前运行在 `AGI` 环境，后续 `check-data`、`fit-physical`、`train-base-pkan`、`train-static` 很可能会误落到 `AGI` 解释器；
2. 这种错误非常隐蔽，因为用户常常会误以为“我明明已经在 `CUDA` 环境里装好了 torch / pykan”；
3. 一旦解释器选择不稳定，后续每个阶段都可能出现不同类型的环境错误，导致排查成本极高；
4. 与之相比，先把解释器路由固定下来，只需要很少的代码修改，却能为所有阶段提供统一、可复现的执行基础。

一句话说，这件事之所以优先级高，是因为：

- **缺文件只会让某一个阶段失败**；
- **解释器绑定错误会让所有阶段都可能失败，而且失败方式还不一致**。

### 6.6 `stage_runner` 的最低落地要求

在第一版 workflow 中，建议把解释器解析逻辑集中在 `stage_runner.py`，不要分散到各个阶段函数里。最低实现要求如下：

1. 提供一个统一函数，例如：

```python
resolve_modeling_python(ctx) -> str
```

2. 该函数优先返回：

- `ctx.agent_configs["DG_TFT_MODELING_PYTHON_EXE"]`

3. 所有建模阶段统一通过它构造命令：

```python
[modeling_python_exe, run_model_path, action, ...]
```

4. 禁止在具体阶段里直接使用：

- `sys.executable`
- 裸 `python`

5. 每次阶段执行都应在日志或阶段 summary 中记录：

- 本次使用的建模解释器路径；
- 本次 Agent 解释器路径；
- 是否启用 CUDA。

这样做的好处是：

- 解释器选择逻辑只有一个入口；
- 后续切换到其他建模环境时改动最小；
- 测试时可以非常容易验证“到底是不是用的 `CUDA` 解释器”。

---

## 7. 当前建模环境验证结论

截至目前，`G:/Anaconda3/envs/CUDA/python.exe` 的验证结果如下。

### 7.1 已验证通过的依赖

- `torch 2.9.1+cu128`
- `numpy`
- `pandas`
- `scipy`
- `matplotlib`
- `openpyxl`
- `PyYAML`
- `scikit-learn`
- `pykan`

### 7.2 CUDA 与 GPU 状态

- `torch.cuda.is_available() == True`
- 可见 GPU：`NVIDIA GeForce RTX 5070 Ti Laptop GPU`
- CUDA Runtime：`12.8`
- GPU 显存约：`11.94 GB`

### 7.3 KAN 可用性验证结论

已确认：

- `from kan import KAN` 成功；
- `KAN.__init__` 参数签名与现有 DG-TFT 脚本的调用方式兼容；
- `KAN` 可以在 `device='cuda'` 下实例化；
- 模型参数和前向输出都成功驻留在 `cuda:0`；
- 已完成最小 GPU forward benchmark。

结论：

- **pykan 在该 `CUDA` 环境中可用**；
- **KAN 可以实际使用 CUDA 加速**。

### 7.4 DG-TFT 脚本层验证结论

已成功：

- `run_model.py --help`
- `run_model.py show-config`
- `validate_dataset.py` 对示例数据集执行成功
- `run_model.py check-data` 基于示例数据的 smoke 检查成功

### 7.5 当前不是环境问题的阻塞项

在继续向训练阶段推进时，当前发现一个**项目内容层**问题，而不是解释器问题：

- `fit_physical_kernel.py` 需要模板文件：
  - `modeling_code/engine/dg_tft_ekv_level2.va`
- 当前交付包扫描结果中未发现该文件
- 导致 `fit-physical` 无法继续

因此当前判断是：

- **建模解释器环境已经可用**；
- **后续训练阶段若失败，应优先排查项目文件完整性和数据路径，而不是再怀疑 Python 环境**。

---

## 8. 为什么不要直接把整个流程交给自由 agent loop

原因很明确：

1. **数值建模流程是强顺序型任务**
   - 阶段之间前后依赖很强；
   - 不适合让 LLM 每一步“临时发挥”。

2. **失败诊断可以交给 agent，但训练执行不应交给 agent 决策**
   - 训练命令、参数、产物路径应确定；
   - agent 更适合解释失败，而不是决定训练算法顺序。

3. **可复现性要求高**
   - 同一份数据应该触发同一套阶段图；
   - 不能因为 prompt 轻微变化导致流程结构变化。

因此推荐模式是：

- **Workflow 决定做什么**；
- **Subagent 帮助解释为什么、失败在哪里、下一步如何建议用户**。

---

## 9. 目录改造建议

## 9.1 建议新增目录

```text
src/device_modeling/dg_tft/
src/workflow/dg_tft/
```

## 9.2 建议的最终结构

```text
src/
  agent/
  context/
  tool/
  utility/
  device_modeling/
    dg_tft/
      config/
      engine/
      dynamic/
      deployment/
      reporting/
      run_model.py
  workflow/
    dg_tft/
      schema.py
      manager.py
      orchestrator.py
      stages.py
      artifacts.py
      dataset_inspector.py
      config_builder.py
      stage_runner.py
      diagnostics.py
      report_builder.py
      subagent_hooks.py
```

## 9.3 迁移原则 ✅ 已落地

### 方案 A：低风险快速接入 ✅
保留原脚本大体不动，只做路径重定位：

- `ref/.../modeling_code/engine/*.py` → `src/device_modeling/dg_tft/engine/`
- `ref/.../modeling_code/dynamic/*.py` → `src/device_modeling/dg_tft/dynamic/`
- `ref/.../modeling_code/config/*.ini` → `src/device_modeling/dg_tft/config/`
- `ref/.../modeling_code/run_model.py` → `src/device_modeling/dg_tft/run_model.py`

### 方案 B：中期重构 ✅（Phase 8 完成）
把原来 `argparse + print + subprocess` 风格，改成：

- `def run_validate(data_dir, ...) -> dict`
- `def run_fit_physical(**kw) -> Path`
- `def run_train_base_pkan(**kw) -> Path`
- `def run_full_model(args) -> Path`
- `def run_train_srh(**kw) -> Path`
- `def run_export_va(**kw) -> Path`

A → B 渐进路径已完成。函数化在 `run_model.py` 内部实现，外层 AGI→CUDA subprocess 不变。

---

## 10. 需要新增的核心数据结构

## 10.1 `WorkflowRun`
建议新增 `src/workflow/dg_tft/schema.py`：

```python
@dataclass
class WorkflowRun:
    run_id: int
    workflow_type: str
    dataset_dir: str
    hysteresis_dir: str | None
    output_root: str
    status: str
    current_stage: str
    created_at: str
    updated_at: str
    hspice_available: bool
    temperature_enabled: bool
    hysteresis_enabled: bool
    artifacts: dict[str, str]
    metrics: dict[str, float]
    warnings: list[str]
    errors: list[str]
```

## 10.2 `StageResult`

```python
@dataclass
class StageResult:
    ok: bool
    stage: str
    summary: str
    outputs: dict[str, str]
    metrics: dict[str, float]
    warnings: list[str]
    error: str | None = None
```

## 10.3 `DatasetProfile`
用于把“数据目录里到底有什么”结构化：

```python
@dataclass
class DatasetProfile:
    static_ok: bool
    cv_ok: bool
    iv_families: list[str]
    has_temperature_iv: bool
    temperature_span_K: float | None
    has_hysteresis: bool
    hysteresis_workbooks: list[str]
    issues: list[str]
    suggested_hysteresis_dir: str | None
```

---

## 11. 工作流管理器设计

建议新增 `src/workflow/dg_tft/manager.py`，仿照现有：

- `Scoreboard`
- `DesignManager`
- `RunManager`

新增一个 `WorkflowManager`，职责：

1. 创建 workflow run；
2. 记录每个阶段状态；
3. 记录产物路径和指标；
4. 持久化到 session；
5. 支持主 agent 查询进度。

### 11.1 持久化文件建议

在 session 目录增加：

```text
session/<uuid>/workflow_runs.json
```

### 11.2 需要支持的方法

```python
create_run(...)
get_run(run_id)
list_runs()
update_stage(run_id, stage_name, status, ...)
attach_artifact(run_id, key, path)
attach_metric(run_id, key, value)
mark_done(run_id)
mark_failed(run_id, reason)
```

---

## 12. 固定工作流设计

下面是最推荐的固定阶段图。

## 阶段 0：请求归一化

### 输入
- 用户给出的数据路径；
- 可选：输出路径、器件名、是否强制跑迟滞、是否跳过 HSPICE。

### 执行逻辑
- 标准化为绝对路径；
- 自动推导默认输出目录；
- 自动生成本次 run 的工作目录。

### 产物
- `WorkflowRun` 初始记录；
- 工作目录，例如：

```text
outputs/dg_tft_runs/run_0001/
```

---

## 阶段 1：数据审计

### 负责模块
- 规则逻辑：`dataset_inspector.py`
- 必要时：`explorer` subagent

### 执行逻辑
1. 调用现有 `validate_dataset.py` 或其函数化版本；
2. 自动检查：
   - IV 三大族是否齐；
   - CV 是否齐；
   - 是否存在温度列；
   - 温度跨度是否足够；
   - 是否存在 I-t 数据；
   - I-t 是否可解析。
3. 生成 `DatasetProfile`。

### Subagent 介入点
仅当规则判断不明确时调用：

- 文件命名不规范但可能可修复；
- I-t 数据目录不明确；
- 多个候选迟滞目录并存；
- 数据审计失败但日志太长，需要自动生成用户可读解释。

### 产物
- `dataset_profile.json`
- `dataset_validation.json`
- 用户可读摘要

---

## 阶段 2：配置生成

### 负责模块
- `config_builder.py`

### 执行逻辑
自动生成本次 run 的独立配置，不直接修改仓库内的固定 `model_config.ini`。

### 强制原则
**不要覆盖原始模板配置。**

### 做法
从模板生成：

```text
outputs/dg_tft_runs/run_0001/runtime_config/model_config.ini
```

自动填入：

- `data_dir`
- `hysteresis_data_dir`
- `output_dir`
- `hspice_exe`
- `hysteresis.enabled`
- `temperature.mode`
- `deployment.run_hspice`

### 自动判定规则

#### 迟滞是否启用
- 若 `DatasetProfile.has_hysteresis == True` 且可解析，自动启用；
- 否则自动关闭。

#### 温度模型是否启用
- 若存在温度数据且跨度满足阈值，设为 `auto` 或 `on`；
- 否则设为 `off`。

#### HSPICE 是否启用
- 只有在可执行路径存在时启用；
- 否则自动跳过，不视为工作流失败。

### 产物
- `resolved_config.ini`
- `config_resolution.json`

---

## 阶段 3：静态数据检查

### 负责模块
- `stage_runner.py`

### 执行逻辑
先执行：

```text
python modeling_code/run_model.py check-data
```

或函数版 `validate_data(...)`。

### 失败处理
- 若失败，工作流直接进入 `FAILED`；
- 由 `explorer` subagent 自动读取 JSON/日志并给用户输出“哪个文件、哪个 sheet、哪一列有问题”。

---

## 阶段 4：物理内核拟合

### 执行逻辑
运行：

```text
fit-physical
```

### 产物
- `01_physical_kernel/`
- `fitted_params.json`
- 对应图和诊断文件

### 自动检查
- `fitted_params.json` 是否生成；
- 关键参数是否为有限值；
- 物理参数是否落在合理范围。

### 异常时 subagent 作用
不是去决定算法，而是：
- 读取日志；
- 归纳失败模式：数据异常 / 初值问题 / 数值不收敛；
- 输出可读建议。

---

## 阶段 5：基础 Parameter KAN 训练

### 执行逻辑
运行：

```text
train-base-pkan
```

### 自动检查
- `pkan_torch_model.pt` 是否生成；
- metadata 是否齐；
- loss 曲线文件是否存在。

### 建议新增
把这一阶段的关键 summary 写成结构化 JSON，例如：

```json
{
  "epochs_shared": 500,
  "epochs_iv": 700,
  "epochs_cv": 700,
  "best_loss": 0.123,
  "output_model": ".../pkan_torch_model.pt"
}
```

---

## 阶段 6：完整静态模型训练

### 执行逻辑
运行：

```text
train-static
```

### 包含内容
- IV/CV 联合训练；
- 温度建模自动判定；
- 电荷域 residual；
- 符号化蒸馏；
- 静态 Verilog-A 导出；
- 若可用则做 HSPICE smoke/equivalence。

### 自动检查
- 最终指标 JSON 是否生成；
- Verilog-A 静态文件是否生成；
- 如果 HSPICE 可用，检查 `DONE`/`TIMEOUT`/`RUNTIME_ERROR`。

### 建议强制输出
新增 `static_stage_summary.json`，统一收集：

- IV RMS
- CV RMS
- 温度是否启用
- 生成的 `.va` 文件路径
- HSPICE 是否执行
- HSPICE 是否通过

---

## 阶段 7：迟滞建模（可选）

### 启动条件
仅在 `DatasetProfile.has_hysteresis == True` 时进入。

### 执行逻辑
依序执行：

1. `analyze_it_hysteresis.py`
2. `train_srh_top2_regime_banks.py`
3. `export_srh_top2_regime_banked_va.py`

必要时保留 `refine_srh_waveform_end_to_end.py` 作为二级增强阶段，不建议在第一版全自动流程中默认启用。

### 自动检查
- 动态参数 bank 是否生成；
- 动态 `.va` / `.inc` 是否生成；
- Python/HSPICE 等价性 JSON 是否生成。

### 建议新增判定逻辑
I-t 数据分三级：

- `disabled`：无数据或不可用；
- `basic`：能跑教师模型和导出；
- `full`：可以跑完整验证。

这样可以减少“有一点 I-t 数据但不够完整就整个失败”的情况。

---

## 阶段 8：部署验证

### 执行逻辑
根据前面产物自动决定：

- 只静态验证；
- 静态 + 动态完整验证；
- HSPICE 可用时跑真实 smoke；
- HSPICE 不可用时仅做文件完整性和源代码可移植性检查。

### 产物
- `deployment_summary.json`
- `artifact_manifest.json`

---

## 阶段 9：最终打包与总结

### 负责模块
- `report_builder.py`
- `worker` 或 `explorer` subagent（用于总结）

### 自动输出
在 run 根目录生成：

```text
RUN_SUMMARY.md
RUN_SUMMARY.json
ARTIFACT_MANIFEST.json
```

### `RUN_SUMMARY.md` 至少应包含
- 数据路径；
- 本次是否启用温度模型；
- 本次是否启用迟滞模型；
- 静态精度结果；
- 动态精度结果；
- 生成的 Verilog-A 文件；
- HSPICE 验证状态；
- 若失败，失败在哪一阶段、主要原因是什么。

---

## 13. Main agent 与 Subagent 的分工

## 13.1 Main agent 负责

1. 接收用户请求；
2. 建立 workflow run；
3. 创建并更新 task scoreboard；
4. 驱动固定阶段状态机；
5. 把阶段性结果回传给用户；
6. 在必要时调用 subagent；
7. 最终汇总结果。

## 13.2 `scheduler` subagent 负责

只做**任务显式化**，不做训练本身。

建议在 run 启动时自动创建以下任务：

1. 审计数据集
2. 生成运行配置
3. 执行静态建模
4. 执行迟滞建模（若启用）
5. 执行部署验证
6. 生成最终报告

这样用户可以继续通过现有 scoreboard 看到工作流进度。

## 13.3 `explorer` subagent 负责

适合做三件事：

1. **异常解释**
   - 读日志；
   - 输出哪一步坏了；
   - 提炼成用户能看懂的错误原因。

2. **模糊情况分析**
   - 多个可疑迟滞目录时，分析哪个最可能正确；
   - 数据命名不规范时，给出修复建议。

3. **结果总结**
   - 从 JSON/图/日志提炼一段结论。

## 13.4 `worker` subagent 负责

第一版里不要让 `worker` subagent 参与训练逻辑本身。

更适合让它做：

- 汇总本次 run 的交付文件；
- 生成面向用户的报告草稿；
- 生成后处理清单；
- 在 run workspace 下写总结文件。

## 13.5 一个非常重要的边界

**训练阶段的 Python 函数和 subprocess 调用由工作流引擎直接执行，不由 subagent 用自由工具调用 bash 来执行。**

这样可以保证：

- 可复现；
- 少 prompt 漂移；
- 不会因为 subagent 自己改命令参数导致结果不可控。

---

## 14. 需要新增的 Tool，而不是全靠聊天触发

建议新增三类正式工具。

## 14.1 `start_dg_tft_workflow`

### 用途
启动一个全自动建模 run。

### 入参建议

```json
{
  "dataset_dir": "...",
  "output_dir": "... optional ...",
  "device_name": "... optional ...",
  "hysteresis_dir": "... optional ...",
  "if_force_hysteresis": false,
  "if_skip_hspice": false
}
```

### 返回
- `run_id`
- 自动判定出的配置摘要
- 当前状态

## 14.2 `query_dg_tft_workflow`

### 用途
查询单个或全部 workflow run 的状态。

### 返回内容
- 当前阶段；
- 完成度；
- 已产出的关键文件；
- 成功/失败。

## 14.3 `read_dg_tft_workflow_report`

### 用途
读取某个 run 的最终 summary。

### 返回内容
- Markdown 文本；
- 或 JSON 摘要。

---

## 15. 建议新增的内建命令

在 `src/utility/command.py` 增加：

- `/workflow_list`
- `/workflow_get <run_id>`
- `/workflow_report <run_id>`

这样即使不走 LLM，也能在 TUI 中直接查询工作流结果。

---

## 16. 需要修改的现有文件

## 16.1 `src/context/agent_context.py`

### 新增内容
- `workflow_man: WorkflowManager`
- 运行中的后台 workflow 线程注册表（如果你希望工作流后台执行）

### 作用
让 workflow 和现有 design/run manager 一样成为上下文的一部分。

---

## 16.2 `src/main.py`

### 修改点
1. 初始化 `WorkflowManager`；
2. 在 session 恢复时加载 workflow run 状态；
3. 在主循环监听中加入 workflow 完成事件；
4. 在保存 session 时一起保存 workflow 状态。

---

## 16.3 `src/tool/tool_def.py`

### 修改点
新增 tool schema：

- `start_dg_tft_workflow`
- `query_dg_tft_workflow`
- `read_dg_tft_workflow_report`

### 同时新增实现函数
例如：

```python
def start_dg_tft_workflow(arguments, ctx, progress):
    ...
```

---

## 16.4 `src/tool/tool_dispatch.py`

注册上述三个新工具的 dispatch 分支。

---

## 16.5 `src/tool/tool_execute.py`

如果 workflow 采用后台线程执行，需要仿照 background subagent 的机制，增加：

- workflow 后台启动；
- workflow 完成时向主消息流注入结果；
- workflow 状态轮询。

也可以复用现有 background agent 的表现方式，但不建议真正把 workflow 封装成“LLM 自由子代理”。

---

## 16.6 `config/agent_configs.json`

建议新增：

```json
{
  "DG_TFT_WORKFLOW_MAX_CONCURRENT_RUNS": 1,
  "DG_TFT_DEFAULT_OUTPUT_ROOT": "./outputs/dg_tft_runs",
  "DG_TFT_DEFAULT_CONFIG_TEMPLATE": "./src/device_modeling/dg_tft/config/model_config_new_device_template.ini",
  "DG_TFT_ENABLE_BACKGROUND_WORKFLOW": true,
  "DG_TFT_HSPICE_REQUIRED": false,
  "DG_TFT_AGENT_PYTHON_EXE": "G:/Anaconda3/envs/AGI/python.exe",
  "DG_TFT_MODELING_PYTHON_EXE": "G:/Anaconda3/envs/CUDA/python.exe",
  "DG_TFT_MODELING_CONDA_ENV": "CUDA"
}
```

其中：

- `DG_TFT_MODELING_PYTHON_EXE` 是模型训练的解释器；
- `DG_TFT_AGENT_PYTHON_EXE` 主要用于记录和排查，不建议业务逻辑依赖它。
- 函数化在 `run_model.py` 内部实现（Phase 8），无需 `USE_IMPORT_API` 开关。

---

## 16.7 `src/context/prompt.py`

### 修改建议
保持 task reminder 机制，但针对 workflow 请求增加一条隐式约束：

- 当识别到是 DG-TFT 自动建模请求时，优先建议调用 `start_dg_tft_workflow`；
- 而不是让主 agent 再自己拆成一堆 bash / read_file / edit_file 临场操作。

这样可以让用户表面上仍然通过自然语言发起，但内部走的是固定工作流工具。

---

## 17. 建模脚本本身的改造重点

这是整个方案最关键的部分。

## 17.1 第一阶段：先包装，不大改算法代码 ✅ 已完成（Phase 0–3）

目标：最短路径跑通。（已实现 — `stage_runner.py` 通过 subprocess 调用 `run_model.py`）

## 17.2 第二阶段：把脚本改成 API ✅ 已完成（Phase 8）

目标：让工作流引擎直接调用 Python 函数。（已实现 — 7 个 engine/dynamic 脚本均有 `run_*()` importable 入口）

### 推荐把以下模块函数化

#### `validate_dataset.py`
提供：

```python
def validate_dataset(data_dir: Path, it_dir: Path | None, json_out: Path | None = None) -> dict:
    ...
```

#### `fit_physical_kernel.py`
提供：

```python
def fit_physical(data_dir: Path, out_dir: Path, *, seed: int, ...) -> StageResult:
    ...
```

#### `train_base_pkan.py`
提供：

```python
def train_base_pkan(data_dir: Path, baseline_json: Path, out_dir: Path, ...) -> StageResult:
    ...
```

#### `dg_tft_full_model.py`
提供：

```python
def train_full_model(data_dir: Path, model_dir: Path, out_dir: Path, ...) -> StageResult:
    ...
```

#### `analyze_it_hysteresis.py`
提供：

```python
def analyze_hysteresis(it_dir: Path, static_data_dir: Path, out_dir: Path) -> StageResult:
    ...
```

### 改造原则
- 保留 CLI 入口；
- CLI 只是包装函数；
- 真正逻辑放在 importable function 里。

这是后续自动化的最佳基础。

---

## 18. 关键的自动判定逻辑

## 18.1 数据目录自动识别

建议支持三种输入：

1. **标准目录**：IV/CV/I-t 已按规范放好；
2. **只有静态目录**：自动关闭迟滞阶段；
3. **混合目录**：自动识别子目录中的 I-t 数据。

建议规则顺序：

1. 若用户显式给 `hysteresis_dir`，优先用它；
2. 否则在 `dataset_dir` 内寻找 I-t 文件；
3. 否则寻找 `dataset_dir/hysteresis_it`；
4. 否则找同级目录中名称包含 `it`, `hysteresis` 的文件夹；
5. 都找不到则关闭迟滞。

## 18.2 温度模型自动启用

基于现有数据检查结果：

- 若 IV 数据中存在温度列，且温度跨度达到阈值，启用温度模型；
- 否则显式关闭。

## 18.3 HSPICE 自动启用

建议规则：

- `hspice_exe` 路径存在且可执行 → 启用；
- 否则自动 skip；
- 不能因为 HSPICE 不可用就判整个 run 失败。

## 18.4 失败后的自动诊断

建议将失败分为：

- `DATA_ERROR`
- `CONFIG_ERROR`
- `TRAINING_NUMERICAL_ERROR`
- `MISSING_ARTIFACT_ERROR`
- `HSPICE_ENV_ERROR`
- `UNKNOWN_ERROR`

由 `diagnostics.py` 统一归类，再让 subagent 输出自然语言说明。

---

## 19. 任务板与工作流状态如何协同

建议不要用 task scoreboard 替代 workflow state。

### 正确做法
- **WorkflowRun** 负责机器态；
- **Task scoreboard** 负责用户可见进度态。

### 具体映射建议

| Workflow stage | 对应 Task |
|---|---|
| dataset_audit | 审计数据集 |
| config_resolve | 生成运行配置 |
| static_train | 执行静态建模 |
| hysteresis_train | 执行迟滞建模 |
| deployment_validate | 执行部署验证 |
| final_report | 生成最终报告 |

这样用户能继续利用现有任务系统观察进度，而工作流引擎本身不受任务板显示逻辑影响。

---

## 20. 推荐的实现顺序（✅ 全部完成）

本方案的 Phase 编号与 `TODO.md` 不同。对应关系：
- 方案 Phase 1 ≈ TODO Phase 0–4
- 方案 Phase 2 ≈ TODO Phase 5–6
- 方案 Phase 3 ≈ TODO Phase 8

```text
Phase 1 ✅ → Phase 2 ✅ → Phase 3 ✅
```
详细进展见 `doc/DG_TFT_Agent全自动建模开发TODO.md`。

## Phase 4：把 subagent 介入做精细化

### 需要完成
1. 数据模糊判定由 `explorer` subagent 辅助；
2. 最终报告由 `worker` subagent 生成草稿；
3. 失败复盘由 `explorer` subagent 分析日志并总结；
4. `scheduler` subagent 自动创建任务树。

---

## 21. 测试方案

## 21.1 单元测试

建议新增：

```text
test/dg_tft_workflow_manager_test.py
test/dg_tft_dataset_inspector_test.py
test/dg_tft_config_builder_test.py
test/dg_tft_stage_runner_test.py
test/dg_tft_diagnostics_test.py
```

重点验证：

- 数据目录识别；
- 迟滞启用/关闭逻辑；
- 配置模板填充；
- 阶段失败时的状态推进；
- 产物 manifest 是否完整。

## 21.2 集成测试

建议用三套数据：

1. **仅静态示例数据**
   - 应该成功跑到静态完成；
   - 自动关闭迟滞。

2. **静态 + I-t 完整数据**
   - 应该成功跑完整动态流程；
   - 应生成完整 VA 及验证结果。

3. **故意损坏的数据**
   - 应在数据检查阶段失败；
   - 应输出明确诊断。

## 21.3 Agent 集成测试

建议新增：

```text
test/dg_tft_workflow_tool_test.py
test/dg_tft_workflow_background_test.py
```

重点验证：

- 主 agent 是否优先走 workflow tool；
- background workflow 完成后是否能回注消息；
- task scoreboard 是否跟随更新。

---

## 22. 风险与规避策略

## 22.1 风险：现有建模脚本耦合太深

### 问题
很多脚本现在是“单文件大脚本”，直接重构成 API 有风险。

### 策略 ✅ 已完成
- 第一阶段先 subprocess 包装（Phase 0–3）；
- 第二阶段再逐个函数化（Phase 8）。

## 22.2 风险：I-t 数据命名和格式波动大

### 策略
- 在 `dataset_inspector.py` 中单独做更强的目录发现和 sheet 兼容；
- 不把所有命名差异压给主训练脚本。

## 22.3 风险：HSPICE 依赖外部环境

### 策略
- 把 HSPICE 视为“增强验证”，不是主流程硬依赖；
- 缺失时自动降级，不阻塞静态/动态建模主流程。

## 22.4 风险：Agent 参与过深导致不可复现

### 策略
- Agent 只参与解释、判断、总结；
- 阶段执行始终由规则工作流控制。

## 22.5 风险：Agent 环境与建模环境耦合错误

### 问题
如果 workflow 默认用 `sys.executable`，很容易误用 `AGI` 环境执行 DG-TFT 数值训练。

### 策略
- 工作流阶段执行必须显式读取 `DG_TFT_MODELING_PYTHON_EXE`；
- 测试中增加“确认运行解释器路径”的检查；
- 日志中打印当前调用的建模解释器路径。

## 22.6 风险：交付包文件不完整

### 问题
当前验证中已经发现 `fit_physical_kernel.py` 依赖的 `dg_tft_ekv_level2.va` 模板文件缺失。

### 策略
- 把“模板/基线/静态文件完整性检查”前置到 workflow 启动阶段；
- 对缺失文件给出明确错误类型：`MISSING_ARTIFACT_ERROR`；
- 不把这类问题误判为 Python 环境问题。

---

## 23. 我最推荐的落地版本

如果你想在最短路径内拿到可用版本，我建议按下面这个组合：

### 第一批真正去改的内容

1. **先确认 DG-TFT 建模代码在 `src` 中的正式落点**
   - 若还没迁移完整，先迁移到：
     - `src/device_modeling/dg_tft/...`

2. **新增固定工作流引擎**
   - `src/workflow/dg_tft/manager.py`
   - `src/workflow/dg_tft/orchestrator.py`
   - `src/workflow/dg_tft/dataset_inspector.py`
   - `src/workflow/dg_tft/config_builder.py`
   - `src/workflow/dg_tft/stage_runner.py`
   - `src/workflow/dg_tft/report_builder.py`

3. **新增三个工具**
   - `start_dg_tft_workflow`
   - `query_dg_tft_workflow`
   - `read_dg_tft_workflow_report`

4. **第一版直接 subprocess 包装 `run_model.py`**
   - 不急着改训练算法实现。

5. **把任务板接上**
   - 用户能看到流程任务进度。

6. **增加失败诊断与最终报告**
   - 让 Agent 真正体现价值。

### 这样做的结果
你会得到一个非常接近目标的系统：

- 用户只给路径；
- Agent 自动启动固定工作流；
- Main agent 负责协调；
- Subagent 负责解释和总结；
- 数值流程仍保持可控和可复现。

---

## 24. 最终建议

如果从工程收益和风险平衡来看，最佳策略不是“让 LLM 接管整个建模流程”，而是：

### 最终原则

1. **把 DG-TFT 流程固化成工作流状态机** ✅；
2. **把 subagent 变成工作流里的柔性插件** ✅；
3. **先包住旧脚本，再逐步函数化** ✅；
4. **把用户可见进度统一投射到 console + scoreboard** ✅；
5. **把最终交付统一收敛为 manifest + summary** ✅。

这套方案是可操作的，而且能和你现在 `DeviceAgent` 里已有的：

- task scoreboard
- background subagent
- tool schema
- session persistence
- Rich TUI

自然接上，不需要推翻现有 Agent 框架。

---

## 25. 可直接执行的下一步清单

建议你下一步按这个顺序实施：

1. 核对 DG-TFT 建模代码在 `src` 的实际位置；
2. 新建 `src/workflow/dg_tft/` 目录；
3. 实现 `DatasetProfile` / `WorkflowRun` / `WorkflowManager`；
4. 实现 `config_builder.py`，先把“自动生成 runtime config”做出来；
5. 实现 `stage_runner.py`，先通过 `DG_TFT_MODELING_PYTHON_EXE` 顺序调用 `check-data` → `fit-physical` → `train-base-pkan` → `train-static`；
6. 再接 `start_dg_tft_workflow` 工具；
7. 最后补 subagent 诊断和总结。

如果你后续愿意，我建议下一步直接做两件事：

1. 我先帮你把这份方案继续细化成**具体到文件/类/函数级别的开发 TODO**；
2. 或者我直接开始在 `src` 里给你搭 **第一版工作流骨架代码**。
