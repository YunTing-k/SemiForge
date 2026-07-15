# 模型部署 | Model Deployment

用户调用命令 / The user-facing command is:

```bash
python src/device_modeling/dg_tft/run_model.py export-va
```

验证通过的文件放置在发布级 `verilog_a` 目录下。对于 HSPICE，单一入口为：
Validated files are placed in the release-level `verilog_a` folder. For HSPICE, the single entry is:

```spice
.include "dg_tft_complete_model.inc"
```

所有五个文件需放置在同一目录下。完整 include 加载静态与动态 Verilog-A 模块，并定义子电路 `dg_tft_complete_model`。
Keep all five files in the same folder. The complete include loads the static and dynamic Verilog-A modules and defines subcircuit `dg_tft_complete_model`.

## 文件清单 | File List

- `dg_tft_static_core.va`：静态 IV/CV 与温度感知器件内核 / static IV/CV and temperature-aware device core
- `dg_tft_dynamic_srh.va`：动态双态 SRH 模块 / dynamic two-state SRH module
- `dg_tft_portable_glue.va`：仿真器无关的电流与状态粘合层 / simulator-neutral current and state glue
- `dg_tft_dynamic_srh.inc`：符号化动态参数表达式 / symbolic dynamic parameter expressions
- `dg_tft_complete_model.inc`：HSPICE 单一入口与集成封装 / single HSPICE entry and integration wrapper

## 其他仿真器 | Other Simulators

对于 Empyrean/Aether 或其他支持 Verilog-A 的仿真器，导入三个 `.va` 文件作为模型源文件。两个 `.inc` 文件为 HSPICE/SPICE 组装封装，非 Verilog-A 源文件。目标仿真器不兼容 HSPICE `.hdl` 语法时，请参阅 `verilog_a/AETHER_DEPLOYMENT_README.md`。
For Empyrean/Aether or another Verilog-A-capable simulator, import the three `.va` files as model sources. The two `.inc` files are HSPICE/SPICE assembly wrappers, not Verilog-A source files. Follow `verilog_a/AETHER_DEPLOYMENT_README.md` when the target simulator does not accept HSPICE `.hdl` syntax.

## 瞬态仿真注意事项 | Transient Simulation Notes

- 显式初始化每个陷阱态节点 / explicitly initialize each trap-state node
- 使用 `maxstep <= min(tau_c, tau_e)/20` / use `maxstep <= min(tau_c, tau_e)/20`
- 保持与 Python 波形仿真相同的预偏置与初始占据惯例 / use the same pre-bias and initial occupancy convention as the Python waveform simulation
- 区分器件本征电容与测量封装增益/偏移及夹具寄生 / distinguish intrinsic device capacitance from measurement-wrapper gain/offset and fixture parasitics

等效性验证位于 `validation` 和 `results/figures/hspice_python_equivalence`。
The equivalence evidence is under `validation` and `results/figures/hspice_python_equivalence`.
