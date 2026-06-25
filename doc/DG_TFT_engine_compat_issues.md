# DG-TFT Engine 兼容性问题汇总

> 由 4 个 explorer agent 扫描 `src/device_modeling/dg_tft/engine/` + `run_model.py` 生成

---

## CRITICAL — 阻塞当前 `train-static` 工作流

### 1. run_model.py 传给 dg_tft_full_model.py 的参数不存在

**文件:** `run_model.py`, lines 174-185

以下参数被传给 `dg_tft_full_model.py`，但该脚本的 `parse_args()` 里没有定义它们：

```python
# run_model.py:174-177
"--early-stop-patience"   # 不在 dg_tft_full_model.py argparse 中
"--early-stop-min-epochs"  # 不在 dg_tft_full_model.py argparse 中

# run_model.py:178-179
"--symbolic-training-weight"  # 不在 dg_tft_full_model.py argparse 中

# run_model.py:184-185
"--hspice-exe"  # 不在 dg_tft_full_model.py argparse 中

# run_model.py:199, 201, 203
"--auto-enable-temperature-kan"  # 不在 dg_tft_full_model.py argparse 中
"--no-auto-enable-temperature-kan"  # 不在 dg_tft_full_model.py argparse 中
```

**后果:** `train-static` / `train-all` 阶段运行 `dg_tft_full_model.py` 时 `argparse` 抛出 `unrecognized arguments`，直接崩溃。

**修复方向:** 从 `run_model.py` 的 `train_full_model()` 中删除这些未识别的参数，或在 `dg_tft_full_model.py` 的 `parse_args()` 中添加对应定义。

---

### 2. _build_and_load_model 旧格式 state_dict 键名映射错误

**文件:** `dg_tft_full_model.py`, `_build_and_load_model()` lines 5597-5622

```python
# 当前代码生成的键名（错误）：
"_kan_shared_center"     # UnifiedIVCVModel 实际是 "kan.center"
"_kan_iv_kan_center"     # 同上
"_kan_cv_kan_center"     # 实际是 "cv_kan.center"
```

另外 buffer 名称也不一致：
- `KANBlock` 用 `output_ranges` (train_base_pkan.py:243)
- `V2StyleKAN` 用 `out_range` (dg_tft_full_model.py:492)

`model.load_state_dict(kan_state, strict=False)` 静默丢弃所有权重，模型从随机初始值开始训练。

**修复方向:** 重写映射逻辑，把 `KANBlock` 的扁平 state dict keys 正确映射到 `UnifiedIVCVModel` 的 `kan.*` / `cv_kan.*` 前缀结构。

---

## HIGH — 环境/可移植性问题

### 3. 裸导入在子进程中可能失败

**文件:** `dg_tft_full_model.py:58-91`, `train_base_pkan.py:28-40`

```python
from fit_physical_kernel import Params, CVTrapParams, ...
from train_base_pkan import KANBlock, ...
```

这些是裸导入（不是 `from .fit_physical_kernel`），依赖 `engine/` 在 `sys.path` 上。`run_model.py:33` 以 `cwd=RELEASE_ROOT` 启动子进程，如果 `PYTHONPATH` 不含 `engine/`，`ModuleNotFoundError`。

**当前环境刚好能跑**是因为环境中已设置 PYTHONPATH，但换机器会崩。

**修复方向:** 在 `run_model.py` 的 `run()` 函数中设置 `env["PYTHONPATH"]` 包含 engine 目录。

---

### 4. 硬编码绝对路径

| 文件 | 行 | 内容 |
|------|-----|------|
| `fit_physical_kernel.py` | 23-31 | `DEFAULT_DATA_DIR = Path("E:/学习/大论文相关材料/器件数据/0521/0521/training_ready_corrected_xlsx")` |
| `model_config.ini` | 3,5,9 | `data_dir`, `hysteresis_data_dir`, `hspice_exe` 全部指向 `E:/` 绝对路径 |
| `model_config_new_device_template.ini` | 6 | `hspice_exe = E:/synopsys/Hspice_S-2021.09/WIN64/hspice.com` |

**修复方向:** 模板文件用占位符（已有 `data_dir = __DATA_DIR__` 模式可扩展），`DEFAULT_DATA_DIR` 改为空或报错提示要求显式传参。

---

### 5. 引擎脚本默认输出目录污染源码树

| 文件 | 默认 `--out-dir` |
|------|-----------------|
| `fit_physical_kernel.py:1224` | `ROOT / "fit_training_ready_outputs"` |
| `train_base_pkan.py:45` | `ROOT / "pkan_ivcv_torch_outputs"` |
| `dg_tft_full_model.py:3383` | `ROOT / "outputs" / "static_training"` |
| `dg_tft_full_model.py:3382` | `DEFAULT_MODEL_DIR = ROOT / "base_physical_pkan"` |

`ROOT = Path(__file__).resolve().parent` = `engine/` 目录。不传 `--out-dir` 就写到源码树里。

**修复方向:** 改为当前工作目录的相对路径，或要求 `--out-dir` 必传。

---

## MEDIUM — 架构不一致

### 6. KANBlock vs V2StyleKAN 是完全不同的架构

| 属性 | KANBlock (train) | V2StyleKAN (load) |
|------|-----------------|-------------------|
| 输出维度 | 任意 `len(output_ranges)` | 固定 `len(CORR_NAMES)` (11 IV) / `len(CV_CORR_NAMES)` (11 CV) |
| 激活函数 | `tanh(normalize) → net → tanh × ranges` | `clamp(normalize) → net → tanh × out_range` |
| normalize | 双 tanh | clamp[-3,3] + tanh |
| 参数 | `hidden`, `output_ranges` (list) | `width` (list), `ranges` (dict) |

`train_base_pkan.py` 用 3 个独立 KANBlock（shared/iv_kan/cv_kan），`dg_tft_full_model.py` 用 `UnifiedIVCVModel` 内含 2 个 V2StyleKAN（kan/cv_kan）+ 物理核 + 寄生网络 + 温度KAN + ...

**两者之间的权重加载在架构上就不可能**——当前 `load_trained_base` 已改成物理参数 + 默认 KAN 初始化的降级方案（commit `c56d3b4`），但基 PKAN 训练产出白费了。

**修复方向 (长远):** 要么让 `train_base_pkan.py` 输出与 `V2StyleKAN` 兼容的格式，要么让 `dg_tft_full_model.py` 加载 `KANBlock` 格式。短期：当前降级方案可接受（至少物理参数正确），`train-static` 从随机 KAN 初始化开始训练。

---

## 修复顺序建议

1. **先修 #1**（unrecognized arguments）— 否则 train-static 无法启动
2. **再修 #3**（PYTHONPATH）— 保证子进程可移植
3. **然后修 #4/#5**（硬编码路径和源码污染）— 模板和默认值
4. **#2** 的映射问题已通过 `c56d3b4` 降级方案绕过，但长期应正确实现
5. **#6** 架构差异需要建模代码开发者介入
