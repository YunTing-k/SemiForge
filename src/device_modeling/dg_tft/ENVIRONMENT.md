# DG-TFT 建模环境说明

## 双解释器分离架构

| 角色 | Conda 环境 | Python 路径 | 用途 |
|------|-----------|-------------|------|
| Agent | `AGI` | `G:/Anaconda3/envs/AGI/python.exe` | 运行 DeviceAgent 主程序、工具系统、subagent |
| 建模 | `CUDA` | `G:/Anaconda3/envs/CUDA/python.exe` | 运行 DG-TFT 建模脚本（静态训练、KAN、动态训练） |

**重要**：两个环境不共享业务依赖。建模阶段执行时必须显式使用 `DG_TFT_MODELING_PYTHON_EXE`，禁止依赖 `sys.executable` 或裸 `python`。

## CUDA 环境已验证依赖

- `torch 2.9.1+cu128`
- `numpy`
- `pandas`
- `scipy`
- `matplotlib`
- `openpyxl`
- `PyYAML`
- `scikit-learn`
- `pykan`

## GPU 状态

- `torch.cuda.is_available() == True`
- GPU: `NVIDIA GeForce RTX 5070 Ti Laptop GPU`
- CUDA Runtime: `12.8`
- VRAM: ~11.94 GB

## KAN 可用性

- `from kan import KAN` 可用
- KAN 支持 `device='cuda'` 实例化与 GPU 前向

## 已验证命令

- `run_model.py --help` ✓
- `run_model.py show-config` ✓
- `run_model.py check-data` (对示例数据) ✓

## 已知阻塞

- `fit_physical_kernel.py` 依赖 `engine/dg_tft_ekv_level2.va` 模板文件（已存在于对应目录）
- 后续训练阶段若失败，优先排查项目文件完整性和数据路径，而非 Python 环境

## 外部依赖

| 依赖 | 必需 | 说明 |
|------|------|------|
| HSPICE | 否 | 增强验证；不可用时自动跳过，不影响主流程 |
