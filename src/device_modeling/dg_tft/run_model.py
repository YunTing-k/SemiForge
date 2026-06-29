from __future__ import annotations

import argparse
import configparser
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd


CODE_ROOT = Path(__file__).resolve().parent
RELEASE_ROOT = CODE_ROOT.parent
ENGINE = CODE_ROOT / "engine"
DYNAMIC = CODE_ROOT / "dynamic"
DEFAULT_CONFIG = CODE_ROOT / "config" / "model_config.ini"
EXAMPLE_DATA_ROOT = (RELEASE_ROOT / "data_templates" / "example_dataset").resolve()

# Add engine and dynamic to sys.path so direct imports work internally.
# (The AGI→CUDA subprocess boundary is preserved by stage_runner.py.)
def _ensure_import_paths() -> None:
    for p in (str(ENGINE), str(DYNAMIC)):
        if p not in sys.path:
            sys.path.insert(0, p)


_ensure_import_paths()


def resolve_path(value: str, config_path: Path) -> Path:
    path = Path(os.path.expandvars(os.path.expanduser(value)))
    if not path.is_absolute():
        path = (config_path.parent / path).resolve()
    return path


def yes(cfg: configparser.ConfigParser, section: str, option: str) -> bool:
    return cfg.getboolean(section, option)


def _log_stage(label: str) -> None:
    """Print a visible stage marker to stdout (captured by outer subprocess)."""
    print(f"\n=== {label} ===", flush=True)


def load_config(path: Path) -> configparser.ConfigParser:
    cfg = configparser.ConfigParser()
    if not path.exists():
        raise FileNotFoundError(f"Configuration file does not exist: {path}")
    cfg.read(path, encoding="utf-8")
    return cfg


def paths(cfg: configparser.ConfigParser, config_path: Path) -> dict[str, Path]:
    output = resolve_path(cfg["paths"]["output_dir"], config_path)
    return {
        "data": resolve_path(cfg["paths"]["data_dir"], config_path),
        "it": resolve_path(cfg["paths"]["hysteresis_data_dir"], config_path),
        "output": output,
        "physical": output / "01_physical_kernel",
        "base_pkan": output / "02_base_pkan",
        "full": output / "03_full_model",
        "dynamic": output / "04_hysteresis",
    }


def _normalize_cv_format(data_dir: Path) -> None:
    """Convert shared-bias CV layout to paired-column format expected by the engine.

    Raw CV files often use: col0=bias, col1=cap1, col2=cap2, ...
    Engine expects:       col0=bias1, col1=cap1, col2=bias2, col3=cap2, ...
    """
    cv_path = data_dir / "CV_train.xlsx"
    if not cv_path.exists():
        return
    try:
        sheets = pd.read_excel(cv_path, sheet_name=None, header=None)
    except Exception:
        return
    rewritten = False
    result: dict[str, pd.DataFrame] = {}
    for sheet_name, df in sheets.items():
        if df.shape[0] < 3 or df.shape[1] < 3:
            result[sheet_name] = df
            continue
        col0_label = str(df.iloc[1, 0]).lower()
        col1_label = str(df.iloc[1, 1]).lower() if df.shape[1] > 1 else ""
        col2_label = str(df.iloc[1, 2]).lower() if df.shape[1] > 2 else ""
        if "bias" not in col0_label or "bias" in col1_label or "bias_node_centered" in col0_label or ("bias" in col0_label and "bias" not in col1_label and "bias" in col2_label):
            result[sheet_name] = df
            continue
        rewritten = True
        desc = str(df.iloc[0, 0])
        bias_raw = pd.to_numeric(df.iloc[2:, 0], errors="coerce").values.astype(float)
        full_desc = desc + " | corrected coordinate, smoothed capacitance only"
        header_pairs: list[str] = []
        data_cols: list[pd.Series] = []
        for c in range(1, df.shape[1]):
            cap_label = str(df.iloc[1, c])
            cap_label_clean = re.sub(r'\s*\(F\)\s*$', '', cap_label, flags=re.I)
            cap_raw = pd.to_numeric(df.iloc[2:, c], errors="coerce").values.astype(float)
            header_pairs.extend([
                f"Bias_node_centered @ {cap_label_clean} (V)",
                f"Capacitance_smoothed @ {cap_label_clean} (F)",
            ])
            data_cols.extend([pd.Series(bias_raw), pd.Series(cap_raw)])
        out = pd.concat(data_cols, axis=1)
        out.columns = header_pairs
        out.index = range(2, 2 + len(bias_raw))
        desc_series = pd.Series([full_desc] + [np.nan] * (out.shape[1] - 1), index=out.columns, name=0)
        header_series = pd.Series(header_pairs, index=out.columns, name=1)
        out = pd.concat([pd.DataFrame([desc_series]), pd.DataFrame([header_series]), out])
        result[sheet_name] = out
    if not rewritten:
        return
    with pd.ExcelWriter(cv_path, engine="openpyxl") as writer:
        for name, df_out in result.items():
            df_out.to_excel(writer, sheet_name=name[:31], index=False, header=False)


# ── Stage functions (functionized — direct imports, no inner subprocess) ──

def validate_data(cfg, config_path, p) -> None:
    from validate_dataset import run_validate

    _log_stage("check-data: normalize CV + validate dataset")
    _normalize_cv_format(p["data"])
    p["output"].mkdir(parents=True, exist_ok=True)
    run_validate(
        data_dir=p["data"],
        it_dir=p["it"] if p["it"].exists() else None,
        json_out=p["output"] / "dataset_validation.json",
    )


def fit_physical(cfg, p) -> None:
    from fit_physical_kernel import run_fit_physical

    section = cfg["physical_fit"]
    p["physical"].mkdir(parents=True, exist_ok=True)
    max_iv = int(section["max_iv_points"])
    max_cv = int(section["max_cv_points"])
    max_nfev = float(section["max_nfev_scale"])
    if yes(cfg, "runtime", "minimal_run"):
        max_iv, max_cv, max_nfev = 1000, 500, 0.1

    _log_stage("fit-physical: physical kernel extraction")
    run_fit_physical(
        data_dir=p["data"],
        out_dir=p["physical"],
        max_iv_points=max_iv,
        max_cv_points=max_cv,
        max_nfev_scale=max_nfev,
        seed=int(cfg["runtime"]["seed"]),
    )


def train_base_pkan(cfg, p) -> None:
    from train_base_pkan import run_train_base_pkan

    section = cfg["base_pkan"]
    baseline = p["physical"] / "fitted_params.json"
    if not baseline.exists():
        raise FileNotFoundError(
            f"{baseline} is missing. Run `python run_model.py fit-physical` first."
        )
    p["base_pkan"].mkdir(parents=True, exist_ok=True)

    _log_stage("train-base-pkan: base PKAN training")
    run_train_base_pkan(
        data_dir=p["data"],
        baseline_json=baseline,
        out_dir=p["base_pkan"],
        device=cfg["runtime"]["device"],
        dtype=cfg["runtime"]["dtype"],
        seed=int(cfg["runtime"]["seed"]),
        epochs_shared=int(section["epochs_shared"]),
        epochs_iv=int(section["epochs_iv"]),
        epochs_cv=int(section["epochs_cv"]),
        kan_hidden=int(section["kan_hidden"]),
        kan_grid=int(section["kan_grid"]),
        kan_k=int(section["kan_k"]),
        batch_iv=int(section["batch_iv"]),
        batch_cv=int(section["batch_cv"]),
    )


# ── Model architecture defaults (used when [base_pkan] section is absent) ──
_FALLBACK_KAN_HIDDEN = 12
_FALLBACK_KAN_GRID = 5
_FALLBACK_KAN_K = 3
_FALLBACK_ETA_N = 0.60


def _prepare_fallback_model_dir(cfg: configparser.ConfigParser, p: dict[str, Path]) -> None:
    import json

    fp = p["physical"] / "fitted_params.json"
    if not fp.exists():
        raise FileNotFoundError(
            f"{fp} is missing. Run `python run_model.py fit-physical` first."
        )
    physical = json.loads(fp.read_text(encoding="utf-8"))
    section = cfg["base_pkan"] if cfg.has_section("base_pkan") else {}
    meta = {
        "description": "Synthesized from fit-physical — no base PKAN weights.",
        "uses_pykan": True,
        "baseline_params": physical["final_params"],
        "baseline_trap_params": physical.get("cv_shared_trap_params")
        or {},
        "baseline_metrics_summary": physical.get("metrics_summary", {}),
        "pkan_metrics_summary": {},
        "final_kan_scale": float(cfg.get("temperature", "kan_scale", fallback="0.35")),
        "training_args": {
            "kan_hidden": int(section.get("kan_hidden", _FALLBACK_KAN_HIDDEN)),
            "kan_grid": int(section.get("kan_grid", _FALLBACK_KAN_GRID)),
            "kan_k": int(section.get("kan_k", _FALLBACK_KAN_K)),
            "eta_n": _FALLBACK_ETA_N,
            "data_dir": str(p["data"]),
        },
    }
    p["base_pkan"].mkdir(parents=True, exist_ok=True)
    (p["base_pkan"] / "pkan_model_metadata.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    try:
        import torch
    except ImportError:
        raise RuntimeError(
            "train-static requires PyTorch to synthesize the fallback model_dir. "
            "Ensure the modeling Python environment has torch installed."
        )
    torch.save({}, p["base_pkan"] / "pkan_torch_model.pt")


def _make_full_model_args(cfg: configparser.ConfigParser, p: dict[str, Path]) -> argparse.Namespace:
    """Build a Namespace for run_full_model from INI config."""
    from dg_tft_full_model import default_args

    section = cfg["full_model"]
    args = default_args()

    # Paths / device
    args.data_dir_override = p["data"]
    args.model_dir = p["base_pkan"]
    args.out_dir = p["full"]
    args.device = cfg["runtime"]["device"]
    args.dtype = cfg["runtime"]["dtype"]
    args.seed = int(cfg["runtime"]["seed"])

    # Core training
    args.epochs = int(section["epochs"])
    args.lr = float(section["learning_rate"])
    args.batch_iv = int(section["batch_iv"])
    args.batch_cv = int(section["batch_cv"])
    args.cv_weight = float(section["cv_weight"])
    args.cv_residual_level = int(section["cv_residual_level"])
    args.early_stop_patience = int(section["early_stop_patience"])
    args.early_stop_min_epochs = int(section["early_stop_min_epochs"])
    args.symbolic_training_weight = float(section["symbolic_training_weight"])
    args.print_every = int(section.get("print_every", "80"))

    # Temperature
    args.temperature_kan_min_span_K = float(cfg["temperature"]["minimum_span_K"])
    args.temperature_kan_scale = float(cfg["temperature"]["kan_scale"])
    args.hspice_exe = cfg["paths"].get("hspice_exe", "")

    # CV residual
    args.enable_cv_residual = yes(cfg, "full_model", "enable_cv_residual")

    # Finetune
    args.finetune_base_pkan = yes(cfg, "full_model", "finetune_base_pkan")

    # Temperature mode
    mode = cfg["temperature"]["mode"].strip().lower()
    if mode == "on":
        args.enable_temperature_kan = True
        args.auto_enable_temperature_kan = False
    elif mode == "off":
        args.enable_temperature_kan = False
        args.auto_enable_temperature_kan = False
    else:
        args.enable_temperature_kan = False
        args.auto_enable_temperature_kan = True

    # HSPICE
    if not yes(cfg, "deployment", "run_hspice"):
        args.skip_hspice_smoke = True
        args.skip_hspice_grid_equivalence = True
        args.skip_hspice_cv_grid_equivalence = True

    # Minimal run
    if yes(cfg, "runtime", "minimal_run"):
        args.quick_run = True
        args.epochs = 1
        args.early_stop_patience = 1
        args.early_stop_min_epochs = 1

    return args


def train_full_model(cfg, p) -> None:
    from dg_tft_full_model import apply_quick_run_preset, run_full_model

    model_dir = p["base_pkan"]
    if not (model_dir / "pkan_torch_model.pt").exists():
        _prepare_fallback_model_dir(cfg, p)
    p["full"].mkdir(parents=True, exist_ok=True)

    _log_stage("train-static: full model training")
    args = _make_full_model_args(cfg, p)
    args = apply_quick_run_preset(args)
    run_full_model(args)


def _setup_dynamic_env(p: dict[str, Path]) -> None:
    """Set environment variables for dynamic/hysteresis stage imports."""
    os.environ["DG_TFT_IT_DATA_DIR"] = str(p["it"])
    os.environ["DG_TFT_STATIC_DATA_DIR"] = str(p["data"])
    os.environ["DG_TFT_DYNAMIC_WORK_DIR"] = str(p["dynamic"])


def train_hysteresis(cfg, p) -> None:
    if not p["it"].exists():
        raise FileNotFoundError(f"Hysteresis data folder does not exist: {p['it']}")
    p["dynamic"].mkdir(parents=True, exist_ok=True)
    _setup_dynamic_env(p)

    from analyze_it_hysteresis import run_analyze_it

    _log_stage("train-hysteresis: analyze I-t data")
    run_analyze_it()

    from train_srh_top2_regime_banks import run_train_srh

    section = cfg["hysteresis"]
    epochs = int(section["epochs"])
    hidden_capture = int(section["hidden_capture"])
    hidden_emission = int(section["hidden_emission"])
    if yes(cfg, "runtime", "minimal_run"):
        epochs, hidden_capture, hidden_emission = 1, 4, 4

    _log_stage("train-hysteresis: train SRH regime banks")
    run_train_srh(
        epochs=epochs,
        target_regime=section["regime"],
        hidden_capture=hidden_capture,
        hidden_emission=hidden_emission,
        amp_parameterization=section["amp_parameterization"],
        print_every=int(section.get("print_every", "1")),
    )

    from export_srh_top2_regime_banked_va import run_export_va

    _log_stage("train-hysteresis: export Verilog-A")
    run_export_va()


def export_supplied_va() -> None:
    print("Validated Verilog-A files are in:")
    print(f"  {RELEASE_ROOT / 'verilog_a'}")


def refuse_accidental_example_training(
    action: str, data_dir: Path, *, allow_example_data: bool
) -> None:
    training_actions = {
        "fit-physical",
        "train-base-pkan",
        "train-static",
        "train-hysteresis",
        "train-all",
    }
    if action not in training_actions or allow_example_data:
        return
    resolved = data_dir.resolve()
    if resolved == EXAMPLE_DATA_ROOT or EXAMPLE_DATA_ROOT in resolved.parents:
        raise RuntimeError(
            "Training is blocked because data_dir points to the small example dataset. "
            "Set data_dir to the real device dataset, or pass --allow-example-data "
            "only for an intentional smoke test."
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Single entry point for the DG-TFT physics + symbolic KAN model."
    )
    parser.add_argument(
        "action",
        choices=[
            "check-data",
            "fit-physical",
            "train-base-pkan",
            "train-static",
            "train-hysteresis",
            "train-all",
            "export-va",
            "show-config",
        ],
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--allow-example-data",
        action="store_true",
        help="Allow training actions to use data_templates/example_dataset.",
    )
    args = parser.parse_args()
    config_path = args.config.resolve()
    cfg = load_config(config_path)
    p = paths(cfg, config_path)
    refuse_accidental_example_training(
        args.action, p["data"], allow_example_data=args.allow_example_data
    )

    if args.action == "show-config":
        for name, value in p.items():
            print(f"{name:12s}: {value}")
        return 0
    if args.action == "check-data":
        validate_data(cfg, config_path, p)
    elif args.action == "fit-physical":
        fit_physical(cfg, p)
    elif args.action == "train-base-pkan":
        train_base_pkan(cfg, p)
    elif args.action == "train-static":
        train_full_model(cfg, p)
    elif args.action == "train-hysteresis":
        train_hysteresis(cfg, p)
    elif args.action == "train-all":
        validate_data(cfg, config_path, p)
        fit_physical(cfg, p)
        train_base_pkan(cfg, p)
        train_full_model(cfg, p)
        if yes(cfg, "hysteresis", "enabled"):
            train_hysteresis(cfg, p)
    elif args.action == "export-va":
        export_supplied_va()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
