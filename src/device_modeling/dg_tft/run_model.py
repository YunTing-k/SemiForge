from __future__ import annotations

import argparse
import configparser
import os
import re
from pathlib import Path
import shlex
import subprocess
import sys

import numpy as np
import pandas as pd


CODE_ROOT = Path(__file__).resolve().parent
RELEASE_ROOT = CODE_ROOT.parent
ENGINE = CODE_ROOT / "engine"
DYNAMIC = CODE_ROOT / "dynamic"
DEFAULT_CONFIG = CODE_ROOT / "config" / "model_config.ini"
EXAMPLE_DATA_ROOT = (RELEASE_ROOT / "data_templates" / "example_dataset").resolve()


def resolve_path(value: str, config_path: Path) -> Path:
    path = Path(os.path.expandvars(os.path.expanduser(value)))
    if not path.is_absolute():
        path = (config_path.parent / path).resolve()
    return path


def yes(cfg: configparser.ConfigParser, section: str, option: str) -> bool:
    return cfg.getboolean(section, option)


def run(command: list[str], *, env: dict[str, str] | None = None) -> None:
    resolved_env = dict(os.environ)
    if env:
        resolved_env.update(env)
    py_paths = [str(ENGINE)]
    existing = resolved_env.get("PYTHONPATH", "")
    if existing:
        py_paths.append(existing)
    resolved_env["PYTHONPATH"] = os.pathsep.join(py_paths)
    print("+", " ".join(shlex.quote(str(x)) for x in command), flush=True)
    subprocess.run(command, check=True, cwd=RELEASE_ROOT, env=resolved_env)


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
        # Build description row + header row + data rows
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
        # Prepend description row (row 0) and header row (row 1)
        desc_series = pd.Series([full_desc] + [np.nan] * (out.shape[1] - 1), index=out.columns, name=0)
        header_series = pd.Series(header_pairs, index=out.columns, name=1)
        out = pd.concat([pd.DataFrame([desc_series]), pd.DataFrame([header_series]), out])
        result[sheet_name] = out
    if not rewritten:
        return
    with pd.ExcelWriter(cv_path, engine="openpyxl") as writer:
        for name, df_out in result.items():
            df_out.to_excel(writer, sheet_name=name[:31], index=False, header=False)


def validate_data(cfg, config_path, p) -> None:
    _normalize_cv_format(p["data"])
    command = [
        sys.executable,
        str(ENGINE / "validate_dataset.py"),
        "--data-dir",
        str(p["data"]),
        "--json-out",
        str(p["output"] / "dataset_validation.json"),
    ]
    if p["it"].exists():
        command += ["--it-dir", str(p["it"])]
    p["output"].mkdir(parents=True, exist_ok=True)
    run(command)


def fit_physical(cfg, p) -> None:
    section = cfg["physical_fit"]
    p["physical"].mkdir(parents=True, exist_ok=True)
    max_iv = section["max_iv_points"]
    max_cv = section["max_cv_points"]
    max_nfev = section["max_nfev_scale"]
    if yes(cfg, "runtime", "debug_run"):
        max_iv = "1000"
        max_cv = "500"
        max_nfev = "0.1"
    elif yes(cfg, "runtime", "quick_run"):
        max_iv = "3000"
        max_cv = "2000"
        max_nfev = "0.3"
    run(
        [
            sys.executable,
            str(ENGINE / "fit_physical_kernel.py"),
            "--data-dir",
            str(p["data"]),
            "--out-dir",
            str(p["physical"]),
            "--max-iv-points",
            max_iv,
            "--max-cv-points",
            max_cv,
            "--max-nfev-scale",
            max_nfev,
            "--seed",
            cfg["runtime"]["seed"],
        ]
    )


def train_base_pkan(cfg, p) -> None:
    section = cfg["base_pkan"]
    baseline = p["physical"] / "fitted_params.json"
    if not baseline.exists():
        raise FileNotFoundError(
            f"{baseline} is missing. Run `python run_model.py fit-physical` first."
        )
    p["base_pkan"].mkdir(parents=True, exist_ok=True)
    run(
        [
            sys.executable,
            str(ENGINE / "train_base_pkan.py"),
            "--data-dir",
            str(p["data"]),
            "--baseline-json",
            str(baseline),
            "--out-dir",
            str(p["base_pkan"]),
            "--device",
            cfg["runtime"]["device"],
            "--dtype",
            cfg["runtime"]["dtype"],
            "--seed",
            cfg["runtime"]["seed"],
            "--epochs-shared",
            section["epochs_shared"],
            "--epochs-iv",
            section["epochs_iv"],
            "--epochs-cv",
            section["epochs_cv"],
            "--kan-hidden",
            section["kan_hidden"],
            "--kan-grid",
            section["kan_grid"],
            "--kan-k",
            section["kan_k"],
            "--batch-iv",
            section["batch_iv"],
            "--batch-cv",
            section["batch_cv"],
        ]
    )


# ── Model architecture defaults (used when [base_pkan] section is absent) ──
_FALLBACK_KAN_HIDDEN = 12
_FALLBACK_KAN_GRID = 5
_FALLBACK_KAN_K = 3
_FALLBACK_ETA_N = 0.60


def _prepare_fallback_model_dir(cfg: configparser.ConfigParser, p: dict[str, Path]) -> None:
    """..."""
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

def train_full_model(cfg, p) -> None:
    section = cfg["full_model"]
    model_dir = p["base_pkan"]
    if not (model_dir / "pkan_torch_model.pt").exists():
        _prepare_fallback_model_dir(cfg, p)
    p["full"].mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(ENGINE / "dg_tft_full_model.py"),
        "--data-dir-override",
        str(p["data"]),
        "--model-dir",
        str(model_dir),
        "--out-dir",
        str(p["full"]),
        "--device",
        cfg["runtime"]["device"],
        "--dtype",
        cfg["runtime"]["dtype"],
        "--seed",
        cfg["runtime"]["seed"],
        "--epochs",
        section["epochs"],
        "--lr",
        section["learning_rate"],
        "--batch-iv",
        section["batch_iv"],
        "--batch-cv",
        section["batch_cv"],
        "--cv-weight",
        section["cv_weight"],
        "--cv-residual-level",
        section["cv_residual_level"],
        "--early-stop-patience",
        section["early_stop_patience"],
        "--early-stop-min-epochs",
        section["early_stop_min_epochs"],
        "--symbolic-training-weight",
        section["symbolic_training_weight"],
        "--temperature-kan-min-span-K",
        cfg["temperature"]["minimum_span_K"],
        "--temperature-kan-scale",
        cfg["temperature"]["kan_scale"],
        "--hspice-exe",
        cfg["paths"]["hspice_exe"],
    ]
    command.append(
        "--enable-cv-residual"
        if yes(cfg, "full_model", "enable_cv_residual")
        else "--disable-cv-residual"
    )
    command.append(
        "--finetune-base-pkan"
        if yes(cfg, "full_model", "finetune_base_pkan")
        else "--no-finetune-base-pkan"
    )
    mode = cfg["temperature"]["mode"].strip().lower()
    if mode == "on":
        command += ["--enable-temperature-kan", "--no-auto-enable-temperature-kan"]
    elif mode == "off":
        command += ["--disable-temperature-kan", "--no-auto-enable-temperature-kan"]
    else:
        command += ["--disable-temperature-kan", "--auto-enable-temperature-kan"]
    if not yes(cfg, "deployment", "run_hspice"):
        command += [
            "--skip-hspice-smoke",
            "--skip-hspice-grid-equivalence",
            "--skip-hspice-cv-grid-equivalence",
        ]
    if yes(cfg, "runtime", "quick_run"):
        command.append("--quick-run")
    if yes(cfg, "runtime", "debug_run"):
        command.append("--quick-run")
        # Override engine --quick-run limits: force absolute minimums
        command += [
            "--epochs", "10",
            "--early-stop-patience", "1",
            "--early-stop-min-epochs", "1",
        ]
    run(command)


def dynamic_env(p) -> dict[str, str]:
    env = dict(os.environ)
    env["DG_TFT_IT_DATA_DIR"] = str(p["it"])
    env["DG_TFT_STATIC_DATA_DIR"] = str(p["data"])
    env["DG_TFT_DYNAMIC_WORK_DIR"] = str(p["dynamic"])
    return env


def train_hysteresis(cfg, p) -> None:
    if not p["it"].exists():
        raise FileNotFoundError(f"Hysteresis data folder does not exist: {p['it']}")
    env = dynamic_env(p)
    p["dynamic"].mkdir(parents=True, exist_ok=True)
    run([sys.executable, str(DYNAMIC / "analyze_it_hysteresis.py")], env=env)
    section = cfg["hysteresis"]
    epochs = section["epochs"]
    hidden_capture = section["hidden_capture"]
    hidden_emission = section["hidden_emission"]
    if yes(cfg, "runtime", "debug_run"):
        epochs = "8"
        hidden_capture = "4"
        hidden_emission = "4"
    elif yes(cfg, "runtime", "quick_run"):
        epochs = "50"
        hidden_capture = "8"
        hidden_emission = "8"
    run(
        [
            sys.executable,
            str(DYNAMIC / "train_srh_top2_regime_banks.py"),
            "--regime",
            section["regime"],
            "--epochs",
            epochs,
            "--hidden-capture",
            hidden_capture,
            "--hidden-emission",
            hidden_emission,
            "--amp-parameterization",
            section["amp_parameterization"],
        ],
        env=env,
    )
    run([sys.executable, str(DYNAMIC / "export_srh_top2_regime_banked_va.py")], env=env)


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
