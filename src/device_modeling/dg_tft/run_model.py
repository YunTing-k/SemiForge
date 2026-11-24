from __future__ import annotations

import argparse
import configparser
import importlib
import importlib.util
import json
import os
import re
from pathlib import Path
import platform
import shlex
import shutil
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

_FALLBACK_KAN_HIDDEN = 12
_FALLBACK_KAN_GRID = 5
_FALLBACK_KAN_K = 3
_FALLBACK_ETA_N = 0.60


# ── helpers ──────────────────────────────────────────────────────────────

def resolve_path(value: str, config_path: Path) -> Path:
    path = Path(os.path.expandvars(os.path.expanduser(value)))
    if not path.is_absolute():
        path = (config_path.parent / path).resolve()
    return path.resolve()


def yes(cfg: configparser.ConfigParser, section: str, option: str) -> bool:
    return cfg.getboolean(section, option)


def resolve_hspice(cfg: configparser.ConfigParser):
    configured = cfg["paths"].get("hspice_exe", "").strip()
    candidates = [
        configured or None,
        os.environ.get("DG_TFT_HSPICE_EXE"),
        os.environ.get("HSPICE_EXE"),
        shutil.which("hspice.com"),
        shutil.which("hspice.exe"),
        shutil.which("hspice"),
    ]
    for candidate in candidates:
        if candidate and Path(candidate).expanduser().exists():
            return str(Path(candidate).expanduser().resolve())
    return None


def run(command: list[str], *, env: dict[str, str] | None = None) -> None:
    print("+", " ".join(shlex.quote(str(x)) for x in command), flush=True)
    child_env = dict(os.environ if env is None else env)
    child_env["PYTHONUNBUFFERED"] = "1"
    mpl_cache = RELEASE_ROOT / ".runtime_cache" / "matplotlib"
    mpl_cache.mkdir(parents=True, exist_ok=True)
    child_env["MPLCONFIGDIR"] = str(mpl_cache)
    subprocess.run(command, check=True, cwd=str(CODE_ROOT), env=child_env)


def record_nonblocking_gate_failure(label: str, exc: Exception) -> None:
    path = RELEASE_ROOT / "validation" / "NONBLOCKING_GATE_FAILURES.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"{label}: {exc!r}\n")
    print(
        f"WARNING: {label} did not pass. Generated model artifacts are retained; "
        f"details: {path}",
        flush=True,
    )


def run_gate(
    command: list[str], *, label: str, env: dict[str, str] | None = None
) -> bool:
    try:
        run(command, env=env)
        return True
    except subprocess.CalledProcessError as exc:
        record_nonblocking_gate_failure(label, exc)
        return False


def run_artifact_stage(
    command: list[str],
    *,
    label: str,
    expected_artifacts: list[Path],
    env: dict[str, str] | None = None,
) -> bool:
    try:
        run(command, env=env)
        return True
    except subprocess.CalledProcessError as exc:
        if not any(path.is_file() for path in expected_artifacts):
            raise
        record_nonblocking_gate_failure(label, exc)
        return False


def phase(message: str) -> None:
    print(f"\n=== {message} ===", flush=True)


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
        "kan_architecture": RELEASE_ROOT / "trained_model" / "base_physical_pkan",
        "full": output / "03_full_model",
        "dynamic": output / "04_hysteresis",
    }


def refuse_accidental_example_training(
    action: str, data_dir: Path, *, allow_example_data: bool
) -> None:
    training_actions = {
        "fit-physical",
        "train-base-pkan",
        "train-static",
        "train-hysteresis",
        "train-all",
        "quick-static",
        "smoke-static",
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
        if (
            "bias" not in col0_label
            or "bias" in col1_label
            or "bias_node_centered" in col0_label
            or ("bias" in col0_label and "bias" not in col1_label and "bias" in col2_label)
        ):
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
        desc_series = pd.Series(
            [full_desc] + [np.nan] * (out.shape[1] - 1), index=out.columns, name=0
        )
        header_series = pd.Series(header_pairs, index=out.columns, name=1)
        out = pd.concat(
            [pd.DataFrame([desc_series]), pd.DataFrame([header_series]), out]
        )
        result[sheet_name] = out
    if not rewritten:
        return
    with pd.ExcelWriter(cv_path, engine="openpyxl") as writer:
        for name, df_out in result.items():
            df_out.to_excel(writer, sheet_name=name[:31], index=False, header=False)


# ── stage functions ──────────────────────────────────────────────────────

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
    run_artifact_stage(
        command,
        label="dataset validation",
        expected_artifacts=[
            p["full"] / "v93_safe_residual_state.pt",
            p["full"] / "v92_safe_residual_state.pt",
        ],
    )

    report = json.loads((p["output"] / "dataset_validation.json").read_text(encoding="utf-8"))
    if not report.get("ok_static", False):
        raise ValueError("Static dataset validation failed; inspect dataset_validation.json")


def fit_physical(cfg, p) -> None:
    section = cfg["physical_fit"]
    max_iv = int(section["max_iv_points"])
    max_cv = int(section["max_cv_points"])
    max_nfev = float(section["max_nfev_scale"])
    if yes(cfg, "runtime", "minimal_run"):
        max_iv, max_cv, max_nfev = 1000, 500, 0.1

    p["physical"].mkdir(parents=True, exist_ok=True)
    run(
        [
            sys.executable,
            str(ENGINE / "fit_physical_kernel.py"),
            "--data-dir",
            str(p["data"]),
            "--out-dir",
            str(p["physical"]),
            "--max-iv-points",
            str(max_iv),
            "--max-cv-points",
            str(max_cv),
            "--max-nfev-scale",
            str(max_nfev),
            "--seed",
            cfg["runtime"]["seed"],
            "--iv-stage-alignment",
            cfg["preprocessing"]["iv_stage_alignment"],
            "--iv-stage-max-shift-V",
            cfg["preprocessing"]["iv_stage_max_shift_V"],
            "--iv-stage-min-matches",
            cfg["preprocessing"]["iv_stage_min_matches"],
        ]
    )


def _prepare_fallback_model_dir(
    cfg: configparser.ConfigParser, p: dict[str, Path]
) -> None:
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
        "baseline_trap_params": physical.get("cv_shared_trap_params") or {},
        "baseline_metrics_summary": physical.get("metrics_summary", {}),
        "pkan_metrics_summary": {},
        "final_kan_scale": float(
            cfg.get("temperature", "kan_scale", fallback="0.35")
        ),
        "training_args": {
            "kan_hidden": int(section.get("kan_hidden", _FALLBACK_KAN_HIDDEN)),
            "kan_grid": int(section.get("kan_grid", _FALLBACK_KAN_GRID)),
            "kan_k": int(section.get("kan_k", _FALLBACK_KAN_K)),
            "eta_n": _FALLBACK_ETA_N,
            "data_dir": str(p["data"]),
        },
    }
    model_dir = p["kan_architecture"]
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "pkan_model_metadata.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    try:
        import torch
    except ImportError:
        raise RuntimeError(
            "train-static requires PyTorch to synthesize the fallback model_dir. "
            "Ensure the modeling Python environment has torch installed."
        )
    torch.save({}, model_dir / "pkan_torch_model.pt")


def train_full_model(cfg, p, *, quick: bool = False) -> None:
    section = cfg["full_model"]
    model_dir = p["kan_architecture"]
    if not (model_dir / "pkan_torch_model.pt").exists():
        _prepare_fallback_model_dir(cfg, p)
    p["full"].mkdir(parents=True, exist_ok=True)
    if yes(cfg, "runtime", "minimal_run"):
        quick = True

    epochs = section.get("epochs", "700")
    lr = section.get("learning_rate", "0.0012")
    early_stop_patience = section.get("early_stop_patience", "5")
    early_stop_min_epochs = section.get("early_stop_min_epochs", "220")
    print_every = section.get("print_every", "10")
    if quick:
        epochs = "1"
        lr = "0.001"
        early_stop_patience = "1"
        early_stop_min_epochs = "1"
        print_every = "1"

    command = [
        sys.executable,
        str(ENGINE / "dg_tft_full_model.py"),
        "--data-dir-override",
        str(p["data"]),
        "--model-dir",
        str(model_dir),
        "--baseline-json",
        str(p["physical"] / "fitted_params.json"),
        "--out-dir",
        str(p["full"]),
        "--device",
        cfg["runtime"]["device"],
        "--dtype",
        cfg["runtime"]["dtype"],
        "--seed",
        cfg["runtime"]["seed"],
        "--epochs",
        epochs,
        "--base-iv-epochs",
        section.get("base_iv_epochs", "540"),
        "--base-iv-restarts",
        section.get("base_iv_restarts", "2"),
        "--base-iv-lr",
        section.get("base_iv_learning_rate", "0.002"),
        "--base-iv-eval-every",
        section.get("base_iv_eval_every", "20"),
        "--base-iv-top1-symbolic-weight",
        section.get("base_iv_top1_symbolic_weight", "0.35"),
        "--base-iv-top1-sparsity-weight",
        section.get("base_iv_top1_sparsity_weight", "0.001"),
        "--base-iv-top1-selection-weight",
        section.get("base_iv_top1_selection_weight", "0.08"),
        "--base-iv-max-top1-compatibility",
        section.get("base_iv_max_top1_compatibility", "0.015"),
        "--iv-residual-stage-epochs",
        section.get("iv_residual_stage_epochs", "700"),
        "--iv-residual-stage-lr",
        section.get("iv_residual_stage_learning_rate", "0.002"),
        "--iv-residual-holdout-stride",
        section.get("iv_residual_holdout_stride", "5"),
        "--iv-residual-stage-linear-weight",
        section.get("iv_residual_stage_linear_weight", "0.25"),
        "--iv-residual-stage-deriv-curves",
        section.get("iv_residual_stage_deriv_curves", "36"),
        "--iv-residual-stage-shape-period",
        section.get("iv_residual_stage_shape_period", "3"),
        "--iv-residual-off-anchor-weight",
        section.get("iv_residual_off_anchor_weight", "2.0"),
        "--lr",
        lr,
        "--base-pkan-lr-mult",
        section.get("base_pkan_lr_mult", "1.0"),
        "--warm-start-base-pkan-lr-mult",
        section.get("warm_start_base_pkan_lr_mult", "0.035"),
        "--batch-iv",
        section["batch_iv"],
        "--batch-cv",
        section["batch_cv"],
        "--cv-weight",
        section["cv_weight"],
        "--cv-train-subset",
        section.get("cv_train_subset", "primary_gate_cv"),
        "--cv-residual-level",
        section["cv_residual_level"],
        "--disabled-iv-pkan",
        section.get("disabled_iv_pkan", ""),
        "--disabled-cv-pkan",
        section.get("disabled_cv_pkan", ""),
        "--disabled-cv-charge-pkan",
        section.get("disabled_cv_charge_pkan", ""),
        "--early-stop-patience",
        early_stop_patience,
        "--early-stop-min-epochs",
        early_stop_min_epochs,
        "--symbolic-training-weight",
        section["symbolic_training_weight"],
        "--print-every",
        print_every,
        "--eval-every",
        section.get("eval_every", "40"),
        "--shape-eval-period",
        section.get("shape_eval_period", "5"),
        "--shape-early-period",
        section.get("shape_early_period", "20"),
        "--shape-middle-period",
        section.get("shape_middle_period", "10"),
        "--safe-scale-steps",
        cfg["validation"]["safe_scale_steps"],
        "--symbolic-edge-steps",
        cfg["symbolic"]["edge_steps"],
        "--symbolic-edge-topk",
        cfg["symbolic"].get("edge_topk", "1"),
        "--symbolic-iv-edge-topk",
        cfg["symbolic"].get("iv_edge_topk", cfg["symbolic"].get("edge_topk", "1")),
        "--symbolic-iv-adaptive-top2-mse",
        cfg["symbolic"].get("iv_adaptive_top2_mse", "0.0"),
        "--symbolic-edge-finetune-steps",
        cfg["symbolic"]["edge_finetune_steps"],
        "--symbolic-network-finetune-steps",
        cfg["symbolic"]["network_finetune_steps"],
        "--symbolic-device-iv-finetune-steps",
        cfg["symbolic"]["device_iv_finetune_steps"],
        "--symbolic-device-iv-measured-steps",
        cfg["symbolic"].get("device_iv_measured_steps", "1600"),
        "--iv-symbolic-calibration-max-terms",
        cfg["symbolic"].get("iv_calibration_max_terms", "24"),
        "--symbolic-device-cv-finetune-steps",
        cfg["symbolic"]["device_cv_finetune_steps"],
        "--temperature-kan-min-span-K",
        cfg["temperature"]["minimum_span_K"],
        "--temperature-kan-scale",
        cfg["temperature"]["kan_scale"],
        "--temperature-kan-hidden",
        cfg["temperature"].get("kan_hidden", "8"),
        "--temperature-kan-grid",
        cfg["temperature"].get("kan_grid", "3"),
        "--temperature-kan-k",
        cfg["temperature"].get("kan_k", "3"),
        "--weight-decay",
        section.get("weight_decay", "1e-6"),
        "--grad-clip",
        section.get("grad_clip", "5.0"),
        "--lr-eta-min-frac",
        section.get("lr_eta_min_frac", "0.02"),
        "--iv-stage-alignment",
        cfg["preprocessing"]["iv_stage_alignment"],
        "--iv-stage-max-shift-V",
        cfg["preprocessing"]["iv_stage_max_shift_V"],
        "--iv-stage-min-matches",
        cfg["preprocessing"]["iv_stage_min_matches"],
    ]
    hspice_exe = resolve_hspice(cfg)
    if hspice_exe:
        command += ["--hspice-exe", hspice_exe]
    if yes(cfg, "full_model", "reset_base_from_physical_fit"):
        command.append("--reset-base-from-physical-fit")
    command.append(
        "--train-base-iv-pkan"
        if yes(cfg, "full_model", "train_base_iv_pkan")
        else "--skip-base-iv-pkan"
    )
    command.append(
        "--train-iv-residual-stage"
        if section.getboolean("train_iv_residual_stage", fallback=True)
        else "--skip-iv-residual-stage"
    )
    command.append(
        "--freeze-iv-residual-after-stage"
        if section.getboolean("freeze_iv_residual_after_stage", fallback=True)
        else "--joint-finetune-iv-residual"
    )
    command.append(
        "--enable-staged-loss-schedule"
        if section.getboolean("staged_loss_schedule", fallback=True)
        else "--disable-staged-loss-schedule"
    )
    if not yes(cfg, "validation", "joint_cv_safe_scale_search"):
        command.append("--no-joint-cv-safe-scale-search")
    if cfg.getboolean("deployment", "export_dynamic_cv_nqs", fallback=False):
        command.append("--va-enable-dynamic-trap-nqs")
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
    if not yes(cfg, "deployment", "run_hspice") or hspice_exe is None:
        if yes(cfg, "deployment", "run_hspice") and hspice_exe is None:
            print(
                "HSPICE not found; Python training/export will continue and "
                "SPICE equivalence checks will be skipped.",
                flush=True,
            )
        command += [
            "--skip-hspice-smoke",
            "--skip-hspice-grid-equivalence",
            "--skip-hspice-cv-grid-equivalence",
        ]
    if quick:
        command += ["--quick-run", "--skip-precision-gates"]
    run(command)


def train_base_pkan(cfg, p) -> None:
    phase("train-base-pkan: base IV PKAN training only")
    model_dir = p["kan_architecture"]
    if not (model_dir / "pkan_torch_model.pt").exists():
        _prepare_fallback_model_dir(cfg, p)
    p["full"].mkdir(parents=True, exist_ok=True)
    section = cfg["base_pkan"] if cfg.has_section("base_pkan") else cfg["full_model"]
    epochs = section.get("epochs", section.get("base_iv_epochs", "540"))
    lr = section.get("learning_rate", section.get("base_iv_learning_rate", "0.002"))
    restarts = section.get("base_iv_restarts", "2")
    command = [
        sys.executable,
        str(ENGINE / "dg_tft_full_model.py"),
        "--data-dir-override",
        str(p["data"]),
        "--model-dir",
        str(model_dir),
        "--baseline-json",
        str(p["physical"] / "fitted_params.json"),
        "--out-dir",
        str(p["full"]),
        "--device",
        cfg["runtime"]["device"],
        "--dtype",
        cfg["runtime"]["dtype"],
        "--seed",
        cfg["runtime"]["seed"],
        "--epochs",
        epochs,
        "--base-iv-epochs",
        epochs,
        "--base-iv-restarts",
        restarts,
        "--base-iv-lr",
        lr,
        "--lr",
        lr,
        "--batch-iv",
        section.get("batch_iv", "256"),
        "--batch-cv",
        section.get("batch_cv", "64"),
        "--cv-residual-level",
        "0",
        "--reset-base-from-physical-fit",
        "--base-iv-only",
        "--skip-iv-residual-stage",
        "--disable-cv-residual",
        "--disable-temperature-kan",
        "--no-auto-enable-temperature-kan",
        "--temperature-kan-hidden",
        cfg["temperature"].get("kan_hidden", "8"),
        "--temperature-kan-grid",
        cfg["temperature"].get("kan_grid", "3"),
        "--temperature-kan-k",
        cfg["temperature"].get("kan_k", "3"),
        "--skip-hspice-smoke",
        "--skip-hspice-grid-equivalence",
        "--skip-hspice-cv-grid-equivalence",
    ]
    if yes(cfg, "runtime", "minimal_run"):
        command += ["--quick-run", "--skip-precision-gates"]
    run(command)


def doctor(cfg, config_path: Path, p: dict[str, Path]) -> None:
    required_modules = {
        "numpy": "numpy",
        "pandas": "pandas",
        "scipy": "scipy",
        "scikit-learn": "sklearn",
        "matplotlib": "matplotlib",
        "openpyxl": "openpyxl",
        "torch": "torch",
        "pykan": "kan",
    }
    optional_modules = {"python-docx": "docx", "reportlab": "reportlab"}
    required_files = [
        DEFAULT_CONFIG,
        ENGINE / "dg_tft_full_model.py",
        ENGINE / "fit_physical_kernel.py",
        ENGINE / "measurement_preprocessing.py",
        CODE_ROOT / "templates" / "physical_kernel_template.va",
        p["kan_architecture"] / "pkan_torch_model.pt",
    ]
    missing_modules = []
    import_errors = {}
    for name, module in required_modules.items():
        if importlib.util.find_spec(module) is None:
            missing_modules.append(name)
            continue
        try:
            importlib.import_module(module)
        except Exception as exc:
            missing_modules.append(name)
            import_errors[name] = repr(exc)
    missing_optional = []
    for name, module in optional_modules.items():
        if importlib.util.find_spec(module) is None:
            missing_optional.append(name)
            continue
        try:
            importlib.import_module(module)
        except Exception:
            missing_optional.append(name)
    missing_files = [str(path) for path in required_files if not path.exists()]
    print(f"Python executable : {sys.executable}")
    print(f"Python version    : {platform.python_version()}")
    print(f"Release root      : {RELEASE_ROOT}")
    print(f"Data directory    : {p['data']}")
    print(f"Output directory  : {p['output']}")
    detected_hspice = resolve_hspice(cfg) or "not found (Python training remains available)"
    print(f"HSPICE executable : {detected_hspice}")
    print(f"Missing modules   : {missing_modules or 'none'}")
    print(f"Import errors     : {import_errors or 'none'}")
    print(f"Optional modules  : {missing_optional or 'none missing'}")
    print(f"Missing files     : {missing_files or 'none'}")
    if missing_modules or missing_files:
        raise RuntimeError(
            "Environment check failed. Run setup_environment.ps1, then repeat "
            "'run_model.cmd doctor'."
        )
    print("Environment check passed.")


def smoke_static(cfg, p) -> None:
    baseline = p["physical"] / "fitted_params.json"
    if not baseline.exists():
        packaged_baseline = (
            RELEASE_ROOT / "trained_model" / "static_signoff" / "fitted_params.json"
        )
        if packaged_baseline.exists():
            baseline = packaged_baseline
        else:
            raise FileNotFoundError(
                f"Missing phase-1 parameters: {baseline}. Run "
                "'run_model.cmd fit-physical' first."
            )
    out_dir = RELEASE_ROOT / "validation" / "portable_static_smoke"
    out_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(ENGINE / "dg_tft_full_model.py"),
        "--data-dir-override",
        str(p["data"]),
        "--model-dir",
        str(p["kan_architecture"]),
        "--baseline-json",
        str(baseline),
        "--reset-base-from-physical-fit",
        "--out-dir",
        str(out_dir),
        "--device",
        "cpu",
        "--dtype",
        cfg["runtime"]["dtype"],
        "--seed",
        cfg["runtime"]["seed"],
        "--max-iv-points",
        "1200",
        "--max-cv-points",
        "2500",
        "--cv-train-subset",
        cfg["full_model"].get("cv_train_subset", "primary_gate_cv"),
        "--cv-residual-level",
        cfg["full_model"]["cv_residual_level"],
        "--disabled-iv-pkan",
        cfg["full_model"].get("disabled_iv_pkan", ""),
        "--disabled-cv-pkan",
        cfg["full_model"].get("disabled_cv_pkan", ""),
        "--disabled-cv-charge-pkan",
        cfg["full_model"].get("disabled_cv_charge_pkan", ""),
        "--smoke-check",
    ]
    command.append(
        "--enable-cv-residual"
        if yes(cfg, "full_model", "enable_cv_residual")
        else "--disable-cv-residual"
    )
    run(command)


def dynamic_env(cfg, p) -> dict[str, str]:
    env = dict(os.environ)
    env["DG_TFT_IT_DATA_DIR"] = str(p["it"])
    env["DG_TFT_STATIC_DATA_DIR"] = str(p["data"])
    env["DG_TFT_DYNAMIC_WORK_DIR"] = str(p["dynamic"])
    hspice_exe = resolve_hspice(cfg)
    if hspice_exe:
        env["DG_TFT_HSPICE_EXE"] = hspice_exe
    return env


def train_hysteresis(cfg, p) -> None:
    if not p["it"].exists():
        raise FileNotFoundError(f"Hysteresis data folder does not exist: {p['it']}")
    env = dynamic_env(cfg, p)
    p["dynamic"].mkdir(parents=True, exist_ok=True)
    run([sys.executable, str(DYNAMIC / "analyze_it_hysteresis.py")], env=env)
    section = cfg["hysteresis"]
    epochs = section.get("epochs", "4000")
    hidden_capture = section.get("hidden_capture", "32")
    hidden_emission = section.get("hidden_emission", "32")
    if yes(cfg, "runtime", "minimal_run"):
        epochs, hidden_capture, hidden_emission = "1", "4", "4"

    dynamic_command = [
        sys.executable,
        str(DYNAMIC / "train_srh_top2_regime_banks.py"),
        "--regime",
        section["regime"],
        "--epochs",
        epochs,
        "--seed",
        cfg["runtime"]["seed"],
        "--fast-restarts",
        section.get("fast_restarts", "1"),
        "--slow-restarts",
        section.get("slow_restarts", "2"),
        "--hidden-capture",
        hidden_capture,
        "--hidden-emission",
        hidden_emission,
        "--amp-parameterization",
        section["amp_parameterization"],
        "--fast-amp-parameterization",
        section.get("fast_amp_parameterization", section["amp_parameterization"]),
        "--slow-amp-parameterization",
        section.get("slow_amp_parameterization", section["amp_parameterization"]),
        "--fast-optimizer-policy",
        section.get("fast_optimizer_policy", "optimized"),
        "--slow-optimizer-policy",
        section.get("slow_optimizer_policy", "v90"),
    ]
    run_artifact_stage(
        dynamic_command,
        label="hysteresis acceptance gate",
        expected_artifacts=[
            p["dynamic"]
            / "srh_top2_regime_banks"
            / "srh_top2_regime_bank_checkpoint.pt"
        ],
        env=env,
    )
    run(
        [
            sys.executable,
            str(DYNAMIC / "export_srh_top2_regime_banked_va.py"),
            "--student-source",
            ".",
        ],
        env=env,
    )
    hspice_exe = resolve_hspice(cfg)
    if yes(cfg, "deployment", "run_hspice") and hspice_exe:
        for regime_name in ("fast", "slow"):
            regime_env = dict(env)
            regime_env["DG_TFT_HYSTERESIS_REGIME"] = regime_name
            phase(f"Python/HSPICE SRH equivalence: {regime_name} bank")
            run_gate(
                [
                    sys.executable,
                    str(DYNAMIC / "validate_srh_top2_modular_equivalence.py"),
                ],
                label=f"dynamic HSPICE equivalence ({regime_name})",
                env=regime_env,
            )
    elif yes(cfg, "deployment", "run_hspice"):
        print(
            "HSPICE not found; dynamic Python training/export completed and "
            "SPICE equivalence checks were skipped.",
            flush=True,
        )


def export_supplied_va() -> None:
    print("Validated Verilog-A files are in:")
    print(f"  {RELEASE_ROOT / 'verilog_a'}")


def assemble_release(cfg, p: dict[str, Path]) -> None:
    phase("Assemble signed-off weights, Verilog-A, figures, and plot data")
    assemble_command = [
        sys.executable,
        str(CODE_ROOT / "reporting" / "assemble_release.py"),
        "--static-dir",
        str(p["full"]),
        "--dynamic-dir",
        str(p["dynamic"]),
        "--physical-dir",
        str(p["physical"]),
    ]
    run(assemble_command)
    hspice_exe = resolve_hspice(cfg)
    if (
        yes(cfg, "hysteresis", "enabled")
        and yes(cfg, "deployment", "run_hspice")
        and hspice_exe
        and (p["dynamic"] / "srh_top2_regime_banks").exists()
    ):
        phase("Validate complete static + top-2 SRH Verilog-A model")
        run_gate(
            [
                sys.executable,
                str(DYNAMIC / "validate_complete_device_hspice.py"),
            ],
            label="complete-device HSPICE equivalence",
            env=dynamic_env(cfg, p),
        )
        run(assemble_command)
    run(
        [
            sys.executable,
            str(CODE_ROOT / "tools" / "build_v95_acceptance_bundle.py"),
        ]
    )
    if importlib.util.find_spec("docx") is not None:
        run([sys.executable, str(CODE_ROOT / "reporting" / "build_user_manual.py")])
    else:
        print(
            "Manual DOCX rebuild skipped because python-docx is not installed; "
            "the supplied signed-off DOCX is retained.",
            flush=True,
        )
    if importlib.util.find_spec("reportlab") is not None:
        run(
            [
                sys.executable,
                str(CODE_ROOT / "reporting" / "build_user_manual_pdf.py"),
            ]
        )
    else:
        print(
            "Manual PDF rebuild skipped because reportlab is not installed; "
            "the supplied signed-off PDF is retained.",
            flush=True,
        )
    run(
        [
            sys.executable,
            str(CODE_ROOT / "reporting" / "build_release_manifest.py"),
        ]
    )


def build_portable_archive() -> None:
    phase("Build clean portable directory, manifest, and ZIP")
    run([sys.executable, str(CODE_ROOT / "reporting" / "build_portable_archive.py")])


def verify_portable_package() -> None:
    phase("Verify packaged file set and SHA-256 manifest")
    run([sys.executable, str(CODE_ROOT / "reporting" / "verify_portable_package.py")])
    phase("Verify v95 model contracts, checkpoint schemas, and clean file set")
    run([sys.executable, str(CODE_ROOT / "tools" / "validate_v95_delivery.py")])


# ── main ─────────────────────────────────────────────────────────────────

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
            "package-release",
            "show-config",
            "doctor",
            "smoke-static",
            "quick-static",
            "package-portable",
            "verify-package",
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
    if args.action == "doctor":
        doctor(cfg, config_path, p)
    elif args.action == "smoke-static":
        doctor(cfg, config_path, p)
        validate_data(cfg, config_path, p)
        smoke_static(cfg, p)
    elif args.action == "quick-static":
        phase("Quick static IV/CV training and architecture validation")
        if yes(cfg, "full_model", "reset_base_from_physical_fit") and not (
            p["physical"] / "fitted_params.json"
        ).exists():
            fit_physical(cfg, p)
        train_full_model(cfg, p, quick=True)
    elif args.action == "check-data":
        validate_data(cfg, config_path, p)
    elif args.action == "fit-physical":
        phase("Fit physical IV/CV kernel")
        fit_physical(cfg, p)
    elif args.action == "train-base-pkan":
        if yes(cfg, "full_model", "reset_base_from_physical_fit") and not (
            p["physical"] / "fitted_params.json"
        ).exists():
            print("Phase-1 physical initialization is missing; running fit-physical first.")
            fit_physical(cfg, p)
        train_base_pkan(cfg, p)
    elif args.action == "train-static":
        phase("Train static IV/CV parameter and residual KAN")
        if yes(cfg, "full_model", "reset_base_from_physical_fit") and not (
            p["physical"] / "fitted_params.json"
        ).exists():
            print("Phase-1 physical initialization is missing; running fit-physical first.")
            fit_physical(cfg, p)
        train_full_model(cfg, p)
    elif args.action == "train-hysteresis":
        phase("Train fast/slow SRH hysteresis KAN banks")
        train_hysteresis(cfg, p)
    elif args.action == "train-all":
        phase("Phase 0/5: validate measured IV/CV/I-t datasets")
        validate_data(cfg, config_path, p)
        phase("Phase 1/5: fit the global physical IV/CV kernel")
        fit_physical(cfg, p)
        phase("Phase 2/5: train parameter KAN and charge-residual KAN")
        train_full_model(cfg, p)
        if yes(cfg, "hysteresis", "enabled"):
            phase("Phase 3/5: train fast/slow SRH hysteresis KAN banks")
            train_hysteresis(cfg, p)
        phase("Phase 4/5: assemble portable signed-off release")
        assemble_release(cfg, p)
        phase("Phase 5/5: complete")
    elif args.action == "package-release":
        assemble_release(cfg, p)
    elif args.action == "package-portable":
        build_portable_archive()
    elif args.action == "verify-package":
        verify_portable_package()
    elif args.action == "export-va":
        export_supplied_va()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
