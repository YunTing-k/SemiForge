from __future__ import annotations

import argparse
import configparser
import os
from pathlib import Path
import shlex
import subprocess
import sys


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
    print("+", " ".join(shlex.quote(str(x)) for x in command), flush=True)
    subprocess.run(command, check=True, cwd=RELEASE_ROOT, env=env)


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


def validate_data(cfg, config_path, p) -> None:
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
    run(
        [
            sys.executable,
            str(ENGINE / "fit_physical_kernel.py"),
            "--data-dir",
            str(p["data"]),
            "--out-dir",
            str(p["physical"]),
            "--max-iv-points",
            section["max_iv_points"],
            "--max-cv-points",
            section["max_cv_points"],
            "--max-nfev-scale",
            section["max_nfev_scale"],
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


def train_full_model(cfg, p) -> None:
    section = cfg["full_model"]
    model_dir = (
        p["base_pkan"]
        if (p["base_pkan"] / "pkan_torch_model.pt").exists()
        else RELEASE_ROOT / "trained_model" / "base_physical_pkan"
    )
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
    run(
        [
            sys.executable,
            str(DYNAMIC / "train_srh_top2_regime_banks.py"),
            "--regime",
            section["regime"],
            "--epochs",
            section["epochs"],
            "--hidden-capture",
            section["hidden_capture"],
            "--hidden-emission",
            section["hidden_emission"],
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
