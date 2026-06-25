from __future__ import annotations

from pathlib import Path
import json
import math
import re
import subprocess

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import refine_srh_waveform_end_to_end as base


ROOT = Path(__file__).resolve().parent
BANK = ROOT / "results_long_timescale" / "srh_top2_regime_banks"
OUT = BANK / "combined"
HSPICE = Path("hspice.com")
CONFIG = ROOT / "DYNAMIC_RELEASE_CONFIG.json"


def parse_hspice_float(token: str) -> float:
    """Parse HSPICE engineering suffixes without importing a legacy exporter."""
    text = token.strip().lower()
    suffixes = {
        "meg": 1.0e6,
        "t": 1.0e12,
        "g": 1.0e9,
        "k": 1.0e3,
        "m": 1.0e-3,
        "u": 1.0e-6,
        "n": 1.0e-9,
        "p": 1.0e-12,
        "f": 1.0e-15,
    }
    for suffix, scale in suffixes.items():
        if text.endswith(suffix):
            return float(text[: -len(suffix)]) * scale
    return float(text)


def formula_eval(formula: dict, x: np.ndarray) -> float:
    raw = float(base._formula_eval(formula, x[None, :])[0])
    return float(formula.get("target_center", 0.0)) + float(
        formula.get("target_scale", 1.0)
    ) * raw


def feature_vector(
    norm: dict,
    family: str,
    process: str,
    branch: str,
    gate_v: float,
    from_v: float,
) -> np.ndarray:
    raw = {
        "interface_top": 1.0 if family == "VTG" else 0.0,
        "process_capture": 1.0 if process == "capture" else 0.0,
        "branch_slow": 1.0 if branch == "slow" else 0.0,
        "temperature_dataset": 0.0,
        "to_v_norm": gate_v,
        "from_v_norm": from_v,
        "delta_v_norm": gate_v - from_v,
        "temp_norm": 293.15,
        "inv_temp_norm": 1000.0 / 293.15,
        "cycle_norm": 1.0,
        "log_hold_s": 1.0,
        "prev_log_id": -9.0,
        "seg_log_id0": -9.0,
        "vds_norm": 0.5,
    }
    values = []
    for name in base.FEATURE_NAMES:
        if name in {
            "interface_top",
            "process_capture",
            "branch_slow",
            "temperature_dataset",
        }:
            values.append(raw[name])
        else:
            mu, scale = norm[name]
            values.append((raw[name] - mu) / scale)
    return np.asarray(values, dtype=float)


def load_summary() -> dict:
    return json.loads(
        (
            BANK
            / "slow_only"
            / "srh_top2_regime_bank_summary.json"
        ).read_text(encoding="utf-8")
    )


def evaluate_parameters(
    summary: dict,
    family: str,
    gate_v: float,
    from_v: float,
) -> dict[str, float]:
    result = {}
    for process in ("capture", "emission"):
        group = summary["formulas"][f"slow|{family}|{process}"]
        for output in ("tau_fast", "tau_slow", "amp_fast", "amp_slow"):
            branch = "slow" if output.endswith("slow") else "fast"
            x = feature_vector(
                summary["normalization"],
                family,
                process,
                branch,
                gate_v,
                from_v,
            )
            result[f"{process}_{output}"] = formula_eval(group[output], x)
    return result


def python_simulation() -> pd.DataFrame:
    summary = load_summary()
    time = np.arange(0.0, 10.0001, 0.001)
    vt = np.where(time <= 0.001, 0.87, np.where(time <= 5.0, 2.50, 0.42))
    states = {"ntf": 0.0, "nts": 0.0, "nbf": 0.0, "nbs": 0.0}
    records = []
    top_cache = {
        float(v): evaluate_parameters(summary, "VTG", float(v), 0.87)
        for v in np.unique(vt)
    }
    bottom = evaluate_parameters(summary, "VBG", 0.0, 0.0)
    for i, (t, vt_i) in enumerate(zip(time, vt)):
        top = top_cache[float(vt_i)]
        params = {
            "ntf": (
                top["capture_tau_fast"],
                top["emission_tau_fast"],
                0.5
                * (
                    -5.0 * math.tanh(top["capture_amp_fast"] / 5.0)
                    + 5.0 * math.tanh(top["emission_amp_fast"] / 5.0)
                ),
            ),
            "nts": (
                top["capture_tau_slow"],
                top["emission_tau_slow"],
                0.5
                * (
                    -5.0 * math.tanh(top["capture_amp_slow"] / 5.0)
                    + 5.0 * math.tanh(top["emission_amp_slow"] / 5.0)
                ),
            ),
            "nbf": (
                bottom["capture_tau_fast"],
                bottom["emission_tau_fast"],
                0.5
                * (
                    -5.0 * math.tanh(bottom["capture_amp_fast"] / 5.0)
                    + 5.0 * math.tanh(bottom["emission_amp_fast"] / 5.0)
                ),
            ),
            "nbs": (
                bottom["capture_tau_slow"],
                bottom["emission_tau_slow"],
                0.5
                * (
                    -5.0 * math.tanh(bottom["capture_amp_slow"] / 5.0)
                    + 5.0 * math.tanh(bottom["emission_amp_slow"] / 5.0)
                ),
            ),
        }
        if i:
            dt = float(time[i] - time[i - 1])
            for name, (log_tc, log_te, _k) in params.items():
                tc = np.clip(10.0**log_tc, 1.0e-8, 1.0e3)
                te = np.clip(10.0**log_te, 1.0e-8, 1.0e3)
                neq = te / (tc + te)
                tau = tc * te / (tc + te)
                states[name] = neq + (states[name] - neq) * math.exp(
                    -dt / max(tau, 1.0e-15)
                )
        dlog = sum(params[name][2] * states[name] for name in states)
        records.append(
            {
                "time_s": t,
                "vt_py": vt_i,
                **{f"{name}_py": value for name, value in states.items()},
                "dlog_py": dlog,
            }
        )
    return pd.DataFrame(records)


def parse_listing(path: Path) -> pd.DataFrame:
    lines = path.read_text(errors="ignore").splitlines()
    num = re.compile(
        r"^[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?(?:meg|[tgkmunpf])?$",
        re.I,
    )

    def table_after(label_line: str, width: int) -> list[list[float]]:
        starts = [i for i, line in enumerate(lines) if label_line in line]
        rows: dict[float, list[float]] = {}
        for start in starts:
            for line in lines[start + 1 :]:
                parts = line.split()
                if len(parts) != width or not all(num.match(x) for x in parts):
                    if rows and ("y" == line.strip() or "job concluded" in line):
                        break
                    continue
                vals = [parse_hspice_float(x) for x in parts]
                rows[vals[0]] = vals
        return [rows[k] for k in sorted(rows)]

    first = table_after("tg         ntf        nts        nbf", 5)
    second = table_after("nbs        dlog", 3)
    a = pd.DataFrame(first, columns=["time_s", "vt_hs", "ntf_hs", "nts_hs", "nbf_hs"])
    b = pd.DataFrame(second, columns=["time_s", "nbs_hs", "dlog_hs"])
    return a.merge(b, on="time_s", how="inner")


def main() -> None:
    deck = OUT / "dg_tft_dynamic_trap_srh_top2_modular_smoke.sp"
    listing = OUT / "dg_tft_dynamic_trap_srh_top2_modular_smoke.lis"
    proc = subprocess.run(
        [str(HSPICE), "-i", deck.name, "-o", listing.name],
        cwd=OUT,
        text=True,
        capture_output=True,
        timeout=900,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stdout + "\n" + proc.stderr)
    hs = parse_listing(listing)
    py = python_simulation()
    merged = pd.merge_asof(
        hs.sort_values("time_s"),
        py.sort_values("time_s"),
        on="time_s",
        direction="nearest",
        tolerance=5.1e-4,
    ).dropna()
    metrics = {}
    for name in ("ntf", "nts", "nbf", "nbs", "dlog"):
        err = merged[f"{name}_hs"] - merged[f"{name}_py"]
        metrics[name] = {
            "rmse": float(np.sqrt(np.mean(err * err))),
            "max_abs": float(np.max(np.abs(err))),
        }
    if CONFIG.exists():
        config = json.loads(CONFIG.read_text(encoding="utf-8"))
        limit = float(config["acceptance"]["python_hspice_dlog_rmse_max"])
        metrics["acceptance"] = {
            "dlog_rmse_limit": limit,
            "pass": bool(metrics["dlog"]["rmse"] <= limit),
        }
    merged.to_csv(
        OUT / "python_vs_hspice_top2_modular.csv",
        index=False,
        encoding="utf-8-sig",
    )
    (OUT / "python_vs_hspice_top2_modular_summary.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 8,
            "axes.linewidth": 0.8,
            "xtick.direction": "in",
            "ytick.direction": "in",
        }
    )
    fig, axes = plt.subplots(2, 1, figsize=(7.16, 4.2), sharex=True)
    for name, color in zip(("ntf", "nts", "nbf", "nbs"), ("#0072B2", "#D55E00", "#009E73", "#CC79A7")):
        axes[0].plot(merged["time_s"], merged[f"{name}_py"], "-", color=color, lw=1.1, label=f"{name} Python")
        axes[0].plot(merged["time_s"], merged[f"{name}_hs"], "--", color=color, lw=0.8)
    axes[0].set_ylabel("Normalized occupancy")
    axes[0].legend(frameon=False, ncol=2, fontsize=7, loc="best")
    axes[0].grid(True, alpha=0.22)
    axes[0].set_title("(a) SRH state trajectories", loc="left", fontsize=8)
    axes[1].plot(merged["time_s"], merged["dlog_py"], "-", color="#0072B2", lw=1.2, label="Python")
    axes[1].plot(merged["time_s"], merged["dlog_hs"], "--", color="#D55E00", lw=0.9, label="HSPICE")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel(r"$\Delta\log_{10}|I_D|$")
    axes[1].legend(frameon=False, fontsize=7, loc="best")
    axes[1].grid(True, alpha=0.22)
    axes[1].set_title("(b) Dynamic current correction", loc="left", fontsize=8)
    fig.subplots_adjust(left=0.11, right=0.985, bottom=0.12, top=0.95, hspace=0.22)
    fig.savefig(OUT / "python_vs_hspice_top2_modular.png", dpi=350)
    plt.close(fig)
    print(json.dumps(metrics, indent=2))
    if "acceptance" in metrics and not metrics["acceptance"]["pass"]:
        raise SystemExit("Python/HSPICE full-SRH equivalence failed.")


if __name__ == "__main__":
    main()
