from __future__ import annotations

import json
from pathlib import Path
import re
import subprocess

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results_long_timescale" / "srh_top2_regime_banks" / "combined"
HSPICE = Path(r"E:\synopsys\Hspice_S-2021.09\WIN64\hspice.exe")


def parse_hspice_float(token: str) -> float:
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


def write_equivalence_deck() -> Path:
    deck = OUT / "dg_tft_complete_model_current_equivalence_v3.sp"
    deck.write_text(
        """* Complete DG-TFT static + slow-SRH current equivalence
.option post=2 accurate runlvl=5 reltol=1e-7 abstol=1e-16 vntol=1e-11 delmax=1m
.temp 20
.hdl "../../../../static_signoff_outputs/dg_tft_static_core.va"
.hdl "dg_tft_dynamic_srh.va"
.include "dg_tft_dynamic_srh.inc"
.include "dg_tft_complete_model.inc"
Vtg tg 0 PWL(0 0.87 1m 0.87 1.001m 2.50 5 2.50 5.001 0.42 10 0.42)
Vbg bg 0 0
Vd d 0 0.5
Xdut tg bg d 0 ntf nts nbf nbs dg_tft_complete_model
+ vfrom_t=0.87 vfrom_b=0 cycle_index=1 hold_s=10
.ic V(ntf)=0 V(nts)=0 V(nbf)=0 V(nbs)=0
.tran 1m 10 UIC
.print tran I(Vd) I(Xdut.Rsense) V(Xdut.dlog)
.end
""",
        encoding="ascii",
    )
    return deck


def parse_equivalence_listing(path: Path) -> pd.DataFrame:
    lines = path.read_text(errors="ignore").splitlines()
    number = re.compile(
        r"^[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?(?:meg|[tgkmunpf])?$",
        re.I,
    )
    starts = [
        i
        for i, line in enumerate(lines)
        if "transient analysis" in line.lower()
    ]
    rows: dict[float, list[float]] = {}
    for start in starts:
        for line in lines[start + 1 :]:
            parts = line.split()
            if len(parts) != 4 or not all(number.match(x) for x in parts):
                if rows and (line.strip() == "y" or "job concluded" in line.lower()):
                    break
                continue
            values = [parse_hspice_float(x) for x in parts]
            rows[values[0]] = values
    if not rows:
        raise RuntimeError("Could not parse complete-device current table.")
    return pd.DataFrame(
        [rows[key] for key in sorted(rows)],
        columns=["time_s", "i_external_a", "i_static_a", "dlog"],
    )


def validate_current_relation(listing: Path) -> dict:
    frame = parse_equivalence_listing(listing)
    predicted_magnitude = np.abs(frame["i_static_a"].to_numpy()) * np.power(
        10.0, frame["dlog"].to_numpy()
    )
    measured_magnitude = np.abs(frame["i_external_a"].to_numpy())
    valid = (
        (frame["time_s"].to_numpy() > 0.0)
        & (measured_magnitude > 1.0e-18)
        & (predicted_magnitude > 1.0e-18)
    )
    if not np.any(valid):
        raise RuntimeError("No valid nonzero current samples were produced.")
    floor = 1.0e-30
    log_error = np.log10(np.maximum(measured_magnitude[valid], floor)) - np.log10(
        np.maximum(predicted_magnitude[valid], floor)
    )
    relative_error = np.abs(
        measured_magnitude[valid] - predicted_magnitude[valid]
    ) / np.maximum(
        measured_magnitude[valid], 1.0e-18
    )
    metrics = {
        "sample_count": int(np.count_nonzero(valid)),
        "log_rmse_dec": float(np.sqrt(np.mean(log_error * log_error))),
        "log_max_abs_dec": float(np.max(np.abs(log_error))),
        "relative_error_p99": float(np.quantile(relative_error, 0.99)),
        "relative_error_max": float(np.max(relative_error)),
        "finite": bool(
            np.isfinite(frame[["i_external_a", "i_static_a", "dlog"]].to_numpy()).all()
        ),
    }
    metrics["pass"] = bool(
        metrics["finite"]
        and metrics["log_rmse_dec"] <= 2.0e-5
        and metrics["log_max_abs_dec"] <= 2.0e-4
    )
    frame["i_relation_a"] = predicted_magnitude
    frame["log_error_dec"] = np.nan
    frame.loc[valid, "log_error_dec"] = log_error
    frame.to_csv(
        OUT / "complete_device_current_equivalence.csv",
        index=False,
        encoding="utf-8-sig",
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
    fig, axes = plt.subplots(2, 1, figsize=(7.16, 4.15), sharex=True)
    axes[0].semilogy(
        frame["time_s"],
        measured_magnitude,
        color="#0072B2",
        lw=1.2,
        label="HSPICE external terminal current",
    )
    axes[0].semilogy(
        frame["time_s"],
        predicted_magnitude,
        "--",
        color="#D55E00",
        lw=0.9,
        label=r"$|I_{\rm static}|10^{\Delta\log I}$",
    )
    axes[0].set_ylabel(r"$|I_D|$ (A)")
    axes[0].legend(frameon=False, loc="lower left", fontsize=7)
    axes[0].grid(True, alpha=0.22)
    axes[0].set_title("(a) Complete terminal-current identity", loc="left", fontsize=8)
    stride = max(1, int(np.ceil(np.count_nonzero(valid) / 1800)))
    axes[1].plot(
        frame.loc[valid, "time_s"].to_numpy()[::stride],
        log_error[::stride],
        color="#009E73",
        lw=0.65,
    )
    axes[1].axhline(0.0, color="0.35", lw=0.6)
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Log error (dec)")
    axes[1].grid(True, alpha=0.22)
    axes[1].set_title("(b) Pointwise logarithmic error", loc="left", fontsize=8)
    fig.subplots_adjust(left=0.11, right=0.985, bottom=0.12, top=0.95, hspace=0.22)
    fig.savefig(OUT / "complete_device_current_equivalence.png", dpi=350)
    plt.close(fig)
    return metrics


def main() -> int:
    deck = write_equivalence_deck()
    listing = OUT / "dg_tft_complete_model_current_equivalence_v3.lis"
    result = subprocess.run(
        [str(HSPICE), "-i", deck.name, "-o", listing.name],
        cwd=OUT,
        text=True,
        capture_output=True,
        timeout=1200,
    )
    text = listing.read_text(errors="ignore") if listing.exists() else ""
    tr0 = OUT / "dg_tft_complete_model_current_equivalence_v3.tr0"
    current_relation = {}
    if result.returncode == 0 and listing.exists():
        current_relation = validate_current_relation(listing)
    summary = {
        "return_code": result.returncode,
        "job_concluded": "job concluded" in text.lower(),
        "hspice_error_present": "**error**" in text.lower()
        or "job aborted" in text.lower(),
        "transient_file_present": tr0.exists(),
        "transient_file_bytes": tr0.stat().st_size if tr0.exists() else 0,
        "complete_current_relation": current_relation,
        "pass": False,
    }
    summary["pass"] = bool(
        result.returncode == 0
        and summary["job_concluded"]
        and not summary["hspice_error_present"]
        and summary["transient_file_bytes"] > 1000
        and current_relation.get("pass", False)
    )
    (OUT / "complete_device_hspice_equivalence_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    if not summary["pass"]:
        raise SystemExit("Complete static + SRH HSPICE current equivalence failed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
