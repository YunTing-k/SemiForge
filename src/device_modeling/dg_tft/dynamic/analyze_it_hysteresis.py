from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import math
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.signal import savgol_filter

try:
    import torch
    import torch.nn as nn
except Exception:  # pragma: no cover
    torch = None
    nn = None


ROOT = Path(__file__).resolve().parent
IT_DATA_DIR = Path(
    os.environ.get(
        "DG_TFT_IT_DATA_DIR",
        str(ROOT / "example_it_data"),
    )
)
EXPECTED_INPUTS = [
    (IT_DATA_DIR / "I-t with different VBG1.xlsx", "VBG", False),
    (IT_DATA_DIR / "I-t with different VBG2.xlsx", "VBG", False),
    (IT_DATA_DIR / "I-t with different VBG3.xlsx", "VBG", False),
    (IT_DATA_DIR / "I-t with different VTG1.xlsx", "VTG", False),
    (IT_DATA_DIR / "I-t with different VTG2.xlsx", "VTG", False),
    (IT_DATA_DIR / "I-t with different VTG3.xlsx", "VTG", False),
    (IT_DATA_DIR / "I-t with different VBG_temperature.xlsx", "VBG", True),
    (IT_DATA_DIR / "I-t with different VTG_temperature.xlsx", "VTG", True),
]
LEGACY_FALLBACK_INPUT = (IT_DATA_DIR / "I-t.xlsx", "VTG", False)
OUT_DIR = Path(os.environ.get("DG_TFT_DYNAMIC_WORK_DIR", ROOT / "results_long_timescale"))
ID_FLOOR_A = 1.0e-15
DEFAULT_TEMP_K = 293.0
STATIC_DATA_DIR = Path(os.environ.get("DG_TFT_STATIC_DATA_DIR", str(IT_DATA_DIR)))


def safe_log10(x: np.ndarray) -> np.ndarray:
    return np.log10(np.clip(np.asarray(x, dtype=float), ID_FLOOR_A, None))


def odd_window(n: int, target: int) -> int:
    w = max(3, min(target, n if n % 2 else n - 1))
    if w % 2 == 0:
        w -= 1
    return max(w, 3)


def hampel_log(y: np.ndarray, k: int = 4, nsigma: float = 4.0) -> np.ndarray:
    y = np.asarray(y, dtype=float).copy()
    out = y.copy()
    n = len(y)
    for i in range(n):
        a = max(0, i - k)
        b = min(n, i + k + 1)
        med = np.median(y[a:b])
        mad = 1.4826 * np.median(np.abs(y[a:b] - med)) + 1.0e-12
        if abs(y[i] - med) > nsigma * mad:
            out[i] = med
    return out


def segment_aware_smooth(t: np.ndarray, gate: np.ndarray, log_i: np.ndarray, seg_id: np.ndarray) -> np.ndarray:
    smooth = np.empty_like(log_i)
    for seg in np.unique(seg_id):
        m = seg_id == seg
        y = hampel_log(log_i[m], k=3, nsigma=4.0)
        n = len(y)
        if n >= 9:
            # Preserve real step transients: short window near segment edges, wider in the flat middle.
            w = odd_window(n, 9 if n < 80 else 13)
            ys = savgol_filter(y, window_length=w, polyorder=2, mode="interp")
            edge = min(6, n // 5)
            if edge > 2:
                we = odd_window(n, 5)
                ye = savgol_filter(y, window_length=we, polyorder=2, mode="interp")
                ys[:edge] = ye[:edge]
                ys[-edge:] = ye[-edge:]
            smooth[m] = ys
        elif n >= 5:
            smooth[m] = savgol_filter(y, window_length=5, polyorder=2, mode="interp")
        else:
            smooth[m] = y
    return smooth


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {}
    for c in df.columns:
        key = str(c).strip().lower().replace(" ", "")
        if key.startswith("t(s") or key in {"time", "times", "time(s)"}:
            mapping[c] = "time_s"
        elif key.startswith("vds"):
            mapping[c] = "vds_V"
        elif key.startswith("vbg"):
            mapping[c] = "vbg_V"
        elif key.startswith("vtg"):
            mapping[c] = "vtg_V"
        elif key in {"t", "t(k)", "temp", "temperature", "temperature(k)"}:
            mapping[c] = "temp_K"
        elif key.startswith("ids") or key.startswith("id") or "current" in key:
            if key in {"id_a", "ida", "id(a)", "ids(a)", "current(a)"} or key.endswith("_a"):
                mapping[c] = "id_A"
            else:
                mapping[c] = "id_nA"
    df = df.rename(columns=mapping)
    required = ["time_s", "vds_V", "vbg_V", "vtg_V", "temp_K"]
    missing = [c for c in required if c not in df.columns and c != "temp_K"]
    if "id_A" not in df.columns and "id_nA" not in df.columns:
        missing.append("ID_A or IDS(nA)")
    if missing:
        raise ValueError(f"Missing columns {missing}; available={list(df.columns)}")
    if "temp_K" not in df.columns:
        df["temp_K"] = DEFAULT_TEMP_K
    current_col = "id_A" if "id_A" in df.columns else "id_nA"
    out = df[required + [current_col]].copy()
    for c in required + [current_col]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out["temp_K"] = out["temp_K"].fillna(DEFAULT_TEMP_K)
    out = out.dropna().reset_index(drop=True)
    if current_col == "id_nA":
        out["id_A"] = out["id_nA"] * 1.0e-9
    else:
        out["id_nA"] = out["id_A"] * 1.0e9
    return out


def read_it_sheet(path: Path, sheet_name: str) -> pd.DataFrame:
    """Read one I-t sheet with a forgiving header detector.

    The recent calibrated files normally have headers on the first row.  Some
    legacy sheets have blank leading rows or only partial columns; those are
    either repaired or explicitly skipped by the caller.
    """
    try:
        return normalize_columns(pd.read_excel(path, sheet_name=sheet_name))
    except Exception as first_exc:
        raw = pd.read_excel(path, sheet_name=sheet_name, header=None)
        header_row = None
        for i in range(min(len(raw), 20)):
            tokens = [str(x).strip().lower().replace(" ", "") for x in raw.iloc[i].tolist()]
            has_time = any(t.startswith("t(s") or t in {"time", "time(s)"} for t in tokens)
            has_id = any(t.startswith("ids") or t in {"id", "id(a)", "id(n/a)"} for t in tokens)
            has_gate = any(t.startswith("vbg") or t.startswith("vtg") for t in tokens)
            if has_time and has_id and has_gate:
                header_row = i
                break
        if header_row is None:
            raise ValueError(f"Cannot identify I-t header in {path.name}:{sheet_name}; first_error={first_exc}") from first_exc
        repaired = pd.read_excel(path, sheet_name=sheet_name, header=header_row)
        return normalize_columns(repaired)


def discover_input_files() -> tuple[list[tuple[Path, str, bool]], pd.DataFrame]:
    rows = []
    found: list[tuple[Path, str, bool]] = []
    for path, family, is_temp in EXPECTED_INPUTS:
        exists = path.exists()
        rows.append(
            {
                "path": str(path),
                "file": path.name,
                "family": family,
                "temperature_dataset": bool(is_temp),
                "exists": bool(exists),
                "used": bool(exists),
                "role": "expected_full_dataset",
            }
        )
        if exists:
            found.append((path, family, is_temp))
    if not found and LEGACY_FALLBACK_INPUT[0].exists():
        path, family, is_temp = LEGACY_FALLBACK_INPUT
        rows.append(
            {
                "path": str(path),
                "file": path.name,
                "family": family,
                "temperature_dataset": bool(is_temp),
                "exists": True,
                "used": True,
                "role": "legacy_fallback_only_because_expected_files_missing",
            }
        )
        found.append(LEGACY_FALLBACK_INPUT)
    if not found and IT_DATA_DIR.exists():
        for path in sorted([*IT_DATA_DIR.glob("*.xlsx"), *IT_DATA_DIR.glob("*.xls")]):
            if path.name.startswith("~$"):
                continue
            is_temp = "temp" in path.stem.lower() or "temperature" in path.stem.lower()
            rows.append(
                {
                    "path": str(path),
                    "file": path.name,
                    "family": "AUTO",
                    "temperature_dataset": bool(is_temp),
                    "exists": True,
                    "used": True,
                    "role": "auto_discovered_user_workbook",
                }
            )
            found.append((path, "AUTO", is_temp))
    return found, pd.DataFrame(rows)


def load_it_workbooks() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    summaries = []
    inputs, input_audit = discover_input_files()
    sheet_audit = []
    if not inputs:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        input_audit.to_csv(OUT_DIR / "it_input_file_audit.csv", index=False, encoding="utf-8-sig")
        raise FileNotFoundError(
            "No I-t input files were found. Expected files are listed in "
            "it_input_file_audit.csv. Put Excel workbooks in the configured "
            "hysteresis_data_dir."
        )
    for path, family, is_temp_dataset in inputs:
        xl = pd.ExcelFile(path)
        for sheet in xl.sheet_names:
            try:
                df = read_it_sheet(path, sheet)
            except Exception as exc:
                sheet_audit.append(
                    {
                        "file": path.name,
                        "sheet": sheet,
                        "family": family,
                        "temperature_dataset": bool(is_temp_dataset),
                        "used": False,
                        "reason": repr(exc),
                    }
                )
                continue
            sheet_family = family
            if sheet_family == "AUTO":
                vbg_span = float(np.ptp(df["vbg_V"].to_numpy(float))) if len(df) else 0.0
                vtg_span = float(np.ptp(df["vtg_V"].to_numpy(float))) if len(df) else 0.0
                sheet_family = "VBG" if vbg_span >= vtg_span else "VTG"
            dv = (
                df["vds_V"].diff().abs().fillna(0.0)
                + df["vbg_V"].diff().abs().fillna(0.0)
                + df["vtg_V"].diff().abs().fillna(0.0)
            )
            seg_id = (dv > 1.0e-10).cumsum().astype(int)
            source_tag = path.stem.replace("I-t with different ", "").replace(" ", "_")
            curve_id = f"{source_tag}_{sheet}"
            sweep_gate = "VBG" if sheet_family == "VBG" else "VTG"
            gate = df["vbg_V"].to_numpy() if sweep_gate == "VBG" else df["vtg_V"].to_numpy()
            log_i = safe_log10(df["id_A"].to_numpy())
            log_i_s = segment_aware_smooth(df["time_s"].to_numpy(), gate, log_i, seg_id.to_numpy())
            df["source_file"] = path.name
            df["temperature_dataset"] = bool(is_temp_dataset)
            df["sheet"] = sheet
            df["family"] = sheet_family
            df["curve_id"] = curve_id
            df["sweep_gate"] = sweep_gate
            df["sweep_V"] = gate
            df["sweep_fit_V"] = gate
            df["voltage_corrected"] = False
            df["seg_id"] = seg_id
            df["id_smooth_A"] = np.power(10.0, log_i_s)
            df["log_id"] = log_i
            df["log_id_smooth"] = log_i_s
            rows.append(df)
            sheet_audit.append(
                {
                    "file": path.name,
                    "sheet": sheet,
                    "family": sheet_family,
                    "temperature_dataset": bool(is_temp_dataset),
                    "used": True,
                    "reason": "",
                    "n_points": int(len(df)),
                    "n_segments": int(seg_id.nunique()),
                    "temp_min_K": float(df["temp_K"].min()),
                    "temp_max_K": float(df["temp_K"].max()),
                }
            )
            for seg, g in df.groupby("seg_id", sort=True):
                summaries.append(
                    {
                        "curve_id": curve_id,
                        "family": sheet_family,
                        "sheet": sheet,
                        "seg_id": int(seg),
                        "n": int(len(g)),
                        "t_start_s": float(g["time_s"].iloc[0]),
                        "t_end_s": float(g["time_s"].iloc[-1]),
                        "vbg_V": float(g["vbg_V"].iloc[0]),
                        "vtg_V": float(g["vtg_V"].iloc[0]),
                        "vds_V": float(g["vds_V"].iloc[0]),
                        "sweep_V": float(g["sweep_V"].iloc[0]),
                        "id_start_nA": float(g["id_nA"].iloc[0]),
                        "id_end_nA": float(g["id_nA"].iloc[-1]),
                        "id_median_nA": float(g["id_nA"].median()),
                        "id_smooth_start_nA": float(g["id_smooth_A"].iloc[0] * 1e9),
                        "id_smooth_end_nA": float(g["id_smooth_A"].iloc[-1] * 1e9),
                    }
                )
    if not rows:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        input_audit.to_csv(OUT_DIR / "it_input_file_audit.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(sheet_audit).to_csv(OUT_DIR / "it_sheet_read_audit.csv", index=False, encoding="utf-8-sig")
        raise RuntimeError("I-t files were found, but no usable sheets were parsed. See it_sheet_read_audit.csv.")
    audit = pd.concat([input_audit, pd.DataFrame(sheet_audit)], ignore_index=True, sort=False)
    return pd.concat(rows, ignore_index=True), pd.DataFrame(summaries), audit


def double_exp_model(t: np.ndarray, p: np.ndarray) -> np.ndarray:
    y_inf, a1, a2, logtau1, logtau2 = p
    tau1 = 10.0**logtau1
    tau2 = 10.0**logtau2
    return y_inf + a1 * np.exp(-t / tau1) + a2 * np.exp(-t / tau2)


def fit_step_response(t_rel: np.ndarray, y: np.ndarray) -> dict:
    t_rel = np.asarray(t_rel, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(y) < 16 or np.nanmax(t_rel) <= 0:
        return {"ok": False, "reason": "too_few_points"}
    # Skip the first sample after a voltage edge only if it is a clear instrument carry-over.
    use = np.ones(len(y), dtype=bool)
    if len(y) > 8:
        d0 = abs(y[0] - np.median(y[1: min(len(y), 8)]))
        span = max(np.percentile(y, 95) - np.percentile(y, 5), 1.0e-6)
        if d0 > 0.35 * span:
            use[0] = False
    tt = t_rel[use] - t_rel[use][0]
    yy = y[use]
    if len(tt) > 2:
        dt = np.diff(tt)
        dt = dt[np.isfinite(dt) & (dt > 0)]
        dt_med = float(np.median(dt)) if len(dt) else float(np.nanmax(tt) / max(len(tt) - 1, 1))
    else:
        dt_med = float(np.nanmax(tt))
    dt_med = max(dt_med, 1.0e-8)
    t_span = max(float(np.nanmax(tt) - np.nanmin(tt)), dt_med)
    # The recalibrated data mixes 4 ms oscilloscope-like traces with
    # 70-166 s stress/recovery traces.  Duration-aware bounds keep the same
    # two-trap equation but prevent the deep trap from being clipped at the old
    # 20 ms upper bound.
    tau_fast_min = max(1.0e-6, 0.02 * dt_med)
    tau_fast_max = max(8.0e-4, min(2.0, 0.8 * t_span))
    tau_slow_min = max(5.0 * tau_fast_min, 0.15 * dt_med)
    tau_slow_max = max(2.0e-2, 20.0 * t_span)
    if tau_slow_min >= tau_slow_max:
        tau_slow_min = max(2.0 * tau_fast_min, 0.1 * tau_slow_max)
    y0 = yy[0]
    yend = np.median(yy[-min(12, len(yy)):])
    amp = y0 - yend
    p0_fast = math.sqrt(tau_fast_min * tau_fast_max)
    p0_slow = math.sqrt(tau_slow_min * tau_slow_max)
    p0 = np.array([yend, 0.55 * amp, 0.45 * amp, math.log10(p0_fast), math.log10(p0_slow)])
    lower = np.array([min(yy) - 2.0, -5.0, -5.0, math.log10(tau_fast_min), math.log10(tau_slow_min)])
    upper = np.array([max(yy) + 2.0, 5.0, 5.0, math.log10(tau_fast_max), math.log10(tau_slow_max)])

    def resid(p):
        return double_exp_model(tt, p) - yy

    res = least_squares(resid, p0, bounds=(lower, upper), loss="soft_l1", f_scale=0.035, max_nfev=8000)
    p = res.x
    tau1 = 10.0 ** p[3]
    tau2 = 10.0 ** p[4]
    if tau1 > tau2:
        tau1, tau2 = tau2, tau1
        p[1], p[2] = p[2], p[1]
    pred = double_exp_model(tt, res.x)
    rmse = float(np.sqrt(np.mean((pred - yy) ** 2)))
    return {
        "ok": bool(res.success),
        "reason": res.message,
        "y_inf": float(p[0]),
        "amp_fast": float(p[1]),
        "amp_slow": float(p[2]),
        "tau_fast_s": float(tau1),
        "tau_slow_s": float(tau2),
        "log_rmse": rmse,
        "n_fit": int(len(yy)),
        "dt_median_s": float(dt_med),
        "t_span_s": float(t_span),
        "tau_fast_lower_s": float(tau_fast_min),
        "tau_fast_upper_s": float(tau_fast_max),
        "tau_slow_lower_s": float(tau_slow_min),
        "tau_slow_upper_s": float(tau_slow_max),
        "fast_tau_unresolved": bool(tau1 <= 2.5 * dt_med),
        "slow_tau_unresolved": bool(tau2 >= 0.6 * tau_slow_max or tau2 >= 5.0 * t_span),
        "skipped_first_point": bool(not use[0]),
    }


def fit_all_steps(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    records = []
    pred_rows = []
    for curve_id, cdf in df.groupby("curve_id", sort=False):
        cdf = cdf.sort_values("time_s")
        segs = sorted(cdf["seg_id"].unique())
        for seg in segs[1:]:
            g = cdf[cdf["seg_id"] == seg].copy()
            prev = cdf[cdf["seg_id"] == seg - 1]
            if prev.empty or len(g) < 16:
                continue
            t_rel = g["time_s"].to_numpy() - g["time_s"].iloc[0]
            fit = fit_step_response(t_rel, g["log_id_smooth"].to_numpy())
            before_v = float(prev["sweep_fit_V"].iloc[-1])
            after_v = float(g["sweep_fit_V"].iloc[0])
            fit.update(
                {
                    "curve_id": curve_id,
                    "source_file": g["source_file"].iloc[0],
                    "family": g["family"].iloc[0],
                    "sheet": g["sheet"].iloc[0],
                    "step_seg_id": int(seg),
                    "from_sweep_V": before_v,
                    "to_sweep_V": after_v,
                    "from_labeled_sweep_V": float(prev["sweep_V"].iloc[-1]),
                    "to_labeled_sweep_V": float(g["sweep_V"].iloc[0]),
                    "delta_sweep_V": after_v - before_v,
                    "direction": "up" if after_v > before_v else "down",
                    "vbg_V": float(g["vbg_V"].iloc[0]),
                    "vtg_V": float(g["vtg_V"].iloc[0]),
                    "vds_V": float(g["vds_V"].iloc[0]),
                    "temp_K": float(g["temp_K"].iloc[0]),
                    "temperature_dataset": bool(g["temperature_dataset"].iloc[0]),
                    "id_start_nA": float(g["id_nA"].iloc[0]),
                    "id_end_nA": float(g["id_nA"].iloc[-1]),
                }
            )
            fit["slow_tau_unresolved"] = bool(fit.get("ok") and fit.get("slow_tau_unresolved", False))
            fit["large_amp_warning"] = bool(
                fit.get("ok") and max(abs(fit.get("amp_fast", 0.0)), abs(fit.get("amp_slow", 0.0))) > 2.0
            )
            fit["reliable_fast_tau"] = bool(fit.get("ok") and fit.get("log_rmse", 9.0) < 0.04 and not fit["large_amp_warning"])
            fit["reliable_full_2trap"] = bool(
                fit["reliable_fast_tau"]
                and not fit["slow_tau_unresolved"]
                and not fit.get("fast_tau_unresolved", False)
            )
            records.append(fit)
            if fit.get("ok"):
                tt = t_rel.copy()
                p = np.array([fit["y_inf"], fit["amp_fast"], fit["amp_slow"], math.log10(fit["tau_fast_s"]), math.log10(fit["tau_slow_s"])])
                yp = double_exp_model(tt, p)
                tmp = g[[
                    "time_s",
                    "curve_id",
                    "source_file",
                    "seg_id",
                    "family",
                    "sweep_V",
                    "vds_V",
                    "temp_K",
                    "temperature_dataset",
                    "id_A",
                    "id_smooth_A",
                    "log_id_smooth",
                ]].copy()
                tmp["t_rel_s"] = tt
                tmp["log_id_fit"] = yp
                tmp["id_fit_A"] = np.power(10.0, yp)
                pred_rows.append(tmp)
    return pd.DataFrame(records), pd.concat(pred_rows, ignore_index=True) if pred_rows else pd.DataFrame()


if torch is not None:

    class TinyTauLabelKAN(nn.Module):
        def __init__(self, in_dim: int, hidden: int = 10):
            super().__init__()
            self.l1 = nn.Linear(in_dim, hidden)
            self.l2 = nn.Linear(hidden, 2)

        def forward(self, x):
            # Symbolic-friendly basis mix: linear + tanh-like hidden nonlinearity.
            h = torch.tanh(self.l1(x))
            return self.l2(h)


def fit_tau_label_kan(step_df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    if torch is None or step_df.empty:
        return pd.DataFrame(), {"enabled": False, "reason": "torch_missing_or_no_steps"}
    if "reliable_fast_tau" in step_df.columns:
        use = step_df[(step_df["ok"] == True) & (step_df["reliable_fast_tau"] == True)].copy()
    else:
        use = step_df[step_df["ok"] == True].copy()
    use = use[np.isfinite(use["tau_fast_s"]) & np.isfinite(use["tau_slow_s"])]
    if len(use) < 6:
        return pd.DataFrame(), {"enabled": False, "reason": "too_few_valid_steps", "n": int(len(use))}
    fam = (use["family"].to_numpy() == "VTG").astype(float)
    direction = (use["direction"].to_numpy() == "up").astype(float)
    x_raw = np.column_stack(
        [
            fam,
            direction,
            use["from_sweep_V"].to_numpy(float) / 10.0,
            use["to_sweep_V"].to_numpy(float) / 10.0,
            use["delta_sweep_V"].to_numpy(float) / 10.0,
            use["vds_V"].to_numpy(float),
            (use["temp_K"].to_numpy(float) - 293.0) / 100.0,
        ]
    )
    y_raw = np.column_stack([np.log10(use["tau_fast_s"]), np.log10(use["tau_slow_s"])])
    fast_w = np.where(use.get("fast_tau_unresolved", False).to_numpy(bool), 0.20, 1.0)
    slow_w = np.where(use.get("slow_tau_unresolved", False).to_numpy(bool), 0.20, 1.0)
    label_w = np.column_stack([fast_w, slow_w])
    mu = x_raw.mean(axis=0)
    sig = np.maximum(x_raw.std(axis=0), 0.25)
    x = (x_raw - mu) / sig
    device = torch.device("cpu")
    dtype = torch.float32
    xt = torch.as_tensor(x, device=device, dtype=dtype)
    yt = torch.as_tensor(y_raw, device=device, dtype=dtype)
    wt = torch.as_tensor(label_w, device=device, dtype=dtype)
    model = TinyTauLabelKAN(x.shape[1], hidden=8).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2.0e-3, weight_decay=2.0e-3)
    best = None
    for ep in range(900):
        opt.zero_grad()
        pred = model(xt)
        loss = torch.sum(wt * (pred - yt) ** 2) / torch.clamp(torch.sum(wt), min=1.0)
        # Smooth symbolic-friendly penalty: discourage excessive hidden weights.
        loss = loss + 2.0e-3 * (torch.mean(torch.abs(model.l1.weight)) + torch.mean(torch.abs(model.l2.weight)))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        val = float(loss.detach())
        if best is None or val < best:
            best = val
    with torch.no_grad():
        pred = model(xt).cpu().numpy()
    keep_cols = ["curve_id", "family", "step_seg_id", "direction", "from_sweep_V", "to_sweep_V"]
    if "from_labeled_sweep_V" in use.columns:
        keep_cols += ["from_labeled_sweep_V", "to_labeled_sweep_V"]
    out = use[keep_cols].copy()
    out["log_tau_fast_fit"] = pred[:, 0]
    out["log_tau_slow_fit"] = pred[:, 1]
    out["tau_fast_kan_s"] = 10.0 ** pred[:, 0]
    out["tau_slow_kan_s"] = 10.0 ** pred[:, 1]
    out["tau_fast_label_s"] = use["tau_fast_s"].to_numpy()
    out["tau_slow_label_s"] = use["tau_slow_s"].to_numpy()
    rmse_fast = float(np.sqrt(np.mean((pred[:, 0] - y_raw[:, 0]) ** 2)))
    rmse_slow = float(np.sqrt(np.mean((pred[:, 1] - y_raw[:, 1]) ** 2)))
    wrmse_fast = float(np.sqrt(np.sum(label_w[:, 0] * (pred[:, 0] - y_raw[:, 0]) ** 2) / np.maximum(np.sum(label_w[:, 0]), 1.0)))
    wrmse_slow = float(np.sqrt(np.sum(label_w[:, 1] * (pred[:, 1] - y_raw[:, 1]) ** 2) / np.maximum(np.sum(label_w[:, 1]), 1.0)))
    metrics = {
        "enabled": True,
        "n_steps": int(len(use)),
        "log_tau_fast_rmse": rmse_fast,
        "log_tau_slow_rmse": rmse_slow,
        "weighted_log_tau_fast_rmse": wrmse_fast,
        "weighted_log_tau_slow_rmse": wrmse_slow,
        "low_weight_fast_labels": int(np.sum(label_w[:, 0] < 1.0)),
        "low_weight_slow_labels": int(np.sum(label_w[:, 1] < 1.0)),
        "feature_mean": mu.tolist(),
        "feature_scale": sig.tolist(),
        "note": "Tiny bias/context KAN for apparent tau labels; use as initialization, not final signoff.",
    }
    return out, metrics


def make_plots(df: pd.DataFrame, step_df: pd.DataFrame, pred_df: pd.DataFrame, tau_kan_df: pd.DataFrame) -> None:
    groups = list(df.groupby("curve_id", sort=False))
    ncols = 3
    nrows = int(math.ceil(max(1, len(groups)) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, max(3.6, 3.4 * nrows)), sharex=False)
    axes_flat = np.ravel(axes)
    for ax, (curve_id, g) in zip(axes_flat, groups):
        ax2 = ax.twinx()
        tscale = 1.0 if g["time_s"].max() > 1.0 else 1.0e3
        tunits = "s" if tscale == 1.0 else "ms"
        ax.semilogy(g["time_s"] * tscale, g["id_A"].clip(lower=ID_FLOOR_A), ".", ms=2, color="0.65", label="raw")
        ax.semilogy(g["time_s"] * tscale, g["id_smooth_A"].clip(lower=ID_FLOOR_A), lw=1.8, label="smoothed")
        ax2.plot(g["time_s"] * tscale, g["sweep_V"], color="#D97706", lw=1.1, alpha=0.55, label="labeled V")
        if "sweep_fit_V" in g.columns and np.any(np.abs(g["sweep_fit_V"].to_numpy() - g["sweep_V"].to_numpy()) > 1e-9):
            ax2.plot(g["time_s"] * tscale, g["sweep_fit_V"], color="#7C3AED", lw=1.2, alpha=0.9, label="fit V")
        ax.set_title(curve_id)
        ax.set_xlabel(f"time ({tunits})")
        ax.set_ylabel("|ID| (A)")
        ax2.set_ylabel("sweep V")
        ax.grid(True, alpha=0.25)
    for ax in axes_flat[len(groups):]:
        ax.axis("off")
    axes_flat[0].legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "it_raw_vs_smoothed.png", dpi=220)
    plt.close(fig)

    if not pred_df.empty:
        groups = list(pred_df.groupby("curve_id", sort=False))
        nrows = int(math.ceil(max(1, len(groups)) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(15, max(3.6, 3.4 * nrows)), sharex=False)
        axes_flat = np.ravel(axes)
        for ax, (curve_id, g) in zip(axes_flat, groups):
            tscale = 1.0 if g["time_s"].max() > 1.0 else 1.0e3
            tunits = "s" if tscale == 1.0 else "ms"
            ax.semilogy(g["time_s"] * tscale, g["id_smooth_A"], ".", ms=2, color="0.55", label="smoothed")
            first = True
            for _, sg in g.groupby("seg_id", sort=True):
                ax.semilogy(sg["time_s"] * tscale, sg["id_fit_A"], lw=1.5, label="2-trap fit" if first else None)
                first = False
            ax.set_title(curve_id)
            ax.set_xlabel(f"time ({tunits})")
            ax.set_ylabel("|ID| (A)")
            ax.grid(True, alpha=0.25)
        for ax in axes_flat[len(groups):]:
            ax.axis("off")
        axes_flat[0].legend(loc="best", fontsize=8)
        fig.tight_layout()
        fig.savefig(OUT_DIR / "it_step_response_fit.png", dpi=220)
        plt.close(fig)

    ok = step_df[step_df.get("ok", False) == True].copy() if not step_df.empty else pd.DataFrame()
    if not ok.empty:
        fig, ax = plt.subplots(figsize=(8.2, 4.8))
        colors = {"VBG": "#2563EB", "VTG": "#DC2626"}
        markers = {"up": "^", "down": "v"}
        for _, r in ok.iterrows():
            ax.scatter(r["to_sweep_V"], r["tau_fast_s"], color=colors[r["family"]], marker=markers[r["direction"]], s=50)
            ax.scatter(r["to_sweep_V"], r["tau_slow_s"], facecolors="none", edgecolors=colors[r["family"]], marker=markers[r["direction"]], s=70)
        ax.set_yscale("log")
        ax.set_xlabel("post-step swept voltage (V)")
        ax.set_ylabel("apparent tau (s)")
        ax.set_title("Fast/slow apparent trap time constants")
        ax.grid(True, alpha=0.25, which="both")
        fig.tight_layout()
        fig.savefig(OUT_DIR / "it_apparent_tau_summary.png", dpi=220)
        plt.close(fig)

    if not tau_kan_df.empty:
        fig, ax = plt.subplots(figsize=(6.8, 5.2))
        ax.loglog(tau_kan_df["tau_fast_label_s"], tau_kan_df["tau_fast_kan_s"], "o", label="fast")
        ax.loglog(tau_kan_df["tau_slow_label_s"], tau_kan_df["tau_slow_kan_s"], "s", label="slow")
        mn = min(tau_kan_df[["tau_fast_label_s", "tau_fast_kan_s", "tau_slow_label_s", "tau_slow_kan_s"]].min())
        mx = max(tau_kan_df[["tau_fast_label_s", "tau_fast_kan_s", "tau_slow_label_s", "tau_slow_kan_s"]].max())
        ax.plot([mn, mx], [mn, mx], "k--", lw=1)
        ax.set_xlabel("step-fit tau label (s)")
        ax.set_ylabel("tau-KAN predicted tau (s)")
        ax.grid(True, alpha=0.25, which="both")
        ax.legend()
        fig.tight_layout()
        fig.savefig(OUT_DIR / "it_tau_kan_label_fit.png", dpi=220)
        plt.close(fig)


def estimate_measurement_issues(df: pd.DataFrame, seg_summary: pd.DataFrame) -> dict:
    issues = []
    repeat_rows = []
    if seg_summary.empty or "curve_id" not in seg_summary.columns or "sweep_V" not in seg_summary.columns:
        return {
            "issues": ["No usable segment summary was available; check input files/sheets and voltage-step segmentation."],
            "repeat_bias_spread": repeat_rows,
            "static_voltage_alignment": [],
        }
    for curve_id, g in seg_summary.groupby("curve_id"):
        for v, gg in g.groupby("sweep_V"):
            if len(gg) >= 2:
                vals = gg["id_median_nA"].to_numpy(float)
                spread_dec = float(np.max(safe_log10(vals * 1e-9)) - np.min(safe_log10(vals * 1e-9)))
                repeat_rows.append({"curve_id": curve_id, "sweep_V": float(v), "repeat_spread_dec": spread_dec, "n": int(len(gg))})
                if spread_dec > 0.15:
                    issues.append(f"{curve_id} repeated V={v:g} has {spread_dec:.2f} dex median-current spread; likely prehistory/hysteresis, not simple noise.")
    static_rows = static_voltage_alignment(seg_summary)
    for r in static_rows:
        if (np.isfinite(r.get("abs_delta_V", np.nan)) and r["abs_delta_V"] > 0.35) or abs(r.get("log_error_dec", 0.0)) > 0.45:
            issues.append(
                f"{r['curve_id']} seg {r['seg_id']} steady current is {r['log_error_dec']:.2f} dex from static IV; "
                f"equivalent {r['sweep_gate']} shift {r['delta_V_equiv']:.2f} V."
            )
    return {"issues": issues, "repeat_bias_spread": repeat_rows, "static_voltage_alignment": static_rows}


def _static_curve_for_family(family: str) -> tuple[np.ndarray, np.ndarray, str]:
    if family == "VBG":
        df = pd.read_excel(STATIC_DATA_DIR / "IDVBG_VDS0.5V_train.xlsx", sheet_name="wide")
        return df["VBG_V"].to_numpy(float), df["ID_A_at_VTG_0V"].to_numpy(float), "VBG"
    df = pd.read_excel(STATIC_DATA_DIR / "IDVTG_VDS0.5V_train.xlsx", sheet_name="wide")
    return df["VTG_V"].to_numpy(float), df["ID_A_at_VBG_0V"].to_numpy(float), "VTG"


def static_voltage_alignment(seg_summary: pd.DataFrame) -> list[dict]:
    rows: list[dict] = []
    if not STATIC_DATA_DIR.exists():
        return rows
    cache: dict[str, tuple[np.ndarray, np.ndarray, str]] = {}
    for _, r in seg_summary.iterrows():
        fam = str(r["family"])
        if fam not in cache:
            try:
                cache[fam] = _static_curve_for_family(fam)
            except Exception:
                continue
        x, y, gate = cache[fam]
        logy = safe_log10(y)
        order = np.argsort(logy)
        x_log_sorted = x[order]
        log_sorted = logy[order]
        measured = max(float(r["id_median_nA"]) * 1.0e-9, ID_FLOOR_A)
        log_meas = float(np.log10(measured))
        static_log_at_label = float(np.interp(float(r["sweep_V"]), x, logy))
        if log_meas < log_sorted[0] or log_meas > log_sorted[-1]:
            v_equiv = float("nan")
            delta = float("nan")
        else:
            v_equiv = float(np.interp(log_meas, log_sorted, x_log_sorted))
            delta = v_equiv - float(r["sweep_V"])
        rows.append(
            {
                "curve_id": r["curve_id"],
                "seg_id": int(r["seg_id"]),
                "sweep_gate": gate,
                "labeled_sweep_V": float(r["sweep_V"]),
                "equivalent_static_sweep_V": v_equiv,
                "delta_V_equiv": delta,
                "abs_delta_V": abs(delta) if np.isfinite(delta) else float("nan"),
                "log_measured_A": log_meas,
                "log_static_at_label_A": static_log_at_label,
                "log_error_dec": log_meas - static_log_at_label,
            }
        )
    return rows


def apply_static_voltage_correction(
    df: pd.DataFrame,
    seg_summary: pd.DataFrame,
    threshold_dec: float = 0.45,
    max_abs_shift_V: float = 1.2,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    """Apply measurement-wrapper voltage correction from static-IV alignment.

    This does not alter the raw measured voltage columns.  It only fills
    `sweep_fit_V`, used by the dynamic tau/KAN features.  Large offsets are
    treated as measurement/prehistory corrections, not as device-core physics.
    """
    if df.empty or seg_summary.empty or "sweep_V" not in df.columns or "sweep_V" not in seg_summary.columns:
        out = df.copy()
        seg_out = seg_summary.copy()
        out["sweep_fit_V"] = out["sweep_V"] if "sweep_V" in out.columns else np.nan
        out["voltage_corrected"] = False
        seg_out["sweep_fit_V"] = seg_out["sweep_V"] if "sweep_V" in seg_out.columns else np.nan
        seg_out["voltage_corrected"] = False
        return out, seg_out, []

    align = static_voltage_alignment(seg_summary)
    corr_map: dict[tuple[str, int], float] = {}
    rows = []
    for r in align:
        de = float(r.get("log_error_dec", 0.0))
        dv = float(r.get("delta_V_equiv", float("nan")))
        use = bool(np.isfinite(dv) and abs(de) >= threshold_dec and abs(dv) <= max_abs_shift_V)
        if use:
            corr_map[(str(r["curve_id"]), int(r["seg_id"]))] = float(r["equivalent_static_sweep_V"])
        rr = dict(r)
        rr["use_voltage_correction"] = use
        rows.append(rr)
    out = df.copy()
    seg_out = seg_summary.copy()
    out["sweep_fit_V"] = out["sweep_V"]
    out["voltage_corrected"] = False
    seg_out["sweep_fit_V"] = seg_out["sweep_V"]
    seg_out["voltage_corrected"] = False
    for (cid, seg), vfit in corr_map.items():
        m = (out["curve_id"] == cid) & (out["seg_id"] == seg)
        out.loc[m, "sweep_fit_V"] = vfit
        out.loc[m, "voltage_corrected"] = True
        ms = (seg_out["curve_id"] == cid) & (seg_out["seg_id"] == seg)
        seg_out.loc[ms, "sweep_fit_V"] = vfit
        seg_out.loc[ms, "voltage_corrected"] = True
    return out, seg_out, rows


def write_outputs(df: pd.DataFrame, seg: pd.DataFrame, step_df: pd.DataFrame, pred_df: pd.DataFrame, tau_kan_df: pd.DataFrame, tau_kan_metrics: dict, audit: dict, input_audit: pd.DataFrame) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    input_audit.to_csv(OUT_DIR / "it_input_file_audit.csv", index=False, encoding="utf-8-sig")
    df.to_csv(OUT_DIR / "it_hysteresis_standardized_smoothed.csv", index=False, encoding="utf-8-sig")
    seg.to_csv(OUT_DIR / "it_segment_summary.csv", index=False, encoding="utf-8-sig")
    step_df.to_csv(OUT_DIR / "it_step_response_fit.csv", index=False, encoding="utf-8-sig")
    pred_df.to_csv(OUT_DIR / "it_step_response_predictions.csv", index=False, encoding="utf-8-sig")
    tau_kan_df.to_csv(OUT_DIR / "it_tau_kan_label_fit.csv", index=False, encoding="utf-8-sig")
    (OUT_DIR / "it_tau_kan_metrics.json").write_text(json.dumps(tau_kan_metrics, indent=2), encoding="utf-8")
    (OUT_DIR / "it_measurement_audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    with pd.ExcelWriter(OUT_DIR / "it_hysteresis_processed_and_fit.xlsx", engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="standardized_smoothed", index=False)
        seg.to_excel(writer, sheet_name="segment_summary", index=False)
        step_df.to_excel(writer, sheet_name="step_response_fit", index=False)
        pred_df.to_excel(writer, sheet_name="fit_predictions", index=False)
        tau_kan_df.to_excel(writer, sheet_name="tau_kan_fit", index=False)
        input_audit.to_excel(writer, sheet_name="input_file_audit", index=False)
        pd.DataFrame(audit.get("repeat_bias_spread", [])).to_excel(writer, sheet_name="repeat_bias_audit", index=False)
        pd.DataFrame(audit.get("static_voltage_alignment", [])).to_excel(writer, sheet_name="static_alignment", index=False)
        pd.DataFrame(audit.get("voltage_correction", [])).to_excel(writer, sheet_name="voltage_correction", index=False)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df, seg, input_audit = load_it_workbooks()
    df, seg, voltage_correction = apply_static_voltage_correction(df, seg)
    step_df, pred_df = fit_all_steps(df)
    tau_kan_df, tau_kan_metrics = fit_tau_label_kan(step_df)
    audit = estimate_measurement_issues(df, seg)
    audit["voltage_correction"] = voltage_correction
    write_outputs(df, seg, step_df, pred_df, tau_kan_df, tau_kan_metrics, audit, input_audit)
    make_plots(df, step_df, pred_df, tau_kan_df)
    if "role" in input_audit.columns:
        expected_file_count = int(input_audit.loc[input_audit["role"] == "expected_full_dataset", "file"].nunique())
    else:
        expected_file_count = None
    if {"used", "path", "file"}.issubset(input_audit.columns):
        used_file_count = int(input_audit.loc[(input_audit["used"] == True) & (input_audit["path"].astype(str).str.len() > 0), "file"].nunique())
    else:
        used_file_count = None
    if {"used", "sheet"}.issubset(input_audit.columns):
        usable_sheet_count = int(input_audit.loc[(input_audit["used"] == True) & input_audit["sheet"].notna()].shape[0])
    else:
        usable_sheet_count = None
    summary = {
        "n_points": int(len(df)),
        "n_curves": int(df["curve_id"].nunique()),
        "n_segments": int(len(seg)),
        "n_step_fits": int(len(step_df)),
        "step_log_rmse_median": float(step_df.loc[step_df["ok"] == True, "log_rmse"].median()) if not step_df.empty else None,
        "tau_kan": tau_kan_metrics,
        "audit_issue_count": int(len(audit["issues"])),
        "audit_issues": audit["issues"],
        "input_files_expected": expected_file_count,
        "input_files_used": used_file_count,
        "usable_sheets": usable_sheet_count,
        "voltage_corrected_segments": int(sum(1 for r in voltage_correction if r.get("use_voltage_correction"))),
        "outputs": str(OUT_DIR),
    }
    (OUT_DIR / "it_hysteresis_analysis_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
