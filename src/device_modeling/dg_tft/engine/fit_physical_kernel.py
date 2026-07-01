from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from cv_preprocessing import (
    centered_coordinate,
    looks_like_shared_bias_wide_table,
    transition_metadata,
)
from measurement_preprocessing import (
    IVAlignmentSettings,
    align_iv_measurement_stages,
    write_iv_alignment_report,
)


ROOT = Path(__file__).resolve().parent
RELEASE_ROOT = ROOT.parents[1]
VA_TEMPLATE = ROOT.parent / "templates" / "physical_kernel_template.va"
DEFAULT_DATA_DIR = RELEASE_ROOT / "data" / "user_dataset"
DEFAULT_OUT_DIR = RELEASE_ROOT / "outputs" / "physical_kernel"

EPS0 = 8.8541878128e-12
K_B = 1.380649e-23
Q_E = 1.602176634e-19
KB_EV = 8.617333262145e-5


@dataclass
class Params:
    W: float = 40e-6
    L: float = 50e-6
    Cox_t: float = EPS0 * 3.9 / 120e-9
    Cox_b: float = EPS0 * 3.9 / 350e-9
    Vth_t0: float = 0.0
    Vth_b0: float = 1.55
    nfac0: float = 4.5
    Vknee_t: float = 1.0e6
    Vknee_b: float = 1.0e6
    Vmodsat_t: float = 1.0e6
    Vmodsat_b: float = 1.0e6
    mu0: float = 1.0e-3
    theta_mu: float = 0.02
    lambda_clm: float = 0.02
    Ioff0: float = 8e-13
    Gleak_ds: float = 1e-13
    Rs0: float = 1.0
    Rd0: float = 1.0
    Tnom: float = 300.0
    kvt_t: float = -2.0e-3
    kvt_b: float = -2.0e-3
    ute: float = -1.5
    Ea_mu: float = 0.0
    knfac: float = 0.0
    trs: float = 0.0
    Ea_rs: float = 0.0
    Cov_tgs_w: float = 0.0
    Cov_tgd_w: float = 0.0
    Cov_bgs_w: float = 0.0
    Cov_bgd_w: float = 0.0
    Cfr_tgs: float = 5e-15
    Cfr_tgd: float = 5e-15
    Cfr_bgs: float = 5e-15
    Cfr_bgd: float = 5e-15


@dataclass
class CVSeries:
    sheet: str
    label: str
    x: np.ndarray
    cap: np.ndarray
    fixed_vtg: float | None = None
    fixed_vbg: float | None = None
    fixed_vds: float = 0.0
    freq_hz: float | None = None
    temp_K: float | None = None
    x_centered: np.ndarray | None = None
    vhalf_V: float | None = None
    transition_centroid_V: float | None = None
    low_F: float | None = None
    high_F: float | None = None
    span_F: float | None = None
    max_abs_step_frac_span: float | None = None
    negative_steps_gt_5pct_span: int = 0
    complete_fraction: float = 1.0
    source_layout: str = "paired_columns"


@dataclass
class CVTrapParams:
    Cit_t: float = 8.0e-14
    Cit_b: float = 8.0e-14
    trap_width: float = 1.0
    tau_s: float = 2.0e-6
    alpha: float = 0.65
    trap_fast_frac: float = 0.25
    default_freq_hz: float = 1.0e5


def smooth_pos(x, s=1e-3):
    x = np.asarray(x, dtype=float)
    return 0.5 * (x + np.sqrt(x * x + s * s))


def one_sided_sat(x, xknee, xsat, s=1e-3):
    p = smooth_pos(np.asarray(x, dtype=float) - xknee, s)
    return np.asarray(x, dtype=float) - p + xsat * (1.0 - np.exp(-p / xsat))


def qsolve(y):
    y = np.asarray(y, dtype=float)
    q = np.empty_like(y, dtype=float)
    strong = y > 2.0
    q[strong] = 0.5 * y[strong] + 1e-12
    q[~strong] = np.exp(np.clip(y[~strong], -80.0, 2.0))
    q = np.maximum(q, 1e-300)
    for _ in range(6):
        f = 2.0 * q + np.log(q) - y
        fp = 2.0 + 1.0 / q
        q = np.maximum(q - f / fp, 1e-300)
    return q


def intrinsic_core(vtg, vbg, vds, p: Params):
    vtg = np.asarray(vtg, dtype=float)
    vbg = np.asarray(vbg, dtype=float)
    vds = np.asarray(vds, dtype=float)
    phi_t = K_B * p.Tnom / Q_E
    cox = p.Cox_t + p.Cox_b
    at = p.Cox_t / cox
    ab = p.Cox_b / cox
    nfac = 1.0 + smooth_pos(p.nfac0 - 1.0, 1e-6)

    vov_t = one_sided_sat(vtg - p.Vth_t0, p.Vknee_t, p.Vmodsat_t)
    vov_b = one_sided_sat(vbg - p.Vth_b0, p.Vknee_b, p.Vmodsat_b)
    vov = at * vov_t + ab * vov_b
    vp = vov / nfac
    qs = qsolve(vp / phi_t)
    qd = qsolve((vp - vds) / phi_t)
    qbar = (
        (2.0 / 3.0) * (qs * qs + qs * qd + qd * qd) + 0.5 * (qs + qd)
    ) / (qs + qd + 1.0)
    return phi_t, cox, at, ab, nfac, vov_t, vov_b, vp, qs, qd, qbar


def intrinsic_ids(vtg, vbg, vds, p: Params):
    phi_t, cox, _, _, nfac, _, _, vp, qs, qd, qbar = intrinsic_core(vtg, vbg, vds, p)
    mu_t = p.mu0
    vmob = 2.0 * nfac * phi_t * qbar
    mu_eff = mu_t / (1.0 + p.theta_mu * vmob)
    ispec = 2.0 * nfac * mu_eff * cox * phi_t * phi_t * p.W / p.L
    id0 = ispec * ((qs * qs + qs) - (qd * qd + qd))

    vabs = np.sqrt(np.asarray(vds, dtype=float) ** 2 + 1.0e-3**2)
    vdsat = smooth_pos(vp, 1.0e-3)
    vexcess = smooth_pos(vabs - vdsat, 1.0e-3)
    return id0 * (1.0 + p.lambda_clm * vexcess)


def leakage_ids(vds, p: Params):
    vds = np.asarray(vds, dtype=float)
    vabs = np.sqrt(vds * vds + 1e-6**2)
    sign = vds / vabs
    turn_on = 1.0 - np.exp(-vabs / 0.02)
    return sign * (p.Ioff0 * turn_on + p.Gleak_ds * vabs)


def external_ids(vtg, vbg, vds, p: Params):
    vtg = np.asarray(vtg, dtype=float)
    vbg = np.asarray(vbg, dtype=float)
    vds = np.asarray(vds, dtype=float)
    rs = math.sqrt(max(p.Rs0, 0.0) ** 2 + 1e-3**2)
    rd = math.sqrt(max(p.Rd0, 0.0) ** 2 + 1e-3**2)
    ids = intrinsic_ids(vtg, vbg, vds, p)
    if max(p.Rs0, p.Rd0) < 10.0:
        return ids + leakage_ids(vds, p)
    for _ in range(30):
        vdi = vds - ids * rd
        vsi = ids * rs
        new_ids = intrinsic_ids(vtg - vsi, vbg - vsi, vdi - vsi, p)
        ids = 0.55 * ids + 0.45 * new_ids
    return ids + leakage_ids(vds, p)


def terminal_charges(vtg, vbg, vds, p: Params):
    vtg = np.asarray(vtg, dtype=float)
    vbg = np.asarray(vbg, dtype=float)
    vds = np.asarray(vds, dtype=float)
    phi_t, cox, at, ab, nfac, vov_t, vov_b, _, qs, qd, qbar = intrinsic_core(vtg, vbg, vds, p)
    delta = vov_t - vov_b
    qtg = p.W * p.L * p.Cox_t * (ab * delta + 2.0 * nfac * phi_t * qbar)
    qbg = p.W * p.L * p.Cox_b * (-at * delta + 2.0 * nfac * phi_t * qbar)

    den = 60.0 * (qs + qd + 1.0) ** 2
    ns = (
        24 * qs**3
        + 48 * qs**2 * qd
        + 32 * qs * qd**2
        + 16 * qd**3
        + 45 * qs**2
        + 50 * qs * qd
        + 25 * qd**2
        + 20 * qs
        + 10 * qd
    )
    nd = (
        16 * qs**3
        + 32 * qs**2 * qd
        + 48 * qs * qd**2
        + 24 * qd**3
        + 25 * qs**2
        + 50 * qs * qd
        + 45 * qd**2
        + 10 * qs
        + 20 * qd
    )
    qs_term = -2.0 * nfac * cox * phi_t * p.W * p.L * ns / den
    qd_term = -2.0 * nfac * cox * phi_t * p.W * p.L * nd / den

    ctgs = p.Cov_tgs_w * p.W + p.Cfr_tgs
    ctgd = p.Cov_tgd_w * p.W + p.Cfr_tgd
    cbgs = p.Cov_bgs_w * p.W + p.Cfr_bgs
    cbgd = p.Cov_bgd_w * p.W + p.Cfr_bgd

    qtg = qtg + ctgs * vtg + ctgd * (vtg - vds)
    qbg = qbg + cbgs * vbg + cbgd * (vbg - vds)
    qd_term = qd_term + ctgd * (vds - vtg) + cbgd * (vds - vbg)
    qs_term = qs_term - ctgs * vtg - cbgs * vbg
    return qtg, qbg, qd_term, qs_term


def load_iv_data(data_dir: Path) -> pd.DataFrame:
    frames = []
    for path in sorted(data_dir.glob("ID*_train.xlsx")):
        if path.name.upper().startswith("CV"):
            continue
        xl = pd.ExcelFile(path)
        if "long" not in xl.sheet_names:
            continue
        df = pd.read_excel(path, sheet_name="long")
        required = {"VD_V", "VBG_V", "VTG_V", "ID_A"}
        missing = required.difference(df.columns)
        if missing:
            raise ValueError(f"{path.name} missing columns: {sorted(missing)}")
        optional_temp_k = [c for c in ["temp_K", "temperature_K", "T_K", "TEMP_K"] if c in df.columns]
        optional_temp_c = [c for c in ["temp_C", "temperature_C", "T_C", "TEMP_C"] if c in df.columns]
        keep_cols = list(required) + optional_temp_k[:1] + optional_temp_c[:1]
        df = df[keep_cols].copy()
        df = df.rename(columns={"VD_V": "VDS_V", "ID_A": "ID_A"})
        if optional_temp_k:
            df = df.rename(columns={optional_temp_k[0]: "temp_K"})
            df["temp_K"] = pd.to_numeric(df["temp_K"], errors="coerce")
        elif optional_temp_c:
            df = df.rename(columns={optional_temp_c[0]: "temp_C"})
            df["temp_K"] = pd.to_numeric(df["temp_C"], errors="coerce") + 273.15
            df = df.drop(columns=["temp_C"])
        df["source_file"] = path.name
        if path.name.startswith("IDVBG"):
            df["family"] = "IDVBG"
        elif path.name.startswith("IDVTG"):
            df["family"] = "IDVTG"
        elif path.name.startswith("IDVD"):
            df["family"] = "IDVD"
        else:
            df["family"] = "IV"
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No *_train.xlsx IV files found in {data_dir}")
    out = pd.concat(frames, ignore_index=True)
    return out.dropna(subset=["VDS_V", "VBG_V", "VTG_V", "ID_A"]).reset_index(drop=True)


def parse_voltage(label: str, key: str) -> float | None:
    m = re.search(rf"{re.escape(key)}\s*=\s*([-+]?\d+(?:\.\d+)?)\s*V", label, flags=re.I)
    return float(m.group(1)) if m else None


def parse_frequency_hz(label: str) -> float | None:
    m = re.search(r"([-+]?\d+(?:\.\d+)?)\s*(kHz|MHz|Hz)", label, flags=re.I)
    if not m:
        return None
    value = float(m.group(1))
    unit = m.group(2).lower()
    if unit == "mhz":
        return value * 1e6
    if unit == "khz":
        return value * 1e3
    return value


def parse_temperature_k(label: str) -> float | None:
    """Parse temperature labels such as T=300K, Temp=25C, or T=-20 degC."""

    m = re.search(r"(?:temp(?:erature)?|T)\s*=\s*([-+]?\d+(?:\.\d+)?)\s*(K|degC|C)?", label, flags=re.I)
    if not m:
        return None
    value = float(m.group(1))
    unit = (m.group(2) or "").lower()
    # Missing-unit labels in lab work are usually Celsius, e.g. "T=25".
    # Values above 170 are interpreted as Kelvin to support labels like T=300.
    if unit in {"c", "degc"} or (not unit and value < 170.0):
        return value + 273.15
    return value


def load_cv_data(data_dir: Path) -> list[CVSeries]:
    path = data_dir / "CV_train.xlsx"
    if not path.exists():
        return []
    series: list[CVSeries] = []
    xl = pd.ExcelFile(path)
    for sheet_name in xl.sheet_names:
        sheet = str(sheet_name).strip().lower()
        raw = pd.read_excel(path, sheet_name=sheet_name, header=None)
        if raw.shape[0] < 3:
            continue
        headers = list(raw.iloc[1])
        if looks_like_shared_bias_wide_table(raw):
            columns = [(0, col, str(headers[col]), "shared_bias_wide") for col in range(1, raw.shape[1])]
        else:
            columns = [
                (col, col + 1, str(headers[col + 1]), "paired_columns")
                for col in range(0, raw.shape[1] - 1, 2)
            ]
        for x_col, cap_col, label, source_layout in columns:
            x = pd.to_numeric(raw.iloc[2:, x_col], errors="coerce")
            cap = pd.to_numeric(raw.iloc[2:, cap_col], errors="coerce")
            mask = x.notna() & cap.notna() & (cap > 0)
            if not mask.any():
                continue
            x_values = x[mask].to_numpy(dtype=float)
            cap_values = cap[mask].to_numpy(dtype=float)
            meta = transition_metadata(x_values, cap_values)
            fixed_vtg = parse_voltage(label, "VTG")
            fixed_vbg = parse_voltage(label, "VBG")
            fixed_vds = parse_voltage(label, "VDS")
            freq_hz = parse_frequency_hz(label)
            temp_k = parse_temperature_k(label)
            series.append(
                CVSeries(
                    sheet=sheet,
                    label=label,
                    x=x_values,
                    cap=cap_values,
                    fixed_vtg=fixed_vtg,
                    fixed_vbg=fixed_vbg,
                    fixed_vds=0.0 if fixed_vds is None else fixed_vds,
                    freq_hz=freq_hz,
                    temp_K=temp_k,
                    x_centered=centered_coordinate(x_values, meta),
                    vhalf_V=meta.vhalf_V,
                    transition_centroid_V=meta.transition_centroid_V,
                    low_F=meta.low_F,
                    high_F=meta.high_F,
                    span_F=meta.span_F,
                    max_abs_step_frac_span=meta.max_abs_step_frac_span,
                    negative_steps_gt_5pct_span=meta.negative_steps_gt_5pct_span,
                    source_layout=source_layout,
                )
            )
    for sheet in sorted(set(s.sheet for s in series)):
        group = [s for s in series if s.sheet == sheet]
        reference_n = max(int(np.median([len(s.x) for s in group])), 1)
        for s in group:
            s.complete_fraction = min(float(len(s.x)) / float(reference_n), 1.0)
    return series


def sample_iv(df: pd.DataFrame, max_points: int, seed: int) -> pd.DataFrame:
    if len(df) <= max_points:
        return df.copy()
    rng = np.random.default_rng(seed)
    pieces = []
    per_file = max(60, max_points // max(df["source_file"].nunique(), 1))
    for _, g in df.groupby("source_file", sort=True):
        n = min(len(g), per_file)
        abs_id = np.maximum(np.abs(g["ID_A"].to_numpy(dtype=float)), 1e-15)
        log_i = np.log10(abs_id)
        weights = np.ones(len(g), dtype=float)
        weights += np.where((-12.2 <= log_i) & (log_i <= -8.5), 3.0, 0.0)
        weights += np.where(log_i > -8.5, 1.5, 0.0)
        weights += np.where(g["family"].isin(["IDVBG", "IDVTG"]), 0.5, 0.0)
        idx = rng.choice(g.index.to_numpy(), size=n, replace=False, p=weights / weights.sum())
        pieces.append(df.loc[np.sort(idx)])
    out = pd.concat(pieces, ignore_index=True)
    if len(out) > max_points:
        out = out.sample(n=max_points, random_state=seed)
    return out.reset_index(drop=True)


def sample_cv(series: list[CVSeries], max_points: int) -> list[CVSeries]:
    total = sum(len(s.x) for s in series)
    if total <= max_points or not series:
        return series
    sampled = []
    for s in series:
        n = max(25, int(round(max_points * len(s.x) / total)))
        n = min(n, len(s.x))
        idx = np.unique(np.linspace(0, len(s.x) - 1, n).astype(int))
        x_centered = s.x_centered[idx] if s.x_centered is not None else None
        sampled.append(replace(s, x=s.x[idx], cap=s.cap[idx], x_centered=x_centered))
    return sampled


def auto_orient_cv_series(
    series: list[CVSeries],
    p: Params,
    span_ratio_min: float = 1.55,
    cox_ratio_min: float = 1.40,
) -> tuple[list[CVSeries], dict]:
    """Correct an evident CBG/CTG worksheet-label interchange for phase 1."""

    def median_span(sheet: str) -> float:
        values = [
            float(s.span_F)
            for s in series
            if s.sheet == sheet and s.span_F is not None and np.isfinite(s.span_F)
        ]
        return float(np.median(values)) if values else 0.0

    cbg_span = median_span("cbg")
    ctg_span = median_span("ctg")
    cox_ratio = float(p.Cox_t / max(p.Cox_b, 1.0e-30))
    span_ratio = cbg_span / max(ctg_span, 1.0e-30)
    swap = bool(cox_ratio >= cox_ratio_min and span_ratio >= span_ratio_min)
    report = {
        "applied_swap": swap,
        "Cox_t_over_Cox_b": cox_ratio,
        "cbg_span_F": cbg_span,
        "ctg_span_F": ctg_span,
        "cbg_span_over_ctg_span": span_ratio,
    }
    if not swap:
        return series, report
    oriented = []
    for s in series:
        if s.sheet == "cbg":
            label = re.sub(r"VTG", "VBG", s.label, flags=re.I)
            oriented.append(
                replace(
                    s,
                    sheet="ctg",
                    label="auto_oriented_" + label,
                    fixed_vbg=s.fixed_vtg,
                    fixed_vtg=None,
                )
            )
        elif s.sheet == "ctg":
            label = re.sub(r"VBG", "VTG", s.label, flags=re.I)
            oriented.append(
                replace(
                    s,
                    sheet="cbg",
                    label="auto_oriented_" + label,
                    fixed_vtg=s.fixed_vbg,
                    fixed_vbg=None,
                )
            )
        else:
            oriented.append(s)
    return oriented, report


def select_primary_cv_series(series: list[CVSeries]) -> tuple[list[CVSeries], list[dict]]:
    """Select clean quasi-static gate CV for the shared physical-kernel fit."""

    selected, decisions = [], []
    for s in series:
        action = "keep"
        reason = "primary_gate_cv"
        if s.sheet == "vds":
            action, reason = "reserve", "drain-terminal/NQS validation"
        elif s.sheet == "cgg" and (s.freq_hz is None or not np.isclose(s.freq_hz, 1.0e4)):
            action, reason = "reserve", "higher-frequency NQS validation"
        elif s.complete_fraction < 0.75:
            action, reason = "drop", f"incomplete_fraction={s.complete_fraction:.3f}"
        elif s.max_abs_step_frac_span is not None and s.max_abs_step_frac_span > 0.20:
            action, reason = "drop", f"single_step_fraction={s.max_abs_step_frac_span:.3f}"
        if action == "keep":
            selected.append(s)
        decisions.append({"sheet": s.sheet, "label": s.label, "action": action, "reason": reason})
    return selected, decisions


PARAM_BOUNDS = {
    "Cox_t": (0.25 * Params.Cox_t, 3.5 * Params.Cox_t),
    "Cox_b": (0.25 * Params.Cox_b, 3.5 * Params.Cox_b),
    "Vth_t0": (-8.0, 8.0),
    "Vth_b0": (-8.0, 12.0),
    "nfac0": (1.15, 10.0),
    "Vknee_t": (2.0, 25.0),
    "Vknee_b": (4.0, 35.0),
    "Vmodsat_t": (3.0, 500.0),
    "Vmodsat_b": (3.0, 700.0),
    "mu0": (1e-8, 0.05),
    "theta_mu": (1e-7, 40.0),
    "lambda_clm": (1e-7, 2.0),
    "Ioff0": (1e-16, 5e-11),
    "Gleak_ds": (1e-17, 5e-11),
    "Rs0": (1e-3, 5e9),
    "Rd0": (1e-3, 5e9),
    "Cfr_tgs": (1e-17, 2e-12),
    "Cfr_tgd": (1e-17, 2e-12),
    "Cfr_bgs": (1e-17, 2e-12),
    "Cfr_bgd": (1e-17, 2e-12),
}
LOG_PARAMS = {
    "Cox_t",
    "Cox_b",
    "mu0",
    "theta_mu",
    "lambda_clm",
    "Ioff0",
    "Gleak_ds",
    "Rs0",
    "Rd0",
    "Vmodsat_t",
    "Vmodsat_b",
    "Cfr_tgs",
    "Cfr_tgd",
    "Cfr_bgs",
    "Cfr_bgd",
}


def pack_params(p: Params, names: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x0, lo, hi = [], [], []
    for name in names:
        value = getattr(p, name)
        low, high = PARAM_BOUNDS[name]
        value = min(max(value, low), high)
        if name in LOG_PARAMS:
            x0.append(math.log(value))
            lo.append(math.log(low))
            hi.append(math.log(high))
        else:
            x0.append(value)
            lo.append(low)
            hi.append(high)
    return np.array(x0), np.array(lo), np.array(hi)


def unpack_params(base: Params, names: list[str], x: np.ndarray) -> Params:
    p = replace(base)
    for name, value in zip(names, x):
        setattr(p, name, math.exp(float(value)) if name in LOG_PARAMS else float(value))
    return p


def predict_iv(df: pd.DataFrame, p: Params) -> np.ndarray:
    return external_ids(
        df["VTG_V"].to_numpy(dtype=float),
        df["VBG_V"].to_numpy(dtype=float),
        df["VDS_V"].to_numpy(dtype=float),
        p,
    )


def iv_residuals(df: pd.DataFrame, p: Params, id_floor: float = 1e-14) -> np.ndarray:
    pred = predict_iv(df, p)
    meas = df["ID_A"].to_numpy(dtype=float)
    abs_meas = np.maximum(np.abs(meas), id_floor)
    abs_pred = np.maximum(np.abs(pred), id_floor)
    log_i = np.log10(abs_meas)
    log_err = np.log10(abs_pred) - np.log10(abs_meas)

    w = np.ones(len(df), dtype=float)
    w *= np.where(df["family"].to_numpy() == "IDVD", 0.90, 1.12)
    w *= np.where((-12.3 <= log_i) & (log_i <= -8.7), 2.35, 1.0)
    w *= np.where(log_i > -8.7, 1.55, 1.0)
    w *= np.where(log_i < -12.8, 0.55, 1.0)
    w *= np.where(np.abs(df["VDS_V"].to_numpy(dtype=float)) < 0.02, 0.25, 1.0)

    scale = np.maximum(abs_meas, np.percentile(abs_meas, 92) * 0.015)
    lin_weight = np.where(abs_meas > 3e-10, 0.28, 0.04)
    lin_err = (pred - meas) / scale
    return np.concatenate([np.sqrt(w) * log_err, lin_weight * lin_err])


def cv_prediction(s: CVSeries, p: Params, h: float = 1e-4) -> np.ndarray:
    return cv_core_prediction_shifted(s, p, 0.0, h)


def cv_core_prediction_shifted(s: CVSeries, p: Params, dv: float = 0.0, h: float = 1e-4) -> np.ndarray:
    x = s.x + dv
    if s.sheet == "cbg":
        vtg = np.full_like(x, 0.0 if s.fixed_vtg is None else s.fixed_vtg)
        vds = np.full_like(x, s.fixed_vds)
        q_plus = terminal_charges(vtg, x + h, vds, p)[1]
        q_minus = terminal_charges(vtg, x - h, vds, p)[1]
        pred = np.abs((q_plus - q_minus) / (2.0 * h))
    elif s.sheet == "ctg":
        vbg = np.full_like(x, 0.0 if s.fixed_vbg is None else s.fixed_vbg)
        vds = np.full_like(x, s.fixed_vds)
        q_plus = terminal_charges(x + h, vbg, vds, p)[0]
        q_minus = terminal_charges(x - h, vbg, vds, p)[0]
        pred = np.abs((q_plus - q_minus) / (2.0 * h))
    else:
        vds = np.full_like(x, s.fixed_vds)
        qtp, qbp, _, _ = terminal_charges(x + h, x + h, vds, p)
        qtm, qbm, _, _ = terminal_charges(x - h, x - h, vds, p)
        pred = np.abs(((qtp + qbp) - (qtm + qbm)) / (2.0 * h))
    return np.maximum(pred, 1e-18)


def cv_residuals(series: list[CVSeries], p: Params) -> np.ndarray:
    if not series:
        return np.array([], dtype=float)
    terms = []
    for s in series:
        pred = cv_prediction(s, p)
        meas = np.maximum(s.cap, 1e-18)
        rel = (pred - meas) / np.maximum(meas, 1e-13)
        log_err = np.log10(pred) - np.log10(meas)
        sheet_weight = {"cgg": 1.1, "ctg": 1.0, "cbg": 1.0, "vds": 0.9}.get(s.sheet, 1.0)
        terms.append(sheet_weight * (0.55 * rel + 0.65 * log_err))
    return np.concatenate(terms)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -70.0, 70.0)))


def cv_biases_for_series(s: CVSeries) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(s.x, dtype=float)
    if s.sheet == "cbg":
        vtg = np.full_like(x, 0.0 if s.fixed_vtg is None else s.fixed_vtg)
        vbg = x
    elif s.sheet == "ctg":
        vtg = x
        vbg = np.full_like(x, 0.0 if s.fixed_vbg is None else s.fixed_vbg)
    else:
        vtg = x
        vbg = x
    vds = np.full_like(x, s.fixed_vds)
    return vtg, vbg, vds


def cv_effective_overdrive(s: CVSeries, p: Params) -> np.ndarray:
    vtg, vbg, _ = cv_biases_for_series(s)
    cox = p.Cox_t + p.Cox_b
    at = p.Cox_t / cox
    ab = p.Cox_b / cox
    # Same threshold and electrostatic weighting as the IV model. No separate CV threshold.
    return at * (vtg - p.Vth_t0) + ab * (vbg - p.Vth_b0)


def trap_frequency_response(freq_hz: float, trap: CVTrapParams) -> float:
    omega_tau = 2.0 * math.pi * max(freq_hz, 1.0) * trap.tau_s
    slow = 1.0 / (1.0 + omega_tau ** trap.alpha)
    return trap.trap_fast_frac + (1.0 - trap.trap_fast_frac) * slow


def cv_trap_capacitance(s: CVSeries, p: Params, trap: CVTrapParams) -> np.ndarray:
    veff = cv_effective_overdrive(s, p)
    occ = sigmoid(veff / trap.trap_width)
    freq = trap.default_freq_hz if s.freq_hz is None else s.freq_hz
    resp = trap_frequency_response(freq, trap)
    if s.sheet == "cbg":
        cit = trap.Cit_b
    elif s.sheet == "ctg":
        cit = trap.Cit_t
    else:
        cit = trap.Cit_t + trap.Cit_b
    return np.maximum(cit * occ * resp, 0.0)


def cv_prediction_shared_trap(s: CVSeries, p: Params, trap: CVTrapParams | None) -> np.ndarray:
    core = cv_prediction(s, p)
    if trap is None:
        return core
    return np.maximum(core + cv_trap_capacitance(s, p, trap), 1e-18)


CV_TRAP_LOG_PARAMS = {"Cit_t", "Cit_b", "trap_width", "tau_s"}
CV_TRAP_BOUNDS = {
    "Cit_t": (1e-16, 6e-13),
    "Cit_b": (1e-16, 6e-13),
    "trap_width": (0.04, 6.0),
    "tau_s": (1e-9, 2e-2),
    "alpha": (0.05, 1.0),
    "trap_fast_frac": (0.0, 1.0),
}


def pack_cv_trap(trap: CVTrapParams, names: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x0, lo, hi = [], [], []
    for name in names:
        value = getattr(trap, name)
        low, high = CV_TRAP_BOUNDS[name]
        value = min(max(value, low), high)
        if name in CV_TRAP_LOG_PARAMS:
            x0.append(math.log(value))
            lo.append(math.log(low))
            hi.append(math.log(high))
        else:
            x0.append(value)
            lo.append(low)
            hi.append(high)
    return np.asarray(x0), np.asarray(lo), np.asarray(hi)


def unpack_cv_trap(base: CVTrapParams, names: list[str], x: np.ndarray) -> CVTrapParams:
    trap = replace(base)
    for name, value in zip(names, x):
        setattr(trap, name, math.exp(float(value)) if name in CV_TRAP_LOG_PARAMS else float(value))
    return trap


def cv_shared_trap_residuals(series: list[CVSeries], p: Params, trap: CVTrapParams) -> np.ndarray:
    terms = []
    for s in series:
        pred = cv_prediction_shared_trap(s, p, trap)
        meas = np.maximum(s.cap, 1e-18)
        rel = (pred - meas) / np.maximum(meas, 1e-13)
        log_err = np.log10(pred) - np.log10(meas)
        weight = {"cgg": 1.05, "cbg": 1.0, "ctg": 1.0, "vds": 0.9}.get(s.sheet, 1.0)
        terms.append(weight * (0.50 * rel + 0.72 * log_err))
    reg = np.array(
        [
            0.10 * math.log(trap.Cit_t / 8e-14) / math.log(10.0),
            0.10 * math.log(trap.Cit_b / 8e-14) / math.log(10.0),
            0.08 * math.log(trap.trap_width / 1.0) / math.log(10.0),
            0.05 * math.log(trap.tau_s / 2e-6) / math.log(10.0),
        ],
        dtype=float,
    )
    return np.concatenate(terms + [reg])


def fit_cv_shared_trap(series: list[CVSeries], p: Params) -> tuple[Params, CVTrapParams | None, dict]:
    if not series:
        return p, None, {"stage": "6_cv_shared_trap", "series": 0}
    base = CVTrapParams()
    p_names = ["Cfr_tgs", "Cfr_tgd", "Cfr_bgs", "Cfr_bgd"]
    trap_names = ["Cit_t", "Cit_b", "trap_width", "tau_s", "alpha", "trap_fast_frac"]
    x0p, lop, hip = pack_params(p, p_names)
    x0t, lot, hit = pack_cv_trap(base, trap_names)
    x0 = np.concatenate([x0p, x0t])
    lo = np.concatenate([lop, lot])
    hi = np.concatenate([hip, hit])
    np_params = len(p_names)

    def residual(x):
        p2 = unpack_params(p, p_names, x[:np_params])
        trap = unpack_cv_trap(base, trap_names, x[np_params:])
        arr = cv_shared_trap_residuals(series, p2, trap)
        fringe_reg = np.array(
            [
                0.05 * math.log(max(p2.Cfr_tgs, 1e-18) / 8e-14) / math.log(10.0),
                0.05 * math.log(max(p2.Cfr_tgd, 1e-18) / 8e-14) / math.log(10.0),
                0.05 * math.log(max(p2.Cfr_bgs, 1e-18) / 2e-14) / math.log(10.0),
                0.05 * math.log(max(p2.Cfr_bgd, 1e-18) / 2e-14) / math.log(10.0),
            ],
            dtype=float,
        )
        arr = np.concatenate([arr, fringe_reg])
        arr[~np.isfinite(arr)] = 1e6
        return np.clip(arr, -1e6, 1e6)

    before = float(np.sqrt(np.mean(residual(x0) ** 2)))
    res = least_squares(
        residual,
        x0,
        bounds=(lo, hi),
        loss="soft_l1",
        f_scale=0.55,
        max_nfev=450,
        x_scale="jac",
    )
    p_out = unpack_params(p, p_names, res.x[:np_params])
    trap = unpack_cv_trap(base, trap_names, res.x[np_params:])
    after = float(np.sqrt(np.mean(residual(res.x) ** 2)))
    print(f"6_cv_shared_trap: rms {before:.4g} -> {after:.4g}, nfev={res.nfev}")
    info = {
        "stage": "6_cv_shared_trap",
        "before_rms": before,
        "after_rms": after,
        "nfev": int(res.nfev),
        "params_after": asdict(trap),
        "fringe_after": {
            "Cfr_tgs": p_out.Cfr_tgs,
            "Cfr_tgd": p_out.Cfr_tgd,
            "Cfr_bgs": p_out.Cfr_bgs,
            "Cfr_bgd": p_out.Cfr_bgd,
        },
    }
    return p_out, trap, info


def regularization(p: Params, process_top: float, process_bottom: float) -> np.ndarray:
    return np.array(
        [
            0.65 * math.log(p.Cox_t / process_top) / math.log(2.0),
            0.65 * math.log(p.Cox_b / process_bottom) / math.log(2.0),
            0.10 * (p.nfac0 - 4.5) / 4.5,
            0.05 * math.log(p.mu0 / 1e-3) / math.log(10.0),
        ],
        dtype=float,
    )


def make_residual_function(
    base: Params,
    names: list[str],
    iv_df: pd.DataFrame | None,
    cv_series: list[CVSeries] | None,
    cv_weight: float,
    reg_weight: float,
    process_top: float,
    process_bottom: float,
) -> Callable[[np.ndarray], np.ndarray]:
    def residual(x):
        p = unpack_params(base, names, x)
        pieces = []
        if iv_df is not None and len(iv_df):
            pieces.append(iv_residuals(iv_df, p))
        if cv_series:
            pieces.append(cv_weight * cv_residuals(cv_series, p))
        if reg_weight:
            pieces.append(reg_weight * regularization(p, process_top, process_bottom))
        if not pieces:
            return np.zeros(1)
        arr = np.concatenate(pieces)
        arr[~np.isfinite(arr)] = 1e6
        return np.clip(arr, -1e6, 1e6)

    return residual


def run_stage(
    stage: str,
    p: Params,
    names: list[str],
    iv_df: pd.DataFrame | None,
    cv_series: list[CVSeries] | None,
    cv_weight: float,
    reg_weight: float,
    process_top: float,
    process_bottom: float,
    max_nfev: int,
    verbose: bool,
) -> tuple[Params, dict]:
    x0, lo, hi = pack_params(p, names)
    residual = make_residual_function(
        p, names, iv_df, cv_series, cv_weight, reg_weight, process_top, process_bottom
    )
    before = float(np.sqrt(np.mean(residual(x0) ** 2)))
    res = least_squares(
        residual,
        x0,
        bounds=(lo, hi),
        loss="soft_l1",
        f_scale=1.0,
        max_nfev=max_nfev,
        x_scale="jac",
        verbose=2 if verbose else 0,
    )
    out = unpack_params(p, names, res.x)
    after = float(np.sqrt(np.mean(residual(res.x) ** 2)))
    accepted = after <= before * 1.002
    if not accepted:
        out = p
        after = before
    info = {
        "stage": stage,
        "parameters": names,
        "before_rms": before,
        "after_rms": after,
        "accepted": accepted,
        "nfev": int(res.nfev),
        "status": int(res.status),
        "message": str(res.message),
    }
    suffix = "" if accepted else " (kept previous parameters)"
    print(f"{stage}: rms {before:.4g} -> {after:.4g}, nfev={res.nfev}{suffix}")
    return out, info


def fit_model(
    iv_fit: pd.DataFrame,
    cv_fit: list[CVSeries],
    max_nfev_scale: float,
    verbose: bool,
) -> tuple[Params, list[dict]]:
    p = Params()
    process_top = p.Cox_t
    process_bottom = p.Cox_b
    history = []

    low_transfer = iv_fit[(iv_fit["family"].isin(["IDVBG", "IDVTG"])) & (iv_fit["VDS_V"].abs() <= 0.55)]
    transfer = iv_fit[iv_fit["family"].isin(["IDVBG", "IDVTG"])]
    idvd = iv_fit[iv_fit["family"] == "IDVD"]

    stages = [
        (
            "1_low_vds_transfer",
            ["Vth_t0", "Vth_b0", "nfac0", "mu0", "Ioff0", "Gleak_ds"],
            low_transfer,
            None,
            0.0,
            0.65,
            70,
        ),
        (
            "2_all_transfer_gate_shape",
            [
                "Vth_t0",
                "Vth_b0",
                "nfac0",
                "mu0",
                "theta_mu",
                "Ioff0",
                "Gleak_ds",
            ],
            transfer,
            None,
            0.0,
            0.35,
            85,
        ),
        (
            "3_output_curves",
            ["mu0", "theta_mu", "lambda_clm", "Ioff0", "Gleak_ds", "Vth_t0", "Vth_b0"],
            pd.concat([idvd, transfer.sample(min(len(transfer), max(len(idvd) // 2, 1)), random_state=12)]),
            None,
            0.0,
            0.20,
            80,
        ),
        (
            "4_cv_capacitance",
            ["Cfr_tgs", "Cfr_tgd", "Cfr_bgs", "Cfr_bgd"],
            None,
            cv_fit,
            1.0,
            0.20,
            90,
        ),
        (
            "5_joint_refine",
            [
                "Vth_t0",
                "Vth_b0",
                "nfac0",
                "mu0",
                "theta_mu",
                "lambda_clm",
                "Ioff0",
                "Gleak_ds",
                "Cfr_tgs",
                "Cfr_tgd",
                "Cfr_bgs",
                "Cfr_bgd",
            ],
            iv_fit,
            cv_fit,
            0.22,
            0.45,
            120,
        ),
    ]
    for stage in stages:
        name, param_names, iv_df, cv_series, cv_weight, reg_weight, base_nfev = stage
        nfev = max(12, int(round(base_nfev * max_nfev_scale)))
        p, info = run_stage(
            name,
            p,
            param_names,
            iv_df,
            cv_series,
            cv_weight,
            reg_weight,
            process_top,
            process_bottom,
            nfev,
            verbose,
        )
        history.append({**info, "params_after": asdict(p)})
    return p, history


def compute_metrics(
    iv_df: pd.DataFrame,
    cv_series: list[CVSeries],
    p: Params,
    cv_trap: CVTrapParams | None = None,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    pred = predict_iv(iv_df, p)
    meas = iv_df["ID_A"].to_numpy(dtype=float)
    abs_meas = np.maximum(np.abs(meas), 1e-14)
    abs_pred = np.maximum(np.abs(pred), 1e-14)
    log_err = np.log10(abs_pred) - np.log10(abs_meas)
    rel_err = (pred - meas) / np.maximum(abs_meas, 1e-14)
    iv_df = iv_df.copy()
    iv_df["ID_fit_A"] = pred
    iv_df["log10_abs_error_dec"] = log_err
    iv_df["relative_error"] = rel_err

    rows = []
    for (family, source), g in iv_df.groupby(["family", "source_file"]):
        rows.append(
            {
                "kind": "IV",
                "family": family,
                "source": source,
                "points": len(g),
                "rms_log10_error_dec": float(np.sqrt(np.mean(g["log10_abs_error_dec"] ** 2))),
                "median_abs_log10_error_dec": float(np.median(np.abs(g["log10_abs_error_dec"]))),
                "median_abs_relative_error": float(np.median(np.abs(g["relative_error"]))),
            }
        )

    nonmonotonic_steps = 0
    checked_steps = 0
    for (family, source), g0 in iv_df.groupby(["family", "source_file"]):
        if family == "IDVBG":
            xcol, fixed_cols = "VBG_V", ["VTG_V"]
        elif family == "IDVTG":
            xcol, fixed_cols = "VTG_V", ["VBG_V"]
        elif family == "IDVD":
            xcol = "VDS_V"
            fixed_cols = ["VTG_V"] if g0["VBG_V"].nunique() == 1 else ["VBG_V"]
        else:
            continue
        for _, gg in g0.groupby(fixed_cols, sort=False):
            gg = gg.sort_values(xcol)
            y = np.log10(np.maximum(np.abs(gg["ID_fit_A"].to_numpy(dtype=float)), 1e-14))
            if len(y) > 1:
                dy = np.diff(y)
                nonmonotonic_steps += int(np.sum(dy < -0.03))
                checked_steps += len(dy)
    for region, mask in {
        "off_lt_1e-12": abs_meas < 1e-12,
        "sub_1e-12_to_1e-9": (abs_meas >= 1e-12) & (abs_meas < 1e-9),
        "on_ge_1e-9": abs_meas >= 1e-9,
    }.items():
        if mask.any():
            rows.append(
                {
                    "kind": "IV",
                    "family": region,
                    "source": "all",
                    "points": int(mask.sum()),
                    "rms_log10_error_dec": float(np.sqrt(np.mean(log_err[mask] ** 2))),
                    "median_abs_log10_error_dec": float(np.median(np.abs(log_err[mask]))),
                    "median_abs_relative_error": float(np.median(np.abs(rel_err[mask]))),
                }
            )

    cv_abs_terms = []
    cv_core_terms = []
    for s in cv_series:
        pred_core = cv_prediction(s, p)
        pred_c = cv_prediction_shared_trap(s, p, cv_trap)
        meas_c = np.maximum(s.cap, 1e-18)
        rel = (pred_c - meas_c) / np.maximum(meas_c, 1e-18)
        logc = np.log10(pred_c) - np.log10(meas_c)
        core_logc = np.log10(np.maximum(pred_core, 1e-18)) - np.log10(meas_c)
        cv_abs_terms.extend(logc.tolist())
        cv_core_terms.extend(core_logc.tolist())
        rows.append(
            {
                "kind": "CV",
                "family": s.sheet,
                "source": s.label,
                "points": len(s.cap),
                "rms_log10_error_dec": float(np.sqrt(np.mean(logc**2))),
                "median_abs_log10_error_dec": float(np.median(np.abs(logc))),
                "median_abs_relative_error": float(np.median(np.abs(rel))),
            }
        )

    summary = {
        "iv_rms_log10_error_dec": float(np.sqrt(np.mean(log_err**2))),
        "iv_median_abs_log10_error_dec": float(np.median(np.abs(log_err))),
        "iv_on_rms_log10_error_dec": float(
            np.sqrt(np.mean(log_err[abs_meas >= 1e-9] ** 2)) if np.any(abs_meas >= 1e-9) else np.nan
        ),
        "iv_subthreshold_rms_log10_error_dec": float(
            np.sqrt(np.mean(log_err[(abs_meas >= 1e-12) & (abs_meas < 1e-9)] ** 2))
            if np.any((abs_meas >= 1e-12) & (abs_meas < 1e-9))
            else np.nan
        ),
        "cv_rms_log10_error_dec": float(np.sqrt(np.mean(np.asarray(cv_abs_terms) ** 2))) if cv_abs_terms else np.nan,
        "cv_core_qs_rms_log10_error_dec": float(np.sqrt(np.mean(np.asarray(cv_core_terms) ** 2)))
        if cv_core_terms
        else np.nan,
        "fit_nonmonotonic_steps": int(nonmonotonic_steps),
        "fit_checked_steps": int(checked_steps),
    }
    return summary, pd.DataFrame(rows), iv_df


def patch_va(template: Path, out: Path, p: Params):
    text = template.read_text(encoding="utf-8")
    if "parameter real Ioff0" not in text:
        text = text.replace(
            "    parameter real lambda_clm = 0.0 from [0:inf);  // 1/V, channel-length modulation\n",
            "    parameter real lambda_clm = 0.0 from [0:inf);  // 1/V, channel-length modulation\n"
            "    parameter real Ioff0     = 0.0 from [0:inf);  // A, off-state leakage floor\n"
            "    parameter real Gleak_ds  = 0.0 from [0:inf);  // A/V, drain-source leakage\n",
        )
    if "real Vds_ext_abs, Ileak_ds;" not in text:
        text = text.replace(
            "    real Vmob, Vabs, Vdsat, Vexcess, clm;\n",
            "    real Vmob, Vabs, Vdsat, Vexcess, clm;\n"
            "    real Vds_ext_abs, Ileak_ds;\n",
        )
        text = text.replace(
            "        I(di, si) <+ Ids;\n",
            "        I(di, si) <+ Ids;\n\n"
            "        Vds_ext_abs = sqrt(V(d, s)*V(d, s) + 1.0e-12);\n"
            "        Ileak_ds = V(d, s)/Vds_ext_abs\n"
            "                 * (Ioff0*(1.0 - exp(-Vds_ext_abs/0.02)) + Gleak_ds*Vds_ext_abs);\n"
            "        I(d, s) <+ Ileak_ds;\n",
        )
    for key, value in asdict(p).items():
        if key == "Tnom":
            continue
        pattern = rf"(parameter\s+real\s+{re.escape(key)}\s*=\s*)([-+0-9.eE]+)"
        text = re.sub(pattern, rf"\g<1>{value:.12g}", text)
    out.write_text(text, encoding="utf-8")


def setup_plot_style():
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "axes.linewidth": 1.5,
            "axes.labelweight": "bold",
            "axes.titleweight": "bold",
            "xtick.direction": "in",
            "ytick.direction": "in",
            "figure.dpi": 120,
        }
    )


def plot_id_family(iv_pred: pd.DataFrame, p: Params, family: str, out: Path):
    setup_plot_style()
    data = iv_pred[iv_pred["family"] == family].copy()
    if data.empty:
        return
    files = list(data["source_file"].drop_duplicates())
    ncols = min(3, len(files))
    nrows = int(math.ceil(len(files) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.2 * ncols, 4.8 * nrows), squeeze=False)
    for ax, source in zip(axes.ravel(), files):
        g = data[data["source_file"] == source]
        if family == "IDVBG":
            xcol, fixed = "VBG_V", "VTG_V"
            xlabel = "VBG (V)"
        elif family == "IDVTG":
            xcol, fixed = "VTG_V", "VBG_V"
            xlabel = "VTG (V)"
        else:
            xcol, fixed = "VDS_V", "VTG_V" if g["VBG_V"].nunique() == 1 else "VBG_V"
            xlabel = "VD (V)"
        fixed_values = sorted(g[fixed].drop_duplicates())
        cmap = plt.get_cmap("viridis")
        for i, val in enumerate(fixed_values):
            gg = g[np.isclose(g[fixed], val)].sort_values(xcol)
            color = cmap(i / max(len(fixed_values) - 1, 1))
            ax.semilogy(gg[xcol], np.maximum(np.abs(gg["ID_A"]), 1e-14), "-", color=color, lw=0.9, alpha=0.55)
            ax.semilogy(gg[xcol], np.maximum(np.abs(gg["ID_fit_A"]), 1e-14), "--", color=color, lw=1.15, alpha=0.9)
        ax.set_title(source.replace("_train.xlsx", ""), fontsize=12)
        ax.set_xlabel(xlabel, fontsize=11, fontweight="bold")
        ax.set_ylabel("|ID| (A)", fontsize=11, fontweight="bold")
        ax.grid(True, which="both", alpha=0.22)
    for ax in axes.ravel()[len(files) :]:
        ax.axis("off")
    fig.suptitle(f"{family}: measured solid, fitted dashed", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out, dpi=220)
    plt.close(fig)


def plot_cv(series: list[CVSeries], p: Params, out: Path, cv_trap: CVTrapParams | None = None):
    if not series:
        return
    setup_plot_style()
    sheets = ["cgg", "cbg", "ctg", "vds"]
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.5), squeeze=False)
    for ax, sheet in zip(axes.ravel(), sheets):
        group = [s for s in series if s.sheet == sheet]
        if not group:
            ax.axis("off")
            continue
        cmap = plt.get_cmap("plasma")
        for i, s in enumerate(group):
            color = cmap(i / max(len(group) - 1, 1))
            pred = cv_prediction_shared_trap(s, p, cv_trap)
            short = re.sub(r"^.*@\s*", "", s.label).replace(" (F) (V)", "").replace(" (F)", "")
            ax.plot(s.x, s.cap, "-", color=color, lw=1.1, alpha=0.7, label=short if i < 8 else None)
            ax.plot(s.x, pred, "--", color=color, lw=1.25)
        ax.set_title(sheet, fontsize=12)
        ax.set_xlabel("Corrected bias coordinate (V)", fontsize=11, fontweight="bold")
        ax.set_ylabel("Capacitance (F)", fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.25)
        if len(group) <= 8:
            ax.legend(fontsize=7, frameon=False)
    fig.suptitle("CV: measured solid, fitted dashed", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out, dpi=220)
    plt.close(fig)


def write_outputs(
    out_dir: Path,
    p: Params,
    history: list[dict],
    summary: dict,
    metrics: pd.DataFrame,
    iv_pred: pd.DataFrame,
    cv_series: list[CVSeries],
    cv_trap: CVTrapParams | None,
    template_va: Path,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in ["cv_enhanced_params.csv"]:
        stale_path = out_dir / stale
        if stale_path.exists():
            stale_path.unlink()
    patch_va(template_va, out_dir / "dg_tft_ekv_level2_fitted.va", p)
    payload = {
        "final_params": asdict(p),
        "cv_shared_trap_params": asdict(cv_trap) if cv_trap is not None else None,
        "metrics_summary": summary,
        "fit_history": history,
        "geometry": {"W_m": p.W, "L_m": p.L, "W_um": p.W * 1e6, "L_um": p.L * 1e6},
    }
    (out_dir / "fitted_params.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    metrics.to_csv(out_dir / "fit_metrics.csv", index=False, encoding="utf-8-sig")
    iv_pred.to_csv(out_dir / "iv_measured_vs_fitted.csv", index=False, encoding="utf-8-sig")
    if cv_trap is not None:
        (out_dir / "cv_shared_trap_params.json").write_text(
            json.dumps(asdict(cv_trap), indent=2), encoding="utf-8"
        )
        pd.DataFrame([asdict(cv_trap)]).to_csv(
            out_dir / "cv_shared_trap_params.csv", index=False, encoding="utf-8-sig"
        )
        cv_rows = []
        for s in cv_series:
            pred = cv_prediction_shared_trap(s, p, cv_trap)
            core = cv_prediction(s, p)
            trap_cap = cv_trap_capacitance(s, p, cv_trap)
            for x, meas, fit, core_fit, trap_fit in zip(s.x, s.cap, pred, core, trap_cap):
                cv_rows.append(
                    {
                        "sheet": s.sheet,
                        "label": s.label,
                        "freq_hz": s.freq_hz,
                        "fixed_vtg": s.fixed_vtg,
                        "fixed_vbg": s.fixed_vbg,
                        "fixed_vds": s.fixed_vds,
                        "bias_V": x,
                        "cap_measured_F": meas,
                        "cap_fit_F": fit,
                        "cap_core_qs_F": core_fit,
                        "cap_trap_F": trap_fit,
                    }
                )
        pd.DataFrame(cv_rows).to_csv(out_dir / "cv_measured_vs_fitted.csv", index=False, encoding="utf-8-sig")
        equation_lines = [
            "Shared trap-dispersion CV model used for compact-model fitting",
            "",
            "The IV and CV paths share W, L, Cox_t, Cox_b, Vth_t0, Vth_b0, nfac0 and the terminal fringe capacitances.",
            "No separate CV threshold, CV Cox, area scale, or per-curve voltage shift is fitted.",
            "",
            "Effective channel overdrive:",
            "  Veff = (Cox_t/(Cox_t+Cox_b))*(Vtg - Vth_t0) + (Cox_b/(Cox_t+Cox_b))*(Vbg - Vth_b0)",
            "",
            "Trap occupancy broadening:",
            "  Ftrap(Veff) = 1/(1 + exp(-Veff/trap_width))",
            "",
            "Frequency response:",
            "  H(f) = trap_fast_frac + (1 - trap_fast_frac)/(1 + (2*pi*f*tau_s)^alpha)",
            "",
            "Additional measured capacitance contribution:",
            "  cgg/vds tied-gate: Ctrap = (Cit_t + Cit_b)*Ftrap*H(f)",
            "  ctg top-gate:       Ctrap = Cit_t*Ftrap*H(f)",
            "  cbg bottom-gate:    Ctrap = Cit_b*Ftrap*H(f)",
            "",
            "Total fit capacitance:",
            "  Cfit = C_QS_from_charge_model + Ctrap",
        ]
        (out_dir / "cv_shared_trap_model_equations.txt").write_text(
            "\n".join(equation_lines), encoding="utf-8"
        )

    lines = [
        "DG-TFT EKV Level-2 training-ready fit",
        f"W/L = {p.W * 1e6:.6g} um / {p.L * 1e6:.6g} um",
        f"IV RMS log10(|ID|) error = {summary['iv_rms_log10_error_dec']:.4g} dec",
        f"IV subthreshold RMS log10(|ID|) error = {summary['iv_subthreshold_rms_log10_error_dec']:.4g} dec",
        f"IV on-state RMS log10(|ID|) error = {summary['iv_on_rms_log10_error_dec']:.4g} dec",
        f"CV shared-trap RMS log10(C) error = {summary['cv_rms_log10_error_dec']:.4g} dec",
        f"CV core-QS RMS log10(C) error = {summary['cv_core_qs_rms_log10_error_dec']:.4g} dec",
        "",
        "Final parameters:",
    ]
    for key, value in asdict(p).items():
        lines.append(f"  {key} = {value:.12g}")
    if cv_trap is not None:
        lines.extend(["", "Shared trap CV parameters:"])
        for key, value in asdict(cv_trap).items():
            lines.append(f"  {key} = {value:.12g}")
    lines.extend(["", "Stage history:"])
    for item in history:
        if "before_rms" in item:
            lines.append(
                f"  {item['stage']}: {item['before_rms']:.4g} -> {item['after_rms']:.4g}, nfev={item['nfev']}"
            )
        else:
            lines.append(f"  {item['stage']}: {item}")
    (out_dir / "fit_report.txt").write_text("\n".join(lines), encoding="utf-8")

    plot_id_family(iv_pred, p, "IDVBG", out_dir / "fit_IDVBG.png")
    plot_id_family(iv_pred, p, "IDVTG", out_dir / "fit_IDVTG.png")
    plot_id_family(iv_pred, p, "IDVD", out_dir / "fit_IDVD.png")
    plot_cv(cv_series, p, out_dir / "fit_CV.png", cv_trap)


def main():
    ap = argparse.ArgumentParser(description="Fit corrected training-ready DG-TFT IV/CV xlsx data.")
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--template-va", type=Path, default=VA_TEMPLATE)
    ap.add_argument("--max-iv-points", type=int, default=10000)
    ap.add_argument("--max-cv-points", type=int, default=1200)
    ap.add_argument("--max-nfev-scale", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=20260522)
    ap.add_argument("--iv-stage-alignment", choices=["off", "auto"], default="auto")
    ap.add_argument("--iv-stage-max-shift-V", type=float, default=0.75)
    ap.add_argument("--iv-stage-min-matches", type=int, default=2)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    if not args.data_dir.exists():
        raise FileNotFoundError(args.data_dir)
    if not args.template_va.exists():
        raise FileNotFoundError(args.template_va)

    iv_all = load_iv_data(args.data_dir)
    iv_all, iv_alignment_rows, iv_alignment_report = align_iv_measurement_stages(
        iv_all,
        IVAlignmentSettings(
            mode=args.iv_stage_alignment,
            max_shift_V=args.iv_stage_max_shift_V,
            min_matched_curves=args.iv_stage_min_matches,
        ),
    )
    cv_all = load_cv_data(args.data_dir)
    cv_all, cv_orientation = auto_orient_cv_series(cv_all, Params())
    cv_all, cv_decisions = select_primary_cv_series(cv_all)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_iv_alignment_report(args.out_dir, iv_alignment_rows, iv_alignment_report)
    (args.out_dir / "v92_phase1_cv_gate_orientation.json").write_text(
        json.dumps(cv_orientation, indent=2),
        encoding="utf-8",
    )
    pd.DataFrame(cv_decisions).to_csv(
        args.out_dir / "v92_phase1_cv_data_decisions.csv",
        index=False,
    )
    iv_fit = sample_iv(iv_all, args.max_iv_points, args.seed)
    cv_fit = sample_cv(cv_all, args.max_cv_points)

    print(f"Loaded IV points: {len(iv_all)} total, {len(iv_fit)} used for fitting")
    print(f"Loaded CV series: {len(cv_all)}, CV points used for fitting: {sum(len(s.x) for s in cv_fit)}")
    params, history = fit_model(iv_fit, cv_fit, args.max_nfev_scale, args.verbose)
    params, cv_trap, cv_info = fit_cv_shared_trap(cv_all, params)
    history.append(cv_info)
    summary, metrics, iv_pred = compute_metrics(iv_all, cv_all, params, cv_trap)
    write_outputs(args.out_dir, params, history, summary, metrics, iv_pred, cv_all, cv_trap, args.template_va)
    print(f"Done. Outputs written to: {args.out_dir}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
