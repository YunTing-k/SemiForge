from __future__ import annotations

"""Shared tensor/data utilities for the current DG-TFT engine.

This module contains only data preparation and numerically stable compact-model
helpers used by the full model. It intentionally contains no KAN training
class, training loop, or checkpoint writer.
"""

import math
import os
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from fit_physical_kernel import CVSeries, CVTrapParams, Params, cv_biases_for_series


ROOT = Path(__file__).resolve().parent
RELEASE_ROOT = ROOT.parents[1]
DEFAULT_BASELINE_JSON = (
    RELEASE_ROOT / "trained_model" / "static_signoff" / "fitted_params.json"
)
ID_FLOOR = 1e-15
C_FLOOR = 1e-18
LOG10 = math.log(10.0)


@dataclass
class Normalizer:
    names: list[str]
    center: list[float]
    scale: list[float]


def set_seed(seed: int) -> None:
    # Configure cuBLAS before the first CUDA context/handle is created. Seeds
    # alone do not control the floating-point workspace/reduction algorithm.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def tensor(x, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.as_tensor(x, dtype=dtype, device=device)


def smooth_pos_t(x: torch.Tensor, s: float = 1e-3) -> torch.Tensor:
    return 0.5 * (x + torch.sqrt(x * x + s * s))


def qsolve_t(y: torch.Tensor) -> torch.Tensor:
    y_safe = torch.clamp(y, min=-35.0)
    q = torch.where(
        y_safe > 2.0,
        0.5 * y_safe + 1e-12,
        torch.exp(torch.clamp(y_safe, -35.0, 2.0)),
    )
    q = torch.clamp(q, min=1e-18)
    for _ in range(7):
        f = 2.0 * q + torch.log(q) - y_safe
        fp = 2.0 + 1.0 / q
        q = torch.clamp(q - f / fp, min=1e-18)
    return q


def make_normalizer(names: list[str], values: np.ndarray) -> Normalizer:
    center: list[float] = []
    scale: list[float] = []
    for i in range(values.shape[1]):
        col = values[:, i]
        col = col[np.isfinite(col)]
        if len(col) == 0:
            center.append(0.0)
            scale.append(1.0)
            continue
        lo, hi = np.percentile(col, [2.0, 98.0])
        c = 0.5 * (lo + hi)
        s = 0.5 * (hi - lo)
        if not np.isfinite(s) or s < 1e-9:
            s = np.std(col)
        if not np.isfinite(s) or s < 1e-9:
            s = 1.0
        center.append(float(c))
        scale.append(float(1.10 * s + 1e-12))
    return Normalizer(names, center, scale)


def gate_weights_np(sheet: str) -> tuple[float, float, int]:
    if sheet == "cbg":
        return 0.0, 1.0, 0
    if sheet == "ctg":
        return 1.0, 0.0, 1
    if sheet == "vds":
        return 1.0, 1.0, 3
    return 1.0, 1.0, 2


def iv_weights_np(df: pd.DataFrame) -> np.ndarray:
    abs_id = np.maximum(np.abs(df["ID_A"].to_numpy(float)), ID_FLOOR)
    log_i = np.log10(abs_id)
    w = np.ones(len(df), dtype=float)
    family = df["family"].to_numpy() if "family" in df else np.full(len(df), "")
    w *= np.where(family == "IDVD", 0.90, 1.10)
    w *= np.where((-12.3 <= log_i) & (log_i <= -8.7), 2.25, 1.0)
    w *= np.where(log_i > -8.7, 1.55, 1.0)
    w *= np.where(log_i < -12.8, 0.60, 1.0)
    if "VDS_V" in df:
        w *= np.where(np.abs(df["VDS_V"].to_numpy(float)) < 0.02, 0.25, 1.0)
    return w


def cv_weights_np(df: pd.DataFrame) -> np.ndarray:
    sheet_w = df["sheet"].map({"cgg": 1.12, "ctg": 1.0, "cbg": 1.0, "vds": 0.9}).fillna(1.0)
    return sheet_w.to_numpy(float)


def make_iv_tensors(df: pd.DataFrame, device: torch.device, dtype: torch.dtype) -> dict[str, torch.Tensor]:
    return {
        "vbg": tensor(df["VBG_V"].to_numpy(float), device, dtype),
        "vtg": tensor(df["VTG_V"].to_numpy(float), device, dtype),
        "vds": tensor(df["VDS_V"].to_numpy(float), device, dtype),
        "id": tensor(df["ID_A"].to_numpy(float), device, dtype),
        "weight": tensor(iv_weights_np(df), device, dtype),
    }


def cv_point_dataframe(series: list[CVSeries], trap: CVTrapParams) -> pd.DataFrame:
    rows = []
    for s in series:
        vtg, vbg, vds = cv_biases_for_series(s)
        gt, gb, mode = gate_weights_np(s.sheet)
        freq = trap.default_freq_hz if s.freq_hz is None else s.freq_hz
        temp_k = getattr(s, "temp_K", None)
        x_centered = getattr(s, "x_centered", None)
        if x_centered is None or len(x_centered) != len(s.x):
            vhalf = getattr(s, "vhalf_V", None)
            x_centered = np.asarray(s.x, dtype=float) - (float(vhalf) if vhalf is not None else 0.0)
        vhalf_value = getattr(s, "vhalf_V", None)
        centroid_value = getattr(s, "transition_centroid_V", None)
        low_value = getattr(s, "low_F", None)
        high_value = getattr(s, "high_F", None)
        span_value = getattr(s, "span_F", None)
        jump_value = getattr(s, "max_abs_step_frac_span", None)
        for i in range(len(s.x)):
            rows.append(
                {
                    "sheet": s.sheet,
                    "label": s.label,
                    "x": float(s.x[i]),
                    "x_raw": float(s.x[i]),
                    "x_centered": float(x_centered[i]),
                    "measured_vhalf_V": float(vhalf_value) if vhalf_value is not None else np.nan,
                    "measured_transition_centroid_V": (
                        float(centroid_value) if centroid_value is not None else np.nan
                    ),
                    "measured_low_F": float(low_value) if low_value is not None else np.nan,
                    "measured_high_F": float(high_value) if high_value is not None else np.nan,
                    "measured_span_F": float(span_value) if span_value is not None else np.nan,
                    "complete_fraction": float(getattr(s, "complete_fraction", 1.0)),
                    "max_abs_step_frac_span": float(jump_value) if jump_value is not None else np.nan,
                    "source_layout": str(getattr(s, "source_layout", "paired_columns")),
                    "VBG_V": float(vbg[i]),
                    "VTG_V": float(vtg[i]),
                    "VDS_V": float(vds[i]),
                    "freq_hz": float(freq),
                    "gate_t": gt,
                    "gate_b": gb,
                    "mode": mode,
                    "cap_F": float(s.cap[i]),
                    "temp_K": float(temp_k) if temp_k is not None else np.nan,
                }
            )
    # Keep the tensor interface well-defined when no CV workbook was supplied.
    columns = [
        "sheet", "label", "x", "x_raw", "x_centered", "VBG_V", "VTG_V",
        "VDS_V", "freq_hz", "gate_t", "gate_b", "mode", "cap_F", "temp_K",
    ]
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=columns)


def make_cv_tensors(df: pd.DataFrame, device: torch.device, dtype: torch.dtype) -> dict[str, torch.Tensor]:
    return {
        "vbg": tensor(df["VBG_V"].to_numpy(float), device, dtype),
        "vtg": tensor(df["VTG_V"].to_numpy(float), device, dtype),
        "vds": tensor(df["VDS_V"].to_numpy(float), device, dtype),
        "freq": tensor(df["freq_hz"].to_numpy(float), device, dtype),
        "gt": tensor(df["gate_t"].to_numpy(float), device, dtype),
        "gb": tensor(df["gate_b"].to_numpy(float), device, dtype),
        "mode": torch.as_tensor(df["mode"].to_numpy(int), dtype=torch.long, device=device),
        "cap": tensor(df["cap_F"].to_numpy(float), device, dtype),
        "weight": tensor(cv_weights_np(df), device, dtype),
    }


def sample_idx(n: int, batch_size: int, device: torch.device) -> torch.Tensor:
    if n <= batch_size:
        return torch.arange(n, device=device)
    return torch.randint(0, n, (batch_size,), device=device)


def log10_abs(x: torch.Tensor, floor: float) -> torch.Tensor:
    return torch.log10(torch.clamp(torch.abs(x), min=floor))


def weighted_mse(err: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    err = torch.nan_to_num(
        torch.clamp(err, -8.0, 8.0),
        nan=0.0,
        posinf=8.0,
        neginf=-8.0,
    )
    return torch.mean(weight * err * err) / torch.clamp(torch.mean(weight), min=1e-12)


def make_iv_anchors(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    if n <= 0:
        return pd.DataFrame(columns=["VBG_V", "VTG_V", "VDS_V"])
    vbg_min, vbg_max = df["VBG_V"].min(), df["VBG_V"].max()
    vtg_min, vtg_max = df["VTG_V"].min(), df["VTG_V"].max()
    vds_min, vds_max = df["VDS_V"].min(), df["VDS_V"].max()
    vbg_span = max(vbg_max - vbg_min, 1.0)
    vtg_span = max(vtg_max - vtg_min, 1.0)
    vds_span = max(vds_max - vds_min, 0.5)
    vds_choices = np.unique(np.round(df["VDS_V"].to_numpy(float), 6))
    return pd.DataFrame(
        {
            "VBG_V": rng.uniform(vbg_min - 0.35 * vbg_span, vbg_max + 0.35 * vbg_span, n),
            "VTG_V": rng.uniform(vtg_min - 0.35 * vtg_span, vtg_max + 0.35 * vtg_span, n),
            "VDS_V": rng.choice(
                np.concatenate(
                    [vds_choices, rng.uniform(max(0.0, vds_min), vds_max + 0.35 * vds_span, 8)]
                ),
                n,
                replace=True,
            ),
        }
    )


def make_cv_anchors(cv_df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    if n <= 0 or cv_df.empty:
        return pd.DataFrame(columns=["VBG_V", "VTG_V", "VDS_V", "freq_hz", "gate_t", "gate_b", "mode"])
    vbg_min, vbg_max = cv_df["VBG_V"].min(), cv_df["VBG_V"].max()
    vtg_min, vtg_max = cv_df["VTG_V"].min(), cv_df["VTG_V"].max()
    vbg_span = max(vbg_max - vbg_min, 1.0)
    vtg_span = max(vtg_max - vtg_min, 1.0)
    vds_choices = np.unique(np.round(cv_df["VDS_V"].to_numpy(float), 6))
    freq_choices = np.unique(np.round(cv_df["freq_hz"].to_numpy(float), 6))
    rows = []
    for _ in range(n):
        mode_name = rng.choice(["cgg", "cbg", "ctg", "vds"])
        gt, gb, mode = gate_weights_np(mode_name)
        xbg = rng.uniform(vbg_min - 0.30 * vbg_span, vbg_max + 0.30 * vbg_span)
        xtg = rng.uniform(vtg_min - 0.30 * vtg_span, vtg_max + 0.30 * vtg_span)
        if mode == 2:
            x = 0.5 * (xbg + xtg)
            vbg, vtg, vds = x, x, float(rng.choice(vds_choices))
        elif mode == 0:
            vbg, vtg, vds = xbg, float(rng.choice([-1.0, -0.5, 0.0, 0.5, 1.0])), float(rng.choice(vds_choices))
        elif mode == 1:
            vbg, vtg, vds = float(rng.choice([0.0, 1.0, 2.0, 3.0])), xtg, float(rng.choice(vds_choices))
        else:
            vbg, vtg, vds = xbg, xbg, float(rng.choice(vds_choices))
        rows.append(
            {
                "VBG_V": float(vbg),
                "VTG_V": float(vtg),
                "VDS_V": float(vds),
                "freq_hz": float(rng.choice(freq_choices)),
                "gate_t": gt,
                "gate_b": gb,
                "mode": mode,
            }
        )
    return pd.DataFrame(rows)
