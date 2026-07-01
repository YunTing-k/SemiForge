"""Raw CV workbook parsing and reversible transition centering.

The device model always evaluates raw terminal voltages.  Centered voltage is
an auxiliary training coordinate used to separate common C-V shape extraction
from held-gate transition motion:

    x_centered = x_raw - Vhalf_measured

No measured capacitance value is replaced by its smoothed value.  Smoothing and
monotone projection are used only to robustly estimate transition metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
import re

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CVTransitionMetadata:
    vhalf_V: float
    transition_centroid_V: float
    low_F: float
    high_F: float
    span_F: float
    n_points: int
    max_abs_step_frac_span: float
    negative_steps_gt_5pct_span: int
    complete_fraction: float = 1.0


def smooth_five_point(y: np.ndarray, passes: int = 3) -> np.ndarray:
    out = np.asarray(y, dtype=float).copy()
    if len(out) < 5:
        return out
    kernel = np.asarray([1.0, 4.0, 6.0, 4.0, 1.0]) / 16.0
    for _ in range(max(0, int(passes))):
        out = np.convolve(np.pad(out, (2, 2), mode="edge"), kernel, mode="valid")
    return out


def transition_metadata(x: np.ndarray, cap: np.ndarray) -> CVTransitionMetadata:
    x = np.asarray(x, dtype=float)
    cap = np.asarray(cap, dtype=float)
    valid = np.isfinite(x) & np.isfinite(cap) & (cap > 0.0)
    x, cap = x[valid], cap[valid]
    if len(x) < 3:
        nan = float("nan")
        return CVTransitionMetadata(nan, nan, nan, nan, nan, int(len(x)), nan, 0)

    order = np.argsort(x, kind="stable")
    xs, ys = x[order], cap[order]
    ys_smooth = smooth_five_point(ys)
    edge_n = max(5, int(round(0.15 * len(xs))))
    edge_n = min(edge_n, max(1, len(xs) // 3))
    low = float(np.median(ys_smooth[:edge_n]))
    high = float(np.median(ys_smooth[-edge_n:]))
    direction = 1.0 if high >= low else -1.0
    monotone = direction * np.maximum.accumulate(direction * ys_smooth)
    span = max(abs(high - low), 1.0e-30)
    target = low + 0.5 * (high - low)
    z = direction * (monotone - target)
    crossings = np.where((z[:-1] <= 0.0) & (z[1:] >= 0.0))[0]
    i = int(crossings[0]) if len(crossings) else int(np.argmin(np.abs(z)))
    if i + 1 < len(xs) and abs(monotone[i + 1] - monotone[i]) > 1.0e-30:
        frac = float(np.clip((target - monotone[i]) / (monotone[i + 1] - monotone[i]), 0.0, 1.0))
        vhalf = float(xs[i] + frac * (xs[i + 1] - xs[i]))
    else:
        vhalf = float(xs[i])

    dx = np.maximum(np.diff(xs), 1.0e-12)
    slope = np.maximum(direction * np.diff(ys_smooth) / dx, 0.0)
    if float(np.sum(slope)) > 1.0e-30:
        centroid = float(np.sum(0.5 * (xs[:-1] + xs[1:]) * slope) / np.sum(slope))
    else:
        centroid = vhalf
    dy = np.diff(ys)
    return CVTransitionMetadata(
        vhalf_V=vhalf,
        transition_centroid_V=centroid,
        low_F=low,
        high_F=high,
        span_F=span,
        n_points=int(len(xs)),
        max_abs_step_frac_span=float(np.max(np.abs(dy)) / span) if len(dy) else 0.0,
        negative_steps_gt_5pct_span=int(np.sum(direction * dy / span < -0.05)) if len(dy) else 0,
    )


def centered_coordinate(x: np.ndarray, metadata: CVTransitionMetadata) -> np.ndarray:
    return np.asarray(x, dtype=float) - float(metadata.vhalf_V)


def parse_numeric_header(label: str, key: str) -> float | None:
    match = re.search(rf"{re.escape(key)}\s*=\s*([-+]?\d+(?:\.\d+)?)", str(label), flags=re.I)
    return float(match.group(1)) if match else None


def looks_like_shared_bias_wide_table(raw: pd.DataFrame) -> bool:
    """Return True for one bias column followed by multiple capacitance columns."""

    if raw.shape[0] < 3 or raw.shape[1] < 3:
        return False
    first_header = str(raw.iloc[1, 0]).lower()
    other_headers = [str(x).lower() for x in raw.iloc[1, 1:]]
    first_is_bias = "bias" in first_header or "voltage" in first_header
    others_are_caps = sum(("cap" in x or "(f)" in x) for x in other_headers) >= max(1, len(other_headers) // 2)
    alternating_bias_headers = sum(("bias" in x or "voltage" in x) for x in other_headers)
    return bool(first_is_bias and others_are_caps and alternating_bias_headers == 0)
