"""Portable preprocessing for raw IV/CV measurement workbooks.

The module distinguishes device physics from measurement-session drift:

* original voltages are always retained in ``*_raw_V`` columns;
* an IV stage shift is applied only when repeated physical bias conditions
  make that shift identifiable;
* unique-VDS files are never aligned against each other because that would
  remove real DIBL/output-bias behavior;
* CV centering remains auxiliary and is implemented in ``cv_preprocessing``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class IVAlignmentSettings:
    mode: str = "auto"
    max_shift_V: float = 0.75
    min_matched_curves: int = 2
    voltage_tolerance_V: float = 0.03
    temperature_tolerance_K: float = 2.0


def _smooth_log_current(y: np.ndarray, passes: int = 2) -> np.ndarray:
    out = np.log10(np.maximum(np.abs(np.asarray(y, dtype=float)), 1.0e-15))
    for _ in range(max(0, int(passes))):
        if len(out) < 3:
            break
        pad = np.pad(out, (1, 1), mode="edge")
        out = 0.25 * pad[:-2] + 0.50 * pad[1:-1] + 0.25 * pad[2:]
    return out


def _transition_center(x: np.ndarray, current: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    current = np.asarray(current, dtype=float)
    valid = np.isfinite(x) & np.isfinite(current)
    if np.count_nonzero(valid) < 8:
        return float("nan")
    x, current = x[valid], current[valid]
    order = np.argsort(x, kind="stable")
    x = x[order]
    log_i = _smooth_log_current(current[order])
    edge_n = max(3, min(len(x) // 4, int(round(0.15 * len(x)))))
    lo = float(np.median(log_i[:edge_n]))
    hi = float(np.median(log_i[-edge_n:]))
    if abs(hi - lo) < 0.5:
        return float("nan")
    direction = 1.0 if hi >= lo else -1.0
    monotone = direction * np.maximum.accumulate(direction * log_i)
    target = 0.5 * (lo + hi)
    z = direction * (monotone - target)
    hits = np.where((z[:-1] <= 0.0) & (z[1:] >= 0.0))[0]
    i = int(hits[0]) if len(hits) else int(np.argmin(np.abs(z)))
    if i + 1 >= len(x) or abs(monotone[i + 1] - monotone[i]) < 1.0e-12:
        return float(x[i])
    frac = float(np.clip((target - monotone[i]) / (monotone[i + 1] - monotone[i]), 0.0, 1.0))
    return float(x[i] + frac * (x[i + 1] - x[i]))


def _curve_table(iv: pd.DataFrame, settings: IVAlignmentSettings) -> pd.DataFrame:
    rows: list[dict] = []
    has_temp = "temp_K" in iv.columns
    for family in ("IDVBG", "IDVTG"):
        sub = iv[iv["family"].eq(family)]
        if sub.empty:
            continue
        sweep = "VBG_V" if family == "IDVBG" else "VTG_V"
        held = "VTG_V" if family == "IDVBG" else "VBG_V"
        group_cols = ["source_file", "VDS_V", held]
        if has_temp:
            group_cols.append("temp_K")
        for keys, g in sub.groupby(group_cols, dropna=False, sort=True):
            if not isinstance(keys, tuple):
                keys = (keys,)
            info = dict(zip(group_cols, keys))
            center = _transition_center(g[sweep].to_numpy(float), g["ID_A"].to_numpy(float))
            if not np.isfinite(center):
                continue
            rows.append(
                {
                    "family": family,
                    "source_file": str(info["source_file"]),
                    "VDS_V": float(info["VDS_V"]),
                    "held_V": float(info[held]),
                    "temp_K": float(info.get("temp_K", 300.0)),
                    "transition_center_V": center,
                    "points": int(len(g)),
                }
            )
    return pd.DataFrame(rows)


def _condition_key(row: pd.Series, settings: IVAlignmentSettings) -> tuple:
    dv = max(settings.voltage_tolerance_V, 1.0e-6)
    dt = max(settings.temperature_tolerance_K, 1.0e-6)
    return (
        str(row["family"]),
        int(round(float(row["VDS_V"]) / dv)),
        int(round(float(row["held_V"]) / dv)),
        int(round(float(row["temp_K"]) / dt)),
    )


def align_iv_measurement_stages(
    iv: pd.DataFrame,
    settings: IVAlignmentSettings | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Apply only identifiable per-source transfer-voltage shifts.

    A source receives a correction only when at least ``min_matched_curves``
    physical bias conditions are repeated in another source of the same
    transfer family. This overlap guard prevents real VDS dependence from
    being mistaken for test-stage Vth drift.
    """

    cfg = settings or IVAlignmentSettings()
    out = iv.copy()
    for name in ("VBG", "VTG", "VDS"):
        col = f"{name}_V"
        out[f"{name}_raw_V"] = pd.to_numeric(out[col], errors="coerce")
    out["iv_stage_shift_V"] = 0.0
    out["iv_stage_alignment_applied"] = False
    out["iv_stage_alignment_reason"] = "disabled" if cfg.mode == "off" else "not_identifiable"

    curves = _curve_table(out, cfg)
    if cfg.mode == "off" or curves.empty:
        report = {
            "mode": cfg.mode,
            "applied_sources": 0,
            "reason": "disabled_or_no_transfer_curves",
        }
        return out, curves, report

    curves = curves.copy()
    curves["condition_key"] = curves.apply(lambda row: _condition_key(row, cfg), axis=1)
    comparisons: list[dict] = []
    source_offsets: dict[tuple[str, str], list[float]] = {}
    for (family, condition), g in curves.groupby(["family", "condition_key"], sort=True):
        if g["source_file"].nunique() < 2:
            continue
        reference = float(np.median(g["transition_center_V"]))
        for _, row in g.iterrows():
            key = (str(family), str(row["source_file"]))
            delta = float(row["transition_center_V"] - reference)
            source_offsets.setdefault(key, []).append(delta)
            comparisons.append(
                {
                    "family": family,
                    "source_file": row["source_file"],
                    "condition_key": str(condition),
                    "transition_center_V": float(row["transition_center_V"]),
                    "condition_reference_V": reference,
                    "raw_delta_V": delta,
                }
            )

    rows = []
    for (family, source), values in sorted(source_offsets.items()):
        count = len(values)
        raw_shift = float(np.median(values))
        applied = count >= int(cfg.min_matched_curves)
        shift = float(np.clip(raw_shift, -cfg.max_shift_V, cfg.max_shift_V)) if applied else 0.0
        rows.append(
            {
                "family": family,
                "source_file": source,
                "matched_conditions": count,
                "raw_stage_shift_V": raw_shift,
                "applied_stage_shift_V": shift,
                "applied": applied,
                "reason": "repeated_bias_conditions" if applied else "insufficient_overlap",
            }
        )
        if not applied:
            continue
        mask = out["family"].eq(family) & out["source_file"].eq(source)
        sweep_col = "VBG_V" if family == "IDVBG" else "VTG_V"
        out.loc[mask, sweep_col] = out.loc[mask, sweep_col] - shift
        out.loc[mask, "iv_stage_shift_V"] = shift
        out.loc[mask, "iv_stage_alignment_applied"] = True
        out.loc[mask, "iv_stage_alignment_reason"] = "repeated_bias_conditions"

    decisions = pd.DataFrame(rows)
    report = {
        "mode": cfg.mode,
        "max_shift_V": cfg.max_shift_V,
        "min_matched_curves": cfg.min_matched_curves,
        "transfer_curves_analyzed": int(len(curves)),
        "matched_condition_rows": int(len(comparisons)),
        "applied_sources": int(decisions["applied"].sum()) if len(decisions) else 0,
        "identifiability_rule": (
            "A stage shift is applied only when the same family, VDS, held-gate "
            "bias and temperature are repeated in another source."
        ),
    }
    return out, decisions, report


def write_iv_alignment_report(
    out_dir: Path,
    decisions: pd.DataFrame,
    report: dict,
    prefix: str = "iv_stage_alignment",
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    decisions.to_csv(out_dir / f"{prefix}.csv", index=False, encoding="utf-8-sig")
    (out_dir / f"{prefix}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

