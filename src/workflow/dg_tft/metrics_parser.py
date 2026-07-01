# -*- coding: utf-8 -*-
"""
Header information
---------
Shanghai Jiao Tong University, School of Integrated Circuits, SMIL Lab
Author: Yu Huang
Create Date: 2026.6.26
Description: Metrics parser for DG-TFT workflow stages.

    Reads quality metrics from stage artifact JSON files (not stdout parsing).
    Each extractor returns a dict of {metric_name: value} or empty dict on failure.

    Metrics are used by quality_guard.py for threshold assessment (Phase 7).

Revision:
---------
2026.6.26      Yu Huang      1.0      Initial implementation (P7-T2)
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from .schema import StageName

sys_log = logging.getLogger("logger")

# ── JSON file paths relative to output_dir ──────────────────────────

_METRIC_SOURCES: dict[str, dict[str, str]] = {
    StageName.FIT_PHYSICAL: {
        "file": "01_physical_kernel/fitted_params.json",
        "keys": "metrics_summary.iv_rms_log10_error_dec",
    },
    StageName.TRAIN_STATIC: {
        "file": "03_full_model/v93_safe_residual_metrics.json",
        "file_fallback": "03_full_model/v92_safe_residual_metrics.json",
        "keys": "residual_iv_rms,residual_cv_rms",
    },
    StageName.TRAIN_HYSTERESIS: {
        "file": "04_hysteresis/it_hysteresis_analysis_summary.json",
        "keys": "step_log_rmse_median",
    },
}

# Default value when a metric key is missing from JSON.
_METRIC_MISSING_SENTINEL: float = float("nan")


def parse_metrics(output_dir: str, stage_name: str) -> dict[str, float]:
    """Extract quality metrics for a stage from its artifact JSON.

    Args:
        output_dir: Root output directory for the workflow run.
        stage_name: Stage identifier (e.g. ``fit-physical``).

    Returns:
        dict of {metric_name: value}.  Missing / unreadable files yield an
        empty dict.
    """
    source = _METRIC_SOURCES.get(stage_name)
    if source is None:
        sys_log.debug("No metric source defined for stage [%s]", stage_name)
        return {}

    json_path = Path(output_dir) / source["file"]
    if not json_path.exists():
        fallback = source.get("file_fallback")
        if fallback:
            json_path = Path(output_dir) / fallback
    if not json_path.exists():
        sys_log.warning("Metric source not found for stage [%s]: %s", stage_name, json_path)
        return {}

    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        sys_log.warning("Failed to read metric source %s: %s", json_path, exc)
        return {}

    spec = source["keys"]
    result: dict[str, float] = {}
    for key_path in _parse_key_list(spec):
        value = _deep_get(data, key_path)
        if value is not None:
            result[key_path] = float(value)
        else:
            result[key_path] = _METRIC_MISSING_SENTINEL

    sys_log.info("Parsed %d metrics from %s for stage [%s]", len(result), source["file"], stage_name)
    return result


def get_primary_metric(stage_name: str) -> str:
    """Return the conventional primary metric key for a stage.

    This is the first key in the ``keys`` spec and is used as the
    single-value gauge for quality assessment.
    """
    source = _METRIC_SOURCES.get(stage_name)
    if source is None:
        return ""
    return _parse_key_list(source["keys"])[0]


# ── Internal helpers ───────────────────────────────────────────────


def _parse_key_list(spec: str) -> list[str]:
    """Split a comma-separated key-path spec into individual paths."""
    return [k.strip() for k in spec.split(",") if k.strip()]


def _deep_get(data: dict, dotted: str) -> float | None:
    """Traverse nested dicts with dot-separated keys.

    Example: ``_deep_get(d, "a.b.c")`` → ``d["a"]["b"]["c"]``
    """
    parts = dotted.split(".")
    cursor = data
    for p in parts:
        if not isinstance(cursor, dict) or p not in cursor:
            return None
        cursor = cursor[p]
    if isinstance(cursor, (int, float)):
        return float(cursor)
    return None
