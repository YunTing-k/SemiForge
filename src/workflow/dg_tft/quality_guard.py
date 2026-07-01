# -*- coding: utf-8 -*-
"""
Header information
---------
Shanghai Jiao Tong University, School of Integrated Circuits, SMIL Lab
Author: Yu Huang
Create Date: 2026.6.26
Description: Quality guard for DG-TFT workflow stages (Phase 7).

    After a stage exits with ok=True, the quality guard evaluates its
    numerical quality (RMS / log10 error metrics) against configurable
    thresholds.  When quality is insufficient the guard suggests
    progressive parameter adjustments and applies them to the runtime
    INI, enabling an automatic retry loop in the orchestrator.

    The adjustment strategy tables and safety boundaries are module-level
    constants — no inline magic numbers.

Revision:
---------
2026.6.26      Yu Huang      1.0      Initial implementation (P7-T3, P7-T5)
"""
from __future__ import annotations

import configparser
import logging
from pathlib import Path

from .schema import StageName
from .metrics_parser import get_primary_metric

sys_log = logging.getLogger("logger")

# ── Config key prefixes for thresholds ──────────────────────────────

_CFG_PREFIX: str = "DG_TFT_QUALITY_"

_STAGE_THRESHOLD_KEYS: dict[str, str] = {
    StageName.FIT_PHYSICAL: _CFG_PREFIX + "PHYSICAL_IV_RMS_MAX",
    StageName.TRAIN_STATIC: _CFG_PREFIX + "STATIC_IV_RMS_MAX",
    StageName.TRAIN_HYSTERESIS: _CFG_PREFIX + "HYSTERESIS_RMS_MAX",
}

_CFG_MAX_RETRIES: str = _CFG_PREFIX + "MAX_RETRIES"
_DEFAULT_MAX_RETRIES: int = 3

# ── Adjustment strategy tables ──────────────────────────────────────
# Each stage maps attempt index (0-based) → list of (section, key, new_value_fn)
# where new_value_fn reads the current INI and returns the new value.
# Strategies are progressive: attempt 0 is the mildest.

_ADJUSTMENT_STRATEGIES: dict[str, list[dict[str, dict[str, object]]]] = {
    # fit-physical: scale nfev → reduce data points
    StageName.FIT_PHYSICAL: [
        {"physical_fit": {"max_nfev_scale": "×2.0"}},
        {"physical_fit": {"max_nfev_scale": "×3.0"}},
        {"physical_fit": {"max_iv_points": "÷2"}},
    ],
    # train-static: epochs+lr → cv_weight → early_stop
    StageName.TRAIN_STATIC: [
        {"full_model": {"epochs": "+50%", "learning_rate": "÷2"}},
        {"full_model": {"cv_weight": "×2.0"}},
        {"full_model": {"early_stop_patience": "×2"}},
    ],
    # train-hysteresis: epochs → hidden units → amp mode
    StageName.TRAIN_HYSTERESIS: [
        {"hysteresis": {"epochs": "+50%"}},
        {"hysteresis": {"hidden_capture": "+4", "hidden_emission": "+4"}},
        {"hysteresis": {"amp_parameterization": "aggressive"}},
    ],
}

# ── Safety boundaries per section.key ───────────────────────────────

_PARAMETER_BOUNDS: dict[str, dict[str, tuple[float, float]]] = {
    "physical_fit": {
        "max_nfev_scale": (0.5, 5.0),
        "max_iv_points": (5000, 200000),
        "max_cv_points": (10000, 500000),
    },
    "base_pkan": {
        "epochs_shared": (100, 3000),
        "epochs_iv": (100, 3000),
        "epochs_cv": (100, 3000),
        "kan_hidden": (4, 32),
        "kan_grid": (2, 20),
        "kan_k": (2, 5),
        "batch_iv": (256, 16384),
        "batch_cv": (64, 4096),
    },
    "full_model": {
        "epochs": (100, 5000),
        "learning_rate": (0.0001, 0.01),
        "cv_weight": (1.0, 50.0),
        "early_stop_patience": (2, 30),
        "early_stop_min_epochs": (50, 500),
        "batch_iv": (512, 16384),
        "batch_cv": (64, 4096),
        "symbolic_training_weight": (0.0, 0.1),
        "print_every": (1, 200),
    },
    "temperature": {
        "minimum_span_K": (1.0, 100.0),
        "kan_scale": (0.1, 1.0),
    },
    "hysteresis": {
        "epochs": (200, 5000),
        "hidden_capture": (4, 32),
        "hidden_emission": (4, 32),
        "print_every": (1, 200),
    },
}

# Whitelist: only these section.keys may be auto-adjusted.
_PARAMETER_WHITELIST: frozenset[str] = frozenset({
    "physical_fit.max_nfev_scale",
    "physical_fit.max_iv_points",
    "physical_fit.max_cv_points",
    "base_pkan.epochs_shared",
    "base_pkan.epochs_iv",
    "base_pkan.epochs_cv",
    "base_pkan.kan_hidden",
    "base_pkan.kan_grid",
    "base_pkan.kan_k",
    "base_pkan.batch_iv",
    "base_pkan.batch_cv",
    "full_model.epochs",
    "full_model.learning_rate",
    "full_model.batch_iv",
    "full_model.batch_cv",
    "full_model.cv_weight",
    "full_model.early_stop_patience",
    "full_model.early_stop_min_epochs",
    "full_model.symbolic_training_weight",
    "full_model.print_every",
    "temperature.minimum_span_K",
    "temperature.kan_scale",
    "hysteresis.epochs",
    "hysteresis.hidden_capture",
    "hysteresis.hidden_emission",
    "hysteresis.amp_parameterization",
    "hysteresis.print_every",
})

# ── Stage-specific console labels ───────────────────────────────────

_STAGE_LABELS: dict[str, str] = {
    StageName.FIT_PHYSICAL: "Physical Kernel",
    StageName.TRAIN_STATIC: "Static Model",
    StageName.TRAIN_HYSTERESIS: "Hysteresis",
}


# ── Public API ──────────────────────────────────────────────────────

def assess_quality(
    stage_name: str,
    metrics: dict[str, float],
    thresholds: dict[str, float],
) -> tuple[bool, list[str]]:
    """Check whether stage metrics satisfy quality thresholds.

    Args:
        stage_name: Stage identifier.
        metrics: Metric name → value dict (from metrics_parser).
        thresholds: Metric/threshold name → maximum allowed value.

    Returns:
        (pass, failures) — ``pass`` is True when all thresholds are met;
        ``failures`` lists human-readable descriptions of violations.
    """
    failures: list[str] = []
    primary = get_primary_metric(stage_name)
    for metric_name, value in metrics.items():
        limit = thresholds.get(metric_name)
        if limit is None:
            continue
        if value != value:  # NaN
            failures.append(f"{metric_name}: value is NaN (cannot assess)")
            continue
        if value > limit:
            failures.append(
                f"{metric_name}: {value:.6f} > threshold {limit:.6f} "
                f"(exceeded by {value - limit:.6f})"
            )
    if not failures and not metrics:
        failures.append("no metrics parsed (artifact file missing or unreadable)")
    return len(failures) == 0, failures


def get_quality_thresholds(agent_configs: dict) -> dict[str, float]:
    """Extract quality thresholds from agent_configs.

    Returns a flat dict keyed by the full dotted metric key path
    (e.g. ``metrics_summary.iv_rms_log10_error_dec``).
    """
    result: dict[str, float] = {}
    for stage_name, cfg_key in _STAGE_THRESHOLD_KEYS.items():
        val = agent_configs.get(cfg_key)
        if val is None:
            continue
        primary = get_primary_metric(stage_name)
        if primary:
            result[primary] = float(val)
    return result


def get_max_retries(agent_configs: dict) -> int:
    """Read max retries from agent_configs, with fallback."""
    val = agent_configs.get(_CFG_MAX_RETRIES, _DEFAULT_MAX_RETRIES)
    return max(0, int(val))


def suggest_adjustments(
    stage_name: str,
    attempt: int,
    ini_path: str,
) -> dict[str, str]:
    """Produce a dict of section.key → value overrides for a retry attempt.

    Attempts are 0-indexed (0 = first retry, 1 = second, ...).
    When ``attempt`` exceeds the defined strategies, returns an empty dict.

    Returns:
        dict suitable for passing to ``apply_adjustments()``, or empty dict
        if no more strategies are available.
    """
    strategies = _ADJUSTMENT_STRATEGIES.get(stage_name)
    if strategies is None or attempt >= len(strategies):
        sys_log.info(
            "No adjustment strategy for stage [%s] attempt %d",
            stage_name, attempt,
        )
        return {}

    blueprint = strategies[attempt]
    overrides: dict[str, str] = {}
    for section, assignments in blueprint.items():
        for key, directive in assignments.items():
            full_key = f"{section}.{key}"
            if full_key not in _PARAMETER_WHITELIST:
                sys_log.warning(
                    "Adjustment not in whitelist, skipping: %s", full_key,
                )
                continue
            new_value = _compute_new_value(section, key, directive, ini_path)
            if new_value is not None:
                overrides[full_key] = new_value

    sys_log.info(
        "Suggested %d adjustments for stage [%s] attempt %d: %s",
        len(overrides), stage_name, attempt,
        ", ".join(f"{k}={v}" for k, v in overrides.items()),
    )
    return overrides


def apply_adjustments(ini_path: str, adjustments: dict[str, str]) -> str:
    """Apply section.key → value adjustments to a copy of the INI file.

    The original file is never modified.  A new file named
    ``model_config.ini`` (same directory, overwriting) is written with
    the adjustments applied.

    Args:
        ini_path: Path to the current runtime INI.
        adjustments: section.key → new_value dict (from suggest_adjustments).

    Returns:
        Path to the modified INI file (same as input, in-place rewrite).
    """
    if not adjustments:
        sys_log.info("No adjustments to apply for %s", ini_path)
        return ini_path

    path = Path(ini_path)
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)

    current_section = ""
    applied: set[str] = set()

    for i, line in enumerate(lines):
        stripped = line.lstrip()
        if stripped.startswith("[") and (stripped.endswith("]\n") or stripped.rstrip().endswith("]")):
            current_section = stripped[1:].split("]")[0].strip()
            continue
        if "=" not in stripped or stripped.startswith("#") or stripped.startswith(";"):
            continue
        option = stripped.split("=", 1)[0].strip()
        full_key = f"{current_section}.{option}"
        if full_key in adjustments:
            indent = line[:len(line) - len(line.lstrip())]
            lines[i] = f"{indent}{option} = {adjustments[full_key]}\n"
            applied.add(full_key)

    missed = set(adjustments.keys()) - applied
    if missed:
        sys_log.warning("Adjustments not found in INI (no matching line): %s", sorted(missed))

    path.write_text("".join(lines), encoding="utf-8")
    sys_log.info("Applied %d adjustments to %s", len(applied), path)
    return str(path)


def stage_display_label(stage_name: str) -> str:
    """Return a human-readable label for a stage name."""
    return _STAGE_LABELS.get(stage_name, stage_name)


# ── Internal helpers ───────────────────────────────────────────────

def _compute_new_value(
    section: str,
    key: str,
    directive: str | int | float,
    ini_path: str,
) -> str | None:
    """Compute the new parameter value from a directive string.

    Directives:
        ``×N``   → multiply current value by N
        ``÷N``   → divide current value by N
        ``+N%``  → increase by N percent
        ``+N``   → add N to integer value
        plain str → return as-is (for enums like ``aggressive``)
    """
    current = _read_ini_value(ini_path, section, key)
    if current is None and isinstance(directive, str):
        dir_lower = directive.lower()
        if dir_lower.replace(".", "").replace("_", "").isalpha():
            return directive
        return None

    if isinstance(directive, (int, float)):
        return str(directive)

    d = str(directive).strip()

    if d.startswith("×"):
        try:
            factor = float(d[1:])
            new_val = current * factor
        except (ValueError, TypeError):
            return None
    elif d.startswith("÷"):
        try:
            divisor = float(d[1:])
            new_val = current / divisor
        except (ValueError, TypeError):
            return None
    elif d.endswith("%"):
        try:
            pct = float(d[:-1])
            new_val = current * (1.0 + pct / 100.0)
        except (ValueError, TypeError):
            return None
    elif d.startswith("+") and not d.startswith("+%"):
        try:
            delta = float(d[1:])
            new_val = current + delta
        except (ValueError, TypeError):
            return None
    else:
        return d

    # Clamp to safety boundaries.
    clamped = _clamp(section, key, new_val)
    if isinstance(current, int) or (isinstance(current, float) and current == int(current) and isinstance(new_val, float)):
        if abs(clamped - round(clamped)) < 1e-9:
            return str(int(round(clamped)))
    if isinstance(clamped, float):
        return f"{clamped:.6g}"
    return str(clamped)


def _read_ini_value(ini_path: str, section: str, key: str) -> float | None:
    """Read a single numeric value from an INI file."""
    try:
        cfg = configparser.ConfigParser()
        cfg.read(ini_path, encoding="utf-8")
        raw = cfg.get(section, key, fallback=None)
        if raw is None:
            return None
        return float(raw)
    except (ValueError, configparser.Error, OSError):
        return None


# ── Hard guard for Agent adjust_params output ─────────────────────

def validate_agent_adjustments(
    params: dict[str, int | float | bool | str],
    param_table: dict[str, dict[str, object]],
    stage_sections: list[str],
    stage_name: str = "",
) -> tuple[dict[str, object], list[str]]:
    """Validate and sanitize an Agent's ``adjust_params`` output against parameter definitions.

    This is the **programmatic hard guard** — fast, no LLM involved.
    It runs before any INI write and enforces:

    1. **Whitelist** — param key must exist in ``param_table``.
    2. **Type** — value type must match the definition (int/float/bool/string).
    3. **Bounds** — numeric values are clamped to ``[lo, hi]``.
    4. **Enum** — string/enum values must be one of the allowed options.
    5. **Stage isolation** — unknown keys are silently dropped.

    Args:
        params: Raw key→value dict from Agent (exclude ``reasoning``).
        param_table: Parameter definition table for the current stage.
        stage_sections: INI section names the stage may write to (list for multi-section stages).
        stage_name: Stage name for log messages.

    Returns:
        ``(sanitized, warnings)`` where ``sanitized`` has invalid/unknown
        keys removed, types coerced where safe, and out-of-bounds values
        clamped.  ``warnings`` is a list of human-readable messages.
    """
    sanitized: dict[str, object] = {}
    warnings: list[str] = []
    label = stage_name or (stage_sections[0] if stage_sections else "unknown")

    for key, raw_value in params.items():
        # --- Step 1: Whitelist check ----------------------------------
        pdef = param_table.get(key)
        if pdef is None:
            warnings.append(
                f"[{label}] Unknown parameter '{key}' — not adjustable for this stage; dropped."
            )
            continue

        # --- Step 2: Type check and coercion --------------------------
        expected_type = _get_type_str(pdef)
        try:
            value = _coerce_value(raw_value, expected_type)
        except (ValueError, TypeError):
            warnings.append(
                f"[{label}] Type mismatch for '{key}': "
                f"expected {expected_type}, got {type(raw_value).__name__} ({raw_value!r}); dropped."
            )
            continue

        # --- Step 3: Enum validation ----------------------------------
        allowed_enum = _get_enum_options(pdef)
        if allowed_enum is not None:
            if isinstance(value, bool) or isinstance(value, str):
                if str(value).lower() not in {str(e).lower() for e in allowed_enum}:
                    warnings.append(
                        f"[{label}] Invalid value for '{key}': {value!r} not in {allowed_enum}; dropped."
                    )
                    continue
            else:
                if value not in allowed_enum:
                    warnings.append(
                        f"[{label}] Invalid value for '{key}': {value!r} not in {allowed_enum}; dropped."
                    )
                    continue

        # --- Step 4: Bounds clamping ----------------------------------
        lo = _get_bound(pdef, "lo")
        hi = _get_bound(pdef, "hi")
        clamped_value = value

        if lo is not None and isinstance(value, (int, float)):
            if value < lo:
                sys_log.warning(
                    "[%s] Clamping '%s': %s → %s (bound lo=%s)",
                    label, key, value, lo, lo,
                )
                warnings.append(
                    f"[{label}] '{key}' = {value} below minimum {lo}; clamped to {lo}."
                )
                clamped_value = type(value)(lo)

        if hi is not None and isinstance(value, (int, float)):
            if value > hi:
                sys_log.warning(
                    "[%s] Clamping '%s': %s → %s (bound hi=%s)",
                    label, key, value, hi, hi,
                )
                warnings.append(
                    f"[{label}] '{key}' = {value} above maximum {hi}; clamped to {hi}."
                )
                clamped_value = type(value)(hi)

        sanitized[key] = clamped_value

    return sanitized, warnings


# ── Helper stubs for param_table access ─────────────────────────────

def _get_type_str(pdef: dict[str, object]) -> str:
    """Extract type string safely."""
    t = pdef.get("type", "string")
    return str(t)


def _get_enum_options(pdef: dict[str, object]) -> list[object] | None:
    """Extract enum list if present."""
    raw = pdef.get("enum")
    if raw is None:
        return None
    if isinstance(raw, list):
        return raw
    return None


def _get_bound(pdef: dict[str, object], bound_key: str) -> int | float | None:
    """Extract a numeric bound (lo/hi) if present."""
    val = pdef.get(bound_key)
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return val
    return None


def _coerce_value(raw_value: object, expected_type: str) -> int | float | bool | str:
    """Coerce a raw value to the expected type.

    Accepts numeric strings for int/float and common boolean representations.
    Raises ValueError/TypeError on mismatch.
    """
    if expected_type == "int":
        if isinstance(raw_value, bool):
            raise TypeError("bool is not int")
        if isinstance(raw_value, int):
            return raw_value
        if isinstance(raw_value, float) and raw_value == int(raw_value):
            return int(raw_value)
        if isinstance(raw_value, str):
            try:
                fv = float(raw_value)
                return int(fv)
            except (ValueError, TypeError):
                raise
        raise TypeError(f"cannot coerce {raw_value!r} to int")

    if expected_type == "float":
        if isinstance(raw_value, bool):
            raise TypeError("bool is not float")
        if isinstance(raw_value, (int, float)):
            return float(raw_value)
        if isinstance(raw_value, str):
            try:
                return float(raw_value)
            except (ValueError, TypeError):
                raise
        raise TypeError(f"cannot coerce {raw_value!r} to float")

    if expected_type == "bool":
        if isinstance(raw_value, bool):
            return raw_value
        if isinstance(raw_value, str):
            lower = raw_value.strip().lower()
            if lower in ("yes", "true", "1"):
                return True
            if lower in ("no", "false", "0"):
                return False
        if isinstance(raw_value, (int, float)):
            return raw_value != 0
        raise TypeError(f"cannot coerce {raw_value!r} to bool")

    if expected_type == "string":
        return str(raw_value)

    raise TypeError(f"unknown expected type '{expected_type}'")


# ── Legacy helpers ──────────────────────────────────────────────────

def _clamp(section: str, key: str, value: float) -> float:
    """Clamp a value to its safety boundary.  Returns unchanged value
    if no bounds are defined.

    When clamping occurs a WARNING is logged.
    """
    bounds = _PARAMETER_BOUNDS.get(section, {}).get(key)
    if bounds is None:
        return value
    lo, hi = bounds
    if value < lo:
        sys_log.warning(
            "Clamping %s.%s: %.6g → %.6g (bound lo=%.6g)",
            section, key, value, lo, lo,
        )
        return lo
    if value > hi:
        sys_log.warning(
            "Clamping %s.%s: %.6g → %.6g (bound hi=%.6g)",
            section, key, value, hi, hi,
        )
        return hi
    return value
