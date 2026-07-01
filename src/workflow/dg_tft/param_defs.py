# -*- coding: utf-8 -*-
"""
Header information
---------
Shanghai Jiao Tong University, School of Integrated Circuits, SMIL Lab
Author: Yu Huang
Create Date: 2026.7.1
Description: Parameter definition tables for DG-TFT workflow stages and
    auto-generation of adjust_params tool JSON schemas.

    Each stage (fit-physical / train-static / train-hysteresis) has a
    canonical parameter table that is the single source of truth for:
      - parameter name, type, bounds, default value, description
      - generating the OpenAI-compatible tool schema
      - driving the hard-guard validation pipe

    When the engine adds a new tunable parameter, only the table below
    needs updating — all downstream consumers (tool schema, validation,
    prompt injection) pick it up automatically.

Revision:
---------
2026.7.1        Yu Huang      1.0      Initial implementation (Task 3.2)
"""
from __future__ import annotations

from typing import Any


# ── Type aliases ─────────────────────────────────────────────────────

ParamDef = dict[str, Any]
"""A single parameter definition::

    {
        "type": "int" | "float" | "bool" | "string",
        "lo": lower_bound  (None for bool / string),
        "hi": upper_bound  (None for bool / string),
        "default": default_value,
        "desc": "Human-readable description for Agent context",
    }
"""


# ═══════════════════════════════════════════════════════════════════════
#  Parameter tables — one per adjustable INI section
# ═══════════════════════════════════════════════════════════════════════

PHYSICAL_FIT_PARAMS: dict[str, ParamDef] = {
    "max_iv_points": {
        "type": "int",
        "lo": 5000,
        "hi": 200000,
        "default": 50000,
        "desc": (
            "Maximum number of IV data points to subsample for physical fitting. "
            "Lower values reduce runtime but may miss device characteristics. "
            "Higher values improve accuracy at the cost of longer fitting time."
        ),
    },
    "max_cv_points": {
        "type": "int",
        "lo": 10000,
        "hi": 500000,
        "default": 100000,
        "desc": (
            "Maximum number of CV data points to subsample for physical fitting. "
            "Lower values reduce runtime; higher values capture more capacitance detail."
        ),
    },
    "max_nfev_scale": {
        "type": "float",
        "lo": 0.5,
        "hi": 5.0,
        "default": 1.0,
        "desc": (
            "Scale factor for maximum number of function evaluations (NFEV) "
            "during physical kernel parameter extraction. "
            "Values > 1.0 allow more optimization iterations per sub-problem. "
            "Increase if fitting returns poor IV curve match; decrease to speed up."
        ),
    },
}


FULL_MODEL_PARAMS: dict[str, ParamDef] = {
    "epochs": {
        "type": "int",
        "lo": 100,
        "hi": 5000,
        "default": 700,
        "desc": (
            "Total training epochs for the full static model (IV + CV). "
            "More epochs improve fit but increase training cost. "
            "Combine with early_stop_patience so training stops when improvement plateaus."
        ),
    },
    "learning_rate": {
        "type": "float",
        "lo": 0.0001,
        "hi": 0.01,
        "default": 0.002,
        "desc": (
            "Learning rate for the Adam optimizer during full model training. "
            "Lower values give more stable convergence; higher values may overshoot. "
            "Typical range: 0.0005–0.005."
        ),
    },
    "batch_iv": {
        "type": "int",
        "lo": 512,
        "hi": 16384,
        "default": 4096,
        "desc": (
            "Batch size for IV training samples. "
            "Larger batches provide more stable gradient estimates but require more GPU memory. "
            "Reduce if OOM; increase if gradient noise is excessive."
        ),
    },
    "batch_cv": {
        "type": "int",
        "lo": 64,
        "hi": 4096,
        "default": 512,
        "desc": (
            "Batch size for CV training samples. "
            "Typically smaller than batch_iv due to higher per-sample memory cost. "
            "Reduce if OOM; increase if gradient noise is excessive."
        ),
    },
    "cv_weight": {
        "type": "float",
        "lo": 1.0,
        "hi": 50.0,
        "default": 8.0,
        "desc": (
            "Weight multiplier for CV loss term relative to IV loss in the combined loss. "
            "Higher values force the model to fit CV curves more closely. "
            "Increase if CV RMS is too high; decrease if IV RMS degrades."
        ),
    },
    "cv_residual_level": {
        "type": "int",
        "lo": 0,
        "hi": 3,
        "default": 1,
        "desc": (
            "Complexity level of the CV residual correction sub-network. "
            "0 = disabled.  Higher levels add more learnable parameters. "
            "Increase if CV fitting needs more flexibility."
        ),
    },
    "enable_cv_residual": {
        "type": "bool",
        "lo": None,
        "hi": None,
        "default": True,
        "desc": (
            "Whether to enable the CV residual learning branch. "
            "When True, a small network learns the CV IV→CV residual correction. "
            "Usually should remain True."
        ),
    },
    "finetune_base_pkan": {
        "type": "bool",
        "lo": None,
        "hi": None,
        "default": True,
        "desc": (
            "Whether to fine-tune the base PKAN weights during full model training. "
            "When True, physical-kernel-fitted PKAN layers continue learning. "
            "Set False to freeze the physical kernel and only train the residual path."
        ),
    },
    "early_stop_patience": {
        "type": "int",
        "lo": 2,
        "hi": 30,
        "default": 5,
        "desc": (
            "Number of consecutive epochs with no validation improvement before early stopping. "
            "Smaller values abort training sooner; larger values allow more exploration. "
            "Increase if training is stopped prematurely."
        ),
    },
    "early_stop_min_epochs": {
        "type": "int",
        "lo": 50,
        "hi": 500,
        "default": 220,
        "desc": (
            "Minimum number of epochs before early stopping is allowed to activate. "
            "Ensures the model has enough training before early-stop logic kicks in. "
            "Increase if model is under-trained; decrease to allow earlier stopping."
        ),
    },
    "symbolic_training_weight": {
        "type": "float",
        "lo": 0.0,
        "hi": 0.1,
        "default": 0.015,
        "desc": (
            "Weight for the symbolic training loss term. "
            "Controls how strongly the model is regularized towards symbolic interpretability. "
            "Higher values produce more interpretable equations at the cost of fitting accuracy."
        ),
    },
}


HYSTERESIS_PARAMS: dict[str, ParamDef] = {
    "enabled": {
        "type": "bool",
        "lo": None,
        "hi": None,
        "default": True,
        "desc": (
            "Whether to run the hysteresis training stage at all. "
            "Set False to skip hysteresis (e.g., no I-t data available)."
        ),
    },
    "regime": {
        "type": "string",
        "lo": None,
        "hi": None,
        "default": "all",
        "enum": ["all", "linear", "sat"],
        "desc": (
            "Target regime for hysteresis training. "
            "'all' — full range; 'linear' — linear region only; 'sat' — saturation region only. "
            "Use 'all' unless specific regime data is problematic."
        ),
    },
    "epochs": {
        "type": "int",
        "lo": 200,
        "hi": 5000,
        "default": 1800,
        "desc": (
            "Training epochs for the hysteresis capture/emission network. "
            "Hysteresis training typically requires more epochs than static fitting. "
            "Increase if dynamic behavior is not well captured."
        ),
    },
    "hidden_capture": {
        "type": "int",
        "lo": 4,
        "hi": 32,
        "default": 16,
        "desc": (
            "Number of hidden units in the capture network (non-linear transform). "
            "Larger networks can model more complex trap dynamics. "
            "Increase if hysteresis waveform shape is poorly reproduced."
        ),
    },
    "hidden_emission": {
        "type": "int",
        "lo": 4,
        "hi": 32,
        "default": 20,
        "desc": (
            "Number of hidden units in the emission network (linear correction). "
            "Larger networks improve time-constant resolution. "
            "Increase if transient response is inaccurate."
        ),
    },
    "amp_parameterization": {
        "type": "string",
        "lo": None,
        "hi": None,
        "default": "conservative",
        "enum": ["conservative", "aggressive"],
        "desc": (
            "Amplitude parameterization style for the hysteresis model. "
            "'conservative' — safer, less prone to oscillation. "
            "'aggressive' — tighter fit at the risk of overshoot artifacts. "
            "Switch to 'aggressive' only if conservative fitting yields poor amplitude match."
        ),
    },
}


# ── Master lookup: stage → (ini_section, param_table) ───────────────

from .schema import StageName

_STAGE_PARAM_MAP: dict[str, tuple[str, dict[str, ParamDef]]] = {
    StageName.FIT_PHYSICAL: ("physical_fit", PHYSICAL_FIT_PARAMS),
    StageName.TRAIN_STATIC: ("full_model", FULL_MODEL_PARAMS),
    StageName.TRAIN_HYSTERESIS: ("hysteresis", HYSTERESIS_PARAMS),
}


# ═══════════════════════════════════════════════════════════════════════
#  JSON Schema auto-generation → adjust_params tool def
# ═══════════════════════════════════════════════════════════════════════

def params_to_tool_schema(
    param_table: dict[str, ParamDef],
    tool_name: str = "adjust_params",
    stage_label: str = "",
) -> dict[str, Any]:
    """Auto-generate an OpenAI-compatible tool schema from a parameter table.

    The generated schema exposes each parameter as an optional property with
    its type, description, and numeric bounds (when applicable).

    Args:
        param_table: {param_name → ParamDef} mapping.
        tool_name: Tool name for the schema (default ``"adjust_params"``).
        stage_label: Human-readable stage name injected into the description.

    Returns:
        A dict matching the ``{"type": "function", "function": {...}}`` format.
    """
    label = stage_label or "current"
    properties: dict[str, dict[str, Any]] = {}
    for name, pdef in param_table.items():
        prop: dict[str, Any] = {}
        json_type = _param_type_to_json_type(pdef["type"])
        prop["type"] = json_type if not isinstance(json_type, list) else json_type[0]
        prop["description"] = pdef["desc"]
        # Numeric bounds
        if pdef.get("lo") is not None:
            prop["minimum"] = pdef["lo"]
        if pdef.get("hi") is not None:
            prop["maximum"] = pdef["hi"]
        # Enum constraint (string / bool)
        if "enum" in pdef:
            prop["enum"] = pdef["enum"]
        properties[name] = prop

    return {
        "type": "function",
        "function": {
            "name": tool_name,
            "description": (
                f"Adjust tunable hyper-parameters for the {label} stage. "
                "Only specify parameters you want to change — omit the rest. "
                "Each value must be within the documented bounds. "
                "Provide a brief reasoning summary."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    **properties,
                    "reasoning": {
                        "type": "string",
                        "description": (
                            "Brief explanation of why these adjustments are made, "
                            "referencing observed metrics or failure symptoms."
                        ),
                    },
                },
                "required": ["reasoning"],
                "additionalProperties": False,
            },
        },
    }


def get_stage_param_schema(stage_name: str) -> dict[str, Any] | None:
    """Get the auto-generated tool schema for a given stage.

    Returns None if the stage has no adjustable parameters (e.g. ``check-data``,
    ``final-report``).
    """
    entry = _STAGE_PARAM_MAP.get(stage_name)
    if entry is None:
        return None
    _ini_section, param_table = entry
    return params_to_tool_schema(
        param_table,
        tool_name="adjust_params",
        stage_label=stage_name,
    )


def get_stage_section(param_table: dict[str, ParamDef]) -> str | None:
    """Reverse-lookup: find the INI section name for a param table."""
    for _stage, (section, pt) in _STAGE_PARAM_MAP.items():
        if pt is param_table:
            return section
    return None


# ── Internal helpers ─────────────────────────────────────────────────

def _param_type_to_json_type(t: str) -> str | list[str]:
    """Map internal type label to JSON Schema type.

    ``bool`` → ``"string"`` with ``enum: ["yes", "no"]`` to match INI convention.
    """
    mapping: dict[str, str | list[str]] = {
        "int": "integer",
        "float": "number",
        "bool": "boolean",
        "string": "string",
    }
    return mapping.get(t, "string")


# ═══════════════════════════════════════════════════════════════════════
#  Soft guard — self-check prompt snippet for Agent system prompt
# ═══════════════════════════════════════════════════════════════════════

_SELF_CHECK_PROMPT: str = (
    "**Before calling `adjust_params`, self-check your proposed values:**\n"
    "1. Bounds — are all values within the documented `minimum`/`maximum`?\n"
    "2. Physical sense — would a domain expert consider these values reasonable "
    "(e.g., epochs=50000 is excessive; lr=1.0 will diverge)?\n"
    "3. Interaction — do any parameter pairs conflict "
    "(e.g., very high learning_rate with very small early_stop_patience may "
    "prevent convergence; very small batch_iv with very large hidden units "
    "may cause OOM)?\n"
    "4. Incrementality — prefer small, incremental changes over extreme jumps. "
    "Doubling epochs is drastic; +25% is usually safer.\n"
    "5. If uncertain, consult the expert knowledge documents for guidance "
    "on reasonable ranges for this device technology.\n"
)


def get_soft_guard_prompt() -> str:
    """Return a self-check prompt snippet for Agent system prompts.

    Inject this into the modeling Agent's system prompt so it performs
    a reasonableness self-audit before emitting ``adjust_params`` calls.
    """
    return _SELF_CHECK_PROMPT
