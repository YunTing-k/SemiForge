# -*- coding: utf-8 -*-
"""
Header information
---------
Shanghai Jiao Tong University, School of Integrated Circuits, SMIL Lab
Author: Yu Huang
Create Date: 2026.6.26
Description: Subagent hook definitions for DG-TFT workflow diagnostics.

    Maps error categories to subagent call points and provides
    parameterised prompt templates for each hook.  The orchestrator
    uses these hooks to invoke ``explorer`` subagents that interpret
    failure logs and suggest remediation to the user.

    Hook definitions and prompt templates are module-level constants —
    no inline hardcoding — so adding a new diagnostic hook only
    requires a single-table edit.

Revision:
---------
2026.6.26      Yu Huang      1.0      Initial implementation (P6-T2)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .schema import ErrorCategory
from src.constants import EXPLORER_AGENT_LABEL

if TYPE_CHECKING:
    pass

# ── Hook type labels ───────────────────────────────────────────────

_HOOK_DATA_ERROR = "on_data_error"
_HOOK_TRAINING_FAILURE = "on_training_failure"
_HOOK_UNKNOWN_ERROR = "on_unknown_error"
_HOOK_MISSING_ARTIFACT = "on_missing_artifact"
_HOOK_CONFIG_ERROR = "on_config_error"
_HOOK_HSPICE_ERROR = "on_hspice_error"


@dataclass
class HookDefinition:
    """Specification for a single diagnostic subagent call point."""

    hook_id: str
    trigger_category: ErrorCategory
    subagent_type: str
    prompt_template: str
    subject: str = ""
    description: str = ""


# ── Prompt templates ───────────────────────────────────────────────
# Each template receives the same set of format variables:
#   {stage_name}   — e.g. "train-static"
#   {category}     — ErrorCategory value string
#   {confidence}   — classification confidence (float)
#   {evidence}     — bullet list of matched keyword sigils
#   {error_message}  — StageResult.error or stderr
#   {log_snippet}  — first N lines of retained stdout/stderr

_PROMPT_DATA_ERROR = """\
You are a diagnostic specialist for DG-TFT semiconductor device modeling. A data validation stage has failed. Your task is to read the relevant output files and explain what is wrong in clear, actionable terms.

**Stage**: {stage_name}
**Error category**: {category} (confidence: {confidence})

**Evidence**:
{evidence}

**Error details**:
{error_message}

**Instructions**:
1. Read the dataset_validation.json file in the stage output directory (if it exists).
2. Identify which file(s) have problems — missing sheets, missing columns, invalid data.
3. Summarise the root cause: which file, which sheet, which column, and why it matters.
4. Suggest the most likely fix the user should apply to their raw data.

Do NOT suggest code changes to the pipeline. Focus on the input data only.
Keep your response under 300 words.
"""

_PROMPT_TRAINING_FAILURE = """\
You are a diagnostic specialist for DG-TFT semiconductor device modeling. A numerical training stage has failed. Your task is to read the relevant training logs and identify the failure mode.

**Stage**: {stage_name}
**Error category**: {category} (confidence: {confidence})

**Evidence**:
{evidence}

**Error details**:
{error_message}

**Instructions**:
1. Read stdout/stderr log files in the stage output directory.
2. Classify the failure mode: OOM (out of memory), NaN/inf propagation, gradient explosion, numerical divergence, or other.
3. Identify whether this is a hyperparameter issue (learning rate, batch size, model size) or a data issue (noisy measurements, scaling).
4. Suggest the most impactful parameter adjustment the user should try first.

Do NOT suggest code changes to the pipeline. Focus on runtime configuration adjustments.
Keep your response under 300 words.
"""

_PROMPT_UNKNOWN_ERROR = """\
You are a diagnostic specialist for DG-TFT semiconductor device modeling. A workflow stage has failed with no clear error signature. Your task is to read the available logs and provide your best interpretation.

**Stage**: {stage_name}
**Error category**: {category} (confidence: {confidence})

**Evidence**:
{evidence}

**Error details**:
{error_message}

**Instructions**:
1. Read any available log files in the stage output directory.
2. Look for Python tracebacks, warning messages, or unusual output patterns.
3. Give your best estimate of what went wrong and why.
4. If you cannot determine the cause, say so clearly and suggest the user check the full log manually.

Keep your response under 300 words.
"""

_PROMPT_MISSING_ARTIFACT = """\
You are a diagnostic specialist for DG-TFT semiconductor device modeling. A workflow stage completed without errors but expected output files are missing. Your task is to diagnose why.

**Stage**: {stage_name}
**Error category**: {category} (confidence: {confidence})

**Evidence**:
{evidence}

**Error details**:
{error_message}

**Instructions**:
1. Check the stage output directory for any generated files.
2. Try to determine why the expected artifacts were not produced — did the training script exit early without raising an error?
3. Suggest whether the stage should be re-run or if manual intervention is needed.

Keep your response under 300 words.
"""

_PROMPT_CONFIG_ERROR = """\
You are a diagnostic specialist for DG-TFT semiconductor device modeling. A configuration error has prevented a workflow stage from running. Your task is to identify the specific configuration problem.

**Stage**: {stage_name}
**Error category**: {category} (confidence: {confidence})

**Evidence**:
{evidence}

**Error details**:
{error_message}

**Instructions**:
1. Identify which configuration key or section is problematic.
2. Check whether the issue is a missing value, wrong type, or invalid path.
3. Suggest the specific fix the user should make to their model_config.ini.

Keep your response under 300 words.
"""

_PROMPT_HSPICE_ERROR = """\
You are a diagnostic specialist for DG-TFT semiconductor device modeling. An HSPICE-related failure has occurred. Your task is to identify whether this is an environment issue or a model issue.

**Stage**: {stage_name}
**Error category**: {category} (confidence: {confidence})

**Evidence**:
{evidence}

**Error details**:
{error_message}

**Instructions**:
1. Determine whether HSPICE is installed and accessible at the configured path.
2. If a simulation failed, check whether the netlist is valid or if the Verilog-A model has issues.
3. Suggest whether the user should fix the HSPICE path or skip HSPICE validation.

Keep your response under 300 words.
"""

# ── Hook registry ──────────────────────────────────────────────────

_HOOKS: list[HookDefinition] = [
    HookDefinition(
        hook_id=_HOOK_DATA_ERROR,
        trigger_category=ErrorCategory.DATA_ERROR,
        subagent_type=EXPLORER_AGENT_LABEL,
        prompt_template=_PROMPT_DATA_ERROR,
        subject="Diagnose data error",
        description="Inspect dataset_validation.json and explain which file/sheet/column has problems.",
    ),
    HookDefinition(
        hook_id=_HOOK_TRAINING_FAILURE,
        trigger_category=ErrorCategory.TRAINING_NUMERICAL_ERROR,
        subagent_type=EXPLORER_AGENT_LABEL,
        prompt_template=_PROMPT_TRAINING_FAILURE,
        subject="Diagnose training failure",
        description="Read training logs and identify divergence/OOM/NaN failure mode.",
    ),
    HookDefinition(
        hook_id=_HOOK_UNKNOWN_ERROR,
        trigger_category=ErrorCategory.UNKNOWN_ERROR,
        subagent_type=EXPLORER_AGENT_LABEL,
        prompt_template=_PROMPT_UNKNOWN_ERROR,
        subject="Diagnose unknown error",
        description="Read available logs and provide best-effort interpretation.",
    ),
    HookDefinition(
        hook_id=_HOOK_MISSING_ARTIFACT,
        trigger_category=ErrorCategory.MISSING_ARTIFACT_ERROR,
        subagent_type=EXPLORER_AGENT_LABEL,
        prompt_template=_PROMPT_MISSING_ARTIFACT,
        subject="Diagnose missing artifacts",
        description="Investigate why expected output files were not created.",
    ),
    HookDefinition(
        hook_id=_HOOK_CONFIG_ERROR,
        trigger_category=ErrorCategory.CONFIG_ERROR,
        subagent_type=EXPLORER_AGENT_LABEL,
        prompt_template=_PROMPT_CONFIG_ERROR,
        subject="Diagnose config error",
        description="Identify the specific configuration key or section that is invalid.",
    ),
    HookDefinition(
        hook_id=_HOOK_HSPICE_ERROR,
        trigger_category=ErrorCategory.HSPICE_ENV_ERROR,
        subagent_type=EXPLORER_AGENT_LABEL,
        prompt_template=_PROMPT_HSPICE_ERROR,
        subject="Diagnose HSPICE error",
        description="Check HSPICE availability and simulation validity.",
    ),
]

# Fast lookup: ErrorCategory → HookDefinition
_HOOK_BY_CATEGORY: dict[ErrorCategory, HookDefinition] = {
    h.trigger_category: h for h in _HOOKS
}


def get_hook(category: ErrorCategory) -> HookDefinition | None:
    """Return the HookDefinition for a given error category, or None."""
    return _HOOK_BY_CATEGORY.get(category)


def list_hooks() -> list[HookDefinition]:
    """Return all registered diagnostic hook definitions."""
    return list(_HOOKS)


# ── Prompt formatter ───────────────────────────────────────────────

# Maximum length for error_message and log_snippet inserted into prompts.
_PROMPT_ERROR_MAX_CHARS: int = 2000
_PROMPT_LOG_SNIPPET_LINES: int = 60


def build_diagnosis_prompt(
    hook: HookDefinition,
    stage_name: str,
    category: ErrorCategory,
    confidence: float,
    evidence: list[str],
    error_message: str = "",
    log_snippet: str = "",
) -> str:
    """Format a hook's prompt template with the current failure context.

    Args:
        hook:           The HookDefinition whose template to render.
        stage_name:     Stage that failed (e.g. ``train-static``).
        category:       Classified error category.
        confidence:     Classification confidence (0.0–1.0).
        evidence:       List of matched keyword sigils.
        error_message:  Human-readable error string.
        log_snippet:    Truncated stdout/stderr for context.

    Returns:
        Formatted prompt string ready for a subagent call.
    """
    evidence_text = _format_evidence(evidence)
    error_text = _truncate(error_message, _PROMPT_ERROR_MAX_CHARS)
    if log_snippet:
        error_text += f"\n\n--- Log excerpt (first {_PROMPT_LOG_SNIPPET_LINES} lines) ---\n"
        error_text += _truncate(log_snippet, _PROMPT_ERROR_MAX_CHARS)

    return hook.prompt_template.format(
        stage_name=stage_name,
        category=category.value,
        confidence=f"{confidence:.0%}",
        evidence=evidence_text,
        error_message=error_text,
    )


# ── Internal helpers ───────────────────────────────────────────────

def _format_evidence(items: list[str]) -> str:
    if not items:
        return "(none)"
    return "\n".join(f"- {item}" for item in items)


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n... [truncated, {len(text)} total chars]"
