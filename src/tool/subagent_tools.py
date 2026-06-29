# -*- coding: utf-8 -*-
"""
Header information
---------
Shanghai Jiao Tong University, School of Integrated Circuits, SMIL Lab
Author: Yu Huang
Create Date: 2026.6.29
Description: Subagent tool schemas and local handlers for DG-TFT modeling agents

Revision:
---------
2026.6.29      Yu Huang      1.0      Initial structure for zero-change-2 subagent enhancement

Details:
---------
This module is the single source of truth for subagent-specific tools. Every tool has two parts:
  - schema: OpenAI JSON function definition (sent to LLM)
  - handler: optional callable(arguments, ctx) -> dict (executed locally, result returned to LLM)

Tools fall into two categories:
  - Investigation tools: have a handler, executed locally, result returned to LLM
    (e.g. SUBAGENT_TOOL_INSPECT_XLSX)
  - Output / stop tools: no handler, arguments captured by SubAgent as final result
    when stop_mode="schema" and the tool name is in stop_schemas
    (e.g. SUBAGENT_TOOL_ADJUST_PARAMS, SUBAGENT_TOOL_FALLBACK_TO_PREVIOUS)

Callers obtain tools via:

    from src.tool.subagent_tools import get_tools, SUBAGENT_TOOL_ADJUST_PARAMS, SUBAGENT_TOOL_FALLBACK_TO_PREVIOUS
    schemas, handlers = get_tools("fit-physical")
    agent = SubAgent(
        ...,
        subagent_tools=schemas,
        tool_handlers=handlers,
        stop_schemas=[SUBAGENT_TOOL_ADJUST_PARAMS, SUBAGENT_TOOL_FALLBACK_TO_PREVIOUS],
    )

Definitions for each modeling stage are registered in _STAGE_TOOLS in Phase 2/3
as defined in doc/TODO.md sections 3.2–3.4.
"""

from typing import Any, Callable

from src.constants import SUCCESS_LABEL

# ---------------------------------------------------------------------------
# Tool name constants — use these everywhere instead of bare strings
# ---------------------------------------------------------------------------

SUBAGENT_TOOL_INSPECT_XLSX: str = "inspect_xlsx"
SUBAGENT_TOOL_ADJUST_PARAMS: str = "adjust_params"
SUBAGENT_TOOL_FALLBACK_TO_PREVIOUS: str = "fallback_to_previous"
SUBAGENT_TOOL_FALLBACK_TO_VERIFIER: str = "fallback_to_verifier"
SUBAGENT_TOOL_SWITCH_KERNEL: str = "switch_kernel"
SUBAGENT_TOOL_REPORT_VERIFICATION: str = "report_verification"
SUBAGENT_TOOL_CONTRIBUTE_KNOWLEDGE: str = "contribute_knowledge"

__all__ = [
    "get_tools",
    "SUBAGENT_TOOL_INSPECT_XLSX",
    "SUBAGENT_TOOL_ADJUST_PARAMS",
    "SUBAGENT_TOOL_FALLBACK_TO_PREVIOUS",
    "SUBAGENT_TOOL_FALLBACK_TO_VERIFIER",
    "SUBAGENT_TOOL_SWITCH_KERNEL",
    "SUBAGENT_TOOL_REPORT_VERIFICATION",
    "SUBAGENT_TOOL_CONTRIBUTE_KNOWLEDGE",
]

# ---------------------------------------------------------------------------
# Handlers (investigation tools)
# ---------------------------------------------------------------------------


def _inspect_xlsx_handler(arguments: dict[str, Any], ctx) -> dict[str, Any]:
    """Placeholder handler for inspect_xlsx — real implementation in Phase 2."""
    return {"status": SUCCESS_LABEL, "path": arguments.get("path", "")}


# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

_INSPECT_XLSX_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": SUBAGENT_TOOL_INSPECT_XLSX,
        "description": "Inspect an XLSX file structure: returns sheet names and columns.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the .xlsx file"},
            },
            "required": ["path"],
        },
    },
}

_ADJUST_PARAMS_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": SUBAGENT_TOOL_ADJUST_PARAMS,
        "description": "Submit parameter adjustments for the current stage.",
        "parameters": {
            "type": "object",
            "properties": {
                "section": {"type": "string", "description": "INI section to modify"},
                "adjustments": {"type": "object", "description": "Key-value adjustments"},
                "reasoning": {"type": "string", "description": "Why this adjustment"},
            },
            "required": ["section", "adjustments", "reasoning"],
        },
    },
}

_FALLBACK_TO_PREVIOUS_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": SUBAGENT_TOOL_FALLBACK_TO_PREVIOUS,
        "description": "Request fallback to the upstream stage.",
        "parameters": {
            "type": "object",
            "properties": {
                "target": {"type": "string", "description": "Stage to fall back to"},
                "reason": {"type": "string", "description": "Why fallback is needed"},
                "evidence": {"type": "string", "description": "Supporting evidence"},
                "suggestion": {"type": "string", "description": "Suggested fix"},
            },
            "required": ["target", "reason"],
        },
    },
}


# ---------------------------------------------------------------------------
# Stage tool registry — populated in Phase 2/3
# ---------------------------------------------------------------------------

_STAGE_TOOLS: dict[str, tuple[list[dict[str, Any]], dict[str, Callable[..., Any]]]] = {
    "fit-physical": (
        [_INSPECT_XLSX_SCHEMA, _ADJUST_PARAMS_SCHEMA, _FALLBACK_TO_PREVIOUS_SCHEMA],
        {SUBAGENT_TOOL_INSPECT_XLSX: _inspect_xlsx_handler},
    ),
    "check-data": (
        [_FALLBACK_TO_PREVIOUS_SCHEMA],
        {},
    ),
}


def get_tools(stage: str) -> tuple[list[dict[str, Any]], dict[str, Callable[..., Any]]]:
    """Return (tool_schemas, tool_handlers) for the given modeling stage.

    stage: one of 'data-prep', 'check-data', 'fit-physical', 'train-static',
           'train-hysteresis', 'final-interpret', or '' for no tools.

    Returns empty lists/dicts for unknown stages (graceful fallback).
    """
    schemas, handlers = _STAGE_TOOLS.get(stage, ([], {}))
    return list(schemas), dict(handlers)
