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

from src.constants import SUCCESS_LABEL, FAIL_LABEL

# ---------------------------------------------------------------------------
# Tool name constants — use these everywhere instead of bare strings
# ---------------------------------------------------------------------------

SUBAGENT_TOOL_INSPECT_XLSX: str = "inspect_xlsx"
SUBAGENT_TOOL_ADJUST_PARAMS: str = "adjust_params"
SUBAGENT_TOOL_FALLBACK_TO_PREVIOUS: str = "fallback_to_previous"
SUBAGENT_TOOL_SWITCH_KERNEL: str = "switch_kernel"
SUBAGENT_TOOL_REPORT_VERIFICATION: str = "report_verification"
SUBAGENT_TOOL_CONTRIBUTE_KNOWLEDGE: str = "contribute_knowledge"
SUBAGENT_TOOL_ACCEPT_QUALITY: str = "accept_quality"

__all__ = [
    "get_tools",
    "SUBAGENT_TOOL_INSPECT_XLSX",
    "SUBAGENT_TOOL_ADJUST_PARAMS",
    "SUBAGENT_TOOL_FALLBACK_TO_PREVIOUS",
    "SUBAGENT_TOOL_SWITCH_KERNEL",
    "SUBAGENT_TOOL_REPORT_VERIFICATION",
    "SUBAGENT_TOOL_CONTRIBUTE_KNOWLEDGE",
    "SUBAGENT_TOOL_ACCEPT_QUALITY",
]

# ---------------------------------------------------------------------------
# Handlers (investigation tools)
# ---------------------------------------------------------------------------


def _inspect_xlsx_handler(arguments: dict[str, Any], ctx) -> dict[str, Any]:
    """Inspect an XLSX file: return sheet names and column headers."""
    try:
        import openpyxl
        file_path = arguments.get("path", "")
        wb = openpyxl.load_workbook(file_path, read_only=True, data_only=True)
        sheets = {}
        for name in wb.sheetnames:
            ws = wb[name]
            headers = [cell.value for cell in next(ws.iter_rows(min_row=1, max_row=1))]
            sheets[name] = {"columns": [str(h) if h is not None else "" for h in headers]}
        wb.close()
        return {"status": SUCCESS_LABEL, "sheets": sheets, "sheet_count": len(sheets)}
    except Exception as e:
        return {"status": FAIL_LABEL, "error": str(e)}


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

_REPORT_VERIFICATION_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": SUBAGENT_TOOL_REPORT_VERIFICATION,
        "description": (
            "Submit structured verification findings for a prepared DG-TFT dataset. "
            "Call this tool exactly once at the end — do not output free text."
        ),
        "parameters": {
            "type": "object",
            "required": ["overall_ok"],
            "additionalProperties": False,
            "properties": {
                "overall_ok": {
                    "type": "boolean",
                    "description": "True if all critical checks pass."
                },
                "iv_families": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "family": {"type": "string"},
                            "files_found": {"type": "array", "items": {"type": "string"}},
                            "long_sheet_ok": {"type": "boolean"},
                            "required_columns_ok": {"type": "boolean"},
                            "missing_columns": {"type": "array", "items": {"type": "string"}},
                            "issues": {"type": "array", "items": {"type": "string"}},
                        },
                    },
                    "description": "Per-family IV check results.",
                },
                "cv_check": {
                    "type": "object",
                    "properties": {
                        "file_found": {"type": "boolean"},
                        "sheets_ok": {"type": "object"},
                        "issues": {"type": "array", "items": {"type": "string"}},
                    },
                    "description": "CV_train.xlsx check results.",
                },
                "it_check": {
                    "type": "object",
                    "properties": {
                        "dir_found": {"type": "boolean"},
                        "file_count": {"type": "integer"},
                        "issues": {"type": "array", "items": {"type": "string"}},
                    },
                    "description": "I-t directory check results.",
                },
                "summary": {"type": "string", "description": "Brief summary of findings."},
            },
        },
    },
}


_SWITCH_KERNEL_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": SUBAGENT_TOOL_SWITCH_KERNEL,
        "description": (
            "Switch the physical kernel used for parameter extraction. "
            "Call this when the current kernel produces poor IV fits "
            "(e.g. numerical oscillations, unrealistic VTO, or convergence failures)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "kernel": {
                    "type": "string",
                    "enum": ["uotft", "ekv_level2"],
                    "description": "Physical kernel name: uotft (default unified-OTFT) or ekv_level2 (EKV Level 2 model).",
                },
                "reasoning": {
                    "type": "string",
                    "description": "Why switching kernels is expected to help.",
                },
            },
            "required": ["kernel", "reasoning"],
        },
    },
}

_ACCEPT_QUALITY_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": SUBAGENT_TOOL_ACCEPT_QUALITY,
        "description": (
            "Accept the current quality metrics and move on without further adjustments. "
            "Use this when quality is close to acceptable, user preferences deprioritize "
            "the failing metric, or continued adjustments are unlikely to help."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "reasoning": {
                    "type": "string",
                    "description": "Why accepting current quality is the right call.",
                },
            },
            "required": ["reasoning"],
        },
    },
}

_CONTRIBUTE_KNOWLEDGE_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": SUBAGENT_TOOL_CONTRIBUTE_KNOWLEDGE,
        "description": (
            "Contribute a factual discovery to the shared knowledge base. "
            "This is persisted to knowledge/_discovered/ and will be available "
            "to future runs' agents as background context."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "Short title summarizing the discovery.",
                },
                "content": {
                    "type": "string",
                    "description": "Full discovery text in Markdown, including conditions, "
                    "observations, metric values, and actionable conclusions.",
                },
            },
            "required": ["title", "content"],
        },
    },
}


# ---------------------------------------------------------------------------
# Handlers — contribute_knowledge
# ---------------------------------------------------------------------------

def _contribute_knowledge_handler(arguments: dict[str, Any], ctx) -> dict[str, Any]:
    """Write a knowledge contribution to the discovered knowledge directory."""
    import os
    from datetime import datetime

    knowledge_dir = os.path.join("knowledge", "_discovered")
    os.makedirs(knowledge_dir, exist_ok=True)

    run_id = getattr(ctx, "run_id", None) or os.urandom(4).hex()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"run_{run_id}_{timestamp}.md"
    filepath = os.path.join(knowledge_dir, filename)

    title = arguments.get("title", "Untitled")
    content = arguments.get("content", "")

    full_text = f"# {title}\n\n{content}\n"

    try:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(full_text)
        return {
            "status": "written",
            "file": filepath,
            "title": title,
        }
    except Exception as e:
        return {"status": FAIL_LABEL, "error": str(e)}


# ---------------------------------------------------------------------------
# Stage tool registry — populated in Phase 2/3
# ---------------------------------------------------------------------------

_STAGE_TOOLS: dict[str, tuple[list[dict[str, Any]], dict[str, Callable[..., Any]]]] = {
    "data-prep": (
        [_INSPECT_XLSX_SCHEMA, _REPORT_VERIFICATION_SCHEMA],
        {SUBAGENT_TOOL_INSPECT_XLSX: _inspect_xlsx_handler},
    ),
    "fit-physical": (
        [
            _INSPECT_XLSX_SCHEMA,
            _ADJUST_PARAMS_SCHEMA,
            _SWITCH_KERNEL_SCHEMA,
            _FALLBACK_TO_PREVIOUS_SCHEMA,
            _ACCEPT_QUALITY_SCHEMA,
        ],
        {SUBAGENT_TOOL_INSPECT_XLSX: _inspect_xlsx_handler},
    ),
    "train-static": (
        [
            _INSPECT_XLSX_SCHEMA,
            _ADJUST_PARAMS_SCHEMA,
            _FALLBACK_TO_PREVIOUS_SCHEMA,
            _ACCEPT_QUALITY_SCHEMA,
        ],
        {SUBAGENT_TOOL_INSPECT_XLSX: _inspect_xlsx_handler},
    ),
    "train-hysteresis": (
        [
            _INSPECT_XLSX_SCHEMA,
            _ADJUST_PARAMS_SCHEMA,
            _FALLBACK_TO_PREVIOUS_SCHEMA,
            _ACCEPT_QUALITY_SCHEMA,
        ],
        {SUBAGENT_TOOL_INSPECT_XLSX: _inspect_xlsx_handler},
    ),
    "check-data": (
        [_FALLBACK_TO_PREVIOUS_SCHEMA],
        {},
    ),
    "final-interpret": (
        [_CONTRIBUTE_KNOWLEDGE_SCHEMA],
        {},
    ),
}


def get_tools(
    stage: str,
    adjust_params_override: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Callable[..., Any]]]:
    """Return (tool_schemas, tool_handlers) for the given modeling stage.

    stage: one of 'data-prep', 'check-data', 'fit-physical', 'train-static',
           'train-hysteresis', 'final-interpret', or '' for no tools.

    adjust_params_override: if provided, replaces the static ``_ADJUST_PARAMS_SCHEMA``
                           in the returned schemas.  This is used by the orchestrator to
                           inject a stage-specific ``adjust_params`` schema auto-generated
                           from parameter definition tables.

    Returns empty lists/dicts for unknown stages (graceful fallback).
    """
    schemas, handlers = _STAGE_TOOLS.get(stage, ([], {}))
    schemas = list(schemas)
    if adjust_params_override is not None:
        for i, s in enumerate(schemas):
            if s.get("function", {}).get("name") == SUBAGENT_TOOL_ADJUST_PARAMS:
                schemas[i] = adjust_params_override
                break
        else:
            schemas.append(adjust_params_override)
    return schemas, dict(handlers)
