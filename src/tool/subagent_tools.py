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
SUBAGENT_TOOL_REPORT_VERIFICATION: str = "report_verification"
SUBAGENT_TOOL_CONTRIBUTE_KNOWLEDGE: str = "contribute_knowledge"
SUBAGENT_TOOL_ACCEPT_QUALITY: str = "accept_quality"
SUBAGENT_TOOL_SUBMIT_HANDOFF: str = "submit_handoff"

__all__ = [
    "get_tools",
    "resolve_agent_type_config",
    "SUBAGENT_TOOL_INSPECT_XLSX",
    "SUBAGENT_TOOL_ADJUST_PARAMS",
    "SUBAGENT_TOOL_FALLBACK_TO_PREVIOUS",
    "SUBAGENT_TOOL_REPORT_VERIFICATION",
    "SUBAGENT_TOOL_CONTRIBUTE_KNOWLEDGE",
    "SUBAGENT_TOOL_ACCEPT_QUALITY",
    "SUBAGENT_TOOL_SUBMIT_HANDOFF",
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

_SUBMIT_HANDOFF_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": SUBAGENT_TOOL_SUBMIT_HANDOFF,
        "description": (
            "Submit a structured handoff report after data preparation. "
            "Call this tool exactly once at the end — do not output free text."
        ),
        "parameters": {
            "type": "object",
            "required": ["observations", "actions"],
            "additionalProperties": False,
            "properties": {
                "observations": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Key observations about the raw data (file types found, format issues, column naming patterns).",
                },
                "actions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "What was done: which files were processed, how they were converted, any normalizations applied.",
                },
                "potential_issues": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Potential problems the verifier should check: missing VDS scans, ambiguous column names, incomplete files.",
                },
                "output_manifest": {
                    "type": "object",
                    "properties": {
                        "iv_files": {"type": "integer", "description": "Number of IV *_train.xlsx files produced."},
                        "cv_files": {"type": "integer", "description": "Number of CV *_train.xlsx files produced."},
                        "it_files": {"type": "integer", "description": "Number of I-t files in the it/ directory, or 0 if none."},
                    },
                    "description": "Count of output files produced, for the verifier to cross-check.",
                },
            },
        },
    },
}

_CONTRIBUTE_KNOWLEDGE_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": SUBAGENT_TOOL_CONTRIBUTE_KNOWLEDGE,
        "description": (
            "Contribute a genuinely novel, generalizable discovery to the shared knowledge base. "
            "Only call this when you have a NON-OBVIOUS finding that future runs would benefit from. "
            "All output MUST be in English. "
            "The file is saved to knowledge/_discovered/ with a YAML header for structured indexing.\n\n"
            "DO NOT CALL THIS for: (a) routine observations like 'learning rate was too high', "
            "(b) one-run metrics without generalizable conditions, (c) facts already documented in "
            "the EXPERT KNOWLEDGE, EXPERIENCE, or DISCOVERED KNOWLEDGE blocks in your prompt.\n\n"
            "ONLY call this when you find a genuinely new PATTERN — something reproducible and "
            "cross-device, with clear triggering conditions, that would guide a future agent to "
            "a better decision.\n\n"
            "The `content` field MUST follow this template (no other format accepted):\n"
            "```\n"
            "## Category: [one of: Diagnostic Signal, Tuning Recipe, Tool Strategy, Engine Behavior, Pipeline Pattern]\n"
            "## Stage(s): [which stage(s) this applies to]\n"
            "## Symptom: [what the agent sees — metric value, error message, log pattern]\n"
            "## Root Cause: [why it happens — the underlying mechanical or physical reason, NOT a guess]\n"
            "## Evidence: [specific metric values, file paths, reproductions]\n"
            "## Actionable Advice:\n"
            "1. [specific action the agent should take first]\n"
            "2. [fallback if action 1 fails]\n"
            "```"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "Short English title summarizing the discovery (used as filename slug and YAML frontmatter)."
                },
                "summary": {
                    "type": "string",
                    "description": (
                        "A short, self-contained summary of the discovery (2-4 sentences). This is the ONLY part "
                        "that will be injected into future agent prompts for indexing. It MUST be specific enough "
                        "to be recognized as relevant when a future agent encounters a similar situation. Include: "
                        "the triggering condition, the recommended action, and the expected outcome."
                    ),
                },
                "content": {
                    "type": "string",
                    "description": "Full discovery text following the required template format above."
                },
            },
            "required": ["title", "summary", "content"],
        },
    },
}


# ---------------------------------------------------------------------------
# Handlers — contribute_knowledge
# ---------------------------------------------------------------------------

def _contribute_knowledge_handler(arguments: dict[str, Any], ctx) -> dict[str, Any]:
    """Write a knowledge contribution with YAML frontmatter to the discovered directory."""
    import os
    import re
    from datetime import datetime

    knowledge_dir = os.path.join("knowledge", "_discovered")
    os.makedirs(knowledge_dir, exist_ok=True)

    title = arguments.get("title", "").strip()
    summary = arguments.get("summary", "").strip()
    content = arguments.get("content", "")

    if title:
        slug = re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")[:80]
    else:
        slug = "discovery"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{slug}_{timestamp}.md"
    filepath = os.path.join(knowledge_dir, filename)

    yaml_header = f"title: {title or 'Untitled'}\n"
    if summary:
        yaml_header += f"summary: {summary}\n"

    full_text = f"---\n{yaml_header}---\n\n# {title or 'Untitled'}\n\n{content}\n"

    try:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(full_text)
        return {
            "status": "written",
            "file": filepath,
            "title": title or "Untitled",
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


# ---------------------------------------------------------------------------
# Agent type → tool configuration (auto-resolved for dedicated types)
# ---------------------------------------------------------------------------
# Dedicated agent types have fixed tool sets determined by their type,
# without requiring callers to pass subagent_tools / tool_handlers params.
# Non-dedicated types (explorer, worker, scheduler) resolve to None —
# callers pass subagent_tools explicitly for those.
# ---------------------------------------------------------------------------

from src.constants import VERIFIER_AGENT_LABEL, PREPARER_AGENT_LABEL, DIAGNOSTIC_AGENT_LABEL, INTERPRETER_AGENT_LABEL

_AGENT_TYPE_CONFIG: dict[str, dict[str, Any]] = {
    PREPARER_AGENT_LABEL: {
        "tool_schemas": [_INSPECT_XLSX_SCHEMA, _SUBMIT_HANDOFF_SCHEMA],
        "tool_handlers": {SUBAGENT_TOOL_INSPECT_XLSX: _inspect_xlsx_handler},
        "stop_mode": "schema",
        "stop_schemas": [SUBAGENT_TOOL_SUBMIT_HANDOFF],
    },
    VERIFIER_AGENT_LABEL: {
        "tool_schemas": [_INSPECT_XLSX_SCHEMA, _REPORT_VERIFICATION_SCHEMA],
        "tool_handlers": {SUBAGENT_TOOL_INSPECT_XLSX: _inspect_xlsx_handler},
        "stop_mode": "schema",
        "stop_schemas": [SUBAGENT_TOOL_REPORT_VERIFICATION],
    },
    DIAGNOSTIC_AGENT_LABEL: {
        "tool_schemas": [_INSPECT_XLSX_SCHEMA, _FALLBACK_TO_PREVIOUS_SCHEMA],
        "tool_handlers": {SUBAGENT_TOOL_INSPECT_XLSX: _inspect_xlsx_handler},
        "stop_mode": "schema",
        "stop_schemas": [SUBAGENT_TOOL_FALLBACK_TO_PREVIOUS],
    },
    INTERPRETER_AGENT_LABEL: {
        "tool_schemas": [_CONTRIBUTE_KNOWLEDGE_SCHEMA],
        "tool_handlers": {SUBAGENT_TOOL_CONTRIBUTE_KNOWLEDGE: _contribute_knowledge_handler},
        "stop_mode": "any",
        "stop_schemas": [],
    },
}


def resolve_agent_type_config(
    agent_type: str,
) -> dict[str, Any] | None:
    """Return tool configuration for a dedicated agent type.

    Returns a dict with keys: tool_schemas, tool_handlers, stop_mode, stop_schemas.
    Returns None for non-dedicated types (callers must provide tool config explicitly).
    """
    return _AGENT_TYPE_CONFIG.get(agent_type)
