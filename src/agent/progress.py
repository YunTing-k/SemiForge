# -*- coding: utf-8 -*-
"""
Header information
---------
Shanghai Jiao Tong University, School of Integrated Circuits, SMIL Lab
Author: Yu Huang
Create Date: 2026.6.12
Description: Agent status and progress types for SemiForge subagent coordination

Revision:
---------
2026.6.12      Yu Huang      1.0      First implementation
2026.6.13      Yu Huang      1.1      Add if_background and subject fields, replace simulate with scheduler agent type

Details:
---------
AgentStatus enum (pending/running/done/error/timeout) and SubAgentProgress dataclass with serialization. Also defines
subagent type configuration: SUPPORTED_TYPES (tool allowlists), PERMISSION_PRESETS (pre-granted permissions), and
SUPPORTED_TYPES_DESC (LLM-facing descriptions).
"""
from enum import Enum
from dataclasses import dataclass
from typing import Any
from src.constants import *


class AgentStatus(str, Enum):
    PENDING = AGENT_PENDING_LABEL
    RUNNING = AGENT_RUNNING_LABEL
    TIMEOUT = AGENT_TIMEOUT_LABEL
    ERROR = AGENT_ERROR_LABEL
    DONE = AGENT_DONE_LABEL


@dataclass
class SubAgentProgress:
    agent_id: str
    subagent_type: str
    subject: str
    status: AgentStatus
    if_background: bool = False
    if_archived: bool = False
    step: int = 0
    current_tool: str = ""
    tool_calls_done: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    start_time: float = 0.0
    elapsed_s: float = 0.0
    last_activity: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "subagent_type": self.subagent_type,
            "subject": self.subject,
            "status": self.status.value,
            "if_background": self.if_background,
            "if_archived": self.if_archived,
            "step": self.step,
            "current_tool": self.current_tool,
            "tool_calls_done": self.tool_calls_done,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "start_time": self.start_time,
            "elapsed_s": self.elapsed_s,
            "last_activity": self.last_activity,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SubAgentProgress":
        return cls(
            agent_id=d["agent_id"],
            subagent_type=d["subagent_type"],
            subject=d.get("subject", ""),
            status=AgentStatus(d["status"]),
            if_background=d.get("if_background", False),
            if_archived=d.get("if_archived", False),
            step=d.get("step", 0),
            current_tool=d.get("current_tool", ""),
            tool_calls_done=d.get("tool_calls_done", 0),
            input_tokens=d.get("input_tokens", 0),
            output_tokens=d.get("output_tokens", 0),
            start_time=d.get("start_time", 0.0),
            elapsed_s=d.get("elapsed_s", 0.0),
            last_activity=d.get("last_activity", 0.0),
        )


SUPPORTED_TYPES: dict[str, tuple[str, ...]] = {
    EXPLORER_AGENT_LABEL: (
        TOOL_NAME_GLOB_FILE,
        TOOL_NAME_GREP_FILE,
        TOOL_NAME_READ_FILE,
        TOOL_NAME_WEB_FETCH,
        TOOL_NAME_WEB_SEARCH,
        TOOL_NAME_BASH,
        TOOL_NAME_CREATE_TASK,
        TOOL_NAME_UPDATE_TASK,
        TOOL_NAME_QUERY_TASK,
    ),
    WORKER_AGENT_LABEL: (),
    SCHEDULER_AGENT_LABEL: (
        TOOL_NAME_CREATE_TASK,
        TOOL_NAME_UPDATE_TASK,
        TOOL_NAME_QUERY_TASK,
        TOOL_NAME_GLOB_FILE,
        TOOL_NAME_GREP_FILE,
        TOOL_NAME_READ_FILE,
        TOOL_NAME_WEB_FETCH,
        TOOL_NAME_WEB_SEARCH,
        TOOL_NAME_BASH,
    ),
    PREPARER_AGENT_LABEL: (
        TOOL_NAME_GLOB_FILE,
        TOOL_NAME_GREP_FILE,
        TOOL_NAME_READ_FILE,
        TOOL_NAME_WRITE_FILE,
        TOOL_NAME_BASH,
        TOOL_NAME_CREATE_TASK,
        TOOL_NAME_UPDATE_TASK,
        TOOL_NAME_QUERY_TASK,
    ),
    VERIFIER_AGENT_LABEL: (
        TOOL_NAME_GLOB_FILE,
        TOOL_NAME_GREP_FILE,
        TOOL_NAME_READ_FILE,
        TOOL_NAME_BASH,
    ),
    DIAGNOSTIC_AGENT_LABEL: (
        TOOL_NAME_GLOB_FILE,
        TOOL_NAME_GREP_FILE,
        TOOL_NAME_READ_FILE,
        TOOL_NAME_BASH,
    ),
    MODELING_AGENT_LABEL: (
        TOOL_NAME_GLOB_FILE,
        TOOL_NAME_GREP_FILE,
        TOOL_NAME_READ_FILE,
        TOOL_NAME_WEB_FETCH,
        TOOL_NAME_WEB_SEARCH,
        TOOL_NAME_BASH,
    ),
    INTERPRETER_AGENT_LABEL: (
        TOOL_NAME_GLOB_FILE,
        TOOL_NAME_GREP_FILE,
        TOOL_NAME_READ_FILE,
        TOOL_NAME_BASH,
    ),
}

PERMISSION_PRESETS: dict[str, tuple[str, ...]] = {
    EXPLORER_AGENT_LABEL: (
        TOOL_NAME_GLOB_FILE,
        TOOL_NAME_GREP_FILE,
        TOOL_NAME_READ_FILE,
        TOOL_NAME_WEB_FETCH,
        TOOL_NAME_WEB_SEARCH,
        TOOL_NAME_CREATE_TASK,
        TOOL_NAME_UPDATE_TASK,
        TOOL_NAME_QUERY_TASK,
        BASH_SAFE_LABEL,
        BASH_NETWORK_LABEL,
    ),
    WORKER_AGENT_LABEL: (
        TOOL_NAME_GLOB_FILE,
        TOOL_NAME_GREP_FILE,
        TOOL_NAME_READ_FILE,
        TOOL_NAME_WRITE_FILE,
        TOOL_NAME_EDIT_FILE,
        TOOL_NAME_WEB_FETCH,
        TOOL_NAME_WEB_SEARCH,
        TOOL_NAME_CREATE_TASK,
        TOOL_NAME_UPDATE_TASK,
        TOOL_NAME_QUERY_TASK,
        BASH_SAFE_LABEL,
        BASH_NETWORK_LABEL,
        BASH_FILE_LABEL,
        BASH_INLINE_SCRIPT_LABEL,
    ),
    SCHEDULER_AGENT_LABEL: (
        TOOL_NAME_CREATE_TASK,
        TOOL_NAME_UPDATE_TASK,
        TOOL_NAME_QUERY_TASK,
        TOOL_NAME_GLOB_FILE,
        TOOL_NAME_GREP_FILE,
        TOOL_NAME_READ_FILE,
        TOOL_NAME_WEB_FETCH,
        TOOL_NAME_WEB_SEARCH,
        BASH_SAFE_LABEL,
    ),
    PREPARER_AGENT_LABEL: (
        TOOL_NAME_GLOB_FILE,
        TOOL_NAME_GREP_FILE,
        TOOL_NAME_READ_FILE,
        TOOL_NAME_WRITE_FILE,
        TOOL_NAME_CREATE_TASK,
        TOOL_NAME_UPDATE_TASK,
        TOOL_NAME_QUERY_TASK,
        BASH_SAFE_LABEL,
        BASH_INLINE_SCRIPT_LABEL,
        BASH_FILE_LABEL,
        BASH_UNKNOWN_LABEL,
    ),
    VERIFIER_AGENT_LABEL: (
        TOOL_NAME_GLOB_FILE,
        TOOL_NAME_GREP_FILE,
        TOOL_NAME_READ_FILE,
        BASH_SAFE_LABEL,
        BASH_INLINE_SCRIPT_LABEL,
    ),
    DIAGNOSTIC_AGENT_LABEL: (
        TOOL_NAME_GLOB_FILE,
        TOOL_NAME_GREP_FILE,
        TOOL_NAME_READ_FILE,
        BASH_SAFE_LABEL,
        BASH_INLINE_SCRIPT_LABEL,
    ),
    MODELING_AGENT_LABEL: (
        TOOL_NAME_GLOB_FILE,
        TOOL_NAME_GREP_FILE,
        TOOL_NAME_READ_FILE,
        BASH_SAFE_LABEL,
        BASH_INLINE_SCRIPT_LABEL,
        BASH_NETWORK_LABEL,
        BASH_FILE_LABEL,
    ),
    INTERPRETER_AGENT_LABEL: (
        TOOL_NAME_GLOB_FILE,
        TOOL_NAME_GREP_FILE,
        TOOL_NAME_READ_FILE,
        BASH_SAFE_LABEL,
        BASH_INLINE_SCRIPT_LABEL,
    ),
}

SUPPORTED_TYPES_DESC: dict[str, str] = {
    EXPLORER_AGENT_LABEL: (
        f"Read-only explorer. Can use {TOOL_NAME_GLOB_FILE}, `{TOOL_NAME_GREP_FILE}`, `{TOOL_NAME_READ_FILE}`, "
        f"`{TOOL_NAME_WEB_FETCH}`, `{TOOL_NAME_WEB_SEARCH}`, `{TOOL_NAME_BASH}`, plus task tools "
        f"(`{TOOL_NAME_CREATE_TASK}` / `{TOOL_NAME_UPDATE_TASK}` / `{TOOL_NAME_QUERY_TASK}`) for own workflow (independent "
        f"scoreboard)."
    ),
    WORKER_AGENT_LABEL: (
        "General-purpose worker agent for research, multi-step tasks, and implementation. Has all tools except spawning other "
        "agents and cron tasks management. Can use task tools for their own workflow (independent scoreboard)."
    ),
    SCHEDULER_AGENT_LABEL: (
        f"Task planning and dependency management agent with shared scoreboard access. Can use {TOOL_NAME_CREATE_TASK} / "
        f"{TOOL_NAME_UPDATE_TASK} / {TOOL_NAME_QUERY_TASK} to plan and organize work, set task dependencies, delete incorrect "
        f"tasks, and can access read-only tools. Tasks it creates are available for main-agent and subagent that have shared "
        f"scoreboard access."
    ),
    PREPARER_AGENT_LABEL: (
        f"Data preparation specialist for DG-TFT device modeling. Converts raw CSV/XLSX measurement files into standardized "
        f"*_train.xlsx format. Has {TOOL_NAME_GLOB_FILE}, {TOOL_NAME_GREP_FILE}, {TOOL_NAME_READ_FILE}, {TOOL_NAME_WRITE_FILE}, "
        f"and {TOOL_NAME_BASH} (with python inline execution) for efficient batch file processing."
    ),
    VERIFIER_AGENT_LABEL: (
        f"Data verification specialist for DG-TFT device modeling. Inspects prepared *_train.xlsx files "
        f"to verify sheet structure, required columns, and data integrity. Has {TOOL_NAME_GLOB_FILE}, "
        f"{TOOL_NAME_GREP_FILE}, {TOOL_NAME_READ_FILE}, and {TOOL_NAME_BASH} (safe + inline python only) "
        f"for file inspection and data validation."
    ),
    DIAGNOSTIC_AGENT_LABEL: (
        f"Diagnostic specialist for check-data failures in DG-TFT modeling. Analyzes dataset structural "
        f"validation results and decides whether to fall back to upstream stages. Has {TOOL_NAME_GLOB_FILE}, "
        f"{TOOL_NAME_GREP_FILE}, {TOOL_NAME_READ_FILE}, and {TOOL_NAME_BASH} (safe + inline python only) "
        f"for inspecting validation outputs, error logs, and xlsx file structure."
    ),
    MODELING_AGENT_LABEL: (
        f"Modeling agent for DG-TFT parameter tuning (Agents B/C/D). Analyzes stage results, quality "
        f"metrics, and output artifacts to decide between adjusting parameters, falling back upstream, "
        f"or accepting current quality. Has {TOOL_NAME_GLOB_FILE}, {TOOL_NAME_GREP_FILE}, "
        f"{TOOL_NAME_READ_FILE}, and {TOOL_NAME_BASH} (safe + inline python only) for inspecting "
        f"output files, logs, metrics JSON, and xlsx file structure."
    ),
    INTERPRETER_AGENT_LABEL: (
        f"Final interpretation specialist for DG-TFT modeling (Agent E). Reads operation_log.json, "
        f"RUN_SUMMARY, and stage output artifacts to produce a comprehensive interpretation: "
        f"recommends the best attempt chain across all stages, diagnoses root causes of failures, "
        f"and optionally contributes novel discoveries to the shared knowledge base. Has "
        f"{TOOL_NAME_GLOB_FILE}, {TOOL_NAME_GREP_FILE}, {TOOL_NAME_READ_FILE}, and "
        f"{TOOL_NAME_BASH} (safe + inline python only) for reading output files and logs."
    ),
}
