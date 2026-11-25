# -*- coding: utf-8 -*-
"""
Header information
---------
Shanghai Jiao Tong University, School of Integrated Circuits, SMIL Lab
Author: Yu Huang
Create Date: 2026.7.1
Description: Operation log — structured per-stage attempt recording for DG-TFT pipeline

The operation log is the single source of truth for Agent E's final interpretation.
Every stage execution (initial or retry) is recorded as a StageAttemptLog entry,
regardless of whether an Agent was involved.  All attempts are preserved in
session/<uuid>/runs/run_NNNN/operation_log.json.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

sys_log = logging.getLogger("logger")


class AttemptTrigger(str, Enum):
    INITIAL = "initial"
    QUALITY_FAIL = "quality_fail"
    AGENT_ADJUST = "agent_adjust"
    FALLBACK_SENT = "fallback_sent"
    FALLBACK_RECEIVED = "fallback_received"
    UPSTREAM_CHANGED = "upstream_changed"
    VERIFIER_FAIL = "verifier_fail"
    PREP_RETRY = "prep_retry"
    AGENT_ACCEPT = "agent_accept"
    EXECUTION_FAILED = "execution_failed"
    INTERRUPTED = "interrupted"
    FEEDBACK_RESOLVED = "feedback_resolved"


@dataclass
class StageAttemptLog:
    n: int
    trigger: AttemptTrigger = AttemptTrigger.INITIAL
    upstream: str = ""
    handoff: dict[str, Any] | None = None
    agent_decision: dict[str, Any] | None = None
    agent_reasoning: str = ""
    ini_snapshot: str = ""
    duration_s: float = 0.0
    metrics: dict[str, float] = field(default_factory=dict)
    adjustments: dict[str, str] | None = None
    ok: bool = True
    error: str = ""
    quality_pass: bool | None = None
    summary: Any = ""
    outputs: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "trigger": self.trigger.value,
            "upstream": self.upstream,
            "handoff": self.handoff,
            "agent_decision": self.agent_decision,
            "agent_reasoning": self.agent_reasoning,
            "ini_snapshot": self.ini_snapshot,
            "duration_s": self.duration_s,
            "metrics": self.metrics,
            "adjustments": self.adjustments,
            "ok": self.ok,
            "error": self.error,
            "quality_pass": self.quality_pass,
            "summary": self.summary,
            "outputs": self.outputs,
            "warnings": self.warnings,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> StageAttemptLog:
        return cls(
            n=d.get("n", 0),
            trigger=AttemptTrigger(d.get("trigger", "initial")),
            upstream=d.get("upstream", ""),
            handoff=d.get("handoff"),
            agent_decision=d.get("agent_decision"),
            agent_reasoning=d.get("agent_reasoning", ""),
            ini_snapshot=d.get("ini_snapshot", ""),
            duration_s=d.get("duration_s", 0.0),
            metrics=d.get("metrics", {}),
            adjustments=d.get("adjustments"),
            ok=d.get("ok", True),
            error=d.get("error", ""),
            quality_pass=d.get("quality_pass"),
            summary=d.get("summary", ""),
            outputs=d.get("outputs", {}),
            warnings=d.get("warnings", []),
        )


class OperationLog:
    """Manages the operation_log.json file for a workflow run.

    Persisted at ``<run_output_dir>/operation_log.json``.
    After :meth:`initialize` is called the first time, subsequent
    :meth:`append_attempt` calls atomically append to the underlying file.
    """

    def __init__(self, output_dir: str | Path):
        self._path = Path(output_dir) / "operation_log.json"
        self._data: dict[str, Any] = {}

    # ── lifecycle ─────────────────────────────────────────────────

    def initialize(self, run_id: int, preferences: str = "") -> None:
        self._data = {
            "run_id": run_id,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "preferences": preferences,
            "final_status": "running",
            "stages": {},
        }
        self._save()

    def load(self) -> None:
        if self._path.exists():
            try:
                with open(self._path, "r", encoding="utf-8") as f:
                    self._data = json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                sys_log.warning("Failed to load operation log %s: %s", self._path, e)
                self._data = {}

    # ── mutators ──────────────────────────────────────────────────

    def append_attempt(self, stage: str, attempt: StageAttemptLog) -> None:
        stages = self._data.setdefault("stages", {})
        stage_attempts: list[dict[str, Any]] = stages.setdefault(stage, [])
        stage_attempts.append(attempt.to_dict())
        self._save()

    def set_final_status(self, status: str) -> None:
        self._data["final_status"] = status
        self._save()

    def append_event(self, kind: str, **details: Any) -> None:
        """Record state transitions without overwriting historical attempts."""
        self._data.setdefault("events", []).append({
            "kind": kind, "timestamp": datetime.now().isoformat(), **details,
        })
        self._save()

    # ── queries ───────────────────────────────────────────────────

    def get_stage_attempts(self, stage: str) -> list[dict[str, Any]]:
        return self._data.get("stages", {}).get(stage, [])

    def get_last_attempt(self, stage: str) -> dict[str, Any] | None:
        attempts = self.get_stage_attempts(stage)
        return attempts[-1] if attempts else None

    def get_last_attempt_n(self, stage: str) -> int:
        attempts = self.get_stage_attempts(stage)
        return len(attempts)

    @property
    def data(self) -> dict[str, Any]:
        return self._data

    # ── private ───────────────────────────────────────────────────

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self._path.with_suffix(".json.tmp")
        with open(temporary, "w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=2, ensure_ascii=False)
        os.replace(temporary, self._path)
