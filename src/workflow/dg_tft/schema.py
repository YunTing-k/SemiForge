# -*- coding: utf-8 -*-
"""
Header information
---------
Shanghai Jiao Tong University, School of Integrated Circuits, SMIL Lab
Author: Yu Huang
Create Date: 2026.6.25
Description: Data structures for DG-TFT workflow runs

Revision:
---------
2026.6.25      Yu Huang      1.0      Initial implementation
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any


class WorkflowStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    DONE = "DONE"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class StageName:
    CHECK_DATA = "check-data"
    FIT_PHYSICAL = "fit-physical"
    TRAIN_BASE_PKAN = "train-base-pkan"
    TRAIN_STATIC = "train-static"
    TRAIN_HYSTERESIS = "train-hysteresis"
    DEPLOYMENT_VALIDATE = "deployment-validate"
    FINAL_REPORT = "final-report"

    STAGE_ORDER: list[str] = [
        CHECK_DATA,
        FIT_PHYSICAL,
        TRAIN_BASE_PKAN,
        TRAIN_STATIC,
        TRAIN_HYSTERESIS,
        DEPLOYMENT_VALIDATE,
        FINAL_REPORT,
    ]


class ErrorCategory(str, Enum):
    DATA_ERROR = "DATA_ERROR"
    CONFIG_ERROR = "CONFIG_ERROR"
    TRAINING_NUMERICAL_ERROR = "TRAINING_NUMERICAL_ERROR"
    MISSING_ARTIFACT_ERROR = "MISSING_ARTIFACT_ERROR"
    HSPICE_ENV_ERROR = "HSPICE_ENV_ERROR"
    UNKNOWN_ERROR = "UNKNOWN_ERROR"


@dataclass
class StageResult:
    ok: bool
    stage: str
    summary: str = ""
    outputs: dict[str, str] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    error: str | None = None
    skipped: bool = False
    skip_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DatasetProfile:
    static_ok: bool = False
    cv_ok: bool = False
    iv_families: list[str] = field(default_factory=list)
    has_temperature_iv: bool = False
    temperature_span_K: float | None = None
    has_hysteresis: bool = False
    hysteresis_workbooks: list[str] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)
    suggested_hysteresis_dir: str | None = None
    total_files: int = 0
    raw_validation: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class WorkflowRun:
    run_id: int
    workflow_type: str = "dg_tft"
    dataset_dir: str = ""
    hysteresis_dir: str | None = None
    output_root: str = ""
    prep_id: int | None = None
    status: WorkflowStatus = WorkflowStatus.PENDING
    current_stage: str = ""
    created_at: str = ""
    updated_at: str = ""
    hspice_available: bool = False
    temperature_enabled: bool = False
    hysteresis_enabled: bool = False
    artifacts: dict[str, str] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    stage_results: dict[str, StageResult] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        raw = {
            "run_id": self.run_id,
            "workflow_type": self.workflow_type,
            "dataset_dir": self.dataset_dir,
            "hysteresis_dir": self.hysteresis_dir,
            "output_root": self.output_root,
            "prep_id": self.prep_id,
            "status": self.status.value,
            "current_stage": self.current_stage,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "hspice_available": self.hspice_available,
            "temperature_enabled": self.temperature_enabled,
            "hysteresis_enabled": self.hysteresis_enabled,
            "artifacts": dict(self.artifacts),
            "metrics": dict(self.metrics),
            "warnings": list(self.warnings),
            "errors": list(self.errors),
            "stage_results": {},
            "extra": dict(self.extra),
        }
        stage_dict = {}
        for k, v in self.stage_results.items():
            if hasattr(v, "__dataclass_fields__"):
                stage_dict[k] = {
                    "ok": getattr(v, "ok", False),
                    "stage": getattr(v, "stage", ""),
                    "summary": getattr(v, "summary", ""),
                    "outputs": dict(getattr(v, "outputs", {})),
                    "metrics": dict(getattr(v, "metrics", {})),
                    "warnings": list(getattr(v, "warnings", [])),
                    "error": getattr(v, "error", None),
                    "skipped": getattr(v, "skipped", False),
                    "skip_reason": getattr(v, "skip_reason", ""),
                }
            elif isinstance(v, dict):
                stage_dict[k] = v
            else:
                stage_dict[k] = v
        raw["stage_results"] = stage_dict
        return raw

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> WorkflowRun:
        stage_results = {}
        for k, v in d.pop("stage_results", {}).items():
            stage_results[k] = StageResult(**v)
        d["status"] = WorkflowStatus(d["status"])
        if "prep_id" not in d:
            d["prep_id"] = None
        run = cls(**{k: v for k, v in d.items() if k != "stage_results"})
        run.stage_results = stage_results
        return run
