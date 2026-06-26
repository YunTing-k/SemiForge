# -*- coding: utf-8 -*-
"""
Header information
---------
Shanghai Jiao Tong University, School of Integrated Circuits, SMIL Lab
Author: Yu Huang
Create Date: 2026.6.26
Description: Error classification engine for DG-TFT workflow failures.

    Maps stage failures to finite error categories using keyword matching
    and exit-code heuristics, enabling downstream diagnostic hooks (Phase 6)
    and quality-guard retry decisions (Phase 7).

    Classification rules are defined as module-level constants — no inline
    hardcoding — so adding a new category or keyword sigil only requires a
    single-table edit.

Revision:
---------
2026.6.26      Yu Huang      1.0      Initial implementation (P6-T1)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .schema import ErrorCategory, StageName, StageResult

if TYPE_CHECKING:
    pass

# ── Keyword → ErrorCategory lookup table ───────────────────────────
# Each tuple: (ErrorCategory, confidence, [keyword sigils ...])
# Keywords are matched case-insensitively against the combined
# error_message + stderr text.  Order matters: earlier rows win.
_KEYWORD_RULES: list[tuple[ErrorCategory, float, list[str]]] = [
    # ── TRAINING_NUMERICAL_ERROR (high-priority — before generic DATA_ERROR) ──
    (ErrorCategory.TRAINING_NUMERICAL_ERROR, 0.9, [
        "cuda out of memory",
        "out of memory",
        "oom",
        "cudnn",
        "nan",
        "inf",
        "overflow encountered",
        "underflow encountered",
        "overflow",
        "gradient",
        "backward",
    ]),
    (ErrorCategory.TRAINING_NUMERICAL_ERROR, 0.7, [
        "diverged",
        "not converging",
        "loss is nan",
        "loss too large",
        "gradient explosion",
    ]),

    # ── HSPICE_ENV_ERROR ──
    (ErrorCategory.HSPICE_ENV_ERROR, 0.9, [
        "hspice.exe",
        "hspice not found",
        "hspice executable",
        "hspice path",
        "hspice simulation failed",
        "hspice timeout",
        "hspice license",
    ]),

    # ── DATA_ERROR ──
    (ErrorCategory.DATA_ERROR, 0.9, [
        "FileNotFoundError",
        "No such file or directory",
        "dataset_validation.json",
        "missing column",
        "missing sheet",
        "invalid sheet",
        "sheet not found",
        "column not found",
        "unexpected header",
        "empty workbook",
        "no valid data",
        "data_dir not found",
        "does not exist",
        "cannot find it directory",
        "xlsx file is empty",
        "excel file format cannot be determined",
    ]),
    (ErrorCategory.DATA_ERROR, 0.6, [
        "keyerror",
        "indexerror",
        "valueerror",
    ]),

    # ── CONFIG_ERROR ──
    (ErrorCategory.CONFIG_ERROR, 0.9, [
        "configparser",
        "no section",
        "no option",
        "invalid config",
        "missing config",
        "configuration error",
        "unknown stage",
        "unknown action",
    ]),

    # ── MISSING_ARTIFACT_ERROR ──
    (ErrorCategory.MISSING_ARTIFACT_ERROR, 0.9, [
        "missing artifact",
        "expected output not found",
        "no such file",
        "cannot find",
        "not generated",
        "output file.*missing",
        "artifact.*not created",
    ]),
]

# Confidence floor when classification falls through to UNKNOWN_ERROR.
_UNKNOWN_CONFIDENCE: float = 0.3

# Exit-code bands for stage-type inference (used when stdout/stderr
# contain no explicit keywords).
_EXIT_CODE_DATA_ERROR_STAGES = frozenset({StageName.CHECK_DATA})
_EXIT_CODE_NUMERICAL_STAGES = frozenset({StageName.FIT_PHYSICAL, StageName.TRAIN_STATIC, StageName.TRAIN_HYSTERESIS})


@dataclass
class DiagnosticResult:
    """Result of classifying a workflow stage failure."""

    category: ErrorCategory = ErrorCategory.UNKNOWN_ERROR
    confidence: float = 0.0
    evidence: list[str] = field(default_factory=list)
    suggestion: str = ""

    def to_dict(self) -> dict:
        return {
            "category": self.category.value,
            "confidence": self.confidence,
            "evidence": self.evidence,
            "suggestion": self.suggestion,
        }


# ── Stage-specific suggestion messages ─────────────────────────────

_STAGE_SUGGESTIONS: dict[str, str] = {
    StageName.CHECK_DATA:
        "Verify the dataset directory contains valid *_train.xlsx files "
        "with required sheets and columns.",
    StageName.FIT_PHYSICAL:
        "Check the physical model fitting inputs — may indicate "
        "incompatible data range or missing template files.",
    StageName.TRAIN_STATIC:
        "Try adjusting static training hyperparameters or checking "
        "for NaN/inf in the training data.",
    StageName.TRAIN_HYSTERESIS:
        "Verify I-t data availability, hysteresis configuration, "
        "and that dynamic engine scripts are present.",
}


def classify_error(
    stage_name: str,
    exit_code: int,
    stdout: str = "",
    stderr: str = "",
    error_message: str = "",
    missing_artifacts: list[str] | None = None,
) -> DiagnosticResult:
    """Classify a stage failure into a finite error category.

    Prioritises explicit keyword matches in the combined error text
    above exit-code heuristics.  When no keywords match the stage
    name is used to guess a plausible category.

    Args:
        stage_name:  e.g.  ``check-data``, ``train-static``.
        exit_code:   Subprocess return code (0 = success).
        stdout:      Captured standard output text.
        stderr:      Captured standard error text.
        error_message:  Pre-parsed error string (e.g. from StageResult.error).
        missing_artifacts:  List of artifact paths that were expected but absent.

    Returns:
        DiagnosticResult with category, confidence, evidence and suggestion.
    """
    combined = _build_search_text(error_message, stderr).lower()
    missing = missing_artifacts or []

    # 1 — Explicit keyword match
    for cat, conf, sigils in _KEYWORD_RULES:
        evidence = [s for s in sigils if s.lower() in combined]
        if evidence:
            return DiagnosticResult(
                category=cat,
                confidence=conf,
                evidence=evidence,
                suggestion=_stage_suggestion(stage_name),
            )

    # 2 — Missing artifacts despite exit_code == 0
    if exit_code == 0 and missing:
        return DiagnosticResult(
            category=ErrorCategory.MISSING_ARTIFACT_ERROR,
            confidence=0.85,
            evidence=[
                f"Stage succeeded (exit 0) but {len(missing)} expected "
                f"artifact(s) missing: {_truncate_list(missing)}"
            ],
            suggestion=_stage_suggestion(stage_name),
        )

    # 3 — Exit-code-guided fallback
    if exit_code != 0:
        if stage_name in _EXIT_CODE_DATA_ERROR_STAGES:
            cat, conf = ErrorCategory.DATA_ERROR, 0.5
        elif stage_name in _EXIT_CODE_NUMERICAL_STAGES:
            cat, conf = ErrorCategory.TRAINING_NUMERICAL_ERROR, 0.4
        else:
            cat, conf = ErrorCategory.UNKNOWN_ERROR, _UNKNOWN_CONFIDENCE
        return DiagnosticResult(
            category=cat,
            confidence=conf,
            evidence=[f"Non-zero exit code {exit_code} on stage '{stage_name}'"],
            suggestion=_stage_suggestion(stage_name),
        )

    # 4 — Complete unknown
    return DiagnosticResult(
        category=ErrorCategory.UNKNOWN_ERROR,
        confidence=_UNKNOWN_CONFIDENCE,
        evidence=["No explicit error signature found in stdout/stderr"],
        suggestion="Review the full stage log for details.",
    )


def classify_from_result(
    stage_result: StageResult,
    exit_code: int = 0,
    stdout: str = "",
    stderr: str = "",
    missing_artifacts: list[str] | None = None,
) -> DiagnosticResult:
    """Convenience wrapper that unpacks a StageResult.

    If the stage_result already has an ``ok == False``, its ``.error``
    field is used as the primary error_message.  Otherwise stdout/stderr
    are searched for failure signatures.

    Args:
        stage_result:  A populated StageResult (usually ok=False).
        exit_code:     Subprocess exit code.
        stdout:        Raw stdout text from the stage subprocess.
        stderr:        Raw stderr text from the stage subprocess.
        missing_artifacts:  Expected artifact paths not found on disk.

    Returns:
        DiagnosticResult.
    """
    return classify_error(
        stage_name=stage_result.stage,
        exit_code=exit_code,
        stdout=stdout,
        stderr=stderr,
        error_message=stage_result.error or "",
        missing_artifacts=missing_artifacts,
    )


# ── Internal helpers ───────────────────────────────────────────────


def _build_search_text(error_message: str, stderr: str) -> str:
    parts: list[str] = []
    if error_message:
        parts.append(error_message)
    if stderr:
        parts.append(stderr)
    return " ".join(parts)


def _stage_suggestion(stage_name: str) -> str:
    return _STAGE_SUGGESTIONS.get(
        stage_name,
        "Check the stage logs for detailed error information.",
    )


def _truncate_list(items: list[str], max_show: int = 3) -> str:
    if len(items) <= max_show:
        return ", ".join(items)
    return ", ".join(items[:max_show]) + f" (+{len(items) - max_show} more)"
