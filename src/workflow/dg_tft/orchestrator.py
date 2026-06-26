# -*- coding: utf-8 -*-
"""
Header information
---------
Shanghai Jiao Tong University, School of Integrated Circuits, SMIL Lab
Author: Yu Huang
Create Date: 2026.6.25
Description: DG-TFT workflow orchestrator — sequential stage execution with
             fail-fast, conditional skip, artifact verification, resume,
             report generation, task board sync, console summary,
             and automatic failure diagnosis (Phase 6).

Revision:
---------
2026.6.25      Yu Huang      1.0      Initial implementation
2026.6.25      Yu Huang      1.1      Phase 5: report builder, task board, console summary
2026.6.26      Yu Huang      1.2      Phase 6: automatic failure diagnosis via subagent hooks
"""
from __future__ import annotations

import configparser
import logging
import os
import re
import time
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

from .schema import (
    WorkflowStatus,
    StageResult,
    StageName,
    DatasetProfile,
)
from .stages import get_stages, StageDef
from .stage_runner import run_stage
from .artifacts import STAGE_DISPLAY_NAMES, get_report_paths
from .report_builder import build_and_write_reports
from .diagnostics import classify_from_result, DiagnosticResult
from .subagent_hooks import get_hook, build_diagnosis_prompt
from src.constants import (
    MAJOR_COLOR2,
    WF_STATUS_ERROR,
    WF_STATUS_SUCCESS,
    WF_STATUS_WARN,
    WF_MISSING_ITEM,
    WF_SECTION_TITLE,
    EXPLORER_AGENT_LABEL,
)

if TYPE_CHECKING:
    from src.context.agent_context import AgentContext

sys_log = logging.getLogger("logger")

# ── Constants ──────────────────────────────────────────────────────

_DIAGNOSIS_CONFIG_KEY: str = "DG_TFT_ENABLE_DIAGNOSIS"
_DIAGNOSIS_SUBAGENT_TIMEOUT_S: int = 300
_DIAGNOSIS_SUBAGENT_MAX_STEPS: int = 12
_DIAGNOSIS_SUBAGENT_ID_PREFIX: str = "diag"
_WF_TASK_PREFIX: str = "[DG-TFT]"

_STAGE_TASK_SUBJECTS: dict[str, str] = {
    StageName.CHECK_DATA: "Audit dataset",
    StageName.FIT_PHYSICAL: "Fit physical kernel",
    StageName.TRAIN_STATIC: "Train static model",
    StageName.TRAIN_HYSTERESIS: "Train hysteresis",
}

_STAGE_TASK_DESCS: dict[str, str] = {
    StageName.CHECK_DATA: "Validate dataset structure and inspect IV/CV/I-t profiles.",
    StageName.FIT_PHYSICAL: "Fit physical equations to extract initial parameter kernels.",
    StageName.TRAIN_STATIC: "Train the full static model and export Verilog-A.",
    StageName.TRAIN_HYSTERESIS: "Train hysteresis-aware dynamic extension and export dynamic VA.",
}


class OrchestratorResult(str, Enum):
    DONE = "DONE"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


def run_workflow(
    ctx: AgentContext,
    run_id: int,
    config_path: str,
    *,
    stage_timeout: int | None = None,
    board=None,
) -> tuple[bool, str]:
    """Execute the DG-TFT modeling pipeline for a single workflow run.

    Stages are executed in order (check-data → fit-physical
    → train-static → train-hysteresis), with fail-fast on required stages and
    automatic skip of hysteresis when no I-t data is available.

    Supports resume: stages that have already completed successfully are
    skipped on re-entry.

    Args:
        ctx: Agent context (for configs, manager, and interpreter resolution).
        run_id: Workflow run ID to execute.
        config_path: Path to the runtime model_config.ini.
        stage_timeout: Override default per-stage timeout (seconds).
        board: Optional Scoreboard for task board sync (P5-T3).

    Returns:
        (ok, summary) — ok is True if all required stages completed.
    """
    wm = ctx.workflow_man
    run = wm.get_run(run_id)
    if run is None:
        sys_log.error("Workflow run %d not found", run_id)
        return False, f"Run {run_id} not found"

    if run.status == WorkflowStatus.CANCELLED:
        sys_log.warning("Workflow run %d is cancelled, will not execute", run_id)
        return False, "Run was cancelled"

    config_path_abs = str(Path(config_path).resolve())
    output_dir = _read_output_dir(config_path_abs)

    profile = _load_profile(run)
    stages = get_stages(profile)
    task_map: dict[str, int] = {}

    sys_log.info(
        "Orchestrator starting run %d with %d stages, output=%s",
        run_id, len(stages), output_dir,
    )

    run.status = WorkflowStatus.RUNNING
    run.current_stage = ""
    run.updated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    _console(ctx, f"DG-TFT workflow run #{run_id} starting [bright_black]({len(stages)} stages)[/bright_black]")
    _console(ctx, f"  data dir  : [bright_black]{run.dataset_dir}[/bright_black]")
    _console(ctx, f"  output dir: [bright_black]{output_dir}[/bright_black]")
    t_start = time.monotonic()

    # ── Create task board entries ──
    task_map = _create_stage_tasks(board, ctx.agent_id, stages, run_id)

    idx = 0
    failed_stage: str = ""
    failed_error: str = ""
    for sd in stages:
        idx += 1

        if _noop_stage(sd):
            sys_log.info("Stage [%s] is a no-op placeholder, skipping", sd.name)
            _resolve_task(board, ctx.agent_id, task_map.get(sd.name), completed=True)
            continue

        if _stage_done(run, sd.name):
            sys_log.info("Stage [%s] already completed, skipping (resume)", sd.name)
            _console(ctx, f"  [{MAJOR_COLOR2}]{idx}/{len(stages)}[/{MAJOR_COLOR2}] {sd.name} [bright_black]skipped (already done)[/bright_black]")
            _resolve_task(board, ctx.agent_id, task_map.get(sd.name), completed=True)
            continue

        run.current_stage = sd.name
        wm.update_stage(run_id, sd.name, WorkflowStatus.RUNNING)
        sys_log.info("--- Stage [%s] starting ---", sd.name)
        _console(ctx, f"  [{MAJOR_COLOR2}]{idx}/{len(stages)}[/{MAJOR_COLOR2}] {sd.name} [bright_black]starting...[/bright_black]")
        _start_task(board, ctx.agent_id, task_map.get(sd.name))

        timeout = stage_timeout or sd.timeout

        if sd.skip_if_profile_false:
            field_val = getattr(profile, sd.skip_if_profile_false, None)
            if field_val is not None and not field_val:
                sr = StageResult(
                    ok=True, stage=sd.name, skipped=True,
                    skip_reason=f"profile.{sd.skip_if_profile_false} is False",
                    summary=f"Stage {sd.name} skipped: no data available",
                )
                wm.update_stage(run_id, sd.name, WorkflowStatus.RUNNING, sr)
                sys_log.info("Stage [%s] skipped: %s", sd.name, sr.skip_reason)
                _console(ctx, f"  [{MAJOR_COLOR2}]{idx}/{len(stages)}[/{MAJOR_COLOR2}] {sd.name} [bright_black]skipped ({sr.skip_reason})[/bright_black]")
                _resolve_task(board, ctx.agent_id, task_map.get(sd.name), completed=True)
                run = wm.get_run(run_id)
                continue

        sr = run_stage(
            ctx, sd.action, config_path_abs,
            cwd=str(Path(config_path_abs).parent),
            timeout=timeout,
            extra_args=sd.extra_args,
        )

        if not sr.ok:
            wm.update_stage(run_id, sd.name, WorkflowStatus.FAILED, sr)
            _resolve_task(board, ctx.agent_id, task_map.get(sd.name), completed=False)
            if sd.required:
                wm.mark_failed(run_id, f"Required stage {sd.name} failed: {sr.error}")
                sys_log.error(
                    "Run %d failed at required stage [%s]: %s",
                    run_id, sd.name, sr.error,
                )
                _console(ctx, f"  [{MAJOR_COLOR2}]{idx}/{len(stages)}[/{MAJOR_COLOR2}] {sd.name} [{WF_STATUS_ERROR}]failed[/{WF_STATUS_ERROR}] [bright_black]({sr.error})[/bright_black]")
                failed_stage = sd.name
                failed_error = sr.error or "unknown error"
                _run_stage_diagnosis(ctx, run_id, sr, output_dir)
                _finalize_run(ctx, run_id, output_dir, t_start, stages,
                              failed=True, failed_stage=failed_stage, failed_error=failed_error)
                return False, f"Failed at {sd.name}: {sr.error}"
            else:
                wm.add_warning(run_id, f"Optional stage {sd.name} failed: {sr.error}")
                sys_log.warning(
                    "Optional stage [%s] failed for run %d, continuing", sd.name, run_id,
                )
                _console(ctx, f"  [{MAJOR_COLOR2}]{idx}/{len(stages)}[/{MAJOR_COLOR2}] {sd.name} [{WF_STATUS_WARN}]failed (non-critical, continuing)[/{WF_STATUS_WARN}]")
                run = wm.get_run(run_id)
                continue

        missing = _verify_artifacts(sd, output_dir)
        if missing:
            wm.add_warning(
                run_id,
                f"Stage {sd.name} reported success but missing artifacts: {missing}",
            )
            sys_log.warning("Stage [%s] succeeded but missing artifacts: %s", sd.name, missing)

        wm.update_stage(run_id, sd.name, WorkflowStatus.RUNNING, sr)
        for p in sd.key_artifacts:
            full_path = Path(output_dir) / sd.expected_output_subdir / p if sd.expected_output_subdir else Path(output_dir) / p
            if full_path.exists():
                wm.attach_artifact(run_id, p, str(full_path))

        sys_log.info("--- Stage [%s] completed successfully ---", sd.name)
        _console(ctx, f"  [{MAJOR_COLOR2}]{idx}/{len(stages)}[/{MAJOR_COLOR2}] {sd.name} [{WF_STATUS_SUCCESS}]✓[/{WF_STATUS_SUCCESS}]")
        _resolve_task(board, ctx.agent_id, task_map.get(sd.name), completed=True)
        run = wm.get_run(run_id)

    # All stages completed
    _finalize_run(ctx, run_id, output_dir, t_start, stages, failed=False)
    return True, f"Run {run_id} completed: {len(stages)} stages"


# ── Finalization (report + console + persistence) ──────────────────

def _finalize_run(
    ctx: AgentContext,
    run_id: int,
    output_dir: str,
    t_start: float,
    stages: list[StageDef],
    *,
    failed: bool = False,
    failed_stage: str = "",
    failed_error: str = "",
) -> None:
    """Generate reports, mark run status, and print console summary."""
    wm = ctx.workflow_man
    elapsed = time.monotonic() - t_start

    run = wm.get_run(run_id)
    if run is None:
        return

    # Build reports from whatever stage data is available
    prep_info = _get_prep_info(ctx, run.prep_id)
    run_data = run.to_dict()
    build_and_write_reports(run_data, output_dir, prep_info=prep_info)

    if failed:
        wm.mark_failed(run_id, failed_error)
        sys_log.error("Workflow run %d failed at %s", run_id, failed_stage)
        _console(ctx, f"[{WF_STATUS_ERROR}]Run #{run_id} FAILED at {STAGE_DISPLAY_NAMES.get(failed_stage, failed_stage)}[/{WF_STATUS_ERROR}]"
                 f" [bright_black]({elapsed:.0f}s)[/bright_black]")
        _console_summary(ctx, run_id, output_dir, failed=True, failed_stage=failed_stage, failed_error=failed_error)
    else:
        wm.mark_done(run_id)
        sys_log.info("Workflow run %d completed successfully", run_id)
        _console(ctx, f"[{WF_STATUS_SUCCESS}]Run #{run_id} DONE[/{WF_STATUS_SUCCESS}] [bright_black]({elapsed:.0f}s)[/bright_black]")
        _console_summary(ctx, run_id, output_dir, failed=False)


# ── Task board helpers ─────────────────────────────────────────────

def _create_stage_tasks(
    board,
    agent_id: str,
    stages: list[StageDef],
    run_id: int,
) -> dict[str, int]:
    """Create scoreboard tasks for each stage. Returns {stage_name: task_id}."""
    if board is None:
        return {}

    task_map: dict[str, int] = {}
    for sd in stages:
        if _noop_stage(sd):
            continue
        subject = _STAGE_TASK_SUBJECTS.get(sd.name, sd.name)
        desc = _STAGE_TASK_DESCS.get(sd.name, "")
        ok, msg = board.create_task(
            f"{_WF_TASK_PREFIX} [{run_id}] {subject}",
            desc,
        )
        if ok:
            m = re.search(r"\d+", msg)
            if m:
                task_map[sd.name] = int(m.group())
    return task_map


def _start_task(board, agent_id: str, task_id: int | None) -> None:
    """Mark a task as in_progress on the scoreboard."""
    if task_id is None or board is None:
        return
    from src.tool.scoreboard import TaskStatus, TaskUpdate
    update = TaskUpdate(
        task_id=task_id,
        if_claim=True,
        requester=agent_id,
        subject=None,
        description=None,
        status=TaskStatus.IN_PROGRESS,
        add_blocks=None,
        add_blocked_by=None,
    )
    board.update_task(update)


def _resolve_task(board, agent_id: str, task_id: int | None, *, completed: bool) -> None:
    """Mark a task as completed (or deleted on failure) on the scoreboard."""
    if task_id is None or board is None:
        return
    from src.tool.scoreboard import TaskStatus, TaskUpdate
    status = TaskStatus.COMPLETED if completed else TaskStatus.DELETED
    update = TaskUpdate(
        task_id=task_id,
        if_claim=True,
        requester=agent_id,
        subject=None,
        description=None,
        status=status,
        add_blocks=None,
        add_blocked_by=None,
    )
    board.update_task(update)


# ── Console summary ────────────────────────────────────────────────

def _console_summary(
    ctx: AgentContext,
    run_id: int,
    output_dir: str,
    *,
    failed: bool = False,
    failed_stage: str = "",
    failed_error: str = "",
) -> None:
    """Print a user-friendly final summary to the console (P5-T4)."""
    if getattr(ctx, "subagent_mute", False):
        return
    c = getattr(ctx, "console", None)
    if c is None:
        return

    wm = ctx.workflow_man
    run = wm.get_run(run_id)
    if run is None:
        return

    reports = get_report_paths(output_dir)
    c.print("")
    c.print(f"[{WF_SECTION_TITLE}]DG-TFT Run Summary[/{WF_SECTION_TITLE}]")
    if failed:
        c.print(f"  [{WF_STATUS_ERROR}]Status   :[/{WF_STATUS_ERROR}] FAILED at {STAGE_DISPLAY_NAMES.get(failed_stage, failed_stage)}")
        c.print(f"  Reason   : [{WF_STATUS_ERROR}]{failed_error[:120]}[/{WF_STATUS_ERROR}]")
    else:
        c.print(f"  [{WF_STATUS_SUCCESS}]Status   :[/{WF_STATUS_SUCCESS}] COMPLETED")

    c.print(f"  Run ID   : {run_id}")
    c.print(f"  Output   : [bright_black]{output_dir}[/bright_black]")
    c.print("  Reports  :")
    for key, label in [("md", "Markdown"), ("json", "JSON"), ("manifest", "Manifest")]:
        p = Path(reports[key])
        if p.exists():
            c.print(f"    [{WF_STATUS_SUCCESS}]✓[/{WF_STATUS_SUCCESS}] {label}: [bright_black]{p.name}[/bright_black]")
        else:
            c.print(f"    [{WF_MISSING_ITEM}]✗[/{WF_MISSING_ITEM}] {label}: (missing)")

    # Key artifacts summary
    if run.artifacts:
        c.print("  Key outputs:")
        for art_key, art_path in run.artifacts.items():
            c.print(f"    [bright_black]{art_key}:[/bright_black] {art_path}")

    # Diagnosis
    if run.extra.get("diagnosis"):
        diag = run.extra["diagnosis"]
        cat = diag.get("category", "UNKNOWN")
        conf = diag.get("confidence", 0)
        evidence = diag.get("evidence", [])
        suggestion = diag.get("suggestion", "")
        subagent_text = diag.get("subagent_result", "")
        c.print(f"  Diagnosis : [{WF_STATUS_WARN}]{cat}[/{WF_STATUS_WARN}] (confidence {conf:.0%})")
        if evidence:
            c.print(f"  Evidence  : [bright_black]{', '.join(evidence[:3])}[/bright_black]")
        if suggestion:
            c.print(f"  Suggestion: [bright_black]{suggestion}[/bright_black]")
        if subagent_text:
            c.print("")
            c.print(f"  [{WF_SECTION_TITLE}]AI Diagnosis[/{WF_SECTION_TITLE}]")
            for line in subagent_text.strip().splitlines()[:10]:
                c.print(f"  [bright_black]{line[:120]}[/bright_black]")

    # Recommendations
    if failed:
        c.print("")
        c.print(f"  [{WF_STATUS_WARN}]Next steps:[/{WF_STATUS_WARN}]")
        _print_failure_recommendation(c, failed_stage)
    c.print("")


def _run_stage_diagnosis(
    ctx: AgentContext,
    run_id: int,
    stage_result: StageResult,
    output_dir: str,
) -> None:
    """Classify a stage failure and optionally invoke an explorer subagent
    for a natural-language diagnosis.

    The classification result is always stored in ``run.extra["diagnosis"]``.
    If ``DG_TFT_ENABLE_DIAGNOSIS`` is true in agent_configs, an explorer
    subagent is launched with the classified hook prompt to read stage output
    files and produce a human-readable diagnosis.

    Failures inside the subagent call are caught and logged — they never
    propagate to the caller.
    """
    wm = ctx.workflow_man
    run = wm.get_run(run_id)
    if run is None:
        return

    # 1 — Classification (always runs, no LLM)
    diag = classify_from_result(stage_result, exit_code=1)
    run.extra["diagnosis"] = diag.to_dict()
    sys_log.info(
        "Stage [%s] classified as %s (confidence %.0f%%)",
        stage_result.stage, diag.category.value, diag.confidence * 100,
    )

    # 2 — Subagent call (opt-in)
    enabled = ctx.agent_configs.get(_DIAGNOSIS_CONFIG_KEY, True)
    if not enabled:
        sys_log.info("Diagnosis subagent disabled via %s", _DIAGNOSIS_CONFIG_KEY)
        return

    hook = get_hook(diag.category)
    if hook is None:
        sys_log.info("No diagnostic hook for category %s", diag.category.value)
        return

    prompt = build_diagnosis_prompt(
        hook,
        stage_name=stage_result.stage,
        category=diag.category,
        confidence=diag.confidence,
        evidence=diag.evidence,
        error_message=stage_result.error or "",
        log_snippet=_collect_log_snippet(output_dir, stage_result.stage),
    )

    console = getattr(ctx, "console", None)
    muted = getattr(ctx, "subagent_mute", False)
    if console and not muted:
        console.print(
            f"  [bright_black]Diagnosing [/{diag.category.value}][bright_black] "
            f"via {hook.subagent_type} subagent ...[/bright_black]"
        )

    try:
        diagnosis_text = _launch_diagnosis_subagent(ctx, hook, prompt, stage_result.stage)
    except Exception:
        sys_log.exception("Diagnosis subagent failed for stage [%s]", stage_result.stage)
        diagnosis_text = (
            f"Diagnosis subagent could not complete. "
            f"Classification: {diag.category.value} (confidence {diag.confidence:.0%}). "
            f"Suggestion: {diag.suggestion}"
        )

    run.extra["diagnosis"]["subagent_result"] = diagnosis_text
    sys_log.info("Diagnosis for stage [%s] complete", stage_result.stage)

    if console and not muted:
        _print_diagnosis(console, diagnosis_text)


def _launch_diagnosis_subagent(
    ctx: AgentContext,
    hook,
    prompt: str,
    stage_name: str,
) -> str:
    """Launch a short-lived explorer subagent to diagnose a stage failure.

    The subagent has only read_file and glob_file tools so it cannot
    modify any files.
    """
    from src.agent.subagent import SubAgent

    agent_id = f"{_DIAGNOSIS_SUBAGENT_ID_PREFIX}_{stage_name}_{os.urandom(4).hex()}"
    agent = SubAgent(
        parent_ctx=ctx,
        subagent_type=EXPLORER_AGENT_LABEL,
        prompt=prompt,
        agent_id=agent_id,
        subject=hook.subject,
        console=None,
    )

    max_steps = ctx.agent_configs.get(
        "DG_TFT_DIAGNOSIS_MAX_STEPS", _DIAGNOSIS_SUBAGENT_MAX_STEPS,
    )
    if isinstance(max_steps, int):
        agent.max_steps = max_steps

    timeout = ctx.agent_configs.get(
        "DG_TFT_DIAGNOSIS_TIMEOUT_S", _DIAGNOSIS_SUBAGENT_TIMEOUT_S,
    )
    if isinstance(timeout, int):
        agent.timeout_s = timeout

    ctx.agent_list[agent_id] = agent.progress
    agent.build_tools()
    agent.build_messages()
    try:
        result = agent.run()
    finally:
        ctx.agent_list.pop(agent_id, None)

    if agent.status.value == "done" and result:
        return result
    err = agent.error or "unknown error"
    raise RuntimeError(f"Diagnosis subagent failed: {err}")


def _collect_log_snippet(output_dir: str, stage_name: str) -> str:
    """Collect a short log snippet from the stage output directory.

    Looks for stdout/stderr log files or any .log/.txt files
    and returns the first few lines.
    """
    base = Path(output_dir)
    candidates = list(base.glob(f"*{stage_name}*.log")) + list(base.glob("*.log")) + list(base.glob("*.txt"))
    for candidate in candidates[:3]:
        try:
            text = candidate.read_text(encoding="utf-8", errors="replace")
            lines = text.splitlines()[:60]
            return "\n".join(lines)
        except Exception:
            continue
    return ""


def _print_diagnosis(console, diagnosis_text: str) -> None:
    """Print a formatted diagnosis block to the console."""
    from textwrap import shorten

    console.print("  [bright_black]── Diagnosis ──[/bright_black]")
    for line in diagnosis_text.strip().splitlines()[:15]:
        shortened = shorten(line, width=120, placeholder="...")
        console.print(f"  [bright_black]{shortened}[/bright_black]")


def _print_failure_recommendation(c, stage_name: str) -> None:
    """Print stage-specific troubleshooting hint."""
    hints = {
        StageName.CHECK_DATA: "Verify the dataset directory contains valid *_train.xlsx files with required sheets.",
        StageName.FIT_PHYSICAL: "Check the physical model fitting inputs — may indicate incompatible data range.",
        StageName.TRAIN_STATIC: "Try adjusting static training hyperparameters or checking for NaN/inf in data.",
        StageName.TRAIN_HYSTERESIS: "Verify I-t data availability and hysteresis configuration settings.",
    }
    hint = hints.get(stage_name, "Check the stage logs for detailed error information.")
    c.print(f"    → {hint}")


# ── Utility helpers ────────────────────────────────────────────────

def _read_output_dir(config_path: str) -> str:
    cfg = configparser.ConfigParser()
    cfg.read(config_path, encoding="utf-8")
    raw = cfg.get("paths", "output_dir", fallback="").strip()
    if raw:
        return raw
    return str(Path(config_path).parent.resolve())


def _load_profile(run) -> DatasetProfile:
    raw = {}
    if run.extra and "dataset_profile" in run.extra:
        raw = run.extra["dataset_profile"]
    elif run.extra and "raw_dataset_profile" in run.extra:
        raw = run.extra["raw_dataset_profile"]
    if isinstance(raw, DatasetProfile):
        return raw
    return DatasetProfile(**raw) if raw else DatasetProfile()


def _stage_done(run, stage_name: str) -> bool:
    sr = run.stage_results.get(stage_name)
    if sr is None:
        return False
    return sr.ok and not sr.skipped


def _verify_artifacts(sd: StageDef, output_dir: str) -> list[str]:
    if not sd.key_artifacts:
        return []
    base = Path(output_dir)
    if sd.expected_output_subdir:
        base = base / sd.expected_output_subdir
    missing = []
    for fname in sd.key_artifacts:
        if not (base / fname).exists():
            missing.append(str(base / fname))
    return missing


def _noop_stage(sd: StageDef) -> bool:
    return sd.name == StageName.FINAL_REPORT


def _get_prep_info(ctx: AgentContext, prep_id: int | None) -> dict | None:
    """Look up data preparation info from the preparation tracker."""
    if prep_id is None:
        return None
    tracker = ctx.workflow_man.preparation_tracker
    prep = tracker.get_prep(prep_id)
    if prep is None:
        return None
    return {
        "ok": prep.get("ok", False),
        "issues": prep.get("issues", []),
        "output_dir": prep.get("output_dir", ""),
        "raw_dataset_dir": prep.get("raw_dataset_dir", ""),
    }


def _console(ctx: AgentContext, message: str) -> None:
    if getattr(ctx, "subagent_mute", False):
        return
    c = getattr(ctx, "console", None)
    if c is not None:
        c.print(message)
