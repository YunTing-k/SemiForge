# -*- coding: utf-8 -*-
"""
Header information
---------
Shanghai Jiao Tong University, School of Integrated Circuits, SMIL Lab
Author: Yu Huang
Create Date: 2026.6.25
Description: DG-TFT workflow orchestrator — sequential stage execution with
             fail-fast, conditional skip, artifact verification, resume,
              report generation, task board sync, and console summary.

Revision:
---------
2026.6.25      Yu Huang      1.0      Initial implementation
2026.6.25      Yu Huang      1.1      Phase 5: report builder, task board, console summary
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
from typing import TYPE_CHECKING, Any

from .schema import (
    WorkflowStatus,
    StageResult,
    StageName,
    DatasetProfile,
)
from .stages import get_stages, get_stage_by_name, StageDef
from .stage_runner import run_stage
from .artifacts import STAGE_DISPLAY_NAMES, get_report_paths
from .report_builder import build_and_write_reports
from .metrics_parser import parse_metrics, get_primary_metric
from .quality_guard import (
    assess_quality,
    get_quality_thresholds,
    get_max_retries as get_quality_max_retries,
    stage_display_label,
    validate_agent_adjustments,
    apply_agent_adjustments,
    snapshot_ini_section,
)
from .operation_log import OperationLog, StageAttemptLog, AttemptTrigger
from src.constants import (
    MAJOR_COLOR2,
    WF_STATUS_ERROR,
    WF_STATUS_SUCCESS,
    WF_STATUS_WARN,
    WF_MISSING_ITEM,
    WF_SECTION_TITLE,
)

if TYPE_CHECKING:
    from src.context.agent_context import AgentContext

sys_log = logging.getLogger("logger")

# ── Constants ──────────────────────────────────────────────────────

_WF_TASK_PREFIX: str = "[DG-TFT]"

_STAGE_TASK_SUBJECTS: dict[str, str] = {
    StageName.DATA_PREP: "Prepare dataset",
    StageName.VERIFIER: "Verify dataset",
    StageName.CHECK_DATA: "Audit dataset",
    StageName.FIT_PHYSICAL: "Fit physical kernel",
    StageName.TRAIN_STATIC: "Train static model",
    StageName.TRAIN_HYSTERESIS: "Train hysteresis",
}

_STAGE_STEP_INDEX: dict[str, int] = {
    StageName.DATA_PREP: 1,
    StageName.VERIFIER: 2,
    StageName.CHECK_DATA: 3,
    StageName.FIT_PHYSICAL: 4,
    StageName.TRAIN_STATIC: 5,
    StageName.TRAIN_HYSTERESIS: 6,
    StageName.FINAL_REPORT: 7,
}

_STAGE_TASK_DESCS: dict[str, str] = {
    StageName.DATA_PREP: "Process raw measurement data into standardized *_train.xlsx format.",
    StageName.VERIFIER: "Verify prepared dataset structure, sheet names, and column requirements.",
    StageName.CHECK_DATA: "Validate dataset structure and inspect IV/CV/I-t profiles.",
    StageName.FIT_PHYSICAL: "Fit physical equations to extract initial parameter kernels.",
    StageName.TRAIN_STATIC: "Train the full static model and export Verilog-A.",
    StageName.TRAIN_HYSTERESIS: "Train hysteresis-aware dynamic extension and export dynamic VA.",
}

_STAGE_HAS_MODELING_AGENT: frozenset[str] = frozenset({
    StageName.FIT_PHYSICAL,
    StageName.TRAIN_STATIC,
    StageName.TRAIN_HYSTERESIS,
})

_UPSTREAM_STAGE: dict[str, str] = {
    StageName.CHECK_DATA: StageName.VERIFIER,
    StageName.FIT_PHYSICAL: StageName.CHECK_DATA,
    StageName.TRAIN_STATIC: StageName.FIT_PHYSICAL,
    StageName.TRAIN_HYSTERESIS: StageName.TRAIN_STATIC,
}

_AGENT_LABELS: dict[str, str] = {
    StageName.CHECK_DATA: "Agent A",
    StageName.FIT_PHYSICAL: "Agent B",
    StageName.TRAIN_STATIC: "Agent C",
    StageName.TRAIN_HYSTERESIS: "Agent D",
}


class OrchestratorResult(str, Enum):
    DONE = "DONE"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


def run_workflow(
    ctx: AgentContext,
    run_id: int,
    *,
    entry: str = "full",
    dataset_dir: str = "",
    raw_dataset_dir: str = "",
    hysteresis_dir: str | None = None,
    device_name: str = "",
    overrides: dict | None = None,
    preferences: str = "",
    config_path: str = "",
    board=None,
) -> tuple[bool, str]:
    """Execute the DG-TFT modeling pipeline using a while+pointer state machine.

    Stages are executed in order.  Agent stages (Preparer, Verifier) spawn
    SubAgents; script stages (check-data through final-report) run subprocess
    scripts.  On hard failure at a script stage, Agent A (check-data) may
    trigger a fallback chain back to the Verifier or Preparer.

    Args:
        ctx: Agent context.
        run_id: Workflow run ID.
        entry: "full" (Preparer→Verifier→check-data→...), "verifier"
               (Verifier→check-data→...), or "check-data" directly.
        dataset_dir: Input dataset directory.
        raw_dataset_dir: Raw data path (for "full" entry or fallback).
        hysteresis_dir: Optional I-t hysteresis data path.
        device_name: Optional device identifier.
        overrides: Optional INI overrides dict.
        preferences: User preferences string.
        board: Optional Scoreboard for task board sync.

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

    # ── Build stage list based on entry mode ──
    stages = get_stages(profile=None, entry=entry)

    # ── Determine output root ──
    from src.constants import SESSION_PATH, DG_TFT_RUNS_SUBDIR
    if config_path:
        output_dir = _read_output_dir(config_path)
    else:
        runs_subdir = ctx.agent_configs.get("DG_TFT_RUNS_SUBDIR", DG_TFT_RUNS_SUBDIR)
        output_dir = os.path.abspath(os.path.join(SESSION_PATH, ctx.session_uuid, runs_subdir, f"run_{run_id:04d}"))
    os.makedirs(output_dir, exist_ok=True)

    # ── Operation log ──
    op_log = OperationLog(output_dir)
    op_log.load()
    if not op_log.data:
        op_log.initialize(run_id, preferences)

    config_path: str = config_path  # keep param (or "" if not provided)
    profile: DatasetProfile | None = _load_profile(run) if config_path else None
    prep_output_dir: str = dataset_dir
    fallback_ctx: dict[str, Any] | None = None
    fallback_depth: int = 0

    sys_log.info(
        "Orchestrator starting run %d (entry=%s) with %d stages, output=%s",
        run_id, entry, len(stages), output_dir,
    )

    run.status = WorkflowStatus.RUNNING
    run.current_stage = ""
    run.updated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    _console(ctx, f"DG-TFT workflow run #{run_id} starting [bright_black](entry={entry}, {len(stages)} stages)[/bright_black]")
    _console(ctx, f"  data dir  : [bright_black]{dataset_dir}[/bright_black]")
    _console(ctx, f"  output dir: [bright_black]{output_dir}[/bright_black]")
    t_start = time.monotonic()

    run_timeout = _compute_run_timeout(ctx, stages)
    _console(ctx, f"  timeouts  : [bright_black]per-stage={{{', '.join(f'{s.name}:{s.timeout}s' for s in stages if s.timeout)}}}, run={run_timeout}s[/bright_black]")

    # ── Create task board entries ──
    task_map = _create_stage_tasks(board, ctx.agent_id, stages, run_id)

    # ── While+pointer state machine ──
    pointer = 0
    max_pointer = len(stages)
    total_stages = len(stages)
    failed_stage: str = ""
    failed_error: str = ""
    retry_counts: dict[str, int] = {}
    prep_max_retries = ctx.agent_configs.get("DG_TFT_PREP_MAX_RETRIES", 3)

    while pointer < max_pointer:
        sd = stages[pointer]
        display_idx = pointer + 1

        # ── Overall run timeout check ──
        if _elapsed_exceeds(t_start, run_timeout):
            _console(ctx, f"  [{WF_STATUS_ERROR}]Run #{run_id} timed out after {run_timeout}s[/{WF_STATUS_ERROR}]")
            failed_stage = sd.name
            failed_error = f"Run overall timeout ({run_timeout}s) exceeded before stage {sd.name}"
            wm.mark_failed(run_id, failed_error)
            op_log.set_final_status("failed")
            _finalize_run(ctx, run_id, output_dir, t_start, stages,
                          failed=True, failed_stage=failed_stage, failed_error=failed_error,
                          board=board, task_map=task_map)
            _spawn_agent_e(ctx, run_id, stages, op_log, output_dir, preferences, failed=True, failed_stage=failed_stage, config_path=config_path)
            return False, failed_error

        # ── No-op stage check ──
        if _noop_stage(sd):
            sys_log.info("Stage [%s] is a no-op placeholder, skipping", sd.name)
            _console(ctx, f"  [{MAJOR_COLOR2}]{display_idx}/{total_stages}[/{MAJOR_COLOR2}] {sd.name} [{WF_STATUS_SUCCESS}]✓[/{WF_STATUS_SUCCESS}] [bright_black](report generation handled by orchestrator)[/bright_black]")
            _resolve_task(board, ctx.agent_id, task_map.get(sd.name), completed=True)
            pointer += 1
            continue

        # ── Resume check ──
        if _stage_done(run, sd.name):
            sys_log.info("Stage [%s] already completed, skipping (resume)", sd.name)
            _console(ctx, f"  [{MAJOR_COLOR2}]{display_idx}/{total_stages}[/{MAJOR_COLOR2}] {sd.name} [bright_black]skipped (already done)[/bright_black]")
            _resolve_task(board, ctx.agent_id, task_map.get(sd.name), completed=True)
            # If we reached here via fallback and the upstream re-run succeeded,
            # consume the fallback context — the downstream stage is about to re-run
            # and should not see its own previous fallback as "why it was re-run"
            pointer += 1
            fallback_ctx = None
            fallback_depth = 0
            continue

        # ── Announce stage ──
        run.current_stage = sd.name
        wm.update_stage(run_id, sd.name, WorkflowStatus.RUNNING)
        sys_log.info("--- Stage [%s] starting ---", sd.name)
        _console(ctx, f"  [{MAJOR_COLOR2}]{display_idx}/{total_stages}[/{MAJOR_COLOR2}] {sd.name} [bright_black]starting...[/bright_black]")
        _start_task(board, ctx.agent_id, task_map.get(sd.name))

        # ── Execute stage ──
        if sd.kind == "agent":
            attempt_dir, agent_ok = _run_agent_stage(ctx, run_id, sd, output_dir, dataset_dir,
                                                     raw_dataset_dir, preferences,
                                                     op_log, fallback_ctx, retry_counts, prep_max_retries)

            if not agent_ok:
                retry_key = f"{sd.name}_retries"
                current = retry_counts.get(retry_key, 0)
                max_r = prep_max_retries if sd.name == StageName.DATA_PREP else prep_max_retries
                if current < max_r:
                    if sd.name == StageName.VERIFIER:
                        _console(ctx, f"  [{WF_STATUS_WARN}]Verifier failed → back to Preparer (attempt {current}/{max_r})[/{WF_STATUS_WARN}]")
                        prep_idx = _find_stage_pointer(stages, StageName.DATA_PREP)
                        if prep_idx < 0:
                            # Non-"full" entry (verifier/check-data): dynamically insert DATA_PREP
                            prep_sd = get_stage_by_name(StageName.DATA_PREP)
                            if prep_sd:
                                stages.insert(0, prep_sd)
                                max_pointer = len(stages)
                                total_stages = len(stages)
                                if board is not None:
                                    _add_stage_task(board, ctx.agent_id, run_id, task_map, prep_sd)
                                prep_idx = 0
                                _console(ctx, f"  [bright_black]({entry} entry: inserted DATA_PREP for fallback)[/bright_black]")
                            else:
                                _console(ctx, f"  [{WF_STATUS_ERROR}]Cannot fallback to DATA_PREP: stage definition not found[/{WF_STATUS_ERROR}]")
                                break
                        if prep_idx >= 0:
                            pointer = prep_idx
                            _revert_tasks(board, ctx.agent_id, task_map, stages, pointer)
                            wm.clear_stage_result(run_id, StageName.DATA_PREP)
                            retry_counts.pop("data-prep_retries", None)
                        fallback_ctx = None
                        fallback_depth = 0
                        run = wm.get_run(run_id)
                        continue
                    else:
                        _console(ctx, f"  [{WF_STATUS_WARN}]{sd.name} attempt {current} failed, retrying...[/{WF_STATUS_WARN}]")
                        fallback_ctx = None
                        fallback_depth = 0
                        run = wm.get_run(run_id)
                        continue
                # Exhausted — fail if required
                if sd.name == StageName.DATA_PREP:
                    _console(ctx, f"  [{WF_STATUS_ERROR}]Preparer exhausted after {max_r} attempts — pipeline cannot continue[/{WF_STATUS_ERROR}]")
                    wm.mark_failed(run_id, f"Preparer exhausted after {max_r} attempts")
                    op_log.set_final_status("failed")
                    _finalize_run(ctx, run_id, output_dir, t_start, stages,
                                  failed=True, failed_stage=sd.name, failed_error="Preparer exhausted",
                                  board=board, task_map=task_map)
                    _spawn_agent_e(ctx, run_id, stages, op_log, output_dir, preferences, failed=True, failed_stage=sd.name, config_path=config_path)
                    return False, f"Failed at {sd.name}: preparer exhausted"
                else:
                    wm.mark_failed(run_id, f"Required stage {sd.name} exhausted after {max_r} attempts")
                    op_log.set_final_status("failed")
                    _finalize_run(ctx, run_id, output_dir, t_start, stages,
                                  failed=True, failed_stage=sd.name, failed_error=f"Verifier exhausted",
                                  board=board, task_map=task_map)
                    _spawn_agent_e(ctx, run_id, stages, op_log, output_dir, preferences, failed=True, failed_stage=sd.name, config_path=config_path)
                    return False, f"Failed at {sd.name}: verifier exhausted"
            else:
                _resolve_task(board, ctx.agent_id, task_map.get(sd.name), completed=True)
                # Record agent stage result in workflow_runs for unified reporting
                agent_sr = StageResult(
                    ok=True, stage=sd.name,
                    summary=f"Completed attempt {retry_counts.get(f'{sd.name}_retries', 1)}",
                )
                wm.update_stage(run_id, sd.name, WorkflowStatus.RUNNING, agent_sr)
                # Update prep output dir after Verifier passes
                if sd.name == StageName.VERIFIER and attempt_dir:
                    prep_output_dir = attempt_dir
                _console(ctx, f"  [{MAJOR_COLOR2}]{display_idx}/{total_stages}[/{MAJOR_COLOR2}] {sd.name} [{WF_STATUS_SUCCESS}]✓[/{WF_STATUS_SUCCESS}]")
                pointer += 1
            fallback_ctx = None
            fallback_depth = 0
            run = wm.get_run(run_id)
            continue

        # ── Script stage ──
        # ── Lazy config building: build runtime config before first script stage ──
        if not config_path or not os.path.isfile(config_path):
            if config_path and not os.path.isfile(config_path):
                sys_log.warning("Config path %s not found, rebuilding", config_path)
            _console(ctx, f"  [bright_black]Building runtime config ...[/bright_black]")
            if not profile:
                from src.workflow.dg_tft.dataset_inspector import inspect_dataset
                profile, _ = inspect_dataset(ctx, prep_output_dir, hysteresis_dir=hysteresis_dir)
                wm.set_dataset_profile(run_id, profile)
            from src.workflow.dg_tft.config_builder import build_runtime_config
            build_result = build_runtime_config(ctx, profile, prep_output_dir,
                                                 run_id=run_id,
                                                 overrides=overrides)
            config_path = build_result["config_path"]
            _console(ctx, f"  [bright_black]Config: {config_path}[/bright_black]")

        timeout = _resolve_stage_timeout(ctx, sd)

        # Profile-based skip
        if sd.skip_if_profile_false and profile:
            field_val = getattr(profile, sd.skip_if_profile_false, None)
            if field_val is not None and not field_val:
                sr = StageResult(
                    ok=True, stage=sd.name, skipped=True,
                    skip_reason=f"profile.{sd.skip_if_profile_false} is False",
                    summary=f"Stage {sd.name} skipped: no data available",
                )
                wm.update_stage(run_id, sd.name, WorkflowStatus.RUNNING, sr)
                sys_log.info("Stage [%s] skipped: %s", sd.name, sr.skip_reason)
                _console(ctx, f"  [{MAJOR_COLOR2}]{display_idx}/{total_stages}[/{MAJOR_COLOR2}] {sd.name} [bright_black]skipped ({sr.skip_reason})[/bright_black]")
                _resolve_task(board, ctx.agent_id, task_map.get(sd.name), completed=True)
                run = wm.get_run(run_id)
                pointer += 1
                continue

        # ── Script stage retry loop (with agent integration) ──
        quality_max_retries = get_quality_max_retries(ctx.agent_configs)
        retry_remaining = quality_max_retries + 1  # +1 for initial run
        has_agent = _stage_has_modeling_agent(sd.name)
        hard_failure_agent_used = False

        while retry_remaining > 0:
            retry_remaining -= 1
            attempt_n = (quality_max_retries + 1) - retry_remaining

            if attempt_n > 1:
                _console(ctx, f"  [bright_black]Retrying stage {sd.name} (attempt {attempt_n})...[/bright_black]")
                _start_task(board, ctx.agent_id, task_map.get(sd.name))

            # ── Direct-agent: downstream fell back to an already-completed stage → spawn agent to review params ──
            if (attempt_n == 1 and has_agent and not hard_failure_agent_used
                    and fallback_ctx and fallback_ctx.get("direct_agent")):
                if fallback_depth > 2:
                    _console(ctx, f"  [{WF_STATUS_ERROR}]Fallback depth exceeded ({fallback_depth}) → unable to resolve[/{WF_STATUS_ERROR}]")
                    failed_stage = sd.name
                    failed_error = f"Fallback chain could not resolve after {fallback_depth} hops"
                    break
                hard_failure_agent_used = True
                fallback_depth += 1
                # Build synthetic result: stage passed but downstream requested review
                downstream_name = ""
                downstream_reason = ""
                from_stage = fallback_ctx.get("from_stage", "")
                if isinstance(fallback_ctx.get("arguments"), dict):
                    downstream_reason = fallback_ctx["arguments"].get("reason", "")[:200]
                direct_sr = StageResult(
                    ok=True, stage=sd.name,
                    error="",
                    summary=f"Downstream stage {from_stage} failed and requested upstream review.",
                )
                _console(ctx, f"  [bright_black]{_AGENT_LABELS.get(sd.name, 'Agent')} ← {from_stage or 'downstream'} fell back → reviewing upstream params...[/bright_black]")
                agent_result = _spawn_modeling_agent(
                    ctx, run_id, sd, direct_sr, output_dir, config_path, op_log,
                    trigger=AttemptTrigger.FALLBACK_RECEIVED,
                    known_failures=[f"Downstream {from_stage} failed: {downstream_reason}"] if downstream_reason else [],
                    preferences=preferences,
                    attempt_n=attempt_n,
                    fallback_ctx=fallback_ctx,
                )
                decision = _process_agent_decision(
                    ctx, run_id, sd, config_path, op_log,
                    agent_result, retry_remaining,
                )
                if decision == "retry":
                    # Agent adjusted params → clear direct_agent, re-run script in next iteration
                    fallback_ctx.pop("direct_agent", None)
                    continue
                if decision == "fallback":
                    fb_target = agent_result.get("arguments", {}).get("target", "")
                    new_pointer = _find_stage_pointer(stages, fb_target)
                    if new_pointer >= 0:
                        pointer = new_pointer
                        _revert_tasks(board, ctx.agent_id, task_map, stages, pointer)
                        if _stage_done(run, fb_target) and _stage_has_modeling_agent(fb_target):
                            wm.clear_stage_result(run_id, fb_target)
                            fallback_ctx = agent_result
                            fallback_ctx["direct_agent"] = True
                        else:
                            wm.clear_stage_result(run_id, fb_target)
                            fallback_ctx = agent_result
                        run = wm.get_run(run_id)
                        _console(ctx, f"  [{WF_STATUS_WARN}]Agent → fallback to {fb_target}[/{WF_STATUS_WARN}]")
                        break
                # accept / error → proceed to run script (params unchanged or adjusted)

            t_stage_start = time.monotonic()
            sr = run_stage(
                ctx, sd.action, config_path,
                cwd=str(Path(config_path).parent),
                timeout=timeout,
                extra_args=sd.extra_args,
                stream_console=True,
            )
            stage_duration = time.monotonic() - t_stage_start

            # ── Hard failure ──
            if not sr.ok:
                wm.update_stage(run_id, sd.name, WorkflowStatus.FAILED, sr)
                _resolve_task(board, ctx.agent_id, task_map.get(sd.name), completed=False)

                if not sd.required:
                    wm.add_warning(run_id, f"Optional stage {sd.name} failed: {sr.error}")
                    sys_log.warning("Optional stage [%s] failed for run %d, continuing", sd.name, run_id)
                    _console(ctx, f"  [{MAJOR_COLOR2}]{display_idx}/{total_stages}[/{MAJOR_COLOR2}] {sd.name} [{WF_STATUS_WARN}]failed (non-critical, continuing)[/{WF_STATUS_WARN}]")
                    break

                _console(ctx, f"  [{MAJOR_COLOR2}]{display_idx}/{total_stages}[/{MAJOR_COLOR2}] {sd.name} [{WF_STATUS_ERROR}]failed[/{WF_STATUS_ERROR}]—continuing")

                # ── Agent A: check-data hard failure → fallback_to_previous ──
                if sd.name == StageName.CHECK_DATA and not _is_minimal_run(config_path):
                    agent_a_result = _spawn_agent_a(ctx, run_id, sd, sr, output_dir, op_log, fallback_ctx=fallback_ctx)
                    if agent_a_result.get("fallback"):
                        fb_target = agent_a_result["fallback"]
                        fb_reason = agent_a_result.get("reason", "")
                        _console(ctx, f"  [{WF_STATUS_WARN}]Agent A → fallback to {fb_target}[/{WF_STATUS_WARN}] [bright_black]({fb_reason[:100]})[/bright_black]")
                        fallback_ctx = agent_a_result
                        new_pointer = _find_stage_pointer(stages, fb_target)
                        if new_pointer < 0 and fb_target == StageName.VERIFIER:
                            # "check-data" entry: dynamically insert VERIFIER
                            verify_sd = get_stage_by_name(StageName.VERIFIER)
                            if verify_sd:
                                check_idx = _find_stage_pointer(stages, StageName.CHECK_DATA)
                                insert_pos = check_idx if check_idx >= 0 else len(stages)
                                stages.insert(insert_pos, verify_sd)
                                max_pointer = len(stages)
                                total_stages = len(stages)
                                if board is not None:
                                    _add_stage_task(board, ctx.agent_id, run_id, task_map, verify_sd)
                                new_pointer = insert_pos
                                _console(ctx, f"  [bright_black]({entry} entry: inserted VERIFIER for fallback)[/bright_black]")
                        if new_pointer >= 0:
                            pointer = new_pointer
                            _revert_tasks(board, ctx.agent_id, task_map, stages, pointer)
                            run = wm.get_run(run_id)
                            break  # jump upstream
                    failed_stage = sd.name
                    failed_error = sr.error or "unknown error"
                    break

                # ── Agent B/C/D: hard failure → adjust_params or fallback (skip under minimal_run) ──
                if not _is_minimal_run(config_path) and has_agent and not hard_failure_agent_used:
                    hard_failure_agent_used = True
                    _console(ctx, f"  [bright_black]Spawning agent for {sd.name} hard failure...[/bright_black]")
                    agent_result = _spawn_modeling_agent(
                        ctx, run_id, sd, sr, output_dir, config_path, op_log,
                        trigger=AttemptTrigger.QUALITY_FAIL,
                        known_failures=[sr.error or "unknown error"],
                        preferences=preferences,
                        attempt_n=attempt_n,
                        fallback_ctx=fallback_ctx,
                    )
                    decision = _process_agent_decision(
                        ctx, run_id, sd, config_path, op_log,
                        agent_result, retry_remaining,
                    )
                    if decision == "retry":
                        continue  # re-run same stage
                    if decision == "fallback":
                        fb_target = agent_result.get("arguments", {}).get("target", "")
                        new_pointer = _find_stage_pointer(stages, fb_target)
                        if new_pointer >= 0:
                            pointer = new_pointer
                            _revert_tasks(board, ctx.agent_id, task_map, stages, pointer)
                            if _stage_done(run, fb_target) and _stage_has_modeling_agent(fb_target):
                                # Upstream already passed — clear result so retry loop enters,
                                # then spawn its agent directly to review params
                                wm.clear_stage_result(run_id, fb_target)
                                fallback_ctx = agent_result
                                fallback_ctx["direct_agent"] = True
                                _console(ctx, f"  [{WF_STATUS_WARN}]Agent → fallback to {fb_target} (direct agent review)[/{WF_STATUS_WARN}]")
                            else:
                                wm.clear_stage_result(run_id, fb_target)
                                _console(ctx, f"  [{WF_STATUS_WARN}]Agent → fallback to {fb_target}[/{WF_STATUS_WARN}]")
                            run = wm.get_run(run_id)
                            fallback_ctx = agent_result
                            break
                    # accept / error / exhausted → mark failed and move on
                    if not failed_stage:
                        failed_stage = sd.name
                    if not failed_error:
                        failed_error = sr.error or "unknown error"

                if not failed_stage:
                    failed_stage = sd.name
                if not failed_error:
                    failed_error = sr.error or "unknown error"
                wm.mark_failed(run_id, f"Required stage {sd.name} failed: {sr.error}")
                sys_log.error("Run %d failed at required stage [%s]: %s", run_id, sd.name, sr.error)
                break

            # ── Success ──
            missing = _verify_artifacts(sd, output_dir)
            if missing:
                wm.add_warning(run_id, f"Stage {sd.name} reported success but missing artifacts: {missing}")
                sys_log.warning("Stage [%s] succeeded but missing artifacts: %s", sd.name, missing)

            if attempt_n == 1:
                _log_initial_attempt(op_log, sd.name, sr, output_dir, stage_duration)

            # ── Quality check ──
            quality_ok, quality_issues = _assess_stage_quality(ctx, sd.name, output_dir, config_path)
            sr.metrics = parse_metrics(output_dir, sd.name)

            wm.update_stage(run_id, sd.name, WorkflowStatus.RUNNING, sr)
            for p in sd.key_artifacts:
                full_path = Path(output_dir) / sd.expected_output_subdir / p if sd.expected_output_subdir else Path(output_dir) / p
                if full_path.exists():
                    wm.attach_artifact(run_id, p, str(full_path))

            if quality_ok:
                sys_log.info("--- Stage [%s] completed successfully ---", sd.name)
                _console(ctx, f"  [{MAJOR_COLOR2}]{display_idx}/{total_stages}[/{MAJOR_COLOR2}] {sd.name} [{WF_STATUS_SUCCESS}]✓[/{WF_STATUS_SUCCESS}]")
                break

            # ── Quality below threshold ──
            stage_label = stage_display_label(sd.name)
            _print_quality_fail(ctx, stage_label, quality_issues)
            wm.add_warning(run_id, f"Stage {sd.name} quality below threshold: {'; '.join(quality_issues)}")
            sys_log.warning("Run %d stage [%s] completed with quality warnings: %s", run_id, sd.name, quality_issues)
            _console(ctx, f"  [{MAJOR_COLOR2}]{display_idx}/{total_stages}[/{MAJOR_COLOR2}] {sd.name} [{WF_STATUS_WARN}]✓ (quality below threshold)[/{WF_STATUS_WARN}]")

            if retry_remaining <= 0:
                _console(ctx, f"  [bright_black]Quality retries exhausted for {sd.name}[/bright_black]")
                break

            if not has_agent:
                break

            # ── Agent B/C/D: quality fail → adjust_params / fallback / accept_quality ──
            _console(ctx, f"  [bright_black]Spawning agent for {sd.name} quality improvement...[/bright_black]")
            agent_result = _spawn_modeling_agent(
                ctx, run_id, sd, sr, output_dir, config_path, op_log,
                trigger=AttemptTrigger.QUALITY_FAIL,
                known_failures=quality_issues,
                preferences=preferences,
                attempt_n=attempt_n,
                fallback_ctx=fallback_ctx,
            )
            decision = _process_agent_decision(
                ctx, run_id, sd, config_path, op_log,
                agent_result, retry_remaining,
            )
            if decision == "retry":
                continue
            if decision == "fallback":
                fb_target = agent_result.get("arguments", {}).get("target", "")
                new_pointer = _find_stage_pointer(stages, fb_target)
                if new_pointer >= 0:
                    pointer = new_pointer
                    _revert_tasks(board, ctx.agent_id, task_map, stages, pointer)
                    if _stage_done(run, fb_target) and _stage_has_modeling_agent(fb_target):
                        wm.clear_stage_result(run_id, fb_target)
                        fallback_ctx = agent_result
                        fallback_ctx["direct_agent"] = True
                        _console(ctx, f"  [{WF_STATUS_WARN}]Agent → fallback to {fb_target} (direct agent review)[/{WF_STATUS_WARN}]")
                    else:
                        wm.clear_stage_result(run_id, fb_target)
                        _console(ctx, f"  [{WF_STATUS_WARN}]Agent → fallback to {fb_target}[/{WF_STATUS_WARN}]")
                    run = wm.get_run(run_id)
                    fallback_ctx = agent_result
                    break
            # accept / error → move on
            break

        # If we exited the retry loop due to fallback, skip pointer advance
        if fallback_ctx is not None:
            _console(ctx, f"  [bright_black]Fallback active → re-entering upstream stage[/bright_black]")
            # If the fallback target stage also hard-failed and the fallback chain
            # is exhausted (e.g. Agent A refused to fallback), abort — don't loop forever.
            if failed_stage:
                _console(ctx, f"  [{WF_STATUS_ERROR}]Fallback chain exhausted at {failed_stage} — pipeline terminating[/{WF_STATUS_ERROR}]")
                wm.mark_failed(run_id, f"Fallback chain exhausted at {failed_stage}: {failed_error}")
                op_log.set_final_status("failed")
                _finalize_run(ctx, run_id, output_dir, t_start, stages,
                              failed=True, failed_stage=failed_stage, failed_error=failed_error or "",
                              board=board, task_map=task_map)
                _spawn_agent_e(ctx, run_id, stages, op_log, output_dir, preferences, failed=True, failed_stage=failed_stage, config_path=config_path)
                return False, f"Failed at {failed_stage}: fallback chain exhausted"
            run = wm.get_run(run_id)
            continue

        _resolve_task(board, ctx.agent_id, task_map.get(sd.name), completed=not bool(failed_stage))
        pointer += 1
        fallback_ctx = None
        run = wm.get_run(run_id)

    # All stages processed — some may have failed
    if failed_stage:
        op_log.set_final_status("failed")
        _finalize_run(ctx, run_id, output_dir, t_start, stages,
                      failed=True, failed_stage=failed_stage, failed_error=failed_error or "",
                      board=board, task_map=task_map)
        _spawn_agent_e(ctx, run_id, stages, op_log, output_dir, preferences, failed=True, failed_stage=failed_stage, config_path=config_path)
        return False, f"Run {run_id} completed with failures: {failed_stage}"
    op_log.set_final_status("done")
    _finalize_run(ctx, run_id, output_dir, t_start, stages, failed=False)
    _spawn_agent_e(ctx, run_id, stages, op_log, output_dir, preferences, failed=False, config_path=config_path)
    return True, f"Run {run_id} completed: {len(stages)} stages"


# ── Agent stage helpers ──────────────────────────────────────────────

def _run_agent_stage(
    ctx: AgentContext,
    run_id: int,
    sd: StageDef,
    output_dir: str,
    dataset_dir: str,
    raw_dataset_dir: str,
    preferences: str,
    op_log: OperationLog | None,
    fallback_ctx: dict[str, Any] | None,
    retry_counts: dict[str, int],
    max_retries: int,
) -> tuple[str, bool]:
    """Execute an agent-type stage (Preparer or Verifier).

    Returns (attempt_dir, ok).  ok=False means the stage should be retried
    if attempts remain, or the pipeline should fail if exhausted.
    """
    from src.constants import PREP_SUBDIR_NAME
    from src.workflow.dg_tft.data_preparer import (
        run_preparer_stage,
        run_verifier_stage,
        copy_dataset_to_data_prep,
    )

    wm = ctx.workflow_man
    stage_name = sd.name
    retry_key = f"{stage_name}_retries"
    current_retry = retry_counts.get(retry_key, 0) + 1
    retry_counts[retry_key] = current_retry

    attempt_n = current_retry
    attempt_dir = os.path.join(output_dir, PREP_SUBDIR_NAME, f"attempt_{attempt_n}")
    os.makedirs(attempt_dir, exist_ok=True)

    trigger = AttemptTrigger.INITIAL if attempt_n == 1 else AttemptTrigger.VERIFIER_FAIL
    if fallback_ctx and fallback_ctx.get("fallback"):
        trigger = AttemptTrigger.FALLBACK_RECEIVED

    t_stage_start = time.monotonic()

    if stage_name == StageName.DATA_PREP:
        # Build handoff history from prior attempts
        handoff_history = _build_handoff_history(op_log)
        prep_result = run_preparer_stage(
            ctx, raw_dataset_dir, attempt_dir,
            handoff_history=handoff_history,
        )
        prep_ok = prep_result.get("ok", False)
        handoff = prep_result.get("handoff")
        error = prep_result.get("error", "")

        op_log.append_attempt("data-prep", StageAttemptLog(
            n=attempt_n, trigger=trigger,
            handoff=handoff,
            ok=prep_ok,
            error=error,
            duration_s=time.monotonic() - t_stage_start,
        ))

        if not prep_ok:
            if current_retry < max_retries:
                sys_log.warning("Preparer attempt %d failed, will retry", attempt_n)
            else:
                sys_log.error("Preparer exhausted after %d attempts", max_retries)
                wm.mark_failed(run_id, error or "Preparer exhausted")
            return attempt_dir, False

        return attempt_dir, True

    elif stage_name == StageName.VERIFIER:
        # In "verifier" entry (no DATA_PREP before us), copy the standard
        # dataset into the attempt dir before verifying.
        if not op_log.get_stage_attempts("data-prep"):
            copy_dataset_to_data_prep(dataset_dir, attempt_dir)

        # Build Verifier prompt context: Preparer handoff + Agent A fallback reason (if any)
        prep_handoff = None
        prep_attempts = op_log.get_stage_attempts("data-prep")
        if prep_attempts:
            prep_handoff = prep_attempts[-1].get("handoff")

        if fallback_ctx and fallback_ctx.get("fallback"):
            agent_a_handoff = {
                "fallback_from": "Agent A (check-data)",
                "reason": fallback_ctx.get("reason", ""),
                "evidence": fallback_ctx.get("evidence", ""),
                "suggestion": fallback_ctx.get("suggestion", ""),
            }
            if prep_handoff:
                prep_handoff = dict(prep_handoff)
                prep_handoff["agent_a_fallback"] = agent_a_handoff
            else:
                prep_handoff = {"agent_a_fallback": agent_a_handoff}

        verify_result = run_verifier_stage(ctx, attempt_dir, handoff=prep_handoff)
        verify_ok = verify_result.get("ok", False)
        feedback = verify_result.get("feedback")
        issues = verify_result.get("issues", [])

        op_log.append_attempt("verifier", StageAttemptLog(
            n=attempt_n, trigger=trigger,
            agent_decision=feedback,
            ok=verify_ok,
            error="; ".join(issues) if issues else "",
            duration_s=time.monotonic() - t_stage_start,
        ))

        if not verify_ok:
            if current_retry < max_retries:
                sys_log.warning("Verifier attempt %d failed, will retry", attempt_n)
            else:
                sys_log.error("Verifier exhausted after %d attempts", max_retries)
                wm.mark_failed(run_id, "Verifier exhausted")
            return attempt_dir, False

        return attempt_dir, True

    return attempt_dir, False


def _build_handoff_history(op_log: OperationLog | None) -> str:
    """Build handoff history text from prior data-prep and verifier attempts.

    Cross-references data-prep attempts with matching verifier feedback
    by attempt number so the Preparer sees what the Verifier found on retry.
    """
    if op_log is None:
        return ""
    import json as _json
    prep_attempts = op_log.get_stage_attempts("data-prep")
    verifier_attempts = op_log.get_stage_attempts("verifier")
    verifier_by_n = {va.get("n"): va for va in verifier_attempts}
    lines: list[str] = []
    for ad in prep_attempts:
        n = ad.get("n", 0)
        h = ad.get("handoff")
        if h:
            lines.append(f"Attempt {n} preparer handoff: {_json.dumps(h, ensure_ascii=False)}")
        va = verifier_by_n.get(n)
        if va:
            fb = va.get("agent_decision")
            if fb:
                lines.append(f"Attempt {n} verifier feedback: {_json.dumps(fb, ensure_ascii=False)}")
    return "\n".join(lines)


def _spawn_agent_a(
    ctx: AgentContext,
    run_id: int,
    sd: StageDef,
    stage_result: StageResult,
    output_dir: str,
    op_log: OperationLog | None = None,
    fallback_ctx: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Spawn Agent A (check-data) to decide whether to fall back to Verifier.

    Agent A uses the dedicated ``diagnostic`` subagent type — base tools
    (glob_file, grep_file, read_file, bash) plus subagent tools (inspect_xlsx
    for investigation, fallback_to_previous for the decision). Tools, stop_mode,
    and stop_schemas are auto-resolved from the type config.

    Args:
        fallback_ctx: If Agent B/C/D fell back to check-data, contains their
                      fallback_to_previous arguments {reason, evidence, suggestion}.

    Returns:
        Dict with keys: fallback (str target or ""), reason, evidence, suggestion.
        If fallback is empty, the orchestrator treats the failure as final.
    """
    from src.agent.subagent import SubAgent
    from src.constants import DIAGNOSTIC_AGENT_LABEL
    from src.workflow.dg_tft.knowledge.loader import build_agent_prompt

    prompt_lines = [
        "You are Agent A (check-data diagnostics).",
        "The check-data stage (dataset structural validation) has failed.",
        f"Error: {stage_result.error}",
    ]
    if stage_result.summary:
        prompt_lines.append(f"Stdout summary (last lines): {stage_result.summary}")
    prompt_lines.append(f"Output directory: {output_dir}")
    prompt_lines.extend([
        "",
        "Your job: decide whether this failure can be fixed upstream by the Verifier.",
        "If the errors suggest that the prepared dataset is malformed (missing columns,",
        "wrong sheet names, corrupt files, etc.), call `fallback_to_previous` to notify",
        "the Verifier stage to re-check the data and possibly fall back to the Preparer.",
        "",
        "If the error is purely environmental (missing executables, configuration issues,",
        "permissions, etc.), do NOT fall back — the failure is final.",
        "",
        "You may use `read_file` and `glob_file` to inspect the output directory and",
        "error logs if needed, but keep it brief — you only have a few steps.",
    ])

    if op_log:
        import json as _json
        prep_attempts = op_log.get_stage_attempts("data-prep")
        verifier_attempts = op_log.get_stage_attempts("verifier")

        if prep_attempts:
            prompt_lines.append("")
            prompt_lines.append("Preparer handoff (what the worker produced):")
            last_prep = prep_attempts[-1]
            h = last_prep.get("handoff")
            if h:
                prompt_lines.append(_json.dumps(h, ensure_ascii=False, indent=2))
            else:
                prompt_lines.append(str(last_prep)[:500])

        if verifier_attempts:
            prompt_lines.append("")
            prompt_lines.append("Verifier report (structured validation results from `report_verification`):")
            last_v = verifier_attempts[-1]
            fb = last_v.get("agent_decision")
            if fb and isinstance(fb, dict):
                prompt_lines.append(_json.dumps(fb, ensure_ascii=False, indent=2))
            else:
                prompt_lines.append(str(last_v)[:500])

    if fallback_ctx and fallback_ctx.get("decision") == "fallback_to_previous":
        fb_args = fallback_ctx.get("arguments", {}) if isinstance(fallback_ctx.get("arguments"), dict) else {}
        prompt_lines.append("")
        prompt_lines.append("**Why you were called (downstream modeling agent fell back):**")
        prompt_lines.append(f"  Reason from downstream agent: {fb_args.get('reason', '')[:200]}")
        prompt_lines.append(f"  Evidence: {fb_args.get('evidence', '')[:200]}")
        prompt_lines.append(f"  Suggestion: {fb_args.get('suggestion', '')[:200]}")

    # ── Expert knowledge ──
    knowledge_block = build_agent_prompt("check-data")
    if knowledge_block:
        prompt_lines.insert(0, knowledge_block)
        prompt_lines.insert(0, "")

    prompt = "\n".join(prompt_lines)

    agent_id = f"agent_a_check_data_{run_id}_{os.urandom(4).hex()}"
    agent = SubAgent(
        parent_ctx=ctx,
        subagent_type=DIAGNOSTIC_AGENT_LABEL,
        prompt=prompt,
        agent_id=agent_id,
        subject="DG-TFT check-data diagnostic",
        console=None,
    )
    ctx.agent_list[agent_id] = agent.progress
    agent.build_tools()
    agent.build_messages()
    try:
        result = agent.run()
    finally:
        agent.progress.status = agent.status
        agent.progress.if_archived = True

    if agent.status.value != "done" or not result:
        err = agent.error or "Agent A did not produce a result"
        sys_log.warning("Agent A failed: %s", err)
        return {"fallback": "", "reason": err, "evidence": "", "suggestion": ""}

    import json as _json
    try:
        decision = _json.loads(result) if isinstance(result, str) else dict(result)
    except (_json.JSONDecodeError, TypeError, ValueError):
        sys_log.warning("Agent A returned non-JSON: %s", str(result)[:200])
        return {"fallback": "", "reason": str(result)[:200], "evidence": "", "suggestion": ""}

    # Log the decision
    if op_log:
        op_log.append_attempt(sd.name, StageAttemptLog(
            n=op_log.get_last_attempt_n(sd.name) + 1,
            trigger=AttemptTrigger.FALLBACK_SENT if decision.get("target") else AttemptTrigger.QUALITY_FAIL,
            agent_decision=decision,
            agent_reasoning=decision.get("reason", ""),
            ok=False,
            error=stage_result.error or "",
        ))

    return {
        "fallback": decision.get("target", ""),
        "reason": decision.get("reason", ""),
        "evidence": decision.get("evidence", ""),
        "suggestion": decision.get("suggestion", ""),
    }


def _find_stage_pointer(stages: list[StageDef], target_name: str) -> int:
    """Find the index of a stage by name. Returns -1 if not found."""
    for i, sd in enumerate(stages):
        if sd.name == target_name:
            return i
    return -1


def _revert_tasks(
    board,
    agent_id: str,
    task_map: dict[str, int],
    stages: list[StageDef],
    fallback_pointer: int,
) -> None:
    """Revert task board entries: mark previously completed stages as in_progress."""
    if board is None:
        return
    from src.tool.scoreboard import TaskStatus
    for i in range(fallback_pointer, len(stages)):
        sd = stages[i]
        tid = task_map.get(sd.name)
        if tid is not None:
            ok, task, _ = board.get_task(tid)
            if ok and task is not None and task.get("status") == TaskStatus.COMPLETED:
                _start_task(board, agent_id, tid)


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
    board = None,
    task_map: dict[str, int] | None = None,
) -> None:
    """Generate reports, mark run status, and print console summary."""
    wm = ctx.workflow_man
    elapsed = time.monotonic() - t_start

    run = wm.get_run(run_id)
    if run is None:
        return

    # Build reports from whatever stage data is available
    prep_info = _get_prep_info(ctx, run.prep_id)
    if prep_info is None and run.run_id is not None:
        prep_info = _get_prep_info(ctx, run.run_id)

    if failed:
        wm.mark_failed(run_id, failed_error)
    else:
        wm.mark_done(run_id)

    # Clean up remaining task board entries on failure
    if failed and board is not None and task_map:
        for sd in stages:
            tid = task_map.get(sd.name)
            if tid is not None:
                _resolve_task(board, ctx.agent_id, tid, completed=False)

    run_data = run.to_dict()
    try:
        build_and_write_reports(run_data, output_dir, prep_info=prep_info)
    except Exception as e:
        sys_log.error("Failed to build reports: %s", e)

    if failed:
        sys_log.error("Workflow run %d failed at %s", run_id, failed_stage)
        _console(ctx, f"[{WF_STATUS_ERROR}]Run #{run_id} FAILED at {STAGE_DISPLAY_NAMES.get(failed_stage, failed_stage)}[/{WF_STATUS_ERROR}]"
                 f" [bright_black]({elapsed:.0f}s)[/bright_black]")
        _console_summary(ctx, run_id, output_dir, failed=True, failed_stage=failed_stage, failed_error=failed_error)
    else:
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
        step = _STAGE_STEP_INDEX.get(sd.name, 0)
        ok, msg = board.create_task(
            f"[DG-TFT, ID: {run_id}] [Step-{step}] {subject}",
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


def _add_stage_task(
    board,
    agent_id: str,
    run_id: int,
    task_map: dict[str, int],
    sd: StageDef,
) -> None:
    """Create a single task board entry for a dynamically inserted stage."""
    if board is None:
        return
    import re as _re
    subject = _STAGE_TASK_SUBJECTS.get(sd.name, sd.name)
    desc = _STAGE_TASK_DESCS.get(sd.name, "")
    step = _STAGE_STEP_INDEX.get(sd.name, 0)
    ok, msg = board.create_task(
        f"[DG-TFT, ID: {run_id}] [Step-{step}] {subject}", desc,
    )
    if ok:
        m = _re.search(r"\d+", msg)
        if m:
            task_map[sd.name] = int(m.group())


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

    from rich.markup import escape as _rich_escape

    wm = ctx.workflow_man
    run = wm.get_run(run_id)
    if run is None:
        return

    reports = get_report_paths(output_dir)
    c.print("")
    c.print(f"[{WF_SECTION_TITLE}]DG-TFT Run Summary[/{WF_SECTION_TITLE}]")
    if failed:
        c.print(f"  [{WF_STATUS_ERROR}]Status   :[/{WF_STATUS_ERROR}] FAILED at {STAGE_DISPLAY_NAMES.get(failed_stage, failed_stage)}")
        c.print(f"  Reason   : [{WF_STATUS_ERROR}]{_rich_escape(failed_error[:120])}[/{WF_STATUS_ERROR}]")
    else:
        c.print(f"  [{WF_STATUS_SUCCESS}]Status   :[/{WF_STATUS_SUCCESS}] COMPLETED")

    c.print(f"  Run ID   : {run_id}")
    c.print(f"  Output   : [bright_black]{_rich_escape(output_dir)}[/bright_black]")
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
            c.print(f"    [bright_black]{art_key}:[/bright_black] {_rich_escape(art_path)}")

    # Recommendations
    if failed:
        c.print("")
        c.print(f"  [{WF_STATUS_WARN}]Next steps:[/{WF_STATUS_WARN}]")
        _print_failure_recommendation(c, failed_stage)
    c.print("")


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


# ── Quality gate (Phase 7) ─────────────────────────────────────────

def _log_initial_attempt(
    op_log: OperationLog | None,
    stage_name: str,
    sr: StageResult,
    output_dir: str,
    duration_s: float = 0.0,
) -> None:
    """Log attempt_1 to the operation log unconditionally.

    Called after every successful stage run, before the quality gate.
    minimal_run runs also get logged — the overhead is negligible.
    """
    if op_log is None:
        return
    metrics = parse_metrics(output_dir, stage_name)
    entry = StageAttemptLog(
        n=1,
        trigger=AttemptTrigger.INITIAL,
        metrics=dict(metrics),
        ok=sr.ok,
        error=sr.error or "",
        duration_s=duration_s,
    )
    op_log.append_attempt(stage_name, entry)

def _assess_stage_quality(
    ctx: AgentContext,
    stage_name: str,
    output_dir: str,
    config_path: str,
) -> tuple[bool, list[str]]:
    """Check quality for a stage; return (ok, failures).

    Avoids LLM entirely — pure rule-driven check.
    """
    if _is_minimal_run(config_path):
        return True, []

    thresholds = get_quality_thresholds(ctx.agent_configs)
    if not thresholds:
        return True, []

    if not get_primary_metric(stage_name):
        return True, []

    metrics = parse_metrics(output_dir, stage_name)
    return assess_quality(stage_name, metrics, thresholds)


def _stage_has_modeling_agent(stage_name: str) -> bool:
    return stage_name in _STAGE_HAS_MODELING_AGENT


def _print_quality_fail(ctx: AgentContext, label: str, failures: list[str]) -> None:
    if getattr(ctx, "subagent_mute", False):
        return
    c = getattr(ctx, "console", None)
    if c is None:
        return
    c.print(f"  [{WF_STATUS_WARN}]Quality[{WF_STATUS_WARN}]: {label} below threshold")
    for f in failures[:3]:
        c.print(f"    [bright_black]{f}[/bright_black]")


def _check_quality_once(
    ctx: AgentContext,
    run_id: int,
    sd: StageDef,
    stage_result: StageResult,
    config_path: str,
    output_dir: str,
) -> tuple[StageResult, list[str]]:
    """Backward-compat wrapper — delegates to _assess_stage_quality."""
    ok, issues = _assess_stage_quality(ctx, sd.name, output_dir, config_path)
    stage_result.metrics = parse_metrics(output_dir, sd.name)
    if not ok:
        _print_quality_fail(ctx, stage_display_label(sd.name), issues)
    return stage_result, issues if not ok else []


# ── Modeling Agent spawn ────────────────────────────────────────────

def _spawn_modeling_agent(
    ctx: AgentContext,
    run_id: int,
    sd: StageDef,
    stage_result: StageResult,
    output_dir: str,
    config_path: str,
    op_log: OperationLog | None = None,
    *,
    trigger: AttemptTrigger = AttemptTrigger.QUALITY_FAIL,
    known_failures: list[str] | None = None,
    preferences: str = "",
    attempt_n: int = 1,
    fallback_ctx: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Spawn a modeling Agent (B/C/D) to decide on parameter adjustments.

    Wraps the actual spawn in a try/except to handle mock/test contexts
    where the full SubAgent infrastructure is not available.
    """
    agent_label = _AGENT_LABELS.get(sd.name, "Agent")
    try:
        return _spawn_modeling_agent_inner(
            ctx, run_id, sd, stage_result, output_dir, config_path,
            op_log, trigger=trigger, known_failures=known_failures,
            preferences=preferences, attempt_n=attempt_n,
            fallback_ctx=fallback_ctx,
        )
    except Exception as e:
        sys_log.warning("%s spawn failed: %s", agent_label, e)
        _console(ctx, f"  [bright_black]{agent_label}: could not spawn (context may be incomplete)[/bright_black])")
        return {"decision": "error", "tool_name": "", "arguments": {}, "reasoning": str(e)}


def _spawn_modeling_agent_inner(
    ctx: AgentContext,
    run_id: int,
    sd: StageDef,
    stage_result: StageResult,
    output_dir: str,
    config_path: str,
    op_log: OperationLog | None,
    *,
    trigger: AttemptTrigger,
    known_failures: list[str] | None,
    preferences: str,
    attempt_n: int,
    fallback_ctx: dict[str, Any] | None = None,
) -> dict[str, Any]:
    from src.agent.subagent import SubAgent
    from src.tool.subagent_tools import (
        get_tools,
        SUBAGENT_TOOL_ADJUST_PARAMS,
        SUBAGENT_TOOL_FALLBACK_TO_PREVIOUS,
        SUBAGENT_TOOL_ACCEPT_QUALITY,
    )
    from src.constants import MODELING_AGENT_LABEL
    from src.workflow.dg_tft.knowledge.loader import build_agent_prompt
    from src.workflow.dg_tft.param_defs import (
        get_stage_param_schema,
        get_stage_sections,
        get_soft_guard_prompt,
    )
    import json as _json

    upstream = _UPSTREAM_STAGE.get(sd.name, "verifier")
    agent_label = _AGENT_LABELS.get(sd.name, "Agent")

    adjust_schema = get_stage_param_schema(sd.name)
    schemas, handlers = get_tools(sd.name, adjust_params_override=adjust_schema)
    stop_schemas = [SUBAGENT_TOOL_ADJUST_PARAMS, SUBAGENT_TOOL_FALLBACK_TO_PREVIOUS, SUBAGENT_TOOL_ACCEPT_QUALITY]

    def _build_prompt() -> str:
        errors = stage_result.error or "none"
        failures_text = ", ".join(known_failures) if known_failures else "none reported"
        soft_guard = get_soft_guard_prompt()

        if trigger == AttemptTrigger.FALLBACK_RECEIVED and stage_result.ok:
            status_line = "completed successfully but a downstream stage has failed and requested your review"
        else:
            status_line = "failed" if trigger == AttemptTrigger.QUALITY_FAIL and not stage_result.ok else "completed with quality below threshold"

        prompt_lines = [
            f"You are {agent_label} ({sd.name} diagnostics).",
            f"The {sd.name} stage has {status_line}.",
            f"Error: {errors}",
            f"Quality failures: {failures_text}",
            f"Attempt number: {attempt_n}",
            f"INI config path: {config_path}",
            f"Output directory: {output_dir}",
            f"Upstream stage for fallback: {upstream}",
            "",
            "Your job: analyze the stage results and decide the best course of action.",
            "",
            "Options (call EXACTLY ONE tool to submit your decision):",
            f"  1. `adjust_params` — if you believe adjusting hyperparameters can improve quality.",
            f"  2. `fallback_to_previous` — if you suspect the problem originates upstream (target: {upstream}).",
            f"  3. `accept_quality` — if the current quality is acceptable, or continued adjustments won't help.",
        ]

        # ── Prior attempts from this stage ──
        if op_log:
            prior = op_log.get_stage_attempts(sd.name)
            if prior:
                prompt_lines.append("")
                prompt_lines.append("**Prior attempts from this stage:**")
                for pa in prior[-3:]:
                    n = pa.get("n", 0)
                    adj = pa.get("adjustments") or {}
                    reason = pa.get("agent_reasoning", "")[:120]
                    trigger_info = pa.get("trigger", "")
                    parts = [f"  Attempt {n}"]
                    if trigger_info:
                        parts.append(f"(trigger={trigger_info})")
                    if adj:
                        adj_str = ", ".join(f"{k}={v}" for k, v in adj.items())
                        parts.append(f"adjustments: {adj_str}")
                    if reason:
                        parts.append(f"reasoning: {reason}")
                    prompt_lines.append(" | ".join(parts))

        # ── Upstream stage context ──
        if op_log:
            upstream_attempts = op_log.get_stage_attempts(upstream)
            if upstream_attempts:
                ua = upstream_attempts[-1]
                prompt_lines.append("")
                prompt_lines.append(f"**Last {upstream} attempt:** OK={ua.get('ok')}, "
                                    f"metrics={_json.dumps(ua.get('metrics', {}))}, "
                                    f"adjustments={_json.dumps(ua.get('adjustments', {}))}")

        # ── Downstream fallback context ──
        if fallback_ctx and fallback_ctx.get("decision") == "fallback_to_previous":
            fb_args = fallback_ctx.get("arguments", {}) if isinstance(fallback_ctx.get("arguments"), dict) else {}
            prompt_lines.append("")
            prompt_lines.append("**Why you were re-run (downstream fallback):**")
            prompt_lines.append(f"  Fallback from downstream stage, reason: {fb_args.get('reason', '')[:200]}")
            prompt_lines.append(f"  Evidence: {fb_args.get('evidence', '')[:200]}")
            prompt_lines.append(f"  Suggestion: {fb_args.get('suggestion', '')[:200]}")

        prompt_lines.extend([
            "",
            "Guidelines:",
            "- You may use `read_file` and `glob_file` to inspect output files and logs.",
            "- Make small, incremental parameter changes — extreme jumps rarely help.",
            "- If the same quality issue persists across multiple attempts, consider fallback or accept.",
            "- If user preferences deprioritize the failing metric, accept_quality is valid.",
            "- You may change as many parameters as needed — there is no limit on the number of adjustments.",
            "",
            soft_guard,
        ])

        if preferences:
            prompt_lines.insert(8, "")
            prompt_lines.insert(9, f"==== USER PREFERENCES ====\n{preferences}\n==== END USER PREFERENCES ====")

        # ── Expert knowledge ──
        kb = build_agent_prompt(sd.name)
        if kb:
            prompt_lines.insert(0, kb)
            prompt_lines.insert(0, "")

        return "\n".join(prompt_lines)

    # ── Build adjust_params handler (validates inline at tool-call level) ──
    param_table = _get_stage_param_table(sd.name)
    sections = get_stage_sections(sd.name)

    def _adjust_params_handler(arguments: dict, _ctx) -> dict:
        raw = arguments.get("adjustments", {}) if isinstance(arguments, dict) else {}
        if not raw:
            return {"status": "fail", "reason": "adjustments must be a non-empty dict of key=value pairs"}
        sanitized, warnings = validate_agent_adjustments(raw, param_table, sections, sd.name)
        if not sanitized:
            return {"status": "fail", "reason": f"All {len(warnings)} params rejected: {'; '.join(warnings[:3])}"}
        primary_section = sections[0]
        snapshot = apply_agent_adjustments(config_path, sanitized, primary_section, param_table)
        applied_str = ", ".join(f"{k}={v}" for k, v in sanitized.items())
        return {
            "status": "ok",
            "applied": dict(sanitized),
            "warnings": warnings,
            "ini_snapshot": snapshot,
            "applied_summary": applied_str,
        }

    handlers[SUBAGENT_TOOL_ADJUST_PARAMS] = _adjust_params_handler

    # ── Single spawn ──
    prompt = _build_prompt()

    agent_id = f"{sd.name}_agent_{run_id}_{os.urandom(4).hex()}"
    agent = SubAgent(
        parent_ctx=ctx,
        subagent_type=MODELING_AGENT_LABEL,
        prompt=prompt,
        agent_id=agent_id,
        subject=f"DG-TFT {sd.name} agent",
        subagent_tools=schemas,
        tool_handlers=handlers,
        stop_mode="schema",
        stop_schemas=stop_schemas,
        console=None,
    )
    ctx.agent_list[agent_id] = agent.progress
    agent.build_tools()
    agent.build_messages()
    try:
        result = agent.run()
    finally:
        agent.progress.status = agent.status
        agent.progress.if_archived = True

    if agent.status.value != "done" or not result:
        err = agent.error or f"{agent_label} did not produce a result"
        sys_log.warning("%s failed: %s", agent_label, err)
        return {"decision": "error", "tool_name": "", "arguments": {}, "reasoning": err}

    try:
        parsed = _json.loads(result) if isinstance(result, str) else dict(result)
    except (_json.JSONDecodeError, TypeError, ValueError):
        sys_log.warning("%s returned non-JSON: %s", agent_label, str(result)[:200])
        return {"decision": "error", "tool_name": "", "arguments": {}, "reasoning": str(result)[:200]}

    tool_name = _normalize_tool_name(parsed)
    reasoning = parsed.get("reasoning", "") if isinstance(parsed, dict) else ""

    return {
        "decision": tool_name,
        "tool_name": tool_name,
        "arguments": parsed,
        "reasoning": reasoning,
    }


def _normalize_tool_name(parsed: dict[str, Any]) -> str:
    """Map raw tool result to a canonical decision name."""
    if not isinstance(parsed, dict):
        return "error"

    tool_map = {
        "adjust_params": "adjust_params",
        "fallback_to_previous": "fallback_to_previous",
        "accept_quality": "accept_quality",
    }
    for key, canonical in tool_map.items():
        if key in parsed or parsed.get("tool") == key or parsed.get("function") == key:
            return canonical

    if parsed.get("applied"):
        return "adjust_params"
    if parsed.get("section") and parsed.get("adjustments"):
        return "adjust_params"
    if parsed.get("target") and parsed.get("reason"):
        return "fallback_to_previous"

    return "error"


def _process_agent_decision(
    ctx: AgentContext,
    run_id: int,
    sd: StageDef,
    config_path: str,
    op_log: OperationLog | None,
    agent_result: dict[str, Any],
    retry_remaining: int,
) -> str:
    """Process an Agent's decision and apply side effects.

    Returns:
        "retry" — re-run the same stage with adjusted params.
        "fallback" — jump to upstream stage.
        "accept" — move on (accept quality or error).
    """
    from src.workflow.dg_tft.param_defs import get_stage_sections

    decision = agent_result.get("decision", "error")
    arguments = agent_result.get("arguments", {})
    reasoning = agent_result.get("reasoning", "")
    agent_label = _AGENT_LABELS.get(sd.name, "Agent")

    if decision == "adjust_params":
        arguments = agent_result.get("arguments", {})
        reasoning = agent_result.get("reasoning", "")
        agent_label = _AGENT_LABELS.get(sd.name, "Agent")

        # Handler output (pre-validated and applied) or legacy raw arguments
        applied = arguments.get("applied", {}) if isinstance(arguments, dict) else {}
        warnings = arguments.get("warnings", []) if isinstance(arguments, dict) else []
        snapshot = arguments.get("ini_snapshot", "") if isinstance(arguments, dict) else ""

        if not applied:
            # Fallback: attempt raw validation (backward compat)
            sections = get_stage_sections(sd.name)
            param_table = _get_stage_param_table(sd.name)
            if sections is None or param_table is None:
                _console(ctx, f"  [{WF_STATUS_WARN}]{agent_label} adjust_params: no param table for {sd.name}[/{WF_STATUS_WARN}]")
                return "accept"
            raw_params = arguments.get("adjustments", {}) if isinstance(arguments, dict) else {}
            if not raw_params:
                _console(ctx, f"  [{WF_STATUS_WARN}]{agent_label} adjust_params called with empty adjustments[/{WF_STATUS_WARN}]")
                return "accept"
            sanitized, warnings = validate_agent_adjustments(raw_params, param_table, sections, sd.name)
            if not sanitized:
                _console(ctx, f"  [{WF_STATUS_WARN}]{agent_label}: all params rejected by guard[/{WF_STATUS_WARN}]")
                return "accept"
            applied = {str(k): v for k, v in sanitized.items()}
            primary_section = sections[0]
            snapshot = apply_agent_adjustments(config_path, sanitized, primary_section, param_table)

        for w in warnings[:3]:
            sys_log.warning("[%s] %s", sd.name, w)
            _console(ctx, f"  [{WF_STATUS_WARN}]{agent_label} guard: {w}[/{WF_STATUS_WARN}]")

        applied_str = arguments.get("applied_summary", "") if isinstance(arguments, dict) else ""
        if not applied_str and applied:
            applied_str = ", ".join(f"{k}={v}" for k, v in applied.items())
        _console(ctx, f"  [{WF_STATUS_SUCCESS}]{agent_label} → adjust_params: {applied_str}[/{WF_STATUS_SUCCESS}]")
        if reasoning:
            _console(ctx, f"  [bright_black]Reasoning: {reasoning[:120]}[/bright_black]")

        if op_log:
            op_log.append_attempt(sd.name, StageAttemptLog(
                n=op_log.get_last_attempt_n(sd.name) + 1,
                trigger=AttemptTrigger.AGENT_ADJUST,
                agent_decision=arguments,
                agent_reasoning=reasoning,
                ini_snapshot=snapshot,
                adjustments={str(k): str(v) for k, v in applied.items()},
                ok=False,
            ))

        return "retry" if retry_remaining > 0 else "accept"

    if decision == "fallback_to_previous":
        target = arguments.get("target", "") if isinstance(arguments, dict) else ""
        _console(ctx, f"  [{WF_STATUS_WARN}]{agent_label} → fallback_to_previous: {target}[/{WF_STATUS_WARN}]")
        if reasoning:
            _console(ctx, f"  [bright_black]Reasoning: {reasoning[:120]}[/bright_black]")

        if op_log:
            op_log.append_attempt(sd.name, StageAttemptLog(
                n=op_log.get_last_attempt_n(sd.name) + 1,
                trigger=AttemptTrigger.FALLBACK_SENT,
                agent_decision=arguments,
                agent_reasoning=reasoning,
                ok=False,
                error="fallback requested",
            ))

        return "fallback"

    if decision == "accept_quality":
        reasoning_text = arguments.get("reasoning", "") if isinstance(arguments, dict) else ""
        _console(ctx, f"  [{WF_STATUS_SUCCESS}]{agent_label} → accept_quality[/{WF_STATUS_SUCCESS}]")
        if reasoning_text:
            _console(ctx, f"  [bright_black]Reasoning: {reasoning_text[:120]}[/bright_black]")

        if op_log:
            op_log.append_attempt(sd.name, StageAttemptLog(
                n=op_log.get_last_attempt_n(sd.name) + 1,
                trigger=AttemptTrigger.AGENT_ACCEPT,
                agent_decision=arguments,
                agent_reasoning=reasoning_text,
                ok=True,
                quality_pass=False,
            ))

        return "accept"

    _console(ctx, f"  [{WF_STATUS_WARN}]{agent_label}: unknown decision '{decision}'[/{WF_STATUS_WARN}]")
    return "accept"


def _get_stage_param_table(stage_name: str):
    """Get the param_def table for a stage (lazy import to avoid circular deps)."""
    from src.workflow.dg_tft.param_defs import _STAGE_PARAM_MAP
    entry = _STAGE_PARAM_MAP.get(stage_name)
    if entry is None:
        return None
    return entry[1]


# ── Agent E — final interpretation ──────────────────────────────────

def _spawn_agent_e(
    ctx: AgentContext,
    run_id: int,
    stages: list[StageDef],
    op_log: OperationLog | None = None,
    output_dir: str = "",
    preferences: str = "",
    *,
    failed: bool = False,
    failed_stage: str = "",
    config_path: str = "",
) -> None:
    """Spawn Agent E for final interpretation and knowledge contribution.

    Agent E runs with stop_mode="any" — it produces a free-text interpretation
    and may call contribute_knowledge to persist discoveries.
    """
    _console(ctx, f"  [bright_black][Agent E] {'pipeline failed' if failed else 'workflow completed'} — "
             f"interpreting results...[/bright_black]")

    try:
        _spawn_agent_e_inner(ctx, run_id, stages, op_log, output_dir, preferences, failed, failed_stage, config_path)
    except Exception as e:
        sys_log.warning("Agent E spawn failed: %s", e)
        _console(ctx, f"  [bright_black]Agent E: could not spawn (context may be incomplete)[/bright_black]")


def _spawn_agent_e_inner(
    ctx: AgentContext,
    run_id: int,
    stages: list[StageDef],
    op_log: OperationLog | None,
    output_dir: str,
    preferences: str,
    failed: bool,
    failed_stage: str,
    config_path: str = "",
) -> None:
    from src.agent.subagent import SubAgent
    from src.constants import INTERPRETER_AGENT_LABEL, FINAL_INTERPRETATION_FILENAME
    from src.workflow.dg_tft.knowledge.loader import build_agent_prompt, load_discovered_summaries
    import json as _json

    status_text = f"FAILED at {failed_stage}" if failed else "COMPLETED successfully"

    # ── Build stage history (all attempts, not just last) ──
    stage_summary_lines: list[str] = []
    if op_log:
        for stage_name, attempts in op_log.data.get("stages", {}).items():
            if not attempts:
                continue
            stage_summary_lines.append(f"  Stage: {stage_name} ({len(attempts)} attempt(s))")
            for a in attempts:
                n = a.get("n", 0)
                ok = "OK" if a.get("ok") else "FAIL"
                trigger = a.get("trigger", "")
                quality = a.get("quality_pass", "")
                metrics = a.get("metrics", {})
                adj = a.get("adjustments", {})
                reason = a.get("agent_reasoning", "")[:120]
                parts = [f"    Attempt {n}: {ok}, trigger={trigger}, quality_pass={quality}"]
                if metrics:
                    parts.append("metrics=" + ", ".join(f"{k}={v:.4f}" for k, v in metrics.items() if v is not None))
                if adj:
                    parts.append("adjustments=" + ", ".join(f"{k}={v}" for k, v in adj.items()))
                if reason:
                    parts.append(f"agent_reasoning=\"{reason}\"")
                stage_summary_lines.append(" | ".join(parts))

    # ── Build prompt ──
    prompt_lines = [
        "You are Agent E (final interpretation) for DG-TFT modeling.",
        f"The pipeline has {status_text}.",
        f"Output directory: {output_dir}",
    ]
    if config_path and _is_minimal_run(config_path):
        prompt_lines.append(
            "minimal_run mode: quick diagnostics only — reduced data, single-epoch training, "
            "no agent-based tuning. All quality gates and Agent A/B/C/D were skipped. "
            "Your job is to flag any hard errors; do NOT critique model quality as if this were a full production run."
        )
    prompt_lines.extend([
        "",
        "LANGUAGE: All output MUST be in English only.",
        "",
        "Your job:",
        "1. Read `operation_log.json` and stage output artifacts to understand the full run.",
        "2. Recommend the BEST chain of attempts across all stages — read ALL attempts,",
        "   not just the last one per stage. For each stage, explain which attempt",
        "   produced the best result and why.",
        "3. If the pipeline failed, identify the root cause stage (the FIRST stage that",
        "   failed independently, not a cascade victim missing upstream artifacts).",
        "4. Assess modeling quality overall and deployment readiness.",
        "5. Optionally (VERY rarely), call `contribute_knowledge` if you discover a truly",
        "   novel, generalizable pattern. Most runs do NOT trigger this.",
        "",
        "Output format — write clear Markdown with these sections:",
        "  ## Recommended Path",
        "  ## Modeling Status",
        "  ## Per-Stage Analysis",
        "  ## Metric Breakdown",
        "  ## Failure Diagnosis (if applicable)",
        "  ## Deployment Readiness",
        "",
        "Be concise and actionable. Focus on what matters for this device.",
        "",
        "IMPORTANT — Injected knowledge: Your prompt includes EXPERT KNOWLEDGE (code-derived",
        "reference covering parameters, engine behavior, and decision frameworks) and",
        "EXPERIENCE (practical patterns from real sessions including diagnostic signals",
        "and tuning recipes). You may also see DISCOVERED KNOWLEDGE summaries from prior",
        "runs. Do NOT repeat or regurgitate this knowledge in your interpretation — it is",
        "context for your reasoning, not content to copy-paste. The same applies to",
        "`contribute_knowledge`: only call it for genuinely novel patterns NOT already",
        "described in the EXPERT KNOWLEDGE, EXPERIENCE, or DISCOVERED KNOWLEDGE blocks.",
    ])

    if op_log:
        prompt_lines.append("")
        prompt_lines.append("**Full operation log summary (all attempts):**")
        prompt_lines.extend(stage_summary_lines)

    # ── Expert knowledge + discovered summaries ──
    knowledge_block = build_agent_prompt("final-report", user_preferences=preferences,
                                         discovered_dir="knowledge/_discovered")
    if knowledge_block:
        prompt_lines.insert(0, knowledge_block)
        prompt_lines.insert(0, "")

    prompt = "\n".join(prompt_lines)

    agent_id = f"agent_e_interpret_{run_id}_{os.urandom(4).hex()}"
    agent = SubAgent(
        parent_ctx=ctx,
        subagent_type=INTERPRETER_AGENT_LABEL,
        prompt=prompt,
        agent_id=agent_id,
        subject="DG-TFT final interpretation",
        console=None,
    )
    ctx.agent_list[agent_id] = agent.progress
    agent.build_tools()
    agent.build_messages()
    try:
        result = agent.run()
    finally:
        agent.progress.status = agent.status
        agent.progress.if_archived = True

    if result and agent.status.value == "done":
        # Write full interpretation to file
        interpretation_path = os.path.join(output_dir, FINAL_INTERPRETATION_FILENAME)
        try:
            with open(interpretation_path, "w", encoding="utf-8") as f:
                f.write(result if isinstance(result, str) else str(result))
            _console(ctx, f"  Agent E → {FINAL_INTERPRETATION_FILENAME} written ({os.path.getsize(interpretation_path)} bytes)")
        except OSError as e:
            sys_log.warning("Agent E: failed to write %s: %s", FINAL_INTERPRETATION_FILENAME, e)
            _console(ctx, f"  [bright_black]Agent E: could not write {FINAL_INTERPRETATION_FILENAME}[/bright_black]")
    else:
        sys_log.warning("Agent E: no interpretation produced (status=%s)", agent.status.value)
        _console(ctx, f"  [bright_black]Agent E: no interpretation produced[/bright_black]")




def _get_prep_info(ctx: AgentContext, prep_id: int | None) -> dict | None:
    """Look up data preparation info from the preparation tracker.

    *prep_id* is typically the workflow run_id (prep and run ids are unified).
    """
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


def _compute_run_timeout(ctx: AgentContext, stages: list[StageDef]) -> int:
    """Run-level timeout = sum of all stage timeouts × 2."""
    total = sum(_resolve_stage_timeout(ctx, sd) or 0 for sd in stages)
    return total * 2


def _resolve_stage_timeout(ctx: AgentContext, sd: StageDef) -> int | None:
    """Resolve stage timeout: agent_configs override > StageDef default.

    Config keys follow the pattern ``DG_TFT_TIMEOUT_S_<STAGE_NAME_UPPER>``
    where STAGE_NAME is the stage name with hyphens replaced by underscores.
    """
    config_key = "DG_TFT_TIMEOUT_S_" + sd.name.upper().replace("-", "_")
    val = ctx.agent_configs.get(config_key)
    if val is not None:
        try:
            return int(val)
        except (ValueError, TypeError):
            sys_log.warning("Invalid timeout value for '%s': %s", config_key, val)
    return sd.timeout


def _elapsed_exceeds(t_start: float, timeout_s: int) -> bool:
    return (time.monotonic() - t_start) > timeout_s


def _is_minimal_run(config_path: str) -> bool:
    """Return True if runtime.minimal_run is set to yes in the given INI file."""
    try:
        cfg = configparser.ConfigParser()
        cfg.read(config_path, encoding="utf-8")
        return cfg.get("runtime", "minimal_run", fallback="no").lower() == "yes"
    except Exception:
        return False
