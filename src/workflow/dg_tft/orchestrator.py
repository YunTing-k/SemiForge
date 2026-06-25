# -*- coding: utf-8 -*-
"""
Header information
---------
Shanghai Jiao Tong University, School of Integrated Circuits, SMIL Lab
Author: Yu Huang
Create Date: 2026.6.25
Description: DG-TFT workflow orchestrator — sequential stage execution with
             fail-fast, conditional skip, artifact verification, and resume.

Revision:
---------
2026.6.25      Yu Huang      1.0      Initial implementation
"""
from __future__ import annotations

import configparser
import logging
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

if TYPE_CHECKING:
    from src.context.agent_context import AgentContext

sys_log = logging.getLogger("logger")


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
) -> tuple[bool, str]:
    """Execute the DG-TFT modeling pipeline for a single workflow run (P3-T8).

    Stages are executed in order (check-data → fit-physical → train-base-pkan
    → train-static → train-hysteresis), with fail-fast on required stages and
    automatic skip of hysteresis when no I-t data is available.

    Supports resume: stages that have already completed successfully are
    skipped on re-entry.

    Args:
        ctx: Agent context (for configs, manager, and interpreter resolution).
        run_id: Workflow run ID to execute.
        config_path: Path to the runtime model_config.ini.
        stage_timeout: Override default per-stage timeout (seconds).

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

    sys_log.info(
        "Orchestrator starting run %d with %d stages, output=%s",
        run_id, len(stages), output_dir,
    )

    run.status = WorkflowStatus.RUNNING
    run.current_stage = ""
    run.updated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    _console(ctx, f"Run #{run_id} [bold]DG-TFT workflow[/bold] 开始 ({len(stages)} 阶段)")
    _console(ctx, f"  数据: {run.dataset_dir}")
    _console(ctx, f"  输出: {output_dir}")
    t_start = time.monotonic()

    idx = 0
    for sd in stages:
        idx += 1

        if _noop_stage(sd):
            sys_log.info("Stage [%s] is a no-op placeholder, skipping", sd.name)
            continue

        if _stage_done(run, sd.name):
            sys_log.info("Stage [%s] already completed, skipping (resume)", sd.name)
            _console(ctx, f"  [{idx}/{len(stages)}] {sd.name} [bright_black]⊘ 跳过 (已完成)[/bright_black]")
            continue

        run.current_stage = sd.name
        wm.update_stage(run_id, sd.name, WorkflowStatus.RUNNING)
        sys_log.info("--- Stage [%s] starting ---", sd.name)
        _console(ctx, f"  [{idx}/{len(stages)}] {sd.name} [bright_black]开始...[/bright_black]")

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
                _console(ctx, f"  [{idx}/{len(stages)}] {sd.name} [bright_black]⊘ 跳过 ({sr.skip_reason})[/bright_black]")
                run = wm.get_run(run_id)  # refresh after manager mutation
                continue

        sr = run_stage(
            ctx, sd.action, config_path_abs,
            cwd=str(Path(config_path_abs).parent),
            timeout=timeout,
            extra_args=sd.extra_args,
        )

        if not sr.ok:
            wm.update_stage(run_id, sd.name, WorkflowStatus.FAILED, sr)
            if sd.required:
                wm.mark_failed(run_id, f"Required stage {sd.name} failed: {sr.error}")
                sys_log.error(
                    "Run %d failed at required stage [%s]: %s",
                    run_id, sd.name, sr.error,
                )
                _console(ctx, f"  [{idx}/{len(stages)}] {sd.name} [red]✗[/red] [bright_black]({sr.error})[/bright_black]")
                _console(ctx, f"[red]✗ Run #{run_id} FAILED at {sd.name}[/red]")
                return False, f"Failed at {sd.name}: {sr.error}"
            else:
                wm.add_warning(run_id, f"Optional stage {sd.name} failed: {sr.error}")
                sys_log.warning(
                    "Optional stage [%s] failed for run %d, continuing", sd.name, run_id,
                )
                _console(ctx, f"  [{idx}/{len(stages)}] {sd.name} [yellow]⚠ 失败（非必需，继续）[/yellow]")
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
        _console(ctx, f"  [{idx}/{len(stages)}] {sd.name} [green]✓[/green]")
        run = wm.get_run(run_id)

    wm.mark_done(run_id)
    elapsed = time.monotonic() - t_start
    sys_log.info("Workflow run %d completed successfully", run_id)
    _console(ctx, f"[green]✓ Run #{run_id} DONE[/green] [bright_black]({elapsed:.0f}s, {len(stages)} 阶段)[/bright_black]")
    return True, f"Run {run_id} completed: {len(stages)} stages"


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


def _console(ctx: AgentContext, message: str) -> None:
    if getattr(ctx, "subagent_mute", False):
        return
    c = getattr(ctx, "console", None)
    if c is not None:
        c.print(message)
