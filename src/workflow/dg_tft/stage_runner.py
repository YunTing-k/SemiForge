# -*- coding: utf-8 -*-
"""
Header information
---------
Shanghai Jiao Tong University, School of Integrated Circuits, SMIL Lab
Author: Yu Huang
Create Date: 2026.6.25
Description: Stage runner — unified subprocess execution for DG-TFT modeling stages

Revision:
---------
2026.6.25      Yu Huang      1.0      Initial implementation
"""
from __future__ import annotations

import logging
import os
import shlex
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING

from rich.markup import escape

from .schema import StageResult

if TYPE_CHECKING:
    from src.context.agent_context import AgentContext

sys_log = logging.getLogger("logger")

RUN_MODEL_PATH = Path(__file__).parent.parent.parent / "device_modeling" / "dg_tft" / "run_model.py"


def resolve_modeling_python(ctx: AgentContext) -> str:
    """Resolve the Python executable path for DG-TFT modeling.

    Priority:
    1. ctx.agent_configs["DG_TFT_MODELING_PYTHON_EXE"]
    2. Falls back to sys.executable with a warning (should not happen in production)
    """
    modeling_exe = ctx.agent_configs.get("DG_TFT_MODELING_PYTHON_EXE", "")
    if modeling_exe and Path(modeling_exe).exists():
        agent_exe = ctx.agent_configs.get("DG_TFT_AGENT_PYTHON_EXE", sys.executable)
        sys_log.info(
            "Resolved modeling python: %s (agent python: %s)",
            modeling_exe, agent_exe,
        )
        return modeling_exe

    sys_log.warning(
        "DG_TFT_MODELING_PYTHON_EXE not set or not found, falling back to sys.executable: %s",
        sys.executable,
    )
    return sys.executable


def run_stage(
    ctx: AgentContext,
    action: str,
    config_path: str,
    *,
    cwd: str | None = None,
    timeout: int | None = None,
    extra_args: list[str] | None = None,
    stream_console: bool = False,
) -> StageResult:
    """Execute a single modeling stage via subprocess.

    Args:
        ctx: Agent context (for configs and logging)
        action: run_model.py action (e.g. 'check-data', 'fit-physical')
        config_path: Path to the runtime config .ini file
        cwd: Working directory for subprocess
        timeout: Timeout in seconds
        extra_args: Additional CLI arguments
        stream_console: When True, print subprocess stdout line-by-line to
                        ``ctx.console`` in ``bright_black`` (used for retry loops
                        to give the user visibility into training progress).

    Returns:
        StageResult with ok, summary, outputs, warnings, error
    """
    modeling_python = resolve_modeling_python(ctx)

    command = [modeling_python, str(RUN_MODEL_PATH), action, "--config", config_path]
    if extra_args:
        command.extend(extra_args)

    stage_name = action

    sys_log.info("Stage [%s] starting: %s", stage_name, " ".join(shlex.quote(str(x)) for x in command))

    if stream_console:
        return _run_stage_streaming(ctx, command, stage_name, cwd, timeout)

    try:
        result = subprocess.run(
            command,
            cwd=cwd or str(RUN_MODEL_PATH.parent),
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        stdout = result.stdout
        stderr = result.stderr
        elapsed = time.monotonic()

        if result.returncode == 0:
            sys_log.info("Stage [%s] succeeded (rc=0)", stage_name)
            return StageResult(
                ok=True,
                stage=stage_name,
                summary=stdout.splitlines()[-5:] if stdout else [],
                warnings=_extract_warnings(stderr),
            )
        else:
            sys_log.error("Stage [%s] failed (rc=%d): %s", stage_name, result.returncode, stderr[:500] if stderr else "")
            return StageResult(
                ok=False,
                stage=stage_name,
                summary=stdout.splitlines()[-10:] if stdout else [],
                error=f"Exit code {result.returncode}: {stderr[:2000]}" if stderr else f"Exit code {result.returncode}",
                warnings=_extract_warnings(stderr),
            )

    except subprocess.TimeoutExpired:
        sys_log.error("Stage [%s] timed out after %s s", stage_name, timeout)
        return StageResult(
            ok=False,
            stage=stage_name,
            error=f"Timed out after {timeout} seconds",
        )
    except FileNotFoundError:
        sys_log.error(
            "Stage [%s] failed: modeling python not found at %s",
            stage_name, modeling_python,
        )
        return StageResult(
            ok=False,
            stage=stage_name,
            error=f"Modeling Python not found: {modeling_python}",
        )
    except Exception as e:
        sys_log.error("Stage [%s] unexpected error: %s", stage_name, e)
        return StageResult(
            ok=False,
            stage=stage_name,
            error=str(e),
        )


def _extract_warnings(stderr: str) -> list[str]:
    if not stderr:
        return []
    lines = stderr.strip().splitlines()
    warnings = []
    for line in lines:
        if "warning" in line.lower():
            warnings.append(line.strip())
    return warnings


def _run_stage_streaming(
    ctx: AgentContext,
    command: list[str],
    stage_name: str,
    cwd: str | None,
    timeout: int | None,
) -> StageResult:
    """Run a stage with streaming stdout/stderr printed to the agent console.

    Uses ``Popen`` with line-by-line reads.  stdout is printed in
    ``bright_black``, stderr in ``bold red``, each prefixed with
    ``[engine]`` to distinguish subprocess output from agent messages.
    """
    muted = getattr(ctx, "subagent_mute", False)
    c = getattr(ctx, "console", None)
    stream_stdout = ctx.agent_configs.get("DG_TFT_STREAM_STDOUT", True)
    stream_stderr = ctx.agent_configs.get("DG_TFT_STREAM_STDERR", True)

    try:
        process = subprocess.Popen(
            command,
            cwd=cwd or str(RUN_MODEL_PATH.parent),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
            start_new_session=(os.name != "nt"),
        )
    except FileNotFoundError:
        sys_log.error("Stage [%s] failed: executable not found", stage_name)
        return StageResult(
            ok=False,
            stage=stage_name,
            error=f"Executable not found: {command[0]}",
        )
    except Exception as e:
        sys_log.error("Stage [%s] unexpected error (spawn): %s", stage_name, e)
        return StageResult(
            ok=False,
            stage=stage_name,
            error=str(e),
        )

    # Drain both pipes concurrently so timeout/cancellation is not blocked by
    # a silent engine waiting inside stdout iteration. Keep full attempt logs.
    log_root = Path(cwd or RUN_MODEL_PATH.parent) / "stage_logs"
    log_root.mkdir(parents=True, exist_ok=True)
    stamp = f"{stage_name}_{time.time_ns()}"
    stdout_path = log_root / f"{stamp}.stdout.log"
    stderr_path = log_root / f"{stamp}.stderr.log"
    stdout_lines: list[str] = []
    stderr_lines: list[str] = []

    def consume(pipe, destination, lines, enabled, style):
        with destination.open("w", encoding="utf-8") as log:
            for line in pipe:
                lines.append(line.rstrip())
                log.write(line)
                log.flush()
                if c and not muted and enabled and line.strip():
                    c.print(f"    [{style}][engine] {escape(line.rstrip())}[/{style}]")

    workers = [
        threading.Thread(target=consume, args=(process.stdout, stdout_path, stdout_lines, stream_stdout, "bright_black"), daemon=True),
        threading.Thread(target=consume, args=(process.stderr, stderr_path, stderr_lines, stream_stderr, "bold red"), daemon=True),
    ]
    for worker in workers:
        worker.start()
    outputs = {"stdout_log": str(stdout_path), "stderr_log": str(stderr_path)}
    try:
        returncode = process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        _stop_process_tree(process)
        return StageResult(ok=False, stage=stage_name, outputs=outputs,
                           error=f"Timed out after {timeout} seconds")
    except BaseException:
        # run_model.py can itself spawn training subprocesses. Cancel the owned
        # process tree before propagating the interrupt to the orchestrator.
        try:
            _stop_process_tree(process)
        except Exception:
            sys_log.exception("Stage [%s] cleanup failed during interruption", stage_name)
        raise
    finally:
        for worker in workers:
            worker.join(timeout=5)
        process.stdout.close()
        process.stderr.close()
    stderr_text = "\n".join(stderr_lines)

    if returncode == 0:
        sys_log.info("Stage [%s] succeeded (rc=0)", stage_name)
        return StageResult(
            ok=True,
            stage=stage_name,
            outputs=outputs,
            summary=stdout_lines[-5:] if stdout_lines else [],
            warnings=_extract_warnings(stderr_text),
        )
    else:
        sys_log.error(
            "Stage [%s] failed (rc=%d): %s",
            stage_name, returncode, stderr_text[:500] if stderr_text else "",
        )
        return StageResult(
            ok=False,
            stage=stage_name,
            outputs=outputs,
            summary=stdout_lines[-10:] if stdout_lines else [],
            error=f"Exit code {returncode}: {stderr_text[:2000]}" if stderr_text else f"Exit code {returncode}",
            warnings=_extract_warnings(stderr_text),
        )


def _stop_process_tree(process) -> None:
    """Terminate only this runner's child tree; never use name-based process kills."""
    if process.poll() is not None:
        return
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                           capture_output=True, timeout=15)
        else:
            os.killpg(process.pid, signal.SIGKILL)
    except (OSError, subprocess.SubprocessError):
        sys_log.exception("Could not terminate child tree for PID %s", process.pid)
    finally:
        # Still reap the direct child if the platform tree helper is unavailable.
        if process.poll() is None:
            process.kill()
        process.wait(timeout=15)
