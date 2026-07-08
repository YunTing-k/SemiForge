# -*- coding: utf-8 -*-
"""
Header information
---------
Shanghai Jiao Tong University, School of Integrated Circuits, SMIL Lab
Author: Yu Huang
Create Date: 2026.6.25
Description: Data preparation — worker subagent + verifier LLM call + programmatic check

Revision:
---------
2026.6.25      Yu Huang      1.0      Initial implementation
"""
from __future__ import annotations

import json
import os
import logging
from pathlib import Path
from typing import Any

from src.context.agent_context import AgentContext
from src.constants import PREPARER_AGENT_LABEL

sys_log = logging.getLogger("logger")

# ── Output filenames ──
_PREP_WORKER_REPORT_NAME = "worker_report.json"
_PREP_VERIFIER_RESULT_NAME = "verifier_result.json"
_PREP_MANIFEST_NAME = "manifest.json"
_PREP_FILE_MANIFEST_NAME = "prep_file_manifest.json"

# ── Data format constants (single source of truth) ──
_PREP_REQUIRED_IV_COLUMNS = ("VD_V", "VBG_V", "VTG_V", "ID_A")
_PREP_IV_SHEET_NAME = "long"
_PREP_CV_EXPECTED_SHEETS = ("cgg", "cbg", "ctg", "vds")

# ── Configurable defaults ──
_PREP_AGENT_ID_PREFIX = "prep_worker"
_PREP_SUBJECT = "DG-TFT data preparation"

WORKER_PROMPT_TEMPLATE = """\
You are a data preparation specialist for DG-TFT device modeling. Convert raw measurement data into the standardized format required by the DG-TFT modeling pipeline.

**Raw data directory**: {raw_dir}
**Output directory**: {output_dir}

The output directory already exists. Write all output files into it.

**Engine requirements your output MUST meet**:

IV files:
  - Filename pattern: `IDVBG*_train.xlsx`, `IDVTG*_train.xlsx`, `IDVD*_train.xlsx`
  - Sheet name: `long` (case-sensitive — files without this sheet are silently ignored by the engine)
  - Required columns: `VD_V`, `VBG_V`, `VTG_V`, `ID_A` (all four mandatory; missing → engine crash)

CV file:
  - Filename: exactly `CV_train.xlsx` (hardcoded in engine, not a wildcard)
  - Internal layout: Row 0 = description text, Row 1 = column headers, Row 2+ = numeric data
    The engine reads with `header=None` and uses `iloc[1]` for headers, `iloc[2:]` for data.
    If your source CV file has headers on Row 0, you MUST fix this — insert an empty Row 0.
  - Column headers must contain text the engine can parse: VTG=X.XXV, VBG=X.XXV, VDS=X.XXV,
    frequency (kHz/MHz/Hz), temperature (T=...K). The engine regex-matches these from Row 1.

I-t files:
  - Exact expected filenames in `{output_dir}/it/`:
    `I-t with different VBG1.xlsx`, `I-t with different VBG2.xlsx`, `I-t with different VBG3.xlsx`,
    `I-t with different VTG1.xlsx`, `I-t with different VTG2.xlsx`, `I-t with different VTG3.xlsx`,
    `I-t with different VBG_temperature.xlsx`, `I-t with different VTG_temperature.xlsx`
  - The engine tries these 8 exact names first, then falls back to `I-t.xlsx`, then auto-discover.
    Missing some is OK (engine skips gracefully), but naming must be correct for the ones you have.

**Python usage rules**:
- Use `{python_exe} -c "..."` for all inline Python code — do NOT write scripts to files
- Always import libraries at the start of every `-c` command: `import pandas as pd, openpyxl, os, shutil, json, glob`
- CSV columns may have leading spaces — use `df.columns = df.columns.str.strip()` to clean them
- Use standard GNU bash commands (rm, not del) for any non-Python operations — the shell is git bash

**Data precision rules — CRITICAL**:
- NEVER round or truncate floating-point values. Measurement data (especially ID_A in nA/µA range) must be preserved at full precision.
- Use `pd.to_excel()` WITHOUT a `float_format` argument — pandas/openpyxl already stores full float64 precision. Adding float_format would silently TRUNCATE significant digits. DO NOT add float_format.
- When reshaping wide→long: verify that no NaN values were introduced. If melt/pivot produces NaN, that indicates incorrect column mapping, not missing data — stop and fix the reshape logic.
- Do NOT use `df.fillna(0.0)` or `df.dropna()` — if you see NaN after reshape, your column mapping is wrong. Fix it, do not mask it.
- When fixing CV/I-t layout, use openpyxl row insertion (`ws.insert_rows()`) — never re-serialize data cells through pandas. Data values in cells must pass through unchanged.
- The engine fits physical parameters (Vth, mobility, subthreshold slope) from this data; any precision loss here directly corrupts the modeling results.

**Step-by-step instructions**:

1. First, `glob_file` to list all files in `{raw_dir}` and `{raw_dir}/it/` (if the it/ subdir exists).

2. **IV files** — process each:
   - Filename contains `IDVBG`, `IDVTG`, or `IDVD` → it's an IV file.
   - If the file already matches `ID*_VDS*V_train.xlsx` with sheet `long` and columns `{iv_columns}`, just copy it.
   - Otherwise, process by format:

   **IV CSV files** (e.g., `IDVBG VDS01V.csv`):
   - Read: `df = pd.read_csv("{raw_dir}/IDVBG VDS01V.csv")`
   - Clean: `df.columns = df.columns.str.strip(); df.columns = [c.replace("(V)","_V").replace("(A)","_A") for c in df.columns]`
   - Save: `df.to_excel("{output_dir}/IDVBG_VDS01V_train.xlsx", sheet_name="{iv_sheet}", index=False)`

   **IV XLSX files** (e.g., `IDVBG_VDS0.1V.xlsx`):
   - XLSX files often have wide-format/pivot layout: column headers are text like
     `ID-VBG, VDS=0.1 V` and rows are measurement values. You MUST reshape these to
     **long format** with exactly these 4 columns: `VD_V, VBG_V, VTG_V, ID_A`.
   - How to reshape: use `pd.read_excel()`, then `pd.melt()` or `df.stack()` to
     convert wide→long, then rename columns to the standard names.
   - **CRITICAL**: After reshaping, verify ALL 4 columns `VD_V, VBG_V, VTG_V, ID_A` exist.
     If any is missing, your output is invalid and the verifier will reject it.

   **Family mapping rules**:
   - `IDVBG_*` -> `IDVBG_VDS<Vvalue>_V_train.xlsx` (extract V from filename)
   - `IDVTG_*` -> `IDVTG_VDS<Vvalue>_V_train.xlsx`
   - `IDVD_*VBG*` -> `IDVD_VBG<Vvalue>_V_train.xlsx` (check for VBG in name)
   - `IDVD_*VTG*` -> `IDVD_VTG<Vvalue>_V_train.xlsx` (check for VTG in name)
   - Skip: `DUAL_REF_*`, `Refernce.xlsx`, `~$*` temp files
   - Normalize V values: 01V = 0.1V, 05V = 0.5V, 5V = 5V.
   - If both CSV and XLSX exist for the same measurement → only process XLSX. Check if output already exists before writing.

3. **CV file** — find and fix:

   a. Use `glob_file` to find the CV data file. Look for filenames containing `CV`, `cv`, `capacitance`, `C-V`, `C_F` (case-insensitive). It's usually an `.xlsx` file. If exactly one match, use it. If multiple, pick the one with `wide` or `tables` in the name; if still ambiguous, pick the largest.

   b. **MUST** use `inspect_xlsx` on this file to check the row layout. Look at the first 3 rows of each sheet:
      - Engine expects: **Row 0 = description text**, **Row 1 = column headers** (with text like "Bias", "Capacitance", "VTG=...", "VBG=..."), **Row 2+ = numeric data**.
      - If Row 0 of your source file **already has** column-header-like text (words like "bias", "cap", "voltage", "vtg", "vbg") AND Row 1 is numeric → the source has headers on Row 0. You MUST shift everything down by inserting a blank row.
      - If Row 0 is short text or empty, and Row 1 has column headers → already correct, no fix needed.

   c. If a fix is needed, use openpyxl to insert a blank row at position 1 (top of each sheet):
      `wb = openpyxl.load_workbook("{raw_dir}/the_cv_file.xlsx"); for ws in wb.worksheets: ws.insert_rows(1); wb.save("{output_dir}/CV_train.xlsx")`

   d. If no fix is needed, just copy: `shutil.copy("{raw_dir}/the_cv_file.xlsx", "{output_dir}/CV_train.xlsx")`

   e. **CRITICAL**: The output MUST be named exactly `CV_train.xlsx`. Do NOT use pandas `to_excel()` — use `shutil.copy` or openpyxl `wb.save()` only. Pandas re-serialization destroys merged cells and internal sheet metadata.

4. **I-t files** — discover, classify, and rename:

   a. If `{raw_dir}/it/` exists, `glob_file` to list all `.xlsx`/`.xls` files there. If no `it/` subdir, scan `{raw_dir}` for files matching `I-t*` or `IT*` (case-insensitive). Skip `~$*` temp files.

   b. For each I-t file, use `inspect_xlsx` to read the first sheet's column names (just the header row — no data needed). Determine the file's **family** and **role**:
      - If column names contain `vbg` or `VBG` → **VBG** family (back-gate sweep)
      - If column names contain `vtg` or `VTG` → **VTG** family (top-gate sweep)
      - If filename or column names contain `temp`, `temperature`, `T(k)`, `T(K)` → **temperature dataset**
      - If you can't determine the family from columns, check the filename — `VBG`/`vbg` vs `VTG`/`vtg`.

   c. Group files by (family, is_temperature):
      - Non-temperature VBG files → name them `I-t with different VBG1.xlsx`, `VBG2`, `VBG3` (sorted alphabetically or by voltage value in filename)
      - Non-temperature VTG files → `I-t with different VTG1.xlsx`, `VTG2`, `VTG3`
      - Temperature VBG → `I-t with different VBG_temperature.xlsx` (only one expected)
      - Temperature VTG → `I-t with different VTG_temperature.xlsx` (only one expected)

   d. If a file cannot be classified as either VBG or VTG, default it to **VTG** (the engine's fallback behavior) and report this in `potential_issues`.

   e. Copy files with their new names to `{output_dir}/it/`:
      `os.makedirs("{output_dir}/it", exist_ok=True); shutil.copy("{raw_dir}/it/original_name.xlsx", "{output_dir}/it/I-t with different VBG1.xlsx")`

5. **Report filename**: Write a `{report_name}` JSON file summarizing what you did.

6. Call `submit_handoff` with a summary of your work:
   - `observations`: What you found (file types, formats, column naming patterns, how many IV/CV/I-t files)
   - `actions`: What you did (which files processed, how reshaped/renamed, CV row-shift if applied)
   - `potential_issues`: Things the verifier should double-check (files you couldn't classify to VBG/VTG, ambiguous IV families, missing VDS scans, too-few I-t files)
   - `output_manifest`: Counts — iv_files, cv_files, it_files

**Each file should take ONE python -c command. After processing all files, call `submit_handoff`.** Do NOT output free text — your plain text response will be ignored.

Do NOT ask the user any questions — work autonomously. Do NOT explore files excessively — read only what you need to classify and transform.
"""

VERIFIER_PROMPT_TEMPLATE = """\
You are a verification specialist for DG-TFT data preparation. Your job is to inspect the output directory and report findings in a structured format via the report_verification tool. You may be called in three scenarios: (1) a worker just processed raw data — check the handoff below for what was done, (2) the pipeline entered at the Verifier stage with a pre-existing dataset — verify it meets standards, (3) Agent A (check-data diagnostics) requested a re-check after check-data failure — focus your checks on the issues identified in the agent_a_fallback section of the handoff.

**Prepared data directory**: {output_dir}

**Engine requirements this data MUST satisfy**:

IV files:
  - Filename pattern: `IDVBG*_train.xlsx`, `IDVTG*_train.xlsx`, `IDVD*_train.xlsx`
  - Each must have a sheet named `long` (exact case) with columns: {iv_columns}
  - Engine uses filename PREFIX to assign IV family — wrong prefix = engine misclassifies the data

CV file:
  - Filename: exactly `CV_train.xlsx` (hardcoded, not a wildcard)
  - Internal layout: Row 0 = description text, Row 1 = column headers, Row 2+ = numeric data
    Engine reads with `header=None` using `iloc[1]` for headers, `iloc[2:]` for data.
    If Row 0 contains column-header-like text ("bias", "cap", "voltage", "vtg", "vbg") instead
    of descriptive text, Row 1's data will be MISREAD as column headers — a silent corruption.
  - Expected sheet names: {cv_sheets} (engine reads all sheets, but these are the standard four)

I-t files:
  - Expected in `{output_dir}/it/`:
    `I-t with different VBG1.xlsx`, `VBG2.xlsx`, `VBG3.xlsx`
    `I-t with different VTG1.xlsx`, `VTG2.xlsx`, `VTG3.xlsx`
    `I-t with different VBG_temperature.xlsx`, `I-t with different VTG_temperature.xlsx`
  - Engine tries these 8 exact names first — file_count alone is insufficient; naming matters

**Instructions**:
1. Use `glob_file` to list all files in the output directory (including the it/ subdirectory).
2. For each `*_train.xlsx` file (NOT CV_train.xlsx):
   - Verify filename matches `IDVBG*_train.xlsx`, `IDVTG*_train.xlsx`, or `IDVD*_train.xlsx`
   - Use `inspect_xlsx` to check sheet `long` exists and contains columns: {iv_columns}
3. For `CV_train.xlsx`:
   - Verify it exists
   - Use `inspect_xlsx` to read the first 3 rows of each sheet. Check: does Row 0 look like description text (short, non-column-header content)? Does Row 1 look like column headers (containing words like bias, cap, voltage)? Report if Row 0 appears to be headers instead of description — this means the layout is shifted and will silently corrupt engine reads.
   - Report sheet names present
4. For the `it/` subdirectory:
   - List files and check against the 8 expected names above. Report which names are present and which are missing.
   - Report file count
5. After inspecting, call `report_verification` with your structured findings.

**CRITICAL**: Call the `report_verification` tool exactly once at the end. Do not output free text. Inspect EVERY *_train.xlsx IV file individually — do not skip any. Check sheets and columns for each one.
"""


def run_data_preparation(ctx: AgentContext, raw_dataset_dir: str, output_dir: str, prep_id: int) -> None:
    """Run the full data preparation pipeline: worker -> verifier -> check.

    Compatibility wrapper — delegates to :func:`run_data_preparation_with_retry`
    with max_retries=1 (no retries). New code should call the retry function directly.

    Args:
        ctx: Agent context.
        raw_dataset_dir: Path to raw measurement data.
        output_dir: Path to write prepared data.
        prep_id: Preparation ID registered by the caller via PreparationTracker.
    """
    console = getattr(ctx, "console", None)
    sys_log.info("Data prep %d started: raw=%s output=%s", prep_id, raw_dataset_dir, output_dir)

    if console and not getattr(ctx, "subagent_mute", False):
        console.print(f"[bold]DG-TFT Data Preparation[/bold] [bright_black]— converting raw measurements to *_train.xlsx format[/bright_black]")

    try:
        from src.tool.subagent_tools import SUBAGENT_TOOL_REPORT_VERIFICATION
        handoff, verifier_feedback, ok, issues = _run_worker_and_verifier(
            ctx, raw_dataset_dir, output_dir, SUBAGENT_TOOL_REPORT_VERIFICATION,
            handoff_history="",
        )
    except Exception as e:
        sys_log.error("Data prep %d failed: %s", prep_id, e)
        if console and not getattr(ctx, "subagent_mute", False):
            console.print(f"  [bold red]Failed[/bold red]: {e}")
        manifest = _build_manifest(output_dir, prep_id, False, [str(e)])
        _write_manifest(output_dir, manifest)
        ctx.workflow_man.preparation_tracker.update_prep(prep_id, False, [str(e)])
        return

    manifest = _build_manifest(output_dir, prep_id, ok, issues)
    _write_manifest(output_dir, manifest)
    _write_file_manifest(output_dir, prep_id)
    ctx.workflow_man.preparation_tracker.update_prep(prep_id, ok, issues)

    if console and not getattr(ctx, "subagent_mute", False):
        if ok:
            console.print(f"  [bold green]Verification[/bold green] [bright_black]passed ({len(issues)} warnings)[/bright_black]")
        else:
            console.print(f"  [bold red]Verification[/bold red] [bright_black]failed[/bright_black]")
            for issue in issues:
                if "CRITICAL" in issue:
                    console.print(f"    [bold red]{issue}[/bold red]")
                else:
                    console.print(f"    [bold yellow]{issue}[/bold yellow]")

    sys_log.info("Data prep %d done: ok=%s issues=%s", prep_id, ok, issues)


def _run_worker(ctx: AgentContext, raw_dir: str, output_dir: str, handoff_history_text: str = "") -> dict[str, Any] | None:
    """Launch a preparer subagent to process raw data into standardized format.

    The preparer uses ``stop_mode="schema"`` with ``submit_handoff`` as its
    output tool (auto-resolved from agent type).

    Args:
        ctx: Agent context.
        raw_dir: Path to raw measurement data.
        output_dir: Destination directory for prepared files.
        handoff_history_text: Previous handoff and verifier feedback to inject
                              into the worker's prompt (for retry attempts).

    Returns:
        Parsed handoff dict (submit_handoff arguments), or None on failure.
    """
    modeling_exe = ctx.agent_configs.get("DG_TFT_MODELING_PYTHON_EXE", "python")
    python_exe = modeling_exe if modeling_exe and Path(modeling_exe).exists() else "python"
    prompt = WORKER_PROMPT_TEMPLATE.format(
        raw_dir=raw_dir, output_dir=output_dir,
        iv_sheet=_PREP_IV_SHEET_NAME,
        iv_columns=", ".join(_PREP_REQUIRED_IV_COLUMNS),
        cv_sheets=", ".join(_PREP_CV_EXPECTED_SHEETS),
        python_exe=python_exe,
        report_name=_PREP_WORKER_REPORT_NAME,
    )
    if handoff_history_text:
        prompt += f"\n\n**Previous attempt feedback** (use this to fix issues):\n{handoff_history_text}"

    from src.agent.subagent import SubAgent

    agent = SubAgent(
        parent_ctx=ctx,
        subagent_type=PREPARER_AGENT_LABEL,
        prompt=prompt,
        agent_id=f"{_PREP_AGENT_ID_PREFIX}_{os.path.basename(output_dir)}",
        subject=_PREP_SUBJECT,
        console=None,
    )
    ctx.agent_list[agent.agent_id] = agent.progress
    agent.build_tools()
    agent.build_messages()
    try:
        result = agent.run()
    finally:
        agent.progress.status = agent.status
        agent.progress.if_archived = True

    handoff: dict[str, Any] | None = None
    if agent.status.value == "done" and result:
        try:
            handoff = json.loads(result) if isinstance(result, str) else dict(result)
        except (json.JSONDecodeError, TypeError, ValueError):
            sys_log.warning("Preparer returned non-JSON result: %s", str(result)[:200])
            handoff = {"observations": [], "actions": [str(result)[:500]], "output_manifest": {}}
        sys_log.info("Data prep worker completed: %s", json.dumps(handoff, ensure_ascii=False)[:300])
    else:
        err = agent.error or "unknown error"
        raise RuntimeError(f"Preparer subagent failed: {err}")

    worker_report_path = os.path.join(output_dir, _PREP_WORKER_REPORT_NAME)
    try:
        with open(worker_report_path, "w", encoding="utf-8") as f:
            json.dump(handoff, f, ensure_ascii=False, indent=2)
    except Exception as e:
        sys_log.warning("Failed to write worker handoff report: %s", e)

    return handoff

def _run_verifier_subagent(ctx: AgentContext, output_dir: str,
                          report_verification_tool_name: str = "report_verification",
                          handoff: dict[str, Any] | None = None) -> dict[str, Any] | None:
    """Run a SubAgent-based verifier to inspect prepared data and report findings.

    The verifier type (``VERIFIER_AGENT_LABEL``) auto-resolves its tools
    (``inspect_xlsx`` + ``report_verification``) and ``stop_mode="schema"``
    from the agent type config — no explicit ``subagent_tools`` or
    ``tool_handlers`` params are needed.

    Args:
        ctx: Agent context.
        output_dir: Directory containing prepared training files.
        report_verification_tool_name: Name constant for the report_verification tool.
        handoff: The Preparer's handoff dict with observations/actions/potential_issues/output_manifest.

    Returns:
        Parsed verifier feedback dict, or None if the verifier failed to produce one.
    """
    prompt = VERIFIER_PROMPT_TEMPLATE.format(
        output_dir=output_dir,
        iv_sheet=_PREP_IV_SHEET_NAME,
        iv_columns=", ".join(_PREP_REQUIRED_IV_COLUMNS),
        cv_sheets=", ".join(_PREP_CV_EXPECTED_SHEETS),
    )
    if handoff:
        handoff_text = json.dumps(handoff, ensure_ascii=False, indent=2)
        prompt += f"\n\n**Preparer handoff** (what the worker found and did — use this to focus your checks):\n```json\n{handoff_text}\n```"

    from src.tool.subagent_tools import get_tools
    from src.agent.subagent import SubAgent
    from src.constants import VERIFIER_AGENT_LABEL

    agent = SubAgent(
        parent_ctx=ctx,
        subagent_type=VERIFIER_AGENT_LABEL,
        prompt=prompt,
        agent_id=f"prep_verifier_{os.path.basename(output_dir)}",
        subject="DG-TFT data verification",
        console=None,
    )
    ctx.agent_list[agent.agent_id] = agent.progress
    agent.build_tools()
    agent.build_messages()
    try:
        result = agent.run()
    finally:
        agent.progress.status = agent.status
        agent.progress.if_archived = True

    if agent.status.value == "done" and result:
        sys_log.info("Verifier SubAgent completed: %s", result[:200] if result else "(empty)")
        try:
            return json.loads(result) if isinstance(result, str) else dict(result)
        except (json.JSONDecodeError, TypeError, ValueError):
            sys_log.warning("Verifier returned non-JSON result, using as summary: %s", str(result)[:200])
            return {"overall_ok": True, "summary": str(result)[:500]}
    else:
        err = agent.error or "unknown error"
        sys_log.warning("Verifier SubAgent did not call report_verification: %s", err)
        fallback = {"overall_ok": False, "summary": f"Verifier failed to produce structured feedback: {err}"}
        _write_verifier_result(output_dir, fallback)
        return fallback


def _run_worker_and_verifier(
    ctx: AgentContext,
    raw_dataset_dir: str,
    output_dir: str,
    report_verification_tool_name: str,
    handoff_history: str = "",
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, bool, list[str]]:
    """Run a single worker→verifier→check cycle.

    Returns:
        (handoff, verifier_feedback, ok, issues).
        handoff is the parsed worker handoff dict.
        verifier_feedback is the parsed verifier result dict.
        ok is True if data prep passed programmatic checks.
        issues is the list of warnings/errors from _check_output.
    """
    console = getattr(ctx, "console", None)
    muted = getattr(ctx, "subagent_mute", False)

    if console and not muted:
        console.print("  [bold]Preparer[/bold] [bright_black]SubAgent processing raw files ...[/bright_black]")
    handoff = _run_worker(ctx, raw_dataset_dir, output_dir, handoff_history_text=handoff_history)
    if console and not muted:
        iv_count = handoff.get("output_manifest", {}).get("iv_files", 0) if handoff else 0
        cv_ok = "✓" if (handoff.get("output_manifest", {}).get("cv_files", 0) if handoff else 0) else "✗"
        console.print(f"  [bold green]Preparer done[/bold green] [bright_black]({iv_count} IV files, CV:{cv_ok})[/bright_black]")

    if console and not muted:
        console.print("  [bold]Verifier[/bold] [bright_black]SubAgent checking output structure ...[/bright_black]")
    verifier_feedback = _run_verifier_subagent(ctx, output_dir, report_verification_tool_name, handoff=handoff)
    _write_verifier_result(output_dir, verifier_feedback)
    if console and not muted:
        vf_ok = verifier_feedback.get("overall_ok", False) if verifier_feedback else False
        status = "[bold green]Verifier done ✓[/bold green]" if vf_ok else "[bold red]Verifier done ✗[/bold red]"
        summary = verifier_feedback.get("summary", "")[:120] if verifier_feedback else ""
        console.print(f"  {status} [bright_black]{summary}[/bright_black]")

    ok, issues = _check_output(output_dir)
    return handoff, verifier_feedback, ok, issues


def run_data_preparation_with_retry(
    ctx: AgentContext,
    raw_dataset_dir: str,
    run_output_dir: str,
    run_id: int,
    preferences: str = "",
    max_retries: int = 3,
) -> dict[str, Any]:
    """Run data preparation with Preparer-Verifier retry loop.

    Each attempt creates a subdirectory ``data_prep/attempt_N/`` under
    ``run_output_dir``.  The operation log is written to
    ``run_output_dir/operation_log.json``.

    Args:
        ctx: Agent context.
        raw_dataset_dir: Path to raw measurement data.
        run_output_dir: Top-level run output directory (session/<uuid>/runs/run_NNNN/).
        run_id: Workflow run ID.
        preferences: User preferences string.
        max_retries: Maximum number of preparer→verifier cycles (default 3).

    Returns:
        Dict with keys: ok, prep_id, output_dir, handoff, issues.
    """
    import time as time_mod
    from src.constants import PREP_SUBDIR_NAME
    from src.tool.subagent_tools import SUBAGENT_TOOL_REPORT_VERIFICATION
    from .operation_log import OperationLog, StageAttemptLog, AttemptTrigger

    console = getattr(ctx, "console", None)
    muted = getattr(ctx, "subagent_mute", False)

    if console and not muted:
        console.print(f"[bold]DG-TFT Data Preparation[/bold] [bright_black]run {run_id} — Preparer-Verifier loop (max {max_retries} retries)[/bright_black]")

    op_log = OperationLog(run_output_dir)
    op_log.initialize(run_id, preferences)

    final_handoff = None
    final_feedback = None
    final_ok = False
    final_issues: list[str] = []
    last_attempt_dir = ""

    for attempt in range(1, max_retries + 1):
        attempt_dir = os.path.join(run_output_dir, PREP_SUBDIR_NAME, f"attempt_{attempt}")
        os.makedirs(attempt_dir, exist_ok=True)
        last_attempt_dir = attempt_dir

        if console and not muted:
            console.print(f"  [bold]Attempt {attempt}/{max_retries}[/bold] [bright_black]→ {attempt_dir}[/bright_black]")

        handoff_history = _get_handoff_history(op_log)

        t_start = time_mod.monotonic()
        try:
            handoff, verifier_feedback, ok, issues = _run_worker_and_verifier(
                ctx, raw_dataset_dir, attempt_dir,
                SUBAGENT_TOOL_REPORT_VERIFICATION,
                handoff_history=handoff_history,
            )
            duration = time_mod.monotonic() - t_start
        except Exception as e:
            sys_log.error("Data prep attempt %d failed: %s", attempt, e)
            trigger = AttemptTrigger.INITIAL if attempt == 1 else AttemptTrigger.VERIFIER_FAIL
            op_log.append_attempt("data-prep", StageAttemptLog(
                n=attempt, trigger=trigger, ok=False, error=str(e),
            ))
            final_issues = [str(e)]
            continue

        vf_ok = verifier_feedback.get("overall_ok", False) if verifier_feedback else False
        all_ok = ok and vf_ok

        trigger = AttemptTrigger.INITIAL if attempt == 1 else AttemptTrigger.VERIFIER_FAIL
        op_log.append_attempt("data-prep", StageAttemptLog(
            n=attempt,
            trigger=trigger,
            handoff=handoff,
            agent_decision=verifier_feedback,
            ok=all_ok,
            error="" if all_ok else "verifier or programmatic check failed",
            duration_s=duration,
        ))

        final_handoff = handoff
        final_feedback = verifier_feedback
        final_ok = all_ok
        final_issues = issues if issues else []

        if all_ok:
            op_log.set_final_status("done")
            if console and not muted:
                console.print(f"  [bold green]Data prep passed[/bold green] [bright_black](attempt {attempt})[/bright_black]")
            break
        else:
            if console and not muted:
                failure_reason = "verifier found issues" if not vf_ok else "programmatic check failed"
                console.print(f"  [bold yellow]Attempt {attempt} failed[/bold yellow] [bright_black]({failure_reason})[/bright_black]")
            if attempt == max_retries:
                op_log.set_final_status("exhausted")
                if console and not muted:
                    console.print(f"  [bold red]Data prep exhausted after {max_retries} attempts[/bold red]")
    else:
        if console and not muted:
            console.print("  [bold]Data prep — no attempts executed[/bold]")

    if not final_ok:
        op_log.set_final_status("failed")

    if last_attempt_dir:
        manifest = _build_manifest(last_attempt_dir, run_id, final_ok, final_issues)
        _write_manifest(last_attempt_dir, manifest)
        _write_file_manifest(last_attempt_dir, run_id)

    return {
        "ok": final_ok,
        "prep_id": run_id,
        "output_dir": last_attempt_dir,
        "handoff": final_handoff,
        "verifier_feedback": final_feedback,
        "issues": final_issues,
    }


def run_verification_only(
    ctx: AgentContext,
    dataset_dir: str,
    run_output_dir: str,
    run_id: int,
    preferences: str = "",
    raw_dataset_dir: str = "",
) -> dict[str, Any]:
    """Standard data entry: verify an already-prepared dataset, skip Preparer.

    If the Verifier passes, the dataset is used as-is.  If it fails and
    ``raw_dataset_dir`` is provided, falls back to the full
    Preparer→Verifier retry loop.

    Args:
        ctx: Agent context.
        dataset_dir: Directory containing prepared *_train.xlsx files.
        run_output_dir: Top-level run output directory.
        run_id: Workflow run ID.
        preferences: User preferences string.
        raw_dataset_dir: Optional raw data path for fallback.

    Returns:
        Dict with keys: ok, prep_id, output_dir, handoff, issues.
    """
    import time as time_mod
    import shutil
    from src.constants import PREP_SUBDIR_NAME
    from src.tool.subagent_tools import SUBAGENT_TOOL_REPORT_VERIFICATION
    from .operation_log import OperationLog, StageAttemptLog, AttemptTrigger

    console = getattr(ctx, "console", None)
    muted = getattr(ctx, "subagent_mute", False)

    if console and not muted:
        console.print(f"[bold]DG-TFT Standard Data Verification[/bold] [bright_black]run {run_id}[/bright_black]")

    op_log = OperationLog(run_output_dir)
    op_log.initialize(run_id, preferences)

    attempt_dir = os.path.join(run_output_dir, PREP_SUBDIR_NAME, "attempt_1")
    os.makedirs(attempt_dir, exist_ok=True)

    # Copy existing dataset into output dir
    for item in Path(dataset_dir).iterdir():
        src = str(item)
        dst = os.path.join(attempt_dir, item.name)
        if item.is_dir():
            shutil.copytree(src, dst, dirs_exist_ok=True)
        elif item.is_file():
            shutil.copy2(src, dst)

    if console and not muted:
        console.print(f"  [bold]Verifier[/bold] [bright_black]checking standard data ...[/bright_black]")
    t_start = time_mod.monotonic()
    verifier_feedback = _run_verifier_subagent(ctx, attempt_dir, SUBAGENT_TOOL_REPORT_VERIFICATION)
    _write_verifier_result(attempt_dir, verifier_feedback)
    duration = time_mod.monotonic() - t_start

    ok, issues = _check_output(attempt_dir)
    vf_ok = verifier_feedback.get("overall_ok", False) if verifier_feedback else False
    all_ok = ok and vf_ok

    op_log.append_attempt("data-prep", StageAttemptLog(
        n=1, trigger=AttemptTrigger.INITIAL,
        agent_decision=verifier_feedback,
        ok=all_ok,
        error="" if all_ok else "standard data verification failed",
        duration_s=duration,
    ))

    if all_ok:
        op_log.set_final_status("done")
        return {
            "ok": True, "prep_id": run_id, "output_dir": attempt_dir,
            "handoff": None, "issues": issues,
        }

    if console and not muted:
        console.print(f"  [bold yellow]Standard data verification failed[/bold yellow] [bright_black]— falling back to Preparer[/bright_black]")

    if not raw_dataset_dir:
        op_log.set_final_status("failed")
        return {
            "ok": False, "prep_id": run_id, "output_dir": attempt_dir,
            "handoff": None, "issues": issues + ["Standard data verification failed, no fallback raw data provided"],
        }

    return run_data_preparation_with_retry(
        ctx, raw_dataset_dir, run_output_dir, run_id,
        preferences=preferences,
    )


def _get_handoff_history(op_log: Any) -> str:
    """Build a handoff history text from previous data-prep and verifier attempts.

    Cross-references verifier feedback with matching data-prep attempts by
    attempt number so the Preparer sees what the Verifier found on retry.
    """
    from .operation_log import OperationLog
    if not isinstance(op_log, OperationLog):
        return ""
    prep_attempts = op_log.get_stage_attempts("data-prep")
    verifier_attempts = op_log.get_stage_attempts("verifier")
    verifier_by_n = {va.get("n"): va for va in verifier_attempts}
    lines: list[str] = []
    for ad in prep_attempts:
        n = ad.get("n", 0)
        h = ad.get("handoff")
        if h:
            lines.append(f"Attempt {n} preparer handoff: {json.dumps(h, ensure_ascii=False)}")
        va = verifier_by_n.get(n)
        if va:
            fb = va.get("agent_decision")
            if fb:
                lines.append(f"Attempt {n} verifier feedback: {json.dumps(fb, ensure_ascii=False)}")
    return "\n".join(lines)


def _read_worker_handoff(output_dir: str) -> dict[str, Any] | None:
    """Read the worker's handoff from worker_report.json and convert to a PrepHandoff dict."""
    worker_path = os.path.join(output_dir, _PREP_WORKER_REPORT_NAME)
    if not os.path.exists(worker_path):
        sys_log.warning("No worker report at %s, cannot build handoff", worker_path)
        return None
    try:
        with open(worker_path, "r", encoding="utf-8") as f:
            report = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        sys_log.warning("Failed to read worker report for handoff: %s", e)
        return None
    from .schema import PrepHandoff
    return PrepHandoff.from_worker_report(report).to_dict()


def _write_verifier_result(output_dir: str, result: dict[str, Any] | None) -> None:
    """Write the verifier's structured feedback to verifier_result.json."""
    if result is None:
        result = {"overall_ok": False, "summary": "Verifier produced no output"}
    verifier_path = os.path.join(output_dir, _PREP_VERIFIER_RESULT_NAME)
    try:
        with open(verifier_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
    except Exception as e:
        sys_log.error("Failed to write %s: %s", _PREP_VERIFIER_RESULT_NAME, e)


def _check_output(output_dir: str) -> tuple[bool, list[str]]:
    """Programmatic CRITICAL/WARNING check of prepared data directory.

    Returns (ok, issues). ok=False means a CRITICAL issue blocks workflow.
    WARNING issues do not set ok=False.
    """
    issues: list[str] = []
    critical = False
    base = Path(output_dir)

    # CRITICAL: at least one IV family file with long sheet and required columns
    iv_files = sorted(base.glob("*_train.xlsx"))
    iv_files = [f for f in iv_files if "CV_train" not in f.name]
    if not iv_files:
        issues.append("[CRITICAL] No IV family *_train.xlsx files found")
        critical = True
    else:
        for f in iv_files:
            try:
                import openpyxl
                wb = openpyxl.load_workbook(str(f), read_only=True, data_only=True)
                if _PREP_IV_SHEET_NAME not in wb.sheetnames:
                    issues.append(f"[CRITICAL] {f.name}: missing '{_PREP_IV_SHEET_NAME}' sheet")
                    critical = True
                    wb.close()
                    continue
                ws = wb[_PREP_IV_SHEET_NAME]
                headers = [cell.value for cell in next(ws.iter_rows(min_row=1, max_row=1))]
                required = set(_PREP_REQUIRED_IV_COLUMNS)
                present = set(h or "" for h in headers)
                missing = required - present
                if missing:
                    issues.append(f"[CRITICAL] {f.name}: missing columns {sorted(missing)}")
                    critical = True
                wb.close()
            except Exception as e:
                issues.append(f"[CRITICAL] {f.name}: cannot read — {e}")
                critical = True

    # WARNING: CV file missing
    cv_file = base / "CV_train.xlsx"
    if not cv_file.exists():
        issues.append("[WARNING] CV_train.xlsx not found (optional)")
    else:
        try:
            import openpyxl
            wb = openpyxl.load_workbook(str(cv_file), read_only=True, data_only=True)
            expected_sheets = set(_PREP_CV_EXPECTED_SHEETS)
            found = set(wb.sheetnames) & expected_sheets
            missing_sheets = expected_sheets - found
            if missing_sheets:
                issues.append(f"[WARNING] CV_train.xlsx: missing sheets {sorted(missing_sheets)}")
            wb.close()
        except Exception as e:
            issues.append(f"[WARNING] CV_train.xlsx: cannot read — {e}")

    # WARNING: I-t directory missing
    it_dir = base / "it"
    if not it_dir.exists() or not it_dir.is_dir():
        issues.append("[WARNING] it/ subdirectory not found (optional)")
    else:
        it_files = list(it_dir.iterdir())
        if len(it_files) == 0:
            issues.append("[WARNING] it/ subdirectory is empty (optional)")

    ok = not critical
    return ok, issues


def _build_manifest(output_dir: str, prep_id: int, ok: bool, issues: list[str]) -> dict[str, Any]:
    import datetime
    return {
        "prep_id": prep_id,
        "ok": ok,
        "issues": issues,
        "created_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


def _write_manifest(output_dir: str, manifest: dict[str, Any]) -> None:
    manifest_path = os.path.join(output_dir, _PREP_MANIFEST_NAME)
    try:
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
    except Exception as e:
        sys_log.error("Failed to write %s: %s", _PREP_MANIFEST_NAME, e)


def _write_file_manifest(output_dir: str, prep_id: int) -> None:
    """Read worker_report.json and produce a file provenance manifest.

    Extracts the file_mapping array from the worker subagent's report
    and enriches it with output directory metadata.
    """
    import datetime
    worker_path = os.path.join(output_dir, _PREP_WORKER_REPORT_NAME)
    if not os.path.exists(worker_path):
        sys_log.warning("No worker report found at %s, skipping file manifest", worker_path)
        return

    try:
        with open(worker_path, "r", encoding="utf-8") as f:
            report = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        sys_log.warning("Failed to read worker report for file manifest: %s", e)
        return

    file_mapping = report.get("file_mapping", [])
    if not file_mapping:
        sys_log.info("No file_mapping in worker report, scanning output directory instead")
        output_files = sorted(Path(output_dir).glob("*_train.xlsx"))
        file_mapping = [{"source": "unknown", "output": f.name, "source_type": "unknown"} for f in output_files]

    manifest = {
        "prep_id": prep_id,
        "output_dir": str(Path(output_dir).resolve()),
        "created_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_outputs": len(file_mapping),
        "duplicates": [m for m in file_mapping if m.get("duplicate_of")],
        "files": file_mapping,
    }

    manifest_path = os.path.join(output_dir, _PREP_FILE_MANIFEST_NAME)
    try:
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        sys_log.info("File manifest written: %d files, %d duplicates", len(file_mapping), len(manifest["duplicates"]))
    except Exception as e:
        sys_log.error("Failed to write %s: %s", _PREP_FILE_MANIFEST_NAME, e)


# ---------------------------------------------------------------------------
# Orchestrator stage functions — called by while+pointer loop (Step 7)
# ---------------------------------------------------------------------------

def run_preparer_stage(
    ctx: AgentContext,
    raw_dataset_dir: str,
    output_dir: str,
    handoff_history: str = "",
) -> dict[str, Any]:
    """Run the Preparer (worker) stage as a standalone pipeline step.

    Spawns a worker SubAgent to process raw measurement data into
    standardized format.  Produces a handoff dict for the Verifier.

    Returns:
        {"ok": bool, "handoff": dict | None, "error": str}
    """
    from .schema import PrepHandoff

    try:
        handoff = _run_worker(ctx, raw_dataset_dir, output_dir, handoff_history_text=handoff_history)
    except Exception as e:
        sys_log.error("Preparer stage failed: %s", e)
        return {"ok": False, "handoff": None, "error": str(e)}

    formatted_handoff = PrepHandoff.from_worker_report(handoff or {}).to_dict()
    return {"ok": True, "handoff": formatted_handoff, "error": ""}


def run_verifier_stage(
    ctx: AgentContext,
    output_dir: str,
    report_verification_tool_name: str = "report_verification",
    handoff: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the Verifier stage as a standalone pipeline step.

    Spawns a Verifier SubAgent to inspect the prepared data and produce
    structured feedback.

    Args:
        ctx: Agent context.
        output_dir: Directory containing prepared files to verify.
        report_verification_tool_name: Name constant for the report_verification tool.
        handoff: The Preparer's handoff dict to inject into the Verifier's prompt.

    Returns:
        {"ok": bool, "feedback": dict | None, "issues": list[str]}
    """
    verifier_feedback = _run_verifier_subagent(ctx, output_dir, report_verification_tool_name, handoff=handoff)
    _write_verifier_result(output_dir, verifier_feedback)

    programmatic_ok, issues = _check_output(output_dir)
    vf_ok = verifier_feedback.get("overall_ok", False) if verifier_feedback else False
    all_ok = programmatic_ok and vf_ok

    return {
        "ok": all_ok,
        "feedback": verifier_feedback,
        "issues": issues if issues else [],
    }


def copy_dataset_to_data_prep(
    dataset_dir: str,
    attempt_dir: str,
) -> None:
    """Copy an existing standard dataset into a data-prep attempt directory."""
    import shutil
    os.makedirs(attempt_dir, exist_ok=True)
    for item in Path(dataset_dir).iterdir():
        src = str(item)
        dst = os.path.join(attempt_dir, item.name)
        if item.is_dir():
            shutil.copytree(src, dst, dirs_exist_ok=True)
        elif item.is_file():
            shutil.copy2(src, dst)
