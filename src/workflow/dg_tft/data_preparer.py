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

from src.utility import client
from src.context.agent_context import AgentContext
from src.context.prompt import deepseek_support
from src.constants import WORKER_AGENT_LABEL

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
_PREP_VERIFIER_MAX_TURNS = 12
_PREP_VERIFIER_MODEL_TIER = "fast"
_PREP_OUTPUT_BASE_DIR = "prepared_data"
_PREP_AGENT_ID_PREFIX = "prep_worker"
_PREP_SUBJECT = "DG-TFT data preparation"

WORKER_PROMPT_TEMPLATE = """\
You are a data preparation specialist for DG-TFT device modeling. Convert raw measurement data into the standardized format required by the DG-TFT modeling pipeline.

**Raw data directory**: {raw_dir}
**Output directory**: {output_dir}

The output directory already exists. Write all output files into it.

**Required output format**:
- IV training file suffix: *_train.xlsx
- Required sheet name: {iv_sheet}
- Required columns: {iv_columns}
- Report filename: {report_name}
- CV output: CV_train.xlsx (expected sheets: {cv_sheets})
- I-t files go to: it/ subdirectory

**Python usage rules**:
- Use `{python_exe} -c "..."` for all inline Python code — do NOT write scripts to files
- Always import libraries at the start of every `-c` command: `import pandas as pd, openpyxl, os, shutil, json`
- CSV columns may have leading spaces — use `df.columns = df.columns.str.strip()` to clean them

**Step-by-step instructions**:

1. First, check what files exist in `{raw_dir}` and `{raw_dir}/it/`. One brief `glob_file` call is enough.

2. Process each file by data type:

   **IV CSV files** (e.g., `IDVBG VDS01V.csv`):
   - Read: `df = pd.read_csv("{raw_dir}/IDVBG VDS01V.csv")`
   - Clean: `df.columns = df.columns.str.strip(); df.columns = [c.replace("(V)","_V").replace("(A)","_A") for c in df.columns]`
   - Save: `df.to_excel("{output_dir}/IDVBG_VDS01V_train.xlsx", sheet_name="{iv_sheet}", index=False)`

   **IV XLSX files** (e.g., `IDVBG_VDS0.1V.xlsx`):
   - Same steps as CSV but use `pd.read_excel()`. Clean column names identically.

   **Family mapping rules**:
   - `IDVBG_*` -> `IDVBG_VDS<Vvalue>_V_train.xlsx` (extract V from filename)
   - `IDVTG_*` -> `IDVTG_VDS<Vvalue>_V_train.xlsx`
   - `IDVD_*VBG*` -> `IDVD_VBG<Vvalue>_V_train.xlsx` (CRITICAL: check for VBG in name)
   - `IDVD_*VTG*` -> `IDVD_VTG<Vvalue>_V_train.xlsx` (CRITICAL: check for VTG in name)
   - Skip: `DUAL_REF_*.csv`, `Refernce.xlsx`
   - **IMPORTANT**: If both CSV and XLSX exist for the same measurement (same V value), only process the XLSX version. Normalize V values: 01V = 0.1V, 05V = 0.5V, 5V = 5V. Check if output file already exists before writing and skip duplicates.

3. **CV file** (e.g., `CV_wide_tables.xlsx`):
   - Read and save as `CV_train.xlsx` with original sheets
   - `df = pd.read_excel("{raw_dir}/CV_wide_tables.xlsx", sheet_name=None); with pd.ExcelWriter("{output_dir}/CV_train.xlsx") as writer: for name, sheet in df.items(): sheet.to_excel(writer, sheet_name=name[:31], index=False)`

4. **I-t files** in `{raw_dir}/it/`:
   - Copy all files to `{output_dir}/it/`: `shutil.copytree("{raw_dir}/it", "{output_dir}/it")`

5. Write `{report_name}` in `{output_dir}` with this exact format:
```json
{{
  "files_processed": ["output_file1.xlsx", "output_file2.xlsx"],
  "file_mapping": [
    {{"source": "IDVBG VDS01V.csv", "output": "IDVBG_VDS0.1V_train.xlsx", "source_type": "csv", "rows": 5117, "duplicate_of": null}},
    {{"source": "IDVBG_VDS0.1V.xlsx", "output": "IDVBG_VDS0.1V_train.xlsx", "source_type": "xlsx", "rows": 5117, "duplicate_of": null}}
  ],
  "iv_families": ["IDVBG", "IDVD", "IDVTG"],
  "cv_found": true,
  "it_found": true,
  "issues": ["..."]
}}
```
IMPORTANT: The `file_mapping` array MUST include every file processed. Set `duplicate_of` to the output filename if this source was skipped because another source already produced the same output. If no duplicate, set `duplicate_of` to null. Set `rows` to the actual row count written to the output file.

**Each file should take ONE python -c command. After processing all files, write {report_name} and output a brief summary.**

Do NOT ask the user any questions — work autonomously. Do NOT explore files excessively — just read, clean, and save.
"""

VERIFIER_PROMPT_TEMPLATE = """\
You are a verification specialist for DG-TFT data preparation. A worker has processed raw measurement data into standardized training files. Your job is to inspect the output directory and report findings in a structured format via the report_verification tool.

**Prepared data directory**: {output_dir}

**Instructions**:
1. Use `glob_file` to list all files in the output directory (including the it/ subdirectory).
2. For each `*_train.xlsx` IV family file (NOT CV_train.xlsx), use `read_file` to check that it has a "{iv_sheet}" sheet with columns: {iv_columns}.
3. For `CV_train.xlsx`, check that it exists and report its sheets (expected: {cv_sheets}).
4. For the `it/` subdirectory, check if it exists and how many files it contains.
5. After inspecting, call `report_verification` with your structured findings.

**CRITICAL**: Call the `report_verification` tool exactly once at the end. Do not output free text. Be efficient — check a few representative files, not every single one.
"""


def _build_verifier_tools() -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": "glob_file",
                "description": "List files in a directory matching a pattern.",
                "parameters": {
                    "type": "object",
                    "required": ["path", "pattern"],
                    "additionalProperties": False,
                    "properties": {
                        "path": {"type": "string", "description": "Directory path to search."},
                        "pattern": {"type": "string", "description": "Glob pattern, e.g. * or *.xlsx."},
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read first few lines of an xlsx file to inspect sheet names and headers.",
                "parameters": {
                    "type": "object",
                    "required": ["file_path"],
                    "additionalProperties": False,
                    "properties": {
                        "file_path": {"type": "string", "description": "Absolute path to the .xlsx file."},
                    },
                },
            },
        },
        build_report_verification_tool(),
    ]


def _execute_verifier_tool(func_name: str, arguments: dict) -> str:
    """Execute a verifier tool inline — returns JSON string result."""
    if func_name == "glob_file":
        base = Path(arguments["path"])
        pattern = arguments.get("pattern", "*")
        matches = sorted(base.glob(pattern))
        result = [str(m.relative_to(base)) if m.is_relative_to(base) else str(m) for m in matches]
        return json.dumps({"files": result, "count": len(result)})
    elif func_name == "read_file":
        try:
            import openpyxl
            wb = openpyxl.load_workbook(arguments["file_path"], read_only=True, data_only=True)
            sheets = {}
            for name in wb.sheetnames:
                ws = wb[name]
                headers = [cell.value for cell in next(ws.iter_rows(min_row=1, max_row=1))]
                sheets[name] = {"columns": [str(h) for h in headers]}
            wb.close()
            return json.dumps(({"sheets": sheets, "sheet_count": len(sheets)}))
        except Exception as e:
            return json.dumps({"error": str(e)})
    else:
        return json.dumps({"error": f"Unknown tool: {func_name}"})


def build_report_verification_tool() -> dict[str, Any]:
    """Build the report_verification tool definition for the verifier LLM."""
    return {
        "type": "function",
        "function": {
            "name": "report_verification",
            "description": (
                "Report the verification findings for prepared DG-TFT dataset. "
                "Call this tool with your structured results — do not output free text."
            ),
            "parameters": {
                "type": "object",
                "required": ["overall_ok"],
                "additionalProperties": False,
                "properties": {
                    "overall_ok": {
                        "type": "boolean",
                        "description": "True if all CRITICAL checks pass; False if any required file/sheet/column is missing."
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
                    },
                    "cv_check": {
                        "type": "object",
                        "properties": {
                            "file_found": {"type": "boolean"},
                            "sheets_ok": {"type": "object"},
                            "issues": {"type": "array", "items": {"type": "string"}},
                        },
                    },
                    "it_check": {
                        "type": "object",
                        "properties": {
                            "dir_found": {"type": "boolean"},
                            "file_count": {"type": "integer"},
                            "issues": {"type": "array", "items": {"type": "string"}},
                        },
                    },
                    "summary": {"type": "string"},
                },
            },
        },
    }


def run_data_preparation(ctx: AgentContext, raw_dataset_dir: str, output_dir: str) -> None:
    """Run the full data preparation pipeline: worker -> verifier -> check.

    Called synchronously from the tool handler. Results are written to
    manifest.json and registered via PreparationTracker.

    Note: prep_id is derived from output_dir (the handler already registered it).
    """
    prep_id = int(Path(output_dir).name)
    console = getattr(ctx, "console", None)
    sys_log.info("Data prep %d started: raw=%s output=%s", prep_id, raw_dataset_dir, output_dir)

    if console and not getattr(ctx, "subagent_mute", False):
        console.print(f"[bold]DG-TFT Data Preparation {prep_id}[/bold] [bright_black]— converting raw measurements to *_train.xlsx format[/bright_black]")

    try:
        if console and not getattr(ctx, "subagent_mute", False):
            console.print("  [bold]Step 1/3[/bold] [bright_black]Worker subagent processing raw files ...[/bright_black]")
        _run_worker(ctx, raw_dataset_dir, output_dir)

        if console and not getattr(ctx, "subagent_mute", False):
            console.print("  [bold]Step 2/3[/bold] [bright_black]Verifier checking output structure ...[/bright_black]")
        _run_verifier(ctx, output_dir)

    except Exception as e:
        sys_log.error("Data prep %d failed: %s", prep_id, e)
        if console and not getattr(ctx, "subagent_mute", False):
            console.print(f"  [bold red]Failed[/bold red]: {e}")
        manifest = _build_manifest(output_dir, prep_id, False, [str(e)])
        _write_manifest(output_dir, manifest)
        ctx.workflow_man.preparation_tracker.update_prep(prep_id, False, [str(e)])
        return

    ok, issues = _check_output(output_dir)
    manifest = _build_manifest(output_dir, prep_id, ok, issues)
    _write_manifest(output_dir, manifest)
    _write_file_manifest(output_dir, prep_id)
    ctx.workflow_man.preparation_tracker.update_prep(prep_id, ok, issues)

    if console and not getattr(ctx, "subagent_mute", False):
        if ok:
            console.print(f"  [bold green]Step 3/3[/bold green] [bright_black]Verification passed ({len(issues)} warnings)[/bright_black]")
        else:
            console.print(f"  [bold red]Step 3/3[/bold red] [bright_black]Verification failed[/bright_black]")
            for issue in issues:
                if "CRITICAL" in issue:
                    console.print(f"    [bold red]{issue}[/bold red]")
                else:
                    console.print(f"    [bold yellow]{issue}[/bold yellow]")

    sys_log.info("Data prep %d done: ok=%s issues=%s", prep_id, ok, issues)


def _run_worker(ctx: AgentContext, raw_dir: str, output_dir: str) -> None:
    """Launch a worker subagent to process raw data into standardized format.

    The worker runs synchronously in the current thread with its own
    SubAgent instance and tool loop.
    """
    modeling_exe = ctx.agent_configs.get("DG_TFT_MODELING_PYTHON_EXE", "python")
    python_exe = modeling_exe if modeling_exe and Path(modeling_exe).exists() else "python"
    prompt = WORKER_PROMPT_TEMPLATE.format(
        raw_dir=raw_dir, output_dir=output_dir,
        iv_sheet=_PREP_IV_SHEET_NAME,
        iv_columns=", ".join(_PREP_REQUIRED_IV_COLUMNS),
        cv_sheets=", ".join(_PREP_CV_EXPECTED_SHEETS),
        report_name=_PREP_WORKER_REPORT_NAME,
        python_exe=python_exe,
    )
    from src.agent.subagent import SubAgent

    agent = SubAgent(
        parent_ctx=ctx,
        subagent_type=WORKER_AGENT_LABEL,
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
        ctx.agent_list.pop(agent.agent_id, None)

    if agent.status.value == "done":
        sys_log.info("Data prep worker completed: %s", result[:200] if result else "(empty)")
    else:
        err = agent.error or "unknown error"
        raise RuntimeError(f"Worker subagent failed: {err}")

    worker_report_path = os.path.join(output_dir, _PREP_WORKER_REPORT_NAME)
    if result:
        try:
            with open(worker_report_path, "w", encoding="utf-8") as f:
                f.write(result)
        except Exception:
            pass


def _run_verifier(ctx: AgentContext, output_dir: str) -> None:
    """Run a lightweight LLM verifier with a mini tool execution loop.

    The verifier has glob_file, read_file (xlsx), and report_verification tools.
    It inspects the output directory and produces a structured report.
    """
    system = {"role": "system", "content": "You are a data verification assistant. Use glob_file and read_file to inspect the output directory, then call report_verification with your structured findings."}
    user = {"role": "user", "content": VERIFIER_PROMPT_TEMPLATE.format(
        output_dir=output_dir,
        iv_sheet=_PREP_IV_SHEET_NAME,
        iv_columns=", ".join(_PREP_REQUIRED_IV_COLUMNS),
        cv_sheets=", ".join(_PREP_CV_EXPECTED_SHEETS),
    )}
    messages = [system, user]

    tools = _build_verifier_tools()
    model_tier: str = ctx.agent_configs.get("DG_TFT_PREP_VERIFIER_MODEL_TIER", _PREP_VERIFIER_MODEL_TIER)
    branch_func, deepseek_key = client.select_branch_func(model_tier)

    max_turns = ctx.agent_configs.get("DG_TFT_VERIFIER_MAX_TURNS", _PREP_VERIFIER_MAX_TURNS)
    if not isinstance(max_turns, int):
        max_turns = _PREP_VERIFIER_MAX_TURNS
    verifier_result = None
    finished = False

    for turn in range(max_turns):
        turns_left = max_turns - turn - 1
        if turns_left == 1:
            messages.append({
                "role": "user",
                "content": "WARNING — Only 1 turn remaining. Call report_verification with your findings NOW. Do not call any other tools.",
            })
        elif turns_left == 2:
            messages.append({
                "role": "user",
                "content": "Only 2 turns remaining. Begin wrapping up — call report_verification soon.",
            })

        try:
            response = branch_func(ctx.llm_client, messages, tools, ctx.api_configs)
        except Exception as e:
            sys_log.warning("Verifier LLM call failed at turn %d: %s", turn, e)
            break

        dumped = response.choices[0].message.model_dump(mode="json")
        if ctx.api_configs.get(deepseek_key, False):
            dumped = deepseek_support(dumped)

        assistant_content = dumped.get("content", "")
        tc_list = dumped.get("tool_calls")

        if tc_list:
            for tc in tc_list:
                func_name = tc["function"]["name"]
                try:
                    func_args = json.loads(tc["function"]["arguments"])
                except Exception:
                    func_args = {}

                if func_name == "report_verification":
                    verifier_result = func_args
                    break

                tool_result = _execute_verifier_tool(func_name, func_args)
                messages.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [tc],
                })
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": tool_result,
                })

            if verifier_result is not None:
                finished = True
                break
        elif assistant_content:
            sys_log.info("Verifier returned text without tool call: %s", assistant_content[:200])
            messages.append({"role": "assistant", "content": assistant_content})
            messages.append({
                "role": "user",
                "content": "You must call the report_verification tool to submit your findings. Do not output free text.",
            })
            continue
        else:
            sys_log.warning("Verifier returned empty response at turn %d", turn)
            break

    verifier_path = os.path.join(output_dir, _PREP_VERIFIER_RESULT_NAME)
    if verifier_result:
        try:
            with open(verifier_path, "w", encoding="utf-8") as f:
                json.dump(verifier_result, f, indent=2, ensure_ascii=False)
        except Exception:
            pass
    else:
        sys_log.warning("Verifier exhausted %d turns without calling report_verification", max_turns)
        fallback = {"overall_ok": False, "summary": f"Verifier did not call report_verification within {max_turns} turns"}
        try:
            with open(verifier_path, "w", encoding="utf-8") as f:
                json.dump(fallback, f, indent=2, ensure_ascii=False)
        except Exception:
            pass


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
