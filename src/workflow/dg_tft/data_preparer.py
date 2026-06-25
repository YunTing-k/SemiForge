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

sys_log = logging.getLogger("logger")

WORKER_PROMPT_TEMPLATE = """\
You are a data preparation specialist for DG-TFT device modeling. Convert raw measurement data into the standardized format required by the DG-TFT modeling pipeline.

**Raw data directory**: {raw_dir}
**Output directory**: {output_dir}

The output directory already exists. Process files and write outputs into it.

**Python setup (use bash to run Python scripts)**:
- Always start every Python command with: `import pandas as pd, openpyxl, os, shutil`
- Use `pd.read_csv()` for CSV files, `pd.read_excel()` for XLSX files
- Columns in CSV have a leading space — use `df.columns = df.columns.str.strip()` to clean them

**Step-by-step instructions**:

1. Scan `{raw_dir}` for all files. Identify data type by filename and content:
   - **IV files**: `IDVBG_*`, `IDVD_*`, `IDVTG_*` (both CSV and XLSX)
   - **CV files**: `CV_*` (e.g., `CV_wide_tables.xlsx`)
   - **I-t files**: `I-t*` files in the `it/` subdirectory
   - Skip: `DUAL_REF_*.csv` and `Refernce.xlsx`

2. For CSV IV files (e.g., `IDVBG VDS01V.csv`):
   - Read with pandas, strip column name whitespace, rename columns to standard names:
     - If header contains lowercase: use `df.columns = [c.strip().upper() for c in df.columns]` then map `VD(V) -> VD_V`, `VBG(V) -> VBG_V`, `VTG(V) -> VTG_V`, `ID(A) -> ID_A`
   - Save as `*_train.xlsx` with sheet name "long"

3. For XLSX IV files (e.g., `IDVBG_VDS0.1V.xlsx`):
   - Read, clean column names similarly, save as `*_train.xlsx` with sheet "long"

4. **Family mapping — use the ORIGINAL FILENAME to determine family**:
   - Files with `IDVBG` in name → output: `IDVBG_VDS<Vvalue>_V_train.xlsx` (extract V after VDS from filename)
   - Files with `IDVTG` in name → output: `IDVTG_VDS<Vvalue>_V_train.xlsx`
   - Files with `IDVD` AND `VBG` in name → output: `IDVD_VBG<Vvalue>_V_train.xlsx`
   - Files with `IDVD` AND `VTG` in name → output: `IDVD_VTG<Vvalue>_V_train.xlsx`
   - **CRITICAL**: Preserve whether the file is VBG or VTG — do NOT mix them up.

5. Each output `*_train.xlsx` MUST have:
   - One sheet named "long"
   - Columns: VD_V, VBG_V, VTG_V, ID_A (in that order)
   - Missing columns → fill with 0

6. If CV file found (e.g., `CV_wide_tables.xlsx`):
   - Read it, check what sheets it has, save as `CV_train.xlsx`
   - Preserve all original sheets (cgg, cbg, ctg, vds if present)
   - If it has only one sheet, name it `cgg`

7. I-t files in `{raw_dir}/it/` → copy into `{output_dir}/it/` (already present, verify)

8. Write `worker_report.json` in `{output_dir}`:
```json
{{
  "files_processed": ["..."],
  "iv_families": ["IDVBG", "IDVD", "IDVTG"],
  "cv_found": true,
  "it_found": false,
  "issues": ["..."]
}}
```

**Constraints**:
- Start with `import pandas as pd, openpyxl, os, shutil` before any pandas operations
- Never modify original measurement values — only reformat column names
- Handle both CSV (space in filename) and XLSX inputs
- If a file cannot be read, log it in issues and skip
- Do NOT ask the user any questions — work autonomously
- When done, output a brief summary text describing what was produced
"""

VERIFIER_PROMPT_TEMPLATE = """\
You are a verification specialist for DG-TFT data preparation. A worker has processed raw measurement data into standardized training files. Your job is to inspect the output and report structural findings.

**Prepared data directory**: {output_dir}

**Instructions**:
1. List the contents of `{output_dir}` (including `it/` subdirectory if it exists).
2. For each `*_train.xlsx` IV family file, check:
   - Does the file exist?
   - Does it contain a "long" sheet?
   - Does the "long" sheet have columns: VD_V, VBG_V, VTG_V, ID_A?
3. For `CV_train.xlsx`, check:
   - Does the file exist?
   - Does it contain sheets: cgg, cbg, ctg, vds?
4. For I-t data, check:
   - Does the `it/` subdirectory exist?
   - How many files does it contain?

**CRITICAL**: You MUST call the `report_verification` tool to report your structured findings. Do not output free-text conclusions — use the tool. Call it exactly once.
"""


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

    Called from a background thread. Does not return a value; results
    are written to manifest.json and registered via PreparationTracker.

    Note: prep_id is derived from output_dir (the handler already registered it).
    """
    prep_id = int(Path(output_dir).name)
    sys_log.info("Data prep %d started: raw=%s output=%s", prep_id, raw_dataset_dir, output_dir)

    try:
        _run_worker(ctx, raw_dataset_dir, output_dir)
        _run_verifier(ctx, output_dir)
    except Exception as e:
        sys_log.error("Data prep %d failed: %s", prep_id, e)
        manifest = _build_manifest(output_dir, prep_id, False, [str(e)])
        _write_manifest(output_dir, manifest)
        ctx.workflow_man.preparation_tracker.update_prep(prep_id, False, [str(e)])
        return

    ok, issues = _check_output(output_dir)
    manifest = _build_manifest(output_dir, prep_id, ok, issues)
    _write_manifest(output_dir, manifest)
    ctx.workflow_man.preparation_tracker.update_prep(prep_id, ok, issues)
    sys_log.info("Data prep %d done: ok=%s issues=%s", prep_id, ok, issues)


def _run_worker(ctx: AgentContext, raw_dir: str, output_dir: str) -> None:
    """Launch a worker subagent to process raw data into standardized format.

    The worker runs synchronously in the current thread with its own
    SubAgent instance and tool loop.
    """
    prompt = WORKER_PROMPT_TEMPLATE.format(raw_dir=raw_dir, output_dir=output_dir)
    from src.agent.subagent import SubAgent

    agent = SubAgent(
        parent_ctx=ctx,
        subagent_type="worker",
        prompt=prompt,
        agent_id="prep_worker",
        subject="DG-TFT data preparation",
        console=None,
    )
    agent.build_tools()
    agent.build_messages()
    result = agent.run()

    if agent.status.value == "done":
        sys_log.info("Data prep worker completed: %s", result[:200] if result else "(empty)")
    else:
        err = agent.error or "unknown error"
        raise RuntimeError(f"Worker subagent failed: {err}")

    worker_report_path = os.path.join(output_dir, "worker_report.json")
    if result:
        try:
            with open(worker_report_path, "w", encoding="utf-8") as f:
                f.write(result)
        except Exception:
            pass


def _run_verifier(ctx: AgentContext, output_dir: str) -> None:
    """Run an independent LLM call to verify the prepared data structure.

    Uses a custom report_verification tool to force structured output.
    Includes retry logic for APIs that don't support tool_choice.
    """
    messages = [
        {"role": "system", "content": "You are a verification assistant. Always call the report_verification tool to output structured results."},
        {"role": "user", "content": VERIFIER_PROMPT_TEMPLATE.format(output_dir=output_dir)},
    ]

    tools = [build_report_verification_tool()]
    model_tier: str = ctx.agent_configs.get("LLM_SUMMARY_MODEL", "fast")
    branch_func, deepseek_key = client.select_branch_func(model_tier)
    if ctx.api_configs.get(deepseek_key, False):
        tool_choice = None
    else:
        tool_choice = {
            "type": "function",
            "function": {"name": "report_verification"},
        }

    max_retries = 3
    tool_calls_result: list[dict[str, Any]] | None = None

    for attempt in range(1, max_retries + 1):
        try:
            response = branch_func(
                ctx.llm_client,
                messages,
                tools,
                ctx.api_configs,
                tool_choice=tool_choice,
            )
        except Exception as e:
            sys_log.warning("Verifier LLM call attempt %d failed: %s", attempt, e)
            if attempt == max_retries:
                break
            continue

        dumped = response.choices[0].message.model_dump(mode="json")
        if ctx.api_configs.get(deepseek_key, False):
            dumped = deepseek_support(dumped)

        tc_list: list[dict[str, Any]] | None = dumped.get("tool_calls")
        if tc_list and len(tc_list) > 0:
            tool_calls_result = []
            for tc in tc_list:
                try:
                    args = json.loads(tc["function"]["arguments"])
                    tc["function"]["parsed_args"] = args
                except Exception:
                    pass
                tool_calls_result.append(tc)
            break

        if attempt < max_retries:
            sys_log.warning("Verifier did not call report_verification, retry %d/%d", attempt, max_retries)
            messages.append({
                "role": "user",
                "content": "You MUST call the `report_verification` tool to report your results. Do not output free text. Call it now.",
            })

    verifier_path = os.path.join(output_dir, "verifier_result.json")
    if tool_calls_result:
        for tc in tool_calls_result:
            parsed = tc.get("function", {}).get("parsed_args")
            if parsed:
                try:
                    with open(verifier_path, "w", encoding="utf-8") as f:
                        json.dump(parsed, f, indent=2, ensure_ascii=False)
                    return
                except Exception:
                    pass

    fallback = {"overall_ok": False, "summary": "LLM refused to call report_verification tool after max retries"}
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
                if "long" not in wb.sheetnames:
                    issues.append(f"[CRITICAL] {f.name}: missing 'long' sheet")
                    critical = True
                    wb.close()
                    continue
                ws = wb["long"]
                headers = [cell.value for cell in next(ws.iter_rows(min_row=1, max_row=1))]
                required = {"VD_V", "VBG_V", "VTG_V", "ID_A"}
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
            expected_sheets = {"cgg", "cbg", "ctg", "vds"}
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
    manifest_path = os.path.join(output_dir, "manifest.json")
    try:
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
    except Exception as e:
        sys_log.error("Failed to write manifest.json: %s", e)
