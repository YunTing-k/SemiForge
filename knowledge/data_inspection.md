# Data Inspection Knowledge

## 1. Global Context

### Pipeline Overview

The DG-TFT modeling pipeline has 7 stages:

```
data-prep → verifier → check-data → fit-physical → train-static → train-hysteresis → final-report
```

| Stage | Kind | Has Modeling Agent | Triggers Agent On |
|-------|------|--------------------|-------------------|
| data-prep | agent | No | — (always runs) |
| verifier | agent | No | — (always runs) |
| **check-data** | **script** | **Diagnostic (Agent A)** | **Hard failure only** |
| fit-physical | script | Yes (Agent B) | Hard failure / quality below threshold |
| train-static | script | Yes (Agent C) | Hard failure / quality below threshold |
| train-hysteresis | script | Yes (Agent D) | Hard failure / quality below threshold |
| final-report | script | No (Agent E) | — (always runs at end) |

Note: Agent A is a **diagnostic agent** distinct from the modeling agents (B/C/D). Agent A has only one tool (`fallback_to_previous`) and uses `stop_mode="schema"` — it does not produce free text. Modeling agents (B/C/D) additionally have `adjust_params`, `accept_quality`, and `inspect_xlsx`.

Script stages run via subprocess; fit-physical uses scipy (CPU-only, no CUDA). train-static, train-hysteresis use PyTorch GPU.
When a script stage hard-fails (exit code != 0) or produces quality below threshold,
a **modeling agent** is spawned to adjust parameters and retry, or fallback upstream.

Fallback is a linear upstream chain: D→C→B→A (check-data) → Verifier → Preparer.
Each agent's fallback target: D→train-static, C→fit-physical, B→check-data, A→Verifier.
This means two hops per agent: e.g., D→C→B→check-data verification → Verifier → data re-prep.

### check-data Position and Role

The **check-data** stage is Stage 3 of 7 — the first script-execution stage,
immediately after the Verifier agent stage. It performs **structural validation**
of the dataset files (IV/CV/I-t) before any heavy modeling computation begins.

This stage has **no adjustable INI parameters** and **no quality thresholds**.
It is a binary gate: pass (all files structurally correct) or hard-fail.

On hard failure, **Agent A** is spawned with a single tool: `fallback_to_previous`.
Agent A decides whether the failure originates from data preparation (→ fallback
to Verifier → potentially Preparer) or is environmental (→ no fallback, pipeline terminates).

**Resume/Skip**: If check-data previously completed successfully (a `StageResult` with `ok=True` and `skipped=False`
exists in the run's task board), it is skipped via `_stage_done()` check — it will not re-run.

### Principles: What Is Being Checked

DG-TFT devices are characterized by three types of electrical measurements:

**IV (Current-Voltage) measurements** — three families:
- `IDVBG`: ID vs VBG sweep, at fixed VDS voltages. Transfer characteristics.
  Naming: `IDVBG VDS0.1V_train.xlsx` (VDS=0.1V), `IDVBG VDS5V_train.xlsx` (VDS=5V).
- `IDVTG`: ID vs VTG sweep, at fixed VDS voltages. Top-gate transfer.
- `IDVD`: ID vs VD sweep. Output characteristics. Two sub-families:
  `IDVD VTG*` (sweep VD at fixed VTG) and `IDVD VBG*` (sweep VD at fixed VBG).

Each IV workbook must contain a **"long" sheet** — every row is one bias point
with standard column names. The engine expects this format; other layouts
(e.g., pivot tables, multi-column VTG sweeps) will fail.

**CV (Capacitance-Voltage) measurements** — four standard sweeps:
- `cgg`: Gate-to-gate capacitance vs bias
- `cbg`: Bottom-gate capacitance vs bias
- `ctg`: Top-gate capacitance vs bias
- `vds`: Drain-source capacitance vs bias

CV data comes in two layout variants:
- **Paired-column**: (bias1, cap1, bias2, cap2, ...) — directly usable
- **Shared-bias wide-table**: one bias column + N capacitance columns — requires
  normalization by the engine before validation

**I-t (Current-Time) hysteresis** — optional, for dynamic modeling:
- Time-domain measurements: `time`, `vds`, `vbg`, `vtg`, `temp`, `id`
- Required only if hysteresis modeling (train-hysteresis stage) is to run

The check-data stage validates that all these files exist, have expected sheets,
and contain required columns — it does NOT check data ranges, units, or physical
plausibility. Those are downstream concerns.

## 2. Upstream/Downstream

### Receives from Verifier

- A verified dataset directory (the Verifier's `attempt_dir`) containing:
  - IV workbooks: `IDVBG*_train.xlsx`, `IDVTG*_train.xlsx`, `IDVD*_train.xlsx`
  - CV workbook: `CV_train.xlsx`
  - Optional I-t directory: `it/` containing `I-t*.xlsx` or `I-t*.xls`
- `model_config.ini` (built lazily by the orchestrator from the first script stage)

### Delivers to fit-physical

- `dataset_validation.json` — per-family validation results (ok_static, ok_with_it flags)
- **Normalized** `CV_train.xlsx` — shared-bias wide-tables are converted to
  paired-column format **in-place** (the engine subprocess rewrites the file — the normalization is destructive and irreversible). The downstream engine reads the normalized file.
- Unchanged `model_config.ini` — check-data writes nothing to INI

### Impact on downstream

- If `ok_static = false`: fit-physical cannot proceed (IV or CV data is structurally invalid)
- If `ok_with_it = false` but `ok_static = true`: fit-physical and train-static can
  proceed, but train-hysteresis will be skipped (no valid I-t data)
- If all ok: full pipeline proceeds normally

## 3. INI Parameters

**This stage has no adjustable INI parameters.**

No `[check-data]` section exists in `model_config.ini`. The stage is not in
`_STAGE_PARAM_MAP` in `param_defs.py`. `get_stage_param_schema("check-data")`
returns `None`.

The only configurable value is the stage timeout:

| Key | Default | Source |
|-----|---------|--------|
| `DG_TFT_TIMEOUT_S_CHECK_DATA` | 600 (10 min) | `agent_configs.json` |

### Why no parameters?

check-data is a pure validation gate. It inspects file structure, not data
content or modeling behavior. Parameter tuning cannot fix a missing column or
wrong sheet name — only data re-preparation can.

## 4. Quality Metrics

**This stage produces no quality metrics.** The quality gate always passes:
`get_primary_metric("check-data")` returns empty string, so `_assess_stage_quality`
returns `True, []` unconditionally.

No entry exists in `_STAGE_THRESHOLD_KEYS` or `_METRIC_SOURCES`.

The only success criterion is: the subprocess exits with code 0 AND the
artifact `dataset_validation.json` is produced on disk.

Note: `minimal_run` also bypasses quality assessment entirely for all stages (returns True immediately).

## 5. Engine Behavior

### Execution Flow

The orchestrator spawns a subprocess. Inside that subprocess, `run_model.py` performs:

1. **CV format normalization** (`run_model.py:_normalize_cv_format`, inside the engine subprocess):
   - Reads `CV_train.xlsx`
   - Uses `looks_like_shared_bias_wide_table()` from `cv_preprocessing.py` to detect the layout:
     checks that header row 1 col 0 contains "bias" or "voltage", at least half of other columns
     contain "cap" or "(f)", and no other columns look like bias/voltage headers.
   - If shared-bias wide-table: rewrites columns to paired-column format using openpyxl
   - Writes back to the same file **in-place and irreversibly** — the original CV format is lost
   - Uses both the detection heuristic plus a secondary check before rewriting. **Caveat**: If `pd.read_excel(cv_path)` raises any Exception, `_normalize_cv_format` silently returns without rewriting the file. The original CV format is preserved but the error is NOT propagated to the orchestrator or Agent A.
   - The orchestrator cannot observe or control this

2. **`validate_dataset.py` is invoked twice within a single workflow run**:
   - **First invocation**: by `dataset_inspector.py` during config building (before the actual stage).
     This uses `capture_output=True` via `subprocess.run` to build a `DatasetProfile` for config generation.
     The output from this invocation is NOT the stage artifact.
   - **Second invocation**: the actual stage run, which produces `dataset_validation.json` as the stage artifact.
     This is the one that counts for pass/fail.
   - The agent should be aware that the first run's result only feeds `DatasetProfile` — it does not produce
     the stage's output and its errors are handled differently (at config-build time).

3. **Subprocess invocation** (the actual stage run):
   ```
   <modeling python> run_model.py check-data --config <model_config.ini>
   ```
   `run_model.py` internally calls `validate_data()` → `_normalize_cv_format()` → invokes `validate_dataset.py`.
   The engine is NOT invoked directly; data-dir and it-dir are resolved from the INI file by `run_model.py`.

4. **Post-execution**: The orchestrator verifies `dataset_validation.json` exists.
   If missing → artifact verification fails → stage marked as failed.

### The DatasetProfile

`inspect_dataset()` builds a `DatasetProfile` with these fields that the orchestrator uses downstream:

| Field | Type | Purpose |
|-------|------|---------|
| `static_ok`, `cv_ok` | bool | Per-category pass/fail |
| `iv_families` | list[str] | Detected family names (IDVBG, IDVTG, IDVD) |
| `has_temperature_iv` | bool | Temperature columns found in any IV file |
| `temperature_span_K` | float | Actual temperature span (not yet populated) |
| `has_hysteresis` | bool | Gate for skipping train-hysteresis |
| `hysteresis_workbooks` | list[str] | Detected I-t file paths |
| `issues` | list[str] | Aggregated validation issues |
| `total_files` | int | Count of IV + CV files |
| `suggested_hysteresis_dir` | str | Auto-discovered I-t directory path |
| `raw_validation` | dict | Raw JSON from validate_dataset.py |

The orchestrator uses `skip_if_profile_false` to skip stages, and the config builder uses
`has_temperature_iv`/`has_hysteresis` to determine temperature/hysteresis INI sections.

### I-t Directory Auto-Discovery

`_discover_it_dir()` searches with this priority:
1. User-specified `hysteresis_dir` (from orchestration parameters)
2. `dataset_dir/hysteresis_it`
3. `dataset_dir/<dir>` where name contains `hysteresis_it`, `it`, `hysteresis`, `I-t`, `it_data`, `hysteresis_data`, `hyst`
4. Sibling directories (under parent) matching keywords `it`, `hysteresis`, `i-t`

**Warning**: The sibling directory keyword `it` is a substring match — it can match unintended directories like `unit`, `digit`, `quality`, etc. If `has_hysteresis=True` but I-t data appears invalid, check if auto-discovery picked up an unrelated directory.

This determines whether I-t data is found or `has_hysteresis` is False.

### HSPICE Availability Check

During dataset inspection, `_check_hspice()` reads the `hspice_exe` path from the config template,
verifies it exists on disk, and appends a warning to `profile.issues` if not found. This affects
`deployment.run_hspice` in the generated config and whether SPICE equivalence validation runs downstream.

### What validate_dataset.py Does — Step by Step

**Step 1 — IV Validation** (3 families, sequential):
- For each family (`IDVBG`, `IDVTG`, `IDVD`):
  - Glob `*_train.xlsx` matching the family name
  - Open each workbook with `pd.ExcelFile()`
  - Reads only **20 rows** (`nrows=20`) for column detection — headers beyond row 20 will be missed
  - Verify sheet `"long"` exists
  - Verify required columns present (fuzzy matching):
    - `VD_V`: matches columns containing `"vd_v"` or `"vd"`
    - `VBG_V`: matches columns containing `"vbg_v"` or `"vbg"`
    - `VTG_V`: matches columns containing `"vtg_v"` or `"vtg"`
    - `ID_A`: matches columns containing `"id_a"` or `"id"`
  - Optionally detect temperature columns (two distinct alias sets):
    - K-scale: `temp_K`, `temperature_K`, `T_K`, `TEMP_K`
    - C-scale: `temp_C`, `temperature_C`, `T_C`, `TEMP_C`
    - Detection only records which are found — no scale conversion or validity check
    - If only C-scale aliases exist (no K-scale), temperature will be "detected" but downstream may use wrong absolute values
  - Report: `files` (found), `issues` (any failures)

**Step 2 — CV Validation**:
- Check `CV_train.xlsx` exists
- Reads only **8 rows** (`nrows=8`) for layout detection
- Verify 4 sheets present: `cgg`, `cbg`, `ctg`, `vds` (case-insensitive)
- For each sheet, detect layout (`looks_like_shared_bias_wide_table`) and count usable bias/capacitance pairs
- Report: `sheets_found`, `issues`

**Step 3 — I-t Validation** (optional, skipped if no `--it-dir`):
- Glob `I-t*.xlsx` / `I-t*.xls` in it_dir
- For each file, verify columns using **fuzzy substring matching** (column names are lowercased first):
  - `time`: `"time" in c` or `"t(" in c` or `c == "t"`
  - `vds`: `"vds" in c` or `"vd" in c`
  - `vbg`: `"vbg" in c` or `"bg" in c`
  - `vtg`: `"vtg" in c` or `"tg" in c`
  - `temp`: `"temp" in c` or `"t/t" in c` or `c` in `{"t", "t(k)", "t_k", "temperature", "temperature_k"}`
  - `id`: `"ids" in c` or `"id" in c` or `"current" in c`
- Note: This fuzzy matching is substantially more permissive than exact column name matching,
  and could produce false positives with non-standard column names
- Report: `files`, `issues`

**Step 4 — Write Output**:
- Combine all results into JSON
- Print to stdout (captured in logs)
- Write to `--json-out` path as `dataset_validation.json`

The script uses `sys.exit(0)` on success (all checks passed) and standard Python
exception propagation on failure (crash → exit code != 0 → error captured in stderr).

### Actual dataset_validation.json Structure

```json
{
  "data_dir": "...",
  "iv": [
    {
      "family": "IDVBG",
      "matched_files": ["IDVBG VDS0.1V_train.xlsx", ...],
      "files": [
        {"file": "...", "exists": true, "ok": true|false, "issues": [...], "temperature_columns": [...], "columns": [...]}
      ],
      "ok": true|false,
      "issues": []
    }
  ],
  "cv": {
    "file": "CV_train.xlsx",
    "exists": true|false,
    "ok": true|false,
    "issues": [...],
    "sheets": {
      "cgg": {"exists": true, "detected_layout": "shared_bias_wide", "usable_curve_pairs_preview": N, "issues": [...]},
      "cbg": {...},
      "ctg": {...},
      "vds": {...}
    }
  },
  "it": {
    "provided": true|false,
    "path": "...",
    "exists": true|false,
    "ok": true|false,
    "files": [...],
    "issues": [...]
  },
  "ok_static": true|false,
  "ok_with_it": true|false
}
```

### How Results Affect Pipeline Flow

| Scenario | `ok_static` | `ok_with_it` | Pipeline Effect |
|----------|-------------|--------------|-----------------|
| All pass | true | true | Full pipeline runs |
| IV or CV fail | false | false | Hard failure → Agent A → fallback or abort |
| I-t fail only | true | false | Static modeling proceeds (fit-physical, train-static). Hysteresis skipped. |
| Script crash | N/A | N/A | No JSON produced → stage failed → Agent A |

### Flow After check-data Failure

When check-data hard-fails:
1. `StageResult` with `ok=False` is recorded
2. `wm.update_stage()` marks the stage as `FAILED` on the task board
3. `_spawn_agent_a()` is called — Agent A decides on fallback
4. **If Agent A returns a fallback target**: the pointer is reset, task board entries reverted, and execution jumps upstream.
   Check-data → fallback → Verifier → may fallback further to Preparer → full data re-preparation → chain of re-execution.
5. **If Agent A returns no fallback** (`{"fallback": ""}`): `failed_stage`/`failed_error` are set, the retry loop breaks.
6. The pipeline does not immediately terminate — the outer `while pointer < max_pointer` loop continues.
   Downstream stages will attempt execution but typically fail quickly due to missing upstream artifacts.
7. Only after all stages are iterated does the orchestrator check `failed_stage` and call `_finalize_run(failed=True)`.

Note: `_run_stage_diagnosis()` does NOT run for check-data failures — the check-data hard-failure path uses `break` before reaching the diagnosis code. Only Agent A's fallback decision runs.

**Agent A prompt content**: Agent A's prompt (orchestrator.py) includes only `stage_result.error` (stderr, truncated to 2000 chars), NOT `stage_result.summary` (stdout last 10 lines which contains validate_dataset.py's JSON output). The agent must proactively `read_file dataset_validation.json` to see the actual validation results — the prompt alone is insufficient for diagnosis.

### Edge Case: Artifact Stage Resume

`validate_data()` in `run_model.py` uses `run_artifact_stage()` instead of `run()`. This means:
if `validate_dataset.py` fails but the expected artifacts already exist on disk from a prior run,
the failure is recorded as non-blocking. The stage may appear to succeed even though the
subprocess crashed, provided existing artifacts are present. **Note**: The expected artifacts checked are NOT `dataset_validation.json` — they are `v93_safe_residual_state.pt` and `v92_safe_residual_state.pt` (train-static checkpoint files). This means a non-blocking failure only occurs when the full model has already been trained successfully in a prior run. It does NOT mask check-data validation failures directly.

## 6. Failure Patterns

> This section will be populated from Zikang's Codex JSONL logs during the
> knowledge extraction phase.

### Error Classification (from diagnostics.py)

When check-data exits with non-zero code:

| Keyword in stderr | Classification | Meaning |
|-------------------|---------------|---------|
| `FileNotFoundError`, `No such file or directory` | DATA_ERROR | Input file missing |
| `missing column`, `column not found` | DATA_ERROR | Required column absent from sheet |
| `missing sheet`, `sheet not found`, `invalid sheet` | DATA_ERROR | Sheet `"long"` or CV sheet missing |
| `unexpected header`, `empty workbook`, `no valid data` | DATA_ERROR | File structure mismatch |
| `data_dir not found`, `does not exist` | DATA_ERROR | Dataset directory path wrong |
| `dataset_validation.json` | DATA_ERROR | Validation output file issue |
| `xlsx file is empty`, `excel file format cannot be determined` | DATA_ERROR | Corrupt workbook |
| `cannot find it directory` | DATA_ERROR | I-t data directory path error |
| `missing artifact`, `expected output not found`, `artifact.*not created` | MISSING_ARTIFACT_ERROR | Expected output file missing |
| `KeyError`, `IndexError`, `ValueError` | DATA_ERROR | Unexpected data structure (confidence 0.6) |
| No keyword match, exit code != 0 | DATA_ERROR | Default classification (confidence 0.5) |
| `configuration error`, `configparser`, `invalid config` | CONFIG_ERROR | Config file problem |
| `missing artifact`, `no such file`, `not generated` | MISSING_ARTIFACT_ERROR | Expected output not found |

The exit-code fallback for check-data defaults to `DATA_ERROR` (confidence 0.5). However, keyword matching runs first and is stage-agnostic — if stderr contains a keyword like `configparser` or `no section`, the classification will be `CONFIG_ERROR` (0.9), not `DATA_ERROR`. Notable: `MISSING_ARTIFACT_ERROR` keywords (`missing artifact`, `expected output not found`, `artifact.*not created`) can also fire for any stage.

### Agent A Decision Framework

Agent A has a single tool: `fallback_to_previous`. It uses `stop_mode="schema"` — this means:

- Agent A must call `fallback_to_previous` to produce a result. Free-text output is **not captured**.
- If Agent A runs out of steps without calling the tool, the orchestrator gets `agent.status.value != "done"`,
  falls through to the error handler, and returns `{"fallback": "", "reason": "Agent A did not produce a result"}`.
  This is treated as "no fallback" — the pipeline terminates.
- The tool requires these fields: `target` (which stage to fall back to), `reason` (why),
  `evidence` (supporting evidence), `suggestion` (suggested fix).

| Decision | Trigger | Consequence |
|----------|---------|-------------|
| `fallback_to_previous` (→ Verifier) | Error indicates malformed data: missing columns, wrong sheet names, corrupt files, data format issues. The root cause is upstream in data preparation. | Verifier re-examines the data and may fallback further to Preparer for re-preparation. The entire pipeline may roll back to data-prep. After fallback, the orchestrator resets the pointer to the target stage and task board entries are reverted. |
| No fallback (do nothing) | Error is environmental: missing executables, filesystem permissions, configuration file corruption. Data re-preparation cannot fix these. | The orchestrator returns `{"fallback": ""}`, sets `failed_stage`/`failed_error`, and after all stages iterate, calls `_finalize_run(failed=True)`. Pipeline terminates with final failure. |

**Key distinction**: Agent A must determine whether the error is *in the data*
(fixable by re-preparation) or *in the environment* (not fixable). Because Agent A
cannot produce free-text diagnostics (stop_mode is schema), it must embed its
reasoning into the `reason` and `evidence` fields of the `fallback_to_previous` tool call.
If it chooses "no fallback", it simply must not call the tool — the orchestrator will
timeout the agent and terminate.

### Diagnostic Tools Available to Agent A

- `read_file`: Read `dataset_validation.json` (if produced), error logs,
  `model_config.ini` (check paths section), IV/CV workbook inspection
- `glob_file`: List files in the dataset directory to verify what exists
- Agent A also has access to standard SubAgent base tools (`bash`, `grep_file`, `write_file`, `edit_file`)
- No `inspect_xlsx` tool — Agent A is the only diagnostic agent without it
