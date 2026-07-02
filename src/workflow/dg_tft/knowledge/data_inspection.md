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
| **check-data** | **script** | **No (Agent A only)** | **Hard failure only** |
| fit-physical | script | Yes (Agent B) | Hard failure / quality below threshold |
| train-static | script | Yes (Agent C) | Hard failure / quality below threshold |
| train-hysteresis | script | Yes (Agent D) | Hard failure / quality below threshold |
| final-report | noop | No (Agent E) | — (always runs at end) |

Agent stages spawn LLM subagents. Script stages run CUDA Python subprocesses.
When a script stage hard-fails (exit code ≠ 0) or produces quality below threshold,
a **modeling agent** is spawned to adjust parameters and retry, or fallback upstream.

Fallback is a linear one-hop chain upstream: D→C→B→A→Verifier→Preparer.

### check-data Position and Role

The **check-data** stage is Stage 3 of 7 — the first script-execution stage,
immediately after the Verifier agent stage. It performs **structural validation**
of the dataset files (IV/CV/I-t) before any heavy modeling computation begins.

This stage has **no adjustable INI parameters** and **no quality thresholds**.
It is a binary gate: pass (all files structurally correct) or hard-fail.

On hard failure, **Agent A** is spawned with a single tool: `fallback_to_previous`.
Agent A decides whether the failure originates from data preparation (→ fallback
to Verifier → potentially Preparer) or is environmental (→ final failure).

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
  paired-column format in-place. The downstream engine reads the normalized file.
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

## 5. Engine Behavior

### Execution Flow

Before the engine script runs, the orchestrator performs one pre-processing step:

1. **CV format normalization** (`run_model.py:_normalize_cv_format`):
   - Reads `CV_train.xlsx`
   - Uses `looks_like_shared_bias_wide_table()` from `cv_preprocessing.py` to
     detect the layout
   - If shared-bias wide-table: rewrites columns to paired-column format
   - Writes back to the same file in-place
   - This is transparent — the agent cannot observe or control it

2. **Subprocess invocation**:
   ```
   <CUDA python> engine/validate_dataset.py
     --data-dir <data_dir>       (from model_config.ini [paths] data_dir)
     --json-out <output>/dataset_validation.json
     [--it-dir <it_dir>]         (only if it/ directory exists)
   ```

3. **Post-execution**: The orchestrator verifies `dataset_validation.json` exists.
   If missing → artifact verification fails → stage marked as failed.

### What validate_dataset.py Does — Step by Step

**Step 1 — IV Validation** (3 families, sequential):
- For each family (`IDVBG`, `IDVTG`, `IDVD`):
  - Glob `*_train.xlsx` matching the family name
  - Open each workbook with `pd.ExcelFile()`
  - Verify sheet `"long"` exists
  - Verify columns `VD_V`, `VBG_V`, `VTG_V`, `ID_A` all present
  - Optionally detect temperature columns (aliases: `temp_K`, `temperature_K`,
    `T_K`, `TEMP_K`, `temp_C`, `temperature_C`, `T_C`, `TEMP_C`)
  - Report: `files` (found), `errors` (any failures)

**Step 2 — CV Validation**:
- Check `CV_train.xlsx` exists
- Verify 4 sheets present: `cgg`, `cbg`, `ctg`, `vds` (case-insensitive)
- For each sheet, detect layout and count usable bias/capacitance pairs
- Report: `sheets_found`, `errors`

**Step 3 — I-t Validation** (optional, skipped if no `--it-dir`):
- Glob `I-t*.xlsx` / `I-t*.xls` in it_dir
- For each file: verify columns `time`, `vds`, `vbg`, `vtg`, `temp`, `id`
- Report: `files_found`, `errors`

**Step 4 — Write Output**:
- Combine all results into JSON
- Print to stdout (captured in logs)
- Write to `--json-out` path as `dataset_validation.json`

The script uses `sys.exit(0)` on success (all checks passed) and standard Python
exception propagation on failure (crash → exit code ≠ 0 → error captured in stderr).

### How Results Affect Pipeline Flow

| Scenario | `ok_static` | `ok_with_it` | Pipeline Effect |
|----------|-------------|--------------|-----------------|
| All pass | true | true | Full pipeline runs |
| IV or CV fail | false | false | Hard failure → Agent A → fallback or abort |
| I-t fail only | true | false | Static modeling proceeds (fit-physical, train-static). Hysteresis skipped. |
| Script crash | N/A | N/A | No JSON produced → stage failed → Agent A |

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
| `xlsx file is empty`, `excel file format cannot be determined` | DATA_ERROR | Corrupt workbook |
| `KeyError`, `IndexError`, `ValueError` | DATA_ERROR | Unexpected data structure (confidence 0.6) |
| No keyword match, exit code ≠ 0 | DATA_ERROR | Default classification (confidence 0.5) |

All errors from check-data are classified as `DATA_ERROR` in `diagnostics.py` —
this is the only classification for this stage.

### Agent A Decision Framework

Agent A is the only agent for this stage. It has a **binary decision**:

| Decision | Trigger | Consequence |
|----------|---------|-------------|
| `fallback_to_previous` (→ Verifier) | Error indicates malformed data: missing columns, wrong sheet names, corrupt files, data format issues. The root cause is upstream in data preparation. | Verifier re-examines the data and may fallback further to Preparer for re-preparation. The entire pipeline may roll back to data-prep. |
| No fallback (do nothing, let pipeline fail) | Error is environmental: missing executables (`python` not found), CUDA not available, filesystem permissions, configuration file corruption. Data re-preparation cannot fix these. | Pipeline terminates with final failure. Agent A should output a brief diagnostic text explaining why fallback won't help. |

**Key distinction**: Agent A must determine whether the error is *in the data*
(fixable by re-preparation) or *in the environment* (not fixable).

### Diagnostic Tools Available to Agent A

- `read_file`: Read `dataset_validation.json` (if produced), error logs,
  `model_config.ini` (check paths section), IV/CV workbook inspection
- `glob_file`: List files in the dataset directory to verify what exists
- `subagent_progress`: Check upstream Verifier and Preparer results for context
