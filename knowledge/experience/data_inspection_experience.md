# Data Inspection Experience (from 2026-06-29 + 05-22 sessions)

> **Agent A tools**: `fallback_to_previous` (→ Verifier), `read_file`, `glob_file`, `bash`, `grep_file`. **No `adjust_params`** — check-data has no INI parameters.  
> **Can influence**: ONLY the fallback decision — is the error in the data (→ fallback to Verifier/Preparer) or environmental (→ no fallback, terminate)?  
> **Principle**: all experience below is organized around the single question Agent A must answer: fallback or not?

## Insights

No generalizable experience patterns found in the 06-29 session.

## Additional Insights from 05-22 session (Turns 280+, 2026-06-22 to 06-29)

### 1. Downstream File Discovery Mismatch — PARTIALLY FIXED
The hysteresis loader's hardcoded 8-filename primary discovery list has been supplemented with a glob-based auto-discovery fallback (`analyze_it_hysteresis.py:197-213`: `IT_DATA_DIR.glob("*.xlsx")` / `glob("*.xls")`). The core mismatch between check-data's broad `I-t*.xlsx` glob and hysteresis's primary hardcoded list still theoretically exists but is mitigated by the tier-3 fallback.

### 2. I-t Column `T` Not Recognized as Temperature — FIXED
Both `analyze_it_hysteresis.py:110` (`normalize_columns()` now includes `"t"`) and `validate_dataset.py:136` (check-data includes `"t"`) recognize column `T` as a valid temperature alias. Also supports `T(K)` and `TEMP_K`.

### 3. INI Path Encoding Corruption with Non-ASCII Characters
**Category**: Principle Insight
**Stage(s)**: check-data (data_dir path resolution)
**Symptom**: check-data fails with "data_dir not found" or "does not exist" errors, even though the data directory exists at the given path.
**Root Cause**: `model_config.ini` paths containing non-ASCII characters (e.g., Chinese directory names on Windows) can become corrupted during file operations — the bytes are read or written with inconsistent encoding, producing garbled path strings. The encoded garbage does not match any real filesystem path.
**Evidence**: Turn 281, line 8883: "model_config.ini 当前确实有编码损坏，路径显示成乱码；这会直接导致迁移后 check-data 或训练失败。我会把配置改成相对路径..." (model_config.ini currently has encoding corruption, paths show as garbled text; this directly causes check-data or training to fail after migration. I will change config to relative paths.)
**Generalization**: This is a general Python `configparser` + Windows encoding issue, not specific to DG-TFT. Any INI file containing non-ASCII characters is vulnerable. The pipeline's error classification system (from `knowledge/data_inspection.md` line 342-360) would classify "data_dir not found" as `DATA_ERROR`, suggesting a fallback chain — but encoding corruption is NOT fixable by data re-preparation. This is an environmental/config error.
**Actionable Advice**: When check-data fails with path-related errors ("data_dir not found", "does not exist"), Agent A should inspect the actual path string from `model_config.ini` for garbled characters before deciding on fallback. Encoding corruption → no fallback (environmental, not fixable by data re-prep). In contrast, a correctly encoded but wrong path → potential fallback to Verifier to fix the path.

### 4. Multi-Bias IV File Naming Convention Variability
**Category**: Tuning Pattern
**Stage(s)**: check-data (IV validation)
**Symptom**: Data validators that expect a single IV file per family (e.g., `IDVBG_train.xlsx`) fail to recognize multi-bias datasets (e.g., `IDVBG VDS0.1V_train.xlsx`, `IDVBG VDS5V_train.xlsx`) even though the files are correctly formatted.
**Root Cause**: DG-TFT datasets typically have multiple IV files per measurement family (one per VDS bias condition), but standalone validators or config checkers may use exact filename matching instead of glob patterns.
**Evidence**: Turn 280+, lines 8847-8851: "数据校验器还错误地只认 IDVBG_train.xlsx 这种单文件名，与你真实目录中的多偏置文件命名不一致。...校验器改为识别 IDVBG_*.xlsx / IDVTG_*.xlsx / IDVD_*.xlsx 整个文件家族。" (The data validator still incorrectly only recognizes single filenames like IDVBG_train.xlsx, inconsistent with multi-bias file naming in the real directory. Fixed to recognize IDVBG_*.xlsx / IDVTG_*.xlsx / IDVD_*.xlsx file families.)
**Generalization**: The pipeline's `validate_dataset.py` already uses `*_train.xlsx` glob patterns (documented in `knowledge/data_inspection.md` line 217), so this issue does NOT affect the pipeline engine. However, any standalone tool or config validator checking for IV files must use glob patterns, not exact filename matching. This applies to all DG-TFT datasets where multiple VDS conditions are measured.
**Actionable Advice**: N/A for pipeline Agent A (engine already uses glob). Relevant for standalone validation tooling outside the pipeline.

### Session Meta-Notes (05-22)
The 05-22 session was dominated by manual data preparation (Turns 1-10), physical model fitting (Turns 11-28), and KAN training architecture (Turns 29-47). Check-data relevant content appears only in late-stage delivery package auditing (Turns 280+, 2026-06-22 to 06-29), where the agent ran the check-data stage as part of delivery validation. The most impactful finding is **Downstream File Discovery Mismatch** — the check-data stage's broad structural validation can mask content-level assumptions in downstream modules, causing check-data to pass but training to fail.

The INI encoding corruption insight is notable because the pipeline's error classification (from `diagnostics.py`) would misclassify it as `DATA_ERROR` and trigger a fallback chain that cannot fix an encoding problem.
