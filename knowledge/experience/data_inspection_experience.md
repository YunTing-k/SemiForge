# Data Inspection Experience

> **Agent A tools**: Only `fallback_to_previous` (→ Verifier). No `read_file`, `glob_file`, or `adjust_params` — check-data has no INI parameters.  
> **Can influence**: ONLY the fallback decision — is the error in the data (→ fallback to Verifier/Preparer) or environmental (→ no fallback, terminate)?  
> **Principle**: all advice below answers one question: fallback or not?

## Diagnostic Patterns

### Path Not Found But Directory Exists → Check for Encoding Corruption

**Symptom**: check-data fails with "data_dir not found" or path errors, but the directory physically exists.

**Agent Impact**: This is an environmental failure, NOT a data preparation issue.

**Root Cause**: `model_config.ini` paths containing non-ASCII characters (e.g., Chinese directory names on Windows) can become garbled during file I/O with inconsistent encoding. The pipeline's error classifier matches "data_dir not found" → `DATA_ERROR`, which would incorrectly suggest fallback to Verifier.

**Correct Action**: Inspect the error message text (passed in the Agent A prompt) for garbled non-ASCII characters in path strings. If the error message itself contains corrupted path characters → **no fallback** (encoding corruption cannot be fixed by data re-preparation). Report as `CONFIG_ERROR`. If paths in the error message appear correctly encoded but point to the wrong location → fallback to Verifier (path correction needed). Note: Agent A cannot read `model_config.ini` directly (no file I/O tools); rely on the error text and stdout summary provided in the prompt.

### Downstream File Discovery Mismatch — check-data Passes But Training Fails

**Symptom**: check-data reports `ok_with_it = true` but downstream hysteresis training cannot find or parse I-t data.

**Root Cause**: check-data performs structural validation only: it confirms `*_train.xlsx` files exist with required columns (exact match against expected column names), checks for the optional `CV_train.xlsx`, and verifies the `it/` subdirectory exists. It does NOT validate I-t file contents or naming inside the `it/` directory. Downstream training stages may have stricter expectations for I-t filenames, current units, or column aliases that check-data cannot catch, leading to a false-positive pass.

**Agent Impact**: Cannot fix the downstream code. The gap is between check-data's structural validation and the training stage's content-level assumptions.

**Correct Action**: If check-data passes but a downstream stage reports file-not-found for I-t data → **no fallback** (data preparation is not the problem — the downstream module's discovery logic is). Report the mismatch so the user can verify the hysteresis loader's expectations match the actual file naming.
