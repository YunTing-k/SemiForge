# Data Inspection Experience

> **Agent A tools**: `fallback_to_previous` (→ Verifier), `read_file`, `glob_file`, `bash`, `grep_file`. **No `adjust_params`** — check-data has no INI parameters.  
> **Can influence**: ONLY the fallback decision — is the error in the data (→ fallback to Verifier/Preparer) or environmental (→ no fallback, terminate)?  
> **Principle**: all advice below answers one question: fallback or not?

## Diagnostic Patterns

### Path Not Found But Directory Exists → Check for Encoding Corruption

**Symptom**: check-data fails with "data_dir not found" or path errors, but the directory physically exists.

**Agent Impact**: This is an environmental failure, NOT a data preparation issue.

**Root Cause**: `model_config.ini` paths containing non-ASCII characters (e.g., Chinese directory names on Windows) can become garbled during file I/O with inconsistent encoding. The pipeline's error classifier matches "data_dir not found" → `DATA_ERROR`, which would incorrectly suggest fallback to Verifier.

**Correct Action**: Before deciding fallback, inspect the path string in `model_config.ini` for garbled characters. If the path looks corrupted → **no fallback** (encoding corruption cannot be fixed by data re-preparation). Report as `CONFIG_ERROR`. If the path is correctly encoded but wrong → fallback to Verifier (path correction needed).

### Downstream File Discovery Mismatch — check-data Passes But Training Fails

**Symptom**: check-data reports `ok_with_it = true` but downstream hysteresis training cannot find or parse I-t data.

**Root Cause**: check-data uses broad fuzzy discovery (`I-t*.xlsx` glob, fuzzy substring column matching), while downstream modules may have stricter expectations for filenames, current units, or column aliases. The broader check-data's matching rules, the more likely this false-positive mismatch.

**Agent Impact**: Cannot fix the downstream code. The gap is between check-data's structural validation and the training stage's content-level assumptions.

**Correct Action**: If check-data passes but a downstream stage reports file-not-found for I-t data → **no fallback** (data preparation is not the problem — the downstream module's discovery logic is). Report the mismatch so the user can verify the hysteresis loader's expectations match the actual file naming.
