# DG-TFT Engine Compatibility Issues

> Originally scanned by 4 explorer agents across `src/device_modeling/dg_tft/engine/` + `run_model.py`
> Last updated: 2026-06-25 (all actionable issues resolved)

---

## RESOLVED

### #1 — Unrecognized CLI args (FALSE ALARM)

All 6 args passed by `run_model.py:train_full_model()` exist in `dg_tft_full_model.py:parse_args()`. Previous scanner misidentified.

### #2 — _build_and_load_model state_dict key mapping (DEGRADED, then RESOLVED)

`KANBlock` (train_base_pkan.py) and `V2StyleKAN` (dg_tft_full_model.py) have incompatible architectures. `_build_and_load_model` now safely returns None for incompatible checkpoints; `load_trained_base` falls through to fresh V2StyleKAN init with physical parameters from `fitted_params.json`. Added `.get("eta_n", 0.60)` and `.get("final_kan_scale", 0.35)` for missing metadata keys. Train-base-pkan stage removed from pipeline per Zikang confirmation that it was superseded by V2StyleKAN.

### #3 — PYTHONPATH in subprocess env (FIXED)

`run_model.py:run()` always adds `ENGINE` directory to `PYTHONPATH` before spawning subprocesses.

### #4 — Hardcoded absolute paths (FIXED)

All `E:/` paths replaced with placeholder/relative defaults across `.ini` templates, engine defaults, and dynamic module constants.

### #5 — Default output directories in source tree (FIXED)

All `ROOT / "outputs/..."` defaults changed to `Path("./outputs/...")`.

### #6 — KANBlock vs V2StyleKAN architecture gap (RESOLVED BY REMOVAL)

Train-base-pkan stage removed from pipeline per Zikang confirmation. `_prepare_fallback_model_dir()` synthesizes model dir from fit-physical output for train-static.

### #7 — CV file format (FIXED)

`_normalize_cv_format()` auto-converts shared-bias CV layout to paired-column format during check-data stage. No preparer changes needed.

### #8 — Config overrides (FIXED)

`start_dg_tft_workflow` supports `overrides` dict (`section.key=value`). Direct text replacement avoids configparser formatting issues. Strict validation rejects keys without section prefix.

### #9 — Console output colors (FIXED)

All console output uses constants from `src/constants.py` (`WF_STATUS_ERROR`, `WF_STATUS_SUCCESS`, etc.), unified with task board color scheme.

### #10 — Preparer subagent permissions (FIXED)

Added `BASH_UNKNOWN_LABEL` to preparer permission presets. Added GNU bash command guidance to both bash tool description and preparer worker prompt.

### #11 — Session isolation in output paths (FIXED)

Output directories now scoped by full session UUID: `outputs/dg_tft_runs/{session_uuid}/run_{XXXX}/`.

### #12 — Deployment-validate placeholder (REMOVED)

Unimplemented placeholder stage removed from `_CORE_STAGES`, `STAGE_ORDER`, and `StageName` constants.
