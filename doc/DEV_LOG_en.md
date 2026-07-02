# SemiForge Development Log

## Development Guidelines

The following conventions apply to all code contributions in this project.

### 1. Test-First
- Every new module must ship with a corresponding test file (`*_test.py`) using the `unittest` framework.
- Run all tests: `python -m unittest discover -s test -p "*_test.py"`.
- All tests must pass before committing.

### 2. Documentation Currency
- Behavioral changes (new tools, prompt routing updates, `agent_configs.json` key semantics, SubAgent lifecycle changes) must be recorded in this document.
- Routing guidance in `src/context/prompt.py` and annotations in `config/agent_configs.json` must stay in sync with the code.

### 3. English-Only Comments and Output
- Code comments, docstrings, `console.print()`, and `sys_log` must use English.
- The only exception is author/institution metadata in file header docstrings.
- Example: `"Stage check-data failed: division by zero"` ✅, Chinese output ❌.

### 4. User Visibility and Console Style
- Every key path checkpoint must emit a `console.print()`. Silent failures are unacceptable.
- Color conventions (use constants from `src/constants.py`):
  - Error: `WF_STATUS_ERROR` (bold red)
  - Warning: `WF_STATUS_WARN` (bold yellow)
  - Success: `WF_STATUS_SUCCESS` (#8CDCA0)
  - Detail / secondary info: `bright_black`
  - **Do not overuse `MAJOR_COLOR1` / `MAJOR_COLOR2`** — reserve them for truly emphasized identifiers like module names and config key names. Never use them for entire sentences.
- When building rich text with `Text.append()`, append each color fragment separately with explicit `style=` rather than nesting Rich markup that may be overridden by an outer style.
- `sys_log` preserves the full audit trail; `console.print()` provides real-time user-facing progress. The two are complementary — neither replaces the other.

### 5. Configuration-Driven; No Hardcoding
- Paths, timeouts, thresholds, and feature flags must be read from `config/agent_configs.json`. Never hardcode.
- Time-related config keys use the `_S` suffix for seconds (e.g., `DG_TFT_TIMEOUT_S_CHECK_DATA`, `SUBAGENT_TIMEOUT_S`).
- String constants (tool names, labels) are defined in `src/constants.py` (convention: `TOOL_NAME_*`, `*_LABEL`).

### 6. Thread Safety
- Any shared mutable state accessed across threads (e.g., `WorkflowManager._runs`, `PreparationTracker._preps`) must be protected by `threading.Lock`.
- All mutation operations (create/update/mark_done/mark_failed/save/load) must execute within the lock.

### 7. Permission System
- Any operation that affects the filesystem or spawns external commands must request user authorization via `pause_for_permission` / `ask_permission_tui`.
- Tool permission mappings use the `PERMISSION_PRESETS` constants. Never handwrite permission lists.

### 8. Import Conventions
- Intra-package imports use relative paths: `from .schema import WorkflowRun`.
- External callers (`main.py`, `agent_context.py`, etc.) use absolute paths: `from src.workflow.dg_tft.manager import WorkflowManager`.
- Never mix both import styles for the same module — Python registers them as two separate module objects, breaking `isinstance` checks.

### 9. Session Persistence
- Data structures that must survive session restart (`WorkflowManager`, `Scoreboard`, etc.) must implement `save_to_file()` / `load_from_file()`.
- Persistence files live under `session/<session_uuid>/` as JSON.
- `to_dict()` and `from_dict()` must not rely on `dataclass.asdict` (its internal `deepcopy` can fail on complex structures).

### 10. SubAgent Lifecycle
- Register: write `SubAgentProgress` into `parent_ctx.agent_list`.
- Execute: `agent.run()` blocks synchronously.
- Unregister: remove from `agent_list` in a `finally` block.
- Mute mode: check `getattr(ctx, "subagent_mute", False)` — when True, suppress console output and permission prompts.
- SubAgent creation uses `clone_context()` for an isolated context copy.

### 11. Error Handling
- Every exception must trigger both `sys_log.error(...)` AND `console.print(..., style="bold red")`.
- Exceptions on non-critical paths (e.g., diagnosis subagent failure) must not propagate — use try/except to contain them.

### 12. Agent / Fixed-Code Boundary
- **Agent/LLM scope**: understanding data formats, verifying output structure, interpreting failure causes.
- **Rule-based scope**: training execution, config generation, quality threshold evaluation, parameter adjustment, report generation.
- Do not let an LLM/SubAgent decide training algorithm parameters or modify training code directly — "leave the openness to the Agent, leave execution to fixed code."

### 13. Git Commit Conventions
- Commit messages in English, concise imperative mood.
- Each commit should focus on a single topic.
- Pre-commit checklist: `git status` → `git diff` → confirm no debug code, no secrets, no unintended changes.

---

## Development Timeline

### 2026-06-25 | Phase 0–2 — Infrastructure + Data Audit + Config Generation

**Phase 0**: DG-TFT modeling code migrated to `src/device_modeling/dg_tft/`. Dual-interpreter strategy locked: Agent (`AGI`) and modeling (`CUDA`) strictly separated via `DG_TFT_MODELING_PYTHON_EXE` in `config/agent_configs.json`.

**Phase 1**: Workflow engine skeleton at `src/workflow/dg_tft/` — `schema.py` (WorkflowRun, StageResult, DatasetProfile), `manager.py` (thread-safe CRUD + session persistence), `stage_runner.py` (unified interpreter resolution). Hooked into `AgentContext` and `main.py`.

**Phase 2**: `dataset_inspector.py` (subprocess CUDA `validate_dataset.py` → DatasetProfile, I-t auto-discovery, temperature/HSPICE detection) + `config_builder.py` (template .ini → per-run runtime config). Tests: 103 total.

---

### 2026-06-25 | Phase 3 — Subprocess-based Full Pipeline

`stages.py` (StageDef dataclass, 7 stages) + `orchestrator.py` (`run_workflow()` with sequential execution, fail-fast, conditional hysteresis skip, resume-from-checkpoint). Console progress output via `ctx.console`. Tests: 135 total.

---

### 2026-06-25 | Phase 4 + 4.5 — Agent Tool Integration + Data Preparation

**Phase 4**: Three workflow tools (`start_dg_tft_workflow`, `query_dg_tft_workflow`, `read_dg_tft_workflow_report`) + dispatch + slash commands (`/workflow_list`, `/workflow_get`, `/workflow_report`). System prompt rewritten from TECoSim/simulator to DeviceModeling/DG-TFT.

**Phase 4.5**: `data_preparer.py` — three-step preparation pipeline:
1. **Preparer subagent** (type=`preparer`): converts raw CSV/XLSX to `*_train.xlsx` format
2. **Verifier LLM** (fast model, mini tool-loop): inspects output with `report_verification` structured schema
3. **`_check_output()`**: pure Python CRITICAL/WARNING check (zero LLM)

Same-day bugfix round: worker subagent uses CUDA python for pandas, verifier progress display, skip-if-already-prepared, synchronous foreground execution, train-static metadata compatibility. Tests: 234 total.

---

### 2026-06-25 | Phase 5 — Reports, Artifacts & Task Board

`artifacts.py` (key constants, path helpers) + `report_builder.py` (`build_and_write_reports()` → ARTIFACT_MANIFEST.json, RUN_SUMMARY.json, RUN_SUMMARY.md). Orchestrator integrated: auto-generates reports on success/failure. Task board sync: stage tasks created/resolved on Scoreboard. Console summary with paths, metrics, and failure suggestions. Tests: 237 total.

---

### 2026-06-26 | Phase 6 — Failure Diagnosis + Subagent Collaboration

`diagnostics.py`: 6-category keyword-based error classification (DATA_ERROR, CONFIG_ERROR, TRAINING_NUMERICAL_ERROR, MISSING_ARTIFACT_ERROR, HSPICE_ENV_ERROR, UNKNOWN_ERROR) with confidence scoring.

`subagent_hooks.py`: 6 explorer subagent hook definitions with parameterized prompt templates. `orchestrator.py` integration: `_run_stage_diagnosis()` auto-classifies → spawns explorer subagent on failure (configurable via `DG_TFT_ENABLE_DIAGNOSIS`). Diagnostic result stored in `run.extra["diagnosis"]`; report builder includes "Diagnosis" section. Tests: 314 total.

---

### 2026-06-26 | Phase 7 — Quality Gate + Auto Parameter Tuning

`metrics_parser.py`: extracts metrics from artifact JSON (fitted_params.json, residual metrics, hysteresis analysis).

`quality_guard.py`: `assess_quality()` threshold check → `suggest_adjustments()` progressive strategy engine → `apply_adjustments()` in-place INI editing. Parameter whitelist + bounds for safety. Retry loop (max 3, configurable) with `_retry_N/` subdirectories; best attempt artifacts preserved.

Orchestrator integration: `_enforce_quality_gate()` between stage success and next stage. Report builder: "Quality" section with retry log table. Thresholds calibrated from real debug run data. Engine stdout/stderr streaming during retries. Tests: 370 total.

---

### 2026-06-26 | Phase 8 — Engine Script Functionization

7 engine/dynamic scripts each gained a `run_*()` importable entry point. `run_model.py` rewritten: inner `subprocess.run()` eliminated, replaced with direct `from engine.xxx import run_yyy; run_yyy()` calls.

AGI→CUDA outer subprocess preserved — only the CUDA-internal double-subprocess removed. No feature flag; functionization is the sole path, transparent to orchestrator. Tests: 393 total (12 skip = CUDA deps not in AGI env).

---

### 2026-06-27 | Phase 9 — Testing, Stability & Cleanup

**P9-T1**: Supplementary unit tests (+26): report quality gate rendering, diagnostics (truncate_list, stdout-only classification), quality_guard (INI read errors, missed adjustments).

**P9-T2**: Agent integration tests (+64, 2 new files): `dg_tft_workflow_tool_test.py` (38 tests — task board linkage, serialization, dispatch, prompt routing) + `dg_tft_workflow_background_test.py` (26 tests — concurrent WorkflowManager, status transitions, DatasetProfile, persistence).

**P9-T3**: Three test datasets (`test_data/{static_only,full,corrupted}/`) + reproducible generator.

**P9-T5**: Cleanup sweep — no debug code, no hardcoded paths, session persistence verified, `.gitignore` updated.

---

### 2026-06-27 | Per-Stage + Overall Run Timeout

Timeout values moved from hardcoded defaults to `agent_configs` (`DG_TFT_TIMEOUT_S_CHECK_DATA=600`, `DG_TFT_TIMEOUT_S_FIT_PHYSICAL=1800`, etc.). Key naming uses `_S` suffix matching existing `SUBAGENT_TIMEOUT_S` convention.

`_resolve_stage_timeout()`: agent_configs override → StageDef.timeout fallback. `_compute_run_timeout()`: Σ(resolved per-stage) × 2. `_elapsed_exceeds()` guard before each stage in orchestrator loop. Tests: +9.

---

### 2026-06-28 | TECoSim Removal + SemiForge Rebrand

Deleted `simulator_param.py` and `simulator_support.py` (DesignManager, RunManager, launch_sim, init_design, read_log — ~1900 lines). Removed 6 simulator tool definitions/handlers/dispatch branches. Purged 13 dead constants, 2 config keys. Renamed all TECoSim-branded code to SemiForge. Added Apache 2.0 LICENSE.txt; updated copyright notices.

---

### 2026-06-29 | Zero-Change One — Output Path Unification to session/\<uuid\>/

Consolidated all workflow output directories from three scattered peer-level paths (`outputs/dg_tft_runs/`, `prepared_data/`, `session/`) into `session/<uuid>/runs/`. 15 subtasks across 8 source files:

**Path reorganization**: `config_builder.py` gained `_runs_base_dir()` to construct `session/<uuid>/runs/`. `tool_def.py`'s `prepare_dg_tft_dataset` handler output directory switched to new path. Removed non-existent `DG_TFT_OUTPUT_ROOT` key references from `agent_configs.json`.

**Code cleanup**: `data_preparer.py` removed `_PREP_OUTPUT_BASE_DIR` — output directory now passed by caller. `manager.py` uses `SESSION_PATH` constant instead of hardcoded `"session"`. Fixed doc comment paths in `PreparationTracker` and `artifacts.py`. `.gitignore` removed `/prepared_data/`, `/outputs/`, `/test_data/` entries.

**minimal_run merge**: `quick_run` and `debug_run` merged into `minimal_run` (INI `[runtime]` section). `run_model.py`, `prompt.py`, and INI template synchronized. 2026-06-29 bugfix: `config_builder.py` `output_dir` uses `Path.resolve()` for absolute paths to prevent engine's `resolve_path()` from nesting relative paths twice.

**Test fixes**: `dg_tft_diagnostics_test.py` completed missing `_make_mock_ctx` attributes + `@patch _launch_diagnosis_subagent` to eliminate MagicMock serialization traceback. `dg_tft_workflow_background_test.py` tearDown added `test_persist` cleanup. `test_subagent_unit.py` includes `session/test-unit-001/` cleanup.

---

### 2026-06-29 | Zero-Change Two — SubAgent Enhancement

**Motivation**: SubAgent could not inject custom tools, enforce structured output, or distinguish between "worker agent => text is fine" vs "decision agent => must use schema."

**Three-layer overhaul**:

1. **Local tool injection** (`subagent.py` + new `subagent_tools.py`):
   - `SubAgent.__init__` gained `subagent_tools`, `tool_handlers`, `stop_mode`, `stop_schemas` — all optional
   - `build_tools()`: appends local tool schemas after `SUPPORTED_TYPES` filtering
   - `execute_tool_calls()`: new `_handle_subagent_tool()` local dispatch — has handler → execute & return; no handler + stop_schema → capture args → DONE; no handler, not stop → FAIL (prevents silent misconfiguration)

2. **Three stop modes**:
   - `plain_text` (default): LLM plain text response → DONE; empty content ⇒ reasoning fallback → reminder to continue
   - `schema`: must call a tool in `stop_schemas` list to terminate; plain text rejected with reminder
   - `any`: any response (text/tool/reasoning) terminates

3. **Unified tool entry point** (`subagent_tools.py`):
   - 7 tool name constants (`SUBAGENT_TOOL_*`)
   - `get_tools(stage)` returns `(schemas, handlers)` tuple — call site doesn't manage two separate dicts
   - `_STAGE_TOOLS` registry: `fit-physical` (3 schemas + 1 handler), `check-data` (1 schema + 0 handler)

**Unit tests**: 15 tests (`test_subagent_unit.py`), covering plain_text/text fallback/schema investigate→output/schema exhaustion/any/FAIL fallback/get_tools integration.

---

### 2026-06-30 | Expert Knowledge System (3.5)

New `src/workflow/dg_tft/knowledge/` directory:
- `loader.py`: `load_knowledge(stage)` reads per-stage `.md` content; `build_agent_prompt(stage, preferences)` assembles **USER PREFERENCES** + **EXPERT KNOWLEDGE** dual-block template, accepts `discovered_dir` parameter for aggregating discovered summaries
- 5 knowledge skeleton files: `data_inspection.md` / `physical_kernel.md` / `static_model.md` / `hysteresis.md` / `final_interpret.md`
- Progressive disclosure design: pre-written knowledge loaded in full into prompt; discovered knowledge produces dual files (detailed `.md` + concise `_summary.md`), summaries aggregated into prompt, full details read on-demand via `read_file`

**Path constants migration**: `DG_TFT_KNOWLEDGE_DIR`, `DG_TFT_RUNS_SUBDIR`, `DG_TFT_DEFAULT_CONFIG_TEMPLATE` moved from `agent_configs.json` into `constants.py`. All source `ctx.agent_configs.get(...)` calls updated to use constants as fallback defaults. 7 source files + 5 test files synchronized. Path keys remaining in `agent_configs.json` are limited to machine/system-dependent executable paths (`DG_TFT_AGENT_PYTHON_EXE`, `DG_TFT_MODELING_PYTHON_EXE`, `BASH_PATH`, `RIPGREP_PATH`).

**Tests**: 14 tests (`dg_tft_knowledge_test.py`), covering valid stages, unmapped stage, missing file, preferences+knowledge combinations, whitespace-only prefs, empty input.

---

#### 2026-07-01 | Stage Attempt Logging (3.6)

**`operation_log.py`** — `StageAttemptLog` + `OperationLog` infrastructure:

- `AttemptTrigger` enum: `INITIAL` / `QUALITY_FAIL` / `AGENT_ADJUST` / `FALLBACK_SENT` / `FALLBACK_RECEIVED` / `UPSTREAM_CHANGED`
- `StageAttemptLog` dataclass: `n` / `trigger` / `upstream` / `handoff` / `agent_decision` / `ini_snapshot` / `duration_s` / `metrics` / `adjustments` / `ok` / `error` / `quality_pass` — full `to_dict()` ↔ `from_dict()` serialization
- `OperationLog`: `initialize(run_id, preferences)` → `append_attempt(stage, attempt)` → `set_final_status(status)` + `load()` / `get_stage_attempts()` / `get_last_attempt()` query API

**Orchestrator integration**:
- `_log_initial_attempt()` — records attempt_1 unconditionally (with metrics), called after `_verify_artifacts` and before `_enforce_quality_gate`. Logged even for minimal_run and stages without quality thresholds
- `_enforce_quality_gate()` handles retries only (n≥2); no longer duplicates the initial attempt log
- `final_status` written on pipeline completion, timeout, or failure
- Retry directory renamed: `_retry_N` → `attempt_{N+1}` (attempt_1 = initial run, attempt_2 = first retry)
- `_cleanup_retry_dirs()` removed — all attempt directories preserved for Agent E interpretation

**`run_id` unification**:
- `PreparationTracker.register_prep()` now accepts optional `run_id` parameter; uses run_id as key when provided
- `prepare_dg_tft_dataset` handler: creates pending run to get run_id; prep output at `session/<uuid>/runs/run_{run_id}/data_prep/attempt_1/`
- `_get_prep_info()` supports run_id fallback lookup

**Tests**: 26 tests (`dg_tft_operation_log_test.py`), covering enum values, dataclass serialization/deserialization/missing-field resilience, OperationLog init/append/multi-stage/persistence/corrupted file/nested dir auto-create/order preservation.

---

### 2026-07-01 | Data Input Stage (TODO 2.5.1–2.5.11) — Version 1.2.0

### 2.5.1 handoff data structures (`schema.py`)
- `PrepHandoff` — structured handoff from Preparer to Verifier: `observations` / `actions` / `potential_issues` / `output_manifest`
- `VerifierFeedback` — structured Verifier feedback: `overall_ok` / `critical_issues` / `suggestion` / `worker_notes` / `summary`
- `PrepHandoff.from_worker_report()` — factory method to convert existing `worker_report.json` into handoff
- `VerifierFeedback.from_dict()` — supports legacy string-list format for `critical_issues`

### 2.5.5 `subagent_tools.py` enhancement
- `inspect_xlsx` handler: replaced placeholder with real `openpyxl` logic (sheet names + column headers); returns `{"sheets": {...}, "sheet_count": N}`
- Added `_REPORT_VERIFICATION_SCHEMA` — output tool schema for Verifier, equivalent to old `build_report_verification_tool()`
- `_STAGE_TOOLS` now includes `"data-prep"` stage: `inspect_xlsx` (investigation tool) + `report_verification` (output tool)
- Imported `FAIL_LABEL` for handler error returns

### 2.5.4 Verifier migrated to SubAgent
- **Deleted**: `_build_verifier_tools()` / `_execute_verifier_tool()` / `_run_verifier()` manual LLM loop (~150 lines)
- **Deleted**: `build_report_verification_tool()` — schema moved to `subagent_tools.py`
- **Added**: `_run_verifier_subagent(ctx, output_dir, report_verification_tool_name)` — `explorer` type + `stop_mode="schema"` + `stop_schemas=["report_verification"]`
  - Verifier SubAgent uses `glob_file` (built-in explorer type) to list files
  - Uses `inspect_xlsx` (subagent tool, with handler) to inspect XLSX structure
  - Uses `report_verification` (subagent tool, no handler, output tool) to submit structured findings
  - Graceful fallback on failure: returns `overall_ok=False` placeholder dict
- Verifier prompt updated: `read_file` → `inspect_xlsx`

### 2.5.2-2.5.3 worker handoff injection
- `_run_worker()` now accepts optional `handoff_history_text` — injected into prompt as "Previous attempt feedback" when non-empty
- `_get_handoff_history(op_log)` — reads prior data-prep attempts from `operation_log.json` and builds a text summary of handoffs + verifier feedback

### 2.5.6 retry loop
- `_run_worker_and_verifier()` — single worker→verifier→check cycle; returns `(handoff, verifier_feedback, ok, issues)`
- `run_data_preparation_with_retry()` — main Preparer-Verifier loop:
  - Each attempt creates `data_prep/attempt_N/` directory
  - Initializes `OperationLog` at `run_output_dir/operation_log.json`
  - Initial attempt trigger = `INITIAL`, retry = `VERIFIER_FAIL`
  - Worker exceptions do not interrupt loop; next attempt continues
  - All attempts recorded to operation_log (handoff + verifier_feedback)
  - Success: `final_status="done"`, exhausted: `final_status="exhausted"`
- `run_data_preparation()` kept as compatibility wrapper, delegates to `_run_worker_and_verifier`

### 2.5.7 standard data entry
- `run_verification_only()` — skips Preparer, Verifier inspects provided dataset directly
  - Pass: use existing data for subsequent stages
  - Fail + `raw_dataset_dir` provided: fallback to `run_data_preparation_with_retry()`
  - Fail + no fallback: return failure immediately

### Operation Log enhancements
- `AttemptTrigger` enum: added `VERIFIER_FAIL` and `PREP_RETRY`
- `OperationLog` now initialized during data prep (before orchestrator); orchestrator's `run_workflow()` can reuse same file

### Other changes
- `tool_def.py`: `prepare_dg_tft_dataset` handler variable renamed to `run_output_dir` / `attempt_1_dir` for clarity
- `data_preparer.py`: removed unused constants `_PREP_VERIFIER_MAX_TURNS` / `_PREP_VERIFIER_MODEL_TIER`; removed unused imports `client` / `deepseek_support`
- `agent_configs.json`: `DG_TFT_PREP_MAX_RETRIES` / `DG_TFT_PREP_HANDOFF_HISTORY_SIZE` already exist
- `DG_TFT_RUNS_SUBDIR` / `PREP_SUBDIR_NAME` constants provided by zero-change-1

### Tests
- Updated `test/dg_tft_preparer_test.py`: removed `TestBuildReportVerificationTool`, mocks patched from `_run_worker`/`_run_verifier` to `_run_worker_and_verifier`
- Added `test/dg_tft_prep_retry_test.py`: 31 tests — `PrepHandoff`/`VerifierFeedback` serialization, `inspect_xlsx` handler, data-prep stage tool registration, `AttemptTrigger` enum values, handoff history building, worker handoff reading, retry loop (success/retry/exhausted/exception continuation/directory creation/operation_log writing/retry triggers), standard data entry (pass/fail no fallback/fallback to Preparer)
- Full suite: 864 tests pass (0 failures, 0 errors)

---

### 2026-07-01 | TECoSimAgent v0.2.6 sync + Data Prep Fixes

### TECoSimAgent sync (commit `72c8038`)

**bash risk detection v1.5 → v1.10** (`bash_support.py`): output redirect detection, `$()`/backtick/`<()` substitution evaluation, quoted-string masking, here-doc detection, `BASH_INLINE_SCRIPT` level 4→3, +85 commands, expanded docker/kubectl/git/pkg-mgr handling.

**Tool result truncation**: `truncate_tool_result()` field-level iterative truncation preserving JSON structure. `SUBAGENT_EARLY_WARN_STEPS` key name unified.

**Markdown rendering theme**: `get_console()` with Theme (H1-H6/code/lists/tables/quotes/links/hr styles), `ContentMD`/`ReasonMD` with custom elements (rounded tables, icon separators).

**prompt.py**: stream collector None sentinel fix, reasoning-only content patch, icon-based resume display (⚑▲❖⬟).

**SubAgent**: `start_time`/`elapsed_s` persisted, GNU bash hint in system prompt, `truncate_tool_result` integration.

**Constants/config**: +25 constants, limits bumped (20000/60/3000).

### Data prep fixes + enhancements

- **operation_log through-line**: handler uses `run_data_preparation_with_retry`, orchestrator loads existing log
- **handoff fix**: no longer overwrites `worker_report.json` written by worker SubAgent
- **manifest writing**: retry loop writes `manifest.json` + `prep_file_manifest.json`
- **Console output**: Preparer/Verifier labels with completion stats
- **agent_list archive**: worker/verifier/diagnosis SubAgent `pop` → `if_archived = True`
- **get_console() enabled**: `main.py` uses themed Console
- **SubAgent TUI display**: `_TOOL_DISPLAY_KEYS` extended for subagent tools

### Tests
- `bash_risk_test.py`: reclassified tar/gzip/unzip/awk/timeout
- `dg_tft_preparer_test.py`: mock targets updated, SESSION_PATH → temp dir, MagicMock leak fixed
- TODO.md: new rule 8 — zero residue in session/
- 864 tests pass, session/ verified clean

---

## Lessons Learned

### Python module double-import trap
`from src.workflow.dg_tft.schema import X` and `from workflow.dg_tft.schema import X` register as two different modules. Use relative imports (`from .schema import`) within the package.

### `dataclass.asdict` + deepcopy
`asdict()` uses `deepcopy()` internally. Complex nested structures with locks/mocks will fail serialization. Manual `to_dict()` is safer.

### Orchestrator design
- Stage definitions (static metadata) separate from execution logic (dynamic orchestration)
- Resume via `_stage_done()` check on `run.stage_results`
- Required stages fail-fast; optional stages log warning and continue
- Quality gate after every trainable stage, not just at pipeline end

### Console output conventions
- Error: `WF_STATUS_ERROR` (bold red) | Warning: `WF_STATUS_WARN` (bold yellow) | Success: `WF_STATUS_SUCCESS` (#8CDCA0) | Detail: `bright_black`
- No Chinese in code output; `sys_log` and `console.print()` complementary

### Test strategy
911 tests pass (11 skipped). Run: `python -m unittest discover -s test -p "*_test.py"`.

### Agent intervention boundaries
LLM/subagent only at 3 points: data preparation, format verification, failure diagnosis. Training execution, config generation, quality gating, and report building are pure rule-based code.

---

### 2026-07-01 | INI Parameter Audit & Exposure (3.2)

### Parameter definition tables (`param_defs.py`)
- `PHYSICAL_FIT_PARAMS` — 3 params (max_iv_points / max_cv_points / max_nfev_scale)
- `FULL_MODEL_PARAMS` — 11 params (epochs / lr / batch_iv / batch_cv / cv_weight / cv_residual_level / enable_cv_residual / finetune_base_pkan / early_stop_patience / early_stop_min_epochs / symbolic_training_weight)
- `HYSTERESIS_PARAMS` — 6 params (enabled / regime / epochs / hidden_capture / hidden_emission / amp_parameterization)
- Each param: type / lo / hi / default / desc; string params have `enum` constraint
- `_STAGE_PARAM_MAP` — stage → (INI section, param_table) lookup

### Auto-generated tool schema
- `params_to_tool_schema(param_table, ...)` → OpenAI-compatible function call schema
- `get_stage_param_schema(stage_name)` → one-liner per stage
- Type mapping: int→"integer", float→"number", bool→"boolean", string→"string" (with enum)
- Auto-injected bounds (minimum/maximum) + description; `reasoning` enforced as required

### Hard guard (`quality_guard.py`)
- `validate_agent_adjustments(params, param_table, stage_section, stage_name)` — 4 checks:
  1. Whitelist — param key must exist in param_table
  2. Type — string→numeric coercion for int/float; yes/no/true/false for bool
  3. Enum — string values checked against allowed options
  4. Bounds — out-of-range values clamped with warnings
- Returns `(sanitized_params, warnings)` — never throws

### Soft guard
- `get_soft_guard_prompt()` — 5 self-check rules (bounds / physical plausibility / interaction / incrementality / expert knowledge)

### Files
- `src/workflow/dg_tft/param_defs.py` — new: param tables + schema gen + soft guard
- `src/workflow/dg_tft/quality_guard.py` — added `validate_agent_adjustments()` + coerce helpers
- `test/dg_tft_param_defs_test.py` — new: 45 tests
- TODO.md — 3.2.1–3.2.7 all complete

---

### 2026-07-01 | DG-TFT Engine v90→v95 Merge

### Engine Upgrade
- `dg_tft_full_model.py`: 13.6K → 19K lines. New: base IV PKAN training pipeline (formerly standalone `train_base_pkan.py`), 6 hard precision gates, ~25 new CLI args, CV charge PKAN split, `CompactKAN` rename, symbolic engine expansion, safe scale search
- `fit_physical_kernel.py`: new IV measurement-stage alignment, CV gate auto-orientation, CV quality selection
- `validate_dataset.py`: new CV layout detection (`looks_like_shared_bias_wide_table`)

### New Engine Modules
- `cv_preprocessing.py` — CV preprocessing utilities (transition metadata, centered coordinates, layout detection)
- `measurement_preprocessing.py` — IV measurement-stage drift alignment
- `tensor_utils.py` — shared tensor/data utilities extracted from old `train_base_pkan.py`

### Removed
- `train_base_pkan.py` (merged into `dg_tft_full_model.py`)
- `dg_tft_ekv_level2.va` (replaced by `templates/physical_kernel_template.va`)

### New Directories
- `templates/` (2 VA templates: physical_kernel, portable_glue)
- `tools/` (7 scripts: acceptance bundle, hysteresis banks, ablation, delivery validation)

### New Actions
- `doctor`: environment diagnostic
- `smoke-static`: lightweight static model validation
- `quick-static`: quick IV/CV training + architecture validation
- `package-release`: full release assembly
- `package-portable`: portable archive
- `verify-package`: package verification

### run_model.py Refactor
- Architecture switched from **direct import** to **subprocess call** (matches v95 canonical architecture — all entries are CLI-only)
- Retained migration adaptations: `_normalize_cv_format()`, `_prepare_fallback_model_dir()`, `minimal_run`, example data guard, `train-base-pkan` action
- v95 helpers: `run()`/`run_gate()`/`run_artifact_stage()`, `resolve_hspice()`, `dynamic_env()`, `doctor()`, release assembly

### Parameter Table Expansion (3.2 Maintenance — per Section 3.2.7)
- `PHYSICAL_FIT_PARAMS`: 3 → 6 (+3 `[preprocessing]` params)
- `FULL_MODEL_PARAMS`: 23 → 44 (+21: base IV training, IV residual, symbolic, LR multipliers, training schedule, cv_train_subset, etc.)
- `HYSTERESIS_PARAMS`: 7 → 13 (+6: fast/slow regime split)
- Total: 32 → 73 params
- `quality_guard.py` `_PARAMETER_BOUNDS`: new `[preprocessing]` section, `full_model` +24 keys, `hysteresis` +2 keys
- `_PARAMETER_WHITELIST`: 30 → 56 entries

### Tests
- `dg_tft_functionized_api_test.py`: rewritten for v95 CLI-only architecture
- `dg_tft_param_defs_test.py`: expected param count 22 → 44; `preprocessing` section added
- `dg_tft_quality_guard_test.py`: string/bool param skip list expanded to 7
- 911 tests pass (11 skipped)

### Files
- `src/device_modeling/dg_tft/engine/` — 4 replaced, 3 added, 2 removed
- `src/device_modeling/dg_tft/{templates,tools}/` — new
- `src/device_modeling/dg_tft/{dynamic,reporting}/` — fully replaced
- `src/device_modeling/dg_tft/{config,run_model.py}` — fully replaced

---

### 2026-07-01 | 3.2 Maintenance — KAN Arch Param Exposure + minimal_run Precision Gate Skip + Artifact Fixes

**KAN Architecture Parameter Light Extension**:
- Parameterized temperature KAN hidden/grid/order: `temperature_kan_hidden` (8), `temperature_kan_grid` (3), `temperature_kan_k` (3) → `[temperature]` section
- Engine `dg_tft_full_model.py`: added `--temperature-kan-hidden/grid/k` argparse; `ResidualDiagnostic.__init__` accepts and passes to `CompactKAN` constructor

**Training Regularization Trio**:
- `weight_decay` (1e-6), `grad_clip` (5.0), `lr_eta_min_frac` (0.02) → `[full_model]` section
- Engine already had argparse; only INI → CLI plumbing was missing

**`--skip-precision-gates` for minimal_run Alignment**:
- Engine: new `--skip-precision-gates` flag, gates 3 precision gates (Base IV-PKAN / IV residual held-out / Deployment HSPICE) → prints warning on failure instead of RuntimeError
- `run_model.py`: passes both `--quick-run` and `--skip-precision-gates` when `quick=True`, ensuring minimal_run does not abort on training quality
- `apply_quick_run_preset`: added `iv_residual_stage_epochs` clamp (≤140)

**Artifact & Gitignore Fixes**:
- `report_builder.py`: ARTIFACT_MANIFEST written last so JSON/MD exist on disk when manifest checks them
- `.gitignore`: ignore `src/device_modeling/.runtime_cache/`, `trained_model/`, `validation/`

**Parameter Count**: `FULL_MODEL_PARAMS` 44→50, total 79 params.

**Files Affected**: `dg_tft_full_model.py`, `run_model.py`, `model_config_new_device_template.ini`, `param_defs.py`, `quality_guard.py`, `report_builder.py`, `.gitignore`, 2 test files.

**Tests**: 911 pass (11 skipped).

---

### 2026-07-01 | Development Order Adjustment — Tool Unification + while+pointer Promoted

**Changes**:
1. Promoted 1.5 Tool Unification from original Step 13 to Step 6 — ensures subsequent Agent development targets unified 3-tool interface
2. Merged while+pointer (3.7) and Agent A (3.4.1) into Step 7 — Agent A depends on fallback, fallback depends on while+pointer, circular dependency resolved in one step
3. Semantic unification: removed `fallback_to_verifier`, unified chain uses only `fallback_to_previous` — every agent in the linear pipeline falls back one hop, no need to distinguish "previous happens to be Verifier"

**New Development Order**:
0-5 ✅ → 6. Tool Unification → 7. while+pointer + Agent A → 8. Agent Infrastructure → 9. Agent B → 10. Agent C → 11. Agent D → 12. Agent E

---

### 2026-07-01 | Step 6 Tool Unification + Multi-Metric Quality Gate + Agent Decision Design

**Step 6: Tool Unification (1.5)** — 4→3 tool convergence:
- `prepare_dg_tft_dataset` + `start_dg_tft_workflow` → **`dg_tft_run`**
- `query_dg_tft_workflow` → **`dg_tft_status`**
- `read_dg_tft_workflow_report` → **`dg_tft_report`**

**`entry` enum** replaces two booleans: `"full"` (default), `"verifier"`, `"check-data"`.

**Bug fixes**: `run_id` :04d padding mismatch, RUN_SUMMARY stale status, TUI 4/5 display, artifact filenames (v90→v93, hysteresis nested path, removed model.va).

**Verifier**: prompt now requires checking ALL files individually.

**Quality guard**: single→multi-metric. Train-static checks both `residual_iv_rms` + `residual_cv_rms`. Dead `DG_TFT_QUALITY_STATIC_CV_RMS_MAX` now wired. Trigger rule: any one metric exceeding threshold wakes the quality gate.

**Agent decisions**: `AttemptTrigger.AGENT_ACCEPT` added. TODO updated with four-way decision flow + handoff map.

**Session rules**: only delete non-UUID mock dirs. `.gitkeep` must be preserved.

### 2026-07-01 | Step 7 while+pointer + Agent A + Pipeline Integration (3.7 + 3.4.1)

**Structural refactor**: for-loop → while+pointer state machine in `orchestrator.py`.

**StageDef.kind**: `"agent"` (Preparer, Verifier) vs `"script"` (check-data through final-report). Agent stages spawn SubAgents; script stages call `run_stage()`.

**Pipeline integration**: Preparer + Verifier are now StageDef entries in `_CORE_STAGES` (uses `get_stages(entry=)` for filtering). Data preparation moved from `tool_def.py:dg_tft_run` into the orchestrator's while+pointer loop.

**Agent A (check-data)**: hard failure → `_spawn_agent_a()` SubAgent with `stop_mode="schema"` → `fallback_to_previous` tool. Agent decides whether the failure warrants a fallback to Verifier or is final.

**Fallback chain**: A → Verifier → Preparer. `_revert_tasks()` re-marks TUI task board entries at fallback pointer. `_find_stage_pointer()` maps stage names to list indices.

**Lazy config build**: `config_path` built before first script stage (after Verifier passes), not in `tool_def.py`.

**Subagent tools**: removed `SUBAGENT_TOOL_FALLBACK_TO_VERIFIER` — all fallback now uses `fallback_to_previous`.

**Orchestrator helpers added**: `run_preparer_stage()`, `run_verifier_stage()`, `copy_dataset_to_data_prep()` in `data_preparer.py`.

**Tests**: 911 pass (0 failures, 0 errors, 11 skipped), zero session residue.

### 2026-07-01 | Engine v95→v96 Merge — Frozen KAN Grid Parameters

**Root cause**: `list(kan.parameters())` passed ALL parameters to Adam, including `act_fun.*.grid` (B-spline knot positions). Adam shifted the knots, creating non-monotonic spacing → visible IDVG/IDVD plateau jumps in IV curves.

**CompactKAN changes**:
- `_optimizer_parameter_names`: captures pykan's constructor-time trainability contract
- `optimizer_parameters(enabled)`: returns only weights pykan declared trainable, filters out grid/mask/affine metadata
- `grid_audit()`: validates strict knot ordering required by B-spline numerics
- `assert_valid_grids(context)`: raises RuntimeError on non-monotonic grid

**Training call sites**: 11 × `list(x.parameters())` → `x.optimizer_parameters(enabled=True/False)`

**Validation**: 15 × `assert_valid_grids()` calls added after checkpoint loads and per training epoch

**Metrics**: 2 × `kan_grid_audit` dicts (4 blocks each) in final metrics output

### 2026-07-01 | Pipeline Stabilization — Agent Stage Checks, Verifier Entry, Report Fixes

**Agent stage result check**: `_run_agent_stage` now returns `(attempt_dir, ok)`. while+pointer only advances on success; on failure retries or falls back to Preparer.

**Verifier entry fix**: standard dataset copied into `data_prep/attempt_N/` before Verifier checks (was checking empty dirs).

**Verifier→Preparer fallback**: Verifier failure → pointer jumps back to DATA_PREP with handoff history for worker re-processing.

**Report fixes**:
- RUN_SUMMARY 0KB: manifest built after report files written, not before
- operation_log duration_s: `run_stage` wrapped with `time.monotonic()` timing
- ARTIFACT_MANIFEST self-reference: excluded its own entry

**Metrics unification**: `metrics_parser` v90→v93. Engine writes only `v93_safe_residual_metrics.json` (removed v92 double-write).

**Agent stages in stage_results**: verifier/data-prep now recorded in `workflow_runs.stage_results` via `wm.update_stage()` — unified with script stages for RUN_SUMMARY.

**Stage display names**: added `data-prep`/`verifier` entries in `STAGE_DISPLAY_NAMES`.

**v96 bug fix**: `kan_grid_audit` loop variable `audit` shadowed DataFrame `audit` — renamed to `block_audit`.

**Tests**: 911 pass (0 failures, 0 errors, 11 skipped)

### 2026-07-02 | Pipeline Stabilization — Round 2: TUI State, Path Fixes, Prompt Enhancement

**TUI Task State Fixes**:
- `_revert_tasks` was marking ALL stages from fallback pointer as IN_PROGRESS, including never-started PENDING tasks. Fixed to only revert `TaskStatus.COMPLETED` tasks.
- Progress counter: `display_idx` changed from monotonically-incrementing `idx_counter` to `pointer + 1`, correctly showing current position on fallback.
- `_finalize_run` now accepts `board`/`task_map`; on failure marks remaining tasks as DELETED. Caller `tool_def.py` except handler also cleans up tasks.
- Task format updated to `[DG-TFT, ID: 1] [Step-1] Prepare dataset`.

**Rich MarkupError Crash Fix**: `_console_summary` / `_print_diagnosis` / `error_exit` / `tool_def.py` exception prints now all escape user/engine text via `rich.markup.escape()` or `markup=False`. Engine error text with `[...]` no longer triggers Rich markup parsing.

**Quality Gate Retry Removed**: `_enforce_quality_gate` (~180 lines) replaced by `_check_quality_once` (check only, no retry). Removed helper functions (`_print_retry_start`, `_print_quality_exhausted`, `_copy_retry_to_main`), unused imports (`shutil`, `suggest_adjustments`). Retry loop delegated to future Agent B. 4 related tests updated to new signature.

**Nested Dir Path Fix**:
- Orchestrator `output_dir` now uses `os.path.abspath` (was relative `./session/uuid/runs/run_0001`)
- `run_model.py:resolve_path` adds final `.resolve()` as defense layer
- Doubling chain: relative path → quality gate retry wrote to INI → `resolve_path` resolved relative to config parent → doubled nesting

**Directory Flattening**: `DG_TFT_RUNS_SUBDIR` changed from `"runs"` → `""`. Paths: `session/<uuid>/runs/run_NNNN` → `session/<uuid>/run_NNNN`.

**Preparer Prompt Enhanced**: Worker prompt now includes explicit wide-format XLSX reshape instructions (`pd.melt`/`df.stack` → standard 4 columns `VD_V VBG_V VTG_V ID_A`).

**Tests**: 915 passed, 11 skipped

### 2026-07-02 | Full v96 Engine Sync + Pre-trained PKAN + First Full Pipeline Pass

**v96 Engine Sync**:
- Replaced dg_tft_full_model.py entirely with v96 baseline (previously only merged CompactKAN grid freeze)
- v96 additions: plateau jump diagnostics + 3 gate checks, KAN grid monotonicity audit, VDS=0 exclusion, 6-threshold quick-run relaxation
- Ported 3 customizations: skip-precision-gates (3 gate locations), Temperature KAN configurability (ResidualDiagnostic + 3 CLI args)
- Remaining 5 engine files + 8 dynamic files: MD5-identical to v96

**Bug Fixes**:
- dg_tft_full_model.py: Fixed variable name clash where loop variable udit overwrote DataFrame udit (v96 inherited bug, re-introduced after sync)
- param_defs.py: 9 defaults aligned to v96 INI (learning_rate 0.002->0.0012, cv_weight 8.0->10.0, symbolic_training_weight 0.015->0.05, etc.)
- run_model.py: Fallback defaults aligned to v96 (epochs, lr, early_stop, shape periods)
- Removed redundant CV_original_train.xlsx backup file that interfered with verifier

**Pre-trained PKAN Model**:
- v96 ships with pre-trained KAN weights at trained_model/base_physical_pkan/ (pkan_torch_model.pt 53KB)
- Correct path in our repo: src/device_modeling/trained_model/base_physical_pkan/ (gitignored)
- Our RELEASE_ROOT = CODE_ROOT.parent resolves to src/device_modeling/, not src/
- Previous fallback metadata (1315 bytes, no real weights) caused base IV-PKAN to train from scratch, converging to on=0.074 > 0.06 gate
- After fixing path, base IV-PKAN converges to on=0.049 (matches v96 standalone), passes all quality gates

**First Full Pipeline Pass**:
- All 5 stages DONE: verifier(ok) check-data(ok) fit-physical(ok) train-static(ok) train-hysteresis(ok)
- train-static ~1h52m, 0 errors, 10/10 artifacts present
- Metrics: iv_rms=0.248, residual_iv_rms=0.229, residual_cv_rms=0.021, hysteresis=0.048

**Tests**: 916 passed, 11 skipped

### 2026-07-02 | Step 8: Agent Infrastructure

**Tool Definitions** (subagent_tools.py):
- 3 new tool schemas: ~~switch_kernel (Agent B kernel switch — removed 2026-07-02, only EKV Level-2 exists)~~, accept_quality (accept current quality), contribute_knowledge (Agent E knowledge feedback)
- New _contribute_knowledge_handler: writes to knowledge/_discovered/run_{id}_{timestamp}.md
- _STAGE_TOOLS registry extended to all 6 stages: train-static/train-hysteresis each with 4 tools (inspect_xlsx + adjust_params + fallback_to_previous + accept_quality), final-interpret with contribute_knowledge
- get_tools() now accepts adjust_params_override parameter for auto-generated stage-specific schemas

**Parameter Validation Pipe** (quality_guard.py):
- apply_agent_adjustments(config_path, adjustments, ini_section) -> snapshot: writes INI + returns operation snapshot
- snapshot_ini_section(config_path, ini_section) -> str: reads INI section snapshot

**Generic Agent Run Loop** (orchestrator.py):
- _spawn_modeling_agent() — generic Agent B/C/D spawn:
  - Auto-generated adjust_params schema (param_defs -> params_to_tool_schema)
  - Per-stage tool sets
  - stop_mode="schema", 4 decisions: adjust_params / fallback_to_previous / accept_quality / error
  - Injects user preferences + soft guard (get_soft_guard_prompt())
  - try/except wrapper for mock/test context resilience
- _process_agent_decision() — decision processing pipe:
  - adjust_params -> validate_agent_adjustments() -> apply_agent_adjustments() -> retry
    - fallback_to_previous -> record FALLBACK_SENT -> pointer jump upstream
  - accept_quality -> record AGENT_ACCEPT -> advance
- _spawn_agent_e() — Agent E final interpretation: stop_mode="any" + contribute_knowledge
- _assess_stage_quality() — pure rule-driven quality check (no LLM)
- while+pointer script stage retry loop: hard failure + quality fail both trigger agent decisions
- 4 mapping tables: _STAGE_HAS_MODELING_AGENT, _UPSTREAM_STAGE, _AGENT_LABELS
- Old placeholders replaced: _check_quality_once / _agent_stage_placeholder / _agent_e_placeholder

**Tests**:
- test_subagent_unit.py: 8 new (train-static/hysteresis/final-interpret tool registry, adjust_params_override, contribute_knowledge handler)
- dg_tft_quality_guard_test.py: 4 new (apply_agent_adjustments write/create section, snapshot_ini_section normal/missing)
- dg_tft_orchestrator_test.py: 9 new (_stage_has_modeling_agent, _normalize_tool_name, _process_agent_decision adjust/accept/fallback/empty)
- **936 passed, 11 skipped** (up from 916, +20 new tests)

**Agent Decision Matrix**:
| Agent | Trigger | Available Tools |
|-------|---------|----------------|
| A | check-data hard failure | fallback_to_previous -> Verifier |
| B | fit-physical hard failure / quality fail | adjust_params / fallback_to_previous / accept_quality |
| C | train-static hard failure / quality fail | adjust_params / fallback_to_previous / accept_quality |
| D | train-hysteresis hard failure / quality fail | adjust_params / fallback_to_previous / accept_quality |
| E | Pipeline end (success or failure) | contribute_knowledge + free text |

### 2026-07-02 | Agent E Knowledge Feedback Fixes

- **contribute_knowledge -> stop_schema**: Changed from handler (LLM called repeatedly, producing many fragment files) to stop_schema tool (called once, producing a single complete contribution)
- **File naming**: Changed from 
un_{run_id:04d}_{timestamp}.md (cross-session collision, meaningless) to {title_slug}_{timestamp}.md (slugified from agent's English title, human readable)
- **minimal_run skips Agent E**: Added _is_minimal_run(config_path) check
- **Contribute only when novel**: system prompt now marks contribute_knowledge as OPTIONAL, only called for genuinely novel findings
- **English enforced**: Agent E prompt + contribute_knowledge tool description require English only
- **gitignore**: Added /knowledge/_discovered/ (not committed to repo)

### 2026-07-02 | Expert Knowledge Files + Three-Way INI/Code/MD Consistency Audit (3 Fix Rounds)

**switch_kernel Removal**:
- Engine only has EKV Level-2 kernel; no kernel abstraction layer exists. switch_kernel wrote to INI but engine never consumed it
- Full removal: subagent_tools.py, orchestrator.py, tests, all docs (6 files, -100 lines)

**5 Tier-1 Expert Knowledge Files Created**:
- Four-dimension knowledge classification: principles/behavior/current capability/experience (JSONL extraction)
- Template: 6 sections (Global Context → Upstream/Downstream → INI Parameters → Quality Metrics → Engine Behavior → Failure Patterns)
- Write-review-audit workflow: subagent researches code → writes MD → review subagent checks facts/omissions → I (human) cross-check INI parameters
- 5 files: data_inspection.md (~240 lines), physical_kernel.md (~400 lines), static_model.md (~550 lines), hysteresis.md (~430 lines), final_interpret.md (~270 lines)
- Directory unified: knowledge/ (repo root), loader _default_knowledge_dir() updated, constants.py updated
- Section 6 (Failure Patterns) left empty, marked for population from Zikang's Codex JSONL logs

**Round 1 Review (subagent cross-validation)**:
- 5 explore subagents ran in parallel, found 40+ factual errors/omissions/oversimplifications
- Critical errors: physical_kernel.md claimed CUDA (pure CPU scipy in reality), static_model.md IV-PKAN input features completely wrong, hysteresis.md ALL enum values wrong, final_interpret.md cascade logic completely backwards (pipeline does NOT stop immediately)

**Round 2 Review (three-way INI/Code/MD consistency audit)**:
- 68 parameters audited across all sources (6 PHY + 50 FULL + 12 HYS)
- Found 8 dead params ([base_pkan] section never passed to subprocess)
- Found 5 default mismatches (base_iv_epochs 700→540, hysteresis epochs 1800→4000, etc.)
- Found critical bug: cv_train_subset enum included "all_gate_cv" but engine only accepts "all" (agent adjust would cause argparse crash)
- Found cv_residual_level hi=3 but engine argparse choices=[0,1,2]
- Found temperature param 4/6 key name mismatch causing agent adjustments to be SILENTLY IGNORED
- param_defs.py fixes: regime/amp_parameterization enums aligned to engine argparse
- Removed dead param: hysteresis print_every (never passed to subprocess)
- quality_guard.py synced: auto-adjust strategy "aggressive"→"independent", removed print_every from bounds/whitelist

**Round 3 (Missing Parameter Gap Fill)**:
- 21 params that run_model.py passes to subprocess but were missing from param_defs now added:
  - [full_model] 9: reset_base_from_physical_fit, train_base_iv_pkan, train_iv_residual_stage, freeze_iv_residual_after_stage, iv_residual_holdout_stride, iv_residual_off_anchor_weight, disabled_iv_pkan/cv_pkan/cv_charge_pkan
  - [symbolic] 10: edge_steps, edge_topk, iv_edge_topk, iv_adaptive_top2_mse, iv_calibration_max_terms, edge_finetune_steps, network_finetune_steps, device_iv_finetune_steps, device_iv_measured_steps, device_cv_finetune_steps
  - [validation] 2: safe_scale_steps, joint_cv_safe_scale_search
- Temperature key naming fixed: param_defs now has ini_key field mapping; apply_agent_adjustments refactored to accept param_table and write to correct section+key
- _STAGE_PARAM_MAP extended with [symbolic] and [validation] sections
- quality_guard bounds + whitelist updated with all new params
- Final: 68 → 71 parameters, agent can now adjust ALL modeling hyperparameters forwarded by the pipeline

**Knowledge MD Sync with Code**:
- physical_kernel.md: CUDA→pure CPU scipy, engine path corrected, IV residual 6-factor weight table, CV trap fit separate coefficients, exact regularization values, f_scale=0.55, CGG 10kHz retained for fitting, auto-orientation span_ratio=1.55, iv_stage_alignment modifies data
- static_model.md: finetune_base_pkan default accuracy, cv_residual_level semantics, IV-PKAN 6 features, two scheduling mechanisms (loss eval + CV parameter phase architecture), CV residual 21 slots/7 active, final-report kind="script", training phase architecture
- hysteresis.md: ALL enum values corrected, epochs/hidden runtime defaults documented, print_every removed from table, knowledge-not-injected status documented
- final_interpret.md: cascade logic fully rewritten, Agent E only receives last attempt per stage, upstream field always empty, contribute_knowledge mechanism, minimal_run skip, pre-computed error classification

**Tests**: 935 passed, 11 skipped

---

## 2026-07-02 — Expert Knowledge File 3-Round Audit + Fix

**Background**: The 5 knowledge files had already been through 3 rounds of subagent review, but agents are not fully reliable — additional verification needed.

**Audit Architecture**: Three rounds of 5 parallel explore subagents, with human review gate between rounds:

- **Round 1 — Positive Claim Extraction**: Exhaustively list every verifiable factual claim in each MD (param defaults, enum values, behavior descriptions, algorithm steps), with source files to cross-check. Output: 187+83+82+68+49 = 469 claims.
- **Round 2 — Claim-by-Claim Verification**: Verify each claim against actual source code, outputting ✅/❌/⚠️/📋. For parameter claims, three-way cross-check of param_defs default → run_model.py passed value → engine argparse default. Output: 18 errors + 27 oversimplifications + 17 TODO items.
- **Round 3 — Reverse Scanning**: Ignore MDs, read only source code to find information critical for agent decisions that is completely absent from the knowledge files. Output: 48 missing items (10+10+7+8+13).

**Key Findings (18 errors)**:

| File | Error | Severity |
|------|-------|----------|
| data_inspection.md | `_run_stage_diagnosis` claimed to run for check-data but code uses `break` first | critical |
| data_inspection.md | "All check-data errors classified as DATA_ERROR" — keyword matching is stage-agnostic, CONFIG_ERROR/etc. keywords take priority | high |
| data_inspection.md | Artifacts checked by `run_artifact_stage` are `.pt` checkpoints (not `dataset_validation.json`) | medium |
| physical_kernel.md | IV sampling subthreshold weight MD says ×3, code: base 1.0 + 3.0 = effective 4.0× | medium |
| physical_kernel.md | IV sampling on-region weight MD says ×1.5, code: base 1.0 + 1.5 = effective 2.5× | medium |
| physical_kernel.md | MISSING_ARTIFACT_ERROR category (6 keywords, 0.9 confidence) entirely absent from error table | medium |
| static_model.md | `cv_residual_level` MD says "2 or 3" but engine choices=[0,1,2], max is 2 | high |
| static_model.md | CV training phase 4 boundary params marked INI-writable but absent from param_defs | high |
| static_model.md | "50 parameters total" — actual count is 51/63/71 depending on counting dimension | low |
| static_model.md | IV-PKAN output names don't match engine CORR_NAMES (dlog_rs_fwd/reverse vs dpsi_V) | medium |
| static_model.md | `base_iv_epochs` MD default 700, engine argparse default 540 | medium |
| hysteresis.md | `fast_amp_parameterization` default stated "None→fallback", actual param_defs default "conservative" | high |
| hysteresis.md | `slow_amp_parameterization` default stated "None→fallback", actual param_defs default "independent" | high |
| final_interpret.md | Successful retries (attempt_n > 1) NEVER logged to operation_log — only agent decision entry | critical |
| final_interpret.md | PREP_RETRY trigger defined in enum but NEVER emitted in production code | medium |
| final_interpret.md | B/C/D quality_fail trigger only used in prompt context; actual log writes AGENT_ADJUST | medium |
| final_interpret.md | Cascade victim description assumes operation_log entries — first-attempt hard failures produce none | high |

**Key Missing Items (48 total)**:

- **data_inspection.md** (10): Agent A prompt missing stdout summary (only truncated 2K-char stderr), CV normalization silently returns on Exception, diagnostic keyword concatenation boundary false positives, CV validation minimum 3 rows gate → "unknown" layout, I-t auto-discovery "it" substring match false positives
- **physical_kernel.md** (10): **Critical**: Agent B prompt gets ZERO engine knowledge (generic 8-line template only), 9 engine-internal hardcoded thresholds not INI-adjustable (RS solve threshold / VDS filter 0.55V / Stage3 transfer ratio / smooth_pos ε etc.), silent data drops (NaN/non-positive capacitance/no "long" sheet — all silently discarded, no warnings)
- **static_model.md** (7): **Critical**: 5 additional precision gates undocumented (base IV region 7-check gate / IV residual heldout gain @ 8e-4 / region degradation @ 0.008 / KAN grid audit / charge passivity), 30+ diagnostic files only 12 listed, engine error messages not classifier-mapped
- **hysteresis.md** (8): **Critical**: Acceptance gate RuntimeError as distinct hard failure path undocumented, 7 acceptance gate thresholds hardcoded not adjustable, data filtering can silently drop ALL segments → "no SRH tau labels" RuntimeError
- **final_interpret.md** (13): **Critical**: inline prompt conflates ok/quality_pass (uses exit code only, PASS may mean quality_pass=False), error text never injected into prompt, Agent E receives NEITHER output_dir NOR config_path, 15-line output truncation (not saved to disk), minimal_run skip broken on failure path

**Knowledge File Fixes Applied** (~40 modifications across 5 files):

| File | Modifications |
|------|---------------|
| data_inspection.md | `_run_stage_diagnosis` paragraph replaced with correct note, error classification fixed to note stage-agnostic keyword matching priority, new "Agent A prompt content" section (stderr 2K-only, stdout summary not in prompt), CV normalization silent-exception caveat, artifact stage artifacts clarification (.pt not dataset_validation.json), I-t substring match warning, missing error keywords added |
| physical_kernel.md | IV sampling weights corrected ×3→×4, ×1.5→×2.5, MISSING_ARTIFACT_ERROR category + other missing keywords added, knowledge-not-injected critical status box, engine-internal hardcoded thresholds section (9 non-adjustable params), adjust_params schema diagnostic-context note, CV auto-orientation hardcoded thresholds warning, silent data filtering section |
| static_model.md | cv_residual_level "or 3"→"2" all occurrences, CV phase params "INI-writable"→"engine-internal only", parameter count "50"→"63/71", IV-PKAN output names corrected to engine CORR_NAMES (dpsi_V replaces dlog_rs_*), new 5-level precision gate suite section, base_iv_epochs default discrepancy footnote, engine error messages classification table |
| hysteresis.md | fast_amp_parameterization default "None→fallback"→"conservative", slow_amp_parameterization default "None→fallback"→"independent", acceptance gate RuntimeError as distinct failure path documented, 7 hardcoded acceptance gate thresholds table, silent data filtering → RuntimeError triggers documented, training constant list (lr/wd/grad_clip not adjustable), HSPICE non-blocking failure note |
| final_interpret.md | operation_log coverage gaps documented (hard failures + successful retries not logged), PREP_RETRY trigger never-emitted note, quality_fail trigger usage clarification (Agent A only), inline prompt ok-vs-quality_pass conflation warning, error text absence from prompt documented, Agent E full tool list (bash/grep_file/web_search/web_fetch), 15-line console truncation warning, minimal_run skip broken on failure path note, agent_reasoning/ini_snapshot fields documented, quality threshold source note (configurable via agent_configs, not constants) |

**Audit Reports Archived**: `doc/audit_r1/` `doc/audit_r2/` (Round 1 claims + Round 2 verification + Round 3 reverse scan results — can be cleaned up later)
