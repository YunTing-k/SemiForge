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
807 tests pass (12 skipped). Run: `python -m unittest discover -s test -p "*_test.py"`.

### Agent intervention boundaries
LLM/subagent only at 3 points: data preparation, format verification, failure diagnosis. Training execution, config generation, quality gating, and report building are pure rule-based code.
