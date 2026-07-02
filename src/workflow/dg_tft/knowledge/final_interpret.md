# Final Interpretation Knowledge

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
| check-data | script | No (Agent A only) | Hard failure only |
| fit-physical | script | Yes (Agent B) | Hard failure / quality below threshold |
| train-static | script | Yes (Agent C) | Hard failure / quality below threshold |
| train-hysteresis | script | Yes (Agent D) | Hard failure / quality below threshold |
| **final-report** | **noop** | **No (Agent E)** | **— (always runs at end)** |

### Agent E Position and Role

Agent E is the **final interpreter** for Stage 7 of 7. Unlike other stages, final-report
is a **noop** — it performs no script execution, has no INI parameters, and has no quality
thresholds of its own. Agent E has exactly **three tasks**:

1. **Path Recommendation** — read `operation_log.json` (all stages, all attempts, all metrics,
   all agent decisions) and recommend the best chain across attempts.
2. **Root Cause Diagnosis** — if the pipeline failed or produced marginal quality, identify the
   FIRST stage in the chain that failed independently (not a cascade victim) and explain why.
3. **Knowledge Contribution** (optional) — if a genuinely novel, generalizable discovery is
   observed, call `contribute_knowledge`. Most runs do NOT trigger this.

Agent E runs with `stop_mode="any"` — it produces a free-text Markdown report and may
optionally call the single `contribute_knowledge` tool. The report is printed to console
(not automatically appended to RUN_SUMMARY.md).

Agent E receives a pre-formatted stage results summary from the orchestrator, including
per-stage pass/fail status and metric values. It has access to standard SubAgent tools
(`read_file`, `glob_file`) to inspect artifacts, but typically reasons from the
operation log alone.

## 2. Cross-Stage Quality Reference

### Unified Metric Table

| Stage | Primary Metric | Source File | Threshold | Below Means | Above → First Suspect |
|-------|---------------|-------------|-----------|-------------|----------------------|
| check-data | (none) | (none) | (none) | Binary gate: structural validation passes | Script crash → data preparation issue (Agent A fallback) |
| fit-physical | `iv_rms_log10_error_dec` | `01_physical_kernel/fitted_params.json` → `metrics_summary.iv_rms_log10_error_dec` | **0.35** | IV fit within ~2.2× factor, acceptable physical baseline | Check `fit_history` for rejected stages → `max_nfev_scale`, then data quality |
| train-static | `residual_iv_rms` | `03_full_model/v93_safe_residual_metrics.json` → `residual_iv_rms` | **0.35** | Static IV residual OK — KAN correction successful | Check `epochs`/`base_iv_epochs`, then upstream fit-physical |
| train-static | `residual_cv_rms` | same file → `residual_cv_rms` | **0.06** | CV derivative error within ~15% relative | Check `cv_weight` >10, then CV residual level, then upstream physical kernel |
| train-hysteresis | `step_log_rmse_median` | `04_hysteresis/it_hysteresis_analysis_summary.json` → `step_log_rmse_median` | **0.08** | Two-trap pre-analysis fits within ~20% relative error | Check I-t data quality (noisy transients, insufficient time points), then upstream static model |

### Secondary Diagnostic Metrics (Not Gated, but Useful for Diagnosis)

| Stage | Metric | Source File | What It Tells You |
|-------|--------|-------------|-------------------|
| fit-physical | `iv_on_rms_log10_error_dec` | `fitted_params.json` | On-region IV fit quality — if good but global is poor, subthreshold is the problem |
| fit-physical | `fit_nonmonotonic_steps` | `fitted_params.json` | Count of non-monotonic fit predictions — >0 indicates convergence to non-physical solution |
| train-static | `base_iv_training.*.top1_compatibility` | `v93_safe_residual_metrics.json` | Per-edge symbolic MSE — identifies which KAN edges failed the precision gate |
| train-static | `safe_iv_residual_scale` | `v93_safe_residual_metrics.json` | 0 means IV residual was disabled (model relies purely on PKAN corrections) |
| train-static | `safe_cv_residual_scale` | `v93_safe_residual_metrics.json` | 0 means CV residual was disabled — PKAN alone couldn't fit CV |
| train-hysteresis | `student_teacher_rmse_dec_mean` | `srh_top2_regime_bank_summary.json` | How well the KAN student learned from teacher fits — high = KAN capacity or training issue |

### Stage Optionality

- **train-hysteresis** is optional: skipped when `DatasetProfile.has_hysteresis` is False
  (no I-t measurement data). Its absence is normal, not a failure.

## 3. Stage Dependency and Cascade Rules

### 3.1 Upstream → Downstream Data Flow

| From | Key Artifact | To | Consumed As |
|------|-------------|----|-------------|
| Verifier | Verified dataset directory (IV/CV workbooks) | check-data | Input data for structural validation |
| check-data | `dataset_validation.json` | fit-physical | Confirms structural integrity; IV/CV file paths |
| check-data | `CV_train.xlsx` (normalized) | fit-physical | Paired-column CV data, ready for loading |
| fit-physical | `01_physical_kernel/fitted_params.json` | train-static | EKV parameters + fit history → physical baseline for PKAN |
| fit-physical | `01_physical_kernel/dg_tft_ekv_level2_fitted.va` | train-static | Physical kernel Verilog-A reference |
| train-static | `03_full_model/v93_safe_residual_state.pt` | train-hysteresis | Accepted static model checkpoint for steady-state current |
| train-static | `03_full_model/dg_tft_static_compact.va` | train-hysteresis | Static Verilog-A used as baseline for dynamic model |
| train-hysteresis | `04_hysteresis/srh_top2_regime_banks/combined/dg_tft_dynamic_srh.va` | final-report | Dynamic SRH Verilog-A for deployment |

### 3.2 Cascade Skip Logic

When a **required** stage fails (hard failure or exhausted retries), the orchestrator
immediately breaks out of the stage loop. **Downstream stages are never executed** — they
do not appear in `operation_log.json` and produce no artifacts.

**Exception**: `train-hysteresis` is `required=False` — if it fails, downstream stages are
not affected (there are none after it), but the pipeline does not break.

**Root cause identification**: the FIRST stage in the pipeline chain that failed
**independently** is the root cause. All later stages absent from the operation log are
cascade victims, not independent failures.

| First Failing Stage | Cascade Victims | Root Cause Presumption |
|--------------------|-----------------|----------------------|
| check-data | fit-physical, train-static, train-hysteresis | Data preparation issue — file structure invalid |
| fit-physical | train-static, train-hysteresis | Physical model cannot fit the data — check fit_history, Vth bounds |
| train-static | train-hysteresis | Static model training failed — check base IV precision gate, CV convergence |
| train-hysteresis | (none — last script stage) | Hysteresis training failed — check I-t data quality, SRH acceptance gate |

**Important**: When fit-physical fails, train-static and train-hysteresis are NOT
independently failed — they were never given the chance to run. Agent E must diagnose
the FIRST failure, not report cascade victims as separate failures.

### 3.3 Attempt Relationships

`operation_log.json` records per-stage attempt sequences. Each attempt includes:

- `n`: attempt number (1-indexed)
- `trigger`: what spawned this attempt (listed below — note this is a subset, some triggers like `fallback_sent` and `agent_accept` are also recorded)
- `upstream`: which upstream attempt this attempt was based on (e.g., `"fit-physical.2"` → "train-static.1 used fit-physical attempt 2 output")
- `metrics`: per-attempt metric values
- `adjustments`: parameter changes applied (from agent decisions)
- `ok`: whether the subprocess exited with code 0
- `quality_pass`: whether metrics met thresholds (`True`/`False`/`None`)

**Common triggers in operation_log**:
- `trigger=initial` + `n=1`: first execution with default parameters
- `trigger=quality_fail` + `n>=2`: agent-adjusted retry after quality below threshold
- `trigger=agent_adjust`: agent called `adjust_params`, INI was updated, this is the retry
- `trigger=fallback_sent`: agent called `fallback_to_previous`, recorded at the stage that initiated the fallback
- `trigger=agent_accept`: agent called `accept_quality`, stage continues without retry
- `trigger=verifier_fail`: data-prep worker retry after Verifier rejection
- `trigger=prep_retry`: data-prep worker internal retry

**Note on re-run triggers**: When a script stage re-runs because upstream changed
(e.g., fit-physical falls back to check-data, then check-data re-executes and fit-physical
re-runs), the re-run is logged with `trigger=initial` at the re-executing stage. The
`upstream` field records which upstream attempt is the dependency.

**Path recommendation**: Agent E should recommend the chain consisting of each stage's
best-scoring attempt. Example: `fit-physical.2 → train-static.1 → train-hysteresis.1`
if `fit-physical.2` had better metrics than `fit-physical.1`, and downstream stages only
used the best upstream output.

## 4. Cross-Stage Diagnostic Patterns

| Symptom Pattern | Likely Root Cause | Evidence to Check |
|----------------|-------------------|-------------------|
| fit-physical `iv_rms` high + train-static `residual_iv_rms` high | Physical kernel is the bottleneck — KAN cannot compensate for a fundamentally wrong EKV baseline | Check `fit_history` in `fitted_params.json` for rejected stages (`accepted=false`). Check if Vth values are at PARAM_BOUNDS limits. Check IV plots for systematic shape mismatch. |
| fit-physical `iv_rms` OK (≤0.35) + train-static `residual_cv_rms` high (>0.06) | CV weight insufficient or CV residual level too low — KAN is underfitting the derivative observable | Check `cv_weight` setting in INI. Check `safe_cv_residual_scale` — if 0, CV residual was force-disabled. Check `cv_residual_level` — increase to 2 or 3. |
| fit-physical OK + base IV precision gate failed (`base_iv_max_top1_compatibility` > 0.015) | PKAN capacity (`kan_hidden`, `kan_grid`) or symbolic training budget insufficient for this device's complexity | Check `base_iv_top1_symbolic_weight` (should be ≥0.3). Check per-edge Top-1 MSE in `base_iv_training.*` — identify which edges fail. Consider increasing `base_iv_epochs` or relaxing gate to 0.03. |
| train-static hard crash — CUDA OOM or NaN in loss | GPU memory exhausted or training numerically unstable | Check `batch_iv` and `batch_cv` — reduce both for OOM. For NaN: check if `grad_clip` is too high, if `learning_rate` is too aggressive, or if the dataset contains extreme outlier values. Check if `cv_residual_level` > 0 — CV charge differentiation can amplify numerical noise into NaN. |
| train-static `residual_iv_rms` OK + `residual_cv_rms` OK + HSPICE equivalence failed | HSPICE environment issue (license, executable path) — not a modeling failure | Check `hspice.log` for license errors. The Verilog-A files are valid; HSPICE is deployment validation, not a training gate. |
| train-static OK + train-hysteresis `step_log_rmse_median` high (>0.08) | I-t data quality is poor — noisy transients, insufficient time points per step, or SRH model fundamentally mismatched | Check `it_measurement_audit.json` for per-bias issues. Check `it_step_response_fit.csv` — if many segments have `ok=false`, the two-trap model cannot fit the data. Parameter tuning CANNOT fix this metric (it comes from pre-analysis, not KAN training). |
| train-hysteresis SRH-KAN acceptance gate failed but `step_log_rmse_median` OK | KAN student failed to learn from viable teacher fits — capacity or training issue | Check `student_teacher_rmse_dec_mean` vs teacher RMSE. High student-teacher gap → increase `epochs` or `hidden_capture`/`hidden_emission`. Check per-group training histories in `srh_top2_regime_bank_summary.json`. |
| All modelling stages failed | Data preparation is the common upstream source of error | Check `dataset_validation.json` from check-data. Check Verifier issues. Check if `CV_train.xlsx` normalization introduced artifacts. Check raw IV files for missing columns or format errors. |
| fit-physical OK + train-static `residual_iv_rms` high + `safe_iv_residual_scale=0` | IV residual KAN was disabled by safety scan — the bare PKAN corrections were insufficient | Check `iv_residual_stage_heldout_gain` — if negative, the residual KAN hurt held-out data and was force-disabled. Increase `iv_residual_stage_epochs` or adjust `iv_residual_stage_linear_weight`. |
| fit-physical `cv_rms_log10_error_dec` high + train-static `residual_cv_rms` high | CV gate orientation may be wrong — CBG↔CTG labels swapped but auto-correction missed it | Check `v92_phase1_cv_gate_orientation.json` in fit-physical output. Check `v92_cv_gate_orientation.json` in train-static output. If `Cox_t/Cox_b < 1.40`, the gate auto-orientation does NOT trigger — a manual orientation correction may be needed. |

## 5. Knowledge Contribution Guidelines

### 5.1 When to Contribute

Agent E should call `contribute_knowledge` **ONLY** when all three conditions are met:

1. **Generalizable pattern** — the observation holds across conditions and is not single-run noise.
   Example: "max_nfev_scale ≥ 3.0 consistently required for datasets with >60000 IV data points."
2. **Specific conditions** — the pattern includes contextual parameters (data size, VDS range,
   device type, temperature range) that define when it applies.
3. **Future utility** — the pattern would help future runs of the same stage make better
   initial decisions or avoid known traps.

### 5.2 What NOT to Contribute

Do NOT call `contribute_knowledge` for:

- Single-run metrics or parameter values without generalizable conditions
  (e.g., "This run had IV RMS 0.04" — not useful alone)
- Known facts already documented in the project knowledge base
  (e.g., "cv_weight=10 is the default" — already in static_model.md)
- Speculation without evidence (e.g., "may be because the subthreshold noise was high")
- Routine parameter-tuning observations (e.g., "learning rate 0.0012 was too high, 0.0006 worked")

### 5.3 Output Format

The `contribute_knowledge` tool accepts:
- `title`: Short English title summarizing the discovery (used as filename slug)
- `content`: Full discovery text in English Markdown — must include conditions,
  observations, metric values, and actionable conclusions

The tool writes to `knowledge/_discovered/{title_slug}_{timestamp}.md`. The content
is injected into future agent system prompts for cross-run learning.

## 6. Report Output Structure

Agent E's Markdown report is output to console. Recommended structure:

### 6.1 Recommended Path
```
fit-physical.2 → train-static.1 → train-hysteresis.1
```
Each stage lists its best attempt with rationale: which metrics improved, which
agent decision drove the improvement, and whether quality was borderline or solid.

### 6.2 Modeling Status
Overall assessment: **success** (all stages passed quality gates), **partial**
(hysteresis skipped or one metric marginally above threshold but accepted),
or **failed** (pipeline terminated at a specific stage).

Include quality grade: **excellent** (all metrics well below thresholds), **good**
(all metrics below thresholds), **borderline** (metrics near thresholds, quality warnings),
**poor** (one or more metrics above thresholds).

### 6.3 Metric Breakdown
Per-stage per-attempt table with:
- Stage and attempt number
- Primary metric value
- Threshold comparison (pass/fail, margin)
- Agent decision (if any) and its effect
- Duration and trigger type

### 6.4 Failure Diagnosis (if applicable)
- First failing stage (root cause)
- Error classification (DATA_ERROR, TRAINING_NUMERICAL_ERROR, CONFIG_ERROR, HSPICE_ENV_ERROR, MISSING_ARTIFACT_ERROR, UNKNOWN_ERROR)
- Cascade victims (stages that never ran)
- Confidence level (high/medium/low based on error keyword matching confidence scores)
- Recommended next action (data re-preparation, parameter tuning, environment fix)

### 6.5 Input/Output Interpretation
- Data source: dataset size, IV/CV families present, I-t data availability
- Generated models: which Verilog-A files were produced and their trustworthiness
  (trusted if all quality gates passed, caution if borderline, untrusted if gates failed)
- Deployment readiness: whether the output model is suitable for circuit simulation

## 7. Common Failure Scenarios and Diagnoses

### fit-physical terminated at stage 1 (low_vds_transfer)
`fit_history[0].accepted = false` → the initial threshold extraction failed. This is
usually a data issue: VDS range is too narrow for the low-VDS filter (≤0.55V),
or ID_A contains only noise in the measurement range. Check if the dataset has
measurements at multiple VDS voltages. If only one VDS is present, the physical
kernel cannot separate Vth_t from Vth_b.

### fit-physical stuck at max_nfev (fit_history shows `status=0, message="\`max_nfev\` reached"`)
The optimizer needs more iterations. This is common on large datasets. Increasing
`max_nfev_scale` from 1.0 to 2.0-3.0 typically resolves it. Upper bound is 5.0
(round(120 × 5.0) = 600 NFEV each for stages 1-5). **Stage 6 (cv_shared_trap) NFEV is
NOT scaled** — fixed at 450 regardless of max_nfev_scale. If the problem persists at
5.0, the issue is not optimization budget — suspect data quality or model mismatch.

### train-static base IV precision gate failed
The PKAN cannot produce Top-1 symbolic replacements of sufficient accuracy. This is
common for devices with unusual transfer characteristics (steep subthreshold swing,
multiple slope regions). Check `base_iv_training.best_restart.top1_compatibility` —
if it's close to 0.015 (e.g., 0.018), relaxing the gate to 0.02 may be acceptable.
If it's >0.05, increase `base_iv_top1_symbolic_weight` to 0.5-0.7 and/or increase
`base_iv_epochs` to 1400-2100.

### train-hysteresis step_log_rmse_median > 0.15
The two-trap pre-analysis cannot fit the I-t transients — likely the data contains
non-exponential transients (e.g., multiple trap species beyond fast/slow, or
significant measurement noise). This metric cannot be improved by KAN parameter
tuning — it comes from `scipy.optimize.least_squares` fits run during pre-analysis.
If `fallback_to_previous` to train-static does not resolve it, the I-t data itself
may need re-measurement with higher time resolution or better signal-to-noise ratio.

### train-static CUDA OOM or NaN in loss
GPU memory exhausted or training numerically unstable. For OOM: reduce `batch_iv` to
2048 or 1024, reduce `batch_cv` to 256 or 128. For NaN: check if the CV charge
differentiation produces extreme values — try `cv_residual_level=0` or `cv_weight=1.0`
temporarily to isolate the NaN source. Reduce `learning_rate` to 0.0005 or increase
`grad_clip` to 2.0. If NaN persists, check the dataset for invalid values (negative
capacitances, zero-bias division).
