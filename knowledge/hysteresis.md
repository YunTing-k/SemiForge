# Hysteresis Knowledge

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
| **train-hysteresis** | **script** | **Yes (Agent D)** | **Hard failure / quality below threshold** |
| final-report | noop | No (Agent E) | — (always runs at end) |

Agent stages spawn LLM subagents. Script stages run subprocesses — fit-physical uses scipy (CPU-only), train-static and train-hysteresis use CUDA Python.
When a script stage hard-fails (exit code ≠ 0) or produces quality below threshold,
a **modeling agent** is spawned to adjust parameters and retry, or fallback upstream.

Fallback is a linear one-hop chain upstream: D→C→B→A→Verifier→Preparer.

### train-hysteresis Position and Role

The **train-hysteresis** stage is Stage 6 of 7 — the **optional dynamic modeling stage**.
It trains a fast/slow SRH (Shockley-Read-Hall) trap-based hysteresis model using KAN
networks to predict trap time constants (tau) and amplitudes from bias conditions.
The output combines with the static Verilog-A model to produce a complete dynamic
compact model suitable for transient circuit simulation.

**This stage is OPTIONAL**: it only runs when I-t (current-time) hysteresis
measurement data is available. The `DatasetProfile.has_hysteresis` flag must be True
in the dataset profile; otherwise the stage is skipped with `skip_if_profile_false`.

**Execution**: `CUDA python engine/train_srh_top2_regime_banks.py`, preceded by an
analysis phase (`analyze_it_hysteresis.py`) that pre-fits step responses to extract
teacher labels for the SRH KAN.

**On failure, Agent D** is spawned with four tools: `adjust_params`, `fallback_to_previous`,
`accept_quality`, and `inspect_xlsx`. Agent D's fallback target is **train-static**.
Timeout is 7200s (2 hours).

### Principles: SRH Trap Dynamics and Double-Exponential Modeling

**What is hysteresis in TFTs?** In double-gate TFT devices, when the gate voltage
changes abruptly (a bias step), the drain current does not instantaneously settle
to its new steady-state value. Instead, it drifts over time — typically microseconds
to seconds — due to charge trapping and de-trapping at semiconductor/dielectric
interface states (slow traps) and within the channel or gate oxide bulk. This
time-dependent current response is called hysteresis. It manifests as a difference
between up-sweep and down-sweep IV curves and as a transient response to bias steps.

**SRH (Shockley-Read-Hall) Trap Model**: The engine models hysteresis using a
two-trap SRH formalism. Each trap contributes an exponential decay term to the
time-dependent current:

```
I(t) = I_∞ + A_fast * (e^{-t/τ_fast} - 1) + A_slow * (e^{-t/τ_slow} - 1)
```

where:
- `I_∞` is the steady-state current
- `A_fast`, `A_slow` are fast/slow trap amplitudes (positive for capture, negative for emission)
- `τ_fast`, `τ_slow` are fast/slow time constants (typically microseconds to seconds)

This is not a stretched-exponential in the traditional sense — it uses two discrete
exponential terms with learned time constants, which collectively provide the
multi-timescale behavior characteristic of TFT traps.

**KAN-based Trap Parameter Prediction**: Instead of fitting physically derived
trap equations, the engine trains CompactKAN networks to predict trap parameters
(τ_fast, τ_slow, A_fast, A_slow) as functions of the bias condition:
- **Input features**: VBG, VTG, VDS, temperature, and derived interaction features
- **Output**: log10(τ_fast), log10(τ_slow), and amplitude(s) for each trap
- **Architecture**: TwoLayerTop1KAN with Top-2 symbolic edges (stricter than the IV
  PKAN's Top-1; hysteresis retains two primitives per symbolic edge for accuracy)

**Fast/Slow Regime Split**: The engine partitions the I-t measurement data into
two regimes based on segment duration:
- **Fast regime** (τ < 50 ms): microsecond-scale transients, typically from
  oscilloscope-like capture. Uses `conservative` amplitude parameterization and
  `optimized` optimizer policy by default.
- **Slow regime** (τ ≥ 50 ms): second-scale transients from stress/recovery traces.
  Uses `independent` per-amplitude parameterization (v90 legacy) and `v90` fixed-AdamW
  optimizer policy by default — this is a protected baseline.

Both regimes use separate KAN banks that are trained independently per
(regime, family, process) group, where family ∈ {VBG, VTG} and process ∈ {capture, emission}.

**Dynamic Verilog-A Export**: After training, the symbolically replaced KAN edges
are exported as individual Verilog-A modules. Each regime gets a wrapper module
that evaluates the KAN at runtime to produce τ and amplitude parameters, which
feed into a continuous-time SRH trap module (`dg_tft_dynamic_srh`). The total
terminal current is `I_static × 10^{dlog}`, where `dlog` is the dynamic correction.

## 2. Upstream/Downstream

### Receives from train-static

- I-t measurement data directory (`hysteresis_data_dir` from `[paths]`)
- `03_full_model/v93_safe_residual_state.pt` — the accepted static model checkpoint
- `model_config.ini` — runtime config with `[hysteresis]`, `[paths]`, `[deployment]` sections
- Dataset profile with `has_hysteresis = True` and valid I-t workbook paths

### Delivers to final-report

- `04_hysteresis/srh_top2_regime_banks/combined/dg_tft_dynamic_srh.va` — dynamic SRH Verilog-A module
- `04_hysteresis/srh_top2_regime_banks/combined/dg_tft_dynamic_srh.inc` — dynamic SRH include file
- `04_hysteresis/it_hysteresis_analysis_summary.json` — step-response analysis summary (contains quality metric)
- `04_hysteresis/srh_top2_regime_banks/srh_top2_regime_bank_summary.json` — SRH training summary
- `04_hysteresis/srh_top2_regime_banks/srh_top2_regime_bank_metrics.csv` — per-segment metrics
- `04_hysteresis/srh_top2_regime_banks/srh_top2_regime_bank_predictions.csv` — per-timepoint predictions
- `04_hysteresis/srh_top2_regime_banks/srh_top2_regime_bank_checkpoint.pt` — checkpoint (key artifact)
- `04_hysteresis/it_hysteresis_processed_and_fit.xlsx` — processed I-t data with fits
- `model_config.ini` — potentially modified by Agent D's `adjust_params`

### Fallback Chain

- Agent D falls back to **train-static** (StageName.TRAIN_STATIC), which is upstream.
  This means: if the hysteresis model cannot be trained because the static model
  checkpoint is invalid or produces unrealistic steady-state currents, fallback
  re-runs the static model training.
- When Agent D accepts quality or the fallback resolves, the pipeline proceeds to
  final-report.

## 3. INI Parameters

All parameters below are from `HYSTERESIS_PARAMS` in `param_defs.py`. The stage
reads the `[hysteresis]` INI section — **12 parameters** total. The `[hysteresis]`
section is the **only** section this stage writes to.

### [hysteresis] Section

| Parameter | Type | Default | Bounds/Enum | Description |
|-----------|------|---------|-------------|-------------|
| `enabled` | bool | True | — | Master switch for hysteresis training. **Set False if**: no I-t data is available or hysteresis modeling is not needed. This is the primary skip mechanism — the orchestrator also checks `DatasetProfile.has_hysteresis`. |
| `regime` | string | all | ["all", "fast", "slow"] | Target regime for hysteresis training. `all` — trains both fast and slow banks. `fast` — trains the fast-regime bank only (segment duration < 50 ms). `slow` — trains the slow-regime bank only (segment duration >= 50 ms). **Use `all`** unless specific regime data is known to be problematic. |
| `epochs` | int | 1800\* | [200, 5000] | Training epochs per SRH-KAN bank group. \*Schema default is 1800, but `run_model.py` has a **runtime fallback of 4000** — if the INI `[hysteresis]` section is missing `epochs`, the effective default is 4000. Hysteresis training typically requires more epochs than static fitting because the KAN must learn bias-dependent trap dynamics from noisy I-t transients. **Increase if**: training history shows improvement at the final checkpoint. **Decrease if**: early stopping consistently fires at ~400 epochs. |
| `hidden_capture` | int | 16\* | [4, 32] | Number of hidden units in the capture-phase KAN networks. \*Schema default is 16, but `run_model.py` runtime fallback is **32**. Capture processes (carriers moving into traps) often require more capacity than emission. **Increase to 20-24 if**: capture waveform shape is poorly reproduced. **Decrease to 8 if**: overfitting causes unrealistic tau predictions. |
| `hidden_emission` | int | 20\* | [4, 32] | Number of hidden units in the emission-phase KAN networks. \*Schema default is 20, but `run_model.py` runtime fallback is **32**. Emission (carriers leaving traps) typically has a wider time-constant range and benefits from larger networks. **Increase to 24-28 if**: emission transient response is inaccurate. **Decrease to 12 if**: emission fitting is acceptable but training time is a concern. |
| `amp_parameterization` | string | independent | ["independent", "conservative"] | Default amplitude parameterization for hysteresis model (overridden by regime-specific settings). **`independent`** (v90 legacy, actual default): per-trap direct amplitude prediction. **`conservative`**: bounded total+exchange form: `A_fast = total/2 + exchange`, `A_slow = total/2 - exchange` with tanh clamping. Less prone to oscillation but may underfit. Use `conservative` for stability. Note: `"aggressive"` does **not exist** in the training script's argparse — do NOT use it. |
| `fast_restarts` | int | 1 | [1, 10] | Number of random restarts for fast-transient SRH bank training. Each restart uses a different random seed; the best result is kept. **Increase to 3-5 if**: fast-regime fitting quality is inconsistent across runs (the KAN converges to poor local minima). Each restart costs `epochs` of training time for fast groups. |
| `slow_restarts` | int | 2 | [1, 10] | Number of random restarts for slow-transient SRH bank training. Slow transients typically benefit from more restarts than fast because the second-scale time constants have higher variance in the teacher labels. **Increase to 4-5 if**: slow-regime amplitude or decay is poorly captured. |
| `fast_amp_parameterization` | string | conservative | ["independent", "conservative"] | Amplitude parameterization for the fast-regime SRH bank. Default is `conservative` (schema default written to INI by config builder). Runtime fallback `section.get(key, section['amp_parameterization'])` only triggers if the key is absent from INI — which won't happen since the INI builder always writes the schema default. |
| `slow_amp_parameterization` | string | independent | ["independent", "conservative"] | Amplitude parameterization for the slow-regime SRH bank. Default is `independent` (v90 legacy, schema default). Same runtime fallback note as `fast_amp_parameterization`. |
| `fast_optimizer_policy` | string | optimized | ["optimized", "v90"] | Optimizer policy for fast-transient SRH bank training. `optimized` — AdamW with CosineAnnealingLR scheduler, bounded learning rate decay. `v90` — legacy fixed-AdamW full-budget policy. **Fall back to `v90`** if the optimized policy shows instability (loss spikes, NaN). |
| `slow_optimizer_policy` | string | v90 | ["optimized", "v90"] | Optimizer policy for slow-transient SRH bank training. `v90` — legacy fixed-AdamW (protected baseline). `optimized` — faster convergence schedule. **Use `optimized`** only if slow-bank training time is a bottleneck and the v90 policy is known to be stable on this device. |

### Training Constants NOT Adjustable via INI

The following are hardcoded in `fit_group()` and cannot be changed by Agent D: `learning_rate=2.5e-3`, `weight_decay=1e-4`, `gradient_clip=5.0`, `eta_min=5e-5`, `early_stop_patience=16`, `early_stop_min_epochs=min(400, max(150, epochs//3))`. Agent D can only change: `epochs`, `hidden sizes`, `amp_parameterization`, `restarts`, and `optimizer_policy` choice. If these levers are exhausted, consider `fallback_to_previous` or `accept_quality`.

### Acceptance Gate Thresholds NOT Adjustable

The following 7 training script acceptance thresholds are hardcoded CLI arguments (never passed by `run_model.py`) and CANNOT be adjusted by Agent D via `adjust_params`:

| CLI argument | Default | Controls |
|---|---|---|
| `max-top2-rmse-mean` | 0.020 | Top-2 KAN vs measurement RMSE |
| `max-student-teacher-rmse-mean` | 0.006 | KAN vs teacher mean RMSE |
| `max-student-teacher-rmse-p90` | 0.012 | KAN vs teacher p90 RMSE |
| `max-endpoint-drift-error-mean` | 0.070 | Endpoint drift mean |
| `max-endpoint-drift-error-p95` | 0.230 | Endpoint drift p95 |

To pass a failed acceptance gate, Agent D must improve the model via training parameters — it cannot relax the gate thresholds themselves.

### minimal_run Override

When `[runtime] minimal_run = yes`, `run_model.py` hard-overrides:
- `epochs`: 1
- `hidden_capture`: 4
- `hidden_emission`: 4

Training completes but results are not meaningful — only for pipeline infrastructure smoke testing.

### Knowledge Injection Status

**This knowledge file is currently NOT injected into Agent D's prompt.** The `build_agent_prompt()` function exists in `knowledge/loader.py` but is never called by the orchestrator. Agent D receives a generic 3-line inline prompt with no stage-specific knowledge. Until the loader is integrated, the tuning guidance and parameter recommendations in this file are not accessible to the agent at runtime.

## 4. Quality Metrics

### step_log_rmse_median

- **What it measures**: The median log10 RMSE (decades) of the two-trap double-exponential
  fits to individual I-t bias-step transients, computed during the pre-analysis phase
  (`analyze_it_hysteresis.py`). Each bias step (a transition between two gate voltage
  levels) is fit independently with the equation `I(t) = I_∞ + A_fast·(e^{-t/τ_fast} − 1)
  + A_slow·(e^{-t/τ_slow} − 1)`. The `log_rmse` is the RMS error in log10(current)
  between the fit and the smoothed measurement. The median across all successful
  step fits (`ok == True`) is reported. A value of 0.08 means the typical two-trap
  fit is within ~0.08 decades of the measured transient — about a 20% relative
  error in absolute current.
- **Source file**: `04_hysteresis/it_hysteresis_analysis_summary.json` → key `step_log_rmse_median`
- **Threshold from agent_configs.json**: `DG_TFT_QUALITY_HYSTERESIS_RMS_MAX` = **0.08** (decades)
- **What "below threshold" means**: The pre-analysis two-trap fits are accurate enough
  to serve as reliable teacher labels for the SRH-KAN training. Values well below 0.03
  indicate clean, well-behaved I-t data. Values approaching 0.08 suggest noisy transients,
  but the KAN student can still learn useful dynamics even with imperfect teacher fits.
  Values above 0.15 indicate extremely noisy or non-exponential transients that may require
  data re-preparation or revisiting the I-t measurement setup.
- **Which parameters typically affect it**:
  - **Primary**: This metric is from the **pre-analysis**, not from the KAN training itself.
    A high `step_log_rmse_median` indicates poor I-t data quality, not poor model training.
    The KAN parameters (epochs, hidden, amp_parameterization) do not affect this metric.
  - **Diagnostic**: Check I-t data files in the hysteresis directory. Issues include: too
    few time points per step (<16), very noisy current traces, incorrect voltage labeling,
    or temperature-dependent transients where the two-trap model is insufficient. The
    pre-analysis report (`it_measurement_audit.json`) lists per-bias issues.
- **Additional metrics produced** (from `it_hysteresis_analysis_summary.json`, diagnostic only):
  - `n_points`: total I-t data points
  - `n_curves`: number of unique I-t curves
  - `n_segments`: number of bias-step segments
  - `n_step_fits`: number of step-response fits attempted
  - `tau_kan.*`: KAN fit metrics for time-constant label prediction (teacher label quality)
  - `audit_issue_count`, `audit_issues`: measurement quality audit
  - `input_files_expected`, `input_files_used`: file availability check
  - `voltage_corrected_segments`: segments with static voltage alignment correction

### Additional SRH-KAN Training Metrics (not gated, diagnostic)

From `srh_top2_regime_bank_summary.json`:
- `teacher_rmse_dec_mean`: mean RMSE between two-trap teacher fits and measured data
- `top2_rmse_dec_mean`: mean RMSE between Top-2 symbolic SRH-KAN predictions and measured data
- `student_teacher_rmse_dec_mean`: mean distillation RMSE (KAN vs teacher) — how well the student learned from the teacher
- `slow_teacher_rmse_dec_mean`, `slow_top2_rmse_dec_mean`, `slow_student_teacher_rmse_dec_mean`: slow-regime breakdowns
- `acceptance.pass`: boolean acceptance gate (engine-internal; the pipeline gate uses `step_log_rmse_median` only)

## 5. Engine Behavior

### Execution Flow

The `train-hysteresis` action in `run_model.py` runs two subprocesses sequentially:

**Silent Data Filtering Before Training**: The pre-analysis → training pipeline applies strict filters that silently exclude segments. If ALL data is excluded, two distinct RuntimeErrors are raised: "No identifiable SRH tau labels after filtering" or "No waveform segments available for end-to-end refinement." In this case, adjusting training parameters does nothing — `fallback_to_previous` is the correct action.

1. **Pre-analysis** (`analyze_it_hysteresis.py`):
   ```
   <CUDA python> dynamic/analyze_it_hysteresis.py
   ```
   Reads I-t workbooks from the hysteresis data directory. This script:
   - Loads I-t measurement files (expected: `I-t with different VBG*.xlsx`,
     `I-t with different VTG*.xlsx`, plus temperature variants; legacy fallback:
     `I-t.xlsx`)
   - Applies static voltage correction (alignment with the static IV model)
   - Segments each curve into bias-step intervals
   - Smooths log10(ID) using Hampel outlier filtering + Savitzky-Golay smoothing
   - Fits each bias step with the two-trap double-exponential model via
     `scipy.optimize.least_squares` with `loss="soft_l1"` and duration-aware
     tau bounds
   - Writes `it_hysteresis_analysis_summary.json` (contains `step_log_rmse_median`)
   - Also writes: `it_step_response_fit.csv`, `it_hysteresis_processed_and_fit.xlsx`,
     `it_tau_kan_label_fit.csv`, `it_tau_kan_metrics.json`, `it_measurement_audit.json`

2. **SRH-KAN Training** (`train_srh_top2_regime_banks.py`):
   ```
   <CUDA python> dynamic/train_srh_top2_regime_banks.py
     --regime <[hysteresis] regime>
     --epochs <[hysteresis] epochs>
     --seed <[runtime] seed>
     --fast-restarts <[hysteresis] fast_restarts>
     --slow-restarts <[hysteresis] slow_restarts>
     --hidden-capture <[hysteresis] hidden_capture>
     --hidden-emission <[hysteresis] hidden_emission>
     --amp-parameterization <[hysteresis] amp_parameterization>
     --fast-amp-parameterization <[hysteresis] fast_amp_parameterization>
     --slow-amp-parameterization <[hysteresis] slow_amp_parameterization>
     --fast-optimizer-policy <[hysteresis] fast_optimizer_policy>
     --slow-optimizer-policy <[hysteresis] slow_optimizer_policy>
   ```
   This script:
   - Builds a waveform label table from the pre-analysis teacher fits
   - Groups segments by (regime, family, process) — at most 8 groups:
     2 regimes × 2 families (VBG, VTG) × 2 processes (capture, emission)
   - For each group, trains a TwoLayerTop1KAN bank predicting
     `tau_fast`, `tau_slow`, and amplitude(s)
   - Training: AdamW optimizer, `lr=2.5e-3`, `weight_decay=1e-4`, with
     CosineAnnealingLR scheduler (when `optimizer_policy="optimized"`)
   - Loss: soft predictions (continuous KAN) → waveform comparison against
     teacher + anchor loss (endpoint/amplitude) + shape quality loss;
     hard predictions (Top-1 symbolic pass) → RMSE + shape penalties
   - Early stopping: patience=16, min_epochs = min(400, max(150, epochs//3))
   - Best checkpoint selected by combined deployment score (waveform RMSE + shape +
     distill quality + teacher fidelity)
   - Acceptance gate checks: teacher RMSE, student-teacher distillation RMSE,
     endpoint drift error, Top-2 symbolic topology verification
   - Writes `srh_top2_regime_banks/srh_top2_regime_bank_checkpoint.pt`, `srh_top2_regime_bank_summary.json`,
     metrics CSV, predictions CSV

3. **Verilog-A Export** (`export_srh_top2_regime_banked_va.py`):
   - Reads the trained bank summary
   - Exports each (regime, family, process) group as a standalone Verilog-A module
   - Generates regime wrapper modules (`dg_tft_dynamic_srh_fast`, `dg_tft_dynamic_srh`)
     and a combined trap module (`dg_tft_dynamic_srh`)
   - Key artifacts: `dg_tft_dynamic_srh.va` and `dg_tft_dynamic_srh.inc`

4. **HSPICE Validation** (optional, only if HSPICE is available and
   `[deployment] run_hspice = yes`):
   - Python/HSPICE equivalence for each regime (fast, slow)
   - Complete static + fast/slow SRH transient smoke test

### How Parameters Change Execution

- **`regime`**: When set to `all` (default), all groups are trained (max 8).
  When set to `fast` or `slow`, only groups matching that regime are trained.
  This reduces training time and focuses capacity, but the missing regime will
  not be available for deployment.

- **`epochs`**: Controls the per-group training budget. With 8 groups at default
  1800 epochs, total weight updates ≈ 8 × 1800 × (segments_per_group) —
  significantly less than the static model's 2100 epochs. The early stopping
  (patience=16, min_epochs=400) typically terminates well before 1800 epochs.
  **Interaction**: `epochs` is the budget per group; `fast_restarts` and
  `slow_restarts` multiply this budget for the corresponding regime groups.

- **`hidden_capture` / `hidden_emission`**: Control KAN capacity separately for
  capture vs emission networks. Emission processes have wider time-constant
  ranges (microseconds to seconds), so the default `hidden_emission=20` is
  larger than `hidden_capture=16`. **Interaction**: larger hidden sizes with
  small group sizes (<50 segments) risk overfitting.

- **`amp_parameterization` (global)**: Sets the default amplitude parameterization
  when the regime-specific override is not set. The `conservative` form
  (`A_fast = total/2 + exchange`, `A_slow = total/2 − exchange`) enforces that
  the two amplitudes are anti-correlated — if one trap captures, the other
  emits. This is physically motivated (charge conservation) and prevents
  the KAN from predicting both traps with large same-sign amplitudes (which
  would violate charge neutrality).

- **`fast_restarts` / `slow_restarts`**: Each restart uses a different random
  seed (`seed + group_index × 1009 + restart × 9176`). The best result (lowest
  deployment score) is kept. Restarts help escape poor local minima but
  multiply training time. Fast groups (µs timescale, fewer segments) benefit
  less from restarts than slow groups.

- **`fast_amp_parameterization` / `slow_amp_parameterization`**: Regime-specific
  overrides. Fast transients default to `conservative` (safer for small amplitude
  changes). Slow transients default to `independent` (v90 legacy — preserves
  compatibility with the validated slow-bank reference).

- **`fast_optimizer_policy` / `slow_optimizer_policy`**: `optimized` uses
  CosineAnnealingLR (eta_min=5e-5). `v90` uses no scheduler — fixed LR throughout.
  The v90 policy is the slow-bank baseline because slow groups are less sensitive
  to learning rate scheduling.

## 6. Failure Patterns

> This section will be populated from Zikang's Codex JSONL logs during the
> knowledge extraction phase.

### Error Classification (from diagnostics.py)

When train-hysteresis exits with non-zero code, keyword classification is applied
against the combined error message + stderr text:

| Keyword in stderr | Classification | Confidence |
|-------------------|---------------|------------|
| `Cuda out of memory`, `out of memory`, `OOM`, `cuDNN` | TRAINING_NUMERICAL_ERROR | 0.9 |
| `NaN`, `inf`, `overflow encountered`, `underflow encountered`, `overflow` | TRAINING_NUMERICAL_ERROR | 0.9 |
| `gradient`, `backward` | TRAINING_NUMERICAL_ERROR | 0.9 |
| `diverged`, `not converging`, `loss is nan`, `loss too large`, `gradient explosion` | TRAINING_NUMERICAL_ERROR | 0.7 |
| `FileNotFoundError`, `No such file or directory`, `does not exist` | DATA_ERROR | 0.9 |
| `missing column`, `missing sheet`, `invalid sheet`, `sheet not found`, `column not found` | DATA_ERROR | 0.9 |
| `unexpected header`, `empty workbook`, `no valid data`, `xlsx file is empty` | DATA_ERROR | 0.9 |
| `data_dir not found`, `cannot find it directory` | DATA_ERROR | 0.9 |
| `excel file format cannot be determined` | DATA_ERROR | 0.9 |
| `KeyError`, `IndexError`, `ValueError` | DATA_ERROR | 0.6 |
| `configparser`, `no section`, `no option`, `invalid config`, `missing config` | CONFIG_ERROR | 0.9 |
| `unknown stage`, `unknown action` | CONFIG_ERROR | 0.9 |
| `hspice.exe`, `hspice not found`, `hspice executable`, `hspice path` | HSPICE_ENV_ERROR | 0.9 |
| `hspice simulation failed`, `hspice timeout`, `hspice license` | HSPICE_ENV_ERROR | 0.9 |
| No keyword match, exit code ≠ 0 | TRAINING_NUMERICAL_ERROR | 0.4 |

The stage suggestion: "Verify I-t data availability, hysteresis configuration,
and that dynamic engine scripts are present."

**Acceptance Gate Hard Failure**: The SRH-KAN training script's internal acceptance gate (line 702-706 in `train_srh_top2_regime_banks.py`) raises `RuntimeError('Top-2 SRH hysteresis acceptance failed: {details}')` when checks fail. This is a HARD failure (exit code != 0) that triggers Agent D, but it is separate from the pipeline quality gate (`step_log_rmse_median`). The error message contains the full acceptance dict. Agent D should inspect `srh_top2_regime_bank_summary.json` → `acceptance.checks` to identify which sub-check(s) failed — the remedy is training parameter tuning, NOT data inspection.

### Agent D Decision Framework

Agent D has four tools and must call EXACTLY ONE:

| Tool | When to Use | Consequences |
|------|-------------|--------------|
| `adjust_params` | Parameter tuning can fix the problem. Use when: (1) quality is close to threshold (e.g., step_log_rmse_median = 0.10 vs threshold 0.08), (2) specific regime groups show poor training results while others are good, (3) the KAN acceptance gate fails on Top-2 topology or student_teacher_rmse. Incremental changes preferred: +25% epochs, not doubling. | INI is updated, stage re-runs with new parameters. Costs a retry slot. |
| `fallback_to_previous` | The upstream outputs are the problem. Use when: (1) the static model checkpoint (`v93_safe_residual_state.pt`) is missing or corrupted, (2) I-t data files are not recognized by the pre-analysis (wrong filenames, missing sheets), (3) the static model produces unrealistic steady-state currents that make transients uninterpretable, (4) multiple parameter adjustments have yielded no improvement. | Stage re-runs from train-static, which regenerates the static model. This is costly — train-static itself is the most expensive stage. |
| `accept_quality` | Current quality is acceptable despite being above threshold. Use when: (1) step_log_rmse_median is only marginally above 0.08 (e.g., 0.09-0.12) and the I-t fits look qualitatively good in individual trace plots, (2) the KAN student model achieves good distillation quality despite imperfect teacher labels, (3) user preferences explicitly deprioritize hysteresis accuracy, (4) all adjustment attempts have been exhausted. | Pipeline moves to final-report. Quality warning is recorded but does not block progress. |
| `inspect_xlsx` | Investigation only — use freely to examine file structure before deciding. Does not consume a decision. Returns `{"status": "success"\|"fail", "sheet_count": N, "sheets": {"Name": {"columns": [...]}}}`. |

**Specific tuning guidance for adjust_params**:

- **step_log_rmse_median too high** (>0.08): This metric comes from the pre-analysis
  (two-trap least_squares fits), not the KAN training. Parameter tuning **cannot** fix
  poor pre-analysis quality. Instead, inspect the I-t data: check
  `it_measurement_audit.json` for issues, verify the hysteresis data directory
  contains the expected files, and review `it_step_response_fit.csv` for per-segment
  RMSE values. If many segments have `ok == False` or `log_rmse > 0.15`, the I-t data
  itself may need re-measurement or re-preparation. In this case, `fallback_to_previous`
  to train-static is the appropriate action — the static model may need regeneration
  with corrected voltage alignment.

- **SRH-KAN acceptance gate failure**: Check `srh_top2_regime_bank_summary.json`
  → `acceptance` block. Common failures:
  - `top2_rmse_mean`: KAN predictions don't match measured transients. Increase
    `epochs` to 2500-3000, or increase `hidden_capture`/`hidden_emission` by 4 each.
  - `student_teacher_rmse_mean` / `student_teacher_rmse_p90`: KAN student didn't
    learn from the teacher fits. Increase `epochs` first, then increase
    `slow_restarts` (if slow-bank led to failure).
   - `endpoint_drift_error_mean` / `endpoint_drift_error_p95`: The KAN predicts
      wrong steady-state levels. Switch `amp_parameterization` from `independent`
      to `conservative` — the bounded total+exchange form enforces charge conservation
      which stabilizes endpoint predictions. **Caveat**: only apply this when
      endpoint drift is the dominant quality issue. Do NOT switch to conservative
      to close small RMSE gaps (< 0.001 dex) — conservative degrades slow-bank
      accuracy and will widen the gap.
  - `strict_top2_symbolic_topology`: Some KAN edges failed to produce Top-2
    symbolic replacements. This is rare but may require increasing `epochs`.

- **CUDA OOM**: Hysteresis training memory is dominated by the per-group batch
  (all segments concatenated). With default hidden sizes (16/20), memory usage
  is modest. If OOM occurs, reduce `hidden_capture` and `hidden_emission` to 8/12
  or reduce the regime to `fast` or `slow` to train fewer groups.

- **HSPICE validation failure**: If the KAN training succeeds but HSPICE
  equivalence fails, this is likely an HSPICE environment issue (license,
  executable path). The Verilog-A files are still valid — the HSPICE check is
  a deployment validation, not a training gate. Use `accept_quality` to proceed.
  **Important**: HSPICE failures do NOT cause the stage to fail — the subprocess
  handles missing HSPICE by printing a warning and continuing (`run_model.py`).
  Agent D should never call `fallback_to_previous` for HSPICE-only failures.

### Diagnostic Tools Available to Agent D

Agent D can use `read_file` and `glob_file` (standard SubAgent tools) to inspect:

- `04_hysteresis/it_hysteresis_analysis_summary.json` — **most important file**. Contains:
  - `step_log_rmse_median` — the quality gate metric
  - `tau_kan` — time-constant label KAN metrics (teacher label quality)
  - `audit_issues` — per-measurement issues (noise, missing data, voltage alignment)
  - `input_files_expected` / `input_files_used` — file availability
  - `voltage_corrected_segments` — count of segments with static alignment correction
- `04_hysteresis/it_measurement_audit.json` — detailed measurement quality audit:
  repeat-bias spread, static voltage alignment status, per-file issues
- `04_hysteresis/it_step_response_fit.csv` — per-segment two-trap fit details:
  `curve_id`, `seg_id`, `family`, `ok`, `log_rmse`, `amp_fast`, `amp_slow`,
  `tau_fast_s`, `tau_slow_s`, `delta_sweep_V`, `direction`, bias voltages
- `04_hysteresis/srh_top2_regime_banks/srh_top2_regime_bank_summary.json` — SRH training results:
  `groups` (per-group training history), `formulas` (symbolic edge formulas),
  `acceptance` (per-gate pass/fail), `teacher_rmse_dec_mean`, `top2_rmse_dec_mean`,
  `student_teacher_rmse_dec_mean`
- `04_hysteresis/srh_top2_regime_banks/srh_top2_regime_bank_metrics.csv` — per-segment distillation metrics
- `04_hysteresis/it_hysteresis_processed_and_fit.xlsx` — full processed I-t data (multisheet workbook)
- `model_config.ini` — current hysteresis parameter values
- `03_full_model/v93_safe_residual_state.pt` — upstream static model checkpoint
- Error logs in the output directory or stderr capture
