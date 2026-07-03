# Hysteresis Experience

> **Agent D tools**: `adjust_params` (12 INI params: epochs, hidden_capture, hidden_emission, amp_parameterization, fast/slow_restarts, fast/slow_amp_parameterization, fast/slow_optimizer_policy), `fallback_to_previous` (→ train-static), `accept_quality`, `inspect_xlsx`.  
> **Can influence**: training budget, KAN capacity, optimizer policy, amplitude parameterization, restart count.  
> **Cannot change**: acceptance gate thresholds, loss function, pre-analysis logic, Verilog-A export code, metadata serialization format.  
> **Principle**: focus on what Agent D can adjust. Engine-hardcoded issues are marked with correct tool choice (usually `fallback_to_previous` or `accept_quality`).

## Principle Insights

### Slow-Bank Amplitude Parameterization Metadata — PARTIALLY MITIGATED
**Status**: The combined summary still stores `amp_parameterization` as a nested dict. However, the VA export script now infers the amplitude policy from formula output names rather than reading the metadata field, avoiding the silent `conservative` default bug. The recommended pre-export assertion has not been added.

**Agent Impact**: **None.** Agent D cannot fix the serialization code or add assertions.

**Correct Action**: If HSPICE dynamic transparency check fails but the export script should have inferred the correct policy → `fallback_to_previous` (→ train-static). Metadata corruption is structural, not fixable by parameter tuning.

### Single Random Initialization Can Account for ~0.0002 dex Gap in Non-Convex SRH-KAN Training
**Category**: Principle Insight  
**Stage(s)**: train-hysteresis (slow-bank training)  
**Symptom**: Fresh slow-bank training produces student→teacher RMSE of 0.00350 dex, while the protected v90 baseline is 0.00333 dex — a gap of ~0.00017 dex (~5%). After switching to v90 optimizer policy (fixed-AdamW, no CosineAnnealingLR), the gap shrinks to ~0.00004 dex (0.0033729 vs 0.0033324, ~1.2%). No amount of epoch budget or hidden-size tuning closes the remaining gap.  
**Root Cause**: The SRH-KAN loss landscape for slow-regime groups (second-scale transients) is non-convex. A single AdamW initialization converges to a local minimum that is ~0.00004 dex from the globally best point reachable with this architecture. The gap is **not** caused by loss function design, training order, or hyperparameter differences — it is inherent stochasticity in the initial weights.  
**Evidence**: Single-restart training with v90 optimizer policy: 0.0033729 dex. Deterministic multi-restart (5 seeds, best validation score): 0.0033399 dex. Gap to v90: 7.53×10⁻⁶ dex (~0.23%) — essentially identical. The `slow_restarts=2` default plus the existing `seed + group_index × 1009 + restart × 9176` seeding formula already implements the mechanism; the session confirmed it eliminates the gap.  
**Generalization**: When Agent D observes a small but persistent quality gap between fresh training and a protected baseline for the slow regime (student_teacher_rmse gap < 0.001 dex), the gap is likely stochastic initialization rather than a parameterization problem. Increasing `slow_restarts` from 2 to 4-5 is the correct action, NOT switching amplitude parameterization or optimizer policy.  
**Actionable Advice**: Agent D should compare the gap magnitude: if `Δ(student_teacher_rmse_dec_mean) < 0.001` dex between fresh and baseline, increase `slow_restarts` by 2 (up to max 5). If `Δ > 0.003` dex, the gap is structural — investigate `slow_amp_parameterization` or `hidden_emission` instead. Do NOT waste retries on `epochs` or `slow_optimizer_policy` for sub-0.001 dex gaps.

### SRH State-Port Interface Versioning — FIXED (auto-detection exists)
**Status**: FIXED. `validate_complete_device_hspice.py` (`state_access_contract()`) reads the deployed `.inc` file at runtime and detects 4-state vs 8-state by checking which subckt names are present. The port list is injected via `{state_ports}` f-string into the SPICE deck. **This entry is retained for historical reference.**

### Missing amp_blend Submodule — FIXED (module exists)
**Status**: FIXED. The `dg_tft_amp_blend` module exists in `dg_tft_portable_glue.va` and is included in all HSPICE complete-device validation decks (`validate_complete_device_hspice.py`). **This entry is retained for historical reference.**

### KAN Training on Pointwise Log-RMSE Alone Produces Non-Physical Oscillations in Predicted Waveforms
**Category**: Principle Insight  
**Stage(s)**: train-hysteresis (SRH-KAN student training), general KAN methodology  
**Symptom**: KAN predictions achieve lower numerical RMSE than the physical baseline, but the predicted curves exhibit non-physical features: negative slopes in saturation, plateau dropoffs, and high-frequency oscillations that look correct in log-space but violate physical continuity. Increasing epoch budget makes RMSE better but makes the oscillations worse.  
**Root Cause**: The pointwise log-RMSE loss function has no notion of curve continuity, monotonicity, or physical smoothness. A KAN with enough capacity can reduce RMSE by overfitting to local noise while creating large second-derivative artifacts that are invisible to the loss function. The KAN "uses its extra degrees of freedom to create non-physical bumps that happen to reduce the average error."  
Pure physical kernel CV curves were smooth and monotonic; adding a parameter-KAN caused minimal distortion, but adding a residual KAN introduced oscillations — the KAN was distorting the waveform to minimize numerical error. The session solved this by progressively adding: ① CV monotonicity constraints (first-derivative ≥ 0), ② IV transconductance (gm) fidelity loss, ③ high-plateau monotonic loss, ④ extrapolation anchor constraints to prevent boundary artifacts, ⑤ base-aware constraints (residual must not worsen shape vs physical baseline).  
**Generalization**: For the hysteresis SRH-KAN, this means: the student-teacher RMSE metric (`student_teacher_rmse_dec_mean`) is a necessary but NOT sufficient quality indicator. A low RMSE with high `endpoint_drift_error` or with visibly oscillatory I(t) predictions indicates that the KAN is "cheating" — using spline flexibility to fit noise rather than learning the physical tau/amplitude relationship.  
**Actionable Advice**: Agent D must not rely solely on `student_teacher_rmse_dec_mean` for quality decisions. When evaluating whether to accept quality or adjust parameters, also check: (a) `endpoint_drift_error_mean`/`p95` — if these are near threshold while RMSE is excellent, the KAN may be oscillating; (b) the per-segment metrics in `srh_top2_regime_bank_metrics.csv` — look for segments where the KAN prediction has higher local curvature than the teacher fit; (c) increase `slow_restarts` rather than `epochs` if shape quality is the concern, since multi-restart selects for deployment-quality rather than pure RMSE. The existing `deployment_score` (waveform RMSE + shape + distill quality + teacher fidelity) already partially addresses this — Agent D should trust it over raw RMSE.

### Edge-Level Symbolic MSE Does Not Predict Terminal Waveform Quality After Replacement
**Category**: Principle Insight  
**Stage(s)**: train-hysteresis (symbolic export phase)  
**Symptom**: Per-edge Top-2 symbolic MSE values pass thresholds but terminal-level I(t) waveform diverges significantly from numeric KAN predictions, especially at bias extremes or regime boundaries.

**Root Cause**: Cumulative approximation error through the 2-layer KAN — edge-level MSE in layer 0 changes feature values, which shifts inputs to layer 1's symbolic edges, producing compounding errors invisible to per-edge metrics.

**Agent Impact**: **None.** Agent D cannot add terminal-level quality gates to the symbolic export engine.

**Correct Action**: If symbolic export passes edge-level gates but HSPICE equivalence fails → `accept_quality` (the numeric model is accurate; symbolic deployment error may be acceptable). Do NOT increase `epochs` for this — more training of numeric KAN does not reduce cumulative symbolic approximation error. Do NOT increase `hidden_capture`/`hidden_emission` — larger networks have MORE edges, making cumulative error worse.

### Physical Model Insufficiency Causes KAN to Produce Non-Physical Corrections, Not Training Failure
**Category**: Principle Insight  
**Stage(s)**: train-hysteresis (pre-analysis → SRH-KAN boundary)  
**Symptom**: KAN training appears to converge (loss decreases, RMSE improves) but the predicted trap parameters (τ, A) exhibit physically impossible patterns: negative capture amplitudes for emission-labeled segments, tau values outside the physically meaningful range, or amplitude predictions that violate charge conservation across bias steps. Tuning training hyperparameters (epochs, hidden sizes, restarts) does not resolve these artifacts.  
**Root Cause**: When the KAN tries to correct for a physical model that is structurally insufficient, it creates non-physical bumps that happen to reduce the average error. In the IV/CV case: the missing CV off-state parasitic capacitance caused KAN to generate spikes at the threshold boundary. In the hysteresis case: if the two-trap teacher model is structurally wrong for certain segments (e.g., a single-trap process is actually happening, or the noise floor masks the second trap), the SRH-KAN student will learn to reproduce a biased teacher in bizarre ways rather than produce smooth, physics-conforming predictions.  
The main CV issue was data heterogeneity — the vds group and the gate-CV charge kernel were different measurement objects, and training them together degraded the general model. The solution was systematic data audit and selective downweighting — not parameter tuning. The KAN was acting more like a heavily constrained patch rather than a neural network fully expressing its residual fitting capability: it had to remain physical while fitting IV/CV, heavily regularized and compressed by shared parameters.  
**Generalization**: For hysteresis, this maps directly to the pre-analysis quality: if `step_log_rmse_median` is high (>0.08), the two-trap teacher fits are unreliable for some segments. The `it_step_response_fit.csv` should be examined for: (a) segments where `ok == False`, (b) segments where `log_rmse > 0.15`, (c) segments where `amp_fast` and `amp_slow` have same sign and similar magnitude (indicating the two-trap model collapsed to single-trap). Segments with these characteristics should be excluded or downweighted before SRH-KAN training.  
**Actionable Advice**: Agent D should distinguish between two failure modes: (1) `step_log_rmse_median` near threshold (0.08-0.10) — the teacher fits are borderline; increasing `epochs` or `slow_restarts` may help the KAN generalize from imperfect labels; (2) `step_log_rmse_median` far above threshold (>0.12) — the teacher labels are structurally unreliable; parameter tuning will NOT help, and `fallback_to_previous` to re-generate the static model with corrected voltage alignment is the correct action. If many individual segments have `ok == False` or `log_rmse > 0.15` in `it_step_response_fit.csv`, the I-t data itself may need re-preparation — adjusting hysteresis training parameters cannot fix bad teacher labels.

### KAN Correction Form — Engine Architecture (Beyond Agent D's Control)
Agent D cannot change the KAN correction architecture (additive vs multiplicative). In brief: different physical parameters need different correction forms — additive for thresholds/time-constants, multiplicative for mobility/gain, bounded sigmoid for dimensionless ratios. This is a future engine architecture consideration.

## Tuning Patterns

### Deterministic Multi-Restart Overcomes Non-Convex Slow-Bank Initialization
**Category**: Tuning Pattern  
**Stage(s)**: train-hysteresis (slow-bank training)  
**Context**: After fresh slow-bank training completes with `slow_optimizer_policy = "v90"` and `slow_amp_parameterization = "independent"`, the aggregated `slow_student_teacher_rmse_dec_mean` is reported as 0.0033729 dex — slightly above the v90 protected baseline of 0.0033324 dex (gap 4.05×10⁻⁵ dex). This gap is consistent across multiple retries with identical hyperparameters but different random seeds. Agent D must decide whether to accept or continue tuning.  
**Pattern**:
1. First attempt: fresh training with `slow_optimizer_policy = "v90"`, `slow_amp_parameterization = "independent"`, `epochs = 1800`, `hidden_emission = 20`. Result: 0.00350 dex (gap 1.7×10⁻⁴ vs v90).
2. Second attempt: switch `slow_optimizer_policy` from `"optimized"` (CosineAnnealingLR) to `"v90"` (fixed-AdamW). Result: 0.0033729 dex (gap 4.05×10⁻⁵ vs v90). The CosineAnnealingLR was confirmed inferior for slow-bank — its LR decay undershoots the flat region of the non-convex loss landscape.
3. Third attempt: increase `slow_restarts` from default 2 to 5, keeping v90 policy and independent parameterization. Result: 0.0033399 dex (gap 7.53×10⁻⁶ vs v90) — essentially identical.
4. Stop here. The residual gap of 7.53×10⁻⁶ dex (0.23% of the baseline) is indistinguishable from rounding noise. Accept quality.
**Key Parameters**: `slow_restarts` (increase from 2 to 5), `slow_optimizer_policy` (use `"v90"`, not `"optimized"`).  
**Success Indicator**: `srh_top2_regime_bank_summary.json` → `slow_student_teacher_rmse_dec_mean` within 1×10⁻⁴ dex of v90 baseline. Full-device transient identity log-RMSE below 1×10⁻⁶ dex.  
**Pitfalls**:
- **Do NOT** switch `slow_amp_parameterization` from `"independent"` to `"conservative"` to close a sub-0.001 dex gap. The `conservative` form changes the algebraic structure (total+exchange clamping) and actually degrades slow-bank accuracy — the gap will widen, not close.
- **Do NOT** increase `epochs` above 2500 for a sub-0.001 dex gap. The early stopping (patience=16, min_epochs=400) already terminates long before 1800 epochs; more budget only wastes time without improvement.
- **Do NOT** confuse the `slow_restarts` seeding with duplicate computation. Each restart's seed uses `seed + group_index × 1009 + restart × 9176`, which is deterministic and spaced apart — not a simple increment.
- Multi-restart value was confirmed through parallel experimentation with IV/CV KAN training. Extended runs with up to 1200 epochs showed that increasing restarts (2→5) consistently closed small quality gaps, while increasing epoch count beyond effective convergence produced no improvement. Per-restart seeding must be deterministic and reproducible — the `seed + group_index × 1009 + restart × 9176` formula used in both codebases satisfies this.

### Amplitude Policy Validation Must Occur Before VA Export, Not After HSPICE Failure
**Category**: Tuning Pattern  
**Stage(s)**: train-hysteresis (assembly → VA export transition)  
**Symptom**: HSPICE detects Python→VA mismatch only after a full HSPICE run, wasting several minutes. The root cause is malformed metadata in `srh_top2_regime_bank_summary.json`.

**Agent Impact**: **Partial.** Agent D can read the summary JSON to validate metadata before export, but cannot fix the serialization code.

**Correct Action**: After training, read `srh_top2_regime_bank_summary.json` and inspect per-group `amp_parameterization`. If values are nested dicts → `fallback_to_previous` (→ train-static). Metadata encoding errors are structural, not fixable by parameter tuning. Do NOT proceed to HSPICE with malformed metadata.
- Metadata serialization bugs are insidious: the training script writes valid data, but the assembly/export path re-interprets it differently. The `Cgg_common` parasitic extraction faced a similar issue — OFF-window parameters extracted correctly but the frozen/unfrozen status was silently overwritten by later training phases. The pre-export validation principle is identical: validate that what was written is what will be read.

### Pre-HSPICE Checklist for Dynamic Model Assembly
**Category**: Tuning Pattern  
**Stage(s)**: train-hysteresis (pre-assembly audit)  
**Checks before assembly**:
1. Module inventory: verify `.va`/`.inc` files exist for static model and hysteresis (typically 5 files for top-1 static + per-regime bank outputs). Missing files → `fallback_to_previous`.
2. Interface audit: verify SRH state port count matches between Python wrapper and VA netlist template. Mismatch → `fallback_to_previous`.
3. Metadata flatness: verify `amp_parameterization` fields are flat strings in combined summary JSON. Nested dict → `fallback_to_previous`.
4. HSPICE reference model: Python reference for equivalence must use deployed symbolic KAN, not numerical spline KAN. If uncertain → `accept_quality` (numeric model accuracy is validated independently).

**Agent Impact**: **Partial.** Agent D can read summary JSON and output files to verify, but cannot fix assembly code.

**Success Indicator**: All 4 checks pass before HSPICE runs. HSPICE IV max error <3%, CV <5%, transient log-RMSE <1e-6 dex.

**Pitfalls**: The most insidious is item 4 — using numerical KAN as HSPICE reference inflates deployment error 4-5x, causing unnecessary parameter tuning or top-k relaxation when the Verilog-A is correct.

### I-t Data Quality Audit Precedes Parameter Tuning for High step_log_rmse_median
**Category**: Tuning Pattern  
**Stage(s)**: train-hysteresis (pre-analysis → training boundary)  
**Context**: When `step_log_rmse_median > 0.08` (threshold), Agent D's default diagnostic path is to inspect `it_hysteresis_analysis_summary.json` and the audit report. However, even with moderate RMSE, certain I-t measurement pathologies are invisible to aggregate metrics.  
**Pattern** (CV data audit methodology, generalized to I-t):
1. **Curve-level monotonicity check**: For each I-t transient segment, verify that the smoothed current is monotonic in the expected direction (emission: monotonically decreasing; capture: monotonically increasing). Non-monotonic points indicate measurement noise or incomplete settling.
2. **Duration consistency**: Check that all segments within the same regime (fast: <50 ms, slow: ≥50 ms) have consistent time bases. Segments with too few time points (<16) cannot produce reliable teacher fits.
3. **Amplitude sign consistency**: Within each process group (capture/emission), verify that the fitted trap amplitudes have consistent signs. A capture segment with negative amplitude or an emission segment with positive amplitude indicates a mislabeled or mis-segmented transient.
4. **Repeat-bias check**: If the same bias condition appears in multiple I-t curves (e.g., different measurement campaigns), the predicted tau values should be within 2× of each other. Larger discrepancies indicate measurement drift or inconsistent voltage alignment.
5. **Action threshold**: If >20% of segments fail any of these checks, data quality — not model architecture — is the bottleneck. The correct action is `fallback_to_previous` to re-prepare data, not increase `epochs` or `slow_restarts`.
**Key Parameters**: N/A — data audit pattern.  
**Success Indicator**: <20% of segments fail per-curve checks. `it_measurement_audit.json` → `audit_issue_count` < 5.  
**Pitfalls**: Confusing data quality issues with model capacity issues. A KAN with more hidden units or epochs WILL memorize noise patterns — producing lower training RMSE but worse generalization. If the per-curve audit shows issues, no amount of parameter tuning will produce a physically meaningful model.

### Caution with Residual-KAN: Better MSE Can Mean Worse Physical Behavior
**Category**: Tuning Pattern  
**Stage(s)**: train-hysteresis (SRH-KAN student training)  
**Context**: This pattern emerges from extensive experimentation with residual KAN for IV/CV, where a secondary KAN (trained on top of a frozen parameter-KAN) consistently reduced MSE but introduced non-physical oscillations. The pattern is directly relevant to hysteresis because the SRH-KAN's student-teacher training is structurally analogous to residual learning.  
**Pattern**:
1. Observe that `student_teacher_rmse_dec_mean` decreases with more epochs or larger hidden sizes — this is expected and positive.
2. Observe that `endpoint_drift_error_mean` or `endpoint_drift_error_p95` remains unchanged or increases — this is the warning sign. The KAN is improving pointwise fit at the expense of waveform correctness.
3. Observe that per-segment metrics (`srh_top2_regime_bank_metrics.csv`) show some segments with KAN predictions that are numerically closer to the teacher but visually "wiggly" — check the `shape_quality` score if available.
4. The correct response: decrease `hidden_capture` or `hidden_emission` by 4 to reduce overfitting capacity, then retry. Alternatively, increase `slow_restarts` to select for better deployment-quality. Do NOT increase epochs.
**Key Parameters**: `hidden_capture`, `hidden_emission` (decrease to reduce overfitting capacity).  
**Success Indicator**: `endpoint_drift_error_mean` does not increase when `student_teacher_rmse_dec_mean` decreases. Both metrics trend downward together.  
**Pitfalls**: The most dangerous outcome is when RMSE improves but endpoint drift worsens — this means the KAN has found a "cheating" solution that fits transient shapes better at individual points while shifting steady-state predictions. Agent D must treat RMSE alone as insufficient for quality decisions — always cross-check with endpoint drift and, if available, per-segment shape metrics.

## Session Meta-Notes

The IV/CV compact model construction session developed KAN methodology through 70+ iterations not directly about hysteresis training, but with generalizable findings. Key lessons for hysteresis: physics-informed loss functions are essential (pure MSE on log-current produces numerically accurate but physically impossible solutions), data quality audit before training is non-negotiable, and multi-restart should select for deployment-quality rather than pure RMSE.
