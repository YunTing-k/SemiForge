# Hysteresis Experience (from 2026-06-29 + 05-22 sessions)

> **Agent D tools**: `adjust_params` (12 INI params: epochs, hidden_capture, hidden_emission, amp_parameterization, fast/slow_restarts, fast/slow_amp_parameterization, fast/slow_optimizer_policy), `fallback_to_previous` (→ train-static), `accept_quality`, `inspect_xlsx`.  
> **Can influence**: training budget, KAN capacity, optimizer policy, amplitude parameterization, restart count.  
> **Cannot change**: acceptance gate thresholds, loss function, pre-analysis logic, Verilog-A export code, metadata serialization format.  
> **Principle**: focus on what Agent D can adjust. Engine-hardcoded issues are marked with correct tool choice (usually `fallback_to_previous` or `accept_quality`).

## Principle Insights

### Slow-Bank Amplitude Parameterization Metadata — PARTIALLY MITIGATED
**Status**: The combined summary still stores `amp_parameterization` as a nested dict (`compose_v95_hysteresis_banks.py`). However, the VA export script (`export_srh_top2_regime_banked_va.py`) now INFERS the amplitude policy from formula output names rather than reading the metadata field — avoiding the silent `conservative` default bug. **The recommended pre-export assertion (`assert isinstance(bank['amp_parameterization'], str)`) has NOT been added.** Any downstream consumer that reads `amp_parameterization` from the combined summary and expects a scalar string will still encounter the nested dict. The advice to add the pre-export assertion remains valid.

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
**Evidence from 05-22 session**: Turn 45-46 explicit observation: "CV 的纯物理内核的曲线多么漂亮，曲线没有什么突变，抖动，加了pkan还好，加了residual就给人一种为了降低而肆意扭曲的感觉。" The session solved this by progressively adding: ① CV monotonicity constraints (first-derivative ≥ 0), ② IV transconductance (gm) fidelity loss, ③ high-plateau monotonic loss, ④ extrapolation anchor constraints to prevent boundary artifacts, ⑤ base-aware constraints (residual must not worsen shape vs physical baseline).  
**Generalization**: For the hysteresis SRH-KAN, this means: the student-teacher RMSE metric (`student_teacher_rmse_dec_mean`) is a necessary but NOT sufficient quality indicator. A low RMSE with high `endpoint_drift_error` or with visibly oscillatory I(t) predictions indicates that the KAN is "cheating" — using spline flexibility to fit noise rather than learning the physical tau/amplitude relationship.  
**Actionable Advice**: Agent D must not rely solely on `student_teacher_rmse_dec_mean` for quality decisions. When evaluating whether to accept quality or adjust parameters, also check: (a) `endpoint_drift_error_mean`/`p95` — if these are near threshold while RMSE is excellent, the KAN may be oscillating; (b) the per-segment metrics in `srh_top2_regime_bank_metrics.csv` — look for segments where the KAN prediction has higher local curvature than the teacher fit; (c) increase `slow_restarts` rather than `epochs` if shape quality is the concern, since multi-restart selects for deployment-quality rather than pure RMSE. The existing `deployment_score` (waveform RMSE + shape + distill quality + teacher fidelity) already partially addresses this — Agent D should trust it over raw RMSE.

### Edge-Level Symbolic MSE Does Not Predict Terminal-Level Waveform Quality After Replacement
**Category**: Principle Insight  
**Stage(s)**: train-hysteresis (symbolic export phase), train-static (symbolic export phase)  
**Symptom**: The per-edge Top-1 (or Top-2 for hysteresis) symbolic MSE values all pass their thresholds (e.g., <0.015 per edge), but after symbolic replacement, the terminal-level predicted I(t) waveform diverges significantly from the numeric KAN prediction — especially at bias extremes or regime boundaries. The per-edge metric appears healthy, but the cumulative symbolic approximation error creates a qualitatively different waveform.  
**Root Cause**: Each KAN edge operates on intermediate features that are nonlinear combinations of bias voltages. The edge-level MSE measures the difference between the raw spline edge and its symbolic surrogate evaluated at the same intermediate feature value. However, these feature values are produced by the preceding layer, which is ALSO being symbolically replaced. The errors compound through the 2-layer KAN: a small edge-MSE in layer 0 changes the feature, which shifts the input to layer 1, which then evaluates a different symbolic edge at a different point than the one it was fitted at. The terminal-level waveform quality (measured in log10(ID) or log10(τ)) reflects this compounding, not individual edge quality.  
**Evidence from 05-22 session**: Turn 46-48: "top_k=1 符号化后退化明显，最终 CV/IV 指标反而不如上一版。这不是 CV 物理核错，而是 KAN spline 边还不够'像单个常见函数'。" The solution was: ① After edge-wise symbolic replacement, re-evaluate on full measured data (terminal-level gate), ② Apply L-BFGS + Adam fine-tuning on the symbolic model with the same loss function used for training, ③ Reject symbolic replacements that increase terminal error by more than a threshold vs the numeric model.  
**Evidence from 06-29 session static_model_experience.md**: Edge-level Top-1 MSE gate passed (all edges <0.015), but terminal current RMS degraded from 0.082 dex to 0.122 dex — a 49% increase.  
**Generalization**: For hysteresis, the existing `strict_top2_symbolic_topology` acceptance gate only verifies that edges CAN be replaced, not that the replacement preserves waveform quality. The SRH-KAN's deployment checkpoint should include a terminal-level gate: compare numeric-KAN and symbolic-KAN predictions on held-out I-t step transients.  
**Actionable Advice**: When Agent D sees `strict_top2_symbolic_topology` pass but deployment waveforms look wrong (or HSPICE equivalence fails), the problem is likely cumulative symbolic approximation error. Increase `epochs` to train edge shapes that are more "single-function-like" (higher similarity to symbolic primitives during training), then redo symbolic replacement. Do NOT increase `hidden_capture`/`hidden_emission` — larger networks have MORE edges, making cumulative error worse, not better. The existing `symbolic_training_weight` and similarity loss in the static model are the right pattern; the hysteresis engine should apply the same approach.

### Physical Model Insufficiency Causes KAN to Produce Non-Physical Corrections, Not Training Failure
**Category**: Principle Insight  
**Stage(s)**: train-hysteresis (pre-analysis → SRH-KAN boundary)  
**Symptom**: KAN training appears to converge (loss decreases, RMSE improves) but the predicted trap parameters (τ, A) exhibit physically impossible patterns: negative capture amplitudes for emission-labeled segments, tau values outside the physically meaningful range, or amplitude predictions that violate charge conservation across bias steps. Tuning training hyperparameters (epochs, hidden sizes, restarts) does not resolve these artifacts.  
**Root Cause from 05-22 session analogy**: When the KAN tries to correct for a physical model that is structurally insufficient, it "creates non-physical bumps that happen to reduce the average error" (Turn 82). In the IV/CV case: the missing CV off-state parasitic capacitance caused KAN to generate spikes at the threshold boundary. In the hysteresis case: if the two-trap teacher model is structurally wrong for certain segments (e.g., a single-trap process is actually happening, or the noise floor masks the second trap), the SRH-KAN student will learn to reproduce a biased teacher in bizarre ways rather than produce smooth, physics-conforming predictions.  
**Evidence from 05-22 session**: Turn 36-40 analysis: "CV 的主要问题不是全部数据都坏，而是 vds 这组和当前 gate-CV charge kernel 不是同一种测量对象，强行一起训会拖累通用模型。" The solution was systematic data audit and selective downweighting — not parameter tuning. Turn 64: "现在 KAN 更像'被强约束的小修小补'，而不是在充分发挥神经网络拟合残差的能力。关键不是 KAN 不够强，而是我们把它放在了一个很难发挥的位置：它既要保持物理、又要拟合 IV/CV、还被很多正则和共享参数压住了。"  
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
- **Additional Evidence from 05-22 session**: The session confirmed multi-restart's value through parallel experimentation with IV/CV KAN training. Extended runs with up to 1200 epochs showed that increasing restarts (2→5) consistently closed small quality gaps, while increasing epoch count beyond effective convergence produced no improvement. The session also demonstrated that per-restart seeding must be deterministic and reproducible — the `seed + group_index × 1009 + restart × 9176` formula used in both codebases satisfies this.

### Amplitude Policy Validation Must Occur Before VA Export, Not After HSPICE Failure
**Category**: Tuning Pattern  
**Stage(s)**: train-hysteresis (assembly → VA export transition)  
**Context**: The combined hysteresis summary (`srh_top2_regime_bank_summary.json`) is written by the training script, then read by the export script. The export script reads amplitude parameterization metadata to generate the Verilog-A module body. If the metadata is malformed, the generated VA uses the wrong amplitude equations but outputs no error — the failure only surfaces when HSPICE detects Python→VA mismatch. This wastes a full HSPICE run (~several minutes).  
**Pattern**:
1. After training completes, read `srh_top2_regime_bank_summary.json` and inspect the per-group `amp_parameterization` field.
2. Assert each group's value is a flat string: `"independent"` or `"conservative"`. If any value is a dict, nested object, or missing: refuse export.
3. Cross-check: the per-regime slow groups should all have identical `amp_parameterization` in the combined summary. If one slow group is `"independent"` and another is `"conservative"`, there's a metadata merge bug.
4. Only after these assertions pass, proceed to VA export and HSPICE.
**Key Parameters**: N/A — this is a metadata validation pattern, not a parameter tuning pattern.  
**Success Indicator**: All per-group `amp_parameterization` fields are flat strings. Export proceeds without defensive defaults.  
**Pitfalls**: Relying on the VA exporter's default fallback (`conservative`) when metadata is ambiguous. The fallback silences the error and produces syntactically valid but semantically incorrect VA — HSPICE will compile but produce wrong transient currents.
- **Additional Evidence from 05-22 session**: Turns 8-10 demonstrated that metadata serialization bugs are insidious: the training script writes valid data, but the assembly/export path re-interprets it differently. The session's `Cgg_common` parasitic extraction faced a similar issue — OFF-window parameters extracted correctly but the frozen/unfrozen status was silently overwritten by later training phases. The pre-export validation principle is identical: validate that what was written is what will be read.

### Pre-HSPICE Checklist for Dynamic Model Assembly
**Category**: Tuning Pattern  
**Stage(s)**: train-hysteresis (pre-assembly audit)  
**Context**: After both static (IV/CV) and hysteresis training complete, the assembly step combines static, fast-SRH, and slow-SRH modules into a complete device. Multiple independent failures (interface mismatch, missing blend module, metadata encoding) can each cause assembly failure, and they compound — fixing one exposes the next.  
**Pattern**:
1. **Module inventory**: List all Verilog-A `.va` and `.inc` files in the static model directory (typically 5 files for top-1 static) and hysteresis directory. Compare against a known-good manifest (v90 reference). Any missing file (especially `amp_blend`, `srh_combiner`, `port_wrapper`) indicates a portability gap.
2. **Interface audit**: Check the number of SRH state ports in the fast-bank, slow-bank, and combined module definitions. They must match between Python wrapper and VA netlist template. If the template is parametric ({4-state / 8-state}), verify the detection logic.
3. **Metadata flatness**: Validate all string-enum fields in the combined summary JSON are flat scalars, not nested dicts. Key fields: `amp_parameterization`, `optimizer_policy`, `regime` for each trained group.
4. **HSPICE reference model**: The Python reference model for HSPICE equivalence must use the **deployed symbolic KAN** (post-calibration), NOT the numerical spline KAN. Using the spline KAN as reference inflates the reported HSPICE error by mixing "numerical→symbolic" error into "Python→HSPICE" error. This was confirmed in the session: after fixing the reference, IV HSPICE max relative error dropped from 2.42% to 0.564%, max log error from 0.00951 dex to 0.00208 dex.
5. **I-t data audit before training** (from 05-22 session evidence): Before launching the SRH-KAN, inspect `it_measurement_audit.json` for: (a) per-file issues (noise, missing segments, voltage alignment), (b) repeat-bias spread (same bias condition across multiple curves should produce similar transients), (c) segments where the two-trap fit RMS is >0.15 dex (these are unreliable teacher labels). If >30% of segments have `ok == False`, the training data quality, not the training parameters, is the bottleneck.
**Key Parameters**: N/A — verification pattern.  
**Success Indicator**: All 5 checklist items pass before HSPICE is run. HSPICE IV max relative error < 3%, CV < 5%, transient log-RMSE < 1×10⁻⁶ dex.  
**Pitfalls**: The most insidious pitfall is item 4 — using the numerical KAN as the HSPICE reference model. This inflates deployment error by 4-5× and causes Agent D to attempt unnecessary parameter tuning or top-k relaxation when the Verilog-A implementation is actually correct.

### I-t Data Quality Audit Precedes Parameter Tuning for High step_log_rmse_median
**Category**: Tuning Pattern  
**Stage(s)**: train-hysteresis (pre-analysis → training boundary)  
**Context**: When `step_log_rmse_median > 0.08` (threshold), Agent D's default diagnostic path is to inspect `it_hysteresis_analysis_summary.json` and the audit report. However, even with moderate RMSE, certain I-t measurement pathologies are invisible to aggregate metrics.  
**Pattern from 05-22 session** (CV data audit methodology, generalized to I-t):
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
**Context**: This pattern emerges from the 05-22 session's extensive experimentation with residual KAN for IV/CV, where a secondary KAN (trained on top of a frozen parameter-KAN) consistently reduced MSE but introduced non-physical oscillations. The pattern is directly relevant to hysteresis because the SRH-KAN's student-teacher training is structurally analogous to residual learning.  
**Pattern**:
1. Observe that `student_teacher_rmse_dec_mean` decreases with more epochs or larger hidden sizes — this is expected and positive.
2. Observe that `endpoint_drift_error_mean` or `endpoint_drift_error_p95` remains unchanged or increases — this is the warning sign. The KAN is improving pointwise fit at the expense of waveform correctness.
3. Observe that per-segment metrics (`srh_top2_regime_bank_metrics.csv`) show some segments with KAN predictions that are numerically closer to the teacher but visually "wiggly" — check the `shape_quality` score if available.
4. The correct response: decrease `hidden_capture` or `hidden_emission` by 4 to reduce overfitting capacity, then retry. Alternatively, increase `slow_restarts` to select for better deployment-quality. Do NOT increase epochs.
**Key Parameters**: `hidden_capture`, `hidden_emission` (decrease to reduce overfitting capacity).  
**Success Indicator**: `endpoint_drift_error_mean` does not increase when `student_teacher_rmse_dec_mean` decreases. Both metrics trend downward together.  
**Pitfalls**: The most dangerous outcome is when RMSE improves but endpoint drift worsens — this means the KAN has found a "cheating" solution that fits transient shapes better at individual points while shifting steady-state predictions. Agent D must treat RMSE alone as insufficient for quality decisions — always cross-check with endpoint drift and, if available, per-segment shape metrics.

## Session Meta-Notes (05-22)

The 05-22 session is a 70+ turn IV/CV compact model construction session — NOT directly about hysteresis training. However, it contains extensive KAN methodology that generalizes. Key methodological takeaways for Agent D:

1. **Physics-informed loss functions are essential**: Pure MSE on log-current produces KAN solutions that are numerically accurate but physically impossible (negative slopes, plateau oscillations, boundary artifacts). The solution was progressively adding: first-derivative monotonicity, second-derivative smoothness, high-plateau flatness, extrapolation anchor, and base-aware constraints (residual must not worsen shape vs physical baseline).

2. **Data quality audit before training is non-negotiable**: The session developed a systematic CV data audit (curve-level monotonicity, jump detection, measurement configuration compatibility) and showed that training on data with incompatible measurement modes (vds as transcapacitance vs gate-CV) degrades the model for ALL curves. For hysteresis, this maps to auditing I-t segments for: monotonicity, amplitude sign consistency, duration consistency, and repeat-bias spread.

3. **Multi-phase training outperforms single-phase**: Phase1 (physics baseline) → Phase2 (parameter KAN with strong shape constraints) → Phase3 (residual KAN with bounded amplitude) → Phase4 (symbolic replacement + fine-tuning) outperformed any single-phase approach.

4. **Edge-wise symbolic replacement needs terminal-level validation**: Per-edge Top-1 MSE is insufficient — cumulative error through 2-layer KAN can produce large terminal-level errors despite all edges passing individually. For hysteresis's Top-2 symbolic edges, this means the acceptance gate should validate at the I(t) waveform level after symbolic replacement.

5. **Multi-restart selection criteria should prefer shape quality over pure MSE**: The session's "safe scale" selection algorithm — preferring scales that satisfy shape constraints even when slightly higher MSE — is directly applicable to hysteresis's `deployment_score` ranking across restarts.

6. **Separate KANs for separate modalities**: The session concluded that a single KAN for both IV and CV created coupling artifacts, and splitting into IV-KAN and CV-KAN improved both. The hysteresis engine already separates capture/emission networks and fast/slow banks — this validates that separation.
