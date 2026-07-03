# Hysteresis Experience (from 2026-06-29 session)

## Principle Insights

### Slow-Bank Amplitude Parameterization Metadata Causes Silent Verilog-A Export Corruption
**Category**: Principle Insight  
**Stage(s)**: train-hysteresis (Verilog-A export phase)  
**Symptom**: Slow-regime HSPICE dynamic transparency check fails (Python vs HSPICE mismatch), while fast-regime HSPICE passes cleanly (2.04×10⁻⁶ dex). No training errors or acceptance gate failures — the failure is only in deployment verification.  
**Root Cause**: The combination/assembly step stored the slow-bank amplitude parameterization as a **nested dictionary** (`{"slow": "independent"}` or similar structured object) rather than as a flat **scalar string** (`"independent"`). The Verilog-A exporter interpreted the nested dict as a missing/unknown policy and defaulted to `conservative`, causing Python and VA to compute trap amplitudes with different algebraic forms. The Python side correctly used `independent` (per-trap direct amplitude prediction), while the VA side used `conservative` (bounded total+exchange form: `A_fast = total/2 + exchange`, `A_slow = total/2 - exchange`).  
**Evidence**: After fixing metadata to scalar `"independent"`, slow-regime Python→HSPICE equivalence reached 0.001266 dex max error. Combined slow metric: 0.0033399 dex, within 7.53×10⁻⁶ dex of v90 baseline (0.0033324 dex). Full device 10,000-point transient current identity check: log-RMSE 8.90×10⁻⁷ dex, max relative error 4.97×10⁻⁶.  
**Generalization**: Any regime-specific configuration that must be identical at both training time and VA export time risks silent corruption if the serialization path uses different data structures. Specifically, nested dicts vs scalars for string enums are a common JSON/YAML serialization pitfall.  
**Actionable Advice**: Before HSPICE verification, validate that `srh_top2_regime_bank_summary.json` stores per-regime `amp_parameterization` as a flat string (one of `"independent"` or `"conservative"`), never as a dict or nested object. Add a pre-export assertion: `assert isinstance(bank['amp_parameterization'], str), "amplitude policy must be flat string"`. If the summary JSON shows `"slow"` or `"fast"` keys inside what should be a string field, refuse export and surface error to Agent D.

### Single Random Initialization Can Account for ~0.0002 dex Gap in Non-Convex SRH-KAN Training
**Category**: Principle Insight  
**Stage(s)**: train-hysteresis (slow-bank training)  
**Symptom**: Fresh slow-bank training produces student→teacher RMSE of 0.00350 dex, while the protected v90 baseline is 0.00333 dex — a gap of ~0.00017 dex (~5%). After switching to v90 optimizer policy (fixed-AdamW, no CosineAnnealingLR), the gap shrinks to ~0.00004 dex (0.0033729 vs 0.0033324, ~1.2%). No amount of epoch budget or hidden-size tuning closes the remaining gap.  
**Root Cause**: The SRH-KAN loss landscape for slow-regime groups (second-scale transients) is non-convex. A single AdamW initialization converges to a local minimum that is ~0.00004 dex from the globally best point reachable with this architecture. The gap is **not** caused by loss function design, training order, or hyperparameter differences — it is inherent stochasticity in the initial weights.  
**Evidence**: Single-restart training with v90 optimizer policy: 0.0033729 dex. Deterministic multi-restart (5 seeds, best validation score): 0.0033399 dex. Gap to v90: 7.53×10⁻⁶ dex (~0.23%) — essentially identical. The `slow_restarts=2` default plus the existing `seed + group_index × 1009 + restart × 9176` seeding formula already implements the mechanism; the session confirmed it eliminates the gap.  
**Generalization**: When Agent D observes a small but persistent quality gap between fresh training and a protected baseline for the slow regime (student_teacher_rmse gap < 0.001 dex), the gap is likely stochastic initialization rather than a parameterization problem. Increasing `slow_restarts` from 2 to 4-5 is the correct action, NOT switching amplitude parameterization or optimizer policy.  
**Actionable Advice**: Agent D should compare the gap magnitude: if `Δ(student_teacher_rmse_dec_mean) < 0.001` dex between fresh and baseline, increase `slow_restarts` by 2 (up to max 5). If `Δ > 0.003` dex, the gap is structural — investigate `slow_amp_parameterization` or `hidden_emission` instead. Do NOT waste retries on `epochs` or `slow_optimizer_policy` for sub-0.001 dex gaps.

### SRH State-Port Interface Versioning: 4-State Cannot Be Substituted for 8-State
**Category**: Principle Insight  
**Stage(s)**: train-hysteresis (HSPICE transient verification)  
**Symptom**: Complete device HSPICE transient simulation fails during interface validation even though the per-regime (fast, slow) HSPICE checks pass individually. Error is at the module instantiation level, not the physics computation level.  
**Root Cause**: The v90 validated deployment uses a 4-state SRH port interface (4 explicit SRH state variable ports), while the newer pipeline hardcodes an 8-state interface. The two interfaces are incompatible at the Verilog-A module boundary — the module names, port lists, and internal state wiring differ. Attempting to connect a 4-state wrapper to an 8-state validator fails at the HSPICE netlist level.  
**Evidence**: Individual fast/slow regime HSPICE checks passed (fast: 2.04×10⁻⁶ dex), but full-device transient failed until a compatibility wrapper with automatic interface recognition was added. After adding auto-detection logic, the complete device passed: 10,000 transient-point current identity log-RMSE 8.90×10⁻⁷ dex.  
**Generalization**: Hardware description language (Verilog-A) module interfaces can change between pipeline versions. When the pipeline is deployed to new datasets, the SRH module interface must be auto-detected or the wrapper must support both 4-state (legacy) and 8-state (current) interfaces. A hardcoded port count assumption will break when combining checkpoint components from different schema versions.  
**Actionable Advice**: The `dg_tft_dynamic_srh` Verilog-A module and its Python wrapper must include explicit interface version detection based on the number of state ports in the trained bank checkpoint. The HSPICE template should be parametric: `{if 4_state: srh_bank_fast_state0_net ... srh_bank_fast_state3_net; else: ... srh_bank_fast_state7_net}`. Agent D should not attempt to fix this via parameter tuning — it is an infrastructure issue requiring code-level interface detection.

### Missing amp_blend Submodule in Portability Glue Breaks Full-Device Assembly
**Category**: Principle Insight  
**Stage(s)**: train-hysteresis (assembly phase)  
**Symptom**: Static (IV/CV/temperature) HSPICE and per-regime hysteresis HSPICE all pass individually, but the combined static+dynamic device module fails to compile or produces incorrect transient current. No obvious error in the hysteresis training logs.  
**Root Cause**: During code portability refactoring (making v90 scripts reusable across datasets), the `amp_blend` submodule was lost from the portable glue layer. This submodule handles the blending of trap amplitudes between fast and slow regimes in the combined `dg_tft_dynamic_srh` module. Without it, amplitudes from one regime can overwrite or fail to combine with the other regime.  
**Evidence**: The session explicitly states: "当前便携 glue 漏掉了 v90 的 amp_blend 模块，已按 v90 原式恢复" — recovered by restoring v90's original formula verbatim.  
**Generalization**: When refactoring multi-module Verilog-A assemblies for portability, the dependency graph is larger than it appears. Submodules may exist only for cross-regime combination logic (amplitude blending, time-constant merging) and are easy to miss during a "keep only the training scripts" cleanup.  
**Actionable Advice**: Before delivering a portable pipeline version, run a full `diff` of the Verilog-A include/module file list between the new portable version and the v90 reference version. Specifically audit for modules whose names contain `blend`, `merge`, `combine`, `wrapper`, or `glue` — these are cross-regime combination modules, not per-regime training modules. A missing `amp_blend` cannot be fixed by Agent D; it requires a code-level restoration of the original module.

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

### Pre-HSPICE Checklist for Dynamic Model Assembly
**Category**: Tuning Pattern  
**Stage(s)**: train-hysteresis (pre-assembly audit)  
**Context**: After both static (IV/CV) and hysteresis training complete, the assembly step combines static, fast-SRH, and slow-SRH modules into a complete device. Multiple independent failures (interface mismatch, missing blend module, metadata encoding) can each cause assembly failure, and they compound — fixing one exposes the next.  
**Pattern**:
1. **Module inventory**: List all Verilog-A `.va` and `.inc` files in the static model directory (typically 5 files for top-1 static) and hysteresis directory. Compare against a known-good manifest (v90 reference). Any missing file (especially `amp_blend`, `srh_combiner`, `port_wrapper`) indicates a portability gap.
2. **Interface audit**: Check the number of SRH state ports in the fast-bank, slow-bank, and combined module definitions. They must match between Python wrapper and VA netlist template. If the template is parametric ({4-state / 8-state}), verify the detection logic.
3. **Metadata flatness**: Validate all string-enum fields in the combined summary JSON are flat scalars, not nested dicts. Key fields: `amp_parameterization`, `optimizer_policy`, `regime` for each trained group.
4. **HSPICE reference model**: The Python reference model for HSPICE equivalence must use the **deployed symbolic KAN** (post-calibration), NOT the numerical spline KAN. Using the spline KAN as reference inflates the reported HSPICE error by mixing "numerical→symbolic" error into "Python→HSPICE" error. This was confirmed in the session: after fixing the reference, IV HSPICE max relative error dropped from 2.42% to 0.564%, max log error from 0.00951 dex to 0.00208 dex.
**Key Parameters**: N/A — verification pattern.  
**Success Indicator**: All 4 checklist items pass before HSPICE is run. HSPICE IV max relative error < 3%, CV < 5%, transient log-RMSE < 1×10⁻⁶ dex.  
**Pitfalls**: The most insidious pitfall is item 4 — using the numerical KAN as the HSPICE reference model. This inflates deployment error by 4-5× and causes Agent D to attempt unnecessary parameter tuning or top-k relaxation when the Verilog-A implementation is actually correct.
