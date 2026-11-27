# Static Model Experience

> **Agent C tools**: `adjust_params` (71 INI params in [full_model]/[base_pkan]/[temperature]/[symbolic]/[validation]), `fallback_to_previous` (→ fit-physical), `accept_quality`, `inspect_xlsx`.  
> **Can influence**: epochs, LR, batch sizes, cv_weight, cv_residual_level, enable_cv_residual, symbolic budgets, freeze flags, warm-start LR multipliers.  
> **Cannot change**: engine training order, optimizer parameter groups, loss function design, safety scan logic, symbolic export architecture.  
> **Principle**: the majority of the actionable advice below focuses on what Agent C can influence via `adjust_params` + correct tool choice. Entries where Agent C's only option is `fallback_to_previous` or `accept_quality` are marked accordingly.

## Principle Insights

### Training Order: Prevent IV Degradation During CV Training

**Diagnostic**: `residual_iv_rms` increases after CV phase vs base IV phase. CV checkpoint selector (weighted by cv_weight=10) picks epochs where CV improved but IV drifted. CV residual may produce negative net contribution (RMS degrades from 0.02073 → 0.020995).

**Agent Impact**: **Partial.** Agent C cannot change the engine training order but CAN control which parameters train in each phase.

**Correct Actions** (what Agent C can do via `adjust_params`):
1. Set `finetune_base_pkan=False` (v96 default) to freeze base PKAN during CV training
2. Request `--freeze-iv-residual-after-stage` to protect IV residual from CV gradients
3. Reduce `cv_weight` to 5.0 during CV residual stage (from default 10.0) to reduce gradient competition
4. If CV residual degrades results: set `enable_cv_residual=False` — CV residual is optional, not mandatory
5. Do NOT set `iv_residual_stage_linear_weight` to 0.34 (default 0.25 is correct — knowledge covers tuning)
6. If IV degradation persists after all above → `fallback_to_previous` (→ fit-physical) — the physical kernel may need regeneration

**Evidence**: Using the current default IV residual parameters (700 epochs, LR=0.002, linear=0.25, deriv=36, shape=3) produced 11.3% held-out IV improvement. Freezing IV before CV training prevented regression. Independent convergence on same staged isolation principle confirmed cross-session.

### Native vs Warm-Start Learning Rate Multipliers
**Category**: Principle Insight
**Stage(s)**: train-static (base IV-PKAN and joint CV training phases)
**Context**: The engine automatically selects between two learning-rate multipliers based on whether the model is trained from scratch or resumed from a checkpoint (see `dg_tft_full_model.py:9898-9901`). The warm-start multiplier is a *protection* mechanism (not an optimization parameter): it exists to prevent CV loss gradients from destroying pre-trained base PKAN weights in the first ~15% of joint training epochs. It has no useful role in fresh-start training.
**Actionable Advice**:
   1. **When `reset_base_from_physical_fit=True`**: `warm_start_base_pkan_lr_mult` is ignored (engine uses `base_pkan_lr_mult`=1.0, see `dg_tft_full_model.py:9898-9901`). Use `base_iv_learning_rate` at default (0.002). During the later joint CV phase, since `finetune_base_pkan=False` (v96 default), the base PKAN will be frozen anyway.
  2. **When `reset_base_from_physical_fit=False`** (resume from checkpoint): Set `warm_start_base_pkan_lr_mult=0.035` and `finetune_base_pkan=True`. This protects the pre-trained PKAN from early destruction when CV loss is introduced. The multiplier ramps from 0.035 to `base_pkan_lr_mult` over the warm-up period.
    3. **Detection**: When `reset_base_from_physical_fit` is True, the base PKAN LR multiplier (`base_pkan_lr_mult`, default 1.0) is used instead of the warm-start multiplier — the engine correctly switches between the two based on this flag.
    4. **What NOT to do**: Do not set `base_iv_learning_rate` above 0.005 — even for fresh-start, rates >0.005 cause training oscillations.

---

### Symbolic Distillation: Locate the Failing Representation

**Diagnostic**: The numerical model passes current-curve criteria; the exported symbolic model fails terminal fidelity or measured-region criteria.

**Agent Impact**: Agent C controls symbolic fitting budgets, IV Top-2 candidate selection, terminal calibration complexity, and early stopping.

**Correct Actions**:
1. Read `network_equivalence.iv_terminal_observable` and `iv_measured_full_curve`. Identify the failing metric, its limit, and its bias region.
2. Compare teacher-to-measurement, student-to-teacher, and student-to-measurement errors. Repair the numerical model when the teacher fails its task criteria; repair export when a qualified teacher loses accuracy during symbolic conversion.
3. Use `v92_edgewise_symbolic_residual_info.json` and the edgewise formula files to inspect the deployed representation. `v92_top1_symbolic_residual_surrogates.*` contains separate low-dimensional diagnostic proxies. A `basis=tanh(VDS_V)` entry describes that proxy.
4. Use latent edge errors to locate approximation difficulty; use terminal and measured-region criteria to evaluate the proposed repair. Internal parameter corrections interact through the device current equation.
5. Preserve the numerical checkpoint when adjusting export parameters. Verify the failing region and all remaining task criteria after execution. Evaluate HSPICE equivalence from the HSPICE result.

**Evidence**: The engine exports legacy residual proxies and edgewise symbolic networks through separate functions. Its `network_equivalence` report separates terminal observables, measured curves, and latent mapping diagnostics.

### Adaptive Top-2: Candidate Threshold and Realized Topology

**Diagnostic**: Strict Top-1 export loses terminal accuracy and individual edge fits show substantial residual MSE.

**Correct Actions**:
1. Set `iv_adaptive_top2_mse` to a positive value to enable candidate fitting. Use the observed edge MSE scale to choose the threshold. Lower positive thresholds admit more candidates; zero selects strict Top-1.
2. Keep `edge_topk=1` and `iv_edge_topk=1`, matching the current engine CLI. The adaptive path fits a second term for eligible IV edges.
3. A Top-2 candidate is adopted when its fitted MSE is below `0.90 * Top-1 MSE`. Read `iv_top2_edge_count` after execution to establish the realized topology.
4. Compare terminal fidelity, measured-region errors, and expression complexity after promotion. Allocate additional edge/network fitting steps when the new representation requires further optimization.

**Evidence**: `symbolize_compact_kan` first tests `edge_loss > adaptive_top2_mse`, then applies the 10% improvement criterion to the fitted candidate.

### Terminal Calibration: Effective Capacity and Model Selection

**Diagnostic**: The symbolic student retains regional current error after edge and network refinement.

**Correct Actions**:
1. Read `iv_symbolic_deployment_calibration.complexity_selection`: candidate scores, `selected_terms`, and the selection rule.
2. The current basis contains 24 terms. The effective `iv_calibration_max_terms` range is 2–24. Increase an active lower cap when larger existing candidates improve held-out scores.
3. The engine selects the smallest candidate whose score is within 3% of the best. When the cap already covers the basis, use edge fitting or terminal optimization controls to improve accuracy.
4. For repeated outcomes near an acceptance limit, compare candidate-score margins and selected term counts. A candidate crossing the 3% boundary can change the selected model abruptly.

**Evidence**: The calibration constructs two sigmoid terms for each of 12 terminal-derived features, then ranks and selects candidate subsets using held-out scores.

### Symbolic Optimization: Budget, Early Stopping, and Checkpoints

**Diagnostic**: Export accuracy remains insufficient after a completed optimization stage.

**Correct Actions**:
1. If validation improves up to the maximum step count and the selected checkpoint is near that limit, increase the budget for that stage.
2. If small improvements continue when patience expires, inspect `min_delta`, `patience`, and `eval_period`. Patience counts validation checks; its nominal observation window is `patience * eval_period` steps.
3. If the selected checkpoint is at the start and later validation scores worsen, inspect the starting symbolic representation and the teacher-versus-measurement objective balance. Preserve the selected checkpoint while repairing that cause.
4. If promising minima fall between widely spaced checks, reduce `eval_period`. Set `min_steps` from the observed optimization trajectory.
5. `device_iv_finetune_steps` fits teacher currents and slopes; `device_iv_measured_steps` fits measured curves with fidelity-aware selection. Match each adjustment to the failing comparison.
6. Quantify borderline outcomes with repeated execution at fixed data, checkpoint, parameters, and seed. Record regional error distributions, pass counts, selected steps, and calibration term counts.

**Evidence**: The two symbolic IV stages retain their selected checkpoints independently. Validation schedules, early stopping, and discrete calibration selection jointly determine the final exported model.

---



## Tuning Patterns

### Residual Safety Scan Rejecting Valid Deployments

**Diagnostic**: `safe_iv_residual_scale=0` despite held-out IV residual showing real improvement (e.g., balanced score 0.09918→0.08796). Base model has extrapolation warnings (ext_bad > 0), and the safety scan requires `ext_bad == 0` (absolute zero) — which no residual can achieve if the base already has warnings.

**Agent Impact**: **None.** Agent C cannot change the safety scan logic (engine code). The main IV-CV joint scan has been partially fixed (now uses `ext_bad <= base_ext_bad`), but CV sub-scans still use absolute-zero.

**Correct Actions**:
1. Check `safe_iv_residual_scale` in metrics JSON. If 0 but residual training logs show improvement → suspect this bug.
2. If residual provides genuine held-out gains → `accept_quality` with justification (document the base model's extrapolation count).
3. Try increasing `iv_residual_stage_epochs` (700→1000) to give the residual more time to reduce extrapolation alerts — small chance of passing the scan.
4. Do NOT waste retries on repeated `adjust_params` — the scan logic is the bottleneck, not parameter values.
5. If residual is essential for deployment quality → **`fallback_to_previous`** (→ fit-physical). A better physical kernel baseline may have fewer shape warnings, allowing the residual to pass.

**Evidence**: Base had 4 extrapolation alerts, all non-zero residuals rejected. After fix to relative criterion: residual accepted at scale 0.5.

---



### CV Passivity Scan: When safe_cv_residual_scale = 0
**Category**: Tuning Pattern
**Stage(s)**: train-static (Phase 5 — Joint CV safe scale search)
**Code Reality (verified 2026-07-03)**: The actual argparse defaults are:
- `joint_cv_passivity_scan_points`: **80**
- `safe_passivity_scan_points`: **160**  
- `passivity_max_points`: **1200**
Note: these argparse defaults are subject to runtime capping (`min(..., 32)`, `min(..., 48)`, `min(..., 400)` respectively via quick-mode limit logic in `_apply_quick_mode_limit`), so effective values may be lower than the declared defaults.
`choose_safe_joint_cv_scales` runs a single exhaustive 2D grid search over `(parameter_scale × mode × residual_scale)` using `joint_cv_passivity_scan_points=80`.
**Context**: Agent C observes that `safe_cv_residual_scale=0` despite residual CV training appearing to converge. The charge conservation / passivity check on the full grid found violations that prevent any non-zero residual scale from being selected.
**Agent Impact**: **None.** Agent C cannot change the scan logic (engine code). The corrective action is adjusting INI parameters below to reduce passivity violation risk, then `accept_quality` if the CV model is adequate without residual.
**Key Parameters**:
  - `joint_cv_safe_scale_search`: Set to `True` only if CV passivity is likely marginal. In most cases, `False` (parameter-then-residual sequential scan) is more robust.
  - `safe_scale_steps`: Default 8.
  - `cv_residual_level`: If the scan consistently finds that ANY residual scale violates passivity, reduce this to 1 (disable inter-gate coupling branch) or 0 (disable all CV residual branches). Evidence shows `cv_residual_level=2` activates the inter-gate reciprocal coupling branch which has higher passivity violation risk.
**Success Indicator**:
  - `safe_cv_residual_scale` is non-zero (0.2 to 0.5 is typical for well-behaved CV models).
  - `max_cross_reciprocity_error` on full passivity grid is below tolerance (typically ~1e-14 F).
  - Charge sum conservation check passes on the full grid.
  - `selected_cv_residual_mode` is not "off".
**Pitfalls**:
  - Do NOT blindly increase `safe_scale_steps` to try to find a safe non-zero scale — if the grid search rejects all scales, the issue is passivity violations, not search granularity.
  - The cross-reciprocity tolerance is physics-based (charge conservation): violation means the model's charge function is non-conservative in a way that would cause SPICE convergence failures. Relaxing the tolerance is never acceptable.

---


### CV Charge Residual Degradation: Residual Hurting CV When Training Order is Wrong
**Category**: Tuning Pattern
**Stage(s)**: train-static (Phase 3 — Joint CV/Residual Training, Phase 5 — CV safety scan)
**Context**: Agent C observes that `residual_cv_rms` is higher after adding CV charge residual than before. Specifically, CV base (parameter-PKAN only) achieves RMS 0.02073, but after charge residual is activated, RMS degrades to 0.020995. The safety scan then selects `safe_cv_residual_scale=0` because any non-zero scale is worse than the no-residual baseline. The CV residual contribution is negative.
**Pattern**:
  1. **Verify IV is frozen**: Before training CV/charge residual, confirm that all IV parameters are frozen (base IV-PKAN weights, phys_delta parameters, IV residual KAN). Use `--freeze-iv-residual-after-stage --no-finetune-base-pkan`. If IV is NOT frozen, the optimizer may shift IV corrections to compensate for CV residual initialization artifacts, causing CV metrics to appear worse.
  2. **Check that CV parameter KAN has converged before introducing residual**: The CV training phase architecture has hardcoded ramp boundaries (engine-internal, not configurable via INI parameters). CV parameter KAN activates from epoch 0.25×total to 0.58×total. CV residual activates from 0.62×total. If the total number of CV training epochs is insufficient, the parameter KAN hasn't converged by the time residual is introduced — the residual "inherits" a poorly-conditioned parameter base.
  3. **Train CV in two explicit stages**: (a) Train CV parameter KAN ONLY for `cv_parameter_epochs` (e.g., 300-400 epochs) with `enable_cv_residual=False` and IV frozen. Select the best checkpoint based on CV RMS on training data. (b) Freeze CV parameter KAN, enable CV charge residual, train for `cv_residual_epochs` (e.g., 300-400 epochs), select based on held-out CV gain.
  4. **If residual still degrades CV**: The residual's contribution should be evaluated on held-out CV data (not training data). If held-out CV RMS with residual is worse than without, accept `safe_cv_residual_scale=0` and deliver the model without CV charge residual. This is a valid outcome — CV residual is an optional correction, not a requirement.
  5. **Report the decision clearly**: In the quality report, distinguish: (a) "CV residual disabled because held-out evaluation showed no improvement" from (b) "CV residual disabled because safety scan found passivity violations". These have different root causes and different follow-up actions.
**Key Parameters**:
  - `enable_cv_residual`: Set to `False` for the CV parameter-only training stage, `True` for the subsequent residual stage.
  - `cv_residual_level`: Start with 1 (shared broadened-DOS + asymmetric terminal redistribution). Level 2 (inter-gate reciprocal coupling) has higher risk of passivity violations without proportional accuracy gain.
  - `cv_weight`: During CV-only training, increase to 10.0-15.0 (CV is the sole objective, no IV competition). During CV residual stage, reduce to 5.0 to allow more careful residual learning.
  - `epochs` for CV phase: At least 400 for parameter-KAN, at least 400 for residual. The total must be ≥800 epochs for the CV phase alone, separate from IV base/residual epochs.
**Success Indicator**:
  - CV residual held-out RMS ≤ CV parameter-only held-out RMS (residual provides gain, not degradation).
  - If residual is rejected: the report clearly states the reason, and the model is delivered with `safe_cv_residual_scale=0` and `selected_cv_residual_mode="off"`.
  - CV RMS meets the quality gate (typically ≤0.06 dex) with or without residual.
**Pitfalls**:
  - Do NOT force CV residual to be non-zero just because "the IV residual helped". CV and IV residuals learn different things: IV residual corrects log-current, which is directly observable. CV residual corrects charge partitioning, which is an abstract intermediate quantity — its impact on the final observable (CV capacitance = dQ/dV) is indirect and can be destabilizing.
  - Do NOT increase `cv_weight` to try to force residual gain. The residual's failure to improve CV on held-out data is a signal that the base CV parameter KAN already captures the available correction space. Higher cv_weight will force the residual to overfit training data while worsening held-out performance.
  - Do NOT blindly increase `cv_residual_level` to 2. Level 2 enables inter-gate reciprocal coupling — a correction that applies to both VTG and VBG sweeps simultaneously. If the base KAN already handles per-gate corrections well, adding coupling can introduce crosstalk degradation.

**Evidence**:
  - After freezing IV, CV base RMS 0.02073 became 0.020995 with charge residual — degradation confirmed.
  - Safety gate will subsequently suppress or zero it, so this still does not satisfy "CV residual has genuine contribution."
  - Agent correctly handled: first do CV-only state ablation, compare existing CV checkpoint with newly trained parameter/residual branches, select minimal combination that does not touch IV.
  - Final result: CV parameter held-out improved from 0.0205225 to 0.0201922 with parameter KAN only; residual was not needed and correctly not forced.
  - Independently confirmed that PKAN+residual had almost no effect on CV curve shapes — safe scale mechanism was zeroing CV residual because CV shape gates triggered on minor non-physical artifacts. IV residual KAN diagnostic showed 50%+ improvement in IDVTG on-region, but CV residual KAN produced oscillatory behavior. CV residual is a marginal contributor that can easily cause more harm than good when CV shape constraints are active.

---

### Engine Architecture Diagnostics — Beyond Agent C's Control
The following patterns are **engine-level architectural concerns** that Agent C cannot address via `adjust_params`:
- **Derivative observables (gm/gds, dC/dV) not in training loss** → post-hoc safe-scale picks scale=0 because residual was never trained to respect derivatives
- **Physical kernel structurally incomplete before KAN** → KAN compensates for missing physics with non-physical oscillations
- **CV charge model using discrete convolution** → produces false multi-step transitions
- **CV measurement calibration (per-configuration gain/offset)** → should be pre-extracted and frozen before KAN training
- **Separate IV/CV residual scale selection** → single global scale zeros both when CV shape gate triggers
- **CV shape template constraints** → plateau-transition-plateau design vs monotonicity-only loss

Agent C's role: if you observe these symptom patterns → `fallback_to_previous` or `accept_quality`. Do not attempt to fix via parameter tuning. 


