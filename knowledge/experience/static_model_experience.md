# Static Model Experience (from 2026-06-29 + 05-22 sessions)

> **Agent C tools**: `adjust_params` (71 INI params in [full_model]/[base_pkan]/[temperature]/[symbolic]/[validation]), `fallback_to_previous` (→ fit-physical), `accept_quality`, `inspect_xlsx`.  
> **Can influence**: epochs, LR, batch sizes, cv_weight, cv_residual_level, enable_cv_residual, symbolic budgets, freeze flags, warm-start LR multipliers.  
> **Cannot change**: engine training order, optimizer parameter groups, loss function design, safety scan logic, symbolic export architecture.  
> **Principle**: the majority of the actionable advice below focuses on what Agent C can influence via `adjust_params` + correct tool choice. Entries where Agent C's only option is `fallback_to_previous` or `accept_quality` are marked accordingly.

## Principle Insights

### KAN Grid Knot Non-Monotonicity → Identification + Correct Tool Choice

**Diagnostic Signal**: IV curves have sharp stair-step jumps (IDVD ~0.12 dex, IDVTG ~0.07 dex) rather than smooth interpolation. RMS metrics may be unchanged — the corruption is invisible to aggregate scores.

**Agent Impact**: **None.** This is an engine optimizer bug (frozen grid tensors incorrectly made trainable). Agent C cannot fix it via `adjust_params`.

**Correct Action**: If you see stair-step IV artifacts + aggregate RMS looks normal → **immediately `fallback_to_previous`** (→ fit-physical). Do NOT waste retries on `adjust_params` (epochs, LR, etc.) — no parameter change can repair non-monotonic B-spline grids. Do NOT attempt post-hoc grid reprojection.

**Evidence**: TURN 26-28: v95 checkpoint had 9 inverted knots (base) + 4 (residual); v90 grids all monotonic. After fix: IDVD jump 0.12175→0.001244 dex (-98.98%), IDVTG 0.07071→0.002966 dex (-95.81%).

---

### Training Order: Prevent IV Degradation During CV Training

**Diagnostic**: `residual_iv_rms` increases after CV phase vs base IV phase. CV checkpoint selector (weighted by cv_weight=10) picks epochs where CV improved but IV drifted. CV residual may produce negative net contribution (RMS degrades from 0.02073 → 0.020995).

**Agent Impact**: **Partial.** Agent C cannot change the engine training order but CAN control which parameters train in each phase.

**Correct Actions** (what Agent C can do via `adjust_params`):
1. Set `finetune_base_pkan=False` (v96 default) to freeze base PKAN during CV training
2. Request `--freeze-iv-residual-after-stage` to protect IV residual from CV gradients
3. Reduce `cv_weight` to 5.0 during CV residual stage (from default 10.0) to reduce gradient competition
4. If CV residual degrades results: set `enable_cv_residual=False` — CV residual is optional, not mandatory
5. Do NOT set `iv_residual_stage_linear_weight` to 0.34 — use 0.25 (v90 proven)
6. If IV degradation persists after all above → `fallback_to_previous` (→ fit-physical) — the physical kernel may need regeneration

**Evidence**: TURN 7-8: restoring v90 parameter values (700 epoch, LR=0.002, linear=0.25, deriv=36, shape=3) produced 11.3% held-out IV improvement. TURN 8: freezing IV before CV training prevented regression. 05-22 session independently converged on same staged isolation principle.

### Native vs Warm-Start Learning Rate Multipliers
**Category**: Principle Insight
**Stage(s)**: train-static (base IV-PKAN and joint CV training phases)
**Symptom**: A freshly trained base IV-PKAN (no pre-existing checkpoint for this device) converges extremely slowly or plateaus at high loss when using `base_iv_learning_rate=0.002` multiplied by `warm_start_base_pkan_lr_mult=0.035` (effective LR = 7×10⁻⁵). The base PKAN effectively learns nothing, and subsequent residual training cannot compensate because the base under-represents the physical corrections. Conversely, using `LR=1.0` (i.e., `warm_start_base_pkan_lr_mult=1.0`) on a fine-tuning scenario from a good pre-trained checkpoint destroys the learned weights in ~15% of epochs.
**Root Cause**: The `warm_start_base_pkan_lr_mult` parameter exists specifically for the case where the pre-trained base PKAN checkpoint already contains good device corrections, and the joint training phase should only make small adjustments. A multiplier of 0.035 means the effective LR = base LR × 0.035 — appropriate for protection. But for a `reset_base_from_physical_fit=True` scenario (fresh device, no prior KAN weights), the base PKAN correction heads are initialized to zero and need to learn from scratch. The 0.035 multiplier starves the optimizer. The v90 workflow avoided this because v90 never did fresh-start — it always fine-tuned from a pre-trained checkpoint. The v93+ workflow introduced fresh-start capability but kept the warm-start multiplier as the default, causing regression.
**Evidence**:
  - TURN 16: "Fresh-start 的 LR 保持 1.0 是必要差异，因为 v90 的 0.035 只适用于已训练 warm-start" (diagnostic.md:490)
  - TURN 18: "唯一不应照抄的是 v90 对已有 checkpoint 的 warm-start 倍率 0.035；新器件 fresh-start 必须用 1.0，否则正是此前退化来源" (diagnostic.md:534)
  - TURN 3: Fresh base training at LR=1.0 reduced high-current RMS from ~0.22 to 0.0388 in ~40 seconds, showing the correct rate for fresh-start (diagnostic.md:55)
  - TURN 7: v90's effective base IV LR was 0.002 × 1.0 = 0.002 for warm-start with pre-trained checkpoint; this was the correct fine-tuning rate (diagnostic.md:192)
**Generalization**: Any pipeline that supports both `reset_base_from_physical_fit=True` (fresh start) and `reset_base_from_physical_fit=False` (resume/warm-start) MUST switch the effective base PKAN learning rate between the two modes. The warm-start multiplier is a *protection* mechanism (not an optimization parameter): it exists to prevent CV loss gradients from destroying pre-trained base PKAN weights in the first 15% of joint training epochs. It has no useful role in fresh-start training.
**Actionable Advice**:
  1. **When `reset_base_from_physical_fit=True`**: Set `warm_start_base_pkan_lr_mult=1.0` (or omit it entirely — same effect). Set `base_iv_learning_rate=0.002` (v90 proven value) and do not apply any LR multiplier during base IV PKAN phase. During the later joint CV phase, since `finetune_base_pkan=False` (v96 default), the base PKAN will be frozen anyway — the warm_start multiplier is irrelevant.
  2. **When `reset_base_from_physical_fit=False`** (resume from checkpoint): Set `warm_start_base_pkan_lr_mult=0.035` and `finetune_base_pkan=True`. This protects the pre-trained PKAN from early destruction when CV loss is introduced. The multiplier ramps from 0.035 to `base_pkan_lr_mult` over the warm-up period.
  3. **Automatic detection**: The engine/pipeline should auto-detect the mode and set the multiplier. If `reset_base_from_physical_fit` is True (or `--reset-base-from-physical-fit` flag is present), override `warm_start_base_pkan_lr_mult` to 1.0 regardless of INI value. Evidence: the session spent multiple turns debugging LR mismatch because the INI had the wrong default for the fresh-start scenario (diagnostic.md:490-534).
  4. **What NOT to do**: Do not blindly copy v90's INI config (which has `warm_start_base_pkan_lr_mult=0.035`) to a fresh-start deployment. This is the exact "copy v90, get worse results" problem the user identified. Do not set `base_iv_learning_rate` above 0.005 — even for fresh-start, rates >0.005 caused oscillations (diagnostic.md:55 shows LR=1.0 was used for a quick test, but the final settled value was 0.002).

---

### Symbolic Distillation: Budget and Parameter Tuning

**Diagnostic**: HSPICE validation shows large errors (12.94% max relative) despite good numeric IV RMS. Edge-local Top-1 compatibility gate passes but terminal-level current accuracy is much worse.

**Agent Impact**: **Partial.** Agent C controls symbolic budget parameters (edge_steps, network_finetune_steps, device_iv_finetune_steps, device_iv_measured_steps) but not the edge-local vs terminal validation approach.

**Correct Actions**:
1. Increase symbolic budget: `edge_steps=180`, `network_finetune_steps=360`, `device_iv_finetune_steps=3000` (v90 values). Do NOT reduce below these.
2. Set `device_iv_measured_steps=0` — the extra 1600-step measured-only phase causes HSPICE error increase (4.886%→0.956% when removed).
3. Restore v90 IV parameters if drifted: `iv_residual_stage_linear_weight=0.25`, `iv_residual_stage_deriv_curves=36`, `iv_residual_stage_shape_period=3`. (Note: these defaults are already restored in current param_defs.py.)
4. Increase `base_iv_top1_symbolic_weight` to 0.35 (from 0.05) if edge-level compatibility is poor — pushes KAN edges toward simpler shapes during training.
5. Relax `base_iv_max_top1_compatibility` to 0.02-0.03 if terminal accuracy is borderline — but only AFTER restoring v90 symbolic budget.
6. If HSPICE errors persist after all above → `accept_quality` with documentation. The numeric model is accurate; deployment errors may be acceptable for some use cases.

**Evidence**: TURN 17: restoring v90 parameters reduced HSPICE IV error from 4.886% → 0.956%. TURN 19: edge-local gate passed (all <0.015) but terminal accuracy was 0.122 dex vs numeric 0.082 dex.

---

### HSPICE Validation Reference Model — OUTDATED (FIXED)
**Status**: FIXED as of current engine. The HSPICE validation function at `dg_tft_full_model.py:16285` wraps its computation inside `with _temporary_deployment_symbolic_nets(diag):`, which explicitly swaps in `_deployment_base_iv_symbolic_net`, `_deployment_iv_residual_symbolic_net`, and `_deployment_cv_symbolic_net` — the **symbolic** KAN students, NOT the numerical spline nets. The smoke report self-documents: `"python_reference_model": "deployed_symbolic_students"`. **This entry is retained for historical reference only — the reference model mismatch has been resolved.**

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

**Evidence**: TURN 19-20: base had 4 extrapolation alerts, all non-zero residuals rejected. After fix to relative criterion: residual accepted at scale 0.5.

---

### CV Finalize Polish Silently Degrading Accepted Models — OUTDATED (FLAG EXISTS)
**Status**: FIXED as of engine v96. The `--skip-post-train-cv-polish` flag exists at `dg_tft_full_model.py:17641` and is actively used in the non-finalize train path (line 18775). The orchestrator should pass this flag during finalize/resume modes. **This entry is retained for historical reference only — the flag is available and should be used.**

---

### CV Passivity Scan: Joint Scan Defaults and Zero-Out Bug — PARTIALLY FIXED
**Category**: Tuning Pattern
**Stage(s)**: train-static (Phase 5 — Joint CV safe scale search)
**Code Reality (verified 2026-07-03)**: The actual defaults are:
- `joint_cv_passivity_scan_points`: **80** (not 48)
- `safe_passivity_scan_points`: **160** (not 96)  
- `passivity_max_points`: **1200** (not 96)
There is NO separate "coarse pass then fine pass" two-stage architecture. `choose_safe_joint_cv_scales` runs a single exhaustive 2D grid search over `(parameter_scale × mode × residual_scale)` using `joint_cv_passivity_scan_points=80`.
**Context**: Agent C observes that `safe_cv_residual_scale=0` despite `residual_cv_rms` appearing reasonable in intermediate training logs. The scan report shows that a joint scan accepted a `cv_residual_scale` value, but the final report has `selected_cv_residual_mode="off"` and `safe_cv_residual_scale=0.0`. The charge conservation / passivity check on the full grid found violations.
**Pattern**:
  1. **Understand the two-stage scan**: The `joint_cv_safe_scale_search` runs a coarse 2D grid over (parameter_scale × residual_scale) using a reduced set of passivity checkpoints (48 points). At each grid point, it checks charge sum, cross-reciprocity (`max_cross_reciprocity_error`), and diagonal negativity. The coarse scan selects the best (parameter_scale, residual_scale) pair.
  2. **Identify the rejection bug**: After coarse scan accepts a pair, the fine scan (full 96-point passivity grid) is run on just that pair. If the fine scan finds violations (e.g., `max_cross_reciprocity_error` exceeds tolerance), the OLD code immediately sets residual_scale=0 instead of searching downward for a smaller-but-nonzero scale. This is the "zero-out on fine rejection" bug.
  3. **Apply the rescan fix**: Change the scan logic to: (a) run coarse scan to select parameter_scale (ignore coarse residual_scale), (b) fix parameter_scale at the selected value, (c) on the FULL 96-point passivity grid, scan all residual_scale values from 0.0 upward, (d) select the LARGEST residual_scale that passes fine-grid passivity. This ensures the scan finds the true maximal safe scale rather than yielding binary 1.0 or 0.0.
  4. **Check for parameter_scale issues**: If even with the rescan fix, no non-zero residual scale passes fine-grid passivity, the problem may be that the coarse scan selected a parameter_scale that already violates passivity on the fine grid. Evidence: "即使 residual=0，联合粗扫选出的 parameter scale=0.3 已使最大交叉互易误差达到 6.53e‑15，超过完整门槛" (diagnostic.md:281). In this case, re-run with a more conservative parameter_scale scan (smaller upper bound).
  5. **Use parameter-only scan as backup**: If the joint scan consistently fails, fall back to parameter-only safety scan (`safe_cv_parameter_scale` selection) with `joint_cv_safe_scale_search=False`, then separately scan CV residual scale with fixed parameter_scale.
**Key Parameters**:
  - `joint_cv_safe_scale_search`: Set to `True` only if CV passivity is likely marginal. In most cases, `False` (parameter-then-residual sequential scan) is more robust.
  - `safe_scale_steps`: Default 8. For the fine-grid rescan after parameter_scale selection, use the same number of steps but with a finer range (0.0 to 1.0) to avoid wasting time on scales that already failed.
  - `cv_residual_level`: If the scan consistently finds that ANY residual scale violates passivity, reduce this to 1 (disable inter-gate coupling branch) or 0 (disable all CV residual branches). Evidence shows `cv_residual_level=2` activates the inter-gate reciprocal coupling branch which has higher passivity violation risk.
**Success Indicator**:
  - `safe_cv_residual_scale` is non-zero (0.2 to 0.5 is typical for well-behaved CV models).
  - `max_cross_reciprocity_error` on full 96-point grid is below tolerance (typically ~1e-14 F).
  - Charge sum conservation check passes on the full grid.
  - `selected_cv_residual_mode` is not "off".
**Pitfalls**:
  - Do NOT force `safe_cv_residual_scale=1.0` based on the coarse scan result. The coarse grid has fewer charge neutrality checkpoints and can miss localized violations that appear only in certain bias regions.
  - Do NOT blindly increase `safe_scale_steps` to try to find a safe non-zero scale. If the coarse scan accepts scale=1.0 but fine scan rejects it, the issue is grid density (number of verification points), not scan linearity. The fix is the rescan logic, not more steps.
  - The cross-reciprocity tolerance is physics-based (charge conservation): violation means the model's charge function is non-conservative in a way that would cause SPICE convergence failures. Relaxing the tolerance is never acceptable.

**Evidence**:
  - TURN 8: "联合粗扫在 48 点 passivity 上接受 CV residual=1，但后续 96 点复核发现最大交叉互易误差超限，于是旧代码直接回退到 0，而没有搜索较小 residual scale。这正是 residual 被错误压到 0 的路径" (diagnostic.md:276)
  - TURN 8: "修复为：联合粗扫确定 parameter scale/模式，随后在完整 passivity 网格上重新搜索所有 residual scale，只有真实通过者才保留" (diagnostic.md:276)
  - TURN 8: "即使 residual=0，联合粗扫选出的 parameter scale=0.3 已使最大交叉互易误差达到 6.53e‑15，超过完整门槛。因此必须回退 parameter scale，而不是把 residual 归零" (diagnostic.md:281)
  - TURN 8: Final routed fix — "改用完整 parameter-only 安全选择（已选 0.1），再在该 parameter scale 上搜索非零 residual" (diagnostic.md:281)

---

### v90 Symbolic Pipeline: Key Default Parameter Differences — OUTDATED (ALL RESTORED)
**Status**: FIXED as of current `param_defs.py`. All three "drifted" defaults have been restored to v90 values: `iv_residual_stage_linear_weight=0.25` (param_defs.py:391), `iv_residual_stage_deriv_curves=36` (param_defs.py:403), `iv_residual_stage_shape_period=3` (param_defs.py:415). Engine argparse defaults also match (e.g., line 17234: `default=0.25`). **This entry is retained for historical reference only — the parameter values are already corrected.**

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
  - TURN 8: "冻结 IV 后，CV base RMS 0.02073，加入 charge residual 后为 0.020995，出现退化" (diagnostic.md:266)
  - TURN 8: "安全门后续会把它压低或置零，因此这仍不满足'CV residual 有真实贡献'" (diagnostic.md:266)
  - TURN 8: Agent correctly handled: "先做 CV-only 状态 ablation，比较现有当前 CV checkpoint 与这次新训练的 parameter/residual 分支，选择不会触碰 IV 的最小组合" (diagnostic.md:266)
  - TURN 8: Final result — CV parameter held-out improved from 0.0205225 to 0.0201922 with parameter KAN only; residual was not needed and correctly not forced (diagnostic.md:270)
  - **Additional Evidence from 05-22 session**: Turns 108-110 independently confirmed that PKAN+residual had almost no effect on CV curve shapes — the safe scale mechanism was zeroing CV residual because CV shape gates triggered on minor non-physical artifacts. Turn 65 residual diagnostic showed IV residual KAN provided 50%+ improvement in IDVTG on-region, but CV residual KAN produced oscillatory behavior. The lesson from the earlier session is the same: CV residual is a marginal contributor that can easily cause more harm than good when CV shape constraints are active. (05-22 diagnostic.md:2670-2684)

---

### Engine Architecture Diagnostics — See Agent E
The following patterns were identified in the 05-22 session but are **engine-level architectural concerns** that Agent C cannot address via `adjust_params`. They have been moved to `knowledge/experience/final_interpret_experience.md` for Agent E's diagnostic reference:
- **Derivative observables (gm/gds, dC/dV) not in training loss** → post-hoc safe-scale picks scale=0 because residual was never trained to respect derivatives
- **Physical kernel structurally incomplete before KAN** → KAN compensates for missing physics with non-physical oscillations
- **CV charge model using discrete convolution** → produces false multi-step transitions
- **CV measurement calibration (per-configuration gain/offset)** → should be pre-extracted and frozen before KAN training
- **Separate IV/CV residual scale selection** → single global scale zeros both when CV shape gate triggers
- **CV shape template constraints** → plateau-transition-plateau design vs monotonicity-only loss

Agent C's role: if you observe these symptom patterns → `fallback_to_previous` or `accept_quality`. Do not attempt to fix via parameter tuning. See Agent E experience for the full diagnostic context.

## Session Meta-Notes (05-22)

The 05-22 session represents the **original development of the Physical Kernel + PKAN methodology** from first principles. It began with raw data formatting (Turns 1-10), proceeded through physical kernel fitting with an empirical EKV model (Turns 11-17), and then spanned approximately 90 turns of iterative PKAN architecture development (Turns 18-110). Key methodological milestones:

- **v5-v12** (Turns 31-52): Initial PKAN training script, CV kernel as shared-trap charge model, discovery that CV must be charge-based (Q->C via autograd) not capacitance-based
- **v14-v20** (Turns 53-58): Move toward EKV-style terminal charge, introduction of terminal-pair parasitic network, recognition that CV data has measurement-configuration dependencies
- **v22-v26** (Turns 59-63): Distributed CV charge kernel with channel participation, CV data quality audit, sheet-level CV quality gating
- **v27-v30** (Turns 65-69): Residual KAN approach, discovery that parameter-KAN is too constrained, strong residual KAN shows 50%+ improvement
- **v31-v35** (Turns 70-81): Safe residual scale mechanism, verbose output system, Verilog-A export
- **v36-v41** (Turns 82-90): Derivative losses (gm/gds, dC/dV) enter main training, CV sheet calibration, shared IV/CV parameter tying, charge conservation diagnostics
- **v42-v50** (Turns 92-109): CV shape template constraints, discovery of discrete convolution artifacts, move to single-transition commercial CV kernel, CBG/CTG label swap detection

The session produced the methodological foundation later refined in the 06-29 session, but at this stage the engine (v96 architecture) and formal quality gates did not yet exist — all quality assurance was done through visual inspection of output curves and manual shape diagnostics.
