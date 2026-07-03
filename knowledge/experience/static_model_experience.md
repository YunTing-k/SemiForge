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

## Principle Insights (from 05-22 session)

### Derivative Observables (gm/gds, dC/dV) Must Enter Main Training Loss — Not Just Post-Hoc Screening
**Category**: Principle Insight
**Stage(s)**: train-static (all PKAN and residual training phases)
**Symptom**: The trained model achieves acceptable IV RMS and CV RMS on point-wise log-error metrics, but the output curves show physically implausible features: IDVTG on-region has unrealistic curvature (excessive gm roll-off or sharp kinks), CV curves have local oscillations or multi-step transitions, and residual KAN contributions visually "twist" the curves in non-physical ways. These artifacts are invisible to aggregated RMS metrics but would cause SPICE convergence failures or wrong small-signal simulation results.
**Root Cause**: The main training loss was designed to minimize point-wise log-error between prediction and measurement — `log(ID_pred/ID_meas)` for IV, `log(C_pred/C_meas)` for CV. Derivatives (transconductance `gm`, output conductance `gds`, `dC/dV`) were computed ONLY during post-training diagnostics and safe-scale screening. This created a fundamental mismatch: the optimizer minimizes what it's told to minimize (MSE on current/capacitance values), but the actual deployment requirement is smooth, monotonic derivatives. The residual KAN's natural response to the MSE-only loss is to overfit local curvature to shave off small numerical errors, producing jagged `gm` and oscillatory `dC/dV` that pass RMS checks but would fail any circuit simulator.
**Evidence**:
  - 05-22 TURN 82: Model shows IDVTG on-region physically implausible curvature after residual KAN — "加了residual之后，IDVTG的开态的曲线变化明显不符合物理，是你没考虑到二阶导吗" (05-22 diagnostic.md:2016)
  - 05-22 TURN 83: Root cause identified — "筛图没有任何意义，不是最终的结果，模型才是最终的结果...导数只用于晒图肯定不对" (05-22 diagnostic.md:2056)
  - 05-22 TURN 83: Code fix — "每一步训练都抽完整曲线子集，主 loss 直接包含 ID/gm/gds、CV 一阶/二阶导、residual 二阶导和 base-aware 约束" (05-22 diagnostic.md:2062)
  - 05-22 TURN 83: Reference to Hu/Kao/Kam IEEE EDL physics-driven NN models — "同时看 ID/gm/gds，把导数拟合提升到主损失，不只是最后筛图" (05-22 diagnostic.md:2038-2040)
  - 05-22 TURN 37: Early foreshadowing — "CV 的不正常抖动...主要是损失函数设置的不对" (05-22 diagnostic.md:945)
**Generalization**: This applies to any compact model training where the final deployment target is a circuit simulator. Circuit simulators solve differential equations — they need smooth `gm = dID/dVG`, `gds = dID/dVD`, and `C = dQ/dV`. A model with 0.02 dex lower IV RMS but jagged second derivatives will fail in SPICE. The loss function MUST include derivative terms as optimization objectives, not just as post-hoc diagnostic metrics. Without this, the model achieves "good numbers" at the cost of "bad physics."
**Actionable Advice**:
  1. **Add gm/gds terms to IV loss**: For each training batch (or periodic full-curve subset), compute finite-difference `gm = dlog(ID)/dVG` and `gds = dlog(ID)/dVD` along each sweep direction. Add these as additional loss terms weighted by `gm_loss_weight` (~0.15-0.25) and `gds_loss_weight` (~0.05-0.10).
  2. **Add dC/dV and d2C/dV2 terms to CV loss**: For CV, work on a smoothed version of measured C (to avoid chasing measurement noise in derivatives). Use a Savitzky-Golay filter or Gaussian-smoothed reference curve for the `dC/dV` target, NOT the raw noisy data. Add second-derivative smoothness penalty on the model's own CV prediction.
  3. **Use base-aware relative constraints**: Do NOT set absolute derivative constraints ("CV slope must be >= 0") — this backfires when the physical kernel itself has minor slope variation. Instead use: `max(0, residual_derivative_worsening - base_derivative_worsening)` — residual must not make derivatives worse than base.
  4. **What NOT to do**: Do not delegate derivative enforcement to the post-training safe-scale selection. If derivatives are only checked during safe-scale screening and not in training loss, the safe-scale will simply pick scale=0 (no residual) because any non-zero residual introduces derivative artifacts it was never trained to avoid. Evidence: 05-22 Turns 82-83 showed this exact failure pattern across v35-v36 (05-22 diagnostic.md:2021-2062).

---

### Physical Kernel Must Be Structurally Sufficient BEFORE KAN — KAN Compensating for Structural Deficiencies Produces Non-Physical Curves
**Category**: Principle Insight
**Stage(s)**: train-static (CV physical kernel design, all PKAN/residual training)
**Symptom**: CV fitting metrics remain poor (RMS > 0.20 dec) despite multiple rounds of adjusting KAN capacity, residual strength, epoch count, and loss function tuning. Visual inspection reveals systematic patterns: CTG predictions are consistently ~2x higher than measurement across all bias conditions; CBG low-capacitance plateau is substantially above the measured floor; CGG shows non-physical double-hump transitions. The KAN and residual networks, instead of learning small device-specific corrections, attempt to compensate for fundamental model structure issues — producing curves with local oscillations, multi-step transitions, and false shoulders.
**Root Cause**: The physical CV kernel (charge model → capacitance via autograd) had three structural gaps: (1) No measurement-configuration de-embedding — the EKV-style charge model produces intrinsic terminal capacitances, but the measured CV includes parasitic, contact, and measurement-path contributions that scale differently per configuration; (2) The transition width of the CV turn-on region was determined entirely by the thermal kT/q sharpness of the EKV charge model, but real IGZO TFTs have disorder/trap-induced broadening; (3) The inter-gate charge partitioning assumed ideal electrostatic coupling through a perfectly conductive channel sheet, which overestimates top-to-bottom gate capacitive coupling in the OFF state. KAN was being asked to perform a role it cannot fulfill: large-scale global corrections that should be in the physical kernel itself.
**Evidence**:
  - 05-22 TURN 37: Full model audit — "KAN 正在替物理模型擦屁股...物理核里有几个结构性缺口" (05-22 diagnostic.md:798)
  - 05-22 TURN 52: "CV 的主要系统误差不是单纯'缺电容'，而是 top/bottom 归属错了...CBG 在中高压严重偏低，而 CTG 同区间严重偏高" (05-22 diagnostic.md:1217)
  - 05-22 TURN 84: v38 diagnosis — "CTG 和 vds 纯物理核整体偏高，说明 residual KAN 不是在微调，而是在试图把一个系统性 offset/scale 错误往下拉" (05-22 diagnostic.md:2089)
  - 05-22 TURN 88: After sheet calibration — "初始化结果很说明问题：CTG 需要约 0.65x 的测量参与系数，vds 需要约 0.49x" (05-22 diagnostic.md:2106)
  - 05-22 TURN 93: "CGG 的 double hump 基本不是器件物理，而是模型先验造成的假结构" (05-22 diagnostic.md:2279)
  - 05-22 TURN 106: User insight — "商业 CV compact model 根本不用 convolution" (05-22 diagnostic.md:2537)
**Generalization**: For any physics+ML hybrid model, the physical component must be structurally complete before the ML component is added. If the physical model is missing a first-order effect (e.g., measurement de-embedding, disorder broadening, interface trap capacitance), the neural network will learn to compensate — but it will do so using high-frequency, non-physical basis functions (spline nodes in KAN), producing curves that fit numerically but violate physical constraints. The correct decomposition is: physical kernel handles all first-order, globally-varying physics; KAN handles second-order, bias-dependent device-specific deviations.
**Actionable Advice**:
  1. **Before training KAN, verify physical kernel baseline**: Run pure physical kernel prediction on ALL CV data configurations. If any configuration shows systematic offset (e.g., >50% scale error across all bias points), this is a kernel deficiency, not a KAN task. Fix the kernel first.
  2. **Add measurement configuration de-embedding to physical kernel**: Each CV measurement configuration (cgg/cbg/ctg/vds) needs a local `gain` and `offset` parameter representing open/short de-embedding, AC participation ratio, and contact parasitics. These are pre-extracted from low-bias/high-bias plateau regions and frozen before KAN training.
  3. **Add disorder/trap broadening to CV transition physics**: The EKV core's sharp thermal transition must be broadened by physical parameters: `dlog_nfac_cv`, `dlog_Cit_t/b`, `dlog_trap_width`. These are global device parameters, not per-curve, and should be allowed substantial tuning range (not just micro-delta of ±1%).
  4. **Add OFF-state channel coupling modeling**: In the OFF state, the channel is depleted — the two gates are NOT electrostatically coupled through a conductive sheet. Add a `channel_participation` factor that smoothly transitions from weak (OFF) to strong (ON) as a function of normalized inversion charge.
  5. **What NOT to do**: Do not increase KAN hidden layer size or add residual CV branches to compensate for poor kernel baseline. Evidence: the 05-22 session spent 30+ turns (v5→v35) adjusting KAN/residual hyperparameters with minimal CV improvement, then gained the largest single improvement from adding sheet-level calibration and fixing CBG/CTG label orientation, not from making KAN stronger (05-22 diagnostic.md:2089-2110).

---

### CV Charge Model Must Have Single Smooth Transition by Physical Construction — Convolution/DOS Artifacts Create False Multi-Step Transitions
**Category**: Principle Insight
**Stage(s)**: train-static (CV kernel architecture)
**Symptom**: CV curves (especially CGG and CBG) show 2-3 distinct steps or "shoulders" in the transition from OFF capacitance to ON capacitance plateaus. These appear as multiple derivative peaks, visible as double-hump or triple-hump structures that do not correspond to any physical mechanism in the device. The artifacts persist even after extensive KAN training and can be traced to the CV kernel implementation, not the PKAN or residual networks.
**Root Cause**: Two distinct mechanisms produced these artifacts at different development stages:
  1. **Discrete Gaussian convolution (v45-v47)**: A 3-point Gaussian convolution applied to the base CV kernel produced 3 shifted copies of the original sharp EKV transition. Since the EKV transition is essentially a sigmoid, convolving it with 3 discrete shifted copies produced 3 visible steps in the output.
  2. **Multiple independent charge sub-components (v26 base)**: The base CV kernel combined `cv_core (EKV inversion) + cv_trap (fast/slow trap) + cgg_tail (background)` as additive components. If these components activate at different gate voltages, the resulting capacitance curve shows a staircase of 3-4 separate transitions.
  3. **Residual KAN with multiple narrow defect branches (v42-v43)**: When CV residual KAN output individual narrow sigmoid contributions, the optimizer could discover parameter values that produce multiple narrow capacitance bumps on top of the base transition.
**Evidence**:
  - 05-22 TURN 103: "CV 依旧出现了奇特的峰...v47 的离散高斯卷积实现方式本身把一个陡峭台阶复制成了 3 个台阶" (05-22 diagnostic.md:2476, 2487)
  - 05-22 TURN 104: Diagnosis — base CV kernel with 3-point convolution produced up to 3 derivative peaks per curve, while single-transition template reduced all curves to exactly 1 peak (05-22 diagnostic.md:2516)
  - 05-22 TURN 105: User direction — "商业 CV compact model 根本不用 convolution...让我重新想，从工业界的标准做法学起" (05-22 diagnostic.md:2537)
  - 05-22 TURN 106: Implementation fix — "基础 CV 回到 base charge kernel，允许受控微调 nfac/Cit/trap_width 这一类展宽参数，并加形状约束确保不会再出现多台阶" (05-22 diagnostic.md:2539)
**Generalization**: The base capacitance function `C(V) = dQ(V)/dV` must produce exactly ONE transition from OFF plateau to ON plateau by construction. Any technique that generates capacitance by summing multiple shifted copies of a sharp transition (convolution, quadrature, DOS broadening with discrete basis) will inherently create false multi-step structures. Commercial compact models achieve wide transitions through intrinsic physics (disorder-broadened surface potential, effective trap density-of-states) where the broadening is continuous and `dQ/dV` is guaranteed to be a single smooth function.
**Actionable Advice**:
  1. **Use a single-transition charge expression**: Define terminal charges as smooth functions with TWO plateau levels (OFF/ON) and ONE continuous transition region. The transition width is controlled by `nfac`, `Cit`, and `trap_width` — global physical parameters shared across all CV curves.
  2. **Avoid discrete convolution for transition broadening**: Do NOT use `cap_broad = sum(w_i * cap(V - dV_i))` — this introduces N copies of each transition feature. If broadening is needed, incorporate it into the charge function itself via continuous functions.
  3. **Avoid additive charge sub-components with independent activation voltages**: If the model has `Q_total = Q_inv + Q_trap_fast + Q_trap_slow + Q_tail`, each `Q_i(V)` must share the SAME effective threshold voltage. Trap and tail contributions should be parameterized as multiplicative corrections on `Q_inv(V)`.
  4. **Detection**: After each CV kernel change, plot `dC/dV` for all CV curves. Count derivative peaks — any curve with >1 distinct peak indicates a false multi-transition and requires kernel redesign.
  5. **What NOT to do**: Do not add more KAN capacity to "smooth out" the multi-step artifacts. The artifacts are deterministic consequences of the kernel structure and cannot be removed by neural network training — KAN can only attenuate them at the cost of introducing compensating oscillations elsewhere. (05-22 diagnostic.md:2476-2598)

---

## Tuning Patterns (from 05-22 session)

### CV Measurement Configuration Calibration — Sheet Gain/Offset as Pre-Training De-Embedding Step
**Category**: Tuning Pattern
**Stage(s)**: train-static (CV data preprocessing and training)
**Context**: CV physical kernel predictions consistently show systematic bias: CTG prediction is ~1.7-2.0x higher than measured across all bias points; VDS prediction is ~2.0x higher; CBG low-plateau capacitance is substantially above measured floor. The residual KAN wastes capacity trying to pull entire curves up/down by large factors, introducing local oscillations instead of learning smooth device-specific features.
**Pattern**:
  1. **Identify config-level bias, not per-point error**: If `mean(predicted/measured)` is far from 1.0 for an entire CV curve, this is a measurement-configuration mismatch. The physical kernel computes intrinsic terminal charge → capacitance; the measured value includes open/short de-embedding error, contact parasitics, effective area uncertainty, and AC frequency response.
  2. **Pre-train per-configuration gain/offset**: Before any PKAN or residual training, estimate a `gain` and `offset` for each measurement configuration (cgg, cbg, ctg, vds): `C_calibrated = gain * C_intrinsic + offset`. Use the OFF plateau (lowest 20% of voltage sweep) and ON plateau (highest 20%) to constrain these.
  3. **Freeze calibration during PKAN/residual training**: The gain/offset represents measurement-path properties, not device physics. Once extracted, freeze these parameters. If left trainable, the optimizer will use them to absorb device physics errors, masking the real modeling problem.
  4. **Document as measurement wrapper, not device model**: In reports and Verilog-A output, clearly separate the "measurement calibration wrapper" from the "device compact model." Only the device model goes into final VA export.
**Evidence**:
  - 05-22 TURN 84: "CTG 和 vds 纯物理核整体偏高，说明 residual KAN 不是在微调，而是在试图把一个系统性 offset/scale 错误往下拉" (05-22 diagnostic.md:2089)
  - 05-22 TURN 88: Sheet calibration initialization — "CTG 需要约 0.65x 的增益，vds 需要约 0.49x 的增益" (05-22 diagnostic.md:2106)
  - 05-22 TURN 88: Result — "CV RMS 从约 0.183 dec 降到 0.129 dec，CTG 从 0.121 dec 降到 0.034 dec，vds 从 0.260 dec 降到 0.079 dec — 最大的改善来源于测量配置校准" (05-22 diagnostic.md:2110)
  - 05-22 TURN 109: Verification — "解冻 sheet calibration 在前几轮几乎不改善 CBG，反而会把测量 wrapper 和器件内核重新缠在一起" (05-22 diagnostic.md:2643)
**Key Parameters**:
  - Per-configuration `cv_sheet_gain`: Typically 0.4-1.2. Initialize from OFF/ON plateau ratio `(C_meas_plateau / C_model_plateau)`.
  - Per-configuration `cv_sheet_offset`: Typically +/- 1e-12 F for open de-embedding residual.
  - `freeze_cv_sheet_calibration`: Set to True after initialization.
**Success Indicator**: CV prediction error becomes balanced (positive and negative residuals of similar magnitude). CTG/vds gain values are physically reasonable (0.4 < gain < 1.2). KAN residual corrections become small.
**Pitfalls**: Do NOT allow per-curve gain/offset — this is per measurement *configuration*, shared across all curves within that config. Do NOT embed measurement calibration into the Verilog-A device model. If gain is <0.3 or >3.0, the measurement configuration mapping is likely wrong (e.g., CBG/CTG labels swapped).

---

### Residual KAN Safety Gate with Separate IV/CV Scale Selection — Single Global Scale Kills IV Residual When CV Shape Gate Triggers
**Category**: Tuning Pattern
**Stage(s)**: train-static (safe residual scale selection)
**Context**: The model trains fine, IV residual KAN shows clear 50%+ improvement in IDVTG on-region RMS, but the safe-scale post-training scan selects `safe_residual_scale = 0.0`, completely disabling ALL residual KAN contributions (both IV and CV). The per-scale scan table reveals: residual provides monotonic IV improvement at every scale, but CV shape diagnostics flag minor derivative artifacts at non-zero scales. The single global scale selector sees "CV not strictly improving" and zeros everything.
**Pattern**:
  1. **Check if CV shape gate is the bottleneck**: Read the per-scale scan table. If IV metrics improve with scale but CV metrics degrade slightly, the bottleneck is specifically CV shape.
  2. **Separate IV and CV residual scales**: Use two independent scales: `safe_iv_residual_scale` (selected by IV shape+accuracy criteria) and `safe_cv_residual_scale` (selected by CV shape+accuracy criteria). IV scale can be non-zero even if CV scale is zero.
  3. **Apply per-domain safety criteria**: IV scale selection considers IV RMS, gm/gds smoothness, IDVD monotonicity, extrapolation anchor errors. CV scale selection considers CV negative slope count, plateau flatness, single-transition peak count.
  4. **Default CV residual to off**: For most devices, CV residual provides marginal benefit (<0.005 dec RMS) at the cost of shape risk. Set `cv_residual_enabled=False` as default.
**Evidence**:
  - 05-22 TURN 65: IV residual alone produced "IDVTG 开态 RMS 减半的提升，非常大" but CV residual produced oscillatory behavior (05-22 diagnostic.md:1609-1611)
  - 05-22 TURN 109: "之前只有一个全局 safe residual scale，CV 只要对 residual 稍微敏感，就会把整个 residual scale 压到 0，连 IV residual 也一起被关掉" (05-22 diagnostic.md:2667)
  - 05-22 TURN 109: After separation — "CV residual 关闭后，safe scale 不再被 CV 拖死，IV residual 仍可启用" (05-22 diagnostic.md:2671)
  - 05-22 TURN 110: User confirms — "PKAN+residual 没作用是对 CV 曲线修正没有作用" (05-22 diagnostic.md:2677)
**Key Parameters**:
  - `enable_iv_residual`: True (default). IV residual is the primary KAN correction target.
  - `enable_cv_residual`: False (default). Only enable when CV shape gates can be satisfied.
  - `cv_residual_safety_margin`: 1.5. Allows small shape degradation in exchange for CV RMS improvement.
  - Separate `iv_residual_scale_steps` and `cv_residual_scale_steps` for independent scan density.
**Success Indicator**: `safe_iv_residual_scale` is non-zero (0.5-1.0). IV metrics improve with residual. CV metrics do not degrade even if CV residual is disabled.
**Pitfalls**: Do NOT force both residuals to be non-zero. It is a valid outcome that CV residual is disabled while IV residual remains active. Do NOT relax CV shape gates just to get a non-zero CV residual scale. Do NOT use a combined IV+CV score for residual scale selection.

---

### CV Shape Template Constraints — Platform-Transition-Platform vs Monotonicity-Only Loss Design
**Category**: Tuning Pattern
**Stage(s)**: train-static (CV loss function design)
**Context**: CV loss functions that only enforce monotonicity (`dC/dV >= 0`) or smoothness (`|d2C/dV2| small`) are insufficient. The model can satisfy both while still producing curves with double-hump transitions, continued slow rise in the high-voltage plateau, and residual KAN twist. These artifacts pass monotonicity checks but violate the expected "plateau-transition-plateau" shape of gate capacitance.
**Pattern**:
  1. **Define shape template per CV type**: For gate CV (CGG/CBG/CTG): OFF plateau (V < Vth-2V) ≈ constant, Transition (Vth-2V to Vth+2V) single monotonic rise, ON plateau (V > Vth+2V) ≈ constant. No second peak, no continued rise. VDS: same but ON plateau may shift with VDS due to channel pinch-off.
  2. **Add template-based loss terms**: Unimodality penalty (dC/dV should have exactly one peak). Plateau flatness penalty (top/bottom 25% of sweep: 10x weight on |dC/dV|). No continued rise penalty (final 20% dC/dV must be near-zero).
  3. **Make shape loss base-aware**: Penalty = `max(0, residual_shape_score - base_shape_score)`. Only penalize if residual makes things worse than base.
  4. **Apply different weights by CV type**: CGG and CBG highest priority, CTG medium, VDS lowest.
**Evidence**:
  - 05-22 TURN 92-93: "CGG 的 double hump 基本不是器件物理，而是模型先验造成的假结构...loss 里没有明确写出双栅 TFT 各种 CV 曲线应有的形状先验" (05-22 diagnostic.md:2279, 2286)
  - 05-22 TURN 93: Shape template design — "CGG/CBG/CTG 的基本形状不是'斜率一直为正'，而是低平台、一个主要转变区、高平台" (05-22 diagnostic.md:2291)
  - 05-22 TURN 93: After template loss — "新的模板 loss 把模型从'追求 CV RMS'推向'更保守的平台型 CV'" (05-22 diagnostic.md:2300)
**Key Parameters**:
  - `cv_template_shape_weight`: 0.15-0.25 (relative to main CV value loss).
  - `cv_plateau_fraction`: 0.25 (fraction of sweep at top/bottom as plateau).
  - `cv_unimodal_penalty_weight`: 0.1 (penalty per additional derivative peak beyond the first).
  - `cv_high_plateau_rise_threshold`: 0.01 (max allowed rise in ON plateau as fraction of total span).
**Success Indicator**: CV curves show exactly 1 derivative peak per curve. ON plateau flat within 1-2% of total span. No double hump visible in CGG or CBG.
**Pitfalls**: Do NOT apply strict unimodality to VDS CV — drain bias can produce legitimate shoulder structures. Accept a 0.01-0.03 dec RMS increase if it eliminates visible artifacts. The "no continued rise" constraint should be relaxed for devices with significant trap-related frequency dispersion.

---

### Data-Driven CV Width Initialization for Base PKAN Transition Parameter Finetuning
**Category**: Tuning Pattern
**Stage(s)**: train-static (CV physical kernel initialization, base PKAN warm-up)
**Context**: The pre-trained base PKAN checkpoint contains CV physical parameters (`nfac`, `Cit_t/b`, `trap_width`) calibrated for a reference device. When training a new device with different transition characteristics, base PKAN finetuning must move these from their reference values. With standard warm-start multipliers (0.035), the physical parameters barely move in the first 100-200 epochs. The residual KAN compensates with non-physical corrections.
**Pattern**:
  1. **Extract measured CV transition width**: For CGG 10kHz and mid-bias CBG/CTG, compute 20%-80% transition span.
  2. **Compare with base model transition width**: Ratio `R = width_measured / width_base`.
  3. **Initialize PKAN transition parameters**: Scale `nfac_cv`, `Cit_t`, `Cit_b`, `trap_width` by `R`: `nfac_init = nfac_base * R`, etc.
  4. **Use elevated LR for transition params in warm-up**: Separate parameter group with 5-10x higher LR during first 50 epochs.
  5. **Freeze after warm-up** (~100 epochs), let residual handle remainder.
**Evidence**:
  - 05-22 TURN 106: "当前短训里 nfac/Cit/trap_width 移动太慢" (05-22 diagnostic.md:2576)
  - 05-22 TURN 106: "从实测 20%-80% 过渡宽度估计...这比特殊处理某条曲线更通用" (05-22 diagnostic.md:2576)
  - 05-22 TURN 107: After initialization — "nfac/Cit/trap_width 已经从一开始被放大，不再只是百分级微调" (05-22 diagnostic.md:2638)
  - 05-22 TURN 48: User instruction — "CV 的 nfac/Cit/trap_width 可能不止要微调，可能需要大调" (05-22 diagnostic.md:2550)
**Key Parameters**: `cv_base_pkan_warmup_epochs=100`, `cv_transition_lr_mult=5.0-10.0`, `cv_transition_range=[0.3, 3.0]`.
**Success Indicator**: Base CV 20%-80% transition width after warm-up within +/-30% of measured width. Base CV RMS decreases >20% during warm-up. Residual corrections remain small.
**Pitfalls**: Only apply to transition-width-controlling parameters (`nfac_cv`, `Cit_t/b`, `trap_width`), not threshold position parameters. If ratio `R` is >5 or <0.2, consider fallback to fit-physical instead.

---

### CBG/CTG Data Label Swap Auto-Detection via Cox Ratio vs CV Span Ratio
**Category**: Tuning Pattern
**Stage(s)**: train-static (CV data preprocessing)
**Context**: The model consistently shows CBG capacitance span >> CTG span, but the top gate has higher Cox and should show LARGER capacitance span. Visual inspection confirms: curves labeled "CBG" actually look like CTG curves and vice versa — the dataset labels are swapped.
**Pattern**:
  1. **Compute expected Cox ratio**: From physical kernel: `R_Cox = Cox_t / Cox_b`.
  2. **Compute measured CV spans**: `span_cbg = max(C_cbg) - min(C_cbg)`, `span_ctg = max(C_ctg) - min(C_ctg)`. Ratio `R_span = span_cbg / span_ctg`.
  3. **Detect swap**: If `(R_Cox > 1 and R_span < 1)` or `(R_Cox < 1 and R_span > 1)`, labels are swapped.
  4. **Auto-correct**: Exchange physical interpretation of CBG and CTG data — "CBG" data assigned to CTG model, "CTG" data assigned to CBG model. Swap swept voltage axis, gate weight matrix, and mode assignment.
  5. **Report**: Write `cv_gate_orientation.json` documenting whether auto-swap was applied.
**Evidence**:
  - 05-22 TURN 109: "Cox_t/Cox_b = 2.92，但原始 CBG_span/CTG_span = 3.27，这和标准双栅 CV 的端口电容大小预期相反" (05-22 diagnostic.md:2663)
  - 05-22 TURN 109: After auto-swap — "CV RMS 从约 0.13 dec 直接降到 0.0745 dec" (05-22 diagnostic.md:2663)
  - 05-22 TURN 109: User suspicion confirmed — "有可能是我标反了，我这个器件底栅栅绝缘层电容小，顶栅绝缘层电容大" (05-22 diagnostic.md:2653)
  - 05-22 TURN 109: After fix — "CBG/CTG 的物理量级也更符合'顶栅 Cox 大'的事实" (05-22 diagnostic.md:2671)
**Key Parameters**: `cv_auto_orientation=True` (default), `cv_orientation_span_threshold=2.0` (minimum discrepancy factor to trigger).
**Success Indicator**: After swap: CBG span ~ Cox_b*W*L, CTG span ~ Cox_t*W*L. Physical kernel CV prediction error decreases >30%. Report clearly documents swap decision.
**Pitfalls**: Do NOT apply auto-swap if both Cox ratios are near 1.0 (symmetric gate). Span comparison must subtract OFF plateau baseline first. If discrepancy factor <2.0, emit warning but do NOT auto-swap.

---

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
