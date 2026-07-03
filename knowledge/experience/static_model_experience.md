# Static Model Experience (from 2026-06-29 session)

## Principle Insights

### KAN Grid Knot Non-Monotonicity: Frozen Grids Becoming Trainable
**Category**: Principle Insight
**Stage(s)**: train-static (affects all downstream stages that rely on trained PKAN checkpoints)
**Symptom**: IV curves show sharp stair-step discontinuities in the IDVD saturation region and abrupt kinks in IDVG/IDVBG high-bias regions. The artifact is visible in `results/plot/*.png` as discrete jumps, not smooth interpolation. IDVD max local 2nd-order jump reaches ~0.12 dex; IDVTG reaches ~0.07 dex.
**Root Cause**: The independent base IV-PKAN training phase calls `base.kan.parameters()` with `requires_grad=True` for all parameters, which inadvertently unfreezes the pykan `grid` and `mask` tensors (designed to be frozen in v96 architecture). AdamW then modifies the B-spline knot positions. Since B-splines require strictly monotonic knot grids, any inversion (e.g., VDS grid `…, -0.8332, -0.9464, …` with descending order, or qbar grid reverse-sorted) causes certain spline edges to abruptly deactivate/reactivate as the input voltage sweeps, producing visible stair-steps.
**Evidence**:
  - TURN 26: "B-spline knot grid 出现了非单调节点（例如 VDS 网格 … -0.8332, -0.9464 …，qbar 网格也倒序）" (diagnostic.md:918)
  - TURN 26: "独立 IV base-PKAN 阶段把 base.kan.parameters() 全部重新设为可训练，误把 pykan 原本冻结的 grid、mask、node metadata 也交给 Adam" (diagnostic.md:923)
  - TURN 28: Fix — after monotonicity restoration, "IDVD 最大局部二阶跳变从 0.12175 dex 降到 0.001244 dex（下降 98.98%），IDVTG 从 0.07071 dex 降到 0.002966 dex（下降 95.81%）" (diagnostic.md:948)
  - TURN 28: v95 checkpoint had 9 inverted knot positions in base IV KAN + 4 in IV residual KAN; v90 grids were all strictly monotonic (diagnostic.md:923)
**Generalization**: This is a code-level bug (not dataset-specific). Any fresh device training that runs `train_current_device_base_iv_pkan` with the v96 frozen-grid architecture will trigger this if the optimizer's parameter group includes grid tensors. It is invisible to the precision gates (RMS may stay reasonable) because the model can still achieve low aggregate error via compensation, but the resulting IV curves have physically implausible discontinuities.
**Actionable Advice**:
  1. **Prevention (code fix)**: When constructing AdamW optimizer groups for KAN training, explicitly exclude `grid`, `mask`, and `node_*` named parameters. Only `fc.weight`, `activation_fn.*`, and `symbolic.*` should be trainable.
  2. **Detection (post-training)**: After base IV-PKAN training, verify all KAN grids are strictly monotonic via `assert_valid_grids()` (already exists in engine as `grid audit`). If the gate passes but visual inspection shows jumps, run a supplementary check: compute max 2nd-order finite difference across each IV curve trace; any local jump >0.01 dex indicates knot damage even if the aggregate audit passes.
  3. **Recovery**: Grid monotonicity failure is NOT recoverable by parameter tuning — the KAN model state is corrupted (`knowledge/static_model.md:564` row "grid audit failed"). Fallback to fit-physical and restart training with the code fix applied. Do NOT attempt to "reproject" or "repair" grids via post-processing — this only masks the issue and keeps degraded edge representations.
  4. **What NOT to do**: Do not apply this fix as a post-hoc grid projection on an already-trained checkpoint and then re-verify. Evidence: the projection approach kept IV nonzero RMS at 0.080191 dex (vs 0.079164 pre-fix, +0.00103 dex) and only masked the grid corruption without fixing root cause (diagnostic.md:948).

---

### Training Order: Independent IV Base → IV Residual → Frozen IV → CV
**Category**: Principle Insight
**Stage(s)**: train-static (also informs Agent D for train-hysteresis: never joint-train with static IV)
**Symptom**: IV degradation after joint IV+CV training completes. Specifically: `residual_iv_rms` increases after the joint phase compared to the base IV PKAN phase, even though the loss formula appears unchanged. The CV-dominated global checkpoint selector picks epochs based on a weighted metric where CV_weight=10 dominates, selecting epochs where IV has drifted from its optimum. Additionally, `residual_cv_rms` shows CV residual with negative net contribution (e.g., CV RMS goes from 0.02073 base to 0.020995 after adding charge residual).
**Root Cause**: Two interacting mechanisms:
  1. **Optimizer competition**: CV is a derivative observable (dQ/dV) — its loss terms involve second-order autograd through the charge model. The gradient magnitudes from CV loss can be 10-100× larger than IV loss gradients (even with cv_weight=1), overwhelming the shared parameter updates and pulling the base IV-PKAN away from its pre-trained optimal point.
  2. **Checkpoint selection bias**: The joint training checkpoint selector uses a combined metric with `cv_weight=10.0`. Epochs where CV improves by 0.001 but IV degrades by 0.005 are incorrectly ranked higher than epochs with good IV and adequate CV.
**Evidence**:
  - TURN 7: "退化主要来自当前把 IV residual 与高权重 CV 联合训练、再用 CV 主导的全局 checkpoint 选择" (diagnostic.md:186)
  - TURN 7: v90 vs current parameter comparison: "v90: 700 epoch, LR=0.002, linear weight=0.25, 36 derivative curves, shape period=3. Current: 160 epoch, LR=0.0012, linear weight=0.34, 24 curves, shape period=5" (diagnostic.md:192)
  - TURN 8: After restoring independent IV base→residual with v90 parameters: held-out balanced score improved from 0.09918 (base) to 0.09097 at epoch 180, and to 0.08796 at epoch 340 — an 11.3% held-out improvement (diagnostic.md:236-241)
  - TURN 8: "冻结 IV 后，CV base RMS 0.02073，加入 charge residual 后为 0.020995，出现退化" (diagnostic.md:266)
  - TURN 15: User explicit instruction: "服用v90 iv训练的流程，损失函数设计，拟合顺序，和符号化的流程！而cv用现在的" (diagnostic.md:483)
**Generalization**: The staged isolation principle applies universally: for any device where the model separates into IV and CV prediction branches sharing a physical kernel, gradient competition from CV (derivative observable) will dominate IV (direct observable) unless CV is trained only after IV is frozen. This is physics-rooted: CV ≡ dQ/dV requires second-order derivatives through the EKV charge model, creating loss surfaces with steep ridges that IV gradients cannot navigate.
**Actionable Advice**:
  1. **Correct training order**: `fit-physical` → `train_current_device_base_iv_pkan` (independent, 700 epoch, LR=0.002) → `train_current_device_iv_residual` (independent, 700 epoch, LR=0.002, held-out split) → **freeze entire IV branch** (base PKAN weights + phys_delta parameters + IV residual KAN) → train CV/charge branch (base CV-PKAN + CV charge residual + CV platform/sheet params).
  2. **Parameter values for IV base phase**: `base_iv_epochs=700`, `base_iv_learning_rate=0.002`, `base_iv_restarts=1` (with data-driven multiple initialization if budget allows). Evidence: v90's 700-epoch base consistently achieved nonzero RMS ~0.09-0.10 dex range; 240 epochs were sufficient but 700 provided margin (diagnostic.md:224-226).
  3. **Parameter values for IV residual phase**: `iv_residual_stage_epochs=700`, `iv_residual_stage_learning_rate=0.002`, `iv_residual_stage_linear_weight=0.25`, `iv_residual_stage_deriv_curves=36`, `iv_residual_stage_shape_period=3`. Evidence: these v90 parameters produced consistent held-out gains across multiple training runs (diagnostic.md:192, 256).
  4. **Freezing strategy**: Set `finetune_base_pkan=False` (already v96 default), AND additionally ensure that `--freeze-iv-residual-after-stage` is passed (not `--joint-finetune-iv-residual`). The CV stage should train only: CV PKAN, CV charge residual, CV platform, CV sheet calibration, temperature KAN (if enabled).
  5. **Checkpoint selection during CV-only phase**: Use a single-objective selector (lowest CV RMS on training data with passivity check) — do NOT include IV metrics since IV is frozen. A combined metric with cv_weight>1 will still produce biased selection.
  6. **What NOT to do**: Do not use a combined IV+CV checkpoint selector during CV training — even if IV is nominally frozen, the selector's tie-breaking will prefer CV-optimal epochs that may show slightly different IV due to random batching noise. Do not set `iv_residual_stage_linear_weight` to 0.34 (the drifted default) — evidence shows 0.25 (v90) produced better held-out gains (diagnostic.md:192).

---

### Symbolic Distillation: Edge-Local MSE Does Not Guarantee Terminal Accuracy
**Category**: Principle Insight
**Stage(s)**: train-static (symbolic phase in Phase 6 of engine execution)
**Symptom**: The Top-1 symbolic edge compatibility gate (`base_iv_max_top1_compatibility=0.015`) passes — all individual KAN edges are replaced by analytic primitives with edge-level MSE <0.015. However, when the full symbolic model predicts IV current on held-out measured curves, the terminal-level (observable) RMS error is dramatically higher: `0.122 dex` vs the numeric model's `0.082 dex`, and on-region RMS degrades from `0.0287 dex` to `0.1133 dex`. The Python→HSPICE grid verification also shows large errors (e.g., 12.94% max relative error) that appear to be a "deployment implementation error" but are actually the cumulative effect of edge-level symbolic approximation.
**Root Cause**: Three compounding factors:
  1. **Error propagation through physics**: The 6 input features to the KAN (`Ut_V, Ub_V, Veff_V, VDS_V, qbar0, qbar0×VDS`) are not independent — they are all functions of the same raw terminal voltages (`Vtg, Vbg, Vds`). A small edge approximation error on, e.g., the `Veff_V → dVth_t_eff_V` edge propagates into the output parameter correction, which then feeds into the EKV drain current equation involving exponential and power-law terms. The resulting current error is amplified by orders of magnitude compared to the edge-local MSE.
  2. **Covariance ignored**: The edges that produce the 11 IV parameter corrections (`dpsi_V, dVth_t_eff_V, dVth_b_eff_V, ...`) are replaced independently. If edge A systematically overestimates by +0.005 and edge B by +0.005, their errors cancel in edge space (both look fine), but add constructively in current space (both bias the same physical parameter direction).
  3. **Training constraint mismatch**: The `base_iv_top1_symbolic_weight` regularizer pushes KAN edges toward Top-1 primitives *during numeric training*, but measures success as edge-local compatibility. The terminal current quality is never directly constrained by the symbolic regularization loss — only edge MSE matters. So the optimizer can find numeric edge shapes that have low edge-Top-1-MSE but poor terminal current performance after symbolic replacement.
**Evidence**:
  - TURN 5: "Top-1 基础 PKAN 的内部 spline 已学得过于复杂...校准后最大误差反而为 26.3%" (diagnostic.md:135)
  - TURN 10: "数值 KAN 的单边误差并不大，但多个相关 edge 经过逐边替换后在终端电流上累积" (diagnostic.md:409)
  - TURN 12: "训练时的 top-1 约束主要约束 latent edge，而不是符号化后的整条 IV；同时 Ut/Ub/Veff/qbar0 高度相关，使逐边最优不等于终端最优" (diagnostic.md:409)
  - TURN 19: "Top‑1 student 相对数值 teacher 的全曲线等效误差是 0.122 dex，开态 RMS 从 0.0287 退化到 0.1133 dex" (diagnostic.md:570)
  - TURN 6: "fresh base 的 top-1 边兼容损失从物理初始化的 0.00309 上升到 0.01856，说明数值训练把样条边推向了难以用单一解析基函数表示的形状" (diagnostic.md:146)
**Generalization**: This gap between edge-local and terminal-level accuracy is inherent to KAN-based symbolic distillation for physics models where KAN outputs are intermediate quantities (parameter corrections) that feed into non-linear physics equations. The gap widens when (a) multiple KAN edges contribute to the same output correction, (b) the downstream physics involves exponentials or power laws, and (c) input features are highly correlated.
**Actionable Advice**:
  1. **Do NOT use edge-local MSE as the sole symbolic gate**. Keep `base_iv_max_top1_compatibility` for structural edge validation only. The primary symbolic acceptance gate MUST be terminal-level: compare symbolic model predictions against numeric model predictions on the full 35,220-point measured IV dataset, broken down by region (non-zero, on-region, subthreshold, transition, off, high-current).
  2. **Add terminal-level terms to the symbolic fine-tuning loss**. After edgewise replacement, the `network_finetune_steps` and `device_iv_finetune_steps` must include: (a) teacher-manifold log-RMS on all IV points, (b) on-region p95 relative error (not just log-RMS), (c) per-curve worst-case error. Evidence: adding terminal on-region p95 to the loss improved from 0.266 to 0.243 dex (diagnostic.md:590-591).
  3. **Validation-set best-state rollback during symbolic finetuning**: The measured finetune stage must save and restore the best validation-set state (not the last training step). Evidence: "后续 measured finetune 没有验证集最佳状态回滚" caused overfitting to training points while worsening held-out curves (diagnostic.md:572).
  4. **Calibration amplitude bounds**: Default calibration sigmoid amplitude bound of ±0.16 dex may be too restrictive for devices where the symbolic→numeric gap is large. Relax to ±0.24 dex for the smooth terminal calibration layer ONLY (not for edge-level symbolic formulas). Evidence: a key sigmoid term hit the ±0.16 bound, preventing necessary correction (diagnostic.md:591). But beware: relaxing too far (to unrestricted) triggers complexity selection to drop too many terms, causing full-curve RMS regression (diagnostic.md:596).
  5. **Remove measured-only finetune phase from default pipeline**: The extra 1600-step post-symbolic measured-only finetune (not present in v90) was the cause of the HSPICE error increase from 0.956% to 4.886%. Evidence: removing it and using only the v90 sequence (edge+network+device finetune, then calibration) reduced HSPICE IV max error from 4.886% to 0.956% (diagnostic.md:522-523). The parameter `device_iv_measured_steps` should be set to 0.
  6. **Layer error decomposition in reports**: The final manifest should report three separate layers: (a) numeric KAN vs measured data, (b) symbolic model vs numeric KAN, (c) HSPICE vs Python symbolic model. Evidence: previously HSPICE verification reported combined (b)+(c) error, misleadingly suggesting large "deployment errors" when the real issue was (b) (diagnostic.md:746).

---

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

### HSPICE Validation Comparing Against Wrong Python Reference Model
**Category**: Principle Insight
**Stage(s)**: train-static (Phase 6 — HSPICE validation)
**Symptom**: HSPICE validation reports large "deployment implementation errors" — IV max relative error 12.94% (vs 3% gate), log RMSE 0.03199 dex — suggesting the Verilog-A export or HSPICE simulation has a bug. Subsequent investigation shows the Verilog-A is actually correct; the validation function was comparing HSPICE output against the *numeric spline KAN model* instead of the *Python symbolic model* (which is what the Verilog-A implements). CV shows the same pattern but with smaller magnitude (smaller errors but same systematic bias).
**Root Cause**: The HSPICE validation function's reference computation (`python_ids`) was calling the numeric KAN's `forward()` method (spline-based), not the symbolic KAN's (analytic formula-based). Since the symbolic model has a ~0.04-0.12 dex discrepancy from the numeric model (see Principle Insight on edge-local vs terminal accuracy), this discrepancy was incorrectly attributed to HSPICE/Verilog-A implementation error. The actual Python-symbolic to HSPICE error is far smaller: 0.564% max relative IV error (vs the misreported 12.94%), 0.0935% median, 0.00477 dex max log error.
**Evidence**:
  - TURN 22: "HSPICE 导出的是符号化 KAN，但验证函数的所谓 python_ids 实际仍在调用数值 spline KAN，因此报告混入了'数值→符号化'误差，并不是真正的'Python 符号表达式→HSPICE'误差" (diagnostic.md:746)
  - TURN 22: After fix — "IV 的 Python 符号模型→HSPICE 最大相对误差从原报告的 2.42% 降为 0.564%，中位数从 0.534% 降为 0.0935%，最大对数误差仅 0.00477 dex" (diagnostic.md:766)
  - TURN 5: Early misdiagnosis — "HSPICE 检查已暴露...符号化 IV 在 13 点网格上的最大 Python–HSPICE 相对误差为 12.94%" led to incorrect conclusion that symbolic export was broken and triggered wasteful re-training (diagnostic.md:88)
  - TURN 24: Final corrected result — "Python 符号模型到 HSPICE 的 IV 网格误差由 0.00951 dex 进一步降到 0.00208 dex" (diagnostic.md:825)
**Generalization**: This is a validation pipeline bug, not a modeling issue. Any pipeline that generates both numeric (spline) and symbolic (analytic) model variants and then validates HSPICE against Python must ensure the reference model matches the exported format. The bug persists silently because both variants produce similar (not identical) outputs — the error is small enough to not trigger NaN/inf but large enough to cause false HSPICE validation failures.
**Actionable Advice**:
  1. **Code fix**: The HSPICE validation function must construct its Python reference model from the same symbolic network that was exported to Verilog-A, not from the original numeric KAN. Specifically, after symbolic export, the reference model should use `SymbolicKANNet.forward()` (or the post-calibration wrapped symbolic network), not `CompactKAN.forward()`.
  2. **Detection**: If HSPICE validation consistently fails with errors in the 2-15% range while the Python numeric→measured error is <5%, suspect this reference mismatch. A quick check: call the exported symbolic model on the same HSPICE grid points and compare against HSPICE output; this should show errors <1% if the Verilog-A is correct.
  3. **Reporting**: All reports must decompose the error into three layers and label each: (a) numeric KAN vs measured (training error), (b) symbolic model vs numeric KAN (compression error), (c) HSPICE vs symbolic Python model (implementation error). Evidence: the session spent multiple turns debugging a false "HSPICE bug" because layer (b) was conflated with layer (c) (diagnostic.md:88, 746, 766).
  4. **What NOT to do**: Do not increase `base_iv_max_top1_compatibility`, `edge_steps`, `network_finetune_steps`, or `device_iv_finetune_steps` in response to inflated HSPICE validation errors — first verify the reference model is correct. Evidence: the early session turns wasted complete training budgets on fixing a "symbolic accuracy" problem that was actually a validation reference problem.

---

## Tuning Patterns

### Residual Safety Scan: Absolute-Zero Requirement Blocking Valid Deployments
**Category**: Tuning Pattern
**Stage(s)**: train-static (Phase 5 — Safety Evaluation, specifically `choose_safe_residual_scales`)
**Context**: Agent C observes that `safe_iv_residual_scale=0` in the metrics JSON (residual is disabled), but held-out IV residual training showed real improvement (e.g., balanced score improved from 0.09918 to 0.08796). The base model passes all numerical gates, but the residual is rejected. The failure log shows "extrapolation warning count > 0" or similar absolute-zero safety condition.
**Pattern**:
  1. **Check whether base already has extrapolation warnings**: Read `v92_safe_shape_diagnostics.csv` or the `base_iv_training` block in the metrics JSON. The column `ext_bad` or equivalent counts non-monotonic/extrapolation violations. If base > 0, the absolute-zero requirement will reject any non-zero residual.
  2. **Understand the gate logic**: The `choose_safe_residual_scales` function iterates residual scales from 0.0 to 2.0. At each scale, it checks shape quality and passivity on a safety holdout split. The original logic required `ext_bad == 0` (ABSOLUTE zero extrapolation alerts) for acceptance — but this is wrong when the base model itself already has extrapolation alerts. Any residual adjustment, even beneficial, will inherit the base alerts.
  3. **Apply the relative-degradation fix**: Change the acceptance criterion from absolute-zero (`ext_bad == 0`) to relative-non-degradation (`candidate_ext_bad <= base_ext_bad`). The residual is accepted if: (a) it does not introduce NEW extrapolation alerts beyond the base count, AND (b) held-out balanced score improves, AND (c) no individual region degrades by more than `iv_residual_max_region_degradation`.
  4. **Re-run safety scan with fixed gate**: The scan should now select a non-zero residual scale (typically 0.5 or 1.0) if the residual provides genuine held-out improvement without new shape issues.
  5. **Validate that the selected scale actually helps**: After acceptance, verify that the residual at the selected scale produces held-out gain AND the shape diagnostics show no increase in `ext_bad` count.
**Key Parameters**:
  - `iv_residual_stage_epochs`: If residual training shows consistent held-out gain but safety scan rejects it, increase this (e.g., 700→1000) before re-running.
  - `iv_residual_max_region_degradation` (engine-internal, default 0.008): If tightened too much (e.g., 0.001), even beneficial residuals will fail. Keep at 0.008.
  - `safe_scale_steps`: Default 8 is sufficient. Do not increase — the scan is expensive.
**Success Indicator**: 
  - `safe_iv_residual_scale` is non-zero (e.g., 0.5 or 1.0) after safety scan.
  - Balance score in `iv_residual_training.heldout_balanced_score` shows improvement vs base.
  - Shape diagnostics show no NEW extrapolation alerts beyond base count.
  - CV passivity (charge sum, cross-reciprocity) passes at the selected scale.
**Pitfalls**:
  - Do NOT set `safe_iv_residual_scale` manually or force it to 1.0. The safety scan exists because non-zero residuals at large scales can introduce non-physical IV curve shapes (negative resistance, non-monotonic gm slopes). Always go through the scan.
  - If the base model has many (>10) extrapolation alerts, the relative-criterion fix may still accept a residual that makes things marginally worse. In this case, investigate whether the base training itself has issues (insufficient epochs, poor physical kernel).
  - Do not confuse "safety scan rejected residual" with "residual training failed". The residual training may have produced a good model, but the safety scan found deployment risks at certain scales. The fix is the scan logic, not the residual training.

**Evidence**:
  - TURN 19: "最终 safety scan 要求'外推绝对零告警'；base 自身已有 4 个外推形状告警，因此所有非零 residual 都被拒绝。这是门控逻辑错误" (diagnostic.md:612)
  - TURN 20: "基线外推本身有 4 个形状告警，旧逻辑却要求 residual 候选 ext_bad == 0，因此任何非零 residual 都不可能入选" (diagnostic.md:622)
  - TURN 20: After fix — "改为'候选告警数不高于基线，且不新增相对异常'" (diagnostic.md:622)
  - TURN 20: Final result — "安全扫描结果已明确选中 IV=0.5、CV=0.5，说明 residual 在真实全曲线收益和不新增异常条件下已恢复贡献" (diagnostic.md:632)

---

### CV Finalize Polish Silently Degrading Accepted Models
**Category**: Tuning Pattern
**Stage(s)**: train-static (Phase 4 — Post-Training CV Polish and Phase 6 finalize)
**Context**: Agent C has an accepted static model that passes all numerical gates. The pipeline then runs `finalize` to produce the deployment artifacts (symbolic export, HSPICE, checkpoint save, VA export). After finalize, the agent observes that `residual_cv_rms` has increased (e.g., from 0.020324 to 0.020383), even though no training should have occurred. The perturbation is small but violates the "no regression" principle.
**Pattern**:
  1. **Identify that polish is running during finalize**: Check the engine logs for lines containing `polish_cv_*`, `finalize polish`, or `cv polish`. If the `finalize` subcommand is being passed (or is the default action), the post-training CV polish stages may execute even with `--resume` or `--finalize-only` flags.
  2. **Disable all polish stages during finalize**: Add `--skip-post-train-cv-polish` to the engine call whenever running in `finalize` or `resume` mode. The polish stages (`polish_cv_neural_corrections`, `polish_cv_charge_residual`, `polish_cv_level2_intergate`, `polish_cv_held_gate_coupling`) are designed to run as part of the initial training pipeline, not as re-entrant correction passes. They use different optimizer settings (higher LR, shorter epochs) that can perturb settled weights.
  3. **Verify that the finalize path does not also trigger warm-start CV prefit**: When `resume` mode is used with a checkpoint, the engine may re-execute CV warm-start prefit (CV platform parameters, sheet calibration, reciprocal charge mapping) before loading the checkpoint state_dict. These prefits are then overwritten by the checkpoint, wasting time and creating a window where the model briefly exists in an intermediate state.
  4. **Check the `finalize` subcommand defaults**: In at least one version, the default `finalize` behavior included polish stages. The fix is to explicitly pass `--skip-post-train-cv-polish` and ensure the code path checks for resume mode before running prefits.
**Key Parameters**:
  - `cv_residual_polish_epochs` (engine-internal): If polish cannot be skipped (legacy code requirement), set this to 0. Evidence shows polish with even 1 epoch can perturb CV RMS.
  - `cv_neural_polish_epochs`, `cv_level2_polish_epochs` (engine-internal): Same — set to 0 if a polish pass is unavoidable.
  - Pipeline-level flag: Ensure `--no-finetune-base-pkan` and `--freeze-iv-residual-after-stage` are passed during finalize to further constrain what can be modified.
**Success Indicator**:
  - `residual_cv_rms` after finalize matches the pre-finalize value to 6+ decimal places.
  - Engine log contains no `polish_cv_*` lines between checkpoint load and artifact save during finalize.
  - The `v93_safe_residual_metrics.json` written by finalize has identical `residual_iv_rms` and `residual_cv_rms` to the pre-finalize metrics.
**Pitfalls**:
  - Do not simply accept the degradation as "small" (e.g., 0.020324→0.020383). This accumulates across multiple pipeline stages. If the user has a CV RMS hard gate of 0.020, a 0.000059 degradation could be the difference between pass and fail.
  - The polish stages do NOT improve CV quality when applied to an already-converged model — they are designed to bootstrap a pre-convergence model. Running them on a converged checkpoint can only perturb.

**Evidence**:
  - TURN 5: "finalize-only 的默认值仍会启动 CV polish，这会改动已验收 checkpoint，而且此前实测该 polish 使 CV 从 0.020324 退化到 0.020383" (diagnostic.md:120)
  - TURN 6: "直接入口的 finalize polish 默认仍会额外训练，和配置层预期不一致" (diagnostic.md:156)
  - TURN 5: "第二个无效步骤也已定位：finalize-only 的默认值仍会启动 CV polish...现在禁止所有 finalize polish，确保终验只读重载并评估" (diagnostic.md:119-120)

---

### CV Passivity Scan: Coarse Grid Accepting Residual, Fine Grid Rejecting → Residual Zeroed
**Category**: Tuning Pattern
**Stage(s)**: train-static (Phase 5 — Joint CV safe scale search)
**Context**: Agent C observes that `safe_cv_residual_scale=0` despite `residual_cv_rms` appearing reasonable in intermediate training logs. The scan report shows that a coarse pass at 48 charge-passivity checkpoints accepted `cv_residual_scale=1.0`, but the final report has `selected_cv_residual_mode="off"` and `safe_cv_residual_scale=0.0`. The charge conservation / passivity check on the full 96-point grid found violations that the coarse grid missed.
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

### v90 Symbolic Pipeline: Key Default Parameter Differences That Cause Symbolic Regression
**Category**: Tuning Pattern
**Stage(s)**: train-static (base IV PKAN Phase 1, IV residual Phase 2, symbolic Phase 6)
**Context**: Agent C observes that numerical IV results are good (base RMS ~0.08 dex, residual held-out gain confirmed) but symbolic export fails with large HSPICE errors or terminal accuracy gaps. The training loss formulas and edge-level symbolic library appear identical to a known-good baseline (v90), yet results are worse. Agent C has already verified that the reference model for HSPICE validation is correct (see tuning pattern on HSPICE reference mismatch).
**Pattern**:
  1. **Check the three drifted defaults**: Compare the current `[full_model]` INI values against the v90 baseline for three critical parameters:
     - `iv_residual_stage_linear_weight`: v90=0.25, drifted to 0.34
     - `iv_residual_stage_deriv_curves`: v90=36, drifted to 24
     - `iv_residual_stage_shape_period`: v90=3, drifted to 5
  2. **Check for extra symbolic phases**: v90's symbolic pipeline was: (a) edgewise replacement (180 steps), (b) edge finetune (80 steps), (c) network finetune (360 steps), (d) device IV finetune (3000 steps). Later versions added a 5th phase: (e) `device_iv_measured_steps=5000` — post-symbolic measured-data finetune. This phase is the single largest contributor to symbolic degradation.
  3. **Restore v90 parameter values**: Set `iv_residual_stage_linear_weight=0.25`, `iv_residual_stage_deriv_curves=36`, `iv_residual_stage_shape_period=3`. Set `device_iv_measured_steps=0` to disable the extra measured finetune.
  4. **Verify symbolic calibration budget**: v90 used `device_iv_finetune_steps=3000` (not 2200) and `edge_steps=180`, `network_finetune_steps=360`. Ensure these are not reduced. The symbolic budget parameters (`edge_steps`, `network_finetune_steps`, `device_iv_finetune_steps`) were reduced to ~1/4-1/6 of v90 in some configurations — this directly caused the edge-local MSE increase.
  5. **Check iv_adaptive_top2_mse**: For strict Top-1 deployment, set `iv_adaptive_top2_mse=0.0` and `iv_edge_topk=1`. For adaptive Top-2 fallback, use `iv_adaptive_top2_mse=0.0015` with the constraint that Top-2 must improve edge MSE by >10% and is limited to edges exceeding the threshold.
  6. **Re-run symbolic from existing numeric checkpoint**: If the numeric model is already good, re-run ONLY the symbolic phase (edge replacement + network finetune + device finetune + calibration) with restored parameters. Do not re-train the numeric model.
**Key Parameters**:
  - `iv_residual_stage_linear_weight`: 0.25 (restored from v90). Higher values (0.34) over-penalize the on-region, causing the residual to sacrifice shape quality in the transition region. This doesn't show in aggregate RMS but degrades symbolic compressibility.
  - `iv_residual_stage_deriv_curves`: 36 (restored from v90). The reduced 24 curves omit VDS=0.1V and VDS=0.5V sheets that capture the transition from linear to saturation. Fewer derivative supervision curves → residual KAN learns less physically constrained shapes → harder to Symbolic-top-1.
  - `iv_residual_stage_shape_period`: 3 (restored from v90). Every 3 epochs (vs 5) means shape monitoring runs ~230 times in 700 epochs instead of ~140. More frequent shape evaluation catches and penalizes non-monotonic edges before they become entrenched.
  - `device_iv_measured_steps`: 0 (disable the extra phase).
  - `device_iv_finetune_steps`: 3000 (v90 value, not reduced).
**Success Indicator**:
  - After symbolic re-export with restored parameters: HSPICE IV max relative error <3% (v90-compatible gate) or ideally <1% (post-fix reference model).
  - Terminal symbolic RMS and p95 pass the full-curve gate.
  - The symbolic model is strictly Top-1 for IV/CV/temperature (0 edges upgraded to Top-2 if `iv_adaptive_top2_mse=0.0`).
**Pitfalls**:
  - Do NOT copy v90's `warm_start_base_pkan_lr_mult=0.035` to the restored config — this applies only to warm-start scenarios (see Principle Insight on native vs warm-start LR). Fresh-start must use 1.0 or the base IV PKAN will not learn.
  - Do NOT reduce `device_iv_finetune_steps` below 3000 even if it "seems high". The session evidence shows that reducing to 2200 was one cause of the symbolic regression (diagnostic.md:93).
  - Do NOT increase `iv_adaptive_top2_mse` as a first resort. The v90 evidence shows Top-1 symbolic export with correct parameters achieves <1% HSPICE error. Only enable adaptive Top-2 if (a) numeric base IV RMS is <0.10 dex, (b) all v90 parameters are restored, (c) Top-1 still fails terminal gate, AND (d) edge-level MSE diagnostics show specific edges >0.015.

**Evidence**:
  - TURN 17: "v94 的 Top-1 候选函数库、edge exhaustive 拟合、180/80/360/2200 精修预算与 v90 完全一致；但我发现三个 IV 默认值被后续版本改过（线性损失 0.25→0.34、导数曲线数 36→24、符号正则周期 3→5），以及多了 v90 没有的 1600 步 measured-only 符号阶段" (diagnostic.md:519)
  - TURN 17: After parameter restoration — "v90 原始 Top-1 顺序的重新符号化已经完成，结果比之前明显改善：HSPICE IV 最大相对误差从 4.886% 降到 0.956%，CV 最大相对误差 0.0578%" (diagnostic.md:522)
  - TURN 18: "我把后续漂移的 iv_linear_weight=0.25、导数曲线数 36、symbolic period 3 恢复。唯一不应照抄的是 v90 对已有 checkpoint 的 warm-start 倍率 0.035" (diagnostic.md:534)
  - TURN 20: "Adaptive Top2 仅升级了 9 条高误差 IV 边（其余 IV 边、全部 CV/温度边仍为 Top1）" — shows adaptive Top-2 is a controlled fallback, not a wholesale relaxation (diagnostic.md:772)

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
