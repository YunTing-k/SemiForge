# Final Interpretation Experience (from 2026-06-29 session)

## Principle Insights

### Fail-Soft Output Policy: Quality Gates Must Mark, Not Block
**Category**: Principle Insight
**Stage(s)**: final-report (Agent E), all script stages with quality gates (train-static, train-hysteresis)
**Symptom**: After 15+ turns of iterative optimization, the user received no deliverable output — the agent kept refining without ever producing a package. User repeatedly demanded: "哪怕不满足门控，你也要能输出模型，而不是什么都不输出" (even if gates fail, you must output a model, not nothing).
**Root Cause**: The pipeline's quality gate logic was implemented as a hard block — if any gate failed (e.g., Top-1 symbolization measured-curve gate), no checkpoint, no Verilog-A, no ZIP was produced. Combined with an optimization loop that never converged to gate-passing state, this created a zero-output deadlock.
**Evidence**:
- Turn 15 (2026-06-30T09:36:30): User explicitly stated "到现在你一次任务闭环没完成" (you haven't completed a single task closure), demanding models be output even on gate failure.
- Turn 16: Agent confirmed the policy change — "门控失败时仍必须保存数值 checkpoint、符号候选、指标和失败原因，不能空交付" (on gate failure, must still save numerical checkpoint, symbolic candidates, metrics, and failure reasons; cannot deliver empty).
- Turn 21: "失败门控不会再导致'什么都不输出'" (failed gates will no longer cause "output nothing").
- Turn 23: Delivery package finally produced after adopting fail-soft policy, even though measured-relative gate had "轻微超限" (slight over-limit) by 0.00151–0.00672 dex.
**Generalization**: Whenever quality gates are implemented as hard blockers with no fallback artifact generation, pipelines can enter infinite refinement loops. This is especially dangerous when: (a) the optimization is non-convex (KAN training with random initialization), (b) gate thresholds are tight relative to achievable precision, or (c) multiple interdependent gates create a conjunctive failure mode where fixing one breaks another.
**Actionable Advice**:
1. Agent E should explicitly check whether artifacts (checkpoints, Verilog-A files, plots) exist in the output directory even when `quality_pass=False`. If artifacts are missing entirely, Agent E should flag this as a pipeline design defect — the stages succeeded at computation but failed at preservation.
2. Agent E should distinguish between "model that didn't train" (no checkpoint → real failure) and "model that trained well but failed a measured-curve symbolization gate" (artifact exists, gate is deployment-accuracy rather than training-accuracy).
3. Agent E should recommend a two-tier delivery model: **always save numerical checkpoints and metrics** (Tier 1), and separately gate the Verilog-A/HSPICE deployment artifacts (Tier 2). Tier 1 should never be empty regardless of gate status.

### Deployment Error Decomposition: Three-Layer vs Conflated Error Reporting
**Category**: Principle Insight
**Stage(s)**: final-report (Agent E), train-static (Agent C)
**Symptom**: HSPICE deployment errors were reported as 2.42%–12.94% maximum relative error, making it appear that the Verilog-A export was fundamentally broken. After fixing the reference model, the same HSPICE showed 0.564%–0.659% max error. Users wasted multiple optimization rounds chasing a non-existent implementation bug.
**Root Cause**: The HSPICE validation function compared HSPICE output against the **numerical spline KAN** (the teacher), not against the **Python symbolic model** (the student — which is what HSPICE actually implements). This conflated three distinct error layers into one metric:
1. **Numerical KAN → Measured data** (training quality, e.g., RMS 0.0802 dex)  
2. **Symbolic model → Numerical KAN** (symbolization loss, e.g., RMS 0.122 dex)  
3. **HSPICE → Python symbolic model** (true deployment implementation error, e.g., max 0.00477 dex)
**Evidence**:
- Turn 9 (TURN 4–5): HSPICE IV max relative error 12.94% → agent spent turns 6–12 trying to fix "deployment" when the real issue was symbolization.
- Turn 22: Agent explicitly identified the reference model mix-up: "HSPICE 导出的是符号化 KAN，但验证函数的所谓 `python_ids` 实际仍在调用数值 spline KAN，因此报告混入了'数值→符号化'误差" (HSPICE exports symbolic KAN, but the validation function's `python_ids` was actually calling numerical spline KAN, so the report mixed in numerical→symbolic error).
- Turn 22 confirmed: after fixing reference, IV Python symbolic→HSPICE max relative error dropped from 2.42% → 0.564%, median from 0.534% → 0.0935%, max log error only 0.00477 dex.
- Turn 24 final confirm: IV HSPICE max relative error 0.659%, CV max 0.0228%.
**Generalization**: When a deployment validation pipeline compares the wrong reference model, "deployment errors" become dominated by upstream symbolization/training errors. This makes debugging impossible because the symptom points to the wrong stage. The condition is: any pipeline where (a) a symbolic/compressed model is generated from a numerical teacher, (b) the compressed model is exported to an external simulator, and (c) validation compares simulator output against the teacher instead of the compressed student.
**Actionable Advice**:
1. Agent E should always decompose reported "deployment errors" into layers. If only a single "HSPICE error" metric is available, it should recommend that the pipeline report all three layers separately: numerical→measured, symbolic→numerical, HSPICE→symbolic.
2. When deployment error is large but numerical training metrics are good, Agent E should suspect reference model conflation before suspecting the Verilog-A export code.
3. Agent E should check the HSPICE validation function's reference source — if `python_ids` references the spline forward pass rather than the selected symbolic expression, the deployment error is systematically inflated.

### Method Reuse vs Artifact Reuse: Process Portability ≠ Result Portability
**Category**: Principle Insight
**Stage(s)**: final-report (Agent E), all stages with modeling agents (B/C/D)
**Symptom**: The agent spent turns 13–16 (2026-06-30T07:54–09:38) mechanically copying v90 checkpoint weights, v90 Verilog-A formulas, and v90 SRH checkpoints into the current run. User explicitly rejected this: "为什么一直在瞎搞" (why are you messing around?) and demanded process reuse, not result reuse.
**Root Cause**: The distinction between "reuse the training recipe" and "reuse the trained weights" was collapsed. The agent was optimizing for "get current device to pass gates using any means" rather than "build a portable pipeline that trains from scratch on any device." The v90 training recipe (independent IV base→residual order, full-curve/partition loss, freeze strategy, Top-1 symbolization, terminal curve calibration) was what made v90 work, but the agent kept reaching for v90's trained artifacts instead of v90's training method.
**Evidence**:
- Turn 15–16 (2026-06-30T09:36–09:38): User explicitly demanded: "你正确的应该是服用v90 iv，迟滞训练的流程，损失函数设计，拟合顺序，和符号化的流程！而cv用现在的" (you should reuse v90's IV and hysteresis training flow, loss function design, fitting order, and symbolization process; CV use the current one). User also stated: "不存在什么直接拿以前训练的结果当成现在结果的现象" (there must be no phenomenon of directly using previously trained results as current results).
- Turn 17 (2026-06-30T10:07): User reinforced: "注意你的ivkan和符号化的拟合参数和拟合顺序及拟合函数有和v90一样吗？别自己调错" (are your IV KAN and symbolization fitting parameters, fitting order, and fitting functions the same as v90? Don't tune them wrong yourself).
- Turn 17 agent commentary confirmed: v94 defaults had drifted from v90 — `iv_linear_weight` 0.25→0.34, derivative curves 36→24, symbolic period 3→5, plus an added 1600-step measured-only phase that v90 didn't have.
- Turn 17 result: After restoring v90 parameters exactly, Top-1 symbolization passed immediately (HSPICE IV max error 0.956% vs previous 4.886%), confirming the drift, not capacity, caused failure.
- Turn 22: Agent found v90's hysteresis script had hardcoded paths to v90 directory, making it non-portable despite "same code."
**Generalization**: When a pipeline is ported across versions, the training recipe (hyperparameters, stage order, loss function composition, validation strategy) often drifts due to accumulated "improvements" that were only tested on one device. The resulting failure looks like a model capacity issue but is actually a recipe regression. This pattern is especially common when: (a) training hyperparameters are spread across multiple config files, (b) default values change between versions without versioning, (c) new "features" (e.g., measured-only finetune phase) are added that weren't part of the validated recipe.
**Actionable Advice**:
1. Agent E should always compare the current pipeline's training recipe (ALL hyperparameters: epoch counts, learning rates, loss weights, stage ordering, frozen/unfrozen flags, validation strategy) against the last known-good version. A table of diff'd hyperparameters should be the FIRST diagnostic output.
2. When a model trained from scratch fails but a previous device passed with "the same code," Agent E should assume recipe drift before model capacity issues. Check: LR, epoch count, loss weights, training phase count/order, validation selection criteria.
3. Agent E should flag any hardcoded directory paths in training scripts as non-portable — these make "same code" a lie across deployments.

### KAN Spline Knot Grid Monotonicity: Silent Cross-Stage Corruption
**Category**: Principle Insight
**Stage(s)**: final-report (Agent E) ← train-static (Agent C) root cause
**Symptom**: Final IV output curves showed visible discontinuities — sharp jumps of 0.07071 dex in IDVTG and 0.12175 dex in IDVD, appearing as "台阶" (steps/flat regions). Users reported this as a plotting or deployment issue, but the root cause was in training.
**Root Cause**: The independent IV base-PKAN training stage incorrectly set `base.kan.parameters()` all to trainable, which included the B-spline `grid` (knot positions) and `mask` that should remain frozen. Adam optimizer then shifted knot positions, creating non-monotonic grids (e.g., VDS grid `… -0.8332, -0.9464 …` — decreasing instead of increasing). B-spline evaluation requires strictly monotonic knots; non-monotonic grids cause certain spline segments to fail or discontinuously switch, producing visible curve jumps. The IV residual stage had the same parameter-freezing bug. Critical: the corruption was invisible in aggregate metrics (RMS, p95) but visually obvious in per-curve plots.
**Evidence**:
- Turn 27 (2026-07-01T10:07): Agent found "非单调节点" (non-monotonic knots) — base grid had 9 reversed knot positions, IV residual grid had 4 reversed positions.
- Turn 27: Agent confirmed "v90 的对应网格全部严格单调，而当前 v95 base 网格有 9 处倒序" (v90's corresponding grids were all strictly monotonic, while current v95 base grid had 9 reversals).
- Turn 28: After fixing grids to monotonic (without retraining), IDVD max local second-order jump dropped 98.98% (0.12175 → 0.001244 dex), IDVTG dropped 95.81% (0.07071 → 0.002966 dex). IV nonzero RMS only changed from 0.079164 → 0.080191 dex (+0.00103), proving the corruption didn't affect aggregate fit quality.
- Turn 28: Agent added "网格单调 + 曲线跳变" (grid monotonicity + curve jump) as a hard gate, preventing future occurrences.
**Generalization**: B-spline KAN models have an invariant that is NOT enforced by typical ML training loops: knot grids must be strictly monotonic. When parameter groups are accidentally unfrozen, optimizers can silently violate this invariant without producing NaN or OOM errors. The resulting corruption produces visible curve artifacts that look like deployment/plotting bugs but originate from a training-level parameter freezing defect. This is inherently cross-stage — the corruption happens in train-static but manifests only in final-report artifact inspection.
**Actionable Advice**:
1. Agent E should check for curve-level artifacts (per-condition IDVG, IDVBG, IDVD plots) even when aggregate metrics are acceptable. Jump discontinuities in otherwise smooth curves are a strong signal of knot grid corruption.
2. Agent E should recommend that the pipeline add a post-training grid monotonicity check as a hard gate in train-static — count `grid[i+1] <= grid[i]` violations and fail the stage if any exist. This is cheap to compute and catches a defect invisible to RMS metrics.
3. When comparing two runs with "identical code" but different curve shapes, Agent E should check KAN internal state (grid monotonicity, activation function sparsity/mask) before assuming training hyperparameter differences.

### Quality Gate Convergence Trap: Perfect Is the Enemy of Good
**Category**: Principle Insight
**Stage(s)**: final-report (Agent E), all stages with quality gates
**Symptom**: The session spanned 28 turns across 23+ hours of real time (2026-06-29T13:52 to 2026-07-01T13:03), with the agent cycling through v92→v93→v94→v95→v96 versions, repeatedly re-training and re-optimizing. The user expressed frustration multiple times about slow convergence and demanded final delivery.
**Root Cause**: Each quality gate failure triggered full retraining from scratch rather than targeted fixes. The agent kept discovering new root causes (warm-start LR mismatch, measured-only phase regression, reference model conflation, knot grid corruption) that required independent fixes, but each fix triggered a new full-pipeline run. The cumulative effect was that "one more fix" became an infinite loop — there was always another sub-threshold metric that could be improved.
**Evidence**:
- Turn 15 (2026-06-30T09:36): User: "到现在你一次任务闭环没完成" — after 15 turns and ~20 hours, no delivery had been made.
- Turn 19 (2026-06-30T11:37): User requested analysis of whether gates were too strict. Agent correctly analyzed and found gate thresholds were reasonable, but the fix loop continued.
- Turn 20–21 (2026-06-30T21:27 to 2026-07-01T00:06): Even after fixing the primary issue (residual gate "absolute zero alert" logic), the agent spent turns 20–24 running full-budget Top-1 symbolization that took 30+ minutes per attempt.
- Turn 28: After 2+ hours of a single task run, user demanded delivery. The slow part was CV second-order autodiff/passivity safety scan on CPU, not IV training — but the agent kept running full-pipeline instead of targeted fixes.
- Turn 24 final delivery: Terminal symbol gate passed, HSPICE IV 0.659%, CV 0.0228%. The final package actually met all deployment-relevant gates once the knot grid fix + reference model fix were applied.
**Generalization**: Multi-stage pipelines with interdependent quality gates create an optimization surface where fixing one gate can break another. When each fix requires a full pipeline re-run, the convergence time grows combinatorially. This is exacerbated when: (a) intermediate artifacts from earlier stages could be reused but aren't, (b) slow stages (CV passivity scans) are re-run even when their inputs haven't changed, (c) the agent treats "one more improvement" as always worthwhile regardless of the marginal gain vs delivery urgency.
**Actionable Advice**:
1. Agent E should explicitly track the **iteration count** and **cumulative wall-clock time** of optimization attempts. If the same stage has been retried >5 times or cumulative time >4 hours, Agent E should recommend delivery with current state and explicit documentation of which gates marginally failed.
2. Agent E should distinguish between **blocking failures** (no trained model exists, training crashed) and **marginal failures** (model trained, gates missed by <10% of threshold margin). Marginal failures on non-deployment gates should trigger a recommendation to deliver with caveats, not to retrain.
3. Agent E should recommend that the pipeline implement **artifact staging**: save intermediate checkpoints after each major phase (base-PKAN, residual, CV, symbolization pre-HSPICE) so that later fixes don't require re-running earlier phases whose inputs haven't changed.

### Pipeline Self-Standing Verification: Hardcoded Paths Break Portability
**Category**: Principle Insight
**Stage(s)**: final-report (Agent E), all stages
**Symptom**: The "same code" that worked for v90 failed for new devices because training scripts contained hardcoded paths to v90 directories (`C:\Users\meizikang\Documents\Modeling\DG_TFT_v90_...`). When run from a clean directory, these scripts could not find intermediate artifacts or wrote output to stale locations.
**Root Cause**: Development iteration tends to accumulate absolute paths — to data directories, checkpoint directories, output directories — that work during interactive development but break in deployment. The v90 hysteresis training script had hardcoded intermediate data paths, and the HSPICE validation had hardcoded metric filenames (e.g., `v92_safe_residual_metrics.json` that should be `v96_safe_residual_metrics.json`).
**Evidence**:
- Turn 22 (2026-07-01T00:37): Agent found v90 hysteresis script "仍把中间数据路径硬编码到 v90 目录，因此不能作为新器件自动流程直接使用" (still hardcoded intermediate data paths to v90 directory, so cannot be used directly as new-device automated flow).
- Turn 28 (2026-07-01T13:03): Agent found `checkpoint-diagnostic-only` hardcoded to look for `v92_safe_residual_metrics.json` while the formal package had `v96_safe_residual_metrics.json` — a versioned filename mismatch that broke post-delivery verification.
- Turn 24: Agent added explicit checks: "正式包不得出现 v90–v94 命名文件" (formal package must not contain v90–v94 named files).
- Turn 24 final delivery: 177 manifest files SHA-256 verified, historical version filename count = 0, delivery structure validation passed.
**Generalization**: Pipelines developed incrementally accumulate version-specific hardcoded references. When the pipeline is delivered as a portable package, these references silently break. The failure mode is pernicious because it only manifests when the package is extracted and run in a different directory — the developer's environment still has the old paths and files. The condition is: any pipeline where (a) version numbers appear in filenames, (b) intermediate file paths are constructed from version strings, and (c) validation scripts reference specific filenames rather than glob patterns.
**Actionable Advice**:
1. Agent E should recommend a "clean-room smoke test": extract the delivery ZIP into a temporary directory, delete all pre-existing outputs, and run the pipeline end-to-end. Any hardcoded path or missing artifact will surface immediately.
2. Agent E should check for hardcoded version strings in validation/reporting scripts. Filenames containing `v92`, `v93`, etc. in scripts (not in data/artifact names) indicate non-portable code.
3. Agent E should verify that all checkpoint loading paths use relative paths constructed from configuration, not absolute paths or hardcoded directory names.
