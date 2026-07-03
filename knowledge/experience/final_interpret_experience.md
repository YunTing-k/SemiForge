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
- Turn 24 final delivery: Terminal symbol gate passed, HSPICE IV 0.659%, CV 0.0228%. The final package actually met all deployment-relevant gates once the knot grid fix + reference model fix were applied.
- **Additional Evidence from 05-22 session**: Turn 100 — user said "不用跑完整的训练，代码给我我自己先检查一下，你只要确保能跑通" (don't run full training, just give me the code so I can check it — just make sure it runs). A deliverable with known limitations (no full convergence check yet) was preferred over blocking delivery until perfect. This reinforces the pattern that the user prioritizes having something usable over waiting for convergence.
**Generalization**: Whenever quality gates are implemented as hard blockers with no fallback artifact generation, pipelines can enter infinite refinement loops. This is especially dangerous when: (a) the optimization is non-convex (KAN training with random initialization), (b) gate thresholds are tight relative to achievable precision, or (c) multiple interdependent gates create a conjunctive failure mode where fixing one breaks another.
**Actionable Advice**:
1. Agent E should explicitly check whether artifacts (checkpoints, Verilog-A files, plots) exist in the output directory even when `quality_pass=False`. If artifacts are missing entirely, Agent E should flag this as a pipeline design defect — the stages succeeded at computation but failed at preservation.
2. Agent E should distinguish between "model that didn't train" (no checkpoint → real failure) and "model that trained well but failed a measured-curve symbolization gate" (artifact exists, gate is deployment-accuracy rather than training-accuracy).
3. Agent E should recommend a two-tier delivery model: **always save numerical checkpoints and metrics** (Tier 1), and separately gate the Verilog-A/HSPICE deployment artifacts (Tier 2). Tier 1 should never be empty regardless of gate status.

### Deployment Error Decomposition: Three-Layer vs Conflated — FIXED (reference model corrected)
**Status**: FIXED. The HSPICE validation at `dg_tft_full_model.py:16285` now uses `_temporary_deployment_symbolic_nets` context manager to swap in symbolic students as the Python reference. The smoke report documents `"python_reference_model": "deployed_symbolic_students"`. **The three-layer decomposition advice remains valuable for Agent E's reporting and is retained below.**

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
- **Additional Evidence from 05-22 session**: Turns 18–21 — the user provided their own notebook (`V2_FINAL.ipynb`) as the reference implementation and demanded the agent understand it line-by-line: "请你仔细研读我这一份代码，每一行都读，理解好。我这个是能运行的很好的" (please carefully study this code, read every line, understand it well. This one runs very well). Turns 22–29: Agent repeatedly drifted from the reference architecture (splitting KAN into separate IV/CV heads, omitting edge-wise symbolization, changing top-k defaults) and was corrected. The user's explicit instruction was: reuse the METHOD (unified KAN→physical parameter corrections, edge-wise top-1 symbolization, Phase1/2/3 training order) not the RESULTS.
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
- **Additional Evidence from 05-22 session**: Turns 30–109 (2026-05-23T03:09 to 2026-05-26T06:28 — over 3 days, 80+ turns, v1→v50+ versions). However, this session shows a CRITICAL DISTINCTION: the iteration was CONSTRUCTIVE because each version discovered and fixed a genuinely new structural physics problem (trap charge model, terminal parasitic matrix, DOS broadening, CV shape templates, derivative-aware loss, discrete convolution artifact, measurement configuration calibration). The session evolved from a simplified CV kernel to a commercial-style compact model CV. This contrasts with the 06-29 session where iteration was parametric tuning in circles. The key diagnostic signal: when each version adds new physical structure rather than just re-tuning hyperparameters, long iteration chains are not a trap — they are convergence toward a correct model architecture. Agent E should distinguish between "structural versioning" (new physics terms added) and "parametric churn" (same code, different LR/epochs).
**Generalization**: Multi-stage pipelines with interdependent quality gates create an optimization surface where fixing one gate can break another. When each fix requires a full pipeline re-run, the convergence time grows combinatorially. This is exacerbated when: (a) intermediate artifacts from earlier stages could be reused but aren't, (b) slow stages (CV passivity scans) are re-run even when their inputs haven't changed, (c) the agent treats "one more improvement" as always worthwhile regardless of the marginal gain vs delivery urgency.
**Actionable Advice**:
1. Agent E should explicitly track the **iteration count** and **cumulative wall-clock time** of optimization attempts. If the same stage has been retried >5 times or cumulative time >4 hours, Agent E should recommend delivery with current state and explicit documentation of which gates marginally failed.
2. Agent E should distinguish between **blocking failures** (no trained model exists, training crashed) and **marginal failures** (model trained, gates missed by <10% of threshold margin). Marginal failures on non-deployment gates should trigger a recommendation to deliver with caveats, not to retrain.
3. Agent E should recommend that the pipeline implement **artifact staging**: save intermediate checkpoints after each major phase (base-PKAN, residual, CV, symbolization pre-HSPICE) so that later fixes don't require re-running earlier phases whose inputs haven't changed.

### Pipeline Self-Standing Verification — FIXED (hardcoded paths removed)
**Status**: FIXED. Grep for `meizikang` across the entire `src/` directory returns zero matches. `assemble_release.py:94` proactively sanitizes any residual absolute paths: `if text.startswith("C:\\Users\\")` → replaces with `<historical_provenance_path_not_required_for_runtime>`. **This entry is retained for historical reference.**

---

### Physical Kernel Completeness Before NN Residual: Missing Physics Causes Non-Physical NN Artifacts
**Category**: Principle Insight
**Stage(s)**: final-report (Agent E), fit-physical (Agent B), train-static (Agent C)
**Symptom**: The KAN residual network produced clearly non-physical CV curves — oscillations, double humps, sharp peaks, negative capacitance steps, and "three-step" staircase artifacts — even as numerical metrics (RMS error) improved. The user repeatedly observed: "加了residualKAN，虽然数据上更接近了，但波动多了" (adding residual KAN made data closer but created more oscillations). The pure physical kernel, by contrast, produced smooth monotonic CV curves.
**Root Cause**: The CV physical kernel was structurally incomplete — missing terminal-pair parasitic extraction, missing measurement-configuration gain/offset calibration, missing DOS broadening physics, and using a discrete convolution implementation that replicated staircase features. The residual KAN was being asked to compensate for systematic model-structural errors rather than learn small residual corrections. When an NN is asked to bridge a large gap between an incomplete physical kernel and measured data, it will: (a) create narrow local features (peaks/dips) to "chase" individual data points, (b) produce oscillations as it switches between fitting different measurement families, (c) manufacture multi-step transitions because the base kernel's single sharp transition can't be "smoothed" by additive NN corrections — it gets replicated as shifted copies. The problem is fundamental: residual correction can subtract/add magnitude but cannot fundamentally change the functional form (e.g., widen a sharp transition into a smooth one).
**Evidence**:
- Turns 35–46 (2026-05-23T05:52–10:22): Agent's CV persistent oscillations traced to CV physical kernel missing trap charge model, terminal parasitic separation, and channel participation physics. Each fix reduced oscillation but the root structure was still incomplete.
- Turn 46 (2026-05-23T10:22): User provided detailed diagnosis: "CV 未饱和电荷核、反型电荷分配、top/bottom Vth 拆分、body 电容、trap Gaussian 峰" — the CV kernel was missing 5+ physical terms.
- Turn 49 (2026-05-23T14:12): User demanded: "cbg仍有不正常的抖动，这一点是必须解决的" (CBG still has abnormal jitter, this MUST be fixed). A narrow trap peak in the CV kernel was causing a sharp derivative spike that KAN couldn't smooth.
- Turn 70 (2026-05-24T12:36): After 70+ turns of CV kernel evolution (adding terminal parasitics, trap charge, measurement calibration, DOS broadening), the CV kernel was finally "complete enough" that residual KAN only needed to make small smooth corrections.
- Turn 82 (2026-05-24T19:47): User's critical observation: "纯物理内核的CV多么漂亮，曲线没有什么突变，抖动，加了pkan还好，加了residual就给人一种为了降低而肆意扭曲的感觉" (the pure physical kernel CV looks so beautiful — no jumps, no jitter; adding PKAN is OK, but adding residual gives a feeling of willful distortion just to lower error).
- Turn 89 (2026-05-25T05:07): User observation: "加了residual之后，IDVTG的开态的曲线变化明显不符合物理" (after adding residual, the IDVTG on-state curve changes clearly violate physics).
- Turn 103–106 (2026-05-25T16:25–16:54): "Three-step artifact" definitively traced to discrete 3-point gaussian convolution in the base CV kernel implementation. A single sharp base transition was convolved with 3 shifted copies, producing 3 visible staircase steps. The fix was to implement a commercial single-transition charge kernel (no convolution needed).
**Generalization**: Neural network residual corrections have a fundamental limitation: they can add/subtract magnitude at given input points but cannot perform structural transformations (widening, smoothing, changing functional form) of the base model's output. When the base physical kernel is structurally incomplete, the NN will compensate by creating narrow local features, oscillations, and multi-step artifacts — "chasing" individual data points at the cost of global physical shape. This occurs whenever: (a) the physical kernel misses entire physical mechanisms (terminal parasitics, trap dynamics, charge partition), (b) the measurement data contains families (cgg/cbg/ctg/vds) that require different base model structures, (c) the base kernel has sharp transitions that the NN tries to widen by stacking shifted copies. Metrics alone (log-RMS) will NOT detect this — the NN overfit may produce lower RMS while the curve shape becomes visibly non-physical.
**Actionable Advice**:
1. Agent E should always compare **pure physical kernel CV curves** against **residual-corrected CV curves** side-by-side. If the residual-corrected version has MORE oscillations, shoulders, peaks, or non-monotonic steps than the pure physical version, the residual KAN is compensating for missing base physics — flag this as a kernel architecture defect, not a training issue.
2. Agent E should check whether each measurement family (cgg, cbg, ctg, vds) has its own explicit measurement-configuration mapping in the model. If a single "unified" CV expression is being forced to fit 4 different measurement configurations without configuration-level gain/offset calibration, the residual NN will oscillate between families.
3. Agent E should verify: can the pure physical kernel (before any NN correction) produce correct curve SHAPES even if magnitudes are off? If shapes are wrong (too sharp, wrong platform levels, multi-step), the kernel is structurally incomplete, and no amount of NN residual training will fix it gracefully.

---

### Derivative-Aware Loss: Physics Constraints Must Enter the Training Objective, Not Post-Hoc Filtering
**Category**: Principle Insight
**Stage(s)**: final-report (Agent E), train-static (Agent C)
**Symptom**: The agent produced models where raw metrics (log-RMS) looked good but the curves had visible non-physical behavior: sharp shoulders on CV, on-state IV curvature distortion, and negative capacitance steps. The agent's safe-scale selection mechanism was "screening" residual KAN outputs after training to pick a version that looked acceptable — but the training loss itself had no physics constraints on derivatives.
**Root Cause**: The training loss was dominated by pointwise log-RMS error between model predictions and measured data. Derivatives (gm = dID/dVG, gds = dID/dVD, dC/dV, d²C/dV²) were computed only during post-hoc evaluation and used to select a "safe" residual scale, not to constrain the optimizer during training. This allowed the residual KAN to learn solutions that lowered pointwise error while creating local curvature artifacts. The user explicitly identified this: "筛图没有任何意义，不是最终的结果，模型才是最终的结果" (filtering plots is meaningless — the model is the final result). Hu Chengming's IEEE EDL papers explicitly include gm, gds, and output conductance in the training objective for compact NN models. The BSIM-NN/NQS framework requires charge derivatives to be continuous and monotonic as a hard constraint.
**Evidence**:
- Turn 82 (2026-05-24T19:47): User: "你之前都是把导数只用于晒图？你这肯定不对呀" (you were only using derivatives for displaying plots? That's definitely wrong). This directly challenged the post-hoc filtering approach.
- Turn 82 continued: User demanded derivative constraints enter the main loss: "ID/gm/gds 和 CV 一阶/二阶导、residual 二阶导和 base-aware 约束" (ID/gm/gds and CV first/second derivatives, residual second derivatives, and base-aware constraints must all be in main loss).
- Turn 83 (2026-05-25T02:20): Agent confirmed Kao/Kam/Hu IEEE EDL papers explicitly include output conductance and transconductance as training targets, not just post-hoc checks. AutoPINN and physics-informed NN literature consistently put smooth loss and monotonic constraints into the training objective.
- Turn 83–84: Agent restructured v37 to include `ivd` (IV derivative) and `cvd` (CV derivative) losses in every training step — "导数项现在确实进入了主优化日志" (derivative terms now genuinely enter the main optimization log).
- Turn 85 (2026-05-25T04:34): Agent added CV raw-data loss, denoised-derivative loss, and plateau loss as separate structured components in the main training loss.
- Turns 91–92: The training shift from "screen after training" to "train with physics" reduced CV artifacts while maintaining accuracy.
**Generalization**: When physical constraints (monotonicity, smoothness, derivative continuity) are applied only as post-hoc filters rather than training objectives, the optimizer can converge to solutions that look good on pointwise metrics but violate physics in derivative space. This is especially dangerous for compact models destined for circuit simulation, where transconductance (gm) and output conductance (gds) determine amplifier gain and linearity. The condition is: any pipeline where (a) the loss function uses only pointwise error metrics, (b) derivative behavior is checked only in evaluation/acceptance code paths, (c) a "safe scale" or post-hoc selection mechanism replaces in-training constraints. The diagnostic signal: the pure physical kernel has smoother curves than the NN-corrected model.
**Actionable Advice**:
1. Agent E should check whether the training loss function includes terms for: (a) first derivatives (gm for IV, dC/dV for CV), (b) second derivatives (smoothness of gm/gds, d²C/dV²), (c) residual magnitude relative to physical kernel output. Absence of these terms means the model may pass metrics but fail circuit simulation.
2. Agent E should verify that any "safe scale" or "acceptance filtering" mechanism is NOT the primary source of physics enforcement. If removing the post-hoc filter causes the model to show non-physical artifacts, the training loss is insufficient.
3. Agent E should recommend that the pipeline report derivative-based metrics alongside pointwise metrics in the final report (e.g., gm RMS, gds RMS, max |dC/dV|, CV monotonicity violation count).

---

### CV Measurement Configuration Calibration: Per-Sheet Offset/Gain Before Intrinsic Fitting
**Category**: Principle Insight
**Stage(s)**: final-report (Agent E), fit-physical (Agent B)
**Symptom**: CTG and VDS CV curves were systematically elevated above measured values by the pure physical kernel. The residual KAN was being forced to "pull down" entire curves by large magnitudes, causing it to create local non-monotonic features and oscillations. CBG showed the opposite — the low-capacitance plateau was consistently under-predicted. The user observed: "ctg的问题和vds的问题都在于纯物理把他们拉的太高了，远高于实测值" (CTG and VDS problems are both that the pure physics pulls them too high, far above measured values).
**Root Cause**: Different CV measurement configurations have different parasitic contributions (pad capacitance, inter-gate coupling, open/short de-embedding residuals, AC participation factors) that are NOT captured by a unified intrinsic charge kernel. The four CV configurations have distinct electrical setups: cgg (S/D short, AC low; TG/BG short, AC high) measures total gate capacitance; cbg (TG fixed bias, BG AC high; S/D short, AC low) measures bottom-gate capacitance with top-gate DC bias; ctg (BG fixed bias, TG AC high; S/D short, AC low) measures top-gate capacitance; vds (TG/BG short, AC high; D fixed bias, S AC low) measures tied-gate capacitance with drain DC bias. Each configuration has a different effective measurement gain (due to AC current path, pad layout, frequency response) and DC offset (due to charge injection, open-circuit calibration residuals). Without per-configuration gain/offset calibration, the intrinsic model's predicted capacitance will have systematic sheet-dependent biases that the NN residual cannot gracefully correct.
**Evidence**:
- Turn 79 (2026-05-24T18:45): User provided exact measurement configuration definitions for all 4 CV types and demanded: "你是不是理解实测数据里CV每条线究竟对应的是什么，拟合模型时是否真的是对应拟合的，没有拟合错" (do you understand what each CV line in the measured data actually corresponds to, and whether the model fit is truly mapping correctly?).
- Turn 84 (2026-05-25T03:12): Agent identified the root cause: CTG needs ~0.65x measurement participation factor, VDS needs ~0.49x. Without this calibration, the residual KAN "不是在微调，而是在试图把一个系统性 offset/scale 错误往下拉" (is not fine-tuning — it's trying to pull down a systematic offset/scale error).
- Turn 84: Agent added CV sheet/config-level calibration with per-configuration `gain + offset` parameters, physically interpreted as open/short de-embedding, AC participation, effective area, and charge offset. After calibration initialization, CV RMS dropped from 0.183 to 0.129 in smoke, CTG from 0.121 to 0.034.
- Turn 85 (2026-05-25T04:34): Full training with calibration: total CV RMS 0.1339 → 0.0953, CBG 0.2041 → 0.1353, CTG 0.0342 → 0.0333, VDS 0.1034 → 0.0841. This was the single largest CV improvement step in the entire 109-turn session.
- Turn 89 (2026-05-25T05:07): User raised concern about how these calibration parameters "在实际verilog里是怎么体现的" (how are these reflected in actual Verilog-A). This prompted separating measurement wrapper (calibration) from device parasitic (physical) in the Verilog-A export.
- Turn 107 (2026-05-25T19:08): User also asked whether CBG/CTG data labels might be SWAPPED, based on the physical expectation that a device with larger top-gate oxide capacitance should show larger CTG than CBG. This is a diagnostic question Agent E should consider.
**Generalization**: Multi-configuration CV measurements contain systematic inter-family offset/scale differences that arise from measurement setup, not from device physics. When these are not explicitly calibrated out before intrinsic model training, the NN residual will be forced to compensate, producing non-physical artifacts. This is analogous to open/short de-embedding in RF measurement and should be treated as a pre-processing calibration step. The condition is: any CV model trained on multiple measurement configurations (cgg, cbg, ctg, vds) without per-configuration gain/offset calibration will produce systematic residual NN artifacts.
**Actionable Advice**:
1. Agent E should verify that the pipeline has explicit per-measurement-configuration gain/offset calibration before intrinsic CV model training. If the same intrinsic charge kernel is asked to fit all CV sheets with only global parameters, systematic inter-sheet biases will force NN residual artifacts.
2. Agent E should check whether CBG/CTG labels are physically consistent: for a device where top-gate oxide capacitance > bottom-gate oxide capacitance, CTG(high-V) > CBG(high-V) is expected. If the opposite is observed, the measurement labels may be swapped — flag this.
3. Agent E should recommend that the Verilog-A export separates measurement wrapper parameters (calibration gain/offset) from device intrinsic parameters (Cox, Vth, Cit). Only device intrinsic parameters belong in the compact model; measurement calibration belongs in the testbench or extraction script.

---

### NN Residual Overfitting to MSE: Lower Error Can Mean Worse Physics
**Category**: Principle Insight
**Stage(s)**: final-report (Agent E), train-static (Agent C)
**Symptom**: Multiple versions across the session showed that adding residual KAN layers reduced CV RMS error while simultaneously creating non-physical curve features. The user observed: "不是一定要过度的追求mse最小" (you don't need to excessively pursue MSE minimum). When the agent compared version v4.1 to v4.2: v4.1 had lower RMS but produced CV with "强烈振荡，CGG、VDS、CTG 都有大量负斜率和跳变" (strong oscillations, CGG/VDS/CTG all had many negative slopes and jumps). v4.2 had even lower RMS but even worse physical shape.
**Root Cause**: The residual KAN has high capacity (98 edges, multi-layer splines) that can fit localized data point errors at the cost of introducing high-frequency local curvature. The MSE loss sees every data point equally but does not penalize local curvature, negative derivative steps, or multi-modal transition shapes. This creates a degenerate optimization landscape where the optimizer can always find a parametric solution that reduces MSE by adding local oscillations — the MSE gradient points toward "fit this region better" without any physics-informed counter-gradient. The safe-scale mechanism (selecting a residual multiplier between 0 and 1) partially mitigated this but didn't eliminate the root cause because the training itself was optimizing toward the wrong objective.
**Evidence**:
- Turn 45 (2026-05-23T10:19): User provided detailed loss function diagnosis: "IV开态的拟合程度很差很差，这是因为你用log去拟合的原因" (IV on-state fit is very very poor, that's because you're fitting in log space). Log-space MSE amplifies subthreshold errors and penalizes on-state errors weakly, causing the optimizer to prioritize the wrong regions.
- Turn 46 (2026-05-23T10:22): Agent compared v4.1 vs physical-stable version: v4.1 had lower loss but worse physical shape — "不是取 loss 最低，而是取物理形状优先" (don't pick the one with lowest loss, pick the one with best physical shape).
- Turn 47 (2026-05-23T12:34): Same pattern with v4.2: "CV RMS 更低到 0.179 dec，但物理形状更差，CGG/VDS/CTG 出现大量尖峰、回落和跳变" (CV RMS lower at 0.179 dec, but physical shape worse — CGG/VDS/CTG showed many spikes, drops, and jumps).
- Turn 83 (2026-05-25T02:20): User explicitly: "不一定要过度追求mse最小" (don't have to excessively pursue MSE minimum). The trade-off was recognized: sometimes a slightly higher RMS with correct physical shape is superior.
- Turn 84 (2026-05-25T03:12): After adding measurement-calibration + derivative loss + shape templates, CV artifacts were reduced while maintaining reasonable RMS — proving the point that the right loss design matters more than model capacity.
**Generalization**: The optimizer minimizes whatever is in the loss function, not what the engineer intends. When the loss is exclusively pointwise log-RMS error, the NN will learn solutions that are locally accurate but globally non-physical. This is a degenerate optimization: any sufficiently expressive NN can reduce MSE by adding local oscillations because MSE has no physics gradient to counterbalance it. The condition is: any pipeline where (a) the loss function is exclusively magnitude-based (RMS, MAE, log-RMSE), (b) the correction network has high capacity relative to the data density, (c) individual measurement points have measurement noise that the NN can "fit" with local curvature. The diagnostic signal: models with lower validation loss have visibly worse curve shapes than models with slightly higher loss.
**Actionable Advice**:
1. Agent E should compare the **shape quality vs RMS error trade-off** across multiple residual scales or training checkpoints. If models with lower error have worse shape characteristics (more negative slope steps, more local peaks, higher second derivatives), the loss function is insufficiently physics-constrained.
2. Agent E should recommend that the loss function include: (a) DERIVATIVE terms (gm, gds, dC/dV) with comparable weight to value error, (b) SHAPE templates (monotonicity, single-transition, plateau flatness), (c) residual MAGNITUDE regularization (penalize large corrections relative to the physical kernel baseline).
3. Agent E should recommend that the pipeline report SHAPE-DERIVED metrics alongside value-derived metrics in the final report: negative slope step count, second derivative RMS, derivative peak count, plateau flatness score.

---

### Iteration Convergence Diagnosis: Structural Evolution vs Parametric Churn
**Category**: Principle Insight
**Stage(s)**: final-report (Agent E), all modeling stages
**Symptom**: The 05-22 session ran 102+ turns across 3+ days, producing v1 through v50+ versions. Superficially this looks like the "Quality Gate Convergence Trap" pattern — endless iteration without convergence. However, the outcome was a genuinely improved modeling framework that evolved from a simplified CV kernel to a commercial-style compact model. The user paradoxically both demanded faster convergence AND repeatedly found major structural issues that required fundamental code changes.
**Root Cause**: The session was NOT pure parametric churn. Each version group addressed structurally distinct physics problems: v1–v30 focused on IV kernel + KAN correction, v31–v40 added CV terminal parasitics, v41–v50 restructured CV into a single-transition commercial charge kernel. The version explosion was a symptom of the PHYSICAL KERNEL HAVING MULTIPLE STRUCTURAL DEFECTS that required sequential discovery and repair. This is categorically different from tuning hyperparameters in circles. The key distinction: when each iteration ADDS NEW PHYSICAL STRUCTURE (new terms in the charge equation, new calibration steps, new derivative constraints), the iteration is PRODUCTIVE convergence. When each iteration only CHANGES EXISTING PARAMETERS (different LR, different epoch count, different weight), the iteration is PARAMETRIC CHURN and a convergence trap.
**Evidence**:
- Turns 1–10 (2026-05-22T07:59–10:10): Data processing — CSV→XLSX, aging correction, smoothing. Pure data preparation, not modeling.
- Turns 11–17 (2026-05-22T10:51–12:42): Physical kernel fitting — EKV level-2 model, IV+CV fitting, CV trap model. First structural phase.
- Turns 18–30 (2026-05-22T12:48 to 2026-05-23T03:09): KAN architecture development — unified KAN→physical parameter corrections, edge-wise symbolization, Phase1/2/3 training. Second structural phase.
- Turns 31–50 (2026-05-23T03:52–14:14): CV kernel evolution v1 — trap charge model, terminal parasitic, channel participation, DOS broadening, body capacitance. Third structural phase.
- Turns 51–75 (2026-05-23T14:46 to 2026-05-24T14:26): Loss function redesign — CV shape templates, derivative-aware loss, measurement calibration, charge-based CV residual. Fourth structural phase.
- Turns 76–95 (2026-05-24T14:29 to 2026-05-25T10:35): Commercial CV restructure — single-transition kernel, CV plateau constraints, sheet-balance, residual capacity reduction. Fifth structural phase.
- Turns 96–109 (2026-05-25T12:37–19:08): Final convergence — discrete convolution removal, commercial unified CV kernel, base PKAN finetuning, delivery packaging. Sixth structural phase.
- Each phase produced a qualitatively different model architecture, not just different hyperparameters. The user's frustration was not about iteration per se but about the agent not recognizing when it was switching between structural problems vs parametric tuning.
**Generalization**: Long iteration chains (10+ turns, 50+ versions) can be either PRODUCTIVE or TRAPPED. The distinction is whether new versions add structural complexity (new model terms, new training phases, new calibration steps) or only change parameter values (different weights, different epoch counts). Structural evolution is a legitimate path to a better model when the initial architecture is incomplete; parametric churn is wasteful. The diagnostic signal: plot the model's structural complexity (number of free parameters in the physical kernel, number of KAN output channels, number of loss terms) against version number. If structural complexity plateaus but versions continue, it's parametric churn. If structural complexity continues to evolve, the iteration is genuine discovery.
**Actionable Advice**:
1. Agent E should classify each version in the operation_log by whether it introduced NEW MODEL STRUCTURE (new physical terms, new calibration stages, new KAN output channels, new loss components) or only changed EXISTING PARAMETERS. Report the ratio of structural changes to parametric changes.
2. Agent E should check whether the model's structural complexity (count of physical parameters, KAN layers/outputs, calibration terms) plateaued before the iteration stopped. If complexity stopped changing but tuning continued, flag as parametric churn.
3. Agent E should recommend that the pipeline track "structural version delta" — a summary of WHAT CHANGED between versions (new features added vs existing features retuned). This is different from tracking metric deltas.

---

### Training Duration Regression: Feature Accumulation Slows Iteration
**Category**: Principle Insight
**Stage(s)**: final-report (Agent E), train-static (Agent C)
**Symptom**: Training time grew from ~20 minutes (Turns 21–30, simple CV kernel + KAN) to ~100+ minutes (Turns 97–109, full commercial CV kernel + derivative losses + edge-wise symbolization). The user asked: "为什么训练要这么长时间，之前不是只要20多分钟吗？" (why does training take so long now? Wasn't it only 20+ minutes before?). This slowdown was not driven by epoch count increase but by computational complexity added incrementally as features accumulated.
**Root Cause**: Several features added for physical correctness introduced quadratic or N× complexity overhead that multiplied training time: (a) discrete Gaussian convolution over CV measurement gates (3-point shifted copy → 3× base CV computation per forward pass, Turns 96–97), (b) full-curve derivative computation in main loss (previously only in evaluation, Turn 83), (c) edge-wise symbolic replacement (98 edges × function search per edge, Turns 25–29), (d) safe-scale exhaustive search over residual multipliers. The agent did not track computational cost per feature addition, resulting in death-by-a-thousand-cuts slowdown. The fix (Turn 97–98) reduced convolution from 3-point to batched single-pass and changed default dtype from float64 to float32, recovering ~2× speed.
**Evidence**:
- Turns 21–30 (2026-05-22T12:59–2026-05-23T03:09): Simple PKAN training took ~2.5 minutes per full run (4600 lines of output, "约 2 分半").
- Turn 31 (2026-05-23T03:52): Training with diagnostic data: ~3–4 minutes.
- Turn 33 (2026-05-23T04:45): Full dataset training began taking longer — "这个会比较久" (this will take longer). Still single-digit minutes.
- Turn 82 (2026-05-24T19:47): Training with derivative losses added: still manageable.
- Turn 96 (2026-05-25T12:37): After adding DOS discrete convolution: "v45运行的太慢了" (v45 is running too slowly) — user explicitly identified the convolution as the bottleneck ("主要原因应该在卷积部分").
- Turn 97–98 (2026-05-25T14:19–14:20): Agent fixed convolution from 3-point loop to batched single-pass, reduced float64→float32 for most ops, cut shape loss frequency. Speed recovered to ~2.5–3 min per smoke run.
- Turn 102 (2026-05-25T15:08): v47 full training: ~102 minutes for 1200 epochs, vs earlier ~20–30 min for fewer features. Profile showed 5-epoch benchmark at ~46 seconds → ~2.5 hours extrapolated vs actual 102 min (early stopping saved time).
- Turn 106 (2026-05-25T16:54): After removing discrete convolution, training returned to ~90–100 minutes for full 1200 epochs — showing that convolution alone added ~30–40% overhead.
**Generalization**: Incremental feature addition without profiling creates silent computational debt. Each new physics term, each new loss component, each new validation check adds to the per-iteration compute cost. When training runs take >1 hour, the iteration cycle becomes untenable — the developer gets <10 attempts per day. This is especially dangerous when each feature individually adds <20% overhead but 5+ such features double training time. The condition: any pipeline where (a) training time was initially fast but has grown over versions, (b) new "features" are added that recompute intermediate results (convolution, per-curve derivatives, exhaustive search), (c) no profiling or timing breakdown exists per training phase.
**Actionable Advice**:
1. Agent E should note the **total training duration** and compare it against previous runs. If duration has grown >2× between versions with similar epoch counts, there is computational bloat — flag it even if model quality improved.
2. Agent E should recommend that the pipeline profile training time by phase (warmup %, PKAN training %, residual training %, symbolization %, safe-scale search %) so that feature-specific slowdowns can be identified.
3. Agent E should flag any feature that requires PER-SAMPLE computation (e.g., per-curve derivative, per-measurement-configuration convolution) that wasn't present in earlier faster versions. These are the typical culprits of training slowdown.

---

### Session Meta-Notes (05-22)

The 05-22 session (2026-05-22T07:59 to 2026-05-26T06:28, 102+ turns, ~3.5 days) is fundamentally a **pipeline BUILDING session**, not a pipeline EXECUTION session like 06-29. The learnings are about:

1. **What makes a model architecture "deliverable"**: The session converged from trial-and-error to a methodology: (a) complete physical kernel FIRST, (b) add measurement calibration SECOND, (c) add NN residual LAST with strong physics constraints, (d) verify derivative behavior in training, not post-hoc.

2. **The role of KAN**: Evolved from "catch-all black-box residual" → "physical parameter modifier (nfac, Cit, trap_width)" → "smooth single-transition CV charge kernel + physics-constrained residual." The user's demand for physical interpretability drove KAN usage toward quasi-physical parameters rather than arbitrary function approximation.

3. **Measurement fidelity awareness**: The session discovered that CV measurement configurations (cgg/cbg/ctg/vds) require explicit handling (gain/offset per sheet, terminal-pair parasitic extraction) because they are electrically different measurement setups, not just "different data columns."

4. **The user's quality standard**: Physical correctness > numerical accuracy. The user explicitly preferred models where "pure physical kernel CV looks beautiful with no jumps" over models with "lower RMS but residual-made oscillations." This is directly actionable for Agent E's quality assessment.

5. **Version explosion classification**: 50+ versions over 102 turns was NOT a trap — it was genuine structural evolution through 6 distinct architectural phases (data prep → physical kernel → KAN correction → CV kernel evolution → loss redesign → commercial CV restructure). Agent E should distinguish these from parametric churn.

---

## Engine Architecture Diagnostics (from B/C/D stage experience, 05-22 session)

The following patterns were identified in the 05-22 development session. They are engine-level architectural concerns that Agents B/C/D cannot fix via their tools — only `fallback_to_previous` or `accept_quality`. They are documented here for **Agent E's diagnostic reference**: when you see these symptom patterns in a pipeline run, you can identify the structural root cause and advise the user accordingly.

### Derivative Observables (gm/gds, dC/dV) Not in Training Loss → Model Fails Deployment Despite Good RMS

**Symptom**: IV RMS and CV RMS metrics look acceptable, but output curves show physically implausible features: IDVTG on-region has unrealistic curvature (gm roll-off or sharp kinks), CV curves have local oscillations or multi-step transitions, residual KAN contributions visually "twist" curves in non-physical ways. These artifacts are invisible to RMS metrics.

**Root Cause**: The training loss minimizes only point-wise log-error (`log(ID_pred/ID_meas)`, `log(C_pred/C_meas)`). Derivatives (transconductance `gm`, output conductance `gds`, `dC/dV`) are computed only during post-training diagnostics and safe-scale screening — not as training objectives. The residual KAN overfits local curvature to reduce pointwise error, producing jagged `gm` and oscillatory `dC/dV`. The safe-scale post-hoc filter then responds by setting scale=0 (disabling residual entirely) because any non-zero residual introduces derivative artifacts.

**Why this blocks Agents B/C/D**: They cannot modify the loss function — it's engine code. They can only try `accept_quality` or `fallback_to_previous`.

**What should be fixed in the engine**: 
- Add `gm` and `gds` terms to IV loss (weight ~0.15-0.25 and ~0.05-0.10)
- Add `dC/dV` and `d²C/dV²` terms to CV loss on smoothed reference curves (not raw noisy data)
- Use base-aware relative constraints: `max(0, residual_derivative_worsening - base_derivative_worsening)`
- Do NOT delegate derivative enforcement to post-hoc safe-scale selection — it will always pick scale=0

**Evidence**: 05-22 TURN 82-83: "筛图没有任何意义，不是最终的结果，模型才是最终的结果...导数只用于晒图肯定不对" — user explicitly demanded derivatives enter main loss. This exact failure pattern repeated across v35-v36.

### Physical Kernel Structurally Incomplete → KAN Produces Non-Physical Oscillations

**Symptom**: CV fitting metrics remain poor (RMS > 0.20 dec) despite multiple rounds of adjusting KAN capacity, residual strength, and epoch count. Visual inspection reveals systematic patterns: CTG ~2x higher than measurement across all bias conditions; CBG low-capacitance plateau substantially above measured floor; CGG shows non-physical double-hump transitions. The KAN, instead of learning small device-specific corrections, compensates for fundamental model structure gaps — producing curves with oscillations, multi-step transitions, and false shoulders.

**Root Cause**: The physical CV kernel was missing three structural elements:
1. Measurement-configuration de-embedding — intrinsic terminal capacitance ≠ measured CV (includes parasitics, contact effects, AC participation ratios)
2. Disorder/trap-induced transition broadening — EKV thermal kT/q sharpness is too narrow for IGZO TFTs
3. OFF-state inter-gate coupling — gates are NOT electrostatically coupled through a depleted channel

The KAN was asked to compensate for first-order global corrections (measurement de-embedding, disorder broadening) that should be in the physical kernel itself. Neural networks can add/subtract magnitude at given points but cannot structurally transform the output functional form.

**What should be fixed in the engine**:
- Add per-configuration `gain` + `offset` for each CV measurement type (cgg/cbg/ctg/vds), pre-extracted from OFF/ON plateaus and frozen before KAN training
- Add disorder/trap broadening physics: `nfac`, `Cit_t/b`, `trap_width` as global device parameters with substantial tuning range (not micro-delta)
- Add channel participation factor: smoothly transitions from weak (OFF, gates decoupled) to strong (ON, gates coupled through conductive sheet)

**Evidence**: 05-22 TURN 37: "KAN 正在替物理模型擦屁股...物理核里有几个结构性缺口." TURN 84: CTG needed ~0.65x gain, VDS needed ~0.49x — confirming measurement-path, not device-physics, errors. Session spent 30+ turns (v5→v35) trying to fix CV through KAN hyperparameters with minimal improvement; largest improvement came from adding sheet-level calibration, not from stronger KAN.

### CV Charge Model Multi-Step Transitions from Discrete Convolution

**Symptom**: CV curves (especially CGG and CBG) show 2-3 distinct steps or "shoulders" in the OFF→ON transition. These appear as double-hump or triple-hump structures that don't correspond to any physical mechanism. Visible as multiple `dC/dV` peaks.

**Root Cause**: Three mechanisms at different development stages:
1. **Discrete Gaussian convolution (v45-v47)**: 3-point convolution on EKV sigmoid transition = 3 shifted copies = 3 visible steps
2. **Multiple additive charge sub-components**: `Q_total = Q_inv + Q_trap_fast + Q_trap_slow + Q_tail` — if each activates at different voltages, produces staircase transitions
3. **Residual KAN narrow branches**: Optimizer can discover parameter values producing multiple narrow capacitance bumps

**What should be fixed in the engine**: The base capacitance function `C(V) = dQ(V)/dV` must produce exactly ONE transition by construction. Do NOT use discrete convolution (`cap_broad = sum(w_i * cap(V - dV_i))`). Use single continuous transition with global physical parameters (`nfac`, `Cit`, `trap_width`) controlling width. Each additive charge sub-component must share the SAME effective threshold voltage. After any kernel change, plot `dC/dV` — any curve with >1 peak indicates false multi-transition.

**Evidence**: 05-22 TURN 103-106: "商业 CV compact model 根本不用 convolution." After replacing 3-point convolution with single-transition kernel, all curves reduced to exactly 1 derivative peak.

### CV Measurement Configuration Calibration — Per-Sheet Gain/Offset

**Symptom**: Systematic per-configuration bias — CTG and VDS consistently ~1.5-2.0x higher than measured across all bias points. Residual KAN wastes capacity trying to pull entire curves up/down.

**Root Cause**: Physical kernel computes intrinsic terminal capacitance. Measured CV includes open/short de-embedding error, contact parasitics, effective area uncertainty, and AC frequency response — different for each measurement configuration.

**Engine fix**: Pre-extract per-configuration `gain` + `offset` from OFF/ON plateaus before KAN training. Freeze these during training. Do NOT embed in Verilog-A device model — they're measurement-path properties. CTG typically 0.65x gain, VDS typically 0.49x gain.

**Evidence**: 05-22 TURN 88: "CV RMS 从约 0.183 dec 降到 0.129 dec，CTG 从 0.121 dec 降到 0.034 dec" — single largest CV improvement in the entire 109-turn session.

### Separate IV/CV Residual Scales — Global Scale Kills Both When One Fails

**Symptom**: IV residual shows 50%+ improvement in IDVTG on-region, but safe-scale scan selects `scale = 0.0`, disabling all residuals. Per-scale table shows IV improves monotonically at every scale, but CV shape diagnostics flag minor artifacts.

**Root Cause**: Single global residual scale selector sees "CV not strictly improving" and zeros everything — including IV residual which was working perfectly.

**Engine fix**: Use two independent scales: `safe_iv_residual_scale` (IV metrics only) and `safe_cv_residual_scale` (CV metrics only). IV can be non-zero while CV stays at zero. Default CV residual to off — it provides marginal benefit (<0.005 dec RMS) for most devices.

**Evidence**: 05-22 TURN 109: "CV 只要对 residual 稍微敏感，就会把整个 residual scale 压到 0，连 IV residual 也一起被关掉."

### CV Shape Template Constraints — Beyond Monotonicity

**Symptom**: CV curves pass monotonicity checks (`dC/dV >= 0`) but still have double-hump transitions, continued slow rise on high-voltage plateau, and residual KAN twist.

**Root Cause**: Monotonicity and smoothness constraints are insufficient. The correct CV shape is "plateau-transition-plateau" — two flat regions connected by a single smooth rise. Without template-based loss, the optimizer finds monotonic but non-plateau solutions.

**Engine fix**: Add template loss: unimodality penalty (exactly 1 peak in `dC/dV`), plateau flatness penalty (10x weight on `|dC/dV|` in top/bottom 25% of sweep), no continued rise penalty. Make shape loss base-aware: only penalize if residual is worse than base. Apply different weights by CV type (CGG/CBG highest priority).

**Evidence**: 05-22 TURN 92-93: "CGG 的 double hump 基本不是器件物理，而是模型先验造成的假结构."

### KAN Correction Form — Additive vs Multiplicative by Parameter Semantics

**Symptom**: Using purely multiplicative KAN corrections causes threshold-sensitive parameters (τ, V-equivalent shifts) to retain non-physical voltage dependence. Using purely additive causes dimensionless parameters (coupling factors, partition ratios) to produce values outside [0,1].

**Root Cause**: Different physical parameters have different natural ranges. Threshold shifts are additive (`Vth_corrected = Vth_base + ΔVth`). Mobility is multiplicative (`μ_corrected = μ_base × exp(dlog_μ)`). Partition ratios must stay in [0,1] (`p_corrected = sigmoid(logit(p_base) + dlogit_p)`).

**Engine fix**: Use mixed correction forms. For hysteresis SRH-KAN: `log10(τ)` additive, `log10(amplitude)` additive, dimensionless ratios bounded sigmoid. This is a future architecture consideration — current code uses direct prediction (raw τ and A values).

**Evidence**: 05-22 TURN 42-43: "KAN 修正物理参数，都是乘性修正，是不是有几个加性修正会好一些."

### CBG/CTG Data Label Swap Detection

**Symptom**: CBG capacitance span >> CTG span, but top gate has higher Cox and should show larger capacitance. Visual inspection: curves labeled "CBG" look like CTG curves and vice versa.

**Detection**: Compute `R_Cox = Cox_t / Cox_b`. Compute `R_span = span_cbg / span_ctg`. If `(R_Cox > 1 and R_span < 1)` or `(R_Cox < 1 and R_span > 1)` → labels swapped.

**Evidence**: 05-22 TURN 109: Cox_t/Cox_b = 2.92 but CBG_span/CTG_span = 3.27 — opposite of expected. After swap: CV RMS dropped from ~0.13 dec to 0.0745 dec. User confirmed: "有可能是我标反了." Engine already has auto-orientation at `v92_phase1_cv_gate_orientation.json` but relies on hardcoded thresholds (Cox ratio ≥1.40, span ratio ≥1.55).

(End of file - total 273 lines)
