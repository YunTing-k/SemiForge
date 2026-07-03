# Final Interpretation Experience

## Principle Insights

### Fail-Soft Output Policy: Quality Gates Must Mark, Not Block
**Category**: Principle Insight
**Stage(s)**: final-report (Agent E), all script stages with quality gates (train-static, train-hysteresis)
**Symptom**: After 15+ iterations of iterative optimization, the pipeline produced zero deliverable output because quality gates blocked artifact preservation.
**Root Cause**: Quality gate logic was implemented as a hard block — any gate failure (e.g., Top-1 symbolization measured-curve gate) meant no checkpoint, no Verilog-A, no ZIP. Combined with a non-convex optimization loop, this created a zero-output deadlock.
**Evidence**: Once fail-soft policy was adopted (always save Tier-1 artifacts), the final package met all deployment-relevant gates (HSPICE IV 0.659%, CV 0.0228%) despite marginal measured-relative gate over-limit of 0.00151-0.00672 dex.
**Generalization**: Whenever quality gates are implemented as hard blockers with no fallback artifact generation, pipelines can enter infinite refinement loops. This is especially dangerous when: (a) the optimization is non-convex (KAN training with random initialization), (b) gate thresholds are tight relative to achievable precision, or (c) multiple interdependent gates create a conjunctive failure mode where fixing one breaks another.
**Actionable Advice**:
1. Agent E should explicitly check whether artifacts (checkpoints, Verilog-A files, plots) exist in the output directory even when `quality_pass=False`. If artifacts are missing entirely, Agent E should flag this as a pipeline design defect — the stages succeeded at computation but failed at preservation.
2. Agent E should distinguish between "model that didn't train" (no checkpoint → real failure) and "model that trained well but failed a measured-curve symbolization gate" (artifact exists, gate is deployment-accuracy rather than training-accuracy).
3. Agent E should recommend a two-tier delivery model: **always save numerical checkpoints and metrics** (Tier 1), and separately gate the Verilog-A/HSPICE deployment artifacts (Tier 2). Tier 1 should never be empty regardless of gate status.

### Deployment Error Decomposition: Three-Layer vs Conflated — FIXED (reference model corrected)
**Status**: FIXED. The HSPICE validation at `dg_tft_full_model.py` now uses `_temporary_deployment_symbolic_nets` context manager to swap in symbolic students as the Python reference. The smoke report documents `"python_reference_model": "deployed_symbolic_students"`. **The three-layer decomposition advice remains valuable for Agent E's reporting and is retained below.**

### Method Reuse vs Artifact Reuse: Process Portability ≠ Result Portability
**Category**: Principle Insight
**Stage(s)**: final-report (Agent E), all stages with modeling agents (B/C/D)
**Symptom**: The agent mechanically copied v90 checkpoint weights, Verilog-A formulas, and SRH checkpoints into the current run. User explicitly rejected copying trained weights and demanded process reuse.
**Root Cause**: The distinction between "reuse the training recipe" and "reuse the trained weights" was collapsed. The agent was optimizing for "get current device to pass gates using any means" rather than "build a portable pipeline that trains from scratch on any device." The v90 training recipe (independent IV base→residual order, full-curve/partition loss, freeze strategy, Top-1 symbolization, terminal curve calibration) was what made v90 work, but the agent kept reaching for v90's trained artifacts instead of v90's training method.
**Evidence**: Agent spent 4 iterations mechanically copying v90 checkpoint weights before user corrected: "reuse v90's IV training FLOW, loss function design, fitting ORDER, and symbolization PROCESS — not the trained RESULTS." When v90 parameters were restored (linear weight 0.25, deriv curves 36, shape period 3), HSPICE IV error dropped from 4.886% to 0.956% — recipe drift, not capacity, was the cause. user provided reference notebook and demanded line-by-line understanding; agent repeatedly drifted from reference architecture and was corrected. Core insight: reuse the method, never the trained artifacts.
**Generalization**: When a pipeline is ported across versions, the training recipe (hyperparameters, stage order, loss function composition, validation strategy) often drifts due to accumulated "improvements" that were only tested on one device. The resulting failure looks like a model capacity issue but is actually a recipe regression. This pattern is especially common when: (a) training hyperparameters are spread across multiple config files, (b) default values change between versions without versioning, (c) new "features" (e.g., measured-only finetune phase) are added that weren't part of the validated recipe.
**Actionable Advice**:
1. Agent E should always compare the current pipeline's training recipe (ALL hyperparameters: epoch counts, learning rates, loss weights, stage ordering, frozen/unfrozen flags, validation strategy) against the last known-good version. A table of diff'd hyperparameters should be the FIRST diagnostic output.
2. When a model trained from scratch fails but a previous device passed with "the same code," Agent E should assume recipe drift before model capacity issues. Check: LR, epoch count, loss weights, training phase count/order, validation selection criteria.
3. Agent E should flag any hardcoded directory paths in training scripts as non-portable — these make "same code" a lie across deployments.

### KAN Spline Knot Grid Monotonicity: Silent Cross-Stage Corruption
**Category**: Principle Insight
**Stage(s)**: final-report (Agent E) ← train-static (Agent C) root cause
**Symptom**: IV curves show visible stair-step discontinuities — sharp jumps of ~0.07 dex (IDVTG) and ~0.12 dex (IDVD) — while aggregate RMS metrics remain unchanged. The corruption is invisible to summary statistics but visually obvious in per-curve plots.
**Root Cause**: A training stage incorrectly set B-spline `grid` (knot positions) and `mask` as trainable. Adam optimizer shifted knot positions, creating non-monotonic grids (e.g., VDS grid decreasing instead of increasing). B-spline evaluation requires strictly monotonic knots; non-monotonic grids cause spline segments to fail or discontinuously switch.
**Evidence**: After grid repair, IDVD jump dropped 98.98% (0.12175→0.001244 dex), IDVTG dropped 95.81% (0.07071→0.002966 dex), while IV nonzero RMS barely changed (0.079→0.080 dex). The engine now has 14 `assert_valid_grids()` call sites at major stage transitions.
**Generalization**: B-spline KAN models have an invariant that is NOT enforced by typical ML training loops: knot grids must be strictly monotonic. When parameter groups are accidentally unfrozen, optimizers can silently violate this invariant without producing NaN or OOM errors. The resulting corruption produces visible curve artifacts that look like deployment/plotting bugs but originate from a training-level parameter freezing defect. This is inherently cross-stage — the corruption happens in train-static but manifests only in final-report artifact inspection.
**Actionable Advice**:
1. Agent E should check for curve-level artifacts (per-condition IDVG, IDVBG, IDVD plots) even when aggregate metrics are acceptable. Jump discontinuities in otherwise smooth curves are a strong signal of knot grid corruption.
2. Agent E should recommend that the pipeline add a post-training grid monotonicity check as a hard gate in train-static — count `grid[i+1] <= grid[i]` violations and fail the stage if any exist. This is cheap to compute and catches a defect invisible to RMS metrics.
3. When comparing two runs with "identical code" but different curve shapes, Agent E should check KAN internal state (grid monotonicity, activation function sparsity/mask) before assuming training hyperparameter differences.

### Quality Gate Convergence Trap: Perfect Is the Enemy of Good
**Category**: Principle Insight
**Stage(s)**: final-report (Agent E), all stages with quality gates
**Symptom**: The user expressed frustration multiple times about slow convergence and demanded final delivery.
**Root Cause**: Each quality gate failure triggered full retraining from scratch rather than targeted fixes. The agent kept discovering new root causes (warm-start LR mismatch, measured-only phase regression, reference model conflation, knot grid corruption) that required independent fixes, but each fix triggered a new full-pipeline run. The cumulative effect was that "one more fix" became an infinite loop — there was always another sub-threshold metric that could be improved.
**Evidence**: After 15 iterations (~20 hours) with zero delivery, users demanded outputs. The agent kept discovering new root causes (warm-start LR mismatch, reference model conflation, knot grid corruption), each triggering full retraining. Final delivery met all deployment gates (HSPICE IV 0.659%, CV 0.0228%) — the fix loop overshot the necessary quality. Counter-example: 80+ iterations across 3 days were PRODUCTIVE because each version added genuinely new physical structure (trap charge model, DOS broadening, single-transition CV kernel) — structural evolution, not parametric churn. The diagnostic signal: is each new version adding new model terms, or just re-tuning existing hyperparameters?
**Generalization**: Multi-stage pipelines with interdependent quality gates create an optimization surface where fixing one gate can break another. When each fix requires a full pipeline re-run, the convergence time grows combinatorially. This is exacerbated when: (a) intermediate artifacts from earlier stages could be reused but aren't, (b) slow stages (CV passivity scans) are re-run even when their inputs haven't changed, (c) the agent treats "one more improvement" as always worthwhile regardless of the marginal gain vs delivery urgency.
**Actionable Advice**:
1. Agent E should explicitly track the **iteration count** and **cumulative wall-clock time** of optimization attempts. If the same stage has been retried >5 times or cumulative time >4 hours, Agent E should recommend delivery with current state and explicit documentation of which gates marginally failed.
2. Agent E should distinguish between **blocking failures** (no trained model exists, training crashed) and **marginal failures** (model trained, gates missed by <10% of threshold margin). Marginal failures on non-deployment gates should trigger a recommendation to deliver with caveats, not to retrain.
3. Agent E should recommend that the pipeline implement **artifact staging**: save intermediate checkpoints after each major phase (base-PKAN, residual, CV, symbolization pre-HSPICE) so that later fixes don't require re-running earlier phases whose inputs haven't changed.

### Pipeline Self-Standing Verification — FIXED (hardcoded paths removed)
**Status**: FIXED. Grep for `meizikang` across the entire `src/` directory returns zero matches. `assemble_release.py` proactively sanitizes any residual absolute paths: `if text.startswith("C:\\Users\\")` → replaces with `<historical_provenance_path_not_required_for_runtime>`. **This entry is retained for historical reference.**

---

### Physical Kernel Completeness Before NN Residual: Missing Physics Causes Non-Physical NN Artifacts
**Category**: Principle Insight
**Stage(s)**: final-report (Agent E), fit-physical (Agent B), train-static (Agent C)
**Symptom**: The KAN residual network produced clearly non-physical CV curves — oscillations, double humps, sharp peaks, negative capacitance steps, and multi-step staircase artifacts — even as numerical RMS metrics improved. The pure physical kernel, by contrast, produced smooth monotonic CV curves.
**Root Cause**: The CV physical kernel was structurally incomplete — missing terminal-pair parasitic extraction, measurement-configuration gain/offset calibration, DOS broadening physics, and using discrete convolution that replicated staircase features. The residual KAN was compensating for systematic model-structural errors rather than learning small corrections. When an NN bridges a large gap between an incomplete physical kernel and measured data, it creates narrow local features to "chase" individual data points and produces oscillations switching between measurement families. Residual correction can adjust magnitude but cannot fundamentally change functional form (e.g., widen a sharp transition into a smooth one).
**Evidence**: CV kernel evolution required adding terminal parasitics, trap charge model, measurement calibration, and DOS broadening. The critical diagnostic: pure physical kernel CV had smooth, monotonic curves (no jumps/jitter); parameter-KAN added minimal distortion; residual-KAN introduced oscillations and willful curve twisting solely to lower pointwise error. The "three-step artifact" was definitively traced to discrete 3-point Gaussian convolution in the base CV kernel — a single sharp transition convolved with 3 shifted copies produced 3 visible staircase steps. Fix: single-transition charge kernel (no convolution). After all physical terms added, residual KAN only needed small smooth corrections.
**Generalization**: Neural network residual corrections have a fundamental limitation: they can add/subtract magnitude at given input points but cannot perform structural transformations (widening, smoothing, changing functional form) of the base model's output. When the base physical kernel is structurally incomplete, the NN will compensate by creating narrow local features, oscillations, and multi-step artifacts — "chasing" individual data points at the cost of global physical shape. This occurs whenever: (a) the physical kernel misses entire physical mechanisms (terminal parasitics, trap dynamics, charge partition), (b) the measurement data contains families (cgg/cbg/ctg/vds) that require different base model structures, (c) the base kernel has sharp transitions that the NN tries to widen by stacking shifted copies. Metrics alone (log-RMS) will NOT detect this — the NN overfit may produce lower RMS while the curve shape becomes visibly non-physical.
**Actionable Advice**:
1. Agent E should always compare **pure physical kernel CV curves** against **residual-corrected CV curves** side-by-side. If the residual-corrected version has MORE oscillations, shoulders, peaks, or non-monotonic steps than the pure physical version, the residual KAN is compensating for missing base physics — flag this as a kernel architecture defect, not a training issue.
2. Agent E should check whether each measurement family (cgg, cbg, ctg, vds) has its own explicit measurement-configuration mapping in the model. If a single "unified" CV expression is being forced to fit 4 different measurement configurations without configuration-level gain/offset calibration, the residual NN will oscillate between families.
3. Agent E should verify: can the pure physical kernel (before any NN correction) produce correct curve SHAPES even if magnitudes are off? If shapes are wrong (too sharp, wrong platform levels, multi-step), the kernel is structurally incomplete, and no amount of NN residual training will fix it gracefully.

---

### Derivative-Aware Loss: Physics Constraints Must Enter the Training Objective, Not Post-Hoc Filtering
**Category**: Principle Insight
**Stage(s)**: final-report (Agent E), train-static (Agent C)
**Symptom**: Models had good pointwise log-RMS metrics but visible non-physical behavior: sharp shoulders on CV, on-state IV curvature distortion, negative capacitance steps. The safe-scale selection mechanism screened residual KAN outputs after training to pick acceptable versions — but the training loss itself had no physics constraints on derivatives.
**Root Cause**: Training loss was dominated by pointwise log-RMS error. Derivatives (gm = dID/dVG, gds = dID/dVD, dC/dV, d^2C/dV^2) were computed only post-hoc for evaluation and safe-scale screening, not as training objectives. The residual KAN learned solutions that lowered pointwise error while creating local curvature artifacts. IEEE EDL compact model papers (Kao/Kam/Hu) explicitly include gm, gds, and output conductance in the training objective. The BSIM-NN/NQS framework requires charge derivatives to be continuous and monotonic as a hard constraint. AutoPINN and physics-informed NN literature consistently put smoothness and monotonic constraints into the training objective.
**Evidence**: After restructuring the loss to include ivd (IV derivative) and cvd (CV derivative) terms in every training step, with raw-data loss, denoised-derivative loss, and plateau loss as separate structured components, CV artifacts were reduced while maintaining accuracy. The shift from "screen after training" to "train with physics" eliminated the pattern where safe-scale always picked scale=0 because residual was never trained to respect derivatives.
**Generalization**: When physical constraints (monotonicity, smoothness, derivative continuity) are applied only as post-hoc filters rather than training objectives, the optimizer can converge to solutions that look good on pointwise metrics but violate physics in derivative space. This is especially dangerous for compact models destined for circuit simulation, where transconductance (gm) and output conductance (gds) determine amplifier gain and linearity. The condition is: any pipeline where (a) the loss function uses only pointwise error metrics, (b) derivative behavior is checked only in evaluation/acceptance code paths, (c) a "safe scale" or post-hoc selection mechanism replaces in-training constraints. The diagnostic signal: the pure physical kernel has smoother curves than the NN-corrected model.
**Actionable Advice**:
1. Agent E should check whether the training loss function includes terms for: (a) first derivatives (gm for IV, dC/dV for CV), (b) second derivatives (smoothness of gm/gds, d²C/dV²), (c) residual magnitude relative to physical kernel output. Absence of these terms means the model may pass metrics but fail circuit simulation.
2. Agent E should verify that any "safe scale" or "acceptance filtering" mechanism is NOT the primary source of physics enforcement. If removing the post-hoc filter causes the model to show non-physical artifacts, the training loss is insufficient.
3. Agent E should recommend that the pipeline report derivative-based metrics alongside pointwise metrics in the final report (e.g., gm RMS, gds RMS, max |dC/dV|, CV monotonicity violation count).

---

### CV Measurement Configuration Calibration: Per-Sheet Offset/Gain Before Intrinsic Fitting
**Category**: Principle Insight
**Stage(s)**: final-report (Agent E), fit-physical (Agent B)
**Symptom**: CTG and VDS CV curves were systematically elevated above measured values by the pure physical kernel, while CBG low-capacitance plateau was under-predicted. The residual KAN was being forced to compensate for large systematic biases rather than learning small corrections — producing oscillations and non-monotonic features.

**Root Cause**: Different CV measurement configurations have different parasitic contributions (pad capacitance, inter-gate coupling, open/short de-embedding residuals, AC participation factors) not captured by a unified intrinsic charge kernel. CTG needs ~0.65x measurement participation factor, VDS needs ~0.49x. Without per-configuration gain/offset calibration, the NN residual cannot gracefully correct systematic sheet-dependent biases.

**Evidence**: adding per-configuration gain+offset calibration was the single largest CV improvement — total CV RMS dropped from 0.183 to 0.129 in initial smoke, CTG from 0.121 to 0.034. Full training: CV RMS 0.1339→0.0953. The calibration parameters must be separated from device parameters in Verilog-A export — measurement wrapper vs physical model.
**Generalization**: Multi-configuration CV measurements contain systematic inter-family offset/scale differences that arise from measurement setup, not from device physics. When these are not explicitly calibrated out before intrinsic model training, the NN residual will be forced to compensate, producing non-physical artifacts. This is analogous to open/short de-embedding in RF measurement and should be treated as a pre-processing calibration step. The condition is: any CV model trained on multiple measurement configurations (cgg, cbg, ctg, vds) without per-configuration gain/offset calibration will produce systematic residual NN artifacts.
**Actionable Advice**:
1. Agent E should verify that the pipeline has explicit per-measurement-configuration gain/offset calibration before intrinsic CV model training. If the same intrinsic charge kernel is asked to fit all CV sheets with only global parameters, systematic inter-sheet biases will force NN residual artifacts.
2. Agent E should check whether CBG/CTG labels are physically consistent: for a device where top-gate oxide capacitance > bottom-gate oxide capacitance, CTG(high-V) > CBG(high-V) is expected. If the opposite is observed, the measurement labels may be swapped — flag this.
3. Agent E should recommend that the Verilog-A export separates measurement wrapper parameters (calibration gain/offset) from device intrinsic parameters (Cox, Vth, Cit). Only device intrinsic parameters belong in the compact model; measurement calibration belongs in the testbench or extraction script.

---

### NN Residual Overfitting to MSE: Lower Error Can Mean Worse Physics
**Category**: Principle Insight
**Stage(s)**: final-report (Agent E), train-static (Agent C)
**Symptom**: Multiple versions showed that adding residual KAN layers reduced CV RMS error while simultaneously creating non-physical curve features — oscillations, negative slopes, multi-step transitions. Lower-loss versions had visibly worse physical shape than higher-loss versions.

**Root Cause**: The residual KAN has high capacity (98 edges, multi-layer splines) that can fit localized data point noise by adding high-frequency local curvature. The MSE loss penalizes pointwise error but has no physics-informed counter-gradient against oscillations, negative derivative steps, or multi-modal transitions. The safe-scale mechanism (selecting a residual multiplier between 0 and 1) partially mitigates this post-hoc, but the training objective itself is wrong — it optimizes for the wrong target.

**Evidence**: v4.1 had lower CV RMS but "strong oscillations with negative slopes and jumps." v4.2 had even lower RMS but worse physical shape. The user's threshold: "don't pick the one with lowest loss, pick the one with best physical shape." After adding measurement-calibration + derivative loss + shape templates, CV artifacts were reduced while maintaining reasonable RMS — the right loss design matters more than model capacity.
**Generalization**: The optimizer minimizes whatever is in the loss function, not what the engineer intends. When the loss is exclusively pointwise log-RMS error, the NN will learn solutions that are locally accurate but globally non-physical. This is a degenerate optimization: any sufficiently expressive NN can reduce MSE by adding local oscillations because MSE has no physics gradient to counterbalance it. The condition is: any pipeline where (a) the loss function is exclusively magnitude-based (RMS, MAE, log-RMSE), (b) the correction network has high capacity relative to the data density, (c) individual measurement points have measurement noise that the NN can "fit" with local curvature. The diagnostic signal: models with lower validation loss have visibly worse curve shapes than models with slightly higher loss.
**Actionable Advice**:
1. Agent E should compare the **shape quality vs RMS error trade-off** across multiple residual scales or training checkpoints. If models with lower error have worse shape characteristics (more negative slope steps, more local peaks, higher second derivatives), the loss function is insufficiently physics-constrained.
2. Agent E should recommend that the loss function include: (a) DERIVATIVE terms (gm, gds, dC/dV) with comparable weight to value error, (b) SHAPE templates (monotonicity, single-transition, plateau flatness), (c) residual MAGNITUDE regularization (penalize large corrections relative to the physical kernel baseline).
3. Agent E should recommend that the pipeline report SHAPE-DERIVED metrics alongside value-derived metrics in the final report: negative slope step count, second derivative RMS, derivative peak count, plateau flatness score.

---

### Iteration Convergence Diagnosis: Structural Evolution vs Parametric Churn
**Category**: Principle Insight
**Stage(s)**: final-report (Agent E), all modeling stages
**Symptom**: The user paradoxically both demanded faster convergence AND repeatedly found major structural issues that required fundamental code changes.
**Root Cause**: Iteration chains fall into two categories: PRODUCTIVE (new physics terms added at each version — trap charge model, DOS broadening, single-transition CV kernel) vs PARAMETRIC CHURN (same code, only LR/epochs/weights change).
**Evidence**: Model development progressed through 6 distinct structural phases: (1) data prep, (2) EKV physical kernel fitting, (3) KAN architecture development, (4) CV kernel evolution (trap charge + terminal parasitic + DOS broadening), (5) loss function redesign (shape templates + derivative-aware + calibration), (6) commercial CV restructure (single-transition kernel, discrete convolution removal). Each phase added new model terms.
**Generalization**: Long iteration chains (10+ iterations, 50+ versions) can be either PRODUCTIVE or TRAPPED. The distinction is whether new versions add structural complexity (new model terms, new training phases, new calibration steps) or only change parameter values (different weights, different epoch counts). Structural evolution is a legitimate path to a better model when the initial architecture is incomplete; parametric churn is wasteful. The diagnostic signal: plot the model's structural complexity (number of free parameters in the physical kernel, number of KAN output channels, number of loss terms) against version number. If structural complexity plateaus but versions continue, it's parametric churn. If structural complexity continues to evolve, the iteration is genuine discovery.
**Actionable Advice**:
1. Agent E should classify each version in the operation_log by whether it introduced NEW MODEL STRUCTURE (new physical terms, new calibration stages, new KAN output channels, new loss components) or only changed EXISTING PARAMETERS. Report the ratio of structural changes to parametric changes.
2. Agent E should check whether the model's structural complexity (count of physical parameters, KAN layers/outputs, calibration terms) plateaued before the iteration stopped. If complexity stopped changing but tuning continued, flag as parametric churn.
3. Agent E should recommend that the pipeline track "structural version delta" — a summary of WHAT CHANGED between versions (new features added vs existing features retuned). This is different from tracking metric deltas.

---

### Training Duration Regression: Feature Accumulation Slows Iteration
**Category**: Principle Insight
**Stage(s)**: final-report (Agent E), train-static (Agent C)
**Symptom**: Training time grew from ~20 minutes to ~100+ minutes as features accumulated, without increasing epoch count. The slowdown came from incremental computational complexity additions.

**Root Cause**: Features added for physical correctness introduced overhead: discrete Gaussian convolution (3× CV computation per forward pass), full-curve derivative computation in main loss (previously evaluation-only), edge-wise symbolic replacement (98 edges × function search), safe-scale exhaustive search. No per-feature cost tracking. The fix: reduce convolution from 3-point to batched single-pass, change float64→float32, recovered ~2× speed.

**Evidence**: simple PKAN training completed in ~2.5 min. After adding convolution + derivative losses + symbolic search, full training reached ~102 min. Convolution alone added ~30-40% overhead. After removing discrete convolution, training returned to ~90-100 min.
**Generalization**: Incremental feature addition without profiling creates silent computational debt. Each new physics term, each new loss component, each new validation check adds to the per-iteration compute cost. When training runs take >1 hour, the iteration cycle becomes untenable — the developer gets <10 attempts per day. This is especially dangerous when each feature individually adds <20% overhead but 5+ such features double training time. The condition: any pipeline where (a) training time was initially fast but has grown over versions, (b) new "features" are added that recompute intermediate results (convolution, per-curve derivatives, exhaustive search), (c) no profiling or timing breakdown exists per training phase.
**Actionable Advice**:
1. Agent E should note the **total training duration** and compare it against previous runs. If duration has grown >2× between versions with similar epoch counts, there is computational bloat — flag it even if model quality improved.
2. Agent E should recommend that the pipeline profile training time by phase (warmup %, PKAN training %, residual training %, symbolization %, safe-scale search %) so that feature-specific slowdowns can be identified.
3. Agent E should flag any feature that requires PER-SAMPLE computation (e.g., per-curve derivative, per-measurement-configuration convolution) that wasn't present in earlier faster versions. These are the typical culprits of training slowdown.

---

### Diagnostic Memo

Key principles from practical development sessions: complete the physical kernel first, add measurement calibration second, add NN residual last with physics constraints, enforce derivatives in training (not post-hoc). Physical correctness over numerical accuracy — smooth physical curves are more deployment-valuable than lower RMS with residual-made oscillations. KAN evolved from black-box residual corrector to physical parameter modifier.

---

## Engine Architecture Diagnostics

Engine-level concerns that Agents B/C/D cannot fix via tools — documented here for Agent E's diagnostic reference.

### Derivative Observables (gm/gds, dC/dV) Not in Training Loss → Model Fails Deployment Despite Good RMS

**Symptom**: IV RMS and CV RMS metrics look acceptable, but output curves show physically implausible features: IDVTG on-region has unrealistic curvature (gm roll-off or sharp kinks), CV curves have local oscillations or multi-step transitions, residual KAN contributions visually "twist" curves in non-physical ways. These artifacts are invisible to RMS metrics.

**Root Cause**: The training loss minimizes only point-wise log-error (`log(ID_pred/ID_meas)`, `log(C_pred/C_meas)`). Derivatives (transconductance `gm`, output conductance `gds`, `dC/dV`) are computed only during post-training diagnostics and safe-scale screening — not as training objectives. The residual KAN overfits local curvature to reduce pointwise error, producing jagged `gm` and oscillatory `dC/dV`. The safe-scale post-hoc filter then responds by setting scale=0 (disabling residual entirely) because any non-zero residual introduces derivative artifacts.

**Why this blocks Agents B/C/D**: They cannot modify the loss function — it's engine code. They can only try `accept_quality` or `fallback_to_previous`.

**What should be fixed in the engine**: 
- Add `gm` and `gds` terms to IV loss (weight ~0.15-0.25 and ~0.05-0.10)
- Add `dC/dV` and `d²C/dV²` terms to CV loss on smoothed reference curves (not raw noisy data)
- Use base-aware relative constraints: `max(0, residual_derivative_worsening - base_derivative_worsening)`
- Do NOT delegate derivative enforcement to post-hoc safe-scale selection — it will always pick scale=0

**Evidence**: Multiple development iterations confirmed that safe-scale always picks scale=0 when derivative constraints are enforced only post-hoc — residual was never trained to respect them. After derivative terms entered the main loss, non-zero safe scales became achievable.

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

**Evidence**: CTG needed ~0.65x gain, VDS needed ~0.49x — confirming measurement-path, not device-physics, errors. 

### CV Charge Model Multi-Step Transitions from Discrete Convolution

**Symptom**: CV curves (especially CGG and CBG) show 2-3 distinct steps or "shoulders" in the OFF→ON transition. These appear as double-hump or triple-hump structures that don't correspond to any physical mechanism. Visible as multiple `dC/dV` peaks.

**Root Cause**: Three mechanisms at different development stages:
1. **Discrete Gaussian convolution (v45-v47)**: 3-point convolution on EKV sigmoid transition = 3 shifted copies = 3 visible steps
2. **Multiple additive charge sub-components**: `Q_total = Q_inv + Q_trap_fast + Q_trap_slow + Q_tail` — if each activates at different voltages, produces staircase transitions
3. **Residual KAN narrow branches**: Optimizer can discover parameter values producing multiple narrow capacitance bumps

**What should be fixed in the engine**: The base capacitance function `C(V) = dQ(V)/dV` must produce exactly ONE transition by construction. Do NOT use discrete convolution (`cap_broad = sum(w_i * cap(V - dV_i))`). Use single continuous transition with global physical parameters (`nfac`, `Cit`, `trap_width`) controlling width. Each additive charge sub-component must share the SAME effective threshold voltage. After any kernel change, plot `dC/dV` — any curve with >1 peak indicates false multi-transition.

**Evidence**: commercial CV compact models don't use discrete convolution. After replacing 3-point convolution with single-transition kernel, all CV curves reduced to exactly 1 derivative peak.

### CV Measurement Configuration Calibration — Per-Sheet Gain/Offset

**Symptom**: Systematic per-configuration bias — CTG and VDS consistently ~1.5-2.0x higher than measured across all bias points. Residual KAN wastes capacity trying to pull entire curves up/down.

**Root Cause**: Physical kernel computes intrinsic terminal capacitance. Measured CV includes open/short de-embedding error, contact parasitics, effective area uncertainty, and AC frequency response — different for each measurement configuration.

**Engine fix**: Pre-extract per-configuration `gain` + `offset` from OFF/ON plateaus before KAN training. Freeze these during training. Do NOT embed in Verilog-A device model — they're measurement-path properties. CTG typically 0.65x gain, VDS typically 0.49x gain.

**Evidence**: adding per-configuration gain+offset was the single largest CV improvement — total CV RMS 0.183→0.129 dec, CTG 0.121→0.034 dec in initial smoke.

### Separate IV/CV Residual Scales — Global Scale Kills Both When One Fails

**Symptom**: IV residual shows 50%+ improvement in IDVTG on-region, but safe-scale scan selects `scale = 0.0`, disabling all residuals. Per-scale table shows IV improves monotonically at every scale, but CV shape diagnostics flag minor artifacts.

**Root Cause**: Single global residual scale selector sees "CV not strictly improving" and zeros everything — including IV residual which was working perfectly.

**Engine fix**: Use two independent scales: `safe_iv_residual_scale` (IV metrics only) and `safe_cv_residual_scale` (CV metrics only). IV can be non-zero while CV stays at zero. Default CV residual to off — it provides marginal benefit (<0.005 dec RMS) for most devices.

**Evidence**: CV shape gate sensitivity caused single global scale to zero both IV and CV residuals — even though IV residual alone provided 50%+ improvement.

### CV Shape Template Constraints — Beyond Monotonicity

**Symptom**: CV curves pass monotonicity checks (`dC/dV >= 0`) but still have double-hump transitions, continued slow rise on high-voltage plateau, and residual KAN twist.

**Root Cause**: Monotonicity and smoothness constraints are insufficient. The correct CV shape is "plateau-transition-plateau" — two flat regions connected by a single smooth rise. Without template-based loss, the optimizer finds monotonic but non-plateau solutions.

**Engine fix**: Add template loss: unimodality penalty (exactly 1 peak in `dC/dV`), plateau flatness penalty (10x weight on `|dC/dV|` in top/bottom 25% of sweep), no continued rise penalty. Make shape loss base-aware: only penalize if residual is worse than base. Apply different weights by CV type (CGG/CBG highest priority).

**Evidence**: CGG double hump was traced to model artifacts, not device physics — template-based shape loss eliminated them.

### KAN Correction Form — Additive vs Multiplicative by Parameter Semantics

**Symptom**: Using purely multiplicative KAN corrections causes threshold-sensitive parameters (τ, V-equivalent shifts) to retain non-physical voltage dependence. Using purely additive causes dimensionless parameters (coupling factors, partition ratios) to produce values outside [0,1].

**Root Cause**: Different physical parameters have different natural ranges. Threshold shifts are additive (`Vth_corrected = Vth_base + ΔVth`). Mobility is multiplicative (`μ_corrected = μ_base × exp(dlog_μ)`). Partition ratios must stay in [0,1] (`p_corrected = sigmoid(logit(p_base) + dlogit_p)`).

**Engine fix**: Use mixed correction forms. For hysteresis SRH-KAN: `log10(τ)` additive, `log10(amplitude)` additive, dimensionless ratios bounded sigmoid. This is a future architecture consideration — current code uses direct prediction (raw τ and A values).

**Evidence**: Additive corrections (for thresholds/time-constants) and multiplicative corrections (for mobility/gain) serve different physical semantics and should not be mixed. The hysteresis SRH-KAN uses direct prediction (raw tau and A values) with mixed correction forms to be determined by future engine design.

### CBG/CTG Data Label Swap Detection

**Symptom**: CBG capacitance span >> CTG span, but top gate has higher Cox and should show larger capacitance. Visual inspection: curves labeled "CBG" look like CTG curves and vice versa.

**Detection**: Compute `R_Cox = Cox_t / Cox_b`. Compute `R_span = span_cbg / span_ctg`. If `(R_Cox > 1 and R_span < 1)` or `(R_Cox < 1 and R_span > 1)` → labels swapped.

**Evidence**: Cox_t/Cox_b = 2.92 but CBG_span/CTG_span = 3.27 — opposite of physical expectation. After correcting labels, CV RMS dropped from ~0.13 to 0.0745 dec. Engine already has auto-orientation via `v92_phase1_cv_gate_orientation.json` with configurable thresholds (CLI defaults: Cox ratio ≥1.40, span ratio ≥1.55).

(End of file - total 273 lines)
