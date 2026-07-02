# Static Model Knowledge

## 1. Global Context

### Pipeline Overview

The DG-TFT modeling pipeline has 7 stages:

```
data-prep → verifier → check-data → fit-physical → train-static → train-hysteresis → final-report
```

| Stage | Kind | Has Modeling Agent | Triggers Agent On |
|-------|------|--------------------|-------------------|
| data-prep | agent | No | — (always runs) |
| verifier | agent | No | — (always runs) |
| check-data | script | No (Agent A only) | Hard failure only |
| fit-physical | script | Yes (Agent B) | Hard failure / quality below threshold |
| **train-static** | **script** | **Yes (Agent C)** | **Hard failure / quality below threshold** |
| train-hysteresis | script | Yes (Agent D) | Hard failure / quality below threshold |
| final-report | script | No (Agent E) | — (always runs at end) |

Agent stages spawn LLM subagents. Script stages run subprocesses — fit-physical uses scipy (CPU-only), train-static and train-hysteresis use CUDA Python.
When a script stage hard-fails (exit code ≠ 0) or produces quality below threshold,
a **modeling agent** is spawned to adjust parameters and retry, or fallback upstream.

Fallback is a linear one-hop chain upstream: D→C→B→A→Verifier→Preparer.

### train-static Position and Role

The **train-static** stage is Stage 5 of 7 — the **second and most complex modeling stage**.
It trains the full static IV+CV model by combining the physical kernel (from fit-physical)
with a PKAN (Physics-informed Kolmogorov-Arnold Network) residual network that learns
device-specific corrections. The output is a Verilog-A compact model suitable for
circuit simulation.

This stage runs via subprocess: `CUDA python engine/dg_tft_full_model.py`.
It is controlled by the `[full_model]`, `[base_pkan]`, `[temperature]`, `[symbolic]`, and `[validation]` INI sections
(five sections, 71 adjustable parameters). Timeout is 7200s (2 hours).

On failure, **Agent C** is spawned with three decision tools (`adjust_params`, `fallback_to_previous`,
`accept_quality`) plus one investigation tool (`inspect_xlsx`). Agent C's fallback target is **fit-physical**.

### Principles: PKAN Residual Modeling

**What is a KAN?** A Kolmogorov-Arnold Network decomposes a multivariate function into
a sum of univariate B-spline activations along learned edges, unlike a standard MLP
which learns fixed activations at nodes. KANs produce interpretable, compact symbolic
formulas and can achieve high accuracy with fewer parameters on low-dimensional physics
regression problems.

**What is PKAN?** "Physics-informed KAN" — the framework packages trained KAN weights
and normalizer metadata alongside physical parameters extracted by the fit-physical stage.
The PKAN does not replace physics; it learns small **additive corrections** to named
physical parameters (Vth, nfac, mu, fringe capacitances, charge trap parameters) using
a CompactKAN network with frozen B-spline knot grids (v96).

**Base PKAN vs. Residual PKAN architecture:**

| Component | Network | Input Features | Output |
|-----------|---------|---------------|--------|
| **IV-PKAN** (`base.kan`) | CompactKAN | `Ut_V, Ub_V, Veff_V, VDS_V, qbar0, VDSxqbar0` (6 physics-derived features) | 11 named IV parameter corrections (dpsi_V, dVth_t_eff_V, dVth_b_eff_V, dlog_mu_dec, dlog_theta_dec, dlambda, dlog_ileak_dec, dlog_Cit_dec, dpartition, dpartition_inv, dgamma_intergate) |
| **CV-PKAN** (`base.cv_kan`) | CompactKAN | Same 6 input features | 10 named CV charge parameter corrections (dlog_Cit_t, dlog_Cit_b, dVfb_t, dVfb_b, dVtrap_center, dlog_tau, dlog_cpl_t, dlog_cpl_b, dlog_Cbody, dlog_wch) — but 6 of these are disabled by default (see `DEFAULT_DISABLED_CV_PKAN`), leaving only 4 active: dVfb_t, dVfb_b, dlog_Cit_t, dlog_Cit_b |
| **IV residual** (`diag.iv_resid`) | CompactKAN | Similar features | 1 scalar: `dlog_id_residual` (additive log-current correction) |
| **CV residual** (`diag.cv_q_resid`) | Single CompactKAN with module routing | Same 6 features | A single CompactKAN with 21 output slots (`CV_RESID_PARAM_NAMES` = 12 param + 9 resid-only). Of these, only **7 are active in release** (`CV_RELEASE_ACTIVE_NAMES`): `dlog_cit_common_raw`, `miss_amp_raw`, `miss_center_raw`, `miss_width_raw`, `miss_bottom_amp_raw`, `miss_bottom_center_raw`, `miss_bottom_width_raw`. The remaining 14 are frozen. Outputs are routed through multiple physical modules (shared broadened-DOS, asymmetric terminal redistribution, inter-gate reciprocal coupling, CV PKAN parameter corrections). |
| **Temperature KAN** (`diag.temperature_kan`) | CompactKAN | `dT_100K, Veff_V, VDS_V, qbar0, dT100_x_qbar0` (5 scaled temperature features) | 7 physics-specific temperature modulation outputs: `dalpha_common_V_per_K` (threshold TC), `dalpha_asym_V_per_K` (asymmetric TC), `dEa_mu_eV` (mobility activation energy), `dmu_power` (mobility exponent), `dEa_ileak_eV` (leakage activation energy), `dcit_tc_per_K` (trap capacitance TC), `dEa_tau_eV` (NQS time-constant activation energy). This is a multiplicative modulator scaling base PKAN outputs, not an additive correction.

All KANs use `CompactKAN` — a pykan wrapper with `forward(x) = tanh(raw) × out_range × scale`,
frozen grid knots (v96), and parameter-name-aware normalization. The grid size (default=5),
spline degree k (default=3), and hidden layer size (default=12) control the B-spline
resolution and model capacity.

**How the model combines IV and CV loss:** The total loss is a weighted sum with a
`cv_weight` multiplier (default 10.0). CV loss terms dominate numerically — CV is a
derivative observable far more sensitive to spline ripple than IV. A staged loss
schedule (`staged_loss_schedule=True`) evolves the IV/CV weight ratio across training
phases: early epochs emphasize IV shape quality, middle epochs introduce CV with
increasing weight, and late epochs balance both.

**Symbolic optimization:** After training, the KAN edges are replaced with Top-1 symbolic
formulas from a library of primitive functions (sin, exp, tanh, abs, polynomial basis, etc.).
This edgewise symbolic replacement (`EdgewiseSymbolicKANNet`) converts the numeric KAN
into a Verilog-A-compatible analytic formula. A hard precision gate
(`base_iv_max_top1_compatibility`, default 0.015 MSE) ensures only edges whose symbolic
approximation error is below a strict threshold are deployed. Failure gates at the
numeric→symbolic boundary can trigger Agent C to adjust symbolic weights.

**Temperature-dependent modeling:** If the dataset spans ≥ `minimum_span_K` (default 20K),
a separate `temperature_kan` network is enabled (mode="auto"). This network learns how
the PKAN output scale varies with device temperature. It is a multiplicative modulator,
not an additive correction — it scales the base PKAN outputs by a learned factor in
[0.1, 1.0] × `kan_scale`. The physical temperature model in the EKV kernel handles the
first-order temperature dependence; the temperature KAN captures residual device-specific
deviation.

## 2. Upstream/Downstream

### Receives from fit-physical

- `01_physical_kernel/fitted_params.json` — final EKV parameters (used via `--reset-base-from-physical-fit` to rebuild the current-device physical base)
- `01_physical_kernel/dg_tft_ekv_level2_fitted.va` — physical kernel Verilog-A template
- A verified dataset directory containing IV workbooks and `CV_train.xlsx`
- `model_config.ini` — runtime config with `[full_model]`, `[base_pkan]`, `[temperature]`, `[validation]`, `[symbolic]`, `[preprocessing]` sections

### Delivers to train-hysteresis

- `03_full_model/v93_safe_residual_state.pt` — accepted PyTorch checkpoint (key artifact)
- `03_full_model/v93_safe_residual_metrics.json` — contains `residual_iv_rms` and `residual_cv_rms` quality metrics
- `03_full_model/dg_tft_static_compact.va` — final Verilog-A compact model (standalone, no dynamic NQS)
- `03_full_model/v92_safe_scale_scan.csv` — safety scale sweep results
- `03_full_model/v92_safe_shape_diagnostics.csv` — shape/passivity check results
- `03_full_model/v92_cv_parameter_head_ablation.csv` — CV PKAN head ablation study
- `03_full_model/v92_cv_gate_orientation.json` — CV gate swap diagnostic
- `03_full_model/iv_stage_alignment_report.csv` / `.json` — IV alignment results
- `03_full_model/v92_cv_kernel_decomposition.csv` — CV charge component breakdown
- `model_config.ini` — potentially modified by Agent C's `adjust_params`

### Fallback Chain

- Agent C falls back to **fit-physical** (StageName.FIT_PHYSICAL), which is upstream.
  This means: if the static model fails because the physical kernel parameters are
  poor (e.g., Vth estimates are off by >0.5V, mobility is unreasonable, or the EKV
  kernel fundamentally cannot describe the device), fallback re-runs the physical fit.
- When train-hysteresis falls back to train-static: the static model checkpoint is
  regenerated, and hysteresis training restarts from the new accepted checkpoint.

## 3. INI Parameters

All parameters below are from `FULL_MODEL_PARAMS` in `param_defs.py`. The stage reads
`[full_model]`, `[base_pkan]`, and `[temperature]` INI sections — 63 parameters total (37 in [full_model] + 8 [base_pkan] + 6 [temperature] + 10 [symbolic] + 2 [validation] = 63 table entries; 71 after param_defs merge of symbolic+validation into FULL_MODEL_PARAMS).

### [full_model] — Primary Training Parameters

| Parameter | Type | Default | Bounds | Description |
|-----------|------|---------|--------|-------------|
| `epochs` | int | 700 | [100, 5000] | Total training epochs for the joint IV+CV residual stage (after base IV PKAN and IV residual stages). Each epoch iterates over IV and CV batches. Higher values allow the optimizer to settle but consume runtime. **Increase if**: residual_iv_rms or residual_cv_rms is still decreasing at epoch 700 (check metrics JSON). **Decrease if**: early stopping consistently fires at ~200 epochs — 500 may suffice. |
| `learning_rate` | float | 0.0012 | [0.0001, 0.01] | Base learning rate for AdamW optimizer during joint IV+CV training. Applied to all trainable parameters (PKAN spline weights, residual KAN, CV platform/warming/coupling params). **Decrease to 0.0005 if**: loss oscillates or spikes (check logs for NaN). **Increase to 0.003 if**: loss plateaus early with no improvement. |
| `batch_iv` | int | 4096 | [512, 16384] | Batch size for IV training samples during joint training. Larger batches stabilize gradient estimates; smaller batches add stochasticity that can help escape local minima. **Reduce if**: CUDA OOM (IV tensors include VBG/VTG/VDS × all bias points). **Increase if**: gradient noise causes loss oscillations. |
| `batch_cv` | int | 512 | [64, 4096] | Batch size for CV training samples during joint training. Typically 1/8 of `batch_iv` because CV per-sample memory cost is higher (CV involves charge differentiation which expands the computation graph). **Reduce if**: CUDA OOM. |
| `cv_weight` | float | 10.0 | [1.0, 50.0] | Weight multiplier for the CV loss term relative to IV loss in the combined objective. CV is a derivative observable — its loss magnitude is naturally smaller than IV log-error, so it needs a booster to contribute meaningfully. **Increase if**: `residual_cv_rms` > 0.06 threshold. **Decrease if**: `residual_iv_rms` degrades (CV loss dominates and pulls the model away from a good IV fit). Values above 30.0 may cause the optimizer to ignore IV entirely. |
| `cv_residual_level` | int | 1 | [0, 2] | Complexity level of the CV residual correction sub-network. **0** = completely disabled (returns zero tensors). **1** = both **shared broadened-DOS** (`cv_shared_resid_amp_raw`) AND **asymmetric terminal redistribution** (`cv_asym_resid_*`) enabled simultaneously. **2** = shared + asymmetric + **inter-gate reciprocal coupling** (`cv_tb_residual`). Individual branches are gated by their respective factor parameters (`cv_shared_residual_factor`, `cv_asym_residual_factor`, `cv_tb_residual_factor`) — each must be non-zero for the branch to receive gradients. **Max=2** (engine argparse restricts to choices=[0,1,2]). **Increase to 2 if**: CV RMS is high and CV plots show inter-gate coupling artifacts. **Set to 0 if**: CV residual introduces instability (check for NaN in `residual_cv_rms`). |
| `enable_cv_residual` | bool | True | — | Master switch for the CV residual learning branch. When True, the `diag.cv_q_resid` CompactKAN is active. When False, CV correction comes only from the base CV-PKAN parameter corrections. **Disable only if**: you suspect the CV residual KAN is causing instability or overfitting. Typically should remain True. |
| `finetune_base_pkan` | bool | False | — | Whether to fine-tune the base IV-PKAN weights during joint training. INI default is `False` (the pipeline always passes this explicitly: `--finetune-base-pkan` if INI says True else `--no-finetune-base-pkan`). Note: the engine's standalone argparse default is `True`, but this is overridden in pipeline context by the INI-supplied value. When **False**: freeze the base PKAN weights only, but continue to train `phys_delta` (29 physical micro-delta parameters, see below), `cv_pair_cpar`, `cgg_common`, `cgg_tail`, `cv_gate_coupling_raw`, and the IV/CV residual KANs. When **True**: the base PKAN splines also train alongside. **In warm-start mode** (resume from prior fit, `reset_base_from_physical_fit=False`): uses `warm_start_base_pkan_lr_mult=0.035` instead of `base_pkan_lr_mult=1.0` — a 29× smaller learning rate to protect the pretrained PKAN from early destruction when CV loss is introduced. |
| `early_stop_patience` | int | 5 | [2, 30] | Number of consecutive epochs with no validation loss improvement before early stopping activates. **Increase to 10-15 if**: the loss curve shows slow but steady decline that patience=5 would prematurely cut off. **Decrease to 2-3 if**: runtime is a bottleneck and convergence is fast. |
| `early_stop_min_epochs` | int | 220 | [50, 500] | Minimum epochs before early stopping is allowed to trigger. This ensures the model has a baseline of training regardless of loss trajectory. The staged loss schedule needs time to ramp through its phases. **Increase if**: model is under-trained at epoch 220 (check if loss is still declining steeply). **Decrease if**: loss plateaus by epoch 100 and runtime matters. |
| `symbolic_training_weight` | float | 0.05 | [0.0, 0.1] | Weight for the symbolic training loss term, which regularizes KAN edges toward deploying Top-1 symbolic formulas. Higher values push KAN edges to learn functions that can be replaced by analytic primitives (sin, exp, tanh, polynomial) without degrading accuracy. **Increase to 0.08-0.10 if**: symbolic equivalence gate fails (edge MSE > `base_iv_max_top1_compatibility` after training). **Decrease to 0.0 if**: symbolic export is not required for this run. |
| `print_every` | int | 10 | [1, 200] | Print training loss and metrics to stdout every N epochs. Larger values reduce log noise. Set to 1 for full visibility during debugging. |
| `eval_every` | int | 40 | [5, 200] | Evaluate full model metrics (IV/CV RMS, shape quality, symbolic compatibility) every N epochs. Evaluation requires a forward pass over all validation data — larger values reduce this overhead. **Decrease to 20 if**: you need finer progress tracking for tuning. |
| `staged_loss_schedule` | bool | True | — | Enable staged loss-weight evolution across training phases. When True, loss weights for IV shape, CV fitting, and symbolic regularization evolve according to `shape_early_period` / `shape_middle_period` / `shape_eval_period`. **Disable only for debugging** — the staged schedule is essential for the model to learn IV shape quality before introducing the destabilizing CV derivative loss. |
| `shape_eval_period` | int | 5 | [1, 20] | Period (epochs) for shape evaluation during the late training phase. Smaller values trigger shape-loss computation more frequently. **Increase to 10 if**: shape evaluation dominates training time. |
| `shape_early_period` | int | 20 | [1, 50] | Shape evaluation period during early training (epochs 0 to ~30% of total). Early epochs focus on IV log-error; shape is checked less frequently to avoid premature shape constraints. |
| `shape_middle_period` | int | 10 | [1, 50] | Shape evaluation period during middle training phase. Intermediate between early (coarse) and late (fine) shape checking. |
| `weight_decay` | float | 1e-6 | [1e-8, 1e-3] | AdamW weight decay (L2 regularization). Prevents large spline coefficients that cause non-monotonic IV curves. **Increase to 1e-5 if**: fitted IV curves show unrealistic wiggles (overfitting to noise). **Increase further to 1e-4 if**: spline knot coefficients are growing unbounded. |
| `grad_clip` | float | 5.0 | [0.1, 50.0] | Maximum gradient norm for gradient clipping. Prevents exploding gradients that can cause loss spikes or NaN. **Decrease to 1.0-2.0 if**: training log shows `loss=nan` or `grad_norm=1e6+`. **Increase to 10.0 if**: gradients are consistently clipped (training stalls). |
| `lr_eta_min_frac` | float | 0.02 | [0.001, 0.5] | Minimum learning rate as a fraction of initial LR (CosineAnnealingLR scheduler). The LR decays from `learning_rate` to `learning_rate × lr_eta_min_frac` over the full epoch budget. **Increase to 0.1 if**: the scheduler drops the LR too low and training stops improving before epochs are exhausted. **Decrease to 0.005 if**: you want a deeper annealing for fine convergence. |
| `cv_train_subset` | string | primary_gate_cv | enum["all","representative","primary_gate_cv"] | CV training data selection. `primary_gate_cv` uses only the quasi-static gate CV sheets (CBG, CTG, CGG). `all` includes all gate capacitance sheets including VDS. `representative` selects representative CV series. **Use `all` if**: VDS capacitance contains useful information and you want the model to fit it. Note: VDS capacitance is typically small/noisy; fitting it may degrade gate CV quality. |

### [full_model] — Base IV PKAN Training (runs before residual/CV)

| Parameter | Type | Default | Bounds | Description |
|-----------|------|---------|--------|-------------|
| `base_iv_epochs` | int | 700 | [100, 5000] | Epochs for the dedicated base IV-PKAN training phase. This trains the IV parameter KAN on the current device's IV data before any CV or residual training. The base IV PKAN has its own precision gate (`base_iv_max_top1_compatibility`) that must pass before proceeding. **Increase if**: base IV PKAN precision gate fails (MSE > 0.015). **Decrease if**: the device is similar to the pre-trained checkpoint and fewer epochs suffice. **Note**: Engine standalone argparse default is 540, and run_model.py fallback uses `'540'`. The INI default (from param_defs) is 700. If the INI key is accidentally missing, the effective default drops to 540 epochs. |
| `base_iv_restarts` | int | 1 | [1, 10] | Number of random restarts for base IV PKAN training. Each restart uses a different random seed and the best result (lowest Top-1 compatibility loss) is kept. **Increase to 3-5 if**: training converges to poor local minima (the Top-1 symbolic edge MSE varies significantly between runs). Each restart costs `base_iv_epochs` of training time. |
| `base_iv_learning_rate` | float | 0.002 | [0.0001, 0.01] | Learning rate for base IV PKAN training (separate from the joint training `learning_rate`). **Decrease if**: base IV loss oscillates. **Increase if**: base IV loss plateaus at high values (>0.05 RMS). |
| `base_iv_eval_every` | int | 20 | [5, 100] | Evaluation frequency during base IV training. Every N epochs, the PKAN is evaluated for Top-1 symbolic compatibility and RMS error. **Decrease to 10 if**: you want finer tracking to debug the precision gate. |
| `base_iv_top1_symbolic_weight` | float | 0.35 | [0.0, 1.0] | Weight for the Top-1 symbolic regularization loss during base IV training. Controls how strongly the PKAN is pushed toward deploying Top-1 symbolic edges. **Increase if**: edgewise symbolic replacement fails the precision gate (MSE > 0.015). **Decrease if**: the symbolic prior distorts the base IV fit (RMS too high). |
| `base_iv_top1_sparsity_weight` | float | 0.001 | [0.0, 0.1] | Sparsity penalty for Top-1 edge selection. Higher values push toward fewer active symbolic edges, producing simpler exported equations. **Increase if**: exported formulas are too complex (many terms per edge). |
| `base_iv_top1_selection_weight` | float | 0.08 | [0.0, 0.5] | Tie-breaker weight when selecting base IV checkpoints. Higher values prefer simpler edge libraries over slightly lower error. |
| `base_iv_max_top1_compatibility` | float | 0.015 | [0.0, 0.1] | **Hard precision gate**: maximum MSE per edge for Top-1 symbolic replacement. If any edge exceeds this threshold, the base IV PKAN training is considered failed and the pipeline cannot proceed. **Relax to 0.03 if**: the current device requires flexible symbolic forms that the Top-1 library cannot capture at 0.015. **Tighten to 0.005 if**: deployment requires extremely precise symbolic equivalence. This is a critical gate parameter — adjusting it changes whether the stage succeeds at all. |

### [full_model] — IV Residual Stage (held-out IV correction)

| Parameter | Type | Default | Bounds | Description |
|-----------|------|---------|--------|-------------|
| `iv_residual_stage_epochs` | int | 700 | [100, 5000] | Epochs for the held-out IV residual training stage. This trains the `diag.iv_resid` CompactKAN on IV data held out from the base PKAN training (stride-based holdout). **Increase if**: residual_iv_rms is too high after this stage. |
| `iv_residual_stage_learning_rate` | float | 0.002 | [0.0001, 0.01] | Learning rate for the IV residual stage. Typically matches the base IV LR. **Adjust if**: residual training oscillates or converges too slowly. |
| `iv_residual_stage_linear_weight` | float | 0.25 | [0.0, 1.0] | Linear (absolute) term weight in IV residual loss. Higher values emphasize matching the absolute current values (A), not just log-error (decades). **Increase if**: the residual model systematically under/over-predicts in the on-region (high current). **Decrease if**: focusing on log-error gives better shape quality. |
| `iv_residual_stage_deriv_curves` | int | 36 | [1, 100] | Number of derivative curves to supervise during IV residual training. The residual KAN learns shape consistency by matching numerically differentiated ID curves. **Increase if**: residual predictions have unrealistic curvature (non-monotonic segments). **Decrease if**: derivative supervision adds too much overhead. |
| `iv_residual_stage_shape_period` | int | 3 | [1, 20] | Shape evaluation period during IV residual training. Every N epochs, the residual model is checked for shape quality (monotonicity, negative tolerance, slope bounds). **Decrease to 1-2 if**: shape quality needs tight monitoring. |

### [full_model] — Training Schedule Controls

| Parameter | Type | Default | Bounds | Description |
|-----------|------|---------|--------|-------------|
| `base_pkan_lr_mult` | float | 1.0 | [0.01, 10.0] | Base PKAN learning-rate multiplier during joint IV+CV training. Applied on top of `learning_rate`. When `finetune_base_pkan=False` this has no effect (the base PKAN is frozen). When `finetune_base_pkan=True`, values > 1.0 let the base PKAN adapt faster. **Reduce to 0.1 if**: you want gentle fine-tuning of the base PKAN without overwriting its pre-trained accuracy. |
| `warm_start_base_pkan_lr_mult` | float | 0.035 | [0.001, 1.0] | Initial LR multiplier for base PKAN during warm-start (first ~15% of epochs). A very small value (0.035) protects the pretrained PKAN from early destruction when CV loss is first introduced. The multiplier ramps from this value to `base_pkan_lr_mult` over the warm-up period. **Increase to 0.1 if**: the base PKAN needs more initial adaptation to CV data. **Use only when `finetune_base_pkan=True`**. |

### [base_pkan] — KAN Hyper-parameters

These parameters control the architecture of the CompactKAN networks. They affect all KAN instances (IV-PKAN, CV-PKAN, IV residual, CV residual) since all use the same base width/grid/k settings. Changes to these parameters require rebuilding KAN models — they cannot be hot-adjusted during training.

**Important**: These `[base_pkan]` parameters are **NOT passed as CLI arguments** to the train-static subprocess by `run_model.py`. They are embedded in the pre-trained PKAN checkpoint (`pkan_torch_model.pt`). Adjusting them via `adjust_params` writes them to the INI but has **no effect on per-device train-static training**. They are only effective when building/retraining the pre-trained base PKAN from scratch (a separate workflow not using the orchestrator).

| Parameter | Type | Default | Bounds | Description |
|-----------|------|---------|--------|-------------|
| `epochs_shared` | int | 500 | [100, 3000] | Shared KAN training epochs for the base PKAN network (used during the pre-trained model preparation, not the per-device training). **Increase if**: base PKAN symbolic formulas look under-fitted at the packaged checkpoint level. |
| `epochs_iv` | int | 700 | [100, 3000] | IV-specific KAN training epochs for the packaged base PKAN. |
| `epochs_cv` | int | 700 | [100, 3000] | CV-specific KAN training epochs for the packaged base PKAN. |
| `kan_hidden` | int | 12 | [4, 32] | Hidden layer size in the CompactKAN. The KAN width vector is `[N_features, kan_hidden, kan_hidden, N_outputs]`. **Increase to 16-24 if**: the device has unusual transfer characteristics that the KAN cannot capture with 12 hidden units. **Decrease to 8 if**: training is memory-bound or the device behavior is simple. Warning: changing `kan_hidden` invalidates pre-trained checkpoints. |
| `kan_grid` | int | 5 | [2, 20] | B-spline grid resolution. Each edge's activation function is represented by a B-spline with this many grid points. **Increase to 8-12 if**: the device has sharp non-linear transitions (e.g., steep subthreshold swing) that low-resolution splines miss. **Decrease to 3 if**: device curves are smooth and you want faster training. Grid points are frozen in v96 (not learned), so this is an architecture parameter. |
| `kan_k` | int | 3 | [2, 5] | B-spline degree/order. k=3 (cubic) is standard — cubic splines are C² continuous and handle most device physics. **Increase to 4 if**: fitting requires higher-order continuity (very smooth derivatives). **Decrease to 2 if**: you want piecewise linear edges for the simplest possible exported equations. |
| `batch_iv_kan` | int | 4096 | [256, 16384] | Batch size for IV KAN training during base PKAN preparation (separate from `full_model.batch_iv`). **Reduce if**: OOM during KAN training. |
| `batch_cv_kan` | int | 512 | [64, 4096] | Batch size for CV KAN training during base PKAN preparation (separate from `full_model.batch_cv`). |

### [temperature] — Temperature KAN Control

**INI key mapping**: `param_defs.py` uses prefixed keys (`temperature_mode`, `temperature_kan_hidden`, etc.) as agent-facing names, with `ini_key` overrides mapping them to the actual INI keys (`mode`, `kan_hidden`, etc.). `apply_agent_adjustments` respects these overrides, so agent adjustments now correctly write to the INI keys that `run_model.py` reads.

| Parameter | Type | Default | Bounds | Description |
|-----------|------|---------|--------|-------------|
| `temperature_mode` | string | auto | enum["auto","on","off"] | Temperature KAN mode. `auto` — enable when measured temperature span ≥ `minimum_span_K`. `on` — force enable (ignores span check). `off` — disable entirely. **Set `on` if**: the dataset has temperature data but the auto-detection fails (e.g., temperature column not found or span is underestimated). **Set `off` if**: you want purely isothermal modeling even when multi-temperature data exists. |
| `minimum_span_K` | float | 20.0 | [1.0, 100.0] | Minimum temperature span (Kelvin) required for auto mode. The engine checks the min and max temperature values across all IV and CV data columns. **Decrease to 10.0 if**: you have a small temperature range (e.g., 300K to 315K) but want temperature-aware modeling. **Increase to 50.0 if**: you only want temperature KAN for wide-range datasets with clear temperature trends. |
| `kan_scale` | float | 0.35 | [0.1, 1.0] | Temperature KAN output scaling factor. The temperature KAN output is multiplied by `kan_scale`, producing a factor in [0.1×kan_scale, 1.0×kan_scale] that modulates the base PKAN output amplitude. **Increase to 0.5 if**: temperature effects are pronounced (the device shows >2× current variation over temperature). **Decrease to 0.15 if**: temperature effects are subtle and you want conservative modulation. |
| `temperature_kan_hidden` | int | 8 | [4, 32] | Hidden layer size for the temperature KAN sub-network. This is a separate architecture from the main PKAN. **Increase to 12-16 if**: temperature dependence is complex (non-monotonic, different slopes for different regimes). |
| `temperature_kan_grid` | int | 3 | [2, 20] | Grid resolution for temperature KAN B-splines. Since temperature is a 1D input (or 2D with device class), a small grid is usually sufficient. **Increase to 5 if**: temperature response is sharply non-linear. |
| `temperature_kan_k` | int | 3 | [2, 5] | Spline degree for temperature KAN. k=3 is standard. **Increase to 4 if**: you need smoother temperature interpolation. |

### [symbolic] — Symbolic Export Parameters

These parameters control the symbolic optimization pipeline that converts trained KAN edges into Verilog-A-compatible analytic formulas. They are passed to the engine by `run_model.py` from the `[symbolic]` INI section.

| Parameter | Type | Default | Bounds | Description |
|-----------|------|---------|--------|-------------|
| `edge_steps` | int | 180 | [10, 2000] | Optimization steps for edgewise symbolic replacement. More steps allow finer symbolic fitting. **Increase if**: Top-1 compatibility gate fails (MSE > threshold). |
| `edge_topk` | int | 1 | [1, 3] | Top-K symbolic edge replacement (1=strict, 2=relaxed). Top-2 applies to CV/temperature edges; IV edges promote when `iv_adaptive_top2_mse > 0`. |
| `iv_edge_topk` | int | 1 | [1, 3] | Top-K symbolic replacement for IV edges specifically. When >1, edges above `iv_adaptive_top2_mse` use Top-2. **Increase if**: IV symbolic edges fail precision gate. |
| `iv_adaptive_top2_mse` | float | 0.0015 | [0.0, 0.1] | MSE threshold for adaptively promoting IV edges to Top-2. Set 0.0 for strict Top-1. **Increase to 0.005-0.01 if**: IV device has unusual transfer characteristics. |
| `iv_calibration_max_terms` | int | 24 | [1, 100] | Max calibration terms in IV symbolic optimization. Higher = more fine-tuning terms in Verilog-A. **Increase if**: deployment fidelity targets are not met post-export. |
| `edge_finetune_steps` | int | 80 | [10, 2000] | Steps for per-edge symbolic fine-tuning. **Increase if**: edge MSE marginally above `base_iv_max_top1_compatibility`. |
| `network_finetune_steps` | int | 360 | [10, 5000] | Steps for full-network symbolic joint fine-tuning. **Increase if**: network-level observables degrade after edgewise replacement. |
| `device_iv_finetune_steps` | int | 3000 | [100, 10000] | Steps for device-specific IV calibration against measured data. **Increase if**: device-level IV predictions show systematic errors post-export. |
| `device_iv_measured_steps` | int | 5000 | [100, 20000] | Steps for measured-data IV symbolic calibration. Fine-tunes formulas against actual measurement curves. **Increase for**: stubborn IV prediction errors. |
| `device_cv_finetune_steps` | int | 650 | [10, 5000] | Steps for CV symbolic calibration. **Increase if**: exported CV predictions have systematic shape errors. |

### [validation] — Safety/Quality Scan Parameters

| Parameter | Type | Default | Bounds | Description |
|-----------|------|---------|--------|-------------|
| `safe_scale_steps` | int | 8 | [2, 50] | Number of scale steps in safety scan for residual output scales. More steps = finer-grained safety envelope. **Increase if**: safe scale is coarsely estimated (large jumps). |
| `joint_cv_safe_scale_search` | bool | False | — | Enable 2D grid search for (parameter_scale, residual_scale) combinations. **Set True if**: CV passivity gate fails with independently scanned scales. |

### minimal_run Override

When `[runtime] minimal_run = yes`, the engine applies `--quick-run` which drastically reduces budgets: epochs=1, base_iv_epochs=1, iv_residual_stage_epochs=1, early_stop_patience=1, early_stop_min_epochs=1, print_every=1. The stage skips HSPICE, edgewise symbolic export, and safety scale scans.

### retry adjustment strategies (auto, non-agent)

When quality fails and no Agent decision is made (retry loop), the auto-adjustment strategies are:
- Attempt 1: `epochs` +50%, `learning_rate` ÷2
- Attempt 2: `cv_weight` ×2.0
- Attempt 3: `early_stop_patience` ×2

These are `quality_guard.py` strategies, applied only when no modeling agent is available.

## 4. Quality Metrics

### residual_iv_rms

- **What it measures**: The RMS log10 error (decades) between measured IV current and the full static model prediction (physical kernel + PKAN corrections + IV residual), evaluated on cross-condition holdout IV data (stride-based split). A value of 0.35 means typical prediction error is ~0.35 decades (2.2× in absolute current).
- **Source file**: `03_full_model/v93_safe_residual_metrics.json` → key `residual_iv_rms`
- **Threshold from agent_configs.json**: `DG_TFT_QUALITY_STATIC_IV_RMS_MAX` = **0.35** (decades)
- **What "below threshold" means**: The static model's IV prediction error is ≤ 0.35 decades, matching the physical kernel's threshold. Since this is a KAN-residual model on top of the physical kernel, it is expected to meet or exceed (lower) the physical kernel's accuracy.
- **Which parameters typically affect it**:
  - **Primary**: `epochs` — insufficient residual training is the most common cause
  - **Secondary**: `base_iv_epochs` — if the base IV PKAN is undertrained, the residual has more to correct
  - **Tertiary**: `iv_residual_stage_epochs`, `iv_residual_stage_linear_weight` — held-out IV residual stage parameters
  - **Architecture**: `kan_hidden`, `kan_grid` — if KAN capacity is insufficient for complex devices
- **Additional related metrics** (from `v93_safe_residual_metrics.json`, diagnostic only):
  - `iv_rms_log10_error_dec`: full-data IV RMS (not holdout)
  - `iv_subthreshold_rms_log10_error_dec`: RMS in subthreshold region
  - `iv_on_rms_log10_error_dec`: RMS in on-region
  - `iv_nonmonotonic_steps`: count of non-monotonic IV predictions
  - `safe_iv_residual_scale`: accepted IV residual output scale (0=disabled)
  - `base_iv_training.*`: base IV PKAN training report (per-restart metrics, Top-1 compatibility)
  - `iv_residual_training.*`: IV residual training report (heldout gain, region degradation)

### residual_cv_rms

- **What it measures**: The RMS log10 error (decades) between measured CV capacitance and the full static model prediction (physical kernel + PKAN CV parameter corrections + CV charge residual), evaluated on the training CV data with safe-selected scales.
- **Source file**: `03_full_model/v93_safe_residual_metrics.json` → key `residual_cv_rms`
- **Threshold from agent_configs.json**: `DG_TFT_QUALITY_STATIC_CV_RMS_MAX` = **0.06** (decades)
- **Why 0.06 vs 0.35 for IV**: CV is a derivative observable — the capacitance is `dQ/dV`. This means the CV error is naturally much smaller in log-space than IV error. A residual_cv_rms of 0.06 means ~15% relative error in capacitance, which is tight for physically consistent charge-based CV prediction.
- **Which parameters typically affect it**:
  - **Primary**: `cv_weight` — the dominant lever for CV quality. Increasing cv_weight forces the optimizer to prioritize CV loss. But be careful: values > 30.0 can cause IV degradation.
  - **Secondary**: `cv_residual_level` — higher levels (2) allow more CV correction degrees of freedom
  - **Tertiary**: `epochs`, `learning_rate`, `staged_loss_schedule` — more training or different scheduling
  - **CV polish stages**: `cv_residual_polish_epochs`, `cv_neural_polish_epochs`, `cv_level2_polish_epochs` — these run after main training and can rescue borderline CV fits
- **Additional related metrics** (from `v93_safe_residual_metrics.json`):
  - `safe_cv_residual_scale`: accepted CV residual output scale
  - `safe_cv_parameter_scale`: accepted CV PKAN parameter correction scale
  - `selected_cv_residual_mode`: "off" | "shared_only" | "persisted_active_factors"
  - `selected_cv_residual_factors.*`: per-branch factor values
  - `cv_vhalf_trend_diagnostics`: vhalf shift trends across bias conditions
  - `cv_platform_amp_F`, `cv_platform_center_V`, `cv_platform_width_V`: gentle strong-accumulation tail parameters
  - `cv_sheet_gain`, `cv_sheet_offset_F`: per-sheet calibration (open/short de-embedding equivalent)
  - `cv_frequency_coverage.*`: multi-frequency diagnostics
  - `cv_acceptance_diagnostics`: per-sheet, per-series acceptance analysis

## 5. Engine Behavior

### Script Entry Point

```
<CUDA python> run_model.py train-static --config <model_config.ini>
```

Which internally constructs and runs:

```
<CUDA python> engine/dg_tft_full_model.py
  --data-dir-override <data_dir>
  --model-dir <kan_architecture>
  --baseline-json <01_physical_kernel/fitted_params.json>
  --reset-base-from-physical-fit (or not if resume)
  --out-dir <03_full_model>
  --device <auto|cpu>
  --dtype <float32|float64>
  --seed <seed>
  --epochs <full_model.epochs>
  --lr <full_model.learning_rate>
  --base-pkan-lr-mult <full_model.base_pkan_lr_mult>
  --warm-start-base-pkan-lr-mult <full_model.warm_start_base_pkan_lr_mult>
  --batch-iv <full_model.batch_iv>
  --batch-cv <full_model.batch_cv>
  --cv-weight <full_model.cv_weight>
  --cv-train-subset <full_model.cv_train_subset>
  --cv-residual-level <full_model.cv_residual_level>
  --early-stop-patience <full_model.early_stop_patience>
  --early-stop-min-epochs <full_model.early_stop_min_epochs>
  --symbolic-training-weight <full_model.symbolic_training_weight>
  --print-every <full_model.print_every>
  --eval-every <full_model.eval_every>
  --shape-eval-period <full_model.shape_eval_period>
  --shape-early-period <full_model.shape_early_period>
  --shape-middle-period <full_model.shape_middle_period>
  --base-iv-epochs <full_model.base_iv_epochs>
  --base-iv-restarts <full_model.base_iv_restarts>
  --base-iv-lr <full_model.base_iv_learning_rate>
  --base-iv-top1-symbolic-weight <full_model.base_iv_top1_symbolic_weight>
  --base-iv-top1-sparsity-weight <full_model.base_iv_top1_sparsity_weight>
  --base-iv-top1-selection-weight <full_model.base_iv_top1_selection_weight>
  --base-iv-max-top1-compatibility <full_model.base_iv_max_top1_compatibility>
  --iv-residual-stage-epochs <full_model.iv_residual_stage_epochs>
  --iv-residual-stage-lr <full_model.iv_residual_stage_learning_rate>
  --iv-residual-stage-linear-weight <full_model.iv_residual_stage_linear_weight>
  --iv-residual-stage-deriv-curves <full_model.iv_residual_stage_deriv_curves>
  --iv-residual-stage-shape-period <full_model.iv_residual_stage_shape_period>
  --weight-decay <full_model.weight_decay>
  --grad-clip <full_model.grad_clip>
  --lr-eta-min-frac <full_model.lr_eta_min_frac>
  --temperature-kan-min-span-K <temperature.minimum_span_K>
  --temperature-kan-scale <temperature.kan_scale>
  --temperature-kan-hidden <temperature.kan_hidden>
  --temperature-kan-grid <temperature.kan_grid>
  --temperature-kan-k <temperature.kan_k>
  --max-iv-points 50000
  --max-cv-points 100000
  --iv-stage-alignment <preprocessing.iv_stage_alignment>
  --iv-stage-max-shift-V <preprocessing.iv_stage_max_shift_V>
  --iv-stage-min-matches <preprocessing.iv_stage_min_matches>
  [--finetune-base-pkan|--no-finetune-base-pkan]
  [--enable-cv-residual|--disable-cv-residual]
  [--enable-staged-loss-schedule|--disable-staged-loss-schedule]
  [--train-base-iv-pkan|--skip-base-iv-pkan]
  [--freeze-iv-residual-after-stage|--joint-finetune-iv-residual]
  [temperature mode flags]
  [HSPICE flags]
  [symbolic-config flags --edge-steps, --symbolic-device-iv-finetune-steps, etc.]
```

### Step-by-Step Execution Flow

**Phase 0 — Model Initialization** (always runs, not skippable):

1. **Load pre-trained base model** (`load_trained_base`): Loads the packaged PYtorch Kan Architecture checkpoint containing the CompactKAN model architecture, trained normalization weights (center/scale for each input feature), and output ranges. This is a versioned `v96_compactkan_architecture` checkpoint with frozen spline grids.

2. **Rebuild from physical fit** (if `--reset-base-from-physical-fit`): Creates a fresh current-device physical+PKAN model by loading the fitted_params.json from the fit-physical stage, patching EKV parameters into the physical kernel, and initializing KAN correction heads at zero. The old-device learned KAN weights are NOT reused — only the architecture and normalizer are preserved.

3. **Load IV and CV data**: Reads IV workbooks (`IDVBG*_train.xlsx`, `IDVTG*_train.xlsx`, `IDVD*_train.xlsx`) and `CV_train.xlsx`. Applies IV stage alignment. Performs CV gate auto-orientation (swap CBG↔CTG if orientation is wrong). Selects CV training subset (primary gate CV or all gate CV). Extracts CV terminal parasitics.

4. **CV preprocessing**: Attaches centered shape templates, performs quality audit (drops outlier series), applies quasi-static frequency policy (weighting multi-frequency data toward the reference frequency if present).

5. **Data sampling**: IV sampling via weighted random subsampling (same algorithm as fit-physical, with subthreshold/on-region emphasis). CV sampling via uniform linspace subsampling. Creates PyTorch tensors for training.

6. **Build diagnostic model** (`ResidualDiagnostic`): Creates the complete trainable model with:
   - Base physical kernel (UnifiedIVCVModel with IV-PKAN and CV-PKAN)
   - IV residual KAN (`diag.iv_resid`) — 1 output (dlog_id_residual)
   - CV parameter residual KAN (`diag.cv_q_resid`) — a single CompactKAN with 21 output slots, of which **only 7 are active in release** (see Architecture table above). The 14 frozen outputs have no effect regardless of KAN hyperparameters.
   - Temperature KAN (`diag.temperature_kan`) — if enabled
   - CV platform stabilizer (gentle strong-accumulation tail)
   - CV sheet calibration (measurement de-embedding)
   - CV DOS broadening, held-gate coupling, charge participation
   - CV dynamic trap/NQS module (if enabled)

7. **CV warm-start initialization**: Initializes CV platform parameters, reciprocal charge mapping, and sheet calibration from raw data before training. These pre-fits provide good initial guesses that prevent the optimizer from getting stuck.

**Phase 1 — Base IV PKAN Training** (`train_current_device_base_iv_pkan`):

The base IV-PKAN is trained independently on the current device's IV data, with `base_iv_restarts` random restarts. Each restart:
- Trains the `base.kan` CompactKAN for `base_iv_epochs` epochs
- Optimizes with AdamW, CosineAnnealingLR scheduler
- Evaluates Top-1 symbolic compatibility, real-parameter RMS, transfer curve shape quality
- Has its own early-stopping (patience=8, min_epochs=260)
- The best restart (lowest Top-1 compatibility loss) is kept

The base IV precision gate checks: if any edge's Top-1 symbolic MSE > `base_iv_max_top1_compatibility` (default 0.015), the training fails with a RuntimeError.

If `--base-iv-only`: the pipeline exits after this phase (used for iterative base KAN development).

**Beyond the quality gate — engine-internal precision gates that cause hard failures**:

- **Base IV region precision gate**: 7 region-specific RMS checks (iv_nonzero_rms≤0.105, iv_on_rms≤0.060, iv_high_rms≤0.055, iv_off_rms≤0.450, iv_subthreshold_rms≤0.125, iv_transition_rms≤0.135, iv_nonzero_p95≤0.220). ANY failure raises RuntimeError.
- **IV residual heldout gain gate**: `iv_residual_min_heldout_gain=8e-4` — residual must improve held-out data by at least this.
- **IV residual region degradation gate**: `iv_residual_max_region_degradation=0.008` — no region may degrade more than this.
- **KAN grid audit** (`assert_valid_grids`): All 4 KANs must pass grid integrity checks. Failure = corrupted model, unrecoverable.
- **Charge conservation/passivity**: Multiple tolerance checks on charge sum, cross-reciprocity, diagonal negativity.

Agent C must know WHICH gate failed to choose the right action: grid audit failure → fallback (unrecoverable); base IV region gate → adjust base_iv_epochs/restarts; IV residual gain gate → adjust iv_residual_stage_* params.

**Phase 2 — Held-Out IV Residual Training** (`train_current_device_iv_residual`):

The `diag.iv_resid` CompactKAN is trained on held-out IV data (stride-based cross-condition split). This stage:
- Freezes the base PKAN and CV PKAN
- Trains only the 1-output IV residual KAN for `iv_residual_stage_epochs` epochs
- Supervision: log-error + linear (absolute) error + derivative curve shape loss
- Has a held-out gain gate: if the residual doesn't improve held-out data by at least `iv_residual_min_heldout_gain` (8e-4), the stage reports failure
- Has a region degradation gate: no region (subthreshold/on) may degrade more than `iv_residual_max_region_degradation` (0.008)

If `--iv-residual-only`: the pipeline exits after this phase.

**Phase 3 — Joint CV/Residual Training** (`train_residuals`):

The full diagnostic model is trained end-to-end (or with frozen base PKAN if `--no-finetune-base-pkan`, which is the v96 default). This is the main training phase:

- Optimizer: AdamW with cosine annealing from `learning_rate` → `learning_rate × lr_eta_min_frac`
- Gradient clipping at `grad_clip`
- Weight decay `weight_decay`
- Early stopping: patience=`early_stop_patience`, min_epochs=`early_stop_min_epochs`
- **Training Phase Architecture** — Two independent scheduling mechanisms control training:

  1. **Staged Loss Evaluation Schedule** (boundaries at 0.30/0.70 of total epochs): Controls **when** shape evaluation, CV derivatives, anchor loss, and reciprocity checks run — not what parameters are trained, but how frequently they're evaluated. `loss_schedule_early_end=0.30`, `loss_schedule_middle_end=0.70`.

  2. **CV Training Phase Architecture** (boundaries controlled by 4 `full_model` parameters):
     - `cv_parameter_ramp_start` (default=0.25) through `cv_parameter_ramp_end` (default=0.58): Trains **global CV physical parameters** (DOS, body, trap, platform, sheet calibration). Global CV physics parameters freeze and bias-dependent PKAN activates at `cv_parameter_ramp_start`.
     - `cv_residual_ramp_start` (default=0.62) through `cv_residual_ramp_end` (default=0.82): **Freezes CV parameter KAN**, trains only residual output edges. At 0.62 onward, only `cv_q_resid` active outputs receive gradients.
     - `cv_parameter_train_max_scale`: Controls maximum CV PKAN training amplitude.
   Note: These 4 boundary parameters are engine-internal only (NOT in param_defs.py, NOT passed by run_model.py, NOT adjustable via `adjust_params`). They control critical CV convergence behavior but Agent C cannot tune them. If CV quality is persistently poor despite parameter changes, these hardcoded ramp boundaries may be the root cause.

- Staged loss schedule (when `staged_loss_schedule=True`):
  - Early phase (0–30% epochs): focus on IV log-error, shape quality, base PKAN consistency. CV is present with a small ramp weight.
  - Middle phase (30–70% epochs): CV weight ramps up to `cv_weight`, symbolic regularization ramps from `symbolic_training_weight × ramp_start` to `symbolic_training_weight × ramp_end`
  - Late phase (70–100% epochs): full CV weight, full symbolic weight, shape monitoring with short periods
- Shape evaluation at configured periods: `shape_early_period` → `shape_middle_period` → `shape_eval_period`
- Metrics printed every `print_every` epochs, full eval every `eval_every` epochs

**Phase 4 — Post-Training CV Polish** (unless `--skip-post-train-cv-polish`):

Four sequential polish stages refine CV quality:
1. `polish_cv_neural_corrections`: Polish CV DOS/pDoping/envelope neural parameters
2. `polish_cv_charge_residual`: Polish the CV charge residual KAN on training data
3. `polish_cv_level2_intergate`: Fit reciprocal inter-gate charge-PKAN controls
4. `polish_cv_held_gate_coupling`: Fit eta_bt/eta_tb transition-center coupling

Each polish stage uses its own learning rate, epochs, and loss weighting — these are engine-internal (not INI parameters for this version).

**Phase 5 — Safety Evaluation and Scale Selection:**

1. **CV parameter head ablation** (`prune_harmful_cv_parameter_heads`): Test each CV PKAN output head independently. Heads that individually degrade IV or CV quality are set to zero and reported.

2. **Safe CV parameter scale selection**: Scan the CV PKAN output scale from 0.0 to `base_scale × 2.0` in `safe_scale_steps` steps (default 8). Select the maximum scale that does not degrade IV or CV passivity (negative capacitance, overshoot).

3. **Joint CV safe scale search** (if `joint_cv_safe_scale_search=yes`): Coarse 2D scan over (parameter_scale, residual_scale). Ensures the combination of CV PKAN parameters + CV charge residual doesn't violate constraints.

4. **Safe residual scale selection** (`choose_safe_residual_scales`): Scan IV residual scale and CV residual scale independently. The selected scale is the maximum that passes shape/passivity checks on the safety holdout split. A failed first attempt at the selected scale triggers a finer search downward.

5. **KAN grid audit**: Verify all 4 KAN networks (base IV, base CV, IV residual, CV residual) pass grid integrity checks (no knot inversion, no zero-variance grids). Failure here raises RuntimeError — the model has a corrupted KAN state.

**Phase 6 — Metrics, Diagnostics, and Export:**

1. **Predict and summarize**: Full forward pass on evaluation IV/CV data with safe-selected scales. Compute all metrics (RMS, per-region breakdown, vhalf trends, temperature trends, etc.).

2. **Write metrics JSON**: `v93_safe_residual_metrics.json` (or `v92_safe_residual_metrics.json` for legacy naming). Contains `residual_iv_rms`, `residual_cv_rms`, all safe scales, all diagnostic blocks.

3. **CV diagnostics**: CV acceptance per-sheet analysis, CV kernel decomposition (which components contribute how much charge), CV vhalf trend analysis.

4. **Symbolic optimization** (unless `--skip-edgewise-symbolic`):
   - Edgewise symbolic replacement: fit Top-1 analytic formulas to each KAN edge
   - Network-level finetuning: fine-tune the resulting symbolic network
   - Device-level finetuning: refine on the current device's measured IV data
   - Calibration: bounded perturbation of selected terms to match deployment fidelity targets
   - Equivalence gate: the symbolic model must match the numeric model within tolerances

5. **Verilog-A export** (`export_complete_veriloga_device`): Generate a standalone Verilog-A compact model file (`dg_tft_static_compact.va`). The VA file contains the EKV physical kernel equations, the symbolic/edge-spline PKAN correction terms, and the charge law with CV residual corrections.

6. **HSPICE validation** (if HSPICE available and not skipped): Smoke test, IV grid equivalence (Python predicted vs HSPICE simulated), CV grid equivalence.

7. **Save checkpoint**: `v93_safe_residual_state.pt` (key artifact for orchestration). Contains the full diagnostic model state_dict, output policy, safe scales, and metadata.

### How Parameters Change Execution

- **`epochs` + `base_iv_epochs` + `iv_residual_stage_epochs`**: The three-phase epoch budget. Total training time is roughly `base_iv_epochs × base_iv_restarts + iv_residual_stage_epochs + epochs`. At default values: 700×1 + 700 + 700 = 2100 total epochs. Each epoch iterates over IV and CV batches — ~0.5–2 seconds per epoch on a modern GPU, giving 20–70 minutes of training. The 7200s timeout provides ample margin.

- **`cv_weight` vs. `symbolic_training_weight`**: These two weights compete. High `cv_weight` pushes CV accuracy; high `symbolic_training_weight` pushes symbolic deployability. If both are high, the optimizer may be pulled in conflicting directions. The staged loss schedule manages this tension by ramping symbolic regularization later.

- **`finetune_base_pkan` = False (v96 default)**: The base IV-PKAN is completely frozen during joint training. Only the IV residual KAN, CV PKAN, CV residual KAN, and physical calibration parameters are trained. This is the safe default — it preserves the base IV precision gate's acceptance.

- **`staged_loss_schedule` = True**: The loss evolves across phases. Early epochs prioritize IV shape for the PKAN; later epochs balance IV and CV. Disabling this (False) applies fixed weights throughout, which can cause CV loss to dominate from epoch 1 and destroy any chance of a good IV fit.

- **Temperature KAN interaction**: When enabled, the temperature KAN modulates base PKAN output scales based on temperature. This affects ALL predictions — IV and CV both see temperature modulation. The temperature KAN is trained jointly; its scale is clamped to [0.1, 1.0] × `kan_scale` for stability.

### minimal_run Behavior

When `--quick-run` is active (triggered by `minimal_run=yes`):
- epochs = 1, base_iv_epochs = 1, iv_residual_stage_epochs = 1
- early_stop_patience = 1, early_stop_min_epochs = 1, print_every = 1
- Safety scale scans still run but with 1 epoch of training — results are not meaningful
- HSPICE, symbolic export, and release packaging are skipped
- This is for smoke-testing pipeline infrastructure, not modeling quality

## 6. Failure Patterns

> This section will be populated from Zikang's Codex JSONL logs during the knowledge extraction phase.

### Error Classification (from diagnostics.py)

When train-static exits with non-zero code, keyword classification is applied:

| Keyword in stderr | Classification | Confidence |
|-------------------|---------------|------------|
| `Cuda out of memory`, `out of memory`, `OOM`, `cuDNN` | TRAINING_NUMERICAL_ERROR | 0.9 |
| `NaN`, `inf`, `overflow encountered`, `underflow encountered`, `overflow` | TRAINING_NUMERICAL_ERROR | 0.9 |
| `gradient`, `backward` | TRAINING_NUMERICAL_ERROR | 0.9 |
| `diverged`, `not converging`, `loss is nan`, `loss too large`, `gradient explosion` | TRAINING_NUMERICAL_ERROR | 0.7 |
| `dataset_validation.json` | DATA_ERROR | 0.9 |
| `FileNotFoundError`, `No such file or directory`, `does not exist` | DATA_ERROR | 0.9 |
| `data_dir not found`, `cannot find it directory` | DATA_ERROR | 0.9 |
| `missing column`, `missing sheet`, `invalid sheet`, `sheet not found`, `column not found` | DATA_ERROR | 0.9 |
| `unexpected header`, `empty workbook`, `no valid data`, `xlsx file is empty` | DATA_ERROR | 0.9 |
| `excel file format cannot be determined` | DATA_ERROR | 0.9 |
| `KeyError`, `IndexError`, `ValueError` | DATA_ERROR | 0.6 |
| `configparser`, `no section`, `no option`, `invalid config`, `missing config` | CONFIG_ERROR | 0.9 |
| `configuration error` | CONFIG_ERROR | 0.9 |
| `unknown stage`, `unknown action` | CONFIG_ERROR | 0.9 |
| `hspice.exe`, `hspice not found`, `hspice executable`, `hspice path` | HSPICE_ENV_ERROR | 0.9 |
| `hspice simulation failed`, `hspice timeout`, `hspice license` | HSPICE_ENV_ERROR | 0.9 |
| No keyword match, exit code ≠ 0 | TRAINING_NUMERICAL_ERROR | 0.4 |
| Exit code = 0 but key artifacts missing | MISSING_ARTIFACT_ERROR | 0.9 |

The stage suggestion: "Try adjusting static training hyperparameters or checking for NaN/inf in the training data."

### Engine-Specific Error Messages

The engine emits specific RuntimeError messages at different failure points. Agent C should recognize these and act accordingly:

| Error Message | Root Cause Gate | Recovery | Agent Action |
|---------------|----------------|----------|--------------|
| `Top-1 compatibility MSE {x} > {threshold}` | `base_iv_max_top1_compatibility` — symbolic edge too inaccurate | Adjustable | `adjust_params`: relax `base_iv_max_top1_compatibility` or increase `base_iv_epochs`/`base_iv_top1_symbolic_weight` |
| `iv_nonzero_rms {x} > {threshold}` | Base IV region precision gate — nonzero region RMS | Adjustable | `adjust_params`: increase `base_iv_epochs` / `base_iv_restarts` |
| `iv_on_rms {x} > {threshold}` | Base IV region gate — on-region RMS | Adjustable | `adjust_params`: increase `base_iv_epochs` / `base_iv_restarts` |
| `iv_off_rms {x} > {threshold}` | Base IV region gate — off-region RMS (hardest) | Adjustable | `adjust_params`: increase `base_iv_epochs` significantly; if persistent, `fallback` |
| `iv_subthreshold_rms {x} > {threshold}` | Base IV region gate — subthreshold RMS | Adjustable | `adjust_params`: increase `base_iv_epochs` |
| `iv_residual heldout gain {x} < min_gain {threshold}` | IV residual heldout gain gate | Adjustable | `adjust_params`: increase `iv_residual_stage_epochs` / `iv_residual_stage_linear_weight` |
| `Region degradation {x} > max_degradation {threshold}` | IV residual region degradation gate | Adjustable | `adjust_params`: increase `iv_residual_stage_epochs`; check CV orientation |
| `grid audit failed` / `invalid grid` | KAN grid integrity — corrupted spline knots | **Unrecoverable** | `fallback_to_previous` — KAN model state is corrupted, no parameter fix possible |
| `CV charge sum exceeds tolerance` / `charge conservation` | Charge conservation gate | Hard to fix | `adjust_params`: reduce `cv_weight` or `cv_residual_level`; if persistent, `fallback` |
| `negative capacitance` / `passivity violation` | CV passivity gate | Adjustable | `adjust_params`: reduce `cv_residual_level` to 1 or 0, or increase `safe_scale_steps` |
| `CUDA out of memory` / `OOM` | Hardware resource | Adjustable | `adjust_params`: reduce `batch_iv`/`batch_cv`, reduce `kan_hidden` (needs checkpoint rebuild) |
| `NaN` / `inf` in loss | Numerical instability | Adjustable | `adjust_params`: reduce `learning_rate`, increase `grad_clip` (tighter), reduce `cv_weight` |
| `FileNotFoundError: data_dir` / `No such file` | Missing upstream data | **Unrecoverable** | `fallback_to_previous` — data integrity issue, not a parameter problem |
| `CV gate orientation swap` | CV terminal misidentification | Adjustable if caught | Engine auto-corrects; if it fails, `fallback_to_previous` |

### Agent C Decision Framework

Agent C has four tools and must call EXACTLY ONE:

| Tool | When to Use | Consequences |
|------|-------------|--------------|
| `adjust_params` | Parameter tuning can fix the problem. Use when: (1) quality is close to threshold (e.g., residual_iv_rms = 0.40 vs threshold 0.35), (2) the base IV precision gate failed (Top-1 MSE > 0.015) and you can relax the gate or increase training budget, (3) CV RMS is high but not catastrophic (>0.06 but <0.15), (4) training appeared to converge but quality is borderline — increase epochs or adjust cv_weight. Make incremental changes: +25% epochs, not ×5; +50% cv_weight, not ×10. | INI is updated, stage re-runs with new parameters. Costs a retry slot. |
| `fallback_to_previous` | The upstream outputs are the problem. Use when: (1) the physical kernel (fit-physical) produced poor parameters — Vth values are at bounds, mobility is 10× off expected, or the fit history shows rejected stages, (2) the fitted_params.json is missing or corrupted, (3) multiple parameter adjustments on train-static have yielded no improvement, suggesting the physical kernel fundamentally cannot model this device, (4) CV orientation was diagnosed as wrong but cannot be corrected by the engine's auto-orientation. | Stage re-runs from fit-physical, which regenerates the physical kernel. Costs the most time (fit-physical + train-static rerun). |
| `accept_quality` | Current quality is acceptable despite being above threshold. Use when: (1) residual_iv_rms is close to 0.35 (e.g., 0.37) and the IV plots look good qualitatively, (2) residual_cv_rms is only slightly above 0.06 (e.g., 0.07) and CV plots match well, (3) the base IV precision gate barely failed (Top-1 MSE = 0.017 vs 0.015) and you judge the symbolic quality acceptable, (4) all adjustment attempts have been exhausted. | Pipeline moves to train-hysteresis. Quality warning is recorded but does not block progress. |
| `inspect_xlsx` | Investigation only — use freely to examine file structure before deciding. Does not consume a decision. Returns `{"status": "success"\|"fail", "sheet_count": N, "sheets": {"Name": {"columns": [...]}}}`. |

### Diagnostic Tools Available to Agent C

Agent C can use `read_file` and `glob_file` (standard SubAgent tools) to inspect:

- `03_full_model/v93_safe_residual_metrics.json` — **most important file**. Contains:
  - `residual_iv_rms`, `residual_cv_rms` — the quality gate metrics
  - `safe_iv_residual_scale`, `safe_cv_residual_scale`, `safe_cv_parameter_scale` — deployment scales
  - `selected_cv_residual_mode`, `selected_cv_residual_factors` — which CV residual branches are active
  - `base_iv_training` — per-restart base IV metrics, Top-1 compatibility per edge
  - `iv_residual_training` — heldout gain, region degradation
  - `cv_vhalf_trend_diagnostics`, `temperature_trend_diagnostics`
  - `cv_platform_amp_F`, `cv_platform_center_V` — physical platform parameters
  - `disabled_iv_pkan_outputs`, `disabled_cv_pkan_outputs` — which PKAN heads were pruned
- `03_full_model/v92_safe_scale_scan.csv` — safety scale sweep: for each IV/CV residual scale, the resulting IV RMS, CV RMS, and pass/fail flags
- `03_full_model/v92_safe_shape_diagnostics.csv` — per-scale shape quality (monotonicity, negative tolerance)
- `03_full_model/v92_cv_parameter_head_ablation.csv` — which CV parameter heads cause degradation
- `03_full_model/v92_cv_gate_orientation.json` — whether CTG↔CBG were swapped
- `01_physical_kernel/fitted_params.json` — upstream physical kernel quality (iv_rms_log10_error_dec, fit_history, fitted parameter values)
- `model_config.ini` — current parameter values for all sections
- Error logs in the output directory or stderr capture
