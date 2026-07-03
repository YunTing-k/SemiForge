# Physical Kernel Knowledge

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
| **fit-physical** | **script** | **Yes (Agent B)** | **Hard failure / quality below threshold** |
| train-static | script | Yes (Agent C) | Hard failure / quality below threshold |
| train-hysteresis | script | Yes (Agent D) | Hard failure / quality below threshold |
| final-report | noop | No (Agent E) | — (always runs at end) |

Agent stages spawn LLM subagents. Script stages run subprocesses — fit-physical uses scipy (CPU-only, no CUDA dependency), while train-static and train-hysteresis use CUDA Python.
When a script stage hard-fails (exit code ≠ 0) or produces quality below threshold,
a **modeling agent** is spawned to adjust parameters and retry, or fallback upstream.

Fallback is a linear one-hop chain upstream: D→C→B→A→Verifier→Preparer.

### fit-physical Position and Role

The **fit-physical** stage is Stage 4 of 7 — the **first modeling stage**. It performs
**EKV Level-2 parameter extraction** by fitting the analytical physical model to
measured IV and CV data using `scipy.optimize.least_squares`.

This stage runs via subprocess: `<python> dg_tft/engine/fit_physical_kernel.py`.
The engine is pure CPU — imports numpy, scipy, pandas, matplotlib (no torch, no CUDA).
It is controlled by the `[physical_fit]` INI section (max_iv_points, max_cv_points,
max_nfev_scale) and reads `[preprocessing]` for IV stage alignment settings.

On failure, **Agent B** is spawned with four stage-specific tools: `adjust_params`, `fallback_to_previous`,
`accept_quality`, and `inspect_xlsx`. It also has all standard SubAgent tools (`read_file`, `glob_file`,
`bash`, `grep_file`, `write_file`, `edit_file`). Agent B's fallback target is **check-data**.

> **CRITICAL — Knowledge Injection Status**: This knowledge file is currently NOT injected into Agent B's system prompt. `_spawn_modeling_agent_inner` (orchestrator.py) builds a generic prompt with error text and tool options but contains zero engine-specific knowledge. Agent B starts blind — it must discover everything through `read_file` calls. Until the knowledge loader is integrated (TODO Step 9), this document's tuning guidance is for human reference only.

### Principles: EKV Level-2 Parameter Extraction

**EKV Formalism**: The engine uses a charge-sheet-based EKV (Enz-Krummenacher-Vittoz)
formalism adapted for double-gate TFT devices. Unlike simple threshold-voltage models,
EKV describes all regions of operation (subthreshold to strong inversion) with a single
set of equations, making it especially suitable for TFTs where the transition between
off and on states is gradual.

**Double-Gate Physics**: The device has two independent gates — top-gate (TG) and
bottom-gate (BG) — with distinct oxide capacitances (`Cox_t`, `Cox_b`) and threshold
voltages (`Vth_t0`, `Vth_b0`). A weighted overdrive voltage combines contributions
from both gates:
```
Vov = (Cox_t/(Cox_t+Cox_b))·Vov_t + (Cox_b/(Cox_t+Cox_b))·Vov_b
```
Each gate has knee and modulation saturation parameters (`Vknee_t/b`, `Vmodsat_t/b`).

**qsolve — Pinch-Off Charge**: The function `qsolve(y)` solves `2q + ln(q) = y`
via 6 Newton iterations. This is the normalized mobile charge density at the source (qs)
and drain (qd), avoiding the explicit piecewise approximations of classical models.
Strong inversion uses a linear approximation; weak inversion uses the Boltzmann exponential.

**34 Physics Parameters**: The `Params` dataclass holds 34 fields covering:
- **Geometry**: W (40 μm), L (50 μm)
- **Gate oxides**: Cox_t, Cox_b (derived from ε_ox / t_ox)
- **Thresholds**: Vth_t0, Vth_b0
- **Subthreshold**: nfac0 (non-ideality factor)
- **Saturation**: Vknee_t/b, Vmodsat_t/b (velocity saturation and knee smoothing)
- **Mobility**: mu0 (low-field), theta_mu (mobility degradation)
- **Channel-length modulation**: lambda_clm
- **Leakage**: Ioff0 (off-state floor), Gleak_ds (drain-source leakage conductance)
- **Series resistance**: Rs0, Rd0 (source/drain resistances; iterative solution when >10Ω)
- **Temperature**: Tnom (reference), kvt_t/b (Vth temp coeff), ute (mobility exponent),
  Ea_mu, knfac, trs, Ea_rs (activation energies for mobility and RS)
- **Fringe capacitances**: Cov_tgs_w, Cov_tgd_w, Cov_bgs_w, Cov_bgd_w (per-width overlap),
  Cfr_tgs, Cfr_tgd, Cfr_bgs, Cfr_bgd (fixed fringe)

Not all 34 parameters are fitted. Only **12 unique parameters** are actually optimized
across the 6 fitting stages (Vth_t0, Vth_b0, nfac0, mu0, Ioff0, Gleak_ds, theta_mu,
lambda_clm, Cfr_tgs, Cfr_tgd, Cfr_bgs, Cfr_bgd). The remaining parameters (W, L, Tnom,
Rs0, Rd0, Vknee_t/b, Vmodsat_t/b, Cox_t/b, Cov_*_w, temperature coefficients knfac,
trs, Ea_rs, Ea_mu, ute, kvt_t/b) are held at default values — either fixed physical
constants (geometry, temperature), intentionally unoptimized (saturation parameters
kept large to approximate "no saturation" in TFT), or only relevant for
non-isothermal simulations (temperature coefficients).

**Least-Squares Optimization**: `scipy.optimize.least_squares` with `loss="soft_l1"`
(Huber-like robust regression) and `x_scale="jac"` (Jacobian-based scaling). Parameters
operating over wide dynamic ranges (mobility, leakage, fringe caps, etc.) are optimized
in log-space via `LOG_PARAMS`. Bounds are enforced by `PARAM_BOUNDS` dict. The optimization
happens in 6 sequential stages (see Engine Behavior below).

**Intrinsic vs External IV**: `intrinsic_ids()` computes the channel current assuming
zero source/drain resistance. `external_ids()` accounts for Rs/Rd via iterative voltage
correction (30 fixed-point iterations with 0.45/0.55 relaxation) and adds the leakage
current. When both Rs0 and Rd0 < 10Ω, the simpler direct calculation is used.

**Terminal Charges for CV**: `terminal_charges()` computes gate and drain/source charges
using the Ward-Dutton charge partitioning scheme. Gate charges combine channel charge with
fringe/overlap contributions. CV capacitance is computed via numerical differentiation
(central difference with h=1e-4), then differentiated to get C = |ΔQ/(2h)|.

**CV Trap Model**: After the 6-stage main fit, a separate shared trap model fit
(`fit_cv_shared_trap`) adds frequency-dependent trap capacitance (interface traps Cit_t,
Cit_b) with a sigmoid occupancy function, Colle-Cole frequency dispersion (parameter α),
and fast/slow trap partitioning.

## 2. Upstream/Downstream

### Receives from check-data

- A verified dataset directory containing:
  - IV workbooks: `IDVBG*_train.xlsx`, `IDVTG*_train.xlsx`, `IDVD*_train.xlsx`
  - CV workbook: `CV_train.xlsx` (already normalized to paired-column format)
  - `dataset_validation.json` — upstream validation results
- `model_config.ini` — runtime config with `[physical_fit]`, `[preprocessing]`, `[paths]`

### Delivers to train-static

- `01_physical_kernel/fitted_params.json` — final parameter values + fit history + metrics summary + CV trap params
- `01_physical_kernel/dg_tft_ekv_level2_fitted.va` — patched Verilog-A template with fitted parameters
- `01_physical_kernel/fit_metrics.csv` — per-family, per-region, per-CV-sheet error metrics
- `01_physical_kernel/iv_measured_vs_fitted.csv` — all IV data with fitted `ID_fit_A`, `log10_abs_error_dec`, `relative_error`
- `01_physical_kernel/cv_measured_vs_fitted.csv` — all CV data with core QS and trap contributions decomposed
- `01_physical_kernel/cv_shared_trap_params.json` / `.csv` — trap model parameters
- `01_physical_kernel/fit_report.txt` — human-readable summary
- `01_physical_kernel/fit_IDVBG.png`, `fit_IDVTG.png`, `fit_IDVD.png`, `fit_CV.png` — measured vs fitted plots
- `01_physical_kernel/v92_phase1_cv_gate_orientation.json` — CV gate swap diagnostic
- `01_physical_kernel/v92_phase1_cv_data_decisions.csv` — which CV series were kept/dropped
- `01_physical_kernel/iv_stage_alignment_report.csv` / `.json` — per-source IV alignment shifts
- `01_physical_kernel/cv_shared_trap_model_equations.txt` — trap model equation reference
- `model_config.ini` — potentially modified by Agent B's `adjust_params`

### Fallback Chain

- Agent B falls back to **check-data** (StageName.CHECK_DATA), which is upstream of fit-physical.
  This means: if the physical fit fails because the data is the problem, the fallback goes to
  check-data, which re-validates the dataset and may itself fallback to Verifier and then Preparer.
  This could trigger a complete data re-preparation cycle.
- When train-static falls back to fit-physical: the fit-physical stage is re-executed, which
  regenerates `fitted_params.json` and the Verilog-A template. The train-static stage should
  then re-examine whether the new physical parameters improve its fitting quality.

## 3. INI Parameters

All parameters below are from `PHYSICAL_FIT_PARAMS` in `param_defs.py`. The stage reads
`[physical_fit]` (max_iv_points, max_cv_points, max_nfev_scale) and `[preprocessing]`
(iv_stage_alignment, iv_stage_max_shift_V, iv_stage_min_matches).

### [physical_fit] Section

| Parameter | Type | Default | Bounds | Description |
|-----------|------|---------|--------|-------------|
| `max_iv_points` | int | 50000 | [5000, 200000] | Max IV data points for fitting. **Increase if**: IV fit misses features (the subsampler may be dropping important bias points). **Decrease if**: runtime exceeds timeout (1800s) or memory is constrained. Set to 1000 in `minimal_run` mode. Each data point contributes a residual term; fewer points = faster iterations but coarser fit. The subsampler weights subthreshold and on-state regions more heavily, so even at low values critical operating points are preserved. |
| `max_cv_points` | int | 100000 | [10000, 500000] | Max CV data points for fitting. **Increase if**: CV capacitance transitions look jagged/undersampled. **Decrease if**: CV fitting stage is slow (CV residuals dominate the combined residual vector). Set to 500 in `minimal_run` mode. CV is subsampled uniformly (not weighted) using `np.linspace` for each series. Note: this controls fitting data, not the final CV dataset cardinality. |
| `max_nfev_scale` | float | 1.0 | [0.5, 5.0] | Scale factor applied to the `max_nfev` budget of each sub-stage. Each of the 6 fitting stages has a base NFEV (70, 85, 80, 90, 120, 450 for CV trap). These are all multiplied by `max_nfev_scale` then rounded (min 12). **Increase (e.g., 2.0, 3.0) if**: the optimizer hits `max_nfev` and converges prematurely (check fit_history in fitted_params.json — if `accepted` is False for a stage, it needs more iterations). **Decrease (e.g., 0.5) if**: runtime is the bottleneck and convergence is acceptable. Set to 0.1 in `minimal_run` mode. This is typically the first parameter to adjust when fit quality is poor — it's cheap (no data reload) and directly controls optimization thoroughness. |

### [preprocessing] Section

These parameters are shared across all script stages but heavily impact fit-physical quality.

| Parameter | Type | Default | Bounds/Enum | Description |
|-----------|------|---------|-------------|-------------|
| `iv_stage_alignment` | string | auto | ["off", "auto"] | IV measurement-stage alignment. **"auto"** automatically detects and corrects per-source-file transfer-voltage shifts using overlapping bias conditions. This is critical when measurement equipment introduces small voltage offsets between measurement stages. **"off"** skips alignment — use only when data is known to be pre-aligned. **Set "off" if**: alignment introduces artifacts (spurious shifts). **Set "auto" if**: different IV files from the same device show systematic voltage offsets that cause the optimizer to converge to physically unreasonable thresholds. |
| `iv_stage_max_shift_V` | float | 0.75V | [0.1, 2.0] | Maximum allowed transfer-voltage shift per source file for IV stage alignment. **Increase if**: measurements have documented large stage offsets (e.g., from different instrument setups). **Decrease if**: you want to be conservative about alignment — larger shifts can mask genuine device-level threshold differences. The alignment algorithm uses shared bias conditions across files to detect shifts; this bound prevents unrealistic corrections. |
| `iv_stage_min_matches` | int | 2 | [1, 10] | Minimum number of matched bias-condition pairs required before applying a stage shift to a source file. **Increase if**: alignment should be more conservative (require stronger evidence). **Decrease to 1 if**: alignment is not triggering despite visible shifts in the IV plots. This is a sensitivity control: too low may over-align noise; too high may miss real offsets. |

### minimal_run Override

When `[runtime] minimal_run = yes`, these values are hard-overridden in `run_model.py`:
- `max_iv_points`: 1000
- `max_cv_points`: 500
- `max_nfev_scale`: 0.1
- All preprocessing settings (iv_stage_alignment, etc.) remain as configured

> **Note on Agent B's adjust_params tool**: The auto-generated `adjust_params` schema provides only parameter names, types, and bounds — it has NO diagnostic context (e.g., it does not explain that Stage 6 NFEV is fixed at 450 regardless of max_nfev_scale, or that reducing max_iv_points below the file cap minimum can drop subthreshold points). This knowledge file's tuning guidance fills this gap, but it is currently not injected into Agent B's prompt. See Knowledge Injection Status note above.

## 4. Quality Metrics

### iv_rms_log10_error_dec

- **What it measures**: The RMS error between log10(measured |ID|) and log10(predicted |ID|)
  across ALL IV data points (all families: IDVBG, IDVTG, IDVD). A value of 0.35 means
  the typical log10 error is ~0.35 decades (a factor of ~2.2× in absolute current).
- **Source file**: `01_physical_kernel/fitted_params.json` → key `metrics_summary.iv_rms_log10_error_dec`
- **Threshold from agent_configs.json**: `DG_TFT_QUALITY_PHYSICAL_IV_RMS_MAX` = **0.35** (decades)
- **What "below threshold" means**: The IV fit has global rms log error ≤ 0.35 decades.
  This is a physically meaningful threshold — a factor of 2.2× in current prediction is
  targeted as the acceptable range for the physical kernel before progressing to the more
  expressive KAN-based static model. Values well below 0.15 are excellent; values above 0.50
  indicate significant model-measurement mismatch.
- **Additional metrics produced** (not used for quality gates, but diagnostically useful):
  - `iv_median_abs_log10_error_dec`: median absolute log10 error
  - `iv_on_rms_log10_error_dec`: RMS error in on-region (|ID| ≥ 1e-9 A)
  - `iv_subthreshold_rms_log10_error_dec`: RMS error in subthreshold (1e-12 ≤ |ID| < 1e-9 A)
  - `cv_rms_log10_error_dec`: RMS log10 error for CV (including trap model)
  - `cv_core_qs_rms_log10_error_dec`: RMS log10 error for core quasi-static CV (no traps)
  - `fit_nonmonotonic_steps`: count of non-monotonic fit predictions (dy < -0.03 decades per step, checked per-family per-source per-fixed-voltage-group). A ratio of `nonmonotonic_steps / checked_steps > 0.1` indicates optimizer found a physically unreasonable local minimum.
  - `fit_checked_steps`: total monotonicity checks performed
- **Which parameter typically affects it**:
  - **Primary**: `max_nfev_scale` — insufficient iterations is the most common cause of poor IV fit
  - **Secondary**: `max_iv_points` — too-aggressive subsampling may miss features in sparse bias regions
  - **Tertiary** (not INI-tunable, but diagnostic): data quality issues like incorrect VDS labeling,
    wrong file family assignment (IDVBG vs IDVTG), or stage misalignment can produce permanently
    high iv_rms_log10_error_dec that parameter tuning cannot fix

Note: When `[runtime] minimal_run = yes`, quality assessment is **fully bypassed** — `_assess_stage_quality` returns `True, []` immediately without reading any metrics or checking thresholds. Agents spawned for quality failures are never triggered in minimal_run mode.

## 5. Engine Behavior

### Subprocess invocation and Engine Argparse vs INI Defaults

**INI defaults** (from `param_defs.py` and `model_config.ini`):
- `max_iv_points`: 50000 (IN), `max_cv_points`: 100000 (INI), `seed`: 20260524 (INI)
- `max_nfev_scale`: 1.0

**Engine argparse defaults** (from fit_physical_kernel.py, used as fallback if INI value is missing):
- `--max-iv-points`: **10000** (1/5 of INI), `--max-cv-points`: **1200** (1/83 of INI)
- `--seed`: **20260522** (2 off from INI 20260524)

The orchestrator passes INI values as explicit CLI arguments, so the engine defaults are only used
if the INI or orchestrator fails to supply the parameter. But the gap is large — an agent should be aware
that if these parameters are missing, fitting will silently use vastly less data.

Oxide thickness defaults (derived from `Cox = ε0 × 3.9 / t_ox`):
- Top oxide: **120 nm** → Cox_t ≈ ε0 × 3.9 / 120e-9
- Bottom oxide: **350 nm** → Cox_b ≈ ε0 × 3.9 / 350e-9
- Cox_t > Cox_b (~2.9×), which affects the voltage weighting ratio at/ab in the overdrive calculation.

### Execution Flow

1. **Subprocess invocation** (run_model.py:fit_physical → engine/fit_physical_kernel.py):

```
<CUDA python> engine/fit_physical_kernel.py
  --data-dir <data_dir>
  --out-dir <output>/01_physical_kernel
  --max-iv-points <max_iv_points>
  --max-cv-points <max_cv_points>
  --max-nfev-scale <max_nfev_scale>
  --seed <runtime.seed>
  --iv-stage-alignment <preprocessing.iv_stage_alignment>
  --iv-stage-max-shift-V <preprocessing.iv_stage_max_shift_V>
  --iv-stage-min-matches <preprocessing.iv_stage_min_matches>
```

2. **Step 1 — Load IV data** (`load_iv_data`):
   - Globs `ID*_train.xlsx` (excluding files starting with "CV")
   - Reads the "long" sheet from each workbook
   - Validates required columns: VD_V, VBG_V, VTG_V, ID_A
   - Optionally detects temperature columns (temp_K, temperature_K, T_K, TEMP_K or Celsius equivalents)
   - Names family based on filename prefix (IDVBG, IDVTG, IDVD)
   - Raises FileNotFoundError if no IV files found

3. **Step 2 — IV stage alignment** (if `--iv-stage-alignment auto`):
    - Detects and corrects per-source transfer-voltage shifts
    - Uses overlapping bias conditions across measurement stages
    - **Modifies the IV dataframe voltages in-place** — the fitting uses aligned (shifted) voltages, not the original measured values
    - Setting alignment to "off" means the original unaligned voltages are used for fitting
    - Writes alignment report to `iv_stage_alignment_report.csv`/`.json`

4. **Step 3 — Load CV data** (`load_cv_data`):
   - Reads `CV_train.xlsx`, all sheets
   - Detects layout (paired-column vs shared-bias wide-table) via `looks_like_shared_bias_wide_table`
   - Parses voltage metadata (VTG=, VBG=, VDS=) and frequency (kHz/MHz/Hz) from column labels
    - Computes transition metadata (centroid, span, vhalf) and centered coordinates
    - Assigns `complete_fraction` relative to the longest series per sheet

> **Silent Data Filtering**: During data loading, several filtering steps silently discard data points with no log warnings. NaN rows in required IV columns are dropped via `dropna()`. CV sheets with <3 rows are skipped. Non-positive capacitances are masked. XLSX files without 'long' sheet are skipped without any log. Agent B should NOT assume all data was loaded — compare loaded point counts with expected file counts if fit quality is unexpectedly poor.

5. **Step 4 — CV gate auto-orientation** (`auto_orient_cv_series`):
    - Swaps CBG↔CTG labels if **both** conditions are met: (1) `Cox_t/Cox_b ≥ 1.40` AND (2) CBG capacitance span / CTG span ≥ **1.55**
    - This corrects common lab worksheet labeling errors where gate labels are swapped
     - Writes diagnostic to `v92_phase1_cv_gate_orientation.json`

> **Warning**: The swap thresholds (Cox ratio ≥1.40, span ratio ≥1.55) are hardcoded. If they produce a false swap, CV fit quality will be systematically poor regardless of parameter tuning. The resulting CV gate label swap error is only visible in `v92_phase1_cv_gate_orientation.json`.

6. **Step 5 — CV series selection** (`select_primary_cv_series`):
    - Keeps only quasi-static gate CV sheets for fitting: CGG at **~10 kHz** IS used for fitting (the filter drops CGG at non-10-kHz frequencies for NQS validation)
    - Drops VDS sheet and high-frequency CGG (not near 10 kHz)
    - Drops incomplete series (<75% complete_fraction) or series with single steps >20% of span

7. **Step 6 — Data subsampling** (`sample_iv`, `sample_cv`):
   - IV: weighted random sampling with emphasis on subthreshold (-12.2 to -8.5 log10|ID| = +3.0 weight on top of base 1.0, effective 4.0× baseline),
     on-region (> -8.5 = +1.5 weight, effective 2.5× baseline), and transfer families (IDVBG/IDVTG = +0.5 weight).
     Per-file cap of `max_points / n_files`, then global cap via final `.sample(n=max_points)`.
   - CV: uniform `np.linspace` subsampling proportional to series length, min 25 points per series.

8. **Step 7 — Build residual function** (`make_residual_function`):
    - Concatenates three residual components with configurable weights:
      - **IV residuals**: `√w × log10_error + linear_weight × (pred - meas) / scale` where `w` is the product of these weights:

| Factor | Condition | Weight |
|--------|-----------|--------|
| Family IDVD | family == "IDVD" | 0.90 (other families 1.12) |
| Subthreshold | -12.3 ≤ log10\|ID\| ≤ -8.7 | 2.35 |
| On-region | log10\|ID\| > -8.7 | 1.55 |
| Deep off | log10\|ID\| < -12.8 | 0.55 |
| Near-zero VDS | \|VDS\| < 0.02V | 0.25 |
| Linear term | |ID_meas\| > 3e-10 A | 0.28 |
| Linear term (weak) | |ID_meas\| ≤ 3e-10 A | 0.04 |

      The linear error scale uses `max(abs_meas, 92nd_percentile × 0.015)`, a dynamic normalization per device.
      - **CV residuals** (main fit): `sheet_weight × (0.55·rel + 0.65·log_err)`
        Sheet weights: cgg=1.1, ctg/cbg=1.0, vds=0.9
      - **CV residuals** (trap fit, stage 6): `sheet_weight × (0.50·rel + 0.72·log_err)`
        Sheet weights: cgg=1.05, ctg/cbg=1.0, vds=0.9. Uses `f_scale=0.55` in `soft_l1` (vs f_scale=1.0 for main stages).
      - **Regularization**: priors keeping parameters near process defaults:
        - Cox_t: `0.65 × log2(Cox_t / process_top)` — strong, 13× weight relative to mu0
        - Cox_b: `0.65 × log2(Cox_b / process_bottom)` — strong
        - nfac0: `0.10 × (nfac0 − 4.5) / 4.5` — moderate linear
        - mu0: `0.05 × log10(mu0 / 1e-3)` — weak
        The regularization references (`process_top`, `process_bottom`) are captured once at init from `Params()` defaults and never updated — regularization always pulls toward original process defaults, even after Cox values are fitted.
    - Non-finite residuals are clamped to ±1e6

9. **Step 8 — Six-stage least_squares fit** (`fit_model`):

| Stage | Name | Parameters Fitted | Data | cv_weight | reg_weight | base_nfev |
|-------|------|-------------------|------|-----------|------------|-----------|
| 1 | low_vds_transfer | Vth_t0, Vth_b0, nfac0, mu0, Ioff0, Gleak_ds | Low-VDS transfer only (\|VDS\|≤0.55V) | 0.0 | 0.65 | 70 |
| 2 | all_transfer_gate_shape | Vth_t0, Vth_b0, nfac0, mu0, theta_mu, Ioff0, Gleak_ds | All transfer (IDVBG+IDVTG) | 0.0 | 0.35 | 85 |
| 3 | output_curves | mu0, theta_mu, lambda_clm, Ioff0, Gleak_ds, Vth_t0, Vth_b0 | Output (IDVD) + transfer sample | 0.0 | 0.20 | 80 |
| 4 | cv_capacitance | Cfr_tgs, Cfr_tgd, Cfr_bgs, Cfr_bgd | CV only | 1.0 | 0.20 | 90 |
| 5 | joint_refine | Vth, nfac0, mu0, theta_mu, lambda_clm, leakage, all fringe | IV + CV joint | 0.22 | 0.45 | 120 |
| 6 | cv_shared_trap | Cfr_*, Cit_t, Cit_b, trap_width, tau_s, alpha, trap_fast_frac | CV only (all series, not just selected) | N/A | N/A | **450 (fixed)** |

Each stage: `run_stage() → least_squares(loss="soft_l1", f_scale=1.0, x_scale="jac", max_nfev=nfev)`
**Stage 6 (trap fit) uses `f_scale=0.55`** (vs 1.0 for main stages) — tighter L1/L2 transition means more residuals are treated with robust L1 loss.
If `after_rms > before_rms × 1.002`, the stage is rejected and previous parameters are kept.
NFEV for stages 1-5 = `round(base_nfev × max_nfev_scale)`, minimum 12.
**Stage 6 NFEV is NOT scaled by max_nfev_scale** — it is fixed at 450 regardless of the scale factor.

10. **Step 9 — CV shared trap fit** (`fit_cv_shared_trap`):
    - Fits ALL CV series (not just selected), including CGG, VDS sheets
    - Jointly optimizes fringe capacitances + trap parameters (Cit, trap_width, tau_s, α, trap_fast_frac)
    - Uses Colle-Cole frequency dispersion for the trap response
    - Includes mild regularization on fringe caps keeping them near expected values

11. **Step 10 — Compute metrics** (`compute_metrics`):
    - Per-family, per-source-file, per-region (off/sub/on) IV metrics
    - Per-sheet CV metrics (with and without trap model)
    - Non-monotonicity check on fitted IV curves

12. **Step 11 — Write outputs** (`write_outputs`):
    - Patches `physical_kernel_template.va` with fitted parameter values → `dg_tft_ekv_level2_fitted.va`
    - Writes `fitted_params.json`, metrics CSV, measured-vs-fitted CSVs, PNG plots
    - Writes `fit_report.txt` with final parameters and stage history
    - Deletes stale `cv_enhanced_params.csv` from previous runs

### Key Log Messages

- `Loaded IV points: N total, M used for fitting` — initial data load summary
- `Loaded CV series: N, CV points used for fitting: M` — CV data summary
- `1_low_vds_transfer: rms X → Y, nfev=Z` — per-stage convergence
- `(kept previous parameters)` suffix — stage rejected (RMS increased)
- `6_cv_shared_trap: rms X → Y, nfev=Z` — final trap fit result
- `Done. Outputs written to: <out_dir>` — final success
- JSON dump of `summary` dict — printed to stdout, captured as stage metrics

### How Parameters Change Execution

- `max_iv_points`: Controls `sample_iv()` which subsamples before fitting. Lower = fewer residual evaluations per iteration = faster per-iteration but coarser fit. The weighted sampling ensures critical regions (subthreshold, on-state) are preserved even at low values.
- `max_cv_points`: Controls `sample_cv()` similarly for CV data.
- **Interaction**: `max_iv_points` + `max_cv_points` together determine the total residual vector length ≈ `2×IV_points + CV_points + regularization`. This affects `least_squares` Jacobian size and memory — reducing both simultaneously cuts runtime more than reducing either alone.
- `max_nfev_scale`: Multiplies per-stage NFEV budgets. Each stage independently gets this multiplier. For example, at scale=2.0, stage 1 gets 140 NFEV instead of 70. This is the most impactful single parameter — doubling it roughly doubles optimization budget with no data reload cost. **Upper bound is 5.0**: stages 1-5 cap at round(120 × 5.0) = 600 NFEV each. Stage 6 is NOT scaled (fixed at 450).
- `iv_stage_alignment`: "auto" enables the alignment pipeline which can produce an `iv_stage_alignment_report.*` artifact. "off" skips it.
- `iv_stage_max_shift_V` / `iv_stage_min_matches`: Control alignment sensitivity. Higher shift + lower matches = more aggressive alignment. Note: INI stores this as `iv_stage_max_shift_v` (configparser lowercases keys; functionally identical).

### Engine-Internal Thresholds NOT Adjustable via INI

The following parameters are hardcoded in the engine and **CANNOT** be changed by Agent B's `adjust_params`:

- RS/RD iterative-solve threshold: 10 Ω (controls whether 30-iteration solver is used vs direct calculation)
- RS/RD relaxation factors: 0.55/0.45
- Stage 1 VDS filter: ≤0.55V
- Stage 3 transfer subsample ratio: `len(idvd)//2` (hardcoded, not seed-dependent — uses `random_state=12`)
- CV numerical diff step h: 1e-4
- qsolve Newton iterations: 6 (fixed)
- `smooth_pos` epsilon: 1e-3 (used in 5+ smoothed equations)

If fit quality is poor and **NO** INI parameter change helps, suspect one of these hardcoded thresholds is the bottleneck.

## 6. Failure Patterns

> This section will be populated from Zikang's Codex JSONL logs during the knowledge extraction phase.

### Error Classification (from diagnostics.py)

When fit-physical exits with non-zero code, keyword classification is applied against
the combined error message + stderr text:

| Keyword in stderr | Classification | Confidence |
|-------------------|---------------|------------|
| `Cuda out of memory`, `out of memory`, `OOM`, `cuDNN` | TRAINING_NUMERICAL_ERROR | 0.9 |
| `NaN`, `inf`, `overflow encountered`, `underflow encountered`, `overflow` | TRAINING_NUMERICAL_ERROR | 0.9 |
| `gradient`, `backward` | TRAINING_NUMERICAL_ERROR | 0.9 |
| `diverged`, `not converging`, `loss is nan`, `loss too large`, `gradient explosion` | TRAINING_NUMERICAL_ERROR | 0.7 |
| `FileNotFoundError`, `No such file or directory`, `does not exist` | DATA_ERROR | 0.9 |
| `missing column`, `missing sheet`, `invalid sheet`, `sheet not found`, `column not found` | DATA_ERROR | 0.9 |
| `unexpected header`, `empty workbook`, `no valid data`, `xlsx file is empty` | DATA_ERROR | 0.9 |
| `dataset_validation.json` | DATA_ERROR | 0.9 |
| `data_dir not found` | DATA_ERROR | 0.9 |
| `cannot find it directory` | DATA_ERROR | 0.9 |
| `excel file format cannot be determined` | DATA_ERROR | 0.9 |
| `KeyError`, `IndexError`, `ValueError` | DATA_ERROR | 0.6 |
| `configparser`, `no section`, `no option`, `invalid config`, `missing config` | CONFIG_ERROR | 0.9 |
| `unknown stage`, `unknown action` | CONFIG_ERROR | 0.9 |
| `hspice.exe`, `hspice not found`, `hspice executable`, `hspice path` | HSPICE_ENV_ERROR | 0.9 |
| `missing artifact`, `expected output not found`, `cannot find`, `not generated`, `output file.*missing`, `artifact.*not created` | MISSING_ARTIFACT_ERROR | 0.9 |
| No keyword match, exit code ≠ 0 | TRAINING_NUMERICAL_ERROR | 0.4 |

Since fit-physical is in `_EXIT_CODE_NUMERICAL_STAGES`, the default exit-code fallback
is `TRAINING_NUMERICAL_ERROR` with confidence 0.4.

**Important**: The keywords `Cuda out of memory`, `cuDNN`, `gradient`, `backward`,
`hspice.exe` will **never** match fit-physical failures — this stage uses
`scipy.optimize.least_squares` (no CUDA, no gradient backpropagation, no HSPICE).
The most relevant keywords for this stage are the DATA_ERROR ones (`FileNotFoundError`,
`missing column`, etc.) and numerical error keywords (`NaN`, `overflow encountered`).

The stage suggestion message: "Check the physical model fitting inputs — may indicate
incompatible data range or missing template files."

### Agent B Decision Framework

Agent B has four tools and must call EXACTLY ONE:

| Tool | When to Use | Consequences |
|------|-------------|--------------|
| `adjust_params` | Parameter tuning can fix the problem. Use when: (1) quality is close to threshold (e.g., iv_rms_log10_error_dec = 0.40 vs threshold 0.35), (2) the fit history shows rejected stages (accepted=false) indicating insufficient iterations, (3) IV plots show systematic over/under-prediction suggesting the optimizer needs more NFEV. Incremental changes preferred — do NOT jump from max_nfev_scale=1.0 to 5.0. | INI is updated, stage re-runs with new parameters. Costs a retry slot. |
| `fallback_to_previous` | The data itself is the problem. Use when: (1) error traces indicate FileNotFoundError or missing columns — the CV_train.xlsx or IV files have structural issues, (2) `dataset_validation.json` from check-data shows validation failures that were ignored/overlooked, (3) IV plots show physically impossible patterns (negative ID, sudden discontinuities) suggesting data collection errors, (4) multiple parameter adjustments have yielded no improvement, suggesting root cause is upstream. | Stage re-runs from check-data, which may propagate further to Verifier/Preparer for data re-preparation. |
| `accept_quality` | Current quality is acceptable despite being above threshold. Use when: (1) iv_rms_log10_error_dec is only marginally above 0.35 (e.g., 0.36-0.42) and the fit plots look qualitatively good, (2) user preferences explicitly deprioritize this metric, (3) the subthreshold region is noisy but the on-region fit is excellent, (4) all adjustment attempts have been exhausted — accept the best result so far. | Pipeline moves to train-static. Quality warning is recorded but does not block progress. |
| `inspect_xlsx` | Investigation only — use freely to examine file structure before deciding. Does not consume a decision. Returns `{"status": "success"|"fail", "sheet_count": N, "sheets": {"Name": {"columns": [...]}}}`. |

**Specific tuning guidance for adjust_params**:

- **iv_rms_log10_error_dec too high** (>0.35): First try `max_nfev_scale` = 2.0. If still failing, try 3.0. The upper bound is 5.0 (600 NFEV per stage for stages 1-5). If the problem persists at 3.0+, the issue is likely not optimization budget — suspect data quality or model mismatch. Consider: is the data subthreshold region abnormally noisy? Are Vth values pinned at bounds? Check the fit_history in fitted_params.json — are all stages accepted? If stage 1 (low_vds_transfer) is rejected, the initial threshold extraction is failing, which will cascade to all subsequent stages.

- **Runtime too long** (>1800s timeout): Reduce `max_iv_points` to 25000 or `max_nfev_scale` to 0.5. These trade accuracy for speed. In minimal_run mode, both are drastically reduced.

- **IV stage alignment artifacts**: If alignment is producing shifts that look wrong (check `iv_stage_alignment_report.csv`), try setting `iv_stage_alignment` to "off" or increasing `iv_stage_max_shift_V` to be more permissive.

### Diagnostic Tools Available

Agent B can use `read_file` and `glob_file` (standard SubAgent tools) to inspect:

- `01_physical_kernel/fitted_params.json` — the most important file. Contains:
  - `final_params`: all 34 parameter values (both fitted and held fixed)
  - `fit_history`: per-stage before/after RMS, accepted flag, NFEV, params_after (full 34-param snapshot), status (scipy convergence code), message (scipy optimizer message)
  - `metrics_summary`: iv_rms_log10_error_dec + per-region breakdowns
  - `cv_shared_trap_params`: trap model parameters
- `01_physical_kernel/fit_metrics.csv` — granular per-family, per-source, per-region metrics
- `01_physical_kernel/fit_report.txt` — human-readable parameter values and history
- `01_physical_kernel/fit_IDVBG.png`, `fit_IDVTG.png`, `fit_IDVD.png`, `fit_CV.png` — visual comparison plots (measured=solid, fitted=dashed). Look for: mismatched slopes in transfer curves, incorrect saturation in output curves, offset errors in capacitance transitions.
- `01_physical_kernel/iv_stage_alignment_report.csv` — per-source shift values and match counts (if alignment is "auto")
- `01_physical_kernel/v92_phase1_cv_data_decisions.csv` — which CV series were kept vs dropped and why
- Error logs in the output directory or stderr capture
- `dataset_validation.json` — upstream check-data results (in output root, not 01_physical_kernel)
- `model_config.ini` — current parameter values, data paths, runtime settings
