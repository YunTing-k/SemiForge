# Physical Kernel Experience (from 2026-06-29 session)

## Principle Insights

### Physical Baseline Divergence Blocks Cross-Version Checkpoint Transfer
**Category**: Principle Insight
**Stage(s)**: fit-physical (primary), train-static (cross-impact)
**Symptom**: Attempting to mechanically splice IV KAN weights from a v90 checkpoint into a v92+ checkpoint produces degraded results (IV nonzero RMS degrades from ~0.10 to ~0.12 dex), even when the individual KAN weight tensors appear structurally compatible.
**Root Cause**: The fitted_params.json produced by fit-physical differs between runs (and across engine versions). These fitted physical parameters (Vth_t0, Vth_b0, nfac0, mu0, Cfr_*) serve as the "phase-1 baseline" for downstream KAN initialization — they determine input feature normalization centers/scales and the initial physical component of the model. When two checkpoints were built on different physical baselines, their KAN weights encode different residual corrections relative to different starting points. Simple tensor transfer ignores this dependency.
**Evidence**: Turn 8 (line 206) — direct checkpoint splicing: "v90 与当前 CV 的 phase-1 物理参数不同，简单替换张量会破坏共享物理核：IV 降到 0.12 dex，不能交付。" Turn 13 (line 434) — restoration: "用正确的 v90 base 物理目录复验后，精确 v90 IV 已恢复：residual scale=1.0，nonzero RMS 0.09992→0.09497，开态 0.04357，高电流 0.01842；先前异常是模型目录与 checkpoint 的物理基准不匹配。"
**Generalization**: Any time either the fit-physical engine changes (different oxide defaults, alignment logic, residual weights) or the input data changes, the resulting fitted_params.json parameters shift. Downstream PKAN checkpoints trained on one baseline cannot be ported to another without full re-initialization from the new physical kernel.
**Actionable Advice**: When Agent B adjusts fit-physical parameters and triggers a re-run, downstream stages (train-static, train-hysteresis) must be re-initialized from scratch — do not attempt to warm-start from a previous checkpoint. Conversely, if Agent C (train-static) falls back to fit-physical, the full static model must be retrained from the new baseline. Direct weight transplantation across physical baselines is systematically unsound.

### Fit-Physical Runtime Calibration
**Category**: Principle Insight
**Stage(s)**: fit-physical
**Symptom**: Agent B or the orchestrator may time out a fit-physical run thinking it is stalled, when in fact it is completing normally.
**Root Cause**: The physical kernel fitting runs entirely on CPU via scipy.optimize.least_squares with 6 sequential stages (plus CV shared trap fit). For a typical device dataset (~50K IV points, ~100K CV points at default max_nfev_scale=1.0), the total runtime is approximately 1 minute.
**Evidence**: Turn 27 (line 931) — fresh-start experiment: "先从原始 Excel 重新拟合物理核（约 1 分钟），再进行 140 epoch 的真实 base-PKAN 训练（约 1 分钟）。" The physical kernel completed in ~1 minute, while the downstream PKAN training began afterward. This timing was consistent across multiple fresh-start invocations throughout the session.
**Generalization**: On typical single-device datasets with default INI parameters, fit-physical is CPU-only and fast (~60s). It should not be the bottleneck in pipeline runtime. If the orchestrator reports a timeout during fit-physical, the problem is NOT that fit-physical is slow — it is either a data loading issue (e.g., malformed Excel files causing infinite parsing), or the timeout value was set too low relative to the data volume (larger max_iv_points / max_cv_points values scale linearly with residual vector size).
**Actionable Advice**: Agent B should NOT attempt to reduce max_nfev_scale or max_iv_points to "speed up" a fit-physical run that is already at defaults — the default runtime is ~1 minute and reducing parameters degrades quality with negligible speed gain. If the run exceeds ~5 minutes at defaults, suspect data issues (missing sheets, malformed columns, extremely large Excel files), not optimization budget. Use `inspect_xlsx` to verify data structure before adjusting parameters.

## Session Meta-Notes

- **Fit-physical stability**: Across all 28 turns of this session, the fit-physical stage never once failed (exit code ≠ 0) or produced quality below threshold. Agent B was never spawned for fit-physical. This suggests the EKV Level-2 physical model + 6-stage least_squares pipeline is robust for the DG-TFT device class, at least for data within expected ranges.
- **Primary session focus**: The session's quality regressions (v92 IV degradation, Top-1 symbolic distillation failures, hysteresis slow-bank divergence) were all in downstream stages (train-static, train-hysteresis), NOT in fit-physical. The physical kernel repeatedly produced usable initial parameters without intervention.
