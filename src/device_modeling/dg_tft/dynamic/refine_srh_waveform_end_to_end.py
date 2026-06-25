from __future__ import annotations

"""End-to-end SRH waveform refinement for DG-TFT I-t hysteresis data.

This stage starts from the denoised I-t step responses and trains the same
top-1-friendly KAN rate model directly against full I-t waveforms.  It keeps
the compact-model interpretation:

    dNta/dt = (Ninf - Nta) / tau_eff

and uses the analytical SRH step solution per shallow/deep branch.  The network
therefore refines log_tau_c/log_tau_e-like rates; it does not draw arbitrary
ID(t) curves.
"""

from pathlib import Path
import argparse
import copy
import json
import math
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from train_srh_history_kan import (
    AdditiveKAN,
    BASIS_NAMES,
    FEATURE_NAMES,
    basis_torch,
    build_label_table,
    basis_np,
)


ROOT = Path(__file__).resolve().parent
RESULTS = Path(os.environ.get("DG_TFT_DYNAMIC_WORK_DIR", ROOT / "results_long_timescale"))
OUT = RESULTS / "srh_waveform_refined"
ID_FLOOR_A = 1.0e-15


class TwoLayerTop1KAN(torch.nn.Module):
    """Compact two-layer KAN with an exact sparse symbolic deployment gate.

    Top-1 is maximally compact, but some SRH rate/coupling maps need two basis
    functions on an edge to reproduce the teacher without discontinuous
    parameter compensation. Top-k remains a finite analytic expression and is
    exported explicitly; no spline table is retained.
    """

    def __init__(
        self,
        n_features: int,
        n_basis: int,
        hidden: int = 12,
        symbolic_topk: int = 2,
    ):
        super().__init__()
        self.hidden = int(hidden)
        self.symbolic_topk = int(max(1, min(symbolic_topk, n_basis)))
        self.hidden_bias = torch.nn.Parameter(torch.zeros(self.hidden))
        self.in_amp = torch.nn.Parameter(
            0.025 * torch.randn(self.hidden, n_features)
        )
        self.in_slope_raw = torch.nn.Parameter(
            torch.zeros(self.hidden, n_features)
        )
        self.in_shift = torch.nn.Parameter(torch.zeros(self.hidden, n_features))
        self.in_logits = torch.nn.Parameter(
            torch.zeros(self.hidden, n_features, n_basis)
        )
        self.out_bias = torch.nn.Parameter(torch.zeros(()))
        self.out_amp = torch.nn.Parameter(
            0.10 * torch.randn(self.hidden)
        )
        self.out_slope_raw = torch.nn.Parameter(torch.zeros(self.hidden))
        self.out_shift = torch.nn.Parameter(torch.zeros(self.hidden))
        self.out_logits = torch.nn.Parameter(
            torch.zeros(self.hidden, n_basis)
        )
        with torch.no_grad():
            # Input edges begin as affine projections. Hidden-output edges use
            # a diverse symbolic activation bank so hard-top1 does not collapse
            # to an all-linear network because of an argmax tie at epoch zero.
            self.in_logits.fill_(-2.0)
            self.in_logits[:, :, BASIS_NAMES.index("linear")] = 2.0
            preferred = [
                "tanh",
                "sigmoid",
                "gaussian",
                "invquad",
                "softplus_decay",
                "linear",
            ]
            self.out_logits.fill_(-2.0)
            for h in range(self.hidden):
                self.out_logits[h, BASIS_NAMES.index(preferred[h % len(preferred)])] = 2.0

    def _gate(self, logits: torch.Tensor, hard_top1: bool) -> torch.Tensor:
        prob = torch.softmax(logits, dim=-1)
        if not hard_top1:
            return prob
        _, idx = torch.topk(logits, k=self.symbolic_topk, dim=-1)
        hard_mask = torch.zeros_like(prob).scatter_(-1, idx, 1.0)
        hard = prob * hard_mask
        hard = hard / torch.clamp(torch.sum(hard, dim=-1, keepdim=True), min=1.0e-12)
        return hard + prob - prob.detach() if logits.requires_grad else hard

    def forward(self, x: torch.Tensor, hard_top1: bool = False) -> torch.Tensor:
        in_slopes = 0.05 + F.softplus(self.in_slope_raw)
        z = (
            x[:, None, :] * in_slopes[None, :, :]
            + self.in_shift[None, :, :]
        )
        vals = torch.stack(
            [basis_torch(name, z) for name in BASIS_NAMES],
            dim=-1,
        )
        gate = self._gate(self.in_logits, hard_top1)
        phi = torch.sum(vals * gate[None, :, :, :], dim=-1)
        hidden = self.hidden_bias[None, :] + torch.sum(
            self.in_amp[None, :, :] * phi,
            dim=-1,
        )

        out_slopes = 0.05 + F.softplus(self.out_slope_raw)
        zout = hidden * out_slopes[None, :] + self.out_shift[None, :]
        out_vals = torch.stack(
            [basis_torch(name, zout) for name in BASIS_NAMES],
            dim=-1,
        )
        out_gate = self._gate(self.out_logits, hard_top1)
        out_phi = torch.sum(out_vals * out_gate[None, :, :], dim=-1)
        return self.out_bias + torch.sum(
            self.out_amp[None, :] * out_phi,
            dim=-1,
        )

    def sparsity_loss(self) -> torch.Tensor:
        pin = torch.softmax(self.in_logits, dim=-1)
        pout = torch.softmax(self.out_logits, dim=-1)
        entropy_in = -torch.sum(
            pin * torch.log(torch.clamp(pin, min=1.0e-9)),
            dim=-1,
        )
        entropy_out = -torch.sum(
            pout * torch.log(torch.clamp(pout, min=1.0e-9)),
            dim=-1,
        )
        return (
            torch.mean(torch.abs(self.in_amp))
            + 0.5 * torch.mean(torch.abs(self.out_amp))
            + 0.015 * torch.mean(entropy_in)
            + 0.015 * torch.mean(entropy_out)
        )


def _robust_scale(v: np.ndarray, floor: float = 0.25) -> tuple[float, float]:
    mu = float(np.nanmedian(v))
    q25, q75 = np.nanpercentile(v, [25, 75])
    sig = float(max((q75 - q25) / 1.349, floor))
    return mu, sig


def _stabilize_temperature_normalization(label: pd.DataFrame, norm: dict) -> tuple[pd.DataFrame, dict]:
    """Keep sparse temperature datasets inside the symbolic KAN training range.

    Most waveforms are measured at 293 K, so an IQR-based scale collapses to
    the generic 0.25 floor and maps 323 K to +120.  Spline training can tolerate
    that poorly and a top-1 symbolic edge extrapolates even worse.  A 25 K
    temperature scale keeps the measured range near [-1, 1.2] while retaining
    the inverse-temperature feature used by Arrhenius-like rate laws.
    """
    label = label.copy()
    norm = dict(norm)
    temp_mu = float(np.nanmedian(label["temp_K"].to_numpy(float)))
    temp_sig = 25.0
    inv = 1000.0 / np.clip(label["temp_K"].to_numpy(float), 1.0, None)
    inv_mu, inv_sig = _robust_scale(inv, floor=0.25)
    norm["temp_norm"] = (temp_mu, temp_sig)
    norm["inv_temp_norm"] = (inv_mu, inv_sig)
    label["temp_norm"] = (label["temp_K"].to_numpy(float) - temp_mu) / temp_sig
    label["inv_temp_norm"] = (inv - inv_mu) / inv_sig
    return label, norm


def _feature_row(step: pd.Series, seg_pred: pd.DataFrame, prev_pred: pd.DataFrame, norm: dict, branch: str) -> np.ndarray:
    family = str(step["family"])
    prev_log = float(prev_pred["log_id_smooth"].median()) if len(prev_pred) else float(seg_pred["log_id_smooth"].iloc[0])
    seg_log0 = float(seg_pred["log_id_smooth"].iloc[0])
    cycle = float(max(int(step["step_seg_id"]) // 2, 0))
    raw = {
        "interface_top": 1.0 if family == "VTG" else 0.0,
        "process_capture": 1.0 if str(step["direction"]) == "up" else 0.0,
        "branch_slow": 1.0 if branch == "slow" else 0.0,
        "temperature_dataset": 1.0 if bool(step.get("temperature_dataset", "temperature" in str(step["curve_id"]).lower())) else 0.0,
        "to_v_norm": float(step["to_sweep_V"]),
        "from_v_norm": float(step["from_sweep_V"]),
        "delta_v_norm": float(step["delta_sweep_V"]),
        "temp_norm": float(step.get("temp_K", 293.0)),
        "inv_temp_norm": 1000.0 / max(float(step.get("temp_K", 293.0)), 1.0),
        "cycle_norm": cycle,
        "log_hold_s": np.log10(max(float(step["t_span_s"]), 1.0e-8)),
        "prev_log_id": prev_log,
        "seg_log_id0": seg_log0,
        "vds_norm": float(step["vds_V"]),
    }
    out = []
    for name in FEATURE_NAMES:
        if name in {"interface_top", "process_capture", "branch_slow", "temperature_dataset"}:
            out.append(raw[name])
        else:
            mu, sig = norm[name]
            out.append((raw[name] - mu) / sig)
    return np.asarray(out, dtype=np.float32)


def _make_waveform_table(norm: dict, max_points_per_segment: int = 240) -> list[dict]:
    step = pd.read_csv(RESULTS / "it_step_response_fit.csv")
    pred = pd.read_csv(RESULTS / "it_step_response_predictions.csv")
    step = step[(step["ok"] == True) & (step["large_amp_warning"] == False) & (step["log_rmse"] < 0.06)].copy()
    rows: list[dict] = []
    for _, r in step.iterrows():
        seg_id = int(r["step_seg_id"])
        seg = pred[(pred["curve_id"] == r["curve_id"]) & (pred["seg_id"] == seg_id)].copy()
        if len(seg) < 16:
            continue
        prev = pred[(pred["curve_id"] == r["curve_id"]) & (pred["seg_id"] == seg_id - 1)].copy()
        if len(seg) > max_points_per_segment:
            idx = np.unique(np.round(np.linspace(0, len(seg) - 1, max_points_per_segment)).astype(int))
            seg = seg.iloc[idx].copy()
        x_fast = _feature_row(r, seg, prev, norm, "fast")
        x_slow = _feature_row(r, seg, prev, norm, "slow")
        y_meas = seg["log_id_smooth"].to_numpy(float)
        net_span = float(y_meas[-1] - y_meas[0])
        reversal = 0.0
        if net_span > 0.0:
            reversal = max(float(np.max(y_meas[:-1]) - y_meas[-1]), 0.0)
        elif net_span < 0.0:
            reversal = max(float(y_meas[-1] - np.min(y_meas[:-1])), 0.0)
        amp_fast_teacher = float(r["amp_fast"])
        amp_slow_teacher = float(r["amp_slow"])
        teacher_competing = (
            amp_fast_teacher * amp_slow_teacher < 0.0
            and min(abs(amp_fast_teacher), abs(amp_slow_teacher)) >= 0.015
            and max(abs(amp_fast_teacher), abs(amp_slow_teacher)) < 4.5
            and float(r.get("log_rmse", 1.0)) < 0.06
        )
        # A slow competing trap may not reverse the current inside a short
        # measurement window. Accept either a resolved measured reversal or a
        # well-conditioned opposite-sign two-trap teacher decomposition.
        allow_competing_branches = (
            reversal > max(0.040, 0.35 * abs(net_span))
            or teacher_competing
        )
        rows.append(
            {
                "curve_id": str(r["curve_id"]),
                "seg_id": seg_id,
                "family": str(r["family"]),
                "process": "capture" if str(r["direction"]) == "up" else "emission",
                "temp_K": float(r.get("temp_K", 293.0)),
                "source_file": str(r.get("source_file", "")),
                "t": torch.as_tensor(seg["t_rel_s"].to_numpy(float), dtype=torch.float32),
                "y": torch.as_tensor(seg["log_id_smooth"].to_numpy(float), dtype=torch.float32),
                "y_teacher": torch.as_tensor(seg["log_id_fit"].to_numpy(float), dtype=torch.float32),
                "y_inf": float(r["y_inf"]),
                "amp_fast": float(r["amp_fast"]),
                "amp_slow": float(r["amp_slow"]),
                "x_fast": torch.as_tensor(x_fast.reshape(1, -1), dtype=torch.float32),
                "x_slow": torch.as_tensor(x_slow.reshape(1, -1), dtype=torch.float32),
                "tau_fast_label": float(r["tau_fast_s"]),
                "tau_slow_label": float(r["tau_slow_s"]),
                "fast_tau_unresolved": bool(r.get("fast_tau_unresolved", False)),
                "slow_tau_unresolved": bool(r.get("slow_tau_unresolved", False)),
                "teacher_fit_rmse_dec": float(r.get("log_rmse", 0.05)),
                "allow_competing_branches": bool(allow_competing_branches),
                "measured_reversal_dec": float(reversal),
            }
        )
    if not rows:
        raise RuntimeError("No waveform segments available for end-to-end refinement.")
    return rows


def _group_key(family: str, process: str, branch: str) -> tuple[str, str, str]:
    return family, branch, process


def _init_models(label: pd.DataFrame) -> dict[tuple[str, str, str], torch.nn.Module]:
    models: dict[tuple[str, str, str], torch.nn.Module] = {}
    for key, g in label.groupby(["family", "branch", "process"]):
        process = str(key[2])
        model = TwoLayerTop1KAN(
            len(FEATURE_NAMES),
            len(BASIS_NAMES),
            hidden=20 if process == "emission" else 12,
        )
        with torch.no_grad():
            model.out_bias.copy_(
                torch.tensor(
                    float(np.median(g["log_tau"].to_numpy(float))),
                    dtype=torch.float32,
                )
            )
        models[tuple(str(x) for x in key)] = model
    return models


def _init_coupling_models(waves: list[dict]) -> dict[tuple[str, str, str], torch.nn.Module]:
    models: dict[tuple[str, str, str], torch.nn.Module] = {}
    for family in sorted({str(row["family"]) for row in waves}):
        for process in sorted({str(row["process"]) for row in waves if str(row["family"]) == family}):
            subset = [row for row in waves if row["family"] == family and row["process"] == process]
            initial = {
                "amp_fast": float(np.median([row["amp_fast"] for row in subset])),
                "amp_slow": float(np.median([row["amp_slow"] for row in subset])),
            }
            for kind in ("amp_fast", "amp_slow"):
                # All three coupling outputs depend on interactions among
                # held bias, step direction, temperature, and history. Keep
                # the physically named decomposition, but do not force the
                # shallow fraction or competing exchange into a one-layer
                # additive GAM that cannot represent those interactions.
                model = TwoLayerTop1KAN(
                    len(FEATURE_NAMES),
                    len(BASIS_NAMES),
                    hidden=20 if process == "emission" else 12,
                )
                with torch.no_grad():
                    model.out_bias.copy_(
                        torch.tensor(initial[kind], dtype=torch.float32)
                    )
                models[(family, process, kind)] = model
    return models


def _inv_softplus(y: float) -> float:
    y = max(float(y), 1.0e-6)
    if y > 30.0:
        return y
    return float(np.log(np.expm1(y)))


def _warm_start_from_history_top1(models: dict[tuple[str, str, str], torch.nn.Module]) -> bool:
    summary_path = RESULTS / "srh_history_kan" / "srh_history_kan_summary.json"
    if not summary_path.exists():
        return False
    payload = json.loads(summary_path.read_text(encoding="utf-8-sig"))
    loaded_any = False
    for item in payload.get("formulas", []):
        key = (str(item["family"]), str(item["branch"]), str(item["process"]))
        if key not in models:
            continue
        model = models[key]
        if not isinstance(model, AdditiveKAN):
            continue
        with torch.no_grad():
            model.bias.copy_(torch.tensor(float(item["bias"]), dtype=torch.float32))
            model.amp.zero_()
            model.slope_raw.zero_()
            model.shift.zero_()
            model.logits.zero_()
            for edge in item.get("edges", []):
                if edge["feature"] not in FEATURE_NAMES or edge["basis"] not in BASIS_NAMES:
                    continue
                j = FEATURE_NAMES.index(edge["feature"])
                b = BASIS_NAMES.index(edge["basis"])
                model.amp[j] = float(edge["amp"])
                model.slope_raw[j] = _inv_softplus(float(edge["slope"]) - 0.05)
                model.shift[j] = float(edge["shift"])
                model.logits[j, :] = -6.0
                model.logits[j, b] = 6.0
                loaded_any = True
    return loaded_any


def _model_tau(models: dict, family: str, process: str, branch: str, x: torch.Tensor, hard_top1: bool = False) -> torch.Tensor:
    key = _group_key(family, process, branch)
    if key not in models:
        # Fallback to process-agnostic branch if a rare label group is absent.
        candidates = [k for k in models if k[0] == family and k[1] == branch]
        if not candidates:
            raise KeyError(f"No tau model for {key}")
        key = candidates[0]
    log_tau = models[key](x, hard_top1=hard_top1).squeeze(0)
    return torch.pow(torch.tensor(10.0, dtype=x.dtype), torch.clamp(log_tau, -8.0, 3.0))


def _project_same_sign_amplitudes(amp_fast: torch.Tensor, amp_slow: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Project unconstrained double-exp amplitudes to a monotone SRH current coupling.

    The per-segment teacher fit is a numerical denoising tool and can assign
    opposite signs to fast/slow amplitudes.  That creates hook-like waveforms
    even when the underlying trapped charge should move monotonically.  Preserve
    the initial total amplitude while assigning both branches the same sign.
    """
    net = amp_fast + amp_slow
    mag_f = torch.abs(amp_fast)
    mag_s = torch.abs(amp_slow)
    denom = torch.clamp(mag_f + mag_s, min=1.0e-9)
    sign = torch.sign(net)
    sign = torch.where(torch.abs(net) > 1.0e-6, sign, torch.sign(amp_fast + 1.0e-12))
    total = torch.abs(net)
    return sign * total * mag_f / denom, sign * total * mag_s / denom


def _predict_segment(
    models: dict,
    row: dict,
    hard_top1: bool = False,
    coupling_models: dict | None = None,
    dispersive: dict | None = None,
) -> torch.Tensor:
    tau_f = _model_tau(models, row["family"], row["process"], "fast", row["x_fast"], hard_top1=hard_top1)
    tau_s = _model_tau(models, row["family"], row["process"], "slow", row["x_slow"], hard_top1=hard_top1)
    t = row["t"]
    if coupling_models is None:
        y_inf = torch.as_tensor(row["y_inf"], dtype=t.dtype)
        amp_fast = torch.as_tensor(row["amp_fast"], dtype=t.dtype)
        amp_slow = torch.as_tensor(row["amp_slow"], dtype=t.dtype)
    else:
        # The post-step static current is supplied by the compact IV model in
        # deployment. The first measured sample can still contain the previous
        # bias point or instrument settling, so it must not be used as a trap
        # state anchor. The teacher intercept is the extraction-time proxy for
        # that post-step static current.
        y0 = row["y_teacher"][0]
        feature = row["x_fast"]
        amp_fast_raw = coupling_models[(row["family"], row["process"], "amp_fast")](
            feature, hard_top1=hard_top1
        ).squeeze(0)
        amp_slow_raw = coupling_models[(row["family"], row["process"], "amp_slow")](
            feature, hard_top1=hard_top1
        ).squeeze(0)
        amp_fast = 5.0 * torch.tanh(amp_fast_raw / 5.0)
        amp_slow = 5.0 * torch.tanh(amp_slow_raw / 5.0)
    if not bool(row.get("allow_competing_branches", False)):
        amp_fast, amp_slow = _project_same_sign_amplitudes(amp_fast, amp_slow)
    slow_resp = torch.exp(-t / torch.clamp(tau_s, min=1.0e-8))
    if dispersive is not None:
        key = f"{row['family']}|{row['process']}"
        if key in dispersive:
            mix = 0.55 * torch.sigmoid(dispersive[key]["mix_raw"])
            rho = 2.0 + 48.0 * torch.sigmoid(dispersive[key]["rho_raw"])
            slow_resp_2 = torch.exp(-t / torch.clamp(tau_s * rho, min=1.0e-8))
            slow_resp = (1.0 - mix) * slow_resp + mix * slow_resp_2
    fast_resp = torch.exp(-t / torch.clamp(tau_f, min=1.0e-8))
    if coupling_models is None:
        return y_inf + amp_fast * fast_resp + amp_slow * slow_resp
    # Anchoring at y0 removes the nondeployable per-segment y_inf calibration.
    # In circuit use, y0 is supplied by the static compact model and the current
    # trap states.
    return y0 + amp_fast * (fast_resp - 1.0) + amp_slow * (slow_resp - 1.0)


def _waveform_physics_penalty(
    pred: torch.Tensor,
    target: torch.Tensor,
    allow_competing_branches: bool = False,
) -> torch.Tensor:
    """Suppress nonphysical hook-like relaxation within a single voltage step.

    The instantaneous current jump belongs to the voltage transition between
    segments.  Once inside one constant-bias segment, SRH trap occupation should
    relax monotonically toward the new quasi-static state.  A small tolerance is
    kept because measured log-current contains noise.
    """
    if pred.numel() < 6:
        return pred.new_tensor(0.0)
    if allow_competing_branches:
        # The measured waveform has a statistically significant reversal.
        # Dynamic-shape supervision controls it; a monotonicity constraint
        # would incorrectly erase the competing capture/emission process.
        return pred.new_tensor(0.0)
    span = target[-1] - target[0]
    if torch.abs(span) < 0.020:
        return pred.new_tensor(0.0)
    direction = torch.sign(span)
    dy = pred[1:] - pred[:-1]
    wrong_slope = torch.relu(-direction * dy - 0.0015)
    # The first few samples are where artificial double-exponential hooks are
    # most visible.  Give them a mild extra weight without forcing a straight
    # line.
    weights = torch.ones_like(wrong_slope)
    weights[: min(5, weights.numel())] = 2.5
    return torch.mean(weights * wrong_slope * wrong_slope)


def _waveform_dynamic_trend_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Fit the fixed-bias relaxation independently of the absolute ID level.

    Pointwise log-current loss is dominated by the static current level.  This
    term instead supervises the measured drift amplitude and the response at
    several normalized times, so the fitted SRH rates must reproduce the
    transient trend rather than merely sit near the correct current decade.
    """
    n = int(pred.numel())
    if n < 8:
        return pred.new_tensor(0.0)

    target_delta = target - target[0]
    pred_delta = pred - pred[0]
    measured_span = target_delta[-1]
    dynamic_scale = torch.clamp(torch.abs(measured_span), min=0.006)

    # Compare the full normalized response.  The scale floor prevents noisy
    # nearly-flat segments from receiving disproportionate weight.
    shape = F.smooth_l1_loss(
        pred_delta / dynamic_scale,
        target_delta / dynamic_scale,
        beta=0.08,
    )

    endpoint = F.smooth_l1_loss(
        (pred[-1] - pred[0]) / dynamic_scale,
        measured_span / dynamic_scale,
        beta=0.06,
    )

    # Multi-time anchors constrain both the fast initial relaxation and the
    # slow tail without differentiating noisy samples point by point.
    anchor_idx = sorted({max(1, min(n - 1, int(round(f * (n - 1))))) for f in (0.03, 0.10, 0.30, 0.60, 1.0)})
    idx = torch.as_tensor(anchor_idx, dtype=torch.long, device=pred.device)
    pred_anchor = pred_delta[idx] / dynamic_scale
    target_anchor = target_delta[idx] / dynamic_scale
    anchors = F.smooth_l1_loss(pred_anchor, target_anchor, beta=0.06)

    pred_inc = pred_anchor[1:] - pred_anchor[:-1]
    target_inc = target_anchor[1:] - target_anchor[:-1]
    multiscale_slope = F.smooth_l1_loss(pred_inc, target_inc, beta=0.05)
    return shape + 0.80 * endpoint + 0.65 * anchors + 0.30 * multiscale_slope


def _endpoint_drift_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Strongly supervise the observable fixed-bias drift and its direction."""
    target_span = target[-1] - target[0]
    pred_span = pred[-1] - pred[0]
    magnitude = F.smooth_l1_loss(pred_span, target_span, beta=0.025)
    if torch.abs(target_span) < 0.003:
        return 0.35 * magnitude
    direction = torch.sign(target_span)
    sign_margin = torch.relu(0.002 - direction * pred_span)
    relative = F.smooth_l1_loss(
        pred_span / torch.clamp(torch.abs(target_span), min=0.003),
        target_span / torch.clamp(torch.abs(target_span), min=0.003),
        beta=0.08,
    )
    return magnitude + 1.25 * relative + 12.0 * sign_margin * sign_margin


def _waveform_importance(row: dict) -> float:
    """Prioritize observable drift while retaining all cleaned waveforms."""
    drift = abs(float(row["y"][-1] - row["y"][0]))
    weight = 1.0 + min(drift / 0.08, 3.0)
    if bool(row.get("allow_competing_branches", False)):
        weight *= 1.25
    if "temperature" in str(row.get("source_file", "")).lower():
        weight *= 1.15
    return float(weight)


def _weighted_loss_mean(losses: list[torch.Tensor], weights: list[float]) -> torch.Tensor:
    if not losses:
        return torch.tensor(0.0)
    values = torch.stack(losses)
    w = values.new_tensor(weights)
    return torch.sum(values * w) / torch.clamp(torch.sum(w), min=1.0e-12)


def _weighted_tail_mean(
    losses: list[torch.Tensor],
    weights: list[float],
    fraction: float = 0.20,
) -> torch.Tensor:
    """CVaR-style mean that keeps catastrophic waveform failures visible."""
    if not losses:
        return torch.tensor(0.0)
    values = torch.stack(losses)
    w = values.new_tensor(weights)
    weighted = values * w / torch.clamp(torch.mean(w), min=1.0e-12)
    k = max(1, int(math.ceil(float(fraction) * weighted.numel())))
    return torch.topk(weighted, k=k, largest=True).values.mean()


def _dynamic_metrics_np(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    scale = max(abs(float(target[-1] - target[0])), 0.006)
    p = (pred - pred[0]) / scale
    t = (target - target[0]) / scale
    endpoint_error = float((pred[-1] - pred[0]) - (target[-1] - target[0]))
    return {
        "dynamic_shape_rmse_norm": float(np.sqrt(np.mean((p - t) ** 2))),
        "endpoint_drift_error_dec": endpoint_error,
        "endpoint_drift_abs_error_dec": abs(endpoint_error),
        "measured_drift_dec": float(target[-1] - target[0]),
        "predicted_drift_dec": float(pred[-1] - pred[0]),
    }


def _top1_formula(model: torch.nn.Module) -> dict:
    if isinstance(model, TwoLayerTop1KAN):
        with torch.no_grad():
            in_prob = torch.softmax(model.in_logits, dim=-1)
            out_prob = torch.softmax(model.out_logits, dim=-1)
            in_weight, in_top = torch.topk(
                in_prob,
                k=model.symbolic_topk,
                dim=-1,
            )
            out_weight, out_top = torch.topk(
                out_prob,
                k=model.symbolic_topk,
                dim=-1,
            )
            in_weight = (
                in_weight
                / torch.clamp(in_weight.sum(dim=-1, keepdim=True), min=1.0e-12)
            ).cpu().numpy()
            out_weight = (
                out_weight
                / torch.clamp(out_weight.sum(dim=-1, keepdim=True), min=1.0e-12)
            ).cpu().numpy()
            in_top = in_top.cpu().numpy()
            out_top = out_top.cpu().numpy()
            in_slopes = (
                0.05 + F.softplus(model.in_slope_raw)
            ).cpu().numpy()
            out_slopes = (
                0.05 + F.softplus(model.out_slope_raw)
            ).cpu().numpy()
            hidden_bias = model.hidden_bias.cpu().numpy()
            in_amp = model.in_amp.cpu().numpy()
            in_shift = model.in_shift.cpu().numpy()
            out_amp = model.out_amp.cpu().numpy()
            out_shift = model.out_shift.cpu().numpy()
            out_bias = float(model.out_bias.cpu())
        hidden_nodes = []
        for h in range(model.hidden):
            edges = []
            for j, name in enumerate(FEATURE_NAMES):
                if abs(float(in_amp[h, j])) < 5.0e-3:
                    continue
                for rank in range(model.symbolic_topk):
                    component_amp = float(in_amp[h, j] * in_weight[h, j, rank])
                    if abs(component_amp) < 2.5e-3:
                        continue
                    edges.append(
                        {
                            "feature": name,
                            "basis": BASIS_NAMES[int(in_top[h, j, rank])],
                            "amp": component_amp,
                            "slope": float(in_slopes[h, j]),
                            "shift": float(in_shift[h, j]),
                        }
                    )
            output_components = []
            for rank in range(model.symbolic_topk):
                component_amp = float(out_amp[h] * out_weight[h, rank])
                if abs(component_amp) < 2.5e-3:
                    continue
                output_components.append(
                    {
                        "basis": BASIS_NAMES[int(out_top[h, rank])],
                        "amp": component_amp,
                        "slope": float(out_slopes[h]),
                        "shift": float(out_shift[h]),
                    }
                )
            if not output_components:
                continue
            hidden_nodes.append(
                {
                    "bias": float(hidden_bias[h]),
                    "edges": edges,
                    "output_components": output_components,
                }
            )
        return {
            "architecture": f"two_layer_top{model.symbolic_topk}",
            "symbolic_topk": model.symbolic_topk,
            "bias": out_bias,
            "hidden_nodes": hidden_nodes,
        }

    with torch.no_grad():
        p = torch.softmax(model.logits, dim=-1)
        top = torch.argmax(p, dim=-1).cpu().numpy()
        slopes = (0.05 + F.softplus(model.slope_raw)).cpu().numpy()
        shifts = model.shift.cpu().numpy()
        amps = model.amp.cpu().numpy()
        bias = float(model.bias.cpu())
    edges = []
    for j, name in enumerate(FEATURE_NAMES):
        if abs(float(amps[j])) < 1.0e-2:
            continue
        edges.append(
            {
                "feature": name,
                "basis": BASIS_NAMES[int(top[j])],
                "amp": float(amps[j]),
                "slope": float(slopes[j]),
                "shift": float(shifts[j]),
            }
        )
    return {"architecture": "additive_top1", "bias": bias, "edges": edges}


def _formula_eval(formula: dict, x: np.ndarray) -> np.ndarray:
    if str(formula.get("architecture", "")).startswith("two_layer_top"):
        y = np.full(x.shape[0], float(formula["bias"]), dtype=float)
        for node in formula["hidden_nodes"]:
            hidden = np.full(x.shape[0], float(node["bias"]), dtype=float)
            for edge in node["edges"]:
                j = FEATURE_NAMES.index(edge["feature"])
                hidden += float(edge["amp"]) * basis_np(
                    edge["basis"],
                    float(edge["slope"]) * x[:, j] + float(edge["shift"]),
                )
            for component in node["output_components"]:
                y += float(component["amp"]) * basis_np(
                    component["basis"],
                    float(component["slope"]) * hidden
                    + float(component["shift"]),
                )
        return y
    y = np.full(x.shape[0], float(formula["bias"]), dtype=float)
    for edge in formula["edges"]:
        j = FEATURE_NAMES.index(edge["feature"])
        y += float(edge["amp"]) * basis_np(edge["basis"], float(edge["slope"]) * x[:, j] + float(edge["shift"]))
    return y


def _append_formula_comments(lines: list[str], label: str, formula: dict) -> None:
    """Write either additive or nested top-1 formulas in a uniform VA-readable form."""
    lines.append(f"// [{label}] bias={formula['bias']:.8e}")
    if str(formula.get("architecture", "")).startswith("two_layer_top"):
        for h, node in enumerate(formula["hidden_nodes"]):
            lines.append(f"//   hidden[{h}] bias={node['bias']:.8e}")
            for edge in node["edges"]:
                lines.append(
                    f"//     + {edge['amp']:.8e} * {edge['basis']}("
                    f"{edge['slope']:.8e}*{edge['feature']} + {edge['shift']:.8e})"
                )
            for component in node["output_components"]:
                lines.append(
                    f"//   output + {component['amp']:.8e} * "
                    f"{component['basis']}({component['slope']:.8e}*"
                    f"hidden[{h}] + {component['shift']:.8e})"
                )
        return
    for edge in formula["edges"]:
        lines.append(
            f"//   + {edge['amp']:.8e} * {edge['basis']}("
            f"{edge['slope']:.8e}*{edge['feature']} + {edge['shift']:.8e})"
        )


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="End-to-end SRH I-t waveform refinement.")
    ap.add_argument("--epochs", type=int, default=1500)
    ap.add_argument("--max-points-per-segment", type=int, default=240)
    ap.add_argument("--print-every", type=int, default=100)
    ap.add_argument("--seed", type=int, default=-1, help="Set >=0 for a deterministic diagnostic rerun.")
    ap.add_argument("--drift-label-weight", type=float, default=7.0)
    ap.add_argument("--fraction-label-weight", type=float, default=1.50)
    ap.add_argument("--competition-label-weight", type=float, default=0.90)
    ap.add_argument("--coupling-label-weight", type=float, default=0.75)
    ap.add_argument("--coupling-hard-weight", type=float, default=0.85)
    ap.add_argument("--endpoint-weight", type=float, default=2.00)
    ap.add_argument("--hard-endpoint-weight", type=float, default=3.00)
    ap.add_argument("--teacher-distill-weight", type=float, default=15.00)
    ap.add_argument("--hard-teacher-distill-weight", type=float, default=20.00)
    ap.add_argument("--tail-distill-weight", type=float, default=10.00)
    ap.add_argument("--tail-distill-fraction", type=float, default=0.20)
    ap.add_argument("--rate-distill-weight", type=float, default=4.00)
    ap.add_argument("--hard-rate-distill-weight", type=float, default=6.00)
    ap.add_argument("--unresolved-rate-confidence", type=float, default=0.35)
    ap.add_argument("--rate-pretrain-epochs", type=int, default=600)
    ap.add_argument("--rate-pretrain-lr", type=float, default=2.5e-3)
    ap.add_argument("--coupling-pretrain-epochs", type=int, default=800)
    ap.add_argument("--coupling-pretrain-lr", type=float, default=3.0e-3)
    ap.add_argument("--drift-lr-mult", type=float, default=2.40)
    ap.add_argument("--fraction-lr-mult", type=float, default=0.50)
    ap.add_argument("--competition-lr-mult", type=float, default=0.30)
    ap.add_argument("--deployment-eval-period", type=int, default=50)
    ap.add_argument("--deployment-endpoint-weight", type=float, default=0.30)
    ap.add_argument("--deployment-shape-weight", type=float, default=0.02)
    ap.add_argument("--deployment-median-weight", type=float, default=0.10)
    ap.add_argument("--deployment-p90-weight", type=float, default=0.80)
    ap.add_argument("--deployment-max-weight", type=float, default=0.30)
    ap.add_argument("--deployment-teacher-weight", type=float, default=4.00)
    ap.add_argument("--drift-polish-epochs", type=int, default=90)
    ap.add_argument("--drift-polish-lr", type=float, default=4.0e-4)
    ap.add_argument(
        "--drift-polish-min-score-improvement",
        type=float,
        default=0.0,
        help=(
            "Minimum deployment-score improvement required to retain the final "
            "drift-only polish. Otherwise the pre-polish checkpoint is restored."
        ),
    )
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    if args.seed >= 0:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
    OUT.mkdir(parents=True, exist_ok=True)
    label, norm = build_label_table()
    label, norm = _stabilize_temperature_normalization(label, norm)
    waves = _make_waveform_table(norm, max_points_per_segment=args.max_points_per_segment)
    for i, row in enumerate(waves):
        row["wave_index"] = i
    models = _init_models(label)
    coupling_models = _init_coupling_models(waves)
    warm_started = _warm_start_from_history_top1(models)
    rate_params = []
    for m in models.values():
        rate_params += list(m.parameters())
    coupling_params: dict[str, list[torch.nn.Parameter]] = {
        "amp_fast": [],
        "amp_slow": [],
    }
    for key, model in coupling_models.items():
        coupling_params[key[2]] += list(model.parameters())
    params = list(rate_params)
    for group in coupling_params.values():
        params += group
    dispersive = {}
    for family in sorted(label["family"].unique()):
        for process in sorted(label["process"].unique()):
            key = f"{family}|{process}"
            # The accepted teacher is a two-trap SRH model. Keep the deployable
            # student on the same two-state manifold. The former trainable
            # slow-trap split introduced an unidentifiable third RC pole and
            # was the main source of teacher/student curvature mismatch.
            dispersive[key] = {
                "mix_raw": torch.tensor(-20.0, dtype=torch.float32),
                "rho_raw": torch.tensor(-20.0, dtype=torch.float32),
            }
    base_lr = 1.6e-3 if warm_started else 2.5e-3
    opt = torch.optim.AdamW(
        [
            {"params": rate_params, "lr": base_lr},
            {"params": coupling_params["amp_fast"], "lr": base_lr * args.drift_lr_mult},
            {"params": coupling_params["amp_slow"], "lr": base_lr * args.drift_lr_mult},
        ],
        weight_decay=2.0e-4,
    )

    # Label tensors keep the learned rates identifiable while waveform loss
    # refines them against every measured I-t point.
    label_tensors = {}
    for key, g in label.groupby(["family", "branch", "process"]):
        key = tuple(str(x) for x in key)
        x = torch.as_tensor(g[FEATURE_NAMES].to_numpy(float), dtype=torch.float32)
        y = torch.as_tensor(g["log_tau"].to_numpy(float), dtype=torch.float32)
        label_tensors[key] = (x, y)
    # Full teacher parameter table used for student distillation. Identifiable
    # labels remain highest confidence, while unresolved poles still provide a
    # weak target instead of being left unconstrained and compensated by
    # erroneous coupling amplitudes.
    rate_wave_label_tensors = {}
    for key in models:
        family, branch, process = key
        subset = [
            row
            for row in waves
            if row["family"] == family and row["process"] == process
        ]
        x_rows = []
        y_rows = []
        weights = []
        for row in subset:
            unresolved = bool(
                row[
                    "fast_tau_unresolved"
                    if branch == "fast"
                    else "slow_tau_unresolved"
                ]
            )
            fit_quality = float(
                np.clip(
                    1.0 - float(row["teacher_fit_rmse_dec"]) / 0.08,
                    0.20,
                    1.0,
                )
            )
            confidence = (
                float(args.unresolved_rate_confidence)
                if unresolved
                else 1.0
            )
            x_rows.append(row["x_fast" if branch == "fast" else "x_slow"])
            y_rows.append(
                math.log10(
                    max(
                        float(
                            row[
                                "tau_fast_label"
                                if branch == "fast"
                                else "tau_slow_label"
                            ]
                        ),
                        1.0e-8,
                    )
                )
            )
            weights.append(confidence * fit_quality)
        rate_wave_label_tensors[key] = (
            torch.cat(x_rows, dim=0),
            torch.as_tensor(y_rows, dtype=torch.float32),
            torch.as_tensor(weights, dtype=torch.float32),
        )
    coupling_label_tensors = {}
    for key in coupling_models:
        family, process, kind = key
        subset = [row for row in waves if row["family"] == family and row["process"] == process]
        targets = []
        target_weights = []
        for row in subset:
            target = float(row[kind])
            targets.append(target)
            duration = float(row["t"][-1])
            both_resolved = not (
                bool(row["fast_tau_unresolved"])
                or bool(row["slow_tau_unresolved"])
            )
            fit_quality = float(
                np.clip(
                    1.0 - float(row["teacher_fit_rmse_dec"]) / 0.08,
                    0.15,
                    1.0,
                )
            )
            if kind == "amp_fast":
                ident = 1.0 if not bool(row["fast_tau_unresolved"]) else 0.25
            else:
                ident = 1.0 if both_resolved and duration >= 0.5 else 0.12
            target_weights.append(ident * fit_quality)
        coupling_label_tensors[key] = (
            torch.cat([row["x_fast"] for row in subset], dim=0),
            torch.as_tensor(targets, dtype=torch.float32),
            torch.as_tensor(target_weights, dtype=torch.float32),
        )

    # Identify each SRH rate law before waveform optimization. A random
    # two-layer student can otherwise compensate a wrong tau with coupling
    # amplitude, producing a deceptively small average error and visibly
    # incorrect relaxation curvature.
    if args.rate_pretrain_epochs > 0:
        rate_pre_opt = torch.optim.AdamW(
            rate_params,
            lr=args.rate_pretrain_lr,
            weight_decay=1.0e-4,
        )
        for pre_ep in range(args.rate_pretrain_epochs):
            rate_pre_opt.zero_grad()
            terms = []
            hard_weight = min(max((pre_ep - 60) / 240.0, 0.0), 1.0)
            for key, model in models.items():
                x, y, confidence = rate_wave_label_tensors[key]
                soft = model(x, hard_top1=False)
                hard = model(x, hard_top1=True)
                soft_vec = F.smooth_l1_loss(
                    soft,
                    y,
                    beta=0.08,
                    reduction="none",
                )
                hard_vec = F.smooth_l1_loss(
                    hard,
                    y,
                    beta=0.10,
                    reduction="none",
                )
                k_tail = max(1, int(math.ceil(0.25 * soft_vec.numel())))
                soft_tail = torch.topk(soft_vec, k=k_tail).values.mean()
                hard_tail = torch.topk(hard_vec, k=k_tail).values.mean()
                terms.append(
                    torch.sum(confidence * soft_vec)
                    / torch.clamp(torch.sum(confidence), min=1.0e-9)
                    + 1.25 * soft_tail
                    + hard_weight
                    * (
                        1.40
                        * torch.sum(confidence * hard_vec)
                        / torch.clamp(torch.sum(confidence), min=1.0e-9)
                        + 1.75 * hard_tail
                        + 0.20 * torch.mean((hard - soft.detach()) ** 2)
                    )
                    + 0.003 * model.sparsity_loss()
                )
            rate_pre_loss = torch.stack(terms).mean()
            rate_pre_loss.backward()
            torch.nn.utils.clip_grad_norm_(rate_params, 5.0)
            rate_pre_opt.step()
        print(
            "SRH rate pretraining complete: "
            f"epochs={args.rate_pretrain_epochs}, "
            f"loss={float(rate_pre_loss.detach()):.6g}"
        )

    # First identify the physically named coupling outputs in parameter space.
    # Starting waveform training from an untrained drift/fraction/competition
    # map made the optimizer settle for a low average error while sacrificing
    # isolated high-drift traces.
    if args.coupling_pretrain_epochs > 0:
        pretrain_params = [
            p for model in coupling_models.values() for p in model.parameters()
        ]
        pre_opt = torch.optim.AdamW(
            pretrain_params,
            lr=args.coupling_pretrain_lr,
            weight_decay=1.0e-4,
        )
        for pre_ep in range(args.coupling_pretrain_epochs):
            pre_opt.zero_grad()
            terms = []
            for key, model in coupling_models.items():
                x, y, label_confidence = coupling_label_tensors[key]
                kind = key[2]
                raw = model(x, hard_top1=False)
                hard_raw = model(x, hard_top1=True)
                if kind in {"amp_fast", "amp_slow"}:
                    soft = 5.0 * torch.tanh(raw / 5.0)
                    hard = 5.0 * torch.tanh(hard_raw / 5.0)
                    sample_weight = 1.0 + torch.clamp(
                        torch.abs(y) / 0.06,
                        max=5.0,
                    )
                    soft_vec = sample_weight * F.smooth_l1_loss(
                        soft,
                        y,
                        beta=0.035,
                        reduction="none",
                    )
                    hard_vec = sample_weight * F.smooth_l1_loss(
                        hard,
                        y,
                        beta=0.045,
                        reduction="none",
                    )
                k_tail = max(1, int(math.ceil(0.25 * soft_vec.numel())))
                tail = torch.topk(soft_vec, k=k_tail).values.mean()
                hard_tail = torch.topk(hard_vec, k=k_tail).values.mean()
                hard_weight = min(
                    max((pre_ep - 100) / 300.0, 0.0),
                    1.0,
                )
                terms.append(
                    torch.sum(label_confidence * soft_vec)
                    / torch.clamp(torch.sum(label_confidence), min=1.0e-9)
                    + 1.5 * tail
                    + hard_weight
                    * (
                        1.25
                        * torch.sum(label_confidence * hard_vec)
                        / torch.clamp(torch.sum(label_confidence), min=1.0e-9)
                        + 1.75 * hard_tail
                        + 0.20 * torch.mean((hard - soft.detach()) ** 2)
                    )
                    + 0.004 * model.sparsity_loss()
                )
            pre_loss = torch.stack(terms).mean()
            pre_loss.backward()
            torch.nn.utils.clip_grad_norm_(pretrain_params, 5.0)
            pre_opt.step()
        print(
            "Coupling parameter pretraining complete: "
            f"epochs={args.coupling_pretrain_epochs}, "
            f"loss={float(pre_loss.detach()):.6g}"
        )

    def deployment_metrics() -> dict[str, float]:
        rmse = []
        teacher_rmse = []
        endpoint = []
        shape = []
        importance = []
        wrong_slope = []
        sign_ok = []
        for row in waves:
            pred = _predict_segment(
                models,
                row,
                hard_top1=True,
                coupling_models=coupling_models,
                dispersive=dispersive,
            )
            target = row["y"]
            rmse.append(float(torch.mean((pred - target) ** 2).sqrt()))
            teacher_rmse.append(
                float(torch.mean((pred - row["y_teacher"]) ** 2).sqrt())
            )
            dm = _dynamic_metrics_np(
                pred.detach().cpu().numpy(),
                row["y_teacher"].detach().cpu().numpy(),
            )
            endpoint.append(dm["endpoint_drift_abs_error_dec"])
            shape.append(dm["dynamic_shape_rmse_norm"])
            importance.append(_waveform_importance(row))
            measured_drift = float(row["y_teacher"][-1] - row["y_teacher"][0])
            predicted_drift = float(pred[-1] - pred[0])
            if (
                abs(measured_drift) >= 0.003
                and not bool(row.get("allow_competing_branches", False))
            ):
                direction = 1.0 if measured_drift > 0.0 else -1.0
                wrong = torch.mean(
                    (
                        direction * (pred[1:] - pred[:-1]) < -0.0015
                    ).to(torch.float32)
                )
            else:
                wrong = pred.new_tensor(0.0)
            wrong_slope.append(float(wrong))
            if abs(measured_drift) >= 0.003:
                sign_ok.append(float(measured_drift * predicted_drift > 0.0))
        mean_rmse = float(np.mean(rmse))
        median_rmse = float(np.median(rmse))
        p90_rmse = float(np.quantile(rmse, 0.90))
        max_rmse = float(np.max(rmse))
        mean_teacher_rmse = float(np.mean(teacher_rmse))
        p90_teacher_rmse = float(np.quantile(teacher_rmse, 0.90))
        mean_endpoint = float(np.mean(endpoint))
        mean_shape = float(np.mean(shape))
        weighted_rmse = float(np.average(rmse, weights=importance))
        weighted_endpoint = float(np.average(endpoint, weights=importance))
        score = (
            0.35 * mean_rmse
            + 0.65 * weighted_rmse
            + args.deployment_endpoint_weight * weighted_endpoint
            + args.deployment_shape_weight * mean_shape
            + args.deployment_median_weight * median_rmse
            + args.deployment_p90_weight * p90_rmse
            + args.deployment_max_weight * max_rmse
            + args.deployment_teacher_weight
            * (0.65 * mean_teacher_rmse + 0.35 * p90_teacher_rmse)
            + 1.25 * (1.0 - (float(np.mean(sign_ok)) if sign_ok else 1.0))
        )
        return {
            "score": score,
            "mean_rmse_dec": mean_rmse,
            "median_rmse_dec": median_rmse,
            "p90_rmse_dec": p90_rmse,
            "max_rmse_dec": max_rmse,
            "mean_teacher_distill_rmse_dec": mean_teacher_rmse,
            "p90_teacher_distill_rmse_dec": p90_teacher_rmse,
            "mean_endpoint_error_dec": mean_endpoint,
            "weighted_rmse_dec": weighted_rmse,
            "weighted_endpoint_error_dec": weighted_endpoint,
            "mean_shape_rmse_norm": mean_shape,
            "mean_wrong_slope_fraction": float(np.mean(wrong_slope)),
            "sign_accuracy": float(np.mean(sign_ok)) if sign_ok else 1.0,
        }

    def snapshot_state() -> dict:
        return {
            "rate_models": {key: copy.deepcopy(model.state_dict()) for key, model in models.items()},
            "coupling_models": {
                key: copy.deepcopy(model.state_dict()) for key, model in coupling_models.items()
            },
            "dispersive": {
                key: {
                    name: value.detach().clone()
                    for name, value in branch.items()
                }
                for key, branch in dispersive.items()
            },
        }

    def restore_state(state: dict) -> None:
        for key, model_state in state["rate_models"].items():
            models[key].load_state_dict(model_state)
        for key, model_state in state["coupling_models"].items():
            coupling_models[key].load_state_dict(model_state)
        with torch.no_grad():
            for key, branch in state["dispersive"].items():
                for name, value in branch.items():
                    dispersive[key][name].copy_(value)

    group_keys = sorted(
        {
            (str(row["family"]), str(row["process"]))
            for row in waves
        }
    )

    def group_score(family: str, process: str) -> dict[str, float]:
        measured_errors = []
        teacher_errors = []
        endpoint_errors = []
        sign_errors = []
        for row in waves:
            if row["family"] != family or row["process"] != process:
                continue
            pred = _predict_segment(
                models,
                row,
                hard_top1=True,
                coupling_models=coupling_models,
                dispersive=dispersive,
            )
            measured_errors.append(
                float(torch.mean((pred - row["y"]) ** 2).sqrt())
            )
            teacher_errors.append(
                float(torch.mean((pred - row["y_teacher"]) ** 2).sqrt())
            )
            endpoint_errors.append(
                abs(
                    float(pred[-1] - pred[0])
                    - float(row["y_teacher"][-1] - row["y_teacher"][0])
                )
            )
            measured_span = float(
                row["y_teacher"][-1] - row["y_teacher"][0]
            )
            predicted_span = float(pred[-1] - pred[0])
            if abs(measured_span) >= 0.003:
                sign_errors.append(float(measured_span * predicted_span <= 0.0))
        if not measured_errors:
            return {"score": float("inf")}
        score = (
            0.15 * float(np.mean(measured_errors))
            + 1.00 * float(np.mean(teacher_errors))
            + 0.80 * float(np.quantile(teacher_errors, 0.90))
            + 0.30 * float(np.max(teacher_errors))
            + 0.10 * float(np.mean(endpoint_errors))
            + 0.80 * (float(np.mean(sign_errors)) if sign_errors else 0.0)
        )
        return {
            "score": score,
            "mean_measured_rmse_dec": float(np.mean(measured_errors)),
            "mean_teacher_rmse_dec": float(np.mean(teacher_errors)),
            "p90_teacher_rmse_dec": float(np.quantile(teacher_errors, 0.90)),
            "max_teacher_rmse_dec": float(np.max(teacher_errors)),
            "mean_endpoint_error_dec": float(np.mean(endpoint_errors)),
        }

    def snapshot_group(family: str, process: str) -> dict:
        return {
            "rate_models": {
                key: copy.deepcopy(model.state_dict())
                for key, model in models.items()
                if key[0] == family and key[2] == process
            },
            "coupling_models": {
                key: copy.deepcopy(model.state_dict())
                for key, model in coupling_models.items()
                if key[0] == family and key[1] == process
            },
            "dispersive": {
                name: value.detach().clone()
                for name, value in dispersive[f"{family}|{process}"].items()
            },
        }

    def restore_group(family: str, process: str, state: dict) -> None:
        for key, model_state in state["rate_models"].items():
            models[key].load_state_dict(model_state)
        for key, model_state in state["coupling_models"].items():
            coupling_models[key].load_state_dict(model_state)
        with torch.no_grad():
            for name, value in state["dispersive"].items():
                dispersive[f"{family}|{process}"][name].copy_(value)

    history = []
    best_deployment: dict[str, float] | None = None
    best_state: dict | None = None
    best_epoch = -1
    best_group_metrics: dict[tuple[str, str], dict[str, float]] = {}
    best_group_states: dict[tuple[str, str], dict] = {}
    best_group_epochs: dict[tuple[str, str], int] = {}
    for ep in range(args.epochs):
        opt.zero_grad()
        wave_losses = []
        phys_losses = []
        trend_losses = []
        endpoint_losses = []
        teacher_losses = []
        rate_distill_losses = []
        wave_weights = []
        for row in waves:
            pred = _predict_segment(
                models, row, hard_top1=False, coupling_models=coupling_models, dispersive=dispersive
            )
            wave_losses.append(F.smooth_l1_loss(pred, row["y"], beta=0.025))
            phys_losses.append(
                _waveform_physics_penalty(
                    pred,
                    row["y_teacher"],
                    row["allow_competing_branches"],
                )
            )
            trend_losses.append(_waveform_dynamic_trend_loss(pred, row["y_teacher"]))
            endpoint_losses.append(_endpoint_drift_loss(pred, row["y_teacher"]))
            teacher_losses.append(
                F.smooth_l1_loss(pred, row["y_teacher"], beta=0.020)
            )
            rate_terms = []
            for branch, x, label_name, flag_name in [
                ("fast", row["x_fast"], "tau_fast_label", "fast_tau_unresolved"),
                ("slow", row["x_slow"], "tau_slow_label", "slow_tau_unresolved"),
            ]:
                tau_pred = _model_tau(
                    models,
                    row["family"],
                    row["process"],
                    branch,
                    x,
                    hard_top1=False,
                )
                log_tau_pred = torch.log10(torch.clamp(tau_pred, min=1.0e-8))
                log_tau_target = log_tau_pred.new_tensor(
                    math.log10(max(float(row[label_name]), 1.0e-8))
                )
                confidence = (
                    float(args.unresolved_rate_confidence)
                    if bool(row[flag_name])
                    else 1.0
                )
                fit_quality = float(
                    np.clip(
                        1.0 - float(row["teacher_fit_rmse_dec"]) / 0.08,
                        0.25,
                        1.0,
                    )
                )
                rate_terms.append(
                    confidence
                    * fit_quality
                    * F.smooth_l1_loss(
                        log_tau_pred,
                        log_tau_target,
                        beta=0.20,
                    )
                )
            rate_distill_losses.append(torch.stack(rate_terms).mean())
            wave_weights.append(_waveform_importance(row))
        wave_loss = _weighted_loss_mean(wave_losses, wave_weights)
        phys_loss = _weighted_loss_mean(phys_losses, wave_weights)
        trend_loss = _weighted_loss_mean(trend_losses, wave_weights)
        endpoint_loss = _weighted_loss_mean(endpoint_losses, wave_weights)
        teacher_loss = _weighted_loss_mean(teacher_losses, wave_weights)
        teacher_tail_loss = _weighted_tail_mean(
            teacher_losses,
            wave_weights,
            fraction=args.tail_distill_fraction,
        )
        rate_distill_loss = _weighted_loss_mean(
            rate_distill_losses,
            wave_weights,
        )
        hard_wave_loss = torch.tensor(0.0)
        hard_phys_loss = torch.tensor(0.0)
        hard_trend_loss = torch.tensor(0.0)
        hard_endpoint_loss = torch.tensor(0.0)
        hard_teacher_loss = torch.tensor(0.0)
        hard_teacher_tail_loss = torch.tensor(0.0)
        hard_rate_distill_loss = torch.tensor(0.0)
        if ep > 80:
            hard_wave_losses = []
            hard_phys_losses = []
            hard_trend_losses = []
            hard_endpoint_losses = []
            hard_teacher_losses = []
            hard_rate_distill_losses = []
            hard_weights = []
            for row in waves:
                hard_pred = _predict_segment(
                    models, row, hard_top1=True, coupling_models=coupling_models, dispersive=dispersive
                )
                hard_wave_losses.append(F.smooth_l1_loss(hard_pred, row["y"], beta=0.030))
                hard_phys_losses.append(
                    _waveform_physics_penalty(
                        hard_pred,
                        row["y_teacher"],
                        row["allow_competing_branches"],
                    )
                )
                hard_trend_losses.append(
                    _waveform_dynamic_trend_loss(hard_pred, row["y_teacher"])
                )
                hard_endpoint_losses.append(
                    _endpoint_drift_loss(hard_pred, row["y_teacher"])
                )
                hard_teacher_losses.append(
                    F.smooth_l1_loss(hard_pred, row["y_teacher"], beta=0.025)
                )
                hard_rate_terms = []
                for branch, x, label_name, flag_name in [
                    ("fast", row["x_fast"], "tau_fast_label", "fast_tau_unresolved"),
                    ("slow", row["x_slow"], "tau_slow_label", "slow_tau_unresolved"),
                ]:
                    tau_pred = _model_tau(
                        models,
                        row["family"],
                        row["process"],
                        branch,
                        x,
                        hard_top1=True,
                    )
                    log_tau_pred = torch.log10(
                        torch.clamp(tau_pred, min=1.0e-8)
                    )
                    log_tau_target = log_tau_pred.new_tensor(
                        math.log10(max(float(row[label_name]), 1.0e-8))
                    )
                    confidence = (
                        float(args.unresolved_rate_confidence)
                        if bool(row[flag_name])
                        else 1.0
                    )
                    fit_quality = float(
                        np.clip(
                            1.0 - float(row["teacher_fit_rmse_dec"]) / 0.08,
                            0.25,
                            1.0,
                        )
                    )
                    hard_rate_terms.append(
                        confidence
                        * fit_quality
                        * F.smooth_l1_loss(
                            log_tau_pred,
                            log_tau_target,
                            beta=0.25,
                        )
                    )
                hard_rate_distill_losses.append(
                    torch.stack(hard_rate_terms).mean()
                )
                hard_weights.append(_waveform_importance(row))
            hard_wave_loss = _weighted_loss_mean(hard_wave_losses, hard_weights)
            hard_phys_loss = _weighted_loss_mean(hard_phys_losses, hard_weights)
            hard_trend_loss = _weighted_loss_mean(hard_trend_losses, hard_weights)
            hard_endpoint_loss = _weighted_loss_mean(hard_endpoint_losses, hard_weights)
            hard_teacher_loss = _weighted_loss_mean(
                hard_teacher_losses,
                hard_weights,
            )
            hard_teacher_tail_loss = _weighted_tail_mean(
                hard_teacher_losses,
                hard_weights,
                fraction=args.tail_distill_fraction,
            )
            hard_rate_distill_loss = _weighted_loss_mean(
                hard_rate_distill_losses,
                hard_weights,
            )
        label_losses = []
        sparse_losses = []
        hard_losses = []
        for key, model in models.items():
            if key not in label_tensors:
                continue
            x, y = label_tensors[key]
            soft = model(x, hard_top1=False)
            label_losses.append(F.smooth_l1_loss(soft, y, beta=0.25))
            sparse_losses.append(model.sparsity_loss())
            if ep > 120:
                hard = model(x, hard_top1=True)
                hard_losses.append(torch.mean((hard - soft.detach()) ** 2))
        label_loss = torch.stack(label_losses).mean()
        sparse_loss = torch.stack(sparse_losses).mean()
        hard_loss = torch.stack(hard_losses).mean() if hard_losses else torch.tensor(0.0)
        coupling_label_losses = []
        coupling_sparse_losses = []
        coupling_hard_losses = []
        for key, model in coupling_models.items():
            x, y, label_confidence = coupling_label_tensors[key]
            kind = key[2]
            soft_raw = model(x, hard_top1=False)
            hard_raw = model(x, hard_top1=True) if ep > 120 else None
            soft = 5.0 * torch.tanh(soft_raw / 5.0)
            hard = (
                5.0 * torch.tanh(hard_raw / 5.0)
                if hard_raw is not None
                else None
            )
            label_weight = args.drift_label_weight
            coupling_label_losses.append(
                label_weight
                * torch.sum(
                    label_confidence
                    * (1.0 + torch.clamp(torch.abs(y) / 0.08, max=3.0))
                    * F.smooth_l1_loss(soft, y, beta=0.08, reduction="none")
                )
                / torch.clamp(torch.sum(label_confidence), min=1.0e-9)
            )
            coupling_sparse_losses.append(model.sparsity_loss())
            if ep > 120:
                coupling_hard_losses.append(
                    label_weight
                    * (
                        0.75
                        * torch.sum(
                            label_confidence
                            * (1.0 + torch.clamp(torch.abs(y) / 0.08, max=3.0))
                            * F.smooth_l1_loss(
                                hard,
                                y,
                                beta=0.10,
                                reduction="none",
                            )
                        )
                        / torch.clamp(torch.sum(label_confidence), min=1.0e-9)
                        + 0.25 * torch.mean((hard - soft.detach()) ** 2)
                    )
                )
        coupling_label_loss = torch.stack(coupling_label_losses).mean()
        coupling_sparse_loss = torch.stack(coupling_sparse_losses).mean()
        coupling_hard_loss = (
            torch.stack(coupling_hard_losses).mean() if coupling_hard_losses else torch.tensor(0.0)
        )
        disp_penalty = torch.stack([
            torch.sigmoid(v["mix_raw"]) ** 2 + 0.02 * torch.sigmoid(v["rho_raw"]) ** 2
            for v in dispersive.values()
        ]).mean()
        loss = (
            wave_loss
            + 0.18 * label_loss
            + 0.010 * sparse_loss
            + 0.45 * hard_loss
            + 1.10 * hard_wave_loss
            + 1.8 * phys_loss
            + 2.5 * hard_phys_loss
            + 0.90 * trend_loss
            + 1.35 * hard_trend_loss
            + args.endpoint_weight * endpoint_loss
            + args.hard_endpoint_weight * hard_endpoint_loss
            + args.teacher_distill_weight * teacher_loss
            + args.hard_teacher_distill_weight * hard_teacher_loss
            + args.tail_distill_weight
            * (teacher_tail_loss + 1.25 * hard_teacher_tail_loss)
            + args.rate_distill_weight * rate_distill_loss
            + args.hard_rate_distill_weight * hard_rate_distill_loss
            + args.coupling_label_weight * coupling_label_loss
            + 0.012 * coupling_sparse_loss
            + args.coupling_hard_weight * coupling_hard_loss
            + 0.010 * disp_penalty
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 5.0)
        opt.step()
        if (
            ep >= 120
            and (
                ep % max(args.deployment_eval_period, 1) == 0
                or ep == args.epochs - 1
            )
        ):
            with torch.no_grad():
                current = deployment_metrics()
            if best_deployment is None or current["score"] < best_deployment["score"]:
                best_deployment = current
                best_state = snapshot_state()
                best_epoch = ep
            for group_key in group_keys:
                group_current = group_score(*group_key)
                if (
                    group_key not in best_group_metrics
                    or group_current["score"]
                    < best_group_metrics[group_key]["score"]
                ):
                    best_group_metrics[group_key] = group_current
                    best_group_states[group_key] = snapshot_group(*group_key)
                    best_group_epochs[group_key] = ep
        if ep % max(args.print_every, 1) == 0 or ep == args.epochs - 1:
            with torch.no_grad():
                hard_wave = torch.stack([
                    F.mse_loss(
                        _predict_segment(
                            models,
                            row,
                            hard_top1=True,
                            coupling_models=coupling_models,
                            dispersive=dispersive,
                        ),
                        row["y"],
                    )
                    for row in waves
                ]).mean().sqrt()
            history.append({
                "epoch": ep,
                "loss": float(loss.detach()),
                "wave_rmse_dec": float(wave_loss.detach()),
                "hard_wave_rmse_dec": float(hard_wave),
                "physics_monotonic_penalty": float(phys_loss.detach()),
                "hard_physics_monotonic_penalty": float(hard_phys_loss.detach()),
                "dynamic_trend_loss": float(trend_loss.detach()),
                "hard_dynamic_trend_loss": float(hard_trend_loss.detach()),
                "endpoint_drift_loss": float(endpoint_loss.detach()),
                "hard_endpoint_drift_loss": float(hard_endpoint_loss.detach()),
                "teacher_distill_loss": float(teacher_loss.detach()),
                "hard_teacher_distill_loss": float(hard_teacher_loss.detach()),
                "teacher_tail_loss": float(teacher_tail_loss.detach()),
                "hard_teacher_tail_loss": float(hard_teacher_tail_loss.detach()),
                "rate_distill_loss": float(rate_distill_loss.detach()),
                "hard_rate_distill_loss": float(hard_rate_distill_loss.detach()),
                "coupling_label_loss": float(coupling_label_loss.detach()),
                "hard_coupling_loss": float(coupling_hard_loss.detach()),
                "deployment_score": (
                    float(current["score"])
                    if "current" in locals()
                    and (
                        ep % max(args.deployment_eval_period, 1) == 0
                        or ep == args.epochs - 1
                    )
                    else None
                ),
            })

    if best_state is not None:
        restore_state(best_state)
    # The four family/process subsystems share no trainable parameters. Restore
    # each from its own best hard-top1 teacher checkpoint instead of forcing all
    # groups to use one global epoch.
    for group_key, state in best_group_states.items():
        restore_group(*group_key, state)
    if best_group_states:
        with torch.no_grad():
            best_deployment = deployment_metrics()

    # The direct Afast/Aslow parameterization already exposes both physical
    # couplings. The legacy total-drift-only polish is intentionally skipped.
    drift_models = [
        model for (family, process, kind), model in coupling_models.items()
        if kind == "drift"
    ]
    drift_params = [p for model in drift_models for p in model.parameters()]
    if args.drift_polish_epochs > 0 and drift_params:
        pre_polish_state = snapshot_state()
        pre_polish_metrics = deployment_metrics()
        drift_opt = torch.optim.AdamW(
            drift_params,
            lr=args.drift_polish_lr,
            weight_decay=1.0e-4,
        )
        for polish_ep in range(args.drift_polish_epochs):
            drift_opt.zero_grad()
            soft_wave = []
            hard_wave = []
            soft_endpoint = []
            hard_endpoint = []
            polish_weights = []
            drift_label_terms = []
            for row in waves:
                pred = _predict_segment(
                    models, row, hard_top1=False,
                    coupling_models=coupling_models, dispersive=dispersive,
                )
                pred_hard = _predict_segment(
                    models, row, hard_top1=True,
                    coupling_models=coupling_models, dispersive=dispersive,
                )
                soft_wave.append(F.smooth_l1_loss(pred, row["y"], beta=0.025))
                hard_wave.append(F.smooth_l1_loss(pred_hard, row["y"], beta=0.030))
                soft_endpoint.append(_endpoint_drift_loss(pred, row["y"]))
                hard_endpoint.append(_endpoint_drift_loss(pred_hard, row["y"]))
                polish_weights.append(_waveform_importance(row))
            for key, model in coupling_models.items():
                if key[2] != "drift":
                    continue
                x, y = coupling_label_tensors[key]
                soft = 5.0 * torch.tanh(model(x, hard_top1=False) / 5.0)
                hard = 5.0 * torch.tanh(model(x, hard_top1=True) / 5.0)
                sample_weight = 1.0 + torch.clamp(torch.abs(y) / 0.08, max=3.0)
                drift_label_terms.append(
                    torch.mean(
                        sample_weight
                        * F.smooth_l1_loss(soft, y, beta=0.06, reduction="none")
                    )
                    + 0.75
                    * torch.mean(
                        sample_weight
                        * F.smooth_l1_loss(hard, y, beta=0.08, reduction="none")
                    )
                )
            polish_loss = (
                0.25 * _weighted_loss_mean(soft_wave, polish_weights)
                + 0.35 * _weighted_loss_mean(hard_wave, polish_weights)
                + 2.2 * _weighted_loss_mean(soft_endpoint, polish_weights)
                + 3.2 * _weighted_loss_mean(hard_endpoint, polish_weights)
                + 0.45 * torch.stack(drift_label_terms).mean()
            )
            polish_loss.backward()
            torch.nn.utils.clip_grad_norm_(drift_params, 3.0)
            drift_opt.step()
            if polish_ep == args.drift_polish_epochs - 1:
                history.append({
                    "epoch": args.epochs + polish_ep,
                    "phase": "drift_polish",
                    "loss": float(polish_loss.detach()),
                })
        post_polish_metrics = deployment_metrics()
        required_score = (
            pre_polish_metrics["score"]
            - float(args.drift_polish_min_score_improvement)
        )
        polish_accepted = post_polish_metrics["score"] <= required_score
        if not polish_accepted:
            restore_state(pre_polish_state)
            post_polish_metrics = pre_polish_metrics
        history.append(
            {
                "epoch": args.epochs + args.drift_polish_epochs,
                "phase": "drift_polish_acceptance",
                "accepted": polish_accepted,
                "pre_score": pre_polish_metrics["score"],
                "post_score": post_polish_metrics["score"],
            }
        )
        print(
            "Drift-only polish "
            f"{'accepted' if polish_accepted else 'rejected and restored'}: "
            f"score {pre_polish_metrics['score']:.6g} -> "
            f"{post_polish_metrics['score']:.6g}"
        )

    rows = []
    pred_rows = []
    with torch.no_grad():
        for row in waves:
            y_refined = _predict_segment(
                models, row, hard_top1=False, coupling_models=coupling_models, dispersive=dispersive
            ).detach().cpu().numpy()
            y_top1 = _predict_segment(
                models, row, hard_top1=True, coupling_models=coupling_models, dispersive=dispersive
            ).detach().cpu().numpy()
            y = row["y"].detach().cpu().numpy()
            yt = row["y_teacher"].detach().cpu().numpy()
            rmse_teacher = float(np.sqrt(np.mean((yt - y) ** 2)))
            rmse_refined = float(np.sqrt(np.mean((y_refined - y) ** 2)))
            rmse_top1 = float(np.sqrt(np.mean((y_top1 - y) ** 2)))
            refined_dynamic = _dynamic_metrics_np(y_refined, y)
            top1_dynamic = _dynamic_metrics_np(y_top1, y)
            sign = np.sign(y[-1] - y[0]) if abs(float(y[-1] - y[0])) >= 0.003 else 0.0
            if (
                sign == 0.0
                or len(y_top1) < 2
                or bool(row.get("allow_competing_branches", False))
            ):
                wrong_slope_frac = 0.0
            else:
                wrong_slope_frac = float(np.mean(sign * np.diff(y_top1) < -0.0015))
            rows.append(
                {
                    "curve_id": row["curve_id"],
                    "seg_id": row["seg_id"],
                    "family": row["family"],
                    "process": row["process"],
                    "temp_K": row["temp_K"],
                    "source_file": row["source_file"],
                    "teacher_rmse_dec": rmse_teacher,
                    "refined_rmse_dec": rmse_refined,
                    "top1_rmse_dec": rmse_top1,
                    "refined_improvement_dec": rmse_teacher - rmse_refined,
                    "top1_improvement_dec": rmse_teacher - rmse_top1,
                    "top1_wrong_slope_frac": wrong_slope_frac,
                    "refined_dynamic_shape_rmse_norm": refined_dynamic["dynamic_shape_rmse_norm"],
                    "top1_dynamic_shape_rmse_norm": top1_dynamic["dynamic_shape_rmse_norm"],
                    "measured_drift_dec": top1_dynamic["measured_drift_dec"],
                    "refined_drift_dec": refined_dynamic["predicted_drift_dec"],
                    "top1_drift_dec": top1_dynamic["predicted_drift_dec"],
                    "refined_endpoint_drift_abs_error_dec": refined_dynamic["endpoint_drift_abs_error_dec"],
                    "top1_endpoint_drift_abs_error_dec": top1_dynamic["endpoint_drift_abs_error_dec"],
                    "allow_competing_branches": row["allow_competing_branches"],
                    "measured_reversal_dec": row["measured_reversal_dec"],
                }
            )
            for t, yy, yr, yh in zip(row["t"].numpy(), y, y_refined, y_top1):
                pred_rows.append(
                    {
                        "curve_id": row["curve_id"],
                        "seg_id": row["seg_id"],
                        "time_rel_s": float(t),
                        "log_id_meas": float(yy),
                        "log_id_refined": float(yr),
                        "log_id_top1": float(yh),
                    }
                )

    metrics = pd.DataFrame(rows)
    preds = pd.DataFrame(pred_rows)
    # Backward-compatible column names are retained for older plotting tools;
    # explicit Top-2 aliases identify the actual release configuration.
    for old, new in [
        ("top1_rmse_dec", "top2_rmse_dec"),
        ("top1_improvement_dec", "top2_improvement_dec"),
        ("top1_wrong_slope_frac", "top2_wrong_slope_frac"),
        ("top1_dynamic_shape_rmse_norm", "top2_dynamic_shape_rmse_norm"),
        ("top1_drift_dec", "top2_drift_dec"),
        (
            "top1_endpoint_drift_abs_error_dec",
            "top2_endpoint_drift_abs_error_dec",
        ),
    ]:
        metrics[new] = metrics[old]
    preds["log_id_top2"] = preds["log_id_top1"]
    formulas = {
        "|".join(key): _top1_formula(model)
        for key, model in models.items()
    }
    coupling_formulas = {
        "|".join(key): _top1_formula(model)
        for key, model in coupling_models.items()
    }
    symbolic_formula_terms = 0
    for formula in [*formulas.values(), *coupling_formulas.values()]:
        if str(formula.get("architecture", "")).startswith("two_layer_top"):
            for node in formula["hidden_nodes"]:
                symbolic_formula_terms += len(node["edges"])
                symbolic_formula_terms += len(node["output_components"])
        else:
            symbolic_formula_terms += len(formula.get("edges", []))
    summary = {
        "n_segments": int(len(metrics)),
        "teacher_rmse_dec_mean": float(metrics["teacher_rmse_dec"].mean()),
        "refined_rmse_dec_mean": float(metrics["refined_rmse_dec"].mean()),
        "top1_rmse_dec_mean": float(metrics["top1_rmse_dec"].mean()),
        "top2_rmse_dec_mean": float(metrics["top2_rmse_dec"].mean()),
        "mean_refined_improvement_dec": float(metrics["refined_improvement_dec"].mean()),
        "mean_top1_improvement_dec": float(metrics["top1_improvement_dec"].mean()),
        "mean_top1_wrong_slope_frac": float(metrics["top1_wrong_slope_frac"].mean()),
        "max_top1_wrong_slope_frac": float(metrics["top1_wrong_slope_frac"].max()),
        "mean_top1_dynamic_shape_rmse_norm": float(metrics["top1_dynamic_shape_rmse_norm"].mean()),
        "mean_top1_endpoint_drift_abs_error_dec": float(metrics["top1_endpoint_drift_abs_error_dec"].mean()),
        "norm": norm,
        "history": history,
        "formulas": formulas,
        "coupling_formulas": coupling_formulas,
        "per_segment_nuisance_calibration": False,
        "warm_started_from_history_top1": bool(warm_started),
        "coupling_target_definition": {
            "amp_fast": "teacher fast-trap log-current coupling coefficient",
            "amp_slow": "teacher slow-trap log-current coupling coefficient",
            "total_drift": "derived exactly as -(Afast+Aslow), not independently fitted",
        },
        "coupling_loss_weights": {
            "drift_label": args.drift_label_weight,
            "fraction_label": args.fraction_label_weight,
            "competition_label": args.competition_label_weight,
            "coupling_label_total": args.coupling_label_weight,
            "coupling_hard_total": args.coupling_hard_weight,
            "endpoint": args.endpoint_weight,
            "hard_endpoint": args.hard_endpoint_weight,
            "teacher_distill": args.teacher_distill_weight,
            "hard_teacher_distill": args.hard_teacher_distill_weight,
            "tail_distill": args.tail_distill_weight,
            "tail_distill_fraction": args.tail_distill_fraction,
            "rate_distill": args.rate_distill_weight,
            "hard_rate_distill": args.hard_rate_distill_weight,
            "unresolved_rate_confidence": args.unresolved_rate_confidence,
            "rate_pretrain_epochs": args.rate_pretrain_epochs,
            "rate_pretrain_lr": args.rate_pretrain_lr,
            "drift_polish_epochs": args.drift_polish_epochs,
            "drift_polish_lr": args.drift_polish_lr,
            "drift_polish_min_score_improvement": args.drift_polish_min_score_improvement,
            "drift_lr_mult": args.drift_lr_mult,
            "fraction_lr_mult": args.fraction_lr_mult,
            "competition_lr_mult": args.competition_lr_mult,
        },
        "deployment_checkpoint": {
            "selected_epoch": best_epoch,
            "selected_group_epochs": {
                "|".join(key): int(value)
                for key, value in best_group_epochs.items()
            },
            "selected_group_metrics": {
                "|".join(key): value
                for key, value in best_group_metrics.items()
            },
            "selection_metric": (
                "mean_top1_rmse + endpoint_weight*mean_endpoint_error + "
                "shape_weight*mean_normalized_shape_rmse + "
                "median_weight*median_top1_rmse"
            ),
            "weights": {
                "endpoint": args.deployment_endpoint_weight,
                "shape": args.deployment_shape_weight,
                "median": args.deployment_median_weight,
                "p90": args.deployment_p90_weight,
                "max": args.deployment_max_weight,
                "teacher": args.deployment_teacher_weight,
            },
            "metrics": best_deployment,
        },
        "dispersive_slow_trap": {
            key: {
                "mix": float((0.55 * torch.sigmoid(v["mix_raw"])).detach().cpu()),
                "rho": float((2.0 + 48.0 * torch.sigmoid(v["rho_raw"])).detach().cpu()),
                "interpretation": "slow trap is implemented as two RC sub-branches: (1-mix)*tau_slow and mix*rho*tau_slow",
            }
            for key, v in dispersive.items()
        },
        "symbolic_topk": 2,
        "symbolic_formula_terms": int(symbolic_formula_terms),
        "model_form": "logI(t)=logI(0)+Afast_KAN*(exp(-t/tau_fast_KAN)-1)+Aslow_KAN*(exp(-t/tau_slow_KAN)-1); the deployable student uses the same two SRH states as the teacher, and every KAN edge is exported as an explicit Top-2 analytic basis mixture.",
    }
    metrics.to_csv(OUT / "srh_waveform_refinement_metrics.csv", index=False, encoding="utf-8-sig")
    preds.to_csv(OUT / "srh_waveform_refined_predictions.csv", index=False, encoding="utf-8-sig")
    (OUT / "srh_waveform_refined_top1_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (OUT / "srh_waveform_refined_top2_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    torch.save(
        {
            "rate_models": {
                "|".join(key): model.state_dict()
                for key, model in models.items()
            },
            "coupling_models": {
                "|".join(key): model.state_dict()
                for key, model in coupling_models.items()
            },
            "normalization": norm,
            "symbolic_topk": 2,
        },
        OUT / "srh_waveform_top2_checkpoint.pt",
    )

    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    axes = axes.ravel()
    for ax, (cid, g) in zip(axes, preds.groupby("curve_id")):
        for seg, s in g.groupby("seg_id"):
            ax.plot(s["time_rel_s"], s["log_id_meas"], ".", ms=2, alpha=0.35)
            ax.plot(s["time_rel_s"], s["log_id_top1"], "-", lw=1.0)
        ax.set_title(cid)
        ax.set_xlabel("t_rel (s)")
        ax.set_ylabel("log10 ID")
    fig.tight_layout()
    fig.savefig(OUT / "srh_waveform_refined_top1_examples.png", dpi=220)
    fig.savefig(OUT / "srh_waveform_refined_top2_examples.png", dpi=220)
    plt.close(fig)

    # VA include note: the full state implementation remains in the SRH exporter;
    # this file lists the refined Top-2 analytic formulas for integration.
    lines = [
        "// Top-2 waveform-refined SRH rate-and-coupling KAN formulas.",
        "// Rate formulas replace log_tau_c/log_tau_e laws.",
        "// Coupling formulas directly predict Afast and Aslow; total drift is -(Afast+Aslow).",
        "// The deployable model contains exactly two SRH state branches.",
        "// Features must be normalized with srh_waveform_refined_top2_summary.json::norm.",
    ]
    lines.append("// ---- SRH rate formulas ----")
    for key, formula in formulas.items():
        _append_formula_comments(lines, f"rate:{key}", formula)
    lines.append("// ---- trap-to-current coupling formulas ----")
    for key, formula in coupling_formulas.items():
        _append_formula_comments(lines, f"coupling:{key}", formula)
    (OUT / "dg_tft_dyntrap_waveform_refined_top1_include.va").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "dg_tft_dyntrap_waveform_refined_top2_include.va").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({k: summary[k] for k in summary if k.endswith("_mean") or k.startswith("mean_") or k == "n_segments"}, indent=2))


if __name__ == "__main__":
    main()
