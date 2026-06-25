from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json
import math
import os
import re
import subprocess

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import least_squares


ROOT = Path(__file__).resolve().parent
RESULTS = Path(os.environ.get("DG_TFT_DYNAMIC_WORK_DIR", ROOT / "results_long_timescale"))
OUT = RESULTS / "srh_history_kan"
HSPICE = Path("hspice.com")

FEATURE_NAMES = [
    "interface_top",
    "process_capture",
    "branch_slow",
    "temperature_dataset",
    "to_v_norm",
    "from_v_norm",
    "delta_v_norm",
    "temp_norm",
    "inv_temp_norm",
    "cycle_norm",
    "log_hold_s",
    "prev_log_id",
    "seg_log_id0",
    "vds_norm",
]

BASIS_NAMES = ["linear", "tanh", "sigmoid", "softplus_decay", "invquad", "gaussian"]


def safe_sigmoid_np(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -80.0, 80.0)))


def basis_torch(name: str, z: torch.Tensor) -> torch.Tensor:
    if name == "linear":
        return z
    if name == "tanh":
        return torch.tanh(z)
    if name == "sigmoid":
        return torch.sigmoid(torch.clamp(z, -60.0, 60.0)) - 0.5
    if name == "softplus_decay":
        return torch.exp(-F.softplus(torch.clamp(z, -60.0, 60.0)))
    if name == "invquad":
        return 1.0 / (1.0 + z * z)
    if name == "gaussian":
        return torch.exp(-torch.clamp(z * z, max=60.0))
    raise ValueError(name)


def basis_np(name: str, z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    if name == "linear":
        return z
    if name == "tanh":
        return np.tanh(z)
    if name == "sigmoid":
        return safe_sigmoid_np(z) - 0.5
    if name == "softplus_decay":
        return np.exp(-np.log1p(np.exp(np.clip(z, -60.0, 60.0))))
    if name == "invquad":
        return 1.0 / (1.0 + z * z)
    if name == "gaussian":
        return np.exp(-np.clip(z * z, 0.0, 60.0))
    raise ValueError(name)


class AdditiveKAN(nn.Module):
    """Small feature-wise KAN/GAM for log10(tau).

    Every edge is univariate and symbolic-library constrained during training
    through a soft candidate gate.  At export each feature keeps only its top-1
    basis, which mirrors the deployment requirement.
    """

    def __init__(self, n_features: int, n_basis: int):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(()))
        self.amp = nn.Parameter(torch.zeros(n_features))
        self.slope_raw = nn.Parameter(torch.zeros(n_features))
        self.shift = nn.Parameter(torch.zeros(n_features))
        self.logits = nn.Parameter(torch.zeros(n_features, n_basis))

    def forward(self, x: torch.Tensor, hard_top1: bool = False) -> torch.Tensor:
        slopes = 0.05 + F.softplus(self.slope_raw)
        z = x * slopes.view(1, -1) + self.shift.view(1, -1)
        vals = torch.stack([basis_torch(name, z) for name in BASIS_NAMES], dim=-1)
        prob = torch.softmax(self.logits, dim=-1)
        if hard_top1:
            idx = torch.argmax(self.logits, dim=-1)
            hard_gate = F.one_hot(idx, num_classes=len(BASIS_NAMES)).to(vals.dtype)
            # Training-time top-1 symbolic projection.  Forward evaluation is
            # already the exact deployable top-1 function, while the
            # straight-through term gives the function-selection logits a
            # useful gradient.  Evaluation/export remains an exact one-hot.
            gate = hard_gate + prob - prob.detach() if self.training else hard_gate
        else:
            gate = prob
        phi = torch.sum(vals * gate.view(1, *gate.shape), dim=-1)
        return self.bias + torch.sum(self.amp.view(1, -1) * phi, dim=-1)

    def sparsity_loss(self) -> torch.Tensor:
        p = torch.softmax(self.logits, dim=-1)
        entropy = -torch.sum(p * torch.log(torch.clamp(p, min=1.0e-9)), dim=-1)
        return torch.mean(torch.abs(self.amp)) + 0.025 * torch.mean(entropy)


@dataclass
class EdgeFormula:
    feature: str
    basis: str
    amp: float
    slope: float
    shift: float


@dataclass
class TauFormula:
    family: str
    branch: str
    process: str
    bias: float
    edges: list[EdgeFormula]
    train_rmse_dec: float
    hard_top1_rmse_dec: float
    n: int

    def eval(self, x: np.ndarray) -> np.ndarray:
        y = np.full(x.shape[0], self.bias, dtype=float)
        for edge in self.edges:
            j = FEATURE_NAMES.index(edge.feature)
            y += edge.amp * basis_np(edge.basis, edge.slope * x[:, j] + edge.shift)
        return y


def _robust_scale(v: np.ndarray, floor: float = 0.25) -> tuple[float, float]:
    mu = float(np.nanmedian(v))
    q25, q75 = np.nanpercentile(v, [25, 75])
    sig = float(max((q75 - q25) / 1.349, floor))
    return mu, sig


def build_label_table() -> tuple[pd.DataFrame, dict[str, list[float]]]:
    step = pd.read_csv(RESULTS / "it_step_response_fit.csv")
    pred = pd.read_csv(RESULTS / "it_step_response_predictions.csv")
    step = step[(step["ok"] == True) & (step["large_amp_warning"] == False) & (step["log_rmse"] < 0.06)].copy()
    rows = []
    for _, r in step.iterrows():
        curve_pred = pred[(pred["curve_id"] == r["curve_id"]) & (pred["seg_id"] == r["step_seg_id"])]
        if curve_pred.empty:
            continue
        prev_seg = int(r["step_seg_id"]) - 1
        prev = pred[(pred["curve_id"] == r["curve_id"]) & (pred["seg_id"] == prev_seg)]
        prev_log = float(prev["log_id_smooth"].median()) if len(prev) else float(curve_pred["log_id_smooth"].iloc[0])
        seg_log0 = float(curve_pred["log_id_smooth"].iloc[0])
        cycle = int(max(int(r["step_seg_id"]) // 2, 0))
        family = str(r["family"])
        for branch, tau_col, unresolved_col in [
            ("fast", "tau_fast_s", "fast_tau_unresolved"),
            ("slow", "tau_slow_s", "slow_tau_unresolved"),
        ]:
            if bool(r.get(unresolved_col, False)):
                continue
            # Respect identifiability: fast labels come from short traces,
            # slow labels from long traces.  Otherwise KAN learns sampling
            # artifacts instead of trap physics.
            if branch == "fast" and float(r["t_span_s"]) > 0.05:
                continue
            if branch == "slow" and float(r["t_span_s"]) < 1.0:
                continue
            if branch == "slow":
                sep = float(r["tau_slow_s"]) / max(float(r["tau_fast_s"]), 1.0e-12)
                if sep < 2.5 or abs(float(r.get("amp_slow", 0.0))) < 0.018:
                    continue
            rows.append(
                {
                    "family": family,
                    "branch": branch,
                    "process": "capture" if str(r["direction"]) == "up" else "emission",
                    "log_tau": float(np.log10(float(r[tau_col]))),
                    "tau_s": float(r[tau_col]),
                    "interface_top": 1.0 if family == "VTG" else 0.0,
                    "process_capture": 1.0 if str(r["direction"]) == "up" else 0.0,
                    "branch_slow": 1.0 if branch == "slow" else 0.0,
                    "temperature_dataset": 1.0 if bool(r.get("temperature_dataset", "temperature" in str(r["curve_id"]).lower())) else 0.0,
                    "to_v": float(r["to_sweep_V"]),
                    "from_v": float(r["from_sweep_V"]),
                    "delta_v": float(r["delta_sweep_V"]),
                    "temp_K": float(r.get("temp_K", 293.0)),
                    "cycle": float(cycle),
                    "hold_s": float(r["t_span_s"]),
                    "prev_log_id": prev_log,
                    "seg_log_id0": seg_log0,
                    "vds": float(r["vds_V"]),
                    "source_file": str(r.get("source_file", "")),
                    "curve_id": r["curve_id"],
                    "step_seg_id": int(r["step_seg_id"]),
                    "fit_log_rmse": float(r["log_rmse"]),
                    "weight": float(1.0 / (0.015 + float(r["log_rmse"]))),
                }
            )
    label = pd.DataFrame(rows)
    if label.empty:
        raise RuntimeError("No identifiable SRH tau labels after filtering.")
    raw_cols = {
        "to_v_norm": label["to_v"].to_numpy(float),
        "from_v_norm": label["from_v"].to_numpy(float),
        "delta_v_norm": label["delta_v"].to_numpy(float),
        "temp_norm": label["temp_K"].to_numpy(float),
        "inv_temp_norm": 1000.0 / np.clip(label["temp_K"].to_numpy(float), 1.0, None),
        "cycle_norm": label["cycle"].to_numpy(float),
        "log_hold_s": np.log10(np.clip(label["hold_s"].to_numpy(float), 1.0e-8, None)),
        "prev_log_id": label["prev_log_id"].to_numpy(float),
        "seg_log_id0": label["seg_log_id0"].to_numpy(float),
        "vds_norm": label["vds"].to_numpy(float),
    }
    norm = {}
    for col, vals in raw_cols.items():
        mu, sig = _robust_scale(vals)
        norm[col] = [mu, sig]
        label[col] = (vals - mu) / sig
    label["weight"] = label["weight"] / np.nanmedian(label["weight"])
    return label, norm


def matrix_for(g: pd.DataFrame) -> np.ndarray:
    return g[FEATURE_NAMES].to_numpy(float)


def weighted_huber(pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor, beta: float) -> torch.Tensor:
    vec = F.smooth_l1_loss(pred, target, beta=beta, reduction="none")
    return torch.sum(vec * weight) / torch.clamp(torch.sum(weight), min=1.0e-9)


def refit_top1_formula(formula: TauFormula, g: pd.DataFrame) -> TauFormula:
    """Optimize the exported top-1 formula itself, not the soft KAN surrogate."""
    x = matrix_for(g)
    y = g["log_tau"].to_numpy(float)
    w = g.get("weight", pd.Series(np.ones(len(g)))).to_numpy(float)
    w = w / max(float(np.nanmedian(w)), 1.0e-9)
    if not formula.edges:
        formula.bias = float(np.average(y, weights=w))
        formula.train_rmse_dec = float(np.sqrt(np.mean((formula.eval(x) - y) ** 2)))
        formula.hard_top1_rmse_dec = formula.train_rmse_dec
        return formula

    # Small label groups cannot identify many univariate edges.  Keep the most
    # useful top-1 edges so the VA formula remains stable and extrapolatable.
    max_edges = max(1, min(5, len(g) // 2 + 1))
    edges = sorted(formula.edges, key=lambda e: abs(e.amp), reverse=True)[:max_edges]
    p0 = [formula.bias]
    lb = [-12.0]
    ub = [6.0]
    for e in edges:
        p0 += [e.amp, math.log(max(e.slope, 0.02)), e.shift]
        lb += [-6.0, math.log(0.02), -8.0]
        ub += [6.0, math.log(8.0), 8.0]

    def pred_from(p: np.ndarray) -> np.ndarray:
        yy = np.full(x.shape[0], p[0], dtype=float)
        k = 1
        for e in edges:
            amp = p[k]
            slope = math.exp(p[k + 1])
            shift = p[k + 2]
            yy += amp * basis_np(e.basis, slope * x[:, FEATURE_NAMES.index(e.feature)] + shift)
            k += 3
        return yy

    def resid(p: np.ndarray) -> np.ndarray:
        yy = pred_from(p)
        data_res = np.sqrt(w) * (yy - y)
        reg = [0.025 * p[0]]
        k = 1
        for _ in edges:
            reg += [0.030 * p[k], 0.010 * p[k + 1], 0.006 * p[k + 2]]
            k += 3
        return np.concatenate([data_res, np.asarray(reg, dtype=float)])

    sol = least_squares(resid, np.asarray(p0, dtype=float), bounds=(np.asarray(lb), np.asarray(ub)), max_nfev=8000)
    p = sol.x
    formula.bias = float(p[0])
    new_edges = []
    k = 1
    for e in edges:
        amp = float(p[k])
        slope = float(math.exp(p[k + 1]))
        shift = float(p[k + 2])
        if abs(amp) >= 5.0e-3:
            new_edges.append(EdgeFormula(e.feature, e.basis, amp, slope, shift))
        k += 3
    formula.edges = new_edges
    pred = formula.eval(x)
    formula.train_rmse_dec = float(np.sqrt(np.mean((pred - y) ** 2)))
    formula.hard_top1_rmse_dec = formula.train_rmse_dec
    return formula


def fit_one_group(g: pd.DataFrame, seed: int = 7) -> tuple[TauFormula, dict]:
    torch.manual_seed(seed)
    x = torch.as_tensor(matrix_for(g), dtype=torch.float32)
    y = torch.as_tensor(g["log_tau"].to_numpy(float), dtype=torch.float32)
    w = torch.as_tensor(g.get("weight", pd.Series(np.ones(len(g)))).to_numpy(float), dtype=torch.float32)
    w = w / torch.clamp(torch.median(w), min=1.0e-9)
    model = AdditiveKAN(x.shape[1], len(BASIS_NAMES))
    with torch.no_grad():
        model.bias.copy_(torch.median(y))
    opt = torch.optim.AdamW(model.parameters(), lr=3.0e-3, weight_decay=2.0e-4)
    best = None
    best_state = None
    for ep in range(1600):
        opt.zero_grad()
        pred = model(x, hard_top1=False)
        loss = weighted_huber(pred, y, w, beta=0.25)
        loss = loss + 0.020 * model.sparsity_loss()
        # A small hard-top1 distillation term makes the final symbolic export
        # faithful instead of being a post-hoc approximation.
        if ep > 500:
            hard = model(x, hard_top1=True)
            loss = loss + 0.30 * torch.mean((hard - pred.detach()) ** 2) + 0.10 * weighted_huber(hard, y, w, beta=0.30)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        val = float(torch.sqrt(torch.mean((model(x, hard_top1=False).detach() - y) ** 2)))
        if best is None or val < best:
            best = val
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    with torch.no_grad():
        pred_soft = model(x, hard_top1=False).numpy()
        pred_hard = model(x, hard_top1=True).numpy()
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
        edges.append(EdgeFormula(name, BASIS_NAMES[int(top[j])], float(amps[j]), float(slopes[j]), float(shifts[j])))
    formula = TauFormula(
        str(g["family"].iloc[0]),
        str(g["branch"].iloc[0]),
        str(g["process"].iloc[0]),
        bias,
        edges,
        float(np.sqrt(np.mean((pred_soft - g["log_tau"].to_numpy(float)) ** 2))),
        float(np.sqrt(np.mean((pred_hard - g["log_tau"].to_numpy(float)) ** 2))),
        int(len(g)),
    )
    formula = refit_top1_formula(formula, g)
    pred_pruned = formula.eval(matrix_for(g))
    formula.hard_top1_rmse_dec = float(np.sqrt(np.mean((pred_pruned - g["log_tau"].to_numpy(float)) ** 2)))
    diag_cols = ["curve_id", "step_seg_id", "tau_s", "log_tau", "temp_K", "source_file", "weight"]
    diag = g[[c for c in diag_cols if c in g.columns]].copy()
    diag["pred_soft_log_tau"] = pred_soft
    diag["pred_top1_log_tau"] = pred_pruned
    diag["pred_top1_tau_s"] = np.power(10.0, pred_pruned)
    return formula, {"diagnostic": diag}


def fit_all(label: pd.DataFrame) -> tuple[list[TauFormula], pd.DataFrame]:
    formulas = []
    diags = []
    for key, g in label.groupby(["family", "branch", "process"], sort=True):
        if len(g) < 2:
            # A single label should become a constant top-1 formula.
            y = float(g["log_tau"].iloc[0])
            formula = TauFormula(key[0], key[1], key[2], y, [], 0.0, 0.0, int(len(g)))
            diag_cols = ["curve_id", "step_seg_id", "tau_s", "log_tau", "temp_K", "source_file", "weight"]
            diag = g[[c for c in diag_cols if c in g.columns]].copy()
            diag["pred_soft_log_tau"] = y
            diag["pred_top1_log_tau"] = y
            diag["pred_top1_tau_s"] = 10.0**y
        else:
            formula, pack = fit_one_group(g)
            diag = pack["diagnostic"]
        formulas.append(formula)
        diag["family"] = key[0]
        diag["branch"] = key[1]
        diag["process"] = key[2]
        diags.append(diag)
    return formulas, pd.concat(diags, ignore_index=True)


def va_feature_assignments(norm: dict[str, list[float]]) -> str:
    def n(var: str, name: str) -> str:
        mu, sig = norm[name]
        return f"(({var}-({mu:.12e}))/({sig:.12e}))"

    return f"""
    // Default deployment uses bias/local state but no explicit measured cycle history.
    // $temperature is treated as Kelvin in HSPICE/Verilog-A deployments.
    x_interface_top_t = 1.0; x_interface_top_b = 0.0;
    x_temperature_dataset = 0.0;
    x_to_v_t = {n("vt", "to_v_norm")}; x_to_v_b = {n("vb", "to_v_norm")};
    x_from_v_t = x_to_v_t; x_from_v_b = x_to_v_b;
    x_delta_v = 0.0;
    x_temp = {n("$temperature", "temp_norm")};
    x_inv_temp = {n("(1000.0/$temperature)", "inv_temp_norm")};
    x_cycle = 0.0;
    x_hold = {n("1.0", "log_hold_s")};
    x_prev_log_id = 0.0;
    x_seg_log_id0 = 0.0;
    x_vds = {n("vds", "vds_norm")};
"""


def edge_expr(edge: EdgeFormula, feature_var: str) -> str:
    z = f"(({edge.slope:.12e})*{feature_var}+({edge.shift:.12e}))"
    if edge.basis == "linear":
        b = z
    elif edge.basis == "tanh":
        b = f"tanh({z})"
    elif edge.basis == "sigmoid":
        b = f"(safe_sigmoid({z})-0.5)"
    elif edge.basis == "softplus_decay":
        b = f"exp(-safe_softplus({z}))"
    elif edge.basis == "invquad":
        b = f"(1.0/(1.0+{z}*{z}))"
    elif edge.basis == "gaussian":
        b = f"exp(-sp_clip({z}*{z},0.0,60.0))"
    else:
        raise ValueError(edge.basis)
    return f"({edge.amp:.12e})*{b}"


def formula_expr(f: TauFormula, prefix: str, process_capture: bool, branch_slow: bool) -> str:
    feature_map = {
        "interface_top": f"x_interface_top_{prefix}",
        "process_capture": "1.0" if process_capture else "0.0",
        "branch_slow": "1.0" if branch_slow else "0.0",
        "temperature_dataset": "x_temperature_dataset",
        "to_v_norm": f"x_to_v_{prefix}",
        "from_v_norm": f"x_from_v_{prefix}",
        "delta_v_norm": "x_delta_v",
        "temp_norm": "x_temp",
        "inv_temp_norm": "x_inv_temp",
        "cycle_norm": "x_cycle",
        "log_hold_s": "x_hold",
        "prev_log_id": "x_prev_log_id",
        "seg_log_id0": "x_seg_log_id0",
        "vds_norm": "x_vds",
    }
    terms = [f"({f.bias:.12e})"]
    terms += [edge_expr(edge, feature_map[edge.feature]) for edge in f.edges]
    return "(" + "+".join(terms) + ")"


def fdict(formulas: list[TauFormula]) -> dict[tuple[str, str, str], TauFormula]:
    return {(f.family, f.branch, f.process): f for f in formulas}


def write_va(formulas: list[TauFormula], norm: dict[str, list[float]], path: Path) -> None:
    f = fdict(formulas)
    text = f"""`include "disciplines.vams"
`include "constants.vams"

// History-aware SRH dynamic trap KAN, top-1 symbolic edge export.
// Python training used cycle/history features; standalone SPICE deployment
// keeps the same symbolic formulas and sets unavailable measurement-history
// terms to neutral defaults.
module dg_tft_dyntrap_history_kan(tg, bg, d, s, nft, nst, nfb, nsb);
  inout tg, bg, d, s, nft, nst, nfb, nsb;
  electrical tg, bg, d, s, nft, nst, nfb, nsb;
  parameter real tau_min = 1.0e-8;
  parameter real tau_max = 1.0e3;

  analog function real safe_softplus;
    input x; real x;
    begin
      if (x > 60.0) safe_softplus = x;
      else if (x < -60.0) safe_softplus = exp(x);
      else safe_softplus = log(1.0+exp(x));
    end
  endfunction

  analog function real safe_sigmoid;
    input x; real x;
    begin
      if (x > 60.0) safe_sigmoid = 1.0;
      else if (x < -60.0) safe_sigmoid = 0.0;
      else safe_sigmoid = 1.0/(1.0+exp(-x));
    end
  endfunction

  analog function real sp_clip;
    input x, lo, hi; real x, lo, hi;
    begin
      if (x < lo) sp_clip = lo;
      else if (x > hi) sp_clip = hi;
      else sp_clip = x;
    end
  endfunction

  real vt, vb, vds;
  real x_interface_top_t, x_interface_top_b, x_temperature_dataset, x_to_v_t, x_to_v_b, x_from_v_t, x_from_v_b, x_delta_v, x_temp, x_inv_temp, x_cycle, x_hold, x_prev_log_id, x_seg_log_id0, x_vds;
  real tc_tf, te_tf, tc_ts, te_ts, tc_bf, te_bf, tc_bs, te_bs;
  analog begin
    vt = V(tg,s); vb = V(bg,s); vds = V(d,s);
{va_feature_assignments(norm)}
    tc_tf = sp_clip(pow(10.0, {formula_expr(f[("VTG", "fast", "capture")], "t", True, False)}), tau_min, tau_max);
    te_tf = sp_clip(pow(10.0, {formula_expr(f[("VTG", "fast", "emission")], "t", False, False)}), tau_min, tau_max);
    tc_ts = sp_clip(pow(10.0, {formula_expr(f[("VTG", "slow", "capture")], "t", True, True)}), tau_min, tau_max);
    te_ts = sp_clip(pow(10.0, {formula_expr(f[("VTG", "slow", "emission")], "t", False, True)}), tau_min, tau_max);
    tc_bf = sp_clip(pow(10.0, {formula_expr(f[("VBG", "fast", "capture")], "b", True, False)}), tau_min, tau_max);
    te_bf = sp_clip(pow(10.0, {formula_expr(f[("VBG", "fast", "emission")], "b", False, False)}), tau_min, tau_max);
    tc_bs = sp_clip(pow(10.0, {formula_expr(f[("VBG", "slow", "capture")], "b", True, True)}), tau_min, tau_max);
    te_bs = sp_clip(pow(10.0, {formula_expr(f[("VBG", "slow", "emission")], "b", False, True)}), tau_min, tau_max);

    I(nft,s) <+ ddt(V(nft,s)) - (1.0-V(nft,s))/tc_tf + V(nft,s)/te_tf;
    I(nst,s) <+ ddt(V(nst,s)) - (1.0-V(nst,s))/tc_ts + V(nst,s)/te_ts;
    I(nfb,s) <+ ddt(V(nfb,s)) - (1.0-V(nfb,s))/tc_bf + V(nfb,s)/te_bf;
    I(nsb,s) <+ ddt(V(nsb,s)) - (1.0-V(nsb,s))/tc_bs + V(nsb,s)/te_bs;
  end
endmodule
"""
    path.write_text(text, encoding="utf-8")


def plot_diagnostics(label: pd.DataFrame, diag: pd.DataFrame, formulas: list[TauFormula]) -> None:
    fig, ax = plt.subplots(figsize=(6.6, 5.2))
    ax.loglog(diag["tau_s"], diag["pred_top1_tau_s"], "o", alpha=0.8)
    mn = min(float(diag["tau_s"].min()), float(diag["pred_top1_tau_s"].min()))
    mx = max(float(diag["tau_s"].max()), float(diag["pred_top1_tau_s"].max()))
    ax.plot([mn, mx], [mn, mx], "k--", lw=1.0)
    ax.set_xlabel("step-fit tau label (s)")
    ax.set_ylabel("history-KAN top-1 tau (s)")
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "history_kan_top1_tau_fit.png", dpi=220)
    plt.close(fig)

    table = []
    for f in formulas:
        table.append(
            {
                "family": f.family,
                "branch": f.branch,
                "process": f.process,
                "n": f.n,
                "soft_rmse_dec": f.train_rmse_dec,
                "top1_rmse_dec": f.hard_top1_rmse_dec,
                "n_edges": len(f.edges),
            }
        )
    pd.DataFrame(table).to_csv(OUT / "history_kan_group_metrics.csv", index=False)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    label, norm = build_label_table()
    formulas, diag = fit_all(label)
    label.to_csv(OUT / "srh_history_kan_training_labels.csv", index=False, encoding="utf-8-sig")
    diag.to_csv(OUT / "srh_history_kan_predictions.csv", index=False, encoding="utf-8-sig")
    payload = {
        "feature_names": FEATURE_NAMES,
        "basis_names": BASIS_NAMES,
        "normalization": norm,
        "formulas": [
            {
                **{k: v for k, v in asdict(f).items() if k != "edges"},
                "edges": [asdict(e) for e in f.edges],
            }
            for f in formulas
        ],
        "overall_top1_rmse_dec": float(np.sqrt(np.mean((diag["pred_top1_log_tau"] - diag["log_tau"]) ** 2))),
        "n_labels": int(len(label)),
        "note": "History-aware SRH KAN: trained on identifiable fast-short and slow-long labels with cycle/history features; exported as top-1 symbolic edge formulas.",
    }
    (OUT / "srh_history_kan_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_va(formulas, norm, OUT / "dg_tft_dyntrap_history_kan.va")
    plot_diagnostics(label, diag, formulas)
    print(json.dumps({"out": str(OUT), "n_labels": len(label), "overall_top1_rmse_dec": payload["overall_top1_rmse_dec"]}, indent=2))


if __name__ == "__main__":
    main()
