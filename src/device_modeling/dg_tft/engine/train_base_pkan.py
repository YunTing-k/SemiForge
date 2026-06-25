from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn

try:
    from kan import KAN

    PYKAN_AVAILABLE = True
except Exception:
    KAN = None
    PYKAN_AVAILABLE = False

from fit_physical_kernel import (
    DEFAULT_DATA_DIR,
    K_B,
    Q_E,
    CVSeries,
    CVTrapParams,
    Params,
    cv_biases_for_series,
    load_cv_data,
    load_iv_data,
    sample_cv,
    sample_iv,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_BASELINE_JSON = ROOT / "fit_training_ready_outputs" / "fitted_params.json"
DEFAULT_OUT_DIR = ROOT / "pkan_ivcv_torch_outputs"
ID_FLOOR = 1e-15
C_FLOOR = 1e-18
LOG10 = math.log(10.0)


@dataclass
class Normalizer:
    names: list[str]
    center: list[float]
    scale: list[float]


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def load_baseline(path: Path) -> tuple[Params, CVTrapParams, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    p_data = payload["final_params"]
    trap_data = payload.get("cv_shared_trap_params") or {}
    p = Params(**{k: p_data[k] for k in asdict(Params()).keys() if k in p_data})
    trap = CVTrapParams(**{k: trap_data[k] for k in asdict(CVTrapParams()).keys() if k in trap_data})
    return p, trap, payload


def tensor(x, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.as_tensor(x, dtype=dtype, device=device)


def smooth_pos_t(x: torch.Tensor, s: float = 1e-3) -> torch.Tensor:
    return 0.5 * (x + torch.sqrt(x * x + s * s))


def one_sided_sat_t(x: torch.Tensor, xknee: float, xsat: float, s: float = 1e-3) -> torch.Tensor:
    p = smooth_pos_t(x - xknee, s)
    return x - p + xsat * (1.0 - torch.exp(-p / xsat))


def qsolve_t(y: torch.Tensor) -> torch.Tensor:
    # The ultra-deep off-state has negligible channel charge and is leakage dominated.
    # Clamping y there prevents Newton's 1/q term from producing unstable autograd
    # gradients while preserving the physically relevant transition/on regions.
    # Do not hard-cap the on-state side: CV differentiates charge, and an upper
    # clamp creates a non-physical capacitance discontinuity when y crosses it.
    y_safe = torch.clamp(y, min=-35.0)
    q = torch.where(y_safe > 2.0, 0.5 * y_safe + 1e-12, torch.exp(torch.clamp(y_safe, -35.0, 2.0)))
    q = torch.clamp(q, min=1e-18)
    for _ in range(7):
        f = 2.0 * q + torch.log(q) - y_safe
        fp = 2.0 + 1.0 / q
        q = torch.clamp(q - f / fp, min=1e-18)
    return q


def effective_overdrive_t(vtg: torch.Tensor, vbg: torch.Tensor, p: Params) -> torch.Tensor:
    cox = p.Cox_t + p.Cox_b
    at = p.Cox_t / cox
    ab = p.Cox_b / cox
    return at * (vtg - p.Vth_t0) + ab * (vbg - p.Vth_b0)


def intrinsic_core_t(
    vtg: torch.Tensor, vbg: torch.Tensor, vds: torch.Tensor, p: Params, dveff: torch.Tensor | float = 0.0
):
    phi_t = K_B * p.Tnom / Q_E
    cox = p.Cox_t + p.Cox_b
    at = p.Cox_t / cox
    ab = p.Cox_b / cox
    nfac = 1.0 + smooth_pos_t(vtg.new_tensor(p.nfac0 - 1.0), 1e-6)
    dveff_t = torch.as_tensor(dveff, dtype=vtg.dtype, device=vtg.device)

    vov_t = one_sided_sat_t(vtg - p.Vth_t0, p.Vknee_t, p.Vmodsat_t)
    vov_b = one_sided_sat_t(vbg - p.Vth_b0, p.Vknee_b, p.Vmodsat_b)
    vov = at * vov_t + ab * vov_b + dveff_t
    vp = vov / nfac
    qs = qsolve_t(vp / phi_t)
    qd = qsolve_t((vp - vds) / phi_t)
    qbar = ((2.0 / 3.0) * (qs * qs + qs * qd + qd * qd) + 0.5 * (qs + qd)) / (qs + qd + 1.0)
    return phi_t, cox, at, ab, nfac, vov_t, vov_b, vp, qs, qd, qbar


def intrinsic_ids_t(vtg, vbg, vds, p: Params, dveff=0.0):
    phi_t, cox, _, _, nfac, _, _, vp, qs, qd, qbar = intrinsic_core_t(vtg, vbg, vds, p, dveff)
    vmob = 2.0 * nfac * phi_t * qbar
    mu_eff = p.mu0 / (1.0 + p.theta_mu * vmob)
    ispec = 2.0 * nfac * mu_eff * cox * phi_t * phi_t * p.W / p.L
    id0 = ispec * ((qs * qs + qs) - (qd * qd + qd))
    vabs = torch.sqrt(vds * vds + 1.0e-3**2)
    vdsat = smooth_pos_t(vp, 1.0e-3)
    vexcess = smooth_pos_t(vabs - vdsat, 1.0e-3)
    return id0 * (1.0 + p.lambda_clm * vexcess)


def leakage_ids_t(vds: torch.Tensor, p: Params) -> torch.Tensor:
    vabs = torch.sqrt(vds * vds + 1e-6**2)
    sign = vds / vabs
    turn_on = 1.0 - torch.exp(-vabs / 0.02)
    return sign * (p.Ioff0 * turn_on + p.Gleak_ds * vabs)


def ids_with_corr_t(vtg, vbg, vds, p: Params, dveff=0.0, dlogid_dec=0.0):
    channel = intrinsic_ids_t(vtg, vbg, vds, p, dveff)
    factor = torch.pow(10.0, torch.clamp(torch.as_tensor(dlogid_dec, dtype=vtg.dtype, device=vtg.device), -2.0, 2.0))
    return channel * factor + leakage_ids_t(vds, p)


def terminal_charges_corr_t(vtg, vbg, vds, p: Params, dveff=0.0):
    phi_t, cox, at, ab, nfac, vov_t, vov_b, _, qs, qd, qbar = intrinsic_core_t(vtg, vbg, vds, p, dveff)
    delta = vov_t - vov_b
    qtg = p.W * p.L * p.Cox_t * (ab * delta + 2.0 * nfac * phi_t * qbar)
    qbg = p.W * p.L * p.Cox_b * (-at * delta + 2.0 * nfac * phi_t * qbar)

    den = 60.0 * (qs + qd + 1.0) ** 2
    ns = (
        24 * qs**3
        + 48 * qs**2 * qd
        + 32 * qs * qd**2
        + 16 * qd**3
        + 45 * qs**2
        + 50 * qs * qd
        + 25 * qd**2
        + 20 * qs
        + 10 * qd
    )
    nd = (
        16 * qs**3
        + 32 * qs**2 * qd
        + 48 * qs * qd**2
        + 24 * qd**3
        + 25 * qs**2
        + 50 * qs * qd
        + 45 * qd**2
        + 10 * qs
        + 20 * qd
    )
    qs_term = -2.0 * nfac * cox * phi_t * p.W * p.L * ns / den
    qd_term = -2.0 * nfac * cox * phi_t * p.W * p.L * nd / den

    ctgs = p.Cov_tgs_w * p.W + p.Cfr_tgs
    ctgd = p.Cov_tgd_w * p.W + p.Cfr_tgd
    cbgs = p.Cov_bgs_w * p.W + p.Cfr_bgs
    cbgd = p.Cov_bgd_w * p.W + p.Cfr_bgd
    qtg = qtg + ctgs * vtg + ctgd * (vtg - vds)
    qbg = qbg + cbgs * vbg + cbgd * (vbg - vds)
    qd_term = qd_term + ctgd * (vds - vtg) + cbgd * (vds - vbg)
    qs_term = qs_term - ctgs * vtg - cbgs * vbg
    return qtg, qbg, qd_term, qs_term


def make_normalizer(names: list[str], values: np.ndarray) -> Normalizer:
    center = []
    scale = []
    for i in range(values.shape[1]):
        col = values[:, i]
        col = col[np.isfinite(col)]
        if len(col) == 0:
            center.append(0.0)
            scale.append(1.0)
            continue
        lo, hi = np.percentile(col, [2.0, 98.0])
        c = 0.5 * (lo + hi)
        s = 0.5 * (hi - lo)
        if not np.isfinite(s) or s < 1e-9:
            s = np.std(col)
        if not np.isfinite(s) or s < 1e-9:
            s = 1.0
        center.append(float(c))
        scale.append(float(1.10 * s + 1e-12))
    return Normalizer(names, center, scale)


class KANBlock(nn.Module):
    def __init__(
        self,
        normalizer: Normalizer,
        output_ranges: list[float],
        hidden: int,
        grid: int,
        k: int,
        seed: int,
        device: torch.device,
        dtype: torch.dtype,
    ):
        super().__init__()
        self.normalizer = normalizer
        self.register_buffer("center", torch.tensor(normalizer.center, dtype=dtype, device=device))
        self.register_buffer("scale", torch.tensor(normalizer.scale, dtype=dtype, device=device))
        self.register_buffer("output_ranges", torch.tensor(output_ranges, dtype=dtype, device=device))
        out_dim = len(output_ranges)
        in_dim = len(normalizer.names)
        if PYKAN_AVAILABLE:
            self.net = KAN(
                width=[in_dim, hidden, out_dim],
                grid=grid,
                k=k,
                seed=seed,
                device=device,
                symbolic_enabled=False,
                save_act=False,
                auto_save=False,
                grid_range=[-1, 1],
            )
            self.net.to(device)
            try:
                self.net.to(dtype=dtype)
            except TypeError:
                self.net = self.net.to(dtype)
            if hasattr(self.net, "speed"):
                try:
                    self.net.speed()
                except Exception:
                    pass
        else:
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden),
                nn.Tanh(),
                nn.Linear(hidden, hidden),
                nn.Tanh(),
                nn.Linear(hidden, out_dim),
            ).to(device=device, dtype=dtype)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh((x - self.center) / self.scale)

    def forward(self, x: torch.Tensor, scale: float | torch.Tensor = 1.0) -> torch.Tensor:
        raw = self.net(self.normalize(x))
        raw = torch.nan_to_num(raw, nan=0.0, posinf=20.0, neginf=-20.0)
        return torch.tanh(raw) * self.output_ranges * scale

    def config(self) -> dict:
        return {
            "normalizer": asdict(self.normalizer),
            "output_ranges": self.output_ranges.detach().cpu().numpy().tolist(),
            "uses_pykan": PYKAN_AVAILABLE,
        }


def gate_weights_np(sheet: str) -> tuple[float, float, int]:
    if sheet == "cbg":
        return 0.0, 1.0, 0
    if sheet == "ctg":
        return 1.0, 0.0, 1
    return 1.0, 1.0, 2


def shared_features_np(vbg, vtg, vds, p: Params) -> np.ndarray:
    vbg = np.asarray(vbg, dtype=float)
    vtg = np.asarray(vtg, dtype=float)
    vds = np.asarray(vds, dtype=float)
    cox = p.Cox_t + p.Cox_b
    at = p.Cox_t / cox
    ab = p.Cox_b / cox
    veff = at * (vtg - p.Vth_t0) + ab * (vbg - p.Vth_b0)
    return np.column_stack([vbg, vtg, vds, veff, vbg + vtg, vds * veff])


def shared_features_t(vbg, vtg, vds, p: Params) -> torch.Tensor:
    veff = effective_overdrive_t(vtg, vbg, p)
    return torch.stack([vbg, vtg, vds, veff, vbg + vtg, vds * veff], dim=1)


def iv_features_t(vbg, vtg, vds, logid_base, p: Params) -> torch.Tensor:
    sf = shared_features_t(vbg, vtg, vds, p)
    return torch.cat([sf, logid_base[:, None]], dim=1)


def iv_features_np(vbg, vtg, vds, logid_base, p: Params) -> np.ndarray:
    return np.column_stack([shared_features_np(vbg, vtg, vds, p), np.asarray(logid_base, dtype=float)])


def cv_features_np(vbg, vtg, vds, freq, gt, gb, p: Params) -> np.ndarray:
    vbg = np.asarray(vbg, dtype=float)
    vtg = np.asarray(vtg, dtype=float)
    vds = np.asarray(vds, dtype=float)
    freq = np.asarray(freq, dtype=float)
    gt_arr = np.asarray(gt, dtype=float) + np.zeros_like(vbg)
    gb_arr = np.asarray(gb, dtype=float) + np.zeros_like(vbg)
    cox = p.Cox_t + p.Cox_b
    at = p.Cox_t / cox
    ab = p.Cox_b / cox
    veff = at * (vtg - p.Vth_t0) + ab * (vbg - p.Vth_b0)
    logf = np.log10(np.maximum(freq, 1.0))
    return np.column_stack([vbg, vtg, vds, veff, logf, gt_arr, gb_arr, veff * gt_arr, veff * gb_arr])


def cv_features_t(vbg, vtg, vds, freq, gt, gb, p: Params) -> torch.Tensor:
    veff = effective_overdrive_t(vtg, vbg, p)
    logf = torch.log10(torch.clamp(freq, min=1.0))
    return torch.stack([vbg, vtg, vds, veff, logf, gt, gb, veff * gt, veff * gb], dim=1)


def make_iv_tensors(df: pd.DataFrame, device: torch.device, dtype: torch.dtype) -> dict[str, torch.Tensor]:
    return {
        "vbg": tensor(df["VBG_V"].to_numpy(float), device, dtype),
        "vtg": tensor(df["VTG_V"].to_numpy(float), device, dtype),
        "vds": tensor(df["VDS_V"].to_numpy(float), device, dtype),
        "id": tensor(df["ID_A"].to_numpy(float), device, dtype),
        "weight": tensor(iv_weights_np(df), device, dtype),
    }


def cv_point_dataframe(series: list[CVSeries], trap: CVTrapParams) -> pd.DataFrame:
    rows = []
    for s in series:
        vtg, vbg, vds = cv_biases_for_series(s)
        gt, gb, mode = gate_weights_np(s.sheet)
        freq = trap.default_freq_hz if s.freq_hz is None else s.freq_hz
        temp_k = getattr(s, "temp_K", None)
        for i in range(len(s.x)):
            rows.append(
                {
                    "sheet": s.sheet,
                    "label": s.label,
                    "x": float(s.x[i]),
                    "VBG_V": float(vbg[i]),
                    "VTG_V": float(vtg[i]),
                    "VDS_V": float(vds[i]),
                    "freq_hz": float(freq),
                    "gate_t": gt,
                    "gate_b": gb,
                    "mode": mode,
                    "cap_F": float(s.cap[i]),
                    "temp_K": float(temp_k) if temp_k is not None else np.nan,
                }
            )
    return pd.DataFrame(rows)


def make_cv_tensors(df: pd.DataFrame, device: torch.device, dtype: torch.dtype) -> dict[str, torch.Tensor]:
    return {
        "vbg": tensor(df["VBG_V"].to_numpy(float), device, dtype),
        "vtg": tensor(df["VTG_V"].to_numpy(float), device, dtype),
        "vds": tensor(df["VDS_V"].to_numpy(float), device, dtype),
        "freq": tensor(df["freq_hz"].to_numpy(float), device, dtype),
        "gt": tensor(df["gate_t"].to_numpy(float), device, dtype),
        "gb": tensor(df["gate_b"].to_numpy(float), device, dtype),
        "mode": torch.as_tensor(df["mode"].to_numpy(int), dtype=torch.long, device=device),
        "cap": tensor(df["cap_F"].to_numpy(float), device, dtype),
        "weight": tensor(cv_weights_np(df), device, dtype),
    }


def iv_weights_np(df: pd.DataFrame) -> np.ndarray:
    abs_id = np.maximum(np.abs(df["ID_A"].to_numpy(float)), ID_FLOOR)
    log_i = np.log10(abs_id)
    w = np.ones(len(df), dtype=float)
    w *= np.where(df["family"].to_numpy() == "IDVD", 0.90, 1.10)
    w *= np.where((-12.3 <= log_i) & (log_i <= -8.7), 2.25, 1.0)
    w *= np.where(log_i > -8.7, 1.55, 1.0)
    w *= np.where(log_i < -12.8, 0.60, 1.0)
    w *= np.where(np.abs(df["VDS_V"].to_numpy(float)) < 0.02, 0.25, 1.0)
    return w


def cv_weights_np(df: pd.DataFrame) -> np.ndarray:
    sheet_w = df["sheet"].map({"cgg": 1.12, "ctg": 1.0, "cbg": 1.0, "vds": 0.9}).fillna(1.0)
    return sheet_w.to_numpy(float)


def batch(data: dict[str, torch.Tensor], idx: torch.Tensor) -> dict[str, torch.Tensor]:
    return {k: v[idx] for k, v in data.items()}


def sample_idx(n: int, batch_size: int, device: torch.device) -> torch.Tensor:
    if n <= batch_size:
        return torch.arange(n, device=device)
    return torch.randint(0, n, (batch_size,), device=device)


def log10_abs(x: torch.Tensor, floor: float) -> torch.Tensor:
    return torch.log10(torch.clamp(torch.abs(x), min=floor))


def weighted_mse(err: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    err = torch.nan_to_num(torch.clamp(err, -8.0, 8.0), nan=0.0, posinf=8.0, neginf=-8.0)
    return torch.sum(weight * err * err) / torch.clamp(torch.sum(weight), min=1e-12)


def calc_dveff(shared: KANBlock, vbg, vtg, vds, p: Params, scale: float = 1.0) -> torch.Tensor:
    return shared(shared_features_t(vbg, vtg, vds, p), scale=scale)[:, 0]


def calc_dlogid(iv_kan: KANBlock, vbg, vtg, vds, base_logid, p: Params, scale: float = 1.0) -> torch.Tensor:
    return iv_kan(iv_features_t(vbg, vtg, vds, base_logid, p), scale=scale)[:, 0]


def calc_dlogcit(cv_kan: KANBlock, vbg, vtg, vds, freq, gt, gb, p: Params, scale: float = 1.0) -> torch.Tensor:
    return cv_kan(cv_features_t(vbg, vtg, vds, freq, gt, gb, p), scale=scale)


def trap_response_t(freq: torch.Tensor, trap: CVTrapParams) -> torch.Tensor:
    omega_tau = 2.0 * math.pi * torch.clamp(freq, min=1.0) * trap.tau_s
    slow = 1.0 / (1.0 + torch.pow(omega_tau, trap.alpha))
    return trap.trap_fast_frac + (1.0 - trap.trap_fast_frac) * slow


def cv_core_t(data: dict[str, torch.Tensor], p: Params, shared: KANBlock | None, scale: float, h: float) -> torch.Tensor:
    vtg = data["vtg"]
    vbg = data["vbg"]
    vds = data["vds"]
    mode = data["mode"]
    tied = mode == 2
    cbg = mode == 0
    ctg = mode == 1

    dh = vtg.new_tensor(h)
    vtg_p = vtg + dh * (ctg | tied).to(vtg.dtype)
    vtg_m = vtg - dh * (ctg | tied).to(vtg.dtype)
    vbg_p = vbg + dh * (cbg | tied).to(vbg.dtype)
    vbg_m = vbg - dh * (cbg | tied).to(vbg.dtype)

    if shared is None:
        dveff_p = 0.0
        dveff_m = 0.0
    else:
        dveff_p = calc_dveff(shared, vbg_p, vtg_p, vds, p, scale)
        dveff_m = calc_dveff(shared, vbg_m, vtg_m, vds, p, scale)
    qtg_p, qbg_p, _, _ = terminal_charges_corr_t(vtg_p, vbg_p, vds, p, dveff_p)
    qtg_m, qbg_m, _, _ = terminal_charges_corr_t(vtg_m, vbg_m, vds, p, dveff_m)
    q_p = torch.where(cbg, qbg_p, torch.where(ctg, qtg_p, qtg_p + qbg_p))
    q_m = torch.where(cbg, qbg_m, torch.where(ctg, qtg_m, qtg_m + qbg_m))
    return torch.clamp(torch.abs((q_p - q_m) / (2.0 * dh)), min=C_FLOOR)


def cv_trap_t(
    data: dict[str, torch.Tensor],
    p: Params,
    trap: CVTrapParams,
    shared: KANBlock | None,
    cv_kan: KANBlock | None,
    scale_shared: float,
    scale_cv: float,
) -> torch.Tensor:
    vbg = data["vbg"]
    vtg = data["vtg"]
    vds = data["vds"]
    freq = data["freq"]
    gt = data["gt"]
    gb = data["gb"]
    if shared is None:
        dveff = torch.zeros_like(vbg)
    else:
        dveff = calc_dveff(shared, vbg, vtg, vds, p, scale_shared)
    if cv_kan is None:
        dlog_t = torch.zeros_like(vbg)
        dlog_b = torch.zeros_like(vbg)
    else:
        dlog = calc_dlogcit(cv_kan, vbg, vtg, vds, freq, gt, gb, p, scale_cv)
        dlog_t = dlog[:, 0]
        dlog_b = dlog[:, 1]
    occ = torch.sigmoid((effective_overdrive_t(vtg, vbg, p) + dveff) / trap.trap_width)
    resp = trap_response_t(freq, trap)
    cit_t = trap.Cit_t * torch.pow(10.0, torch.clamp(dlog_t, -2.0, 2.0))
    cit_b = trap.Cit_b * torch.pow(10.0, torch.clamp(dlog_b, -2.0, 2.0))
    cit = gt * cit_t + gb * cit_b
    return torch.clamp(cit * occ * resp, min=0.0)


def cv_pred_t(
    data: dict[str, torch.Tensor],
    p: Params,
    trap: CVTrapParams,
    shared: KANBlock | None,
    cv_kan: KANBlock | None,
    scale_shared: float,
    scale_cv: float,
    h: float,
) -> torch.Tensor:
    return torch.clamp(
        cv_core_t(data, p, shared, scale_shared, h)
        + cv_trap_t(data, p, trap, shared, cv_kan, scale_shared, scale_cv),
        min=C_FLOOR,
    )


def make_iv_anchors(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    if n <= 0:
        return pd.DataFrame(columns=["VBG_V", "VTG_V", "VDS_V"])
    vbg_min, vbg_max = df["VBG_V"].min(), df["VBG_V"].max()
    vtg_min, vtg_max = df["VTG_V"].min(), df["VTG_V"].max()
    vds_min, vds_max = df["VDS_V"].min(), df["VDS_V"].max()
    vbg_span = max(vbg_max - vbg_min, 1.0)
    vtg_span = max(vtg_max - vtg_min, 1.0)
    vds_span = max(vds_max - vds_min, 0.5)
    vds_choices = np.unique(np.round(df["VDS_V"].to_numpy(float), 6))
    return pd.DataFrame(
        {
            "VBG_V": rng.uniform(vbg_min - 0.35 * vbg_span, vbg_max + 0.35 * vbg_span, n),
            "VTG_V": rng.uniform(vtg_min - 0.35 * vtg_span, vtg_max + 0.35 * vtg_span, n),
            "VDS_V": rng.choice(
                np.concatenate([vds_choices, rng.uniform(max(0.0, vds_min), vds_max + 0.35 * vds_span, 8)]),
                n,
                replace=True,
            ),
        }
    )


def make_cv_anchors(cv_df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    if n <= 0 or cv_df.empty:
        return pd.DataFrame(columns=["VBG_V", "VTG_V", "VDS_V", "freq_hz", "gate_t", "gate_b", "mode"])
    vbg_min, vbg_max = cv_df["VBG_V"].min(), cv_df["VBG_V"].max()
    vtg_min, vtg_max = cv_df["VTG_V"].min(), cv_df["VTG_V"].max()
    vbg_span = max(vbg_max - vbg_min, 1.0)
    vtg_span = max(vtg_max - vtg_min, 1.0)
    vds_choices = np.unique(np.round(cv_df["VDS_V"].to_numpy(float), 6))
    freq_choices = np.unique(np.round(cv_df["freq_hz"].to_numpy(float), 6))
    rows = []
    for _ in range(n):
        mode_name = rng.choice(["cgg", "cbg", "ctg", "vds"])
        gt, gb, mode = gate_weights_np(mode_name)
        xbg = rng.uniform(vbg_min - 0.30 * vbg_span, vbg_max + 0.30 * vbg_span)
        xtg = rng.uniform(vtg_min - 0.30 * vtg_span, vtg_max + 0.30 * vtg_span)
        if mode == 2:
            x = 0.5 * (xbg + xtg)
            vbg = x
            vtg = x
        elif mode == 0:
            vbg = xbg
            vtg = rng.choice([-1.0, -0.5, 0.0, 0.5, 1.0])
        else:
            vtg = xtg
            vbg = rng.choice([0.0, 1.0, 2.0, 3.0])
        rows.append(
            {
                "VBG_V": vbg,
                "VTG_V": vtg,
                "VDS_V": float(rng.choice(vds_choices)),
                "freq_hz": float(rng.choice(freq_choices)),
                "gate_t": gt,
                "gate_b": gb,
                "mode": mode,
            }
        )
    return pd.DataFrame(rows)


def regularize_params(modules: list[nn.Module]) -> torch.Tensor:
    terms = []
    for module in modules:
        for p in module.parameters():
            if p.requires_grad:
                terms.append(torch.mean(p * p))
    if not terms:
        return torch.tensor(0.0)
    return sum(terms) / len(terms)


def stage_shared(
    shared: KANBlock,
    iv: dict[str, torch.Tensor],
    cv: dict[str, torch.Tensor],
    iv_anchor: dict[str, torch.Tensor],
    cv_anchor: dict[str, torch.Tensor],
    p: Params,
    trap: CVTrapParams,
    args,
):
    opt = torch.optim.AdamW(shared.parameters(), lr=args.lr_shared, weight_decay=0.0)
    n_iv = len(iv["id"])
    n_cv = len(cv["cap"])
    for ep in range(1, args.epochs_shared + 1):
        scale = args.kan_scale * min(1.0, ep / max(args.warmup_epochs, 1))
        opt.zero_grad(set_to_none=True)
        ib = batch(iv, sample_idx(n_iv, args.batch_iv, iv["id"].device))
        cb = batch(cv, sample_idx(n_cv, args.batch_cv, cv["cap"].device))
        dveff = calc_dveff(shared, ib["vbg"], ib["vtg"], ib["vds"], p, scale)
        pred_id = ids_with_corr_t(ib["vtg"], ib["vbg"], ib["vds"], p, dveff, 0.0)
        err_iv = log10_abs(pred_id, ID_FLOOR) - log10_abs(ib["id"], ID_FLOOR)
        loss_iv = weighted_mse(err_iv, ib["weight"])

        pred_c = cv_pred_t(cb, p, trap, shared, None, scale, 0.0, args.cv_diff_h)
        err_cv = torch.log10(pred_c) - torch.log10(torch.clamp(cb["cap"], min=C_FLOOR))
        loss_cv = weighted_mse(err_cv, cb["weight"])

        a_iv = calc_dveff(shared, iv_anchor["vbg"], iv_anchor["vtg"], iv_anchor["vds"], p, scale)
        a_cv = calc_dveff(shared, cv_anchor["vbg"], cv_anchor["vtg"], cv_anchor["vds"], p, scale)
        loss_anchor = torch.mean((a_iv / args.dveff_range) ** 2) + torch.mean((a_cv / args.dveff_range) ** 2)
        loss_small = torch.mean((dveff / args.dveff_range) ** 2)
        loss_reg = regularize_params([shared]).to(loss_iv.device)
        loss = loss_iv + args.cv_weight_shared * loss_cv + args.anchor_weight * loss_anchor + args.small_weight * loss_small
        loss = loss + args.param_l2 * loss_reg
        loss.backward()
        torch.nn.utils.clip_grad_norm_(shared.parameters(), args.grad_clip)
        opt.step()
        if ep == 1 or ep % args.print_every == 0 or ep == args.epochs_shared:
            print(
                f"shared {ep:5d}/{args.epochs_shared}: "
                f"loss={loss.item():.5g} iv={loss_iv.item():.4g} cv={loss_cv.item():.4g} "
                f"anchor={loss_anchor.item():.4g} scale={scale:.3f}"
            )


def stage_iv(
    shared: KANBlock,
    iv_kan: KANBlock,
    iv: dict[str, torch.Tensor],
    iv_anchor: dict[str, torch.Tensor],
    p: Params,
    args,
):
    shared.eval()
    for param in shared.parameters():
        param.requires_grad_(False)
    opt = torch.optim.AdamW(iv_kan.parameters(), lr=args.lr_iv, weight_decay=0.0)
    n_iv = len(iv["id"])
    for ep in range(1, args.epochs_iv + 1):
        scale = args.kan_scale * min(1.0, ep / max(args.warmup_epochs, 1))
        opt.zero_grad(set_to_none=True)
        ib = batch(iv, sample_idx(n_iv, args.batch_iv, iv["id"].device))
        with torch.no_grad():
            dveff = calc_dveff(shared, ib["vbg"], ib["vtg"], ib["vds"], p, args.kan_scale)
            base = ids_with_corr_t(ib["vtg"], ib["vbg"], ib["vds"], p, dveff, 0.0)
            base_log = log10_abs(base, ID_FLOOR)
        dlog = calc_dlogid(iv_kan, ib["vbg"], ib["vtg"], ib["vds"], base_log, p, scale)
        pred_id = ids_with_corr_t(ib["vtg"], ib["vbg"], ib["vds"], p, dveff, dlog)
        err_iv = log10_abs(pred_id, ID_FLOOR) - log10_abs(ib["id"], ID_FLOOR)
        loss_iv = weighted_mse(err_iv, ib["weight"])

        with torch.no_grad():
            a_dveff = calc_dveff(shared, iv_anchor["vbg"], iv_anchor["vtg"], iv_anchor["vds"], p, args.kan_scale)
            a_base = ids_with_corr_t(iv_anchor["vtg"], iv_anchor["vbg"], iv_anchor["vds"], p, a_dveff, 0.0)
            a_base_log = log10_abs(a_base, ID_FLOOR)
        a_dlog = calc_dlogid(iv_kan, iv_anchor["vbg"], iv_anchor["vtg"], iv_anchor["vds"], a_base_log, p, scale)
        loss_anchor = torch.mean((a_dlog / args.dlogid_range) ** 2)
        loss_small = torch.mean((dlog / args.dlogid_range) ** 2)
        loss_reg = regularize_params([iv_kan]).to(loss_iv.device)
        loss = loss_iv + args.anchor_weight * loss_anchor + args.small_weight * loss_small + args.param_l2 * loss_reg
        loss.backward()
        torch.nn.utils.clip_grad_norm_(iv_kan.parameters(), args.grad_clip)
        opt.step()
        if ep == 1 or ep % args.print_every == 0 or ep == args.epochs_iv:
            print(
                f"iv      {ep:5d}/{args.epochs_iv}: "
                f"loss={loss.item():.5g} iv={loss_iv.item():.4g} anchor={loss_anchor.item():.4g} scale={scale:.3f}"
            )


def stage_cv(
    shared: KANBlock,
    cv_kan: KANBlock,
    cv: dict[str, torch.Tensor],
    cv_anchor: dict[str, torch.Tensor],
    p: Params,
    trap: CVTrapParams,
    args,
):
    shared.eval()
    for param in shared.parameters():
        param.requires_grad_(False)
    opt = torch.optim.AdamW(cv_kan.parameters(), lr=args.lr_cv, weight_decay=0.0)
    n_cv = len(cv["cap"])
    for ep in range(1, args.epochs_cv + 1):
        scale = args.kan_scale * min(1.0, ep / max(args.warmup_epochs, 1))
        opt.zero_grad(set_to_none=True)
        cb = batch(cv, sample_idx(n_cv, args.batch_cv, cv["cap"].device))
        pred_c = cv_pred_t(cb, p, trap, shared, cv_kan, args.kan_scale, scale, args.cv_diff_h)
        err_cv = torch.log10(pred_c) - torch.log10(torch.clamp(cb["cap"], min=C_FLOOR))
        loss_cv = weighted_mse(err_cv, cb["weight"])

        a_dlog = calc_dlogcit(
            cv_kan,
            cv_anchor["vbg"],
            cv_anchor["vtg"],
            cv_anchor["vds"],
            cv_anchor["freq"],
            cv_anchor["gt"],
            cv_anchor["gb"],
            p,
            scale,
        )
        loss_anchor = torch.mean((a_dlog / args.dlogcit_range) ** 2)
        loss_small = torch.mean((calc_dlogcit(cv_kan, cb["vbg"], cb["vtg"], cb["vds"], cb["freq"], cb["gt"], cb["gb"], p, scale) / args.dlogcit_range) ** 2)
        loss_reg = regularize_params([cv_kan]).to(loss_cv.device)
        loss = loss_cv + args.anchor_weight * loss_anchor + args.small_weight * loss_small + args.param_l2 * loss_reg
        loss.backward()
        torch.nn.utils.clip_grad_norm_(cv_kan.parameters(), args.grad_clip)
        opt.step()
        if ep == 1 or ep % args.print_every == 0 or ep == args.epochs_cv:
            print(
                f"cv      {ep:5d}/{args.epochs_cv}: "
                f"loss={loss.item():.5g} cv={loss_cv.item():.4g} anchor={loss_anchor.item():.4g} scale={scale:.3f}"
            )


def eval_iv(iv_t: dict[str, torch.Tensor], p: Params, shared: KANBlock, iv_kan: KANBlock, args) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    outs = []
    dveffs = []
    dlogs = []
    n = len(iv_t["id"])
    with torch.no_grad():
        for start in range(0, n, args.eval_batch):
            sl = slice(start, min(start + args.eval_batch, n))
            b = {k: v[sl] for k, v in iv_t.items()}
            dveff = calc_dveff(shared, b["vbg"], b["vtg"], b["vds"], p, args.kan_scale)
            base = ids_with_corr_t(b["vtg"], b["vbg"], b["vds"], p, dveff, 0.0)
            base_log = log10_abs(base, ID_FLOOR)
            dlog = calc_dlogid(iv_kan, b["vbg"], b["vtg"], b["vds"], base_log, p, args.kan_scale)
            pred = ids_with_corr_t(b["vtg"], b["vbg"], b["vds"], p, dveff, dlog)
            outs.append(pred.detach().cpu().numpy())
            dveffs.append(dveff.detach().cpu().numpy())
            dlogs.append(dlog.detach().cpu().numpy())
    return np.concatenate(outs), np.concatenate(dveffs), np.concatenate(dlogs)


def eval_cv(cv_t: dict[str, torch.Tensor], p: Params, trap: CVTrapParams, shared: KANBlock, cv_kan: KANBlock, args):
    caps = []
    dveffs = []
    dlogts = []
    dlogbs = []
    n = len(cv_t["cap"])
    with torch.no_grad():
        for start in range(0, n, args.eval_batch):
            sl = slice(start, min(start + args.eval_batch, n))
            b = {k: v[sl] for k, v in cv_t.items()}
            pred = cv_pred_t(b, p, trap, shared, cv_kan, args.kan_scale, args.kan_scale, args.cv_diff_h)
            d = calc_dveff(shared, b["vbg"], b["vtg"], b["vds"], p, args.kan_scale)
            dlog = calc_dlogcit(cv_kan, b["vbg"], b["vtg"], b["vds"], b["freq"], b["gt"], b["gb"], p, args.kan_scale)
            caps.append(pred.detach().cpu().numpy())
            dveffs.append(d.detach().cpu().numpy())
            dlogts.append(dlog[:, 0].detach().cpu().numpy())
            dlogbs.append(dlog[:, 1].detach().cpu().numpy())
    return np.concatenate(caps), np.concatenate(dveffs), np.concatenate(dlogts), np.concatenate(dlogbs)


def make_models(
    iv_fit: pd.DataFrame,
    cv_fit_df: pd.DataFrame,
    iv_anchor_df: pd.DataFrame,
    cv_anchor_df: pd.DataFrame,
    p: Params,
    args,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[KANBlock, KANBlock, KANBlock]:
    shared_names = ["VBG", "VTG", "VDS", "Veff", "VBG_plus_VTG", "VDS_times_Veff"]
    shared_values = [shared_features_np(iv_fit["VBG_V"], iv_fit["VTG_V"], iv_fit["VDS_V"], p)]
    if not cv_fit_df.empty:
        shared_values.append(shared_features_np(cv_fit_df["VBG_V"], cv_fit_df["VTG_V"], cv_fit_df["VDS_V"], p))
    shared_values.append(shared_features_np(iv_anchor_df["VBG_V"], iv_anchor_df["VTG_V"], iv_anchor_df["VDS_V"], p))
    shared_values.append(shared_features_np(cv_anchor_df["VBG_V"], cv_anchor_df["VTG_V"], cv_anchor_df["VDS_V"], p))
    shared_norm = make_normalizer(shared_names, np.vstack(shared_values))
    shared = KANBlock(
        shared_norm,
        [args.dveff_range],
        args.kan_hidden,
        args.kan_grid,
        args.kan_k,
        args.seed,
        device,
        dtype,
    )

    # The IV KAN receives a baseline-current feature. Use the pure baseline for normalization;
    # the actual training feature is recomputed after the shared KAN is trained.
    vtg = iv_fit["VTG_V"].to_numpy(float)
    vbg = iv_fit["VBG_V"].to_numpy(float)
    vds = iv_fit["VDS_V"].to_numpy(float)
    id_base = np.full(len(iv_fit), -11.0, dtype=float)
    iv_norm = make_normalizer(shared_names + ["log10_ID_base"], iv_features_np(vbg, vtg, vds, id_base, p))
    iv_kan = KANBlock(
        iv_norm,
        [args.dlogid_range],
        args.kan_hidden,
        args.kan_grid,
        args.kan_k,
        args.seed + 11,
        device,
        dtype,
    )

    cv_names = ["VBG", "VTG", "VDS", "Veff", "log10_freq", "gate_t", "gate_b", "Veff_gate_t", "Veff_gate_b"]
    cv_values = cv_features_np(
        cv_fit_df["VBG_V"],
        cv_fit_df["VTG_V"],
        cv_fit_df["VDS_V"],
        cv_fit_df["freq_hz"],
        cv_fit_df["gate_t"],
        cv_fit_df["gate_b"],
        p,
    )
    cv_anchor_values = cv_features_np(
        cv_anchor_df["VBG_V"],
        cv_anchor_df["VTG_V"],
        cv_anchor_df["VDS_V"],
        cv_anchor_df["freq_hz"],
        cv_anchor_df["gate_t"],
        cv_anchor_df["gate_b"],
        p,
    )
    cv_norm = make_normalizer(cv_names, np.vstack([cv_values, cv_anchor_values]))
    cv_kan = KANBlock(
        cv_norm,
        [args.dlogcit_range, args.dlogcit_range],
        args.kan_hidden,
        args.kan_grid,
        args.kan_k,
        args.seed + 23,
        device,
        dtype,
    )
    return shared, iv_kan, cv_kan


def recompute_iv_normalizer(iv_kan: KANBlock, iv_fit_t: dict[str, torch.Tensor], p: Params, shared: KANBlock, args):
    with torch.no_grad():
        d = calc_dveff(shared, iv_fit_t["vbg"], iv_fit_t["vtg"], iv_fit_t["vds"], p, args.kan_scale)
        base = ids_with_corr_t(iv_fit_t["vtg"], iv_fit_t["vbg"], iv_fit_t["vds"], p, d, 0.0)
        logbase = log10_abs(base, ID_FLOOR).detach().cpu().numpy()
    vbg = iv_fit_t["vbg"].detach().cpu().numpy()
    vtg = iv_fit_t["vtg"].detach().cpu().numpy()
    vds = iv_fit_t["vds"].detach().cpu().numpy()
    norm = make_normalizer(iv_kan.normalizer.names, iv_features_np(vbg, vtg, vds, logbase, p))
    iv_kan.normalizer = norm
    dtype = iv_kan.center.dtype
    device = iv_kan.center.device
    iv_kan.center.data = torch.tensor(norm.center, dtype=dtype, device=device)
    iv_kan.scale.data = torch.tensor(norm.scale, dtype=dtype, device=device)


def metrics(iv_df: pd.DataFrame, cv_df: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    err = iv_df["log10_abs_error_dec"].to_numpy(float)
    log_i = np.log10(np.maximum(np.abs(iv_df["ID_A"].to_numpy(float)), ID_FLOOR))
    sub = (-12.3 <= log_i) & (log_i <= -8.7)
    on = log_i > -8.7
    off = log_i < -12.8
    cerr = cv_df["log10_cap_error_dec"].to_numpy(float)
    summary = {
        "iv_rms_log10_error_dec": float(np.sqrt(np.mean(err**2))),
        "iv_median_abs_log10_error_dec": float(np.median(np.abs(err))),
        "iv_subthreshold_rms_log10_error_dec": float(np.sqrt(np.mean(err[sub] ** 2))) if sub.any() else None,
        "iv_on_rms_log10_error_dec": float(np.sqrt(np.mean(err[on] ** 2))) if on.any() else None,
        "iv_off_rms_log10_error_dec": float(np.sqrt(np.mean(err[off] ** 2))) if off.any() else None,
        "cv_rms_log10_error_dec": float(np.sqrt(np.mean(cerr**2))) if len(cerr) else None,
        "cv_median_abs_log10_error_dec": float(np.median(np.abs(cerr))) if len(cerr) else None,
        "dveff_abs_p95_V": float(np.percentile(np.abs(iv_df["dveff_KAN_V"]), 95)),
        "dlogid_abs_p95_dec": float(np.percentile(np.abs(iv_df["dlogid_KAN_dec"]), 95)),
        "dlogCit_t_abs_p95_dec": float(np.percentile(np.abs(cv_df["dlogCit_t_KAN_dec"]), 95)) if len(cv_df) else None,
        "dlogCit_b_abs_p95_dec": float(np.percentile(np.abs(cv_df["dlogCit_b_KAN_dec"]), 95)) if len(cv_df) else None,
    }
    rows = []
    for (family, source), g in iv_df.groupby(["family", "source_file"]):
        e = g["log10_abs_error_dec"].to_numpy(float)
        rows.append(
            {
                "type": "IV",
                "family": family,
                "source": source,
                "n": len(g),
                "rms_log10_error_dec": float(np.sqrt(np.mean(e**2))),
                "median_abs_log10_error_dec": float(np.median(np.abs(e))),
            }
        )
    for (sheet, label), g in cv_df.groupby(["sheet", "label"]):
        e = g["log10_cap_error_dec"].to_numpy(float)
        rows.append(
            {
                "type": "CV",
                "family": sheet,
                "source": label,
                "n": len(g),
                "rms_log10_error_dec": float(np.sqrt(np.mean(e**2))),
                "median_abs_log10_error_dec": float(np.median(np.abs(e))),
            }
        )
    return summary, pd.DataFrame(rows)


def setup_plot_style():
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 9,
            "axes.linewidth": 1.1,
            "axes.grid": True,
            "grid.alpha": 0.25,
        }
    )


def plot_iv_family(iv_df: pd.DataFrame, family: str, out: Path):
    data = iv_df[iv_df["family"] == family].copy()
    if data.empty:
        return
    xcol = {"IDVBG": "VBG_V", "IDVTG": "VTG_V", "IDVD": "VDS_V"}.get(family, "VBG_V")
    group_cols = {
        "IDVBG": ["source_file", "VDS_V", "VTG_V"],
        "IDVTG": ["source_file", "VDS_V", "VBG_V"],
        "IDVD": ["source_file", "VBG_V", "VTG_V"],
    }.get(family, ["source_file"])
    fig, ax = plt.subplots(figsize=(7.0, 5.6))
    for idx, (_, g) in enumerate(data.groupby(group_cols, sort=True)):
        if idx >= 80:
            break
        g = g.sort_values(xcol)
        ax.semilogy(g[xcol], np.maximum(np.abs(g["ID_A"]), ID_FLOOR), ".", ms=3, alpha=0.50)
        ax.semilogy(g[xcol], np.maximum(np.abs(g["ID_pkan_A"]), ID_FLOOR), "-", lw=1.1, alpha=0.88)
    ax.set_xlabel(xcol.replace("_V", " (V)"))
    ax.set_ylabel("|ID| (A)")
    ax.set_title(f"{family}: measured points and PKAN fit")
    fig.tight_layout()
    fig.savefig(out, dpi=220)
    plt.close(fig)


def plot_cv(cv_df: pd.DataFrame, out: Path):
    if cv_df.empty:
        return
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 7.0), sharey=False)
    axes = axes.ravel()
    for ax, sheet in zip(axes, ["cgg", "cbg", "ctg", "vds"]):
        data = cv_df[cv_df["sheet"] == sheet]
        if data.empty:
            ax.set_visible(False)
            continue
        for _, g in data.groupby("label", sort=True):
            g = g.sort_values("x")
            ax.semilogy(g["x"], np.maximum(g["cap_meas_F"], C_FLOOR), ".", ms=3, alpha=0.50)
            ax.semilogy(g["x"], np.maximum(g["cap_pkan_F"], C_FLOOR), "-", lw=1.1, alpha=0.88)
        ax.set_title(sheet)
        ax.set_xlabel("sweep voltage (V)")
        ax.set_ylabel("C (F)")
    fig.suptitle("CV: measured points and PKAN fit")
    fig.tight_layout()
    fig.savefig(out, dpi=220)
    plt.close(fig)


def write_outputs(
    out_dir: Path,
    p: Params,
    trap: CVTrapParams,
    baseline_payload: dict,
    shared: KANBlock,
    iv_kan: KANBlock,
    cv_kan: KANBlock,
    iv_df: pd.DataFrame,
    cv_df: pd.DataFrame,
    summary: dict,
    by_curve: pd.DataFrame,
    args,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    iv_df.to_csv(out_dir / "iv_measured_vs_pkan.csv", index=False, encoding="utf-8-sig")
    cv_df.to_csv(out_dir / "cv_measured_vs_pkan.csv", index=False, encoding="utf-8-sig")
    by_curve.to_csv(out_dir / "pkan_metrics_by_curve.csv", index=False, encoding="utf-8-sig")
    serializable_args = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    payload = {
        "description": "PyTorch/pykan IV-CV PKAN correction on frozen shared EKV/trap physical core.",
        "uses_pykan": PYKAN_AVAILABLE,
        "baseline_params": asdict(p),
        "baseline_trap_params": asdict(trap),
        "baseline_metrics_summary": baseline_payload.get("metrics_summary", {}),
        "pkan_metrics_summary": summary,
        "training_args": serializable_args,
        "shared_dveff_kan": shared.config(),
        "iv_transport_kan": iv_kan.config(),
        "cv_trap_kan": cv_kan.config(),
    }
    (out_dir / "pkan_model_metadata.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    torch.save(
        {
            "metadata": payload,
            "shared_state_dict": shared.state_dict(),
            "iv_kan_state_dict": iv_kan.state_dict(),
            "cv_kan_state_dict": cv_kan.state_dict(),
        },
        out_dir / "pkan_torch_model.pt",
    )
    lines = [
        "PyTorch/pykan IV-CV PKAN fit",
        "",
        "Baseline physical core: frozen EKV + shared trap model from fitted_params.json.",
        "KAN correction variables:",
        "  dVeff: shared by IV and CV",
        "  dlogID: IV transport multiplier",
        "  dlogCit_t / dlogCit_b: CV shared-trap amplitude correction",
        "",
        "Baseline metrics:",
    ]
    for k, v in baseline_payload.get("metrics_summary", {}).items():
        lines.append(f"  {k}: {v}")
    lines.append("")
    lines.append("PKAN metrics:")
    for k, v in summary.items():
        lines.append(f"  {k}: {v}")
    (out_dir / "pkan_fit_report.txt").write_text("\n".join(lines), encoding="utf-8")
    setup_plot_style()
    plot_iv_family(iv_df, "IDVBG", out_dir / "pkan_IDVBG.png")
    plot_iv_family(iv_df, "IDVTG", out_dir / "pkan_IDVTG.png")
    plot_iv_family(iv_df, "IDVD", out_dir / "pkan_IDVD.png")
    plot_cv(cv_df, out_dir / "pkan_CV.png")


def parse_args():
    ap = argparse.ArgumentParser(description="Train a PyTorch/pykan PKAN correction for shared DG TFT IV/CV model.")
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    ap.add_argument("--baseline-json", type=Path, default=DEFAULT_BASELINE_JSON)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--seed", type=int, default=20260522)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    ap.add_argument("--max-iv-points", type=int, default=12000)
    ap.add_argument("--max-cv-points", type=int, default=1800)
    ap.add_argument("--kan-hidden", type=int, default=7)
    ap.add_argument("--kan-grid", type=int, default=5)
    ap.add_argument("--kan-k", type=int, default=3)
    ap.add_argument("--kan-scale", type=float, default=0.85)
    ap.add_argument("--dveff-range", type=float, default=0.45)
    ap.add_argument("--dlogid-range", type=float, default=0.55)
    ap.add_argument("--dlogcit-range", type=float, default=0.75)
    ap.add_argument("--epochs-shared", type=int, default=1000)
    ap.add_argument("--epochs-iv", type=int, default=1200)
    ap.add_argument("--epochs-cv", type=int, default=1000)
    ap.add_argument("--warmup-epochs", type=int, default=250)
    ap.add_argument("--batch-iv", type=int, default=2048)
    ap.add_argument("--batch-cv", type=int, default=1024)
    ap.add_argument("--eval-batch", type=int, default=4096)
    ap.add_argument("--lr-shared", type=float, default=1.5e-3)
    ap.add_argument("--lr-iv", type=float, default=1.5e-3)
    ap.add_argument("--lr-cv", type=float, default=1.2e-3)
    ap.add_argument("--cv-weight-shared", type=float, default=0.45)
    ap.add_argument("--anchor-iv", type=int, default=1000)
    ap.add_argument("--anchor-cv", type=int, default=900)
    ap.add_argument("--anchor-weight", type=float, default=0.75)
    ap.add_argument("--small-weight", type=float, default=0.015)
    ap.add_argument("--param-l2", type=float, default=1e-7)
    ap.add_argument("--grad-clip", type=float, default=5.0)
    ap.add_argument("--cv-diff-h", type=float, default=1e-4)
    ap.add_argument("--print-every", type=int, default=100)
    return ap.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    print(f"Python uses pykan: {PYKAN_AVAILABLE}")
    print(f"Device: {device}, dtype: {dtype}")

    p, trap, baseline_payload = load_baseline(args.baseline_json)
    iv_all = load_iv_data(args.data_dir)
    cv_all = load_cv_data(args.data_dir)
    iv_fit = sample_iv(iv_all, args.max_iv_points, args.seed)
    cv_fit = sample_cv(cv_all, args.max_cv_points)
    cv_fit_df = cv_point_dataframe(cv_fit, trap)
    cv_all_df = cv_point_dataframe(cv_all, trap)

    print(f"Loaded IV points: {len(iv_all)} total, {len(iv_fit)} used")
    print(f"Loaded CV points: {len(cv_all_df)} total, {len(cv_fit_df)} used")

    iv_anchor_df = make_iv_anchors(iv_fit, args.anchor_iv, args.seed + 101)
    cv_anchor_df = make_cv_anchors(cv_fit_df, args.anchor_cv, args.seed + 211)

    shared, iv_kan, cv_kan = make_models(iv_fit, cv_fit_df, iv_anchor_df, cv_anchor_df, p, args, device, dtype)
    iv_fit_t = make_iv_tensors(iv_fit, device, dtype)
    cv_fit_t = make_cv_tensors(cv_fit_df, device, dtype)
    iv_anchor_t = {
        "vbg": tensor(iv_anchor_df["VBG_V"].to_numpy(float), device, dtype),
        "vtg": tensor(iv_anchor_df["VTG_V"].to_numpy(float), device, dtype),
        "vds": tensor(iv_anchor_df["VDS_V"].to_numpy(float), device, dtype),
    }
    cv_anchor_t = {
        "vbg": tensor(cv_anchor_df["VBG_V"].to_numpy(float), device, dtype),
        "vtg": tensor(cv_anchor_df["VTG_V"].to_numpy(float), device, dtype),
        "vds": tensor(cv_anchor_df["VDS_V"].to_numpy(float), device, dtype),
        "freq": tensor(cv_anchor_df["freq_hz"].to_numpy(float), device, dtype),
        "gt": tensor(cv_anchor_df["gate_t"].to_numpy(float), device, dtype),
        "gb": tensor(cv_anchor_df["gate_b"].to_numpy(float), device, dtype),
        "mode": torch.as_tensor(cv_anchor_df["mode"].to_numpy(int), dtype=torch.long, device=device),
    }

    print("Stage A: shared dVeff KAN")
    stage_shared(shared, iv_fit_t, cv_fit_t, iv_anchor_t, cv_anchor_t, p, trap, args)
    print("Stage B: IV transport dlogID KAN")
    recompute_iv_normalizer(iv_kan, iv_fit_t, p, shared, args)
    stage_iv(shared, iv_kan, iv_fit_t, iv_anchor_t, p, args)
    print("Stage C: CV trap dlogCit KAN")
    stage_cv(shared, cv_kan, cv_fit_t, cv_anchor_t, p, trap, args)

    print("Evaluating all data")
    iv_all_t = make_iv_tensors(iv_all, device, dtype)
    cv_all_t = make_cv_tensors(cv_all_df, device, dtype)
    pred_id, dveff_iv, dlogid = eval_iv(iv_all_t, p, shared, iv_kan, args)
    pred_c, dveff_cv, dlogt, dlogb = eval_cv(cv_all_t, p, trap, shared, cv_kan, args)

    iv_pred = iv_all.copy()
    iv_pred["ID_pkan_A"] = pred_id
    iv_pred["dveff_KAN_V"] = dveff_iv
    iv_pred["dlogid_KAN_dec"] = dlogid
    iv_pred["log10_abs_error_dec"] = np.log10(np.maximum(np.abs(pred_id), ID_FLOOR)) - np.log10(
        np.maximum(np.abs(iv_pred["ID_A"].to_numpy(float)), ID_FLOOR)
    )
    cv_pred = cv_all_df.copy()
    cv_pred["cap_meas_F"] = cv_pred["cap_F"]
    cv_pred["cap_pkan_F"] = pred_c
    cv_pred["dveff_KAN_V"] = dveff_cv
    cv_pred["dlogCit_t_KAN_dec"] = dlogt
    cv_pred["dlogCit_b_KAN_dec"] = dlogb
    cv_pred["log10_cap_error_dec"] = np.log10(np.maximum(pred_c, C_FLOOR)) - np.log10(
        np.maximum(cv_pred["cap_F"].to_numpy(float), C_FLOOR)
    )

    summary, by_curve = metrics(iv_pred, cv_pred)
    write_outputs(
        args.out_dir,
        p,
        trap,
        baseline_payload,
        shared,
        iv_kan,
        cv_kan,
        iv_pred,
        cv_pred,
        summary,
        by_curve,
        args,
    )
    print(json.dumps(summary, indent=2))
    print(f"Outputs written to: {args.out_dir}")


if __name__ == "__main__":
    main()
