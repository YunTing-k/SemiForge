"""Prepared complete sweeps and batched, curve-balanced IV constraints.

Curve plans retain only measurement-derived tensors. The independent residual
stage may additionally retain frozen baseline outputs in its own local cache;
trainable predictions and their autograd graphs are always fresh.
"""
from __future__ import annotations

import torch
from contextlib import contextmanager
from torch.nn import functional as F


class PreparedIVBatch(dict):
    """Tensor dictionary with private sweep metadata (never a data column)."""

    def __init__(self, data, layout, source_indices=None):
        super().__init__(data)
        self.layout = layout
        self.source_indices = source_indices


class IVCurveLayout:
    """Padded sorted indices; masks preserve equal weight per eligible curve."""

    @torch.no_grad()
    def __init__(self, data, smooth_passes, current_floor):
        self.smooth_passes = smooth_passes
        codes, inverse = torch.unique(data["curve_code"], sorted=True, return_inverse=True)
        self.codes, self.inverse = codes, inverse
        # One host transfer prepares ragged indexing once, instead of scalar
        # GPU reads, boolean indexing and sorting for every curve every epoch.
        buckets = [[] for _ in range(len(codes))]
        for point, curve in enumerate(inverse.cpu().tolist()):
            buckets[curve].append(point)
        device = data["vbg"].device
        self.lengths = torch.tensor([len(row) for row in buckets], device=device)
        width = max(map(len, buckets), default=0)
        indices, families = [], []
        for row in buckets:
            idx = torch.tensor(row, device=device)
            family = int(data["family"][idx[0]]) if "family" in data else 0
            voltage = data["vds"] if family == 2 else data["vtg"] if family == 1 else data["vbg"]
            idx = idx[torch.argsort(voltage[idx])]
            indices.append(torch.cat([idx, idx[-1:].expand(width - len(idx))]))
            families.append(family)
        self.index = torch.stack(indices) if indices else torch.empty((0, 0), device=device, dtype=torch.long)
        self.family = torch.tensor(families, device=device, dtype=torch.long)
        self.x = torch.where(
            self.family[:, None] == 2, data["vds"][self.index],
            torch.where(self.family[:, None] == 1, data["vtg"][self.index], data["vbg"][self.index]),
        )
        self.target = torch.log10(data["id"].abs().clamp_min(current_floor))[self.index]
        if smooth_passes > 0:
            kernel = self.target.new_tensor([1., 2., 3., 2., 1.]) / 9.
            # Smooth equal-length sweeps together, retaining reflect boundaries
            # at each real sweep end rather than at the padded tensor boundary.
            for length in sorted(set(map(len, buckets))):
                if length < 5:
                    continue
                rows = self.lengths == length
                values = self.target[rows, :length, None].transpose(1, 2)
                for _ in range(smooth_passes):
                    values = F.conv1d(F.pad(values, (2, 2), mode="reflect"), kernel[None, None, :])
                self.target[rows, :length] = values[:, 0]
        self.dx = (self.x[:, 1:] - self.x[:, :-1]).clamp_min(1.e-3)
        self.target_slope = (self.target[:, 1:] - self.target[:, :-1]) / self.dx
        self.target_curv = self.target[:, 2:] - 2. * self.target[:, 1:-1] + self.target[:, :-2]
        self.pairs = torch.arange(max(width - 1, 0), device=device)[None, :] < self.lengths[:, None] - 1
        self.mids = torch.arange(max(width - 2, 0), device=device)[None, :] < self.lengths[:, None] - 2
        self.eligible = self.lengths >= 8

    def sample(self, data, n_curves):
        if len(self.codes) <= n_curves:
            return PreparedIVBatch(data, self)
        # Same randperm and original point order as curve_batch. Sorting only
        # the chosen curve IDs also preserves the original loss reduction order.
        pick = torch.randperm(len(self.codes), device=self.codes.device)[:n_curves].sort().values
        chosen = torch.zeros(len(self.codes), device=self.codes.device, dtype=torch.bool)
        chosen[pick] = True
        rows = torch.nonzero(chosen[self.inverse], as_tuple=True)[0]
        subset = object.__new__(IVCurveLayout)
        subset.smooth_passes = self.smooth_passes
        for name in ("codes", "lengths", "family", "x", "target", "dx", "target_slope", "target_curv", "pairs", "mids", "eligible"):
            setattr(subset, name, getattr(self, name)[pick])
        subset.index = torch.searchsorted(rows, self.index[pick])
        # Sampled layouts are used for loss evaluation, never recursively sampled.
        subset.inverse = None
        return PreparedIVBatch({key: value[rows] for key, value in data.items()}, subset, rows)


@contextmanager
def residual_only_gradients(diag, params):
    """Freeze unused parameters for this stage; restore flags even on failure."""
    active = {id(p) for p in params}
    flags = [(p, p.requires_grad) for p in diag.parameters()]
    try:
        for p, _ in flags:
            if id(p) not in active:
                p.requires_grad_(False)
        yield
    finally:
        for p, enabled in flags:
            p.requires_grad_(enabled)


class ResidualBaselineCache:
    """Stage-local, detached baselines for fixed measured terminal conditions.

    Used only while the optimizer updates iv_resid. This is not a general model
    cache: temperature/physical parameters, data and preprocessing must stay
    fixed, and input-voltage autograd is deliberately unsupported.
    """

    @torch.no_grad()
    def __init__(self, diag, data, layout, chunk_size=4096):
        active = {id(p) for p in diag.iv_resid.parameters()}
        if any(p.requires_grad for p in diag.parameters() if id(p) not in active):
            raise ValueError("Residual baselines require frozen non-residual parameters")
        if any(v.requires_grad for v in data.values() if torch.is_tensor(v)):
            raise ValueError("Residual baseline cache does not support differentiable inputs")
        if chunk_size <= 0 or not len(data["vbg"]):
            raise ValueError("Residual baseline cache requires data and a positive chunk size")
        self.diag, self.data, self.layout = diag, data, layout
        base, physical = [], []
        # Bound the preparation peak for larger measured data sets. Preserve
        # original point order so both point and full-sweep sampling are exact.
        for start in range(0, len(data["vbg"]), chunk_size):
            subset = {key: value[start:start + chunk_size] for key, value in data.items()}
            base.append(diag.base_ids(subset).detach())
            physical.append(diag.physical_ids(subset).detach())
        self.base = torch.cat(base)
        self.physical = torch.cat(physical)

    def attach(self, batch, indices):
        batch.residual_baseline_owner = self.diag
        batch.residual_baselines = (
            self.base if indices is None else self.base[indices],
            self.physical if indices is None else self.physical[indices],
        )
        return batch

    def point_batch(self, n):
        total = len(self.data["vbg"])
        idx = torch.randint(total, (min(n, total),), device=self.data["vbg"].device)
        batch = PreparedIVBatch({key: value[idx] for key, value in self.data.items()}, None, idx)
        return self.attach(batch, idx)

    def curve_batch(self, n):
        batch = self.layout.sample(self.data, n)
        return self.attach(batch, batch.source_indices)


def masked_mean(values, mask):
    """One mean per curve; an empty region contributes zero, as before."""
    return torch.where(mask, values, 0.).sum(dim=1) / mask.sum(dim=1).clamp_min(1)


def huber_zero(value, beta):
    return F.smooth_l1_loss(value, torch.zeros_like(value), beta=beta, reduction="none")


def curve_constraint_losses(layout, logp, logb, logphys, args, *, derivative=True, shape=True):
    """Preserve the original derivative/base-detach and shape/base-gradient rules."""
    p = layout
    y, base = logp[p.index], logb[p.index]
    slope = (y[:, 1:] - y[:, :-1]) / p.dx
    curv = y[:, 2:] - 2. * y[:, 1:-1] + y[:, :-2]
    pair_target = .5 * (p.target[:, 1:] + p.target[:, :-1])
    is_output = p.family[:, None] == 2

    def residual_terms(base_value):
        r = y - base_value
        return ((r[:, 1:] - r[:, :-1]) / p.dx,
                r[:, 2:] - 2. * r[:, 1:-1] + r[:, :-2])

    def finish(per_curve):
        # Short sweeps must not dilute the mean of eligible complete sweeps.
        return torch.where(p.eligible, per_curve, 0.).sum() / p.eligible.sum().clamp_min(1)

    derivative_loss = shape_loss = logp.new_zeros(())
    if derivative:
        r_slope, r_curv = residual_terms(base.detach())
        active = p.pairs & (pair_target > args.iv_deriv_log_floor)
        target = p.target_slope.clamp(max=args.iv_deriv_slope_clip)
        target = torch.maximum(target, torch.where(is_output, -args.iv_deriv_slope_clip, 0.))
        derivative_loss = finish(
            args.iv_deriv_slope_weight * masked_mean(huber_zero(slope - target, .20), active)
            + args.iv_deriv_mono_weight * masked_mean(torch.relu(-slope - args.iv_shape_negative_tol).square(), active & ~is_output)
            + args.iv_deriv_curv_weight * masked_mean(huber_zero(curv - p.target_curv.clamp(-args.iv_deriv_curv_clip, args.iv_deriv_curv_clip), .050), p.mids)
            + args.iv_residual_curv_weight * masked_mean(huber_zero(r_curv, .025), p.mids)
            + args.iv_residual_slope_weight * masked_mean(torch.relu(r_slope.abs() - args.iv_residual_slope_tol).square(), p.pairs)
        )
    if shape:
        r_slope, r_curv = residual_terms(base)
        rb = base - logphys[p.index].detach()
        rb_curv = rb[:, 2:] - 2. * rb[:, 1:-1] + rb[:, :-2]
        active = p.pairs & (pair_target > args.iv_shape_active_log_floor)
        target = p.target_slope.clamp(max=args.iv_slope_target_max)
        target = torch.maximum(target, torch.where(is_output, -args.iv_slope_target_max, 0.))
        on_pair = p.pairs & (p.family[:, None] == 1) & (p.x[:, 1:] >= args.idvtg_on_x_min) & (p.x[:, 1:] <= args.idvtg_on_x_max) & (pair_target > -9.2)
        on_mid = p.mids & (p.family[:, None] == 1) & (p.x[:, 1:-1] >= args.idvtg_on_x_min) & (p.x[:, 1:-1] <= args.idvtg_on_x_max) & (p.target[:, 1:-1] > -9.2)
        sat_pair = p.pairs & is_output & (p.x[:, 1:] >= args.idvd_smooth_vds_min)
        sat_mid = p.mids & is_output & (p.x[:, 1:-1] >= args.idvd_smooth_vds_min)
        # Sorted voltage makes the saturation region contiguous. Require three
        # selected slopes, exactly as the scalar implementation does.
        sat_tv = sat_pair[:, 1:] & sat_pair[:, :-1] & (sat_pair.sum(dim=1, keepdim=True) >= 3)
        shape_loss = finish(
            args.iv_shape_negative_weight * masked_mean(torch.relu(-slope - args.iv_shape_negative_tol).square(), active & ~is_output)
            + args.iv_slope_match_weight * masked_mean(huber_zero(slope - target, .22), active)
            + args.iv_residual_curv_weight * masked_mean(huber_zero(r_curv, .030), p.mids)
            + args.iv_residual_slope_weight * masked_mean(torch.relu(r_slope.abs() - args.iv_residual_slope_tol).square(), p.pairs)
            + args.idvtg_on_slope_match_weight * masked_mean(huber_zero(slope - p.target_slope.clamp(0., args.idvtg_on_slope_target_max), .18), on_pair)
            + args.idvtg_on_curv_weight * masked_mean(torch.relu(curv.abs() - args.idvtg_on_curv_tol).square(), on_mid)
            + args.idvd_slope_tv_weight * masked_mean(huber_zero(slope[:, 1:] - slope[:, :-1], args.idvd_slope_tv_tol), sat_tv)
            + args.idvd_curv_weight * masked_mean(torch.relu(r_curv.abs() - args.idvd_curv_tol).square(), sat_mid)
            + args.idvd_base_pkan_curv_weight * masked_mean(torch.relu(rb_curv.abs() - args.idvd_base_pkan_curv_tol).square(), sat_mid)
        )
    return derivative_loss, shape_loss
