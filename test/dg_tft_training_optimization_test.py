"""Numerical contracts for batched fitting, priors and complete-sweep losses.

Scalar reference formulas below are test-only oracles from the pre-optimization
engine. Keeping independent reductions catches padding, curve weighting,
detachment and candidate-order regressions; production has only the new path.
"""
from __future__ import annotations
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

try:
    import torch
    from torch import nn
    from torch.nn import functional as F
except ImportError:
    raise unittest.SkipTest("Requires the modeling environment (PyTorch)")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src/device_modeling/dg_tft/engine"))
import dg_tft_full_model as engine
from iv_curve_optimization import IVCurveLayout, PreparedIVBatch, ResidualBaselineCache, residual_only_gradients
ID_FLOOR = engine.ID_FLOOR
_reflect_smooth_1d = engine._reflect_smooth_1d
symbolic_func = engine.symbolic_func
CompositeSymbolicEdge = engine.CompositeSymbolicEdge
PYKAN_AVAILABLE = True
_top1_symbolic_library_bank = engine._top1_symbolic_library_bank

def reference_fit_top1_symbolic_edge_exhaustive(
    target: torch.Tensor,
    x: torch.Tensor,
    names: list[str],
    finetune_steps: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[CompositeSymbolicEdge, float]:
    """Select the best one-function symbolic edge before nonlinear polishing.

    A mixture fit followed by coefficient pruning is a weak top-1 procedure:
    correlated basis functions can split one spline edge, and the largest
    coefficient is not necessarily the best standalone function.  For the
    deployable top-1 path we instead enumerate a small affine grid for each
    safe symbolic primitive, solve its amplitude/bias by least squares, keep
    the globally best candidate, and only then fine-tune its affine arguments.
    """

    best = None
    # Top-1 means one primitive per spline edge, not one fixed argument.
    # Enumerating f(a*x+b) before polishing is much closer to how compact
    # models use analytic functions and greatly improves top-1 precision.
    b_grid = [0.25, 0.50, 0.75, 1.0, 1.5, 2.0, 3.0, -0.25, -0.50, -0.75, -1.0, -1.5, -2.0, -3.0]
    c_grid = [-2.0, -1.0, -0.50, 0.0, 0.50, 1.0, 2.0]
    with torch.no_grad():
        y = target.reshape(-1)
        y_mean = torch.mean(y)
        y_center = y - y_mean
        for idx, name in enumerate(names):
            affine_grid = [(1.0, 0.0)] if name == "const" else [(b, c) for b in b_grid for c in c_grid]
            for b0, c0 in affine_grid:
                phi = symbolic_func(name, x * b0 + c0).reshape(-1)
                phi_mean = torch.mean(phi)
                phi_center = phi - phi_mean
                var = torch.mean(phi_center * phi_center)
                if float(var.detach().cpu()) < 1.0e-18:
                    a = phi.new_tensor(0.0)
                else:
                    a = torch.mean(phi_center * y_center) / var
                bias = y_mean - a * phi_mean
                pred = a * phi + bias
                mse = float(torch.mean((pred - y) ** 2).detach().cpu())
                if best is None or mse < best[0]:
                    best = (mse, idx, float(a.detach().cpu()), b0, c0, float(bias.detach().cpu()))
    assert best is not None
    _, keep_idx, a0, b0, c0, bias0 = best
    edge = CompositeSymbolicEdge(names, device, dtype)
    with torch.no_grad():
        edge.a.zero_()
        edge.a[keep_idx] = edge.a.new_tensor(a0)
        edge.b[keep_idx] = edge.b.new_tensor(b0)
        edge.c[keep_idx] = edge.c.new_tensor(c0)
        edge.bias.copy_(edge.bias.new_tensor(bias0))
        edge.active_indices = [keep_idx]
    opt = torch.optim.Adam([edge.a, edge.b, edge.c, edge.bias], lr=0.012)
    for _ in range(max(0, int(finetune_steps))):
        opt.zero_grad(set_to_none=True)
        loss = torch.mean((edge(x) - target) ** 2)
        loss.backward()
        if edge.a.grad is not None:
            with torch.no_grad():
                inactive = torch.ones_like(edge.a, dtype=torch.bool)
                inactive[keep_idx] = False
                edge.a.grad[inactive] = 0.0
                if edge.b.grad is not None:
                    edge.b.grad[inactive] = 0.0
                if edge.c.grad is not None:
                    edge.c.grad[inactive] = 0.0
        opt.step()
    with torch.no_grad():
        final = torch.mean((edge(x) - target) ** 2).detach()
    return edge, float(final.cpu())


def reference_kan_top1_symbolic_compatibility_loss(kan: CompactKAN, args) -> torch.Tensor:
    """Pull spline edges toward one safe analytic primitive during training.

    This is deliberately a soft prior.  The spline KAN remains free to fit the
    device, but edges that can explain the data equally well with a top-1
    primitive are preferred before the deployment-time symbolic replacement.
    """

    if not PYKAN_AVAILABLE or kan is None or not hasattr(kan, "net") or not hasattr(kan.net, "act_fun"):
        return torch.as_tensor(0.0, device=kan.center.device, dtype=kan.center.dtype)
    n = max(25, int(getattr(args, "symbolic_training_grid_points", 49)))
    x = torch.linspace(-2.5, 2.5, n, device=kan.center.device, dtype=kan.center.dtype)
    bank = _top1_symbolic_library_bank(x)
    pieces = []
    weights = []
    for layer in kan.net.act_fun:
        in_dim = int(getattr(layer, "in_dim", 0))
        out_dim = int(getattr(layer, "out_dim", 0))
        for i in range(in_dim):
            inp = torch.zeros(n, in_dim, device=x.device, dtype=x.dtype)
            inp[:, i] = x
            y = layer(inp)
            y0 = layer(torch.zeros_like(inp))
            if isinstance(y, tuple):
                y = y[0]
            if isinstance(y0, tuple):
                y0 = y0[0]
            for j in range(out_dim):
                edge = y[:, j] - y0[:, j]
                std = torch.std(edge)
                if float(std.detach().cpu()) < 1.0e-6:
                    continue
                edge = (edge - torch.mean(edge)) / torch.clamp(std, min=edge.new_tensor(1.0e-6))
                dist = torch.mean((bank - edge.reshape(1, -1)) ** 2, dim=1)
                # Weak edges should not dominate the compatibility score.  A
                # detached energy weight keeps this a shape prior rather than
                # an accidental amplitude penalty.
                weight = torch.clamp(std.detach(), min=std.new_tensor(1.0e-4), max=std.new_tensor(10.0))
                pieces.append(weight * torch.min(dist))
                weights.append(weight)
    if not pieces:
        return torch.as_tensor(0.0, device=kan.center.device, dtype=kan.center.dtype)
    return torch.stack(pieces).sum() / torch.clamp(torch.stack(weights).sum(), min=kan.center.new_tensor(1.0e-8))


def reference_kan_edge_group_sparsity_loss(kan: CompactKAN, args) -> torch.Tensor:
    """Group sparsity on spline-edge variation for compact top-1 deployment."""

    if not PYKAN_AVAILABLE or kan is None or not hasattr(kan, "net") or not hasattr(kan.net, "act_fun"):
        return torch.as_tensor(0.0, device=kan.center.device, dtype=kan.center.dtype)
    n = max(25, int(getattr(args, "symbolic_training_grid_points", 49)))
    x = torch.linspace(-2.5, 2.5, n, device=kan.center.device, dtype=kan.center.dtype)
    energies = []
    for layer in kan.net.act_fun:
        in_dim = int(getattr(layer, "in_dim", 0))
        out_dim = int(getattr(layer, "out_dim", 0))
        for i in range(in_dim):
            inp = torch.zeros(n, in_dim, device=x.device, dtype=x.dtype)
            inp[:, i] = x
            y = layer(inp)
            y0 = layer(torch.zeros_like(inp))
            if isinstance(y, tuple):
                y = y[0]
            if isinstance(y0, tuple):
                y0 = y0[0]
            for j in range(out_dim):
                edge = y[:, j] - y0[:, j]
                edge = edge - torch.mean(edge)
                energies.append(torch.sqrt(torch.mean(edge * edge) + edge.new_tensor(1.0e-10)))
    if not energies:
        return torch.as_tensor(0.0, device=kan.center.device, dtype=kan.center.dtype)
    return torch.stack(energies).mean()


def reference_iv_ordered_shape_loss(diag: ResidualDiagnostic, data: dict[str, torch.Tensor], args) -> torch.Tensor:
    """Physics-informed IV residual shape regularizer.

    The residual is allowed to improve gm and the on-current, but it should not
    create high-frequency curvature.  Penalizing the second difference of the
    *residual* preserves the compact kernel's natural EKV-like S-shape while
    still allowing broad bias-dependent corrections.
    """

    if "curve_code" not in data or len(data["vbg"]) < 8:
        return torch.as_tensor(0.0, device=data["vbg"].device, dtype=data["vbg"].dtype)
    pred = diag.ids_residual(data, 1.0)
    # Keep the base PKAN graph alive here.  IDVD saturation ripple is often
    # introduced by the parameter PKAN itself, before the external residual is
    # applied.  Detaching base_ids made the dedicated rb_curv penalty a
    # diagnostic-only number with no training effect.
    base = diag.base_ids(data)
    physical = diag.physical_ids(data).detach()
    logp = torch.log10(torch.clamp(torch.abs(pred), min=ID_FLOOR))
    logb = torch.log10(torch.clamp(torch.abs(base), min=ID_FLOOR))
    logphys = torch.log10(torch.clamp(torch.abs(physical), min=ID_FLOOR))
    logt = torch.log10(torch.clamp(torch.abs(data["id"]), min=ID_FLOOR))
    residual = logp - logb
    pieces = []
    for code in torch.unique(data["curve_code"]):
        mask = data["curve_code"] == code
        n = int(mask.sum().detach().cpu())
        if n < 8:
            continue
        fam = int(data["family"][mask][0].detach().cpu()) if "family" in data else 0
        x_all = data["vds"][mask] if fam == 2 else (data["vtg"][mask] if fam == 1 else data["vbg"][mask])
        idx = torch.argsort(x_all)
        x = x_all[idx]
        y = logp[mask][idx]
        yt_raw = logt[mask][idx]
        yt = _reflect_smooth_1d(yt_raw, args.iv_deriv_target_smooth_passes) if args.iv_deriv_target_smooth_passes > 0 else yt_raw
        r = residual[mask][idx]
        rb = (logb - logphys)[mask][idx]
        dx = torch.clamp(x[1:] - x[:-1], min=x.new_tensor(1.0e-3))
        slope = (y[1:] - y[:-1]) / dx
        slope_t = (yt[1:] - yt[:-1]) / dx
        r_slope = (r[1:] - r[:-1]) / dx
        r_curv = r[2:] - 2.0 * r[1:-1] + r[:-2]
        rb_curv = rb[2:] - 2.0 * rb[1:-1] + rb[:-2]
        y_curv = y[2:] - 2.0 * y[1:-1] + y[:-2]

        active_pair = 0.5 * (yt[1:] + yt[:-1]) > args.iv_shape_active_log_floor
        neg = torch.as_tensor(0.0, device=y.device, dtype=y.dtype)
        if bool(active_pair.any().detach().cpu()) and fam != 2:
            neg = torch.mean(torch.relu(-slope[active_pair] - args.iv_shape_negative_tol) ** 2)
        slope_match = torch.as_tensor(0.0, device=y.device, dtype=y.dtype)
        if bool(active_pair.any().detach().cpu()):
            tgt = torch.clamp(
                slope_t[active_pair],
                min=-args.iv_slope_target_max if fam == 2 else 0.0,
                max=args.iv_slope_target_max,
            )
            serr = slope[active_pair] - tgt
            slope_match = torch.mean(torch.nn.functional.smooth_l1_loss(serr, torch.zeros_like(serr), beta=0.22, reduction="none"))

        r_curv_loss = torch.mean(torch.nn.functional.smooth_l1_loss(r_curv, torch.zeros_like(r_curv), beta=0.030, reduction="none"))
        r_slope_loss = torch.mean(torch.relu(torch.abs(r_slope) - args.iv_residual_slope_tol) ** 2)

        on_shape = torch.as_tensor(0.0, device=y.device, dtype=y.dtype)
        if fam == 1:
            on_pair = (x[1:] >= args.idvtg_on_x_min) & (x[1:] <= args.idvtg_on_x_max) & (0.5 * (yt[1:] + yt[:-1]) > -9.2)
            if bool(on_pair.any().detach().cpu()):
                slope_err = slope[on_pair] - torch.clamp(slope_t[on_pair], min=0.0, max=args.idvtg_on_slope_target_max)
                on_shape = on_shape + args.idvtg_on_slope_match_weight * torch.mean(
                    torch.nn.functional.smooth_l1_loss(slope_err, torch.zeros_like(slope_err), beta=0.18, reduction="none")
                )
            on_mid = (x[1:-1] >= args.idvtg_on_x_min) & (x[1:-1] <= args.idvtg_on_x_max) & (yt[1:-1] > -9.2)
            if bool(on_mid.any().detach().cpu()):
                on_shape = on_shape + args.idvtg_on_curv_weight * torch.mean(
                    torch.relu(torch.abs(y_curv[on_mid]) - args.idvtg_on_curv_tol) ** 2
                )
        elif fam == 2:
            # IDVD may have a small measured roll-off in saturation, so do not
            # force gds positive.  It should, however, remain smooth after the
            # low-VDS knee.  This suppresses residual PKAN ripples without
            # flattening the physical output-conductance trend.
            sat_pair = x[1:] >= args.idvd_smooth_vds_min
            sat_mid = x[1:-1] >= args.idvd_smooth_vds_min
            if bool(sat_pair.any().detach().cpu()) and int(sat_pair.sum().detach().cpu()) >= 3:
                sat_slope = slope[sat_pair]
                on_shape = on_shape + args.idvd_slope_tv_weight * torch.mean(
                    F.smooth_l1_loss(
                        sat_slope[1:] - sat_slope[:-1],
                        torch.zeros_like(sat_slope[1:]),
                        beta=args.idvd_slope_tv_tol,
                        reduction="none",
                    )
                )
            if bool(sat_mid.any().detach().cpu()):
                on_shape = on_shape + args.idvd_curv_weight * torch.mean(
                    torch.relu(torch.abs(r_curv[sat_mid]) - args.idvd_curv_tol) ** 2
                )
                # The pretrained parameter-PKAN can itself introduce a shallow
                # saturation ripple before the external residual is applied.
                # Constrain only that added curvature, not the physical CLM
                # trend, so a smooth nonzero output conductance remains legal.
                on_shape = on_shape + args.idvd_base_pkan_curv_weight * torch.mean(
                    torch.relu(torch.abs(rb_curv[sat_mid]) - args.idvd_base_pkan_curv_tol) ** 2
                )

        pieces.append(
            args.iv_shape_negative_weight * neg
            + args.iv_slope_match_weight * slope_match
            + args.iv_residual_curv_weight * r_curv_loss
            + args.iv_residual_slope_weight * r_slope_loss
            + on_shape
        )
    if not pieces:
        return torch.as_tensor(0.0, device=data["vbg"].device, dtype=data["vbg"].dtype)
    return torch.stack(pieces).mean()


def reference_iv_derivative_consistency_loss(diag: ResidualDiagnostic, data: dict[str, torch.Tensor], args) -> torch.Tensor:
    """Main training loss on gm/gds-like derivatives.

    Literature-style physics-guided NN compact modeling usually evaluates not
    only I-V but also transconductance/output conductance.  This discrete
    derivative loss does that directly on full measured sweeps.
    """

    if "curve_code" not in data or len(data["vbg"]) < 8:
        return torch.as_tensor(0.0, device=data["vbg"].device, dtype=data["vbg"].dtype)
    pred = diag.ids_residual(data, 1.0)
    base = diag.base_ids(data).detach()
    logp = torch.log10(torch.clamp(torch.abs(pred), min=ID_FLOOR))
    logb = torch.log10(torch.clamp(torch.abs(base), min=ID_FLOOR))
    logt = torch.log10(torch.clamp(torch.abs(data["id"]), min=ID_FLOOR))
    pieces = []
    for code in torch.unique(data["curve_code"]):
        mask = data["curve_code"] == code
        if int(mask.sum().detach().cpu()) < 8:
            continue
        fam = int(data["family"][mask][0].detach().cpu()) if "family" in data else 0
        x_all = data["vds"][mask] if fam == 2 else (data["vtg"][mask] if fam == 1 else data["vbg"][mask])
        idx = torch.argsort(x_all)
        x = x_all[idx]
        yp = logp[mask][idx]
        yb = logb[mask][idx]
        yt_raw = logt[mask][idx]
        yt = _reflect_smooth_1d(yt_raw, args.iv_deriv_target_smooth_passes) if args.iv_deriv_target_smooth_passes > 0 else yt_raw
        dx = torch.clamp(x[1:] - x[:-1], min=x.new_tensor(1.0e-3))
        sp = (yp[1:] - yp[:-1]) / dx
        st = (yt[1:] - yt[:-1]) / dx
        rb = yp - yb
        rb_slope = (rb[1:] - rb[:-1]) / dx
        rb_curv = rb[2:] - 2.0 * rb[1:-1] + rb[:-2]
        yp_curv = yp[2:] - 2.0 * yp[1:-1] + yp[:-2]
        yt_curv = yt[2:] - 2.0 * yt[1:-1] + yt[:-2]

        active = 0.5 * (yt[1:] + yt[:-1]) > args.iv_deriv_log_floor
        if bool(active.any().detach().cpu()):
            st_clip = torch.clamp(st[active], min=0.0 if fam != 2 else -args.iv_deriv_slope_clip, max=args.iv_deriv_slope_clip)
            slope_loss = torch.mean(
                torch.nn.functional.smooth_l1_loss(sp[active] - st_clip, torch.zeros_like(st_clip), beta=0.20, reduction="none")
            )
            mono_loss = (
                torch.as_tensor(0.0, device=yp.device, dtype=yp.dtype)
                if fam == 2
                else torch.mean(torch.relu(-sp[active] - args.iv_shape_negative_tol) ** 2)
            )
        else:
            slope_loss = torch.as_tensor(0.0, device=yp.device, dtype=yp.dtype)
            mono_loss = torch.as_tensor(0.0, device=yp.device, dtype=yp.dtype)

        curv_target = torch.clamp(yt_curv, -args.iv_deriv_curv_clip, args.iv_deriv_curv_clip)
        curv_loss = torch.mean(
            torch.nn.functional.smooth_l1_loss(yp_curv - curv_target, torch.zeros_like(curv_target), beta=0.050, reduction="none")
        )
        residual_curv = torch.mean(
            torch.nn.functional.smooth_l1_loss(rb_curv, torch.zeros_like(rb_curv), beta=0.025, reduction="none")
        )
        residual_slope = torch.mean(torch.relu(torch.abs(rb_slope) - args.iv_residual_slope_tol) ** 2)
        pieces.append(
            args.iv_deriv_slope_weight * slope_loss
            + args.iv_deriv_mono_weight * mono_loss
            + args.iv_deriv_curv_weight * curv_loss
            + args.iv_residual_curv_weight * residual_curv
            + args.iv_residual_slope_weight * residual_slope
        )
    if not pieces:
        return torch.as_tensor(0.0, device=data["vbg"].device, dtype=data["vbg"].dtype)
    return torch.stack(pieces).mean()


class AnalyticDevice(nn.Module):
    def __init__(self, device, dtype):
        super().__init__()
        self.base = nn.Parameter(torch.tensor(.08, device=device, dtype=dtype))
        self.residual = nn.Parameter(torch.tensor(.12, device=device, dtype=dtype))

    def base_ids(self, data):
        x = data["vbg"] + .3 * data["vtg"] + .2 * data["vds"]
        return torch.sign(data["vds"]) * torch.pow(10., -11. + .4*x + self.base*torch.sin(x))

    def physical_ids(self, data):
        return torch.sign(data["vds"]) * torch.pow(10., -11. + .4 * (data["vbg"] + .3*data["vtg"] + .2*data["vds"]))

    def ids_residual(self, data, scale=1.):
        return self.base_ids(data) * torch.pow(10., self.residual * torch.sin(2.1*data["vbg"] + data["vds"]))


class CacheTestDevice(AnalyticDevice):
    """Independent temperature-dependent baseline and a trainable correction."""

    def __init__(self, device, dtype):
        super().__init__(device, dtype)
        del self.residual
        self.iv_resid = nn.Linear(2, 1, device=device, dtype=dtype)
        with torch.no_grad():
            self.iv_resid.weight.fill_(.03)
            self.iv_resid.bias.fill_(.01)

    def base_ids(self, data):
        temp = torch.nan_to_num(data.get("temp_K", torch.full_like(data["vbg"], 300.)), nan=300.)
        return super().base_ids(data) * torch.exp((temp - 300.) * self.base * .01)

    def logid_residual(self, data, scale=1.):
        x = torch.stack([data["vbg"], data["vtg"] + data["vds"]], dim=1)
        return self.iv_resid(x)[:, 0] * scale

    ids_residual = engine.ResidualDiagnostic.ids_residual


class TestTrainingOptimization(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        with patch.object(sys, "argv", ["test"]):
            cls.args = engine.parse_args()

    def devices(self):
        # CPU-only mode allows independent tests while a GPU timing run is active.
        import os
        return ["cpu", "cuda"] if torch.cuda.is_available() and not os.environ.get("DEVICEAGENT_TEST_CPU_ONLY") else ["cpu"]

    def data(self, device, dtype, lengths=(3, 7, 8, 9, 17, 25)):
        rows = []
        for curve, length in enumerate(lengths):
            x = torch.linspace(-2., 7., length, device=device, dtype=dtype)
            if length > 8:
                x[3] = x[2]  # Repeated bias must retain the original dx floor.
            target = torch.pow(10., -12. + .7*x + .03*torch.sin(1.3*x))
            rows.append({"vbg": x, "vtg": x+.1, "vds": x,
                         "id": target, "curve_code": torch.full_like(x, curve*3).long(),
                         "family": torch.full_like(x, curve % 3).long()})
        result = {key: torch.cat([row[key] for row in rows]) for key in rows[0]}
        torch.manual_seed(31)
        order = torch.randperm(len(result["vbg"]), device=device)
        return {key: value[order] for key, value in result.items()}

    def test_curve_losses_and_gradients_ragged_sweeps(self):
        for device in self.devices():
            for dtype in (torch.float32, torch.float64):
                for passes in (0, 2):
                    with self.subTest(device=device, dtype=dtype, passes=passes):
                        data = self.data(device, dtype);diag = AnalyticDevice(device, dtype)
                        self.args.iv_deriv_target_smooth_passes = passes
                        layout = IVCurveLayout(data, passes, ID_FLOOR)
                        prepared = PreparedIVBatch(data, layout)
                        for old, new in [(reference_iv_derivative_consistency_loss, engine.iv_derivative_consistency_loss),
                                         (reference_iv_ordered_shape_loss, engine.iv_ordered_shape_loss)]:
                            a, b = old(diag, data, self.args), new(diag, prepared, self.args)
                            tol = 2e-4 if dtype == torch.float32 else 1e-10
                            torch.testing.assert_close(a, b, atol=tol, rtol=tol)
                            ga = torch.autograd.grad(a, tuple(diag.parameters()))
                            gb = torch.autograd.grad(b, tuple(diag.parameters()))
                            for x, y in zip(ga, gb):
                                torch.testing.assert_close(x, y, atol=tol, rtol=tol)
                        expected = reference_iv_derivative_consistency_loss(diag, data, self.args) + reference_iv_ordered_shape_loss(diag, data, self.args)
                        actual = sum(engine._iv_curve_constraint_losses(diag, prepared, self.args))
                        torch.testing.assert_close(expected, actual, atol=tol, rtol=tol)

    def test_sampling_preserves_rng_point_order_and_loss(self):
        for device in self.devices():
            data = self.data(device, torch.float64);diag = AnalyticDevice(device, torch.float64)
            plan = IVCurveLayout(data, self.args.iv_deriv_target_smooth_passes, ID_FLOOR)
            for count in (1, 3, 6, 20):
                torch.manual_seed(54);old = engine.curve_batch(data, count);after_old = torch.rand(5, device=device)
                torch.manual_seed(54);new = plan.sample(data, count);after_new = torch.rand(5, device=device)
                torch.testing.assert_close(after_old, after_new, rtol=0, atol=0)
                for key in data:
                    torch.testing.assert_close(old[key], new[key], rtol=0, atol=0)
                a = reference_iv_ordered_shape_loss(diag, old, self.args)
                b = engine.iv_ordered_shape_loss(diag, new, self.args)
                torch.testing.assert_close(a, b, rtol=1e-10, atol=1e-10)

    def test_no_eligible_curves_missing_metadata_and_new_smoothing(self):
        data = self.data("cpu", torch.float64, (3, 4, 7));diag = AnalyticDevice("cpu", torch.float64)
        for fn in (engine.iv_ordered_shape_loss, engine.iv_derivative_consistency_loss):
            self.assertEqual(float(fn(diag, data, self.args).detach()), 0.)
            no_codes = {k:v for k,v in data.items() if k != "curve_code"}
            self.assertEqual(float(fn(diag, no_codes, self.args)), 0.)
        data = self.data("cpu", torch.float64)
        del data["family"]
        prepared = PreparedIVBatch(data, IVCurveLayout(data, 0, ID_FLOOR))
        self.args.iv_deriv_target_smooth_passes = 2
        torch.testing.assert_close(reference_iv_ordered_shape_loss(diag, data, self.args),
                                   engine.iv_ordered_shape_loss(diag, prepared, self.args), atol=1e-10, rtol=1e-10)

    def test_symbolic_priors_values_gradients_and_zero_edges(self):
        class ProbeLayer(nn.Module):
            def __init__(self, device):
                super().__init__()
                self.in_dim, self.out_dim = 2, 3
                self.weight = nn.Parameter(torch.tensor([[.3, 0.], [-.2, .5], [1.e-8, .1]], device=device, dtype=torch.float64))

            def forward(self, x):
                return torch.sin(x[:, None, :] * self.weight[None, :, :]).sum(dim=2), None

        for device in self.devices():
            layer = ProbeLayer(device)
            kan = SimpleNamespace(center=torch.zeros(2, device=device, dtype=torch.float64), net=SimpleNamespace(act_fun=[layer]))
            for old, new in [(reference_kan_top1_symbolic_compatibility_loss, engine._kan_top1_symbolic_compatibility_loss),
                             (reference_kan_edge_group_sparsity_loss, engine._kan_edge_group_sparsity_loss)]:
                a, b = old(kan, self.args), new(kan, self.args)
                torch.testing.assert_close(a, b, atol=1e-10, rtol=1e-10)
                ga = torch.autograd.grad(a, layer.weight)[0]
                gb = torch.autograd.grad(b, layer.weight)[0]
                torch.testing.assert_close(ga, gb, atol=1e-10, rtol=1e-10)
            with torch.no_grad():
                layer.weight.zero_()
            self.assertEqual(float(engine._kan_top1_symbolic_compatibility_loss(kan, self.args).detach()), 0.)

    def test_candidate_selection_and_polishing(self):
        names = ["const", "x", "x2", "tanh", "softplus", "asinh", "gauss"]
        for device in self.devices():
            x = torch.linspace(-2., 2., 53, device=device, dtype=torch.float64)
            for name in ("const", "x2", "asinh"):
                target = .7 * symbolic_func(name, x*.75+.5) + .12
                a, ae = reference_fit_top1_symbolic_edge_exhaustive(target, x, names, 5, x.device, x.dtype)
                b, be = engine.fit_top1_symbolic_edge_exhaustive(target, x, names, 5, x.device, x.dtype)
                self.assertEqual(a.active_indices, b.active_indices)
                torch.testing.assert_close(a(x), b(x), atol=1e-9, rtol=1e-9)
                self.assertAlmostEqual(ae, be, places=10)

    def test_joint_polish_short_reachable_sample_preserves_modules(self):
        # A reachable-domain sample can be smaller than the requested budget.
        # The packed optimizer must still copy back into the same edge modules.
        for device in self.devices():
            torch.manual_seed(27)
            dtype = torch.float64
            kan = SimpleNamespace(center=torch.zeros(3, device=device, dtype=dtype),
                                  out_range=torch.tensor([.5, .2], device=device, dtype=dtype),
                                  net=nn.Linear(3, 2, device=device, dtype=dtype))
            layers = nn.ModuleList([engine.EdgewiseSymbolicKANLayer(3, 2, ["x", "tanh"], torch.device(device), dtype)])
            schema = list(layers.state_dict())
            identities = [id(p) for p in layers.parameters()]
            args = SimpleNamespace(symbolic_network_finetune_steps=2, symbolic_network_finetune_points=512)
            samples = torch.randn(17, 3, device=device, dtype=dtype)
            engine._joint_finetune_symbolic_network(kan, layers, device, dtype, args, samples)
            self.assertEqual(schema, list(layers.state_dict()))
            self.assertEqual(identities, [id(p) for p in layers.parameters()])
            self.assertTrue(bool(torch.isfinite(engine._forward_symbolic_layers(layers, samples)).all()))

    def test_residual_cache_temperature_sampling_and_gradients(self):
        for device in self.devices():
            for dtype in (torch.float32, torch.float64):
                with self.subTest(device=device, dtype=dtype):
                    data = self.data(device, dtype)
                    data["temp_K"] = torch.linspace(270., 400., len(data["vbg"]), device=device, dtype=dtype)
                    data["temp_K"][0] = float("nan")
                    data["vds"][1] = 0.
                    data["weight"] = torch.ones_like(data["vbg"])
                    diag = CacheTestDevice(device, dtype)
                    params = list(diag.iv_resid.parameters())
                    layout = IVCurveLayout(data, self.args.iv_deriv_target_smooth_passes, ID_FLOOR)
                    with residual_only_gradients(diag, params):
                        cache = ResidualBaselineCache(diag, data, layout, chunk_size=13)
                        for curves in (False, True):
                            torch.manual_seed(61)
                            expected = engine.curve_batch(data, 3) if curves else engine.batch(data, 23)
                            rng_expected = torch.rand(7, device=device)
                            torch.manual_seed(61)
                            actual = cache.curve_batch(3) if curves else cache.point_batch(23)
                            torch.testing.assert_close(rng_expected, torch.rand(7, device=device), atol=0, rtol=0)
                            for key in data:
                                torch.testing.assert_close(expected[key], actual[key], atol=0, rtol=0, equal_nan=True)
                            torch.testing.assert_close(engine._cached_residual_current(diag, actual)[0], diag.ids_residual(expected))
                            loss_a = engine.iv_residual_loss(diag, expected, self.args)[0]
                            loss_b = engine.iv_residual_loss(diag, actual, self.args)[0]
                            if curves:
                                loss_a = loss_a + sum(engine._iv_curve_constraint_losses(diag, expected, self.args))
                                loss_b = loss_b + sum(engine._iv_curve_constraint_losses(diag, actual, self.args))
                            torch.testing.assert_close(loss_a, loss_b)
                            for ga, gb in zip(torch.autograd.grad(loss_a, params), torch.autograd.grad(loss_b, params)):
                                torch.testing.assert_close(ga, gb)
                        all_curves = cache.curve_batch(100)
                        torch.testing.assert_close(all_curves.residual_baselines[0], diag.base_ids(data))
                        self.assertFalse(cache.base.requires_grad)
                        self.assertFalse(cache.physical.requires_grad)
                    self.assertTrue(diag.base.requires_grad)
                    self.assertIsNone(engine._cached_residual_current(CacheTestDevice(device, dtype), all_curves))
                    self.assertIsNone(engine._cached_residual_current(diag, dict(all_curves)))

    def test_residual_freeze_exception_and_cache_preconditions(self):
        diag = CacheTestDevice("cpu", torch.float64)
        params = list(diag.iv_resid.parameters())
        data = self.data("cpu", torch.float64)
        plan = IVCurveLayout(data, 0, ID_FLOOR)
        flags = [p.requires_grad for p in diag.parameters()]
        with self.assertRaises(ValueError):
            ResidualBaselineCache(diag, data, plan)
        with self.assertRaisesRegex(RuntimeError, "injected"):
            with residual_only_gradients(diag, params):
                self.assertFalse(diag.base.requires_grad)
                raise RuntimeError("injected")
        self.assertEqual(flags, [p.requires_grad for p in diag.parameters()])
        with residual_only_gradients(diag, params):
            data["vbg"].requires_grad_(True)
            with self.assertRaises(ValueError):
                ResidualBaselineCache(diag, data, plan)

    def test_cached_current_preserves_zero_sign_floor_and_clipping(self):
        diag = CacheTestDevice("cpu", torch.float64)
        data = {"vbg": torch.tensor([-30., -1., 0., 1., 30.], dtype=torch.float64),
                "vtg": torch.zeros(5, dtype=torch.float64),
                "vds": torch.tensor([-1., -1., 0., 1., 1.], dtype=torch.float64)}
        with torch.no_grad():
            diag.iv_resid.weight.fill_(2.)
        cached = PreparedIVBatch(data, None)
        cached.residual_baseline_owner = diag
        with torch.no_grad():
            cached.residual_baselines = (diag.base_ids(data), diag.physical_ids(data))
        expected = diag.ids_residual(data)
        actual, raw_residual, _, _ = engine._cached_residual_current(diag, cached)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        self.assertEqual(float(actual[2].detach()), 0.)
        self.assertLess(float(actual[0].detach()), 0.)
        self.assertGreater(float(raw_residual[-1].detach()), 2.)
        self.assertAlmostEqual(float(actual[-1].detach()), .01)

    def test_batched_anchor_preserves_finite_difference_objective(self):
        for device in self.devices():
            data = self.data(device, torch.float64)
            data["temp_K"] = torch.full_like(data["vbg"], 340.)
            data["gt"] = torch.zeros_like(data["vbg"])
            diag = CacheTestDevice(device, torch.float64)
            r = diag.logid_residual(data)
            def offset(key, amount):
                return diag.logid_residual({**data, key: data[key] + amount})
            rb = offset("vbg", .08) - 2*r + offset("vbg", -.08)
            rt = offset("vtg", .08) - 2*r + offset("vtg", -.08)
            old = r.square().mean() + self.args.extrap_iv_curv_weight * (rb.square().mean() + rt.square().mean())
            new = engine.iv_extrap_anchor_loss(diag, data, self.args)
            torch.testing.assert_close(old, new, atol=1e-12, rtol=1e-12)
            params = tuple(diag.iv_resid.parameters())
            for a, b in zip(torch.autograd.grad(old, params), torch.autograd.grad(new, params)):
                torch.testing.assert_close(a, b, atol=1e-12, rtol=1e-12)
            self.assertEqual(float(engine.iv_extrap_anchor_loss(diag, {k:v[:0] for k,v in data.items()}, self.args)), 0.)


if __name__ == "__main__":
    unittest.main()
