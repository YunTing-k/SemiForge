"""Batched execution and stopping policy for fixed-topology symbolic students.

The packed representation is temporary: coefficients are copied back into the
existing edge modules so checkpoints, formulas and Verilog-A retain their schema.
It is rebuilt for each calibration, after any topology selection or restoration.
"""
from __future__ import annotations

from contextlib import contextmanager
import math

import torch
from torch import nn
from torch.overrides import TorchFunctionMode


class _CaptureConstants(TorchFunctionMode):
    """Own immutable scalar tensors during one symbolic graph's preparation.

    Scalar tensor factories in the physical kernel otherwise copy CPU constants
    to CUDA during capture. This thread-local dispatch mode is active only while
    warming/capturing our deterministic objective; it never patches torch APIs.
    Warmup allocates the constants normally, then capture only reads them.
    """

    def __init__(self):
        super().__init__()
        self.values = {}
        self.sealed = False

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        # Reject known synchronization/dynamic-output operations in warmup,
        # before entering CUDA capture (some runtimes retain a sticky CUDA
        # error after attempting host reads inside a captured stream).
        if func in (torch.Tensor.item, torch.Tensor.cpu, torch.Tensor.numpy,
                    torch.Tensor.__bool__, torch.Tensor.__float__, torch.Tensor.__int__):
            raise RuntimeError("Host tensor read is unsupported in symbolic CUDA graph")
        if func is torch.Tensor.__getitem__:
            indices = args[1] if isinstance(args[1], tuple) else (args[1],)
            if any(torch.is_tensor(index) and (index.device != args[0].device or index.dtype == torch.bool)
                   for index in indices):
                raise RuntimeError("Host or boolean tensor indexing is unsupported in symbolic CUDA graph")
        key = None
        if func is torch.Tensor.new_tensor and len(args) == 2:
            reference, value = args
            key = (func, reference.device, reference.dtype, reference.layout)
        elif func in (torch.tensor, torch.as_tensor) and len(args) == 1:
            value = args[0]
            key = (func, torch.get_default_dtype(), torch.get_default_device())
        else:
            value = None
        if (key is None or type(value) not in (float, int, bool)
                or not math.isfinite(value) or kwargs.get("requires_grad", False)):
            return func(*args, **kwargs)
        key += (type(value), value, tuple(sorted(kwargs.items())))
        if key not in self.values:
            if self.sealed:
                raise RuntimeError("Unprepared scalar constant in symbolic CUDA graph")
            self.values[key] = func(*args, **kwargs)
        return self.values[key]


class SymbolicBackwardRunner:
    """Replay only deterministic forward/backward; keep all training policy eager.

    Sampling, Adam, gradient clipping, scheduler and validation stay in the
    caller. Each replay copies the *new* batch/targets into fixed input buffers,
    zeros captured gradients and uses current parameter storage. Warmup/capture
    never steps the optimizer and preserves RNG state. CPU and capture failures
    retain the identical ordinary objective, not a second numerical formula.
    """

    def __init__(self, objective, params, enabled=True):
        self.objective, self.params = objective, list(params)
        self.enabled = enabled
        self.attempted = False
        self.graph = None
        self.fallback_reason = None
        self.replays = 0

    def _capture(self, batch, target):
        self.batch = {k: v.detach().clone() for k, v in batch.items()}
        self.target = {k: v.detach().clone() for k, v in target.items()}
        self.constants = _CaptureConstants()
        device = next(iter(batch.values())).device
        # Autograd requires a side-stream warmup before capture. No RNG advances
        # and no optimizer state/parameter update is allowed in this setup.
        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream(device))
        with torch.random.fork_rng(devices=[device.index]):
            with torch.cuda.stream(stream), self.constants:
                for _ in range(3):
                    for p in self.params:
                        p.grad = None
                    self.objective(self.batch, self.target).backward()
            torch.cuda.current_stream(device).wait_stream(stream)
            versions = {key: value._version for key, value in self.constants.values.items()}
            self.constants.sealed = True
            graph = torch.cuda.CUDAGraph()
            # Manage the stream separately: some PyTorch graph context versions
            # fail to restore it when capture_end raises on an unsupported op.
            with torch.cuda.stream(stream), self.constants:
                graph.capture_begin()
                try:
                    for p in self.params:
                        if p.grad is not None:
                            p.grad.zero_()
                    self.loss = self.objective(self.batch, self.target)
                    self.loss.backward()
                finally:
                    graph.capture_end()
            torch.cuda.current_stream(device).wait_stream(stream)
            if any(value._version != versions[key] for key, value in self.constants.values.items()):
                raise RuntimeError("Symbolic objective mutated a captured constant")
            self.gradients = [p.grad for p in self.params]
            self.graph = graph

    @staticmethod
    def _compatible(source, buffers):
        return source.keys() == buffers.keys() and all(
            v.shape == buffers[k].shape and v.dtype == buffers[k].dtype and v.device == buffers[k].device
            and not v.requires_grad for k, v in source.items()
        )

    def backward(self, batch, target):
        if not self.attempted:
            self.attempted = True
            cuda = bool(batch) and all(v.is_cuda and not v.requires_grad for v in (*batch.values(), *target.values()))
            if self.enabled and cuda:
                previous_grads = [p.grad for p in self.params]
                try:
                    self._capture(batch, target)
                except (RuntimeError, NotImplementedError) as exc:
                    self.graph = None
                    self.fallback_reason = f"{type(exc).__name__}: {exc}"
                finally:
                    for p, grad in zip(self.params, previous_grads):
                        p.grad = grad
        if self.graph is not None:
            if self._compatible(batch, self.batch) and self._compatible(target, self.target):
                for key, value in batch.items():
                    self.batch[key].copy_(value)
                for key, value in target.items():
                    self.target[key].copy_(value)
                for p, grad in zip(self.params, self.gradients):
                    p.grad = grad
                self.graph.replay()
                self.replays += 1
                return self.loss
            self.graph = None
            self.fallback_reason = "Input shape, dtype, device or gradient requirement changed"
        # Always match zero_grad(set_to_none=True), including after a failed
        # capture or shape change; never leak warmup gradients into a real step.
        for p in self.params:
            p.grad = None
        loss = self.objective(batch, target)
        loss.backward()
        return loss


class PackedSymbolicLayer(nn.Module):
    """Evaluate active edges grouped by analytic function, without scalar reads."""

    def __init__(self, source, symbolic_func):
        super().__init__()
        self.symbolic_func = symbolic_func
        self.bias = nn.Parameter(source.bias.detach().clone())
        mask = (source.importance.detach().cpu() > 0.5).tolist()
        self.edge_indices = [(j, i) for j, row in enumerate(mask) for i, active in enumerate(row) if active]
        device = source.bias.device
        self.edge_bias = nn.Parameter(
            torch.stack([source.edges[j][i].bias.detach() for j, i in self.edge_indices])
            if self.edge_indices else source.bias.new_empty(0)
        )
        self.register_buffer("bias_outputs", torch.tensor([j for j, _ in self.edge_indices], device=device, dtype=torch.long))
        self.groups = []
        self.coefficients = nn.ParameterList()
        for k, name in enumerate(source.symbol_names):
            terms = []
            for j, i in self.edge_indices:
                edge = source.edges[j][i]
                active = edge.active_indices if edge.active_indices is not None else range(len(edge.symbol_names))
                if k in active:
                    terms.append((j, i, k))
            if not terms:
                continue
            group = len(self.groups)
            self.groups.append((name, terms))
            self.register_buffer(f"inputs_{group}", torch.tensor([i for _, i, _ in terms], device=device))
            self.register_buffer(f"outputs_{group}", torch.tensor([j for j, _, _ in terms], device=device))
            self.coefficients.append(nn.Parameter(torch.stack([
                torch.stack([getattr(source.edges[j][i], field)[k].detach() for j, i, k in terms])
                for field in ("a", "b", "c")
            ])))

    def forward(self, x):
        result = self.bias.unsqueeze(0).expand(x.shape[0], -1).clone()
        result = result.index_add(1, self.bias_outputs, self.edge_bias.unsqueeze(0).expand(x.shape[0], -1))
        for group, (name, _) in enumerate(self.groups):
            a, b, c = self.coefficients[group].unbind(0)
            z = x.index_select(1, getattr(self, f"inputs_{group}")) * b + c
            result = result.index_add(1, getattr(self, f"outputs_{group}"), a * self.symbolic_func(name, z))
        return result

    @torch.no_grad()
    def copy_to(self, target):
        target.bias.copy_(self.bias)
        for n, (j, i) in enumerate(self.edge_indices):
            target.edges[j][i].bias.copy_(self.edge_bias[n])
        for group, (_, terms) in enumerate(self.groups):
            for field_id, field in enumerate(("a", "b", "c")):
                for n, (j, i, k) in enumerate(terms):
                    getattr(target.edges[j][i], field)[k].copy_(self.coefficients[group][field_id, n])


class PackedSymbolicNet(nn.Sequential):
    def __init__(self, source, symbolic_func):
        super().__init__(*[PackedSymbolicLayer(layer, symbolic_func) for layer in source])

    def copy_to(self, target):
        for layer, destination in zip(self, target):
            layer.copy_to(destination)


@contextmanager
def symbolic_student(diag, base_layers, residual_layers, symbolic_func, scale):
    """Install a student temporarily; restore the numeric teacher even on failure."""
    base = PackedSymbolicNet(base_layers, symbolic_func)
    residual = PackedSymbolicNet(residual_layers, symbolic_func)
    teacher_base, teacher_residual = diag.base.kan.net, diag.iv_resid.net
    diag.base.kan.net, diag.iv_resid.net = base, residual
    params = list(base.parameters()) + (list(residual.parameters()) if scale > 0 else [])
    # Only the symbolic coefficients are optimized. Avoid accumulating gradients
    # in frozen physical/temperature parameters that are absent from Adam.
    flags = [(p, p.requires_grad) for p in diag.parameters()]
    trainable = {id(p) for p in params}
    for p, _ in flags:
        p.requires_grad_(id(p) in trainable)
    try:
        yield base, residual, params
        base.copy_to(base_layers)
        residual.copy_to(residual_layers)
    finally:
        for p, enabled in flags:
            p.requires_grad_(enabled)
        diag.base.kan.net, diag.iv_resid.net = teacher_base, teacher_residual


def evaluate_groups(diag, groups, scale):
    """Merge pointwise device evaluations; metadata-only columns are not needed.

    Explicit default temperature and gate flags preserve the behavior when only
    some groups (e.g. measured data versus teacher anchors) contain those fields.
    """
    merged = {}
    for key in ("vbg", "vtg", "vds", "gt", "gb", "temp_K"):
        values = []
        for group in groups:
            if key in group:
                values.append(group[key])
            elif key in ("gt", "gb"):
                values.append(torch.ones_like(group["vbg"]))
            elif key == "temp_K":
                values.append(torch.full_like(group["vbg"], float(diag.base.p.Tnom)))
            else:
                raise KeyError(f"Missing terminal voltage: {key}")
        merged[key] = torch.cat(values)
    return list(diag.ids_residual(merged, scale).split([len(group["vbg"]) for group in groups]))


class EarlyStop:
    """Track the true best model separately from significant-improvement patience.

    Patience counts evaluation events, not optimizer steps. Minimum steps gate
    stopping, while checks during warmup still establish the best state. Zero
    patience disables stopping. Snapshots remain on-device to avoid many copies.
    """

    def __init__(self, patience, min_steps, min_delta):
        if patience < 0 or min_steps < 0 or not math.isfinite(min_delta) or min_delta < 0:
            raise ValueError("Early-stop patience/min_steps/min_delta must be finite and nonnegative")
        self.patience, self.min_steps, self.min_delta = patience, min_steps, min_delta
        self.best_score = self.reference_score = float("inf")
        self.best_step = 0
        self.bad_checks = 0
        self.states = None

    def observe(self, step, score, modules):
        if math.isfinite(score) and score < self.best_score:
            self.best_score, self.best_step = score, step
            self.states = [{k: v.detach().clone() for k, v in module.state_dict().items()} for module in modules]
        if math.isfinite(score) and score < self.reference_score - self.min_delta:
            self.reference_score, self.bad_checks = score, 0
        else:
            self.bad_checks += 1
        return bool(self.patience and step >= self.min_steps and self.bad_checks >= self.patience)

    def restore(self, modules):
        if self.states is None:
            raise FloatingPointError("No finite symbolic validation score; refusing to export an invalid student")
        for module, state in zip(modules, self.states):
            module.load_state_dict(state, strict=True)
