"""Training-policy equivalence for deterministic symbolic CUDA replay."""
from pathlib import Path
import copy
import sys
import unittest
from unittest.mock import patch

try:
    import torch
    from torch import nn
except ImportError:
    raise unittest.SkipTest("Requires the modeling environment (PyTorch)")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src/device_modeling/dg_tft/engine"))
from symbolic_optimization import SymbolicBackwardRunner, EarlyStop


class Objective(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([.4, -.2], device=device))
        self.bias = nn.Parameter(torch.tensor(.1, device=device))
        self.unused = nn.Parameter(torch.tensor(2., device=device))

    def forward(self, batch, target):
        x = batch["x"]
        # Exercise all scalar factory forms used in the physical kernel. These
        # factories must stay normal outside the private capture context.
        offset = x.new_tensor(.2) + torch.tensor(.1, device=x.device, dtype=x.dtype)
        scale = torch.as_tensor(.7, device=x.device, dtype=x.dtype)
        pred = torch.tanh((x * self.weight).sum(dim=1) + self.bias + offset) * scale
        return ((pred - target["y"]).square() * batch["weights"]).mean()


class TestSymbolicBackwardRunner(unittest.TestCase):
    def test_cpu_ordinary_execution_and_no_rng_change(self):
        model = Objective("cpu")
        runner = SymbolicBackwardRunner(model, model.parameters())
        x = {"x": torch.ones(5, 2), "weights": torch.ones(5)}
        target = {"y": torch.zeros(5)}
        state = torch.get_rng_state().clone()
        for _ in range(3):
            actual = runner.backward(x, target)
            torch.testing.assert_close(actual, model(x, target))
            self.assertIsNone(model.unused.grad)
        self.assertEqual(runner.replays, 0)
        torch.testing.assert_close(state, torch.get_rng_state(), atol=0, rtol=0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_changing_batches_adam_clipping_scheduler_and_restore(self):
        torch.manual_seed(63)
        eager = Objective("cuda")
        graph_model = copy.deepcopy(eager)
        models = (eager, graph_model)
        runners = [SymbolicBackwardRunner(model, model.parameters(), enabled=enabled)
                   for model, enabled in zip(models, (False, True))]
        opts = [torch.optim.Adam(model.parameters(), lr=.03) for model in models]
        schedules = [torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=16, eta_min=.001) for opt in opts]
        stops = [EarlyStop(0, 0, 0.) for _ in models]
        val = {"x": torch.randn(13, 2, device="cuda"), "weights": torch.ones(13, device="cuda")}
        val_target = {"y": torch.randn(13, device="cuda")}
        for step in range(16):
            batch = {"x": torch.randn(9, 2, device="cuda") * (step+1),
                     "weights": torch.rand(9, device="cuda") + .2}
            target = {"y": torch.randn(9, device="cuda")}
            cpu_rng, cuda_rng = torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone()
            losses = []
            for model, runner, opt, schedule in zip(models, runners, opts, schedules):
                opt.zero_grad(set_to_none=True)
                losses.append(runner.backward(batch, target).detach().clone())
                torch.nn.utils.clip_grad_norm_(model.parameters(), .05)
                opt.step()
                schedule.step()
            torch.testing.assert_close(cpu_rng, torch.get_rng_state(), atol=0, rtol=0)
            torch.testing.assert_close(cuda_rng, torch.cuda.get_rng_state(), atol=0, rtol=0)
            torch.testing.assert_close(losses[0], losses[1], atol=2e-6, rtol=1e-5)
            self.assertEqual(schedules[0].get_last_lr(), schedules[1].get_last_lr())
            for p, q in zip(eager.parameters(), graph_model.parameters()):
                torch.testing.assert_close(p, q, atol=3e-6, rtol=1e-5)
                if p.grad is None:
                    self.assertIsNone(q.grad)
                else:
                    torch.testing.assert_close(p.grad, q.grad, atol=1e-6, rtol=1e-4)
                    for key in ("step", "exp_avg", "exp_avg_sq"):
                        torch.testing.assert_close(opts[0].state[p][key], opts[1].state[q][key], atol=1e-6, rtol=1e-4)
            if step % 3 == 0:
                with torch.no_grad():
                    for model, stop in zip(models, stops):
                        stop.observe(step, float(model(val, val_target)), [model])
            if step == 9:
                # load_state_dict copies into captured parameter storage; a
                # restored checkpoint must affect the very next graph replay.
                for model, stop in zip(models, stops):
                    stop.restore([model])
        self.assertEqual(stops[0].best_step, stops[1].best_step)
        self.assertEqual(runners[1].replays, 16)
        self.assertIsNone(runners[1].fallback_reason)
        self.assertIsNone(graph_model.unused.grad)
        self.assertNotIn(graph_model.unused, opts[1].state)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_shape_change_falls_back_without_stale_gradients(self):
        model = Objective("cuda")
        runner = SymbolicBackwardRunner(model, model.parameters())
        for n in (7, 11, 11):
            batch = {"x": torch.randn(n, 2, device="cuda"), "weights": torch.ones(n, device="cuda")}
            target = {"y": torch.randn(n, device="cuda")}
            loss = runner.backward(batch, target)
            actual = [None if p.grad is None else p.grad.clone() for p in model.parameters()]
            model.zero_grad(set_to_none=True)
            ref = model(batch, target)
            ref.backward()
            torch.testing.assert_close(ref, loss)
            for p, g in zip(model.parameters(), actual):
                if g is not None:
                    torch.testing.assert_close(p.grad, g)
        self.assertEqual(runner.replays, 1)
        self.assertIn("shape", runner.fallback_reason)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_capture_failure_restores_rng_and_discards_warmup_gradients(self):
        model = Objective("cuda")
        runner = SymbolicBackwardRunner(model, model.parameters())
        batch = {"x": torch.ones(7, 2, device="cuda"), "weights": torch.ones(7, device="cuda")}
        target = {"y": torch.zeros(7, device="cuda")}
        state = torch.cuda.get_rng_state().clone()
        # Fail after warmup, before capture starts, to model an unavailable
        # graph runtime without deliberately poisoning the CUDA context.
        with patch("torch.cuda.CUDAGraph", side_effect=RuntimeError("capture unavailable")):
            loss = runner.backward(batch, target)
        grads = [p.grad.clone() if p.grad is not None else None for p in model.parameters()]
        torch.testing.assert_close(state, torch.cuda.get_rng_state(), atol=0, rtol=0)
        model.zero_grad(set_to_none=True)
        ref = model(batch, target)
        ref.backward()
        torch.testing.assert_close(ref, loss)
        for p, g in zip(model.parameters(), grads):
            if g is not None:
                torch.testing.assert_close(p.grad, g)
        self.assertIn("capture unavailable", runner.fallback_reason)
        self.assertEqual(runner.replays, 0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_failure_inside_capture_can_resume_ordinary_training(self):
        model = Objective("cuda")
        def objective(batch, target):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("unsupported capture branch")
            return model(batch, target)
        runner = SymbolicBackwardRunner(objective, model.parameters())
        batch = {"x": torch.ones(7, 2, device="cuda"), "weights": torch.ones(7, device="cuda")}
        target = {"y": torch.zeros(7, device="cuda")}
        opt = torch.optim.Adam(model.parameters(), lr=.01)
        for _ in range(2):
            opt.zero_grad(set_to_none=True)
            loss = runner.backward(batch, target)
            self.assertTrue(bool(torch.isfinite(loss)))
            opt.step()
        self.assertIn("unsupported capture branch", runner.fallback_reason)
        self.assertEqual(runner.replays, 0)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_host_index_and_item_fall_back_before_capture(self):
        for host_read in (False, True):
            model = Objective("cuda")
            index = torch.tensor([0, 1, 2])
            def objective(batch, target):
                if host_read:
                    return model(batch, target) + batch["weights"][0].item()
                return model({"x": batch["x"][index], "weights": batch["weights"][index]},
                             {"y": target["y"][index]})
            runner = SymbolicBackwardRunner(objective, model.parameters())
            batch = {"x": torch.ones(7, 2, device="cuda"), "weights": torch.ones(7, device="cuda")}
            target = {"y": torch.zeros(7, device="cuda")}
            actual = runner.backward(batch, target)
            torch.testing.assert_close(actual, objective(batch, target))
            self.assertEqual(runner.replays, 0)
            self.assertIn("unsupported", runner.fallback_reason)


if __name__ == "__main__":
    unittest.main()
