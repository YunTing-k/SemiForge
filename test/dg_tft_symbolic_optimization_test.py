"""Numerical, lifecycle and workflow regression checks for symbolic acceleration."""
from pathlib import Path
import copy
import json
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

try:
    import torch
    from torch import nn
except ImportError:
    raise unittest.SkipTest("Requires the modeling environment (PyTorch)")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src/device_modeling/dg_tft/engine"))
import dg_tft_full_model as engine
from symbolic_optimization import EarlyStop, PackedSymbolicNet, evaluate_groups, symbolic_student


def layers(device="cpu"):
    names = ["const", "x", "x2", "tanh", "sigmoid", "softplus", "silu", "softsign", "asinh", "decay", "invquad", "gauss", "log_1p"]
    result = nn.ModuleList([
        engine.EdgewiseSymbolicKANLayer(3, 4, names, torch.device(device), torch.float64),
        engine.EdgewiseSymbolicKANLayer(4, 2, names, torch.device(device), torch.float64),
    ])
    with torch.no_grad():
        for layer in result:
            for j, row in enumerate(layer.edges):
                for i, edge in enumerate(row):
                    edge.a.normal_(0, .1)
                    edge.b.uniform_(.2, .8)
                    edge.c.uniform_(-.2, .2)
                    edge.bias.fill_(.03)
                    edge_index = j * layer.in_dim + i
                    edge.active_indices = [edge_index % len(names), (edge_index + 5) % len(names)]
            layer.importance[0, 0] = 0
    return result


class TestPackedSymbolic(unittest.TestCase):
    def test_outputs_gradients_adam_and_export_roundtrip(self):
        for device in (["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]):
            with self.subTest(device=device):
                torch.manual_seed(74)
                source = layers(device)
                packed = PackedSymbolicNet(source, engine.symbolic_func)
                reference = engine.EdgewiseSymbolicKANNet(source)
                optimizers = [torch.optim.Adam(net.parameters(), lr=.001) for net in (reference, packed)]
                for _ in range(3):
                    x = torch.randn(19, 3, device=device, dtype=torch.float64, requires_grad=True)
                    xx = x.detach().clone().requires_grad_()
                    for opt in optimizers:
                        opt.zero_grad(set_to_none=True)
                    y, yy = reference(x), packed(xx)
                    torch.testing.assert_close(y, yy, atol=1e-10, rtol=1e-10)
                    y.square().mean().backward()
                    yy.square().mean().backward()
                    torch.testing.assert_close(x.grad, xx.grad, atol=1e-10, rtol=1e-10)
                    for sl, pl in zip(source, packed):
                        for group, (_, terms) in enumerate(pl.groups):
                            expected = torch.stack([torch.stack([
                                getattr(sl.edges[j][i], field).grad[k]
                                if getattr(sl.edges[j][i], field).grad is not None else x.new_zeros(())
                                for j, i, k in terms]) for field in ("a", "b", "c")])
                            torch.testing.assert_close(expected, pl.coefficients[group].grad, atol=1e-10, rtol=1e-10)
                    for net, opt in zip((reference, packed), optimizers):
                        torch.nn.utils.clip_grad_norm_(net.parameters(), 5.)
                        opt.step()
                destination = copy.deepcopy(source)
                packed.copy_to(destination)
                torch.testing.assert_close(reference(x), engine.EdgewiseSymbolicKANNet(destination)(x), atol=1e-9, rtol=1e-9)
                self.assertEqual(set(source.state_dict()), set(destination.state_dict()))
                self.assertEqual(engine._symbolic_active_index_tree(source), engine._symbolic_active_index_tree(destination))

    def test_empty_layer_and_topology_rebuild(self):
        source = layers()
        source[0].importance.zero_()
        packed = PackedSymbolicNet(source, engine.symbolic_func)
        x = torch.randn(4, 3, dtype=torch.float64)
        torch.testing.assert_close(packed(x), engine.EdgewiseSymbolicKANNet(source)(x))
        source[0].importance.fill_(1)
        source[0].edges[0][0].active_indices = None
        source[0].edges[0][1].active_indices = []
        rebuilt = PackedSymbolicNet(source, engine.symbolic_func)
        torch.testing.assert_close(rebuilt(x), engine.EdgewiseSymbolicKANNet(source)(x))

    def test_context_restores_teacher_and_parameter_flags_on_error(self):
        diag = nn.Module()
        diag.base = nn.Module(); diag.base.kan = nn.Module(); diag.base.kan.net = nn.Linear(3, 2)
        diag.iv_resid = nn.Module(); diag.iv_resid.net = nn.Linear(3, 2)
        diag.physical = nn.Parameter(torch.tensor(2.))
        teacher = diag.base.kan.net
        source, residual = layers(), layers()
        before = copy.deepcopy(source.state_dict())
        with self.assertRaisesRegex(RuntimeError, "intentional"):
            with symbolic_student(diag, source, residual, engine.symbolic_func, 1) as (base, _, _):
                self.assertFalse(diag.physical.requires_grad)
                with torch.no_grad(): base[0].bias.add_(10)
                raise RuntimeError("intentional")
        self.assertIs(diag.base.kan.net, teacher)
        self.assertTrue(diag.physical.requires_grad)
        for k, v in before.items(): torch.testing.assert_close(source.state_dict()[k], v)

    def test_grouping_preserves_mixed_temperature_and_gate_defaults(self):
        class Diag:
            base = SimpleNamespace(p=SimpleNamespace(Tnom=300.))
            def ids_residual(self, data, scale):
                return scale * (data["vbg"] * data.get("gt", 1.) + data["vtg"] * data.get("gb", 1.) + data["vds"] + data.get("temp_K", 300.))
        groups = [{"vbg": torch.arange(3.), "vtg": torch.ones(3), "vds": torch.ones(3)},
                  {"vbg": torch.arange(2.), "vtg": torch.ones(2), "vds": torch.ones(2), "temp_K": torch.tensor([240., 380.]), "gt": torch.zeros(2)}]
        for a, b in zip([Diag().ids_residual(group, .7) for group in groups], evaluate_groups(Diag(), groups, .7)):
            torch.testing.assert_close(a, b)


class TestEarlyStop(unittest.TestCase):
    def test_minimum_patience_delta_and_true_best_restore(self):
        module = nn.Linear(1, 1)
        stop = EarlyStop(2, 30, .1)
        for step, score, expected in [(0, 1., False), (10, .96, False), (20, .95, False), (30, .97, True)]:
            with torch.no_grad(): module.weight.fill_(step)
            self.assertEqual(stop.observe(step, score, [module]), expected)
        stop.restore([module])
        self.assertEqual(stop.best_step, 20)
        self.assertEqual(module.weight.item(), 20)

    def test_disabled_and_nonfinite_scores(self):
        module = nn.Linear(1, 1)
        stop = EarlyStop(0, 0, 0)
        self.assertFalse(stop.observe(0, float("nan"), [module]))
        with self.assertRaises(FloatingPointError): stop.restore([module])
        stop.observe(1, 2., [module])
        self.assertFalse(stop.observe(100, float("inf"), [module]))
        stop.restore([module])
        for args in [(-1, 0, 0), (1, -1, 0), (1, 1, float("nan"))]:
            with self.assertRaises(ValueError): EarlyStop(*args)


class TestSymbolicWorkflow(unittest.TestCase):
    def test_agent_bounds_ini_and_engine_command_roundtrip(self):
        from src.workflow.dg_tft.param_defs import get_stage_param_table, get_stage_param_schema, get_stage_sections
        from src.workflow.dg_tft.quality_guard import validate_agent_adjustments, apply_adjustments
        from src.workflow.dg_tft.schema import StageName
        from src.device_modeling.dg_tft import run_model
        table = get_stage_param_table(StageName.TRAIN_STATIC, task_mode="iv_only")
        properties = get_stage_param_schema(StageName.TRAIN_STATIC, task_mode="iv_only")["function"]["parameters"]["properties"]
        overrides = {"device_iv_finetune_patience": 0,
                     "device_iv_measured_patience": 3, "device_iv_measured_min_steps": 120,
                     "device_iv_finetune_eval_period": 30, "device_iv_measured_eval_period": 20,
                     "device_iv_finetune_min_steps": 200, "device_iv_finetune_min_delta": .002,
                     "device_iv_measured_min_delta": .003}
        for key in overrides: self.assertIn(key, properties)
        accepted, warnings = validate_agent_adjustments(overrides, table, get_stage_sections(StageName.TRAIN_STATIC), StageName.TRAIN_STATIC)
        self.assertEqual(accepted, overrides); self.assertFalse(warnings)
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / "model.ini"
            template = Path(__file__).resolve().parents[1] / 'src/device_modeling/dg_tft/config/model_config.ini'
            config.write_bytes(template.read_bytes())
            apply_adjustments(str(config), {f"symbolic.{k}": str(v) for k, v in accepted.items()})
            cfg = run_model.load_config(config)
            cfg["runtime"]["minimal_run"] = "no"
            paths = run_model.paths(cfg, config)
            with patch.object(run_model, "run") as execute:
                run_model.train_full_model(cfg, paths)
            command = execute.call_args.args[0]
            for key, value in overrides.items():
                flag = '--symbolic-' + key.replace('_', '-')
                self.assertEqual(command[command.index(flag) + 1], str(value))
            with patch.object(sys, "argv", ["engine", *command[2:]]):
                parsed = engine.parse_args()
            for key, value in overrides.items():
                self.assertEqual(getattr(parsed, "symbolic_" + key), value)
        bad, warnings = validate_agent_adjustments({"device_iv_measured_patience": -2}, table, get_stage_sections(StageName.TRAIN_STATIC), StageName.TRAIN_STATIC)
        self.assertEqual(bad["device_iv_measured_patience"], 0)
        self.assertTrue(warnings)


if __name__ == "__main__":
    unittest.main()
