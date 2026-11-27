"""Coverage-aware IV acceptance: absent regions differ from invalid predictions.

Run in the modeling environment. Small CPU models exercise real metric and
checkpoint selection paths without API calls or a full device training run.
"""
from contextlib import ExitStack, redirect_stdout
from dataclasses import dataclass
import io
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src/device_modeling/dg_tft/engine"))
import dg_tft_full_model as engine


@dataclass
class PhysicalStub:
    value: float = 1.0


class TinyKAN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.4, dtype=torch.float64))

    def optimizer_parameters(self, enabled=True):
        self.weight.requires_grad_(enabled)
        return [self.weight]

    def assert_valid_grids(self, context):
        pass

    def config(self):
        return {}


class TinyBase(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.kan = TinyKAN()
        self.cv_kan = TinyKAN()
        self.phys_delta = torch.nn.Linear(1, 1, dtype=torch.float64)
        self.p = PhysicalStub()
        self.trap = PhysicalStub()

    def ids(self, data, scale):
        return data["id"] * 10.0 ** self.kan.weight


def data():
    return {
        "id": torch.tensor([1e-12, 1e-10, 1e-8, 1e-6], dtype=torch.float64),
        "vds": torch.ones(4, dtype=torch.float64),
        "vbg": torch.arange(4, dtype=torch.float64),
        "vtg": torch.ones(4, dtype=torch.float64),
        "family": torch.zeros(4, dtype=torch.int64),
        "curve_code": torch.zeros(4, dtype=torch.int64),
    }


def limits():
    return {"base_iv_max_" + name: 1.0 for name in (
        "nonzero_rms", "nonzero_p95", "off_rms", "subthreshold_rms",
        "transition_rms", "on_rms", "high_rms",
    )}


def output_frame(error=0.01):
    d = data()
    frame = pd.DataFrame({
        "VDS_V": d["vds"].numpy(), "VBG_V": d["vbg"].numpy(),
        "VTG_V": d["vtg"].numpy(), "ID_A": d["id"].numpy(),
        "family": "IDVBG", "source_file": "test.csv",
    })
    for prefix, col in [("physical", "ID_physical_A"), ("base", "ID_base_pkan_A"), ("residual", "ID_residual_A")]:
        value = error if prefix == "residual" else 0.4
        frame[col] = frame.ID_A * 10 ** value
        frame[f"log10_error_{prefix}_dec"] = value
    return frame


def default_args():
    with patch.object(sys, "argv", ["engine"]):
        return engine.parse_args()


class RegionCoverageTests(unittest.TestCase):
    def setUp(self):
        self.base = TinyBase()
        self.metrics = engine._base_iv_accuracy(self.base, data(), 1.0)

    def test_empty_off_region_is_unverified_and_score_stays_finite(self):
        self.assertEqual(self.metrics["iv_off_rms_count"], 0)
        self.assertTrue(math.isnan(self.metrics["iv_off_rms"]))
        self.assertAlmostEqual(engine._base_iv_selection_score(self.metrics), 0.4)
        passed, checks = engine._base_iv_precision_gate(self.metrics, SimpleNamespace(**limits()))
        self.assertTrue(passed)
        self.assertEqual(checks["iv_off_rms"]["status"], "not_applicable")

    def test_nonfinite_populated_region_fails_instead_of_being_skipped(self):
        for invalid in [float("nan"), float("inf")]:
            metrics = dict(self.metrics, iv_on_rms=invalid)
            self.assertEqual(engine._base_iv_selection_score(metrics), float("inf"))
            self.assertFalse(engine._base_iv_precision_gate(metrics, SimpleNamespace(**limits()))[0])
            self.assertFalse(engine._iv_region_comparison(self.metrics, metrics, "iv_on_rms", 0.1)["pass"])
            self.assertFalse(engine._iv_metric_limit_check(metrics, "iv_on_rms", 1.0)["pass"])

    def test_unknown_coverage_and_mandatory_empty_metrics_fail(self):
        metrics = dict(self.metrics)
        del metrics["iv_off_rms_count"]
        self.assertFalse(engine._base_iv_precision_gate(metrics, SimpleNamespace(**limits()))[0])
        self.assertEqual(engine._base_iv_selection_score(metrics), float("inf"))
        metrics.update(iv_nonzero_rms=float("nan"), iv_nonzero_rms_count=0)
        self.assertFalse(engine._iv_metric_limit_check(metrics, "iv_nonzero_rms", 1.0)["pass"])

    def test_full_coverage_preserves_original_score_and_thresholds(self):
        keys = ["iv_nonzero_rms", "iv_off_rms", "iv_subthreshold_rms", "iv_transition_rms", "iv_on_rms", "iv_high_rms"]
        values = [0.21, 0.13, 0.17, 0.09, 0.08, 0.04]
        metrics = dict(zip(keys, values))
        expected = 0.24*values[0]+0.10*values[1]+0.22*values[2]+0.20*values[3]+0.14*values[4]+0.10*values[5]
        self.assertEqual(engine._base_iv_selection_score(metrics), expected)
        bad = dict(self.metrics, iv_on_rms=2.0)
        self.assertFalse(engine._base_iv_precision_gate(bad, SimpleNamespace(**limits()))[0])

    def test_residual_and_symbolic_comparisons_skip_only_shared_empty_region(self):
        check = engine._iv_region_comparison(self.metrics, self.metrics, "iv_off_rms", 0.02)
        self.assertEqual(check["status"], "not_applicable")
        other = dict(self.metrics, iv_off_rms_count=3, iv_off_rms=0.0)
        self.assertFalse(engine._iv_region_comparison(self.metrics, other, "iv_off_rms", 0.02)["pass"])
        self.assertFalse(engine._iv_region_comparison(other, self.metrics, "iv_off_rms", 0.02)["pass"])

    def test_dataframe_summary_propagates_coverage_and_prediction_nan(self):
        d = data()
        frame = pd.DataFrame({
            "VDS_V": d["vds"].numpy(), "VBG_V": d["vbg"].numpy(),
            "VTG_V": d["vtg"].numpy(), "ID_A": d["id"].numpy(),
            "family": "IDVBG", "source_file": "test.csv",
        })
        for prefix, col in [("physical", "ID_physical_A"), ("base", "ID_base_pkan_A"), ("residual", "ID_residual_A")]:
            frame[col] = frame.ID_A * 10 ** 0.1
            frame[f"log10_error_{prefix}_dec"] = 0.1
        metrics = engine.summarize(frame, pd.DataFrame())
        self.assertEqual(metrics["residual_iv_off_rms_count"], 0)
        self.assertEqual(metrics["residual_idvtg_high_rms_count"], 0)
        unprefixed = {k.removeprefix("residual_"): v for k,v in metrics.items() if k.startswith("residual_")}
        self.assertAlmostEqual(engine._base_iv_selection_score(unprefixed), 0.1)
        frame.loc[3, "log10_error_residual_dec"] = np.nan
        metrics = engine.summarize(frame, pd.DataFrame())
        self.assertFalse(engine._iv_metric_limit_check(metrics, "residual_iv_high_rms", 1.0)["pass"])

    def test_residual_accuracy_retains_empty_region_counts(self):
        diag = SimpleNamespace(eval=lambda: None, ids_residual=lambda batch, scale: self.base.ids(batch, scale))
        metrics = engine._iv_residual_accuracy(diag, data(), 1.0)
        self.assertEqual(metrics["iv_off_rms_count"], 0)
        self.assertTrue(math.isfinite(engine._base_iv_selection_score(metrics)))

    def test_safe_scale_selection_retains_real_improvement_without_off_samples(self):
        args = default_args()
        args.safe_scale_steps = 2
        shape = pd.DataFrame({"domain": ["IV"]})
        def predict(*a):
            return output_frame(0.4 - 0.39 * a[-2]), pd.DataFrame()
        with patch.object(engine, "predict_outputs", side_effect=predict), \
             patch.object(engine, "_diagnostic_frames", return_value=(None, None)), \
             patch.object(engine, "extrapolation_shape_diagnostics", return_value=shape), \
             patch.object(engine, "_shape_iv_fail_count", return_value=0), \
             patch.object(engine, "_shape_fail_count_relative", return_value=0), \
             patch.object(engine, "strict_cv_negative_count", return_value=0):
            scale, _, scan, _ = engine.choose_safe_residual_scales(
                None, output_frame(), pd.DataFrame(), output_frame(), pd.DataFrame(),
                torch.device("cpu"), torch.float64, args,
            )
        self.assertEqual(scale, 1.0)
        self.assertTrue(np.isfinite(scan.score).all())

    def test_release_reports_unmeasured_regions_but_rejects_populated_nan(self):
        args = default_args()
        args.require_hspice_smoke = False
        args.require_hspice_grid_equivalence = False
        args.require_hspice_cv_grid_equivalence = False
        metrics = engine.summarize(output_frame(), pd.DataFrame())
        metrics.update(task_mode="iv_only", safe_residual_scale=1.0,
            symbolic_edge_equivalence={"temperature_pkan": {"pass": True}, "iv_terminal_observable": {"pass": True}})
        scan = pd.DataFrame([{"scale": 1.0, "train_bad": 0, "extrap_bad": 0, "extrap_worse": 0}])
        report = engine._release_pass_fail(scan, pd.DataFrame(), metrics, args)
        self.assertTrue(report["pass"], report["checks"])
        self.assertEqual(report["iv_metric_checks"]["residual_iv_off_rms"]["status"], "not_applicable")
        metrics["residual_iv_high_rms"] = float("nan")
        report = engine._release_pass_fail(scan, pd.DataFrame(), metrics, args)
        self.assertFalse(report["pass"])
        self.assertFalse(report["checks"]["iv_high_absolute_accuracy"])

    def test_symbolic_audit_accepts_matching_curves_without_off_region(self):
        d = data()
        diag = SimpleNamespace(
            base=SimpleNamespace(kan=SimpleNamespace(net=None)),
            iv_resid=SimpleNamespace(net=None),
            ids_residual=lambda batch, scale: batch["id"],
        )
        with patch.object(engine, "make_iv_tensors_v2", return_value=d), \
             patch.object(engine, "EdgewiseSymbolicKANNet", return_value=None), \
             patch.object(engine, "_symbolic_top2_edge_count", return_value=0):
            report = engine._symbolic_iv_measured_curve_report(
                diag, [], [], output_frame(), torch.device("cpu"), torch.float64, default_args(),
            )
        self.assertTrue(report["pass"])
        self.assertEqual(report["regional_degradation_checks"]["iv_off_rms"]["status"], "not_applicable")

    def test_real_training_selects_trained_checkpoint_with_empty_off_region(self):
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            args = SimpleNamespace(**limits(), out_dir=Path(tmp), base_iv_epochs=3,
                base_iv_restarts=1, seed=1, dtype="float64", base_iv_lr=0.05,
                base_iv_phys_lr_mult=1.0, weight_decay=0.0, batch_iv=4,
                base_iv_small_weight=0.0, base_iv_coupling_weight=0.0,
                base_iv_sparse_weight=0.0, base_iv_top1_symbolic_weight=0.0,
                base_iv_top1_sparsity_weight=0.0, base_iv_top1_selection_weight=0.0,
                grad_clip=5.0, base_iv_eval_every=1, base_iv_min_delta=0.0,
                base_iv_early_stop_min_epochs=3, base_iv_early_stop_patience=5,
                symbolic_training_ramp_start=0.0, symbolic_training_ramp_end=1.0,
                base_iv_max_top1_compatibility=1.0)
            diag = SimpleNamespace(base=self.base, base_scale=1.0, iv_resid=TinyKAN())
            zero = lambda *a: torch.tensor(0.0, dtype=torch.float64)
            for name in ["correction_smallness", "physical_coupling_loss",
                         "_kan_top1_symbolic_compatibility_loss", "_kan_edge_group_sparsity_loss"]:
                stack.enter_context(patch.object(engine, name, side_effect=zero))
            stack.enter_context(patch.object(engine, "_symbolic_training_ramp", return_value=0.0))
            stack.enter_context(patch.object(engine, "zero_compact_kan_output_head"))
            stack.enter_context(patch.object(engine, "base_checkpoint_schema", return_value={}))
            stack.enter_context(patch.object(engine, "iv_loss", side_effect=lambda base,*a: (base.kan.weight.square(), {})))
            stack.enter_context(redirect_stdout(io.StringIO()))
            report = engine.train_current_device_base_iv_pkan(diag, data(), {}, args)
            self.assertGreater(report["selected_epoch"], 0)
            self.assertLess(report["selected_metrics"]["iv_nonzero_rms"], report["initial_metrics"]["iv_nonzero_rms"])
            self.assertTrue(all(math.isfinite(h["score"]) for h in report["history"]))
            self.assertTrue((Path(tmp)/"v93_base_pkan_current_device.pt").exists())

    def test_operator_base_limits_reach_engine_without_changing_release_limits(self):
        from src.device_modeling.dg_tft import run_model
        from src.workflow.dg_tft.param_defs import get_stage_param_table
        from src.workflow.dg_tft.schema import StageName
        repo = Path(__file__).resolve().parents[1]
        template = repo / "src/device_modeling/dg_tft/config/model_config_new_device_template.ini"
        overrides = {"base_iv_max_nonzero_rms": "0.23", "base_iv_max_nonzero_p95": "0.55",
            "base_iv_max_subthreshold_rms": "0.30", "base_iv_max_transition_rms": "0.45",
            "base_iv_max_on_rms": "0.18", "base_iv_max_high_rms": "0.15"}
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / "model.ini"
            config.write_bytes(template.read_bytes())
            cfg = run_model.load_config(config)
            cfg["full_model"].update(overrides)
            cfg["runtime"]["minimal_run"] = "no"
            paths = run_model.paths(cfg, config)
            with patch.object(run_model, "run") as execute:
                run_model.train_full_model(cfg, paths)
            command = execute.call_args.args[0]
            with patch.object(sys, "argv", ["engine", *command[2:]]):
                parsed = engine.parse_args()
            defaults = default_args()
            for key, value in overrides.items():
                self.assertEqual(getattr(parsed, key), float(value))
            for key, value in vars(defaults).items():
                if key.startswith("final_"):
                    self.assertEqual(getattr(parsed, key), value)
            self.assertTrue(set(overrides).isdisjoint(get_stage_param_table(StageName.TRAIN_STATIC)))


if __name__ == "__main__":
    unittest.main()
