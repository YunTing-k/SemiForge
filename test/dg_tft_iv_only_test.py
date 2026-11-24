"""Regression tests for input-derived IV-only modeling and Agent constraints.

Run workflow tests in the Agent environment. Engine tests require the modeling
environment (PyTorch, scipy, pandas and openpyxl).
"""
from __future__ import annotations

import importlib.util
import json
import os
import contextlib
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from src.workflow.dg_tft.dataset_inspector import _build_profile
from src.workflow.dg_tft.metrics_parser import parse_metrics
from src.workflow.dg_tft.param_defs import (
    get_stage_param_schema,
    get_stage_param_table,
    get_stage_sections,
    get_static_task_mode,
)
from src.workflow.dg_tft.quality_guard import assess_quality, validate_agent_adjustments
from src.workflow.dg_tft.report_builder import _model_scope
from src.workflow.dg_tft.schema import StageName


class TestIVOnlyWorkflow(unittest.TestCase):
    def test_profile_distinguishes_absent_and_invalid_cv(self):
        for present, ok, expected in [(False, False, "iv_only"), (True, False, "iv_cv"), (True, True, "iv_cv")]:
            with self.subTest(present=present, ok=ok):
                raw = {"cv": {"exists": present, "ok": ok}, "ok_static": not present or ok}
                profile = _build_profile(Path("."), raw, None)
                self.assertEqual(profile.task_mode, expected)
                self.assertEqual(profile.static_ok, not present or ok)

    def test_schema_and_hard_guard_both_reject_cv_adjustments(self):
        stage = StageName.TRAIN_STATIC
        table = get_stage_param_table(stage, task_mode="iv_only")
        schema = get_stage_param_schema(stage, task_mode="iv_only")
        properties = schema["function"]["parameters"]["properties"]
        for name in ["cv_weight", "batch_cv", "device_cv_finetune_steps", "enable_cv_residual"]:
            self.assertNotIn(name, properties)
        self.assertIn("kan_scale", table)
        accepted, warnings = validate_agent_adjustments(
            {"cv_weight": 1.0, "learning_rate": 0.001}, table, get_stage_sections(stage), stage,
        )
        self.assertEqual(accepted, {"learning_rate": 0.001})
        self.assertTrue(warnings)
        self.assertIn("cv_weight", get_stage_param_table(stage))

    def test_quality_omits_only_explicitly_inapplicable_cv(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "03_full_model"
            output.mkdir()
            metrics_file = output / "v93_safe_residual_metrics.json"
            limits = {"residual_iv_rms": 0.1, "residual_cv_rms": 0.1}
            for mode, expected_pass in [("iv_only", True), ("iv_cv", False)]:
                metrics_file.write_text(json.dumps({"task_mode": mode, "residual_iv_rms": 0.05}))
                metrics = parse_metrics(tmp, StageName.TRAIN_STATIC)
                self.assertEqual(assess_quality(StageName.TRAIN_STATIC, metrics, limits)[0], expected_pass)
            metrics_file.write_text(json.dumps({"task_mode": "iv_only"}))
            self.assertFalse(assess_quality(StageName.TRAIN_STATIC, parse_metrics(tmp, StageName.TRAIN_STATIC), limits)[0])

    def test_legacy_config_uses_actual_cv_presence(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            config = base / "run.ini"
            config.write_text(f"[paths]\ndata_dir = {base}\n")
            self.assertEqual(get_static_task_mode(str(config)), "iv_only")
            # Presence is enough here; malformed input must be repaired, not skipped.
            (base / "CV_train.xlsx").write_bytes(b"invalid workbook")
            self.assertEqual(get_static_task_mode(str(config)), "iv_cv")

    def test_report_does_not_claim_validated_capacitance(self):
        scope = _model_scope({"extra": {"dataset_profile": {"cv_present": False, "cv_ok": False}}})
        self.assertEqual(scope["cv_status"], "not_applicable")
        self.assertIn("uncalibrated", scope["limitation"])


@unittest.skipUnless(importlib.util.find_spec("torch"), "Requires the modeling environment")
class TestIVOnlyEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        engine_dir = Path(__file__).resolve().parents[1] / "src/device_modeling/dg_tft/engine"
        sys.path.insert(0, str(engine_dir))
        import fit_physical_kernel
        import dg_tft_full_model
        import validate_dataset
        cls.physical = fit_physical_kernel
        cls.engine = dg_tft_full_model
        cls.validator = validate_dataset

    def test_absent_cv_has_valid_empty_tensor_interface(self):
        import torch
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(self.physical.load_cv_data(Path(tmp)), [])
        frame = self.engine.cv_point_dataframe([], self.physical.CVTrapParams())
        tensors = self.engine.make_cv_tensors_v2(frame, torch.device("cpu"), torch.float64)
        self.assertEqual(tensors["vbg"].numel(), 0)
        self.assertIn("temp_K", tensors)

    def test_present_malformed_cv_is_rejected(self):
        import pandas as pd
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "CV_train.xlsx"
            pd.DataFrame({"not_capacitance": [1, 2]}).to_excel(target, index=False)
            self.assertFalse(self.validator.check_cv_file(target)["ok"])
            with self.assertRaisesRegex(ValueError, "Invalid CV workbook"):
                self.physical.load_cv_data(Path(tmp))

    def test_valid_cv_remains_supported(self):
        import pandas as pd
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "CV_train.xlsx"
            pd.DataFrame([["description", None], ["V", "VDS=0V T=300K 1kHz"], [-1, 1e-12], [0, 2e-12], [1, 3e-12]]).to_excel(
                target, sheet_name="cgg", header=False, index=False,
            )
            self.assertTrue(self.validator.check_cv_file(target)["ok"])
            self.assertEqual(len(self.physical.load_cv_data(Path(tmp))), 1)

    def test_iv_physical_fit_does_not_optimize_fringe_capacitance(self):
        import pandas as pd
        frame = pd.DataFrame({"family": ["IDVBG", "IDVTG", "IDVD"], "VDS_V": [0.1, 0.1, 1.0]})
        calls = []

        def run_stage(name, params, names, *args):
            calls.append((name, names))
            return params, {"stage": name}

        with patch.object(self.physical, "run_stage", side_effect=run_stage):
            self.physical.fit_model(frame, [], 0.1, False)
        self.assertEqual(len(calls), 4)
        self.assertTrue(all(not name.startswith("Cfr_") for _, names in calls for name in names))
        with patch.object(self.physical, "run_stage", side_effect=run_stage):
            self.physical.fit_model(frame, [object()], 0.1, False)
        self.assertEqual(len(calls), 9)
        self.assertIn("Cfr_tgs", calls[-1][1])


@unittest.skipUnless(os.environ.get("DEVICEAGENT_RUN_ENGINE_SMOKE") == "1", "Opt-in reduced-budget engine integration")
class TestStaticEngineSmoke(unittest.TestCase):
    """Run real fitting/export in temporary directories, without an LLM or HSPICE.

    Set DEVICEAGENT_RUN_ENGINE_SMOKE=1 in the modeling environment. Precision
    gates are explicitly bypassed for these tiny budgets; no accuracy claim is
    made. CV routines must not be called at all in an IV-only run.
    """

    def _run(self, *, temperature: bool, cv: bool, export: bool):
        import numpy as np
        import pandas as pd
        import torch
        engine_dir = Path(__file__).resolve().parents[1] / "src/device_modeling/dg_tft/engine"
        sys.path.insert(0, str(engine_dir))
        import fit_physical_kernel as physical
        import dg_tft_full_model as engine
        from src.device_modeling.dg_tft import run_model

        with tempfile.TemporaryDirectory(prefix="deviceagent_static_test_") as tmp:
            root = Path(tmp)
            data = root / "data"
            data.mkdir()
            for family in ["IDVBG", "IDVTG", "IDVD"]:
                rows = []
                for temp in ([293.15, 353.15] if temperature else [300.0]):
                    for bias in [0.1, 1.0, 4.0]:
                        for x in np.linspace(0.05 if family == "IDVD" else -8.0, 8.0, 32):
                            rows.append(dict(
                                family=family, source_file=family, temp_K=temp,
                                VBG_V=x if family == "IDVBG" else (bias if family == "IDVD" else 0.0),
                                VTG_V=x if family == "IDVTG" else 0.0,
                                VDS_V=x if family == "IDVD" else bias,
                            ))
                frame = pd.DataFrame(rows)
                frame["ID_A"] = physical.predict_iv(frame, physical.Params())
                if not temperature:
                    frame = frame.drop(columns="temp_K")
                frame.rename(columns={"VDS_V": "VD_V"}).to_excel(data / f"{family}_train.xlsx", sheet_name="long", index=False)
            if cv:
                with pd.ExcelWriter(data / "CV_train.xlsx") as writer:
                    for sheet in ["cgg", "cbg", "ctg"]:
                        rows = [["description", None], ["V", "VTG=0V VBG=0V VDS=0V 1kHz"]]
                        rows += [[x, 1e-12 + 1e-12 / (1 + np.exp(-x))] for x in np.linspace(-8, 8, 16)]
                        pd.DataFrame(rows).to_excel(writer, sheet_name=sheet, header=False, index=False)
            paths = {
                "data": data, "physical": root / "01_physical_kernel", "kan_architecture": root / "architecture",
                "full": root / "03_full_model", "output": root, "it": root / "it",
            }
            cfg = run_model.load_config(engine_dir.parent / "config/model_config.ini")
            cfg["runtime"]["minimal_run"] = "yes"
            cfg["base_pkan"] = {"kan_hidden": "4", "kan_grid": "3", "kan_k": "2"}
            with contextlib.redirect_stdout(io.StringIO()):
                run_model.validate_data(cfg, root / "run.ini", paths)
                run_model.fit_physical(cfg, paths)
                run_model._prepare_fallback_model_dir(cfg, paths)
            validation = json.loads((root / "dataset_validation.json").read_text())
            self.assertTrue(validation["ok_static"])
            self.assertEqual(validation["task_mode"], "iv_cv" if cv else "iv_only")
            with patch.object(sys, "argv", ["smoke"]):
                args = engine.parse_args()
            settings = dict(
                data_dir_override=data, model_dir=paths["kan_architecture"],
                baseline_json=paths["physical"] / "fitted_params.json", out_dir=paths["full"],
                device="cpu", dtype="float64", reset_base_from_physical_fit=True,
                epochs=2, base_iv_epochs=2, base_iv_restarts=1, base_iv_eval_every=1,
                base_iv_early_stop_min_epochs=1, iv_residual_stage_epochs=2, batch_iv=64, batch_cv=16,
                eval_iv_curves=2, deriv_iv_curves=2, idvd_deriv_curves=2, iv_residual_stage_deriv_curves=2,
                eval_every=1, print_every=1, early_stop_min_epochs=1, safe_scale_steps=1,
                symbolic_training_weight=0.0, symbolic_training_sparsity_weight=0.0,
                symbolic_edge_steps=1, symbolic_edge_finetune_steps=1, symbolic_network_finetune_steps=1,
                symbolic_device_iv_finetune_steps=1, symbolic_device_iv_measured_steps=1,
                symbolic_edge_sample_points=9, symbolic_network_finetune_points=128, symbolic_equiv_points=64,
                iv_symbolic_calibration_steps=1, temperature_kan_hidden=4, temperature_kan_grid=3,
                temperature_kan_k=2, auto_enable_temperature_kan=True,
                skip_hspice_smoke=True, skip_hspice_grid_equivalence=True, skip_hspice_cv_grid_equivalence=True,
                skip_precision_gates=True, training_only=not export, skip_post_train_cv_polish=True,
                init_commercial_cv_broadening=False, init_cv_sheet_calibration=False,
            )
            for name, value in settings.items():
                setattr(args, name, value)
            observed = {}
            original_train = engine.train_residuals

            def checked_train(diag, *train_args):
                temp_before = diag.temp_mu_ea_raw.detach().clone()
                cv_before = {name: value.detach().clone() for name, value in diag.named_parameters()
                             if name.startswith(("cv_", "trap_", "base.cv_", "base.cgg_"))}
                original_train(diag, *train_args)
                observed["temperature_kan_enabled"] = diag.temperature_kan_enabled
                observed["temperature_updated"] = not torch.equal(temp_before, diag.temp_mu_ea_raw)
                if not cv:
                    for name, value in diag.named_parameters():
                        if name in cv_before:
                            self.assertTrue(torch.equal(cv_before[name], value), name)
                    self.assertEqual(float(diag.temperature_kan.out_range[-1]), 0.0)

            with contextlib.ExitStack() as stack:
                stack.enter_context(patch.object(engine, "parse_args", return_value=args))
                stack.enter_context(patch.object(engine, "train_residuals", side_effect=checked_train))
                if not cv:
                    for name in ["initialize_commercial_cv_broadening", "initialize_cv_reciprocal_charge_from_raw",
                                 "initialize_cv_sheet_calibration", "polish_cv_neural_corrections",
                                 "choose_safe_cv_parameter_scale", "choose_safe_joint_cv_scales",
                                 "_device_level_finetune_cv_symbolic", "_symbolic_cv_observable_equivalence_report"]:
                        stack.enter_context(patch.object(engine, name, side_effect=AssertionError(f"Unexpected CV operation: {name}")))
                stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                engine.main()
            self.assertEqual(observed["temperature_kan_enabled"], temperature)
            if temperature:
                self.assertTrue(observed["temperature_updated"])
            metric_name = "v93_safe_residual_metrics.json" if export else "training_only_metrics.json"
            metrics = json.loads((paths["full"] / metric_name).read_text())
            self.assertTrue(np.isfinite(metrics["residual_iv_rms"]))
            self.assertEqual("residual_cv_rms" in metrics, cv)
            if export:
                manifest = json.loads((paths["full"] / "v93_release_manifest.json").read_text())
                self.assertEqual(manifest["cv_status"], "not_applicable")
                self.assertTrue(manifest["edgewise_symbolic"]["enabled"])
                self.assertEqual(manifest["edgewise_symbolic"]["network_equivalence"]["cv_terminal_observable"]["status"], "not_applicable")
                va = paths["full"] / manifest["standalone_veriloga"]["standalone_va_file"]
                self.assertIn("uncalibrated", va.read_text())
                self.assertTrue((paths["full"] / "v93_safe_residual_state.pt").exists())

    def test_iv_temperature_full_export(self):
        self._run(temperature=True, cv=False, export=True)

    def test_iv_without_temperature_training(self):
        self._run(temperature=False, cv=False, export=False)

    def test_joint_iv_cv_training(self):
        self._run(temperature=False, cv=True, export=False)


if __name__ == "__main__":
    unittest.main()
