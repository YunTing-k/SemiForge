# -*- coding: utf-8 -*-
"""
Unit tests for Phase 8 functionized engine API — verify importable run_*()
entry points exist with correct signatures and defaults.

Run: python -m unittest test.dg_tft_functionized_api_test -v
"""
import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Inject engine and dynamic paths for import tests
_engine_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "src", "device_modeling", "dg_tft", "engine")
)
_dynamic_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "src", "device_modeling", "dg_tft", "dynamic")
)
for p in (_engine_path, _dynamic_path):
    if p not in sys.path:
        sys.path.insert(0, p)


def _can_import(module_name: str) -> bool:
    """Check if a module can be imported without error."""
    import importlib
    try:
        importlib.import_module(module_name)
        return True
    except Exception:
        return False


_has_engine_deps = _can_import("matplotlib")
_has_torch = _can_import("torch")
_has_dynamic_deps = _can_import("analyze_it_hysteresis")
_has_full_model_deps = _can_import("dg_tft_full_model")


class TestValidateDatasetAPI(unittest.TestCase):
    def test_run_validate_importable(self):
        from validate_dataset import run_validate
        self.assertTrue(callable(run_validate))

    def test_run_validate_signature(self):
        from validate_dataset import run_validate
        import inspect
        sig = inspect.signature(run_validate)
        params = list(sig.parameters.keys())
        self.assertIn("data_dir", params)
        self.assertIn("it_dir", params)
        self.assertIn("json_out", params)

    def test_run_validate_returns_dict_with_missing_dir(self):
        from validate_dataset import run_validate
        result = run_validate(
            data_dir=Path("/nonexistent/path/for/test"),
            it_dir=None,
            json_out=None,
        )
        self.assertIsInstance(result, dict)
        self.assertIn("ok_static", result)
        self.assertIn("ok_with_it", result)


@unittest.skipUnless(_has_engine_deps, "Engine deps (matplotlib, scipy, torch) not available in AGI env")
class TestFitPhysicalAPI(unittest.TestCase):
    def test_run_fit_physical_importable(self):
        from fit_physical_kernel import run_fit_physical
        self.assertTrue(callable(run_fit_physical))

    def test_run_fit_physical_keyword_only(self):
        from fit_physical_kernel import run_fit_physical
        import inspect
        sig = inspect.signature(run_fit_physical)
        params = list(sig.parameters.keys())
        self.assertIn("data_dir", params)
        self.assertIn("out_dir", params)
        self.assertIn("max_iv_points", params)
        self.assertIn("max_cv_points", params)
        self.assertIn("max_nfev_scale", params)
        self.assertIn("seed", params)


@unittest.skipUnless(_has_engine_deps, "Engine deps (matplotlib, scipy, torch) not available in AGI env")
class TestTrainBasePKANAPI(unittest.TestCase):
    def test_run_train_base_pkan_importable(self):
        from train_base_pkan import run_train_base_pkan
        self.assertTrue(callable(run_train_base_pkan))

    def test_run_train_base_pkan_keyword_only(self):
        from train_base_pkan import run_train_base_pkan
        import inspect
        sig = inspect.signature(run_train_base_pkan)
        params = list(sig.parameters.keys())
        self.assertIn("data_dir", params)
        self.assertIn("baseline_json", params)
        self.assertIn("out_dir", params)
        self.assertIn("epochs_shared", params)
        self.assertIn("epochs_iv", params)
        self.assertIn("epochs_cv", params)
        self.assertIn("kan_hidden", params)
        self.assertIn("kan_grid", params)
        self.assertIn("kan_k", params)


@unittest.skipUnless(_has_full_model_deps, "Full model deps not available in AGI env")
class TestFullModelAPI(unittest.TestCase):
    def test_modules_importable(self):
        from dg_tft_full_model import run_full_model, default_args, apply_quick_run_preset
        self.assertTrue(callable(run_full_model))
        self.assertTrue(callable(default_args))
        self.assertTrue(callable(apply_quick_run_preset))

    def test_default_args_returns_namespace_with_key_fields(self):
        from dg_tft_full_model import default_args
        args = default_args()
        self.assertIsNotNone(args)
        self.assertTrue(hasattr(args, "epochs"))
        self.assertTrue(hasattr(args, "lr"))
        self.assertTrue(hasattr(args, "device"))
        self.assertTrue(hasattr(args, "dtype"))
        self.assertTrue(hasattr(args, "seed"))
        self.assertTrue(hasattr(args, "out_dir"))

    def test_apply_quick_run_preset_only_when_quick_run_true(self):
        from dg_tft_full_model import default_args, apply_quick_run_preset
        args = default_args()
        args.quick_run = False
        original = args.epochs
        result = apply_quick_run_preset(args)
        self.assertEqual(result.epochs, original)

    def test_apply_quick_run_preset_reduces_epochs(self):
        from dg_tft_full_model import default_args, apply_quick_run_preset
        args = default_args()
        args.quick_run = True
        args.epochs = 700
        result = apply_quick_run_preset(args)
        self.assertLessEqual(result.epochs, 100)

    def test_default_args_matches_parse_args_shape(self):
        """default_args() should return a Namespace we can pass to run_full_model."""
        import dg_tft_full_model as engine
        defaults = engine.default_args()
        self.assertIsNotNone(defaults)
        self.assertTrue(hasattr(defaults, "model_dir"))


@unittest.skipUnless(_has_dynamic_deps, "Dynamic deps not available in AGI env")
class TestDynamicAnalyzeAPI(unittest.TestCase):
    def test_run_analyze_it_importable(self):
        from analyze_it_hysteresis import run_analyze_it
        self.assertTrue(callable(run_analyze_it))


@unittest.skipUnless(_has_torch, "Torch not available in AGI env")
class TestTrainSRHAPI(unittest.TestCase):
    def test_run_train_srh_importable(self):
        from train_srh_top2_regime_banks import run_train_srh
        self.assertTrue(callable(run_train_srh))

    def test_run_train_srh_signature(self):
        from train_srh_top2_regime_banks import run_train_srh
        import inspect
        sig = inspect.signature(run_train_srh)
        params = list(sig.parameters.keys())
        self.assertIn("target_regime", params)
        self.assertIn("epochs", params)
        self.assertIn("hidden_capture", params)
        self.assertIn("hidden_emission", params)
        self.assertIn("amp_parameterization", params)
        self.assertIn("print_every", params)


class TestExportVAAPI(unittest.TestCase):
    def test_run_export_va_importable(self):
        from export_srh_top2_regime_banked_va import run_export_va
        self.assertTrue(callable(run_export_va))

    def test_run_export_va_signature(self):
        from export_srh_top2_regime_banked_va import run_export_va
        import inspect
        sig = inspect.signature(run_export_va)
        params = list(sig.parameters.keys())
        self.assertIn("student_source", params)
        self.assertIn("include_legacy_banked", params)


class TestRunModelImportPaths(unittest.TestCase):
    def test_run_model_importable(self):
        """Verify the refactored run_model.py is importable."""
        import importlib
        mod = importlib.import_module("src.device_modeling.dg_tft.run_model")
        self.assertIsNotNone(mod)

    def test_run_model_has_all_actions(self):
        import importlib
        mod = importlib.import_module("src.device_modeling.dg_tft.run_model")
        actions = ["validate_data", "fit_physical", "train_base_pkan",
                   "train_full_model", "train_hysteresis"]
        for action in actions:
            self.assertTrue(callable(getattr(mod, action, None)),
                           f"Action '{action}' not found or not callable")

    def test_run_model_has_log_stage(self):
        import importlib
        mod = importlib.import_module("src.device_modeling.dg_tft.run_model")
        self.assertTrue(callable(mod._log_stage))

    def test_run_model_has_make_full_model_args(self):
        import importlib
        mod = importlib.import_module("src.device_modeling.dg_tft.run_model")
        self.assertTrue(callable(mod._make_full_model_args))

    def test_run_model_has_setup_dynamic_env(self):
        import importlib
        mod = importlib.import_module("src.device_modeling.dg_tft.run_model")
        self.assertTrue(callable(mod._setup_dynamic_env))

    def test_run_model_engine_paths_in_sys_path(self):
        """Verify ENGINE and DYNAMIC are added to sys.path on import."""
        import importlib
        importlib.import_module("src.device_modeling.dg_tft.run_model")
        self.assertIn(_engine_path, sys.path)
        self.assertIn(_dynamic_path, sys.path)


if __name__ == "__main__":
    unittest.main()
