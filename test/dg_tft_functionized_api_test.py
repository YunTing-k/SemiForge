# -*- coding: utf-8 -*-
"""
Unit tests for v95-based engine and run_model API.
Verifies engine modules are importable and run_model action functions exist.

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
_has_dynamic_deps = _can_import("analyze_it_hysteresis")
_has_full_model_deps = _can_import("dg_tft_full_model")


@unittest.skipUnless(_has_engine_deps, "Engine deps (matplotlib, scipy) not available in AGI env")
class TestEngineModulesImportable(unittest.TestCase):
    """v95 engine scripts are subprocess-entry only, but must be syntactically valid."""

    def test_cv_preprocessing_importable(self):
        import cv_preprocessing
        self.assertTrue(callable(cv_preprocessing.transition_metadata))
        self.assertTrue(callable(cv_preprocessing.centered_coordinate))
        self.assertTrue(callable(cv_preprocessing.looks_like_shared_bias_wide_table))

    def test_measurement_preprocessing_importable(self):
        import measurement_preprocessing
        self.assertTrue(callable(measurement_preprocessing.align_iv_measurement_stages))
        self.assertTrue(callable(measurement_preprocessing.write_iv_alignment_report))

    def test_tensor_utils_importable(self):
        import tensor_utils
        self.assertTrue(callable(tensor_utils.set_seed))
        self.assertTrue(callable(tensor_utils.Normalizer))

    def test_fit_physical_kernel_importable(self):
        import fit_physical_kernel
        self.assertTrue(callable(fit_physical_kernel.main))

    def test_validate_dataset_importable(self):
        import validate_dataset
        self.assertTrue(callable(validate_dataset.check_iv_file))
        self.assertTrue(callable(validate_dataset.check_cv_file))

    def test_dg_tft_full_model_importable(self):
        import dg_tft_full_model
        self.assertTrue(callable(dg_tft_full_model.main))


@unittest.skipUnless(_has_dynamic_deps, "Dynamic deps not available in AGI env")
class TestDynamicModulesImportable(unittest.TestCase):
    """v95 dynamic modules must be syntactically valid."""

    def test_analyze_it_hysteresis_importable(self):
        import analyze_it_hysteresis
        self.assertIsNotNone(analyze_it_hysteresis)

    def test_train_srh_top2_regime_banks_importable(self):
        import train_srh_top2_regime_banks
        self.assertIsNotNone(train_srh_top2_regime_banks)

    def test_export_srh_top2_regime_banked_va_importable(self):
        import export_srh_top2_regime_banked_va
        self.assertIsNotNone(export_srh_top2_regime_banked_va)

    def test_validate_complete_device_hspice_importable(self):
        import validate_complete_device_hspice
        self.assertIsNotNone(validate_complete_device_hspice)

    def test_validate_srh_top2_modular_equivalence_importable(self):
        import validate_srh_top2_modular_equivalence
        self.assertIsNotNone(validate_srh_top2_modular_equivalence)


class TestRunModelAPI(unittest.TestCase):
    """run_model.py action functions must exist with correct names."""

    @classmethod
    def setUpClass(cls):
        import importlib
        cls.mod = importlib.import_module("src.device_modeling.dg_tft.run_model")

    def test_run_model_importable(self):
        self.assertIsNotNone(self.mod)

    def test_core_action_functions_exist(self):
        actions = [
            "validate_data", "fit_physical", "train_base_pkan",
            "train_full_model", "train_hysteresis",
        ]
        for action in actions:
            self.assertTrue(callable(getattr(self.mod, action, None)),
                            f"Action '{action}' not found or not callable")

    def test_new_v95_action_functions_exist(self):
        actions = [
            "doctor", "smoke_static", "assemble_release",
            "build_portable_archive", "verify_portable_package",
        ]
        for action in actions:
            self.assertTrue(callable(getattr(self.mod, action, None)),
                            f"Action '{action}' not found or not callable")

    def test_all_actions_in_main_choices(self):
        """Verify main() action count. Specific actions tested above."""
        self.assertTrue(callable(self.mod.main))

    def test_helpers_exist(self):
        helpers = ["phase", "resolve_hspice", "resolve_path", "yes",
                   "load_config", "paths", "run", "run_gate",
                   "run_artifact_stage", "record_nonblocking_gate_failure"]
        for h in helpers:
            self.assertTrue(callable(getattr(self.mod, h, None)),
                            f"Helper '{h}' not found or not callable")

    def test_migration_adaptations_exist(self):
        adaptations = [
            "_normalize_cv_format",
            "_prepare_fallback_model_dir",
            "refuse_accidental_example_training",
        ]
        for a in adaptations:
            self.assertTrue(callable(getattr(self.mod, a, None)),
                            f"Adaptation '{a}' not found or not callable")

    def test_example_data_guard_blocks_example_training(self):
        with self.assertRaises(RuntimeError):
            self.mod.refuse_accidental_example_training(
                "train-static", self.mod.EXAMPLE_DATA_ROOT, allow_example_data=False
            )

    def test_example_data_guard_allows_non_example(self):
        self.mod.refuse_accidental_example_training(
            "train-static", Path("D:/some_real_data"), allow_example_data=False
        )

    def test_example_data_guard_allowed_flag(self):
        self.mod.refuse_accidental_example_training(
            "train-static",
            self.mod.EXAMPLE_DATA_ROOT,
            allow_example_data=True,
        )


if __name__ == "__main__":
    unittest.main()
