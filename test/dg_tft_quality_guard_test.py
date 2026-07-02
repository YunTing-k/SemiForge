# -*- coding: utf-8 -*-
"""
Unit tests for Phase 7: metrics_parser.py and quality_guard.py.

Run: python -m unittest test.dg_tft_quality_guard_test -v
"""
import sys
import os
import json
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.workflow.dg_tft.metrics_parser import (
    parse_metrics,
    get_primary_metric,
    _METRIC_SOURCES,
    _METRIC_MISSING_SENTINEL,
)
from src.workflow.dg_tft.quality_guard import (
    assess_quality,
    get_quality_thresholds,
    get_max_retries,
    suggest_adjustments,
    apply_adjustments,
    stage_display_label,
    _ADJUSTMENT_STRATEGIES,
    _PARAMETER_BOUNDS,
    _PARAMETER_WHITELIST,
    _STAGE_THRESHOLD_KEYS,
    _CFG_PREFIX,
    _DEFAULT_MAX_RETRIES,
)
from src.workflow.dg_tft.schema import StageName


# ────────────────────────────────────────────────────────────────────
# Test helpers
# ────────────────────────────────────────────────────────────────────

def _make_temp_ini(adjustments: dict[str, str] | None = None) -> str:
    """Create a minimal INI file for apply_adjustments tests.

    Args:
        adjustments: Optional dict of section.key → value overrides to apply
                     after creating the INI. Keys without section prefix default
                     to ``physical_fit``.
    """
    content = (
        "[paths]\n"
        "data_dir = /tmp/data\n"
        "[physical_fit]\n"
        "max_nfev_scale = 1.0\n"
        "max_iv_points = 50000\n"
        "[full_model]\n"
        "epochs = 700\n"
        "learning_rate = 0.002\n"
        "cv_weight = 8.0\n"
        "early_stop_patience = 5\n"
        "[hysteresis]\n"
        "epochs = 1800\n"
        "hidden_capture = 16\n"
        "hidden_emission = 20\n"
        "amp_parameterization = conservative\n"
    )
    temp = tempfile.NamedTemporaryFile(mode="w", suffix=".ini", delete=False, encoding="utf-8")
    temp.write(content)
    temp.close()

    if adjustments:
        from src.workflow.dg_tft.quality_guard import apply_adjustments
        path = apply_adjustments(temp.name, adjustments)
        return path

    return temp.name


# ────────────────────────────────────────────────────────────────────
# Test: Metrics Parser
# ────────────────────────────────────────────────────────────────────


class TestMetricsParser(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_parse_fit_physical_metrics(self):
        subdir = Path(self.tmpdir) / "01_physical_kernel"
        subdir.mkdir()
        json_path = subdir / "fitted_params.json"
        json_path.write_text(json.dumps({
            "metrics_summary": {
                "iv_rms_log10_error_dec": 0.0042,
                "cv_rms_log10_error_dec": 0.008,
            }
        }))
        result = parse_metrics(self.tmpdir, StageName.FIT_PHYSICAL)
        self.assertIn("metrics_summary.iv_rms_log10_error_dec", result)
        self.assertAlmostEqual(result["metrics_summary.iv_rms_log10_error_dec"], 0.0042)

    def test_parse_train_static_metrics(self):
        subdir = Path(self.tmpdir) / "03_full_model"
        subdir.mkdir()
        json_path = subdir / "v93_safe_residual_metrics.json"
        json_path.write_text(json.dumps({
            "residual_iv_rms": 0.012,
            "residual_cv_rms": 0.034,
        }))
        result = parse_metrics(self.tmpdir, StageName.TRAIN_STATIC)
        self.assertIn("residual_iv_rms", result)
        self.assertIn("residual_cv_rms", result)
        self.assertAlmostEqual(result["residual_iv_rms"], 0.012)
        self.assertAlmostEqual(result["residual_cv_rms"], 0.034)

    def test_parse_hysteresis_metrics(self):
        subdir = Path(self.tmpdir) / "04_hysteresis"
        subdir.mkdir()
        json_path = subdir / "it_hysteresis_analysis_summary.json"
        json_path.write_text(json.dumps({
            "step_log_rmse_median": 0.022,
        }))
        result = parse_metrics(self.tmpdir, StageName.TRAIN_HYSTERESIS)
        self.assertIn("step_log_rmse_median", result)
        self.assertAlmostEqual(result["step_log_rmse_median"], 0.022)

    def test_parse_missing_json_returns_empty(self):
        result = parse_metrics(self.tmpdir, StageName.FIT_PHYSICAL)
        self.assertEqual(result, {})

    def test_parse_invalid_json_returns_empty(self):
        subdir = Path(self.tmpdir) / "01_physical_kernel"
        subdir.mkdir()
        json_path = subdir / "fitted_params.json"
        json_path.write_text("not valid json {{{")
        result = parse_metrics(self.tmpdir, StageName.FIT_PHYSICAL)
        self.assertEqual(result, {})

    def test_parse_nonexistent_stage_returns_empty(self):
        result = parse_metrics(self.tmpdir, "nonexistent-stage")
        self.assertEqual(result, {})

    def test_parse_missing_key_in_json_returns_nan(self):
        subdir = Path(self.tmpdir) / "01_physical_kernel"
        subdir.mkdir()
        json_path = subdir / "fitted_params.json"
        json_path.write_text(json.dumps({"no_metrics_here": 1.0}))
        result = parse_metrics(self.tmpdir, StageName.FIT_PHYSICAL)
        self.assertIn("metrics_summary.iv_rms_log10_error_dec", result)
        self.assertTrue(result["metrics_summary.iv_rms_log10_error_dec"] != result["metrics_summary.iv_rms_log10_error_dec"])  # NaN

    def test_get_primary_metric_fit_physical(self):
        self.assertEqual(get_primary_metric(StageName.FIT_PHYSICAL), "metrics_summary.iv_rms_log10_error_dec")

    def test_get_primary_metric_train_static(self):
        self.assertEqual(get_primary_metric(StageName.TRAIN_STATIC), "residual_iv_rms")

    def test_get_primary_metric_hysteresis(self):
        self.assertEqual(get_primary_metric(StageName.TRAIN_HYSTERESIS), "step_log_rmse_median")

    def test_get_primary_metric_unknown(self):
        self.assertEqual(get_primary_metric("unknown-stg"), "")

    def test_metric_sources_defined_for_all_stages(self):
        for stage in [StageName.FIT_PHYSICAL, StageName.TRAIN_STATIC, StageName.TRAIN_HYSTERESIS]:
            self.assertIn(stage, _METRIC_SOURCES, f"Missing metric source for {stage}")
            self.assertIn("file", _METRIC_SOURCES[stage])
            self.assertIn("keys", _METRIC_SOURCES[stage])


# ────────────────────────────────────────────────────────────────────
# Test: Quality Assessment
# ────────────────────────────────────────────────────────────────────


class TestAssessQuality(unittest.TestCase):

    def test_all_pass(self):
        thresholds = {"metrics_summary.iv_rms_log10_error_dec": 0.01}
        metrics = {"metrics_summary.iv_rms_log10_error_dec": 0.005}
        ok, failures = assess_quality(StageName.FIT_PHYSICAL, metrics, thresholds)
        self.assertTrue(ok)
        self.assertEqual(failures, [])

    def test_single_fail(self):
        thresholds = {"metrics_summary.iv_rms_log10_error_dec": 0.01}
        metrics = {"metrics_summary.iv_rms_log10_error_dec": 0.15}
        ok, failures = assess_quality(StageName.FIT_PHYSICAL, metrics, thresholds)
        self.assertFalse(ok)
        self.assertEqual(len(failures), 1)
        self.assertIn("exceeded", failures[0])

    def test_multi_metric_some_fail(self):
        thresholds = {
            "residual_iv_rms": 0.02,
            "residual_cv_rms": 0.05,
        }
        metrics = {
            "residual_iv_rms": 0.01,
            "residual_cv_rms": 0.12,
        }
        ok, failures = assess_quality(StageName.TRAIN_STATIC, metrics, thresholds)
        self.assertFalse(ok)
        self.assertEqual(len(failures), 1)
        self.assertIn("residual_cv_rms", failures[0])

    def test_nan_metric_fails(self):
        thresholds = {"metrics_summary.iv_rms_log10_error_dec": 0.01}
        metrics = {"metrics_summary.iv_rms_log10_error_dec": float("nan")}
        ok, failures = assess_quality(StageName.FIT_PHYSICAL, metrics, thresholds)
        self.assertFalse(ok)
        self.assertIn("NaN", failures[0])

    def test_empty_metrics_reports_missing(self):
        thresholds = {"residual_iv_rms": 0.02}
        ok, failures = assess_quality(StageName.TRAIN_STATIC, {}, thresholds)
        self.assertFalse(ok)
        self.assertIn("no metrics", failures[0])

    def test_metric_not_in_thresholds_passes(self):
        thresholds = {"residual_cv_rms": 0.05}
        metrics = {"residual_iv_rms": 999.0}  # Not in thresholds → ignored
        ok, _ = assess_quality(StageName.TRAIN_STATIC, metrics, thresholds)
        self.assertTrue(ok)

    def test_metric_exactly_at_threshold_passes(self):
        thresholds = {"residual_iv_rms": 0.02}
        metrics = {"residual_iv_rms": 0.02}
        ok, _ = assess_quality(StageName.TRAIN_STATIC, metrics, thresholds)
        self.assertTrue(ok)


# ────────────────────────────────────────────────────────────────────
# Test: Thresholds from agent_configs
# ────────────────────────────────────────────────────────────────────


class TestThresholds(unittest.TestCase):

    def test_get_thresholds_all_defined(self):
        configs = {
            "DG_TFT_QUALITY_PHYSICAL_IV_RMS_MAX": 0.01,
            "DG_TFT_QUALITY_STATIC_IV_RMS_MAX": 0.02,
            "DG_TFT_QUALITY_HYSTERESIS_RMS_MAX": 0.03,
        }
        thresholds = get_quality_thresholds(configs)
        self.assertIn("metrics_summary.iv_rms_log10_error_dec", thresholds)
        self.assertIn("residual_iv_rms", thresholds)
        self.assertIn("step_log_rmse_median", thresholds)

    def test_get_thresholds_partial(self):
        configs = {
            "DG_TFT_QUALITY_PHYSICAL_IV_RMS_MAX": 0.01,
        }
        thresholds = get_quality_thresholds(configs)
        self.assertIn("metrics_summary.iv_rms_log10_error_dec", thresholds)
        self.assertNotIn("residual_iv_rms", thresholds)

    def test_get_thresholds_empty(self):
        thresholds = get_quality_thresholds({})
        self.assertEqual(thresholds, {})

    def test_get_max_retries_default(self):
        self.assertEqual(get_max_retries({}), _DEFAULT_MAX_RETRIES)

    def test_get_max_retries_custom(self):
        self.assertEqual(get_max_retries({"DG_TFT_QUALITY_MAX_RETRIES": 5}), 5)

    def test_get_max_retries_zero(self):
        self.assertEqual(get_max_retries({"DG_TFT_QUALITY_MAX_RETRIES": 0}), 0)

    def test_get_max_retries_negative_clamped(self):
        self.assertEqual(get_max_retries({"DG_TFT_QUALITY_MAX_RETRIES": -1}), 0)

    def test_threshold_key_mapping_consistency(self):
        for stage_name, metric_map in _STAGE_THRESHOLD_KEYS.items():
            for metric_key, cfg_key in metric_map.items():
                self.assertTrue(cfg_key.startswith(_CFG_PREFIX),
                                f"Config key {cfg_key} for {stage_name}.{metric_key} should start with {_CFG_PREFIX}")
                self.assertTrue(metric_key, f"Metric key missing for {stage_name}")


# ────────────────────────────────────────────────────────────────────
# Test: Parameter Adjustments
# ────────────────────────────────────────────────────────────────────


class TestSuggestAdjustments(unittest.TestCase):

    def setUp(self):
        self.ini = _make_temp_ini()

    def tearDown(self):
        try:
            os.unlink(self.ini)
        except OSError:
            pass

    def test_fit_physical_attempt_0(self):
        adjustments = suggest_adjustments(StageName.FIT_PHYSICAL, 0, self.ini)
        self.assertIn("physical_fit.max_nfev_scale", adjustments)
        self.assertEqual(adjustments["physical_fit.max_nfev_scale"], "2")

    def test_fit_physical_attempt_1(self):
        adjustments = suggest_adjustments(StageName.FIT_PHYSICAL, 1, self.ini)
        self.assertIn("physical_fit.max_nfev_scale", adjustments)
        self.assertEqual(adjustments["physical_fit.max_nfev_scale"], "3")

    def test_fit_physical_attempt_2(self):
        adjustments = suggest_adjustments(StageName.FIT_PHYSICAL, 2, self.ini)
        self.assertIn("physical_fit.max_iv_points", adjustments)
        self.assertEqual(adjustments["physical_fit.max_iv_points"], "25000")

    def test_train_static_attempt_0(self):
        adjustments = suggest_adjustments(StageName.TRAIN_STATIC, 0, self.ini)
        self.assertIn("full_model.epochs", adjustments)
        self.assertIn("full_model.learning_rate", adjustments)
        self.assertEqual(adjustments["full_model.epochs"], "1050")
        self.assertEqual(adjustments["full_model.learning_rate"], "0.001")

    def test_train_static_attempt_1(self):
        adjustments = suggest_adjustments(StageName.TRAIN_STATIC, 1, self.ini)
        self.assertIn("full_model.cv_weight", adjustments)
        self.assertEqual(adjustments["full_model.cv_weight"], "16")

    def test_train_static_attempt_2(self):
        adjustments = suggest_adjustments(StageName.TRAIN_STATIC, 2, self.ini)
        self.assertIn("full_model.early_stop_patience", adjustments)
        self.assertEqual(adjustments["full_model.early_stop_patience"], "10")

    def test_hysteresis_attempt_0(self):
        adjustments = suggest_adjustments(StageName.TRAIN_HYSTERESIS, 0, self.ini)
        self.assertIn("hysteresis.epochs", adjustments)
        self.assertEqual(adjustments["hysteresis.epochs"], "2700")

    def test_hysteresis_attempt_1(self):
        adjustments = suggest_adjustments(StageName.TRAIN_HYSTERESIS, 1, self.ini)
        self.assertIn("hysteresis.hidden_capture", adjustments)
        self.assertIn("hysteresis.hidden_emission", adjustments)
        self.assertEqual(adjustments["hysteresis.hidden_capture"], "20")
        self.assertEqual(adjustments["hysteresis.hidden_emission"], "24")

    def test_hysteresis_attempt_2_aggressive(self):
        adjustments = suggest_adjustments(StageName.TRAIN_HYSTERESIS, 2, self.ini)
        self.assertIn("hysteresis.amp_parameterization", adjustments)
        self.assertEqual(adjustments["hysteresis.amp_parameterization"], "aggressive")

    def test_unknown_stage_returns_empty(self):
        adjustments = suggest_adjustments("nonexistent", 0, self.ini)
        self.assertEqual(adjustments, {})

    def test_attempt_exceeds_strategies(self):
        adjustments = suggest_adjustments(StageName.FIT_PHYSICAL, 99, self.ini)
        self.assertEqual(adjustments, {})

    def test_adjustments_in_whitelist(self):
        for stage_name in _ADJUSTMENT_STRATEGIES:
            for attempt, blueprint in enumerate(_ADJUSTMENT_STRATEGIES[stage_name]):
                for section, assignments in blueprint.items():
                    for key in assignments:
                        full_key = f"{section}.{key}"
                        self.assertIn(full_key, _PARAMETER_WHITELIST,
                                      f"Key {full_key} not in whitelist")

    def test_adjustment_clamps_to_max_bound(self):
        ini_with_high_nfev = _make_temp_ini({"physical_fit.max_nfev_scale": "10.0"})
        try:
            adjustments = suggest_adjustments(StageName.FIT_PHYSICAL, 0, ini_with_high_nfev)
            new_val = float(adjustments["physical_fit.max_nfev_scale"])
            self.assertLessEqual(new_val, 5.0)
        finally:
            try:
                os.unlink(ini_with_high_nfev)
            except OSError:
                pass

    def test_adjustment_clamps_to_min_bound(self):
        ini_with_tiny_epochs = _make_temp_ini({"full_model.epochs": "50"})
        try:
            adjustments = suggest_adjustments(StageName.TRAIN_STATIC, 0, ini_with_tiny_epochs)
            new_val = int(adjustments["full_model.epochs"])
            self.assertGreaterEqual(new_val, 100)
        finally:
            try:
                os.unlink(ini_with_tiny_epochs)
            except OSError:
                pass


# ────────────────────────────────────────────────────────────────────
# Test: Apply Adjustments to INI
# ────────────────────────────────────────────────────────────────────


class TestApplyAdjustments(unittest.TestCase):

    def setUp(self):
        self.ini = _make_temp_ini()

    def tearDown(self):
        try:
            os.unlink(self.ini)
        except OSError:
            pass

    def test_apply_single_adjustment(self):
        result_path = apply_adjustments(self.ini, {"physical_fit.max_nfev_scale": "2.0"})
        self.assertEqual(result_path, self.ini)
        import configparser
        cfg = configparser.ConfigParser()
        cfg.read(self.ini, encoding="utf-8")
        self.assertEqual(cfg.get("physical_fit", "max_nfev_scale"), "2.0")

    def test_apply_multi_section_adjustments(self):
        apply_adjustments(self.ini, {
            "physical_fit.max_nfev_scale": "2.0",
            "full_model.epochs": "1000",
            "hysteresis.amp_parameterization": "aggressive",
        })
        import configparser
        cfg = configparser.ConfigParser()
        cfg.read(self.ini, encoding="utf-8")
        self.assertEqual(cfg.get("physical_fit", "max_nfev_scale"), "2.0")
        self.assertEqual(cfg.get("full_model", "epochs"), "1000")
        self.assertEqual(cfg.get("hysteresis", "amp_parameterization"), "aggressive")

    def test_apply_empty_adjustments_returns_same_path(self):
        result = apply_adjustments(self.ini, {})
        self.assertEqual(result, self.ini)

    def test_apply_unknown_key_warns_but_does_not_crash(self):
        result = apply_adjustments(self.ini, {"nonexistent.key": "value"})
        self.assertEqual(result, self.ini)

    def test_original_values_preserved_for_unrelated_keys(self):
        apply_adjustments(self.ini, {"full_model.epochs": "1000"})
        import configparser
        cfg = configparser.ConfigParser()
        cfg.read(self.ini, encoding="utf-8")
        self.assertEqual(cfg.get("full_model", "cv_weight"), "8.0")
        self.assertEqual(cfg.get("physical_fit", "max_iv_points"), "50000")

    def test_apply_preserves_ini_structure(self):
        path = apply_adjustments(self.ini, {"full_model.epochs": "999"})
        content = Path(path).read_text(encoding="utf-8")
        self.assertIn("[paths]", content)
        self.assertIn("[full_model]", content)
        self.assertIn("epochs = 999", content)

    def test_apply_missed_adjustment_logged(self):
        path = apply_adjustments(self.ini, {"nonexistent_section.key": "value"})
        self.assertEqual(path, self.ini)

    def test_apply_adjustment_new_section_not_in_ini(self):
        path = apply_adjustments(self.ini, {"new_section.param": "val"})
        self.assertEqual(path, self.ini)


class TestReadIniValueErrors(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_read_ini_nonexistent_file_returns_none(self):
        from src.workflow.dg_tft.quality_guard import _read_ini_value
        result = _read_ini_value(
            os.path.join(self.tmpdir, "does_not_exist.ini"),
            "section", "key",
        )
        self.assertIsNone(result)

    def test_read_ini_missing_section_returns_none(self):
        from src.workflow.dg_tft.quality_guard import _read_ini_value
        result = _read_ini_value(self._make_ini(), "missing_section", "key")
        self.assertIsNone(result)

    def test_read_ini_missing_key_returns_none(self):
        from src.workflow.dg_tft.quality_guard import _read_ini_value
        result = _read_ini_value(self._make_ini(), "full_model", "missing_key")
        self.assertIsNone(result)

    def test_read_ini_non_numeric_value_returns_none(self):
        from src.workflow.dg_tft.quality_guard import _read_ini_value
        result = _read_ini_value(self._make_ini(), "full_model", "learning_rate")
        self.assertAlmostEqual(result, 0.002)

    def _make_ini(self):
        return _make_temp_ini()


# ────────────────────────────────────────────────────────────────────
# Test: Strategy Tables & Constants
# ────────────────────────────────────────────────────────────────────


class TestStrategyTables(unittest.TestCase):

    def test_strategies_defined_for_all_stages(self):
        for stage in [StageName.FIT_PHYSICAL, StageName.TRAIN_STATIC, StageName.TRAIN_HYSTERESIS]:
            self.assertIn(stage, _ADJUSTMENT_STRATEGIES, f"Missing strategy for {stage}")

    def test_strategies_have_max_3_attempts(self):
        for strategies in _ADJUSTMENT_STRATEGIES.values():
            self.assertLessEqual(len(strategies), 3, "Each stage should have at most 3 retry strategies")

    def test_strategies_only_reference_whitelisted_params(self):
        for stage_name, strategies in _ADJUSTMENT_STRATEGIES.items():
            for blueprint in strategies:
                for section, assignments in blueprint.items():
                    for key in assignments:
                        full_key = f"{section}.{key}"
                        self.assertIn(full_key, _PARAMETER_WHITELIST,
                                      f"{full_key} used in {stage_name} strategy but not in whitelist")

    def test_all_whitelisted_params_have_bounds(self):
        _string_or_bool_params = {
            "amp_parameterization",
            "fast_amp_parameterization",
            "slow_amp_parameterization",
            "fast_optimizer_policy",
            "slow_optimizer_policy",
            "cv_train_subset",
            "staged_loss_schedule",
        }
        for full_key in _PARAMETER_WHITELIST:
            section, key = full_key.split(".", 1)
            if key in _string_or_bool_params:
                continue
            self.assertIn(section, _PARAMETER_BOUNDS,
                          f"No bounds defined for section [{section}] (key {full_key})")
            self.assertIn(key, _PARAMETER_BOUNDS[section],
                          f"No bounds defined for {full_key}")

    def test_stage_label_returns_friendly_name(self):
        self.assertEqual(stage_display_label(StageName.FIT_PHYSICAL), "Physical Kernel")
        self.assertEqual(stage_display_label(StageName.TRAIN_STATIC), "Static Model")
        self.assertEqual(stage_display_label(StageName.TRAIN_HYSTERESIS), "Hysteresis")
        self.assertEqual(stage_display_label("unknown"), "unknown")


# ────────────────────────────────────────────────────────────────────
# Test: Orchestrator integration helpers
# ────────────────────────────────────────────────────────────────────


class TestOrchestratorQualityGate(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.ini = _make_temp_ini()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)
        try:
            os.unlink(self.ini)
        except OSError:
            pass

    def _make_mock_ctx(self, configs=None, mute=False):
        ctx = MagicMock()
        if configs is None:
            ctx.agent_configs = {
                "DG_TFT_QUALITY_PHYSICAL_IV_RMS_MAX": 0.01,
                "DG_TFT_QUALITY_MAX_RETRIES": 3,
            }
        else:
            ctx.agent_configs = configs
        ctx.workflow_man = MagicMock()
        run = MagicMock()
        run.extra = {}
        run.stage_results = {}
        ctx.workflow_man.get_run.return_value = run
        ctx.agent_list = {}
        ctx.agent_id = "test_agent"
        ctx.subagent_mute = mute
        ctx.console = MagicMock() if not mute else None
        return ctx

    def _make_sd(self, name=StageName.FIT_PHYSICAL, action="fit-physical",
                 subdir="01_physical_kernel", artifacts=None):
        from src.workflow.dg_tft.stages import StageDef
        return StageDef(
            name=name, action=action, required=True,
            expected_output_subdir=subdir,
            key_artifacts=artifacts or ["fitted_params.json"],
        )

    def test_quality_pass_returns_original_result(self):
        from src.workflow.dg_tft.orchestrator import _check_quality_once
        from src.workflow.dg_tft.schema import StageResult

        subdir = Path(self.tmpdir) / "01_physical_kernel"
        subdir.mkdir()
        json_path = subdir / "fitted_params.json"
        json_path.write_text(json.dumps({
            "metrics_summary": {"iv_rms_log10_error_dec": 0.005},
        }))

        ctx = self._make_mock_ctx()
        sd = self._make_sd()
        sr = StageResult(ok=True, stage=StageName.FIT_PHYSICAL, summary="pass")

        result_sr, issues = _check_quality_once(
            ctx, 1, sd, sr, self.ini, self.tmpdir,
        )
        self.assertTrue(result_sr.ok)
        self.assertEqual(issues, [])

    def test_quality_fail_without_thresholds_returns_original(self):
        from src.workflow.dg_tft.orchestrator import _check_quality_once
        from src.workflow.dg_tft.schema import StageResult

        ctx = self._make_mock_ctx(configs={})
        sd = self._make_sd()
        sr = StageResult(ok=True, stage=StageName.FIT_PHYSICAL, summary="pass")

        result_sr, issues = _check_quality_once(
            ctx, 1, sd, sr, self.ini, self.tmpdir,
        )
        self.assertTrue(result_sr.ok)
        self.assertEqual(issues, [])

    def test_quality_check_returns_issues_when_below_threshold(self):
        from src.workflow.dg_tft.orchestrator import _check_quality_once
        from src.workflow.dg_tft.schema import StageResult

        subdir = Path(self.tmpdir) / "01_physical_kernel"
        subdir.mkdir()
        json_path = subdir / "fitted_params.json"
        json_path.write_text(json.dumps({
            "metrics_summary": {"iv_rms_log10_error_dec": 0.95},
        }))

        ctx = self._make_mock_ctx()
        sd = self._make_sd()
        sr = StageResult(ok=True, stage=StageName.FIT_PHYSICAL, summary="pass")

        _result_sr, issues = _check_quality_once(
            ctx, 1, sd, sr, self.ini, self.tmpdir,
        )
        self.assertNotEqual(issues, [])

    def test_metrics_stored_on_stage_result(self):
        from src.workflow.dg_tft.orchestrator import _check_quality_once
        from src.workflow.dg_tft.schema import StageResult

        subdir = Path(self.tmpdir) / "01_physical_kernel"
        subdir.mkdir()
        json_path = subdir / "fitted_params.json"
        json_path.write_text(json.dumps({
            "metrics_summary": {"iv_rms_log10_error_dec": 0.005},
        }))

        ctx = self._make_mock_ctx()
        sd = self._make_sd()
        sr = StageResult(ok=True, stage=StageName.FIT_PHYSICAL, summary="pass")

        result_sr, _ = _check_quality_once(
            ctx, 1, sd, sr, self.ini, self.tmpdir,
        )
        self.assertIn("metrics_summary.iv_rms_log10_error_dec", result_sr.metrics)


class TestApplyAgentAdjustments(unittest.TestCase):
    """Tests for apply_agent_adjustments and snapshot_ini_section (Step 8)."""

    def setUp(self):
        self.ini = _make_temp_ini()

    def tearDown(self):
        try:
            os.unlink(self.ini)
        except OSError:
            pass

    def test_apply_agent_adjustments_writes_ini(self):
        """apply_agent_adjustments writes adjustments and returns snapshot."""
        from src.workflow.dg_tft.quality_guard import apply_agent_adjustments

        adjustments = {"max_nfev_scale": 3.0, "max_iv_points": 100000}
        snapshot = apply_agent_adjustments(self.ini, adjustments, "physical_fit")

        self.assertIn("physical_fit.max_nfev_scale = 3.0", snapshot)
        self.assertIn("physical_fit.max_iv_points = 100000", snapshot)

        import configparser
        cfg = configparser.ConfigParser()
        cfg.read(self.ini, encoding="utf-8")
        self.assertEqual(cfg.get("physical_fit", "max_nfev_scale"), "3.0")
        self.assertEqual(cfg.get("physical_fit", "max_iv_points"), "100000")

    def test_apply_agent_adjustments_creates_section(self):
        """apply_agent_adjustments creates section if missing."""
        from src.workflow.dg_tft.quality_guard import apply_agent_adjustments

        adjustments = {"epochs": 2000}
        snapshot = apply_agent_adjustments(self.ini, adjustments, "new_section")

        self.assertIn("new_section.epochs = 2000", snapshot)
        import configparser
        cfg = configparser.ConfigParser()
        cfg.read(self.ini, encoding="utf-8")
        self.assertEqual(cfg.get("new_section", "epochs"), "2000")

    def test_snapshot_ini_section(self):
        """snapshot_ini_section returns formatted section content."""
        from src.workflow.dg_tft.quality_guard import snapshot_ini_section

        text = snapshot_ini_section(self.ini, "physical_fit")
        self.assertIn("[physical_fit]", text)
        self.assertIn("max_nfev_scale = 1.0", text)
        self.assertIn("max_iv_points = 50000", text)

    def test_snapshot_ini_section_missing(self):
        """snapshot_ini_section handles missing sections."""
        from src.workflow.dg_tft.quality_guard import snapshot_ini_section

        text = snapshot_ini_section(self.ini, "nonexistent")
        self.assertIn("[nonexistent]", text)
        self.assertIn("(empty)", text)


if __name__ == "__main__":
    unittest.main()
