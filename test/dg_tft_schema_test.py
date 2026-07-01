# -*- coding: utf-8 -*-
"""
Unit tests for DG-TFT workflow data structures (schema.py).
Run: python -m unittest test.dg_tft_schema_test
"""
import sys
import os
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.workflow.dg_tft.schema import (
    WorkflowRun,
    WorkflowStatus,
    StageResult,
    DatasetProfile,
    StageName,
    ErrorCategory,
)


class TestStageResult(unittest.TestCase):
    def test_default_values(self):
        sr = StageResult(ok=True, stage="check-data")
        self.assertEqual(sr.summary, "")
        self.assertEqual(sr.outputs, {})
        self.assertEqual(sr.metrics, {})
        self.assertEqual(sr.warnings, [])
        self.assertIsNone(sr.error)
        self.assertFalse(sr.skipped)
        self.assertEqual(sr.skip_reason, "")

    def test_to_dict_roundtrip(self):
        sr = StageResult(
            ok=True,
            stage="fit-physical",
            summary="Fitting succeeded",
            outputs={"params": "/path/params.json"},
            metrics={"rms": 0.001},
            warnings=["high residual"],
        )
        d = sr.to_dict()
        self.assertIsInstance(d, dict)
        self.assertEqual(d["ok"], True)
        self.assertEqual(d["stage"], "fit-physical")
        self.assertEqual(d["metrics"]["rms"], 0.001)

    def test_to_dict_from_dict_roundtrip(self):
        sr = StageResult(
            ok=False,
            stage="train-static",
            summary="Failed",
            error="Numerical instability",
            skipped=False,
        )
        d = sr.to_dict()
        sr2 = StageResult(**d)
        self.assertEqual(sr2.ok, False)
        self.assertEqual(sr2.stage, "train-static")
        self.assertEqual(sr2.error, "Numerical instability")

    def test_skipped_stage(self):
        sr = StageResult(
            ok=False,
            stage="train-hysteresis",
            skipped=True,
            skip_reason="No I-t data",
        )
        self.assertTrue(sr.skipped)
        self.assertEqual(sr.skip_reason, "No I-t data")


class TestWorkflowRun(unittest.TestCase):
    def setUp(self):
        self.run = WorkflowRun(
            run_id=1,
            dataset_dir="C:/data/my_device",
            output_root="C:/output/run_0001",
        )

    def test_default_status_is_pending(self):
        self.assertEqual(self.run.status, WorkflowStatus.PENDING)

    def test_default_stage_results_empty(self):
        self.assertEqual(self.run.stage_results, {})

    def test_default_artifacts_empty(self):
        self.assertEqual(self.run.artifacts, {})

    def test_default_metrics_empty(self):
        self.assertEqual(self.run.metrics, {})

    def test_mutate_and_serialize(self):
        sr = StageResult(ok=True, stage="check-data", summary="OK")
        self.run.stage_results["check-data"] = sr
        self.run.artifacts["config"] = "/path/config.ini"
        self.run.metrics["iv_rms"] = 0.005
        self.run.status = WorkflowStatus.DONE
        self.run.current_stage = "check-data"

        d = self.run.to_dict()
        self.assertEqual(d["status"], "DONE")
        self.assertIn("check-data", d["stage_results"])
        self.assertEqual(d["stage_results"]["check-data"]["ok"], True)

    def test_to_dict_from_dict_roundtrip(self):
        sr = StageResult(ok=True, stage="check-data", summary="OK")
        self.run.stage_results["check-data"] = sr
        self.run.artifacts["config"] = "/path/config.ini"
        self.run.metrics["iv_rms"] = 0.005
        self.run.status = WorkflowStatus.RUNNING
        self.run.current_stage = "check-data"

        d = self.run.to_dict()
        run2 = WorkflowRun.from_dict(d)

        self.assertEqual(run2.run_id, 1)
        self.assertEqual(run2.status, WorkflowStatus.RUNNING)
        self.assertEqual(run2.metrics["iv_rms"], 0.005)
        self.assertIn("check-data", run2.stage_results)
        self.assertTrue(run2.stage_results["check-data"].ok)

    def test_from_dict_without_stage_results(self):
        d = self.run.to_dict()
        run2 = WorkflowRun.from_dict(d)
        self.assertEqual(run2.stage_results, {})

    def test_multiple_stage_results(self):
        for stage in ["check-data", "fit-physical", "train-static"]:
            self.run.stage_results[stage] = StageResult(ok=True, stage=stage, summary="OK")
        d = self.run.to_dict()
        run2 = WorkflowRun.from_dict(d)
        self.assertEqual(len(run2.stage_results), 3)


class TestDatasetProfile(unittest.TestCase):
    def test_default_values(self):
        dp = DatasetProfile()
        self.assertFalse(dp.static_ok)
        self.assertFalse(dp.cv_ok)
        self.assertEqual(dp.iv_families, [])
        self.assertFalse(dp.has_temperature_iv)
        self.assertIsNone(dp.temperature_span_K)
        self.assertFalse(dp.has_hysteresis)

    def test_to_dict_roundtrip(self):
        dp = DatasetProfile(
            static_ok=True,
            cv_ok=True,
            iv_families=["idvbg", "idvd", "idvtg"],
            has_temperature_iv=True,
            temperature_span_K=150.0,
            has_hysteresis=True,
            hysteresis_workbooks=["I-t_1.xlsx", "I-t_2.xlsx"],
        )
        d = dp.to_dict()
        self.assertTrue(d["static_ok"])
        self.assertEqual(d["temperature_span_K"], 150.0)
        self.assertEqual(len(d["iv_families"]), 3)


class TestWorkflowStatus(unittest.TestCase):
    def test_all_statuses(self):
        self.assertIn(WorkflowStatus.PENDING, WorkflowStatus)
        self.assertIn(WorkflowStatus.RUNNING, WorkflowStatus)
        self.assertIn(WorkflowStatus.DONE, WorkflowStatus)
        self.assertIn(WorkflowStatus.FAILED, WorkflowStatus)
        self.assertIn(WorkflowStatus.CANCELLED, WorkflowStatus)

    def test_str_value(self):
        self.assertEqual(WorkflowStatus.PENDING.value, "PENDING")
        self.assertEqual(WorkflowStatus.DONE.value, "DONE")


class TestStageName(unittest.TestCase):
    def test_stage_order_length(self):
        self.assertEqual(len(StageName.STAGE_ORDER), 7)

    def test_stage_order_contains_all(self):
        expected = [
            "data-prep",
            "verifier",
            "check-data",
            "fit-physical",
            "train-static",
            "train-hysteresis",
            "final-report",
        ]
        self.assertEqual(StageName.STAGE_ORDER, expected)


class TestErrorCategory(unittest.TestCase):
    def test_all_categories(self):
        categories = [
            ErrorCategory.DATA_ERROR,
            ErrorCategory.CONFIG_ERROR,
            ErrorCategory.TRAINING_NUMERICAL_ERROR,
            ErrorCategory.MISSING_ARTIFACT_ERROR,
            ErrorCategory.HSPICE_ENV_ERROR,
            ErrorCategory.UNKNOWN_ERROR,
        ]
        self.assertEqual(len(categories), 6)


if __name__ == "__main__":
    unittest.main()
