# -*- coding: utf-8 -*-
"""
Unit tests for WorkflowManager (direct methods, without session persistence).
Run: python -m unittest test.dg_tft_manager_test
"""
import sys
import os
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.workflow.dg_tft.manager import WorkflowManager
from src.workflow.dg_tft.schema import (
    WorkflowStatus,
    StageResult,
    DatasetProfile,
)


class TestWorkflowManagerCreate(unittest.TestCase):
    def setUp(self):
        self.wm = WorkflowManager()

    def test_create_run(self):
        ok, rid, ts = self.wm.create_run("C:/data/my_device")
        self.assertTrue(ok)
        self.assertEqual(rid, 1)
        self.assertIsNotNone(ts)

    def test_create_multiple_runs_auto_increment(self):
        self.wm.create_run("C:/data/A")
        self.wm.create_run("C:/data/B")
        self.wm.create_run("C:/data/C")
        self.assertEqual(len(self.wm.list_runs()), 3)
        self.assertIsNotNone(self.wm.get_run(1))
        self.assertIsNotNone(self.wm.get_run(3))
        self.assertIsNone(self.wm.get_run(4))

    def test_create_run_with_device_name(self):
        ok, rid, _ = self.wm.create_run("C:/data/d", device_name="my_tft")
        run = self.wm.get_run(rid)
        self.assertEqual(run.extra.get("device_name"), "my_tft")

    def test_create_run_with_hysteresis_dir(self):
        ok, rid, _ = self.wm.create_run("C:/data/d", hysteresis_dir="C:/data/d/it")
        run = self.wm.get_run(rid)
        self.assertEqual(run.hysteresis_dir, "C:/data/d/it")

    def test_create_run_with_output_root(self):
        ok, rid, _ = self.wm.create_run("C:/data/d", output_root="C:/output/test")
        run = self.wm.get_run(rid)
        self.assertEqual(run.output_root, "C:/output/test")

    def test_create_run_default_status_is_pending(self):
        ok, rid, _ = self.wm.create_run("C:/data/d")
        run = self.wm.get_run(rid)
        self.assertEqual(run.status, WorkflowStatus.PENDING)


class TestWorkflowManagerMutation(unittest.TestCase):
    def setUp(self):
        self.wm = WorkflowManager()
        self.wm.create_run("C:/data/test")

    def test_update_stage(self):
        sr = StageResult(ok=True, stage="check-data", summary="OK")
        ok = self.wm.update_stage(1, "check-data", WorkflowStatus.RUNNING, sr)
        self.assertTrue(ok)
        run = self.wm.get_run(1)
        self.assertEqual(run.current_stage, "check-data")
        self.assertEqual(run.status, WorkflowStatus.RUNNING)
        self.assertIn("check-data", run.stage_results)

    def test_update_stage_nonexistent(self):
        ok = self.wm.update_stage(999, "check-data", WorkflowStatus.RUNNING)
        self.assertFalse(ok)

    def test_attach_artifact(self):
        ok = self.wm.attach_artifact(1, "config", "/path/config.ini")
        self.assertTrue(ok)
        run = self.wm.get_run(1)
        self.assertEqual(run.artifacts["config"], "/path/config.ini")

    def test_attach_artifact_nonexistent(self):
        ok = self.wm.attach_artifact(999, "config", "/path/config.ini")
        self.assertFalse(ok)

    def test_attach_metric(self):
        ok = self.wm.attach_metric(1, "iv_rms", 0.001)
        self.assertTrue(ok)
        run = self.wm.get_run(1)
        self.assertAlmostEqual(run.metrics["iv_rms"], 0.001)

    def test_attach_metric_nonexistent(self):
        ok = self.wm.attach_metric(999, "iv_rms", 0.001)
        self.assertFalse(ok)

    def test_mark_done(self):
        ok = self.wm.mark_done(1)
        self.assertTrue(ok)
        run = self.wm.get_run(1)
        self.assertEqual(run.status, WorkflowStatus.DONE)

    def test_mark_done_nonexistent(self):
        ok = self.wm.mark_done(999)
        self.assertFalse(ok)

    def test_mark_failed(self):
        ok = self.wm.mark_failed(1, "Numerical error")
        self.assertTrue(ok)
        run = self.wm.get_run(1)
        self.assertEqual(run.status, WorkflowStatus.FAILED)
        self.assertIn("Numerical error", run.errors)

    def test_mark_failed_nonexistent(self):
        ok = self.wm.mark_failed(999)
        self.assertFalse(ok)

    def test_cancel_run(self):
        ok = self.wm.cancel_run(1)
        self.assertTrue(ok)
        run = self.wm.get_run(1)
        self.assertEqual(run.status, WorkflowStatus.CANCELLED)

    def test_add_warning(self):
        ok = self.wm.add_warning(1, "HSPICE not available")
        self.assertTrue(ok)
        run = self.wm.get_run(1)
        self.assertIn("HSPICE not available", run.warnings)

    def test_set_dataset_profile(self):
        dp = DatasetProfile(
            static_ok=True,
            cv_ok=True,
            has_temperature_iv=True,
            temperature_span_K=100.0,
            has_hysteresis=True,
        )
        ok = self.wm.set_dataset_profile(1, dp)
        self.assertTrue(ok)
        run = self.wm.get_run(1)
        self.assertTrue(run.temperature_enabled)
        self.assertTrue(run.hysteresis_enabled)
        self.assertIn("dataset_profile", run.extra)


class TestWorkflowManagerSerialization(unittest.TestCase):
    def test_to_dict_from_dict_roundtrip(self):
        wm = WorkflowManager()
        wm.create_run("C:/data/A")
        wm.create_run("C:/data/B")
        sr = StageResult(ok=True, stage="check-data", summary="OK")
        wm.update_stage(1, "check-data", WorkflowStatus.RUNNING, sr)
        wm.attach_artifact(1, "config", "/path/cfg.ini")
        wm.attach_metric(1, "iv_rms", 0.005)
        wm.mark_done(1)

        d = wm.to_dict()
        wm2 = WorkflowManager()
        wm2.from_dict(d)

        self.assertEqual(len(wm2.list_runs()), 2)
        run = wm2.get_run(1)
        self.assertEqual(run.status, WorkflowStatus.DONE)
        self.assertIn("config", run.artifacts)
        self.assertAlmostEqual(run.metrics["iv_rms"], 0.005)
        self.assertIn("check-data", run.stage_results)

        run2 = wm2.get_run(2)
        self.assertEqual(run2.status, WorkflowStatus.PENDING)

    def test_from_dict_empty(self):
        wm = WorkflowManager()
        wm.from_dict({"next_run_id": 1, "runs": {}})
        self.assertEqual(len(wm.list_runs()), 0)


class TestWorkflowManagerListRuns(unittest.TestCase):
    def test_list_runs_empty(self):
        wm = WorkflowManager()
        self.assertEqual(wm.list_runs(), [])

    def test_list_runs_summary(self):
        wm = WorkflowManager()
        wm.create_run("C:/data/A")
        wm.create_run("C:/data/B")

        runs = wm.list_runs()
        self.assertEqual(len(runs), 2)
        self.assertIn("run_id", runs[0])
        self.assertIn("status", runs[0])
        self.assertIn("dataset_dir", runs[0])
        self.assertIn("current_stage", runs[0])
        self.assertIn("created_at", runs[0])


if __name__ == "__main__":
    unittest.main()
