# -*- coding: utf-8 -*-
"""
Phase 9 integration tests: Background execution tracking and async patterns.

Run: python -m unittest test.dg_tft_workflow_background_test -v
"""
import sys
import os
import time
import unittest
import threading
import tempfile
import shutil
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.workflow.dg_tft.schema import (
    WorkflowRun, WorkflowStatus, StageResult, StageName, DatasetProfile,
)
from src.constants import SESSION_PATH
from src.workflow.dg_tft.manager import WorkflowManager


# ────────────────────────────────────────────────────────────────────
# Test: _running_threads dict operations
# ────────────────────────────────────────────────────────────────────


class TestRunningThreadsDict(unittest.TestCase):

    def setUp(self):
        self.wm = WorkflowManager()
        self.wm._save_runs = lambda: None

    def test_running_threads_starts_empty(self):
        self.assertIsInstance(self.wm._running_threads, dict)
        self.assertEqual(len(self.wm._running_threads), 0)

    def test_register_and_remove_thread(self):
        def dummy_work():
            pass
        t = threading.Thread(target=dummy_work, daemon=True)
        run_id = "bg_run_1"
        self.wm._running_threads[run_id] = t
        self.assertIn(run_id, self.wm._running_threads)
        del self.wm._running_threads[run_id]
        self.assertNotIn(run_id, self.wm._running_threads)

    def test_multiple_threads_coexist(self):
        self.wm._running_threads["run_A"] = "thread_A"
        self.wm._running_threads["run_B"] = "thread_B"
        self.wm._running_threads["run_C"] = "thread_C"
        self.assertEqual(len(self.wm._running_threads), 3)


# ────────────────────────────────────────────────────────────────────
# Test: Concurrent WorkflowManager access
# ────────────────────────────────────────────────────────────────────


class TestConcurrentManagerAccess(unittest.TestCase):

    def setUp(self):
        self.wm = WorkflowManager()
        self.wm._save_runs = lambda: None
        self.wm.session_uuid = "test_concurrent_access"

    def test_concurrent_create_runs(self):
        results = []
        barrier = threading.Barrier(5, timeout=5)

        def create_run(i):
            barrier.wait()
            ok, rid, _ = self.wm.create_run(dataset_dir=f"/tmp/concurrent_{i}")
            results.append((ok, rid))

        threads = [threading.Thread(target=create_run, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        self.assertEqual(len(results), 5)
        all_ok = all(ok for ok, _ in results)
        self.assertTrue(all_ok, "All concurrent creates should succeed")
        ids = {rid for _, rid in results}
        self.assertEqual(len(ids), 5, "All run IDs should be unique")

    def test_concurrent_update_and_query(self):
        ok, run_id, _ = self.wm.create_run(dataset_dir="/tmp/shared")
        results = []

        def worker(i):
            self.wm.attach_metric(run_id, f"metric_{i}", float(i))
            run = self.wm.get_run(run_id)
            results.append(run is not None)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        self.assertTrue(all(results))
        run = self.wm.get_run(run_id)
        metric_count = len(run.metrics)
        self.assertEqual(metric_count, 10, f"Should have 10 metrics, got {metric_count}")

    def test_concurrent_lock_prevents_race_on_next_run_id(self):
        ids_seen = []

        def create_and_collect():
            ok, rid, _ = self.wm.create_run(dataset_dir="/tmp/race")
            ids_seen.append(rid)

        threads = [threading.Thread(target=create_and_collect) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        self.assertEqual(len(ids_seen), 20)
        self.assertEqual(len(set(ids_seen)), 20, "All run IDs must be unique")


# ────────────────────────────────────────────────────────────────────
# Test: Status transitions
# ────────────────────────────────────────────────────────────────────


class TestStatusTransitions(unittest.TestCase):

    def setUp(self):
        self.wm = WorkflowManager()
        self.wm._save_runs = lambda: None
        ok, self.run_id, _ = self.wm.create_run(dataset_dir="/tmp/status_test")

    def test_valid_transition_sequence(self):
        self.assertEqual(self.wm.get_run(self.run_id).status, WorkflowStatus.PENDING)

        self.wm.update_stage(self.run_id, StageName.CHECK_DATA, WorkflowStatus.RUNNING)
        self.assertEqual(self.wm.get_run(self.run_id).status, WorkflowStatus.RUNNING)

        self.wm.mark_done(self.run_id)
        self.assertEqual(self.wm.get_run(self.run_id).status, WorkflowStatus.DONE)

    def test_mark_done_overrides_running(self):
        self.wm.update_stage(self.run_id, StageName.FIT_PHYSICAL, WorkflowStatus.RUNNING)
        self.wm.mark_done(self.run_id)
        self.assertEqual(self.wm.get_run(self.run_id).status, WorkflowStatus.DONE)

    def test_mark_failed_overrides_running(self):
        self.wm.update_stage(self.run_id, StageName.TRAIN_STATIC, WorkflowStatus.RUNNING)
        self.wm.mark_failed(self.run_id, "oops")
        self.assertEqual(self.wm.get_run(self.run_id).status, WorkflowStatus.FAILED)

    def test_cancel_overrides_running(self):
        self.wm.update_stage(self.run_id, StageName.FIT_PHYSICAL, WorkflowStatus.RUNNING)
        self.wm.cancel_run(self.run_id)
        self.assertEqual(self.wm.get_run(self.run_id).status, WorkflowStatus.CANCELLED)

    def test_can_mark_failed_after_pending(self):
        self.wm.mark_failed(self.run_id, "immediate crash")
        self.assertEqual(self.wm.get_run(self.run_id).status, WorkflowStatus.FAILED)

    def test_mark_failed_stores_reason_in_errors(self):
        self.wm.mark_failed(self.run_id, "convergence failure")
        run = self.wm.get_run(self.run_id)
        self.assertIn("convergence failure", run.errors)

    def test_mark_failed_no_reason(self):
        self.wm.mark_failed(self.run_id)
        run = self.wm.get_run(self.run_id)
        self.assertEqual(run.status, WorkflowStatus.FAILED)

    def test_update_stage_stores_stage_result(self):
        sr = StageResult(
            ok=True, stage=StageName.CHECK_DATA,
            summary="dataset audited", metrics={"sheets": 6},
        )
        self.wm.update_stage(
            self.run_id, StageName.CHECK_DATA, WorkflowStatus.RUNNING,
            stage_result=sr,
        )
        run = self.wm.get_run(self.run_id)
        self.assertIn(StageName.CHECK_DATA, run.stage_results)
        self.assertEqual(run.stage_results[StageName.CHECK_DATA].ok, True)
        self.assertEqual(
            run.stage_results[StageName.CHECK_DATA].metrics["sheets"], 6,
        )


# ────────────────────────────────────────────────────────────────────
# Test: DatasetProfile integration
# ────────────────────────────────────────────────────────────────────


class TestDatasetProfileIntegration(unittest.TestCase):

    def setUp(self):
        self.wm = WorkflowManager()
        self.wm._save_runs = lambda: None
        ok, self.run_id, _ = self.wm.create_run(dataset_dir="/tmp/ds_profile")

    def test_set_dataset_profile_with_hysteresis(self):
        profile = DatasetProfile(static_ok=True, has_hysteresis=True)
        self.wm.set_dataset_profile(self.run_id, profile)
        run = self.wm.get_run(self.run_id)
        self.assertTrue(run.hysteresis_enabled)
        self.assertIn("dataset_profile", run.extra)

    def test_set_dataset_profile_with_temperature(self):
        profile = DatasetProfile(static_ok=True, has_temperature_iv=True)
        self.wm.set_dataset_profile(self.run_id, profile)
        run = self.wm.get_run(self.run_id)
        self.assertTrue(run.temperature_enabled)

    def test_set_dataset_profile_static_only(self):
        profile = DatasetProfile(static_ok=True)
        self.wm.set_dataset_profile(self.run_id, profile)
        run = self.wm.get_run(self.run_id)
        self.assertFalse(run.hysteresis_enabled)
        self.assertFalse(run.temperature_enabled)

    def test_set_dataset_profile_failed(self):
        profile = DatasetProfile(static_ok=False)
        profile.ok_static = False
        self.wm.set_dataset_profile(self.run_id, profile)
        run = self.wm.get_run(self.run_id)
        self.assertFalse(run.hysteresis_enabled)
        self.assertFalse(run.temperature_enabled)

    def test_set_profile_nonexistent_run(self):
        profile = DatasetProfile(static_ok=True)
        result = self.wm.set_dataset_profile(999, profile)
        self.assertFalse(result)


# ────────────────────────────────────────────────────────────────────
# Test: WorkflowManager persistence hooks
# ────────────────────────────────────────────────────────────────────


class TestPersistenceHooks(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)
        shutil.rmtree(os.path.join(SESSION_PATH, "test_persist"), ignore_errors=True)

    def test_save_load_runs_basic(self):
        wm = WorkflowManager()
        wm.session_uuid = "test_persist"
        wm.create_run(dataset_dir="/tmp/ds_save")
        wm.create_run(dataset_dir="/tmp/ds_save2")

        wm2 = WorkflowManager()
        wm2.from_dict(wm.to_dict())
        self.assertEqual(len(wm2.list_runs()), 2)

    def test_empty_persistence(self):
        wm = WorkflowManager()
        wm.session_uuid = "test_empty"
        data = wm.to_dict()
        self.assertIn("runs", data)
        self.assertEqual(len(data["runs"]), 0)


# ────────────────────────────────────────────────────────────────────
# Test: Edge cases for tool result structures
# ────────────────────────────────────────────────────────────────────


class TestToolResultEdgeCases(unittest.TestCase):

    def setUp(self):
        self.ctx = MagicMock()
        self.ctx.agent_configs = {}
        self.ctx.workflow_man = WorkflowManager()
        self.ctx.workflow_man.session_uuid = "test_edge"
        self.ctx.workflow_man._save_runs = lambda: None
        self.ctx.subagent_mute = False
        self.ctx.console = MagicMock()
        self.ctx.agent_list = {}
        self.ctx.agent_id = "test_edge_agent"
        self.ctx.mcp_router = MagicMock()
        self.ctx.mcp_router.tool_registry = {}

    def _mock_progress(self):
        p = MagicMock()
        p.console = MagicMock()
        p.console.print = MagicMock()
        return p

    def test_query_workflow_stage_results_include_metrics(self):
        wm = self.ctx.workflow_man
        ok, run_id, _ = wm.create_run(dataset_dir="/tmp/ds", output_root="/tmp/out")
        sr = StageResult(ok=True, stage=StageName.FIT_PHYSICAL,
                         summary="fit ok", metrics={"iv_rms": 0.005})
        wm.update_stage(run_id, StageName.FIT_PHYSICAL, WorkflowStatus.RUNNING,
                        stage_result=sr)
        wm.mark_done(run_id)

        from src.tool.tool_def import dg_tft_status
        result = dg_tft_status(
            {"run_id": run_id}, self.ctx, self._mock_progress(),
        )
        self.assertEqual(result["status"], "SUCCESS")
        self.assertIn("stage_results", result)

    def test_query_workflow_shows_hysteresis_enabled(self):
        wm = self.ctx.workflow_man
        ok, run_id, _ = wm.create_run(dataset_dir="/tmp/ds", hysteresis_dir="/tmp/hys")
        profile = DatasetProfile(static_ok=True, has_hysteresis=True)
        wm.set_dataset_profile(run_id, profile)

        from src.tool.tool_def import dg_tft_status
        result = dg_tft_status(
            {"run_id": run_id}, self.ctx, self._mock_progress(),
        )
        self.assertTrue(result["hysteresis_enabled"])

    @patch("src.workflow.dg_tft.data_preparer.run_verification_only")
    @patch("src.workflow.dg_tft.orchestrator.run_workflow")
    @patch("src.workflow.dg_tft.config_builder.build_runtime_config")
    @patch("src.workflow.dg_tft.dataset_inspector.inspect_dataset")
    @patch("src.tool.tool_def.ask_permission_tui")
    @patch("src.tool.tool_def.pause_for_permission")
    @patch("src.tool.tool_def.resume_from_permission")
    def test_dg_tft_run_failure_returns_success_with_details(
        self, mock_resume, mock_pause, mock_ask,
        mock_inspect, mock_build, mock_run_wf, mock_verify,
    ):
        from src.tool.tool_def import dg_tft_run

        mock_ask.return_value = (True, None)
        mock_inspect.return_value = (DatasetProfile(static_ok=True), {})
        mock_build.return_value = {"config_path": "/tmp/cfg.ini"}
        mock_run_wf.return_value = (False, "pipeline crashed")
        mock_verify.return_value = {"ok": True, "output_dir": "/tmp/ds", "issues": []}

        result = dg_tft_run(
            {"dataset_dir": "/tmp/ds", "entry": "verifier"}, self.ctx, self._mock_progress(),
        )
        self.assertEqual(result["status"], "SUCCESS")
        self.assertFalse(result.get("workflow_ok"))


if __name__ == "__main__":
    unittest.main()
