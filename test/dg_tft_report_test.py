# -*- coding: utf-8 -*-
"""
Unit tests for Phase 5: artifacts, report_builder, task board sync, console summary.
Run: python -m unittest test.dg_tft_report_test -v
"""
import sys
import os
import json
import unittest
import tempfile
import shutil
from unittest.mock import patch, MagicMock, call
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.workflow.dg_tft.artifacts import (
    RUN_SUMMARY_MD_NAME,
    RUN_SUMMARY_JSON_NAME,
    ARTIFACT_MANIFEST_NAME,
    ARTIFACT_KEY_DATASET_VALIDATION,
    ARTIFACT_KEY_FITTED_PARAMS,
    ARTIFACT_KEY_PKAN_MODEL,
    ARTIFACT_KEY_STATIC_MODEL_CKPT,
    ARTIFACT_KEY_STATIC_MODEL_VA,
    ARTIFACT_KEY_HYSTERESIS_VA,
    ARTIFACT_KEY_HYSTERESIS_INC,
    ARTIFACT_KEY_RUN_SUMMARY,
    ARTIFACT_KEY_ARTIFACT_MANIFEST,
    ARTIFACT_FILENAMES,
    STAGE_ARTIFACTS,
    STAGE_DISPLAY_NAMES,
    artifact_path,
    verify_stage_artifacts,
    get_report_paths,
)
from src.workflow.dg_tft.report_builder import (
    build_and_write_reports,
    _build_artifact_manifest,
    _build_json_summary,
    _build_md_summary,
)
from src.workflow.dg_tft.schema import (
    WorkflowRun, WorkflowStatus, StageResult, StageName, DatasetProfile,
)
from src.workflow.dg_tft.stages import StageDef, get_stages
from src.workflow.dg_tft.orchestrator import (
    _create_stage_tasks, _start_task, _resolve_task,
    _console_summary, _finalize_run, _get_prep_info,
)


# ────────────────────────────────────────────────────────────────────
# Test: Artifact Constants & Helpers
# ────────────────────────────────────────────────────────────────────

class TestArtifactConstants(unittest.TestCase):

    def test_report_names_defined(self):
        self.assertEqual(RUN_SUMMARY_MD_NAME, "RUN_SUMMARY.md")
        self.assertEqual(RUN_SUMMARY_JSON_NAME, "RUN_SUMMARY.json")
        self.assertEqual(ARTIFACT_MANIFEST_NAME, "ARTIFACT_MANIFEST.json")

    def test_artifact_keys_unique(self):
        keys = [
            ARTIFACT_KEY_DATASET_VALIDATION,
            ARTIFACT_KEY_FITTED_PARAMS,
            ARTIFACT_KEY_PKAN_MODEL,
            ARTIFACT_KEY_STATIC_MODEL_CKPT,
            ARTIFACT_KEY_STATIC_MODEL_VA,
            ARTIFACT_KEY_HYSTERESIS_VA,
            ARTIFACT_KEY_HYSTERESIS_INC,
            ARTIFACT_KEY_RUN_SUMMARY,
            ARTIFACT_KEY_ARTIFACT_MANIFEST,
        ]
        self.assertEqual(len(keys), len(set(keys)))

    def test_artifact_filenames_maps_all_keys(self):
        for key in [
            ARTIFACT_KEY_DATASET_VALIDATION,
            ARTIFACT_KEY_FITTED_PARAMS,
            ARTIFACT_KEY_PKAN_MODEL,
            ARTIFACT_KEY_STATIC_MODEL_CKPT,
            ARTIFACT_KEY_STATIC_MODEL_VA,
            ARTIFACT_KEY_HYSTERESIS_VA,
            ARTIFACT_KEY_HYSTERESIS_INC,
        ]:
            self.assertIn(key, ARTIFACT_FILENAMES)

    def test_stage_artifacts_has_expected_stages(self):
        expected_stages = ["check-data", "fit-physical",
                           "train-static", "train-hysteresis"]
        for s in expected_stages:
            self.assertIn(s, STAGE_ARTIFACTS)

    def test_stage_display_names_covers_stages(self):
        for s in ["check-data", "fit-physical",
                  "train-static", "train-hysteresis"]:
            self.assertIn(s, STAGE_DISPLAY_NAMES)
            self.assertIsInstance(STAGE_DISPLAY_NAMES[s], str)
            self.assertGreater(len(STAGE_DISPLAY_NAMES[s]), 0)


class TestArtifactPath(unittest.TestCase):

    def test_artifact_path_no_subdir(self):
        p = artifact_path("/out", "", "fitted_params")
        self.assertEqual(p.name, "fitted_params.json")
        self.assertTrue(str(p).startswith(str(Path("/out").resolve())))

    def test_artifact_path_with_subdir(self):
        p = artifact_path("/out", "01_physical_kernel", "fitted_params")
        self.assertTrue("01_physical_kernel" in str(p))
        self.assertEqual(p.name, "fitted_params.json")

    def test_artifact_path_unknown_key_falls_back(self):
        p = artifact_path("/out", "", "unknown_key")
        self.assertEqual(p.name, "unknown_key")


class TestVerifyStageArtifacts(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_all_missing_when_empty_dir(self):
        missing = verify_stage_artifacts(self.tmpdir, "check-data")
        self.assertEqual(len(missing), 1)
        self.assertIn("dataset_validation.json", missing[0])

    def test_all_present(self):
        base = os.path.join(self.tmpdir, "01_physical_kernel")
        os.makedirs(base, exist_ok=True)
        Path(os.path.join(base, "fitted_params.json")).write_text("{}")
        missing = verify_stage_artifacts(self.tmpdir, "fit-physical")
        self.assertEqual(missing, [])

    def test_partial_missing(self):
        base = os.path.join(self.tmpdir, "03_full_model")
        os.makedirs(base, exist_ok=True)
        Path(os.path.join(base, "static_model_checkpoint.pt")).write_text("")
        missing = verify_stage_artifacts(self.tmpdir, "train-static")
        self.assertEqual(len(missing), 1)
        self.assertIn("model.va", missing[0])

    def test_unknown_stage(self):
        missing = verify_stage_artifacts(self.tmpdir, "nonexistent")
        self.assertEqual(missing, [])


class TestGetReportPaths(unittest.TestCase):

    def test_returns_three_paths(self):
        paths = get_report_paths("/tmp/out")
        self.assertIn("md", paths)
        self.assertIn("json", paths)
        self.assertIn("manifest", paths)
        self.assertTrue(paths["md"].endswith(RUN_SUMMARY_MD_NAME))
        self.assertTrue(paths["json"].endswith(RUN_SUMMARY_JSON_NAME))
        self.assertTrue(paths["manifest"].endswith(ARTIFACT_MANIFEST_NAME))


# ────────────────────────────────────────────────────────────────────
# Test: Report Builder
# ────────────────────────────────────────────────────────────────────

class TestReportBuilder(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _make_run_data(self, status="DONE", stages_ok=True):
        return {
            "run_id": 1,
            "status": status,
            "created_at": "2026-06-25 10:00:00",
            "updated_at": "2026-06-25 10:30:00",
            "dataset_dir": "/data/test",
            "output_root": self.tmpdir,
            "prep_id": None,
            "hysteresis_enabled": False,
            "temperature_enabled": False,
            "warnings": [],
            "errors": [],
            "stage_results": {
                "check-data": {
                    "ok": True, "stage": "check-data", "skipped": False,
                    "skip_reason": "", "error": None,
                    "summary": "Dataset valid", "metrics": {"files": 10},
                    "outputs": {}, "warnings": [],
                },
                "fit-physical": {
                    "ok": stages_ok, "stage": "fit-physical", "skipped": False,
                    "skip_reason": "", "error": None if stages_ok else "convergence error",
                    "summary": "Fit done", "metrics": {"r2": 0.95},
                    "outputs": {}, "warnings": [],
                },
                "train-static": {
                    "ok": stages_ok, "stage": "train-static", "skipped": False,
                    "skip_reason": "", "error": None,
                    "summary": "Static model trained", "metrics": {"loss": 0.01},
                    "outputs": {}, "warnings": [],
                },
            },
            "artifacts": {},
            "metrics": {"final_rmse": 0.02},
        }

    def test_build_artifact_manifest_structure(self):
        manifest = _build_artifact_manifest(self._make_run_data(), self.tmpdir)
        self.assertEqual(manifest["run_id"], 1)
        self.assertIn("total_items", manifest)
        self.assertIn("missing_items", manifest)
        self.assertIn("items", manifest)
        self.assertIsInstance(manifest["items"], list)
        self.assertGreater(len(manifest["items"]), 0)
        for item in manifest["items"]:
            self.assertIn("stage", item)
            self.assertIn("artifact_key", item)
            self.assertIn("filename", item)
            self.assertIn("path", item)
            self.assertIn("exists", item)

    def test_build_manifest_marks_existing_files(self):
        base = os.path.join(self.tmpdir, "01_physical_kernel")
        os.makedirs(base, exist_ok=True)
        Path(os.path.join(base, "fitted_params.json")).write_text("{}")
        run_data = self._make_run_data()
        manifest = _build_artifact_manifest(run_data, self.tmpdir)
        fp_item = [it for it in manifest["items"] if it["artifact_key"] == "fitted_params"]
        self.assertEqual(len(fp_item), 1)
        self.assertTrue(fp_item[0]["exists"])
        self.assertGreater(fp_item[0]["size_bytes"], 0)

    def test_build_json_summary_structure(self):
        run_data = self._make_run_data()
        manifest = _build_artifact_manifest(run_data, self.tmpdir)
        summary = _build_json_summary(run_data, manifest, None)
        self.assertEqual(summary["run_id"], 1)
        self.assertEqual(summary["status"], "DONE")
        self.assertEqual(len(summary["stages"]), 3)
        self.assertEqual(summary["stages"][0]["display_name"], "Dataset Audit")

    def test_build_json_summary_includes_prep_info(self):
        run_data = self._make_run_data()
        manifest = _build_artifact_manifest(run_data, self.tmpdir)
        prep = {"ok": True, "issues": ["low-quality IV"]}
        summary = _build_json_summary(run_data, manifest, prep)
        self.assertIn("preparation", summary)
        self.assertTrue(summary["preparation"]["ok"])
        self.assertIn("low-quality IV", summary["preparation"]["issues"])

    def test_build_json_summary_failed_run(self):
        run_data = self._make_run_data(status="FAILED", stages_ok=False)
        manifest = _build_artifact_manifest(run_data, self.tmpdir)
        summary = _build_json_summary(run_data, manifest, None)
        self.assertEqual(summary["status"], "FAILED")
        self.assertEqual(summary["errors_count"], 0)

    def test_build_md_summary_has_expected_sections(self):
        run_data = self._make_run_data()
        manifest = _build_artifact_manifest(run_data, self.tmpdir)
        md = _build_md_summary(run_data, manifest, None)
        self.assertIn("# DG-TFT Modeling Run #1", md)
        self.assertIn("## 1. Input", md)
        self.assertIn("## 2. Stage Results", md)
        self.assertIn("## 3. Artifacts", md)
        self.assertIn("[PASS]", md)

    def test_build_md_summary_failed_shows_error(self):
        run_data = self._make_run_data(stages_ok=False)
        manifest = _build_artifact_manifest(run_data, self.tmpdir)
        md = _build_md_summary(run_data, manifest, None)
        self.assertIn("[FAIL]", md)
        self.assertIn("convergence error", md)

    def test_build_md_summary_skipped_stage(self):
        run_data = self._make_run_data()
        run_data["stage_results"]["train-hysteresis"] = {
            "ok": True, "stage": "train-hysteresis", "skipped": True,
            "skip_reason": "No I-t data", "error": None,
            "summary": "", "metrics": {}, "outputs": {}, "warnings": [],
        }
        manifest = _build_artifact_manifest(run_data, self.tmpdir)
        md = _build_md_summary(run_data, manifest, None)
        self.assertIn("[SKIPPED]", md)
        self.assertIn("No I-t data", md)

    def test_build_and_write_reports_writes_files(self):
        run_data = self._make_run_data()
        ok = build_and_write_reports(run_data, self.tmpdir)
        self.assertTrue(ok)
        self.assertTrue(Path(self.tmpdir, RUN_SUMMARY_MD_NAME).exists())
        self.assertTrue(Path(self.tmpdir, RUN_SUMMARY_JSON_NAME).exists())
        self.assertTrue(Path(self.tmpdir, ARTIFACT_MANIFEST_NAME).exists())

    def test_build_and_write_reports_json_is_valid(self):
        run_data = self._make_run_data()
        build_and_write_reports(run_data, self.tmpdir)
        with open(Path(self.tmpdir, RUN_SUMMARY_JSON_NAME), "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual(data["run_id"], 1)

    def test_build_and_write_reports_manifest_is_valid(self):
        run_data = self._make_run_data()
        build_and_write_reports(run_data, self.tmpdir)
        with open(Path(self.tmpdir, ARTIFACT_MANIFEST_NAME), "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertIn("items", data)
        self.assertGreaterEqual(len(data["items"]), 1)


# ────────────────────────────────────────────────────────────────────
# Test: Task Board Helpers
# ────────────────────────────────────────────────────────────────────

class TestTaskBoardHelpers(unittest.TestCase):

    def setUp(self):
        self.mock_board = MagicMock()
        self.agent_id = "test_agent"

    def test_create_stage_tasks_no_board(self):
        stages = list(get_stages())
        task_map = _create_stage_tasks(None, self.agent_id, stages, 1)
        self.assertEqual(task_map, {})

    def test_create_stage_tasks_calls_board(self):
        self.mock_board.create_task.return_value = (True, "Task with id 42 created")
        stages = list(get_stages())[:2]
        task_map = _create_stage_tasks(self.mock_board, self.agent_id, stages, 1)
        self.mock_board.create_task.assert_called()
        self.assertIn(StageName.CHECK_DATA, task_map)

    def test_create_stage_tasks_skips_noop_stages(self):
        self.mock_board.create_task.return_value = (True, "Task with id 1 created")
        stages = [StageDef(name=StageName.FINAL_REPORT, action="final-report", required=False)]
        task_map = _create_stage_tasks(self.mock_board, self.agent_id, stages, 1)
        self.assertEqual(task_map, {})

    def test_start_task_none(self):
        _start_task(self.mock_board, self.agent_id, None)
        self.mock_board.update_task.assert_not_called()

    def test_start_task_no_board(self):
        _start_task(None, self.agent_id, 1)
        # Should not raise

    def test_start_task_calls_update(self):
        _start_task(self.mock_board, self.agent_id, 42)
        self.mock_board.update_task.assert_called_once()
        update = self.mock_board.update_task.call_args[0][0]
        self.assertEqual(update["task_id"], 42)
        self.assertEqual(update["if_claim"], True)
        self.assertEqual(update["requester"], self.agent_id)

    def test_resolve_task_completed(self):
        _resolve_task(self.mock_board, self.agent_id, 42, completed=True)
        self.mock_board.update_task.assert_called_once()
        update = self.mock_board.update_task.call_args[0][0]
        from src.tool.scoreboard import TaskStatus
        self.assertEqual(update["status"], TaskStatus.COMPLETED)

    def test_resolve_task_failed(self):
        _resolve_task(self.mock_board, self.agent_id, 42, completed=False)
        self.mock_board.update_task.assert_called_once()
        update = self.mock_board.update_task.call_args[0][0]
        from src.tool.scoreboard import TaskStatus
        self.assertEqual(update["status"], TaskStatus.DELETED)

    def test_resolve_task_none(self):
        _resolve_task(self.mock_board, self.agent_id, None, completed=True)
        self.mock_board.update_task.assert_not_called()

    def test_resolve_task_no_board(self):
        _resolve_task(None, self.agent_id, 42, completed=True)
        # Should not raise


# ────────────────────────────────────────────────────────────────────
# Test: Console Summary
# ────────────────────────────────────────────────────────────────────

class MockCtx:
    def __init__(self, wm, console=None):
        self.workflow_man = wm
        self.console = console
        self.agent_id = "test_agent"
        self.subagent_mute = False


class TestConsoleSummary(unittest.TestCase):

    def setUp(self):
        from src.workflow.dg_tft.manager import WorkflowManager
        self.tmpdir = tempfile.mkdtemp()
        self.wm = WorkflowManager()
        self.wm.session_uuid = "test_session"
        self.wm._save_runs = lambda: None

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_console_summary_success_writes_output(self):
        self.wm.create_run("/data/test", output_root=self.tmpdir)
        mock_console = MagicMock()
        ctx = MockCtx(self.wm, console=mock_console)
        _console_summary(ctx, 1, self.tmpdir, failed=False)
        output = str(mock_console.print.call_args_list)
        self.assertIn("COMPLETED", output)

    def test_console_summary_failure_writes_output(self):
        self.wm.create_run("/data/test", output_root=self.tmpdir)
        mock_console = MagicMock()
        ctx = MockCtx(self.wm, console=mock_console)
        _console_summary(ctx, 1, self.tmpdir, failed=True,
                         failed_stage="fit-physical", failed_error="test error")
        output = str(mock_console.print.call_args_list)
        self.assertIn("FAILED", output)
        self.assertIn("test error", output)

    def test_console_summary_muted(self):
        self.wm.create_run("/data/test", output_root=self.tmpdir)
        mock_console = MagicMock()
        ctx = MockCtx(self.wm, console=mock_console)
        ctx.subagent_mute = True
        _console_summary(ctx, 1, self.tmpdir, failed=False)
        mock_console.print.assert_not_called()

    def test_get_prep_info_no_prep_id(self):
        ctx = MockCtx(self.wm)
        self.assertIsNone(_get_prep_info(ctx, None))

    def test_get_prep_info_valid(self):
        tracker = self.wm.preparation_tracker
        tracker.register_prep("/raw", "/prepared")
        tracker.update_prep(1, True, ["minor issue"])
        ctx = MockCtx(self.wm)
        info = _get_prep_info(ctx, 1)
        self.assertIsNotNone(info)
        self.assertTrue(info["ok"])
        self.assertIn("minor issue", info["issues"])


# ────────────────────────────────────────────────────────────────────
# Test: Orchestrator Finalize with Reports
# ────────────────────────────────────────────────────────────────────

class TestOrchestratorFinalize(unittest.TestCase):

    def setUp(self):
        from src.workflow.dg_tft.manager import WorkflowManager
        self.tmpdir = tempfile.mkdtemp()
        self.wm = WorkflowManager()
        self.wm.session_uuid = "test_session"
        self.wm._save_runs = lambda: None

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_finalize_run_success_writes_reports(self):
        self.wm.create_run("/data/test", output_root=self.tmpdir)
        stages = list(get_stages())
        mock_console = MagicMock()
        ctx = MockCtx(self.wm, console=mock_console)
        ctx.agent_configs = {}
        _finalize_run(ctx, 1, self.tmpdir, 0.0, stages, failed=False)
        self.assertTrue(Path(self.tmpdir, RUN_SUMMARY_MD_NAME).exists())
        self.assertTrue(Path(self.tmpdir, RUN_SUMMARY_JSON_NAME).exists())
        self.assertTrue(Path(self.tmpdir, ARTIFACT_MANIFEST_NAME).exists())

    def test_finalize_run_failure_writes_reports(self):
        self.wm.create_run("/data/test", output_root=self.tmpdir)
        stages = list(get_stages())
        mock_console = MagicMock()
        ctx = MockCtx(self.wm, console=mock_console)
        ctx.agent_configs = {}
        _finalize_run(ctx, 1, self.tmpdir, 0.0, stages,
                      failed=True, failed_stage="fit-physical", failed_error="oops")
        self.assertTrue(Path(self.tmpdir, RUN_SUMMARY_MD_NAME).exists())
        run = self.wm.get_run(1)
        self.assertEqual(run.status, WorkflowStatus.FAILED)

    def test_finalize_run_muted_still_writes_reports(self):
        self.wm.create_run("/data/test", output_root=self.tmpdir)
        stages = list(get_stages())
        mock_console = MagicMock()
        ctx = MockCtx(self.wm, console=mock_console)
        ctx.subagent_mute = True
        ctx.agent_configs = {}
        _finalize_run(ctx, 1, self.tmpdir, 0.0, stages, failed=False)
        self.assertTrue(Path(self.tmpdir, RUN_SUMMARY_MD_NAME).exists())


if __name__ == "__main__":
    unittest.main()
