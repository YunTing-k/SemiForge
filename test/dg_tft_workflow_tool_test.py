# -*- coding: utf-8 -*-
"""
Phase 9 integration tests: Agent tool dispatch with task board linkage.

Run: python -m unittest test.dg_tft_workflow_tool_test -v
"""
import sys
import os
import unittest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.workflow.dg_tft.schema import (
    WorkflowRun, WorkflowStatus, StageResult, StageName, DatasetProfile,
)
from src.workflow.dg_tft.manager import WorkflowManager
from src.constants import (
    TOOL_NAME_START_WORKFLOW,
    TOOL_NAME_QUERY_WORKFLOW,
    TOOL_NAME_READ_WORKFLOW_REPORT,
    TOOL_NAME_PREPARE_DATASET,
    SUCCESS_LABEL,
    FAIL_LABEL,
    DENIED_LABEL,
)


def _make_mock_progress():
    p = MagicMock()
    p.console = MagicMock()
    p.console.print = MagicMock()
    return p


def _make_mock_context(configs=None):
    ctx = MagicMock()
    ctx.agent_configs = configs or {"DG_TFT_RUNS_SUBDIR": "runs"}
    ctx.workflow_man = WorkflowManager()
    ctx.workflow_man.session_uuid = "test_integration"
    ctx.workflow_man._save_runs = lambda: None
    ctx.subagent_mute = False
    ctx.console = MagicMock()
    ctx.agent_list = {}
    ctx.agent_id = "main_test_agent"
    ctx.mcp_router = MagicMock()
    ctx.mcp_router.tool_registry = {}
    ctx.session_uuid = "test_integration"
    return ctx


# ────────────────────────────────────────────────────────────────────
# Test: Task board linkage with workflow tools
# ────────────────────────────────────────────────────────────────────


class TestTaskBoardLinkage(unittest.TestCase):

    def setUp(self):
        self.ctx = _make_mock_context()

    @patch("src.workflow.dg_tft.orchestrator.run_workflow")
    @patch("src.workflow.dg_tft.config_builder.build_runtime_config")
    @patch("src.workflow.dg_tft.dataset_inspector.inspect_dataset")
    @patch("src.tool.tool_def.ask_permission_tui")
    @patch("src.tool.tool_def.pause_for_permission")
    @patch("src.tool.tool_def.resume_from_permission")
    def test_start_workflow_fails_without_dataset_dir(
        self, mock_resume, mock_pause, mock_ask,
        mock_inspect, mock_build, mock_run_wf,
    ):
        from src.tool.tool_def import start_dg_tft_workflow

        mock_ask.return_value = (True, None)
        result = start_dg_tft_workflow(
            {}, self.ctx, _make_mock_progress(),
        )
        self.assertEqual(result["status"], FAIL_LABEL)

    @patch("src.workflow.dg_tft.orchestrator.run_workflow")
    @patch("src.workflow.dg_tft.config_builder.build_runtime_config")
    @patch("src.workflow.dg_tft.dataset_inspector.inspect_dataset")
    @patch("src.tool.tool_def.ask_permission_tui")
    @patch("src.tool.tool_def.pause_for_permission")
    @patch("src.tool.tool_def.resume_from_permission")
    def test_start_workflow_board_not_required(
        self, mock_resume, mock_pause, mock_ask,
        mock_inspect, mock_build, mock_run_wf,
    ):
        from src.tool.tool_def import start_dg_tft_workflow

        mock_ask.return_value = (True, None)
        mock_inspect.return_value = (DatasetProfile(static_ok=True), {})
        mock_build.return_value = {"config_path": "/tmp/cfg.ini"}
        mock_run_wf.return_value = (True, "all done")

        result = start_dg_tft_workflow(
            {"dataset_dir": "/tmp/ds"}, self.ctx, _make_mock_progress(),
            board=None,
        )
        self.assertEqual(result["status"], SUCCESS_LABEL)
        self.assertIn("run_id", result)

    @patch("src.workflow.dg_tft.orchestrator.run_workflow")
    @patch("src.workflow.dg_tft.config_builder.build_runtime_config")
    @patch("src.workflow.dg_tft.dataset_inspector.inspect_dataset")
    @patch("src.tool.tool_def.ask_permission_tui")
    @patch("src.tool.tool_def.pause_for_permission")
    @patch("src.tool.tool_def.resume_from_permission")
    def test_start_workflow_with_overrides(
        self, mock_resume, mock_pause, mock_ask,
        mock_inspect, mock_build, mock_run_wf,
    ):
        from src.tool.tool_def import start_dg_tft_workflow

        mock_ask.return_value = (True, None)
        mock_inspect.return_value = (DatasetProfile(static_ok=True), {})
        mock_build.return_value = {"config_path": "/tmp/cfg.ini"}
        mock_run_wf.return_value = (True, "all done")

        overrides = {"full_model.epochs": "1000", "physical_fit.max_nfev_scale": "2.0"}
        result = start_dg_tft_workflow(
            {"dataset_dir": "/tmp/ds", "overrides": overrides},
            self.ctx, _make_mock_progress(),
        )
        self.assertEqual(result["status"], SUCCESS_LABEL)

    def test_prepare_dataset_schema(self):
        from src.tool.tool_def import tool_prepare_dg_tft_dataset_def
        schema = tool_prepare_dg_tft_dataset_def()
        fn = schema["function"]
        self.assertEqual(fn["name"], TOOL_NAME_PREPARE_DATASET)
        self.assertIn("raw_dataset_dir", fn["parameters"]["properties"])
        self.assertIn("raw_dataset_dir", fn["parameters"]["required"])
        self.assertEqual(
            fn["parameters"]["properties"]["raw_dataset_dir"]["type"], "string"
        )

    @patch("src.tool.tool_def.ask_permission_tui")
    @patch("src.tool.tool_def.pause_for_permission")
    @patch("src.tool.tool_def.resume_from_permission")
    def test_prepare_dataset_permission_denied(
        self, mock_resume, mock_pause, mock_ask,
    ):
        from src.tool.tool_def import prepare_dg_tft_dataset

        mock_ask.return_value = (False, "permission denied by user")
        result = prepare_dg_tft_dataset(
            {"raw_dataset_dir": "/tmp/raw_data"}, self.ctx, _make_mock_progress(),
        )
        self.assertEqual(result["status"], DENIED_LABEL)

    def test_query_prep_not_found(self):
        from src.tool.tool_def import query_dg_tft_workflow
        result = query_dg_tft_workflow(
            {"prep_id": 99999}, self.ctx, _make_mock_progress(),
        )
        self.assertEqual(result["status"], FAIL_LABEL)
        self.assertIn("not found", result["info"].lower())

    def test_query_prep_found(self):
        tracker = self.ctx.workflow_man.preparation_tracker
        tracker.register_prep("/tmp/raw", "/tmp/out")
        tracker.update_prep(1, True, ["minor warning"])

        from src.tool.tool_def import query_dg_tft_workflow
        result = query_dg_tft_workflow(
            {"prep_id": 1}, self.ctx, _make_mock_progress(),
        )
        self.assertEqual(result["status"], SUCCESS_LABEL)
        self.assertTrue(result["ok"])
        self.assertIn("minor warning", result["issues"])

    def test_query_neither_run_id_nor_prep_id(self):
        from src.tool.tool_def import query_dg_tft_workflow
        result = query_dg_tft_workflow(
            {}, self.ctx, _make_mock_progress(),
        )
        self.assertEqual(result["status"], FAIL_LABEL)

    def test_read_report_with_stage_results(self):
        wm = self.ctx.workflow_man
        ok, run_id, _ = wm.create_run(
            dataset_dir="/tmp/rds", output_root="/tmp/out",
        )
        wm.update_stage(run_id, StageName.CHECK_DATA, WorkflowStatus.RUNNING)
        sr = StageResult(ok=True, stage=StageName.CHECK_DATA,
                         summary="dataset valid", metrics={"files": 5})
        wm.update_stage(run_id, StageName.CHECK_DATA, WorkflowStatus.RUNNING,
                        stage_result=sr)
        wm.add_warning(run_id, "low sample count")
        wm.mark_done(run_id)

        from src.tool.tool_def import read_dg_tft_workflow_report
        result = read_dg_tft_workflow_report(
            {"run_id": run_id}, self.ctx, _make_mock_progress(),
        )
        self.assertEqual(result["status"], SUCCESS_LABEL)
        self.assertIn("warnings", result)
        self.assertIn("low sample count", result["warnings"])

    def test_read_report_failed_run(self):
        wm = self.ctx.workflow_man
        ok, run_id, _ = wm.create_run(
            dataset_dir="/tmp/fail_ds", output_root="/tmp/out",
        )
        wm.mark_failed(run_id, "pipeline crashed at fit-physical")

        from src.tool.tool_def import read_dg_tft_workflow_report
        result = read_dg_tft_workflow_report(
            {"run_id": run_id}, self.ctx, _make_mock_progress(),
        )
        self.assertEqual(result["status"], SUCCESS_LABEL)
        self.assertEqual(result["workflow_status"], "FAILED")


# ────────────────────────────────────────────────────────────────────
# Test: WorkflowManager concurrency / lock safety
# ────────────────────────────────────────────────────────────────────


class TestWorkflowManagerConcurrency(unittest.TestCase):

    def setUp(self):
        self.wm = WorkflowManager()
        self.wm._save_runs = lambda: None
        self.wm.session_uuid = "test_concurrent"

    def test_create_multiple_runs_safe(self):
        ids = set()
        for i in range(10):
            ok, run_id, _ = self.wm.create_run(dataset_dir=f"/tmp/ds_{i}")
            self.assertTrue(ok)
            ids.add(run_id)
        self.assertEqual(len(ids), 10)

    def test_update_stage_on_nonexistent_run(self):
        result = self.wm.update_stage(999, "check-data", WorkflowStatus.RUNNING)
        self.assertFalse(result)

    def test_attach_artifact_nonexistent(self):
        result = self.wm.attach_artifact(999, "key", "/tmp/path")
        self.assertFalse(result)

    def test_attach_metric_nonexistent(self):
        result = self.wm.attach_metric(999, "r2", 0.95)
        self.assertFalse(result)

    def test_mark_done_nonexistent(self):
        result = self.wm.mark_done(999)
        self.assertFalse(result)

    def test_mark_failed_nonexistent(self):
        result = self.wm.mark_failed(999)
        self.assertFalse(result)

    def test_cancel_run_nonexistent(self):
        result = self.wm.cancel_run(999)
        self.assertFalse(result)

    def test_add_warning_nonexistent(self):
        result = self.wm.add_warning(999, "warning")
        self.assertFalse(result)

    def test_get_run_nonexistent(self):
        self.assertIsNone(self.wm.get_run(999))

    def test_list_runs_empty(self):
        runs = self.wm.list_runs()
        self.assertIsInstance(runs, list)
        self.assertEqual(len(runs), 0)

    def test_list_runs_with_entries(self):
        self.wm.create_run(dataset_dir="/tmp/a")
        self.wm.create_run(dataset_dir="/tmp/b")
        runs = self.wm.list_runs()
        self.assertEqual(len(runs), 2)

    def test_full_lifecycle(self):
        ok, run_id, _ = self.wm.create_run(dataset_dir="/tmp/lifecycle")
        self.assertEqual(self.wm.get_run(run_id).status, WorkflowStatus.PENDING)

        self.wm.update_stage(run_id, StageName.CHECK_DATA, WorkflowStatus.RUNNING)
        self.assertEqual(self.wm.get_run(run_id).status, WorkflowStatus.RUNNING)
        self.assertEqual(self.wm.get_run(run_id).current_stage, StageName.CHECK_DATA)

        self.wm.attach_artifact(run_id, "config_path", "/tmp/config.ini")
        self.assertIn("config_path", self.wm.get_run(run_id).artifacts)

        self.wm.attach_metric(run_id, "iv_rms", 0.01)
        self.assertIn("iv_rms", self.wm.get_run(run_id).metrics)
        self.assertEqual(self.wm.get_run(run_id).metrics["iv_rms"], 0.01)

        self.wm.add_warning(run_id, "test warning")
        self.assertIn("test warning", self.wm.get_run(run_id).warnings)

        self.wm.mark_done(run_id)
        self.assertEqual(self.wm.get_run(run_id).status, WorkflowStatus.DONE)

    def test_failure_lifecycle(self):
        ok, run_id, _ = self.wm.create_run(dataset_dir="/tmp/fail_life")
        self.wm.update_stage(run_id, StageName.FIT_PHYSICAL, WorkflowStatus.RUNNING)
        self.wm.mark_failed(run_id, "divergence")
        self.assertEqual(self.wm.get_run(run_id).status, WorkflowStatus.FAILED)
        self.assertIn("divergence", self.wm.get_run(run_id).errors)

    def test_cancel_lifecycle(self):
        ok, run_id, _ = self.wm.create_run(dataset_dir="/tmp/cancel_life")
        self.wm.update_stage(run_id, StageName.CHECK_DATA, WorkflowStatus.RUNNING)
        self.wm.cancel_run(run_id)
        self.assertEqual(self.wm.get_run(run_id).status, WorkflowStatus.CANCELLED)


# ────────────────────────────────────────────────────────────────────
# Test: Serialization round-trips
# ────────────────────────────────────────────────────────────────────


class TestSerializationRoundTrip(unittest.TestCase):

    def setUp(self):
        self.wm = WorkflowManager()
        self.wm._save_runs = lambda: None
        self.wm.session_uuid = "test_serial"

    def test_empty_wm_round_trip(self):
        data = self.wm.to_dict()
        wm2 = WorkflowManager()
        wm2.from_dict(data)
        self.assertIn("runs", data)
        self.assertEqual(len(data["runs"]), 0)

    def test_wm_with_runs_round_trip(self):
        self.wm.create_run(dataset_dir="/tmp/ds1", device_name="DeviceA")
        self.wm.create_run(dataset_dir="/tmp/ds2", device_name="DeviceB")
        self.wm.update_stage(1, StageName.CHECK_DATA, WorkflowStatus.RUNNING)
        self.wm.mark_done(1)
        self.wm.mark_failed(2, "error")

        data = self.wm.to_dict()
        wm2 = WorkflowManager()
        wm2.from_dict(data)
        self.assertEqual(len(wm2.list_runs()), 2)
        self.assertEqual(wm2.get_run(1).status, WorkflowStatus.DONE)
        self.assertEqual(wm2.get_run(2).status, WorkflowStatus.FAILED)

    def test_to_dict_includes_preparation_tracker(self):
        tracker = self.wm.preparation_tracker
        tracker.register_prep("/tmp/raw", "/tmp/out")
        tracker.update_prep(1, True, [])

        data = self.wm.to_dict()
        self.assertIn("preparation_tracker", data)
        self.assertIn("preps", data["preparation_tracker"])

        wm2 = WorkflowManager()
        wm2.from_dict(data)
        prep = wm2.preparation_tracker.get_prep(1)
        self.assertIsNotNone(prep)
        self.assertTrue(prep["ok"])

    def test_stage_result_dict_format(self):
        sr = StageResult(
            ok=False, stage=StageName.TRAIN_STATIC,
            summary="training failed", error="nan at epoch 42",
            metrics={"loss": float("inf")}, warnings=["low data"],
        )
        d = sr.to_dict()
        self.assertEqual(d["ok"], False)
        self.assertEqual(d["stage"], StageName.TRAIN_STATIC)
        self.assertEqual(d["error"], "nan at epoch 42")
        self.assertIn("low data", d["warnings"])
        self.assertEqual(d["metrics"]["loss"], float("inf"))

    def test_workflow_run_round_trip(self):
        run = WorkflowRun(
            run_id=5, dataset_dir="/tmp/x", status=WorkflowStatus.RUNNING,
            current_stage=StageName.FIT_PHYSICAL,
            hysteresis_enabled=True, temperature_enabled=False,
            artifacts={"config": "/tmp/config.ini"}, metrics={"r2": 0.95},
            warnings=["test warn"], errors=["test err"],
            stage_results={
                StageName.CHECK_DATA: StageResult(ok=True, stage=StageName.CHECK_DATA),
            },
        )
        d = run.to_dict()
        self.assertEqual(d["run_id"], 5)
        self.assertEqual(d["status"], "RUNNING")
        self.assertEqual(d["current_stage"], StageName.FIT_PHYSICAL)
        self.assertTrue(d["hysteresis_enabled"])
        self.assertFalse(d["temperature_enabled"])
        self.assertIn("config", d["artifacts"])
        self.assertIn("r2", d["metrics"])
        self.assertIn("test warn", d["warnings"])


# ────────────────────────────────────────────────────────────────────
# Test: PreparationTracker operations
# ────────────────────────────────────────────────────────────────────


class TestPreparationTracker(unittest.TestCase):

    def setUp(self):
        from src.workflow.dg_tft.manager import PreparationTracker
        self.tracker = PreparationTracker()

    def test_register_prep_returns_incrementing_ids(self):
        id1 = self.tracker.register_prep("/tmp/r1", "/tmp/o1")
        id2 = self.tracker.register_prep("/tmp/r2", "/tmp/o2")
        self.assertEqual(id1, 1)
        self.assertEqual(id2, 2)

    def test_get_prep_nonexistent(self):
        self.assertIsNone(self.tracker.get_prep(999))

    def test_update_prep_sets_fields(self):
        pid = self.tracker.register_prep("/tmp/r", "/tmp/o")
        self.tracker.update_prep(pid, True, ["warning1", "warning2"])
        prep = self.tracker.get_prep(pid)
        self.assertTrue(prep["ok"])
        self.assertEqual(len(prep["issues"]), 2)
        self.assertIn("warning1", prep["issues"])

    def test_update_prep_nonexistent_silent(self):
        self.tracker.update_prep(999, True, [])

    def test_serialization_round_trip(self):
        self.tracker.register_prep("/tmp/r1", "/tmp/o1")
        self.tracker.update_prep(1, False, ["bad data"])

        from src.workflow.dg_tft.manager import PreparationTracker
        tracker2 = PreparationTracker()
        tracker2.from_dict(self.tracker.to_dict())

        prep = tracker2.get_prep(1)
        self.assertIsNotNone(prep)
        self.assertFalse(prep["ok"])
        self.assertEqual(tracker2._next_prep_id, self.tracker._next_prep_id)


# ────────────────────────────────────────────────────────────────────
# Test: Dispatch fallback behavior
# ────────────────────────────────────────────────────────────────────


class TestDispatchFallback(unittest.TestCase):

    def setUp(self):
        self.ctx = _make_mock_context()

    def test_undefined_tool_returns_fail(self):
        from src.tool.tool_dispatch import call_tools
        from src.tool.scoreboard import Scoreboard

        result, addon = call_tools(
            "non_existent_tool_name", {},
            self.ctx, Scoreboard(), _make_mock_progress(),
        )
        self.assertEqual(result["status"], FAIL_LABEL)
        self.assertIn("undefined", result["info"].lower())

    def test_query_with_non_numeric_run_id(self):
        from src.tool.tool_def import query_dg_tft_workflow
        result = query_dg_tft_workflow(
            {"run_id": "not_an_int"}, self.ctx, _make_mock_progress(),
        )
        self.assertEqual(result["status"], FAIL_LABEL)
        self.assertIn("not found", result["info"].lower())

    def test_read_report_with_bad_run_id_type(self):
        from src.tool.tool_def import read_dg_tft_workflow_report
        result = read_dg_tft_workflow_report(
            {"run_id": "not_an_int"}, self.ctx, _make_mock_progress(),
        )
        self.assertEqual(result["status"], FAIL_LABEL)


# ────────────────────────────────────────────────────────────────────
# Test: Prompt routing priorities
# ────────────────────────────────────────────────────────────────────


class TestPromptRoutingIntegration(unittest.TestCase):

    def test_routing_prompt_mentions_workflow_before_manual_bash(self):
        from src.context.prompt import get_agent_guideline_prompts
        prompts = get_agent_guideline_prompts()
        content = prompts[0]["content"]

        self.assertIn("DG-TFT Workflow Routing", content)
        self.assertIn(TOOL_NAME_START_WORKFLOW, content)
        workflow_index = content.index("DG-TFT Workflow Routing")
        manual_index = content.index("instead of manually")
        self.assertLess(workflow_index, manual_index,
                        "Workflow routing section should appear before manual bash reminder")

    def test_prompt_references_all_four_workflow_tools(self):
        from src.context.prompt import get_agent_guideline_prompts
        prompts = get_agent_guideline_prompts()
        content = prompts[0]["content"]

        for tool_name in [TOOL_NAME_START_WORKFLOW, TOOL_NAME_QUERY_WORKFLOW,
                          TOOL_NAME_READ_WORKFLOW_REPORT, TOOL_NAME_PREPARE_DATASET]:
            self.assertIn(tool_name, content,
                          f"Prompt should reference {tool_name}")

    def test_prompt_advises_against_manual_bash(self):
        from src.context.prompt import get_agent_guideline_prompts
        prompts = get_agent_guideline_prompts()
        content = prompts[0]["content"]
        self.assertIn("instead of manually", content.lower())


if __name__ == "__main__":
    unittest.main()
