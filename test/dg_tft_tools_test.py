# -*- coding: utf-8 -*-
"""
Unit tests for Phase 4: Agent tool integration (tools, dispatch, commands, prompts).
Run: python -m unittest test.dg_tft_tools_test -v
"""
import sys
import os
import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.workflow.dg_tft.schema import (
    WorkflowRun,
    WorkflowStatus,
    StageResult,
    StageName,
)
from src.workflow.dg_tft.manager import WorkflowManager
from src.constants import (
    TOOL_NAME_START_WORKFLOW,
    TOOL_NAME_QUERY_WORKFLOW,
    TOOL_NAME_READ_WORKFLOW_REPORT,
    SUCCESS_LABEL,
    FAIL_LABEL,
    DENIED_LABEL,
)


class MockContext:
    def __init__(self, configs=None):
        self.agent_configs = configs or {"DG_TFT_OUTPUT_ROOT": "/tmp/dg_tft_out"}
        self.workflow_man = WorkflowManager()
        self.workflow_man.session_uuid = "test_session_tools"
        self.subagent_mute = False
        self.console = None
        self.agent_list = {}
        self.mcp_router = MagicMock()
        self.mcp_router.tool_registry = {}
        self.mcp_router.reg_tools = []
        self.messages = []
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_tokens = 0
        self.total_uncached_tokens = 0
        self.last_input_tokens = 0
        self.last_output_tokens = 0
        self.last_tokens = 0
        self.system_read_only_paths = []
        self.read_only_paths = []
        self.permissions = {}
        self.url_caches = []
        self.durable_crons = []
        self.session_crons = []
        self.cron_tasks = []
        self.cron_ids = []
        self.active_cron = 0
        self.design_man = MagicMock()
        self.run_man = MagicMock()
        self.tools = []
        self.skills = []
        self.loaded_skills = []
        self.api_configs = {}
        self.mcps_configs = []
        self.args = MagicMock()
        self.args.dangerously_allow_all = False
        self.session_uuid = "test_tools"
        self.session_title = "test"
        self.system_prompts = 0
        self.tools_prompts = 0
        self.user_prompts = 0
        self.content_prompts = 0
        self.reasoning_prompts = 0
        self.tool_calls_prompts = 0
        self.tool_results_prompts = 0
        self.total_llm_requests = 0
        self.task_tool_unuse = 0
        self.files_read = {}
        self.if_summarized = False
        self.task_end = False
        self.background_agents = []


class TestToolSchemas(unittest.TestCase):
    """P4-T1: Verify workflow tool schemas have correct structure."""

    def test_start_workflow_schema(self):
        from src.tool.tool_def import tool_start_dg_tft_workflow_def
        schema = tool_start_dg_tft_workflow_def()
        self.assertEqual(schema["type"], "function")
        fn = schema["function"]
        self.assertEqual(fn["name"], TOOL_NAME_START_WORKFLOW)
        self.assertIn("dataset_dir", fn["parameters"]["properties"])
        self.assertIn("dataset_dir", fn["parameters"]["required"])
        self.assertIn("hysteresis_dir", fn["parameters"]["properties"])
        self.assertIn("device_name", fn["parameters"]["properties"])
        self.assertFalse(fn["parameters"]["additionalProperties"])

    def test_query_workflow_schema(self):
        from src.tool.tool_def import tool_query_dg_tft_workflow_def
        schema = tool_query_dg_tft_workflow_def()
        fn = schema["function"]
        self.assertEqual(fn["name"], TOOL_NAME_QUERY_WORKFLOW)
        self.assertIn("run_id", fn["parameters"]["properties"])
        self.assertIn("run_id", fn["parameters"]["required"])
        self.assertEqual(
            fn["parameters"]["properties"]["run_id"]["type"], "integer"
        )

    def test_read_report_schema(self):
        from src.tool.tool_def import tool_read_dg_tft_workflow_report_def
        schema = tool_read_dg_tft_workflow_report_def()
        fn = schema["function"]
        self.assertEqual(fn["name"], TOOL_NAME_READ_WORKFLOW_REPORT)
        self.assertIn("run_id", fn["parameters"]["properties"])
        self.assertIn("run_id", fn["parameters"]["required"])
        self.assertEqual(
            fn["parameters"]["properties"]["run_id"]["type"], "integer"
        )

    def test_tools_in_create_prompts(self):
        from src.tool.tool_def import create_tools_prompts
        ctx = MockContext()
        prompts = create_tools_prompts(ctx)
        names = [p["function"]["name"] for p in prompts]
        self.assertIn(TOOL_NAME_START_WORKFLOW, names)
        self.assertIn(TOOL_NAME_QUERY_WORKFLOW, names)
        self.assertIn(TOOL_NAME_READ_WORKFLOW_REPORT, names)


class TestToolHandlers(unittest.TestCase):
    """P4-T2: Verify tool handler behavior with mocked dependencies."""

    def setUp(self):
        self.ctx = MockContext()
        self.ctx.workflow_man.create_run(
            dataset_dir="/tmp/test_dataset",
            output_root="/tmp/dg_tft_out",
            device_name="test_device",
        )

    def _mock_progress(self):
        p = MagicMock()
        p.console.print = MagicMock()
        return p

    @patch("src.tool.tool_def.ask_permission_tui")
    @patch("src.tool.tool_def.pause_for_permission")
    @patch("src.tool.tool_def.resume_from_permission")
    @patch("threading.Thread")
    def test_start_workflow_creates_run_and_returns_run_id(
        self, mock_thread, mock_resume, mock_pause, mock_ask
    ):
        mock_ask.return_value = (True, None)
        mock_thread_obj = MagicMock()
        mock_thread.return_value = mock_thread_obj

        from src.tool.tool_def import start_dg_tft_workflow
        result = start_dg_tft_workflow(
            {"dataset_dir": "/tmp/new_dataset"},
            self.ctx,
            self._mock_progress(),
        )
        self.assertEqual(result["status"], SUCCESS_LABEL)
        self.assertIn("run_id", result)
        self.assertGreater(result["run_id"], 1)

    @patch("src.tool.tool_def.ask_permission_tui")
    @patch("src.tool.tool_def.pause_for_permission")
    @patch("src.tool.tool_def.resume_from_permission")
    @patch("threading.Thread")
    def test_start_workflow_with_hysteresis(
        self, mock_thread, mock_resume, mock_pause, mock_ask
    ):
        mock_ask.return_value = (True, None)
        mock_thread_obj = MagicMock()
        mock_thread.return_value = mock_thread_obj

        from src.tool.tool_def import start_dg_tft_workflow
        result = start_dg_tft_workflow(
            {
                "dataset_dir": "/tmp/ds",
                "hysteresis_dir": "/tmp/hyst",
                "device_name": "mytft",
            },
            self.ctx,
            self._mock_progress(),
        )
        self.assertEqual(result["status"], SUCCESS_LABEL)
        run = self.ctx.workflow_man.get_run(result["run_id"])
        self.assertEqual(run.hysteresis_dir, "/tmp/hyst")
        self.assertEqual(run.extra["device_name"], "mytft")

    @patch("src.tool.tool_def.ask_permission_tui")
    @patch("src.tool.tool_def.pause_for_permission")
    @patch("src.tool.tool_def.resume_from_permission")
    def test_start_workflow_permission_denied(
        self, mock_resume, mock_pause, mock_ask
    ):
        mock_ask.return_value = (False, "not now")

        from src.tool.tool_def import start_dg_tft_workflow
        result = start_dg_tft_workflow(
            {"dataset_dir": "/tmp/ds"},
            self.ctx,
            self._mock_progress(),
        )
        self.assertEqual(result["status"], DENIED_LABEL)
        self.assertIn("not now", result["info"])

    def test_query_workflow_returns_status(self):
        wm = self.ctx.workflow_man
        ok, run_id, _ = wm.create_run(dataset_dir="/tmp/qds")
        wm.update_stage(run_id, StageName.CHECK_DATA, WorkflowStatus.RUNNING)

        from src.tool.tool_def import query_dg_tft_workflow
        result = query_dg_tft_workflow(
            {"run_id": run_id}, self.ctx, self._mock_progress()
        )
        self.assertEqual(result["status"], SUCCESS_LABEL)
        self.assertEqual(result["run_id"], run_id)
        self.assertEqual(result["workflow_status"], "RUNNING")
        self.assertEqual(result["current_stage"], StageName.CHECK_DATA)

    def test_query_workflow_not_found(self):
        from src.tool.tool_def import query_dg_tft_workflow
        result = query_dg_tft_workflow(
            {"run_id": 9999}, self.ctx, self._mock_progress()
        )
        self.assertEqual(result["status"], FAIL_LABEL)
        self.assertIn("not found", result["info"])

    def test_read_report_returns_run_data(self):
        wm = self.ctx.workflow_man
        ok, run_id, _ = wm.create_run(
            dataset_dir="/tmp/rds", output_root="/tmp/out"
        )
        wm.update_stage(run_id, StageName.CHECK_DATA, WorkflowStatus.RUNNING)
        wm.attach_artifact(run_id, "config_path", "/tmp/out/config.ini")
        wm.attach_metric(run_id, "r2", 0.95)
        wm.mark_done(run_id)

        from src.tool.tool_def import read_dg_tft_workflow_report
        result = read_dg_tft_workflow_report(
            {"run_id": run_id}, self.ctx, self._mock_progress()
        )
        self.assertEqual(result["status"], SUCCESS_LABEL)
        self.assertEqual(result["run_id"], run_id)
        self.assertEqual(result["workflow_status"], "DONE")
        self.assertIn("stage_summaries", result)

    def test_read_report_not_found(self):
        from src.tool.tool_def import read_dg_tft_workflow_report
        result = read_dg_tft_workflow_report(
            {"run_id": 9999}, self.ctx, self._mock_progress()
        )
        self.assertEqual(result["status"], FAIL_LABEL)


class TestToolDispatch(unittest.TestCase):
    """P4-T4: Verify dispatch routes to correct handler."""

    def setUp(self):
        self.ctx = MockContext()
        self.ctx.workflow_man.create_run(
            dataset_dir="/tmp/dd_dataset", output_root="/tmp/dd_out"
        )

    def _mock_progress(self):
        p = MagicMock()
        p.console.print = MagicMock()
        return p

    @patch("src.tool.tool_def.ask_permission_tui")
    @patch("src.tool.tool_def.pause_for_permission")
    @patch("src.tool.tool_def.resume_from_permission")
    @patch("threading.Thread")
    def test_dispatch_start_workflow(
        self, mock_thread, mock_resume, mock_pause, mock_ask
    ):
        mock_ask.return_value = (True, None)
        mock_thread_obj = MagicMock()
        mock_thread.return_value = mock_thread_obj

        from src.tool.tool_dispatch import call_tools
        from src.tool.scoreboard import Scoreboard

        board = Scoreboard()
        progress = self._mock_progress()
        result, addon = call_tools(
            TOOL_NAME_START_WORKFLOW,
            {"dataset_dir": "/tmp/ds"},
            self.ctx,
            board,
            progress,
        )
        self.assertEqual(result["status"], SUCCESS_LABEL)
        self.assertIn("run_id", result)
        self.assertIsNone(addon)

    @patch("src.tool.tool_def.ask_permission_tui")
    @patch("src.tool.tool_def.pause_for_permission")
    @patch("src.tool.tool_def.resume_from_permission")
    @patch("threading.Thread")
    def test_dispatch_query_workflow(
        self, mock_thread, mock_resume, mock_pause, mock_ask
    ):
        mock_ask.return_value = (True, None)
        mock_thread_obj = MagicMock()
        mock_thread.return_value = mock_thread_obj

        from src.tool.tool_dispatch import call_tools
        from src.tool.scoreboard import Scoreboard

        board = Scoreboard()
        progress = self._mock_progress()

        start_result, _ = call_tools(
            TOOL_NAME_START_WORKFLOW,
            {"dataset_dir": "/tmp/ds"},
            self.ctx,
            board,
            progress,
        )
        run_id = start_result["run_id"]

        result, addon = call_tools(
            TOOL_NAME_QUERY_WORKFLOW,
            {"run_id": run_id},
            self.ctx,
            board,
            progress,
        )
        self.assertEqual(result["status"], SUCCESS_LABEL)
        self.assertEqual(result["run_id"], run_id)

    @patch("src.tool.tool_def.ask_permission_tui")
    @patch("src.tool.tool_def.pause_for_permission")
    @patch("src.tool.tool_def.resume_from_permission")
    @patch("threading.Thread")
    def test_dispatch_read_report(
        self, mock_thread, mock_resume, mock_pause, mock_ask
    ):
        mock_ask.return_value = (True, None)
        mock_thread_obj = MagicMock()
        mock_thread.return_value = mock_thread_obj

        from src.tool.tool_dispatch import call_tools
        from src.tool.scoreboard import Scoreboard

        board = Scoreboard()
        progress = self._mock_progress()

        start_result, _ = call_tools(
            TOOL_NAME_START_WORKFLOW,
            {"dataset_dir": "/tmp/ds"},
            self.ctx,
            board,
            progress,
        )
        run_id = start_result["run_id"]

        result, addon = call_tools(
            TOOL_NAME_READ_WORKFLOW_REPORT,
            {"run_id": run_id},
            self.ctx,
            board,
            progress,
        )
        self.assertEqual(result["status"], SUCCESS_LABEL)
        self.assertEqual(result["run_id"], run_id)


class TestCommands(unittest.TestCase):
    """P4-T5: Verify builtin commands for workflow management."""

    def setUp(self):
        self.ctx = MockContext()

    def test_command_registered(self):
        from src.utility.command import BuiltinCommands
        cmd = BuiltinCommands(MagicMock())
        registered = dict(cmd._commands)
        self.assertIn("workflow_list", registered)
        self.assertIn("workflow_get", registered)
        self.assertIn("workflow_report", registered)

    def test_workflow_list_empty(self):
        from src.utility.command import cmd_workflow_list
        console = MagicMock()
        console.print = MagicMock()
        from src.tool.scoreboard import Scoreboard

        cmd_workflow_list([], self.ctx, Scoreboard(), console)
        self.assertTrue(console.print.called)

    def test_workflow_list_with_runs(self):
        from src.utility.command import cmd_workflow_list
        from src.tool.scoreboard import Scoreboard

        self.ctx.workflow_man.create_run(dataset_dir="/tmp/ds1")
        self.ctx.workflow_man.create_run(dataset_dir="/tmp/ds2")

        console = MagicMock()
        console.print = MagicMock()
        cmd_workflow_list([], self.ctx, Scoreboard(), console)
        self.assertTrue(console.print.called)

    def test_workflow_get_valid(self):
        from src.utility.command import cmd_workflow_get
        from src.tool.scoreboard import Scoreboard

        ok, run_id, _ = self.ctx.workflow_man.create_run(dataset_dir="/tmp/gds")
        console = MagicMock()
        console.print = MagicMock()
        cmd_workflow_get([str(run_id)], self.ctx, Scoreboard(), console)
        self.assertTrue(console.print.called)

    def test_workflow_get_not_found(self):
        from src.utility.command import cmd_workflow_get
        from src.tool.scoreboard import Scoreboard

        console = MagicMock()
        console.print = MagicMock()
        cmd_workflow_get(["99999"], self.ctx, Scoreboard(), console)
        self.assertTrue(console.print.called)

    def test_workflow_get_no_args(self):
        from src.utility.command import cmd_workflow_get
        from src.tool.scoreboard import Scoreboard

        console = MagicMock()
        console.print = MagicMock()
        cmd_workflow_get([], self.ctx, Scoreboard(), console)
        self.assertTrue(console.print.called)

    def test_workflow_report_no_args(self):
        from src.utility.command import cmd_workflow_report
        from src.tool.scoreboard import Scoreboard

        console = MagicMock()
        console.print = MagicMock()
        cmd_workflow_report([], self.ctx, Scoreboard(), console)
        self.assertTrue(console.print.called)

    def test_workflow_report_not_found(self):
        from src.utility.command import cmd_workflow_report
        from src.tool.scoreboard import Scoreboard

        console = MagicMock()
        console.print = MagicMock()
        cmd_workflow_report(["99999"], self.ctx, Scoreboard(), console)
        self.assertTrue(console.print.called)


class TestPromptRouting(unittest.TestCase):
    """P4-T6: Verify prompt includes DG-TFT routing hint."""

    def test_guideline_prompts_include_workflow_routing(self):
        from src.context.prompt import get_agent_guideline_prompts
        prompts = get_agent_guideline_prompts()
        content = prompts[0]["content"]
        self.assertIn("DG-TFT Workflow Routing", content)
        self.assertIn(TOOL_NAME_START_WORKFLOW, content)
        self.assertIn(TOOL_NAME_QUERY_WORKFLOW, content)
        self.assertIn(TOOL_NAME_READ_WORKFLOW_REPORT, content)

    def test_guideline_prompts_include_routing_instructions(self):
        from src.context.prompt import get_agent_guideline_prompts
        prompts = get_agent_guideline_prompts()
        content = prompts[0]["content"]
        self.assertIn("MUST use", content)
        self.assertIn("auto-modeling", content)
        self.assertIn("DONE status", content)


class TestRunningThreadTracker(unittest.TestCase):
    """P4-T3: Verify RunningRunTracker in WorkflowManager."""

    def test_manager_has_running_threads_dict(self):
        wm = WorkflowManager()
        self.assertTrue(hasattr(wm, "_running_threads"))
        self.assertIsInstance(wm._running_threads, dict)
        self.assertEqual(len(wm._running_threads), 0)

    @patch("threading.Thread")
    def test_thread_stored_after_start(self, mock_thread):
        mock_thread_obj = MagicMock()
        mock_thread.return_value = mock_thread_obj

        wm = WorkflowManager()
        wm.session_uuid = "test_threads"
        ok, run_id, _ = wm.create_run(dataset_dir="/tmp/tds")

        thread = mock_thread.return_value
        wm._running_threads[str(run_id)] = thread

        self.assertIn(str(run_id), wm._running_threads)
        self.assertEqual(wm._running_threads[str(run_id)], thread)


class TestBackgroundThreadFunction(unittest.TestCase):
    """P4-T3: Verify _run_workflow_thread clears thread on completion."""

    @patch("src.workflow.dg_tft.orchestrator.run_workflow")
    @patch("src.workflow.dg_tft.config_builder.build_runtime_config")
    @patch("src.workflow.dg_tft.dataset_inspector.inspect_dataset")
    def test_thread_clears_tracker_on_success(
        self, mock_inspect, mock_build, mock_run_wf
    ):
        from src.tool.tool_def import _run_workflow_thread
        from src.workflow.dg_tft.schema import DatasetProfile

        profile = DatasetProfile(static_ok=True, cv_ok=True)
        mock_inspect.return_value = (profile, {})
        mock_build.return_value = {"config_path": "/tmp/cfg.ini"}
        mock_run_wf.return_value = (True, "all done")

        wm = WorkflowManager()
        wm.session_uuid = "test_bg"
        ok, run_id, _ = wm.create_run(dataset_dir="/tmp/bgds")
        wm._running_threads[str(run_id)] = MagicMock()

        ctx = MockContext()
        ctx.workflow_man = wm

        _run_workflow_thread(ctx, run_id, wm)

        self.assertNotIn(str(run_id), wm._running_threads)
        mock_inspect.assert_called_once()
        mock_build.assert_called_once()
        mock_run_wf.assert_called_once()

    @patch("src.workflow.dg_tft.config_builder.build_runtime_config")
    @patch("src.workflow.dg_tft.dataset_inspector.inspect_dataset")
    def test_thread_marks_failed_on_exception(self, mock_inspect, mock_build):
        from src.tool.tool_def import _run_workflow_thread

        mock_inspect.side_effect = RuntimeError("inspection failure")

        wm = WorkflowManager()
        wm.session_uuid = "test_bg_fail"
        ok, run_id, _ = wm.create_run(dataset_dir="/tmp/bgfail")
        wm._running_threads[str(run_id)] = MagicMock()

        ctx = MockContext()
        ctx.workflow_man = wm

        _run_workflow_thread(ctx, run_id, wm)

        self.assertNotIn(str(run_id), wm._running_threads)
        run = wm.get_run(run_id)
        self.assertEqual(run.status, WorkflowStatus.FAILED)
        self.assertIn("inspection failure", run.errors[0])


if __name__ == "__main__":
    unittest.main()
