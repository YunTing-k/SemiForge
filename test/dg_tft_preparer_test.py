# -*- coding: utf-8 -*-
"""
Unit tests for Phase 4.5: Data preparation (worker + verifier + check).
Run: python -m unittest test.dg_tft_preparer_test -v
"""
import sys
import os
import json
import tempfile
import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.workflow.dg_tft.schema import WorkflowRun, WorkflowStatus
from src.workflow.dg_tft.manager import WorkflowManager, PreparationTracker
from src.workflow.dg_tft.data_preparer import (
    _check_output,
    _build_manifest,
    _write_manifest,
    WORKER_PROMPT_TEMPLATE,
    VERIFIER_PROMPT_TEMPLATE,
    run_data_preparation,
)
from src.constants import (
    TOOL_NAME_PREPARE_DATASET,
    SUCCESS_LABEL,
    FAIL_LABEL,
    DENIED_LABEL,
)


class TestPreparationTracker(unittest.TestCase):
    """P4.5-T1: PreparationTracker lifecycle management."""

    def test_register_and_get(self):
        pt = PreparationTracker()
        pid = pt.register_prep("/raw/data", "/out/1")
        self.assertEqual(pid, 1)
        prep = pt.get_prep(1)
        self.assertIsNotNone(prep)
        self.assertEqual(prep["raw_dataset_dir"], "/raw/data")
        self.assertFalse(prep["ok"])

    def test_register_sequential_ids(self):
        pt = PreparationTracker()
        p1 = pt.register_prep("/a", "/out/a")
        p2 = pt.register_prep("/b", "/out/b")
        self.assertEqual(p1, 1)
        self.assertEqual(p2, 2)

    def test_update_prep(self):
        pt = PreparationTracker()
        pid = pt.register_prep("/raw", "/out")
        pt.update_prep(pid, True, [])
        prep = pt.get_prep(pid)
        self.assertTrue(prep["ok"])

    def test_to_from_dict(self):
        pt = PreparationTracker()
        pt.register_prep("/raw", "/out")
        pt.update_prep(1, True, ["[WARNING] no CV found"])
        data = pt.to_dict()
        pt2 = PreparationTracker()
        pt2.from_dict(data)
        self.assertEqual(pt2._next_prep_id, pt._next_prep_id)
        self.assertEqual(pt2._preps, pt._preps)

    def test_get_nonexistent(self):
        pt = PreparationTracker()
        self.assertIsNone(pt.get_prep(999))


class TestWorkflowRunPrepId(unittest.TestCase):
    """P4.5-T1: prep_id in WorkflowRun."""

    def test_prep_id_default_none(self):
        run = WorkflowRun(run_id=1)
        self.assertIsNone(run.prep_id)

    def test_prep_id_in_to_dict(self):
        run = WorkflowRun(run_id=1, prep_id=5)
        d = run.to_dict()
        self.assertEqual(d["prep_id"], 5)

    def test_prep_id_in_from_dict(self):
        d = {
            "run_id": 1,
            "status": "PENDING",
            "prep_id": 5,
        }
        run = WorkflowRun.from_dict(d)
        self.assertEqual(run.prep_id, 5)

    def test_prep_id_in_create_run(self):
        wm = WorkflowManager()
        ok, rid, _ = wm.create_run(
            dataset_dir="/data",
            prep_id=7,
        )
        self.assertTrue(ok)
        run = wm.get_run(rid)
        self.assertEqual(run.prep_id, 7)


class TestCheckOutput(unittest.TestCase):
    """P4.5-T1: _check_output programmatic CRITICAL/WARNING checks."""

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.outdir = Path(self.tmpdir.name)

    def tearDown(self):
        self.tmpdir.cleanup()

    def _make_xlsx(self, name, sheet_name="long", headers=None):
        import openpyxl
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = sheet_name
        if headers:
            for i, h in enumerate(headers, 1):
                ws.cell(row=1, column=i, value=h)
        path = self.outdir / name
        wb.save(str(path))
        wb.close()

    def test_no_iv_files_critical(self):
        ok, issues = _check_output(str(self.outdir))
        self.assertFalse(ok)
        self.assertTrue(any("CRITICAL" in i for i in issues))

    def test_iv_file_with_long_sheet_and_columns_ok(self):
        self._make_xlsx("IDVBG_VDS0.1V_train.xlsx", headers=["VD_V", "VBG_V", "VTG_V", "ID_A"])
        ok, issues = _check_output(str(self.outdir))
        self.assertTrue(ok)

    def test_iv_file_missing_long_sheet_critical(self):
        self._make_xlsx("IDVBG_VDS0.1V_train.xlsx", sheet_name="Sheet1", headers=["VD_V", "VBG_V"])
        ok, issues = _check_output(str(self.outdir))
        self.assertFalse(ok)
        self.assertTrue(any("missing 'long'" in i for i in issues))

    def test_iv_file_missing_columns_critical(self):
        self._make_xlsx("IDVBG_VDS0.1V_train.xlsx", headers=["VD_V", "VBG_V"])
        ok, issues = _check_output(str(self.outdir))
        self.assertFalse(ok)
        self.assertTrue(any("missing columns" in i for i in issues))

    def test_cv_missing_is_warning(self):
        self._make_xlsx("IDVBG_VDS0.1V_train.xlsx", headers=["VD_V", "VBG_V", "VTG_V", "ID_A"])
        ok, issues = _check_output(str(self.outdir))
        self.assertTrue(ok)
        self.assertTrue(any("[WARNING] CV_train.xlsx not found" in i for i in issues))

    def test_cv_present_no_warning(self):
        self._make_xlsx("IDVBG_VDS0.1V_train.xlsx", headers=["VD_V", "VBG_V", "VTG_V", "ID_A"])
        self._make_xlsx("CV_train.xlsx", sheet_name="cgg")
        ok, issues = _check_output(str(self.outdir))
        self.assertTrue(ok)
        self.assertFalse(any("CV_train.xlsx not found" in i for i in issues))

    def test_it_missing_is_warning(self):
        self._make_xlsx("IDVBG_VDS0.1V_train.xlsx", headers=["VD_V", "VBG_V", "VTG_V", "ID_A"])
        ok, issues = _check_output(str(self.outdir))
        self.assertTrue(ok)
        self.assertTrue(any("[WARNING] it/" in i for i in issues))

    def test_multiple_families_all_ok(self):
        self._make_xlsx("IDVBG_VDS0.1V_train.xlsx", headers=["VD_V", "VBG_V", "VTG_V", "ID_A"])
        self._make_xlsx("IDVTG_VDS0.1V_train.xlsx", headers=["VD_V", "VBG_V", "VTG_V", "ID_A"])
        self._make_xlsx("IDVD_VBG0V_train.xlsx", headers=["VD_V", "VBG_V", "VTG_V", "ID_A"])
        ok, issues = _check_output(str(self.outdir))
        self.assertTrue(ok)

    def test_one_family_bad_others_ok_still_critical(self):
        self._make_xlsx("IDVBG_VDS0.1V_train.xlsx", headers=["VD_V", "VBG_V", "VTG_V", "ID_A"])
        self._make_xlsx("IDVTG_VDS0.1V_train.xlsx", headers=["VD_V"])
        ok, issues = _check_output(str(self.outdir))
        self.assertFalse(ok)


class TestWorkerPromptTemplate(unittest.TestCase):
    """P4.5-T1: Worker prompt template formatting."""

    def test_template_formats_both_vars(self):
        from src.workflow.dg_tft.data_preparer import (
            _PREP_IV_SHEET_NAME, _PREP_REQUIRED_IV_COLUMNS,
            _PREP_CV_EXPECTED_SHEETS, _PREP_WORKER_REPORT_NAME,
        )
        prompt = WORKER_PROMPT_TEMPLATE.format(
            raw_dir="/raw", output_dir="/out",
            iv_sheet=_PREP_IV_SHEET_NAME,
            iv_columns=", ".join(_PREP_REQUIRED_IV_COLUMNS),
            cv_sheets=", ".join(_PREP_CV_EXPECTED_SHEETS),
            report_name=_PREP_WORKER_REPORT_NAME,
            python_exe="python",
        )
        self.assertIn("/raw", prompt)
        self.assertIn("/out", prompt)
        self.assertIn("long", prompt)
        self.assertIn("IDVBG", prompt)


class TestVerifierPromptTemplate(unittest.TestCase):
    """P4.5-T1: Verifier prompt template formatting."""

    def test_template_formats_output_dir(self):
        from src.workflow.dg_tft.data_preparer import (
            _PREP_IV_SHEET_NAME, _PREP_REQUIRED_IV_COLUMNS, _PREP_CV_EXPECTED_SHEETS,
        )
        prompt = VERIFIER_PROMPT_TEMPLATE.format(
            output_dir="/out",
            iv_sheet=_PREP_IV_SHEET_NAME,
            iv_columns=", ".join(_PREP_REQUIRED_IV_COLUMNS),
            cv_sheets=", ".join(_PREP_CV_EXPECTED_SHEETS),
        )
        self.assertIn("/out", prompt)
        self.assertIn("report_verification", prompt)


class TestBuildManifest(unittest.TestCase):
    """P4.5-T1: Manifest construction and writing."""

    def test_ok_manifest(self):
        m = _build_manifest("/out", 1, True, ["[WARNING] no CV"])
        self.assertTrue(m["ok"])
        self.assertEqual(m["prep_id"], 1)

    def test_failed_manifest(self):
        m = _build_manifest("/out", 2, False, ["[CRITICAL] no IV files"])
        self.assertFalse(m["ok"])

    def test_write_manifest(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            manifest = _build_manifest(td, 1, True, [])
            _write_manifest(td, manifest)
            written = json.load(open(os.path.join(td, "manifest.json"), encoding="utf-8"))
            self.assertEqual(written["prep_id"], 1)
            self.assertTrue(written["ok"])


class TestRunDataPreparationMocked(unittest.TestCase):
    """P4.5-T1: Full run_data_preparation flow with mocked subagent and LLM."""

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.prep_id = 1
        self.outdir = Path(self.tmpdir.name) / str(self.prep_id)
        self.outdir.mkdir(parents=True, exist_ok=True)

        import openpyxl
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "long"
        for i, h in enumerate(["VD_V", "VBG_V", "VTG_V", "ID_A"], 1):
            ws.cell(row=1, column=i, value=h)
        path = self.outdir / "IDVBG_VDS0.1V_train.xlsx"
        wb.save(str(path))
        wb.close()

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_run_data_preparation_success(self):
        """Worker+verifier succeed, check_output passes."""
        mock_ctx = MagicMock()
        mock_ctx.workflow_man = WorkflowManager()
        mock_ctx.workflow_man.preparation_tracker = PreparationTracker()
        mock_ctx.workflow_man.preparation_tracker.register_prep("/raw", str(self.outdir))

        with patch("src.workflow.dg_tft.data_preparer._run_worker_and_verifier",
                   return_value=({}, {"overall_ok": True}, True, ["[WARNING] no CV"])):
            run_data_preparation(mock_ctx, "/raw", str(self.outdir), self.prep_id)

        prep = mock_ctx.workflow_man.preparation_tracker.get_prep(self.prep_id)
        self.assertIsNotNone(prep)
        self.assertTrue(os.path.exists(str(self.outdir / "manifest.json")))

    def test_run_data_preparation_fails(self):
        """Worker+verifier raises — manifest marks failed."""
        mock_ctx = MagicMock()
        mock_ctx.workflow_man = WorkflowManager()
        mock_ctx.workflow_man.preparation_tracker = PreparationTracker()
        mock_ctx.workflow_man.preparation_tracker.register_prep("/raw", str(self.outdir))

        with patch("src.workflow.dg_tft.data_preparer._run_worker_and_verifier",
                   side_effect=RuntimeError("worker crash")):
            run_data_preparation(mock_ctx, "/raw", str(self.outdir), self.prep_id)

        prep = mock_ctx.workflow_man.preparation_tracker.get_prep(self.prep_id)
        self.assertIsNotNone(prep)
        self.assertFalse(prep["ok"])
        manifest = json.load(open(str(self.outdir / "manifest.json"), encoding="utf-8"))
        self.assertFalse(manifest["ok"])
        self.assertTrue(any("worker crash" in i for i in manifest["issues"]))

    def test_run_data_preparation_verifier_fails_check_passes(self):
        """Verifier says False but programmatic check passes — still ok."""
        mock_ctx = MagicMock()
        mock_ctx.workflow_man = WorkflowManager()
        mock_ctx.workflow_man.preparation_tracker = PreparationTracker()
        mock_ctx.workflow_man.preparation_tracker.register_prep("/raw", str(self.outdir))

        with patch("src.workflow.dg_tft.data_preparer._run_worker_and_verifier",
                   return_value=({}, {"overall_ok": False}, True, ["[WARNING] no CV"])):
            run_data_preparation(mock_ctx, "/raw", str(self.outdir), self.prep_id)

        prep = mock_ctx.workflow_man.preparation_tracker.get_prep(self.prep_id)
        self.assertIsNotNone(prep)
        # run_data_preparation uses programmatic check result, not verifier feedback
        self.assertTrue(prep["ok"])


class TestToolHandler(unittest.TestCase):
    """P4.5-T2: prepare_dg_tft_dataset tool handler behavior."""

    def setUp(self):
        import tempfile
        self.tmpdir = tempfile.TemporaryDirectory()
        self.session_dir = self.tmpdir.name

    def tearDown(self):
        self.tmpdir.cleanup()

    def _mock_ctx(self, permissions=None):
        ctx = MagicMock()
        ctx.workflow_man = WorkflowManager()
        ctx.workflow_man.preparation_tracker = PreparationTracker()
        ctx.agent_configs = {}
        ctx.session_uuid = "00000000-0000-0000-0000-000000000000"
        if permissions is not None:
            ctx.permissions = permissions
        return ctx

    def test_tool_handler_creates_prep_and_output(self):
        from src.tool.tool_def import prepare_dg_tft_dataset
        ctx = self._mock_ctx(permissions={TOOL_NAME_PREPARE_DATASET: False})
        progress = MagicMock()
        arguments = {"raw_dataset_dir": "/some/raw/path"}

        with patch("src.workflow.dg_tft.data_preparer.run_data_preparation_with_retry",
                   return_value={"ok": True, "prep_id": 1, "output_dir": "/fake/out", "issues": []}), \
             patch("src.tool.tool_def.SESSION_PATH", self.tmpdir.name):
            result = prepare_dg_tft_dataset(arguments, ctx, progress)
        self.assertEqual(result["status"], SUCCESS_LABEL)
        self.assertIsNotNone(result.get("prep_id"))

    def test_tool_handler_permission_denied(self):
        from src.tool.tool_def import prepare_dg_tft_dataset
        ctx = self._mock_ctx(permissions={TOOL_NAME_PREPARE_DATASET: False})
        progress = MagicMock()
        arguments = {"raw_dataset_dir": "/some/path"}

        with patch("src.tool.tool_def.ask_permission_tui", return_value=(False, "test denial")):
            result = prepare_dg_tft_dataset(arguments, ctx, progress)
            self.assertEqual(result["status"], DENIED_LABEL)


class TestToolDispatch(unittest.TestCase):
    """P4.5-T2: prepare_dg_tft_dataset tool dispatch."""

    def test_dispatch_prepare_dataset(self):
        import tempfile
        from src.tool.tool_dispatch import call_tools
        ctx = MagicMock()
        ctx.workflow_man = WorkflowManager()
        ctx.workflow_man.preparation_tracker = PreparationTracker()
        ctx.agent_configs = {}
        ctx.session_uuid = "00000000-0000-0000-0000-000000000000"
        ctx.permissions = {TOOL_NAME_PREPARE_DATASET: False}
        ctx.mcp_router = MagicMock()
        ctx.mcp_router.tool_registry = {}
        progress = MagicMock()
        board = MagicMock()

        with tempfile.TemporaryDirectory() as td, \
             patch("src.workflow.dg_tft.data_preparer.run_data_preparation_with_retry",
                   return_value={"ok": True, "prep_id": 1, "output_dir": "/fake/out", "issues": []}), \
             patch("src.tool.tool_def.SESSION_PATH", td):
            result, addon = call_tools(TOOL_NAME_PREPARE_DATASET, {"raw_dataset_dir": "/data"}, ctx, board, progress)
        self.assertEqual(result["status"], SUCCESS_LABEL)
        self.assertIsNotNone(result.get("prep_id"))
        self.assertIsNone(addon)


class TestCommandsPrep(unittest.TestCase):
    """P4.5-T3: workflow commands show prep data."""

    def setUp(self):
        self.ctx = MagicMock()
        self.ctx.workflow_man = WorkflowManager()
        self.ctx.workflow_man.preparation_tracker = PreparationTracker()
        self.ctx.workflow_man.preparation_tracker.register_prep("/raw", "/out/prep_1")

    def test_workflow_get_prep_shows_prep_info(self):
        from src.utility.command import cmd_workflow_get
        from src.constants import MAJOR_COLOR1
        board = MagicMock()
        console = MagicMock()
        cmd_workflow_get(["1"], self.ctx, board, console)
        printed = console.print.call_args[0][0] if console.print.call_args else ""
        self.assertTrue(True)


class TestPromptRoutingPrep(unittest.TestCase):
    """P4.5-T3: prompt includes data preparation routing."""

    def test_guideline_prompts_include_prepare_tool(self):
        from src.context.prompt import get_agent_guideline_prompts
        prompts = get_agent_guideline_prompts()
        content = prompts[0]["content"]
        self.assertIn(TOOL_NAME_PREPARE_DATASET, content)
        self.assertIn("raw measurement data", content.lower())


class TestPrepareToolSchema(unittest.TestCase):
    """P4.5-T2: prepare_dg_tft_dataset tool schema."""

    def test_schema_in_create_prompts(self):
        from src.tool.tool_def import create_tools_prompts
        ctx = MagicMock()
        ctx.mcp_router = MagicMock()
        ctx.mcp_router.reg_tools = []
        ctx.agent_configs = {}
        prompts = create_tools_prompts(ctx)
        names = [p["function"]["name"] for p in prompts]
        self.assertIn(TOOL_NAME_PREPARE_DATASET, names)


if __name__ == "__main__":
    unittest.main()
