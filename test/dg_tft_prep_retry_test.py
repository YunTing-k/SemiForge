# -*- coding: utf-8 -*-
"""
Unit tests for Phase 4: Preparer-Verifier retry loop (TODO 2.5).

Run: python -m unittest test.dg_tft_prep_retry_test -v
"""
import sys
import os
import json
import tempfile
import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# ---------------------------------------------------------------------------
# 2.5.1: PrepHandoff / VerifierFeedback serialization
# ---------------------------------------------------------------------------

class TestPrepHandoff(unittest.TestCase):

    def test_to_dict_roundtrip(self):
        from src.workflow.dg_tft.schema import PrepHandoff
        ho = PrepHandoff(
            observations=["Found 3 IV CSVs"],
            actions=["Processed IDVBG_VDS0.1V"],
            potential_issues=["IDVD series may be incomplete"],
            output_manifest={"iv_files": 12, "cv_files": 1, "it_files": 0},
        )
        d = ho.to_dict()
        self.assertEqual(d["observations"], ["Found 3 IV CSVs"])
        ho2 = PrepHandoff.from_dict(d)
        self.assertEqual(ho2.observations, ho.observations)
        self.assertEqual(ho2.output_manifest, ho.output_manifest)

    def test_from_dict_empty(self):
        from src.workflow.dg_tft.schema import PrepHandoff
        ho = PrepHandoff.from_dict({})
        self.assertEqual(ho.observations, [])
        self.assertEqual(ho.actions, [])
        self.assertEqual(ho.potential_issues, [])
        self.assertEqual(ho.output_manifest, {})

    def test_from_worker_report(self):
        from src.workflow.dg_tft.schema import PrepHandoff
        report = {
            "observations": ["Found 3 IV CSVs, 1 CV file, 8 I-t files"],
            "actions": ["Processed IDVBG_VDS0.1V_train.xlsx", "Processed IDVTG_VDS5V_train.xlsx"],
            "potential_issues": ["IDVD series missing"],
            "output_manifest": {"iv_files": 10, "cv_files": 1, "it_files": 8},
        }
        ho = PrepHandoff.from_worker_report(report)
        self.assertIn("Found 3 IV CSVs", ho.observations[0])
        self.assertEqual(ho.output_manifest["iv_files"], 10)
        self.assertEqual(ho.output_manifest["cv_files"], 1)
        self.assertEqual(ho.output_manifest["it_files"], 8)
        self.assertEqual(ho.potential_issues, ["IDVD series missing"])


class TestVerifierFeedback(unittest.TestCase):

    def test_to_dict_roundtrip(self):
        from src.workflow.dg_tft.schema import VerifierFeedback
        vf = VerifierFeedback(
            overall_ok=False,
            critical_issues=[{"file": "IDVBG_VDS0.1V_train.xlsx", "issue": "missing VD_V column"}],
            suggestion="Re-process IDVBG with column name normalization",
            worker_notes="Previous attempt was missing VTG_V too",
            summary="IV file IDVBG_VDS0.1V_train.xlsx missing critical column VD_V",
        )
        d = vf.to_dict()
        vf2 = VerifierFeedback.from_dict(d)
        self.assertEqual(vf2.overall_ok, False)
        self.assertEqual(len(vf2.critical_issues), 1)
        self.assertEqual(vf2.suggestion, vf.suggestion)

    def test_from_dict_with_string_critical_issues(self):
        from src.workflow.dg_tft.schema import VerifierFeedback
        vf = VerifierFeedback.from_dict({
            "overall_ok": False,
            "critical_issues": ["missing VD_V", "no long sheet"],
        })
        self.assertEqual(len(vf.critical_issues), 2)
        self.assertEqual(vf.critical_issues[0]["issue"], "missing VD_V")

    def test_from_dict_empty(self):
        from src.workflow.dg_tft.schema import VerifierFeedback
        vf = VerifierFeedback.from_dict({})
        self.assertFalse(vf.overall_ok)  # missing field = not ok
        self.assertEqual(vf.critical_issues, [])

    def test_report_verification_ok(self):
        from src.workflow.dg_tft.schema import VerifierFeedback
        vf = VerifierFeedback.from_dict({
            "overall_ok": True,
            "summary": "All checks pass",
        })
        self.assertTrue(vf.overall_ok)

    def test_report_verification_failed(self):
        from src.workflow.dg_tft.schema import VerifierFeedback
        vf = VerifierFeedback.from_dict({
            "overall_ok": False,
            "critical_issues": [{"file": "test.xlsx", "issue": "bad"}],
        })
        self.assertFalse(vf.overall_ok)


# ---------------------------------------------------------------------------
# 2.5.5: subagent_tools data-prep stage + inspect_xlsx handler
# ---------------------------------------------------------------------------

class TestSubagentToolsDataPrep(unittest.TestCase):

    def test_data_prep_stage_has_tools(self):
        from src.tool.subagent_tools import get_tools, SUBAGENT_TOOL_INSPECT_XLSX, SUBAGENT_TOOL_REPORT_VERIFICATION
        schemas, handlers = get_tools("data-prep")
        self.assertEqual(len(schemas), 2)
        names = [s["function"]["name"] for s in schemas]
        self.assertIn(SUBAGENT_TOOL_INSPECT_XLSX, names)
        self.assertIn(SUBAGENT_TOOL_REPORT_VERIFICATION, names)
        self.assertEqual(len(handlers), 1)
        self.assertIn(SUBAGENT_TOOL_INSPECT_XLSX, handlers)

    def test_inspect_xlsx_handler_real_file(self):
        from src.tool.subagent_tools import _inspect_xlsx_handler
        import openpyxl
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "test.xlsx")
            wb = openpyxl.Workbook()
            ws = wb.active
            ws.title = "long"
            for i, h in enumerate(["VD_V", "VBG_V", "VTG_V", "ID_A"], 1):
                ws.cell(row=1, column=i, value=h)
            wb.save(path)
            wb.close()
            result = _inspect_xlsx_handler({"path": path}, None)
            self.assertEqual(result["sheet_count"], 1)
            self.assertIn("long", result["sheets"])
            self.assertEqual(result["sheets"]["long"]["columns"], ["VD_V", "VBG_V", "VTG_V", "ID_A"])

    def test_inspect_xlsx_handler_missing_file(self):
        from src.tool.subagent_tools import _inspect_xlsx_handler
        result = _inspect_xlsx_handler({"path": "/no/such/file.xlsx"}, None)
        self.assertIn("error", result)

    def test_inspect_xlsx_handler_multiple_sheets(self):
        from src.tool.subagent_tools import _inspect_xlsx_handler
        import openpyxl
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "multi.xlsx")
            wb = openpyxl.Workbook()
            ws1 = wb.active
            ws1.title = "Sheet1"
            ws1.cell(row=1, column=1, value="A")
            ws2 = wb.create_sheet("Sheet2")
            ws2.cell(row=1, column=1, value="B")
            wb.save(path)
            wb.close()
            result = _inspect_xlsx_handler({"path": path}, None)
            self.assertEqual(result["sheet_count"], 2)


# ---------------------------------------------------------------------------
# 3.6: AttemptTrigger enum values
# ---------------------------------------------------------------------------

class TestAttemptTrigger(unittest.TestCase):

    def test_verifier_fail_trigger_value(self):
        from src.workflow.dg_tft.operation_log import AttemptTrigger
        self.assertEqual(AttemptTrigger.VERIFIER_FAIL.value, "verifier_fail")

    def test_prep_retry_trigger_value(self):
        from src.workflow.dg_tft.operation_log import AttemptTrigger
        self.assertEqual(AttemptTrigger.PREP_RETRY.value, "prep_retry")


# ---------------------------------------------------------------------------
# 2.5.2: _get_handoff_history
# ---------------------------------------------------------------------------

class TestGetHandoffHistory(unittest.TestCase):

    def test_empty_on_no_attempts(self):
        from src.workflow.dg_tft.data_preparer import _get_handoff_history
        from src.workflow.dg_tft.operation_log import OperationLog
        with tempfile.TemporaryDirectory() as td:
            op_log = OperationLog(td)
            op_log.initialize(1, "")
            result = _get_handoff_history(op_log)
            self.assertEqual(result, "")

    def test_builds_history_from_attempts(self):
        from src.workflow.dg_tft.data_preparer import _get_handoff_history
        from src.workflow.dg_tft.operation_log import OperationLog, StageAttemptLog, AttemptTrigger
        with tempfile.TemporaryDirectory() as td:
            op_log = OperationLog(td)
            op_log.initialize(1, "")
            op_log.append_attempt("data-prep", StageAttemptLog(
                n=1, trigger=AttemptTrigger.INITIAL,
                handoff={"observations": ["found files"]},
            ))
            op_log.append_attempt("verifier", StageAttemptLog(
                n=1, trigger=AttemptTrigger.VERIFIER_FAIL,
                agent_decision={"overall_ok": False, "summary": "fail"},
            ))
            result = _get_handoff_history(op_log)
            self.assertIn("found files", result)
            self.assertIn("fail", result)

    def test_returns_empty_for_non_oplog(self):
        from src.workflow.dg_tft.data_preparer import _get_handoff_history
        result = _get_handoff_history(None)
        self.assertEqual(result, "")


# ---------------------------------------------------------------------------
# 2.5.3: _read_worker_handoff
# ---------------------------------------------------------------------------

class TestReadWorkerHandoff(unittest.TestCase):

    def test_reads_valid_report(self):
        from src.workflow.dg_tft.data_preparer import _read_worker_handoff
        with tempfile.TemporaryDirectory() as td:
            report = {
                "observations": ["Found 1 IV file"],
                "actions": ["Processed a.xlsx"],
                "potential_issues": ["missing CV"],
                "output_manifest": {"iv_files": 1, "cv_files": 0, "it_files": 0},
            }
            with open(os.path.join(td, "worker_report.json"), "w", encoding="utf-8") as f:
                json.dump(report, f)
            handoff = _read_worker_handoff(td)
            self.assertIsNotNone(handoff)
            self.assertIn("observations", handoff)
            self.assertIn("actions", handoff)
            self.assertEqual(handoff["potential_issues"], ["missing CV"])

    def test_returns_none_for_missing_report(self):
        from src.workflow.dg_tft.data_preparer import _read_worker_handoff
        with tempfile.TemporaryDirectory() as td:
            handoff = _read_worker_handoff(td)
            self.assertIsNone(handoff)

    def test_returns_none_for_invalid_json(self):
        from src.workflow.dg_tft.data_preparer import _read_worker_handoff
        with tempfile.TemporaryDirectory() as td:
            with open(os.path.join(td, "worker_report.json"), "w", encoding="utf-8") as f:
                f.write("not json")
            handoff = _read_worker_handoff(td)
            self.assertIsNone(handoff)


# ---------------------------------------------------------------------------
# 2.5.6: run_data_preparation_with_retry
# ---------------------------------------------------------------------------

class TestRunDataPreparationWithRetry(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.run_output_dir = Path(self.tmpdir.name) / "session" / "test-uuid" / "run_1"
        self.run_output_dir.mkdir(parents=True, exist_ok=True)
        self.ctx = MagicMock()

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_success_first_attempt(self):
        from src.workflow.dg_tft.data_preparer import run_data_preparation_with_retry
        with patch("src.workflow.dg_tft.data_preparer._run_worker_and_verifier",
                   return_value=(
                       {"observations": ["ok"]},
                       {"overall_ok": True},
                       True,
                       ["[WARNING] no CV"],
                   )):
            result = run_data_preparation_with_retry(
                self.ctx, "/raw", str(self.run_output_dir), 1, max_retries=3,
            )
        self.assertTrue(result["ok"])
        self.assertEqual(result["prep_id"], 1)

    def test_success_after_retry(self):
        from src.workflow.dg_tft.data_preparer import run_data_preparation_with_retry
        fail_result = ({}, {"overall_ok": False}, False, ["[CRITICAL] bad"])
        pass_result = ({}, {"overall_ok": True}, True, [])
        with patch("src.workflow.dg_tft.data_preparer._run_worker_and_verifier",
                   side_effect=[fail_result, pass_result]):
            result = run_data_preparation_with_retry(
                self.ctx, "/raw", str(self.run_output_dir), 1, max_retries=3,
            )
        self.assertTrue(result["ok"])

    def test_exhausted_all_retries(self):
        from src.workflow.dg_tft.data_preparer import run_data_preparation_with_retry
        fail_result = ({}, {"overall_ok": False}, False, ["[CRITICAL] bad"])
        with patch("src.workflow.dg_tft.data_preparer._run_worker_and_verifier",
                   return_value=fail_result):
            result = run_data_preparation_with_retry(
                self.ctx, "/raw", str(self.run_output_dir), 1, max_retries=2,
            )
        self.assertFalse(result["ok"])

    def test_worker_exception_continue_retry(self):
        from src.workflow.dg_tft.data_preparer import run_data_preparation_with_retry
        pass_result = ({}, {"overall_ok": True}, True, [])
        with patch("src.workflow.dg_tft.data_preparer._run_worker_and_verifier",
                   side_effect=[RuntimeError("crash"), pass_result]):
            result = run_data_preparation_with_retry(
                self.ctx, "/raw", str(self.run_output_dir), 1, max_retries=3,
            )
        self.assertTrue(result["ok"])

    def test_creates_attempt_dirs(self):
        from src.workflow.dg_tft.data_preparer import run_data_preparation_with_retry
        from src.constants import PREP_SUBDIR_NAME
        fail_result = ({}, {"overall_ok": False}, False, ["fail"])
        pass_result = ({}, {"overall_ok": True}, True, [])
        with patch("src.workflow.dg_tft.data_preparer._run_worker_and_verifier",
                   side_effect=[fail_result, pass_result]):
            run_data_preparation_with_retry(
                self.ctx, "/raw", str(self.run_output_dir), 1, max_retries=3,
            )
        self.assertTrue(os.path.isdir(os.path.join(str(self.run_output_dir), PREP_SUBDIR_NAME, "attempt_1")))
        self.assertTrue(os.path.isdir(os.path.join(str(self.run_output_dir), PREP_SUBDIR_NAME, "attempt_2")))

    def test_writes_operation_log(self):
        from src.workflow.dg_tft.data_preparer import run_data_preparation_with_retry
        pass_result = ({}, {"overall_ok": True}, True, [])
        with patch("src.workflow.dg_tft.data_preparer._run_worker_and_verifier",
                   return_value=pass_result):
            run_data_preparation_with_retry(
                self.ctx, "/raw", str(self.run_output_dir), 1, max_retries=3,
            )
        oplog_path = os.path.join(str(self.run_output_dir), "operation_log.json")
        self.assertTrue(os.path.exists(oplog_path))
        with open(oplog_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual(data["run_id"], 1)
        self.assertIn("data-prep", data["stages"])
        attempts = data["stages"]["data-prep"]
        self.assertEqual(len(attempts), 1)
        self.assertEqual(attempts[0]["n"], 1)
        self.assertEqual(attempts[0]["trigger"], "initial")

    def test_retry_trigger_is_verifier_fail(self):
        from src.workflow.dg_tft.data_preparer import run_data_preparation_with_retry
        fail_result = ({}, {"overall_ok": False}, False, ["fail"])
        pass_result = ({}, {"overall_ok": True}, True, [])
        with patch("src.workflow.dg_tft.data_preparer._run_worker_and_verifier",
                   side_effect=[fail_result, pass_result]):
            run_data_preparation_with_retry(
                self.ctx, "/raw", str(self.run_output_dir), 1, max_retries=3,
            )
        oplog_path = os.path.join(str(self.run_output_dir), "operation_log.json")
        with open(oplog_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        attempts = data["stages"]["data-prep"]
        self.assertEqual(len(attempts), 2)
        self.assertEqual(attempts[0]["trigger"], "initial")
        self.assertEqual(attempts[1]["trigger"], "verifier_fail")

    def test_zero_retries_never_runs(self):
        from src.workflow.dg_tft.data_preparer import run_data_preparation_with_retry
        result = run_data_preparation_with_retry(
            self.ctx, "/raw", str(self.run_output_dir), 1, max_retries=0,
        )
        self.assertFalse(result["ok"])


# ---------------------------------------------------------------------------
# 2.5.7: run_verification_only
# ---------------------------------------------------------------------------

class TestRunVerificationOnly(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.run_output_dir = Path(self.tmpdir.name) / "session" / "test-uuid" / "run_2"
        self.run_output_dir.mkdir(parents=True, exist_ok=True)
        self.dataset_dir = Path(self.tmpdir.name) / "standard_data"
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        import openpyxl
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "long"
        for i, h in enumerate(["VD_V", "VBG_V", "VTG_V", "ID_A"], 1):
            ws.cell(row=1, column=i, value=h)
        wb.save(str(self.dataset_dir / "IDVBG_VDS0.1V_train.xlsx"))
        wb.close()
        self.ctx = MagicMock()

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_passes_when_verifier_ok(self):
        from src.workflow.dg_tft.data_preparer import run_verification_only
        with patch("src.workflow.dg_tft.data_preparer._run_verifier_subagent",
                   return_value={"overall_ok": True}):
            result = run_verification_only(
                self.ctx, str(self.dataset_dir), str(self.run_output_dir), 2,
            )
        self.assertTrue(result["ok"])
        self.assertEqual(result["prep_id"], 2)

    def test_fails_without_fallback_when_no_raw_dir(self):
        from src.workflow.dg_tft.data_preparer import run_verification_only
        with patch("src.workflow.dg_tft.data_preparer._run_verifier_subagent",
                   return_value={"overall_ok": False}):
            result = run_verification_only(
                self.ctx, str(self.dataset_dir), str(self.run_output_dir), 2,
            )
        self.assertFalse(result["ok"])

    def test_falls_back_to_preparer_when_raw_dir_provided(self):
        from src.workflow.dg_tft.data_preparer import run_verification_only
        with patch("src.workflow.dg_tft.data_preparer._run_verifier_subagent",
                   return_value={"overall_ok": False}), \
             patch("src.workflow.dg_tft.data_preparer.run_data_preparation_with_retry",
                   return_value={"ok": True, "prep_id": 2}):
            result = run_verification_only(
                self.ctx, str(self.dataset_dir), str(self.run_output_dir), 2,
                raw_dataset_dir="/raw_fallback",
            )
        self.assertTrue(result["ok"])


if __name__ == "__main__":
    unittest.main()
