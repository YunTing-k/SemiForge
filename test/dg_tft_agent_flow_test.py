# -*- coding: utf-8 -*-
"""
Unit tests for Agent A-E interaction flows — fallback chains, handoff,
tool decisions, minimal_run gating, and operation_log recording.

All tests avoid real LLM calls (mock SubAgent) and real subprocess
execution (mock run_stage).  No files or directories are left behind.
"""
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import openpyxl

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.workflow.dg_tft.manager import WorkflowManager
from src.workflow.dg_tft.operation_log import OperationLog, AttemptTrigger, StageAttemptLog
from src.workflow.dg_tft.orchestrator import (
    _normalize_tool_name,
    _process_agent_decision,
    _spawn_agent_a,
    _spawn_agent_e,
    _run_agent_stage,
    run_workflow,
    _stage_has_modeling_agent,
)
from src.workflow.dg_tft.schema import (
    DatasetProfile,
    StageName,
    StageResult,
    WorkflowRun,
    WorkflowStatus,
)
from src.workflow.dg_tft.stages import StageDef
from src.constants import (
    DIAGNOSTIC_AGENT_LABEL,
    INTERPRETER_AGENT_LABEL,
    MODELING_AGENT_LABEL,
    PREPARER_AGENT_LABEL,
    VERIFIER_AGENT_LABEL,
)


# ── helpers ──────────────────────────────────────────────────────────

def _make_mock_subagent(return_value):
    """Create a MagicMock SubAgent that returns *return_value* from .run()."""
    agent = MagicMock()
    agent.build_tools = MagicMock()
    agent.build_messages = MagicMock()
    agent.error = None
    agent.progress = MagicMock()
    agent.status = MagicMock()
    agent.status.value = "done"
    agent.run.return_value = return_value
    return agent


def _make_mock_ctx(extra_configs=None):
    """Return a MockContext suitable for orchestrator tests."""
    configs = {
        "DG_TFT_MODELING_PYTHON_EXE": sys.executable,
        "DG_TFT_AGENT_PYTHON_EXE": sys.executable,
        "DG_TFT_QUALITY_PHYSICAL_IV_RMS_MAX": 0.35,
        "DG_TFT_QUALITY_STATIC_IV_RMS_MAX": 0.35,
        "DG_TFT_QUALITY_HYSTERESIS_RMS_MAX": 0.08,
        "DG_TFT_QUALITY_MAX_RETRIES": 3,
    }
    if extra_configs:
        configs.update(extra_configs)

    class Ctx:
        def __init__(self):
            self.agent_configs = configs
            self.workflow_man = WorkflowManager()
            self.workflow_man.session_uuid = "test_agent_flow"
            self.workflow_man._save_runs = lambda: None
            self.agent_list = {}
            self.agent_id = "test_agent"
            self.subagent_mute = True
            self.console = None
            self.session_uuid = "test_agent_flow"

    return Ctx()


def _write_minimal_config(tmpdir, extra_lines="", minimial_run=False):
    """Write a minimal model_config.ini for orchestrator tests."""
    lines = [
        "[paths]",
        "data_dir = /data",
        "output_dir = " + tmpdir.replace("\\", "/") + "/output",
        "hysteresis_data_dir = /data",
        "hspice_exe =",
        "[hysteresis]",
        "enabled = yes",
        "[temperature]",
        "mode = auto",
        "[deployment]",
        "run_hspice = no",
    ]
    if minimial_run:
        lines.append("[runtime]")
        lines.append("minimal_run = yes")
    if extra_lines:
        lines.extend(extra_lines.strip().splitlines())
    path = os.path.join(tmpdir, "model_config.ini")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


# ======================================================================
# _normalize_tool_name
# ======================================================================

class TestNormalizeToolName(unittest.TestCase):

    def test_adjust_params_explicit(self):
        self.assertEqual(_normalize_tool_name({"adjust_params": {"a": 1}}), "adjust_params")

    def test_fallback_to_previous(self):
        self.assertEqual(_normalize_tool_name({"fallback_to_previous": {}}), "fallback_to_previous")

    def test_accept_quality(self):
        self.assertEqual(_normalize_tool_name({"accept_quality": {}}), "accept_quality")

    def test_applied_heuristic(self):
        self.assertEqual(_normalize_tool_name({"applied": {"epochs": 50}}), "adjust_params")

    def test_section_adjustments_heuristic(self):
        self.assertEqual(_normalize_tool_name({"section": "fit", "adjustments": {"lr": 0.001}}), "adjust_params")

    def test_target_reason_heuristic(self):
        self.assertEqual(_normalize_tool_name({"target": "check-data", "reason": "bad data"}), "fallback_to_previous")

    def test_error_on_empty(self):
        self.assertEqual(_normalize_tool_name({}), "error")

    def test_error_on_unknown(self):
        self.assertEqual(_normalize_tool_name({"garbage": 1}), "error")

    def test_error_on_non_dict(self):
        self.assertEqual(_normalize_tool_name("not a dict"), "error")


# ======================================================================
# _process_agent_decision
# ======================================================================

class TestProcessAgentDecision(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.config_path = _write_minimal_config(self.tmpdir)
        os.makedirs(os.path.join(self.tmpdir, "output"), exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _make_sd(self, name="fit-physical"):
        return StageDef(name=name, action=name, required=True)

    def test_adjust_params_with_applied(self):
        ctx = _make_mock_ctx()
        sd = self._make_sd()
        op_log = OperationLog(self.tmpdir)
        result = {
            "decision": "adjust_params",
            "tool_name": "adjust_params",
            "arguments": {
                "applied": {"epochs": 10},
                "applied_summary": "epochs=10",
                "warnings": [],
                "ini_snapshot": "[section]\nepochs = 10\n",
            },
            "reasoning": "need more epochs",
        }
        decision = _process_agent_decision(ctx, 1, sd, self.config_path, op_log, result, 2)
        self.assertEqual(decision, "retry")

    def test_accept_quality(self):
        ctx = _make_mock_ctx()
        sd = self._make_sd()
        op_log = OperationLog(self.tmpdir)
        result = {"decision": "accept_quality", "arguments": {}, "reasoning": "good enough"}
        decision = _process_agent_decision(ctx, 1, sd, self.config_path, op_log, result, 2)
        self.assertEqual(decision, "accept")

    def test_fallback_to_previous(self):
        ctx = _make_mock_ctx()
        sd = self._make_sd()
        op_log = OperationLog(self.tmpdir)
        result = {
            "decision": "fallback_to_previous",
            "tool_name": "fallback_to_previous",
            "arguments": {"target": "check-data", "reason": "upstream data issue"},
            "reasoning": "go back",
        }
        decision = _process_agent_decision(ctx, 1, sd, self.config_path, op_log, result, 2)
        self.assertEqual(decision, "fallback")

    def test_error_falls_through_to_accept(self):
        ctx = _make_mock_ctx()
        sd = self._make_sd()
        op_log = OperationLog(self.tmpdir)
        result = {"decision": "error", "arguments": {}, "reasoning": "could not parse"}
        decision = _process_agent_decision(ctx, 1, sd, self.config_path, op_log, result, 0)
        self.assertEqual(decision, "accept")


# ======================================================================
# Agent A — check-data diagnostic
# ======================================================================

class TestAgentA(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.ctx = _make_mock_ctx()
        os.makedirs(os.path.join(self.tmpdir, "output"), exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _make_sd(self):
        return StageDef(name=StageName.CHECK_DATA, action="check-data", required=True)

    @patch("src.agent.subagent.SubAgent")
    def test_agent_a_decides_fallback_to_verifier(self, mock_sa_class):
        mock_sa_class.return_value = _make_mock_subagent(
            json.dumps({"target": "verifier", "reason": "missing columns", "evidence": "VG column absent"}),
        )
        sd = self._make_sd()
        sr = StageResult(ok=False, stage="check-data", error="Validation failed")
        result = _spawn_agent_a(self.ctx, 1, sd, sr, self.tmpdir)
        self.assertEqual(result["fallback"], "verifier")
        self.assertIn("missing columns", result["reason"])
        self.assertIn("VG column absent", result["evidence"])

    @patch("src.agent.subagent.SubAgent")
    def test_agent_a_decides_no_fallback(self, mock_sa_class):
        mock_sa_class.return_value = _make_mock_subagent(
            json.dumps({"target": "", "reason": "permission denied", "evidence": "hspice executable missing"}),
        )
        sd = self._make_sd()
        sr = StageResult(ok=False, stage="check-data", error="Permission denied")
        result = _spawn_agent_a(self.ctx, 1, sd, sr, self.tmpdir)
        self.assertEqual(result["fallback"], "")
        self.assertIn("permission denied", result["reason"])

    @patch("src.agent.subagent.SubAgent")
    def test_agent_a_subagent_error(self, mock_sa_class):
        agent = _make_mock_subagent("garbage")
        agent.status.value = "error"
        agent.error = "LLM timeout"
        mock_sa_class.return_value = agent
        sd = self._make_sd()
        sr = StageResult(ok=False, stage="check-data", error="Validation failed")
        result = _spawn_agent_a(self.ctx, 1, sd, sr, self.tmpdir)
        self.assertEqual(result["fallback"], "")

    @patch("src.agent.subagent.SubAgent")
    def test_agent_a_receives_preparer_handoff(self, mock_sa_class):
        captured_prompt = []
        def capture(*args, **kwargs):
            captured_prompt.append(kwargs.get("prompt", ""))
            return _make_mock_subagent(json.dumps({"target": "verifier", "reason": "test"}))
        mock_sa_class.side_effect = capture

        op_log = OperationLog(self.tmpdir)
        op_log.initialize(1, "")
        op_log.append_attempt("data-prep", StageAttemptLog(
            n=1, trigger=AttemptTrigger.INITIAL, handoff={"observations": ["ok"]},
            ok=True, error="", duration_s=1.0,
        ))
        op_log.append_attempt("verifier", StageAttemptLog(
            n=1, trigger=AttemptTrigger.INITIAL,
            agent_decision={"overall_ok": True, "summary": "all good"},
            ok=True, error="", duration_s=1.0,
        ))

        sd = self._make_sd()
        sr = StageResult(ok=False, stage="check-data", error="Validation failed")
        result = _spawn_agent_a(self.ctx, 1, sd, sr, self.tmpdir, op_log=op_log)
        self.assertEqual(result["fallback"], "verifier")
        prompt = captured_prompt[0]
        self.assertIn("Preparer handoff", prompt)
        self.assertIn("observations", prompt)

    @patch("src.agent.subagent.SubAgent")
    def test_agent_a_receives_downstream_fallback_ctx(self, mock_sa_class):
        captured_prompt = []
        def capture(*args, **kwargs):
            captured_prompt.append(kwargs.get("prompt", ""))
            return _make_mock_subagent(json.dumps({"target": "", "reason": "test"}))
        mock_sa_class.side_effect = capture

        fallback_ctx = {
            "decision": "fallback_to_previous",
            "arguments": {
                "reason": "CV shows wrong charge model",
                "evidence": "derivative mismatch",
                "suggestion": "re-check parameter bounds",
            },
        }
        sd = self._make_sd()
        sr = StageResult(ok=False, stage="check-data", error="Validation failed")
        _spawn_agent_a(self.ctx, 1, sd, sr, self.tmpdir, fallback_ctx=fallback_ctx)
        prompt = captured_prompt[0]
        self.assertIn("CV shows wrong charge model", prompt)
        self.assertIn("derivative mismatch", prompt)


# ======================================================================
# Agent E — final interpretation
# ======================================================================

class TestAgentE(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.ctx = _make_mock_ctx()
        os.makedirs(os.path.join(self.tmpdir, "output"), exist_ok=True)
        self.config_path = _write_minimal_config(self.tmpdir)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    @patch("src.agent.subagent.SubAgent")
    def test_agent_e_spawns_on_success(self, mock_sa_class):
        mock_sa_class.return_value = _make_mock_subagent("# Interpretation\nTest run completed.")
        stages = [StageDef(name="check-data", action="check-data", required=True)]
        op_log = OperationLog(self.tmpdir)
        op_log.initialize(1, "")
        _spawn_agent_e(self.ctx, 1, stages, op_log, self.tmpdir, "", failed=False)
        # Verify FINAL_INTERPRETATION.md was written
        fi_path = os.path.join(self.tmpdir, "FINAL_INTERPRETATION.md")
        self.assertTrue(Path(fi_path).exists())
        content = Path(fi_path).read_text(encoding="utf-8")
        self.assertIn("Interpretation", content)

    @patch("src.agent.subagent.SubAgent")
    def test_agent_e_spawns_on_failure(self, mock_sa_class):
        mock_sa_class.return_value = _make_mock_subagent("# Interpretation\nPipeline failed at check-data.")
        stages = [StageDef(name="check-data", action="check-data", required=True)]
        op_log = OperationLog(self.tmpdir)
        op_log.initialize(1, "")
        _spawn_agent_e(self.ctx, 1, stages, op_log, self.tmpdir, "", failed=True,
                       failed_stage="check-data", config_path=self.config_path)
        fi_path = os.path.join(self.tmpdir, "FINAL_INTERPRETATION.md")
        self.assertTrue(Path(fi_path).exists())

    @patch("src.agent.subagent.SubAgent")
    def test_agent_e_minimal_run_receives_hint(self, mock_sa_class):
        captured_prompt = []
        def capture(*args, **kwargs):
            captured_prompt.append(kwargs.get("prompt", ""))
            return _make_mock_subagent("# Interpretation\nMinimal run.")
        mock_sa_class.side_effect = capture

        minimial_config = _write_minimal_config(self.tmpdir, minimial_run=True)
        stages = [StageDef(name="check-data", action="check-data", required=True)]
        op_log = OperationLog(self.tmpdir)
        op_log.initialize(1, "")
        _spawn_agent_e(self.ctx, 1, stages, op_log, self.tmpdir, "", failed=False,
                       config_path=minimial_config)
        prompt = captured_prompt[0]
        self.assertIn("quick diagnostics only", prompt)
        self.assertIn("do NOT critique model quality as if this were a full production run", prompt)

    @patch("src.agent.subagent.SubAgent")
    def test_agent_e_non_minimal_run_no_hint(self, mock_sa_class):
        captured_prompt = []
        def capture(*args, **kwargs):
            captured_prompt.append(kwargs.get("prompt", ""))
            return _make_mock_subagent("# Interpretation\nFull run.")
        mock_sa_class.side_effect = capture

        stages = [StageDef(name="check-data", action="check-data", required=True)]
        op_log = OperationLog(self.tmpdir)
        op_log.initialize(1, "")
        _spawn_agent_e(self.ctx, 1, stages, op_log, self.tmpdir, "", failed=False,
                       config_path=self.config_path)
        prompt = captured_prompt[0]
        self.assertNotIn("quick diagnostics only", prompt)


# ======================================================================
# Orchestrator: Agent A / check-data hard failure paths
# ======================================================================

class TestOrchestratorAgentAFlow(unittest.TestCase):
    """Integration tests for check-data failure → Agent A decisions."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.config_path = _write_minimal_config(self.tmpdir)
        os.makedirs(os.path.join(self.tmpdir, "output"), exist_ok=True)
        self.dataset_dir = os.path.join(self.tmpdir, "dataset")
        os.makedirs(self.dataset_dir, exist_ok=True)
        # Create a minimal dataset file so copy_dataset_to_data_prep works
        dummy_xlsx = os.path.join(self.dataset_dir, "IDVD_VDS01_train.xlsx")
        wb = openpyxl.Workbook()
        wb.active.title = "long"
        wb.save(dummy_xlsx)
        wb.close()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _create_run(self, ctx):
        ok, rid, _ = ctx.workflow_man.create_run(self.dataset_dir, output_root=self.tmpdir)
        profile = DatasetProfile(static_ok=True, cv_ok=True, has_hysteresis=False)
        ctx.workflow_man.set_dataset_profile(rid, profile)
        return rid

    @patch("src.workflow.dg_tft.data_preparer.run_verifier_stage")
    @patch("src.workflow.dg_tft.data_preparer.run_preparer_stage")
    @patch("src.agent.subagent.SubAgent")
    @patch("src.workflow.dg_tft.orchestrator.run_stage")
    def test_check_data_fail_agent_a_fallback_to_verifier(
        self, mock_run, mock_sa, mock_prep, mock_verify,
    ):
        mock_run.return_value = StageResult(ok=False, stage="check-data", error="missing columns")
        mock_sa.return_value = _make_mock_subagent(
            json.dumps({"target": "verifier", "reason": "bad data", "evidence": "sheet missing"}),
        )
        mock_prep.return_value = {"ok": True, "handoff": {"observations": ["ok"]}, "error": ""}
        mock_verify.return_value = {"ok": False, "feedback": {}, "issues": ["critical defect"], "worker_handoff": {}}
        ctx = _make_mock_ctx({"DG_TFT_PREP_MAX_RETRIES": 2})
        rid = self._create_run(ctx)
        result, summary = run_workflow(ctx, rid, entry="check-data", dataset_dir=self.dataset_dir,
                                       config_path=self.config_path)
        self.assertFalse(result)
        self.assertIn("verifier", summary.lower())
        self.assertIn("exhausted", summary.lower())

    @patch("src.agent.subagent.SubAgent")
    @patch("src.workflow.dg_tft.orchestrator.run_stage")
    def test_check_data_fail_agent_a_refuses_fallback(self, mock_run, mock_sa):
        mock_run.return_value = StageResult(ok=False, stage="check-data", error="hspice missing")
        mock_sa.return_value = _make_mock_subagent(
            json.dumps({"target": "", "reason": "environmental error", "evidence": "no hspice"}),
        )
        ctx = _make_mock_ctx()
        rid = self._create_run(ctx)
        result, summary = run_workflow(ctx, rid, entry="check-data", dataset_dir=self.dataset_dir,
                                       config_path=self.config_path)
        self.assertFalse(result)
        self.assertIn("check-data", summary)


# ======================================================================
# Orchestrator: Agent B/C/D hard failure / quality fail paths  
# ======================================================================

class TestOrchestratorModelingAgentFlow(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.config_path = _write_minimal_config(self.tmpdir)
        os.makedirs(os.path.join(self.tmpdir, "output"), exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _create_run(self, ctx):
        ok, rid, _ = ctx.workflow_man.create_run("/data/test", output_root=self.tmpdir)
        profile = DatasetProfile(static_ok=True, cv_ok=True, has_hysteresis=False)
        ctx.workflow_man.set_dataset_profile(rid, profile)
        return rid

    @patch("src.agent.subagent.SubAgent")
    @patch("src.workflow.dg_tft.orchestrator.run_stage")
    @patch("src.workflow.dg_tft.orchestrator._assess_stage_quality")
    def test_fit_physical_hard_fail_agent_b_adjust_params(self, mock_q, mock_run, mock_sa):
        mock_sa.return_value = _make_mock_subagent(
            json.dumps({"applied": {"epochs": 50}, "applied_summary": "epochs=50", "warnings": [], "ini_snapshot": ""}),
        )
        responses = [
            StageResult(ok=True, stage="check-data", summary="OK"),
            StageResult(ok=False, stage="fit-physical", error="diverged"),
            StageResult(ok=True, stage="fit-physical", summary="recovered"),  # retry success
            StageResult(ok=True, stage="train-static", summary="OK"),
        ]
        mock_run.side_effect = responses
        mock_q.return_value = (True, [])  # quality always ok to focus on hard-failure path

        ctx = _make_mock_ctx()
        rid = self._create_run(ctx)
        result, summary = run_workflow(ctx, rid, entry="check-data", dataset_dir="/data/test",
                                       config_path=self.config_path)
        self.assertTrue(result)
        self.assertIn("completed", summary)
        self.assertEqual(mock_run.call_count, 4)

    @patch("src.agent.subagent.SubAgent")
    @patch("src.workflow.dg_tft.orchestrator.run_stage")
    @patch("src.workflow.dg_tft.orchestrator._assess_stage_quality")
    def test_fit_physical_quality_fail_agent_b_accept(self, mock_q, mock_run, mock_sa):
        mock_sa.return_value = _make_mock_subagent(
            json.dumps({"accept_quality": {}}),
        )
        mock_run.side_effect = [
            StageResult(ok=True, stage="check-data", summary="OK"),
            StageResult(ok=True, stage="fit-physical", summary="ok but low quality"),
            StageResult(ok=True, stage="train-static", summary="OK"),
        ]
        mock_q.return_value = (False, ["High RMS error"])  # quality fail triggers agent

        ctx = _make_mock_ctx()
        rid = self._create_run(ctx)
        result, summary = run_workflow(ctx, rid, entry="check-data", dataset_dir="/data/test",
                                       config_path=self.config_path)
        self.assertTrue(result)

    @patch("src.agent.subagent.SubAgent")
    @patch("src.workflow.dg_tft.orchestrator.run_stage")
    @patch("src.workflow.dg_tft.orchestrator._assess_stage_quality")
    def test_agent_b_fallback_to_checkdata_then_agent_a_refuses(self, mock_q, mock_run, mock_sa):
        """B quality fail → fallback to check-data → check-data fails → Agent A refuses."""
        call_count = {"n": 0}
        def make_agent(*args, **kwargs):
            call_count["n"] += 1
            st = kwargs.get("subagent_type", "")
            if st == DIAGNOSTIC_AGENT_LABEL:
                return _make_mock_subagent(json.dumps(
                    {"target": "", "reason": "env error", "evidence": "hspice missing"},
                ))
            else:
                return _make_mock_subagent(json.dumps(
                    {"target": "check-data", "reason": "upstream data issue", "evidence": "CV anomalies"},
                ))
        mock_sa.side_effect = make_agent

        mock_run.side_effect = [
            StageResult(ok=True, stage="check-data", summary="OK"),
            StageResult(ok=True, stage="fit-physical", summary="ok but low"),
            StageResult(ok=False, stage="check-data", error="still bad"),
        ]
        mock_q.return_value = (False, ["High RMS"])

        ctx = _make_mock_ctx()
        rid = self._create_run(ctx)
        result, summary = run_workflow(ctx, rid, entry="check-data", dataset_dir="/data/test",
                                       config_path=self.config_path)
        self.assertFalse(result)
        run = ctx.workflow_man.get_run(rid)
        self.assertEqual(run.status, WorkflowStatus.FAILED)
        self.assertGreaterEqual(call_count["n"], 2, "Agent B and Agent A should both have spawned")


# ======================================================================
# Orchestrator: Two-stage fallback (D → C)
# ======================================================================

class TestOrchestratorTwoStageFallback(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.config_path = _write_minimal_config(self.tmpdir)
        os.makedirs(os.path.join(self.tmpdir, "output"), exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _create_run(self, ctx):
        ok, rid, _ = ctx.workflow_man.create_run("/data/test", output_root=self.tmpdir)
        profile = DatasetProfile(static_ok=True, cv_ok=True, has_hysteresis=True,
                                 suggested_hysteresis_dir="/data/hyst")
        ctx.workflow_man.set_dataset_profile(rid, profile)
        return rid

    @patch("src.agent.subagent.SubAgent")
    @patch("src.workflow.dg_tft.orchestrator.run_stage")
    @patch("src.workflow.dg_tft.orchestrator._assess_stage_quality")
    def test_d_fallback_to_c_then_c_adjusts_and_recovers(self, mock_q, mock_run, mock_sa):
        """D quality fail → fallback to C → C adjusts and re-runs → D re-runs → success."""
        call_order = []
        def make_agent(*args, **kwargs):
            st = kwargs.get("subagent_type", "")
            call_order.append(st)
            if st == MODELING_AGENT_LABEL and "train-hysteresis" in str(kwargs.get("prompt", "")):
                return _make_mock_subagent(json.dumps(
                    {"target": "train-static", "reason": "upstream data issue", "evidence": ""},
                ))
            return _make_mock_subagent(json.dumps(
                {"applied": {"epochs": 50}, "applied_summary": "epochs=50", "warnings": [], "ini_snapshot": ""},
            ))
        mock_sa.side_effect = make_agent

        mock_run.side_effect = [
            StageResult(ok=True, stage="check-data", summary="OK"),
            StageResult(ok=True, stage="fit-physical", summary="OK"),
            StageResult(ok=True, stage="train-static", summary="OK"),
            StageResult(ok=True, stage="train-hysteresis", summary="low"),
            StageResult(ok=True, stage="train-static", summary="re-run"),
            StageResult(ok=True, stage="train-static", summary="re-run ok"),
            StageResult(ok=True, stage="train-hysteresis", summary="ok now"),
        ]
        def quality(ctx, stage_name, output_dir, config_path):
            if stage_name == "train-hysteresis":
                return False, ["High RMS"]
            return True, []
        mock_q.side_effect = quality

        ctx = _make_mock_ctx()
        rid = self._create_run(ctx)
        result, summary = run_workflow(ctx, rid, entry="check-data", dataset_dir="/data/test",
                                       config_path=self.config_path)
        self.assertTrue(result)


# ======================================================================
# Orchestrator: minimal_run gating
# ======================================================================

class TestOrchestratorMinimalRun(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.config_path = _write_minimal_config(self.tmpdir, minimial_run=True)
        os.makedirs(os.path.join(self.tmpdir, "output"), exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _create_run(self, ctx):
        ok, rid, _ = ctx.workflow_man.create_run("/data/test", output_root=self.tmpdir)
        profile = DatasetProfile(static_ok=True, cv_ok=True, has_hysteresis=False)
        ctx.workflow_man.set_dataset_profile(rid, profile)
        return rid

    @patch("src.agent.subagent.SubAgent")
    @patch("src.workflow.dg_tft.orchestrator.run_stage")
    def test_minimal_run_agents_abcd_skipped_e_runs(self, mock_run, mock_sa):
        mock_sa.return_value = _make_mock_subagent("# Interpretation\nQuick test run.")
        # Hard failure at fit-physical — would normally trigger Agent B
        mock_run.side_effect = [
            StageResult(ok=True, stage="check-data", summary="OK"),
            StageResult(ok=False, stage="fit-physical", error="diverged"),
            StageResult(ok=True, stage="train-static", summary="OK"),
        ]
        ctx = _make_mock_ctx()
        rid = self._create_run(ctx)
        result, summary = run_workflow(ctx, rid, entry="check-data", dataset_dir="/data/test",
                                       config_path=self.config_path)
        self.assertFalse(result)
        # Agent E should still have been spawned (SubAgent called at least once)
        self.assertGreaterEqual(mock_sa.call_count, 1)
        # Verify FINAL_INTERPRETATION.md exists
        fi_path = os.path.join(self.tmpdir, "output", "FINAL_INTERPRETATION.md")
        self.assertTrue(Path(fi_path).exists() or
                        any("FINAL_INTERPRETATION.md" in str(c) for c in mock_sa.call_args_list)
                        if mock_sa.call_args_list else False)

    @patch("src.agent.subagent.SubAgent")
    @patch("src.workflow.dg_tft.orchestrator.run_stage")
    def test_minimal_run_check_data_hard_fail_no_agent_a(self, mock_run, mock_sa):
        expected_responses = []
        def track_subagent_type(*args, **kwargs):
            expected_responses.append(kwargs.get("subagent_type", ""))
            return _make_mock_subagent("# Interpretation")
        mock_sa.side_effect = track_subagent_type

        mock_run.return_value = StageResult(ok=False, stage="check-data", error="missing data")
        ctx = _make_mock_ctx()
        rid = self._create_run(ctx)
        result, summary = run_workflow(ctx, rid, entry="check-data", dataset_dir="/data/test",
                                       config_path=self.config_path)
        self.assertFalse(result)
        # Agent A (diagnostic type) should NOT have been spawned
        diag_calls = [t for t in expected_responses if t == DIAGNOSTIC_AGENT_LABEL]
        self.assertEqual(len(diag_calls), 0, "Agent A should be skipped in minimal_run")
        # Agent E (interpreter type) SHOULD have been spawned
        interp_calls = [t for t in expected_responses if t == INTERPRETER_AGENT_LABEL]
        self.assertGreaterEqual(len(interp_calls), 1, "Agent E should always run")


# ======================================================================
# _run_agent_stage — Preparer/Verifier retry + handoff propagation
# ======================================================================

class TestRunAgentStage(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.output_dir = os.path.join(self.tmpdir, "output")
        os.makedirs(self.output_dir, exist_ok=True)
        self.dataset_dir = os.path.join(self.tmpdir, "dataset")
        os.makedirs(self.dataset_dir, exist_ok=True)
        self.raw_dir = os.path.join(self.tmpdir, "raw")
        os.makedirs(self.raw_dir, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    @patch("src.workflow.dg_tft.data_preparer.run_preparer_stage")
    @patch("src.workflow.dg_tft.data_preparer.run_verifier_stage")
    def test_preparer_ok_verifier_ok(self, mock_verify, mock_prep):
        mock_prep.return_value = {"ok": True, "handoff": {"observations": ["processed"]}, "error": ""}
        mock_verify.return_value = {
            "ok": True, "feedback": {"overall_ok": True},
            "issues": [], "worker_handoff": {"observations": ["processed"]},
        }

        ctx = _make_mock_ctx()
        wm = ctx.workflow_man
        ok, rid, _ = wm.create_run(self.dataset_dir, output_root=self.output_dir)
        wm.set_dataset_profile(rid, DatasetProfile(static_ok=True, cv_ok=True))
        op_log = OperationLog(self.output_dir)
        op_log.initialize(rid, "")
        retry_counts = {}

        sd = StageDef(name=StageName.DATA_PREP, action="data-prep", required=True, kind="agent")
        attempt_dir, ok = _run_agent_stage(ctx, rid, sd, self.output_dir, self.dataset_dir,
                                           self.raw_dir, "", op_log, None, retry_counts, 3)
        self.assertTrue(ok)
        self.assertIn("data_prep", attempt_dir)
        self.assertIn("data-prep", op_log.data.get("stages", {}))
        self.assertEqual(len(op_log.data["stages"]["data-prep"]), 1)

    @patch("src.workflow.dg_tft.data_preparer.run_preparer_stage")
    @patch("src.workflow.dg_tft.data_preparer.run_verifier_stage")
    def test_verifier_fail_returns_false_for_orchestrator_retry(self, mock_verify, mock_prep):
        """_run_agent_stage returns (attempt_dir, False) on verifier fail so that
        run_workflow's while+pointer loop can reset back to DATA_PREP."""
        mock_prep.return_value = {"ok": True, "handoff": {"observations": ["ok"]}, "error": ""}
        mock_verify.return_value = {
            "ok": False, "feedback": {"overall_ok": False}, "issues": ["bad CV"], "worker_handoff": {},
        }

        ctx = _make_mock_ctx()
        wm = ctx.workflow_man
        ok, rid, _ = wm.create_run(self.dataset_dir, output_root=self.output_dir)
        wm.set_dataset_profile(rid, DatasetProfile(static_ok=True, cv_ok=True))
        op_log = OperationLog(self.output_dir)
        op_log.initialize(rid, "")
        retry_counts = {}
        sd = StageDef(name=StageName.VERIFIER, action="verifier", required=True, kind="agent")
        attempt_dir, is_ok = _run_agent_stage(ctx, rid, sd, self.output_dir, self.dataset_dir,
                                           self.raw_dir, "", op_log, None, retry_counts, 3)
        self.assertFalse(is_ok, "_run_agent_stage should return False on verifier fail")
        self.assertIn("verifier", op_log.data.get("stages", {}))
        self.assertEqual(len(op_log.data["stages"]["verifier"]), 1)

    @patch("src.workflow.dg_tft.data_preparer.run_preparer_stage")
    @patch("src.workflow.dg_tft.data_preparer.run_verifier_stage")
    def test_agent_a_fallback_injected_into_verifier_handoff(self, mock_verify, mock_prep):
        captured_handoff = []
        mock_prep.return_value = {"ok": True, "handoff": {"observations": ["processed"], "actions": ["merged sheets"]}, "error": ""}
        def verify_capture(*args, **kwargs):
            captured_handoff.append(kwargs.get("handoff"))
            return {"ok": True, "feedback": {"overall_ok": True}, "issues": [], "worker_handoff": {}}
        mock_verify.side_effect = verify_capture

        ctx = _make_mock_ctx()
        wm = ctx.workflow_man
        ok, rid, _ = wm.create_run(self.dataset_dir, output_root=self.output_dir)
        wm.set_dataset_profile(rid, DatasetProfile(static_ok=True, cv_ok=True))
        op_log = OperationLog(self.output_dir)
        op_log.initialize(rid, "")
        # Simulate Preparer already done
        op_log.append_attempt("data-prep", StageAttemptLog(
            n=1, trigger=AttemptTrigger.INITIAL,
            handoff={"observations": ["processed"], "actions": ["merged sheets"]},
            ok=True, error="", duration_s=1.0,
        ))
        retry_counts = {}

        # Build Agent A fallback context
        fallback_ctx = {
            "fallback": "verifier",
            "reason": "missing VG column",
            "evidence": "Sheet long",
            "suggestion": "re-check export",
        }
        sd = StageDef(name=StageName.VERIFIER, action="verifier", required=True, kind="agent")
        _run_agent_stage(ctx, rid, sd, self.output_dir, self.dataset_dir,
                         self.raw_dir, "", op_log, fallback_ctx, retry_counts, 3)
        self.assertEqual(len(captured_handoff), 1)
        handoff = captured_handoff[0]
        self.assertIn("observations", handoff)
        self.assertIn("agent_a_fallback", handoff)
        self.assertEqual(handoff["agent_a_fallback"]["fallback_from"], "Agent A (check-data)")
        self.assertEqual(handoff["agent_a_fallback"]["reason"], "missing VG column")

    def test_stage_has_modeling_agent(self):
        self.assertTrue(_stage_has_modeling_agent("fit-physical"))
        self.assertTrue(_stage_has_modeling_agent("train-static"))
        self.assertTrue(_stage_has_modeling_agent("train-hysteresis"))
        self.assertFalse(_stage_has_modeling_agent("check-data"))
        self.assertFalse(_stage_has_modeling_agent("data-prep"))
        self.assertFalse(_stage_has_modeling_agent("verifier"))


# ======================================================================
# Operation log: ensure agent decisions and fallback are recorded
# ======================================================================

class TestOperationLogAgentEntries(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        os.makedirs(os.path.join(self.tmpdir, "output"), exist_ok=True)
        self.config_path = _write_minimal_config(self.tmpdir)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    @patch("src.agent.subagent.SubAgent")
    @patch("src.workflow.dg_tft.orchestrator.run_stage")
    @patch("src.workflow.dg_tft.orchestrator._assess_stage_quality")
    def test_agent_adjust_params_recorded_in_oplog(self, mock_q, mock_run, mock_sa):
        mock_run.side_effect = [
            StageResult(ok=True, stage="check-data", summary="OK"),
            StageResult(ok=True, stage="fit-physical", summary="low quality"),
            StageResult(ok=True, stage="fit-physical", summary="re-run 1"),
            StageResult(ok=True, stage="fit-physical", summary="re-run 2"),
            StageResult(ok=True, stage="fit-physical", summary="re-run 3"),
            StageResult(ok=True, stage="train-static", summary="OK"),
        ]
        mock_sa.return_value = _make_mock_subagent(
            json.dumps({
                "applied": {"epochs": 50},
                "applied_summary": "epochs=50",
                "warnings": [],
                "ini_snapshot": "",
            }),
        )
        fit_q_count = {"n": 0}
        def quality(ctx, stage_name, output_dir, config_path):
            if stage_name == "fit-physical":
                fit_q_count["n"] += 1
                if fit_q_count["n"] <= 3:
                    return False, ["High RMS"]
            return True, []
        mock_q.side_effect = quality

        ctx = _make_mock_ctx()
        ok, rid, _ = ctx.workflow_man.create_run("/data/test", output_root=self.tmpdir)
        profile = DatasetProfile(static_ok=True, cv_ok=True, has_hysteresis=False)
        ctx.workflow_man.set_dataset_profile(rid, profile)
        result, summary = run_workflow(ctx, rid, entry="check-data", dataset_dir="/data/test",
                                       config_path=self.config_path)
        self.assertTrue(result)
        op_log_path = os.path.join(self.tmpdir, "output", "operation_log.json")
        self.assertTrue(Path(op_log_path).exists(), "operation_log.json should be written")
        with open(op_log_path, "r", encoding="utf-8") as f:
            log_data = json.load(f)
        stages = log_data.get("stages", {})
        self.assertIn("fit-physical", stages)
        triggers = [a.get("trigger") for a in stages["fit-physical"]]
        self.assertIn("agent_adjust", triggers)

    @patch("src.agent.subagent.SubAgent")
    @patch("src.workflow.dg_tft.orchestrator.run_stage")
    @patch("src.workflow.dg_tft.orchestrator._assess_stage_quality")
    def test_agent_fallback_recorded_in_oplog(self, mock_q, mock_run, mock_sa):
        call_count = {"n": 0}
        def make_agent(*args, **kwargs):
            call_count["n"] += 1
            st = kwargs.get("subagent_type", "")
            if st == DIAGNOSTIC_AGENT_LABEL:
                return _make_mock_subagent(json.dumps(
                    {"target": "", "reason": "env", "evidence": ""},
                ))
            return _make_mock_subagent(json.dumps(
                {"target": "check-data", "reason": "upstream data issue", "evidence": "blocky CV"},
            ))
        mock_sa.side_effect = make_agent

        mock_run.side_effect = [
            StageResult(ok=True, stage="check-data", summary="OK"),
            StageResult(ok=True, stage="fit-physical", summary="low quality"),
            StageResult(ok=False, stage="check-data", error="bad"),
        ]
        def quality(ctx, stage_name, output_dir, config_path):
            if stage_name == "fit-physical":
                return False, ["High RMS"]
            return True, []
        mock_q.side_effect = quality

        ctx = _make_mock_ctx()
        ok, rid, _ = ctx.workflow_man.create_run("/data/test", output_root=self.tmpdir)
        profile = DatasetProfile(static_ok=True, cv_ok=True, has_hysteresis=False)
        ctx.workflow_man.set_dataset_profile(rid, profile)
        result, summary = run_workflow(ctx, rid, entry="check-data", dataset_dir="/data/test",
                                       config_path=self.config_path)
        self.assertFalse(result)
        op_log_path = os.path.join(self.tmpdir, "output", "operation_log.json")
        self.assertTrue(Path(op_log_path).exists())
        with open(op_log_path, "r", encoding="utf-8") as f:
            log_data = json.load(f)
        stages = log_data.get("stages", {})
        self.assertIn("fit-physical", stages)
        triggers = [a.get("trigger") for a in stages["fit-physical"]]
        self.assertIn("fallback_sent", triggers)


if __name__ == "__main__":
    unittest.main()
