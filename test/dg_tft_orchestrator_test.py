# -*- coding: utf-8 -*-
"""
Unit tests for stages.py and orchestrator.py.
Run: python -m unittest test.dg_tft_orchestrator_test -v
"""
import sys
import os
import json
import unittest
import tempfile
import shutil
from unittest.mock import patch, MagicMock
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.workflow.dg_tft.control import WorkflowStopped
from src.workflow.dg_tft.schema import (
    WorkflowRun,
    WorkflowStatus,
    StageResult,
    StageName,
    DatasetProfile,
)
from src.workflow.dg_tft.manager import WorkflowManager
from src.workflow.dg_tft.stages import (
    get_stages,
    get_stage_by_name,
    StageDef,
    _CORE_STAGES,
)
from src.workflow.dg_tft.orchestrator import (
    run_workflow,
    OrchestratorResult,
    _read_output_dir,
    _verify_artifacts,
    _noop_stage,
    _stage_done,
    _load_profile,
)


class MockContext:
    def __init__(self, configs=None):
        self.agent_configs = configs or {}
        self.workflow_man = WorkflowManager()
        self.agent_list = {}
        self.workflow_man.session_uuid = "test_session"
        self.workflow_man._save_runs = lambda: None
        self.agent_id = "test_agent"
        self.subagent_mute = False
        self.console = None
        self.session_uuid = "test_session"


class TestStageDefinitions(unittest.TestCase):

    def test_core_stages_not_empty(self):
        self.assertGreater(len(_CORE_STAGES), 0)

    def test_all_stage_names_defined(self):
        names = {sd.name for sd in _CORE_STAGES}
        for attr_name in dir(StageName):
            if attr_name.startswith("_") or attr_name == "STAGE_ORDER":
                continue
            val = getattr(StageName, attr_name)
            if isinstance(val, str):
                self.assertIn(val, names, f"Stage {val} not in _CORE_STAGES")

    def test_first_script_stage_is_check_data(self):
        # Preparer and Verifier are agent stages at the head;
        # check-data is the first script stage
        check_idx = next(i for i, sd in enumerate(_CORE_STAGES) if sd.name == StageName.CHECK_DATA)
        self.assertGreater(check_idx, 0)
        self.assertEqual(_CORE_STAGES[check_idx].name, StageName.CHECK_DATA)
        self.assertTrue(_CORE_STAGES[check_idx].required)

    def test_check_data_artifacts(self):
        sd = get_stage_by_name(StageName.CHECK_DATA)
        self.assertIn("dataset_validation.json", sd.key_artifacts)

    def test_fit_physical_artifacts(self):
        sd = get_stage_by_name(StageName.FIT_PHYSICAL)
        self.assertIn("fitted_params.json", sd.key_artifacts)
        self.assertEqual(sd.expected_output_subdir, "01_physical_kernel")

    def test_train_static_artifacts(self):
        sd = get_stage_by_name(StageName.TRAIN_STATIC)
        self.assertEqual(sd.expected_output_subdir, "03_full_model")

    def test_train_hysteresis_optional(self):
        sd = get_stage_by_name(StageName.TRAIN_HYSTERESIS)
        self.assertFalse(sd.required)
        self.assertEqual(sd.skip_if_profile_false, "has_hysteresis")

    def test_depends_on_stage(self):
        sd = get_stage_by_name(StageName.FIT_PHYSICAL)
        self.assertIn(StageName.CHECK_DATA, sd.depends_on)
        sd_verifier = get_stage_by_name(StageName.CHECK_DATA)
        self.assertIn(StageName.VERIFIER, sd_verifier.depends_on)
        sd_prep = get_stage_by_name(StageName.VERIFIER)
        self.assertIn(StageName.DATA_PREP, sd_prep.depends_on)

    def test_get_stages_all_without_profile(self):
        stages = get_stages()
        self.assertEqual(len(stages), len(_CORE_STAGES))

    def test_get_stages_filters_hysteresis_when_no_data(self):
        profile = DatasetProfile(has_hysteresis=False)
        stages = get_stages(profile)
        names = [sd.name for sd in stages]
        self.assertNotIn(StageName.TRAIN_HYSTERESIS, names)

    def test_get_stages_keeps_hysteresis_when_has_data(self):
        profile = DatasetProfile(has_hysteresis=True)
        stages = get_stages(profile)
        names = [sd.name for sd in stages]
        self.assertIn(StageName.TRAIN_HYSTERESIS, names)

    def test_get_stage_by_name_unknown(self):
        self.assertIsNone(get_stage_by_name("nonexistent"))


class TestOrchestratorHelpers(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_read_output_dir_from_config(self):
        ini_path = os.path.join(self.tmpdir, "config.ini")
        Path(ini_path).write_text(
            "[paths]\ndata_dir = /data\noutput_dir = /output/run\n",
            encoding="utf-8",
        )
        self.assertEqual(_read_output_dir(ini_path), "/output/run")

    def test_read_output_dir_missing_section(self):
        ini_path = os.path.join(self.tmpdir, "config.ini")
        Path(ini_path).write_text("[runtime]\nseed=42\n", encoding="utf-8")
        result = _read_output_dir(ini_path)
        self.assertTrue(os.path.isabs(result))

    def test_verify_artifacts_all_exist(self):
        sd = StageDef(name="test", action="test", required=True,
                      key_artifacts=["a.txt", "b.txt"],
                      expected_output_subdir="out")
        base = os.path.join(self.tmpdir, "out")
        os.makedirs(base, exist_ok=True)
        Path(os.path.join(base, "a.txt")).write_text("")
        Path(os.path.join(base, "b.txt")).write_text("")
        missing = _verify_artifacts(sd, self.tmpdir)
        self.assertEqual(missing, [])

    def test_verify_artifacts_missing(self):
        sd = StageDef(name="test", action="test", required=True,
                      key_artifacts=["a.txt", "missing.txt"])
        base = os.path.join(self.tmpdir, "out")
        os.makedirs(base, exist_ok=True)
        Path(os.path.join(base, "a.txt")).write_text("")
        missing = _verify_artifacts(sd, os.path.join(self.tmpdir, "out"))
        self.assertEqual(len(missing), 1)

    def test_verify_artifacts_no_artifacts(self):
        sd = StageDef(name="test", action="test", required=False,
                      key_artifacts=[])
        self.assertEqual(_verify_artifacts(sd, self.tmpdir), [])

    def test_noop_stage_final_report(self):
        sd = StageDef(name=StageName.FINAL_REPORT, action="final-report", required=False)
        self.assertTrue(_noop_stage(sd))

    def test_noop_stage_regular(self):
        sd = StageDef(name=StageName.CHECK_DATA, action="check-data", required=True)
        self.assertFalse(_noop_stage(sd))

    def test_stage_done_true(self):
        run = WorkflowRun(run_id=1, stage_results={
            "check-data": StageResult(ok=True, stage="check-data", summary="OK"),
        })
        self.assertTrue(_stage_done(run, "check-data"))

    def test_stage_done_false_not_ok(self):
        run = WorkflowRun(run_id=1, stage_results={
            "check-data": StageResult(ok=False, stage="check-data", error="err"),
        })
        self.assertFalse(_stage_done(run, "check-data"))

    def test_stage_done_false_skipped(self):
        run = WorkflowRun(run_id=1, stage_results={
            "check-data": StageResult(ok=True, stage="check-data", skipped=True, skip_reason="no data"),
        })
        self.assertFalse(_stage_done(run, "check-data"))

    def test_stage_done_false_missing(self):
        run = WorkflowRun(run_id=1)
        self.assertFalse(_stage_done(run, "check-data"))


class TestOrchestratorRunWorkflow(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.ctx = MockContext({
            "DG_TFT_MODELING_PYTHON_EXE": sys.executable,
            "DG_TFT_AGENT_PYTHON_EXE": sys.executable,
        })
        self.config_path = os.path.join(self.tmpdir, "model_config.ini")
        Path(self.config_path).write_text(
            "[paths]\ndata_dir = /data\noutput_dir = " + self.tmpdir.replace("\\", "/") + "/output\n"
            "hysteresis_data_dir = /data\nhspice_exe =\n"
            "[hysteresis]\nenabled = yes\n"
            "[temperature]\nmode = auto\n"
            "[deployment]\nrun_hspice = no\n",
            encoding="utf-8",
        )
        os.makedirs(os.path.join(self.tmpdir, "output"), exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _mock_stage_ok(self, ctx, action, *args, **kwargs):
        return StageResult(ok=True, stage=action, summary=f"{action} done")

    def _mock_stage_fail(self, ctx, action, *args, **kwargs):
        return StageResult(ok=False, stage=action, error="test error")

    @patch("src.workflow.dg_tft.orchestrator.run_stage")
    def test_full_success_flow(self, mock_run):
        mock_run.side_effect = self._mock_stage_ok
        ok, rid, _ = self.ctx.workflow_man.create_run(
            "/data/test", output_root=self.tmpdir,
        )
        profile = DatasetProfile(
            static_ok=True, cv_ok=True,
            iv_families=["idvd"], has_hysteresis=True,
            suggested_hysteresis_dir="/data/hyst",
        )
        self.ctx.workflow_man.set_dataset_profile(rid, profile)

        result, summary = run_workflow(self.ctx, rid, entry="check-data", dataset_dir="/data/test", config_path=self.config_path)

        self.assertTrue(result)
        self.assertIn("completed", summary)
        run = self.ctx.workflow_man.get_run(rid)
        self.assertEqual(run.status, WorkflowStatus.DONE)

    @patch("src.workflow.dg_tft.orchestrator.run_stage")
    def test_required_stage_fails_but_pipeline_continues(self, mock_run):
        # Pipeline continues after failure; all stages run.
        # entry="check-data" → check-data, fit-physical, train-static, final-report(noop)
        responses = [
            StageResult(ok=True, stage="check-data", summary="OK"),
            StageResult(ok=False, stage="fit-physical", error="divergence"),
            StageResult(ok=True, stage="train-static", summary="OK"),
        ]
        mock_run.side_effect = responses
        ok, rid, _ = self.ctx.workflow_man.create_run(
            "/data/test", output_root=self.tmpdir,
        )
        profile = DatasetProfile(static_ok=True, cv_ok=True, has_hysteresis=False)
        self.ctx.workflow_man.set_dataset_profile(rid, profile)

        result, summary = run_workflow(self.ctx, rid, entry="check-data", dataset_dir="/data/test", config_path=self.config_path)

        self.assertFalse(result)
        self.assertIn("fit-physical", summary)
        run = self.ctx.workflow_man.get_run(rid)
        self.assertEqual(run.status, WorkflowStatus.FAILED)
        self.assertIn("check-data", run.stage_results)
        self.assertIn("fit-physical", run.stage_results)

    @patch("src.workflow.dg_tft.orchestrator.run_stage")
    def test_skip_hysteresis_when_no_data(self, mock_run):
        mock_run.side_effect = self._mock_stage_ok
        ok, rid, _ = self.ctx.workflow_man.create_run(
            "/data/test", output_root=self.tmpdir,
        )
        profile = DatasetProfile(static_ok=True, cv_ok=True, has_hysteresis=False)
        self.ctx.workflow_man.set_dataset_profile(rid, profile)

        result, summary = run_workflow(self.ctx, rid, entry="check-data", dataset_dir="/data/test", config_path=self.config_path)

        self.assertTrue(result)
        run = self.ctx.workflow_man.get_run(rid)
        self.assertEqual(run.status, WorkflowStatus.DONE)
        # hysteresis is skipped (not filtered) — recorded as skipped in stage_results
        hyster_sr = run.stage_results.get(StageName.TRAIN_HYSTERESIS)
        self.assertTrue(hyster_sr is None or hyster_sr.skipped)

    @patch("src.workflow.dg_tft.orchestrator.run_stage")
    def test_optional_stage_failure_continues(self, mock_run):
        responses = [
            StageResult(ok=True, stage="check-data", summary="OK"),
            StageResult(ok=True, stage="fit-physical", summary="OK"),
            StageResult(ok=True, stage="train-static", summary="OK"),
            StageResult(ok=False, stage="train-hysteresis", error="HSPICE crash"),
        ]
        mock_run.side_effect = responses
        ok, rid, _ = self.ctx.workflow_man.create_run(
            "/data/test", output_root=self.tmpdir,
        )
        profile = DatasetProfile(static_ok=True, cv_ok=True, has_hysteresis=True)
        self.ctx.workflow_man.set_dataset_profile(rid, profile)

        result, summary = run_workflow(self.ctx, rid, entry="check-data", dataset_dir="/data/test", config_path=self.config_path)

        self.assertTrue(result)
        run = self.ctx.workflow_man.get_run(rid)
        self.assertEqual(run.status, WorkflowStatus.DONE)
        self.assertIn(StageName.TRAIN_HYSTERESIS, run.stage_results)
        self.assertFalse(run.stage_results[StageName.TRAIN_HYSTERESIS].ok)
        self.assertGreater(len(run.warnings), 0)

    @patch("src.workflow.dg_tft.orchestrator.run_stage")
    def test_resume_skips_completed_stages(self, mock_run):
        mock_run.side_effect = self._mock_stage_ok
        ok, rid, _ = self.ctx.workflow_man.create_run(
            "/data/test", output_root=self.tmpdir,
        )
        profile = DatasetProfile(static_ok=True, cv_ok=True, has_hysteresis=False)

        run = self.ctx.workflow_man.get_run(rid)
        run.stage_results = {
            StageName.CHECK_DATA: StageResult(ok=True, stage=StageName.CHECK_DATA, summary="done"),
            StageName.FIT_PHYSICAL: StageResult(ok=True, stage=StageName.FIT_PHYSICAL, summary="done"),
        }
        self.ctx.workflow_man.set_dataset_profile(rid, profile)

        result, summary = run_workflow(self.ctx, rid, entry="check-data", dataset_dir="/data/test", config_path=self.config_path)

        self.assertTrue(result)
        self.assertGreater(mock_run.call_count, 0)
        actions_called = [call[0][1] for call in mock_run.call_args_list]
        self.assertNotIn("check-data", actions_called)
        self.assertNotIn("fit-physical", actions_called)

    def test_run_not_found(self):
        result, summary = run_workflow(self.ctx, 999, entry="check-data", dataset_dir="/data/test")
        self.assertFalse(result)
        self.assertIn("not found", summary)

    def test_run_cancelled(self):
        ok, rid, _ = self.ctx.workflow_man.create_run("/data/test")
        self.ctx.workflow_man.cancel_run(rid)
        result, summary = run_workflow(self.ctx, rid, entry="check-data", dataset_dir="/data/test", config_path=self.config_path)
        self.assertFalse(result)
        self.assertIn("cancelled", summary.lower())

    @patch("src.workflow.dg_tft.orchestrator.run_stage")
    def test_artifact_attached_on_success(self, mock_run):
        mock_run.side_effect = self._mock_stage_ok
        ok, rid, _ = self.ctx.workflow_man.create_run("/data/test", output_root=self.tmpdir)

        output_dir = os.path.join(self.tmpdir, "output")
        for sd_name in ["01_physical_kernel", "03_full_model"]:
            os.makedirs(os.path.join(output_dir, sd_name), exist_ok=True)

        artifacts_to_create = [
            ("dataset_validation.json", ""),
            ("fitted_params.json", "01_physical_kernel"),
            ("v93_safe_residual_state.pt", "03_full_model"),
        ]
        for fname, subdir in artifacts_to_create:
            target_dir = os.path.join(output_dir, subdir)
            os.makedirs(target_dir, exist_ok=True)
            Path(os.path.join(target_dir, fname)).write_text("mock")

        profile = DatasetProfile(static_ok=True, cv_ok=True, has_hysteresis=False)
        self.ctx.workflow_man.set_dataset_profile(rid, profile)

        result, _ = run_workflow(self.ctx, rid, entry="check-data", dataset_dir="/data/test", config_path=self.config_path)

        self.assertTrue(result)
        run = self.ctx.workflow_man.get_run(rid)
        self.assertEqual(run.status, WorkflowStatus.DONE)
        self.assertGreater(len(run.artifacts), 0)


# ────────────────────────────────────────────────────────────────────
# Test: Timeout helpers (Phase 9)
# ────────────────────────────────────────────────────────────────────


class TestTimeoutHelpers(unittest.TestCase):

    def _make_mock_ctx_for_timeout(self, configs=None):
        ctx = MagicMock()
        ctx.agent_configs = configs or {}
        return ctx

    def test_compute_run_timeout_zero_stages(self):
        from src.workflow.dg_tft.orchestrator import _compute_run_timeout
        ctx = self._make_mock_ctx_for_timeout()
        self.assertEqual(_compute_run_timeout(ctx, []), 0)

    def test_compute_run_timeout_single_stage(self):
        from src.workflow.dg_tft.orchestrator import _compute_run_timeout
        ctx = self._make_mock_ctx_for_timeout()
        sd = StageDef(name="test", action="act", required=True, timeout=600)
        result = _compute_run_timeout(ctx, [sd])
        self.assertEqual(result, 1200)

    def test_compute_run_timeout_multiple_stages(self):
        from src.workflow.dg_tft.orchestrator import _compute_run_timeout
        ctx = self._make_mock_ctx_for_timeout()
        stages = [
            StageDef(name="s1", action="a1", required=True, timeout=600),
            StageDef(name="s2", action="a2", required=True, timeout=1800),
            StageDef(name="s3", action="a3", required=True, timeout=7200),
        ]
        result = _compute_run_timeout(ctx, stages)
        self.assertEqual(result, (600 + 1800 + 7200) * 2)

    def test_compute_run_timeout_none_timeout_skipped(self):
        from src.workflow.dg_tft.orchestrator import _compute_run_timeout
        ctx = self._make_mock_ctx_for_timeout()
        stages = [
            StageDef(name="s1", action="a1", required=True, timeout=600),
            StageDef(name="s2", action="a2", required=True, timeout=None),
        ]
        result = _compute_run_timeout(ctx, stages)
        self.assertEqual(result, 1200)

    def test_compute_run_timeout_agent_configs_override(self):
        from src.workflow.dg_tft.orchestrator import _compute_run_timeout
        ctx = self._make_mock_ctx_for_timeout({
            "DG_TFT_TIMEOUT_S_S1": "120",
            "DG_TFT_TIMEOUT_S_S2": "240",
        })
        stages = [
            StageDef(name="s1", action="a1", required=True, timeout=600),
            StageDef(name="s2", action="a2", required=True, timeout=1800),
        ]
        result = _compute_run_timeout(ctx, stages)
        self.assertEqual(result, (120 + 240) * 2)

    def test_resolve_stage_timeout_override(self):
        from src.workflow.dg_tft.orchestrator import _resolve_stage_timeout
        ctx = self._make_mock_ctx_for_timeout({
            "DG_TFT_TIMEOUT_S_CHECK_DATA": "300",
        })
        sd = StageDef(name="check-data", action="act", required=True, timeout=600)
        result = _resolve_stage_timeout(ctx, sd)
        self.assertEqual(result, 300)

    def test_resolve_stage_timeout_fallback(self):
        from src.workflow.dg_tft.orchestrator import _resolve_stage_timeout
        ctx = self._make_mock_ctx_for_timeout({})
        sd = StageDef(name="check-data", action="act", required=True, timeout=600)
        result = _resolve_stage_timeout(ctx, sd)
        self.assertEqual(result, 600)

    def test_elapsed_exceeds_true(self):
        from src.workflow.dg_tft.orchestrator import _elapsed_exceeds
        self.assertTrue(_elapsed_exceeds(0, -1))

    def test_elapsed_exceeds_false(self):
        from src.workflow.dg_tft.orchestrator import _elapsed_exceeds
        import time
        self.assertFalse(_elapsed_exceeds(time.monotonic(), 999999))


class TestModelingAgentHelpers(unittest.TestCase):
    """Tests for Step 8 agent infrastructure helpers."""

    def test_stage_has_modeling_agent(self):
        from src.workflow.dg_tft.orchestrator import _stage_has_modeling_agent
        self.assertTrue(_stage_has_modeling_agent(StageName.FIT_PHYSICAL))
        self.assertTrue(_stage_has_modeling_agent(StageName.TRAIN_STATIC))
        self.assertTrue(_stage_has_modeling_agent(StageName.TRAIN_HYSTERESIS))
        self.assertFalse(_stage_has_modeling_agent(StageName.CHECK_DATA))
        self.assertFalse(_stage_has_modeling_agent(StageName.DATA_PREP))
        self.assertFalse(_stage_has_modeling_agent(StageName.VERIFIER))

    def test_normalize_tool_name_adjust_params(self):
        from src.workflow.dg_tft.orchestrator import _normalize_tool_name
        self.assertEqual(_normalize_tool_name({"section": "physical_fit", "adjustments": {"lr": 0.001}, "reasoning": "test"}), "adjust_params")

    def test_normalize_tool_name_fallback(self):
        from src.workflow.dg_tft.orchestrator import _normalize_tool_name
        self.assertEqual(_normalize_tool_name({"target": "check-data", "reason": "upstream issue"}), "fallback_to_previous")

    def test_normalize_tool_name_accept_quality(self):
        from src.workflow.dg_tft.orchestrator import _normalize_tool_name
        self.assertEqual(_normalize_tool_name({"accept_quality": True, "reasoning": "ok"}), "accept_quality")

    def test_normalize_tool_name_unknown(self):
        from src.workflow.dg_tft.orchestrator import _normalize_tool_name
        self.assertEqual(_normalize_tool_name({"foo": "bar"}), "error")

    def test_normalize_tool_name_non_dict(self):
        from src.workflow.dg_tft.orchestrator import _normalize_tool_name
        self.assertEqual(_normalize_tool_name("just text"), "error")

    def test_process_agent_decision_adjust_params_retry(self):
        """_process_agent_decision returns 'retry' for valid adjust_params."""
        import tempfile
        from src.workflow.dg_tft.orchestrator import _process_agent_decision

        with tempfile.TemporaryDirectory() as tmpdir:
            ini_path = os.path.join(tmpdir, "test.ini")
            with open(ini_path, "w") as f:
                f.write("[physical_fit]\nmax_nfev_scale = 1.0\nmax_iv_points = 50000\n")
            sd = StageDef(name="fit-physical", action="act", required=True)

            # Need to mock ctx.console, ctx.subagent_mute
            ctx = MockContext()
            ctx.subagent_mute = True  # suppress console output

            from src.agent.subagent import SubAgent  # mock out SubAgent so it doesn't try real LLM
            from src.workflow.dg_tft.stage_runner import run_stage

            # Handler output format: adjust_params handler writes INI inline and returns
            # {"status": "ok", "applied": {...}, "warnings": [...], ...}
            agent_result = {
                "decision": "adjust_params",
                "arguments": {
                    "status": "ok",
                    "applied": {"max_nfev_scale": 3.0},
                    "warnings": [],
                    "ini_snapshot": "physical_fit.max_nfev_scale = 3.0",
                    "applied_summary": "max_nfev_scale=3.0",
                },
                "reasoning": "Need more NF evaluations",
            }
            result = _process_agent_decision(
                ctx, 1, sd, ini_path, None, agent_result, 3,
            )
            self.assertEqual(result, "retry")

    def test_process_agent_decision_accept_quality(self):
        from src.workflow.dg_tft.orchestrator import _process_agent_decision

        ctx = MockContext()
        ctx.subagent_mute = True
        sd = StageDef(name="fit-physical", action="act", required=True)
        agent_result = {
            "decision": "accept_quality",
            "arguments": {"reasoning": "IV is close enough"},
            "reasoning": "IV is close enough",
        }
        result = _process_agent_decision(ctx, 1, sd, "", None, agent_result, 0)
        self.assertEqual(result, "accept")

    def test_process_agent_decision_fallback(self):
        from src.workflow.dg_tft.orchestrator import _process_agent_decision

        ctx = MockContext()
        ctx.subagent_mute = True
        sd = StageDef(name="fit-physical", action="act", required=True)
        agent_result = {
            "decision": "fallback_to_previous",
            "arguments": {"target": "check-data", "reason": "data issues"},
            "reasoning": "data issues",
        }
        result = _process_agent_decision(ctx, 1, sd, "", None, agent_result, 0)
        self.assertEqual(result, "fallback")

    def test_process_agent_decision_empty_adjustments(self):
        """Rejected adjustments must not implicitly accept an unresolved failure."""
        from src.workflow.dg_tft.orchestrator import _process_agent_decision

        ctx = MockContext()
        ctx.subagent_mute = True
        sd = StageDef(name="fit-physical", action="act", required=True)
        agent_result = {
            "decision": "adjust_params",
            "arguments": {
                "status": "fail",
                "applied": {},
                "warnings": ["all params rejected"],
            },
            "reasoning": "no changes",
        }
        with self.assertRaises(WorkflowStopped):
            _process_agent_decision(ctx, 1, sd, "", None, agent_result, 0)


# ======================================================================
# Handler integration test — exercises the REAL adjust_params path
# (schema → validate → apply → INI write on disk)
# ======================================================================

class TestAdjustParamsHandlerIntegration(unittest.TestCase):
    """Test the adjust_params handler logic end-to-end with real param tables,
    real INI files on disk, and LLM-flat argument format.
    """

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.ini_path = os.path.join(self.tmpdir, "model_config.ini")
        self._write_ini(self.ini_path)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _write_ini(self, path):
        content = (
            "[paths]\n"
            "data_dir = /tmp/data\n"
            "[full_model]\n"
            "epochs = 700\n"
            "learning_rate = 0.002\n"
            "cv_weight = 8.0\n"
            "early_stop_patience = 5\n"
        )
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)

    def _build_handler(self, stage_name, config_path):
        from src.workflow.dg_tft.orchestrator import _get_stage_param_table
        from src.workflow.dg_tft.param_defs import get_stage_sections
        from src.workflow.dg_tft.quality_guard import (
            validate_agent_adjustments,
            apply_agent_adjustments,
        )
        from src.workflow.dg_tft.stages import StageDef

        param_table = _get_stage_param_table(stage_name)
        sections = get_stage_sections(stage_name)
        sd = StageDef(name=stage_name, action="act", required=True)

        def handler(arguments: dict, _ctx=None) -> dict:
            if not isinstance(arguments, dict):
                return {"status": "fail", "reason": "adjust_params arguments must be a dict"}
            reasoning = arguments.get("reasoning", "")
            raw = {k: v for k, v in arguments.items() if k != "reasoning"}
            if not raw:
                return {"status": "fail", "reason": "no parameter values provided (only 'reasoning' found)"}
            sanitized, warnings = validate_agent_adjustments(raw, param_table, sections, sd.name)
            if not sanitized:
                return {"status": "fail", "reason": f"All {len(warnings)} params rejected: {'; '.join(warnings[:3])}"}
            primary_section = sections[0]
            snapshot = apply_agent_adjustments(config_path, sanitized, primary_section, param_table)
            applied_str = ", ".join(f"{k}={v}" for k, v in sanitized.items())
            return {
                "status": "ok",
                "applied": dict(sanitized),
                "warnings": warnings,
                "ini_snapshot": snapshot,
                "applied_summary": applied_str,
                "reasoning": reasoning,
            }
        return handler

    def test_handler_accepts_flat_llm_args(self):
        """Handler parses flat LLM args ({epochs: 2000, reasoning: ...}) and writes INI."""
        handler = self._build_handler("train-static", self.ini_path)

        result = handler({"epochs": 1500, "learning_rate": 0.001, "reasoning": "more epochs, lower LR"})

        self.assertEqual(result["status"], "ok")
        self.assertIn("epochs", result["applied"])
        self.assertIn("learning_rate", result["applied"])
        self.assertEqual(result["applied"]["epochs"], 1500)
        self.assertEqual(result["applied"]["learning_rate"], 0.001)
        self.assertIn("reasoning", result)
        self.assertEqual(result["reasoning"], "more epochs, lower LR")

        import configparser
        cfg = configparser.ConfigParser()
        cfg.read(self.ini_path, encoding="utf-8")
        self.assertEqual(cfg.get("full_model", "epochs"), "1500")
        self.assertEqual(cfg.get("full_model", "learning_rate"), "0.001")
        self.assertEqual(cfg.get("full_model", "cv_weight"), "8.0")

    def test_handler_only_reasoning_returns_fail(self):
        """Calling adjust_params with ONLY reasoning → fail (nothing to apply)."""
        handler = self._build_handler("train-static", self.ini_path)
        result = handler({"reasoning": "everything looks fine"})
        self.assertEqual(result["status"], "fail")
        self.assertIn("no parameter values", result["reason"])

    def test_handler_non_dict_returns_fail(self):
        handler = self._build_handler("train-static", self.ini_path)
        result = handler("not a dict")
        self.assertEqual(result["status"], "fail")

    def test_handler_full_cv_weight_adjustment(self):
        """cv_weight is a float param — verify type coercion in the full chain."""
        handler = self._build_handler("train-static", self.ini_path)

        result = handler({"cv_weight": 12.5, "reasoning": "CV needs more weight"})

        self.assertEqual(result["status"], "ok")
        self.assertIn("cv_weight", result["applied"])
        import configparser
        cfg = configparser.ConfigParser()
        cfg.read(self.ini_path, encoding="utf-8")
        self.assertIn(cfg.get("full_model", "cv_weight"), ("12.5", "12.5"))

    def test_handler_hysteresis_stage_params(self):
        """Hysteresis stage has different param set — verify schema isolation."""
        import configparser

        # Need to add [hysteresis] section to the INI
        self._write_ini(self.ini_path)
        with open(self.ini_path, "a", encoding="utf-8") as f:
            f.write("[hysteresis]\nepochs = 1800\nhidden_capture = 16\namp_parameterization = conservative\n")

        handler = self._build_handler("train-hysteresis", self.ini_path)
        result = handler({"epochs": 2000, "hidden_capture": 32, "reasoning": "stronger hysteresis model"})

        self.assertEqual(result["status"], "ok")
        cfg = configparser.ConfigParser()
        cfg.read(self.ini_path, encoding="utf-8")
        self.assertEqual(cfg.get("hysteresis", "epochs"), "2000")
        self.assertEqual(cfg.get("hysteresis", "hidden_capture"), "32")
        # full_model should be untouched
        self.assertEqual(cfg.get("full_model", "epochs"), "700")


if __name__ == "__main__":
    unittest.main()
