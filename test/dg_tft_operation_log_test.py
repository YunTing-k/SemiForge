# -*- coding: utf-8 -*-
"""
Unit tests for DG-TFT operation log module (operation_log.py).
Run: python -m unittest test.dg_tft_operation_log_test -v
"""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.workflow.dg_tft.operation_log import (
    AttemptTrigger,
    StageAttemptLog,
    OperationLog,
)


class TestAttemptTrigger(unittest.TestCase):
    """3.6-T1: AttemptTrigger enum values."""

    def test_all_triggers_defined(self):
        expected = {"initial", "quality_fail", "agent_adjust", "fallback_sent",
                     "fallback_received", "upstream_changed", "verifier_fail",
                     "prep_retry", "agent_accept", "execution_failed", "interrupted", "feedback_resolved"}
        self.assertEqual(set(AttemptTrigger._value2member_map_), expected)

    def test_trigger_is_string_enum(self):
        t = AttemptTrigger.INITIAL
        self.assertEqual(t.value, "initial")
        self.assertIsInstance(t.value, str)

    def test_all_triggers_present(self):
        self.assertEqual(len(AttemptTrigger), 12)

    def test_trigger_from_string(self):
        for val in ("initial", "quality_fail", "agent_adjust", "fallback_sent",
                     "fallback_received", "upstream_changed", "verifier_fail",
                     "prep_retry", "agent_accept"):
            t = AttemptTrigger(val)
            self.assertEqual(t.value, val)


class TestStageAttemptLog(unittest.TestCase):
    """3.6-T2: StageAttemptLog serialization round-trip."""

    def test_default_construction(self):
        entry = StageAttemptLog(n=1)
        self.assertEqual(entry.n, 1)
        self.assertEqual(entry.trigger, AttemptTrigger.INITIAL)
        self.assertEqual(entry.upstream, "")
        self.assertIsNone(entry.handoff)
        self.assertIsNone(entry.agent_decision)
        self.assertEqual(entry.ini_snapshot, "")
        self.assertEqual(entry.duration_s, 0.0)
        self.assertEqual(entry.metrics, {})
        self.assertIsNone(entry.adjustments)
        self.assertTrue(entry.ok)
        self.assertEqual(entry.error, "")
        self.assertIsNone(entry.quality_pass)

    def test_full_construction(self):
        entry = StageAttemptLog(
            n=2,
            trigger=AttemptTrigger.QUALITY_FAIL,
            upstream="fit-physical.1",
            handoff={"status": "ok"},
            agent_decision={"tool": "adjust_params", "adjustments": {"lr": 0.001}},
            ini_snapshot="/path/to/ini",
            duration_s=45.2,
            metrics={"iv_rms": 0.0021},
            adjustments={"physical_fit.max_nfev_scale": "2.0"},
            ok=True,
            error="",
            quality_pass=True,
        )
        self.assertEqual(entry.n, 2)
        self.assertEqual(entry.trigger, AttemptTrigger.QUALITY_FAIL)
        self.assertEqual(entry.upstream, "fit-physical.1")
        self.assertEqual(entry.handoff["status"], "ok")
        self.assertEqual(entry.agent_decision["tool"], "adjust_params")
        self.assertEqual(entry.duration_s, 45.2)
        self.assertEqual(entry.metrics["iv_rms"], 0.0021)
        self.assertTrue(entry.quality_pass)

    def test_to_dict_basic(self):
        entry = StageAttemptLog(n=1, metrics={"a": 1.0, "b": 2.0})
        d = entry.to_dict()
        self.assertEqual(d["n"], 1)
        self.assertEqual(d["trigger"], "initial")
        self.assertEqual(d["metrics"], {"a": 1.0, "b": 2.0})
        self.assertTrue(d["ok"])

    def test_to_dict_with_agent_decision(self):
        entry = StageAttemptLog(n=3, trigger=AttemptTrigger.AGENT_ADJUST,
                                agent_decision={"tool": "fallback", "reason": "bad vto"})
        d = entry.to_dict()
        self.assertEqual(d["trigger"], "agent_adjust")
        self.assertEqual(d["agent_decision"]["tool"], "fallback")

    def test_from_dict_roundtrip(self):
        original = StageAttemptLog(
            n=2,
            trigger=AttemptTrigger.QUALITY_FAIL,
            upstream="check-data.1",
            handoff={"status": "warn"},
            agent_decision=None,
            ini_snapshot="/tmp/ini",
            duration_s=33.0,
            metrics={"cv_rms": 0.015},
            adjustments={"full_model.learning_rate": "0.0005"},
            ok=True,
            error="",
            quality_pass=False,
        )
        d = original.to_dict()
        restored = StageAttemptLog.from_dict(d)
        self.assertEqual(restored.n, original.n)
        self.assertEqual(restored.trigger, original.trigger)
        self.assertEqual(restored.upstream, original.upstream)
        self.assertEqual(restored.handoff, original.handoff)
        self.assertIsNone(restored.agent_decision)
        self.assertEqual(restored.ini_snapshot, original.ini_snapshot)
        self.assertEqual(restored.duration_s, original.duration_s)
        self.assertEqual(restored.metrics, original.metrics)
        self.assertEqual(restored.adjustments, original.adjustments)
        self.assertEqual(restored.ok, original.ok)
        self.assertEqual(restored.error, original.error)
        self.assertEqual(restored.quality_pass, original.quality_pass)

    def test_from_dict_minimal(self):
        d = {"n": 1}
        entry = StageAttemptLog.from_dict(d)
        self.assertEqual(entry.n, 1)
        self.assertEqual(entry.trigger, AttemptTrigger.INITIAL)

    def test_from_dict_handles_missing_fields(self):
        d = {"n": 5, "trigger": "fallback_received"}
        entry = StageAttemptLog.from_dict(d)
        self.assertEqual(entry.n, 5)
        self.assertEqual(entry.trigger, AttemptTrigger.FALLBACK_RECEIVED)
        self.assertEqual(entry.metrics, {})


class TestOperationLog(unittest.TestCase):
    """3.6-T3: OperationLog lifecycle (initialize, append, load, query)."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.log_path = Path(self.tmpdir) / "operation_log.json"

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_initialize_creates_file(self):
        op = OperationLog(self.tmpdir)
        op.initialize(run_id=42, preferences="hysteresis first")
        self.assertTrue(self.log_path.exists())
        data = json.loads(self.log_path.read_text(encoding="utf-8"))
        self.assertEqual(data["run_id"], 42)
        self.assertEqual(data["preferences"], "hysteresis first")
        self.assertEqual(data["final_status"], "running")
        self.assertEqual(data["stages"], {})

    def test_initialize_with_empty_preferences(self):
        op = OperationLog(self.tmpdir)
        op.initialize(run_id=1)
        data = json.loads(self.log_path.read_text(encoding="utf-8"))
        self.assertEqual(data["preferences"], "")

    def test_append_attempt_single_stage(self):
        op = OperationLog(self.tmpdir)
        op.initialize(run_id=42)
        entry = StageAttemptLog(n=1, metrics={"cv_rms": 0.015})
        op.append_attempt("train-static", entry)

        data = json.loads(self.log_path.read_text(encoding="utf-8"))
        self.assertIn("train-static", data["stages"])
        self.assertEqual(len(data["stages"]["train-static"]), 1)

    def test_append_attempt_multiple(self):
        op = OperationLog(self.tmpdir)
        op.initialize(run_id=42)
        op.append_attempt("fit-physical", StageAttemptLog(n=1, metrics={"iv_rms": 0.002}))
        op.append_attempt("fit-physical", StageAttemptLog(n=2, trigger=AttemptTrigger.QUALITY_FAIL,
                                                           metrics={"iv_rms": 0.001}))

        data = json.loads(self.log_path.read_text(encoding="utf-8"))
        self.assertEqual(len(data["stages"]["fit-physical"]), 2)
        self.assertEqual(data["stages"]["fit-physical"][1]["n"], 2)

    def test_append_to_multiple_stages(self):
        op = OperationLog(self.tmpdir)
        op.initialize(run_id=42)
        op.append_attempt("check-data", StageAttemptLog(n=1))
        op.append_attempt("fit-physical", StageAttemptLog(n=1))
        op.append_attempt("train-static", StageAttemptLog(n=1))

        data = json.loads(self.log_path.read_text(encoding="utf-8"))
        self.assertEqual(len(data["stages"]), 3)

    def test_get_stage_attempts_returns_empty_for_unknown(self):
        op = OperationLog(self.tmpdir)
        op.initialize(run_id=42)
        self.assertEqual(op.get_stage_attempts("unknown"), [])

    def test_get_last_attempt_n_initial(self):
        op = OperationLog(self.tmpdir)
        op.initialize(run_id=42)
        self.assertEqual(op.get_last_attempt_n("any-stage"), 0)

    def test_get_last_attempt_n_after_append(self):
        op = OperationLog(self.tmpdir)
        op.initialize(run_id=42)
        op.append_attempt("check-data", StageAttemptLog(n=1))
        op.append_attempt("check-data", StageAttemptLog(n=2))
        self.assertEqual(op.get_last_attempt_n("check-data"), 2)

    def test_get_last_attempt_returns_none_for_unknown(self):
        op = OperationLog(self.tmpdir)
        op.initialize(run_id=42)
        self.assertIsNone(op.get_last_attempt("unknown"))

    def test_get_last_attempt_returns_latest(self):
        op = OperationLog(self.tmpdir)
        op.initialize(run_id=42)
        op.append_attempt("fit-physical", StageAttemptLog(n=1, metrics={"iv_rms": 0.003}))
        op.append_attempt("fit-physical", StageAttemptLog(n=2, metrics={"iv_rms": 0.001}))
        last = op.get_last_attempt("fit-physical")
        self.assertIsNotNone(last)
        self.assertEqual(last["n"], 2)
        self.assertEqual(last["metrics"]["iv_rms"], 0.001)

    def test_set_final_status(self):
        op = OperationLog(self.tmpdir)
        op.initialize(run_id=42)
        op.set_final_status("done")
        data = json.loads(self.log_path.read_text(encoding="utf-8"))
        self.assertEqual(data["final_status"], "done")

    def test_set_final_status_failed(self):
        op = OperationLog(self.tmpdir)
        op.initialize(run_id=42)
        op.set_final_status("failed")
        data = json.loads(self.log_path.read_text(encoding="utf-8"))
        self.assertEqual(data["final_status"], "failed")

    def test_data_property_exposes_all(self):
        op = OperationLog(self.tmpdir)
        op.initialize(run_id=10, preferences="cv first")
        self.assertEqual(op.data["run_id"], 10)

    def test_load_existing_log(self):
        op1 = OperationLog(self.tmpdir)
        op1.initialize(run_id=77, preferences="iv focus")
        op1.append_attempt("check-data", StageAttemptLog(n=1))
        op1.set_final_status("done")

        op2 = OperationLog(self.tmpdir)
        op2.load()
        self.assertEqual(op2.data["run_id"], 77)
        self.assertEqual(op2.data["final_status"], "done")
        self.assertEqual(len(op2.data["stages"]["check-data"]), 1)

    def test_load_corrupted_file_handles_gracefully(self):
        self.tmpdir_path = Path(self.tmpdir)
        self.tmpdir_path.mkdir(parents=True, exist_ok=True)
        (self.tmpdir_path / "operation_log.json").write_text("not valid json {{{", encoding="utf-8")
        op = OperationLog(self.tmpdir)
        op.load()
        self.assertEqual(op.data, {})

    def test_directory_created_if_missing(self):
        nested = Path(self.tmpdir) / "a" / "b" / "c"
        op = OperationLog(nested)
        op.initialize(run_id=1)
        self.assertTrue((nested / "operation_log.json").exists())

    def test_append_preserves_attempt_order(self):
        op = OperationLog(self.tmpdir)
        op.initialize(run_id=1)
        for i in range(5):
            op.append_attempt("test-stage", StageAttemptLog(n=i + 1))
        attempts = op.get_stage_attempts("test-stage")
        self.assertEqual(len(attempts), 5)
        self.assertEqual(attempts[0]["n"], 1)
        self.assertEqual(attempts[4]["n"], 5)


class TestStageAttemptLogAuditGaps(unittest.TestCase):
    """Tests for audit-discovered gaps in StageAttemptLog semantics."""

    def test_upstream_defaults_to_empty_string(self):
        entry = StageAttemptLog(n=1)
        self.assertEqual(entry.upstream, "")
        d = entry.to_dict()
        self.assertEqual(d["upstream"], "")

    def test_upstream_accepts_and_roundtrips_value(self):
        entry = StageAttemptLog(n=1, upstream="fit-physical.3")
        self.assertEqual(entry.upstream, "fit-physical.3")
        d = entry.to_dict()
        restored = StageAttemptLog.from_dict(d)
        self.assertEqual(restored.upstream, "fit-physical.3")

    def test_quality_pass_none_semantic(self):
        entry = StageAttemptLog(n=1)
        self.assertIsNone(entry.quality_pass)
        d = entry.to_dict()
        self.assertIsNone(d["quality_pass"])
        restored = StageAttemptLog.from_dict(d)
        self.assertIsNone(restored.quality_pass)

    def test_quality_pass_true_semantic(self):
        entry = StageAttemptLog(n=1, quality_pass=True)
        self.assertTrue(entry.quality_pass)
        d = entry.to_dict()
        self.assertTrue(d["quality_pass"])

    def test_quality_pass_false_semantic(self):
        entry = StageAttemptLog(n=1, quality_pass=False)
        self.assertFalse(entry.quality_pass)
        d = entry.to_dict()
        self.assertFalse(d["quality_pass"])

    def test_ok_true_quality_pass_none_distinction(self):
        ok_true_quality_none = StageAttemptLog(n=1, ok=True, quality_pass=None)
        ok_true_quality_true = StageAttemptLog(n=1, ok=True, quality_pass=True)
        self.assertEqual(ok_true_quality_none.to_dict()["ok"], True)
        self.assertIsNone(ok_true_quality_none.to_dict()["quality_pass"])
        self.assertEqual(ok_true_quality_true.to_dict()["quality_pass"], True)

    def test_agent_reasoning_field_serialized(self):
        entry = StageAttemptLog(n=2, trigger=AttemptTrigger.AGENT_ADJUST,
                                 agent_reasoning="VTO out of range, suspected data issue")
        d = entry.to_dict()
        self.assertEqual(d["agent_reasoning"], "VTO out of range, suspected data issue")

    def test_agent_reasoning_default_empty(self):
        entry = StageAttemptLog(n=1)
        self.assertEqual(entry.agent_reasoning, "")
        self.assertEqual(entry.to_dict()["agent_reasoning"], "")

    def test_ini_snapshot_serialized(self):
        entry = StageAttemptLog(n=2, ini_snapshot="[physical_fit]\nmax_nfev_scale=2.0\n")
        d = entry.to_dict()
        self.assertIn("max_nfev_scale=2.0", d["ini_snapshot"])

    def test_error_field_preserved_in_failed_entry(self):
        entry = StageAttemptLog(n=1, ok=False,
                                 error="FileNotFoundError: IDVBG_train.xlsx")
        d = entry.to_dict()
        self.assertEqual(d["error"], "FileNotFoundError: IDVBG_train.xlsx")
        self.assertFalse(d["ok"])

    def test_duration_s_preserved(self):
        entry = StageAttemptLog(n=1, duration_s=67.5)
        d = entry.to_dict()
        self.assertEqual(d["duration_s"], 67.5)


class TestOperationLogStateMachine(unittest.TestCase):
    """State-machine tests for typical pipeline attempt sequences.
    These tests demonstrate the data model's correct-behavior specification.
    Where the orchestrator currently falls short of this spec, the docstring
    notes the gap explicitly."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.op = OperationLog(self.tmpdir)
        self.op.initialize(run_id=1)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    # ── correct-behavior specification tests ────────────────────────

    def test_every_stage_execution_logs_an_entry(self):
        """Spec: every stage execution (init, agent decision, retry)
        produces a log entry with that execution's actual ok/metrics.
        Gap: orchestrator._log_initial_attempt only fires on ok=True and
        attempt_n==1, so retry results after agent_adjust are never logged."""
        self.op.append_attempt("fit-physical", StageAttemptLog(
            n=1, trigger=AttemptTrigger.INITIAL, ok=True,
            metrics={"iv_rms_log10_error_dec": 0.40}))
        self.op.append_attempt("fit-physical", StageAttemptLog(
            n=2, trigger=AttemptTrigger.AGENT_ADJUST, ok=False,
            agent_decision={"tool": "adjust_params"},
            agent_reasoning="need more nfev"))
        # The actual retry execution — should be logged with its own ok=True
        self.op.append_attempt("fit-physical", StageAttemptLog(
            n=2, trigger=AttemptTrigger.INITIAL, ok=True,
            metrics={"iv_rms_log10_error_dec": 0.0015}))

        attempts = self.op.get_stage_attempts("fit-physical")
        self.assertEqual(len(attempts), 3)
        self.assertTrue(attempts[2]["ok"])
        self.assertEqual(attempts[2]["metrics"]["iv_rms_log10_error_dec"], 0.0015)

    def test_hard_failed_stage_appears_in_log(self):
        """Spec: every stage execution logs its outcome, including hard failures.
        Gap: orchestrator._log_initial_attempt only called when ok=True."""
        self.op.append_attempt("check-data", StageAttemptLog(
            n=1, trigger=AttemptTrigger.INITIAL, ok=False,
            error="ValueError: missing column VD_V"))
        self.assertEqual(len(self.op.get_stage_attempts("check-data")), 1)
        entry = self.op.get_last_attempt("check-data")
        self.assertFalse(entry["ok"])
        self.assertIn("VD_V", entry["error"])

    def test_cascade_victim_stage_logs_failure(self):
        """Spec: when downstream stages fail because upstream failed,
        their failure is still logged. Gap: orchestrator never appends
        for cascade victims that hard-fail on their first attempt."""
        self.op.append_attempt("fit-physical", StageAttemptLog(
            n=1, trigger=AttemptTrigger.INITIAL, ok=False,
            error="ValueError: missing column VD_V"))
        self.op.append_attempt("train-static", StageAttemptLog(
            n=1, trigger=AttemptTrigger.INITIAL, ok=False,
            error="FileNotFoundError: fitted_params.json"))
        self.assertEqual(len(self.op.get_stage_attempts("train-static")), 1)

    def test_last_attempt_ok_reflects_actual_execution(self):
        """Spec: the last log entry reflects the most recent execution's
        actual outcome (not an intermediate agent decision).
        Gap: Agent E's prompt reads attempts[-1] which may be an
        AGENT_ADJUST entry with ok=False even after a successful retry."""
        self.op.append_attempt("train-static", StageAttemptLog(
            n=1, trigger=AttemptTrigger.INITIAL, ok=True,
            metrics={"residual_iv_rms": 0.30, "residual_cv_rms": 0.07}))
        self.op.append_attempt("train-static", StageAttemptLog(
            n=2, trigger=AttemptTrigger.AGENT_ADJUST, ok=False,
            agent_decision={"tool": "adjust_params"}))
        self.op.append_attempt("train-static", StageAttemptLog(
            n=2, trigger=AttemptTrigger.INITIAL, ok=True,
            metrics={"residual_iv_rms": 0.30, "residual_cv_rms": 0.05}))

        last = self.op.get_last_attempt("train-static")
        self.assertTrue(last["ok"])
        self.assertEqual(last["metrics"]["residual_cv_rms"], 0.05)

    # ── behaving correctly (no expected failure) ────────────────────

    def test_full_pipeline_success_sequence(self):
        stages = ["data-prep", "verifier", "check-data", "fit-physical",
                   "train-static", "train-hysteresis"]
        for s in stages:
            self.op.append_attempt(s, StageAttemptLog(n=1, ok=True))
        for s in stages:
            attempts = self.op.get_stage_attempts(s)
            self.assertEqual(len(attempts), 1)
            self.assertTrue(attempts[0]["ok"])

    def test_agent_decision_preserves_reasoning_and_decision(self):
        self.op.append_attempt("fit-physical", StageAttemptLog(
            n=2, trigger=AttemptTrigger.AGENT_ADJUST, ok=False,
            agent_decision={"tool": "adjust_params", "adjustments": {"max_nfev_scale": 2.0}},
            agent_reasoning="VTO at bound -5.0, suspected data issue",
            adjustments={"physical_fit.max_nfev_scale": "2.0"}))
        entry = self.op.get_last_attempt("fit-physical")
        self.assertEqual(entry["agent_reasoning"], "VTO at bound -5.0, suspected data issue")
        self.assertEqual(entry["agent_decision"]["tool"], "adjust_params")
        self.assertEqual(entry["adjustments"]["physical_fit.max_nfev_scale"], "2.0")

    def test_fallback_sequence_preserves_target_and_reason(self):
        self.op.append_attempt("train-hysteresis", StageAttemptLog(
            n=1, trigger=AttemptTrigger.INITIAL, ok=False,
            error="RuntimeError: Top-2 SRH hysteresis acceptance failed"))
        self.op.append_attempt("train-hysteresis", StageAttemptLog(
            n=2, trigger=AttemptTrigger.FALLBACK_SENT,
            agent_decision={"tool": "fallback_to_previous", "target": "train-static",
                            "reason": "Static model checkpoint inadequate"}))
        self.op.append_attempt("train-static", StageAttemptLog(
            n=1, trigger=AttemptTrigger.INITIAL, ok=True))
        self.op.append_attempt("train-static", StageAttemptLog(
            n=2, trigger=AttemptTrigger.FALLBACK_RECEIVED, ok=True,
            metrics={"residual_iv_rms": 0.20, "residual_cv_rms": 0.04}))

        self.assertEqual(len(self.op.get_stage_attempts("train-hysteresis")), 2)
        self.assertEqual(len(self.op.get_stage_attempts("train-static")), 2)
        self.assertEqual(
            self.op.get_last_attempt("train-hysteresis")["trigger"], "fallback_sent")
        self.assertEqual(
            self.op.get_last_attempt("train-static")["trigger"], "fallback_received")

    def test_quality_pass_tri_state_behavior(self):
        self.op.append_attempt("train-static", StageAttemptLog(
            n=1, trigger=AttemptTrigger.INITIAL, ok=True, quality_pass=None))
        self.op.append_attempt("train-static", StageAttemptLog(
            n=2, trigger=AttemptTrigger.AGENT_ADJUST, ok=False, quality_pass=True))
        self.op.append_attempt("train-static", StageAttemptLog(
            n=3, trigger=AttemptTrigger.AGENT_ACCEPT, ok=False, quality_pass=False))

        entries = self.op.get_stage_attempts("train-static")
        self.assertIsNone(entries[0]["quality_pass"])
        self.assertTrue(entries[1]["quality_pass"])
        self.assertFalse(entries[2]["quality_pass"])


if __name__ == "__main__":
    unittest.main()
