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
        for val in ("initial", "quality_fail", "agent_adjust", "fallback_sent",
                     "fallback_received", "upstream_changed"):
            self.assertIn(val, AttemptTrigger._value2member_map_)

    def test_trigger_is_string_enum(self):
        t = AttemptTrigger.INITIAL
        self.assertEqual(t.value, "initial")
        self.assertIsInstance(t.value, str)


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


if __name__ == "__main__":
    unittest.main()
