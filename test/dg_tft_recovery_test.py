"""Regressions from the joint-data trial: recovery, evidenced feedback and exits.

Use actual state/log persistence with deterministic engine/agent responses.
No LLM requests or expensive device training are needed to test orchestration.
"""
import asyncio
import json
import sys
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.workflow.dg_tft.control import WorkflowStopped
from src.workflow.dg_tft.orchestrator import run_workflow
from src.workflow.dg_tft.schema import DatasetProfile, StageResult, WorkflowStatus
from src.workflow.dg_tft.stage_runner import _run_stage_streaming
from src.workflow.dg_tft.operation_log import OperationLog
from src.workflow.dg_tft.orchestrator import _archive_stage_outputs
from src.workflow.dg_tft.stages import get_stage_by_name
from test.dg_tft_agent_flow_test import _make_mock_ctx, _make_mock_subagent, _write_minimal_config


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.output = self.root / "output"
        self.output.mkdir()
        self.config = _write_minimal_config(str(self.root))
        self.ctx = _make_mock_ctx()
        _, self.rid, _ = self.ctx.workflow_man.create_run("/data", output_root=str(self.output))
        self.ctx.workflow_man.set_dataset_profile(self.rid, DatasetProfile(static_ok=True, has_hysteresis=False))
        self.calls = []
        self.patchers = [
            patch("src.workflow.dg_tft.orchestrator._spawn_agent_e"),
            patch("src.workflow.dg_tft.orchestrator._assess_stage_quality", side_effect=self.quality),
            patch("src.workflow.dg_tft.orchestrator.run_stage", side_effect=self.engine),
        ]
        for patcher in self.patchers:
            patcher.start()
            self.addCleanup(patcher.stop)

    def engine(self, ctx, action, *args, **kwargs):
        self.calls.append(action)
        if action == "fit-physical":
            dest = self.output / "01_physical_kernel"
            dest.mkdir(exist_ok=True)
            (dest / "fitted_params.json").write_text(json.dumps({
                "metrics_summary": {"iv_rms_log10_error_dec": .8 if self.calls.count(action) == 1 else .1}
            }))
        return StageResult(ok=True, stage=action, summary="executed")

    def quality(self, ctx, stage, *args):
        failed = stage == "fit-physical" and self.calls.count(stage) == 1
        return not failed, ["inconsistent source data"] if failed else []

    def execute(self):
        return run_workflow(self.ctx, self.rid, entry="check-data", config_path=self.config, dataset_dir="/data")

    def log(self):
        return json.loads((self.output / "operation_log.json").read_text(encoding="utf-8"))

    @staticmethod
    def fallback(*args, **kwargs):
        return {"decision": "fallback_to_previous", "arguments": {
            "_tool": "fallback_to_previous", "target": "check-data",
            "reason": "Cross-source bias collision", "evidence": "same bias, different currents",
        }}

    def test_structural_pass_requires_review_and_reruns_physical(self):
        def review(ctx, rid, sd, result, output, log, fallback_ctx):
            self.assertTrue(result.ok)  # The prior bug skipped diagnosis here.
            self.assertEqual(fallback_ctx["from_stage"], "fit-physical")
            self.assertNotIn("fit-physical", ctx.workflow_man.get_run(rid).stage_results)
            return {"fallback": "", "resolution": {
                "outcome": "not_applicable", "reason": "Source labels refer to distinct bias sweeps",
                "evidence": "Matched-bias inspection disproves the initial collision",
            }}
        with patch("src.workflow.dg_tft.orchestrator._spawn_modeling_agent", side_effect=self.fallback), \
             patch("src.workflow.dg_tft.orchestrator._spawn_agent_a", side_effect=review):
            ok, summary = self.execute()
        self.assertTrue(ok, summary)
        self.assertEqual(self.calls, ["check-data", "fit-physical", "check-data", "fit-physical", "train-static"])
        log = self.log()
        self.assertEqual(log["final_status"], "done")
        self.assertEqual(len([a for a in log["stages"]["fit-physical"] if a["summary"] == "executed"]), 2)
        invalidation = next(e for e in log["events"] if e["kind"] == "invalidation")
        self.assertIn("train-static", invalidation["stages"])
        self.assertTrue(invalidation["results"]["fit-physical"]["ok"])
        archived = list((self.output / "attempt_archive").glob("*/fitted_params.json"))
        self.assertEqual(len(archived), 1)
        self.assertEqual(json.loads(archived[0].read_text())["metrics_summary"]["iv_rms_log10_error_dec"], .8)

    def test_format_pass_without_feedback_resolution_cannot_continue(self):
        with patch("src.workflow.dg_tft.orchestrator._spawn_modeling_agent", side_effect=self.fallback), \
             patch("src.workflow.dg_tft.orchestrator._spawn_agent_a", return_value={"fallback": ""}):
            ok, _ = self.execute()
        self.assertFalse(ok)
        self.assertNotIn("train-static", self.calls)
        self.assertEqual(self.ctx.workflow_man.get_run(self.rid).status, WorkflowStatus.FAILED)
        self.assertIn("resolution", self.log()["events"][-1]["reason"])

    def test_missing_measurement_facts_stop_with_actionable_report(self):
        from src.constants import FINAL_INTERPRETATION_FILENAME
        (self.output / FINAL_INTERPRETATION_FILENAME).write_text("Previous successful model")
        agent = _make_mock_subagent(json.dumps({
            "_tool": "stop_workflow", "outcome": "needs_input",
            "reason": "Device identities cannot be confirmed",
            "evidence": "Matched bias currents differ by seven decades",
            "required_information": "Confirm device identifiers for both measurement groups",
        }))
        with patch("src.workflow.dg_tft.orchestrator._spawn_modeling_agent", side_effect=self.fallback), \
             patch("src.agent.subagent.SubAgent", return_value=agent):
            ok, _ = self.execute()
        self.assertFalse(ok)
        self.assertNotIn("train-static", self.calls)
        self.assertEqual(self.ctx.workflow_man.get_run(self.rid).status, WorkflowStatus.NEEDS_INPUT)
        self.assertEqual(self.log()["final_status"], "needs_input")
        report = json.loads((self.output / "RUN_SUMMARY.json").read_text(encoding="utf-8"))
        self.assertEqual(report["status"], "NEEDS_INPUT")
        self.assertIn("device identifiers", report["termination"]["required_information"])
        from src.constants import FINAL_INTERPRETATION_FILENAME
        self.assertIn("NEEDS_INPUT", (self.output / FINAL_INTERPRETATION_FILENAME).read_text(encoding="utf-8"))

    def test_hard_failure_is_logged_before_diagnostic_stop(self):
        def fail(ctx, action, *args, **kwargs):
            self.calls.append(action)
            return StageResult(ok=False, stage=action, error="precision gate failed",
                               summary=["last training epoch"], outputs={"stderr_log": "saved.log"})
        with patch("src.workflow.dg_tft.orchestrator.run_stage", side_effect=fail), \
             patch("src.workflow.dg_tft.orchestrator._spawn_agent_a",
                   side_effect=WorkflowStopped("Unrecoverable input", evidence="confirmed mismatch")):
            ok, _ = self.execute()
        self.assertFalse(ok)
        attempt = self.log()["stages"]["check-data"][0]
        self.assertFalse(attempt["ok"])
        self.assertEqual(attempt["trigger"], "execution_failed")
        self.assertEqual(attempt["summary"], ["last training epoch"])
        self.assertEqual(attempt["outputs"]["stderr_log"], "saved.log")

    def test_cancellation_persists_terminal_status_and_history(self):
        for exception in (KeyboardInterrupt, asyncio.CancelledError):
            with self.subTest(exception=exception.__name__):
                self.ctx.workflow_man.update_stage(self.rid, "", WorkflowStatus.PENDING)
                with patch("src.workflow.dg_tft.orchestrator.run_stage", side_effect=exception):
                    with self.assertRaises(exception):
                        self.execute()
                self.assertEqual(self.ctx.workflow_man.get_run(self.rid).status, WorkflowStatus.CANCELLED)
                self.assertEqual(self.log()["final_status"], "cancelled")
                self.assertEqual(self.log()["stages"]["check-data"][-1]["trigger"], "interrupted")
                report = json.loads((self.output / "RUN_SUMMARY.json").read_text(encoding="utf-8"))
                self.assertEqual(report["status"], "CANCELLED")

    def test_unresolved_quality_is_not_reused_on_resume(self):
        from src.workflow.dg_tft.orchestrator import _stage_done
        with patch("src.workflow.dg_tft.orchestrator._spawn_modeling_agent",
                   return_value={"decision": "error", "reasoning": "No valid decision"}):
            ok, _ = self.execute()
        self.assertFalse(ok)
        run = self.ctx.workflow_man.get_run(self.rid)
        self.assertTrue(run.stage_results["fit-physical"].ok)
        self.assertFalse(_stage_done(run, "fit-physical"))
        ok, summary = self.execute()
        self.assertTrue(ok, summary)
        self.assertEqual(self.calls.count("fit-physical"), 2)
        self.assertTrue(_stage_done(run, "fit-physical"))

    def test_preparer_can_stop_entire_run_without_retry(self):
        with patch("src.workflow.dg_tft.data_preparer.run_preparer_stage",
                   side_effect=WorkflowStopped("Missing gate reference", evidence="CSV has only VG",
                                               needs_input=True, required_information="Identify VG terminal")) as preparer:
            ok, _ = run_workflow(self.ctx, self.rid, entry="full", config_path=self.config)
        self.assertFalse(ok)
        self.assertEqual(preparer.call_count, 1)
        self.assertEqual(self.ctx.workflow_man.get_run(self.rid).status, WorkflowStatus.NEEDS_INPUT)

    def test_verifier_fallback_uses_new_prepared_data_and_preserves_feedback(self):
        new_data = self.output / "data_prep" / "attempt_2"
        new_data.mkdir(parents=True)
        received = []

        def agent_stage(ctx, rid, sd, output, dataset, raw, preferences, log, feedback, retries, limit):
            received.append((sd.name, feedback))
            count = retries.get(sd.name + "_retries", 0) + 1
            retries[sd.name + "_retries"] = count
            # First verifier rejects the data; preparer fixes it; verifier passes.
            return str(new_data), sd.name != "verifier" or count > 1

        def build(ctx, profile, dataset, **kwargs):
            self.assertEqual(dataset, str(new_data))
            self.assertTrue(profile.static_ok)
            return {"config_path": self.config}

        with patch("src.workflow.dg_tft.orchestrator._spawn_modeling_agent", side_effect=self.fallback), \
             patch("src.workflow.dg_tft.orchestrator._spawn_agent_a", return_value={
                 "fallback": "verifier", "reason": "Source mapping requires repair",
                 "evidence": "Matched bias discrepancy", "suggestion": "Check original mapping"}), \
             patch("src.workflow.dg_tft.orchestrator._run_agent_stage", side_effect=agent_stage), \
             patch("src.workflow.dg_tft.dataset_inspector.inspect_dataset", return_value=(DatasetProfile(static_ok=True), {})), \
             patch("src.workflow.dg_tft.config_builder.build_runtime_config", side_effect=build) as builder:
            ok, summary = self.execute()
        self.assertTrue(ok, summary)
        self.assertEqual([name for name, _ in received], ["verifier", "data-prep", "verifier"])
        self.assertTrue(all(feedback["reason"] == "Source mapping requires repair" for _, feedback in received))
        self.assertEqual(builder.call_count, 1)
        self.assertEqual(self.calls.count("fit-physical"), 2)

    def test_same_stage_retry_preserves_warm_start_and_snapshots_history(self):
        directory = self.output / "03_full_model"
        directory.mkdir()
        checkpoint = directory / "checkpoint.pt"
        checkpoint.write_bytes(b"existing checkpoint")
        log = OperationLog(self.output)
        log.initialize(self.rid)
        _archive_stage_outputs(get_stage_by_name("train-static"), str(self.output), log)
        self.assertEqual(checkpoint.read_bytes(), b"existing checkpoint")
        copies = list((self.output / "attempt_archive").glob("*/checkpoint.pt"))
        self.assertEqual(len(copies), 1)
        self.assertEqual(copies[0].read_bytes(), checkpoint.read_bytes())

    def test_stop_tool_is_available_to_all_execution_agents(self):
        from src.tool.subagent_tools import get_tools, resolve_agent_type_config
        from src.constants import PREPARER_AGENT_LABEL, VERIFIER_AGENT_LABEL, DIAGNOSTIC_AGENT_LABEL
        for stage in ("fit-physical", "train-static", "train-hysteresis"):
            schemas, _ = get_tools(stage)
            self.assertIn("stop_workflow", [s["function"]["name"] for s in schemas])
        for label in (PREPARER_AGENT_LABEL, VERIFIER_AGENT_LABEL, DIAGNOSTIC_AGENT_LABEL):
            config = resolve_agent_type_config(label)
            self.assertIn("stop_workflow", config["stop_schemas"])

    def test_preparer_stop_passes_through_real_wrapper(self):
        from src.workflow.dg_tft.data_preparer import run_preparer_stage
        agent = _make_mock_subagent(json.dumps({
            "_tool": "stop_workflow", "outcome": "needs_input",
            "reason": "Missing gate mapping", "evidence": "CSV only names VG",
            "required_information": "Which gate is VG?",
        }))
        with patch("src.agent.subagent.SubAgent", return_value=agent):
            with self.assertRaises(WorkflowStopped):
                run_preparer_stage(self.ctx, "/raw", str(self.output))


class StreamingLifecycleTests(unittest.TestCase):
    def test_silent_process_timeout_and_full_logs(self):
        with tempfile.TemporaryDirectory() as root:
            result = _run_stage_streaming(_make_mock_ctx(),
                [sys.executable, "-u", "-c", "import time; print('started', flush=True); time.sleep(30)"],
                "probe", root, .3)
            self.assertFalse(result.ok)
            self.assertIn("Timed out", result.error)
            self.assertIn("started", Path(result.outputs["stdout_log"]).read_text())

    def test_interrupt_terminates_owned_process_and_preserves_logs(self):
        real_popen = subprocess.Popen
        processes = []

        def spawn(*args, **kwargs):
            process = real_popen(*args, **kwargs)
            # Cleanup invokes taskkill via subprocess.run/Popen as well. Only
            # interrupt the probe's wait, never the cleanup helper's wait.
            if processes or args[0][0] != sys.executable:
                return process
            processes.append(process)
            original_wait = process.wait
            first = True

            def interrupt_once(*args, **kwargs):
                nonlocal first
                if first:
                    first = False
                    raise KeyboardInterrupt()
                return original_wait(*args, **kwargs)
            process.wait = interrupt_once
            return process

        with tempfile.TemporaryDirectory() as root:
            with patch("src.workflow.dg_tft.stage_runner.subprocess.Popen", side_effect=spawn):
                with self.assertRaises(KeyboardInterrupt):
                    _run_stage_streaming(_make_mock_ctx(),
                        [sys.executable, "-c", "import time; time.sleep(30)"], "probe", root, 10)
            self.assertIsNotNone(processes[0].poll())
            self.assertEqual(len(list((Path(root) / "stage_logs").glob("*.log"))), 2)


if __name__ == "__main__":
    unittest.main()
