"""User source constraints must reach both preparation agents on every attempt."""
import json
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from src.workflow.dg_tft.data_preparer import run_preparer_stage, run_verifier_stage
from src.workflow.dg_tft.orchestrator import _run_agent_stage
from src.workflow.dg_tft.operation_log import OperationLog
from src.workflow.dg_tft.stages import get_stage_by_name
from test.dg_tft_agent_flow_test import _make_mock_ctx, _make_mock_subagent


class InputScopeTests(unittest.TestCase):
    def test_run_tool_keeps_user_request_when_preferences_are_omitted(self):
        from src.tool.tool_def import dg_tft_run
        ctx = _make_mock_ctx()
        request = "Only use measurements under /final."
        ctx.messages = [{"role": "user", "content": request},
                        {"role": "user", "content": "<system_reminder>Update tasks</system_reminder>"}]
        with tempfile.TemporaryDirectory() as output, \
             patch("src.constants.SESSION_PATH", output), \
             patch("src.tool.tool_def.pause_for_permission"), \
             patch("src.tool.tool_def.resume_from_permission"), \
             patch("src.tool.tool_def.ask_permission_tui", return_value=(True, None)), \
             patch("src.workflow.dg_tft.orchestrator.run_workflow", return_value=(True, "done")) as run:
            result = dg_tft_run({"dataset_dir": "/final"}, ctx, MagicMock())
        self.assertTrue(result["workflow_ok"])
        self.assertEqual(run.call_args.kwargs["preferences"], request)

    def test_orchestrator_preserves_source_contract_for_preparer_and_verifier(self):
        ctx = _make_mock_ctx()
        request = "Only use final and its subdirectories; do not import external measurements."
        with tempfile.TemporaryDirectory() as output:
            _, rid, _ = ctx.workflow_man.create_run("/final", output_root=output)
            log = OperationLog(output)
            log.initialize(rid, request)
            retries = {}
            with patch("src.workflow.dg_tft.data_preparer.run_preparer_stage",
                       return_value={"ok": True, "handoff": {}}) as prep, \
                 patch("src.workflow.dg_tft.data_preparer.run_verifier_stage",
                       return_value={"ok": True, "feedback": {}, "issues": []}) as verify:
                for stage in ("data-prep", "verifier", "data-prep", "verifier"):
                    _run_agent_stage(ctx, rid, get_stage_by_name(stage), output,
                                     "/final", "/final", request, log, None, retries, 3)
                for call in prep.call_args_list + verify.call_args_list:
                    self.assertEqual(call.kwargs["preferences"], request)
                self.assertEqual(verify.call_args.kwargs["raw_dataset_dir"], "/final")

    def test_actual_subagent_prompts_include_scope_before_handoff_claims(self):
        ctx = _make_mock_ctx()
        request = "仅使用 final 下的测量数据，不得引入目录之外的数据。"
        agent = _make_mock_subagent(json.dumps({"observations": [], "actions": [],
                                                "output_manifest": {}, "overall_ok": True}))
        with tempfile.TemporaryDirectory() as output, \
             patch("src.agent.subagent.SubAgent", return_value=agent) as spawn, \
             patch("src.workflow.dg_tft.data_preparer._check_output", return_value=(True, [])):
            run_preparer_stage(ctx, "/final", output, preferences=request)
            prep_prompt = spawn.call_args.kwargs["prompt"]
            self.assertIn(request, prep_prompt)
            self.assertIn("Measurement source root: /final", prep_prompt)
            run_verifier_stage(ctx, output, preferences=request, raw_dataset_dir="/final",
                               handoff={"actions": ["Imported sibling CV because it is the same device"]})
            verify_prompt = spawn.call_args.kwargs["prompt"]
            self.assertIn(request, verify_prompt)
            self.assertIn("Authorized measurement source root: /final", verify_prompt)
            self.assertIn("verification failure", verify_prompt)


if __name__ == "__main__":
    unittest.main()
