"""Exercise the production adjustment handler and its SubAgent retry boundary.

The handler closes over a workflow run. Extract its unchanged function AST to
bind a temporary INI without starting the orchestrator, a model or a gateway.
Run: python -m unittest test.dg_tft_agent_adjustment_test -v
"""
import ast
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from src.agent.progress import AgentStatus
from src.agent.subagent import SubAgent
from src.constants import FAIL_LABEL
from src.workflow.dg_tft.param_defs import get_stage_param_table, get_stage_sections
from src.workflow.dg_tft.quality_guard import apply_agent_adjustments, validate_agent_adjustments


class TestAgentAdjustmentFeedback(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "config.ini"
        self.original = "[full_model]\nbase_iv_epochs = 100\nbase_iv_learning_rate = 0.0001\n"
        self.path.write_text(self.original, encoding="utf-8")
        self.table = get_stage_param_table("train-static", task_mode="iv_only")
        source = Path(__file__).resolve().parents[1] / "src/workflow/dg_tft/orchestrator.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        handler = next(n for n in ast.walk(tree)
                       if isinstance(n, ast.FunctionDef) and n.name == "_adjust_params_handler")
        namespace = {
            "validate_agent_adjustments": validate_agent_adjustments,
            "apply_agent_adjustments": apply_agent_adjustments,
            "param_table": self.table,
            "sections": get_stage_sections("train-static"),
            "sd": SimpleNamespace(name="train-static"),
            "config_path": str(self.path),
            "FAIL_LABEL": FAIL_LABEL,
        }
        exec(compile(ast.Module(body=[handler], type_ignores=[]), str(source), "exec"), namespace)
        self.agent = SubAgent.__new__(SubAgent)
        self.agent.ctx = SimpleNamespace(messages=[])
        self.agent._tool_handlers = {"adjust_params": namespace["_adjust_params_handler"]}
        self.agent.agent_id = "test-adjustment-feedback"
        self.agent.stop_mode = "schema"
        self.agent.stop_schemas = ["adjust_params"]
        self.agent.status = None
        self.agent.result = None
        self.agent.progress = SimpleNamespace(status=None)

    def submit(self, arguments):
        call = {"id": "test-call", "type": "function", "function": {
            "name": "adjust_params", "arguments": json.dumps(arguments)}}
        self.agent._handle_subagent_tool(call, "adjust_params", arguments)
        return json.loads(self.agent.ctx.messages[-1]["content"])

    def test_rejected_operation_can_be_repaired_in_same_agent(self):
        rejected = self.submit({"base_epochs": 700, "reasoning": "Increase base IV budget."})
        self.assertEqual(rejected["status"], FAIL_LABEL)
        self.assertNotEqual(self.agent.status, AgentStatus.DONE)
        self.assertEqual(self.path.read_text(encoding="utf-8"), self.original)
        accepted = self.submit({"base_iv_epochs": 700, "reasoning": "Use the supported parameter."})
        self.assertEqual(accepted["applied"], {"base_iv_epochs": 700})
        self.assertEqual(self.agent.status, AgentStatus.DONE)
        self.assertIn("base_iv_epochs = 700", self.path.read_text(encoding="utf-8"))

    def test_partial_invalid_operation_never_writes_valid_subset(self):
        result = self.submit({"base_iv_epochs": 700, "cv_weight": 8,
                              "reasoning": "An IV-only task must reject CV tuning."})
        self.assertEqual(result["status"], FAIL_LABEL)
        self.assertEqual(result["rejected"], ["cv_weight"])
        self.assertNotEqual(self.agent.status, AgentStatus.DONE)
        self.assertEqual(self.path.read_text(encoding="utf-8"), self.original)

    def test_missing_reason_or_values_returns_feedback(self):
        for args in ({"base_iv_epochs": 700}, {"base_iv_epochs": 700, "reasoning": 42},
                     {"base_iv_epochs": 700, "reasoning": " "}, {"reasoning": "No change."}, []):
            with self.subTest(args=args):
                self.assertEqual(self.submit(args)["status"], FAIL_LABEL)
                self.assertNotEqual(self.agent.status, AgentStatus.DONE)
                self.assertEqual(self.path.read_text(encoding="utf-8"), self.original)

    def test_safe_coercion_and_clamping_still_execute(self):
        result = self.submit({"base_iv_epochs": "700.0", "base_iv_learning_rate": 1e-8,
                              "reasoning": "Valid integer coercion and documented bounds."})
        self.assertEqual(result["applied"], {"base_iv_epochs": 700, "base_iv_learning_rate": 0.0001})
        self.assertTrue(result["warnings"])
        self.assertEqual(self.agent.status, AgentStatus.DONE)

    def test_nonfinite_and_fractional_integers_are_rejected(self):
        for key, values in {
            "base_iv_epochs": ["700.5", 700.5, "nan", float("inf"), True],
            "base_iv_learning_rate": ["NaN", "Infinity", "-Infinity", float("nan"), True],
        }.items():
            for value in values:
                with self.subTest(key=key, value=value):
                    result = self.submit({key: value, "reasoning": "Reject invalid numeric configuration."})
                    self.assertEqual(result["status"], FAIL_LABEL)
                    self.assertNotEqual(self.agent.status, AgentStatus.DONE)
                    self.assertEqual(self.path.read_text(encoding="utf-8"), self.original)


if __name__ == "__main__":
    unittest.main()
