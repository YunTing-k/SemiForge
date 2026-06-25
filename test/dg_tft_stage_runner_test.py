# -*- coding: utf-8 -*-
"""
Unit tests for stage_runner.py — interpreter resolution and stage execution.
Run: python -m unittest test.dg_tft_stage_runner_test
"""
import sys
import os
import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.workflow.dg_tft.stage_runner import (
    resolve_modeling_python,
    run_stage,
    RUN_MODEL_PATH,
)
from src.workflow.dg_tft.schema import StageResult


class MockContext:
    """Minimal mock of AgentContext for testing stage_runner."""
    def __init__(self, configs=None):
        self.agent_configs = configs or {}


class TestResolveModelingPython(unittest.TestCase):
    def test_uses_configured_path_when_exists(self):
        ctx = MockContext({
            "DG_TFT_MODELING_PYTHON_EXE": sys.executable,
            "DG_TFT_AGENT_PYTHON_EXE": sys.executable,
        })
        result = resolve_modeling_python(ctx)
        self.assertEqual(result, sys.executable)

    def test_falls_back_to_sys_executable_when_not_configured(self):
        ctx = MockContext({})
        result = resolve_modeling_python(ctx)
        self.assertEqual(result, sys.executable)

    def test_falls_back_when_configured_path_missing(self):
        ctx = MockContext({
            "DG_TFT_MODELING_PYTHON_EXE": "C:/nonexistent/python.exe",
        })
        result = resolve_modeling_python(ctx)
        self.assertEqual(result, sys.executable)

    def test_returns_absolute_path(self):
        ctx = MockContext({
            "DG_TFT_MODELING_PYTHON_EXE": sys.executable,
        })
        result = resolve_modeling_python(ctx)
        self.assertTrue(os.path.isabs(result))


class TestRunStageCommand(unittest.TestCase):
    def setUp(self):
        self.ctx = MockContext({
            "DG_TFT_MODELING_PYTHON_EXE": sys.executable,
            "DG_TFT_AGENT_PYTHON_EXE": sys.executable,
        })

    @patch("subprocess.run")
    def test_successful_stage(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="Training complete\nLoss: 0.001\n",
            stderr="",
        )
        result = run_stage(self.ctx, "check-data", "/tmp/config.ini", cwd="/tmp")
        self.assertTrue(result.ok)
        self.assertEqual(result.stage, "check-data")
        self.assertIsNone(result.error)

    @patch("subprocess.run")
    def test_failed_stage(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=1,
            stdout="Starting...\n",
            stderr="Error: file not found\n",
        )
        result = run_stage(self.ctx, "fit-physical", "/tmp/config.ini", cwd="/tmp")
        self.assertFalse(result.ok)
        self.assertEqual(result.stage, "fit-physical")
        self.assertIsNotNone(result.error)
        self.assertIn("Exit code 1", result.error)

    @patch("subprocess.run")
    def test_stage_with_warnings(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="OK\n",
            stderr="Warning: convergence slow\nwarning: high residual\n",
        )
        result = run_stage(self.ctx, "train-static", "/tmp/config.ini", cwd="/tmp")
        self.assertTrue(result.ok)
        self.assertGreater(len(result.warnings), 0)

    @patch("subprocess.run", side_effect=FileNotFoundError("python not found"))
    def test_python_not_found(self, mock_run):
        result = run_stage(self.ctx, "check-data", "/tmp/config.ini", cwd="/tmp")
        self.assertFalse(result.ok)
        self.assertIn("not found", result.error)

    @patch("subprocess.run")
    def test_command_format(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="OK", stderr="")
        run_stage(self.ctx, "check-data", "C:/data/config.ini", cwd="C:/work")

        args = mock_run.call_args[0][0]
        self.assertTrue(args[0].endswith("python.exe") or args[0].endswith("python"))
        self.assertIn("check-data", args)
        self.assertIn("--config", args)
        self.assertIn("C:/data/config.ini", args)

    @patch("subprocess.run")
    def test_extra_args_passed(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="OK", stderr="")
        run_stage(self.ctx, "train-static", "/tmp/config.ini",
                  extra_args=["--seed", "42"], cwd="/tmp")

        args = mock_run.call_args[0][0]
        self.assertIn("--seed", args)
        self.assertIn("42", args)


class TestRunModelPath(unittest.TestCase):
    def test_run_model_path_exists(self):
        self.assertTrue(RUN_MODEL_PATH.resolve().exists(),
                        f"run_model.py not found at {RUN_MODEL_PATH}")

    def test_run_model_path_is_file(self):
        self.assertTrue(RUN_MODEL_PATH.resolve().is_file())

    def test_run_model_path_ends_correctly(self):
        self.assertTrue(str(RUN_MODEL_PATH).endswith(
            os.path.join("device_modeling", "dg_tft", "run_model.py")
        ))


if __name__ == "__main__":
    unittest.main()
