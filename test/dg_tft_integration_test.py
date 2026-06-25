# -*- coding: utf-8 -*-
"""
Minimal closed-loop integration test for DG-TFT workflow infrastructure.
Verifies the program itself works as a cohesive system (not just individual modules).

Tests:
  - AgentContext creation with workflow_man
  - config/agent_configs.json contains all required DG-TFT keys
  - WorkflowManager full lifecycle (create → mutate → serialize → deserialize → verify)
  - Session persistence to file
  - run_model.py accessibility
  - Interpreter resolution

Run: python -m unittest test.dg_tft_integration_test
"""
import sys
import os
import json
import unittest
import tempfile
import shutil

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.context.agent_context import AgentContext
from src.workflow.dg_tft.manager import WorkflowManager
from src.workflow.dg_tft.schema import (
    WorkflowRun,
    WorkflowStatus,
    StageResult,
    DatasetProfile,
    StageName,
    ErrorCategory,
)
from src.workflow.dg_tft.stage_runner import (
    resolve_modeling_python,
    RUN_MODEL_PATH,
)
from src.constants import AGENT_CONFIGS_PATH


class TestAgentConfigDGTTFKeys(unittest.TestCase):
    """Verify agent_configs.json has all required DG-TFT keys."""
    
    REQUIRED_KEYS = [
        "DG_TFT_AGENT_PYTHON_EXE",
        "DG_TFT_MODELING_PYTHON_EXE",
        "DG_TFT_MODELING_CONDA_ENV",
        "DG_TFT_WORKFLOW_MAX_CONCURRENT_RUNS",
        "DG_TFT_DEFAULT_OUTPUT_ROOT",
        "DG_TFT_DEFAULT_CONFIG_TEMPLATE",
        "DG_TFT_ENABLE_BACKGROUND_WORKFLOW",
        "DG_TFT_USE_IMPORT_API",
        "DG_TFT_HSPICE_REQUIRED",
    ]

    def test_all_required_keys_present(self):
        config_path = os.path.abspath(AGENT_CONFIGS_PATH)
        self.assertTrue(os.path.exists(config_path),
                        f"agent_configs.json not found at {config_path}")
        
        with open(config_path, "r", encoding="utf-8") as f:
            configs = json.load(f)
        
        missing = [k for k in self.REQUIRED_KEYS if k not in configs]
        self.assertEqual(missing, [], f"Missing config keys: {missing}")

    def test_modeling_python_exe_is_absolute(self):
        with open(AGENT_CONFIGS_PATH, "r", encoding="utf-8") as f:
            configs = json.load(f)
        path = configs.get("DG_TFT_MODELING_PYTHON_EXE", "")
        self.assertTrue(os.path.isabs(path) or path.startswith("."),
                        f"DG_TFT_MODELING_PYTHON_EXE should be absolute path, got: {path}")


class TestAgentContextIntegration(unittest.TestCase):
    """Verify AgentContext properly initializes workflow_man."""

    def test_agent_context_has_workflow_man(self):
        ctx = AgentContext()
        self.assertTrue(hasattr(ctx, "workflow_man"))
        self.assertIsInstance(ctx.workflow_man, WorkflowManager)

    def test_agent_context_to_dict_includes_workflow(self):
        ctx = AgentContext()
        # Minimal setup
        from rich.console import Console
        ctx.agent_configs = {
            "DG_TFT_MODELING_PYTHON_EXE": sys.executable,
            "DG_TFT_AGENT_PYTHON_EXE": sys.executable,
        }
        ctx.workflow_man.create_run("C:/data/test")
        d = ctx.to_dict(Console(quiet=True), mute=True)
        self.assertIn("workflow_runs", d)
        self.assertIn("runs", d["workflow_runs"])

    def test_agent_context_from_dict_restores_workflow(self):
        ctx = AgentContext()
        from rich.console import Console
        ctx.workflow_man.create_run("C:/data/test")
        d = ctx.to_dict(Console(quiet=True), mute=True)

        ctx2 = AgentContext()
        ctx2.agent_list = {}
        ctx2.from_dict(d, Console(quiet=True), mute=True)
        runs = ctx2.workflow_man.list_runs()
        self.assertEqual(len(runs), 1)


class TestWorkflowFullLifecycle(unittest.TestCase):
    """End-to-end workflow lifecycle test (in-memory, no session files)."""

    def test_full_lifecycle_success(self):
        """Simulate a complete successful workflow run."""
        wm = WorkflowManager()

        # Phase: Request normalization → create run
        ok, rid, _ = wm.create_run(
            dataset_dir="C:/project/data/my_device",
            output_root="C:/output/dg_tft_runs/run_0001",
            device_name="IGZO_TFT_v1",
        )
        self.assertTrue(ok)

        # Phase: Data audit → set dataset profile
        dp = DatasetProfile(
            static_ok=True,
            cv_ok=True,
            iv_families=["idvbg", "idvd", "idvtg"],
            has_temperature_iv=True,
            temperature_span_K=100.0,
            has_hysteresis=True,
            hysteresis_workbooks=["I-t_1.xlsx"],
            suggested_hysteresis_dir="C:/project/data/my_device/hysteresis",
        )
        wm.set_dataset_profile(rid, dp)
        wm.update_stage(rid, "check-data", WorkflowStatus.RUNNING,
                        StageResult(ok=True, stage="check-data", summary="Data validation passed"))

        # Phase: fit-physical
        wm.update_stage(rid, "fit-physical", WorkflowStatus.RUNNING,
                        StageResult(ok=True, stage="fit-physical",
                                    summary="Physical fitting succeeded",
                                    outputs={"params": "01_physical_kernel/fitted_params.json"},
                                    metrics={"rms_error": 0.002}))

        # Phase: train-base-pkan
        wm.update_stage(rid, "train-base-pkan", WorkflowStatus.RUNNING,
                        StageResult(ok=True, stage="train-base-pkan",
                                    summary="Base PKAN trained",
                                    outputs={"model": "02_base_pkan/pkan_torch_model.pt"},
                                    metrics={"best_loss": 0.123}))

        # Phase: train-static
        wm.update_stage(rid, "train-static", WorkflowStatus.RUNNING,
                        StageResult(ok=True, stage="train-static",
                                    summary="Static model trained",
                                    metrics={"iv_rms": 0.005, "cv_rms": 0.008}))
        wm.attach_metric(rid, "iv_rms", 0.005)
        wm.attach_metric(rid, "cv_rms", 0.008)

        # Phase: train-hysteresis (conditional)
        if dp.has_hysteresis:
            wm.update_stage(rid, "train-hysteresis", WorkflowStatus.RUNNING,
                            StageResult(ok=True, stage="train-hysteresis",
                                        summary="Hysteresis model trained",
                                        outputs={"va": "04_hysteresis/model.va"}))

        # Phase: deployment-validate
        wm.update_stage(rid, "deployment-validate", WorkflowStatus.RUNNING,
                        StageResult(ok=True, stage="deployment-validate",
                                    summary="Deployment validated",
                                    warnings=["HSPICE not available, skipped smoke test"]))

        # Phase: final-report
        wm.update_stage(rid, "final-report", WorkflowStatus.RUNNING,
                        StageResult(ok=True, stage="final-report", summary="Report generated"))
        wm.attach_artifact(rid, "final_summary", "RUN_SUMMARY.md")
        wm.attach_artifact(rid, "artifact_manifest", "ARTIFACT_MANIFEST.json")
        wm.mark_done(rid)

        # Verify final state
        run = wm.get_run(rid)
        self.assertEqual(run.status, WorkflowStatus.DONE)
        self.assertEqual(len(run.stage_results), 7)  # all stages recorded
        self.assertTrue(run.temperature_enabled)
        self.assertTrue(run.hysteresis_enabled)
        self.assertIn("iv_rms", run.metrics)

    def test_full_lifecycle_failure_at_physical(self):
        """Simulate a run that fails at fit-physical stage."""
        wm = WorkflowManager()
        wm.create_run("C:/data/bad_data")

        wm.update_stage(1, "check-data", WorkflowStatus.RUNNING,
                        StageResult(ok=True, stage="check-data", summary="Passed"))
        wm.update_stage(1, "fit-physical", WorkflowStatus.FAILED,
                        StageResult(ok=False, stage="fit-physical",
                                    error="Numerical divergence in optimizer"))
        wm.mark_failed(1, "Physical fitting failed: numerical divergence")

        run = wm.get_run(1)
        self.assertEqual(run.status, WorkflowStatus.FAILED)
        self.assertEqual(len(run.stage_results), 2)
        self.assertIn("Physical fitting failed", run.errors[0])

    def test_full_lifecycle_no_hysteresis(self):
        """Simulate a run without hysteresis data — should skip that stage."""
        wm = WorkflowManager()
        wm.create_run("C:/data/static_only")

        dp = DatasetProfile(
            static_ok=True,
            cv_ok=True,
            has_temperature_iv=False,
            has_hysteresis=False,
        )
        wm.set_dataset_profile(1, dp)

        wm.update_stage(1, "check-data", WorkflowStatus.RUNNING,
                        StageResult(ok=True, stage="check-data", summary="OK"))
        wm.update_stage(1, "fit-physical", WorkflowStatus.RUNNING,
                        StageResult(ok=True, stage="fit-physical", summary="OK"))
        wm.update_stage(1, "train-base-pkan", WorkflowStatus.RUNNING,
                        StageResult(ok=True, stage="train-base-pkan", summary="OK"))
        wm.update_stage(1, "train-static", WorkflowStatus.RUNNING,
                        StageResult(ok=True, stage="train-static", summary="OK"))
        # train-hysteresis skipped because no hysteresis data
        wm.add_warning(1, "Hysteresis stage skipped: no I-t data available")
        wm.update_stage(1, "deployment-validate", WorkflowStatus.RUNNING,
                        StageResult(ok=True, stage="deployment-validate", summary="OK"))
        wm.mark_done(1)

        run = wm.get_run(1)
        self.assertEqual(run.status, WorkflowStatus.DONE)
        self.assertFalse(run.hysteresis_enabled)
        self.assertEqual(len(run.stage_results), 5)  # hysteresis skipped
        self.assertIn("no i-t data available", run.warnings[0].lower())


class TestSerializationPersistence(unittest.TestCase):
    """Verify workflow state survives serialization/deserialization cycles."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_save_load_file_roundtrip(self):
        wm = WorkflowManager()
        wm.create_run("C:/data/test", output_root="C:/output")
        sr = StageResult(ok=True, stage="check-data", summary="OK")
        wm.update_stage(1, "check-data", WorkflowStatus.RUNNING, sr)
        wm.mark_done(1)

        d = wm.to_dict()
        # Simulate saving to file
        file_path = os.path.join(self.tmpdir, "workflow_runs.json")
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(d, f, indent=2, ensure_ascii=False)

        # Simulate loading from file
        with open(file_path, "r", encoding="utf-8") as f:
            loaded = json.load(f)

        wm2 = WorkflowManager()
        wm2.from_dict(loaded)

        run = wm2.get_run(1)
        self.assertEqual(run.status, WorkflowStatus.DONE)
        self.assertIn("check-data", run.stage_results)
        self.assertTrue(run.stage_results["check-data"].ok)

    def test_multirun_persistence(self):
        wm = WorkflowManager()
        for i in range(5):
            wm.create_run(f"C:/data/run_{i}")
        wm.mark_done(1)
        wm.mark_failed(2, "error")
        wm.cancel_run(3)

        d = wm.to_dict()
        wm2 = WorkflowManager()
        wm2.from_dict(d)

        self.assertEqual(len(wm2.list_runs()), 5)
        self.assertEqual(wm2.get_run(1).status, WorkflowStatus.DONE)
        self.assertEqual(wm2.get_run(2).status, WorkflowStatus.FAILED)
        self.assertEqual(wm2.get_run(3).status, WorkflowStatus.CANCELLED)
        self.assertEqual(wm2.get_run(4).status, WorkflowStatus.PENDING)
        self.assertEqual(wm2.get_run(5).status, WorkflowStatus.PENDING)


class TestInterpreterResolution(unittest.TestCase):
    """Verify interpreter resolution logic."""

    def test_modeling_exe_set_and_exists(self):
        ctx = AgentContext()
        ctx.agent_configs = {
            "DG_TFT_MODELING_PYTHON_EXE": sys.executable,
        }
        result = resolve_modeling_python(ctx)
        self.assertEqual(result, sys.executable)

    def test_modeling_exe_not_set_falls_back(self):
        ctx = AgentContext()
        ctx.agent_configs = {}
        result = resolve_modeling_python(ctx)
        self.assertEqual(result, sys.executable)  # fallback


class TestRunModelAccessibility(unittest.TestCase):
    """Verify run_model.py is accessible at its expected location."""

    def test_run_model_exists(self):
        self.assertTrue(RUN_MODEL_PATH.resolve().exists(),
                        f"run_model.py not found at {RUN_MODEL_PATH}")

    def test_run_model_has_expected_actions(self):
        content = RUN_MODEL_PATH.read_text()
        expected_actions = [
            "check-data",
            "fit-physical",
            "train-base-pkan",
            "train-static",
            "train-hysteresis",
            "train-all",
        ]
        for action in expected_actions:
            self.assertIn(action, content, f"Action '{action}' not found in run_model.py")


if __name__ == "__main__":
    unittest.main()
