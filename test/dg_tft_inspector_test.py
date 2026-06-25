# -*- coding: utf-8 -*-
"""
Unit tests for dataset_inspector.py (P2-T1 through P2-T5).

Tests:
  - _discover_it_dir: directory discovery logic
  - _build_profile: raw JSON to DatasetProfile
  - inspect_dataset: subprocess integration (mocked)
  - _check_hspice / is_hspice_available: HSPICE detection

Run: python -m unittest test.dg_tft_inspector_test -v
"""
import sys
import os
import json
import unittest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.workflow.dg_tft.dataset_inspector import (
    _discover_it_dir,
    _build_profile,
    inspect_dataset,
    _check_hspice,
    is_hspice_available,
)
from src.workflow.dg_tft.schema import DatasetProfile
from src.context.agent_context import AgentContext


_VALID_IV = {
    "family": "IDVBG",
    "matched_files": ["IDVBG_train.xlsx"],
    "files": [
        {
            "file": "IDVBG_train.xlsx",
            "exists": True,
            "ok": True,
            "issues": [],
            "temperature_columns": ["temp_K"],
            "columns": ["VD_V", "VBG_V", "VTG_V", "ID_A", "temp_K"],
        }
    ],
    "ok": True,
    "issues": [],
}

_VALID_CV = {
    "file": "CV_train.xlsx",
    "exists": True,
    "ok": True,
    "issues": [],
    "sheets": {
        "cgg": {"exists": True, "usable_curve_pairs_preview": 3, "issues": []},
        "cbg": {"exists": True, "usable_curve_pairs_preview": 2, "issues": []},
        "ctg": {"exists": True, "usable_curve_pairs_preview": 2, "issues": []},
        "vds": {"exists": True, "usable_curve_pairs_preview": 1, "issues": []},
    },
}

_VALID_IT = {
    "provided": True,
    "path": "/data/hysteresis_it",
    "exists": True,
    "ok": True,
    "files": [{"file": "I-t_1.xlsx", "sheets": {"raw": {"columns": [], "missing_normalized_fields": []}}}],
    "issues": [],
}

_FULL_VALID_JSON = {
    "data_dir": "/data/device",
    "iv": [_VALID_IV.copy()],
    "cv": _VALID_CV.copy(),
    "it": _VALID_IT.copy(),
    "ok_static": True,
    "ok_with_it": True,
}


class TestDiscoverItDir(unittest.TestCase):
    """P2-T3: I-t directory auto-discovery."""

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())
        self.data_dir = self.tmpdir / "device_data"
        self.data_dir.mkdir()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_user_specified_exists(self):
        it_dir = self.tmpdir / "my_it"
        it_dir.mkdir()
        result = _discover_it_dir(self.data_dir, str(it_dir))
        self.assertEqual(result, it_dir)

    def test_user_specified_not_found(self):
        result = _discover_it_dir(self.data_dir, str(self.tmpdir / "nonexistent"))
        self.assertIsNone(result)

    def test_hysteresis_it_subdir(self):
        it_dir = self.data_dir / "hysteresis_it"
        it_dir.mkdir()
        result = _discover_it_dir(self.data_dir, None)
        self.assertEqual(result, it_dir)

    def test_it_subdir(self):
        it_dir = self.data_dir / "it"
        it_dir.mkdir()
        result = _discover_it_dir(self.data_dir, None)
        self.assertEqual(result, it_dir)

    def test_hysteresis_subdir(self):
        it_dir = self.data_dir / "hysteresis"
        it_dir.mkdir()
        result = _discover_it_dir(self.data_dir, None)
        self.assertEqual(result, it_dir)

    def test_I_t_subdir(self):
        it_dir = self.data_dir / "I-t"
        it_dir.mkdir()
        result = _discover_it_dir(self.data_dir, None)
        self.assertEqual(result, it_dir)

    def test_sibling_dir_with_it_keyword(self):
        sibling_it = self.tmpdir / "device_hysteresis"
        sibling_it.mkdir()
        result = _discover_it_dir(self.data_dir, None)
        self.assertEqual(result, sibling_it)

    def test_no_it_dir_found(self):
        result = _discover_it_dir(self.data_dir, None)
        self.assertIsNone(result)

    def test_multiple_candidates_uses_first(self):
        it1 = self.data_dir / "hysteresis_it"
        it1.mkdir()
        it2 = self.data_dir / "it"
        it2.mkdir()
        result = _discover_it_dir(self.data_dir, None)
        self.assertEqual(result, it1)


class TestBuildProfile(unittest.TestCase):
    """P2-T1: build DatasetProfile from raw validation JSON."""

    def test_full_valid_profile(self):
        profile = _build_profile(Path("/data/device"), _FULL_VALID_JSON, Path("/data/device/hysteresis_it"))
        self.assertTrue(profile.static_ok)
        self.assertTrue(profile.cv_ok)
        self.assertEqual(profile.iv_families, ["IDVBG"])
        self.assertTrue(profile.has_temperature_iv)
        self.assertTrue(profile.has_hysteresis)
        self.assertEqual(profile.hysteresis_workbooks, ["I-t_1.xlsx"])
        self.assertEqual(profile.suggested_hysteresis_dir, str(Path("/data/device/hysteresis_it")))
        self.assertEqual(profile.total_files, 2)
        self.assertEqual(profile.issues, [])

    def test_no_temperature(self):
        raw = {
            "data_dir": "/data/device",
            "iv": [
                {
                    "family": "IDVD",
                    "ok": True,
                    "files": [
                        {"ok": True, "temperature_columns": [], "issues": []}
                    ],
                    "issues": [],
                }
            ],
            "cv": {"file": "CV.xlsx", "exists": False, "ok": False, "issues": [], "sheets": {}},
            "it": {"provided": False, "ok": True, "issues": ["not checked"]},
            "ok_static": False,
        }
        profile = _build_profile(Path("/data/device"), raw, None)
        self.assertFalse(profile.has_temperature_iv)
        self.assertFalse(profile.has_hysteresis)
        self.assertFalse(profile.static_ok)
        self.assertIsNone(profile.temperature_span_K)

    def test_issues_collected(self):
        raw = {
            "data_dir": "/data/bad",
            "iv": [
                {
                    "family": "IDVBG",
                    "ok": False,
                    "files": [],
                    "issues": ["no workbook matched", "missing file"],
                }
            ],
            "cv": {
                "file": "CV.xlsx",
                "exists": False,
                "ok": False,
                "issues": ["missing file; CV fitting will be skipped"],
                "sheets": {},
            },
            "it": {"provided": False, "ok": True, "issues": ["not checked"]},
            "ok_static": False,
        }
        profile = _build_profile(Path("/data/bad"), raw, None)
        self.assertIn("no workbook matched", profile.issues)
        self.assertIn("missing file", profile.issues)
        self.assertIn("missing file; CV fitting will be skipped", profile.issues)
        self.assertEqual(profile.total_files, 0)

    def test_no_hysteresis_data(self):
        raw_no_it = dict(_FULL_VALID_JSON)
        raw_no_it["it"] = {"provided": False, "ok": True, "issues": ["not checked"]}
        profile = _build_profile(Path("/data/device"), raw_no_it, None)
        self.assertFalse(profile.has_hysteresis)
        self.assertEqual(profile.hysteresis_workbooks, [])

    def test_raw_validation_stored(self):
        profile = _build_profile(Path("/data/device"), _FULL_VALID_JSON, None)
        self.assertEqual(profile.raw_validation, _FULL_VALID_JSON)


class TestHspiceCheck(unittest.TestCase):
    """P2-T5: HSPICE availability detection."""

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())
        self.template_path = self.tmpdir / "model_config.ini"
        self.fake_hspice = self.tmpdir / "hspice.com"

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _make_ctx(self, hspice_path=""):
        ctx = AgentContext()
        if hspice_path:
            self.template_path.write_text(
                f"[paths]\nhspice_exe = {hspice_path}\n[temperature]\nmode = auto\n[hysteresis]\nenabled = no\n[deployment]\nrun_hspice = yes\n",
                encoding="utf-8",
            )
        else:
            self.template_path.write_text(
                "[paths]\n[temperature]\nmode = auto\n[hysteresis]\nenabled = no\n[deployment]\nrun_hspice = yes\n",
                encoding="utf-8",
            )
        ctx.agent_configs = {"DG_TFT_DEFAULT_CONFIG_TEMPLATE": str(self.template_path)}
        return ctx

    def test_hspice_exists(self):
        self.fake_hspice.write_text("")
        ctx = self._make_ctx(str(self.fake_hspice))
        profile = DatasetProfile()
        _check_hspice(ctx, profile)
        self.assertNotIn(
            "HSPICE executable not found",
            "\n".join(profile.issues) if profile.issues else "",
        )

    def test_hspice_not_found(self):
        ctx = self._make_ctx(str(self.tmpdir / "nonexistent_hspice.com"))
        profile = DatasetProfile()
        _check_hspice(ctx, profile)
        self.assertTrue(any("not found" in i for i in profile.issues))

    def test_is_hspice_available_true(self):
        self.fake_hspice.write_text("")
        ctx = self._make_ctx(str(self.fake_hspice))
        self.assertTrue(is_hspice_available(ctx))

    def test_is_hspice_available_false(self):
        ctx = self._make_ctx(str(self.tmpdir / "no_hspice.exe"))
        self.assertFalse(is_hspice_available(ctx))

    def test_no_template_config_key(self):
        ctx = AgentContext()
        ctx.agent_configs = {}
        profile = DatasetProfile()
        _check_hspice(ctx, profile)
        self.assertIn("HSPICE path not configured", profile.issues[0])


class TestInspectDatasetMocked(unittest.TestCase):
    """P2-T1/P2-T2: full inspect_dataset with mocked subprocess."""

    def _make_ctx(self):
        ctx = AgentContext()
        ctx.agent_configs = {
            "DG_TFT_MODELING_PYTHON_EXE": sys.executable,
            "DG_TFT_DEFAULT_CONFIG_TEMPLATE": str(Path(tempfile.gettempdir()) / "dummy.ini"),
        }
        return ctx

    @patch("src.workflow.dg_tft.dataset_inspector.subprocess.run")
    def test_successful_inspection(self, mock_run):
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps(_FULL_VALID_JSON)
        mock_result.stderr = ""
        mock_run.return_value = mock_result

        ctx = self._make_ctx()
        profile, raw = inspect_dataset(ctx, "/fake/data")

        self.assertTrue(profile.static_ok)
        self.assertTrue(profile.has_temperature_iv)
        self.assertTrue(profile.has_hysteresis)
        self.assertIn("IDVBG", profile.iv_families)
        self.assertEqual(raw["ok_static"], True)

    @patch("src.workflow.dg_tft.dataset_inspector.subprocess.run")
    def test_validation_failure(self, mock_run):
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = ""
        mock_result.stderr = "Traceback: some error"
        mock_run.return_value = mock_result

        ctx = self._make_ctx()
        profile, raw = inspect_dataset(ctx, "/fake/data")

        self.assertFalse(profile.static_ok)
        self.assertIn("exited 1", profile.issues[0])
        self.assertEqual(raw, {})

    @patch("src.workflow.dg_tft.dataset_inspector.subprocess.run")
    def test_empty_output(self, mock_run):
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = ""
        mock_result.stderr = ""
        mock_run.return_value = mock_result

        ctx = self._make_ctx()
        profile, raw = inspect_dataset(ctx, "/fake/data")

        self.assertFalse(profile.static_ok)
        self.assertIn("empty output", profile.issues[0].lower())
        self.assertEqual(raw, {})

    @patch("src.workflow.dg_tft.dataset_inspector.subprocess.run")
    def test_invalid_json(self, mock_run):
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "not valid json {{{"
        mock_result.stderr = ""
        mock_run.return_value = mock_result

        ctx = self._make_ctx()
        profile, raw = inspect_dataset(ctx, "/fake/data")

        self.assertFalse(profile.static_ok)
        self.assertIn("Failed to parse", profile.issues[0])
        self.assertEqual(raw, {})

    @patch("src.workflow.dg_tft.dataset_inspector.subprocess.run")
    def test_json_out_path(self, mock_run):
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps(_FULL_VALID_JSON)
        mock_result.stderr = ""
        mock_run.return_value = mock_result

        ctx = self._make_ctx()
        tmp = tempfile.mkdtemp()
        try:
            json_out = os.path.join(tmp, "validation.json")
            profile, raw = inspect_dataset(ctx, "/fake/data", json_out_path=json_out)
            mock_run.assert_called_once()
            cmd_args = mock_run.call_args[0][0]
            self.assertIn("--json-out", cmd_args)
            self.assertIn(json_out, cmd_args)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
