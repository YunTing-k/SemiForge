# -*- coding: utf-8 -*-
"""
Unit tests for config_builder.py (P2-T6, P2-T7).

Tests:
  - build_runtime_config: full config generation + file output
  - _build_resolution: resolution metadata computation
  - _write_ini: INI section/field correctness
  - Edge cases: missing template, empty profile, HSPICE disabled

Run: python -m unittest test.dg_tft_builder_test -v
"""
import sys
import os
import json
import unittest
import tempfile
import shutil
import configparser
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.workflow.dg_tft.config_builder import build_runtime_config
from src.workflow.dg_tft.schema import DatasetProfile
from src.context.agent_context import AgentContext


_TEMPLATE_INI = """[paths]
data_dir = D:/replace_with_new_device/static_iv_cv
hysteresis_data_dir = D:/replace_with_new_device/hysteresis_it
output_dir = ../../outputs/new_device
hspice_exe = E:/synopsys/Hspice_S-2021.09/WIN64/hspice.com

[runtime]
device = auto
dtype = float32
seed = 20260524

[physical_fit]
max_iv_points = 50000

[base_pkan]
epochs_shared = 500

[full_model]
epochs = 700

[temperature]
mode = auto
minimum_span_K = 20.0

[hysteresis]
enabled = no
regime = all

[deployment]
run_hspice = yes
maxstep_tau_divisor = 20
"""


def _make_profile(*, static_ok=True, cv_ok=True, has_temp=False, has_hyst=False, hyst_dir=None):
    return DatasetProfile(
        static_ok=static_ok,
        cv_ok=cv_ok,
        iv_families=["IDVBG", "IDVD"],
        has_temperature_iv=has_temp,
        has_hysteresis=has_hyst,
        hysteresis_workbooks=["I-t_1.xlsx"] if has_hyst else [],
        suggested_hysteresis_dir=hyst_dir,
        total_files=3,
        issues=[],
    )


def _make_ctx(tmpdir: Path, *, hspice_path=None, ensure_hspice=True):
    ctx = AgentContext()
    template_path = tmpdir / "template.ini"

    ini_content = _TEMPLATE_INI
    if hspice_path:
        ini_content = ini_content.replace(
            "hspice_exe = E:/synopsys/Hspice_S-2021.09/WIN64/hspice.com",
            f"hspice_exe = {hspice_path}",
        )
        if ensure_hspice:
            hspice_file = Path(hspice_path)
            if not hspice_file.exists():
                hspice_file.parent.mkdir(parents=True, exist_ok=True)
                hspice_file.write_text("")

    template_path.write_text(ini_content, encoding="utf-8")
    ctx.agent_configs = {
        "DG_TFT_DEFAULT_CONFIG_TEMPLATE": str(template_path),
    }
    if hspice_path:
        ctx.agent_configs["DG_TFT_HSPICE_EXE"] = hspice_path
    return ctx, template_path


class TestBuildRuntimeConfig(unittest.TestCase):
    """P2-T6: full config generation."""

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_generates_config_with_temperature_and_hysteresis(self):
        ctx, template = _make_ctx(self.tmpdir, hspice_path=str(self.tmpdir / "hspice.exe"))
        profile = _make_profile(has_temp=True, has_hyst=True, hyst_dir="/data/it")

        result = build_runtime_config(ctx, profile, "/data/device", run_id=42, work_dir=str(self.tmpdir / "work"))

        self.assertTrue(os.path.exists(result["config_path"]))
        self.assertTrue(os.path.exists(result["profile_path"]))
        self.assertTrue(os.path.exists(result["resolution_path"]))

        cfg = configparser.ConfigParser()
        cfg.read(result["config_path"], encoding="utf-8")
        self.assertEqual(cfg.get("paths", "data_dir"), str(Path("/data/device").resolve()))
        self.assertEqual(cfg.get("paths", "hysteresis_data_dir"), "/data/it")
        self.assertEqual(cfg.get("temperature", "mode"), "auto")
        self.assertEqual(cfg.get("hysteresis", "enabled"), "yes")
        self.assertEqual(cfg.get("deployment", "run_hspice"), "yes")

    def test_no_temperature_sets_mode_off(self):
        ctx, template = _make_ctx(self.tmpdir, hspice_path=str(self.tmpdir / "hspice.exe"))
        profile = _make_profile(has_temp=False, has_hyst=False)

        result = build_runtime_config(ctx, profile, "/data/device", run_id=1, work_dir=str(self.tmpdir / "work"))

        cfg = configparser.ConfigParser()
        cfg.read(result["config_path"], encoding="utf-8")
        self.assertEqual(cfg.get("temperature", "mode"), "off")
        self.assertEqual(cfg.get("hysteresis", "enabled"), "no")

    def test_hspice_unavailable_sets_run_hspice_no(self):
        nonexistent = str(self.tmpdir / "hspice" / "does_not_exist.exe")
        ctx, template = _make_ctx(self.tmpdir, hspice_path=nonexistent, ensure_hspice=False)
        profile = _make_profile()

        result = build_runtime_config(ctx, profile, "/data/device", run_id=1, work_dir=str(self.tmpdir / "work"))

        cfg = configparser.ConfigParser()
        cfg.read(result["config_path"], encoding="utf-8")
        self.assertEqual(cfg.get("deployment", "run_hspice"), "no")

    def test_output_dir_contains_run_tag(self):
        ctx, template = _make_ctx(self.tmpdir, hspice_path=str(self.tmpdir / "hspice.exe"))
        profile = _make_profile()

        result = build_runtime_config(
            ctx, profile, "/data/device",
            output_root=str(self.tmpdir / "outputs"),
            run_id=7,
            work_dir=str(self.tmpdir / "work"),
        )

        cfg = configparser.ConfigParser()
        cfg.read(result["config_path"], encoding="utf-8")
        output = cfg.get("paths", "output_dir")
        self.assertIn("run_0007", output)


class TestResolution(unittest.TestCase):
    """P2-T7: resolution metadata correctness."""

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_resolution_json_content(self):
        ctx, template = _make_ctx(self.tmpdir, hspice_path=str(self.tmpdir / "hspice.exe"))
        profile = _make_profile(has_temp=True, has_hyst=True, hyst_dir="/data/it")

        result = build_runtime_config(ctx, profile, "/data/device", run_id=1, work_dir=str(self.tmpdir / "work"))

        with open(result["resolution_path"], "r", encoding="utf-8") as f:
            resolution = json.load(f)

        self.assertTrue(resolution["temperature_enabled"])
        self.assertTrue(resolution["hysteresis_enabled"])
        self.assertEqual(resolution["temperature_mode"], "auto")
        self.assertEqual(resolution["run_hspice"], "yes")
        self.assertEqual(resolution["iv_families"], ["IDVBG", "IDVD"])
        self.assertEqual(resolution["total_files"], 3)

    def test_resolution_with_issues(self):
        nonexistent = str(self.tmpdir / "hspice" / "does_not_exist.exe")
        ctx, template = _make_ctx(self.tmpdir, hspice_path=nonexistent, ensure_hspice=False)
        profile = _make_profile()
        profile.issues = ["Test warning", "Another issue"]

        result = build_runtime_config(ctx, profile, "/data/device", run_id=1, work_dir=str(self.tmpdir / "work"))

        with open(result["resolution_path"], "r", encoding="utf-8") as f:
            resolution = json.load(f)

        self.assertIn("Test warning", resolution["issues"])
        self.assertFalse(resolution["hspice_available"])
        self.assertEqual(resolution["run_hspice"], "no")

    def test_profile_json_content(self):
        ctx, template = _make_ctx(self.tmpdir, hspice_path=str(self.tmpdir / "hspice.exe"))
        profile = _make_profile(has_temp=True, static_ok=False, cv_ok=False)
        profile.iv_families = ["IDVD"]

        result = build_runtime_config(ctx, profile, "/data/device", run_id=1, work_dir=str(self.tmpdir / "work"))

        with open(result["profile_path"], "r", encoding="utf-8") as f:
            saved = json.load(f)

        self.assertFalse(saved["static_ok"])
        self.assertFalse(saved["cv_ok"])
        self.assertTrue(saved["has_temperature_iv"])
        self.assertEqual(saved["iv_families"], ["IDVD"])


class TestWriteIni(unittest.TestCase):
    """P2-T6: INI file content validation."""

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_template_sections_preserved(self):
        """Ensure non-overridden sections/keys are preserved."""
        ctx, template = _make_ctx(self.tmpdir, hspice_path=str(self.tmpdir / "hspice.exe"))
        profile = _make_profile()

        result = build_runtime_config(ctx, profile, "/data/device", run_id=1, work_dir=str(self.tmpdir / "work"))

        cfg = configparser.ConfigParser()
        cfg.read(result["config_path"], encoding="utf-8")

        self.assertEqual(cfg.get("runtime", "device"), "auto")
        self.assertEqual(cfg.get("runtime", "seed"), "20260524")
        self.assertEqual(cfg.get("full_model", "epochs"), "700")
        self.assertEqual(cfg.get("physical_fit", "max_iv_points"), "50000")
        self.assertEqual(cfg.get("deployment", "maxstep_tau_divisor"), "20")

    def test_template_not_found_raises(self):
        ctx = AgentContext()
        ctx.agent_configs = {"DG_TFT_DEFAULT_CONFIG_TEMPLATE": str(self.tmpdir / "nonexistent.ini")}
        profile = _make_profile()

        with self.assertRaises(FileNotFoundError):
            build_runtime_config(ctx, profile, "/data/device", run_id=1, work_dir=str(self.tmpdir / "work"))

    def test_template_key_not_set_raises(self):
        ctx = AgentContext()
        ctx.agent_configs = {}
        profile = _make_profile()

        with self.assertRaises(ValueError):
            build_runtime_config(ctx, profile, "/data/device", run_id=1, work_dir=str(self.tmpdir / "work"))


if __name__ == "__main__":
    unittest.main()
