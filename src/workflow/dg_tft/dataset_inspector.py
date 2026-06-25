# -*- coding: utf-8 -*-
"""
Header information
---------
Shanghai Jiao Tong University, School of Integrated Circuits, SMIL Lab
Author: Yu Huang
Create Date: 2026.6.25
Description: Dataset inspector — calls CUDA-env validate_dataset.py via subprocess,
             auto-discovers I-t directories, and builds DatasetProfile.

Revision:
---------
2026.6.25      Yu Huang      1.0      Initial implementation
"""
from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

from .schema import DatasetProfile

if TYPE_CHECKING:
    from src.context.agent_context import AgentContext

sys_log = logging.getLogger("logger")

_RUN_MODEL_DIR = Path(__file__).parent.parent.parent / "device_modeling" / "dg_tft"
_VALIDATE_SCRIPT = _RUN_MODEL_DIR / "engine" / "validate_dataset.py"

_IT_CANDIDATE_NAMES = ["hysteresis_it", "it", "hysteresis", "I-t", "it_data", "hysteresis_data", "hyst"]


def inspect_dataset(
    ctx: AgentContext,
    dataset_dir: str,
    *,
    hysteresis_dir: str | None = None,
    json_out_path: str | None = None,
) -> tuple[DatasetProfile, dict]:
    """Inspect a dataset directory and build a DatasetProfile (P2-T1 through P2-T5).

    Calls validate_dataset.py in the CUDA modeling environment via subprocess,
    parses its JSON output, discovers I-t data, analyzes temperature support,
    and checks HSPICE availability.

    Args:
        ctx: Agent context (for configs and interpreter resolution).
        dataset_dir: Path to the static IV/CV dataset directory.
        hysteresis_dir: Optional user-specified I-t / hysteresis data directory.
        json_out_path: If given, save the raw validation JSON to this file.

    Returns:
        (DatasetProfile, raw_validation_dict)
    """
    from .stage_runner import resolve_modeling_python

    modeling_python = resolve_modeling_python(ctx)
    data_path = Path(dataset_dir).resolve()

    it_dir = _discover_it_dir(data_path, hysteresis_dir)

    cmd = [modeling_python, str(_VALIDATE_SCRIPT), "--data-dir", str(data_path)]
    if json_out_path:
        out_path = Path(json_out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cmd += ["--json-out", str(out_path)]
    if it_dir and it_dir.exists():
        cmd += ["--it-dir", str(it_dir)]

    sys_log.info("dataset_inspector: %s", " ".join(cmd))
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(data_path.parent),
    )

    if result.returncode != 0:
        sys_log.error("validate_dataset.py failed (rc=%d): %s", result.returncode, result.stderr[:500])
        return (
            DatasetProfile(
                static_ok=False,
                issues=[f"validate_dataset.py exited {result.returncode}: {result.stderr[:500]}"],
            ),
            {},
        )

    stdout = result.stdout.strip()
    if not stdout:
        return DatasetProfile(static_ok=False, issues=["validate_dataset.py returned empty output"]), {}

    try:
        raw = json.loads(stdout)
    except json.JSONDecodeError as e:
        return DatasetProfile(static_ok=False, issues=[f"Failed to parse validation JSON: {e}"]), {}

    profile = _build_profile(data_path, raw, it_dir)
    _check_hspice(ctx, profile)

    return profile, raw


def _discover_it_dir(data_path: Path, user_dir: str | None) -> Path | None:
    """Auto-discover I-t / hysteresis directory (P2-T3).

    Priority:
      1. User-specified hysteresis_dir (if valid).
      2. dataset_dir/hysteresis_it.
      3. dataset_dir/<name> where name contains hysteresis/it/I-t keywords.
      4. Sibling directories whose names match hysteresis/it keywords.
    """
    if user_dir:
        p = Path(user_dir)
        if p.exists():
            sys_log.info("Using user-specified hysteresis dir: %s", p)
            return p
        sys_log.warning("User hysteresis_dir not found, falling back to auto-discovery: %s", user_dir)

    candidates: list[Path] = []

    it_dir = data_path / "hysteresis_it"
    if it_dir.exists():
        candidates.append(it_dir)

    for name in _IT_CANDIDATE_NAMES:
        p = data_path / name
        if p.exists() and p not in candidates:
            candidates.append(p)

    parent = data_path.parent
    if parent.resolve() != data_path.resolve():
        try:
            for child in sorted(parent.iterdir()):
                if not child.is_dir() or child.resolve() == data_path.resolve():
                    continue
                name_lower = child.name.lower()
                if any(kw in name_lower for kw in ["it", "hysteresis", "i-t"]):
                    if child not in candidates:
                        candidates.append(child)
        except (PermissionError, FileNotFoundError, OSError):
            sys_log.debug("Cannot scan parent directory: %s", parent)

    if candidates:
        sys_log.info(
            "Auto-discovered %d I-t candidate(s): %s, using first",
            len(candidates), [str(c) for c in candidates],
        )
        return candidates[0]

    return None


def _build_profile(data_path: Path, raw: dict, it_dir: Path | None) -> DatasetProfile:
    """Build DatasetProfile from raw validation JSON (P2-T1, P2-T4)."""
    profile = DatasetProfile()
    profile.raw_validation = raw

    iv_data = raw.get("iv", [])
    profile.static_ok = raw.get("ok_static", False)
    profile.iv_families = [item["family"] for item in iv_data]

    cv_data = raw.get("cv", {})
    profile.cv_ok = cv_data.get("ok", False)

    file_count = sum(len(f.get("files", [])) for f in iv_data)
    if cv_data.get("exists"):
        file_count += 1
    profile.total_files = file_count

    has_temp = False
    for family in iv_data:
        for f in family.get("files", []):
            if f.get("temperature_columns"):
                has_temp = True
    profile.has_temperature_iv = has_temp

    it_data = raw.get("it", {})
    if it_data.get("provided") and it_data.get("ok"):
        profile.has_hysteresis = True
        profile.hysteresis_workbooks = [
            f.get("file", str(f)) for f in it_data.get("files", [])
        ]

    if it_dir:
        profile.suggested_hysteresis_dir = str(it_dir)

    issues: list[str] = []
    for family in iv_data:
        issues.extend(family.get("issues", []))
        for f in family.get("files", []):
            issues.extend(f.get("issues", []))
    for sheet_name, sheet_info in cv_data.get("sheets", {}).items():
        if isinstance(sheet_info, dict):
            issues.extend(sheet_info.get("issues", []))
    if isinstance(cv_data.get("issues"), list):
        issues.extend(cv_data["issues"])
    it_issues = it_data.get("issues", [])
    if isinstance(it_issues, list) and it_issues != ["not checked"]:
        issues.extend(it_issues)
    profile.issues = issues

    return profile


def _check_hspice(ctx: AgentContext, profile: DatasetProfile) -> None:
    """Check HSPICE executable availability (P2-T5).

    Reads hspice_exe from the modeling config template and verifies it exists.
    Appends a warning to profile.issues if not found.
    """
    template_path = _read_template_hspice_path(ctx)
    if not template_path:
        profile.issues.append("HSPICE path not configured; deployment.run_hspice will be set to no")
        return

    if Path(template_path).exists():
        return

    msg = f"HSPICE executable not found at {template_path}; run_hspice will be set to no"
    sys_log.warning(msg)
    profile.issues.append(msg)


def _read_template_hspice_path(ctx: AgentContext) -> str:
    """Extract hspice_exe from the default config template .ini."""
    import configparser

    template_key = ctx.agent_configs.get("DG_TFT_DEFAULT_CONFIG_TEMPLATE", "")
    if not template_key:
        return ""
    template_path = Path(template_key)
    if not template_path.exists():
        return ""

    cfg = configparser.ConfigParser()
    cfg.read(template_path, encoding="utf-8")
    return cfg.get("paths", "hspice_exe", fallback="")


def is_hspice_available(ctx: AgentContext) -> bool:
    """Convenience: check whether HSPICE is available (P2-T5)."""
    path = _read_template_hspice_path(ctx)
    return bool(path) and Path(path).exists()
