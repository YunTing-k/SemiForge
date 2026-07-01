# -*- coding: utf-8 -*-
"""
Header information
---------
Shanghai Jiao Tong University, School of Integrated Circuits, SMIL Lab
Author: Yu Huang
Create Date: 2026.6.25
Description: Runtime config builder — generates per-run .ini files from
             a template and DatasetProfile.

Revision:
---------
2026.6.25      Yu Huang      1.0      Initial implementation
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from .dataset_inspector import is_hspice_available
from .schema import DatasetProfile
from src.constants import SESSION_PATH, DG_TFT_RUNS_SUBDIR, DG_TFT_DEFAULT_CONFIG_TEMPLATE

if TYPE_CHECKING:
    from src.context.agent_context import AgentContext

sys_log = logging.getLogger("logger")


def _runs_base_dir(ctx: AgentContext) -> Path:
    """Return the base directory for workflow run outputs under session/<uuid>/runs/."""
    session_uuid = ctx.session_uuid or "unknown"
    runs_subdir = ctx.agent_configs.get("DG_TFT_RUNS_SUBDIR", DG_TFT_RUNS_SUBDIR)
    return Path(SESSION_PATH) / session_uuid / runs_subdir


def build_runtime_config(
    ctx: AgentContext,
    profile: DatasetProfile,
    dataset_dir: str,
    *,
    output_root: str | None = None,
    run_id: int | None = None,
    work_dir: str | None = None,
    overrides: dict[str, str] | None = None,
) -> dict:
    """Build and persist a run-specific DG-TFT .ini configuration (P2-T6, P2-T7).

    Reads the template .ini, fills in dynamic fields based on the DatasetProfile,
    applies optional section.key overrides, and writes the runtime config to disk.

    Args:
        ctx: Agent context (for default config keys).
        profile: DatasetProfile from dataset_inspector.
        dataset_dir: Absolute path to the dataset being modeled.
        output_root: Override for output root directory (default: session/<uuid>/runs/).
        run_id: Optional run ID for generating a unique output subdirectory.
        work_dir: If given, write config files inside this directory.
                   Otherwise uses runs_base/run_<run_id>/.
        overrides: Optional dict of "section.key" → "value" to apply after
                   writing the template-based config.

    Returns:
        dict with keys:
          - config_path: str — path to the generated runtime .ini
          - profile_path: str — path to the saved dataset_profile.json
          - resolution: dict — config_resolution metadata
    """
    data_path = Path(dataset_dir).resolve()
    run_tag = f"run_{run_id:04d}" if run_id is not None else "run_0001"

    if output_root:
        runs_base = Path(output_root).resolve()
    else:
        runs_base = _runs_base_dir(ctx)

    if work_dir:
        config_dir = Path(work_dir).resolve()
    else:
        config_dir = runs_base / run_tag

    config_dir.mkdir(parents=True, exist_ok=True)

    output_dir = str((runs_base / run_tag).resolve())
    resolution = _build_resolution(ctx, profile, data_path, output_dir)

    runtime_config_path = config_dir / "model_config.ini"
    _write_ini(ctx, profile, data_path, resolution, runtime_config_path)
    if overrides:
        _apply_overrides(runtime_config_path, overrides)
    sys_log.info("Runtime config written: %s", runtime_config_path)

    profile_path = config_dir / "dataset_profile.json"
    profile_path.write_text(json.dumps(profile.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    sys_log.info("Dataset profile written: %s", profile_path)

    resolution["profile_path"] = str(profile_path.resolve())

    resolution_path = config_dir / "config_resolution.json"
    resolution_path.write_text(json.dumps(resolution, indent=2, ensure_ascii=False), encoding="utf-8")
    sys_log.info("Config resolution written: %s", resolution_path)

    return {
        "config_path": str(runtime_config_path.resolve()),
        "profile_path": str(profile_path.resolve()),
        "resolution_path": str(resolution_path.resolve()),
        "resolution": resolution,
    }


def _build_resolution(
    ctx: AgentContext,
    profile: DatasetProfile,
    data_path: Path,
    output_dir: str,
) -> dict:
    """Compute all auto-filled config values (P2-T4, P2-T5, P2-T7)."""
    hspice_ok = is_hspice_available(ctx)

    temperature_enabled = profile.has_temperature_iv
    temperature_mode = "auto" if temperature_enabled else "off"

    hysteresis_enabled = profile.has_hysteresis
    hysteresis_dir = profile.suggested_hysteresis_dir or str(data_path)

    return {
        "data_dir": str(data_path),
        "hysteresis_data_dir": hysteresis_dir,
        "output_dir": output_dir,
        "hspice_exe": _read_template_hspice_path(ctx),
        "hspice_available": hspice_ok,
        "run_hspice": "yes" if hspice_ok else "no",
        "temperature_enabled": temperature_enabled,
        "temperature_mode": temperature_mode,
        "hysteresis_enabled": "yes" if hysteresis_enabled else "no",
        "static_ok": profile.static_ok,
        "cv_ok": profile.cv_ok,
        "has_temperature_iv": profile.has_temperature_iv,
        "has_hysteresis": profile.has_hysteresis,
        "iv_families": profile.iv_families,
        "total_files": profile.total_files,
        "issues": profile.issues,
        "template_config": ctx.agent_configs.get("DG_TFT_DEFAULT_CONFIG_TEMPLATE", DG_TFT_DEFAULT_CONFIG_TEMPLATE),
    }


def _write_ini(
    ctx: AgentContext,
    profile: DatasetProfile,
    data_path: Path,
    resolution: dict,
    out_path: Path,
) -> None:
    """Read template .ini, fill dynamic fields, write to out_path (P2-T6).

    Uses direct text replacement to avoid configparser formatting issues
    (tab indentation, comment stripping, case folding).
    """
    template_key = ctx.agent_configs.get("DG_TFT_DEFAULT_CONFIG_TEMPLATE", DG_TFT_DEFAULT_CONFIG_TEMPLATE)
    if not template_key:
        raise ValueError("DG_TFT_DEFAULT_CONFIG_TEMPLATE not set")

    template_path = Path(template_key)
    if not template_path.exists():
        raise FileNotFoundError(f"Config template not found: {template_path}")

    text = template_path.read_text(encoding="utf-8")

    replacements = {
        "data_dir": resolution["data_dir"],
        "hysteresis_data_dir": resolution["hysteresis_data_dir"],
        "output_dir": resolution["output_dir"],
        "hspice_exe": resolution["hspice_exe"],
    }

    lines = text.splitlines(keepends=True)
    current_section = ""

    for i, line in enumerate(lines):
        stripped = line.lstrip()
        if stripped.startswith("[") and stripped.rstrip().endswith("]"):
            current_section = stripped[1:].split("]")[0].strip()
            continue
        if "=" not in stripped or stripped.startswith("#") or stripped.startswith(";"):
            continue
        option = stripped.split("=", 1)[0].strip()

        if current_section == "paths" and option in replacements:
            indent = line[:len(line) - len(line.lstrip())]
            lines[i] = f"{indent}{option} = {replacements[option]}\n"
        elif current_section == "hysteresis" and option == "enabled":
            indent = line[:len(line) - len(line.lstrip())]
            lines[i] = f"{indent}enabled = {resolution['hysteresis_enabled']}\n"
        elif current_section == "temperature" and option == "mode":
            indent = line[:len(line) - len(line.lstrip())]
            lines[i] = f"{indent}mode = {resolution['temperature_mode']}\n"
        elif current_section == "deployment" and option == "run_hspice":
            indent = line[:len(line) - len(line.lstrip())]
            lines[i] = f"{indent}run_hspice = {resolution['run_hspice']}\n"

    out_path.write_text("".join(lines), encoding="utf-8")


def _read_template_hspice_path(ctx: AgentContext) -> str:
    import configparser

    template_key = ctx.agent_configs.get("DG_TFT_DEFAULT_CONFIG_TEMPLATE", DG_TFT_DEFAULT_CONFIG_TEMPLATE)
    if not template_key:
        return ""
    template_path = Path(template_key)
    if not template_path.exists():
        return ""
    cfg = configparser.ConfigParser()
    cfg.read(template_path, encoding="utf-8")
    return cfg.get("paths", "hspice_exe", fallback="")


def _apply_overrides(config_path: Path, overrides: dict[str, str]) -> None:
    """Apply section.key → value overrides by direct text manipulation.

    Uses line-by-line replacement to avoid configparser formatting issues
    (tab indentation, keyword lowercasing, comment stripping).
    """
    if not overrides:
        return
    invalid = []
    for key in overrides:
        if "." not in key:
            invalid.append(key)
    if invalid:
        raise ValueError(
            f"Overrides must use section.key format. "
            f"Invalid keys (missing section prefix): {invalid}"
        )

    lines = config_path.read_text(encoding="utf-8").splitlines(keepends=True)
    current_section = ""
    replaced = set()

    for i, line in enumerate(lines):
        stripped = line.lstrip()
        if stripped.startswith("[") and stripped.endswith("]\n") or stripped.startswith("[") and stripped.rstrip().endswith("]"):
            current_section = stripped[1:].split("]")[0].strip()
            continue
        if "=" not in stripped or stripped.startswith("#") or stripped.startswith(";"):
            continue
        option = stripped.split("=", 1)[0].strip()
        full_key = f"{current_section}.{option}"
        if full_key in overrides:
            indent = line[:len(line) - len(line.lstrip())]
            lines[i] = f"{indent}{option} = {overrides[full_key]}\n"
            replaced.add(full_key)

    # Warn about keys that didn't match any line
    missed = set(overrides.keys()) - replaced
    if missed:
        sys_log.warning("Overrides not found in config (no matching line): %s", sorted(missed))

    config_path.write_text("".join(lines), encoding="utf-8")
    sys_log.info("Applied %d config overrides to %s", len(replaced), config_path)
