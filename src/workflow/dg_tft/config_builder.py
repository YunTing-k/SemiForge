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

import configparser
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from .dataset_inspector import is_hspice_available
from .schema import DatasetProfile

if TYPE_CHECKING:
    from src.context.agent_context import AgentContext

sys_log = logging.getLogger("logger")


def build_runtime_config(
    ctx: AgentContext,
    profile: DatasetProfile,
    dataset_dir: str,
    *,
    output_root: str | None = None,
    run_id: int | None = None,
    work_dir: str | None = None,
) -> dict:
    """Build and persist a run-specific DG-TFT .ini configuration (P2-T6, P2-T7).

    Reads the template .ini, fills in dynamic fields based on the DatasetProfile,
    and writes the runtime config to disk.

    Args:
        ctx: Agent context (for default config keys).
        profile: DatasetProfile from dataset_inspector.
        dataset_dir: Absolute path to the dataset being modeled.
        output_root: Override for output root directory.
        run_id: Optional run ID for generating a unique output subdirectory.
        work_dir: If given, write config files inside this directory.
                  Otherwise uses output_root/run_<run_id>/.

    Returns:
        dict with keys:
          - config_path: str — path to the generated runtime .ini
          - profile_path: str — path to the saved dataset_profile.json
          - resolution: dict — config_resolution metadata
    """
    data_path = Path(dataset_dir).resolve()
    run_tag = f"run_{run_id:04d}" if run_id is not None else "run_0001"

    default_output_root = ctx.agent_configs.get("DG_TFT_DEFAULT_OUTPUT_ROOT", "./outputs/dg_tft_runs")
    out_root = Path(output_root or default_output_root).resolve()

    if work_dir:
        config_dir = Path(work_dir).resolve()
    else:
        config_dir = out_root / run_tag

    config_dir.mkdir(parents=True, exist_ok=True)

    resolution = _build_resolution(ctx, profile, data_path, out_root, run_tag)

    runtime_config_path = config_dir / "model_config.ini"
    _write_ini(ctx, profile, data_path, resolution, runtime_config_path)
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
    out_root: Path,
    run_tag: str,
) -> dict:
    """Compute all auto-filled config values (P2-T4, P2-T5, P2-T7)."""
    hspice_ok = is_hspice_available(ctx)

    temperature_enabled = profile.has_temperature_iv
    temperature_mode = "auto" if temperature_enabled else "off"

    hysteresis_enabled = profile.has_hysteresis
    hysteresis_dir = profile.suggested_hysteresis_dir or str(data_path)

    output_dir = str(out_root / run_tag)

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
        "template_config": ctx.agent_configs.get("DG_TFT_DEFAULT_CONFIG_TEMPLATE", ""),
    }


def _write_ini(
    ctx: AgentContext,
    profile: DatasetProfile,
    data_path: Path,
    resolution: dict,
    out_path: Path,
) -> None:
    """Read template .ini, fill dynamic fields, write to out_path (P2-T6)."""
    template_key = ctx.agent_configs.get("DG_TFT_DEFAULT_CONFIG_TEMPLATE", "")
    if not template_key:
        raise ValueError("DG_TFT_DEFAULT_CONFIG_TEMPLATE not set in agent_configs.json")

    template_path = Path(template_key)
    if not template_path.exists():
        raise FileNotFoundError(f"Config template not found: {template_path}")

    cfg = configparser.ConfigParser()
    cfg.read(template_path, encoding="utf-8")

    cfg.set("paths", "data_dir", str(data_path))
    cfg.set("paths", "hysteresis_data_dir", resolution["hysteresis_data_dir"])
    cfg.set("paths", "output_dir", resolution["output_dir"])
    cfg.set("paths", "hspice_exe", resolution["hspice_exe"])

    cfg.set("hysteresis", "enabled", resolution["hysteresis_enabled"])
    cfg.set("temperature", "mode", resolution["temperature_mode"])
    cfg.set("deployment", "run_hspice", resolution["run_hspice"])

    with open(out_path, "w", encoding="utf-8") as f:
        cfg.write(f)


def _read_template_hspice_path(ctx: AgentContext) -> str:
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
