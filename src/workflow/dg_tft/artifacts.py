# -*- coding: utf-8 -*-
"""
Header information
---------
Shanghai Jiao Tong University, School of Integrated Circuits, SMIL Lab
Author: Yu Huang
Create Date: 2026.6.25
Description: Artifact key definitions, naming conventions, and path helpers
              for the DG-TFT automated modeling pipeline.

Revision:
---------
2026.6.25      Yu Huang      1.0      Initial implementation
"""
from __future__ import annotations

from pathlib import Path

# ── Report file names ──────────────────────────────────────────────
RUN_SUMMARY_MD_NAME: str = "RUN_SUMMARY.md"
RUN_SUMMARY_JSON_NAME: str = "RUN_SUMMARY.json"
ARTIFACT_MANIFEST_NAME: str = "ARTIFACT_MANIFEST.json"

# ── Artifact keys by stage ─────────────────────────────────────────
ARTIFACT_KEY_DATASET_VALIDATION: str = "dataset_validation"
ARTIFACT_KEY_FITTED_PARAMS: str = "fitted_params"
ARTIFACT_KEY_PKAN_MODEL: str = "pkan_model"
ARTIFACT_KEY_STATIC_MODEL_CKPT: str = "static_model_ckpt"
ARTIFACT_KEY_HYSTERESIS_VA: str = "hysteresis_va"
ARTIFACT_KEY_HYSTERESIS_INC: str = "hysteresis_inc"
ARTIFACT_KEY_RUN_SUMMARY: str = "run_summary"
ARTIFACT_KEY_ARTIFACT_MANIFEST: str = "artifact_manifest"

# ── Actual artifact filenames (with extensions) ────────────────────
ARTIFACT_FILENAMES: dict[str, str] = {
    ARTIFACT_KEY_DATASET_VALIDATION: "dataset_validation.json",
    ARTIFACT_KEY_FITTED_PARAMS: "fitted_params.json",
    ARTIFACT_KEY_PKAN_MODEL: "pkan_torch_model.pt",
    ARTIFACT_KEY_STATIC_MODEL_CKPT: "v93_safe_residual_state.pt",
    ARTIFACT_KEY_HYSTERESIS_VA: "dg_tft_dynamic_srh.va",
    ARTIFACT_KEY_HYSTERESIS_INC: "dg_tft_dynamic_srh.inc",
    ARTIFACT_KEY_RUN_SUMMARY: RUN_SUMMARY_MD_NAME,
    ARTIFACT_KEY_ARTIFACT_MANIFEST: ARTIFACT_MANIFEST_NAME,
}

# Stage -> { subdir -> [artifact keys] }
STAGE_ARTIFACTS: dict[str, dict[str, list[str]]] = {
    "check-data": {
        "": [ARTIFACT_KEY_DATASET_VALIDATION],
    },
    "fit-physical": {
        "01_physical_kernel": [ARTIFACT_KEY_FITTED_PARAMS],
    },
    "train-static": {
        "03_full_model": [ARTIFACT_KEY_STATIC_MODEL_CKPT],
    },
    "train-hysteresis": {
        "04_hysteresis/srh_top2_regime_banks/combined": [ARTIFACT_KEY_HYSTERESIS_VA, ARTIFACT_KEY_HYSTERESIS_INC],
    },
}

# Friendly display names for each stage
STAGE_DISPLAY_NAMES: dict[str, str] = {
    "data-prep": "Prepare Dataset",
    "verifier": "Verify Dataset",
    "check-data": "Audit Dataset",
    "fit-physical": "Fit Physical Kernel",
    "train-static": "Train Static Model",
    "train-hysteresis": "Train Hysteresis",
    "deployment-validate": "Deployment Validation",
    "final-report": "Final Report",
}


def artifact_path(output_dir: str, subdir: str, artifact_key: str) -> Path:
    """Return the absolute path for an artifact by its logical key.

    Args:
        output_dir: Root output directory (e.g., session/<uuid>/runs/run_0001).
        subdir: Stage subdirectory (e.g., '01_physical_kernel') or ''.
        artifact_key: Logical artifact key (e.g., 'fitted_params').

    Returns:
        Resolved absolute Path to the artifact file.
    """
    filename = ARTIFACT_FILENAMES.get(artifact_key, artifact_key)
    base = Path(output_dir)
    if subdir:
        base = base / subdir
    return (base / filename).resolve()


def verify_stage_artifacts(output_dir: str, stage_name: str) -> list[str]:
    """Return list of missing artifact paths for a stage. Empty = all present.

    Uses the logical artifact keys and their mapped filenames.
    """
    entry = STAGE_ARTIFACTS.get(stage_name)
    if entry is None:
        return []

    missing: list[str] = []
    for subdir, keys in entry.items():
        for key in keys:
            full = artifact_path(output_dir, subdir, key)
            if not full.exists():
                missing.append(str(full))
    return missing


def get_report_paths(output_dir: str) -> dict[str, str]:
    """Return the absolute paths for the three report files.

    Returns:
        dict with keys: 'md', 'json', 'manifest'
    """
    base = Path(output_dir).resolve()
    return {
        "md": str(base / RUN_SUMMARY_MD_NAME),
        "json": str(base / RUN_SUMMARY_JSON_NAME),
        "manifest": str(base / ARTIFACT_MANIFEST_NAME),
    }
