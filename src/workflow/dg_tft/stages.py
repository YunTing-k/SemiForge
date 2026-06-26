# -*- coding: utf-8 -*-
"""
Header information
---------
Shanghai Jiao Tong University, School of Integrated Circuits, SMIL Lab
Author: Yu Huang
Create Date: 2026.6.25
Description: Standard DG-TFT workflow stage definitions and ordering.

Revision:
---------
2026.6.25      Yu Huang      1.0      Initial implementation
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .schema import StageName, DatasetProfile

if TYPE_CHECKING:
    pass


@dataclass
class StageDef:
    name: str
    action: str
    required: bool
    depends_on: list[str] = field(default_factory=list)
    expected_output_subdir: str = ""
    key_artifacts: list[str] = field(default_factory=list)
    timeout: int | None = None
    extra_args: list[str] = field(default_factory=list)
    skip_if_profile_false: str = ""


_CORE_STAGES: list[StageDef] = [
    StageDef(
        name=StageName.CHECK_DATA,
        action="check-data",
        required=True,
        expected_output_subdir="",
        key_artifacts=["dataset_validation.json"],
        timeout=600,
    ),
    StageDef(
        name=StageName.FIT_PHYSICAL,
        action="fit-physical",
        required=True,
        depends_on=[StageName.CHECK_DATA],
        expected_output_subdir="01_physical_kernel",
        key_artifacts=["fitted_params.json"],
        timeout=1800,
    ),
    StageDef(
        name=StageName.TRAIN_STATIC,
        action="train-static",
        required=True,
        depends_on=[StageName.FIT_PHYSICAL],
        expected_output_subdir="03_full_model",
        key_artifacts=["v90_safe_residual_state.pt", "v90_veriloga_residual_include.va"],
        timeout=7200,
        extra_args=["--print-every", "1"],
    ),
    StageDef(
        name=StageName.TRAIN_HYSTERESIS,
        action="train-hysteresis",
        required=False,
        depends_on=[StageName.TRAIN_STATIC],
        expected_output_subdir="04_hysteresis",
        key_artifacts=["srh_top2_regime_banks/combined/dg_tft_dynamic_srh.va", "srh_top2_regime_banks/combined/dg_tft_dynamic_srh.inc"],
        timeout=7200,
        skip_if_profile_false="has_hysteresis",
    ),
    StageDef(
        name=StageName.FINAL_REPORT,
        action="final-report",
        required=False,
        depends_on=[StageName.TRAIN_HYSTERESIS],
        expected_output_subdir="",
        key_artifacts=["RUN_SUMMARY.md"],
        timeout=600,
    ),
]


def get_stages(profile: DatasetProfile | None = None) -> list[StageDef]:
    """Return the ordered pipeline stage list with skip conditions evaluated.

    Args:
        profile: Optional DatasetProfile; when provided, skip_if_profile_false
                 is checked against profile fields and disabled stages are
                 filtered from the returned list.

    Returns:
        Ordered list of active StageDef objects.
    """
    if profile is None:
        return list(_CORE_STAGES)

    result: list[StageDef] = []
    for sd in _CORE_STAGES:
        if sd.skip_if_profile_false:
            attr = getattr(profile, sd.skip_if_profile_false, None)
            if attr is not None and not attr:
                continue
        result.append(sd)
    return result


def get_stage_by_name(name: str) -> StageDef | None:
    for sd in _CORE_STAGES:
        if sd.name == name:
            return sd
    return None
