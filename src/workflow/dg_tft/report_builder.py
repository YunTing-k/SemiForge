# -*- coding: utf-8 -*-
"""
Header information
---------
Shanghai Jiao Tong University, School of Integrated Circuits, SMIL Lab
Author: Yu Huang
Create Date: 2026.6.25
Description: Report builder — generates RUN_SUMMARY.md, RUN_SUMMARY.json,
              and ARTIFACT_MANIFEST.json from workflow run data.

Revision:
---------
2026.6.25      Yu Huang      1.0      Initial implementation
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from .artifacts import (
    ARTIFACT_FILENAMES,
    ARTIFACT_KEY_RUN_SUMMARY,
    ARTIFACT_KEY_ARTIFACT_MANIFEST,
    RUN_SUMMARY_MD_NAME,
    RUN_SUMMARY_JSON_NAME,
    ARTIFACT_MANIFEST_NAME,
    STAGE_DISPLAY_NAMES,
    STAGE_ARTIFACTS,
    artifact_path,
    get_report_paths,
)

sys_log = logging.getLogger("logger")


def build_and_write_reports(
    run_data: dict[str, Any],
    output_dir: str,
    *,
    prep_info: dict[str, Any] | None = None,
) -> bool:
    """Generate all three report files from a completed (or failed) workflow run.

    Args:
        run_data: Full WorkflowRun.to_dict() output.
        output_dir: Root output directory for this run.
        prep_info: Optional data preparation summary.

    Returns:
        True if all files were written successfully.
    """
    paths = get_report_paths(output_dir)
    success = True

    # Build artifact-only manifest (no report files yet — they don't exist).
    # The full manifest with report file sizes is written last.
    artifact_manifest = _build_artifact_manifest(run_data, output_dir, include_reports=False)

    # Compute stage results with proper error/warning/skip fields for RUN_SUMMARY
    stage_results_data = _prepare_stage_results_for_report(run_data)

    try:
        json_summary = _build_json_summary(run_data, artifact_manifest, prep_info, stage_results_data)
        with open(paths["json"], "w", encoding="utf-8") as f:
            json.dump(json_summary, f, indent=2, ensure_ascii=False)
        sys_log.info("RUN_SUMMARY.json written: %s", paths["json"])
    except Exception:
        sys_log.exception("Failed to write RUN_SUMMARY.json")
        success = False

    try:
        md_text = _build_md_summary(run_data, artifact_manifest, prep_info, stage_results_data)
        with open(paths["md"], "w", encoding="utf-8") as f:
            f.write(md_text)
        sys_log.info("RUN_SUMMARY.md written: %s", paths["md"])
    except Exception:
        sys_log.exception("Failed to write RUN_SUMMARY.md")
        success = False

    try:
        # Full manifest — now that json/md exist on disk, their sizes are correct.
        # Exclude the manifest itself (it can't stat itself before being written).
        full_manifest = _build_artifact_manifest(run_data, output_dir, include_reports=True, include_self=False)
        with open(paths["manifest"], "w", encoding="utf-8") as f:
            json.dump(full_manifest, f, indent=2, ensure_ascii=False)
        sys_log.info("ARTIFACT_MANIFEST written: %s", paths["manifest"])
    except Exception:
        sys_log.exception("Failed to write ARTIFACT_MANIFEST")
        success = False

    return success


# ─── Manifest ───────────────────────────────────────────────────────

def _build_artifact_manifest(
    run_data: dict[str, Any],
    output_dir: str,
    *,
    include_reports: bool = True,
    include_self: bool = True,
) -> dict[str, Any]:
    """Build ARTIFACT_MANIFEST.json content."""

    stage_results: dict[str, dict[str, Any]] = run_data.get("stage_results", {})
    attached_artifacts: dict[str, str] = run_data.get("artifacts", {})

    items: list[dict[str, Any]] = []

    for stage_name in [
        "check-data", "fit-physical",
        "train-static", "train-hysteresis",
    ]:
        entry = STAGE_ARTIFACTS.get(stage_name)
        if entry is None:
            continue

        for subdir, keys in entry.items():
            for key in keys:
                full_path = artifact_path(output_dir, subdir, key)
                exists = full_path.exists()
                invalidated = stage_name in run_data.get("extra", {}).get("invalidated_stages", [])
                items.append({
                    "stage": stage_name,
                    "artifact_key": key,
                    "filename": ARTIFACT_FILENAMES.get(key, key),
                    "path": str(full_path),
                    "subdir": subdir,
                    "exists": exists,
                    "current": not invalidated,
                    "size_bytes": full_path.stat().st_size if exists else 0,
                })

    for key, path in attached_artifacts.items():
        p = Path(path)
        exists = p.exists()
        items.append({
            "stage": "attached",
            "artifact_key": key,
            "filename": p.name,
            "path": str(p.resolve()),
            "subdir": "",
            "exists": exists,
            "size_bytes": p.stat().st_size if exists else 0,
        })

    if include_reports:
        report_paths = get_report_paths(output_dir)
        for report_key, report_path in report_paths.items():
            if report_key == "manifest" and not include_self:
                continue
            p = Path(report_path)
            exists = p.exists()
            items.append({
                "stage": "report",
                "artifact_key": ARTIFACT_KEY_RUN_SUMMARY if report_key != "manifest" else ARTIFACT_KEY_ARTIFACT_MANIFEST,
                "filename": p.name,
                "path": str(p),
                "subdir": "",
                "exists": exists,
                "size_bytes": p.stat().st_size if exists else 0,
            })

    return {
        "run_id": run_data.get("run_id"),
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "output_dir": str(Path(output_dir).resolve()),
        "total_items": len(items),
        "missing_items": sum(1 for it in items if not it["exists"]),
        "items": items,
    }


# ─── JSON summary ───────────────────────────────────────────────────

def _prepare_stage_results_for_report(
    run_data: dict[str, Any],
) -> list[dict[str, Any]]:
    """Normalize stage results for use in JSON/MD report builders."""
    stage_results = run_data.get("stage_results", {})
    stages_list: list[dict[str, Any]] = []
    for sn in ["data-prep", "verifier", "check-data", "fit-physical", "train-static", "train-hysteresis"]:
        sr = stage_results.get(sn)
        if sr is None:
            continue
        if isinstance(sr, dict):
            stages_list.append({
                "stage": sn,
                "display_name": STAGE_DISPLAY_NAMES.get(sn, sn),
                "ok": sr.get("ok", False),
                "skipped": sr.get("skipped", False),
                "skip_reason": sr.get("skip_reason", ""),
                "error": sr.get("error"),
                "warnings_count": len(sr.get("warnings", [])),
                "metrics": sr.get("metrics", {}),
            })
    return stages_list


def _model_scope(run_data: dict[str, Any]) -> dict[str, Any]:
    """Report the dataset contract without equating absent CV with zero error."""
    from .schema import DatasetProfile

    profile = run_data.get("extra", {}).get("dataset_profile")
    if not isinstance(profile, dict):
        return {"task_mode": "unknown"}
    cv = profile.get("raw_validation", {}).get("cv", {})
    present = profile.get("cv_present", cv.get("exists", profile.get("cv_ok", False)))
    mode = DatasetProfile(cv_present=present, cv_ok=profile.get("cv_ok", False)).task_mode
    return {
        "task_mode": mode,
        "cv_status": "not_applicable" if mode == "iv_only" else "see_stage_results",
        "limitation": ("No measured CV: charge/capacitance defaults are uncalibrated; AC/transient accuracy is unvalidated."
                       if mode == "iv_only" else ""),
    }


def _build_json_summary(
    run_data: dict[str, Any],
    manifest: dict[str, Any],
    prep_info: dict[str, Any] | None,
    stage_results_data: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build RUN_SUMMARY.json content."""
    if stage_results_data is None:
        stage_results_data = _prepare_stage_results_for_report(run_data)
    status = run_data.get("status", "UNKNOWN")

    summary: dict[str, Any] = {
        "run_id": run_data.get("run_id"),
        "model_scope": _model_scope(run_data),
        "termination": run_data.get("extra", {}).get("termination"),
        "status": status,
        "created_at": run_data.get("created_at"),
        "updated_at": run_data.get("updated_at"),
        "dataset_dir": run_data.get("dataset_dir"),
        "output_root": run_data.get("output_root"),
        "prep_id": run_data.get("prep_id"),
        "hysteresis_enabled": run_data.get("hysteresis_enabled", False),
        "temperature_enabled": run_data.get("temperature_enabled", False),
        "warnings_count": len(run_data.get("warnings", [])),
        "errors_count": len(run_data.get("errors", [])),
        "stages": stage_results_data,
        "artifacts_summary": {
            "total": manifest.get("total_items", 0),
            "missing": manifest.get("missing_items", 0),
        },
    }

    if run_data.get("metrics"):
        summary["metrics"] = dict(run_data["metrics"])

    if prep_info:
        summary["preparation"] = {
            "ok": prep_info.get("ok", False),
            "issues": prep_info.get("issues", []),
        }

    extra = run_data.get("extra", {})
    if isinstance(extra, dict) and "quality_retry_history" in extra:
        summary["quality"] = {
            "retries": len(extra["quality_retry_history"]),
            "history": extra["quality_retry_history"],
        }

    return summary


# ─── Markdown summary ───────────────────────────────────────────────

def _build_md_summary(
    run_data: dict[str, Any],
    manifest: dict[str, Any],
    prep_info: dict[str, Any] | None,
    stage_results_data: list[dict[str, Any]] | None = None,
) -> str:
    """Build RUN_SUMMARY.md content."""
    if stage_results_data is None:
        stage_results_data = _prepare_stage_results_for_report(run_data)
    lines: list[str] = []

    lines.append(f"# DG-TFT Modeling Run #{run_data.get('run_id', '?')}")
    lines.append("")
    lines.append(f"**Status**: {run_data.get('status', 'UNKNOWN')}")
    lines.append(f"**Created**: {run_data.get('created_at', '—')}")
    lines.append(f"**Updated**: {run_data.get('updated_at', '—')}")
    lines.append("")

    # ── Input ──
    lines.append("## 1. Input")
    lines.append(f"- **Dataset directory**: `{run_data.get('dataset_dir', '—')}`")
    lines.append(f"- **Output root**: `{run_data.get('output_root', '—')}`")
    if run_data.get("prep_id") is not None:
        lines.append(f"- **Preparation ID**: {run_data['prep_id']}")
    if prep_info:
        lines.append(f"- **Preparation OK**: {'yes' if prep_info.get('ok') else 'no'}")
        issues = prep_info.get("issues", [])
        if issues:
            for iss in issues:
                lines.append(f"  - [WARN] {iss}")
    lines.append(f"- **Hysteresis enabled**: {run_data.get('hysteresis_enabled', False)}")
    lines.append(f"- **Temperature enabled**: {run_data.get('temperature_enabled', False)}")
    termination = run_data.get("extra", {}).get("termination")
    if termination:
        lines.append("")
        lines.append("## Termination")
        for key in ("reason", "evidence", "required_information"):
            if termination.get(key):
                lines.append(f"- **{key.replace('_', ' ').title()}**: {termination[key]}")

    scope = _model_scope(run_data)
    lines.append(f"- **Static modeling scope**: {scope['task_mode']}")
    if scope.get("limitation"):
        lines.append(f"- **CV: NOT APPLICABLE**. {scope['limitation']}")
    lines.append("")

    # ── Stage results ──
    lines.append("## 2. Stage Results")
    lines.append("")
    for sr in stage_results_data:
        display = sr.get("display_name", sr["stage"])
        if sr.get("skipped"):
            lines.append(f"### {display} — [SKIPPED] {sr.get('skip_reason', '')}")
        elif sr.get("ok"):
            lines.append(f"### {display} — [PASS]")
        else:
            lines.append(f"### {display} — [FAIL]")
            if sr.get("error"):
                lines.append("")
                lines.append("```text")
                lines.append(sr["error"])
                lines.append("```")
        lines.append("")
        metrics = sr.get("metrics", {})
        if metrics:
            for k, v in metrics.items():
                lines.append(f"- **{k}**: {v}")
            lines.append("")

    # ── Artifacts ──
    lines.append("## 3. Artifacts")
    lines.append("")
    items = manifest.get("items", [])
    stage_items: dict[str, list[dict[str, Any]]] = {}
    for it in items:
        st = it.get("stage", "other")
        stage_items.setdefault(st, []).append(it)

    for sn in [    "check-data", "fit-physical", "train-static", "train-hysteresis", "report", "attached"]:
        si = stage_items.get(sn)
        if not si:
            continue
        display = STAGE_DISPLAY_NAMES.get(sn, sn.title())
        lines.append(f"### {display}")
        lines.append("")
        for it in si:
            icon = "✓" if it["exists"] else "✗"
            size_kb = it.get("size_bytes", 0) / 1024
            lines.append(f"- {icon} `{it['filename']}` ({size_kb:.1f} KB)")
        lines.append("")

    # ── Quality (Phase 7) ──
    extra = run_data.get("extra", {})
    has_quality = isinstance(extra, dict) and "quality_retry_history" in extra
    if has_quality:
        q_section = 4
        q_history = extra["quality_retry_history"]
        lines.append(f"## {q_section}. Quality Gate")
        lines.append("")
        lines.append(f"- **Retries attempted**: {len(q_history)}")
        passes = sum(1 for r in q_history if r.get("quality_pass"))
        lines.append(f"- **Retries passed**: {passes}")
        lines.append("")
        lines.append("### Retry Log")
        lines.append("")
        lines.append("| Attempt | Pass | Adjustments | Metrics | Error |")
        lines.append("|---------|------|-------------|---------|-------|")
        for entry in q_history:
            att = entry.get("attempt", "?")
            passed = "✓" if entry.get("quality_pass") else "✗"
            if not entry.get("ok"):
                passed = "✗ (crashed)"
            adj_list = entry.get("adjustments", {})
            adj_str = ", ".join(f"{k}={v}" for k, v in adj_list.items()) if adj_list else "—"
            met_list = entry.get("metrics", {})
            met_str = ", ".join(f"{k}: {v:.4f}".rstrip("0").rstrip(".") for k, v in met_list.items()) if met_list else "—"
            err_str = entry.get("error") or "—"
            if len(err_str) > 60:
                err_str = err_str[:57] + "..."
            lines.append(f"| {att} | {passed} | {adj_str} | {met_str} | {err_str} |")
        lines.append("")

        # Stage metrics section for metrics in stage_results
        lines.append("### Final Metrics")
        lines.append("")
        stage_results = run_data.get("stage_results", {})
        for sn in ["fit-physical", "train-static", "train-hysteresis"]:
            sr = stage_results.get(sn)
            if not isinstance(sr, dict):
                continue
            met = sr.get("metrics", {})
            if not met:
                continue
            display = STAGE_DISPLAY_NAMES.get(sn, sn)
            lines.append(f"**{display}**:")
            for k, v in met.items():
                if v != v:
                    lines.append(f"- {k}: N/A")
                else:
                    lines.append(f"- {k}: {v:.6f}")
            lines.append("")

    # ── Warnings / Errors ──
    warnings = run_data.get("warnings", [])
    errors = run_data.get("errors", [])
    issue_base = 4
    if has_quality:
        issue_base += 1
    issue_section = issue_base
    if warnings or errors:
        lines.append(f"## {issue_section}. Issues")
        lines.append("")
        if errors:
            lines.append("### Errors")
            for e in errors:
                lines.append(f"- {e}")
            lines.append("")
        if warnings:
            lines.append("### Warnings")
            for w in warnings:
                lines.append(f"- {w}")
            lines.append("")

    lines.append("---")
    lines.append(f"*Report generated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")

    return "\n".join(lines) + "\n"
