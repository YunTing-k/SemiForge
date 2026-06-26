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

    try:
        # 1) Artifact manifest (must run first so the other reports can reference it)
        manifest = _build_artifact_manifest(run_data, output_dir)
        with open(paths["manifest"], "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        sys_log.info("ARTIFACT_MANIFEST written: %s", paths["manifest"])
    except Exception:
        sys_log.exception("Failed to write ARTIFACT_MANIFEST")
        success = False

    try:
        # 2) JSON summary
        json_summary = _build_json_summary(run_data, manifest, prep_info)
        with open(paths["json"], "w", encoding="utf-8") as f:
            json.dump(json_summary, f, indent=2, ensure_ascii=False)
        sys_log.info("RUN_SUMMARY.json written: %s", paths["json"])
    except Exception:
        sys_log.exception("Failed to write RUN_SUMMARY.json")
        success = False

    try:
        # 3) Markdown summary
        md_text = _build_md_summary(run_data, manifest, prep_info)
        with open(paths["md"], "w", encoding="utf-8") as f:
            f.write(md_text)
        sys_log.info("RUN_SUMMARY.md written: %s", paths["md"])
    except Exception:
        sys_log.exception("Failed to write RUN_SUMMARY.md")
        success = False

    return success


# ─── Manifest ───────────────────────────────────────────────────────

def _build_artifact_manifest(
    run_data: dict[str, Any],
    output_dir: str,
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
                items.append({
                    "stage": stage_name,
                    "artifact_key": key,
                    "filename": ARTIFACT_FILENAMES.get(key, key),
                    "path": str(full_path),
                    "subdir": subdir,
                    "exists": exists,
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

    report_paths = get_report_paths(output_dir)
    for report_key, report_path in report_paths.items():
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

def _build_json_summary(
    run_data: dict[str, Any],
    manifest: dict[str, Any],
    prep_info: dict[str, Any] | None,
) -> dict[str, Any]:
    """Build RUN_SUMMARY.json content."""
    status = run_data.get("status", "UNKNOWN")
    stage_results = run_data.get("stage_results", {})

    stages_list: list[dict[str, Any]] = []
    for sn in ["check-data", "fit-physical", "train-static", "train-hysteresis"]:
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

    summary: dict[str, Any] = {
        "run_id": run_data.get("run_id"),
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
        "stages": stages_list,
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
    if isinstance(extra, dict) and "diagnosis" in extra:
        summary["diagnosis"] = extra["diagnosis"]

    return summary


# ─── Markdown summary ───────────────────────────────────────────────

def _build_md_summary(
    run_data: dict[str, Any],
    manifest: dict[str, Any],
    prep_info: dict[str, Any] | None,
) -> str:
    """Build RUN_SUMMARY.md content."""
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
    lines.append("")

    # ── Stage results ──
    lines.append("## 2. Stage Results")
    lines.append("")
    stage_results = run_data.get("stage_results", {})
    for sn in ["check-data", "fit-physical", "train-static", "train-hysteresis"]:
        sr = stage_results.get(sn)
        if sr is None:
            continue
        if isinstance(sr, dict):
            display = STAGE_DISPLAY_NAMES.get(sn, sn)
            if sr.get("skipped"):
                lines.append(f"### {display} — [SKIPPED] {sr.get('skip_reason', '')}")
            elif sr.get("ok"):
                lines.append(f"### {display} — [PASS]")
            else:
                lines.append(f"### {display} — [FAIL]")
                if sr.get("error"):
                    lines.append(f"")
                    lines.append(f"```text")
                    lines.append(f"{sr['error']}")
                    lines.append(f"```")
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

    # ── Diagnosis ──
    extra = run_data.get("extra", {})
    if isinstance(extra, dict) and "diagnosis" in extra:
        diag = extra["diagnosis"]
        lines.append("## 4. Diagnosis")
        lines.append("")
        lines.append(f"- **Category**: {diag.get('category', 'UNKNOWN')}")
        lines.append(f"- **Confidence**: {diag.get('confidence', 0):.0%}")
        evidence = diag.get("evidence", [])
        if evidence:
            lines.append(f"- **Evidence**: {', '.join(evidence[:5])}")
        suggestion = diag.get("suggestion", "")
        if suggestion:
            lines.append(f"- **Suggestion**: {suggestion}")
        subagent_text = diag.get("subagent_result", "")
        if subagent_text:
            lines.append("")
            lines.append("### AI Diagnosis")
            lines.append("")
            for line in subagent_text.strip().splitlines():
                lines.append(f"> {line}")
        lines.append("")

    # ── Warnings / Errors ──
    warnings = run_data.get("warnings", [])
    errors = run_data.get("errors", [])
    issue_section = 5 if (isinstance(extra, dict) and "diagnosis" in extra) else 4
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
