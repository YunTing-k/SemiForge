from __future__ import annotations

from pathlib import Path
import json

import pandas as pd


ROOT = Path(__file__).resolve().parent
BASE = ROOT / "results_long_timescale" / "srh_top2_regime_banks"
OUT = BASE / "combined"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fast_metrics = pd.read_csv(BASE / "srh_top2_regime_bank_metrics.csv")
    fast_pred = pd.read_csv(BASE / "srh_top2_regime_bank_predictions.csv")
    slow_metrics = pd.read_csv(BASE / "slow_only" / "srh_top2_regime_bank_metrics.csv")
    slow_pred = pd.read_csv(BASE / "slow_only" / "srh_top2_regime_bank_predictions.csv")

    fast_metrics = fast_metrics[fast_metrics["regime"] == "fast"].copy()
    fast_keys = set(zip(fast_metrics["curve_id"], fast_metrics["seg_id"].astype(int)))
    fast_pred = fast_pred[
        [
            (curve_id, int(seg_id)) in fast_keys
            for curve_id, seg_id in zip(fast_pred["curve_id"], fast_pred["seg_id"])
        ]
    ].copy()

    legacy_path = ROOT / "results_long_timescale" / "srh_waveform_refined" / "srh_waveform_refinement_metrics.csv"
    if legacy_path.exists():
        legacy = pd.read_csv(legacy_path)
        existing_cols = set(fast_metrics.columns) | set(slow_metrics.columns)
        metadata_cols = [
            c
            for c in legacy.columns
            if c in {"curve_id", "seg_id"} or c not in existing_cols
            if c not in {
                "teacher_rmse_dec",
                "top1_rmse_dec",
                "top2_rmse_dec",
                "student_teacher_rmse_dec",
                "top1_dynamic_shape_rmse_norm",
                "top2_dynamic_shape_rmse_norm",
                "top1_drift_dec",
                "top2_drift_dec",
                "top1_endpoint_drift_abs_error_dec",
                "top2_endpoint_drift_abs_error_dec",
            }
        ]
        metadata = legacy[metadata_cols].drop_duplicates(["curve_id", "seg_id"])
        fast_metrics = fast_metrics.merge(metadata, on=["curve_id", "seg_id"], how="left")
        slow_metrics = slow_metrics.merge(metadata, on=["curve_id", "seg_id"], how="left")

    metrics = pd.concat([fast_metrics, slow_metrics], ignore_index=True, sort=False)
    predictions = pd.concat([fast_pred, slow_pred], ignore_index=True, sort=False)
    metrics.to_csv(OUT / "srh_top2_regime_bank_metrics.csv", index=False, encoding="utf-8-sig")
    predictions.to_csv(OUT / "srh_top2_regime_bank_predictions.csv", index=False, encoding="utf-8-sig")

    summary = {
        "architecture": "regime-aware dual-state SRH: fast us-ms bank plus slow seconds bank",
        "n_fast": int((metrics["regime"] == "fast").sum()),
        "n_slow": int((metrics["regime"] == "slow").sum()),
        "teacher_rmse_dec_mean": float(metrics["teacher_rmse_dec"].mean()),
        "top2_rmse_dec_mean": float(metrics["top2_rmse_dec"].mean()),
        "student_teacher_rmse_dec_mean": float(metrics["student_teacher_rmse_dec"].mean()),
        "student_teacher_rmse_dec_p90": float(metrics["student_teacher_rmse_dec"].quantile(0.9)),
        "slow_student_teacher_rmse_dec_mean": float(
            metrics.loc[metrics["regime"] == "slow", "student_teacher_rmse_dec"].mean()
        ),
        "slow_student_teacher_rmse_dec_max": float(
            metrics.loc[metrics["regime"] == "slow", "student_teacher_rmse_dec"].max()
        ),
    }
    (OUT / "srh_top2_regime_bank_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
