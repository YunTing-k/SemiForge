from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[2]
V90_SLOW_REFERENCE_RMSE_DEC = 0.0033323642404866406
V90_EQUIVALENCE_ABS_TOL_DEC = 1.0e-4


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def same_normalization(left: dict, right: dict) -> bool:
    if left.keys() != right.keys():
        return False
    return all(
        np.allclose(
            np.asarray(left[key], dtype=float),
            np.asarray(right[key], dtype=float),
            rtol=0.0,
            atol=1.0e-12,
        )
        for key in left
    )


def select_rows(frame: pd.DataFrame, regime: str) -> pd.DataFrame:
    return frame.loc[frame["regime"].astype(str).eq(regime)].copy()


def select_prediction_rows(
    predictions: pd.DataFrame,
    metrics: pd.DataFrame,
) -> pd.DataFrame:
    keys = set(
        zip(metrics["curve_id"].astype(str), metrics["seg_id"].astype(int))
    )
    keep = [
        (str(curve_id), int(seg_id)) in keys
        for curve_id, seg_id in zip(
            predictions["curve_id"], predictions["seg_id"]
        )
    ]
    return predictions.loc[keep].copy()


def regime_policy(value: object, regime: str) -> str:
    if isinstance(value, dict):
        return str(value[regime])
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compose a fresh v95 hysteresis release from a validated fast bank "
            "and a separately retrained v90-method slow bank."
        )
    )
    parser.add_argument("--fast-bank", type=Path, required=True)
    parser.add_argument("--slow-bank", type=Path, required=True)
    parser.add_argument("--out-bank", type=Path, required=True)
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()

    fast = args.fast_bank.resolve()
    slow = args.slow_bank.resolve()
    out = args.out_bank.resolve()
    outputs_root = (ROOT / "outputs").resolve()
    if outputs_root not in out.parents:
        raise RuntimeError(f"Output must remain under {outputs_root}: {out}")
    if out.exists():
        if not args.replace:
            raise FileExistsError(out)
        shutil.rmtree(out)
    out.mkdir(parents=True)

    fast_summary = read_json(fast / "srh_top2_regime_bank_summary.json")
    slow_summary = read_json(slow / "srh_top2_regime_bank_summary.json")
    if not same_normalization(
        fast_summary["normalization"], slow_summary["normalization"]
    ):
        raise RuntimeError("Fast/slow normalization metadata are incompatible.")

    fast_metrics = select_rows(
        pd.read_csv(fast / "srh_top2_regime_bank_metrics.csv"), "fast"
    )
    slow_metrics = select_rows(
        pd.read_csv(slow / "srh_top2_regime_bank_metrics.csv"), "slow"
    )
    fast_predictions = select_prediction_rows(
        pd.read_csv(fast / "srh_top2_regime_bank_predictions.csv"),
        fast_metrics,
    )
    slow_predictions = select_prediction_rows(
        pd.read_csv(slow / "srh_top2_regime_bank_predictions.csv"),
        slow_metrics,
    )
    metrics = pd.concat([fast_metrics, slow_metrics], ignore_index=True)
    predictions = pd.concat(
        [fast_predictions, slow_predictions], ignore_index=True
    )

    formulas = {
        **{
            key: value
            for key, value in fast_summary["formulas"].items()
            if key.startswith("fast|")
        },
        **{
            key: value
            for key, value in slow_summary["formulas"].items()
            if key.startswith("slow|")
        },
    }
    groups = {
        **{
            key: value
            for key, value in fast_summary.get("groups", {}).items()
            if key.startswith("fast|")
        },
        **{
            key: value
            for key, value in slow_summary.get("groups", {}).items()
            if key.startswith("slow|")
        },
    }
    if len(formulas) != 8:
        raise RuntimeError(
            f"Expected 8 fast/slow family/process formula groups, got {len(formulas)}."
        )

    top2_values = [
        int(formula.get("symbolic_topk", 0))
        for group in formulas.values()
        for formula in group.values()
    ]
    endpoint_mean = float(
        metrics["top2_endpoint_drift_abs_error_dec"].mean()
    )
    endpoint_p95 = float(
        metrics["top2_endpoint_drift_abs_error_dec"].quantile(0.95)
    )
    summary = {
        "architecture": (
            "fresh hybrid-method dual-state SRH: optimized conservative fast "
            "banks plus v90-compatible independent-amplitude slow banks"
        ),
        "trained_regime": "all",
        "hidden_capture": fast_summary["hidden_capture"],
        "hidden_emission": fast_summary["hidden_emission"],
        "amp_parameterization": {
            "fast": regime_policy(
                fast_summary["amp_parameterization"], "fast"
            ),
            "slow": regime_policy(
                slow_summary["amp_parameterization"], "slow"
            ),
        },
        "normalization": fast_summary["normalization"],
        "formulas": formulas,
        "groups": groups,
        "teacher_rmse_dec_mean": float(metrics["teacher_rmse_dec"].mean()),
        "top2_rmse_dec_mean": float(metrics["top2_rmse_dec"].mean()),
        "student_teacher_rmse_dec_mean": float(
            metrics["student_teacher_rmse_dec"].mean()
        ),
        "student_teacher_rmse_dec_p90": float(
            metrics["student_teacher_rmse_dec"].quantile(0.9)
        ),
        "student_teacher_rmse_dec_max": float(
            metrics["student_teacher_rmse_dec"].max()
        ),
        "slow_teacher_rmse_dec_mean": float(
            slow_metrics["teacher_rmse_dec"].mean()
        ),
        "slow_top2_rmse_dec_mean": float(
            slow_metrics["top2_rmse_dec"].mean()
        ),
        "slow_student_teacher_rmse_dec_mean": float(
            slow_metrics["student_teacher_rmse_dec"].mean()
        ),
        "top2_symbolic_formula_count": len(top2_values),
        "top2_endpoint_drift_abs_error_dec_mean": endpoint_mean,
        "top2_endpoint_drift_abs_error_dec_p95": endpoint_p95,
    }
    checks = {
        "strict_top2_symbolic_topology": bool(
            top2_values and all(value == 2 for value in top2_values)
        ),
        "top2_rmse_mean": summary["top2_rmse_dec_mean"] <= 0.020,
        "student_teacher_rmse_mean": (
            summary["student_teacher_rmse_dec_mean"] <= 0.006
        ),
        "student_teacher_rmse_p90": (
            summary["student_teacher_rmse_dec_p90"] <= 0.012
        ),
        "endpoint_drift_error_mean": endpoint_mean <= 0.070,
        "endpoint_drift_error_p95": endpoint_p95 <= 0.230,
        "slow_equivalent_to_v90": (
            summary["slow_top2_rmse_dec_mean"]
            <= V90_SLOW_REFERENCE_RMSE_DEC + V90_EQUIVALENCE_ABS_TOL_DEC
        ),
    }
    summary["acceptance"] = {
        "pass": all(checks.values()),
        "checks": checks,
        "limits": {
            "top2_rmse_dec_mean": 0.020,
            "student_teacher_rmse_dec_mean": 0.006,
            "student_teacher_rmse_dec_p90": 0.012,
            "endpoint_drift_abs_error_dec_mean": 0.070,
            "endpoint_drift_abs_error_dec_p95": 0.230,
            "v90_slow_top2_rmse_dec_mean": V90_SLOW_REFERENCE_RMSE_DEC,
            "v90_equivalence_abs_tolerance_dec": V90_EQUIVALENCE_ABS_TOL_DEC,
        },
    }

    fast_checkpoint = torch.load(
        fast / "srh_top2_regime_bank_checkpoint.pt",
        map_location="cpu",
        weights_only=False,
    )
    slow_checkpoint = torch.load(
        slow / "srh_top2_regime_bank_checkpoint.pt",
        map_location="cpu",
        weights_only=False,
    )
    checkpoint = {
        **{
            key: value
            for key, value in fast_checkpoint.items()
            if key.startswith("fast|")
        },
        **{
            key: value
            for key, value in slow_checkpoint.items()
            if key.startswith("slow|")
        },
    }
    if len(checkpoint) != 8:
        raise RuntimeError(f"Expected 8 checkpoint groups, got {len(checkpoint)}.")

    metrics.to_csv(
        out / "srh_top2_regime_bank_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    predictions.to_csv(
        out / "srh_top2_regime_bank_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    (out / "srh_top2_regime_bank_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    torch.save(checkpoint, out / "srh_top2_regime_bank_checkpoint.pt")
    (out / "v95_hysteresis_composition_provenance.json").write_text(
        json.dumps(
            {
                "fast_bank": str(fast),
                "slow_bank": str(slow),
                "historical_checkpoint_reused": False,
                "historical_methodology_only": True,
                "selection": {
                    "fast": "fresh conservative-amplitude optimized training",
                    "slow": "fresh v90-compatible independent-amplitude training",
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    if not summary["acceptance"]["pass"]:
        raise RuntimeError("Composed hysteresis bank failed acceptance.")


if __name__ == "__main__":
    main()
