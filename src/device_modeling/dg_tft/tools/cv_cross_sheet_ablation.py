from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def rms_log(pred: np.ndarray, measured: np.ndarray) -> float:
    floor = 1.0e-18
    error = np.log10(np.maximum(pred, floor)) - np.log10(
        np.maximum(measured, floor)
    )
    return float(np.sqrt(np.mean(error * error)))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Leave-one-CV-sheet-out validation of residual scale selection."
    )
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--deployed-scale", type=float, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    frame = pd.read_csv(args.predictions)
    base = frame["cap_base_pkan_F"].to_numpy(float)
    deployed = frame["cap_residual_F"].to_numpy(float)
    measured = frame["cap_F"].to_numpy(float)
    sheet = frame["sheet"].astype(str).to_numpy()
    rows = []
    scales = np.linspace(0.0, 1.0, 11)
    for heldout in sorted(set(sheet)):
        train = sheet != heldout
        test = ~train
        candidates = []
        for scale in scales:
            pred = base + (scale / args.deployed_scale) * (deployed - base)
            candidates.append((rms_log(pred[train], measured[train]), float(scale)))
        _, selected_scale = min(candidates)
        selected = base + (selected_scale / args.deployed_scale) * (deployed - base)
        base_rms = rms_log(base[test], measured[test])
        selected_rms = rms_log(selected[test], measured[test])
        rows.append(
            {
                "heldout_sheet": heldout,
                "heldout_points": int(np.sum(test)),
                "scale_selected_without_heldout_sheet": selected_scale,
                "heldout_base_rms_dec": base_rms,
                "heldout_base_plus_residual_rms_dec": selected_rms,
                "heldout_gain_dec": base_rms - selected_rms,
            }
        )
    result = pd.DataFrame(rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(
        args.out_dir / "v93_cv_cross_sheet_selection_ablation.csv",
        index=False,
        encoding="utf-8-sig",
    )
    summary = {
        "split": "leave_one_cv_sheet_out_for_residual_scale_selection",
        "important_scope_note": (
            "The fixed CV model was trained on the current data; only residual "
            "scale selection excludes each held-out sheet. IV residual has the "
            "separate true curve-condition training holdout."
        ),
        "sheets": int(len(result)),
        "positive_heldout_sheets": int((result["heldout_gain_dec"] > 0.0).sum()),
        "mean_heldout_gain_dec": float(result["heldout_gain_dec"].mean()),
        "pass": bool((result["heldout_gain_dec"] > 0.0).all()),
    }
    (args.out_dir / "v93_cv_cross_sheet_selection_ablation.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
