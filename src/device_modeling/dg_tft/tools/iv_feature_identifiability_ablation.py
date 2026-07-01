from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


REGIONS = {
    "off": (-np.inf, -12.3),
    "subthreshold": (-12.3, -10.5),
    "transition": (-10.5, -8.7),
    "on": (-8.7, -7.0),
    "high": (-7.0, np.inf),
}


def standardize(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = np.nanmedian(train, axis=0)
    scale = np.nanstd(train, axis=0)
    scale = np.where(scale > 1.0e-9, scale, 1.0)
    return (train - center) / scale, (test - center) / scale


def design(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -8.0, 8.0)
    return np.column_stack([np.ones(len(x)), x, x * x, np.tanh(x)])


def ridge_fit(x: np.ndarray, y: np.ndarray, ridge: float = 3.0e-3) -> np.ndarray:
    reg = ridge * np.eye(x.shape[1])
    reg[0, 0] = 0.0
    return np.linalg.solve(x.T @ x + reg, x.T @ y)


def rms(value: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(value)))) if len(value) else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cross-condition screening and rank audit for proposed IV features."
    )
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--fitted-params", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    frame = pd.read_csv(args.predictions)
    params = json.loads(args.fitted_params.read_text(encoding="utf-8"))["final_params"]
    vtg = frame["VTG_V"].to_numpy(float)
    vbg = frame["VBG_V"].to_numpy(float)
    vds = frame["VDS_V"].to_numpy(float)
    measured = np.log10(np.maximum(np.abs(frame["ID_A"].to_numpy(float)), 1.0e-15))
    residual_error = frame["log10_error_residual_dec"].to_numpy(float)

    at = float(params["Cox_t"]) / (float(params["Cox_t"]) + float(params["Cox_b"]))
    ab = 1.0 - at
    ut = vtg - float(params["Vth_t0"])
    ub = vbg - float(params["Vth_b0"])
    veff = at * ut + ab * ub

    # qbar0 is already present in the production feature tensor.  Its exact
    # monotone proxy is sufficient for rank/duplication screening here; the
    # production implementation remains the charge-based qsolve expression.
    qbar_proxy = np.log1p(np.exp(np.clip(veff / 0.18, -40.0, 40.0)))
    existing = np.column_stack(
        [ut, ub, veff, vds, qbar_proxy, vds * qbar_proxy]
    )
    candidates = {
        "no_extra_features": np.zeros(len(frame)),
        "log_qbar0_safe": qbar_proxy,
        "Veff_over_VDS_safe": veff / np.maximum(np.abs(vds), 0.10),
        "gate_asymmetry": ut - ub,
        "vds_qbar_interaction": vds * qbar_proxy,
    }

    condition = (
        frame["source_file"].astype(str)
        + "|"
        + frame["family"].astype(str)
        + "|"
        + frame["VDS_V"].round(5).astype(str)
    )
    fold = condition.map(
        lambda value: int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:8], 16) % 5
    ).to_numpy()

    rows: list[dict] = []
    for name, candidate in candidates.items():
        fold_errors: list[np.ndarray] = []
        fold_measured: list[np.ndarray] = []
        for heldout in range(5):
            train = fold != heldout
            test = ~train
            if name == "no_extra_features":
                pred_correction = np.full(np.sum(test), -np.mean(residual_error[train]))
            else:
                train_x, test_x = standardize(
                    candidate[train, None], candidate[test, None]
                )
                beta = ridge_fit(design(train_x[:, 0]), -residual_error[train])
                pred_correction = design(test_x[:, 0]) @ beta
            fold_errors.append(residual_error[test] + pred_correction)
            fold_measured.append(measured[test])
        error = np.concatenate(fold_errors)
        measured_by_fold = np.concatenate(fold_measured)
        row = {
            "feature": name,
            "cross_condition_rms_dec": rms(error),
            "gain_vs_uncorrected_dec": rms(residual_error) - rms(error),
            "symbolic_basis_terms_screened": 0 if name == "no_extra_features" else 3,
        }
        for region, (lo, hi) in REGIONS.items():
            mask = (measured_by_fold >= lo) & (measured_by_fold < hi)
            row[f"{region}_rms_dec"] = rms(error[mask])
        rows.append(row)

    base_scaled, _ = standardize(existing, existing)
    base_rank = int(np.linalg.matrix_rank(base_scaled, tol=1.0e-7))
    base_condition = float(np.linalg.cond(base_scaled))
    rank_rows = []
    for name, candidate in candidates.items():
        augmented = (
            existing
            if name == "no_extra_features"
            else np.column_stack([existing, candidate])
        )
        scaled, _ = standardize(augmented, augmented)
        rank_rows.append(
            {
                "feature": name,
                "existing_effective_rank": base_rank,
                "augmented_effective_rank": int(
                    np.linalg.matrix_rank(scaled, tol=1.0e-7)
                ),
                "existing_condition_number": base_condition,
                "augmented_condition_number": float(np.linalg.cond(scaled)),
                "exact_existing_feature": name
                in {"log_qbar0_safe", "vds_qbar_interaction"},
                "exact_linear_combination_of_existing": name == "gate_asymmetry",
            }
        )

    result = pd.DataFrame(rows)
    rank = pd.DataFrame(rank_rows)
    baseline = float(
        result.loc[
            result["feature"] == "no_extra_features", "cross_condition_rms_dec"
        ].iloc[0]
    )
    nonredundant = {
        row["feature"]
        for row in rank_rows
        if not row["exact_existing_feature"]
        and not row["exact_linear_combination_of_existing"]
    }
    selected = result[
        (result["cross_condition_rms_dec"] < baseline - 1.0e-4)
        & (result["feature"].isin(nonredundant))
    ]["feature"].tolist()
    payload = {
        "method": "five-fold cross-condition residual screening with a three-term analytic candidate basis",
        "scope": "screening ablation; a feature is not admitted to the production KAN without stable held-out gain",
        "existing_production_features": [
            "Ut",
            "Ub",
            "Veff",
            "VDS",
            "qbar0=log1p(charge_proxy)",
            "VDS*qbar0",
        ],
        "selected_feature_set": selected,
        "pass_no_unjustified_feature_addition": len(selected) == 0,
        "rows": rows,
        "rank_audit": rank_rows,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.out_dir / "v93_iv_feature_ablation.csv", index=False)
    rank.to_csv(args.out_dir / "v93_iv_feature_rank_audit.csv", index=False)
    (args.out_dir / "v93_iv_feature_ablation.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
