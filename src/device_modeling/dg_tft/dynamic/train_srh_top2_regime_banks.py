from __future__ import annotations

"""Distill fast and slow two-trap teachers into separate symbolic SRH banks."""

from pathlib import Path
import argparse
import copy
import json
import math
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import refine_srh_waveform_end_to_end as base


ROOT = Path(__file__).resolve().parent
DYNAMIC_WORK = Path(os.environ.get("DG_TFT_DYNAMIC_WORK_DIR", ROOT / "results_long_timescale"))
OUT = DYNAMIC_WORK / "srh_top2_regime_banks"


def regime(row: dict) -> str:
    return "slow" if float(row["t"][-1]) >= 0.05 else "fast"


def predict(
    models: dict,
    row: dict,
    hard: bool,
    amp_parameterization: str = "independent",
) -> torch.Tensor:
    x_fast = row["x_fast"]
    x_slow = row["x_slow"]
    def decoded(name: str, x: torch.Tensor) -> torch.Tensor:
        model = models[name]
        z = model(x, hard_top1=hard).squeeze(0)
        return z * float(model.target_scale) + float(model.target_center)

    log_tf = decoded("tau_fast", x_fast)
    log_ts = decoded("tau_slow", x_slow)
    if amp_parameterization == "conservative":
        total_raw = decoded("total_amp", x_fast)
        exchange_raw = decoded("exchange_amp", x_fast)
        total = 5.0 * torch.tanh(total_raw / 5.0)
        exchange = 2.5 * torch.tanh(exchange_raw / 2.5)
        af = 0.5 * total + exchange
        ass = 0.5 * total - exchange
    else:
        af_raw = decoded("amp_fast", x_fast)
        ass_raw = decoded("amp_slow", x_fast)
        af = 5.0 * torch.tanh(af_raw / 5.0)
        ass = 5.0 * torch.tanh(ass_raw / 5.0)
    tf = torch.pow(log_tf.new_tensor(10.0), torch.clamp(log_tf, -8.0, 3.0))
    ts = torch.pow(log_ts.new_tensor(10.0), torch.clamp(log_ts, -8.0, 3.0))
    t = row["t"]
    y0 = row["y_teacher"][0]
    return (
        y0
        + af * (torch.exp(-t / torch.clamp(tf, min=1.0e-8)) - 1.0)
        + ass * (torch.exp(-t / torch.clamp(ts, min=1.0e-8)) - 1.0)
    )


def init_group(
    rows: list[dict],
    hidden_capture: int,
    hidden_emission: int,
    amp_parameterization: str,
) -> dict[str, base.TwoLayerTop1KAN]:
    n_features = len(base.FEATURE_NAMES)
    n_basis = len(base.BASIS_NAMES)
    amp_names = (
        ("total_amp", "exchange_amp")
        if amp_parameterization == "conservative"
        else ("amp_fast", "amp_slow")
    )
    models = {
        name: base.TwoLayerTop1KAN(
            n_features,
            n_basis,
            hidden=hidden_emission if rows[0]["process"] == "emission" else hidden_capture,
            symbolic_topk=2,
        )
        for name in ("tau_fast", "tau_slow", *amp_names)
    }
    targets = {
        "tau_fast": np.asarray([math.log10(max(r["tau_fast_label"], 1.0e-8)) for r in rows]),
        "tau_slow": np.asarray([math.log10(max(r["tau_slow_label"], 1.0e-8)) for r in rows]),
    }
    if amp_parameterization == "conservative":
        targets["total_amp"] = np.asarray(
            [r["amp_fast"] + r["amp_slow"] for r in rows]
        )
        targets["exchange_amp"] = np.asarray(
            [0.5 * (r["amp_fast"] - r["amp_slow"]) for r in rows]
        )
    else:
        targets["amp_fast"] = np.asarray([r["amp_fast"] for r in rows])
        targets["amp_slow"] = np.asarray([r["amp_slow"] for r in rows])
    with torch.no_grad():
        for name, model in models.items():
            center = float(np.median(targets[name]))
            q25, q75 = np.percentile(targets[name], [25.0, 75.0])
            floor = 0.20 if name.startswith("tau") else 0.010
            scale = float(max((q75 - q25) / 1.349, floor))
            model.target_center = center
            model.target_scale = scale
            model.out_bias.zero_()
    return models


def fit_group(
    rows: list[dict],
    epochs: int = 1800,
    hidden_capture: int = 16,
    hidden_emission: int = 20,
    amp_parameterization: str = "independent",
    print_every: int = 1,
) -> tuple[dict, dict]:
    models = init_group(
        rows,
        hidden_capture,
        hidden_emission,
        amp_parameterization,
    )
    params = [p for model in models.values() for p in model.parameters()]
    opt = torch.optim.AdamW(params, lr=2.5e-3, weight_decay=1.0e-4)
    best_score = float("inf")
    best_state = None
    history = []
    for ep in range(epochs):
        opt.zero_grad()
        soft_wave = []
        hard_wave = []
        anchor_terms = []
        shape_terms = []
        endpoint_terms = []
        param_terms = []
        hard_param_terms = []
        for row in rows:
            soft = predict(models, row, False, amp_parameterization)
            hard = predict(models, row, True, amp_parameterization)
            teacher = row["y_teacher"]
            soft_wave.append(F.smooth_l1_loss(soft, teacher, beta=0.010))
            hard_wave.append(F.smooth_l1_loss(hard, teacher, beta=0.012))
            anchor_idx = sorted(
                {
                    0,
                    max(1, int(round(0.05 * (teacher.numel() - 1)))),
                    max(1, int(round(0.25 * (teacher.numel() - 1)))),
                    max(1, int(round(0.50 * (teacher.numel() - 1)))),
                    teacher.numel() - 1,
                }
            )
            idx = torch.as_tensor(anchor_idx, device=teacher.device)
            anchor_terms.append(
                F.smooth_l1_loss(hard[idx], teacher[idx], beta=0.004)
            )
            teacher_delta = teacher - teacher[0]
            hard_delta = hard - hard[0]
            shape_scale = torch.clamp(
                torch.abs(teacher_delta[-1]),
                min=teacher.new_tensor(0.006),
            )
            shape_terms.append(
                torch.mean(((hard_delta - teacher_delta) / shape_scale) ** 2)
            )
            endpoint_terms.append(
                F.smooth_l1_loss(
                    hard_delta[-1],
                    teacher_delta[-1],
                    beta=0.002,
                )
            )
            target_items = [
                ("tau_fast", row["x_fast"], math.log10(max(row["tau_fast_label"], 1.0e-8))),
                ("tau_slow", row["x_slow"], math.log10(max(row["tau_slow_label"], 1.0e-8))),
            ]
            if amp_parameterization == "conservative":
                target_items.extend(
                    [
                        ("total_amp", row["x_fast"], row["amp_fast"] + row["amp_slow"]),
                        ("exchange_amp", row["x_fast"], 0.5 * (row["amp_fast"] - row["amp_slow"])),
                    ]
                )
            else:
                target_items.extend(
                    [
                        ("amp_fast", row["x_fast"], row["amp_fast"]),
                        ("amp_slow", row["x_fast"], row["amp_slow"]),
                    ]
                )
            for name, x, target in target_items:
                model = models[name]
                raw = model(x, hard_top1=False).squeeze(0)
                raw_h = model(x, hard_top1=True).squeeze(0)
                target_z = (float(target) - float(model.target_center)) / float(model.target_scale)
                target_t = raw.new_tensor(target_z)
                param_terms.append(F.smooth_l1_loss(raw, target_t, beta=0.10))
                hard_param_terms.append(F.smooth_l1_loss(raw_h, target_t, beta=0.12))
        soft_wave_loss = torch.stack(soft_wave).mean()
        hard_wave_loss = torch.stack(hard_wave).mean()
        anchor_loss = torch.stack(anchor_terms).mean()
        shape_loss = torch.stack(shape_terms).mean()
        endpoint_loss = torch.stack(endpoint_terms).mean()
        param_loss = torch.stack(param_terms).mean()
        hard_param_loss = torch.stack(hard_param_terms).mean()
        tail = torch.topk(
            torch.stack(
                [
                    torch.sqrt(
                        torch.mean(
                            (
                                predict(
                                    models,
                                    row,
                                    True,
                                    amp_parameterization,
                                )
                                - row["y_teacher"]
                            )
                            ** 2
                        )
                    )
                    for row in rows
                ]
            ),
            k=max(1, int(math.ceil(0.25 * len(rows)))),
        ).values.mean()
        sparsity = torch.stack([m.sparsity_loss() for m in models.values()]).mean()
        # Per-segment two-exponential parameters can be weakly identifiable.
        # Treat the complete teacher waveform as the authoritative target.
        loss = (
            12.0 * soft_wave_loss
            + 26.0 * hard_wave_loss
            + 18.0 * anchor_loss
            + 1.8 * shape_loss
            + 28.0 * endpoint_loss
            + 0.35 * param_loss
            + 0.55 * hard_param_loss
            + 7.0 * tail
            + 0.003 * sparsity
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 5.0)
        opt.step()
        if ep % print_every == 0 or ep == epochs - 1:
            with torch.no_grad():
                rms = np.asarray(
                    [
                        float(
                            torch.sqrt(
                                torch.mean(
                                    (
                                        predict(
                                            models,
                                            row,
                                            True,
                                            amp_parameterization,
                                        )
                                        - row["y_teacher"]
                                    )
                                    ** 2
                                )
                            )
                        )
                        for row in rows
                    ]
                )
            mean_rms = float(np.mean(rms))
            p90_rms = float(np.quantile(rms, 0.9))
            max_rms = float(np.max(rms))
            score = mean_rms + 0.8 * p90_rms + 0.3 * max_rms
            if score < best_score:
                best_score = score
                best_state = {
                    name: copy.deepcopy(model.state_dict())
                    for name, model in models.items()
                }
            print(
                f"Hysteresis ep {ep:5d}/{epochs} "
                f"loss={loss.item():.5g} "
                f"soft={float(soft_wave_loss):.4g} hard={float(hard_wave_loss):.4g} "
                f"endpoint={float(endpoint_loss):.4g} "
                f"mean={mean_rms:.4g} p90={p90_rms:.4g} max={max_rms:.4g} "
                f"best={best_score:.4g}"
            )
            if ep % 200 == 0 or ep == epochs - 1:
                history.append(
                    {
                        "epoch": ep,
                        "mean_teacher_rmse_dec": mean_rms,
                        "p90_teacher_rmse_dec": p90_rms,
                        "max_teacher_rmse_dec": max_rms,
                    }
                )
    for name, state in best_state.items():
        models[name].load_state_dict(state)
    return models, {"score": best_score, "history": history}


def _dynamic_metrics(pred: np.ndarray, measured: np.ndarray) -> dict[str, float]:
    return base._dynamic_metrics_np(np.asarray(pred), np.asarray(measured))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1800)
    parser.add_argument("--regime", choices=("all", "fast", "slow"), default="all")
    parser.add_argument("--print-every", type=int, default=1)
    parser.add_argument("--hidden-capture", type=int, default=16)
    parser.add_argument("--hidden-emission", type=int, default=20)
    parser.add_argument("--tag", default="")
    parser.add_argument(
        "--amp-parameterization",
        choices=("independent", "conservative"),
        default="independent",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    suffix = f"_{args.tag}" if args.tag else ""
    out_dir = OUT if args.regime == "all" and not suffix else OUT / f"{args.regime}_only{suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)
    label, norm = base.build_label_table()
    label, norm = base._stabilize_temperature_normalization(label, norm)
    rows = base._make_waveform_table(norm, max_points_per_segment=240)
    grouped: dict[tuple[str, str, str], list[dict]] = {}
    for row in rows:
        row_regime = regime(row)
        if args.regime != "all" and row_regime != args.regime:
            continue
        key = (row_regime, row["family"], row["process"])
        grouped.setdefault(key, []).append(row)

    all_models = {}
    group_reports = {}
    metric_rows = []
    pred_rows = []
    formulas = {}
    for key, subset in sorted(grouped.items()):
        models, report = fit_group(
            subset,
            epochs=args.epochs,
            hidden_capture=args.hidden_capture,
            hidden_emission=args.hidden_emission,
            amp_parameterization=args.amp_parameterization,
            print_every=args.print_every,
        )
        all_models[key] = models
        group_reports["|".join(key)] = report
        formulas["|".join(key)] = {
            name: {
                **base._top1_formula(model),
                "target_center": float(model.target_center),
                "target_scale": float(model.target_scale),
            }
            for name, model in models.items()
        }
        with torch.no_grad():
            for row in subset:
                student = predict(
                    models,
                    row,
                    True,
                    args.amp_parameterization,
                ).cpu().numpy()
                teacher = row["y_teacher"].cpu().numpy()
                measured = row["y"].cpu().numpy()
                teacher_rmse = float(np.sqrt(np.mean((teacher - measured) ** 2)))
                student_rmse = float(np.sqrt(np.mean((student - measured) ** 2)))
                distill_rmse = float(np.sqrt(np.mean((student - teacher) ** 2)))
                dynamic = _dynamic_metrics(student, measured)
                metric_rows.append(
                    {
                        "curve_id": row["curve_id"],
                        "seg_id": row["seg_id"],
                        "regime": key[0],
                        "family": key[1],
                        "process": key[2],
                        "teacher_rmse_dec": teacher_rmse,
                        "top2_rmse_dec": student_rmse,
                        "student_teacher_rmse_dec": distill_rmse,
                        "top2_dynamic_shape_rmse_norm": dynamic["dynamic_shape_rmse_norm"],
                        "measured_drift_dec": dynamic["measured_drift_dec"],
                        "top2_drift_dec": dynamic["predicted_drift_dec"],
                        "top2_endpoint_drift_abs_error_dec": dynamic["endpoint_drift_abs_error_dec"],
                    }
                )
                for t, ym, yt, ys in zip(row["t"].numpy(), measured, teacher, student):
                    pred_rows.append(
                        {
                            "curve_id": row["curve_id"],
                            "seg_id": row["seg_id"],
                            "time_rel_s": float(t),
                            "log_id_meas": float(ym),
                            "log_id_teacher": float(yt),
                            "log_id_top2": float(ys),
                        }
                    )
    metrics = pd.DataFrame(metric_rows)
    predictions = pd.DataFrame(pred_rows)
    # Compatibility aliases for the release plotting and audit pipeline.
    metrics["top1_rmse_dec"] = metrics["top2_rmse_dec"]
    metrics["top1_dynamic_shape_rmse_norm"] = metrics["top2_dynamic_shape_rmse_norm"]
    metrics["top1_drift_dec"] = metrics["top2_drift_dec"]
    metrics["top1_endpoint_drift_abs_error_dec"] = metrics["top2_endpoint_drift_abs_error_dec"]
    predictions["log_id_top1"] = predictions["log_id_top2"]
    metrics.to_csv(out_dir / "srh_top2_regime_bank_metrics.csv", index=False, encoding="utf-8-sig")
    predictions.to_csv(out_dir / "srh_top2_regime_bank_predictions.csv", index=False, encoding="utf-8-sig")
    summary = {
        "architecture": "two independent two-trap SRH banks: fast us-ms and slow seconds",
        "trained_regime": args.regime,
        "hidden_capture": args.hidden_capture,
        "hidden_emission": args.hidden_emission,
        "amp_parameterization": args.amp_parameterization,
        "normalization": norm,
        "formulas": formulas,
        "groups": group_reports,
        "teacher_rmse_dec_mean": float(metrics["teacher_rmse_dec"].mean()),
        "top2_rmse_dec_mean": float(metrics["top2_rmse_dec"].mean()),
        "student_teacher_rmse_dec_mean": float(metrics["student_teacher_rmse_dec"].mean()),
        "student_teacher_rmse_dec_p90": float(metrics["student_teacher_rmse_dec"].quantile(0.9)),
        "student_teacher_rmse_dec_max": float(metrics["student_teacher_rmse_dec"].max()),
    }
    (out_dir / "srh_top2_regime_bank_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    torch.save(
        {
            "|".join(key): {
                name: {
                    "state_dict": model.state_dict(),
                    "target_center": float(model.target_center),
                    "target_scale": float(model.target_scale),
                }
                for name, model in models.items()
            }
            for key, models in all_models.items()
        },
        out_dir / "srh_top2_regime_bank_checkpoint.pt",
    )
    print(json.dumps({k: v for k, v in summary.items() if k.endswith("_mean") or k.endswith("_p90") or k.endswith("_max")}, indent=2))


if __name__ == "__main__":
    main()
