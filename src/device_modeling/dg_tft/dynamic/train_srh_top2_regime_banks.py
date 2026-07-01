from __future__ import annotations

"""Distill fast and slow two-trap teachers into separate symbolic SRH banks."""

from pathlib import Path
import argparse
import copy
import json
import math
import os
import time

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


def predict_batch(
    models: dict,
    rows: list[dict],
    hard: bool,
    amp_parameterization: str = "independent",
) -> tuple[list[torch.Tensor], dict[str, torch.Tensor]]:
    """Evaluate each KAN once for the complete segment bank.

    Segment time vectors have different lengths, so only the inexpensive SRH
    exponentials remain in a Python loop.  The costly KAN forwards are batched
    over all segments and reused by waveform, parameter, anchor and tail
    losses.
    """

    x_fast = torch.cat([row["x_fast"] for row in rows], dim=0)
    x_slow = torch.cat([row["x_slow"] for row in rows], dim=0)
    raw: dict[str, torch.Tensor] = {}
    decoded: dict[str, torch.Tensor] = {}
    for name, model in models.items():
        x = x_slow if name == "tau_slow" else x_fast
        z = model(x, hard_top1=hard).reshape(-1)
        raw[name] = z
        decoded[name] = z * float(model.target_scale) + float(model.target_center)

    log_tf = decoded["tau_fast"]
    log_ts = decoded["tau_slow"]
    if amp_parameterization == "conservative":
        total_raw = decoded["total_amp"]
        exchange_raw = decoded["exchange_amp"]
        total = 5.0 * torch.tanh(total_raw / 5.0)
        exchange = 2.5 * torch.tanh(exchange_raw / 2.5)
        af = 0.5 * total + exchange
        ass = 0.5 * total - exchange
    else:
        af = 5.0 * torch.tanh(decoded["amp_fast"] / 5.0)
        ass = 5.0 * torch.tanh(decoded["amp_slow"] / 5.0)
    tf = torch.pow(log_tf.new_tensor(10.0), torch.clamp(log_tf, -8.0, 3.0))
    ts = torch.pow(log_ts.new_tensor(10.0), torch.clamp(log_ts, -8.0, 3.0))

    waves: list[torch.Tensor] = []
    for i, row in enumerate(rows):
        t = row["t"]
        y0 = row["y_teacher"][0]
        waves.append(
            y0
            + af[i] * (torch.exp(-t / torch.clamp(tf[i], min=1.0e-8)) - 1.0)
            + ass[i] * (torch.exp(-t / torch.clamp(ts[i], min=1.0e-8)) - 1.0)
        )
    return waves, raw


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
    optimizer_policy: str = "optimized",
) -> tuple[dict, dict]:
    models = init_group(
        rows,
        hidden_capture,
        hidden_emission,
        amp_parameterization,
    )
    params = [p for model in models.values() for p in model.parameters()]
    opt = torch.optim.AdamW(params, lr=2.5e-3, weight_decay=1.0e-4)
    sched = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            opt,
            T_max=max(epochs, 1),
            eta_min=5.0e-5,
        )
        if optimizer_policy == "optimized"
        else None
    )
    best_score = float("inf")
    best_state = None
    history = []
    no_improve_checks = 0
    early_stop_patience = 16
    early_stop_min_epochs = min(400, max(150, epochs // 3))
    started = time.perf_counter()
    print(
        f"  SRH-KAN bank: segments={len(rows)}, epochs={epochs}, "
        f"hidden=[{hidden_capture},{hidden_emission}], amp={amp_parameterization}, "
        f"optimizer={optimizer_policy}",
        flush=True,
    )
    for ep in range(epochs):
        opt.zero_grad()
        soft_predictions, soft_raw = predict_batch(
            models, rows, False, amp_parameterization
        )
        hard_predictions, hard_raw = predict_batch(
            models, rows, True, amp_parameterization
        )
        soft_wave = []
        hard_wave = []
        anchor_terms = []
        shape_terms = []
        endpoint_terms = []
        param_terms = []
        hard_param_terms = []
        for row_idx, row in enumerate(rows):
            soft = soft_predictions[row_idx]
            hard = hard_predictions[row_idx]
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
                raw = soft_raw[name][row_idx]
                raw_h = hard_raw[name][row_idx]
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
        hard_rms = torch.stack(
            [
                torch.sqrt(torch.mean((pred - row["y_teacher"]) ** 2))
                for pred, row in zip(hard_predictions, rows)
            ]
        )
        tail = torch.topk(
            hard_rms,
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
        if sched is not None:
            sched.step()
        if ep % 25 == 0 or ep == epochs - 1:
            with torch.no_grad():
                eval_waves, _ = predict_batch(
                    models, rows, True, amp_parameterization
                )
                rms = np.asarray(
                    [
                        float(torch.sqrt(torch.mean((pred - row["y_teacher"]) ** 2)))
                        for pred, row in zip(eval_waves, rows)
                    ]
                )
            score = float(np.mean(rms) + 0.8 * np.quantile(rms, 0.9) + 0.3 * np.max(rms))
            minimum_improvement = 1.0e-5 if optimizer_policy == "optimized" else 0.0
            if score < best_score - minimum_improvement:
                best_score = score
                best_state = {
                    name: copy.deepcopy(model.state_dict())
                    for name, model in models.items()
                }
                no_improve_checks = 0
            else:
                no_improve_checks += 1
            if ep % 100 == 0 or ep == epochs - 1:
                elapsed = time.perf_counter() - started
                eta = elapsed / max(ep + 1, 1) * max(epochs - ep - 1, 0)
                history.append(
                    {
                        "epoch": ep,
                        "mean_teacher_rmse_dec": float(np.mean(rms)),
                        "p90_teacher_rmse_dec": float(np.quantile(rms, 0.9)),
                        "max_teacher_rmse_dec": float(np.max(rms)),
                    }
                )
                print(
                    f"  epoch {ep + 1:4d}/{epochs}: mean={np.mean(rms):.5f} "
                    f"p90={np.quantile(rms, 0.9):.5f} "
                    f"max={np.max(rms):.5f} dec, "
                    f"elapsed={elapsed/60.0:.1f}m ETA={eta/60.0:.1f}m",
                    flush=True,
                )
            if (
                optimizer_policy == "optimized"
                and ep >= early_stop_min_epochs
                and no_improve_checks >= early_stop_patience
            ):
                history.append(
                    {
                        "epoch": ep,
                        "early_stop": True,
                        "best_score": best_score,
                    }
                )
                print(
                    f"  early stop at epoch {ep}; best score={best_score:.6g}",
                    flush=True,
                )
                break
    for name, state in best_state.items():
        models[name].load_state_dict(state)
    return models, {"score": best_score, "history": history}


def _dynamic_metrics(pred: np.ndarray, measured: np.ndarray) -> dict[str, float]:
    return base._dynamic_metrics_np(np.asarray(pred), np.asarray(measured))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1800)
    parser.add_argument("--regime", choices=("all", "fast", "slow"), default="all")
    parser.add_argument("--hidden-capture", type=int, default=16)
    parser.add_argument("--hidden-emission", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260524)
    parser.add_argument("--fast-restarts", type=int, default=1)
    parser.add_argument("--slow-restarts", type=int, default=2)
    parser.add_argument("--tag", default="")
    parser.add_argument(
        "--amp-parameterization",
        choices=("independent", "conservative"),
        default="independent",
    )
    parser.add_argument(
        "--optimizer-policy",
        choices=("optimized", "v90"),
        default="optimized",
        help=(
            "v90 preserves the historical fixed-AdamW full-budget policy; "
            "optimized adds cosine decay and validation early stopping."
        ),
    )
    parser.add_argument(
        "--fast-amp-parameterization",
        choices=("independent", "conservative"),
        default=None,
    )
    parser.add_argument(
        "--slow-amp-parameterization",
        choices=("independent", "conservative"),
        default=None,
    )
    parser.add_argument(
        "--fast-optimizer-policy",
        choices=("optimized", "v90"),
        default=None,
    )
    parser.add_argument(
        "--slow-optimizer-policy",
        choices=("optimized", "v90"),
        default=None,
    )
    parser.add_argument("--max-top2-rmse-mean", type=float, default=0.020)
    parser.add_argument("--max-student-teacher-rmse-mean", type=float, default=0.006)
    parser.add_argument("--max-student-teacher-rmse-p90", type=float, default=0.012)
    parser.add_argument("--max-endpoint-drift-error-mean", type=float, default=0.070)
    parser.add_argument("--max-endpoint-drift-error-p95", type=float, default=0.230)
    parser.add_argument(
        "--max-v90-slow-top2-rmse-mean",
        type=float,
        default=0.0033323642404866406,
    )
    parser.add_argument(
        "--enforce-v90-slow-reference",
        action="store_true",
        help=(
            "Optional release-audit comparison for the bundled validation "
            "dataset. Leave disabled for a new device, whose quality is gated "
            "only by its own teacher, waveform, endpoint, and holdout metrics."
        ),
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
    total_groups = len(grouped)
    selected_amp_policies: dict[str, str] = {}
    selected_optimizer_policies: dict[str, str] = {}
    for group_index, (key, subset) in enumerate(sorted(grouped.items()), start=1):
        group_regime = str(key[0])
        group_amp_parameterization = (
            getattr(args, f"{group_regime}_amp_parameterization")
            or args.amp_parameterization
        )
        group_optimizer_policy = (
            getattr(args, f"{group_regime}_optimizer_policy")
            or args.optimizer_policy
        )
        selected_amp_policies[group_regime] = group_amp_parameterization
        selected_optimizer_policies[group_regime] = group_optimizer_policy
        print(
            f"[hysteresis bank {group_index}/{total_groups}] "
            f"{'|'.join(key)} segments={len(subset)}",
            flush=True,
        )
        restart_count = max(
            1,
            int(
                args.fast_restarts
                if group_regime == "fast"
                else args.slow_restarts
            ),
        )
        models = None
        report = None
        restart_reports = []
        for restart in range(restart_count):
            restart_seed = (
                int(args.seed)
                + group_index * 1009
                + restart * 9176
            )
            np.random.seed(restart_seed)
            torch.manual_seed(restart_seed)
            candidate_models, candidate_report = fit_group(
                subset,
                epochs=args.epochs,
                hidden_capture=args.hidden_capture,
                hidden_emission=args.hidden_emission,
                amp_parameterization=group_amp_parameterization,
                optimizer_policy=group_optimizer_policy,
            )
            restart_reports.append(
                {
                    "restart": restart,
                    "seed": restart_seed,
                    "score": float(candidate_report["score"]),
                }
            )
            if report is None or candidate_report["score"] < report["score"]:
                models = candidate_models
                report = candidate_report
        if models is None or report is None:
            raise RuntimeError(f"No restart completed for {'|'.join(key)}")
        report["restart_selection"] = {
            "count": restart_count,
            "candidates": restart_reports,
            "selected": min(restart_reports, key=lambda item: item["score"]),
        }
        all_models[key] = models
        print(
            f"[hysteresis bank {group_index}/{total_groups}] complete; "
            f"score={report['score']:.6g}",
            flush=True,
        )
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
                    group_amp_parameterization,
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
    # Hysteresis is intentionally strict top-2.  Do not publish misleading
    # top-1 aliases: IV/CV/temperature are top-1, while both SRH state banks
    # retain two primitives per symbolic edge.
    metrics.to_csv(out_dir / "srh_top2_regime_bank_metrics.csv", index=False, encoding="utf-8-sig")
    predictions.to_csv(out_dir / "srh_top2_regime_bank_predictions.csv", index=False, encoding="utf-8-sig")
    summary = {
        "architecture": "two independent two-trap SRH banks: fast us-ms and slow seconds",
        "trained_regime": args.regime,
        "hidden_capture": args.hidden_capture,
        "hidden_emission": args.hidden_emission,
        "amp_parameterization": selected_amp_policies,
        "optimizer_policy": selected_optimizer_policies,
        "seed": args.seed,
        "restarts": {
            "fast": args.fast_restarts,
            "slow": args.slow_restarts,
        },
        "normalization": norm,
        "formulas": formulas,
        "groups": group_reports,
        "teacher_rmse_dec_mean": float(metrics["teacher_rmse_dec"].mean()),
        "top2_rmse_dec_mean": float(metrics["top2_rmse_dec"].mean()),
        "student_teacher_rmse_dec_mean": float(metrics["student_teacher_rmse_dec"].mean()),
        "student_teacher_rmse_dec_p90": float(metrics["student_teacher_rmse_dec"].quantile(0.9)),
        "student_teacher_rmse_dec_max": float(metrics["student_teacher_rmse_dec"].max()),
    }
    slow_metrics = metrics.loc[metrics["regime"].astype(str).eq("slow")]
    if len(slow_metrics):
        summary["slow_teacher_rmse_dec_mean"] = float(
            slow_metrics["teacher_rmse_dec"].mean()
        )
        summary["slow_top2_rmse_dec_mean"] = float(
            slow_metrics["top2_rmse_dec"].mean()
        )
        summary["slow_student_teacher_rmse_dec_mean"] = float(
            slow_metrics["student_teacher_rmse_dec"].mean()
        )
    topology_topk = []
    for group_formula in formulas.values():
        for formula in group_formula.values():
            if isinstance(formula, dict) and "symbolic_topk" in formula:
                topology_topk.append(int(formula["symbolic_topk"]))
    endpoint_mean = float(metrics["top2_endpoint_drift_abs_error_dec"].mean())
    endpoint_p95 = float(metrics["top2_endpoint_drift_abs_error_dec"].quantile(0.95))
    acceptance_checks = {
        "strict_top2_symbolic_topology": bool(
            topology_topk and all(value == 2 for value in topology_topk)
        ),
        "top2_rmse_mean": bool(summary["top2_rmse_dec_mean"] <= args.max_top2_rmse_mean),
        "student_teacher_rmse_mean": bool(
            summary["student_teacher_rmse_dec_mean"]
            <= args.max_student_teacher_rmse_mean
        ),
        "student_teacher_rmse_p90": bool(
            summary["student_teacher_rmse_dec_p90"]
            <= args.max_student_teacher_rmse_p90
        ),
        "endpoint_drift_error_mean": bool(
            endpoint_mean <= args.max_endpoint_drift_error_mean
        ),
        "endpoint_drift_error_p95": bool(
            endpoint_p95 <= args.max_endpoint_drift_error_p95
        ),
    }
    if len(slow_metrics) and args.enforce_v90_slow_reference:
        acceptance_checks["slow_not_worse_than_v90"] = bool(
            summary["slow_top2_rmse_dec_mean"]
            <= args.max_v90_slow_top2_rmse_mean
        )
    summary["top2_symbolic_formula_count"] = len(topology_topk)
    summary["top2_endpoint_drift_abs_error_dec_mean"] = endpoint_mean
    summary["top2_endpoint_drift_abs_error_dec_p95"] = endpoint_p95
    summary["acceptance"] = {
        "pass": all(acceptance_checks.values()),
        "checks": acceptance_checks,
        "limits": {
            "top2_rmse_dec_mean": args.max_top2_rmse_mean,
            "student_teacher_rmse_dec_mean": args.max_student_teacher_rmse_mean,
            "student_teacher_rmse_dec_p90": args.max_student_teacher_rmse_p90,
            "endpoint_drift_abs_error_dec_mean": args.max_endpoint_drift_error_mean,
            "endpoint_drift_abs_error_dec_p95": args.max_endpoint_drift_error_p95,
            "v90_slow_top2_rmse_dec_mean": (
                args.max_v90_slow_top2_rmse_mean
                if args.enforce_v90_slow_reference
                else None
            ),
        },
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
    if not summary["acceptance"]["pass"]:
        raise RuntimeError(
            "Top-2 SRH hysteresis acceptance failed: "
            f"{summary['acceptance']}"
        )


if __name__ == "__main__":
    main()
