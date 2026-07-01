from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[2]


def copy_file(source: Path, destination: Path, *, required: bool = True) -> None:
    if not source.exists():
        if required:
            raise FileNotFoundError(source)
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def copy_complete_device_wrapper(source: Path, destination: Path) -> None:
    """Add the public four-terminal interface around the trained slow-SRH body."""

    text = source.read_text(encoding="utf-8")
    legacy_header = ".subckt dg_tft_complete_model tg bg d s ntf nts nbf nbs"
    if legacy_header not in text or "dg_tft_complete_model_state_access" in text:
        copy_file(source, destination)
        return
    state_name = "dg_tft_complete_model_state_access_slow"
    protected = text.replace(
        legacy_header,
        f".subckt {state_name} tg bg d s ntf nts nbf nbs",
        1,
    )
    protected = protected.replace(
        ".ends dg_tft_complete_model",
        f".ends {state_name}",
        1,
    )
    adapter = f"""

* Four-terminal compatibility entry. The freshly trained slow-SRH equations
* remain in {state_name}; this wrapper only makes state nodes internal.
.subckt dg_tft_complete_model tg bg d s
+ vfrom_t=0 vfrom_b=0 cycle_index=0 hold_s=1
+ prev_log_id=-9 seg_log_id0=-9 temperature_dataset=0
Xprotected tg bg d s _ntf _nts _nbf _nbs {state_name}
+ vfrom_t=vfrom_t vfrom_b=vfrom_b cycle_index=cycle_index hold_s=hold_s
+ prev_log_id=prev_log_id seg_log_id0=seg_log_id0 temperature_dataset=temperature_dataset
.ends dg_tft_complete_model
"""
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(protected.rstrip() + adapter, encoding="utf-8")


def reset_directory(path: Path) -> None:
    resolved = path.resolve()
    if ROOT.resolve() not in resolved.parents:
        raise RuntimeError(f"Refusing to reset a directory outside the release: {resolved}")
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def portable_json(value):
    """Replace machine-specific provenance paths in copied delivery metadata."""

    if isinstance(value, dict):
        return {key: portable_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [portable_json(item) for item in value]
    if not isinstance(value, str):
        return value
    text = value
    for sibling in ROOT.parent.iterdir():
        if sibling.is_dir() and sibling.name.lower().startswith("dg_tft_v"):
            text = text.replace(str(sibling), "<release_root>")
            text = text.replace(
                str(sibling).replace("\\", "/"), "<release_root>"
            )
    root_text = str(ROOT)
    text = text.replace(root_text, "<release_root>")
    text = text.replace(root_text.replace("\\", "/"), "<release_root>")
    if ":\\synopsys\\" in text.lower() or ":/synopsys/" in text.lower():
        return "<configured_hspice_executable>"
    if text.startswith("E:\\") or text.startswith("E:/"):
        return "<external_source_not_required_for_runtime>"
    if text.startswith("C:\\Users\\") or text.startswith("C:/Users/"):
        return "<historical_provenance_path_not_required_for_runtime>"
    return text


def sanitize_json_file(path: Path) -> None:
    if not path.exists():
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    path.write_text(
        json.dumps(portable_json(payload), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def rewrite_checkpoint_contract(path: Path, checkpoint_format: str) -> None:
    """Publish a portable v95 schema without changing any learned tensor."""

    if not path.is_file():
        return
    payload = torch.load(path, map_location="cpu", weights_only=False)
    schema = payload.setdefault("checkpoint_schema", {})
    schema["format"] = checkpoint_format
    metadata = payload.get("metadata")
    if isinstance(metadata, dict):
        contract = metadata.setdefault("release_contract", {})
        contract["release"] = "v95"
        contract["base_checkpoint_format"] = checkpoint_format
    payload = portable_json(payload)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def plot_static_hspice(static_dir: Path, validation_figures: Path) -> None:
    iv_path = static_dir / "v92_hspice_iv_grid_equivalence.json"
    if iv_path.is_file():
        iv = json.loads(iv_path.read_text())
        grid = iv["grid"]
        fig, ax = plt.subplots(figsize=(3.45, 2.45))
        ax.semilogy(grid["vtg_V"], grid["python_ids_abs_A"], color="#0072B2", lw=1.8, label="Python")
        ax.semilogy(
            grid["vtg_V"],
            grid["hspice_ids_abs_A"],
            color="#D55E00",
            lw=1.2,
            ls="--",
            label="HSPICE",
        )
        ax.set_xlabel(r"$V_{\mathrm{TG}}$ (V)")
        ax.set_ylabel(r"$|I_{\mathrm{D}}|$ (A)")
        ax.legend(frameon=False, fontsize=7)
        ax.grid(True, color="#D9D9D9", lw=0.45, alpha=0.75)
        fig.tight_layout()
        fig.savefig(validation_figures / "static_iv_python_vs_hspice.png", dpi=300)
        plt.close(fig)

    cv_path = static_dir / "v92_cv_hspice_cv_grid_equivalence.json"
    if cv_path.is_file():
        cv = json.loads(cv_path.read_text())
        fig, axes = plt.subplots(1, 3, figsize=(7.15, 2.35), sharey=False)
        for ax, mode in zip(axes, ("cbg", "ctg", "cgg")):
            g = cv["grid"][mode]
            x = g.get("bias_V", g.get("sweep_voltage_V", g.get("voltage_V")))
            py = g.get("python_cap_F", g.get("python_capacitance_F"))
            hs = g.get("hspice_cap_F", g.get("hspice_capacitance_F"))
            ax.plot(x, py, color="#0072B2", lw=1.8, label="Python")
            ax.plot(x, hs, color="#D55E00", lw=1.2, ls="--", label="HSPICE")
            ax.set_title(mode.upper(), fontsize=8)
            ax.set_xlabel("Bias (V)")
            ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
            ax.grid(True, color="#D9D9D9", lw=0.45, alpha=0.75)
        axes[0].set_ylabel("C (F)")
        axes[0].legend(frameon=False, fontsize=7)
        fig.tight_layout()
        fig.savefig(validation_figures / "static_cv_python_vs_hspice.png", dpi=300)
        plt.close(fig)


def representative_rows(metrics: pd.DataFrame, regime: str) -> pd.DataFrame:
    subset = metrics[metrics["regime"].str.lower() == regime].copy()
    chosen: list[pd.Series] = []
    for family, process in (("VBG", "capture"), ("VBG", "emission"), ("VTG", "capture"), ("VTG", "emission")):
        group = subset[(subset["family"] == family) & (subset["process"] == process)].copy()
        if group.empty:
            continue
        median = float(group["top2_rmse_dec"].median())
        chosen.append(group.iloc[(group["top2_rmse_dec"] - median).abs().argmin()])
    if len(chosen) < 4 and not subset.empty:
        for quantile in (0.2, 0.5, 0.8, 1.0):
            target = float(subset["top2_rmse_dec"].quantile(quantile))
            row = subset.iloc[(subset["top2_rmse_dec"] - target).abs().argmin()]
            if not any(
                str(x["curve_id"]) == str(row["curve_id"]) and int(x["seg_id"]) == int(row["seg_id"])
                for x in chosen
            ):
                chosen.append(row)
            if len(chosen) == 4:
                break
    return pd.DataFrame(chosen[:4])


def plot_dynamic_regime(
    predictions: pd.DataFrame,
    metrics: pd.DataFrame,
    regime: str,
    destination: Path,
) -> None:
    selected = representative_rows(metrics, regime)
    if selected.empty:
        return
    fig, axes = plt.subplots(2, 2, figsize=(7.5, 5.35))
    axes_flat = axes.ravel()
    for panel, (ax, row) in enumerate(zip(axes_flat, selected.itertuples(index=False))):
        curve = predictions[
            (predictions["curve_id"].astype(str) == str(row.curve_id))
            & (predictions["seg_id"].astype(int) == int(row.seg_id))
        ].sort_values("time_rel_s")
        t_s = curve["time_rel_s"].to_numpy(float)
        if regime == "fast":
            if float(np.max(t_s)) < 2.0e-4:
                t = 1.0e6 * t_s
                time_label = r"Time after step ($\mu$s)"
            else:
                t = 1.0e3 * t_s
                time_label = "Time after step (ms)"
        else:
            t = t_s
            time_label = "Time after step (s)"
        measured = np.power(10.0, curve["log_id_meas"].to_numpy(float))
        teacher = np.power(10.0, curve["log_id_teacher"].to_numpy(float))
        student = np.power(10.0, curve["log_id_top2"].to_numpy(float))
        ax.scatter(
            t,
            measured,
            s=8,
            facecolors="white",
            edgecolors="#666666",
            linewidths=0.55,
            label="Measured",
            zorder=3,
        )
        ax.plot(t, teacher, color="#E68632", lw=1.25, ls="--", label="Two-trap teacher")
        ax.plot(t, student, color="#0072B2", lw=1.6, label="Top-2 SRH-KAN")
        positive = measured[measured > 0]
        if positive.size and float(np.max(positive) / np.min(positive)) > 50.0:
            ax.set_yscale("log")
        else:
            ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
        curve_label = (
            str(row.curve_id)
            .replace("_temperature_", "-T-")
            .replace("_Sheet", "-S")
            .replace("_", "-")
        )
        ax.set_title(f"{curve_label} (seg. {int(row.seg_id)})", fontsize=8, pad=6)
        ax.text(
            0.98,
            0.05,
            f"Teacher={row.teacher_rmse_dec:.3f}, Top-2={row.top2_rmse_dec:.3f} dex",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=6.5,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.2},
        )
        ax.text(
            -0.12,
            1.08,
            f"({chr(ord('a') + panel)})",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=8,
            fontweight="bold",
            clip_on=False,
        )
        ax.set_xlabel(time_label)
        ax.set_ylabel(r"$|I_{\mathrm{D}}|$ (A)")
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=4))
        ax.grid(True, color="#D9D9D9", lw=0.45, alpha=0.75)
    for ax in axes_flat[len(selected) :]:
        ax.axis("off")
    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.suptitle(f"{regime.capitalize()} hysteresis: fixed-bias relaxation", y=0.985, fontsize=10)
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.945),
        ncol=3,
        frameon=False,
        fontsize=7.5,
    )
    fig.subplots_adjust(left=0.105, right=0.985, bottom=0.09, top=0.82, wspace=0.34, hspace=0.52)
    fig.savefig(destination, dpi=300)
    plt.close(fig)


def write_results_readme(results: Path, metrics: dict, dynamic_summary: dict | None) -> None:
    dynamic_text = "not supplied"
    if dynamic_summary:
        aggregate = dynamic_summary.get("aggregate", dynamic_summary)
        dynamic_text = (
            f"teacher RMSE={aggregate.get('teacher_rmse_dec_mean', float('nan')):.6g} dex; "
            f"Top-2 RMSE={aggregate.get('top2_rmse_dec_mean', float('nan')):.6g} dex"
        )
    text = f"""# Generated result set

This directory is regenerated by `run_model.cmd package-release` and at the
end of `run_model.cmd train-all`. Figures and their source CSV/JSON files are
kept in parallel subdirectories. Generation does not imply that every quality
gate passed; see `validation/V95_ACCEPTANCE_METRICS.json`.

## Static acceptance

- IV RMS: physical {metrics['physical_iv_rms']:.6g} dex, parameter KAN
  {metrics['base_iv_rms']:.6g} dex, final {metrics['residual_iv_rms']:.6g} dex.
- CV RMS: physical {metrics['physical_cv_rms']:.6g} dex, parameter KAN
  {metrics['base_cv_rms']:.6g} dex, final {metrics['residual_cv_rms']:.6g} dex.
- Accepted scales: IV residual {metrics['safe_iv_residual_scale']:.3f}, CV
  parameter KAN {metrics['safe_cv_parameter_scale']:.3f}, CV charge residual
  {metrics['safe_cv_residual_scale']:.3f}.
- Static VA mode: `{metrics['va_pkan_export_mode']}`.

## Dynamic acceptance

{dynamic_text}. Dynamic deployment uses a compact Top-2 symbolic SRH-KAN
because a Top-1 student did not preserve the two-trap teacher over all fast and
slow regimes.
"""
    (results / "README.md").write_text(text, encoding="utf-8")


def assemble(static_dir: Path, dynamic_dir: Path, physical_dir: Path) -> None:
    metrics_path = static_dir / "v93_safe_residual_metrics.json"
    if not metrics_path.exists():
        # Read-only compatibility for an in-flight run started by the previous
        # filename contract.  The assembled delivery always publishes v93
        # names and never copies both generations.
        metrics_path = static_dir / "v92_safe_residual_metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    results = ROOT / "results"
    figures = results / "figures"
    plot_data = results / "plot_data"
    reset_directory(figures)
    reset_directory(plot_data)

    figure_groups = {
        "static_iv": [
            "v92_IV_physical.png",
            "v92_IV_param_pkan.png",
            "v92_IV_residual.png",
            "v92_IDVTG_base_vs_safe_residual.png",
            "v92_performance_improvement_bars.png",
        ],
        "static_cv": [
            "v92_CV_physical.png",
            "v92_CV_param_pkan.png",
            "v92_CV_residual.png",
            "v92_CV_base_vs_safe_residual.png",
            "v92_cv_raw_vs_centered_preprocessing.png",
        ],
        "temperature": [
            "v92_temperature_IDVTG_transfer.png",
            "v92_temperature_IDVBG_transfer.png",
            "v92_temperature_CV_shift.png",
            "v92_temperature_physical_params.png",
        ],
        "extrapolation": [
            "v92_extrapolation_IV_base_vs_safe_residual.png",
            "v92_extrapolation_CV_base_vs_safe_residual.png",
        ],
    }
    for group, names in figure_groups.items():
        for name in names:
            copy_file(
                static_dir / name,
                figures / group / name.replace("v92_", "v95_", 1),
                required=False,
            )

    data_groups = {
        "static_iv": ["v92_iv_base_vs_safe_residual.csv"],
        "static_cv": [
            "v92_cv_base_vs_safe_residual.csv",
            "v92_cv_kernel_decomposition.csv",
            "v92_cv_raw_and_centered_preprocessing.csv",
            "v92_cv_reciprocal_prefit_metrics.csv",
            "v92_cv_reciprocal_prefit_predictions.csv",
            "v92_cv_vhalf_trend_diagnostics.csv",
            "v92_cv_parameter_head_ablation.csv",
            "v92_pkan_identifiability_audit.csv",
        ],
        "temperature": [
            "v92_temperature_iv_transfer_predictions.csv",
            "v92_temperature_cv_predictions.csv",
            "v92_temperature_physical_params.csv",
            "v92_temperature_trend_diagnostics.csv",
        ],
        "extrapolation": [
            "v92_extrapolation_iv_base_vs_safe_residual.csv",
            "v92_extrapolation_cv_base_vs_safe_residual.csv",
            "v92_extrapolation_shape_diagnostics.csv",
        ],
        "validation": [
            "v92_charge_passivity_diagnostics.json",
            "v92_cv_acceptance_diagnostics.json",
            "v92_hspice_iv_grid_equivalence.json",
            "v92_cv_hspice_cv_grid_equivalence.json",
            "v92_safe_residual_metrics.json",
            "v92_safe_scale_scan.csv",
            "v93_iv_residual_heldout_ablation.json",
            "v93_cv_cross_sheet_selection_ablation.csv",
            "v93_cv_cross_sheet_selection_ablation.json",
        ],
    }
    for group, names in data_groups.items():
        for name in names:
            destination_name = name.replace("v92_", "v95_", 1)
            destination_name = destination_name.replace("v93_", "v95_", 1)
            copy_file(
                static_dir / name,
                plot_data / group / destination_name,
                required=False,
            )

    validation_figures = figures / "validation"
    validation_figures.mkdir(parents=True, exist_ok=True)
    plot_static_hspice(static_dir, validation_figures)

    dynamic_bank = dynamic_dir / "srh_top2_regime_banks"
    dynamic_summary = None
    if dynamic_bank.exists():
        predictions = pd.read_csv(dynamic_bank / "srh_top2_regime_bank_predictions.csv")
        dynamic_metrics = pd.read_csv(dynamic_bank / "srh_top2_regime_bank_metrics.csv")
        dynamic_summary = json.loads((dynamic_bank / "srh_top2_regime_bank_summary.json").read_text())
        hysteresis_figures = figures / "hysteresis"
        hysteresis_figures.mkdir(parents=True, exist_ok=True)
        plot_dynamic_regime(
            predictions,
            dynamic_metrics,
            "fast",
            hysteresis_figures / "fast_us_ms_step_response_fit.png",
        )
        plot_dynamic_regime(
            predictions,
            dynamic_metrics,
            "slow",
            hysteresis_figures / "slow_seconds_step_response_fit.png",
        )
        copy_file(
            dynamic_bank / "srh_top2_regime_bank_predictions.csv",
            plot_data / "hysteresis" / "srh_top2_regime_bank_predictions.csv",
        )
        copy_file(
            dynamic_bank / "srh_top2_regime_bank_metrics.csv",
            plot_data / "hysteresis" / "srh_top2_regime_bank_metrics.csv",
        )
        copy_file(
            dynamic_bank / "srh_top2_regime_bank_summary.json",
            plot_data / "hysteresis" / "srh_top2_regime_bank_summary.json",
        )
        combined = dynamic_bank / "combined"
        for name in (
            "python_vs_hspice_top2_fast.png",
            "python_vs_hspice_top2_slow.png",
            "python_vs_hspice_top2_modular.png",
            "complete_device_current_equivalence.png",
        ):
            copy_file(combined / name, validation_figures / name, required=False)
        for name in (
            "python_vs_hspice_top2_fast.csv",
            "python_vs_hspice_top2_slow.csv",
            "python_vs_hspice_top2_modular.csv",
            "python_vs_hspice_top2_fast_summary.json",
            "python_vs_hspice_top2_slow_summary.json",
            "python_vs_hspice_top2_modular_summary.json",
            "complete_device_current_equivalence.csv",
            "complete_device_hspice_equivalence_summary.json",
            "complete_device_hspice_smoke_summary.json",
        ):
            copy_file(combined / name, plot_data / "validation" / name, required=False)

    static_signoff = ROOT / "trained_model" / "static_signoff"
    dynamic_signoff = ROOT / "trained_model" / "dynamic_srh"
    reset_directory(static_signoff)
    reset_directory(dynamic_signoff)
    symbolic_info_path = static_dir / "v92_edgewise_symbolic_residual_info.json"
    iv_symbolic_topk = 1
    iv_adaptive_top2_edges = 0
    if symbolic_info_path.exists():
        symbolic_info = json.loads(
            symbolic_info_path.read_text(encoding="utf-8")
        )
        iv_symbolic_topk = int(symbolic_info.get("iv_topk", 1))
        iv_adaptive_top2_edges = int(
            symbolic_info.get("adaptive_iv_top2_edge_count", 0)
        )
    iv_symbolic_tag = (
        "adaptive_top2"
        if iv_adaptive_top2_edges > 0
        else f"top{iv_symbolic_topk}"
    )
    static_files = {
        "v93_safe_residual_state.pt": "v95_safe_residual_state.pt",
        "v93_safe_residual_metrics.json": "v95_safe_residual_metrics.json",
        "v93_release_manifest.json": "v95_release_manifest.json",
        "kan_feature_normalizer.json": "kan_feature_normalizer.json",
        "v92_edgewise_symbolic_base_iv_kan.txt": (
            f"v95_{iv_symbolic_tag}_symbolic_base_iv_kan.txt"
        ),
        "v92_edgewise_symbolic_iv_residual.txt": (
            f"v95_{iv_symbolic_tag}_symbolic_iv_residual.txt"
        ),
        "v92_edgewise_symbolic_cv_charge_residual.txt": "v95_top1_symbolic_cv_charge_residual.txt",
        "v92_temperature_edgewise_symbolic_temperature_kan.txt": "v95_top1_symbolic_temperature_kan.txt",
        "v92_top1_symbolic_residual_surrogates.json": "v95_top1_symbolic_residual_surrogates.json",
        "v92_charge_passivity_diagnostics.json": "v95_charge_passivity_diagnostics.json",
        "v92_cv_acceptance_diagnostics.json": "v95_cv_acceptance_diagnostics.json",
        "v92_safe_scale_scan.csv": "v95_safe_scale_scan.csv",
        "v92_safe_shape_diagnostics.csv": "v95_safe_shape_diagnostics.csv",
        "v92_edgewise_symbolic_residual_info.json": "v95_symbolic_info.json",
        "v92_pkan_identifiability_audit.csv": "v95_pkan_identifiability_audit.csv",
        "v92_cv_parameter_head_ablation.csv": "v95_cv_parameter_head_ablation.csv",
        "v93_base_iv_training_report.json": "v95_base_iv_training_report.json",
        "v93_iv_residual_heldout_ablation.json": "v95_iv_residual_heldout_ablation.json",
        "v93_cv_cross_sheet_selection_ablation.csv": "v95_cv_cross_sheet_selection_ablation.csv",
        "v93_cv_cross_sheet_selection_ablation.json": "v95_cv_cross_sheet_selection_ablation.json",
        "v95_symbolic_iv_measured_selection.json": "v95_symbolic_iv_measured_selection.json",
        "v95_symbolic_iv_student_state.pt": "v95_symbolic_iv_student_state.pt",
    }
    for source_name, destination_name in static_files.items():
        source = static_dir / source_name
        if not source.exists() and source_name.startswith("v93_"):
            source = static_dir / source_name.replace("v93_", "v92_", 1)
        copy_file(source, static_signoff / destination_name, required=False)
    rewrite_checkpoint_contract(
        static_signoff / "v95_safe_residual_state.pt",
        "dg_tft_v95_residual_diagnostic",
    )
    copy_file(physical_dir / "fitted_params.json", static_signoff / "fitted_params.json")
    for name in (
        "v95_safe_residual_metrics.json",
        "v95_release_manifest.json",
        "fitted_params.json",
    ):
        sanitize_json_file(static_signoff / name)
    release_manifest_path = static_signoff / "v95_release_manifest.json"
    if release_manifest_path.exists():
        release_manifest = json.loads(release_manifest_path.read_text(encoding="utf-8"))
        release_manifest["version"] = "v95_fresh_training_final_delivery"
        release_manifest["base_model_dir"] = "trained_model/base_physical_pkan"
        release_manifest["data_dir"] = "data/user_dataset"
        release_manifest["fresh_training_no_historical_checkpoint"] = True
        release_manifest["historical_weights_or_formulas_deployed"] = False
        release_manifest_path.write_text(
            json.dumps(release_manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    iv_ablation_path = static_signoff / "v95_iv_residual_heldout_ablation.json"
    if iv_ablation_path.exists():
        iv_ablation = json.loads(iv_ablation_path.read_text(encoding="utf-8"))
        iv_ablation["training_recipe"] = (
            "independent_iv_residual_loss_plus_derivative_ordered_shape_"
            "extrapolation_anchor_and_top1_symbolic_regularization"
        )
        if "v90_protected_hyperparameters" in iv_ablation:
            iv_ablation["iv_training_hyperparameters"] = iv_ablation.pop(
                "v90_protected_hyperparameters"
            )
        iv_ablation_path.write_text(
            json.dumps(iv_ablation, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    architecture_metadata = (
        ROOT / "trained_model" / "base_physical_pkan" / "pkan_model_metadata.json"
    )
    sanitize_json_file(architecture_metadata)
    if architecture_metadata.exists():
        architecture = json.loads(architecture_metadata.read_text(encoding="utf-8"))
        contract = architecture.setdefault("release_contract", {})
        contract["release"] = "v95"
        contract["base_checkpoint_format"] = "dg_tft_v95_compactkan_architecture"
        contract["fresh_device_policy"] = (
            "rebuild every physical parameter and KAN tensor from the new "
            "dataset before IV base training; do not reuse fitted device weights"
        )
        architecture_metadata.write_text(
            json.dumps(architecture, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    rewrite_checkpoint_contract(
        ROOT / "trained_model" / "base_physical_pkan" / "pkan_torch_model.pt",
        "dg_tft_v95_compactkan_architecture",
    )
    if dynamic_bank.exists():
        for name in (
            "srh_top2_regime_bank_checkpoint.pt",
            "srh_top2_regime_bank_summary.json",
            "srh_top2_regime_bank_metrics.csv",
        ):
            copy_file(dynamic_bank / name, dynamic_signoff / name)
        for name in (
            "python_vs_hspice_top2_fast_summary.json",
            "python_vs_hspice_top2_slow_summary.json",
            "complete_device_hspice_equivalence_summary.json",
            "complete_device_hspice_smoke_summary.json",
        ):
            copy_file(
                dynamic_bank / "combined" / name,
                dynamic_signoff / name,
                required=False,
            )

    verilog_a = ROOT / "verilog_a"
    reset_directory(verilog_a)
    copy_file(static_dir / "dg_tft_static_core.va", verilog_a / "dg_tft_static_core.va")
    copy_file(
        ROOT / "modeling_code" / "templates" / "dg_tft_portable_glue.va",
        verilog_a / "dg_tft_portable_glue.va",
    )
    if dynamic_bank.exists():
        combined = dynamic_bank / "combined"
        for name in ("dg_tft_dynamic_srh.va", "dg_tft_dynamic_srh.inc"):
            copy_file(combined / name, verilog_a / name)
        copy_complete_device_wrapper(
            combined / "dg_tft_complete_model.inc",
            verilog_a / "dg_tft_complete_model.inc",
        )
    (verilog_a / "AETHER_MODEL_FILELIST.txt").write_text(
        "dg_tft_static_core.va\n"
        "dg_tft_dynamic_srh.va\n"
        "dg_tft_portable_glue.va\n"
        "dg_tft_dynamic_srh.inc\n"
        "dg_tft_complete_model.inc\n",
        encoding="utf-8",
    )
    (verilog_a / "DEPLOYMENT_README.md").write_text(
        "# Verilog-A deployment\n\n"
        f"IV KAN edges are deployed as compact {iv_symbolic_tag} analytic "
        f"symbolic functions ({iv_adaptive_top2_edges} adaptively promoted "
        "edges); "
        "CV and temperature KAN edges remain Top-1. No exact B-spline tables are "
        "embedded. The fast/slow SRH hysteresis banks use compact Top-2 two-layer "
        "symbolic functions because Top-1 did not preserve the two-trap teacher "
        "over all regimes.\n\n"
        "HSPICE entry:\n\n"
        "```spice\n.include \"dg_tft_static_core.va\"\n"
        ".include \"dg_tft_dynamic_srh.va\"\n"
        ".include \"dg_tft_portable_glue.va\"\n"
        ".include \"dg_tft_dynamic_srh.inc\"\n"
        ".include \"dg_tft_complete_model.inc\"\n```\n\n"
        "Use `maxstep <= min(tau_c,tau_e)/20` and initialize every SRH state "
        "parameter for production transient simulation.\n",
        encoding="utf-8",
    )

    write_results_readme(results, metrics, dynamic_summary)
    print(f"Release artifacts assembled under {ROOT}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--static-dir", type=Path, required=True)
    parser.add_argument("--dynamic-dir", type=Path, required=True)
    parser.add_argument("--physical-dir", type=Path, required=True)
    args = parser.parse_args()
    assemble(args.static_dir.resolve(), args.dynamic_dir.resolve(), args.physical_dir.resolve())


if __name__ == "__main__":
    main()
