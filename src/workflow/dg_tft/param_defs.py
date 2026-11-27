# -*- coding: utf-8 -*-
"""
Header information
---------
Shanghai Jiao Tong University, School of Integrated Circuits, SMIL Lab
Author: Yu Huang
Create Date: 2026.7.1
Description: Parameter definition tables for DG-TFT workflow stages and
    auto-generation of adjust_params tool JSON schemas.

    Each stage (fit-physical / train-static / train-hysteresis) has a
    canonical parameter table that is the single source of truth for:
      - parameter name, type, bounds, default value, description
      - generating the OpenAI-compatible tool schema
      - driving the hard-guard validation pipe

    When the engine adds a new tunable parameter, only the table below
    needs updating — all downstream consumers (tool schema, validation,
    prompt injection) pick it up automatically.

Revision:
---------
2026.7.1        Yu Huang      1.0      Initial implementation (Task 3.2)
"""
from __future__ import annotations

from typing import Any


# ── Type aliases ─────────────────────────────────────────────────────

ParamDef = dict[str, Any]
"""A single parameter definition::

    {
        "type": "int" | "float" | "bool" | "string",
        "lo": lower_bound  (None for bool / string),
        "hi": upper_bound  (None for bool / string),
        "default": default_value,
        "desc": "Human-readable description for Agent context",
        "section": "ini_section_name",  # optional — defaults to the stage's primary section
    }
"""


# ═══════════════════════════════════════════════════════════════════════
#  Parameter tables — one per adjustable INI section
# ═══════════════════════════════════════════════════════════════════════

PHYSICAL_FIT_PARAMS: dict[str, ParamDef] = {
    "max_iv_points": {
        "type": "int",
        "lo": 5000,
        "hi": 200000,
        "default": 50000,
        "desc": (
            "Maximum number of IV data points to subsample for physical fitting. "
            "Lower values reduce runtime but may miss device characteristics. "
            "Higher values improve accuracy at the cost of longer fitting time."
        ),
    },
    "max_cv_points": {
        "type": "int",
        "lo": 10000,
        "hi": 500000,
        "default": 100000,
        "desc": (
            "Maximum number of CV data points to subsample for physical fitting. "
            "Lower values reduce runtime; higher values capture more capacitance detail."
        ),
    },
    "max_nfev_scale": {
        "type": "float",
        "lo": 0.5,
        "hi": 5.0,
        "default": 1.0,
        "desc": (
            "Scale factor for maximum number of function evaluations (NFEV) "
            "during physical kernel parameter extraction. "
            "Values > 1.0 allow more optimization iterations per sub-problem. "
            "Increase if fitting returns poor IV curve match; decrease to speed up."
        ),
    },

    # ── [preprocessing] — shared measurement-stage alignment ──
    "iv_stage_alignment": {
        "type": "string",
        "lo": None,
        "hi": None,
        "default": "auto",
        "enum": ["off", "auto"],
        "section": "preprocessing",
        "desc": (
            "IV measurement-stage alignment mode. "
            "'auto' — automatically detect and correct per-source transfer-voltage shifts. "
            "'off' — no alignment. "
            "Set 'off' if data is known to be pre-aligned."
        ),
    },
    "iv_stage_max_shift_V": {
        "type": "float",
        "lo": 0.1,
        "hi": 2.0,
        "default": 0.75,
        "section": "preprocessing",
        "desc": (
            "Maximum allowed transfer-voltage shift per source (V) for IV stage alignment. "
            "Lower values are more conservative. "
            "Increase only if measurement has documented large offsets."
        ),
    },
    "iv_stage_min_matches": {
        "type": "int",
        "lo": 1,
        "hi": 10,
        "default": 2,
        "section": "preprocessing",
        "desc": (
            "Minimum number of matched bias pairs required before applying a stage shift. "
            "Higher values require more evidence before alignment. "
            "Reduce if alignment is not triggering when expected."
        ),
    },
}


FULL_MODEL_PARAMS: dict[str, ParamDef] = {
    "epochs": {
        "type": "int",
        "lo": 100,
        "hi": 5000,
        "default": 700,
        "desc": (
            "Total training epochs for the full static model (IV + CV). "
            "More epochs improve fit but increase training cost. "
            "Combine with early_stop_patience so training stops when improvement plateaus."
        ),
    },
    "learning_rate": {
        "type": "float",
        "lo": 0.0001,
        "hi": 0.01,
        "default": 0.0012,
        "desc": (
            "Learning rate for the Adam optimizer during full model training. "
            "Lower values give more stable convergence; higher values may overshoot. "
            "Typical range: 0.0005–0.005."
        ),
    },
    "batch_iv": {
        "type": "int",
        "lo": 512,
        "hi": 16384,
        "default": 4096,
        "desc": (
            "Batch size for IV training samples. "
            "Larger batches provide more stable gradient estimates but require more GPU memory. "
            "Reduce if OOM; increase if gradient noise is excessive."
        ),
    },
    "batch_cv": {
        "type": "int",
        "lo": 64,
        "hi": 4096,
        "default": 512,
        "desc": (
            "Batch size for CV training samples. "
            "Typically smaller than batch_iv due to higher per-sample memory cost. "
            "Reduce if OOM; increase if gradient noise is excessive."
        ),
    },
    "cv_weight": {
        "type": "float",
        "lo": 1.0,
        "hi": 50.0,
        "default": 10.0,
        "desc": (
            "Weight multiplier for CV loss term relative to IV loss in the combined loss. "
            "Higher values force the model to fit CV curves more closely. "
            "Increase if CV RMS is too high; decrease if IV RMS degrades."
        ),
    },
    "cv_residual_level": {
        "type": "int",
        "lo": 0,
        "hi": 2,
        "default": 1,
        "desc": (
            "Complexity level of the CV residual correction sub-network. "
            "0 = disabled. 1 = shared broadened-DOS + asymmetric. 2 = shared + asymmetric + inter-gate reciprocal. "
            "Higher levels add more learnable parameters. "
            "Increase if CV fitting needs more flexibility. Max=2 (engine argparse restriction)."
        ),
    },
    "enable_cv_residual": {
        "type": "bool",
        "lo": None,
        "hi": None,
        "default": True,
        "desc": (
            "Whether to enable the CV residual learning branch. "
            "When True, a small network learns the CV IV→CV residual correction. "
            "Usually should remain True."
        ),
    },
    "finetune_base_pkan": {
        "type": "bool",
        "lo": None,
        "hi": None,
        "default": False,
        "desc": (
            "Whether to fine-tune the base PKAN weights during full model training. "
            "When True, physical-kernel-fitted PKAN layers continue learning. "
            "When False (v96 default), freeze the physical kernel and only train the residual path "
            "so that CV gradient corrections do not erode the accepted IV accuracy."
        ),
    },
    "early_stop_patience": {
        "type": "int",
        "lo": 2,
        "hi": 30,
        "default": 5,
        "desc": (
            "Number of consecutive epochs with no validation improvement before early stopping. "
            "Smaller values abort training sooner; larger values allow more exploration. "
            "Increase if training is stopped prematurely."
        ),
    },
    "early_stop_min_epochs": {
        "type": "int",
        "lo": 50,
        "hi": 500,
        "default": 220,
        "desc": (
            "Minimum number of epochs before early stopping is allowed to activate. "
            "Ensures the model has enough training before early-stop logic kicks in. "
            "Increase if model is under-trained; decrease to allow earlier stopping."
        ),
    },
    "symbolic_training_weight": {
        "type": "float",
        "lo": 0.0,
        "hi": 0.1,
        "default": 0.05,
        "desc": (
            "Weight for the symbolic training loss term. "
            "Controls how strongly the model is regularized towards symbolic interpretability. "
            "Higher values produce more interpretable equations at the cost of fitting accuracy."
        ),
    },
    "print_every": {
        "type": "int",
        "lo": 1,
        "hi": 200,
        "default": 10,
        "section": "full_model",
        "desc": (
            "Print training loss/metrics every N epochs. "
            "Larger values reduce log noise; set to 1 for full visibility."
        ),
    },

    # ── [full_model] — base IV PKAN training ──
    "base_iv_epochs": {
        "type": "int",
        "lo": 100,
        "hi": 5000,
        "default": 700,
        "section": "full_model",
        "desc": (
            "Epochs for the base IV PKAN training phase (runs before residual/CV training). "
            "More epochs improve the current-device base KAN quality. "
            "Increase if the base IV PKAN is undertrained."
        ),
    },
    "base_iv_restarts": {
        "type": "int",
        "lo": 1,
        "hi": 10,
        "default": 1,
        "section": "full_model",
        "desc": (
            "Number of restarts for base IV PKAN training. "
            "Each restart uses a different random seed. "
            "Increase if base IV training converges to poor local minima."
        ),
    },
    "base_iv_learning_rate": {
        "type": "float",
        "lo": 0.0001,
        "hi": 0.01,
        "default": 0.002,
        "section": "full_model",
        "desc": (
            "Learning rate for the base IV PKAN training phase. "
            "Lower values improve stability; higher values speed up convergence. "
            "Adjust if base IV loss oscillates or plateaus early."
        ),
    },
    "base_iv_eval_every": {
        "type": "int",
        "lo": 5,
        "hi": 100,
        "default": 20,
        "section": "full_model",
        "desc": (
            "Evaluate base IV PKAN metrics every N epochs. "
            "Smaller values provide finer progress tracking. "
            "Increase to reduce eval overhead."
        ),
    },
    "base_iv_top1_symbolic_weight": {
        "type": "float",
        "lo": 0.0,
        "hi": 1.0,
        "default": 0.35,
        "section": "full_model",
        "desc": (
            "Weight of the Top-1 compatibility regularizer during numerical base IV training. "
            "Higher values favor symbolically representable edge functions in the learned numerical model."
        ),
    },
    "base_iv_top1_sparsity_weight": {
        "type": "float",
        "lo": 0.0,
        "hi": 0.1,
        "default": 0.001,
        "section": "full_model",
        "desc": (
            "Sparsity penalty weight for base IV Top-1 symbolic selection. "
            "Higher values push towards fewer active symbolic edges. "
            "Increase if exported equations are too complex."
        ),
    },
    "base_iv_top1_selection_weight": {
        "type": "float",
        "lo": 0.0,
        "hi": 0.5,
        "default": 0.08,
        "section": "full_model",
        "desc": (
            "Weight of Top-1 compatibility in numerical base IV checkpoint selection. "
            "The selection score is numeric_score plus this weight times compatibility loss."
        ),
    },
    "base_iv_max_top1_compatibility": {
        "type": "float",
        "lo": 0.0,
        "hi": 0.1,
        "default": 0.015,
        "section": "full_model",
        "desc": (
            "Diagnostic reference for base IV Top-1 compatibility loss. "
            "The base training report marks this check diagnostic_only. "
            "Complete-curve current metrics determine base IV acceptance."
        ),
    },

    # ── [full_model] v95 — IV residual stage ──
    "iv_residual_stage_epochs": {
        "type": "int",
        "lo": 100,
        "hi": 5000,
        "default": 700,
        "section": "full_model",
        "desc": (
            "Epochs for the held-out IV residual training stage. "
            "This stage trains the residual KAN on held-out IV data. "
            "Increase if residual IV fit is poor."
        ),
    },
    "iv_residual_stage_learning_rate": {
        "type": "float",
        "lo": 0.0001,
        "hi": 0.01,
        "default": 0.002,
        "section": "full_model",
        "desc": (
            "Learning rate for the IV residual stage. "
            "Controls the optimization step size for held-out residual fitting. "
            "Adjust if residual training oscillates or converges too slowly."
        ),
    },
    "iv_residual_stage_linear_weight": {
        "type": "float",
        "lo": 0.0,
        "hi": 1.0,
        "default": 0.25,
        "section": "full_model",
        "desc": (
            "Linear term weight in the IV residual stage loss. "
            "Higher values emphasize matching the absolute curve values. "
            "Increase if the residual model misses the baseline level."
        ),
    },
    "iv_residual_stage_deriv_curves": {
        "type": "int",
        "lo": 1,
        "hi": 100,
        "default": 36,
        "section": "full_model",
        "desc": (
            "Number of derivative curves to supervise in IV residual training. "
            "More curves enforce tighter shape consistency. "
            "Increase if residual predictions have unrealistic curvature."
        ),
    },
    "iv_residual_stage_shape_period": {
        "type": "int",
        "lo": 1,
        "hi": 20,
        "default": 3,
        "section": "full_model",
        "desc": (
            "Shape evaluation period during IV residual training. "
            "Smaller values trigger shape-loss more frequently. "
            "Increase to reduce shape-matching overhead."
        ),
    },

    # ── [full_model] v95 — LR multipliers & training schedule ──
    "base_pkan_lr_mult": {
        "type": "float",
        "lo": 0.01,
        "hi": 10.0,
        "default": 1.0,
        "section": "full_model",
        "desc": (
            "Base PKAN learning-rate multiplier applied during full model training. "
            ">1.0 makes the base KAN learn faster; <1.0 freezes it more. "
            "Reduce if base KAN overfits to the current device data."
        ),
    },
    "warm_start_base_pkan_lr_mult": {
        "type": "float",
        "lo": 0.001,
        "hi": 1.0,
        "default": 0.035,
        "section": "full_model",
        "desc": (
            "Initial learning-rate multiplier for base PKAN during warm-start. "
            "Small values protect the pretrained KAN from early destruction. "
            "Increase if the base PKAN needs more initial adaptation."
        ),
    },
    "eval_every": {
        "type": "int",
        "lo": 5,
        "hi": 200,
        "default": 40,
        "section": "full_model",
        "desc": (
            "Evaluate full model metrics every N epochs. "
            "Smaller values give finer progress tracking but increase overhead. "
            "Set higher if training time is dominated by evaluation."
        ),
    },
    "shape_eval_period": {
        "type": "int",
        "lo": 1,
        "hi": 20,
        "default": 5,
        "section": "full_model",
        "desc": (
            "Period (in epochs) for shape evaluation during full model training. "
            "Smaller values check shape quality more frequently. "
            "Increase to reduce shape-eval overhead."
        ),
    },
    "shape_early_period": {
        "type": "int",
        "lo": 1,
        "hi": 50,
        "default": 20,
        "section": "full_model",
        "desc": (
            "Shape evaluation period during the early training phase. "
            "Larger values reduce early-epoch shape-checking overhead. "
            "Decrease if shape quality needs early monitoring."
        ),
    },
    "shape_middle_period": {
        "type": "int",
        "lo": 1,
        "hi": 50,
        "default": 10,
        "section": "full_model",
        "desc": (
            "Shape evaluation period during the middle training phase. "
            "Intermediate shape-checking frequency between early and late phases. "
            "Adjust to balance shape quality monitoring with training speed."
        ),
    },
    "staged_loss_schedule": {
        "type": "bool",
        "lo": None,
        "hi": None,
        "default": True,
        "section": "full_model",
        "desc": (
            "Enable staged loss-weight schedule during full model training. "
            "When True, loss weights evolve across training phases. "
            "Disable only for debugging or if loss weights should remain static."
        ),
    },
    "weight_decay": {
        "type": "float",
        "lo": 1e-8,
        "hi": 1e-3,
        "default": 1e-6,
        "section": "full_model",
        "desc": (
            "AdamW weight decay (L2 regularization) for full model training. "
            "Higher values penalize large weights more strongly. "
            "Increase if model overfits the training data."
        ),
    },
    "grad_clip": {
        "type": "float",
        "lo": 0.1,
        "hi": 50.0,
        "default": 5.0,
        "section": "full_model",
        "desc": (
            "Maximum gradient norm for gradient clipping. "
            "Prevents exploding gradients during training. "
            "Decrease if loss spikes occur; increase if training is too slow."
        ),
    },
    "lr_eta_min_frac": {
        "type": "float",
        "lo": 0.001,
        "hi": 0.5,
        "default": 0.02,
        "section": "full_model",
        "desc": (
            "Minimum learning rate as a fraction of initial LR "
            "(CosineAnnealingLR scheduler). "
            "Lower values allow the scheduler to decay further. "
            "Increase if training converges too quickly to a suboptimal minimum."
        ),
    },
    "cv_train_subset": {
        "type": "string",
        "lo": None,
        "hi": None,
        "default": "primary_gate_cv",
        "enum": ["all", "representative", "primary_gate_cv"],
        "section": "full_model",
        "desc": (
            "CV training data subset selection. "
            "'primary_gate_cv' — use only the primary gate CV sheets (CBG, CTG, CGG). "
            "'all' — include all gate capacitance sheets including VDS. "
            "'representative' — select representative CV series. "
            "Use 'all' if secondary gates contain useful information."
        ),
    },

    # ── [full_model] — training-stage control flags and masking ──
    "reset_base_from_physical_fit": {
        "type": "bool",
        "lo": None,
        "hi": None,
        "default": True,
        "section": "full_model",
        "desc": (
            "Whether to rebuild the PKAN from the physical kernel fitted_params.json. "
            "When True, creates a fresh current-device physical+PKAN model. "
            "When False, resume from existing PKAN checkpoint (warm-start). "
            "Set False to fine-tune an existing checkpoint."
        ),
    },
    "train_base_iv_pkan": {
        "type": "bool",
        "lo": None,
        "hi": None,
        "default": True,
        "section": "full_model",
        "desc": (
            "Whether to train the base IV-PKAN on current-device data. "
            "When True, runs the base IV PKAN training phase before residual/CV training. "
            "When False, skip base IV training (use pre-trained weights only)."
        ),
    },
    "train_iv_residual_stage": {
        "type": "bool",
        "lo": None,
        "hi": None,
        "default": True,
        "section": "full_model",
        "desc": (
            "Whether to train the held-out IV residual stage. "
            "When True, trains the diag.iv_resid CompactKAN on held-out IV data. "
            "When False, skip IV residual training (rely on PKAN corrections only)."
        ),
    },
    "freeze_iv_residual_after_stage": {
        "type": "bool",
        "lo": None,
        "hi": None,
        "default": True,
        "section": "full_model",
        "desc": (
            "Whether to freeze the IV residual KAN after its dedicated stage. "
            "When True, IV residual weights are frozen during joint CV training. "
            "When False, joint-finetune the IV residual alongside CV (risk IV degradation)."
        ),
    },
    "iv_residual_holdout_stride": {
        "type": "int",
        "lo": 1,
        "hi": 20,
        "default": 5,
        "section": "full_model",
        "desc": (
            "Stride for cross-condition holdout split in IV residual training. "
            "Data points every N steps are held out for validation. "
            "Lower values = larger holdout set, more conservative residual evaluation."
        ),
    },
    "iv_residual_off_anchor_weight": {
        "type": "float",
        "lo": 0.0,
        "hi": 20.0,
        "default": 2.0,
        "section": "full_model",
        "desc": (
            "Weight for the off-state anchor loss in IV residual training. "
            "Higher values enforce accurate deep-off current prediction. "
            "Increase if the residual model produces unrealistic off-state leakage."
        ),
    },
    "disabled_iv_pkan": {
        "type": "string",
        "lo": None,
        "hi": None,
        "default": "",
        "section": "full_model",
        "desc": (
            "Comma-separated list of IV-PKAN output names to disable (set to zero). "
            "Empty string means all outputs active. "
            "Use to selectively remove problematic PKAN correction heads. "
            "Example: 'dlog_rs_fwd,dpartition_inv'"
        ),
    },
    "disabled_cv_pkan": {
        "type": "string",
        "lo": None,
        "hi": None,
        "default": "",
        "section": "full_model",
        "desc": (
            "Comma-separated list of CV-PKAN output names to disable (set to zero). "
            "Empty string means all outputs active (subject to engine DEFAULT_DISABLED_CV_PKAN). "
            "Additional heads can be disabled here beyond engine defaults."
        ),
    },
    "disabled_cv_charge_pkan": {
        "type": "string",
        "lo": None,
        "hi": None,
        "default": "",
        "section": "full_model",
        "desc": (
            "Comma-separated list of CV charge residual output names to disable. "
            "Empty string means all outputs active (subject to engine CV_RELEASE_ACTIVE_NAMES). "
            "Use to selectively disable specific CV residual branches."
        ),
    },

    # ── [base_pkan] — KAN hyper-parameters (read during full model training) ──
    "epochs_shared": {
        "type": "int",
        "lo": 100,
        "hi": 3000,
        "default": 500,
        "section": "base_pkan",
        "desc": (
            "Shared KAN training epochs for the base PKAN network. "
            "More epochs improve the symbolic representation quality. "
            "Increase if symbolic formulas look under-fitted."
        ),
    },
    "epochs_iv": {
        "type": "int",
        "lo": 100,
        "hi": 3000,
        "default": 700,
        "section": "base_pkan",
        "desc": (
            "IV-specific KAN training epochs. "
            "Controls how many epochs the KAN trains specifically on IV data. "
            "Increase if IV symbolic fit is poor."
        ),
    },
    "epochs_cv": {
        "type": "int",
        "lo": 100,
        "hi": 3000,
        "default": 700,
        "section": "base_pkan",
        "desc": (
            "CV-specific KAN training epochs. "
            "Controls how many epochs the KAN trains specifically on CV data. "
            "Increase if CV symbolic fit is poor."
        ),
    },
    "kan_hidden": {
        "type": "int",
        "lo": 4,
        "hi": 32,
        "default": 12,
        "section": "base_pkan",
        "desc": (
            "KAN hidden layer size. "
            "Larger values increase model capacity for learning complex device physics. "
            "Increase if symbolic formulas are too simple to capture device behavior."
        ),
    },
    "kan_grid": {
        "type": "int",
        "lo": 2,
        "hi": 20,
        "default": 5,
        "section": "base_pkan",
        "desc": (
            "KAN grid resolution. "
            "Higher values provide finer spline granularity. "
            "Increase for devices with sharp non-linear transitions."
        ),
    },
    "kan_k": {
        "type": "int",
        "lo": 2,
        "hi": 5,
        "default": 3,
        "section": "base_pkan",
        "desc": (
            "KAN spline degree/order (k). "
            "Higher-order splines are smoother but may oscillate. "
            "k=3 (cubic) is standard; increase only if fitting requires higher-order continuity."
        ),
    },
    "batch_iv_kan": {
        "type": "int",
        "lo": 256,
        "hi": 16384,
        "default": 4096,
        "section": "base_pkan",
        "desc": (
            "Batch size for IV KAN training ([base_pkan] section, separate from full_model.batch_iv). "
            "Larger batches give more stable spline fitting. "
            "Reduce if OOM during KAN training."
        ),
    },
    "batch_cv_kan": {
        "type": "int",
        "lo": 64,
        "hi": 4096,
        "default": 512,
        "section": "base_pkan",
        "desc": (
            "Batch size for CV KAN training ([base_pkan] section, separate from full_model.batch_cv). "
            "Larger batches give more stable spline fitting. "
            "Reduce if OOM during KAN training."
        ),
    },

    # ── [temperature] — temperature KAN control ──
    "temperature_mode": {
        "type": "string",
        "lo": None,
        "hi": None,
        "default": "auto",
        "enum": ["auto", "on", "off"],
        "section": "temperature",
        "desc": (
            "Temperature KAN mode. "
            "'auto' — enable when temperature span ≥ minimum_span_K. "
            "'on' — force enable. "
            "'off' — disable entirely. "
            "Use 'on' if the dataset has temperature data but auto-detection fails."
        ),
    },
    "minimum_span_K": {
        "type": "float",
        "lo": 1.0,
        "hi": 100.0,
        "default": 20.0,
        "section": "temperature",
        "desc": (
            "Minimum temperature span (Kelvin) required for auto mode to enable temperature KAN. "
            "Lower values allow temperature KAN with narrower temperature sweeps. "
            "Decrease if you have a small temperature range but want temperature-aware modeling."
        ),
    },
    "kan_scale": {
        "type": "float",
        "lo": 0.1,
        "hi": 1.0,
        "default": 0.35,
        "section": "temperature",
        "desc": (
            "Temperature KAN scaling factor. "
            "Controls the contribution weight of temperature-dependent terms. "
            "Increase to make temperature effects more prominent."
        ),
    },
    "temperature_kan_hidden": {
        "type": "int",
        "lo": 4,
        "hi": 32,
        "default": 8,
        "section": "temperature",
        "ini_key": "kan_hidden",
        "desc": (
            "Hidden layer size for the temperature KAN sub-network. "
            "Larger values increase capacity for complex temperature-dependence. "
            "Increase if temperature trends are poorly captured."
        ),
    },
    "temperature_kan_grid": {
        "type": "int",
        "lo": 2,
        "hi": 20,
        "default": 3,
        "section": "temperature",
        "ini_key": "kan_grid",
        "desc": (
            "Grid resolution for temperature KAN B-splines. "
            "Higher values provide finer temperature interpolation. "
            "Increase for devices with sharp temperature-dependent transitions."
        ),
    },
    "temperature_kan_k": {
        "type": "int",
        "lo": 2,
        "hi": 5,
        "default": 3,
        "section": "temperature",
        "ini_key": "kan_k",
        "desc": (
            "Spline degree for temperature KAN B-splines. "
            "Higher-order splines are smoother but may oscillate. "
            "k=3 (cubic) is standard; increase only for higher-order continuity."
        ),
    },
}

# ── [symbolic] — symbolic export parameters ──
SYMBOLIC_PARAMS: dict[str, ParamDef] = {
    "edge_steps": {
        "type": "int",
        "lo": 10,
        "hi": 2000,
        "default": 180,
        "section": "symbolic",
        "desc": (
            "Number of optimization steps for edgewise symbolic replacement. "
            "More steps allow finer symbolic formula fitting. "
            "Increase when edge fitting remains under-optimized at the step limit."
        ),
    },
    "edge_topk": {
        "type": "int",
        "lo": 1,
        "hi": 3,
        "default": 1,
        "section": "symbolic",
        "desc": (
            "Initial symbolic edge multiplicity. The current engine CLI accepts 1. "
            "IV candidate promotion is controlled by iv_adaptive_top2_mse."
        ),
    },
    "iv_edge_topk": {
        "type": "int",
        "lo": 1,
        "hi": 3,
        "default": 1,
        "section": "symbolic",
        "desc": (
            "Initial symbolic edge multiplicity for IV. The current engine CLI accepts 1. "
            "IV candidate promotion is controlled by iv_adaptive_top2_mse."
        ),
    },
    "iv_adaptive_top2_mse": {
        "type": "float",
        "lo": 0.0,
        "hi": 0.1,
        "default": 0.0015,
        "section": "symbolic",
        "desc": (
            "Top-1 edge MSE threshold for attempting an IV Top-2 fit. "
            "Zero disables promotion. Among positive thresholds, lower values admit more candidates. "
            "A candidate is adopted when its Top-2 MSE is below 0.90 times its Top-1 MSE. "
            "The export report records the actual promoted edge count."
        ),
    },
    "iv_calibration_max_terms": {
        "type": "int",
        "lo": 1,
        "hi": 100,
        "default": 24,
        "section": "symbolic",
        "desc": (
            "Upper bound on IV terminal calibration terms. The current basis has 24 terms; "
            "the effective bound is clamped to 2 through 24. "
            "The engine selects the smallest candidate within 3 percent of the best validation score. "
            "The export report records candidate scores and selected_terms."
        ),
    },
    "edge_finetune_steps": {
        "type": "int",
        "lo": 10,
        "hi": 2000,
        "default": 80,
        "section": "symbolic",
        "desc": (
            "Training steps for edge-level symbolic fine-tuning. "
            "More steps improve per-edge symbolic accuracy. "
            "Increase if edgewise replacement MSE is marginally above threshold."
        ),
    },
    "network_finetune_steps": {
        "type": "int",
        "lo": 10,
        "hi": 5000,
        "default": 360,
        "section": "symbolic",
        "desc": (
            "Training steps for full-network symbolic fine-tuning. "
            "After edgewise replacement, the complete symbolic network is jointly refined. "
            "Increase if network-level symbolic observables degrade after edge replacement."
        ),
    },
    "device_iv_finetune_steps": {
        "type": "int",
        "lo": 100,
        "hi": 10000,
        "default": 3000,
        "section": "symbolic",
        "desc": (
            "Training steps for device-specific IV symbolic calibration. "
            "The symbolic network matches the numeric teacher current and slopes. "
            "Increase if device-level IV fidelity is poor after symbolic export."
        ),
    },
    "device_iv_measured_steps": {
        "type": "int",
        "lo": 100,
        "hi": 20000,
        "default": 5000,
        "section": "symbolic",
        "desc": (
            "Training steps for measured-data IV symbolic calibration. "
            "Fine-tunes symbolic formulas directly against measured IV curves. "
            "Increase when measured-curve refinement is still improving at the step limit."
        ),
    },
    # Independent stopping policies for the two IV symbolic stages.
    "device_iv_finetune_eval_period": {
        "type": "int", "lo": 1, "hi": 1000, "default": 100,
        "section": "symbolic",
        "desc": "IV symbolic teacher current and slope calibration: Optimizer steps between fixed validation checks. Patience counts checks, not steps.",
    },
    "device_iv_finetune_patience": {
        "type": "int", "lo": 0, "hi": 100, "default": 6,
        "section": "symbolic",
        "desc": "IV symbolic teacher current and slope calibration: Consecutive validation checks without significant improvement before stopping; 0 disables early stopping.",
    },
    "device_iv_finetune_min_steps": {
        "type": "int", "lo": 0, "hi": 20000, "default": 500,
        "section": "symbolic",
        "desc": "IV symbolic teacher current and slope calibration: Minimum optimizer steps before early stopping can trigger; maximum step budget still applies.",
    },
    "device_iv_finetune_min_delta": {
        "type": "float", "lo": 0.0, "hi": 1.0, "default": 0.0001,
        "section": "symbolic",
        "desc": "IV symbolic teacher current and slope calibration: Absolute validation-score improvement required to reset patience; the true best model is always retained.",
    },
    "device_iv_measured_eval_period": {
        "type": "int", "lo": 1, "hi": 1000, "default": 50,
        "section": "symbolic",
        "desc": "IV symbolic measured-curve refinement: Optimizer steps between fixed validation checks. Patience counts checks, not steps.",
    },
    "device_iv_measured_patience": {
        "type": "int", "lo": 0, "hi": 100, "default": 6,
        "section": "symbolic",
        "desc": "IV symbolic measured-curve refinement: Consecutive validation checks without significant improvement before stopping; 0 disables early stopping.",
    },
    "device_iv_measured_min_steps": {
        "type": "int", "lo": 0, "hi": 20000, "default": 300,
        "section": "symbolic",
        "desc": "IV symbolic measured-curve refinement: Minimum optimizer steps before early stopping can trigger; maximum step budget still applies.",
    },
    "device_iv_measured_min_delta": {
        "type": "float", "lo": 0.0, "hi": 1.0, "default": 0.0001,
        "section": "symbolic",
        "desc": "IV symbolic measured-curve refinement: Absolute validation-score improvement required to reset patience; the true best model is always retained.",
    },
    "device_cv_finetune_steps": {
        "type": "int",
        "lo": 10,
        "hi": 5000,
        "default": 650,
        "section": "symbolic",
        "desc": (
            "Training steps for device-specific CV symbolic calibration. "
            "The symbolic CV model is fine-tuned on measured CV data. "
            "Increase if exported CV predictions have systematic errors."
        ),
    },
}

# ── [validation] — safety/quality scan parameters ──
VALIDATION_PARAMS: dict[str, ParamDef] = {
    "safe_scale_steps": {
        "type": "int",
        "lo": 2,
        "hi": 50,
        "default": 8,
        "section": "validation",
        "desc": (
            "Number of scale steps in the safety scan for residual output scales. "
            "More steps produce finer-grained safety envelope discovery. "
            "Increase if the safe scale is coarsely estimated."
        ),
    },
    "joint_cv_safe_scale_search": {
        "type": "bool",
        "lo": None,
        "hi": None,
        "default": False,
        "section": "validation",
        "desc": (
            "Enable 2D joint (parameter_scale, residual_scale) safety scan for CV. "
            "When True, runs a coarse grid search for CV scale combinations. "
            "When False, scan CV parameter and residual scales independently. "
            "Set True if CV passivity gate fails with independent scales."
        ),
    },
}


HYSTERESIS_PARAMS: dict[str, ParamDef] = {
    "enabled": {
        "type": "bool",
        "lo": None,
        "hi": None,
        "default": True,
        "desc": (
            "Whether to run the hysteresis training stage at all. "
            "Set False to skip hysteresis (e.g., no I-t data available)."
        ),
    },
    "regime": {
        "type": "string",
        "lo": None,
        "hi": None,
        "default": "all",
        "enum": ["all", "fast", "slow"],
        "desc": (
            "Target regime for hysteresis training. "
            "'all' — train both fast and slow SRH banks. "
            "'fast' — fast-regime only (segment duration < 50 ms). "
            "'slow' — slow-regime only (segment duration >= 50 ms). "
            "Use 'all' unless specific regime data is problematic."
        ),
    },
    "epochs": {
        "type": "int",
        "lo": 200,
        "hi": 5000,
        "default": 1800,
        "desc": (
            "Training epochs for the hysteresis capture/emission network. "
            "Hysteresis training typically requires more epochs than static fitting. "
            "Increase if dynamic behavior is not well captured."
        ),
    },
    "hidden_capture": {
        "type": "int",
        "lo": 4,
        "hi": 32,
        "default": 16,
        "desc": (
            "Number of hidden units in the capture network (non-linear transform). "
            "Larger networks can model more complex trap dynamics. "
            "Increase if hysteresis waveform shape is poorly reproduced."
        ),
    },
    "hidden_emission": {
        "type": "int",
        "lo": 4,
        "hi": 32,
        "default": 20,
        "desc": (
            "Number of hidden units in the emission network (full non-linear KAN, same TwoLayerTop1KAN architecture as capture). "
            "Larger networks improve time-constant resolution. "
            "Increase if transient response is inaccurate."
        ),
    },
    "amp_parameterization": {
        "type": "string",
        "lo": None,
        "hi": None,
        "default": "independent",
        "enum": ["independent", "conservative"],
        "desc": (
            "Amplitude parameterization style for the hysteresis model. "
            "'independent' — legacy v90 per-amplitude head (the engine default). "
            "'conservative' — bounded total+exchange form, less prone to oscillation. "
            "Use 'conservative' for deployable charge-conserving models."
        ),
    },


    # ── [hysteresis] v95 — fast/slow regime split ──
    "fast_restarts": {
        "type": "int",
        "lo": 1,
        "hi": 10,
        "default": 1,
        "section": "hysteresis",
        "desc": (
            "Number of random restarts for fast-transient SRH bank training. "
            "More restarts improve robustness at the cost of training time. "
            "Increase if fast-regime fitting quality is inconsistent across runs."
        ),
    },
    "slow_restarts": {
        "type": "int",
        "lo": 1,
        "hi": 10,
        "default": 2,
        "section": "hysteresis",
        "desc": (
            "Number of random restarts for slow-transient SRH bank training. "
            "Slow transients typically benefit from more restarts than fast. "
            "Increase if slow-regime amplitude/decay is poorly captured."
        ),
    },
    "fast_amp_parameterization": {
        "type": "string",
        "lo": None,
        "hi": None,
        "default": "conservative",
        "enum": ["independent", "conservative"],
        "section": "hysteresis",
        "desc": (
            "Amplitude parameterization for the fast-transient SRH bank. "
            "'conservative' — bounded total/exchange form, less prone to oscillation. "
            "'independent' — legacy v90 per-amplitude head. "
            "Use 'conservative' for stability."
        ),
    },
    "slow_amp_parameterization": {
        "type": "string",
        "lo": None,
        "hi": None,
        "default": "independent",
        "enum": ["independent", "conservative"],
        "section": "hysteresis",
        "desc": (
            "Amplitude parameterization for the slow-transient SRH bank. "
            "'independent' — legacy v90 per-amplitude head (protected baseline). "
            "'conservative' — bounded total/exchange form. "
            "Use 'conservative' if slow-bank amplitudes oscillate."
        ),
    },
    "fast_optimizer_policy": {
        "type": "string",
        "lo": None,
        "hi": None,
        "default": "optimized",
        "enum": ["optimized", "v90"],
        "section": "hysteresis",
        "desc": (
            "Optimizer policy for fast-transient SRH bank training. "
            "'optimized' — bounded schedule with faster convergence. "
            "'v90' — legacy fixed-AdamW full-budget policy. "
            "Fall back to 'v90' if optimized policy shows instability."
        ),
    },
    "slow_optimizer_policy": {
        "type": "string",
        "lo": None,
        "hi": None,
        "default": "v90",
        "enum": ["optimized", "v90"],
        "section": "hysteresis",
        "desc": (
            "Optimizer policy for slow-transient SRH bank training. "
            "'v90' — legacy fixed-AdamW (protected baseline). "
            "'optimized' — faster convergence schedule. "
            "Use 'optimized' only if slow-bank training time is a bottleneck."
        ),
    },
}


# ── Master lookup: stage → ([ini_sections], param_table) ────────────

from .schema import StageName

# Merge symbolic and validation params into the full-model table so they appear in
# the adjust_params tool schema for Agent C (train-static).
FULL_MODEL_PARAMS.update(SYMBOLIC_PARAMS)
FULL_MODEL_PARAMS.update(VALIDATION_PARAMS)

_STAGE_PARAM_MAP: dict[str, tuple[list[str], dict[str, ParamDef]]] = {
    StageName.FIT_PHYSICAL: (["physical_fit", "preprocessing"], PHYSICAL_FIT_PARAMS),
    StageName.TRAIN_STATIC: (["full_model", "base_pkan", "temperature", "symbolic", "validation"], FULL_MODEL_PARAMS),
    StageName.TRAIN_HYSTERESIS: (["hysteresis"], HYSTERESIS_PARAMS),
}


# ═══════════════════════════════════════════════════════════════════════
#  JSON Schema auto-generation → adjust_params tool def
# ═══════════════════════════════════════════════════════════════════════

def params_to_tool_schema(
    param_table: dict[str, ParamDef],
    tool_name: str = "adjust_params",
    stage_label: str = "",
) -> dict[str, Any]:
    """Auto-generate an OpenAI-compatible tool schema from a parameter table.

    The generated schema exposes each parameter as an optional property with
    its type, description, and numeric bounds (when applicable).

    Args:
        param_table: {param_name → ParamDef} mapping.
        tool_name: Tool name for the schema (default ``"adjust_params"``).
        stage_label: Human-readable stage name injected into the description.

    Returns:
        A dict matching the ``{"type": "function", "function": {...}}`` format.
    """
    label = stage_label or "current"
    properties: dict[str, dict[str, Any]] = {}
    for name, pdef in param_table.items():
        prop: dict[str, Any] = {}
        json_type = _param_type_to_json_type(pdef["type"])
        prop["type"] = json_type if not isinstance(json_type, list) else json_type[0]
        prop["description"] = pdef["desc"]
        # Numeric bounds
        if pdef.get("lo") is not None:
            prop["minimum"] = pdef["lo"]
        if pdef.get("hi") is not None:
            prop["maximum"] = pdef["hi"]
        # Enum constraint (string / bool)
        if "enum" in pdef:
            prop["enum"] = pdef["enum"]
        properties[name] = prop

    return {
        "type": "function",
        "function": {
            "name": tool_name,
            "description": (
                f"Adjust tunable hyper-parameters for the {label} stage. "
                "Only specify parameters you want to change — omit the rest. "
                "Each value must be within the documented bounds. "
                "Provide a brief reasoning summary."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    **properties,
                    "reasoning": {
                        "type": "string",
                        "description": (
                            "Brief explanation of why these adjustments are made, "
                            "referencing observed metrics or failure symptoms."
                        ),
                    },
                },
                "required": ["reasoning"],
                "additionalProperties": False,
            },
        },
    }


def get_stage_param_table(stage_name: str, *, task_mode: str = "iv_cv") -> dict[str, ParamDef] | None:
    """Use the same applicability filter for the tool schema and hard guard."""
    entry = _STAGE_PARAM_MAP.get(stage_name)
    if entry is None:
        return None
    table = entry[1]
    if task_mode != "iv_only":
        return table
    # CV-specific options consistently use a separate ``cv`` name component.
    return {name: definition for name, definition in table.items() if "cv" not in name.split("_")}


def get_static_task_mode(config_path: str) -> str:
    """Read input-derived scope, with backwards compatibility for older INIs."""
    import configparser
    import os
    from pathlib import Path

    cfg = configparser.ConfigParser()
    cfg.read(config_path, encoding="utf-8")
    data_dir = cfg.get("paths", "data_dir", fallback="")
    if data_dir:
        data_path = Path(os.path.expandvars(os.path.expanduser(data_dir)))
        if not data_path.is_absolute():
            data_path = Path(config_path).parent / data_path
        if data_path.is_dir():
            return "iv_cv" if (data_path / "CV_train.xlsx").exists() else "iv_only"
    return cfg.get("runtime", "static_task_mode", fallback="iv_cv")


def get_stage_param_schema(stage_name: str, *, task_mode: str = "iv_cv") -> dict[str, Any] | None:
    """Get the auto-generated tool schema for a given stage.

    Returns None if the stage has no adjustable parameters (e.g. ``check-data``,
    ``final-report``).
    """
    param_table = get_stage_param_table(stage_name, task_mode=task_mode)
    if param_table is None:
        return None
    return params_to_tool_schema(
        param_table,
        tool_name="adjust_params",
        stage_label=stage_name,
    )


def get_stage_sections(stage_name: str) -> list[str] | None:
    """Return the list of INI section names a stage may write to.

    Returns None if the stage has no adjustable parameters.
    """
    entry = _STAGE_PARAM_MAP.get(stage_name)
    if entry is None:
        return None
    return entry[0]


def get_stage_section(param_table: dict[str, ParamDef]) -> str | None:
    """Reverse-lookup: find the primary INI section name for a param table."""
    for _stage, (sections, pt) in _STAGE_PARAM_MAP.items():
        if pt is param_table:
            return sections[0]
    return None


# ── Internal helpers ─────────────────────────────────────────────────

def _param_type_to_json_type(t: str) -> str | list[str]:
    """Map internal type label to JSON Schema type.

    ``bool`` → ``"string"`` with ``enum: ["yes", "no"]`` to match INI convention.
    """
    mapping: dict[str, str | list[str]] = {
        "int": "integer",
        "float": "number",
        "bool": "boolean",
        "string": "string",
    }
    return mapping.get(t, "string")


# ═══════════════════════════════════════════════════════════════════════
#  Soft guard — self-check prompt snippet for Agent system prompt
# ═══════════════════════════════════════════════════════════════════════

_SELF_CHECK_PROMPT: str = (
    "**Before calling `adjust_params`, self-check your proposed values:**\n"
    "1. Bounds — are all values within the documented `minimum`/`maximum`?\n"
    "2. Physical sense — would a domain expert consider these values reasonable "
    "(e.g., epochs=50000 is excessive; lr=1.0 will diverge)?\n"
    "3. Interaction — do any parameter pairs conflict "
    "(e.g., very high learning_rate with very small early_stop_patience may "
    "prevent convergence; very small batch_iv with very large hidden units "
    "may cause OOM)?\n"
    "4. Incrementality — prefer small, incremental changes over extreme jumps. "
    "Doubling epochs is drastic; +25% is usually safer.\n"
    "5. If uncertain, consult the expert knowledge documents for guidance "
    "on reasonable ranges for this device technology.\n"
)


def get_soft_guard_prompt() -> str:
    """Return a self-check prompt snippet for Agent system prompts.

    Inject this into the modeling Agent's system prompt so it performs
    a reasonableness self-audit before emitting ``adjust_params`` calls.
    """
    return _SELF_CHECK_PROMPT
