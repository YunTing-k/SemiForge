from __future__ import annotations

import configparser
import json
from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[2]
HISTORICAL_TAGS = ("v90", "v91", "v92", "v93", "v94")


def main() -> int:
    errors: list[str] = []
    forbidden_names = []
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(ROOT)
        if any(part in {".venv", ".runtime_cache", "__pycache__"} for part in relative.parts):
            continue
        if any(tag in part.lower() for part in relative.parts for tag in HISTORICAL_TAGS):
            forbidden_names.append(relative.as_posix())
    if forbidden_names:
        errors.append(f"historical-version filenames present: {forbidden_names[:20]}")

    required = [
        ROOT / "modeling_code" / "config" / "model_config.ini",
        ROOT / "trained_model" / "static_signoff" / "v95_safe_residual_state.pt",
        ROOT / "trained_model" / "static_signoff" / "v95_symbolic_info.json",
        ROOT / "trained_model" / "static_signoff" / "v95_symbolic_iv_student_state.pt",
        ROOT / "trained_model" / "dynamic_srh" / "srh_top2_regime_bank_checkpoint.pt",
        ROOT / "validation" / "V95_ACCEPTANCE_METRICS.json",
        ROOT / "verilog_a" / "dg_tft_static_core.va",
        ROOT / "verilog_a" / "dg_tft_dynamic_srh.va",
    ]
    missing = [str(path.relative_to(ROOT)) for path in required if not path.is_file()]
    if missing:
        errors.append(f"missing required files: {missing}")

    cfg = configparser.ConfigParser()
    cfg.read(ROOT / "modeling_code" / "config" / "model_config.ini", encoding="utf-8")
    if cfg.get("symbolic", "iv_edge_topk", fallback="") != "1":
        errors.append("IV symbolic base topology is not Top-1")
    if cfg.get("symbolic", "edge_topk", fallback="") != "1":
        errors.append("CV/temperature symbolic topology is not Top-1")
    if not cfg.getboolean("full_model", "reset_base_from_physical_fit", fallback=False):
        errors.append("fresh-device physical reset is disabled")
    if cfg.getfloat("full_model", "base_pkan_lr_mult", fallback=0.0) < 0.5:
        errors.append("fresh-device base-PKAN LR multiplier is still warm-start sized")

    symbolic_info_path = (
        ROOT / "trained_model" / "static_signoff" / "v95_symbolic_info.json"
    )
    if symbolic_info_path.is_file():
        symbolic_info = json.loads(
            symbolic_info_path.read_text(encoding="utf-8")
        )
        adaptive_edges = int(
            symbolic_info.get("adaptive_iv_top2_edge_count", 0)
        )
        formula_tag = "adaptive_top2" if adaptive_edges else "top1"
        for name in (
            f"v95_{formula_tag}_symbolic_base_iv_kan.txt",
            f"v95_{formula_tag}_symbolic_iv_residual.txt",
        ):
            if not (
                ROOT / "trained_model" / "static_signoff" / name
            ).is_file():
                errors.append(f"missing symbolic formula: {name}")

    acceptance_path = ROOT / "validation" / "V95_ACCEPTANCE_METRICS.json"
    if acceptance_path.is_file():
        acceptance = json.loads(acceptance_path.read_text(encoding="utf-8"))
        if acceptance.get("delivery_status") != "accepted":
            errors.append(
                f"delivery status is {acceptance.get('delivery_status')!r}, not 'accepted'"
            )

    for relative, expected in (
        (
            Path("trained_model/base_physical_pkan/pkan_torch_model.pt"),
            "dg_tft_v95_compactkan_architecture",
        ),
        (
            Path("trained_model/static_signoff/v95_safe_residual_state.pt"),
            "dg_tft_v95_residual_diagnostic",
        ),
    ):
        path = ROOT / relative
        if not path.is_file():
            continue
        payload = torch.load(path, map_location="cpu", weights_only=False)
        actual = payload.get("checkpoint_schema", {}).get("format")
        if actual != expected:
            errors.append(f"{relative.as_posix()} schema={actual!r}, expected={expected!r}")

    report = {
        "status": "pass" if not errors else "fail",
        "root": str(ROOT),
        "historical_filename_count": len(forbidden_names),
        "required_file_count": len(required),
        "errors": errors,
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
