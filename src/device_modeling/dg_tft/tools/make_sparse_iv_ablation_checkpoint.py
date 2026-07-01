from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


IV_OUTPUTS = [
    "dpsi_V",
    "dVth_t_eff_V",
    "dVth_b_eff_V",
    "dlog_mu_dec",
    "dlog_theta_dec",
    "dlambda",
    "dlog_ileak_dec",
    "dlog_Cit_dec",
    "dpartition",
    "dpartition_inv",
    "dgamma_intergate",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create an isolated IV-output pruning ablation checkpoint."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--disable", required=True)
    args = parser.parse_args()

    disabled = [name.strip() for name in args.disable.split(",") if name.strip()]
    unknown = sorted(set(disabled) - set(IV_OUTPUTS))
    if unknown:
        raise ValueError(f"Unknown IV outputs: {unknown}")
    payload = torch.load(args.input, map_location="cpu", weights_only=False)
    state = payload["diag_state_dict"]
    out_range = state["base.kan.out_range"].clone()
    for name in disabled:
        out_range[IV_OUTPUTS.index(name)] = 0.0
    state["base.kan.out_range"] = out_range
    policy = dict(payload.get("output_policy", {}))
    policy["disabled_iv_pkan"] = sorted(disabled)
    payload["output_policy"] = policy
    payload["sparse_iv_ablation"] = {
        "disabled_outputs": sorted(disabled),
        "source_checkpoint": str(args.input),
        "note": "Isolated ablation only; not a production checkpoint until full-curve validation passes.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    print(json.dumps(payload["sparse_iv_ablation"], indent=2))


if __name__ == "__main__":
    main()
