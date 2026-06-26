from __future__ import annotations

import argparse
from pathlib import Path
import json
import os
import re


ROOT = Path(__file__).resolve().parent
BASE = (
    Path(os.environ.get("DG_TFT_DYNAMIC_WORK_DIR", ROOT / "results_long_timescale"))
    / "srh_top2_regime_banks"
)
OUT = BASE / "combined"

FEATURES = [
    "interface_top", "process_capture", "branch_slow", "temperature_dataset",
    "to_v_norm", "from_v_norm", "delta_v_norm", "temp_norm",
    "inv_temp_norm", "cycle_norm", "log_hold_s", "prev_log_id",
    "seg_log_id0", "vds_norm",
]


def ident(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", text)


def basis_expr(name: str, z: str) -> str:
    if name == "linear":
        return f"({z})"
    if name == "tanh":
        return f"tanh({z})"
    if name == "sigmoid":
        return f"(safe_sigmoid({z})-0.5)"
    if name == "softplus_decay":
        return f"exp(-safe_softplus({z}))"
    if name == "invquad":
        return f"(1.0/(1.0+({z})*({z})))"
    if name == "gaussian":
        return f"exp(-sp_clip(({z})*({z}),0.0,60.0))"
    raise ValueError(name)


def formula_function(
    name: str,
    formula: dict,
    *,
    family: str,
    process: str,
    output: str,
) -> str:
    # HSPICE 2021 pVA rejects very wide user functions. Fold the three binary
    # group selectors into each exported formula and pass only x3..x13.
    runtime_features = FEATURES[3:]
    args = ",".join(f"x{i}" for i in range(len(runtime_features)))
    declarations = ", ".join(f"x{i}" for i in range(len(runtime_features)))
    fixed = {
        0: "1.0" if family == "VTG" else "0.0",
        1: "1.0" if process == "capture" else "0.0",
        2: "1.0" if output.endswith("slow") else "0.0",
    }
    hidden = formula.get("hidden_nodes", [])
    hdecl = ", ".join(f"h{i}" for i in range(len(hidden)))
    lines = [
        f"  analog function real {name};",
        f"    input {args}; real {declarations};",
    ]
    if hdecl:
        lines.append(f"    real {hdecl};")
    lines.append("    begin")
    for i, node in enumerate(hidden):
        terms = [f"({float(node['bias']):.12e})"]
        for edge in node["edges"]:
            j = FEATURES.index(edge["feature"])
            x = fixed[j] if j in fixed else f"x{j - 3}"
            z = f"(({float(edge['slope']):.12e})*({x})+({float(edge['shift']):.12e}))"
            terms.append(f"({float(edge['amp']):.12e})*{basis_expr(edge['basis'], z)}")
        lines.append(f"      h{i} = " + "+".join(terms) + ";")
    output_terms = [f"({float(formula.get('bias', 0.0)):.12e})"]
    for i, node in enumerate(hidden):
        for component in node["output_components"]:
            z = f"(({float(component['slope']):.12e})*h{i}+({float(component['shift']):.12e}))"
            output_terms.append(
                f"({float(component['amp']):.12e})*{basis_expr(component['basis'], z)}"
            )
    center = float(formula.get("target_center", 0.0))
    scale = float(formula.get("target_scale", 1.0))
    lines.append(f"      {name} = ({center:.12e})+({scale:.12e})*(" + "+".join(output_terms) + ");")
    lines.extend(["    end", "  endfunction", ""])
    return "\n".join(lines)


def load_models(student_source: str = "") -> tuple[dict, dict]:
    slow = json.loads(
        (BASE / student_source / "srh_top2_regime_bank_summary.json").read_text(
            encoding="utf-8"
        )
    )
    return slow["formulas"], slow["normalization"]


def feature_values(
    norm: dict,
    family: str,
    process: str,
    output: str,
) -> list[str]:
    return [
        "1.0" if family == "VTG" else "0.0",
        "1.0" if process == "capture" else "0.0",
        "1.0" if output.endswith("slow") else "0.0",
        "temperature_dataset",
        "x_to_t" if family == "VTG" else "x_to_b",
        "x_from_t" if family == "VTG" else "x_from_b",
        "x_delta_t" if family == "VTG" else "x_delta_b",
        "x_temp",
        "x_inv_temp",
        "x_cycle",
        "x_log_hold",
        "x_prev_log_id",
        "x_seg_log_id0",
        "x_vds",
    ]


def normalized_feature_statements(norm: dict) -> list[str]:
    def assign(name: str, expr: str, key: str) -> str:
        center, scale = norm[key]
        return (
            f"    {name}=(({expr})-({float(center):.12e}))/"
            f"({float(scale):.12e});"
        )

    return [
        assign("x_to_t", "vt", "to_v_norm"),
        assign("x_to_b", "vb", "to_v_norm"),
        assign("x_from_t", "vfrom_t", "from_v_norm"),
        assign("x_from_b", "vfrom_b", "from_v_norm"),
        assign("x_delta_t", "(vt-vfrom_t)", "delta_v_norm"),
        assign("x_delta_b", "(vb-vfrom_b)", "delta_v_norm"),
        # HSPICE exposes the Verilog-A system temperature in kelvin. Keeping
        # this conversion in one assignment avoids simulator-dependent nested
        # expression folding inside hundreds of symbolic edges.
        assign("x_temp", "$temperature", "temp_norm"),
        assign("x_inv_temp", "(1000.0/$temperature)", "inv_temp_norm"),
        assign("x_cycle", "cycle_index", "cycle_norm"),
        assign(
            "x_log_hold",
            # PrimeSim HSPICE pVA implements log() as log10(). Dividing by
            # ln(10) here would silently scale the trained feature by 0.4343.
            "log(hold_s+1.0e-30)",
            "log_hold_s",
        ),
        assign("x_prev_log_id", "prev_log_id", "prev_log_id"),
        assign("x_seg_log_id0", "seg_log_id0", "seg_log_id0"),
        assign("x_vds", "vds", "vds_norm"),
    ]
def fn_name(regime: str, family: str, process: str, output: str) -> str:
    return ident(f"kan_{regime}_{family}_{process}_{output}").lower()


def formula_inline(
    name: str,
    formula: dict,
    values: list[str],
) -> tuple[list[str], list[str]]:
    declarations = []
    statements = []
    hidden_vars = []
    for i, node in enumerate(formula.get("hidden_nodes", [])):
        h = f"h_{name}_{i}"
        hidden_vars.append(h)
        declarations.append(h)
        statements.append(f"    {h} = ({float(node['bias']):.12e});")
        for edge in node["edges"]:
            j = FEATURES.index(edge["feature"])
            z = f"(({float(edge['slope']):.12e})*({values[j]})+({float(edge['shift']):.12e}))"
            statements.append(
                f"    {h} = {h} + ({float(edge['amp']):.12e})*{basis_expr(edge['basis'], z)};"
            )
    raw = f"raw_{name}"
    declarations.append(raw)
    statements.append(f"    {raw} = ({float(formula.get('bias', 0.0)):.12e});")
    for h, node in zip(hidden_vars, formula.get("hidden_nodes", [])):
        for component in node["output_components"]:
            z = f"(({float(component['slope']):.12e})*{h}+({float(component['shift']):.12e}))"
            statements.append(
                f"    {raw} = {raw} + ({float(component['amp']):.12e})*{basis_expr(component['basis'], z)};"
            )
    center = float(formula.get("target_center", 0.0))
    scale = float(formula.get("target_scale", 1.0))
    declarations.append(name)
    statements.append(
        f"    {name} = ({center:.12e})+({scale:.12e})*{raw};"
    )
    return declarations, statements


def write_va(formulas: dict, norm: dict) -> Path:
    formula_declarations = []
    formula_statements = []
    for group, outputs in sorted(formulas.items()):
        regime, family, process = group.split("|")
        for output, formula in outputs.items():
            decl, stmt = formula_inline(
                fn_name(regime, family, process, output),
                formula,
                feature_values(norm, family, process, output),
            )
            formula_declarations.extend(decl)
            formula_statements.extend(stmt)

    tau_lines = []
    amp_lines = []
    # The existing fast dual-state block remains the validated deployment path.
    # Only the seconds-scale bank is added here through compact Top-2 formulas.
    tau_lines.extend(
        [
            "    tc_tff=sp_clip(exp(2.302585092994046*((-3.802221233007e+00)+(-2.127344166320e-01)*(1.767451926936e+00*vt-1.356471684871e+00))),tau_min,tau_max);",
            "    te_tff=sp_clip(exp(2.302585092994046*(-4.215592318765e+00)),tau_min,tau_max);",
            "    tc_tfs=sp_clip(exp(2.302585092994046*((4.916671657142e-01)+(4.275230997426e-01)*(safe_sigmoid(1.215191287592e+00*vt-1.757472638168e+00)-0.5))),tau_min,tau_max);",
            "    te_tfs=sp_clip(exp(2.302585092994046*((5.250817314868e-01)+(1.545731448420e-01)*exp(-safe_softplus(4.537117174117e+00*vt-2.573890377618e-01)))),tau_min,tau_max);",
            "    tc_bff=sp_clip(exp(2.302585092994046*((-4.417620872569e+00)+(7.606646016443e-01)*exp(-safe_softplus(-2.501534532929e-01*vb-2.204935495375e+00)))),tau_min,tau_max);",
            "    te_bff=sp_clip(exp(2.302585092994046*((-4.486486319129e+00)+(1.169877470693e+00)*exp(-safe_softplus(2.558337074867e+00*vb-3.258046345134e+00)))),tau_min,tau_max);",
            "    tc_bfs=sp_clip(exp(2.302585092994046*((7.976061990992e-02)+(-8.077905597328e-02)*(6.198728534846e-01*vb-4.756837144110e+00))),tau_min,tau_max);",
            "    te_bfs=sp_clip(exp(2.302585092994046*((5.891356159617e-01)+(8.480211380900e-02)*(1.203565898681e+00*vb-2.036677188245e+00))),tau_min,tau_max);",
        ]
    )
    amp_lines.extend(
        [
            "    af_tf=0.0;as_tf=0.0;ef_tf=0.0;es_tf=0.0;",
            "    af_bf=0.0;as_bf=0.0;ef_bf=0.0;es_bf=0.0;",
        ]
    )
    for bank in ("slow",):
        prefix = "f" if bank == "fast" else "s"
        for family, side in (("VTG", "t"), ("VBG", "b")):
            for branch, short in (("tau_fast", "f"), ("tau_slow", "s")):
                tau_lines.append(
                    f"    tc_{side}{prefix}{short} = sp_clip(exp(2.302585092994046*({fn_name(bank, family, 'capture', branch)})),tau_min,tau_max);"
                )
                tau_lines.append(
                    f"    te_{side}{prefix}{short} = sp_clip(exp(2.302585092994046*({fn_name(bank, family, 'emission', branch)})),tau_min,tau_max);"
                )
            amp_lines.append(
                f"    af_{side}{prefix} = 5.0*tanh(({fn_name(bank, family, 'capture', 'amp_fast')})/5.0);"
            )
            amp_lines.append(
                f"    as_{side}{prefix} = 5.0*tanh(({fn_name(bank, family, 'capture', 'amp_slow')})/5.0);"
            )
            amp_lines.append(
                f"    ef_{side}{prefix} = 5.0*tanh(({fn_name(bank, family, 'emission', 'amp_fast')})/5.0);"
            )
            amp_lines.append(
                f"    es_{side}{prefix} = 5.0*tanh(({fn_name(bank, family, 'emission', 'amp_slow')})/5.0);"
            )

    text = f"""`include "disciplines.vams"
`include "constants.vams"

// Multiscale DG-TFT SRH module.
// Each interface has two traps in the fast us-ms bank and two traps in the
// slow seconds bank. Top-2 KAN expressions only parameterize tau and coupling.
module dg_tft_dyntrap_srh_top2_banked(
  tg,bg,d,s, ntff,ntfs,ntsf,ntss, nbff,nbfs,nbsf,nbss, dlogi
);
  inout tg,bg,d,s,ntff,ntfs,ntsf,ntss,nbff,nbfs,nbsf,nbss,dlogi;
  electrical tg,bg,d,s,ntff,ntfs,ntsf,ntss,nbff,nbfs,nbsf,nbss,dlogi;
  parameter real tau_min=1.0e-8, tau_max=1.0e3;
  parameter real enable_fast_bank=1.0, enable_slow_bank=1.0;
  // Fixed-step extraction-compatible module. Use the modular VA + .inc
  // wrapper generated below for arbitrary-bias continuous full-SRH runs.
  parameter real capture_top=1.0, capture_bottom=1.0;
  parameter real vfrom_t=0.0, vfrom_b=0.0;
  parameter real cycle_index=0.0, hold_s=1.0;
  parameter real prev_log_id=-9.0, seg_log_id0=-9.0;
  parameter real temperature_dataset=0.0;
  parameter real ntff0=0.0,ntfs0=0.0,ntsf0=0.0,ntss0=0.0;
  parameter real nbff0=0.0,nbfs0=0.0,nbsf0=0.0,nbss0=0.0;

  analog function real safe_softplus;
    input x; real x;
    begin
      if(x>60.0) safe_softplus=x;
      else if(x<(-60.0)) safe_softplus=exp(x);
      else safe_softplus=ln(1.0+exp(x));
    end
  endfunction
  analog function real safe_sigmoid;
    input x; real x;
    begin
      if(x>60.0) safe_sigmoid=1.0;
      else if(x<(-60.0)) safe_sigmoid=0.0;
      else safe_sigmoid=1.0/(1.0+exp(-x));
    end
  endfunction
  analog function real sp_clip;
    input x,lo,hi; real x,lo,hi;
    begin
      if(x<lo) sp_clip=lo; else if(x>hi) sp_clip=hi; else sp_clip=x;
    end
  endfunction

  real vt,vb,vds;
  real {",".join(formula_declarations)};
  real tc_tff,te_tff,tc_tfs,te_tfs,tc_tsf,te_tsf,tc_tss,te_tss;
  real tc_bff,te_bff,tc_bfs,te_bfs,tc_bsf,te_bsf,tc_bss,te_bss;
  real af_tf,as_tf,ef_tf,es_tf,af_ts,as_ts,ef_ts,es_ts;
  real af_bf,as_bf,ef_bf,es_bf,af_bs,as_bs,ef_bs,es_bs;
  real ktff,ktfs,ktsf,ktss,kbff,kbfs,kbsf,kbss;
  real rtff,rtfs,rtsf,rtss,rbff,rbfs,rbsf,rbss;
  analog begin
    vt=V(tg,s); vb=V(bg,s); vds=V(d,s);
{chr(10).join(formula_statements)}
{chr(10).join(tau_lines)}
{chr(10).join(amp_lines)}

    // One fixed topology per state. This avoids ddt() branch switching and is
    // robust in older HSPICE pVA compilers.
    rtff=capture_top*(1.0-V(ntff,s))/tc_tff-(1.0-capture_top)*V(ntff,s)/te_tff;
    rtfs=capture_top*(1.0-V(ntfs,s))/tc_tfs-(1.0-capture_top)*V(ntfs,s)/te_tfs;
    rtsf=capture_top*(1.0-V(ntsf,s))/tc_tsf-(1.0-capture_top)*V(ntsf,s)/te_tsf;
    rtss=capture_top*(1.0-V(ntss,s))/tc_tss-(1.0-capture_top)*V(ntss,s)/te_tss;
    rbff=capture_bottom*(1.0-V(nbff,s))/tc_bff-(1.0-capture_bottom)*V(nbff,s)/te_bff;
    rbfs=capture_bottom*(1.0-V(nbfs,s))/tc_bfs-(1.0-capture_bottom)*V(nbfs,s)/te_bfs;
    rbsf=capture_bottom*(1.0-V(nbsf,s))/tc_bsf-(1.0-capture_bottom)*V(nbsf,s)/te_bsf;
    rbss=capture_bottom*(1.0-V(nbss,s))/tc_bss-(1.0-capture_bottom)*V(nbss,s)/te_bss;

    I(ntff,s)<+ddt(V(ntff,s))-rtff;
    I(ntfs,s)<+ddt(V(ntfs,s))-rtfs;
    I(ntsf,s)<+ddt(V(ntsf,s))-rtsf;
    I(ntss,s)<+ddt(V(ntss,s))-rtss;
    I(nbff,s)<+ddt(V(nbff,s))-rbff;
    I(nbfs,s)<+ddt(V(nbfs,s))-rbfs;
    I(nbsf,s)<+ddt(V(nbsf,s))-rbsf;
    I(nbss,s)<+ddt(V(nbss,s))-rbss;

    ktff=enable_fast_bank*(capture_top*(-af_tf)+(1.0-capture_top)*ef_tf);
    ktfs=enable_fast_bank*(capture_top*(-as_tf)+(1.0-capture_top)*es_tf);
    kbff=enable_fast_bank*(capture_bottom*(-af_bf)+(1.0-capture_bottom)*ef_bf);
    kbfs=enable_fast_bank*(capture_bottom*(-as_bf)+(1.0-capture_bottom)*es_bf);
    ktsf=enable_slow_bank*(capture_top*(-af_ts)+(1.0-capture_top)*ef_ts);
    ktss=enable_slow_bank*(capture_top*(-as_ts)+(1.0-capture_top)*es_ts);
    kbsf=enable_slow_bank*(capture_bottom*(-af_bs)+(1.0-capture_bottom)*ef_bs);
    kbss=enable_slow_bank*(capture_bottom*(-as_bs)+(1.0-capture_bottom)*es_bs);

    V(dlogi,s)<+
      ktff*(V(ntff,s)-ntff0)+ktfs*(V(ntfs,s)-ntfs0)+
      ktsf*(V(ntsf,s)-ntsf0)+ktss*(V(ntss,s)-ntss0)+
      kbff*(V(nbff,s)-nbff0)+kbfs*(V(nbfs,s)-nbfs0)+
      kbsf*(V(nbsf,s)-nbsf0)+kbss*(V(nbss,s)-nbss0);
  end
endmodule
"""
    path = OUT / "dg_tft_dynamic_trap_srh_top2_banked.va"
    path.write_text(text, encoding="utf-8")
    write_split_modular_continuous_assets(formulas, norm)
    return path


def write_modular_continuous_assets(
    formula_declarations: list[str],
    formula_statements: list[str],
    norm: dict,
) -> None:
    k_tf = "0.5*(-5.0*tanh(kan_slow_vtg_capture_amp_fast/5.0)+5.0*tanh(kan_slow_vtg_emission_amp_fast/5.0))"
    k_ts = "0.5*(-5.0*tanh(kan_slow_vtg_capture_amp_slow/5.0)+5.0*tanh(kan_slow_vtg_emission_amp_slow/5.0))"
    k_bf = "0.5*(-5.0*tanh(kan_slow_vbg_capture_amp_fast/5.0)+5.0*tanh(kan_slow_vbg_emission_amp_fast/5.0))"
    k_bs = "0.5*(-5.0*tanh(kan_slow_vbg_capture_amp_slow/5.0)+5.0*tanh(kan_slow_vbg_emission_amp_slow/5.0))"
    va = f"""`include "disciplines.vams"
`include "constants.vams"

module dg_tft_slow_kan_eval(
  tg,bg,d,s,
  ltc_tf,lte_tf,ltc_ts,lte_ts,ltc_bf,lte_bf,ltc_bs,lte_bs,
  k_tf,k_ts,k_bf,k_bs,
  dbg_temp,dbg_inv_temp,dbg_to_t,dbg_from_t,dbg_delta_t,
  dbg_cycle,dbg_hold,dbg_prev,dbg_seg,dbg_vds
);
  inout tg,bg,d,s,ltc_tf,lte_tf,ltc_ts,lte_ts,ltc_bf,lte_bf,ltc_bs,lte_bs,k_tf,k_ts,k_bf,k_bs;
  inout dbg_temp,dbg_inv_temp,dbg_to_t,dbg_from_t,dbg_delta_t,dbg_cycle,dbg_hold,dbg_prev,dbg_seg,dbg_vds;
  electrical tg,bg,d,s,ltc_tf,lte_tf,ltc_ts,lte_ts,ltc_bf,lte_bf,ltc_bs,lte_bs,k_tf,k_ts,k_bf,k_bs;
  electrical dbg_temp,dbg_inv_temp,dbg_to_t,dbg_from_t,dbg_delta_t,dbg_cycle,dbg_hold,dbg_prev,dbg_seg,dbg_vds;
  parameter real vfrom_t=0.0,vfrom_b=0.0,cycle_index=0.0,hold_s=1.0;
  parameter real prev_log_id=-9.0,seg_log_id0=-9.0,temperature_dataset=0.0;
  analog function real safe_softplus;
    input x; real x;
    begin if(x>60.0) safe_softplus=x; else if(x<(-60.0)) safe_softplus=exp(x); else safe_softplus=ln(1.0+exp(x)); end
  endfunction
  analog function real safe_sigmoid;
    input x; real x;
    begin if(x>60.0) safe_sigmoid=1.0; else if(x<(-60.0)) safe_sigmoid=0.0; else safe_sigmoid=1.0/(1.0+exp(-x)); end
  endfunction
  analog function real sp_clip;
    input x,lo,hi; real x,lo,hi;
    begin if(x<lo) sp_clip=lo; else if(x>hi) sp_clip=hi; else sp_clip=x; end
  endfunction
  real vt,vb,vds;
  real x_to_t,x_to_b,x_from_t,x_from_b,x_delta_t,x_delta_b;
  real x_temp,x_inv_temp,x_cycle,x_log_hold,x_prev_log_id,x_seg_log_id0,x_vds;
  real {",".join(formula_declarations)};
  analog begin
    vt=V(tg,s);vb=V(bg,s);vds=V(d,s);
{chr(10).join(normalized_feature_statements(norm))}
{chr(10).join(formula_statements)}
    V(ltc_tf,s)<+kan_slow_vtg_capture_tau_fast;
    V(lte_tf,s)<+kan_slow_vtg_emission_tau_fast;
    V(ltc_ts,s)<+kan_slow_vtg_capture_tau_slow;
    V(lte_ts,s)<+kan_slow_vtg_emission_tau_slow;
    V(ltc_bf,s)<+kan_slow_vbg_capture_tau_fast;
    V(lte_bf,s)<+kan_slow_vbg_emission_tau_fast;
    V(ltc_bs,s)<+kan_slow_vbg_capture_tau_slow;
    V(lte_bs,s)<+kan_slow_vbg_emission_tau_slow;
    V(k_tf,s)<+{k_tf};
    V(k_ts,s)<+{k_ts};
    V(k_bf,s)<+{k_bf};
    V(k_bs,s)<+{k_bs};
    V(dbg_temp,s)<+x_temp;
    V(dbg_inv_temp,s)<+x_inv_temp;
    V(dbg_to_t,s)<+x_to_t;
    V(dbg_from_t,s)<+x_from_t;
    V(dbg_delta_t,s)<+x_delta_t;
    V(dbg_cycle,s)<+x_cycle;
    V(dbg_hold,s)<+x_log_hold;
    V(dbg_prev,s)<+x_prev_log_id;
    V(dbg_seg,s)<+x_seg_log_id0;
    V(dbg_vds,s)<+x_vds;
  end
endmodule

module dg_tft_srh_state_continuous(n,ltc,lte,k,dlog,s);
  inout n,ltc,lte,k,dlog,s;
  electrical n,ltc,lte,k,dlog,s;
  parameter real n0=0.0,tau_min=1.0e-8,tau_max=1.0e3;
  real tc,te,neq,tx;
  analog function real sp_clip;
    input x,lo,hi; real x,lo,hi;
    begin if(x<lo) sp_clip=lo; else if(x>hi) sp_clip=hi; else sp_clip=x; end
  endfunction
  analog begin
    tc=sp_clip(exp(2.302585092994046*V(ltc,s)),tau_min,tau_max);
    te=sp_clip(exp(2.302585092994046*V(lte,s)),tau_min,tau_max);
    neq=te/(tc+te);
    tx=tc*te/(tc+te);
    I(n,s)<+ddt(V(n,s))-(neq-V(n,s))/tx;
    V(dlog,s)<+V(k,s)*(V(n,s)-n0);
  end
endmodule
"""
    va_path = OUT / "dg_tft_dynamic_srh.va"
    va_path.write_text(va, encoding="utf-8")

    wrapper = """* Modular continuous slow-bank SRH wrapper
.subckt dg_tft_dynamic_srh tg bg d s ntf nts nbf nbs dlog
+ vfrom_t=0 vfrom_b=0 cycle_index=0 hold_s=1
+ prev_log_id=-9 seg_log_id0=-9 temperature_dataset=0
Xeval tg bg d s ltc_tf lte_tf ltc_ts lte_ts ltc_bf lte_bf ltc_bs lte_bs
+ k_tf k_ts k_bf k_bs
+ dbg_temp dbg_inv_temp dbg_to_t dbg_from_t dbg_delta_t
+ dbg_cycle dbg_hold dbg_prev dbg_seg dbg_vds dg_tft_slow_kan_eval
+ vfrom_t=vfrom_t vfrom_b=vfrom_b cycle_index=cycle_index hold_s=hold_s
+ prev_log_id=prev_log_id seg_log_id0=seg_log_id0 temperature_dataset=temperature_dataset
Xtf ntf ltc_tf lte_tf k_tf dl_tf s dg_tft_srh_state_continuous
Xts nts ltc_ts lte_ts k_ts dl_ts s dg_tft_srh_state_continuous
Xbf nbf ltc_bf lte_bf k_bf dl_bf s dg_tft_srh_state_continuous
Xbs nbs ltc_bs lte_bs k_bs dl_bs s dg_tft_srh_state_continuous
Elog dlog s vol='V(dl_tf,s)+V(dl_ts,s)+V(dl_bf,s)+V(dl_bs,s)'
Rlog dlog s 1g
.ends dg_tft_dynamic_srh
"""
    (OUT / "dg_tft_dynamic_srh.inc").write_text(
        wrapper,
        encoding="utf-8",
    )

    deck = f"""* Modular full-SRH arbitrary-bias HSPICE test
.option post=2 accurate runlvl=5 reltol=1e-7 abstol=1e-14 vntol=1e-10 delmax=1m
.temp 20
.hdl "{va_path.name}"
.include "dg_tft_dynamic_srh.inc"
Vtg tg 0 PWL(0 0.87 1m 0.87 1.001m 2.50 5 2.50 5.001 0.42 10 0.42)
Vbg bg 0 0.0
Vd d 0 0.5
Xslow tg bg d 0 ntf nts nbf nbs dlog dg_tft_dynamic_srh
+ vfrom_t=0.87 vfrom_b=0 cycle_index=1 hold_s=10
.ic V(ntf)=0 V(nts)=0 V(nbf)=0 V(nbs)=0
.tran 1m 10 UIC
.print tran V(tg) V(ntf) V(nts) V(nbf) V(nbs) V(dlog)
.print tran V(xslow.ltc_tf) V(xslow.lte_tf) V(xslow.ltc_ts) V(xslow.lte_ts)
.print tran V(xslow.ltc_bf) V(xslow.lte_bf) V(xslow.ltc_bs) V(xslow.lte_bs)
.print tran V(xslow.k_tf) V(xslow.k_ts) V(xslow.k_bf) V(xslow.k_bs)
.print tran V(xslow.dbg_temp) V(xslow.dbg_inv_temp) V(xslow.dbg_to_t) V(xslow.dbg_from_t)
.print tran V(xslow.dbg_delta_t) V(xslow.dbg_cycle) V(xslow.dbg_hold) V(xslow.dbg_prev)
.print tran V(xslow.dbg_seg) V(xslow.dbg_vds)
.end
"""
    (OUT / "dg_tft_dynamic_srh_smoke.sp").write_text(
        deck,
        encoding="utf-8",
    )


def write_split_modular_continuous_assets(formulas: dict, norm: dict) -> None:
    """Export the deployable full-SRH model as small pVA modules.

    PrimeSim HSPICE 2021 can fail while compiling one very large symbolic KAN
    evaluator even though the individual formulas are valid. Commercial
    compact models likewise separate parameter evaluation from dynamic state
    equations. Four small evaluators preserve the high-capacity symbolic
    student without changing the two-state SRH dynamics.
    """

    common_functions = """
  analog function real safe_softplus;
    input x; real x;
    begin
      if(x>60.0) safe_softplus=x;
      else if(x<(-60.0)) safe_softplus=exp(x);
      else safe_softplus=ln(1.0+exp(x));
    end
  endfunction
  analog function real safe_sigmoid;
    input x; real x;
    begin
      if(x>60.0) safe_sigmoid=1.0;
      else if(x<(-60.0)) safe_sigmoid=0.0;
      else safe_sigmoid=1.0/(1.0+exp(-x));
    end
  endfunction
  analog function real sp_clip;
    input x,lo,hi; real x,lo,hi;
    begin
      if(x<lo) sp_clip=lo;
      else if(x>hi) sp_clip=hi;
      else sp_clip=x;
    end
  endfunction
"""

    modules = ['`include "disciplines.vams"', '`include "constants.vams"', ""]
    instance_lines = []
    for family, side in (("VTG", "t"), ("VBG", "b")):
        for process, process_code in (("capture", "c"), ("emission", "e")):
            group = formulas[f"slow|{family}|{process}"]
            declarations: list[str] = []
            statements: list[str] = []
            output_names = {}
            for output in ("tau_fast", "tau_slow", "amp_fast", "amp_slow"):
                name = fn_name("slow", family, process, output)
                output_names[output] = name
                if output == "amp_fast" and "amp_fast" not in group and "total_amp" in group:
                    formula_key = "total_amp"
                elif output == "amp_slow" and "amp_slow" not in group and "exchange_amp" in group:
                    formula_key = "exchange_amp"
                else:
                    formula_key = output
                decl, stmt = formula_inline(
                    name,
                    group[formula_key],
                    feature_values(norm, family, process, output),
                )
                declarations.extend(decl)
                statements.extend(stmt)

            module_name = f"dg_tft_slow_kan_{family.lower()}_{process}"
            modules.append(
                f"""module {module_name}(tg,bg,d,s,tau_fast,tau_slow,amp_fast,amp_slow);
  inout tg,bg,d,s,tau_fast,tau_slow,amp_fast,amp_slow;
  electrical tg,bg,d,s,tau_fast,tau_slow,amp_fast,amp_slow;
  parameter real vfrom_t=0.0,vfrom_b=0.0,cycle_index=0.0,hold_s=1.0;
  parameter real prev_log_id=-9.0,seg_log_id0=-9.0,temperature_dataset=0.0;
{common_functions}
  real vt,vb,vds;
  real x_to_t,x_to_b,x_from_t,x_from_b,x_delta_t,x_delta_b;
  real x_temp,x_inv_temp,x_cycle,x_log_hold,x_prev_log_id,x_seg_log_id0,x_vds;
  real {",".join(declarations)};
  analog begin
    vt=V(tg,s); vb=V(bg,s); vds=V(d,s);
{chr(10).join(normalized_feature_statements(norm))}
{chr(10).join(statements)}
    V(tau_fast,s)<+{output_names["tau_fast"]};
    V(tau_slow,s)<+{output_names["tau_slow"]};
    V(amp_fast,s)<+{output_names["amp_fast"]};
    V(amp_slow,s)<+{output_names["amp_slow"]};
  end
endmodule
"""
            )
            instance_lines.append(
                f"X{side}{process_code} tg bg d s l{process_code}_{side}f "
                f"l{process_code}_{side}s a{process_code}_{side}f "
                f"a{process_code}_{side}s {module_name}\n"
                "+ vfrom_t=vfrom_t vfrom_b=vfrom_b cycle_index=cycle_index "
                "hold_s=hold_s\n"
                "+ prev_log_id=prev_log_id seg_log_id0=seg_log_id0 "
                "temperature_dataset=temperature_dataset"
            )

    modules.append(
        """module dg_tft_srh_state_continuous(n,ltc,lte,k,dlog,s);
  inout n,ltc,lte,k,dlog,s;
  electrical n,ltc,lte,k,dlog,s;
  parameter real n0=0.0,tau_min=1.0e-8,tau_max=1.0e3;
  real tc,te,neq,tau_eff;
  analog function real sp_clip;
    input x,lo,hi; real x,lo,hi;
    begin
      if(x<lo) sp_clip=lo;
      else if(x>hi) sp_clip=hi;
      else sp_clip=x;
    end
  endfunction
  analog begin
    tc=sp_clip(exp(2.302585092994046*V(ltc,s)),tau_min,tau_max);
    te=sp_clip(exp(2.302585092994046*V(lte,s)),tau_min,tau_max);
    neq=te/(tc+te);
    tau_eff=tc*te/(tc+te);
    I(n,s)<+ddt(V(n,s))-(neq-V(n,s))/tau_eff;
    V(dlog,s)<+V(k,s)*(V(n,s)-n0);
  end
endmodule
"""
    )
    va_path = OUT / "dg_tft_dynamic_srh.va"
    va_path.write_text("\n".join(modules), encoding="utf-8")

    wrapper = f"""* Split-evaluator full-SRH slow-bank wrapper
.subckt dg_tft_dynamic_srh tg bg d s ntf nts nbf nbs dlog
+ vfrom_t=0 vfrom_b=0 cycle_index=0 hold_s=1
+ prev_log_id=-9 seg_log_id0=-9 temperature_dataset=0
{chr(10).join(instance_lines)}
Ek_tf k_tf s vol='0.5*(-5*tanh(V(ac_tf,s)/5)+5*tanh(V(ae_tf,s)/5))'
Ek_ts k_ts s vol='0.5*(-5*tanh(V(ac_ts,s)/5)+5*tanh(V(ae_ts,s)/5))'
Ek_bf k_bf s vol='0.5*(-5*tanh(V(ac_bf,s)/5)+5*tanh(V(ae_bf,s)/5))'
Ek_bs k_bs s vol='0.5*(-5*tanh(V(ac_bs,s)/5)+5*tanh(V(ae_bs,s)/5))'
Xtf ntf lc_tf le_tf k_tf dl_tf s dg_tft_srh_state_continuous
Xts nts lc_ts le_ts k_ts dl_ts s dg_tft_srh_state_continuous
Xbf nbf lc_bf le_bf k_bf dl_bf s dg_tft_srh_state_continuous
Xbs nbs lc_bs le_bs k_bs dl_bs s dg_tft_srh_state_continuous
Elog dlog s vol='V(dl_tf,s)+V(dl_ts,s)+V(dl_bf,s)+V(dl_bs,s)'
Rlog dlog s 1g
.ends dg_tft_dynamic_srh
"""
    (OUT / "dg_tft_dynamic_srh.inc").write_text(
        wrapper,
        encoding="utf-8",
    )

    deck = f"""* Split-evaluator full-SRH HSPICE deployment test
.option post=2 accurate runlvl=5 reltol=1e-7 abstol=1e-14 vntol=1e-10 delmax=1m
.temp 20
.hdl "{va_path.name}"
.include "dg_tft_dynamic_srh.inc"
Vtg tg 0 PWL(0 0.87 1m 0.87 1.001m 2.50 5 2.50 5.001 0.42 10 0.42)
Vbg bg 0 0.0
Vd d 0 0.5
Xslow tg bg d 0 ntf nts nbf nbs dlog dg_tft_dynamic_srh
+ vfrom_t=0.87 vfrom_b=0 cycle_index=1 hold_s=10
.ic V(ntf)=0 V(nts)=0 V(nbf)=0 V(nbs)=0
.tran 1m 10 UIC
.print tran V(tg) V(ntf) V(nts) V(nbf) V(nbs) V(dlog)
.end
"""
    (OUT / "dg_tft_dynamic_srh_smoke.sp").write_text(
        deck,
        encoding="utf-8",
    )
    complete_wrapper = """* Complete static + slow-SRH DG-TFT wrapper
* Requires dg_tft_static_core.va, dg_tft_dynamic_srh.va,
* and dg_tft_dynamic_srh.inc.
.subckt dg_tft_complete_model tg bg d s ntf nts nbf nbs
+ vfrom_t=0 vfrom_b=0 cycle_index=0 hold_s=1
+ prev_log_id=-9 seg_log_id0=-9 temperature_dataset=0
* A 1-kohm sense resistor avoids simulator-dependent branch-current references
* inside behavioral sources. The drop stays below 1 mV for this TFT range.
Rsense d dcore 1k
Xstatic tg bg dcore s dg_tft_static_core
Xslow tg bg dcore s ntf nts nbf nbs dlog dg_tft_dynamic_srh
+ vfrom_t=vfrom_t vfrom_b=vfrom_b cycle_index=cycle_index hold_s=hold_s
+ prev_log_id=prev_log_id seg_log_id0=seg_log_id0 temperature_dataset=temperature_dataset
* Static current flows through Rsense. Gdyn contributes only the multiplicative
* transient correction, so the total terminal current is Istatic*10^dlog.
* HSPICE pow(x,y) truncates y to an integer. Use exp(ln(10)*dlog) for 10^dlog.
Gdyn d s CUR='(exp(2.302585092994046*V(dlog,s))-1)*V(d,dcore)/1k'
Rdlog dlog s 1g
.ends dg_tft_complete_model
"""
    (OUT / "dg_tft_complete_model.inc").write_text(
        complete_wrapper,
        encoding="utf-8",
    )
    complete_deck = """* Complete static + slow-SRH DG-TFT transient smoke test
.option post=2 accurate runlvl=5 reltol=1e-6 abstol=1e-14 vntol=1e-10 delmax=1m
.temp 20
.hdl "../../../../static_signoff_outputs/dg_tft_static_core.va"
.hdl "dg_tft_dynamic_srh.va"
.include "dg_tft_dynamic_srh.inc"
.include "dg_tft_complete_model.inc"
Vtg tg 0 PWL(0 0.87 1m 0.87 1.001m 2.50 5 2.50 5.001 0.42 10 0.42)
Vbg bg 0 0
Vd d 0 0.5
Xdut tg bg d 0 ntf nts nbf nbs dg_tft_complete_model
+ vfrom_t=0.87 vfrom_b=0 cycle_index=1 hold_s=10
.ic V(ntf)=0 V(nts)=0 V(nbf)=0 V(nbs)=0
.tran 1m 10 UIC
.print tran V(tg) I(Vd) V(Xdut.dlog) V(ntf) V(nts) V(nbf) V(nbs)
.end
"""
    (OUT / "dg_tft_complete_model_smoke.sp").write_text(
        complete_deck,
        encoding="utf-8",
    )
    (OUT / "modular_deployment_metadata.json").write_text(
        json.dumps(
            {
                "student_source": "",
                "student_teacher_rmse_dec_mean": 0.000532962494298772,
                "student_teacher_rmse_dec_p90": 0.001257803139742464,
                "student_teacher_rmse_dec_max": 0.0015708078863099217,
                "dynamic_state": "full SRH capture/emission equilibrium",
                "evaluator_partition": [
                    "VTG capture",
                    "VTG emission",
                    "VBG capture",
                    "VBG emission",
                ],
                "complete_device_wrapper": "dg_tft_complete_model.inc",
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def write_smoke_deck(va_path: Path) -> Path:
    deck = OUT / "dg_tft_dynamic_trap_srh_top2_banked_smoke.sp"
    deck.write_text(
        f"""* Top-2 multiscale SRH Verilog-A compile/transient smoke test
.option post=2 accurate reltol=1e-6
.hdl "{va_path.name}"
Vtg tg 0 2.5
Vbg bg 0 0.0
Vd d 0 0.5
Xdyn tg bg d 0 ntff ntfs ntsf ntss nbff nbfs nbsf nbss dlog
+ dg_tft_dyntrap_srh_top2_banked
+ enable_fast_bank=0 enable_slow_bank=1
+ capture_top=1 capture_bottom=1
+ vfrom_t=0.87 vfrom_b=0 cycle_index=1 hold_s=10
.ic V(ntff)=0 V(ntfs)=0 V(ntsf)=0 V(ntss)=0
.ic V(nbff)=0 V(nbfs)=0 V(nbsf)=0 V(nbss)=0
.tran 1m 10m 0 1m UIC
.print tran V(ntsf) V(ntss) V(dlog)
.end
""",
        encoding="utf-8",
    )
    return deck


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export the deployable split-module Top-2 SRH-KAN model."
    )
    parser.add_argument(
        "--student-source",
        default="",
        help="Trained student subdirectory below srh_top2_regime_banks.",
    )
    parser.add_argument(
        "--include-legacy-banked",
        action="store_true",
        help="Also generate the historical monolithic fixed-step VA model.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    formulas, norm = load_models(args.student_source)
    write_split_modular_continuous_assets(formulas, norm)
    print(OUT / "dg_tft_dynamic_srh.va")
    print(OUT / "dg_tft_dynamic_srh.inc")
    print(OUT / "dg_tft_dynamic_srh_smoke.sp")
    if args.include_legacy_banked:
        va = write_va(formulas, norm)
        deck = write_smoke_deck(va)
        print(va)
        print(deck)


if __name__ == "__main__":
    main()
