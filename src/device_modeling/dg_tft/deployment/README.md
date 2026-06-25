# Deployment

The user-facing command is:

```powershell
python modeling_code/run_model.py export-va
```

Validated files are placed in the release-level `verilog_a` folder. For
HSPICE, the single entry is:

```spice
.include "dg_tft_complete_model.inc"
```

Keep all five files in the same folder. The complete include loads the static
and dynamic Verilog-A modules and defines subcircuit `dg_tft_complete_model`.

Files:

- `dg_tft_static_core.va`: static IV/CV and temperature-aware device core.
- `dg_tft_dynamic_srh.va`: dynamic two-state SRH module.
- `dg_tft_portable_glue.va`: simulator-neutral current and state glue.
- `dg_tft_dynamic_srh.inc`: symbolic dynamic parameter expressions.
- `dg_tft_complete_model.inc`: single HSPICE entry and integration wrapper.

For Empyrean/Aether or another Verilog-A-capable simulator, import the three
`.va` files as model sources.  The two `.inc` files are HSPICE/SPICE assembly
wrappers, not Verilog-A source files.  Follow
`verilog_a/AETHER_DEPLOYMENT_README.md` when the target simulator does not
accept HSPICE `.hdl` syntax.

For transient production simulation:

- explicitly initialize each trap-state node;
- use `maxstep <= min(tau_c, tau_e)/20`;
- use the same pre-bias and initial occupancy convention as the Python
  waveform simulation;
- distinguish intrinsic device capacitance from measurement-wrapper
  gain/offset and fixture parasitics.

The equivalence evidence is under `validation` and
`results/figures/hspice_python_equivalence`.
