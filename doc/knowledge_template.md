# {Stage Display Name} Knowledge

## 1. Global Context

### Pipeline Overview

The DG-TFT modeling pipeline has 7 stages:

```
data-prep → verifier → check-data → fit-physical → train-static → train-hysteresis → final-report
```

| Stage | Kind | Has Modeling Agent | Triggers Agent On |
|-------|------|--------------------|-------------------|
| data-prep | agent | No | — (always runs) |
| verifier | agent | No | — (always runs) |
| check-data | script | No (Agent A only) | Hard failure only |
| fit-physical | script | Yes (Agent B) | Hard failure / quality below threshold |
| train-static | script | Yes (Agent C) | Hard failure / quality below threshold |
| train-hysteresis | script | Yes (Agent D) | Hard failure / quality below threshold |
| final-report | noop | No (Agent E) | — (always runs at end) |

Script stages run CUDA Python subprocesses via `run_model.py <action>`. When a
script stage hard-fails (exit code ≠ 0) or produces quality below threshold, a
**modeling agent** (B/C/D) is spawned to adjust parameters and retry, fallback
upstream, or accept current quality.

Fallback is a linear one-hop chain upstream: D→C→B→A→Verifier→Preparer.

### {Stage Name} Position and Role

{Description of what this stage does, its kind (script/agent/noop), whether it
has a modeling agent, and what triggers agent intervention.}

{If this stage has a modeling agent ({agent_label}): what tools are available,
what INI section it controls, what decisions it can make.}

### Principles: {Underlying Theory}

{Physical/mathematical principles of what this stage does. What is being modeled?
What are the key physical quantities? What is the fundamental approach?
Not a textbook — operational knowledge for an LLM agent that needs to make
parameter tuning decisions.}

## 2. Upstream/Downstream

### Receives from {Upstream Stage}

{What files, directories, INI config this stage receives from upstream. What
does ok/fail of upstream mean for this stage?}

### Delivers to {Downstream Stage}

{What artifacts this stage produces. What the downstream stage depends on.
What happens if this stage fails or produces sub-threshold quality?}

### Fallback Chain

{If this stage has a modeling agent: what upstream stage it falls back to.
If this stage receives fallback from downstream: what to check.}

## 3. INI Parameters

{If no parameters: state explicitly "This stage has no adjustable INI parameters"
and explain why.}

{If has parameters: table with columns Parameter, Section, Type, Range, Default,
Description, Tuning Guidance. For each parameter explain: what it controls in
the algorithm, what happens when you increase/decrease it, what signals indicate
it needs adjustment.}

{Include stage timeout configuration.}

## 4. Quality Metrics

{If no metrics: state explicitly "This stage produces no quality metrics" and
explain what determines success instead.}

{If has metrics: table with columns Metric, Source File, Threshold Key, Meaning.
For each metric explain: what physical/modeling quantity it measures, what
"below threshold" means in practice (e.g., "IV model deviates >0.1 dex from
measurement"), what parameter typically affects this metric.}

## 5. Engine Behavior

{Step-by-step execution flow: what the subprocess does, in order.}

{Key subprocess invocation with actual command line.}

{What the engine script checks/computes, and in what order.}

{What files are read from, what files are written to.}

{Diagnostic signals: how to interpret exit codes, what each log file means,
what to look for when things go wrong.}

{Impact of parameters on execution flow: e.g., "increasing epochs extends runtime
linearly but may improve convergence."}

## 6. Failure Patterns

> This section will be populated from Zikang's Codex JSONL logs during the
> knowledge extraction phase.

### Error Classification (from diagnostics.py)

{Classification keywords for this stage from diagnostics.py, if any.}

### Agent Decision Framework

{If this stage has a modeling agent: decision matrix showing when to use
adjust_params vs fallback_to_previous vs accept_quality. Include specific
guidance: what error patterns suggest parameter adjustment, what patterns
suggest an upstream problem.}

### Diagnostic Tools Available

{Tools the agent can use: read_file, glob_file, subagent_progress, etc.
List specific files the agent should inspect first when diagnosing failures.}
