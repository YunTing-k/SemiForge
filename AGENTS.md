# AGENTS.md — SemiForge Agent Instructions

## On session start or after compact

1. **Read `doc/TODO.md` in full** before starting any work. The TODO is the single source of truth for architecture, development order, and design decisions. Do NOT rely solely on the compacted conversation summary — it may omit critical context.

2. **Check the development order** (section 五 in TODO.md). Know which step is current and what its dependencies are.

3. **Replay key decisions** from `doc/DEV_LOG_zh.md` and `doc/DEV_LOG_en.md` if uncertain about why something was done a certain way.

## Code conventions (from TODO.md principles)

- No hardcoding — constants go in `constants.py`, user-configurable in `agent_configs.json`
- English only for code comments and console output (Chinese only in docstrings' author metadata)
- `minimal_run` priority — new pipeline features must respect the `[runtime] minimal_run = yes` flag
- Tests must use temp dirs, no writes to `session/`
- Git: concise English commit messages, single-topic commits
- Version: 1.x.x, not bumping to 2.0.0 until this development phase is complete

## Current state

- Steps 0-5 complete (911 tests pass)
- Step 6 next: Tool unification (dg_tft_run / dg_tft_status / dg_tft_report)
- Engine: v95 merged, 79 parameters exposed
- Agent infrastructure (Step 8) follows while+pointer refactor (Step 7)
