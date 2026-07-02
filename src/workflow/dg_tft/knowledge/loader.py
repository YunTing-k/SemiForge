# -*- coding: utf-8 -*-
"""
Header information
---------
Shanghai Jiao Tong University, School of Integrated Circuits, SMIL Lab
Author: Yu Huang
Create Date: 2026.6.30
Description: Expert knowledge loader for DG-TFT modeling agents.

    Reads pre-written knowledge .md files and builds agent system prompts
    with user preferences and expert knowledge blocks injected.

    load_knowledge(stage) → raw pre-written knowledge text
    build_agent_prompt(stage, user_preferences) → preferences + knowledge blocks
    load_discovered_summaries(stage) → aggregated discovered-summary text (progressive disclosure)

    Two knowledge types, different injection strategies:
      · Pre-written (knowledge/*.md)        — curated, bounded → full text in system prompt
      · Discovered (knowledge/_discovered/) — grows with runs   → summaries injected, details on-disk (Agent read_file)

Revision:
---------
2026.6.30      Yu Huang      1.0      Initial implementation (3.5)
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

# ── Stage → knowledge file mapping ──────────────────────────────────

_STAGE_KNOWLEDGE_MAP: dict[str, str] = {
    "check-data": "data_inspection.md",
    "fit-physical": "physical_kernel.md",
    "train-static": "static_model.md",
    "train-hysteresis": "hysteresis.md",
    "final-report": "final_interpret.md",
}

# ── Prompt block templates ──────────────────────────────────────────

_USER_PREFERENCES_BLOCK = """\
==== USER PREFERENCES ====
{user_preferences}
==== END USER PREFERENCES ====
"""

_EXPERT_KNOWLEDGE_BLOCK = """\
==== EXPERT KNOWLEDGE (DO NOT OUTPUT THIS SECTION) ====
{expert_knowledge}
==== END EXPERT KNOWLEDGE ====
"""

# ── Helpers ─────────────────────────────────────────────────────────

def _default_knowledge_dir() -> Path:
    """Project-root knowledge/ directory (expert + discovered knowledge)."""
    return Path(__file__).resolve().parent.parent.parent.parent.parent / "knowledge"


def load_knowledge(
    stage: str,
    knowledge_dir: Optional[str | Path] = None,
) -> str:
    """Load expert knowledge text for a given pipeline stage.

    Args:
        stage: Stage name (e.g. "fit-physical", "train-static").
        knowledge_dir: Path to the knowledge directory. If None, uses
                       the directory containing this loader module.

    Returns:
        Knowledge file content as a string, or empty string if the
        stage has no mapped knowledge file or the file does not exist.
    """
    filename = _STAGE_KNOWLEDGE_MAP.get(stage)
    if filename is None:
        return ""

    root = Path(knowledge_dir) if knowledge_dir else _default_knowledge_dir()
    file_path = root / filename
    if not file_path.is_file():
        return ""

    return file_path.read_text(encoding="utf-8").strip()


def build_agent_prompt(
    stage: str,
    user_preferences: str = "",
    knowledge_dir: Optional[str | Path] = None,
    discovered_dir: Optional[str | Path] = None,
) -> str:
    """Build the preferences + expert knowledge portion of an agent system prompt.

    Returns a string containing the USER PREFERENCES and EXPERT KNOWLEDGE
    blocks (only including non-empty sections), suitable for injection
    into a stage-specific agent prompt template.

    The EXPERT KNOWLEDGE block combines:
      1. Pre-written knowledge (knowledge/*.md) — full text, always loaded
      2. Discovered knowledge summaries (knowledge/_discovered/*_summary.md)
         — aggregated per-stage, injected for progressive disclosure

    Args:
        stage: Stage name.
        user_preferences: User-specified modeling priorities. If empty,
                          the USER PREFERENCES block is omitted.
        knowledge_dir: Optional knowledge directory path.
        discovered_dir: Optional discovered knowledge directory path. If
                        provided, summaries are aggregated and appended.

    Returns:
        Formatted string for injection into an agent system prompt.
        Returns empty string if all sections are empty.
    """
    parts: list[str] = []

    if user_preferences.strip():
        parts.append(_USER_PREFERENCES_BLOCK.format(
            user_preferences=user_preferences.strip(),
        ))

    # ── pre-written knowledge (full text) ──
    pre_written = load_knowledge(stage, knowledge_dir=knowledge_dir)

    # ── discovered knowledge summaries (progressive disclosure) ──
    discovered = ""
    if discovered_dir:
        discovered = load_discovered_summaries(stage, discovered_dir=discovered_dir)

    combined = pre_written or ""
    if discovered:
        if combined:
            combined += "\n\n## Discovered Knowledge (from previous runs)\n\n"
        combined += discovered

    if combined:
        parts.append(_EXPERT_KNOWLEDGE_BLOCK.format(
            expert_knowledge=combined,
        ))

    if not parts:
        return ""

    return "\n".join(parts)


def list_known_stages() -> list[str]:
    """Return list of stages that have mapped knowledge files."""
    return list(_STAGE_KNOWLEDGE_MAP.keys())


def load_discovered_summaries(
    stage: str,
    discovered_dir: str | Path,
) -> str:
    """Aggregate discovered knowledge summaries for a stage (progressive disclosure).

    Scans ``discovered_dir`` for ``*_summary.md`` files, reads their content,
    and appends them into a single text block for injection into the agent
    system prompt.  Detailed per-run reports (*.md without ``_summary`` suffix)
    are left on disk for the agent to read on-demand via ``read_file``.

    Implemented when 4.4 (knowledge contribution) is built.

    Args:
        stage: Stage name for filtering (future; currently returns all).
        discovered_dir: Path to ``knowledge/_discovered/``.

    Returns:
        Concatenated summary text, or empty string if no summaries exist.
    """
    # TODO (4.4): scan discovered_dir for *_summary.md,
    # filter by stage-relevant keywords, return concatenated text.
    return ""
