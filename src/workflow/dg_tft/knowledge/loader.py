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

    Two knowledge sources, injected separately:
      · knowledge/*.md        — code-derived reference (parameters, engine behavior, decision frameworks)
      · knowledge/experience/*.md — practical patterns from real sessions (diagnostic signals, tuning recipes)

    load_knowledge(stage)   → raw pre-written knowledge text
    load_experience(stage)  → raw experience text
    build_agent_prompt(stage, user_preferences) → preferences + knowledge + experience blocks

Revision:
---------
2026.6.30      Yu Huang      1.0      Initial implementation (3.5)
2026.7.03                  1.1      Add experience/ directory support (Step 8+)
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

_STAGE_EXPERIENCE_MAP: dict[str, str] = {
    "check-data": "data_inspection_experience.md",
    "fit-physical": "physical_kernel_experience.md",
    "train-static": "static_model_experience.md",
    "train-hysteresis": "hysteresis_experience.md",
    "final-report": "final_interpret_experience.md",
}

# ── Prompt block templates ──────────────────────────────────────────

_USER_PREFERENCES_BLOCK = """\
==== USER PREFERENCES ====
{user_preferences}
==== END USER PREFERENCES ====
"""

_EXPERT_KNOWLEDGE_BLOCK = """\
==== EXPERT KNOWLEDGE (code-derived reference — parameters, engine behavior, decision frameworks) ====
{expert_knowledge}
==== END EXPERT KNOWLEDGE ====
"""

_EXPERIENCE_BLOCK = """\
==== EXPERIENCE (practical patterns from real sessions — diagnostic signals, tuning recipes, tool choices) ====
{experience}
==== END EXPERIENCE ====
"""

# ── Helpers ─────────────────────────────────────────────────────────

def _default_knowledge_dir() -> Path:
    """Project-root knowledge/ directory (expert + experience knowledge)."""
    return Path(__file__).resolve().parent.parent.parent.parent.parent / "knowledge"


def load_knowledge(
    stage: str,
    knowledge_dir: Optional[str | Path] = None,
) -> str:
    """Load code-derived expert knowledge text for a given pipeline stage.

    Reads from ``knowledge/*.md``.

    Args:
        stage: Stage name (e.g. "fit-physical", "train-static").
        knowledge_dir: Path to the knowledge directory. If None, uses
                       the project-root knowledge/ directory.

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


def load_experience(
    stage: str,
    knowledge_dir: Optional[str | Path] = None,
) -> str:
    """Load practical experience patterns for a given pipeline stage.

    Reads from ``knowledge/experience/*.md``.

    Args:
        stage: Stage name (e.g. "fit-physical", "train-static").
        knowledge_dir: Path to the knowledge directory. If None, uses
                       the project-root knowledge/ directory.

    Returns:
        Experience file content as a string, or empty string if the
        stage has no mapped experience file or the file does not exist.
    """
    filename = _STAGE_EXPERIENCE_MAP.get(stage)
    if filename is None:
        return ""

    root = Path(knowledge_dir) if knowledge_dir else _default_knowledge_dir()
    file_path = root / "experience" / filename
    if not file_path.is_file():
        return ""

    return file_path.read_text(encoding="utf-8").strip()


def build_agent_prompt(
    stage: str,
    user_preferences: str = "",
    knowledge_dir: Optional[str | Path] = None,
    discovered_dir: Optional[str | Path] = None,
) -> str:
    """Build the preferences + knowledge + experience portion of an agent system prompt.

    Returns a string containing the USER PREFERENCES, EXPERT KNOWLEDGE, and EXPERIENCE
    blocks (only including non-empty sections), suitable for injection
    into a stage-specific agent prompt template.

    Block order:
      1. USER PREFERENCES — modelling priorities from the user
      2. EXPERT KNOWLEDGE — code-derived reference (knowledge/*.md)
      3. EXPERIENCE — practical patterns from real sessions (knowledge/experience/*.md)

    Args:
        stage: Stage name.
        user_preferences: User-specified modeling priorities. If empty,
                          the USER PREFERENCES block is omitted.
        knowledge_dir: Optional knowledge directory path.
        discovered_dir: Optional discovered knowledge directory path (reserved for 4.4).

    Returns:
        Formatted string for injection into an agent system prompt.
        Returns empty string if all sections are empty.
    """
    parts: list[str] = []

    if user_preferences.strip():
        parts.append(_USER_PREFERENCES_BLOCK.format(
            user_preferences=user_preferences.strip(),
        ))

    # ── code-derived knowledge ──
    pre_written = load_knowledge(stage, knowledge_dir=knowledge_dir)
    if pre_written:
        parts.append(_EXPERT_KNOWLEDGE_BLOCK.format(
            expert_knowledge=pre_written,
        ))

    # ── practical experience patterns ──
    experience = load_experience(stage, knowledge_dir=knowledge_dir)
    if experience:
        parts.append(_EXPERIENCE_BLOCK.format(
            experience=experience,
        ))

    # ── discovered knowledge summaries (progressive disclosure, reserved for 4.4) ──
    discovered = ""
    if discovered_dir:
        discovered = load_discovered_summaries(stage, discovered_dir=discovered_dir)
    if discovered:
        parts.append(discovered)

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
    """Scan discovered knowledge files for YAML frontmatter summaries.

    Reads ``knowledge/_discovered/*.md``, parses ``---\n...\n---`` YAML
    headers, and returns a formatted list of summaries for injection into
    the agent system prompt.  The agent can then ``read_file`` individual
    files on-demand.

    Args:
        stage: Stage name for filtering (future; currently returns all).
        discovered_dir: Path to ``knowledge/_discovered/``.

    Returns:
        Formatted text block listing discovered knowledge items, or empty
        string if no summaries exist.
    """
    import glob
    import yaml

    root = Path(discovered_dir)
    if not root.is_dir():
        return ""

    items: list[dict[str, str]] = []
    for md_file in sorted(glob.glob(str(root / "*.md"))):
        try:
            content = Path(md_file).read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        if not content.startswith("---"):
            continue
        try:
            _, fm, _ = content.split("---", 2)
            data = yaml.safe_load(fm)
            if isinstance(data, dict):
                title = str(data.get("title", ""))
                summary = str(data.get("summary", ""))
                if title or summary:
                    items.append({"file": Path(md_file).name, "title": title, "summary": summary})
        except (ValueError, yaml.YAMLError):
            continue

    if not items:
        return ""

    lines = [
        "==== DISCOVERED KNOWLEDGE (summaries — use `read_file` to load full content on-demand) ====",
        "",
    ]
    for item in items:
        lines.append(f"- **{item['title']}** ({item['file']})")
        lines.append(f"  {item['summary']}")
        lines.append("")
    return "\n".join(lines)
