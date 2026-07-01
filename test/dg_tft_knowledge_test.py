# -*- coding: utf-8 -*-
"""
Unit tests for knowledge/loader.py (3.5).

Tests:
  - load_knowledge: valid stages, unknown stage, missing file
  - build_agent_prompt: with/without preferences, with/without knowledge
  - list_known_stages: returns correct list

Run: python -m unittest test.dg_tft_knowledge_test -v
"""
import sys
import os
import unittest
import tempfile
import shutil
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.workflow.dg_tft.knowledge.loader import (
    load_knowledge,
    build_agent_prompt,
    list_known_stages,
)


class TestLoadKnowledge(unittest.TestCase):

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp(prefix="dg_tft_knowledge_test_")

    def tearDown(self):
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def _write(self, name: str, content: str) -> Path:
        p = Path(self._tmpdir) / name
        p.write_text(content, encoding="utf-8")
        return p

    # ── load_knowledge ──────────────────────────────────────────────

    def test_load_known_stage(self):
        """3.5-T1: load_knowledge returns content for a mapped stage."""
        self._write("data_inspection.md", "## check-data rules")
        result = load_knowledge("check-data", knowledge_dir=self._tmpdir)
        self.assertIn("check-data rules", result)

    def test_load_another_stage(self):
        """3.5-T2: load_knowledge works for fit-physical stage."""
        self._write("physical_kernel.md", "## fit-physical rules")
        result = load_knowledge("fit-physical", knowledge_dir=self._tmpdir)
        self.assertIn("fit-physical rules", result)

    def test_load_unknown_stage(self):
        """3.5-T3: load_knowledge returns empty for unmapped stage."""
        result = load_knowledge("no-such-stage", knowledge_dir=self._tmpdir)
        self.assertEqual(result, "")

    def test_load_missing_file(self):
        """3.5-T4: load_knowledge returns empty when file not on disk."""
        result = load_knowledge("check-data", knowledge_dir=self._tmpdir)
        self.assertEqual(result, "")

    def test_load_hysteresis(self):
        """3.5-T5: load_knowledge works for train-hysteresis."""
        self._write("hysteresis.md", "# Hysteresis model")
        result = load_knowledge("train-hysteresis", knowledge_dir=self._tmpdir)
        self.assertIn("Hysteresis model", result)

    def test_load_train_static(self):
        """3.5-T6: load_knowledge works for train-static."""
        self._write("static_model.md", "# Static model")
        result = load_knowledge("train-static", knowledge_dir=self._tmpdir)
        self.assertIn("Static model", result)

    def test_load_final_report(self):
        """3.5-T7: load_knowledge works for final-report."""
        self._write("final_interpret.md", "# Final interpret")
        result = load_knowledge("final-report", knowledge_dir=self._tmpdir)
        self.assertIn("Final interpret", result)

    # ── build_agent_prompt ──────────────────────────────────────────

    def test_prompt_with_prefs_and_knowledge(self):
        """3.5-T8: build_agent_prompt includes both blocks."""
        self._write("physical_kernel.md", "# fit-physical expert knowledge")
        prompt = build_agent_prompt(
            "fit-physical",
            user_preferences="hysteresis priority first",
            knowledge_dir=self._tmpdir,
        )
        self.assertIn("USER PREFERENCES", prompt)
        self.assertIn("hysteresis priority first", prompt)
        self.assertIn("EXPERT KNOWLEDGE", prompt)
        self.assertIn("fit-physical expert knowledge", prompt)

    def test_prompt_with_prefs_only(self):
        """3.5-T9: build_agent_prompt omits knowledge block when no file."""
        prompt = build_agent_prompt(
            "check-data",
            user_preferences="IV accuracy first",
            knowledge_dir=self._tmpdir,
        )
        self.assertIn("USER PREFERENCES", prompt)
        self.assertIn("IV accuracy first", prompt)
        self.assertNotIn("EXPERT KNOWLEDGE", prompt)

    def test_prompt_with_knowledge_only(self):
        """3.5-T10: build_agent_prompt omits preferences block when empty."""
        self._write("data_inspection.md", "# data rules")
        prompt = build_agent_prompt(
            "check-data",
            user_preferences="",
            knowledge_dir=self._tmpdir,
        )
        self.assertNotIn("USER PREFERENCES", prompt)
        self.assertIn("EXPERT KNOWLEDGE", prompt)
        self.assertIn("data rules", prompt)

    def test_prompt_empty_both(self):
        """3.5-T11: build_agent_prompt returns empty when both absent."""
        prompt = build_agent_prompt(
            "check-data",
            user_preferences="",
            knowledge_dir=self._tmpdir,
        )
        self.assertEqual(prompt, "")

    def test_prompt_whitespace_only_prefs(self):
        """3.5-T12: whitespace-only preferences treated as empty."""
        self._write("static_model.md", "# static rules")
        prompt = build_agent_prompt(
            "train-static",
            user_preferences="   ",
            knowledge_dir=self._tmpdir,
        )
        self.assertNotIn("USER PREFERENCES", prompt)
        self.assertIn("EXPERT KNOWLEDGE", prompt)

    def test_prompt_unknown_stage(self):
        """3.5-T13: build_agent_prompt handles unmapped stage gracefully."""
        prompt = build_agent_prompt(
            "no-such-stage",
            user_preferences="test pref",
            knowledge_dir=self._tmpdir,
        )
        self.assertIn("USER PREFERENCES", prompt)
        self.assertNotIn("EXPERT KNOWLEDGE", prompt)

    # ── list_known_stages ───────────────────────────────────────────

    def test_list_known_stages(self):
        """3.5-T14: list_known_stages returns expected set."""
        stages = list_known_stages()
        expected = {
            "check-data",
            "fit-physical",
            "train-static",
            "train-hysteresis",
            "final-report",
        }
        self.assertEqual(set(stages), expected)


if __name__ == "__main__":
    unittest.main()
