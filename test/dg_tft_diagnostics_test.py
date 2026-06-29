# -*- coding: utf-8 -*-
"""
Unit tests for Phase 6: diagnostics.py (error classification) and
subagent_hooks.py (hook definitions + prompt templates).

Run: python -m unittest test.dg_tft_diagnostics_test -v
"""
import sys
import os
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.workflow.dg_tft.diagnostics import (
    classify_error,
    classify_from_result,
    DiagnosticResult,
    _KEYWORD_RULES,
    _UNKNOWN_CONFIDENCE,
    _STAGE_SUGGESTIONS,
    _EXIT_CODE_NUMERICAL_STAGES,
    _EXIT_CODE_DATA_ERROR_STAGES,
)
from src.workflow.dg_tft.subagent_hooks import (
    HookDefinition,
    get_hook,
    list_hooks,
    build_diagnosis_prompt,
    _HOOKS,
    _HOOK_BY_CATEGORY,
    _PROMPT_ERROR_MAX_CHARS,
)
from src.workflow.dg_tft.schema import (
    ErrorCategory,
    StageName,
    StageResult,
    WorkflowRun,
)


# ────────────────────────────────────────────────────────────────────
# Test: Error classification — keyword matching
# ────────────────────────────────────────────────────────────────────


class TestKeywordClassification(unittest.TestCase):

    def test_data_error_file_not_found(self):
        result = classify_error(
            stage_name=StageName.CHECK_DATA,
            exit_code=1,
            stderr="FileNotFoundError: [Errno 2] No such file or directory: 'data.csv'",
        )
        self.assertEqual(result.category, ErrorCategory.DATA_ERROR)
        self.assertGreaterEqual(result.confidence, 0.7)

    def test_data_error_missing_sheet(self):
        result = classify_error(
            stage_name=StageName.CHECK_DATA,
            exit_code=1,
            error_message="Validation failed: missing sheet 'long' in IDVD_train.xlsx",
        )
        self.assertEqual(result.category, ErrorCategory.DATA_ERROR)
        self.assertGreaterEqual(result.confidence, 0.7)

    def test_data_error_missing_column(self):
        result = classify_error(
            stage_name=StageName.CHECK_DATA,
            exit_code=1,
            error_message="column not found: VBG_V",
        )
        self.assertEqual(result.category, ErrorCategory.DATA_ERROR)
        self.assertGreaterEqual(result.confidence, 0.7)

    def test_data_error_keyerror_valueerror(self):
        result = classify_error(
            stage_name=StageName.CHECK_DATA,
            exit_code=1,
            error_message="KeyError: 'long'",
        )
        self.assertEqual(result.category, ErrorCategory.DATA_ERROR)
        self.assertAlmostEqual(result.confidence, 0.6)

    def test_config_error_section_missing(self):
        result = classify_error(
            stage_name=StageName.FIT_PHYSICAL,
            exit_code=1,
            error_message="configparser.NoSectionError: No section: 'paths'",
        )
        self.assertEqual(result.category, ErrorCategory.CONFIG_ERROR)
        self.assertGreaterEqual(result.confidence, 0.7)

    def test_config_error_no_option(self):
        result = classify_error(
            stage_name=StageName.TRAIN_STATIC,
            exit_code=1,
            error_message="No option 'data_dir' in section: 'paths'",
        )
        self.assertEqual(result.category, ErrorCategory.CONFIG_ERROR)
        self.assertGreaterEqual(result.confidence, 0.7)

    def test_training_error_oom(self):
        result = classify_error(
            stage_name=StageName.TRAIN_STATIC,
            exit_code=1,
            stderr="RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB",
        )
        self.assertEqual(result.category, ErrorCategory.TRAINING_NUMERICAL_ERROR)
        self.assertGreaterEqual(result.confidence, 0.7)

    def test_training_error_nan(self):
        result = classify_error(
            stage_name=StageName.TRAIN_STATIC,
            exit_code=1,
            stderr="loss is nan at epoch 42",
        )
        self.assertEqual(result.category, ErrorCategory.TRAINING_NUMERICAL_ERROR)
        self.assertAlmostEqual(result.confidence, 0.9)

    def test_training_error_inf(self):
        result = classify_error(
            stage_name=StageName.FIT_PHYSICAL,
            exit_code=1,
            stderr="ValueError: inf values in parameter vector",
        )
        self.assertEqual(result.category, ErrorCategory.TRAINING_NUMERICAL_ERROR)
        self.assertAlmostEqual(result.confidence, 0.9)

    def test_training_error_overflow(self):
        result = classify_error(
            stage_name=StageName.TRAIN_HYSTERESIS,
            exit_code=1,
            stderr="overflow encountered in multiply",
        )
        self.assertEqual(result.category, ErrorCategory.TRAINING_NUMERICAL_ERROR)
        self.assertGreaterEqual(result.confidence, 0.7)

    def test_hspice_env_error_not_found(self):
        result = classify_error(
            stage_name=StageName.TRAIN_STATIC,
            exit_code=1,
            error_message="hspice.exe not found",
        )
        self.assertEqual(result.category, ErrorCategory.HSPICE_ENV_ERROR)
        self.assertGreaterEqual(result.confidence, 0.7)

    def test_hspice_timeout(self):
        result = classify_error(
            stage_name=StageName.TRAIN_STATIC,
            exit_code=1,
            stderr="hspice simulation failed: timeout",
        )
        self.assertEqual(result.category, ErrorCategory.HSPICE_ENV_ERROR)
        self.assertGreaterEqual(result.confidence, 0.7)


# ────────────────────────────────────────────────────────────────────
# Test: Error classification — missing artifacts
# ────────────────────────────────────────────────────────────────────


class TestMissingArtifactClassification(unittest.TestCase):

    def test_missing_artifacts_despite_exit_zero(self):
        result = classify_error(
            stage_name=StageName.TRAIN_STATIC,
            exit_code=0,
            missing_artifacts=["v90_safe_residual_state.pt"],
        )
        self.assertEqual(result.category, ErrorCategory.MISSING_ARTIFACT_ERROR)
        self.assertGreaterEqual(result.confidence, 0.7)

    def test_missing_artifacts_multiple(self):
        result = classify_error(
            stage_name=StageName.TRAIN_HYSTERESIS,
            exit_code=0,
            missing_artifacts=["dg_tft_dynamic_srh.va", "dg_tft_dynamic_srh.inc"],
        )
        self.assertEqual(result.category, ErrorCategory.MISSING_ARTIFACT_ERROR)
        self.assertAlmostEqual(result.confidence, 0.85)
        self.assertIn("2 expected", result.evidence[0])

    def test_keyword_match_takes_priority_over_missing_artifacts(self):
        result = classify_error(
            stage_name=StageName.CHECK_DATA,
            exit_code=0,
            error_message="FileNotFoundError: data.csv",
            missing_artifacts=["dataset_validation.json"],
        )
        self.assertEqual(result.category, ErrorCategory.DATA_ERROR)


# ────────────────────────────────────────────────────────────────────
# Test: Error classification — exit-code fallback
# ────────────────────────────────────────────────────────────────────


class TestExitCodeFallback(unittest.TestCase):

    def test_check_data_exit_nonzero_defaults_to_data_error(self):
        result = classify_error(
            stage_name=StageName.CHECK_DATA,
            exit_code=1,
            stderr="Something went wrong (no keyword match)",
        )
        self.assertEqual(result.category, ErrorCategory.DATA_ERROR)
        self.assertEqual(result.confidence, 0.5)

    def test_fit_physical_exit_nonzero_defaults_to_numerical(self):
        result = classify_error(
            stage_name=StageName.FIT_PHYSICAL,
            exit_code=1,
            stderr="Something went wrong (no keyword match)",
        )
        self.assertEqual(result.category, ErrorCategory.TRAINING_NUMERICAL_ERROR)
        self.assertEqual(result.confidence, 0.4)

    def test_train_static_exit_nonzero_defaults_to_numerical(self):
        result = classify_error(
            stage_name=StageName.TRAIN_STATIC,
            exit_code=2,
            stderr="Unknown failure",
        )
        self.assertEqual(result.category, ErrorCategory.TRAINING_NUMERICAL_ERROR)
        self.assertEqual(result.confidence, 0.4)

    def test_train_hysteresis_exit_nonzero_defaults_to_numerical(self):
        result = classify_error(
            stage_name=StageName.TRAIN_HYSTERESIS,
            exit_code=3,
            stderr="Unknown failure",
        )
        self.assertEqual(result.category, ErrorCategory.TRAINING_NUMERICAL_ERROR)
        self.assertEqual(result.confidence, 0.4)

    def test_unknown_stage_falls_to_unknown(self):
        result = classify_error(
            stage_name="deploy-va",
            exit_code=1,
            stderr="Something went wrong",
        )
        self.assertEqual(result.category, ErrorCategory.UNKNOWN_ERROR)
        self.assertEqual(result.confidence, _UNKNOWN_CONFIDENCE)


# ────────────────────────────────────────────────────────────────────
# Test: Error classification — edge cases
# ────────────────────────────────────────────────────────────────────


class TestTruncateList(unittest.TestCase):

    def test_truncate_short_list(self):
        from src.workflow.dg_tft.diagnostics import _truncate_list
        result = _truncate_list(["a.txt", "b.txt"])
        self.assertEqual(result, "a.txt, b.txt")

    def test_truncate_long_list(self):
        from src.workflow.dg_tft.diagnostics import _truncate_list
        items = [f"file_{i}.txt" for i in range(10)]
        result = _truncate_list(items, max_show=3)
        self.assertIn("file_0.txt", result)
        self.assertIn("file_2.txt", result)
        self.assertIn("+7 more", result)
        self.assertNotIn("file_3.txt", result)

    def test_truncate_empty_list(self):
        from src.workflow.dg_tft.diagnostics import _truncate_list
        result = _truncate_list([])
        self.assertEqual(result, "")

    def test_truncate_list_at_boundary(self):
        from src.workflow.dg_tft.diagnostics import _truncate_list
        result = _truncate_list(["a", "b", "c"], max_show=3)
        self.assertEqual(result, "a, b, c")
        self.assertNotIn("more", result)


class TestStdoutOnlyClassification(unittest.TestCase):

    def test_stdout_only_keyword_match(self):
        result = classify_error(
            stage_name=StageName.CHECK_DATA,
            exit_code=1,
            stdout="FileNotFoundError: data.csv not found in /data",
        )
        self.assertEqual(result.category, ErrorCategory.DATA_ERROR)

    def test_stdout_only_no_match_falls_back(self):
        result = classify_error(
            stage_name=StageName.CHECK_DATA,
            exit_code=1,
            stdout="Processing completed with warnings",
        )
        self.assertEqual(result.category, ErrorCategory.DATA_ERROR)
        self.assertEqual(result.confidence, 0.5)

    def test_error_message_overrides_stdout(self):
        result = classify_error(
            stage_name=StageName.CHECK_DATA,
            exit_code=1,
            stdout="Some benign message",
            error_message="FileNotFoundError: real_error.csv",
        )
        self.assertEqual(result.category, ErrorCategory.DATA_ERROR)
        self.assertIn("fileNotFoundError".lower(), result.evidence[0].lower())

    def test_empty_text_complete_unknown(self):
        result = classify_error(
            stage_name="deploy-va",
            exit_code=0,
        )
        self.assertEqual(result.category, ErrorCategory.UNKNOWN_ERROR)
        self.assertEqual(result.confidence, _UNKNOWN_CONFIDENCE)

    def test_none_values_treated_as_empty(self):
        result = classify_error(
            stage_name=StageName.CHECK_DATA,
            exit_code=0,
            stdout="",
            stderr="",
            error_message="",
        )
        self.assertEqual(result.category, ErrorCategory.UNKNOWN_ERROR)

    def test_case_insensitive_keyword_matching(self):
        result = classify_error(
            stage_name=StageName.CHECK_DATA,
            exit_code=1,
            error_message="FILENOTFOUNDERROR: data/missing.csv",
        )
        self.assertEqual(result.category, ErrorCategory.DATA_ERROR)

    def test_first_matching_rule_wins(self):
        result = classify_error(
            stage_name=StageName.CHECK_DATA,
            exit_code=1,
            error_message="configparser.NoSectionError: 'paths' section missing",
        )
        self.assertEqual(result.category, ErrorCategory.CONFIG_ERROR)
        self.assertGreaterEqual(result.confidence, 0.9)

    def test_confidence_high_for_explicit_match(self):
        result = classify_error(
            stage_name=StageName.CHECK_DATA,
            exit_code=1,
            error_message="FileNotFoundError: missing.xlsx",
        )
        self.assertGreaterEqual(result.confidence, 0.85)

    def test_suggestion_provided_on_classification(self):
        result = classify_error(
            stage_name=StageName.CHECK_DATA,
            exit_code=1,
            error_message="FileNotFoundError: missing.xlsx",
        )
        self.assertIsInstance(result.suggestion, str)
        self.assertGreater(len(result.suggestion), 0)

    def test_unknown_stage_gets_generic_suggestion(self):
        result = classify_error(
            stage_name="custom-stage",
            exit_code=0,
        )
        self.assertIn("log", result.suggestion.lower())


# ────────────────────────────────────────────────────────────────────
# Test: classify_from_result wrapper
# ────────────────────────────────────────────────────────────────────


class TestClassifyFromResult(unittest.TestCase):

    def test_uses_stage_result_error_field(self):
        sr = StageResult(
            ok=False,
            stage=StageName.CHECK_DATA,
            error="FileNotFoundError: missing.csv",
        )
        result = classify_from_result(sr, exit_code=1)
        self.assertEqual(result.category, ErrorCategory.DATA_ERROR)

    def test_uses_stderr_when_error_is_none(self):
        sr = StageResult(
            ok=False,
            stage=StageName.TRAIN_STATIC,
            error="",
        )
        result = classify_from_result(
            sr,
            exit_code=1,
            stderr="loss is nan at epoch 100",
        )
        self.assertEqual(result.category, ErrorCategory.TRAINING_NUMERICAL_ERROR)

    def test_passes_missing_artifacts_through(self):
        sr = StageResult(
            ok=True,
            stage=StageName.TRAIN_HYSTERESIS,
        )
        result = classify_from_result(
            sr,
            exit_code=0,
            missing_artifacts=["va_file.va"],
        )
        self.assertEqual(result.category, ErrorCategory.MISSING_ARTIFACT_ERROR)

    def test_ok_true_does_not_overwrite_classification(self):
        sr = StageResult(
            ok=True,
            stage=StageName.CHECK_DATA,
            error="FileNotFoundError: bad.csv",
        )
        result = classify_from_result(sr, exit_code=1)
        self.assertEqual(result.category, ErrorCategory.DATA_ERROR)


# ────────────────────────────────────────────────────────────────────
# Test: DiagnosticResult serialisation
# ────────────────────────────────────────────────────────────────────


class TestDiagnosticResultSerialisation(unittest.TestCase):

    def test_to_dict_contains_all_fields(self):
        result = DiagnosticResult(
            category=ErrorCategory.DATA_ERROR,
            confidence=0.9,
            evidence=["FileNotFoundError"],
            suggestion="Check data path.",
        )
        d = result.to_dict()
        self.assertEqual(d["category"], "DATA_ERROR")
        self.assertEqual(d["confidence"], 0.9)
        self.assertEqual(len(d["evidence"]), 1)
        self.assertIn("Check data path", d["suggestion"])

    def test_default_values(self):
        result = DiagnosticResult()
        self.assertEqual(result.category, ErrorCategory.UNKNOWN_ERROR)
        self.assertEqual(result.confidence, 0.0)
        self.assertEqual(result.evidence, [])
        self.assertEqual(result.suggestion, "")


# ────────────────────────────────────────────────────────────────────
# Test: Module-level constants integrity
# ────────────────────────────────────────────────────────────────────


class TestConstantsIntegrity(unittest.TestCase):

    def test_keyword_rules_have_valid_categories(self):
        valid = set(ErrorCategory)
        for cat, _, _ in _KEYWORD_RULES:
            self.assertIn(cat, valid)

    def test_keyword_rules_confidence_in_range(self):
        for _, conf, _ in _KEYWORD_RULES:
            self.assertGreaterEqual(conf, 0.0)
            self.assertLessEqual(conf, 1.0)

    def test_keyword_rules_non_empty_sigils(self):
        for cat, _, sigils in _KEYWORD_RULES:
            self.assertGreater(len(sigils), 0, f"Empty sigil list for {cat}")

    def test_stage_suggestions_cover_core_stages(self):
        for stage in (StageName.CHECK_DATA, StageName.FIT_PHYSICAL,
                       StageName.TRAIN_STATIC, StageName.TRAIN_HYSTERESIS):
            self.assertIn(stage, _STAGE_SUGGESTIONS)

    def test_data_error_stages_set_is_correct(self):
        self.assertIn(StageName.CHECK_DATA, _EXIT_CODE_DATA_ERROR_STAGES)

    def test_numerical_stages_set_is_correct(self):
        for stage in (StageName.FIT_PHYSICAL, StageName.TRAIN_STATIC,
                       StageName.TRAIN_HYSTERESIS):
            self.assertIn(stage, _EXIT_CODE_NUMERICAL_STAGES)

    def test_all_error_categories_have_keyword_rules(self):
        covered = {cat for cat, _, _ in _KEYWORD_RULES}
        for cat in ErrorCategory:
            if cat == ErrorCategory.UNKNOWN_ERROR:
                continue  # UNKNOWN_ERROR is the fallback, no explicit keywords needed
            self.assertIn(cat, covered, f"No keyword rules for {cat}")


# ════════════════════════════════════════════════════════════════════
# Tests: subagent_hooks
# ════════════════════════════════════════════════════════════════════


class TestHookRegistry(unittest.TestCase):

    def test_six_hooks_registered(self):
        self.assertEqual(len(_HOOKS), 6)

    def test_list_hooks_returns_all(self):
        hooks = list_hooks()
        self.assertEqual(len(hooks), 6)
        for h in hooks:
            self.assertIsInstance(h, HookDefinition)

    def test_every_error_category_has_a_hook(self):
        for cat in ErrorCategory:
            self.assertIn(cat, _HOOK_BY_CATEGORY, f"No hook for {cat}")

    def test_hooks_use_explorer_subagent_type(self):
        for h in _HOOKS:
            self.assertEqual(
                h.subagent_type,
                "explorer",
                f"Hook {h.hook_id} should use explorer subagent",
            )

    def test_hooks_have_unique_ids(self):
        ids = [h.hook_id for h in _HOOKS]
        self.assertEqual(len(ids), len(set(ids)))

    def test_hook_ids_are_stable(self):
        expected = {
            "on_data_error", "on_training_failure", "on_unknown_error",
            "on_missing_artifact", "on_config_error", "on_hspice_error",
        }
        actual = {h.hook_id for h in _HOOKS}
        self.assertEqual(actual, expected)

    def test_get_hook_known_category(self):
        hook = get_hook(ErrorCategory.DATA_ERROR)
        self.assertIsNotNone(hook)
        self.assertEqual(hook.hook_id, "on_data_error")

    def test_get_hook_unknown_category_returns_none(self):
        self.assertIsNone(get_hook(None))

    def test_hooks_have_subject(self):
        for h in _HOOKS:
            self.assertIsInstance(h.subject, str)
            self.assertGreater(len(h.subject), 0,
                               f"Hook {h.hook_id} has empty subject")

    def test_hooks_have_description(self):
        for h in _HOOKS:
            self.assertIsInstance(h.description, str)
            self.assertGreater(len(h.description), 0,
                               f"Hook {h.hook_id} has empty description")


class TestBuildDiagnosisPrompt(unittest.TestCase):

    def setUp(self):
        self.hook = get_hook(ErrorCategory.TRAINING_NUMERICAL_ERROR)
        self.assertIsNotNone(self.hook)

    def test_prompt_contains_stage_name(self):
        prompt = build_diagnosis_prompt(
            self.hook,
            stage_name="train-static",
            category=ErrorCategory.TRAINING_NUMERICAL_ERROR,
            confidence=0.9,
            evidence=["loss is nan"],
            error_message="nan at epoch 42",
        )
        self.assertIn("train-static", prompt)
        self.assertIn("TRAINING_NUMERICAL_ERROR", prompt)
        self.assertIn("90%", prompt)
        self.assertIn("loss is nan", prompt)
        self.assertIn("epoch 42", prompt)

    def test_prompt_truncates_long_error_message(self):
        long_error = "x" * (_PROMPT_ERROR_MAX_CHARS + 500)
        prompt = build_diagnosis_prompt(
            self.hook,
            stage_name="train-static",
            category=ErrorCategory.TRAINING_NUMERICAL_ERROR,
            confidence=0.5,
            evidence=[],
            error_message=long_error,
        )
        self.assertIn("truncated", prompt)
        self.assertLess(len(prompt), len(long_error) + 2000)

    def test_prompt_handles_empty_evidence(self):
        prompt = build_diagnosis_prompt(
            self.hook,
            stage_name="train-static",
            category=ErrorCategory.TRAINING_NUMERICAL_ERROR,
            confidence=0.5,
            evidence=[],
        )
        self.assertIn("(none)", prompt.lower())

    def test_prompt_handles_empty_error_message(self):
        prompt = build_diagnosis_prompt(
            self.hook,
            stage_name="train-static",
            category=ErrorCategory.TRAINING_NUMERICAL_ERROR,
            confidence=0.5,
            evidence=["nan detected"],
            error_message="",
        )
        self.assertIn("nan detected", prompt)

    def test_prompt_includes_log_snippet_when_provided(self):
        prompt = build_diagnosis_prompt(
            self.hook,
            stage_name="train-static",
            category=ErrorCategory.TRAINING_NUMERICAL_ERROR,
            confidence=0.9,
            evidence=["overflow"],
            error_message="inf found",
            log_snippet="Epoch 1 loss=1.5\nEpoch 2 loss=NaN",
        )
        self.assertIn("Log excerpt", prompt)
        self.assertIn("Epoch 1", prompt)

    def test_confidence_formatting(self):
        prompt = build_diagnosis_prompt(
            self.hook,
            stage_name="train-static",
            category=ErrorCategory.TRAINING_NUMERICAL_ERROR,
            confidence=1.0,
            evidence=["nan"],
        )
        self.assertIn("100%", prompt)

        prompt_50 = build_diagnosis_prompt(
            self.hook,
            stage_name="train-static",
            category=ErrorCategory.TRAINING_NUMERICAL_ERROR,
            confidence=0.5,
            evidence=["nan"],
        )
        self.assertIn("50%", prompt_50)

    def test_data_error_prompt_template_complete(self):
        hook = get_hook(ErrorCategory.DATA_ERROR)
        prompt = build_diagnosis_prompt(
            hook,
            stage_name="check-data",
            category=ErrorCategory.DATA_ERROR,
            confidence=0.9,
            evidence=["missing sheet: long"],
            error_message="Validation failed for IDVBG_train.xlsx",
        )
        self.assertIn("check-data", prompt)
        self.assertIn("missing sheet: long", prompt)
        self.assertIn("IDVBG_train.xlsx", prompt)
        self.assertNotIn("{stage_name}", prompt)
        self.assertNotIn("{category}", prompt)

    def test_config_error_prompt_template_complete(self):
        hook = get_hook(ErrorCategory.CONFIG_ERROR)
        prompt = build_diagnosis_prompt(
            hook,
            stage_name="fit-physical",
            category=ErrorCategory.CONFIG_ERROR,
            confidence=0.9,
            evidence=["No section: 'paths'"],
            error_message="configparser.NoSectionError",
        )
        self.assertIn("configuration", prompt.lower())
        self.assertNotIn("{", prompt)  # no unrendered placeholders

    def test_missing_artifact_prompt_template_complete(self):
        hook = get_hook(ErrorCategory.MISSING_ARTIFACT_ERROR)
        prompt = build_diagnosis_prompt(
            hook,
            stage_name="train-static",
            category=ErrorCategory.MISSING_ARTIFACT_ERROR,
            confidence=0.85,
            evidence=["v90_safe_residual_state.pt missing"],
            error_message="Expected outputs not found",
        )
        self.assertIn("missing", prompt.lower())
        self.assertNotIn("{", prompt)

    def test_hspice_error_prompt_template_complete(self):
        hook = get_hook(ErrorCategory.HSPICE_ENV_ERROR)
        prompt = build_diagnosis_prompt(
            hook,
            stage_name="train-static",
            category=ErrorCategory.HSPICE_ENV_ERROR,
            confidence=0.9,
            evidence=["hspice.exe not found"],
            error_message="",
        )
        self.assertIn("HSPICE", prompt)
        self.assertNotIn("{", prompt)

    def test_unknown_error_prompt_template_complete(self):
        hook = get_hook(ErrorCategory.UNKNOWN_ERROR)
        prompt = build_diagnosis_prompt(
            hook,
            stage_name="train-static",
            category=ErrorCategory.UNKNOWN_ERROR,
            confidence=0.3,
            evidence=[],
            error_message="exit code 137",
        )
        self.assertIn("UNKNOWN_ERROR", prompt)
        self.assertIn("exit code 137", prompt)
        self.assertNotIn("{", prompt)


class TestPromptTemplateImmutability(unittest.TestCase):

    def test_templates_have_no_hardcoded_stage_names(self):
        for h in _HOOKS:
            self.assertNotIn("check-data", h.prompt_template,
                             f"Hook {h.hook_id} hardcodes stage name")
            self.assertNotIn("train-static", h.prompt_template,
                             f"Hook {h.hook_id} hardcodes stage name")

    def test_templates_use_format_placeholders(self):
        for h in _HOOKS:
            for var in ("{stage_name}", "{category}", "{evidence}", "{error_message}"):
                self.assertIn(var, h.prompt_template,
                              f"Hook {h.hook_id} missing {var} placeholder")


# ────────────────────────────────────────────────────────────────────
# Test: Full integration — classify then hook
# ────────────────────────────────────────────────────────────────────


class TestDiagnoseHookIntegration(unittest.TestCase):

    def test_classify_then_get_hook(self):
        result = classify_error(
            stage_name=StageName.CHECK_DATA,
            exit_code=1,
            error_message="FileNotFoundError: missing.csv",
        )
        hook = get_hook(result.category)
        self.assertIsNotNone(hook)
        self.assertEqual(hook.hook_id, "on_data_error")

    def test_classify_numerical_then_get_hook(self):
        result = classify_error(
            stage_name=StageName.TRAIN_STATIC,
            exit_code=1,
            stderr="loss is nan at epoch 42",
        )
        hook = get_hook(result.category)
        self.assertIsNotNone(hook)
        self.assertEqual(hook.hook_id, "on_training_failure")

    def test_build_prompt_from_diagnostic_result(self):
        result = classify_error(
            stage_name=StageName.FIT_PHYSICAL,
            exit_code=1,
            stderr="overflow encountered in exp",
        )
        hook = get_hook(result.category)
        prompt = build_diagnosis_prompt(
            hook,
            stage_name=StageName.FIT_PHYSICAL,
            category=result.category,
            confidence=result.confidence,
            evidence=result.evidence,
            error_message="overflow encountered in exp",
        )
        self.assertIn("fit-physical", prompt)
        self.assertIn("fit-physical", prompt)
        self.assertIn("overflow", prompt)


# ════════════════════════════════════════════════════════════════════
# Tests: Orchestrator diagnosis (P6-T3 + P6-T4)
# ════════════════════════════════════════════════════════════════════


class TestRunStageDiagnosis(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.fake_ini = os.path.join(self.tmpdir, "dummy.ini")

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _make_mock_ctx(self, enable_diagnosis=True):
        mock_ctx = MagicMock()
        mock_ctx.agent_configs = {
            "DG_TFT_ENABLE_DIAGNOSIS": enable_diagnosis,
            "DG_TFT_MODELING_PYTHON_EXE": sys.executable,
        }
        mock_ctx.subagent_mute = True
        mock_ctx.console = None
        mock_ctx.agent_list = {}

        wm = MagicMock()
        run = WorkflowRun(run_id=1)
        wm.get_run.return_value = run
        mock_ctx.workflow_man = wm
        return mock_ctx

    def _make_failed_stage_result(self, stage_name="fit-physical", error="divergence"):
        return StageResult(
            ok=False,
            stage=stage_name,
            error=error,
            summary=f"Stage {stage_name} failed",
        )

    def test_classification_stored_when_diagnosis_disabled(self):
        from src.workflow.dg_tft.orchestrator import _run_stage_diagnosis

        ctx = self._make_mock_ctx(enable_diagnosis=False)
        sr = self._make_failed_stage_result(stage_name="check-data",
                                             error="FileNotFoundError: missing.csv")
        _run_stage_diagnosis(ctx, 1, sr, self.tmpdir, self.fake_ini)

        run = ctx.workflow_man.get_run(1)
        self.assertIn("diagnosis", run.extra)
        diag = run.extra["diagnosis"]
        self.assertEqual(diag["category"], "DATA_ERROR")
        self.assertGreater(diag["confidence"], 0.0)
        self.assertIn("FileNotFoundError", diag["evidence"])

    def test_classification_stored_for_numerical_error(self):
        from src.workflow.dg_tft.orchestrator import _run_stage_diagnosis

        ctx = self._make_mock_ctx(enable_diagnosis=False)
        sr = self._make_failed_stage_result(stage_name="train-static",
                                             error="loss is nan at epoch 42")
        _run_stage_diagnosis(ctx, 1, sr, self.tmpdir, self.fake_ini)

        run = ctx.workflow_man.get_run(1)
        diag = run.extra["diagnosis"]
        self.assertEqual(diag["category"], "TRAINING_NUMERICAL_ERROR")
        self.assertIn("nan", diag["evidence"])

    def test_classification_for_config_error(self):
        from src.workflow.dg_tft.orchestrator import _run_stage_diagnosis

        ctx = self._make_mock_ctx(enable_diagnosis=False)
        sr = self._make_failed_stage_result(stage_name="fit-physical",
                                             error="configparser.NoSectionError: 'paths'")
        _run_stage_diagnosis(ctx, 1, sr, self.tmpdir, self.fake_ini)

        run = ctx.workflow_man.get_run(1)
        diag = run.extra["diagnosis"]
        self.assertEqual(diag["category"], "CONFIG_ERROR")
        self.assertIn("configparser", diag["evidence"])

    def test_suggestion_included(self):
        from src.workflow.dg_tft.orchestrator import _run_stage_diagnosis

        ctx = self._make_mock_ctx(enable_diagnosis=False)
        sr = self._make_failed_stage_result(stage_name="check-data",
                                             error="missing sheet 'long'")
        _run_stage_diagnosis(ctx, 1, sr, self.tmpdir, self.fake_ini)

        run = ctx.workflow_man.get_run(1)
        diag = run.extra["diagnosis"]
        self.assertIsInstance(diag["suggestion"], str)
        self.assertGreater(len(diag["suggestion"]), 0)

    def test_handles_missing_run(self):
        from src.workflow.dg_tft.orchestrator import _run_stage_diagnosis

        ctx = self._make_mock_ctx(enable_diagnosis=False)
        ctx.workflow_man.get_run.return_value = None
        sr = self._make_failed_stage_result()
        _run_stage_diagnosis(ctx, 999, sr, self.tmpdir, self.fake_ini)

    def test_subagent_launch_handles_failure_gracefully(self):
        from src.workflow.dg_tft.orchestrator import _run_stage_diagnosis

        ctx = self._make_mock_ctx(enable_diagnosis=True)
        sr = self._make_failed_stage_result(stage_name="check-data",
                                             error="FileNotFoundError: missing.csv")
        _run_stage_diagnosis(ctx, 1, sr, self.tmpdir, self.fake_ini)

        run = ctx.workflow_man.get_run(1)
        self.assertIn("diagnosis", run.extra)
        diag = run.extra["diagnosis"]
        self.assertIn("subagent_result", diag)
        self.assertIn("Diagnosis subagent could not complete", diag["subagent_result"])

    def test_classification_runs_even_when_subagent_fails(self):
        from src.workflow.dg_tft.orchestrator import _run_stage_diagnosis

        ctx = self._make_mock_ctx(enable_diagnosis=True)
        sr = self._make_failed_stage_result(stage_name="fit-physical",
                                             error="overflow encountered")
        _run_stage_diagnosis(ctx, 1, sr, self.tmpdir, self.fake_ini)

        run = ctx.workflow_man.get_run(1)
        diag = run.extra["diagnosis"]
        self.assertEqual(diag["category"], "TRAINING_NUMERICAL_ERROR")


class TestCollectLogSnippet(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_returns_empty_when_no_logs(self):
        from src.workflow.dg_tft.orchestrator import _collect_log_snippet
        result = _collect_log_snippet(self.tmpdir, "check-data")
        self.assertEqual(result, "")

    def test_reads_stage_log_file(self):
        from src.workflow.dg_tft.orchestrator import _collect_log_snippet
        log_path = os.path.join(self.tmpdir, "check-data_stderr.log")
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("line1\nline2\nline3\n")
        result = _collect_log_snippet(self.tmpdir, "check-data")
        self.assertIn("line1", result)
        self.assertIn("line2", result)

    def test_truncates_long_log(self):
        from src.workflow.dg_tft.orchestrator import _collect_log_snippet
        log_path = os.path.join(self.tmpdir, "output.log")
        lines = [f"line{i}" for i in range(100)]
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        result = _collect_log_snippet(self.tmpdir, "fit-physical")
        result_lines = result.splitlines()
        self.assertLessEqual(len(result_lines), 60)


# ════════════════════════════════════════════════════════════════════
# Tests: Report builder diagnosis integration (P6-T5)
# ════════════════════════════════════════════════════════════════════


class TestReportDiagnosisIntegration(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _make_run_data(self, diagnosis=None):
        run_data = {
            "run_id": 1,
            "status": "FAILED",
            "created_at": "2026-06-26 10:00:00",
            "updated_at": "2026-06-26 10:05:00",
            "dataset_dir": "/data",
            "output_root": self.tmpdir,
            "prep_id": None,
            "hysteresis_enabled": False,
            "temperature_enabled": False,
            "warnings": [],
            "errors": ["Stage check-data failed"],
            "metrics": {},
            "stage_results": {
                "check-data": {
                    "ok": False,
                    "skipped": False,
                    "skip_reason": "",
                    "error": "FileNotFoundError: missing.csv",
                    "warnings": [],
                    "metrics": {},
                },
                "fit-physical": None,
                "train-static": None,
                "train-hysteresis": None,
            },
            "artifacts": {},
            "extra": {},
        }
        if diagnosis is not None:
            run_data["extra"]["diagnosis"] = diagnosis
        return run_data

    def test_json_summary_includes_diagnosis_when_present(self):
        from src.workflow.dg_tft.report_builder import _build_json_summary

        diag = {
            "category": "DATA_ERROR",
            "confidence": 0.9,
            "evidence": ["FileNotFoundError"],
            "suggestion": "Check data path",
            "subagent_result": "The file missing.csv was not found.",
        }
        run_data = self._make_run_data(diagnosis=diag)
        manifest = {"items": [], "total_items": 0, "missing_items": 0}
        summary = _build_json_summary(run_data, manifest, None)
        self.assertIn("diagnosis", summary)
        self.assertEqual(summary["diagnosis"]["category"], "DATA_ERROR")
        self.assertEqual(summary["diagnosis"]["confidence"], 0.9)

    def test_json_summary_no_diagnosis_when_absent(self):
        from src.workflow.dg_tft.report_builder import _build_json_summary

        run_data = self._make_run_data()
        manifest = {"items": [], "total_items": 0, "missing_items": 0}
        summary = _build_json_summary(run_data, manifest, None)
        self.assertNotIn("diagnosis", summary)

    def test_md_summary_includes_diagnosis_section(self):
        from src.workflow.dg_tft.report_builder import _build_md_summary

        diag = {
            "category": "CONFIG_ERROR",
            "confidence": 0.9,
            "evidence": ["configparser", "no section"],
            "suggestion": "Check model_config.ini paths section",
            "subagent_result": "The config file is missing the [paths] section.",
        }
        run_data = self._make_run_data(diagnosis=diag)
        manifest = {"items": [], "total_items": 0, "missing_items": 0}
        md = _build_md_summary(run_data, manifest, None)
        self.assertIn("## 4. Diagnosis", md)
        self.assertIn("CONFIG_ERROR", md)
        self.assertIn("configparser", md)
        self.assertIn("AI Diagnosis", md)
        self.assertIn("The config file is missing", md)

    def test_md_summary_no_diagnosis_when_absent(self):
        from src.workflow.dg_tft.report_builder import _build_md_summary

        run_data = self._make_run_data()
        manifest = {"items": [], "total_items": 0, "missing_items": 0}
        md = _build_md_summary(run_data, manifest, None)
        self.assertNotIn("Diagnosis", md)

    def test_md_summary_diagnosis_without_subagent_result(self):
        from src.workflow.dg_tft.report_builder import _build_md_summary

        diag = {
            "category": "DATA_ERROR",
            "confidence": 0.9,
            "evidence": ["missing sheet"],
            "suggestion": "Add 'long' sheet",
        }
        run_data = self._make_run_data(diagnosis=diag)
        manifest = {"items": [], "total_items": 0, "missing_items": 0}
        md = _build_md_summary(run_data, manifest, None)
        self.assertIn("## 4. Diagnosis", md)
        self.assertIn("DATA_ERROR", md)
        self.assertNotIn("AI Diagnosis", md)


if __name__ == "__main__":
    unittest.main()
