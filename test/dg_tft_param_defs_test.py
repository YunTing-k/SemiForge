# -*- coding: utf-8 -*-
"""
Unit tests for parameter definitions and validation (TODO 3.2).

Run: python -m unittest test.dg_tft_param_defs_test -v
"""
import sys
import os
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# ---------------------------------------------------------------------------
# 3.2.1–3.2.3: Parameter table integrity
# ---------------------------------------------------------------------------

class TestParamTables(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from src.workflow.dg_tft import param_defs as _pd
        cls.PHYSICAL_FIT_PARAMS = _pd.PHYSICAL_FIT_PARAMS
        cls.FULL_MODEL_PARAMS = _pd.FULL_MODEL_PARAMS
        cls.HYSTERESIS_PARAMS = _pd.HYSTERESIS_PARAMS

    def _check_table(self, table, name):
        """Verify a param table has all required keys for each entry."""
        required = {"type", "lo", "hi", "default", "desc"}
        for key, pdef in table.items():
            missing = required - set(pdef.keys())
            self.assertEqual(
                missing, set(),
                f"[{name}] Param '{key}' missing keys: {missing}"
            )
            self.assertIn(pdef["type"], ("int", "float", "bool", "string"),
                          f"[{name}] Param '{key}' has unknown type '{pdef['type']}'")

    def test_physical_fit_table_structure(self):
        self._check_table(self.PHYSICAL_FIT_PARAMS, "PHYSICAL_FIT")
        self.assertIn("max_iv_points", self.PHYSICAL_FIT_PARAMS)
        self.assertIn("max_cv_points", self.PHYSICAL_FIT_PARAMS)
        self.assertIn("max_nfev_scale", self.PHYSICAL_FIT_PARAMS)

    def test_kan_params_present(self):
        """TODO 3.2 says '含 KAN 超参' — [base_pkan] params must be in FULL_MODEL_PARAMS."""
        for p in ("epochs_shared", "epochs_iv", "epochs_cv", "kan_hidden", "kan_grid", "kan_k",
                  "batch_iv_kan", "batch_cv_kan"):
            self.assertIn(p, self.FULL_MODEL_PARAMS, f"Missing KAN param: {p}")

    def test_temperature_params_present(self):
        for p in ("temperature_mode", "minimum_span_K", "kan_scale"):
            self.assertIn(p, self.FULL_MODEL_PARAMS, f"Missing temperature param: {p}")

    def test_print_every_present(self):
        self.assertIn("print_every", self.FULL_MODEL_PARAMS)
        self.assertIn("print_every", self.HYSTERESIS_PARAMS)

    def test_cross_section_params_have_section_field(self):
        """Params from non-primary sections must declare their 'section'."""
        cross = {k: v for k, v in self.FULL_MODEL_PARAMS.items() if "section" in v}
        self.assertIn("epochs_shared", cross)
        self.assertEqual(cross["epochs_shared"]["section"], "base_pkan")
        self.assertIn("temperature_mode", cross)
        self.assertEqual(cross["temperature_mode"]["section"], "temperature")

    def test_primary_section_params_have_no_section_field(self):
        """Params in the primary [full_model] section should NOT have a section field."""
        primary_params = ["epochs", "learning_rate", "cv_weight", "early_stop_patience"]
        for p in primary_params:
            self.assertNotIn("section", self.FULL_MODEL_PARAMS[p],
                             f"Primary param '{p}' should not have explicit section")

    def test_full_model_table_structure(self):
        self._check_table(self.FULL_MODEL_PARAMS, "FULL_MODEL")
        self.assertGreaterEqual(len(self.FULL_MODEL_PARAMS), 44)

    def test_hysteresis_table_structure(self):
        self._check_table(self.HYSTERESIS_PARAMS, "HYSTERESIS")
        self.assertIn("amp_parameterization", self.HYSTERESIS_PARAMS)
        self.assertIn("hidden_capture", self.HYSTERESIS_PARAMS)

    def test_bounds_consistency(self):
        """lo must be <= hi for numeric params."""
        for name, table in [
            ("PHYSICAL_FIT", self.PHYSICAL_FIT_PARAMS),
            ("FULL_MODEL", self.FULL_MODEL_PARAMS),
            ("HYSTERESIS", self.HYSTERESIS_PARAMS),
        ]:
            for key, pdef in table.items():
                if pdef["type"] in ("int", "float"):
                    self.assertIsNotNone(pdef["lo"], f"[{name}] {key} numeric but lo is None")
                    self.assertIsNotNone(pdef["hi"], f"[{name}] {key} numeric but hi is None")
                    self.assertLessEqual(pdef["lo"], pdef["hi"],
                                         f"[{name}] {key} lo={pdef['lo']} > hi={pdef['hi']}")

    def test_bool_params_have_no_bounds(self):
        """Bool params should have lo=None, hi=None."""
        for name, table in [
            ("FULL_MODEL", self.FULL_MODEL_PARAMS),
            ("HYSTERESIS", self.HYSTERESIS_PARAMS),
        ]:
            for key, pdef in table.items():
                if pdef["type"] == "bool":
                    self.assertIsNone(pdef["lo"], f"[{name}] bool param {key} lo should be None")
                    self.assertIsNone(pdef["hi"], f"[{name}] bool param {key} hi should be None")

    def test_enum_params_have_enum_list(self):
        """String params with enum should have a non-empty enum list."""
        for name, table in [
            ("HYSTERESIS", self.HYSTERESIS_PARAMS),
            ("FULL_MODEL", self.FULL_MODEL_PARAMS),
            ("PHYSICAL_FIT", self.PHYSICAL_FIT_PARAMS),
        ]:
            for key, pdef in table.items():
                if "enum" in pdef:
                    self.assertIsInstance(pdef["enum"], list, f"[{name}] {key} enum is not list")
                    self.assertGreater(len(pdef["enum"]), 0, f"[{name}] {key} enum is empty")
                    self.assertIn(pdef["default"], pdef["enum"],
                                  f"[{name}] {key} default not in enum")

    def test_default_within_bounds(self):
        """Default values must be within [lo, hi]."""
        for name, table in [
            ("PHYSICAL_FIT", self.PHYSICAL_FIT_PARAMS),
            ("FULL_MODEL", self.FULL_MODEL_PARAMS),
            ("HYSTERESIS", self.HYSTERESIS_PARAMS),
        ]:
            for key, pdef in table.items():
                if pdef["type"] in ("int", "float") and pdef.get("lo") is not None:
                    self.assertGreaterEqual(
                        pdef["default"], pdef["lo"],
                        f"[{name}] {key} default={pdef['default']} < lo={pdef['lo']}"
                    )
                    self.assertLessEqual(
                        pdef["default"], pdef["hi"],
                        f"[{name}] {key} default={pdef['default']} > hi={pdef['hi']}"
                    )


# ---------------------------------------------------------------------------
# 3.2.4: Schema auto-generation
# ---------------------------------------------------------------------------

class TestSchemaGeneration(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from src.workflow.dg_tft.param_defs import (
            params_to_tool_schema,
            get_stage_param_schema,
            PHYSICAL_FIT_PARAMS,
        )
        cls.params_to_tool_schema = staticmethod(params_to_tool_schema)
        cls.get_stage_param_schema = staticmethod(get_stage_param_schema)
        cls.PHYSICAL_FIT_PARAMS = PHYSICAL_FIT_PARAMS

    def test_schema_overall_structure(self):
        schema = self.params_to_tool_schema(self.PHYSICAL_FIT_PARAMS, stage_label="fit-physical")
        self.assertEqual(schema["type"], "function")
        fn = schema["function"]
        self.assertEqual(fn["name"], "adjust_params")
        self.assertIn("description", fn)
        self.assertEqual(fn["parameters"]["type"], "object")
        self.assertIn("reasoning", fn["parameters"]["properties"])
        self.assertEqual(fn["parameters"]["required"], ["reasoning"])
        self.assertIs(fn["parameters"]["additionalProperties"], False)

    def test_schema_int_param(self):
        schema = self.params_to_tool_schema(self.PHYSICAL_FIT_PARAMS)
        props = schema["function"]["parameters"]["properties"]
        iv = props["max_iv_points"]
        self.assertEqual(iv["type"], "integer")
        self.assertIn("minimum", iv)
        self.assertIn("maximum", iv)
        self.assertEqual(iv["minimum"], 5000)
        self.assertEqual(iv["maximum"], 200000)

    def test_schema_float_param(self):
        schema = self.params_to_tool_schema(self.PHYSICAL_FIT_PARAMS)
        props = schema["function"]["parameters"]["properties"]
        nfev = props["max_nfev_scale"]
        self.assertEqual(nfev["type"], "number")
        self.assertEqual(nfev["minimum"], 0.5)
        self.assertEqual(nfev["maximum"], 5.0)

    def test_schema_bool_param(self):
        from src.workflow.dg_tft.param_defs import HYSTERESIS_PARAMS
        schema = self.params_to_tool_schema(HYSTERESIS_PARAMS)
        props = schema["function"]["parameters"]["properties"]
        en = props["enabled"]
        self.assertEqual(en["type"], "boolean")
        self.assertNotIn("minimum", en)
        self.assertNotIn("maximum", en)

    def test_schema_enum_param(self):
        from src.workflow.dg_tft.param_defs import HYSTERESIS_PARAMS
        schema = self.params_to_tool_schema(HYSTERESIS_PARAMS)
        props = schema["function"]["parameters"]["properties"]
        amp = props["amp_parameterization"]
        self.assertEqual(amp["type"], "string")
        self.assertIn("enum", amp)
        self.assertIn("conservative", amp["enum"])
        self.assertIn("aggressive", amp["enum"])

    def test_schema_description_contains_stage_label(self):
        schema = self.params_to_tool_schema(self.PHYSICAL_FIT_PARAMS, stage_label="fit-physical")
        desc = schema["function"]["description"]
        self.assertIn("fit-physical", desc)

    def test_get_stage_param_schema_valid(self):
        schema = self.get_stage_param_schema("fit-physical")
        self.assertIsNotNone(schema)
        self.assertIn("max_nfev_scale", schema["function"]["parameters"]["properties"])

    def test_get_stage_param_schema_train_static(self):
        schema = self.get_stage_param_schema("train-static")
        self.assertIsNotNone(schema)
        self.assertIn("learning_rate", schema["function"]["parameters"]["properties"])

    def test_get_stage_param_schema_train_hysteresis(self):
        schema = self.get_stage_param_schema("train-hysteresis")
        self.assertIsNotNone(schema)
        self.assertIn("hidden_capture", schema["function"]["parameters"]["properties"])

    def test_get_stage_param_schema_none_for_non_adjustable(self):
        self.assertIsNone(self.get_stage_param_schema("check-data"))
        self.assertIsNone(self.get_stage_param_schema("final-report"))
        self.assertIsNone(self.get_stage_param_schema("nonexistent"))


# ---------------------------------------------------------------------------
# 3.2.5: Hard guard validation
# ---------------------------------------------------------------------------

class TestHardGuardValidation(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from src.workflow.dg_tft.quality_guard import validate_agent_adjustments
        from src.workflow.dg_tft.param_defs import (
            PHYSICAL_FIT_PARAMS,
            FULL_MODEL_PARAMS,
            HYSTERESIS_PARAMS,
        )
        cls.validate = staticmethod(validate_agent_adjustments)
        cls.PHYSICAL_FIT_PARAMS = PHYSICAL_FIT_PARAMS
        cls.FULL_MODEL_PARAMS = FULL_MODEL_PARAMS
        cls.HYSTERESIS_PARAMS = HYSTERESIS_PARAMS
        cls.PF_SECTIONS = ["physical_fit"]
        cls.FM_SECTIONS = ["full_model", "base_pkan", "temperature"]
        cls.HY_SECTIONS = ["hysteresis"]

    # ── Normal valid params pass through ─────────────────────────

    def test_valid_params_pass_through(self):
        sanitized, warnings = self.validate(
            {"max_nfev_scale": 2.0},
            self.PHYSICAL_FIT_PARAMS,
            self.PF_SECTIONS,
            "fit-physical",
        )
        self.assertEqual(warnings, [])
        self.assertEqual(sanitized, {"max_nfev_scale": 2.0})

    def test_valid_int_param(self):
        sanitized, warnings = self.validate(
            {"max_iv_points": 100000},
            self.PHYSICAL_FIT_PARAMS,
            self.PF_SECTIONS,
        )
        self.assertEqual(warnings, [])
        self.assertEqual(sanitized["max_iv_points"], 100000)

    def test_valid_string_enum(self):
        sanitized, warnings = self.validate(
            {"amp_parameterization": "aggressive"},
            self.HYSTERESIS_PARAMS,
            self.HY_SECTIONS,
        )
        self.assertEqual(warnings, [])
        self.assertEqual(sanitized["amp_parameterization"], "aggressive")

    def test_valid_bool(self):
        sanitized, warnings = self.validate(
            {"enable_cv_residual": False},
            self.FULL_MODEL_PARAMS,
            self.FM_SECTIONS,
        )
        self.assertEqual(warnings, [])
        self.assertEqual(sanitized["enable_cv_residual"], False)

    def test_multiple_valid_params(self):
        sanitized, warnings = self.validate(
            {
                "max_nfev_scale": 3.0,
                "max_iv_points": 75000,
            },
            self.PHYSICAL_FIT_PARAMS,
            self.PF_SECTIONS,
        )
        self.assertEqual(warnings, [])
        self.assertEqual(len(sanitized), 2)

    # ── Unknown params dropped ───────────────────────────────────

    def test_unknown_param_dropped(self):
        sanitized, warnings = self.validate(
            {"nonexistent_key": 42},
            self.PHYSICAL_FIT_PARAMS,
            self.PF_SECTIONS,
        )
        self.assertEqual(sanitized, {})
        self.assertEqual(len(warnings), 1)
        self.assertIn("nonexistent_key", warnings[0])

    def test_unknown_params_dropped_known_kept(self):
        sanitized, warnings = self.validate(
            {"max_nfev_scale": 1.5, "fake": 999},
            self.PHYSICAL_FIT_PARAMS,
            self.PF_SECTIONS,
        )
        self.assertEqual(sanitized, {"max_nfev_scale": 1.5})
        self.assertEqual(len(warnings), 1)

    # ── Type coercion ────────────────────────────────────────────

    def test_string_int_coerced(self):
        sanitized, warnings = self.validate(
            {"max_iv_points": "100000"},
            self.PHYSICAL_FIT_PARAMS,
            self.PF_SECTIONS,
        )
        self.assertEqual(warnings, [])
        self.assertIsInstance(sanitized["max_iv_points"], int)
        self.assertEqual(sanitized["max_iv_points"], 100000)

    def test_string_float_coerced(self):
        sanitized, warnings = self.validate(
            {"max_nfev_scale": "2.5"},
            self.PHYSICAL_FIT_PARAMS,
            self.PF_SECTIONS,
        )
        self.assertEqual(warnings, [])
        self.assertIsInstance(sanitized["max_nfev_scale"], float)
        self.assertEqual(sanitized["max_nfev_scale"], 2.5)

    def test_float_to_int_coerced_when_whole(self):
        sanitized, warnings = self.validate(
            {"epochs": 500.0},
            self.FULL_MODEL_PARAMS,
            self.FM_SECTIONS,
        )
        self.assertEqual(warnings, [])
        self.assertIsInstance(sanitized["epochs"], int)
        self.assertEqual(sanitized["epochs"], 500)

    def test_float_to_int_fails_when_fractional(self):
        sanitized, warnings = self.validate(
            {"epochs": 500.5},
            self.FULL_MODEL_PARAMS,
            self.FM_SECTIONS,
        )
        self.assertEqual(sanitized, {})
        self.assertEqual(len(warnings), 1)
        self.assertIn("Type mismatch", warnings[0])

    def test_string_bool_coerced(self):
        sanitized, warnings = self.validate(
            {"enabled": "yes"},
            self.HYSTERESIS_PARAMS,
            self.HY_SECTIONS,
        )
        self.assertEqual(warnings, [])
        self.assertTrue(sanitized["enabled"])

    def test_string_bool_no_coerced(self):
        sanitized, warnings = self.validate(
            {"enabled": "false"},
            self.HYSTERESIS_PARAMS,
            self.HY_SECTIONS,
        )
        self.assertEqual(sanitized["enabled"], False)

    def test_bad_string_for_int_dropped(self):
        sanitized, warnings = self.validate(
            {"max_iv_points": "not_a_number"},
            self.PHYSICAL_FIT_PARAMS,
            self.PF_SECTIONS,
        )
        self.assertEqual(sanitized, {})
        self.assertEqual(len(warnings), 1)
        self.assertIn("Type mismatch", warnings[0])

    # ── Enum validation ──────────────────────────────────────────

    def test_enum_valid(self):
        sanitized, warnings = self.validate(
            {"regime": "all"},
            self.HYSTERESIS_PARAMS,
            self.HY_SECTIONS,
        )
        self.assertEqual(warnings, [])
        self.assertEqual(sanitized["regime"], "all")

    def test_enum_invalid_dropped(self):
        sanitized, warnings = self.validate(
            {"regime": "super_saturation"},
            self.HYSTERESIS_PARAMS,
            self.HY_SECTIONS,
        )
        self.assertEqual(sanitized, {})
        self.assertEqual(len(warnings), 1)
        self.assertIn("not in", warnings[0])

    def test_enum_case_insensitive(self):
        sanitized, warnings = self.validate(
            {"amp_parameterization": "CONSERVATIVE"},
            self.HYSTERESIS_PARAMS,
            self.HY_SECTIONS,
        )
        self.assertEqual(warnings, [])
        self.assertEqual(sanitized["amp_parameterization"], "CONSERVATIVE")

    # ── Bounds clamping ──────────────────────────────────────────

    def test_value_below_lo_clamped(self):
        sanitized, warnings = self.validate(
            {"max_nfev_scale": 0.1},
            self.PHYSICAL_FIT_PARAMS,
            self.PF_SECTIONS,
        )
        self.assertEqual(len(warnings), 1)
        self.assertIn("clamped", warnings[0])
        self.assertEqual(sanitized["max_nfev_scale"], 0.5)

    def test_value_above_hi_clamped(self):
        sanitized, warnings = self.validate(
            {"max_nfev_scale": 100.0},
            self.PHYSICAL_FIT_PARAMS,
            self.PF_SECTIONS,
        )
        self.assertEqual(len(warnings), 1)
        self.assertIn("clamped", warnings[0])
        self.assertEqual(sanitized["max_nfev_scale"], 5.0)

    def test_value_at_boundary_passes(self):
        sanitized, warnings = self.validate(
            {"max_nfev_scale": 5.0},
            self.PHYSICAL_FIT_PARAMS,
            self.PF_SECTIONS,
        )
        self.assertEqual(warnings, [])
        self.assertEqual(sanitized["max_nfev_scale"], 5.0)

    def test_value_at_lo_boundary_passes(self):
        sanitized, warnings = self.validate(
            {"max_iv_points": 5000},
            self.PHYSICAL_FIT_PARAMS,
            self.PF_SECTIONS,
        )
        self.assertEqual(warnings, [])
        self.assertEqual(sanitized["max_iv_points"], 5000)

    def test_int_value_clamped_preserves_int(self):
        sanitized, warnings = self.validate(
            {"max_iv_points": 1},
            self.PHYSICAL_FIT_PARAMS,
            self.PF_SECTIONS,
        )
        self.assertEqual(sanitized["max_iv_points"], 5000)
        self.assertIsInstance(sanitized["max_iv_points"], int)

    # ── Mixed scenarios ──────────────────────────────────────────

    def test_empty_params(self):
        sanitized, warnings = self.validate(
            {},
            self.PHYSICAL_FIT_PARAMS,
            self.PF_SECTIONS,
        )
        self.assertEqual(sanitized, {})
        self.assertEqual(warnings, [])

    def test_all_invalid_dropped(self):
        sanitized, warnings = self.validate(
            {"fake1": 1, "fake2": 2},
            self.PHYSICAL_FIT_PARAMS,
            self.PF_SECTIONS,
        )
        self.assertEqual(sanitized, {})
        self.assertEqual(len(warnings), 2)


# ---------------------------------------------------------------------------
# 3.2.6: Soft guard prompt
# ---------------------------------------------------------------------------

class TestSoftGuardPrompt(unittest.TestCase):

    def test_prompt_returns_string(self):
        from src.workflow.dg_tft.param_defs import get_soft_guard_prompt
        prompt = get_soft_guard_prompt()
        self.assertIsInstance(prompt, str)
        self.assertGreater(len(prompt), 100)

    def test_prompt_mentions_key_topics(self):
        from src.workflow.dg_tft.param_defs import get_soft_guard_prompt
        prompt = get_soft_guard_prompt()
        self.assertIn("self-check", prompt)
        self.assertIn("adjust_params", prompt)
        self.assertIn("Bounds", prompt)


# ---------------------------------------------------------------------------
# _STAGE_PARAM_MAP integrity
# ---------------------------------------------------------------------------

class TestStageParamMap(unittest.TestCase):

    def test_map_coverage(self):
        from src.workflow.dg_tft.param_defs import (
            _STAGE_PARAM_MAP,
            PHYSICAL_FIT_PARAMS,
            FULL_MODEL_PARAMS,
            HYSTERESIS_PARAMS,
        )
        self.assertIn("fit-physical", _STAGE_PARAM_MAP)
        self.assertIn("train-static", _STAGE_PARAM_MAP)
        self.assertIn("train-hysteresis", _STAGE_PARAM_MAP)
        # check each map entry points to the correct table
        self.assertIs(_STAGE_PARAM_MAP["fit-physical"][1], PHYSICAL_FIT_PARAMS)
        self.assertIs(_STAGE_PARAM_MAP["train-static"][1], FULL_MODEL_PARAMS)
        self.assertIs(_STAGE_PARAM_MAP["train-hysteresis"][1], HYSTERESIS_PARAMS)

    def test_section_names_match_ini(self):
        from src.workflow.dg_tft.param_defs import _STAGE_PARAM_MAP
        all_sections = set()
        for sections in (v[0] for v in _STAGE_PARAM_MAP.values()):
            all_sections.update(sections)
        self.assertEqual(all_sections, {"physical_fit", "preprocessing", "full_model", "base_pkan", "temperature", "hysteresis"})


if __name__ == "__main__":
    unittest.main()
