"""Isothermal IV sweeps must stay intact in training and curve evaluation."""
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src/device_modeling/dg_tft/engine"))
import dg_tft_full_model as engine
from iv_curve_optimization import IVCurveLayout


def sweeps():
    rows = []
    for family, swept in [("IDVBG", "VBG_V"), ("IDVTG", "VTG_V"), ("IDVD", "VDS_V")]:
        for temperature in [293.15, 333.15]:
            for voltage in np.linspace(0.5, 2.5, 11):
                row = dict(family=family, source_file="combined.xlsx", VBG_V=2., VTG_V=1., VDS_V=1.)
                row[swept] = voltage
                row.update(temp_K=temperature, ID_A=10 ** (-9. + .5 * voltage + (temperature - 293.15) / 40.))
                rows.append(row)
    return pd.DataFrame(rows)


def tensors(frame):
    return engine.make_iv_tensors_v2(frame, torch.device("cpu"), torch.float64)


class TemperatureCurveTests(unittest.TestCase):
    def test_training_curves_are_isothermal_and_have_physical_slopes(self):
        data = tensors(sweeps())
        self.assertEqual(len(torch.unique(data["curve_code"])), 6)
        for code in torch.unique(data["curve_code"]):
            self.assertEqual(len(torch.unique(data["temp_K"][data["curve_code"] == code])), 1)
        layout = IVCurveLayout(data, 0, engine.ID_FLOOR)
        torch.testing.assert_close(layout.target_slope[layout.pairs], torch.full((60,), .5, dtype=torch.float64))
        self.assertTrue(bool((layout.dx[layout.pairs] > .19).all()))

    def test_temperature_aliases_and_celsius_give_identical_curves(self):
        frame = sweeps()
        expected = tensors(frame)
        for name in ["temperature_K", "T_K", "TEMP_K", "temp_C", "temperature_C", "T_C", "TEMP_C"]:
            other = frame.rename(columns={"temp_K": name})
            if name.endswith("C"):
                other[name] -= 273.15
            actual = tensors(other)
            torch.testing.assert_close(actual["temp_K"], expected["temp_K"])
            torch.testing.assert_close(actual["curve_code"], expected["curve_code"])

    def test_nominal_and_single_temperature_keep_existing_curve_order(self):
        frame = sweeps().query("temp_K == 293.15").copy()
        actual = tensors(frame)
        nominal = tensors(frame.drop(columns="temp_K"))
        torch.testing.assert_close(actual["curve_code"], nominal["curve_code"])
        self.assertEqual(len(torch.unique(nominal["curve_code"])), 3)
        self.assertTrue(bool(torch.isnan(nominal["temp_K"]).all()))

    def test_missing_temperature_is_retained_as_its_own_group(self):
        frame = sweeps().query("family == 'IDVD'").copy()
        frame.loc[frame.temp_K > 300, "temp_K"] = np.nan
        groups = list(engine._iv_curve_groups(frame, ["source_file", "VBG_V", "VTG_V"]))
        self.assertEqual(sorted(len(group) for _, group in groups), [11, 11])
        self.assertEqual(len(torch.unique(tensors(frame)["curve_code"])), 2)

    def test_holdout_selects_whole_isothermal_curves(self):
        frame = sweeps()
        heldout, report = engine.iv_cross_condition_holdout(frame, stride=2)
        self.assertEqual(report["heldout_conditions"], 3)
        self.assertEqual(len(heldout), 33)
        for _, group in heldout.groupby(["family", "temp_K"]):
            self.assertEqual(len(group), 11)

    def test_summary_does_not_turn_temperature_offsets_into_voltage_slopes(self):
        frame = sweeps()
        # A constant error on each isothermal sweep has exactly zero slope
        # error, even when its offset differs between temperatures.
        offset = (frame.temp_K - 293.15) / 200.
        for prefix, pred in [("physical", "ID_physical_A"), ("base", "ID_base_pkan_A"), ("residual", "ID_residual_A")]:
            frame[pred] = frame.ID_A * 10 ** offset
            frame[f"log10_error_{prefix}_dec"] = offset
        result = engine.summarize(frame, pd.DataFrame())
        self.assertEqual(result["residual_iv_curve_rms_p95_count"], 6)
        self.assertEqual(result["residual_iv_slope_rmse_dec_per_v_count"], 60)
        self.assertLess(result["residual_iv_slope_rmse_dec_per_v"], 1.e-12)
        self.assertNotIn("_iv_curve_temperature_K", frame.columns)


if __name__ == "__main__":
    unittest.main()
