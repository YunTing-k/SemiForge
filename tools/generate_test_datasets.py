# -*- coding: utf-8 -*-
"""
Generate 3 test datasets for Phase 9 DG-TFT integration testing.

Usage: python tools/generate_test_datasets.py
"""
import sys
import shutil
from pathlib import Path

try:
    from openpyxl import Workbook
except ImportError:
    print("openpyxl not available - cannot generate test datasets")
    sys.exit(1)

TOOL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TOOL_DIR.parent
TEST_DATA_DIR = PROJECT_ROOT / "test_data"

IV_HEADERS = ["VD_V", "VBG_V", "VTG_V", "ID_A"]
IV_DATA = [
    [0.1, -5.0, -5.0, 1.0e-6],
    [0.1, -4.0, -4.0, 2.0e-6],
    [0.1, -3.0, -3.0, 3.5e-6],
    [0.1, -2.0, -2.0, 5.0e-6],
    [0.1, -1.0, -1.0, 7.0e-6],
]
CV_DATA = [
    [-5.0, 1.2e-12, -5.0, 1.3e-12, -5.0, 1.4e-12, -5.0, 1.5e-12],
    [-4.0, 1.3e-12, -4.0, 1.4e-12, -4.0, 1.5e-12, -4.0, 1.6e-12],
    [-3.0, 1.4e-12, -3.0, 1.5e-12, -3.0, 1.6e-12, -3.0, 1.7e-12],
    [-2.0, 1.5e-12, -2.0, 1.6e-12, -2.0, 1.7e-12, -2.0, 1.8e-12],
    [-1.0, 1.6e-12, -1.0, 1.7e-12, -1.0, 1.8e-12, -1.0, 1.9e-12],
]
IT_HEADERS = ["time(s)", "vds_V", "vbg_V", "vtg_V", "id_A", "temp_K"]
IT_DATA = [
    [0.0e-6, 0.1, -5.0, -5.0, 1.0e-6, 300],
    [1.0e-6, 0.1, -5.0, -4.0, 1.5e-6, 300],
    [2.0e-6, 0.1, -5.0, -3.0, 2.0e-6, 300],
    [3.0e-6, 0.1, -5.0, -2.0, 2.5e-6, 300],
    [4.0e-6, 0.1, -5.0, -1.0, 3.0e-6, 300],
]


def _make_iv_xlsx(path, sheet_name="long"):
    if path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
    wb = Workbook()
    ws = wb.active
    ws.title = sheet_name
    for i, h in enumerate(IV_HEADERS, 1):
        ws.cell(row=1, column=i, value=h)
    for r, row in enumerate(IV_DATA, 2):
        for c, val in enumerate(row, 1):
            ws.cell(row=r, column=c, value=val)
    wb.save(str(path))
    wb.close()


def _make_cv_xlsx(path):
    wb = Workbook()
    first = True
    for sn in ["cgg", "cbg", "ctg", "vds"]:
        if first:
            ws = wb.active
            ws.title = sn
            first = False
        else:
            ws = wb.create_sheet(title=sn)
        for r, row in enumerate(CV_DATA):
            for c, val in enumerate(row, 1):
                ws.cell(row=r + 3, column=c, value=val)
    wb.save(str(path))
    wb.close()


def _make_it_xlsx(path):
    wb = Workbook()
    ws = wb.active
    ws.title = "data"
    for i, h in enumerate(IT_HEADERS, 1):
        ws.cell(row=1, column=i, value=h)
    for r, row in enumerate(IT_DATA, 2):
        for c, val in enumerate(row, 1):
            ws.cell(row=r, column=c, value=val)
    wb.save(str(path))
    wb.close()


def create_datasets():
    if TEST_DATA_DIR.exists():
        shutil.rmtree(str(TEST_DATA_DIR))
    TEST_DATA_DIR.mkdir(parents=True)

    # --- Dataset 1: static-only ---
    ds1 = TEST_DATA_DIR / "static_only"
    ds1.mkdir()
    _make_iv_xlsx(ds1 / "IDVBG_VDS0.1V_train.xlsx")
    _make_iv_xlsx(ds1 / "IDVTG_VDS0.1V_train.xlsx")
    _make_iv_xlsx(ds1 / "IDVD_VBG0V_train.xlsx")
    _make_cv_xlsx(ds1 / "CV_train.xlsx")
    print(f"[OK] {ds1}")

    # --- Dataset 2: full (static + I-t) ---
    ds2 = TEST_DATA_DIR / "full"
    ds2.mkdir()
    _make_iv_xlsx(ds2 / "IDVBG_VDS0.1V_train.xlsx")
    _make_iv_xlsx(ds2 / "IDVTG_VDS0.1V_train.xlsx")
    _make_iv_xlsx(ds2 / "IDVD_VBG0V_train.xlsx")
    _make_cv_xlsx(ds2 / "CV_train.xlsx")
    it_dir = ds2 / "hysteresis_it"
    it_dir.mkdir()
    _make_it_xlsx(it_dir / "I-t_sample1.xlsx")
    _make_it_xlsx(it_dir / "I-t_sample2.xlsx")
    print(f"[OK] {ds2}")

    # --- Dataset 3: corrupted ---
    ds3 = TEST_DATA_DIR / "corrupted"
    ds3.mkdir()
    _make_iv_xlsx(ds3 / "IDVTG_VDS0.1V_train.xlsx")
    _make_iv_xlsx(ds3 / "IDVD_VBG0V_train.xlsx")

    wb = Workbook()
    for sn in ["cgg", "cbg"]:
        ws = wb.active if wb.active.title else wb.create_sheet(title=sn)
        if sn == "cgg":
            ws.title = sn
        for r, row in enumerate(CV_DATA):
            for c, val in enumerate(row, 1):
                ws.cell(row=r + 3, column=c, value=val)
    wb.save(str(ds3 / "CV_train.xlsx"))
    wb.close()

    bad_it = ds3 / "hysteresis_it"
    bad_it.mkdir()
    wb = Workbook()
    ws = wb.active
    ws.title = "data"
    ws.cell(row=1, column=1, value="time(s)")
    ws.cell(row=1, column=2, value="id_A")
    for r in range(5):
        ws.cell(row=r + 2, column=1, value=float(r) * 1e-6)
        ws.cell(row=r + 2, column=2, value=1e-6 + r * 5e-7)
    wb.save(str(bad_it / "I-t_bad.xlsx"))
    wb.close()
    print(f"[OK] {ds3}")

    print("\nDone. 3 test datasets under", str(TEST_DATA_DIR))


if __name__ == "__main__":
    create_datasets()
