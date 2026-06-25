from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


IV_FAMILIES = {
    "IDVBG": ("IDVBG*_train.xlsx", "IDVBG_train.xlsx"),
    "IDVTG": ("IDVTG*_train.xlsx", "IDVTG_train.xlsx"),
    "IDVD": ("IDVD*_train.xlsx", "IDVD_train.xlsx"),
}
IV_REQUIRED = {"VD_V", "VBG_V", "VTG_V", "ID_A"}
TEMP_K_ALIASES = {"temp_K", "temperature_K", "T_K", "TEMP_K"}
TEMP_C_ALIASES = {"temp_C", "temperature_C", "T_C", "TEMP_C"}
CV_SHEETS = ["cgg", "cbg", "ctg", "vds"]


def check_iv_file(path: Path) -> dict:
    item: dict = {"file": path.name, "exists": path.exists(), "ok": False, "issues": []}
    if not path.exists():
        item["issues"].append("missing file")
        return item
    try:
        xl = pd.ExcelFile(path)
    except Exception as exc:
        item["issues"].append(f"cannot open workbook: {exc}")
        return item
    if "long" not in xl.sheet_names:
        item["issues"].append("missing sheet named 'long'")
        return item
    df = pd.read_excel(path, sheet_name="long", nrows=20)
    missing = sorted(IV_REQUIRED.difference(df.columns))
    if missing:
        item["issues"].append(f"missing columns: {missing}")
    temp_cols = sorted((TEMP_K_ALIASES | TEMP_C_ALIASES).intersection(df.columns))
    item["temperature_columns"] = temp_cols
    item["columns"] = list(map(str, df.columns))
    item["ok"] = not item["issues"]
    return item


def check_iv_family(data_dir: Path, family: str, patterns: tuple[str, ...]) -> dict:
    paths: list[Path] = []
    for pattern in patterns:
        paths.extend(data_dir.glob(pattern))
    paths = sorted(set(paths))
    files = [check_iv_file(path) for path in paths]
    issues = [] if paths else [f"no {family} workbook matched: {patterns}"]
    if any(not item["ok"] for item in files):
        issues.append("one or more workbooks failed validation")
    return {
        "family": family,
        "matched_files": [path.name for path in paths],
        "files": files,
        "ok": bool(paths) and all(item["ok"] for item in files),
        "issues": issues,
    }


def check_cv_file(path: Path) -> dict:
    item: dict = {"file": path.name, "exists": path.exists(), "ok": False, "issues": [], "sheets": {}}
    if not path.exists():
        item["issues"].append("missing file; CV fitting will be skipped")
        return item
    try:
        xl = pd.ExcelFile(path)
    except Exception as exc:
        item["issues"].append(f"cannot open workbook: {exc}")
        return item
    lower_to_real = {s.lower(): s for s in xl.sheet_names}
    for sheet in CV_SHEETS:
        if sheet not in lower_to_real:
            item["sheets"][sheet] = {"exists": False, "issues": ["missing sheet"]}
            continue
        raw = pd.read_excel(path, sheet_name=lower_to_real[sheet], header=None, nrows=8)
        usable_pairs = 0
        if raw.shape[0] >= 3 and raw.shape[1] >= 2:
            for col in range(0, raw.shape[1] - 1, 2):
                x = pd.to_numeric(raw.iloc[2:, col], errors="coerce")
                c = pd.to_numeric(raw.iloc[2:, col + 1], errors="coerce")
                if (x.notna() & c.notna()).any():
                    usable_pairs += 1
        issues = [] if usable_pairs else ["no usable x/C paired columns found"]
        item["sheets"][sheet] = {"exists": True, "usable_curve_pairs_preview": usable_pairs, "issues": issues}
    item["ok"] = not item["issues"] and any(v.get("usable_curve_pairs_preview", 0) for v in item["sheets"].values())
    return item


def check_it_dir(path: Path | None) -> dict:
    if path is None:
        return {"provided": False, "ok": True, "issues": ["not checked"]}
    item: dict = {"provided": True, "path": str(path), "exists": path.exists(), "ok": False, "files": [], "issues": []}
    if not path.exists():
        item["issues"].append("I-t directory does not exist")
        return item
    files = sorted(list(path.glob("I-t*.xlsx")) + list(path.glob("I-t*.xls")))
    if not files:
        item["issues"].append("no Excel files found")
        return item
    required_norm = {"time", "vds", "vbg", "vtg", "temp", "id"}
    for f in files:
        info = {"file": f.name, "sheets": {}}
        try:
            xl = pd.ExcelFile(f)
            for sheet in xl.sheet_names:
                df = pd.read_excel(f, sheet_name=sheet, nrows=10)
                cols = [str(c).lower() for c in df.columns]
                flags = {
                    "time": any("t(" in c or "time" in c or c == "t" for c in cols),
                    "vds": any("vds" in c or "vd" in c for c in cols),
                    "vbg": any("vbg" in c or "bg" in c for c in cols),
                    "vtg": any("vtg" in c or "tg" in c for c in cols),
                    "temp": any(
                        "temp" in c
                        or "t/t" in c
                        or c in {"t", "t(k)", "t_k", "temperature", "temperature_k"}
                        for c in cols
                    ),
                    "id": any("ids" in c or "id" in c or "current" in c for c in cols),
                }
                missing = sorted(k for k in required_norm if not flags[k])
                info["sheets"][sheet] = {"columns": list(map(str, df.columns)), "missing_normalized_fields": missing}
        except Exception as exc:
            info["error"] = str(exc)
        item["files"].append(info)
    item["ok"] = not item["issues"] and all(
        not sheet_info.get("missing_normalized_fields")
        for f in item["files"]
        for sheet_info in f.get("sheets", {}).values()
    )
    return item


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate DG-TFT compact-model training dataset layout.")
    ap.add_argument("--data-dir", type=Path, required=True, help="Folder containing IDVBG/IDVTG/IDVD/CV training workbooks.")
    ap.add_argument("--it-dir", type=Path, default=None, help="Optional folder containing raw I-t hysteresis workbooks.")
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args()

    report = {
        "data_dir": str(args.data_dir),
        "iv": [
            check_iv_family(args.data_dir, family, patterns)
            for family, patterns in IV_FAMILIES.items()
        ],
        "cv": check_cv_file(args.data_dir / "CV_train.xlsx"),
        "it": check_it_dir(args.it_dir),
    }
    report["ok_static"] = all(x["ok"] for x in report["iv"]) and bool(report["cv"]["ok"])
    report["ok_with_it"] = bool(report["ok_static"]) and bool(report["it"]["ok"])
    text = json.dumps(report, indent=2, ensure_ascii=False)
    print(text)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
