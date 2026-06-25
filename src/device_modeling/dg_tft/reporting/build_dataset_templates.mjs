import fs from "node:fs/promises";
import path from "node:path";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const here = path.dirname(new URL(import.meta.url).pathname.replace(/^\/([A-Za-z]:)/, "$1"));
const root = path.resolve(here, "..", "..");
const outDir = path.join(root, "data_templates");

function styleDataSheet(sheet, range) {
  sheet.showGridLines = false;
  sheet.getRange(range).getRow(0).format = {
    fill: "#D9EAF7",
    font: { bold: true, color: "#17365D" },
    borders: { preset: "bottom", style: "medium", color: "#7F8C8D" },
  };
  sheet.getRange(range).format.font = { name: "Arial", size: 10 };
  sheet.getRange(range).format.autofitColumns();
  sheet.freezePanes.freezeRows(1);
}

async function saveWorkbook(workbook, name, previewSheet, previewRange) {
  const file = await SpreadsheetFile.exportXlsx(workbook);
  await file.save(path.join(outDir, name));
  const preview = await workbook.render({
    sheetName: previewSheet,
    range: previewRange,
    scale: 1.5,
    format: "png",
  });
  await fs.writeFile(
    path.join(outDir, name.replace(".xlsx", "_preview.png")),
    new Uint8Array(await preview.arrayBuffer()),
  );
}

function ivWorkbook(family) {
  const wb = Workbook.create();
  const sheet = wb.worksheets.add("long");
  const defaults = {
    IDVBG: [[0.1, -5.0, 0.0, 1e-12, 300], [0.1, -4.9, 0.0, 1e-12, 300]],
    IDVTG: [[0.1, 0.0, -4.0, 1e-12, 300], [0.1, 0.0, -3.9, 1e-12, 300]],
    IDVD: [[0.0, 0.0, 0.0, 0.0, 300], [0.1, 0.0, 0.0, 1e-12, 300]],
  };
  sheet.getRange("A1:E3").values = [
    ["VD_V", "VBG_V", "VTG_V", "ID_A", "temp_K"],
    ...defaults[family],
  ];
  styleDataSheet(sheet, "A1:E3");
  sheet.getRange("A:A").format.numberFormat = "0.000";
  sheet.getRange("B:C").format.numberFormat = "0.000";
  sheet.getRange("D:D").format.numberFormat = "0.000E+00";
  sheet.getRange("E:E").format.numberFormat = "0.0";
  return wb;
}

function cvWorkbook() {
  const wb = Workbook.create();
  for (const mode of ["cgg", "cbg", "ctg", "vds"]) {
    const sheet = wb.worksheets.add(mode);
    const label =
      mode === "cbg" ? "cbg_VTG=0V_VDS=0V_100kHz_T=300K" :
      mode === "ctg" ? "ctg_VBG=0V_VDS=0V_100kHz_T=300K" :
      mode === "cgg" ? "cgg_10kHz_VDS=0V_T=300K" :
      "vds_VDS=0.5V_100kHz_T=300K";
    sheet.getRange("A1:B5").values = [
      ["Optional note", null],
      [label, label],
      [-5.0, 3.0e-13],
      [-4.9, 3.0e-13],
      [-4.8, 3.0e-13],
    ];
    styleDataSheet(sheet, "A2:B5");
    sheet.getRange("A:A").format.numberFormat = "0.000";
    sheet.getRange("B:B").format.numberFormat = "0.000E+00";
  }
  return wb;
}

function itWorkbook() {
  const wb = Workbook.create();
  const sheet = wb.worksheets.add("waveform_1");
  sheet.getRange("A1:F5").values = [
    ["time_s", "VDS_V", "VBG_V", "VTG_V", "temp_K", "ID_A"],
    [0.0, 0.1, 0.0, 0.0, 300, 1e-12],
    [0.001, 0.1, 0.0, 0.0, 300, 1e-12],
    [0.002, 0.1, 2.0, 0.0, 300, 2e-10],
    [0.003, 0.1, 2.0, 0.0, 300, 1.8e-10],
  ];
  styleDataSheet(sheet, "A1:F5");
  sheet.getRange("A:E").format.numberFormat = "0.000000";
  sheet.getRange("F:F").format.numberFormat = "0.000E+00";
  return wb;
}

await fs.mkdir(outDir, { recursive: true });
for (const family of ["IDVBG", "IDVTG", "IDVD"]) {
  await saveWorkbook(ivWorkbook(family), `${family}_train_template.xlsx`, "long", "A1:E6");
}
await saveWorkbook(cvWorkbook(), "CV_train_template.xlsx", "cgg", "A1:B7");
await saveWorkbook(itWorkbook(), "IT_hysteresis_template.xlsx", "waveform_1", "A1:F7");
console.log(outDir);
