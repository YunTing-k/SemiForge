import fs from "node:fs/promises";
import path from "node:path";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const here = path.dirname(new URL(import.meta.url).pathname.replace(/^\/([A-Za-z]:)/, "$1"));
const releaseRoot = path.resolve(here, "..", "..");
const csvDir = path.join(releaseRoot, "results", "plot_data", "csv");
const outputPath = path.join(releaseRoot, "results", "plot_data", "DG_TFT_plot_data.xlsx");

const files = [
  ["Static_IV", "static_iv_measured_physics_pkan_residual.csv", "Static ID-VBG/ID-VTG/ID-VD measured and three model stages"],
  ["Static_CV", "static_cv_measured_physics_pkan_residual.csv", "Cgg/Cbg/Ctg/Vds measured and three model stages"],
  ["IV_Extrap", "static_iv_extrapolation.csv", "Static IV extrapolation grid"],
  ["CV_Extrap", "static_cv_extrapolation.csv", "Static CV extrapolation grid"],
  ["CV_Vhalf", "cv_vhalf_held_gate_trends.csv", "Held-gate coupling and Vhalf trend"],
  ["Temp_IV", "temperature_iv_predictions.csv", "Temperature-dependent transfer curves"],
  ["Temp_CV", "temperature_cv_predictions.csv", "Temperature-dependent capacitance"],
  ["Temp_Params", "temperature_physical_parameters.csv", "Physical parameters versus temperature"],
  ["Hyst_Fast", "hysteresis_fast_predictions.csv", "Microsecond-millisecond two-trap teacher and symbolic SRH-KAN"],
  ["Hyst_Slow", "hysteresis_slow_predictions.csv", "Seconds-scale two-trap teacher and symbolic SRH-KAN"],
  ["Hyst_Metrics", "hysteresis_slow_metrics.csv", "Per-waveform slow-hysteresis metrics"],
  ["SRH_HSPICE", "python_hspice_srh_states.csv", "Python versus HSPICE four-state SRH response"],
  ["Current_HSPICE", "python_hspice_complete_current.csv", "Complete Python versus HSPICE terminal current"],
  ["Stage_Loss", "stage_loss_summary.csv", "Physics, parameter-KAN and residual-KAN progression"],
  ["Hyst_Summary", "hysteresis_validation_summary.csv", "Fast/slow dynamic validation summary"],
  ["Temp_Diagnostics", "temperature_trend_diagnostics.csv", "Temperature physical-trend acceptance checks"],
];

function parseCSV(text) {
  const rows = [];
  let row = [];
  let field = "";
  let quoted = false;
  for (let i = 0; i < text.length; i += 1) {
    const ch = text[i];
    if (quoted) {
      if (ch === '"' && text[i + 1] === '"') {
        field += '"';
        i += 1;
      } else if (ch === '"') {
        quoted = false;
      } else {
        field += ch;
      }
    } else if (ch === '"') {
      quoted = true;
    } else if (ch === ",") {
      row.push(field);
      field = "";
    } else if (ch === "\n") {
      row.push(field.replace(/\r$/, ""));
      rows.push(row);
      row = [];
      field = "";
    } else {
      field += ch;
    }
  }
  if (field.length || row.length) {
    row.push(field.replace(/\r$/, ""));
    rows.push(row);
  }
  const width = Math.max(...rows.map((r) => r.length));
  return rows.map((r, rowIndex) => {
    const padded = [...r, ...Array(width - r.length).fill("")];
    if (rowIndex === 0) return padded;
    return padded.map((value) => {
      const trimmed = value.trim();
      if (trimmed === "") return null;
      const numeric = Number(trimmed);
      return Number.isFinite(numeric) ? numeric : value;
    });
  });
}

const workbook = Workbook.create();
const readme = workbook.worksheets.add("README");
readme.showGridLines = false;
readme.getRange("A1").values = [["DG-TFT plot-data workbook"]];
readme.getRange("A1:C1").format = {
  font: { bold: true, color: "#17365D", size: 16 },
  rowHeight: 28,
  verticalAlignment: "center",
};
readme.getRange("A3:C3").values = [["Sheet", "Source CSV", "Purpose"]];
readme.getRange("A3:C3").format = {
  fill: "#D9EAF7",
  font: { bold: true, color: "#17365D" },
  borders: { preset: "bottom", style: "medium", color: "#7F8C8D" },
};
readme.getRange(`A4:C${3 + files.length}`).values = files.map(([sheet, file, purpose]) => [
  sheet,
  file,
  purpose,
]);
readme.getRange(`A4:C${3 + files.length}`).format = {
  borders: { preset: "insideHorizontal", style: "thin", color: "#D9D9D9" },
  verticalAlignment: "top",
};
readme.getRange("A21:F21").merge();
readme.getRange("A21").values = [[
  "Units: voltage V; current A; capacitance F; time s; temperature K. Raw measured values are not overwritten."
]];
readme.getRange("A21:F21").format = { fill: "#FFF2CC", wrapText: true };
readme.getRange("A:A").format.columnWidth = 20;
readme.getRange("B:B").format.columnWidth = 52;
readme.getRange("C:C").format.columnWidth = 72;
readme.freezePanes.freezeRows(3);

for (const [sheetName, fileName] of files) {
  const csvText = await fs.readFile(path.join(csvDir, fileName), "utf8");
  const rows = parseCSV(csvText.replace(/^\uFEFF/, ""));
  const sheet = workbook.worksheets.add(sheetName);
  const width = rows[0].length;
  const chunkSize = 4000;
  for (let start = 0; start < rows.length; start += chunkSize) {
    const chunk = rows.slice(start, start + chunkSize);
    sheet.getRangeByIndexes(start, 0, chunk.length, width).values = chunk;
  }
  sheet.showGridLines = false;
  const used = sheet.getUsedRange(true);
  if (used) {
    const header = used.getRow(0);
    header.format = {
      fill: "#D9EAF7",
      font: { bold: true, color: "#17365D" },
      borders: { preset: "bottom", style: "medium", color: "#7F8C8D" },
      wrapText: true,
    };
    used.format.font = { name: "Arial", size: 9 };
    used.format.autofitColumns();
    used.format.autofitRows();
  }
  sheet.freezePanes.freezeRows(1);
}

await fs.mkdir(path.dirname(outputPath), { recursive: true });
const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(outputPath);

const preview = await workbook.render({
  sheetName: "README",
  range: "A1:C21",
  scale: 1.5,
  format: "png",
});
await fs.writeFile(
  path.join(releaseRoot, "results", "plot_data", "plot_data_workbook_preview.png"),
  new Uint8Array(await preview.arrayBuffer()),
);

const inspection = await workbook.inspect({
  kind: "table",
  range: "README!A1:C21",
  include: "values,formulas",
  tableMaxRows: 25,
  tableMaxCols: 4,
});
console.log(inspection.ndjson);
console.log(outputPath);
