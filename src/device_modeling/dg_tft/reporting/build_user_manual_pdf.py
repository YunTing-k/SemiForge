from __future__ import annotations

from pathlib import Path
import re

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    KeepTogether,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "manual" / "DG_TFT_v95_用户操作与技术手册.md"
OUTPUT = ROOT / "manual" / "DG_TFT_v95_用户操作与技术手册.pdf"


def esc(text: str) -> str:
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = re.sub(r"`([^`]+)`", r"<font name='Courier'>\1</font>", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", text)
    return text


def footer(canvas, doc) -> None:
    canvas.saveState()
    canvas.setFont("STSong-Light", 8)
    canvas.setFillColor(colors.HexColor("#64748B"))
    canvas.drawString(18 * mm, 10 * mm, "DG-TFT v95 Fresh-Training Manual")
    canvas.drawRightString(A4[0] - 18 * mm, 10 * mm, f"{doc.page}")
    canvas.restoreState()


def build() -> None:
    pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
    styles = getSampleStyleSheet()
    body = ParagraphStyle(
        "BodyCN",
        parent=styles["BodyText"],
        fontName="STSong-Light",
        fontSize=9.2,
        leading=13.6,
        textColor=colors.HexColor("#1F2937"),
        spaceAfter=5,
    )
    title = ParagraphStyle(
        "TitleCN",
        parent=body,
        fontSize=21,
        leading=27,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#17365D"),
        spaceAfter=12,
    )
    h1 = ParagraphStyle(
        "H1CN",
        parent=body,
        fontSize=14,
        leading=18,
        textColor=colors.HexColor("#2E74B5"),
        spaceBefore=10,
        spaceAfter=6,
        keepWithNext=True,
    )
    h2 = ParagraphStyle(
        "H2CN",
        parent=body,
        fontSize=11.5,
        leading=15,
        textColor=colors.HexColor("#1F4D78"),
        spaceBefore=7,
        spaceAfter=4,
        keepWithNext=True,
    )
    bullet = ParagraphStyle(
        "BulletCN",
        parent=body,
        leftIndent=12,
        firstLineIndent=-7,
        bulletIndent=4,
    )
    code = ParagraphStyle(
        "Code",
        parent=body,
        fontName="Courier",
        fontSize=7.6,
        leading=10.2,
        leftIndent=7,
        rightIndent=7,
        backColor=colors.HexColor("#F2F4F7"),
        borderPadding=5,
        spaceBefore=2,
        spaceAfter=6,
    )

    doc = BaseDocTemplate(
        str(OUTPUT),
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=17 * mm,
        title="DG-TFT Physics and Symbolic KAN Modeling Manual",
        author="DG-TFT Modeling Project",
    )
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="body")
    doc.addPageTemplates([PageTemplate(id="main", frames=[frame], onPage=footer)])

    lines = SOURCE.read_text(encoding="utf-8").splitlines()
    story = []
    in_code = False
    code_lines: list[str] = []
    table_rows: list[list[str]] = []

    def flush_code() -> None:
        nonlocal code_lines
        if code_lines:
            story.append(Paragraph("<br/>".join(esc(x) for x in code_lines), code))
            code_lines = []

    def flush_table() -> None:
        nonlocal table_rows
        if not table_rows:
            return
        rows = [
            row
            for row in table_rows
            if not all(set(cell) <= {"-", ":", " "} for cell in row)
        ]
        table_rows = []
        if not rows:
            return
        ncols = max(len(row) for row in rows)
        normalized = [row + [""] * (ncols - len(row)) for row in rows]
        cell_style = ParagraphStyle("CellCN", parent=body, fontSize=7.8, leading=10.5, spaceAfter=0)
        data = [[Paragraph(esc(cell), cell_style) for cell in row] for row in normalized]
        first = 0.24 * doc.width if ncols == 2 else doc.width / ncols
        widths = [first, doc.width - first] if ncols == 2 else [doc.width / ncols] * ncols
        table = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E8EEF5")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#17365D")),
                    ("FONTNAME", (0, 0), (-1, -1), "STSong-Light"),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#CBD5E1")),
                    ("LEFTPADDING", (0, 0), (-1, -1), 5),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ]
            )
        )
        story.append(KeepTogether([table, Spacer(1, 5)]))

    for raw in lines:
        line = raw.rstrip()
        if line.startswith("```"):
            flush_table()
            if in_code:
                flush_code()
            in_code = not in_code
            continue
        if in_code:
            code_lines.append(line)
            continue
        if line.startswith("|") and line.endswith("|"):
            table_rows.append([x.strip() for x in line.strip("|").split("|")])
            continue
        flush_table()
        if not line or line == "---":
            story.append(Spacer(1, 2))
        elif line.startswith("# "):
            story.append(Paragraph(esc(line[2:]), title))
        elif line.startswith("## "):
            story.append(Paragraph(esc(line[3:]), h1))
        elif line.startswith("### "):
            story.append(Paragraph(esc(line[4:]), h2))
        elif re.match(r"^\d+\.\s", line):
            text = re.sub(r"^\d+\.\s+", "", line)
            story.append(Paragraph(esc(text), bullet, bulletText="1."))
        elif line.startswith("- "):
            story.append(Paragraph(esc(line[2:]), bullet, bulletText="-"))
        else:
            story.append(Paragraph(esc(line), body))

    flush_table()
    flush_code()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    doc.build(story)
    print(OUTPUT)


if __name__ == "__main__":
    build()
