"""
Validation report -> PDF.
=========================

Renders a report written by :mod:`validation.report` (``REPORT.md`` or the
``--passing-only`` excerpt ``REPORT_passing.md``) to PDF with ReportLab, so the
markdown stays the single source and the PDF is never hand-assembled.

The markdown produced by ``validation.report`` is a small, known subset --
``#``/``##``/``###`` headings, pipe tables, ``![alt](plots/x.png)`` images,
``**bold**``/``*italic*``/``_italic_`` spans, ``1.`` numbered lines and plain
paragraphs -- so this parses that subset directly rather than pulling in a full
markdown engine.

Usage:
    python -m validation.report_pdf                       # REPORT.md -> REPORT.pdf
    python -m validation.report_pdf --passing-only        # the excerpt
    python -m validation.report_pdf --input path/to.md --output path/to.pdf
"""
from __future__ import annotations

import argparse
import html
import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from reportlab.platypus import (Image, PageBreak, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)

REPORT_DIR = Path(__file__).resolve().parent / "report"

PAGE_W, PAGE_H = A4
MARGIN = 18 * mm
CONTENT_W = PAGE_W - 2 * MARGIN

INK = colors.HexColor("#1b1f24")
MUTED = colors.HexColor("#57606a")
ACCENT = colors.HexColor("#0b5cad")
PASS_C = colors.HexColor("#1a7f37")
FAIL_C = colors.HexColor("#b42318")
RULE = colors.HexColor("#d0d7de")
HEAD_BG = colors.HexColor("#eef2f6")
ZEBRA = colors.HexColor("#f7f9fb")


# -- inline markdown -> reportlab markup --------------------------------------

def _inline(text: str) -> str:
    """Convert the inline markdown subset to ReportLab's mini-HTML."""
    text = text.replace("&nbsp;", " ")
    text = html.escape(text, quote=False)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"(?<!\w)\*(?!\s)(.+?)(?<!\s)\*(?!\w)", r"<i>\1</i>", text)
    text = re.sub(r"(?<!\w)_(?!\s)(.+?)(?<!\s)_(?!\w)", r"<i>\1</i>", text)
    text = re.sub(r"`(.+?)`", r"<font face='Courier'>\1</font>", text)
    return text


def _register_fonts() -> tuple:
    """(regular, bold) font names, preferring a Unicode-capable TTF.

    The report is full of glyphs the base-14 fonts do not carry -- the per-row
    status ticks, the one-sided tolerance arrows, math signs in source cells --
    which Helvetica silently drops. DejaVu ships with matplotlib, already a hard
    dependency here, so use it and fall back only if that lookup fails.
    """
    try:
        from matplotlib import get_data_path
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont

        ttf = Path(get_data_path()) / "fonts" / "ttf"
        reg, bold = ttf / "DejaVuSans.ttf", ttf / "DejaVuSans-Bold.ttf"
        if not (reg.exists() and bold.exists()):
            return "Helvetica", "Helvetica-Bold"
        pdfmetrics.registerFont(TTFont("DejaVu", str(reg)))
        pdfmetrics.registerFont(TTFont("DejaVu-Bold", str(bold)))
        pdfmetrics.registerFontFamily("DejaVu", normal="DejaVu", bold="DejaVu-Bold",
                                      italic="DejaVu", boldItalic="DejaVu-Bold")
        return "DejaVu", "DejaVu-Bold"
    except Exception:
        return "Helvetica", "Helvetica-Bold"


def _styles() -> dict:
    font, font_b = _register_fonts()
    base = getSampleStyleSheet()["BodyText"]
    mk = lambda **kw: ParagraphStyle(parent=base, **kw)
    return {
        "h1": mk(name="h1", fontName=font_b, fontSize=19, leading=24,
                 textColor=INK, spaceBefore=0, spaceAfter=6),
        "h2": mk(name="h2", fontName=font_b, fontSize=13, leading=17,
                 textColor=ACCENT, spaceBefore=16, spaceAfter=6),
        "h3": mk(name="h3", fontName=font_b, fontSize=10.5, leading=14,
                 textColor=INK, spaceBefore=12, spaceAfter=4),
        "body": mk(name="body", fontName=font, fontSize=8.6, leading=12.2,
                   textColor=INK, alignment=TA_LEFT, spaceAfter=5),
        "muted": mk(name="muted", fontName=font, fontSize=8.2, leading=11.2,
                    textColor=MUTED, spaceAfter=5),
        "cell": mk(name="cell", fontName=font, fontSize=7.2, leading=9.4,
                   textColor=INK),
        "cellhead": mk(name="cellhead", fontName=font_b, fontSize=7.2,
                       leading=9.4, textColor=INK),
        "_font": font,
    }


# -- block parsing -------------------------------------------------------------

_IMG_RE = re.compile(r"^!\[(?P<alt>[^\]]*)\]\((?P<src>[^)]+)\)\s*$")
_ROW_RE = re.compile(r"^\|(.*)\|\s*$")
_SEP_RE = re.compile(r"^\|[\s:|-]+\|\s*$")


def _split_row(line: str) -> list:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def _status_colour(cell: str):
    """Colour for a status cell, so pass/fail reads at a glance."""
    plain = cell.strip()
    if plain in ("PASS", "✓"):
        return PASS_C
    if plain in ("FAIL", "✗"):
        return FAIL_C
    return None


def _table_flowable(rows: list, st: dict) -> Table:
    header, body = rows[0], rows[1:]
    ncols = len(header)
    data = [[Paragraph(_inline(c), st["cellhead"]) for c in header]]
    for r in body:
        r = (r + [""] * ncols)[:ncols]
        data.append([Paragraph(_inline(c), st["cell"]) for c in r])

    # Weight columns by the longest cell so wide source/reference columns get
    # the room instead of every column sharing the width equally.
    widths = []
    for i in range(ncols):
        longest = max(len(str(r[i])) if i < len(r) else 0 for r in rows)
        widths.append(max(longest, 4))
    total = sum(widths)
    col_w = [CONTENT_W * w / total for w in widths]

    # A narrow column still has to hold its own header and its widest short
    # word: purely proportional widths wrap "Status"/"PASS" into "St at us".
    min_w = 15 * mm
    deficit = sum(max(0.0, min_w - w) for w in col_w)
    if deficit:
        wide = [i for i, w in enumerate(col_w) if w > min_w]
        slack = sum(col_w[i] - min_w for i in wide)
        for i in wide:
            col_w[i] -= deficit * (col_w[i] - min_w) / slack
        col_w = [max(w, min_w) for w in col_w]

    style = [
        ("BACKGROUND", (0, 0), (-1, 0), HEAD_BG),
        ("LINEBELOW", (0, 0), (-1, 0), 0.6, RULE),
        ("GRID", (0, 0), (-1, -1), 0.25, RULE),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]
    for r_i, r in enumerate(body, start=1):
        if r_i % 2 == 0:
            style.append(("BACKGROUND", (0, r_i), (-1, r_i), ZEBRA))
        for c_i, cell in enumerate(r[:ncols]):
            col = _status_colour(cell)
            if col is not None:
                data[r_i][c_i] = Paragraph(
                    f"<font color='#{col.hexval()[2:]}'><b>{_inline(cell)}</b></font>",
                    st["cell"])
    t = Table(data, colWidths=col_w, repeatRows=1, hAlign="LEFT")
    t.setStyle(TableStyle(style))
    return t


def _image_flowable(src: Path, max_w: float = CONTENT_W * 0.72):
    if not src.exists():
        return None
    iw, ih = ImageReader(str(src)).getSize()
    w = min(max_w, iw)
    return Image(str(src), width=w, height=ih * w / iw, hAlign="LEFT")


def build_story(md: str, base_dir: Path, st: dict) -> list:
    story, lines, i = [], md.split("\n"), 0
    while i < len(lines):
        line = lines[i].rstrip()
        if not line.strip():
            i += 1
            continue

        if _ROW_RE.match(line):
            rows = []
            while i < len(lines) and _ROW_RE.match(lines[i].rstrip()):
                if not _SEP_RE.match(lines[i].rstrip()):
                    rows.append(_split_row(lines[i]))
                i += 1
            if rows:
                story += [_table_flowable(rows, st), Spacer(1, 7)]
            continue

        m = _IMG_RE.match(line)
        if m:
            img = _image_flowable(base_dir / m.group("src"))
            if img is not None:
                story += [img, Spacer(1, 8)]
            i += 1
            continue

        if line.startswith("### "):
            story.append(Paragraph(_inline(line[4:]), st["h3"]))
        elif line.startswith("## "):
            story.append(Paragraph(_inline(line[3:]), st["h2"]))
        elif line.startswith("# "):
            story.append(Paragraph(_inline(line[2:]), st["h1"]))
        else:
            style = st["muted"] if line.startswith("_") else st["body"]
            story.append(Paragraph(_inline(line), style))
        i += 1
    return story


# -- page furniture ------------------------------------------------------------

def _page_furniture(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(MUTED)
    canvas.drawString(MARGIN, 11 * mm, "K2 AeroSim - physics validation")
    canvas.drawRightString(PAGE_W - MARGIN, 11 * mm, f"{doc.page}")
    canvas.setStrokeColor(RULE)
    canvas.setLineWidth(0.4)
    canvas.line(MARGIN, 14 * mm, PAGE_W - MARGIN, 14 * mm)
    canvas.restoreState()


def render_pdf(md_path: Path, pdf_path: Path) -> Path:
    md = Path(md_path).read_text(encoding="utf-8")
    st = _styles()
    doc = SimpleDocTemplate(
        str(pdf_path), pagesize=A4,
        leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=MARGIN, bottomMargin=20 * mm,
        title="K2 Physics Validation Report", author="K2 AeroSim",
    )
    doc.build(build_story(md, Path(md_path).parent, st),
              onFirstPage=_page_furniture, onLaterPages=_page_furniture)
    return pdf_path


def main():
    ap = argparse.ArgumentParser(description="Render a validation report to PDF.")
    ap.add_argument("--passing-only", action="store_true",
                    help="render REPORT_passing.md instead of REPORT.md")
    ap.add_argument("--input", type=Path, help="markdown to render")
    ap.add_argument("--output", type=Path, help="PDF to write")
    args = ap.parse_args()

    stem = "REPORT_passing" if args.passing_only else "REPORT"
    src = args.input or REPORT_DIR / f"{stem}.md"
    if not src.exists():
        raise SystemExit(f"No report at {src}; run python -m validation.report first.")
    out = args.output or src.with_suffix(".pdf")
    print(f"Wrote {render_pdf(src, out)}")


if __name__ == "__main__":
    main()
