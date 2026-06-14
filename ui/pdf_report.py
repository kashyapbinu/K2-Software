"""
Lightweight multi-page PDF report builder.

Captures the matplotlib figures a workspace already renders (plus optional
image arrays, e.g. a pyvista contour screenshot) into a single PDF, with a
cover page carrying a title and a key/value summary table. Used by the CFD,
Monte-Carlo and Optimization workspaces so users can save an analysis as a PDF.
"""

from __future__ import annotations

import datetime
import logging

import matplotlib
matplotlib.use("Agg", force=False)
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

logger = logging.getLogger("K2.PDFReport")

_A4_PORTRAIT = (8.27, 11.69)
_A4_LANDSCAPE = (11.69, 8.27)


def save_report(path, title, subtitle="", kv_rows=None, figures=None, images=None):
    """Write a PDF report.

    Parameters
    ----------
    path : str
        Output ``.pdf`` path.
    title, subtitle : str
        Cover-page heading.
    kv_rows : list[tuple[str, str]]
        Summary rows shown as a table on the cover.
    figures : list[matplotlib.figure.Figure]
        Existing workspace figures — one PDF page each.
    images : list[tuple[str, ndarray]]
        (caption, RGB image) pairs — e.g. a 3D contour screenshot.
    """
    try:
        with PdfPages(path) as pdf:
            # ── Cover page ──
            fig = plt.figure(figsize=_A4_PORTRAIT)
            fig.patch.set_facecolor("white")
            fig.text(0.5, 0.93, title, ha="center", fontsize=22, fontweight="bold")
            if subtitle:
                fig.text(0.5, 0.895, subtitle, ha="center", fontsize=12, color="#555555")
            stamp = "K2 AeroSim  ·  " + datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
            fig.text(0.5, 0.87, stamp, ha="center", fontsize=9, color="#999999")
            if kv_rows:
                ax = fig.add_axes([0.1, 0.12, 0.8, 0.7])
                ax.axis("off")
                tbl = ax.table(
                    cellText=[[str(k), str(v)] for k, v in kv_rows],
                    colLabels=["Parameter", "Value"],
                    cellLoc="left", colLoc="left", loc="upper center",
                    colWidths=[0.6, 0.4])
                tbl.auto_set_font_size(False)
                tbl.set_fontsize(9)
                tbl.scale(1, 1.6)
                for (r, c), cell in tbl.get_celld().items():
                    cell.set_edgecolor("#dddddd")
                    if r == 0:
                        cell.set_facecolor("#222a35")
                        cell.set_text_props(color="white", fontweight="bold")
            pdf.savefig(fig)
            plt.close(fig)

            # ── Figure pages ──
            for f in (figures or []):
                if f is None:
                    continue
                try:
                    pdf.savefig(f, facecolor=f.get_facecolor())
                except Exception as e:
                    logger.debug("skip figure in report: %s", e)

            # ── Image pages (screenshots) ──
            for caption, img in (images or []):
                if img is None:
                    continue
                try:
                    fig = plt.figure(figsize=_A4_LANDSCAPE)
                    fig.patch.set_facecolor("white")
                    ax = fig.add_axes([0.03, 0.04, 0.94, 0.88])
                    ax.axis("off")
                    ax.imshow(img)
                    if caption:
                        ax.set_title(caption, fontsize=12)
                    pdf.savefig(fig)
                    plt.close(fig)
                except Exception as e:
                    logger.debug("skip image in report: %s", e)
        return True
    except Exception as e:
        logger.error("PDF report failed: %s", e)
        return False
