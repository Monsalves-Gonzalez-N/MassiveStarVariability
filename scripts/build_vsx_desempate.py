#!/usr/bin/env python
"""Desempate de un puñado de estrellas VSX: `per_vsx` y la alternativa anotada.

Para las estrellas con nota "half"/"mitad" o "doble" se agrega un tercer panel
con el fold en `per_vsx`/2 o `per_vsx`*2. Las que no tienen ninguna nota (los
conflictos `ok`+`maybe` sin explicación) muestran solo `per_vsx` dos veces
para dejar espacio simétrico. Mismos checkboxes que `build_vsx_pdf_review.py`.

    PYTHONPATH=src python scripts/build_vsx_desempate.py --tics 91701376,264729815,...
"""
import argparse
import re
import sys
from pathlib import Path

import fitz
import matplotlib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _preview import open_in_preview  # noqa: E402
from build_vsx_pdf_review import (  # noqa: E402
    CHOICES, RAW_DIR, REVIEW_CSV, add_form_widgets, load_one_sector_per_star,
    rasterize_points, read_existing_answers,
)

from msv.config import RESULTS_DIR
from msv.viz import plot_phase_fold

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

ROW_HEIGHT = 3.2
RASTER_DPI = 130
WIDTH_RATIOS = [2.0, 2.0, 2.0, 1.3]
FIGURE_WIDTH = 15.0
BOX_SIDE = 0.20
BOX_LEFT = 0.15
FIRST_BOX_TOP = 0.28
BOX_SPACING = 0.28
NOTES_TOP_GAP = 0.28
NOTES_BOTTOM = 0.18


def alternate_period(per_vsx, note):
    note = str(note).lower() if isinstance(note, str) else ""
    if "mitad" in note or re.search(r"\bhalf\b", note):
        return per_vsx / 2, "P_VSX / 2"
    if "doble" in note:
        return per_vsx * 2, "P_VSX x 2"
    return None, None


def annotation_rectangles(position, figure_size):
    width, height = figure_size
    left = position.x0 * width + BOX_LEFT
    top = position.y1 * height - FIRST_BOX_TOP

    def to_fraction(x0, y0, x1, y1):
        return (x0 / width, y0 / height, x1 / width, y1 / height)

    boxes = []
    for index in range(len(CHOICES)):
        box_top = top - index * BOX_SPACING
        boxes.append(to_fraction(left, box_top - BOX_SIDE, left + BOX_SIDE, box_top))
    notes_top = top - len(CHOICES) * BOX_SPACING - NOTES_TOP_GAP
    notes = to_fraction(left, position.y0 * height + NOTES_BOTTOM,
                        position.x1 * width - BOX_LEFT, notes_top)
    return boxes, notes


def draw_annotation_labels(figure, boxes, notes):
    for (x0, y0, x1, y1), label in zip(boxes, CHOICES):
        figure.text(x1 + 0.006, 0.5 * (y0 + y1), label, fontsize=9,
                    va="center", ha="left")
    figure.text(notes[0], notes[3] + 0.004, "notas", fontsize=8,
                va="bottom", ha="left", color="0.35")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tics", required=True)
    parser.add_argument("--fits-dir", default=str(RAW_DIR / "vsx_review"))
    parser.add_argument("--review", default=str(REVIEW_CSV))
    parser.add_argument("--out", default=str(RESULTS_DIR / "figures" / "vsx_review_desempate.pdf"))
    args = parser.parse_args()

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    previous = read_existing_answers(output)

    wanted = [int(t) for t in args.tics.split(",")]
    tab = pd.read_csv(args.review).set_index("TIC")
    curves = load_one_sector_per_star(args.fits_dir)

    entries = []
    with PdfPages(output) as pdf:
        for tic in wanted:
            if tic not in curves or tic not in tab.index:
                print(f"salteada TIC {tic}: sin curva o sin fila en {args.review}")
                continue
            row = tab.loc[tic]
            sector, frame = curves[tic]
            time = frame["Time"].to_numpy()
            flux = frame["flux"].to_numpy()
            good = np.isfinite(time) & np.isfinite(flux)
            time, flux = time[good], flux[good]
            flux = flux / np.median(flux)

            per_vsx = row.per_vsx
            note = row.get("vis_notes")
            note = note if isinstance(note, str) else ""
            alt_period, alt_label = alternate_period(per_vsx, note)

            figure_size = (FIGURE_WIDTH, ROW_HEIGHT)
            figure, axes = plt.subplots(1, 4, figsize=figure_size,
                                        gridspec_kw={"width_ratios": WIDTH_RATIOS})
            axes[0].plot(time, flux, ".k", ms=1.5, alpha=0.6, rasterized=True)
            axes[0].set_xlabel("BTJD")
            axes[0].set_ylabel("Flux")
            axes[0].grid(alpha=0.3)
            axes[0].set_title(
                f"TIC {tic}   VSX {row.Type}   sector {int(sector)}"
                f"   nota: {note or '-'}", fontsize=10, loc="left")

            plot_phase_fold(time, flux, per_vsx, ax=axes[1], phase_bins=40,
                            title=f"P_VSX = {per_vsx:.5f} d")
            rasterize_points(axes[1])

            if alt_period is not None:
                plot_phase_fold(time, flux, alt_period, ax=axes[2], phase_bins=40,
                                title=f"{alt_label} = {alt_period:.5f} d")
                rasterize_points(axes[2])
            else:
                axes[2].axis("off")
                axes[2].set_title("sin alternativa anotada", fontsize=10)

            axes[3].set_axis_off()
            figure.tight_layout()
            boxes, notes = annotation_rectangles(axes[3].get_position(), figure_size)
            draw_annotation_labels(figure, boxes, notes)
            entries.append({"page": len(entries), "name": str(tic),
                            "boxes": boxes, "notes": notes,
                            "figure_size": figure_size})
            pdf.savefig(figure, dpi=RASTER_DPI)
            plt.close(figure)

    add_form_widgets(output, entries, previous)
    print(f"escrito {output}  ({len(entries)} estrellas)")
    open_in_preview(output)


if __name__ == "__main__":
    main()
