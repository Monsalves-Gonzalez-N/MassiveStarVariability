#!/usr/bin/env python
"""Revisión visual del benchmark VSX, ciega a cualquier predicción propia.

Una estrella por fila, un solo sector (el de más puntos), dos paneles: la
curva en tiempo y el fold en `per_vsx`. No se muestra ninguna clase ni período
del pipeline propio. La pregunta es una sola: ¿el fold en el período de VSX se
ve liso (una forma coherente y repetible), o no? No clasifica morfología —
eso se deja para después, sobre las que pasen este filtro (ver
docs/PLAN_clasificacion_4clases.md, sección del pivote a solo-CNN).

Checkboxes por fila (excluyentes, se leen con `read_vsx_veredicto.py`):

    ok              el fold es liso: el período de VSX describe la curva
    maybe           hay algo de estructura pero no cierra del todo
    multiperiodica  se ve batido de varias frecuencias cercanas: la
                    envolvente cambia de ciclo a ciclo, ningún período único
                    explica el fold
    bad             no es liso y no es batido: ruido, irregular, sin señal
    sin_lc          no hay curva utilizable

Más un campo de notas.

Regenerar el PDF NO pierde lo ya contestado: si el archivo de salida existe,
se leen sus respuestas antes de escribirlo y se vuelven a poner por nombre de
campo. Con `--fresh` se descartan.

    PYTHONPATH=src python scripts/build_vsx_pdf_review.py
"""
import argparse
import sys
from pathlib import Path

import fitz
import matplotlib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _preview import open_in_preview  # noqa: E402

from msv.config import CATALOGS_DIR, RAW_DIR, RESULTS_DIR
from msv.io import parse_fits_name, read_lc_fits
from msv.viz import plot_phase_fold

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

REVIEW_CSV = CATALOGS_DIR / "vsx_visual_review.csv"

ROWS_PER_PAGE = 4
ROW_HEIGHT = 3.2
RASTER_DPI = 130
WIDTH_RATIOS = [2.3, 2.3, 1.5]
FIGURE_WIDTH = 13.0

CHOICES = ["ok", "maybe", "multiperiodica", "bad", "sin_lc"]
BOX_SIDE = 0.20
BOX_LEFT = 0.15
FIRST_BOX_TOP = 0.28
BOX_SPACING = 0.28
NOTES_TOP_GAP = 0.28
NOTES_BOTTOM = 0.18


def load_one_sector_per_star(fits_dir):
    """Curva del sector con más puntos, por TIC, desde un directorio de FITS."""
    paths = sorted(Path(fits_dir).glob("*_lc.fits"))
    if not paths:
        raise SystemExit(f"sin FITS en {fits_dir}")
    best = {}
    for path in paths:
        try:
            tic, sector = parse_fits_name(path)
            frame = read_lc_fits(path)
        except Exception:
            continue
        if tic not in best or len(frame) > len(best[tic][1]):
            best[tic] = (sector, frame)
    return best


def draw_light_curve(axis, time, flux):
    axis.plot(time, flux, ".k", ms=1.5, alpha=0.6, rasterized=True)
    axis.set_xlabel("BTJD")
    axis.set_ylabel("Flux")
    axis.grid(alpha=0.3)


def rasterize_points(axis):
    for line in axis.get_lines():
        if line.get_linestyle() == "None":
            line.set_rasterized(True)


def annotation_rectangles(position, figure_size):
    width, height = figure_size
    left = position.x0 * width + BOX_LEFT
    top = position.y1 * height - FIRST_BOX_TOP

    def to_fraction(x0_inches, y0_inches, x1_inches, y1_inches):
        return (x0_inches / width, y0_inches / height,
                x1_inches / width, y1_inches / height)

    boxes = []
    for index in range(len(CHOICES)):
        box_top = top - index * BOX_SPACING
        boxes.append(to_fraction(left, box_top - BOX_SIDE,
                                 left + BOX_SIDE, box_top))
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


def read_existing_answers(pdf_path):
    if not Path(pdf_path).exists():
        return {}
    document = fitz.open(pdf_path)
    answers = {}
    for page in document:
        for widget in page.widgets():
            value = widget.field_value
            if value not in (None, False, "", "Off"):
                answers[widget.field_name] = value
    document.close()
    return answers


def add_form_widgets(pdf_path, entries, previous=None):
    previous = previous or {}
    document = fitz.open(pdf_path)
    written = set()
    for entry in entries:
        page = document[entry["page"]]
        width, height = entry["figure_size"]

        def to_rect(fraction):
            x0, y0, x1, y1 = fraction
            return fitz.Rect(x0 * width * 72, (1 - y1) * height * 72,
                             x1 * width * 72, (1 - y0) * height * 72)

        for choice, fraction in zip(CHOICES, entry["boxes"]):
            widget = fitz.Widget()
            widget.field_type = fitz.PDF_WIDGET_TYPE_CHECKBOX
            widget.field_name = f"{entry['name']}__{choice.replace(' ', '_')}"
            widget.rect = to_rect(fraction)
            widget.field_value = widget.field_name in previous
            widget.border_width = 1.0
            widget.border_color = (0.25, 0.25, 0.25)
            widget.fill_color = (1.0, 1.0, 1.0)
            page.add_widget(widget)
            written.add(widget.field_name)

        widget = fitz.Widget()
        widget.field_type = fitz.PDF_WIDGET_TYPE_TEXT
        widget.field_name = f"{entry['name']}__nota"
        widget.rect = to_rect(entry["notes"])
        widget.field_value = previous.get(widget.field_name, "")
        widget.text_fontsize = 9
        widget.field_flags = fitz.PDF_TX_FIELD_IS_MULTILINE
        widget.border_width = 0.8
        widget.border_color = (0.65, 0.65, 0.65)
        widget.fill_color = (1.0, 1.0, 1.0)
        page.add_widget(widget)
        written.add(widget.field_name)

    document.xref_set_key(document.pdf_catalog(), "AcroForm/NeedAppearances",
                          "true")
    document.saveIncr()
    document.close()
    return written


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fits-dir", default=str(RAW_DIR / "vsx_review"))
    parser.add_argument("--review", default=str(REVIEW_CSV))
    parser.add_argument("--out", default=str(RESULTS_DIR / "figures" / "vsx_review.pdf"))
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--no-open", action="store_true")
    parser.add_argument("--tics", default=None,
                        help="lista de TIC separados por coma; ignora el corte P<13d")
    args = parser.parse_args()

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    previous = {} if args.fresh else read_existing_answers(output)
    if previous:
        print(f"{len(previous)} campos ya contestados en el PDF anterior")

    tab = pd.read_csv(args.review)
    curves = load_one_sector_per_star(args.fits_dir)
    if args.tics:
        wanted = {int(t) for t in args.tics.split(",")}
        rows = [row for _, row in tab.sort_values("TIC").iterrows()
               if int(row.TIC) in curves and int(row.TIC) in wanted]
        print(f"{len(rows)} de {len(wanted)} TIC pedidos con curva en {args.fits_dir}")
    else:
        # P >= 13 d no completa ni un ciclo por sector TESS: no hay nada que
        # foldear con una sola curva.
        rows = [row for _, row in tab.sort_values("TIC").iterrows()
               if int(row.TIC) in curves and row.per_vsx < 13]
        print(f"{len(rows)} de {len(tab)} estrellas con curva en {args.fits_dir} y P_VSX < 13 d")

    entries = []
    with PdfPages(output) as pdf:
        for start in range(0, len(rows), ROWS_PER_PAGE):
            page_rows = rows[start:start + ROWS_PER_PAGE]
            figure_size = (FIGURE_WIDTH, ROW_HEIGHT * len(page_rows))
            figure, axes = plt.subplots(
                len(page_rows), 3, figsize=figure_size, squeeze=False,
                gridspec_kw={"width_ratios": WIDTH_RATIOS})
            for position, row in enumerate(page_rows):
                tic = int(row.TIC)
                sector, frame = curves[tic]
                time = frame["Time"].to_numpy()
                flux = frame["flux"].to_numpy()
                good = np.isfinite(time) & np.isfinite(flux)
                time, flux = time[good], flux[good]
                flux = flux / np.median(flux)
                draw_light_curve(axes[position][0], time, flux)
                per_vsx = row.per_vsx
                if np.isfinite(per_vsx) and per_vsx > 0:
                    plot_phase_fold(time, flux, per_vsx, ax=axes[position][1],
                                    phase_bins=40,
                                    title=f"P_VSX = {per_vsx:.4f} d")
                    rasterize_points(axes[position][1])
                else:
                    axes[position][1].axis("off")
                    axes[position][1].set_title("P_VSX: no disponible", fontsize=10)
                axes[position][2].set_axis_off()
                axes[position][0].set_title(
                    f"TIC {tic}   VSX {row.Type}   sector {int(sector)}",
                    fontsize=10, loc="left")
            figure.tight_layout()
            for position, row in enumerate(page_rows):
                boxes, notes = annotation_rectangles(
                    axes[position][2].get_position(), figure_size)
                draw_annotation_labels(figure, boxes, notes)
                entries.append({
                    "page": start // ROWS_PER_PAGE,
                    "name": str(int(row.TIC)),
                    "boxes": boxes,
                    "notes": notes,
                    "figure_size": figure_size,
                })
            pdf.savefig(figure, dpi=RASTER_DPI)
            plt.close(figure)

    written = add_form_widgets(output, entries, previous)
    print(f"escrito {output}  ({len(entries)} formularios)")
    if previous:
        lost = sorted(set(previous) - written)
        print(f"restauradas {len(previous) - len(lost)} respuestas")
        for name in lost:
            print(f"  PERDIDA (ya no existe el campo): {name} = {previous[name]!r}")
    print("llenar en Preview, guardar (cmd+S) y correr "
          "scripts/read_vsx_veredicto.py")

    if not args.no_open:
        open_in_preview(output)


if __name__ == "__main__":
    main()
