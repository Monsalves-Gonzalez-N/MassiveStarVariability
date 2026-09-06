#!/usr/bin/env python
"""Etapa 1: revisar a ciegas el período que reporta el paper.

Una estrella-sector por fila, dos paneles: la curva en tiempo y el fold en el
período PUBLICADO. Nuestro período, nuestra clase y nuestras probabilidades no
aparecen a propósito — el juicio es sobre el período del paper y solo, si se
muestra el nuestro al lado, la respuesta queda anclada.

Cada fila lleva un formulario en el PDF (Preview lo llena y lo guarda):

    periodico        el fold cierra: el período del paper es el bueno
    multi periodico  hay señal coherente pero una sola frecuencia no la
                     describe; el período del paper puede ser una de varias
    irregular        no hay señal coherente a ese período
    half period      el fold muestra dos ciclos distintos superpuestos, o sea
                     que el período verdadero es el doble del publicado

Regenerar el PDF NO pierde lo ya contestado: si el archivo de salida existe, se
leen sus respuestas antes de escribirlo y se vuelven a poner por nombre de
campo. Con `--fresh` se descartan.

Después se leen las respuestas con `read_veredicto.py`, que las agrega a nivel
estrella y avisa si dos sectores de la misma se contradicen.

    PYTHONPATH=src python scripts/golden/build_2p_review.py
"""
import argparse
import sys
from pathlib import Path

import fitz
import matplotlib
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _preview import open_in_preview  # noqa: E402

from msv.config import RESULTS_DIR
from msv.viz import plot_phase_fold

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROWS_PER_PAGE = 4
ROW_HEIGHT = 3.2
# Cada panel son miles de puntos: en vectorial el PDF pesa 29 MB y Preview se
# arrastra al pasar de página. Rasterizados quedan unos pocos MB y los ejes,
# los textos y los widgets siguen siendo vectoriales.
RASTER_DPI = 130
WIDTH_RATIOS = [2.3, 2.3, 1.5]
FIGURE_WIDTH = 13.0

CHOICES = ["periodico", "multi periodico", "irregular", "half period"]
# Todo en pulgadas sobre el rectángulo reservado, porque los widgets del PDF se
# posicionan en puntos absolutos y no en fracción de eje.
BOX_SIDE = 0.20
BOX_LEFT = 0.15
FIRST_BOX_TOP = 0.28
BOX_SPACING = 0.31
NOTES_TOP_GAP = 0.28
NOTES_BOTTOM = 0.18


def draw_light_curve(axis, time, flux):
    axis.plot(time, flux, ".k", ms=1.5, alpha=0.6, rasterized=True)
    axis.set_xlabel("BTJD")
    axis.set_ylabel("Flux")
    axis.grid(alpha=0.3)


def rasterize_points(axis):
    """Rasterizar las nubes de puntos y dejar vectorial la mediana por bin."""
    for line in axis.get_lines():
        if line.get_linestyle() == "None":
            line.set_rasterized(True)


def annotation_rectangles(position, figure_size):
    """Rectángulos de los widgets, en fracción de figura.

    `position` es el bbox del eje reservado. Devuelve un rect por opción más el
    de las notas.
    """
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
    """Valores ya cargados en una versión anterior del mismo PDF.

    Un checkbox sin marcar vuelve como el string "Off", que en Python es
    verdadero: hay que comparar contra el estado y no evaluar el valor.
    """
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
    """Checkboxes y campo de texto sobre el PDF ya escrito.

    Los widgets se agregan después porque matplotlib no escribe AcroForm. Las
    coordenadas de fitz van desde arriba, las de matplotlib desde abajo.
    """
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

    # Sin NeedAppearances algunos visores dibujan el campo vacío aunque tenga
    # valor guardado.
    document.xref_set_key(document.pdf_catalog(), "AcroForm/NeedAppearances",
                          "true")
    document.saveIncr()
    document.close()
    return written


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--peaks", default=str(RESULTS_DIR / "golden" / "match_peaks.csv"))
    parser.add_argument("--curves", default=str(RESULTS_DIR / "golden" / "curves.pkl"))
    parser.add_argument("--tag", default="2P", help="tag del pico reportado a revisar")
    parser.add_argument("--out", default=None)
    parser.add_argument("--fresh", action="store_true",
                        help="descartar las respuestas del PDF anterior")
    parser.add_argument("--no-open", action="store_true")
    args = parser.parse_args()

    peaks = pd.read_csv(args.peaks)
    curves = pd.read_pickle(args.curves)
    output_default = (
        RESULTS_DIR / "figures" / f"golden_review_{args.tag.replace('/', '')}.pdf")
    output = Path(args.out) if args.out else output_default
    output.parent.mkdir(parents=True, exist_ok=True)

    previous = {} if args.fresh else read_existing_answers(output)
    if previous:
        print(f"{len(previous)} campos ya contestados en el PDF anterior")

    peaks["key"] = list(zip(peaks.TIC.astype(int), peaks.sector.astype(int)))
    selected = peaks[peaks.reportado & (peaks.tag == args.tag)]
    selected = selected.sort_values(["TIC", "sector"])
    rows = [row for _, row in selected.iterrows() if row.key in curves]
    print(f"{len(rows)} estrella-sector con tag {args.tag}, "
          f"{selected.TIC.nunique()} TIC")

    entries = []
    with PdfPages(output) as pdf:
        for start in range(0, len(rows), ROWS_PER_PAGE):
            page_rows = rows[start:start + ROWS_PER_PAGE]
            figure_size = (FIGURE_WIDTH, ROW_HEIGHT * len(page_rows))
            figure, axes = plt.subplots(
                len(page_rows), 3, figsize=figure_size, squeeze=False,
                gridspec_kw={"width_ratios": WIDTH_RATIOS})
            for position, row in enumerate(page_rows):
                time, flux = curves[row.key]
                good = np.isfinite(time) & np.isfinite(flux)
                time, flux = time[good], flux[good]
                draw_light_curve(axes[position][0], time, flux)
                plot_phase_fold(time, flux, row.period_gold,
                                ax=axes[position][1], phase_bins=40,
                                title=f"P = {row.period_gold:.4f} d")
                rasterize_points(axes[position][1])
                axes[position][2].set_axis_off()
                axes[position][0].set_title(
                    f"TIC {int(row.TIC)}   sector {int(row.sector)}   "
                    f"{row.baseline / row.period_gold:.0f} ciclos",
                    fontsize=10, loc="left")
            figure.tight_layout()
            for position, row in enumerate(page_rows):
                boxes, notes = annotation_rectangles(
                    axes[position][2].get_position(), figure_size)
                draw_annotation_labels(figure, boxes, notes)
                entries.append({
                    "page": start // ROWS_PER_PAGE,
                    "name": f"{int(row.TIC)}_{int(row.sector)}",
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
          "scripts/golden/read_veredicto.py")

    if not args.no_open:
        open_in_preview(output)


if __name__ == "__main__":
    main()
