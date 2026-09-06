#!/usr/bin/env python
"""PDF de revisión del pipeline de una sola red.

Una ESTRELLA por fila, cuatro paneles:

  1. la curva limpia en tiempo, para ver de qué se está hablando
  2. el fold en el período REPORTADO, que es el de menor `p_LPV` entre los
     candidatos de clase periódica — no el de mayor probabilidad de clase, que
     con mediana 0.996 no ordena nada
  3. el histograma 2D de 32x32 que es exactamente lo que entra a la CNN
  4. la distribución de probabilidad de las 5 clases en el pico reportado, o
     con `--promedio` promediada sobre TODOS los picos de la estrella-sector,
     con el valor del pico reportado como marca vertical negra
  5. `-log10 p_LPV` de los 12 mejores candidatos, ordenados como ranking, con
     el reportado en negro y el umbral de nivel `alta` como línea vertical

El panel 4 es el que hace revisable la elección: si el candidato elegido está
apenas por encima del segundo, la estrella merece mirarse a mano. Ver los
paneles 2 y 3 juntos importa porque la ambigüedad P vs 2P es obvia en el fold
(en 2P se ven dos ciclos idénticos) y no en el histograma.

El encabezado da la clase, su probabilidad, el % irregular de la estrella y el
nivel. El % irregular se lee según la clase: en ELL, por encima de ~10% la
clase es sospechosa (una elipsoidal es UNA modulación continua); en E es sólo
descripción del objeto, porque un eclipse sobrevive superpuesto a otra
variabilidad. Ver `docs/HANDOFF_contaminacion_ELL.md` §3.

    python scripts/build_review_una_red.py results/clasificacion_una_red.csv \
        --curves results/phasefold_curves.pkl
"""
import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _preview import open_in_preview  # noqa: E402

from msv import config
from msv.features import phase_fold_hist2d
from msv.viz import GROUP_COLORS, plot_phase_fold

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

LOG_PLPV_MIN = 12.0
ROWS_PER_PAGE = 4


def draw_light_curve(axis, time, flux):
    axis.plot(time, flux, ".k", ms=1.5, alpha=0.6)
    axis.set_xlabel("BTJD")
    axis.set_ylabel("Flux")
    axis.grid(alpha=0.3)


def draw_hist2d(axis, time, flux, period):
    image = phase_fold_hist2d(time, flux, period, norm=config.HIST_NORM)
    axis.imshow(image, aspect="auto", cmap="viridis",
                extent=(0, 1, flux.min(), flux.max()))
    axis.set_xlabel("Phase")
    axis.set_title(f"hist2d 32x32 ({config.HIST_NORM})", fontsize=8)
    axis.set_yticks([])


def draw_distribution(axis, row, names, block=None):
    """La barra de probabilidades del pico reportado.

    Con `block` las barras pasan a ser el promedio sobre TODOS los picos de la
    estrella-sector y el pico reportado queda como marca vertical. Son dos
    preguntas distintas: la barra del pico dice qué vio la red en ESE fold, el
    promedio dice hacia dónde tira la estrella cuando se le prueban todos los
    períodos candidatos. `p_LPV` promediado así es exactamente la columna
    `irregular`.
    """
    if block is None:
        values = [row[f"p_{name}"] for name in names]
    else:
        values = [block[f"p_{name}"].mean() for name in names]
    axis.barh(range(len(names)), values,
              color=[GROUP_COLORS.get(name, "0.6") for name in names], height=0.7)
    if block is not None:
        for position, name in enumerate(names):
            axis.plot([row[f"p_{name}"]] * 2, [position - 0.42, position + 0.42],
                      color="k", linewidth=1.2)
    axis.set_yticks(range(len(names)))
    axis.set_yticklabels(names, fontsize=7)
    axis.invert_yaxis()
    # 1.03 y no 1: la marca del pico reportado cae en p=1.000 y en el borde
    # exacto del eje no se ve.
    axis.set_xlim(0, 1.03 if block is not None else 1)
    axis.set_xlabel("p (CNN)" if block is None
                    else f"p media de {len(block)} picos")
    axis.grid(alpha=0.3, axis="x")
    for position, value in enumerate(values):
        if value > 0.02:
            axis.text(min(value + 0.02, 0.7), position, f"{value:.2f}",
                      va="center", fontsize=6.5)


def draw_candidates(axis, block, chosen_index, top=12):
    """Los mejores candidatos como ranking: el eje y es el orden, no el período.

    Mostrar los ~25 picos de la estrella deja las etiquetas ilegibles y los de
    abajo no aportan; lo que hay que poder leer es por cuánto le ganó el
    elegido al segundo.
    """
    ranked = block.sort_values("log_pLPV", ascending=False).head(top)
    limit = ranked.log_pLPV.max() * 1.55
    for position, (index, row) in enumerate(ranked.iterrows()):
        is_chosen = index == chosen_index
        axis.barh(-position, row.log_pLPV,
                  color="k" if is_chosen else GROUP_COLORS.get(row.clase, "0.6"),
                  alpha=1.0 if is_chosen else 0.55, height=0.72)
        axis.text(row.log_pLPV + 0.02 * limit, -position,
                  f"{row.per:.3f} {row.clase}", va="center", fontsize=6.5,
                  fontweight="bold" if is_chosen else "normal", color="0.1")
    axis.axvline(LOG_PLPV_MIN, color="tab:red", lw=1.0, ls="--")
    axis.set_yticks([])
    axis.set_xlabel(r"$-\log_{10}\ p_{\rm LPV}$")
    axis.set_xlim(0, limit)
    axis.set_ylim(-len(ranked) + 0.4, 0.6)
    axis.grid(alpha=0.3, axis="x")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv", help="salida de step_clasificar_una_red.py")
    parser.add_argument("--curves", default=str(config.RESULTS_DIR / "phasefold_curves.pkl"))
    parser.add_argument("--out", default=None)
    parser.add_argument("--promedio", action="store_true",
                        help="el panel de probabilidades promediado sobre "
                             "TODOS los picos, con el reportado como marca")
    parser.add_argument("--no-open", action="store_true")
    args = parser.parse_args()

    table = pd.read_csv(args.csv)
    curves = pd.read_pickle(args.curves)
    output = Path(args.out) if args.out else config.RESULTS_DIR / "figures" / "review_una_red.pdf"
    output.parent.mkdir(parents=True, exist_ok=True)

    table["key"] = list(zip(table.TIC.astype(int), table.sector.astype(int)))
    reported = table[table.reportado].sort_values(["nivel", "log_pLPV"],
                                                  ascending=[True, False])
    stars = [row for _, row in reported.iterrows() if row.key in curves]
    print(f"{len(stars)} estrellas con curva de las {len(reported)} reportadas")

    with PdfPages(output) as pdf:
        for start in range(0, len(stars), ROWS_PER_PAGE):
            page = stars[start:start + ROWS_PER_PAGE]
            figure, axes = plt.subplots(
                len(page), 5, figsize=(19, 3.1 * len(page)), squeeze=False,
                gridspec_kw={"width_ratios": [2.2, 2.0, 1.2, 1.2, 1.4]})
            for row_position, row in enumerate(page):
                time, flux = curves[row.key]
                good = np.isfinite(time) & np.isfinite(flux)
                time, flux = time[good], flux[good]
                block = table[table.key == row.key]
                draw_light_curve(axes[row_position][0], time, flux)
                plot_phase_fold(time, flux, row.per, ax=axes[row_position][1],
                                phase_bins=40, title=f"P = {row.per:.4f} d")
                draw_hist2d(axes[row_position][2], time, flux, row.per)
                draw_distribution(axes[row_position][3], row,
                                  ["ELL", "Pulsating", "E", "LPV", "Rndm"],
                                  block=block if args.promedio else None)
                draw_candidates(axes[row_position][4], block, row.name)
                axes[row_position][0].set_title(
                    f"TIC {int(row.TIC)} s{int(row.sector)}   "
                    f"{row.clase} (p={row.prob:.3f})   "
                    f"{100 * row.irregular:.0f}% irregular   "
                    r"$-\log_{10}p_{\rm LPV}$=" f"{row.log_pLPV:.1f}   "
                    f"[{row.nivel}]",
                    fontsize=9, loc="left",
                    color="k" if row.nivel == "alta" else "tab:red")
            figure.tight_layout()
            pdf.savefig(figure)
            plt.close(figure)
    print(f"escrito {output}")
    if not args.no_open:
        open_in_preview(output)


if __name__ == "__main__":
    main()
