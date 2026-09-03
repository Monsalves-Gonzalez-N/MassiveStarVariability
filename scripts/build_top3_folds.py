#!/usr/bin/env python
"""PDF para arbitrar el período a mano: los N mejores candidatos de cada
estrella, doblados uno al lado del otro, con un recuadro en blanco para anotar.

Una ESTRELLA por fila y un fold por candidato, ordenados de mejor a peor. El
punto es que la decisión de período se toma COMPARANDO folds, no mirando uno:
un armónico 2x de una eclipsante doblado solo se ve perfectamente sano, y solo
al ponerlo al lado del fundamental se ve que ahí los dos mínimos tenían
profundidades distintas.

Cada panel lleva un ID corto (`p03-2b`: página 3, fila 2, candidato b) para que
las notas escritas en el PDF se puedan referenciar sin ambigüedad, y el mismo
ID va en el CSV `..._panels.csv` que se escribe al lado del PDF, con el TIC,
el sector, el período y la clase de cada panel.

El orden de los candidatos lo elige `--rank`:
  prob         la probabilidad media de su mejor clase no-Rndm (default): son
               los candidatos que de verdad compiten por la clasificación.
  prominence   la prominencia del pico en el periodograma: es el orden en que
               los propuso la detección, útil para ver si el ranking por
               probabilidad está reordenando cosas.

    python scripts/build_top3_folds.py results/clasificacion_review20.csv \
        --curves results/phasefold_curves.pkl --stat median
"""
import argparse
import string
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _preview import open_in_preview  # noqa: E402
from build_lc_prob_review import (STATS, best_non_random, group_columns,  # noqa: E402
                                  winning_votes)

from msv import config
from msv.viz import GROUP_COLORS, plot_phase_fold

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

LETTERS = string.ascii_lowercase


def panel_label(index):
    """a..z, luego aa, ab, ... Hay estrellas con más de 26 candidatos."""
    label = ""
    index += 1
    while index:
        index, remainder = divmod(index - 1, len(LETTERS))
        label = LETTERS[remainder] + label
    return label


def rank_candidates(group, names, how, how_many=None):
    """Los `how_many` mejores candidatos de una estrella, mejor primero.

    `how_many=None` (o 0) devuelve todos los candidatos de la estrella.
    """
    if how == "prominence":
        order = group.sort_values("prominence", ascending=False)
    else:
        _, probabilities, _, _ = best_non_random(group, names)
        order = group.iloc[np.argsort(-probabilities)]
    return order if not how_many else order.head(how_many)


def draw_note_box(axis, panel_id):
    for spine in axis.spines.values():
        spine.set_edgecolor("0.75")
        spine.set_linestyle((0, (3, 3)))
    axis.set_xticks([]); axis.set_yticks([])
    axis.text(0.03, 0.94, panel_id, transform=axis.transAxes, fontsize=8,
              fontweight="bold", va="top", color="0.45")
    axis.text(0.03, 0.80, "nota:", transform=axis.transAxes, fontsize=7.5,
              va="top", color="0.6")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("classification", help="CSV de step_brf.py")
    parser.add_argument("--curves", required=True,
                        help="pickle {(TIC, sector): (time, flux)}")
    parser.add_argument("--top", type=int, default=3,
                        help="candidatos por estrella")
    parser.add_argument("--rank", default="prob", choices=["prob", "prominence"])
    parser.add_argument("--stat", default="median", choices=list(STATS))
    parser.add_argument("--per-page", type=int, default=3,
                        help="estrellas por página")
    parser.add_argument("--phase-bins", type=int, default=50)
    parser.add_argument("--outdir", default=str(config.RESULTS_DIR / "figures"))
    parser.add_argument("--outname", default="top_folds.pdf")
    parser.add_argument("--no-open", action="store_true")
    args = parser.parse_args()

    table = pd.read_csv(args.classification)
    names = group_columns(table)
    if not names:
        raise SystemExit("el CSV no trae columnas p_<clase>: "
                         "regenerarlo con la versión actual de step_brf.py")
    curves = pd.read_pickle(args.curves)
    period_col = "period" if "period" in table.columns else "per"

    stars = []
    for (tic, sector), group in table.groupby(["TIC", "sector"], sort=True):
        chosen = rank_candidates(group, names, args.rank, args.top)
        classes, centers, low, high = best_non_random(chosen, names, args.stat)
        votes = winning_votes(chosen, names)
        stars.append((int(tic), int(sector), chosen, classes, centers,
                      low, high, votes))

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / args.outname

    panel_rows = []
    with PdfPages(outpath) as pdf:
        for start in range(0, len(stars), args.per_page):
            page_number = start // args.per_page + 1
            page = stars[start:start + args.per_page]
            figure, axes = plt.subplots(
                len(page), args.top + 1, squeeze=False,
                figsize=(4.0 * args.top + 3.2, 2.6 * len(page)),
                gridspec_kw={"width_ratios": [3] * args.top + [2.2]})
            for row_number, (row_axes, star) in enumerate(zip(axes, page), start=1):
                tic, sector, chosen, classes, centers, low, high, votes = star
                for column, (axis, (_, candidate)) in enumerate(
                        zip(row_axes, chosen.iterrows())):
                    panel_id = f"p{page_number:02d}-{row_number}{panel_label(column)}"
                    period = candidate[period_col]
                    time, flux = curves[(tic, sector)]
                    good = np.isfinite(time) & np.isfinite(flux)
                    plot_phase_fold(time[good], flux[good], period, ax=axis,
                                    phase_bins=args.phase_bins, color="0.6",
                                    ms=1.2)
                    spread = ("" if low is None else
                              (f"  [{low[column]:.2f}, {high[column]:.2f}]"))
                    vote = "" if votes is None else f"  v={votes[column]:.2f}"
                    axis.set_title(
                        f"{panel_id}   P = {period:.4f} d ({candidate['source']})\n"
                        f"{classes[column]} {centers[column]:.3f}{spread}{vote}",
                        fontsize=8.5,
                        color=GROUP_COLORS.get(classes[column], "k"))
                    axis.set_xlabel("phase", fontsize=8)
                    axis.set_ylabel("flux" if column == 0 else "", fontsize=8)
                    axis.tick_params(labelsize=7)
                    panel_rows.append({
                        "panel": panel_id, "TIC": tic, "sector": sector,
                        "source": candidate["source"], "per": period,
                        "prominence": candidate["prominence"],
                        "clase": classes[column], "centro": centers[column],
                        "lo": None if low is None else low[column],
                        "hi": None if high is None else high[column],
                        "voto": None if votes is None else votes[column],
                        "nota": ""})
                draw_note_box(row_axes[-1], f"TIC {tic} s{sector}")
            figure.tight_layout()
            pdf.savefig(figure, dpi=140)
            plt.close(figure)

    panels = pd.DataFrame(panel_rows)
    panels_path = outpath.with_name(outpath.stem + "_panels.csv")
    panels.to_csv(panels_path, index=False)
    print(f"{outpath}  ({len(stars)} estrellas x {args.top} candidatos, "
          f"orden por {args.rank})")
    print(f"{panels_path}  ({len(panels)} paneles; la columna `nota` es para "
          f"volcar lo que escribas en el PDF)")
    if not args.no_open:
        open_in_preview(outpath)


if __name__ == "__main__":
    main()
