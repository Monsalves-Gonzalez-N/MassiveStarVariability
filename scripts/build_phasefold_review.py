#!/usr/bin/env python
"""PDF de phase-folds con la clasificación CNN+BRF encima, para arbitrar
visualmente qué candidato de período es el correcto.

Una ESTRELLA por fila, doblada en el candidato de MAYOR probabilidad con el
argmax tomado SIN `Rndm` — el mismo que marca ◄ en build_lc_prob_review.py,
para poder leer los dos PDF juntos: las estrellas van en el mismo orden y con
la misma paginación, así que página N / panel M es la misma estrella en ambos.
Descartar `Rndm` del argmax importa: un
`Rndm 0.99` dice "doblando a este período no se ve nada", que no es una
afirmación sobre la estrella y taparía al candidato que sí lo es.

El título reporta el resumen elegido con `--stat`: la media con su intervalo
p16-p84, o la mediana con la caja Q1-Q3, y SIEMPRE el voto `v` (fracción de las
pasadas MC en que esa clase gana). El voto va siempre porque la distribución
por pico es bimodal y ningún par centro+ancho la describe: en un pico con masa
en 0 y en 1 la mediana se va al modo mayoritario y la caja se estira de 0 a 1,
mientras `v` dice directamente cuánta masa tiene cada modo.

A la izquierda la curva doblada, a la derecha el histograma 2D de 32x32 que es
exactamente lo que entra a la CNN. Ver los dos importa: la ambigüedad P vs P/2
de una eclipsante es invisible para la red (doblar en P/2 superpone primario y
secundario y sigue pareciendo un eclipse por ciclo) pero es obvia en el fold,
donde en P se ven dos profundidades.

    python scripts/build_phasefold_review.py clasificacion.csv --data-dir test_data
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
from build_lc_prob_review import (STATS, best_candidate, best_non_random,  # noqa: E402
                                   group_columns, winning_votes)

from msv import config
from msv.features import phase_fold_hist2d
from msv.viz import plot_phase_fold

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def draw_hist2d(axis, time, flux, period):
    axis.imshow(phase_fold_hist2d(time, flux, period),
                cmap="viridis", aspect="auto", interpolation="nearest")
    axis.set_xticks([]); axis.set_yticks([])
    axis.set_xlabel("hist2d 32x32 (input CNN)", fontsize=8)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("classification",
                        help="CSV de step_brf.py — el MISMO que build_lc_prob_review.py, "
                             "para que el período graficado sea el ◄ de ese PDF")
    parser.add_argument("--data-dir", default="test_data",
                        help="directorio con lightcurves_all.parquet")
    parser.add_argument("--curves", default=None,
                        help="pickle {(TIC, sector): (time, flux)} con curvas ya "
                             "limpias, en vez de leer el parquet")
    parser.add_argument("--per-page", type=int, default=4,
                        help="estrellas por página; el default coincide con el "
                             "de build_lc_prob_review.py para que página N / "
                             "panel M sea la misma estrella en los dos PDF")
    parser.add_argument("--outdir", default=None)
    parser.add_argument("--outname", default="phasefold_review.pdf",
                        help="nombre del PDF dentro de --outdir")
    parser.add_argument("--stat", default="mean", choices=list(STATS),
                        help="resumen en el título: media +/- sigma o "
                             "mediana Q1-Q3; el voto va siempre")
    parser.add_argument("--no-open", action="store_true")
    args = parser.parse_args()

    table = pd.read_csv(args.classification)
    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = config.REPO_ROOT / data_dir
    if args.curves:
        curves = pd.read_pickle(args.curves)
    else:
        lc_path = data_dir / "lightcurves_all.parquet"
        if not lc_path.exists() or lc_path.stat().st_size == 0:
            raise SystemExit(f"{lc_path} falta o está en 0 bytes "
                             "(Dropbox sin sincronizar?)")
        curves = {key: (group["Time"].to_numpy(), group["flux"].to_numpy())
                  for key, group in pd.read_parquet(lc_path).groupby(["TIC", "sector"])}

    outdir = config.REPO_ROOT / (args.outdir or (config.RESULTS_DIR / "figures"))
    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / args.outname

    period_col = "period" if "period" in table.columns else "per"
    names = group_columns(table)
    if not names:
        raise SystemExit("el CSV no trae columnas p_<clase>: "
                         "regenerarlo con la versión actual de step_brf.py")

    chosen = [best_candidate(group, names)
              for _, group in table.groupby(["TIC", "sector"], sort=False)]
    longest = table.loc[chosen].copy()
    classes, probabilities, low, high = best_non_random(longest, names, args.stat)
    votes = winning_votes(longest, names)
    longest["clase_sin_rndm"] = classes
    longest["prob_sin_rndm"] = probabilities
    longest["lo_sin_rndm"] = np.nan if low is None else low
    longest["hi_sin_rndm"] = np.nan if high is None else high
    longest["voto_sin_rndm"] = np.nan if votes is None else votes
    longest = longest.sort_values(["TIC", "sector"])
    rows = list(longest.itertuples(index=False))
    with PdfPages(outpath) as pdf:
        for start in range(0, len(rows), args.per_page):
            page = rows[start:start + args.per_page]
            figure, axes = plt.subplots(len(page), 2, squeeze=False,
                                        figsize=(11, 2.3 * len(page)),
                                        gridspec_kw={"width_ratios": [3, 1]})
            for (left, right), row in zip(axes, page):
                time, flux = curves[(row.TIC, row.sector)]
                good = np.isfinite(time) & np.isfinite(flux)
                time, flux = time[good], flux[good]
                period = getattr(row, period_col)
                plot_phase_fold(time, flux, period, ax=left, phase_bins=50,
                                color="0.55", ms=1.5)
                draw_hist2d(right, time, flux, period)
                kind = getattr(row, "kind", None)
                harmonic = getattr(row, "harmonic_order", None)
                detalle = f", {kind}" if isinstance(kind, str) else ""
                if harmonic is not None and not pd.isna(harmonic):
                    detalle += f" {int(harmonic)}x"
                left.set_xlabel("phase", fontsize=8)
                interval = ("" if pd.isna(row.lo_sin_rndm) else
                            f" [{row.lo_sin_rndm:.3f}, {row.hi_sin_rndm:.3f}]")
                vote = ("" if pd.isna(row.voto_sin_rndm)
                        else f"  v={row.voto_sin_rndm:.2f}")
                left.set_title(
                    f"TIC {row.TIC} s{row.sector}   P = {period:.4f} d  "
                    f"({row.source}{detalle})   →   "
                    f"{row.clase_sin_rndm}   {row.prob_sin_rndm:.3f}"
                    f"{interval}{vote}",
                    fontsize=9)
                left.set_ylabel("flux", fontsize=8)
            figure.tight_layout()
            pdf.savefig(figure, dpi=140)
            plt.close(figure)

    print(f"{outpath}  ({len(rows)} estrellas, en su candidato más probable)")
    if not args.no_open:
        open_in_preview(outpath)


if __name__ == "__main__":
    main()
