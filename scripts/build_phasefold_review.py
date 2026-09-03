#!/usr/bin/env python
"""PDF de phase-folds con la clasificación CNN+BRF encima, para arbitrar
visualmente qué candidato de período es el correcto.

Cada fila es un candidato: a la izquierda la curva doblada, a la derecha el
histograma 2D de 32x32 que es exactamente lo que entra a la CNN. Ver los dos
importa: la ambigüedad P vs P/2 de una eclipsante es invisible para la red
(doblar en P/2 superpone primario y secundario y sigue pareciendo un eclipse
por ciclo) pero es obvia en el fold, donde en P se ven dos profundidades.

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
    parser.add_argument("classification", help="CSV de step_brf / classify_peaks")
    parser.add_argument("--data-dir", default="test_data",
                        help="directorio con lightcurves_all.parquet")
    parser.add_argument("--curves", default=None,
                        help="pickle {(TIC, sector): (time, flux)} con curvas ya "
                             "limpias, en vez de leer el parquet")
    parser.add_argument("--per-page", type=int, default=5, help="candidatos por página")
    parser.add_argument("--outdir", default=None)
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
    outpath = outdir / "phasefold_review.pdf"

    table = table.sort_values(["TIC", "sector", "prob"], ascending=[True, True, False])
    rows = list(table.itertuples(index=False))
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
                plot_phase_fold(time, flux, row.period, ax=left, phase_bins=50,
                                color="0.55", ms=1.5)
                draw_hist2d(right, time, flux, row.period)
                orden = ("" if pd.isna(getattr(row, "harmonic_order", None))
                         else f" {int(row.harmonic_order)}x")
                left.set_xlabel("phase", fontsize=8)
                left.set_title(
                    f"TIC {row.TIC} s{row.sector}   P = {row.period:.4f} d  "
                    f"({row.kind}{orden})   →   {row.clase}   "
                    f"prob {row.prob:.3f} ± {row.sigma:.3f}   "
                    f"instability {row.instability:.2f}", fontsize=9)
                left.set_ylabel("flux", fontsize=8)
            figure.tight_layout()
            pdf.savefig(figure, dpi=140)
            plt.close(figure)

    print(f"{outpath}  ({len(rows)} candidatos)")
    if not args.no_open:
        open_in_preview(outpath)


if __name__ == "__main__":
    main()
