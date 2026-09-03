#!/usr/bin/env python
"""PDF diagnóstico de `clean_lightcurve`: qué puntos quita el sigma-clip y dónde.

Un panel por (TIC, sector) en tiempo vs flujo: en negro lo que sobrevive y en
rojo lo que se lleva `sigma_clip_gap_edges`. Las líneas verticales punteadas
marcan los gaps > 2 d, que es donde el sigma-clip actúa.

    python scripts/build_cleaning_review.py --dir review_50 --worst 10
"""
import argparse
import sys
from pathlib import Path

import matplotlib
from matplotlib.backends.backend_pdf import PdfPages

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _preview import open_in_preview  # noqa: E402

from msv import config
from msv.cleaning import clean_lightcurve
from msv.io import parse_fits_name, read_lc_fits
from msv.viz import plot_cleaning_curve

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dir", default="review_50",
                        help="subdirectorio de RAW_DIR con los FITS")
    parser.add_argument("--worst", type=int, default=None,
                        help="solo las N con mayor fracción removida")
    parser.add_argument("--per-page", type=int, default=4)
    parser.add_argument("--no-open", action="store_true")
    args = parser.parse_args()

    paths = sorted((config.RAW_DIR / args.dir).glob("*_lc.fits"))
    if not paths:
        raise SystemExit(f"sin FITS en {config.RAW_DIR / args.dir}")

    entries = []
    for path in paths:
        frame = read_lc_fits(path)
        time = frame["Time"].to_numpy()
        flux = frame["flux"].to_numpy()
        _, _, _, keep = clean_lightcurve(time, flux, return_mask=True)
        entries.append((path, time, flux, (~keep).sum() / keep.size))
    entries.sort(key=lambda e: -e[-1])
    if args.worst:
        entries = entries[:args.worst]

    outdir = config.RESULTS_DIR / "figures"
    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / "cleaning_review.pdf"
    with PdfPages(outpath) as pdf:
        for start in range(0, len(entries), args.per_page):
            page = entries[start:start + args.per_page]
            figure, axes = plt.subplots(len(page), 1, squeeze=False,
                                        figsize=(12, 2.6 * len(page)))
            for axis, (path, time, flux, frac) in zip(axes[:, 0], page):
                tic, sector = parse_fits_name(path)
                plot_cleaning_curve(time, flux, ax=axis,
                                    title=f"TIC {tic} — sector {sector}   "
                                          f"{frac * 100:.2f}% removido")
            figure.tight_layout()
            pdf.savefig(figure, dpi=140)
            plt.close(figure)
    print(f"{outpath}  ({len(entries)} curvas)")

    if not args.no_open:
        open_in_preview(outpath)


if __name__ == "__main__":
    main()
