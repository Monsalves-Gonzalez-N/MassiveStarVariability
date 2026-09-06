#!/usr/bin/env python
"""El prewhitening paso a paso, para una estrella-sector.

Existe porque la operación se malentiende fácil: no selecciona PUNTOS de la
curva. En cada instante el brillo es la suma de todas las variaciones más el
ruido, así que no hay un subconjunto de puntos que sea una variación y otro que
sea la otra — todos los puntos participan de todas. Lo que se separa es en
FRECUENCIA, y el ajuste que se resta está evaluado en todos los puntos.

Cada fila es una iteración:

  izquierda   el espectro de amplitud del residuo actual (Lomb-Scargle, el
              mismo de `msv.periodograms`, en ppt en vez de potencia
              normalizada), la curva de corte `snr_min` x ruido local, y el
              pico elegido
  derecha     ese mismo residuo en el tiempo, con el modelo del paso encima —
              sinusoide del pico más sus armónicos, ajustada por mínimos
              cuadrados sobre TODOS los puntos

La última fila es donde se para: el pico más alto ya no llega al corte y el
espectro queda por debajo de la línea en todas partes.

    PYTHONPATH=src python scripts/golden/plot_prewhiten_steps.py --tic 42889751 --sector 6
"""
import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _preview import open_in_preview  # noqa: E402

from msv.cleaning import clean_lightcurve
from msv.config import RESULTS_DIR
from msv.prewhiten import (DEFAULT_HARMONICS, DEFAULT_SNR_MIN,
                           amplitude_spectrum, extract_components,
                           frequency_grid, noise_spectrum, to_relative)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RASTER_DPI = 140
ZOOM_DAYS = 8.0
ROW_HEIGHT = 2.5
WIDTH_RATIOS = [2.0, 1.5]


def draw_step(spectrum_axis, curve_axis, time, entry, grid, snr_min,
              frequency_limit, flux_limit):
    noise = noise_spectrum(grid, entry["spectrum"])
    spectrum_axis.plot(grid, entry["spectrum"], "-k", lw=0.8)
    spectrum_axis.plot(grid, snr_min * noise, "-", color="tab:blue", lw=1.1,
                       label=f"corte = {snr_min:.0f} x ruido local")
    color = "tab:red" if entry["accepted"] else "0.5"
    spectrum_axis.plot([entry["frequency"]], [entry["amplitude"]], "v",
                       color=color, ms=8)
    spectrum_axis.set_xlim(0, frequency_limit)
    spectrum_axis.set_ylim(0, flux_limit)
    spectrum_axis.set_ylabel("Amplitud [ppt]")
    spectrum_axis.grid(alpha=0.3)
    verdict = ("se resta" if entry["accepted"]
               else f"SNR < {snr_min:.0f}: se para, no se resta")
    spectrum_axis.set_title(
        f"Paso {entry['step']}   pico en f={entry['frequency']:.4f}/d "
        f"(P={1 / entry['frequency']:.4f} d)   A={entry['amplitude']:.2f} ppt   "
        f"SNR={entry['snr']:.1f}   ->  {verdict}", fontsize=9, loc="left")
    if entry["step"] == 1:
        spectrum_axis.legend(fontsize=7, loc="upper right")

    window = time <= time.min() + ZOOM_DAYS
    curve_axis.plot(time[window], entry["residual"][window], ".", color="0.35",
                    ms=1.8, alpha=0.6, rasterized=True)
    if entry["accepted"]:
        curve_axis.plot(time[window], entry["model"][window], "-",
                        color="tab:red", lw=1.4)
    curve_axis.set_ylabel("ppt")
    curve_axis.grid(alpha=0.3)
    curve_axis.set_title("residuo que entra al paso" +
                         ("  +  el modelo que se le resta" if entry["accepted"]
                          else "  (queda asi)"), fontsize=8.5, loc="left")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tic", type=int, default=42889751)
    parser.add_argument("--sector", type=int, default=6)
    parser.add_argument("--lc", default=str(RESULTS_DIR / "golden" / "lc.parquet"))
    parser.add_argument("--n-max", type=int, default=8,
                        help="tope de iteraciones; conviene dejarlo por encima "
                             "del número de componentes para que se vea el paso "
                             "en el que se para")
    parser.add_argument("--n-harmonics", type=int, default=DEFAULT_HARMONICS)
    parser.add_argument("--snr-min", type=float, default=DEFAULT_SNR_MIN)
    parser.add_argument("--out", default=None)
    parser.add_argument("--no-open", action="store_true")
    args = parser.parse_args()

    block = pd.read_parquet(
        args.lc, filters=[("TIC", "=", args.tic), ("sector", "=", args.sector)],
        columns=["Time", "flux", "flux_err"])
    if block.empty:
        raise SystemExit(f"TIC {args.tic} sector {args.sector} no esta en {args.lc}")
    time, flux, error = clean_lightcurve(block.Time.to_numpy(),
                                         block.flux.to_numpy(),
                                         block.flux_err.to_numpy())
    relative_flux, _ = to_relative(flux)
    components, residual, history = extract_components(
        time, relative_flux, n_max=args.n_max, n_harmonics=args.n_harmonics,
        snr_min=args.snr_min, return_history=True)

    grid = frequency_grid(time)
    final_spectrum = amplitude_spectrum(time, residual, grid)
    frequency_limit = min(grid.max(),
                          1.5 * max(entry["frequency"] for entry in history))
    flux_limit = 1.15 * max(entry["spectrum"].max() for entry in history)

    output = Path(args.out) if args.out else (
        RESULTS_DIR / "figures" / f"prewhiten_pasos_TIC{args.tic}_s{args.sector}.pdf")
    output.parent.mkdir(parents=True, exist_ok=True)

    n_rows = len(history) + 1
    figure, axes = plt.subplots(n_rows, 2, figsize=(14.0, 1.4 + ROW_HEIGHT * n_rows),
                               gridspec_kw={"width_ratios": WIDTH_RATIOS,
                                            "hspace": 0.62, "wspace": 0.20},
                               squeeze=False)
    for position, entry in enumerate(history):
        draw_step(axes[position][0], axes[position][1], time, entry, grid,
                  args.snr_min, frequency_limit, flux_limit)

    noise = noise_spectrum(grid, final_spectrum)
    axes[-1][0].plot(grid, final_spectrum, "-k", lw=0.8)
    axes[-1][0].plot(grid, args.snr_min * noise, "-", color="tab:blue", lw=1.1)
    axes[-1][0].set_xlim(0, frequency_limit)
    axes[-1][0].set_ylim(0, flux_limit)
    axes[-1][0].set_ylabel("Amplitud [ppt]")
    axes[-1][0].grid(alpha=0.3)
    axes[-1][0].set_title(
        f"Residuo final: {len(components)} componentes extraidas, "
        f"sigma {relative_flux.std():.2f} -> {residual.std():.2f} ppt "
        f"({1 - residual.var() / relative_flux.var():.0%} de la varianza "
        f"explicada)", fontsize=9, loc="left")
    window = time <= time.min() + ZOOM_DAYS
    axes[-1][1].plot(time[window], residual[window], ".", color="0.35", ms=1.8,
                     alpha=0.6, rasterized=True)
    axes[-1][1].set_ylabel("ppt")
    axes[-1][1].grid(alpha=0.3)
    axes[-1][1].set_title("lo que queda", fontsize=8.5, loc="left")
    for axis in (axes[-1][0], axes[-1][1]):
        axis.set_xlabel("Frecuencia [1/d]" if axis is axes[-1][0] else "BTJD")
    axes[-2][0].set_xlabel("Frecuencia [1/d]")
    axes[-2][1].set_xlabel("BTJD")

    figure.suptitle(
        f"TIC {args.tic} sector {args.sector} — el prewhitening paso a paso\n"
        f"No se seleccionan puntos: se ajusta un modelo sobre TODOS los puntos y "
        f"se resta.  El corte de {args.snr_min:.0f} sigma es sobre el ESPECTRO, "
        f"no sobre la curva.", fontsize=10.5, y=0.997, va="top")
    figure.subplots_adjust(left=0.06, right=0.985,
                           top=1 - 0.85 / figure.get_figheight(),
                           bottom=0.45 / figure.get_figheight(), hspace=0.62,
                           wspace=0.20)
    with PdfPages(output) as pdf:
        pdf.savefig(figure, dpi=RASTER_DPI)
    plt.close(figure)

    print(f"-> {output}   {len(history)} pasos")
    if not args.no_open:
        open_in_preview(output)


if __name__ == "__main__":
    main()
