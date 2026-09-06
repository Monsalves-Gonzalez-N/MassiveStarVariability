#!/usr/bin/env python
"""Barrido del criterio de ancho del ACF sobre el benchmark VSX.

`select_peaks_acf` exige que un pico dure al menos
`min(ACF_WIDTH_FRAC * P / cadencia, ACF_WIDTH_MAX_SAMPLES)` muestras. El
número sale de que la correlación de una modulación CONTINUA persiste ~0.27*P;
en una eclipsante separada el pico del ACF es tan angosto como el eclipse, que
no escala con P. Este barrido mide las dos caras de relajarlo:

    ganancia   recall del período de VSX (1:1, tolerancia 5%) con el ACF SOLO
    costo      picos por estrella, que es lo que después tiene que clasificar
               la CNN, y picos espurios en las estrellas `bad`

Se recalcula el ACF una sola vez por estrella y se evalúa cada configuración
sobre la misma grilla.

    PYTHONPATH=src python scripts/vsx/barrido_ancho_acf.py
"""
import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from msv.cleaning import clean_lightcurve
from msv.config import CATALOGS_DIR, RESULTS_DIR
from msv.peaks import select_peaks_acf
from msv.periodograms import acf_periodogram

TOLERANCE = 0.05
CONFIGURACIONES = [
    ("actual   frac=0.05 tope=20", 0.05, 20),
    ("frac=0.05 tope=12", 0.05, 12),
    ("frac=0.05 tope=8", 0.05, 8),
    ("frac=0.025 tope=20", 0.025, 20),
    ("frac=0.025 tope=8", 0.025, 8),
    ("frac=0.01 tope=20", 0.01, 20),
    ("sin ancho (piso 3)", None, None),
]


def cachear_grillas(cache_path, tics):
    if cache_path.exists():
        with open(cache_path, "rb") as handle:
            return pickle.load(handle)
    lightcurves = pd.read_parquet(RESULTS_DIR / "vsx" / "lc.parquet")
    grids = {}
    for tic, block in lightcurves.groupby("TIC"):
        if int(tic) not in tics:
            continue
        time, flux, error = clean_lightcurve(block.Time.values, block.flux.values,
                                             block.flux_err.values)
        if len(time) < 20:
            continue
        table = acf_periodogram(time, flux, error)
        grids[int(tic)] = (table.per.values, table.power.values, table.fap.values,
                           table.attrs["cadence_days"])
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as handle:
        pickle.dump(grids, handle)
    return grids


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache",
                        default=str(RESULTS_DIR / "vsx" / "acf_grids.pkl"))
    args = parser.parse_args()

    review = pd.read_csv(CATALOGS_DIR / "vsx_visual_review.csv")
    review = review[review.per_vsx.notna()
                    & review.vis_verdict.isin(["ok", "maybe", "unconstrained", "bad"])]
    periods = dict(zip(review.TIC.astype(int), review.per_vsx))
    verdicts = dict(zip(review.TIC.astype(int), review.vis_verdict))

    grids = cachear_grillas(Path(args.cache), set(periods))
    print(f"{len(grids)} estrellas con grilla de ACF")

    filas = []
    for etiqueta, width_frac, tope in CONFIGURACIONES:
        encontrados = {}
        n_picos = {}
        for tic, (per, power, fap, cadence) in grids.items():
            if width_frac is None:
                peaks = select_peaks_acf(per, power, fap, cadence=cadence,
                                         width_frac=None, width=3)
            else:
                peaks = select_peaks_acf(per, power, fap, cadence=cadence,
                                         width_frac=width_frac,
                                         width_max_samples=tope)
            verdict = verdicts[tic]
            n_picos.setdefault(verdict, []).append(len(peaks))
            if peaks.empty:
                encontrados.setdefault(verdict, []).append(False)
                continue
            ratio = np.abs(peaks.per.values / periods[tic] - 1.0)
            encontrados.setdefault(verdict, []).append(bool((ratio < TOLERANCE).any()))

        fila = {"configuracion": etiqueta}
        for verdict in ["ok", "maybe", "unconstrained", "bad"]:
            if verdict not in encontrados:
                continue
            fila[f"recall {verdict}"] = round(
                100 * float(np.mean(encontrados[verdict])), 1)
            fila[f"picos {verdict}"] = round(float(np.mean(n_picos[verdict])), 1)
        filas.append(fila)

    tabla = pd.DataFrame(filas).set_index("configuracion")
    columnas = ([c for c in tabla.columns if c.startswith("recall")]
                + [c for c in tabla.columns if c.startswith("picos")])
    print("\nrecall 1:1 del ACF SOLO [%] y picos por estrella")
    print(tabla[columnas].to_string())
    tabla.to_csv(RESULTS_DIR / "vsx" / "barrido_ancho_acf.csv")


if __name__ == "__main__":
    main()
