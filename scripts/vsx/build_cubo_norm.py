#!/usr/bin/env python
"""Rehacer el cubo de entrada a la CNN con otra normalizacion del hist2d.

Los picos NO cambian: el periodograma es el mismo y las filas salen en el mismo
orden que la clasificacion de referencia, para que todo lo demas (la sonda, las
etiquetas de VSX, la revision visual) se pueda cruzar sin reordenar nada.

    PYTHONPATH=src python scripts/vsx/build_cubo_norm.py --norm min_max \
        --out results/vsx/cnn_input_w001_minmax.npz
"""
import argparse

import numpy as np
import pandas as pd

from msv.cleaning import clean_lightcurve
from msv.config import RESULTS_DIR
from msv.features import phase_fold_hist2d

META_COLS = ["TIC", "sector", "source", "per", "power", "prominence", "width",
             "amplitude"]


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--clasificacion",
                        default=str(RESULTS_DIR / "vsx" /
                                    "clasificacion_w001.csv"))
    parser.add_argument("--lc", default=str(RESULTS_DIR / "vsx" / "lc.parquet"))
    parser.add_argument("--norm", default="min_max",
                        choices=["min_max", "log"])
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    peaks = pd.read_csv(args.clasificacion)
    lightcurves = pd.read_parquet(
        args.lc, filters=[("TIC", "in", sorted(peaks.TIC.unique().tolist()))],
        columns=["TIC", "sector", "Time", "flux", "flux_err"])
    curvas = {}
    for (tic, sector), bloque in lightcurves.groupby(["TIC", "sector"]):
        if len(bloque) < 20:
            continue
        time, flux, _ = clean_lightcurve(bloque.Time.to_numpy(),
                                         bloque.flux.to_numpy(),
                                         bloque.flux_err.to_numpy())
        curvas[(tic, sector)] = (time, flux)

    cubo = np.zeros((len(peaks), 32, 32), dtype=np.float32)
    sin_curva = 0
    for posicion, fila in enumerate(peaks.itertuples()):
        curva = curvas.get((fila.TIC, fila.sector))
        if curva is None or not np.isfinite(fila.per) or fila.per <= 0:
            sin_curva += 1
            continue
        time, flux = curva
        cubo[posicion] = phase_fold_hist2d(time, flux, fila.per,
                                           norm=args.norm)
        if posicion % 2000 == 0:
            print(f"  {posicion}/{len(peaks)}")

    np.savez_compressed(
        args.out, X=cubo[..., np.newaxis],
        **{column: peaks[column].to_numpy() for column in META_COLS})
    print(f"-> {args.out}   X{cubo.shape}  norm={args.norm}  "
          f"{sin_curva} picos sin curva")


if __name__ == "__main__":
    main()
