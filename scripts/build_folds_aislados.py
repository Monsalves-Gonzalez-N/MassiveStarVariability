#!/usr/bin/env python
"""Paso 5 del plan: rehacer el fold quitando las OTRAS variaciones.

La CNN se entrenó con estrellas cuya curva ES una sola variación. En una
multiperiódica el fold en cualquier período lleva las otras encima como
dispersión, así que la imagen que recibe está fuera de la distribución con la
que se entrenó. `msv.prewhiten.isolate` la devuelve adentro: resta las demás
componentes y conserva las conmensurables, porque el primer armónico de una
elipsoidal es lo que le da sus dos máximos y quitarlo borraría la forma que el
fold tiene que mostrar.

Aislar NO es una mejora universal — sube la mediana de `log_pLPV` de 0.27 a
1.03 en las estrella-sector multiperiódicas y la BAJA de 1.07 a 0.82 sobre el
conjunto completo. Ayuda donde hay varias variaciones y estorba donde no. Por
eso se aísla **sólo donde el triage de `step_descriptores.py` dice
`multiperiodica`**; el resto de las filas se copian tal cual del parquet de
entrada.

Sale el mismo schema de picos que `run_peaks.py`/`collapse_comb.py`, así que
step_cnn_export -> step_cnn -> step_clasificar_una_red corre sin tocar nada.

    PYTHONPATH=src python scripts/build_folds_aislados.py \
        --peaks results/golden/peaks_collapsed.parquet \
        --descriptores results/golden/descriptores.csv \
        --lc results/golden/lc.parquet \
        --out results/golden/peaks_aislados.parquet
"""
import argparse

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from msv.cleaning import clean_lightcurve
from msv.config import RESULTS_DIR
from msv.features import HIST_SIZE, phase_fold_hist2d
from msv.prewhiten import (DEFAULT_HARMONICS, DEFAULT_MAX_COMPONENTS,
                           DEFAULT_SNR_MIN, extract_components, from_relative,
                           isolate, to_relative)
from msv.structure import MULTIPERIODICA

SCHEMA_COLS = ["TIC", "sector", "source", "per", "power", "power_effective",
               "prominence", "width", "amplitude"]
MIN_POINTS = 20


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--peaks",
                        default=str(RESULTS_DIR / "golden" / "peaks_collapsed.parquet"))
    parser.add_argument("--descriptores",
                        default=str(RESULTS_DIR / "golden" / "descriptores.csv"))
    parser.add_argument("--lc", default=str(RESULTS_DIR / "golden" / "lc.parquet"))
    parser.add_argument("--out",
                        default=str(RESULTS_DIR / "golden" / "peaks_aislados.parquet"))
    parser.add_argument("--estructura", default=MULTIPERIODICA,
                        help="qué estructuras se aíslan; `todas` para no gatear")
    parser.add_argument("--estructura-col", default="estructura",
                        help="columna de triage a usar (ver step_clase_final.py)")
    parser.add_argument("--n-max", type=int, default=DEFAULT_MAX_COMPONENTS)
    parser.add_argument("--n-harmonics", type=int, default=DEFAULT_HARMONICS)
    parser.add_argument("--snr-min", type=float, default=DEFAULT_SNR_MIN)
    args = parser.parse_args()

    peaks = pd.read_parquet(args.peaks).reset_index(drop=True)
    descriptors = pd.read_csv(args.descriptores)
    if args.estructura == "todas":
        selected = descriptors
    else:
        selected = descriptors[descriptors[args.estructura_col] == args.estructura]
    targets = set(map(tuple, selected[["TIC", "sector"]].values))
    print(f"{len(peaks)} picos; {len(targets)} estrella-sector a aislar "
          f"de {len(descriptors)}")

    lightcurves = pd.read_parquet(
        args.lc, columns=["TIC", "sector", "Time", "flux", "flux_err"],
        filters=[("TIC", "in", sorted({tic for tic, _ in targets}))]
        if targets else None)

    histograms = list(peaks.hist2d.values)
    isolated_rows = 0
    for position, ((tic, sector), block) in enumerate(
            peaks.groupby(["TIC", "sector"], sort=True)):
        if (tic, sector) not in targets:
            continue
        curve = lightcurves[(lightcurves.TIC == tic)
                            & (lightcurves.sector == sector)]
        if len(curve) < MIN_POINTS:
            continue
        time, flux, _ = clean_lightcurve(curve.Time.to_numpy(),
                                         curve.flux.to_numpy(),
                                         curve.flux_err.to_numpy())
        relative_flux, mean_flux = to_relative(flux)
        components, _ = extract_components(
            time, relative_flux, n_max=args.n_max,
            n_harmonics=args.n_harmonics, snr_min=args.snr_min)
        if not components:
            continue
        for index, peak in block.iterrows():
            if not np.isfinite(peak.per) or peak.per <= 0:
                continue
            cleaned = isolate(relative_flux, components, 1.0 / peak.per)
            in_flux_units = from_relative(cleaned, mean_flux)
            histograms[index] = phase_fold_hist2d(time, in_flux_units,
                                                 peak.per).ravel()
            isolated_rows += 1
        if position % 50 == 0:
            print(f"  TIC {tic} s{sector}: {len(components)} componentes, "
                  f"{len(block)} folds aislados")

    flat = np.concatenate([np.asarray(hist, dtype=np.float32)
                           for hist in histograms])
    table = pa.table({
        column: pa.array(peaks[column].to_numpy()) for column in SCHEMA_COLS
    })
    table = table.append_column(
        "hist2d", pa.FixedSizeListArray.from_arrays(
            pa.array(flat, type=pa.float32()), HIST_SIZE))
    pq.write_table(table, args.out, compression="snappy")
    print(f"\n-> {args.out}   {len(peaks)} picos, {isolated_rows} con el fold "
          f"aislado ({isolated_rows / len(peaks):.1%})")


if __name__ == "__main__":
    main()
