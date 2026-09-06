#!/usr/bin/env python
"""Paso 1/4 de la clasificación en 4 clases: descriptores de estructura.

Corre el prewhitening iterativo (`msv.prewhiten.extract_components`) sobre cada
estrella-sector y escribe SOLO los descriptores: cuántas componentes coherentes
hay, en cuántos grupos resueltos caen, y qué fracción de la varianza explican.
`scripts/golden/prewhiten.py` ya calcula lo mismo, pero está orientado a
revisión visual — pliega, arma el parquet de picos y guarda las curvas — y eso
no se puede correr sobre toda la muestra.

De ahí sale la columna `estructura` (`msv.structure.triage`), que es la
pregunta que la CNN no puede contestar:

    sin_senal        ninguna componente llega a SNR 4
    irregular        todo en un grupo apiñado: una joroba ancha, no modos
    multiperiodica   >= 2 grupos resueltos
    uniperiodica     un grupo con 1-2 componentes

Salidas: `descriptores.csv` (una fila por estrella-sector) y, con
`--componentes`, una fila por componente extraída con su frecuencia, amplitud,
SNR y grupo.

    PYTHONPATH=src python scripts/step_descriptores.py \
        --lc results/golden/lc.parquet \
        --out results/golden/descriptores.csv \
        --componentes results/golden/descriptores_componentes.csv
"""
import argparse

import numpy as np
import pandas as pd

from msv.cleaning import clean_lightcurve
from msv.config import RESULTS_DIR
from msv.prewhiten import (DEFAULT_HARMONICS, DEFAULT_MAX_COMPONENTS,
                           DEFAULT_SNR_MIN, extract_components, to_relative)
from msv.structure import describe_structure, triage

DESCRIPTOR_COLS = ["TIC", "sector", "estructura", "estructura_apinado",
                   "n_components", "n_clusters",
                   "max_cluster_size", "cluster_sizes", "coherent_fraction",
                   "separacion_mediana_rayleigh", "separacion_minima_rayleigh",
                   "dominante", "frecuencia_dominante", "periodo_dominante",
                   "amplitud_dominante_ppt", "snr_dominante", "n_points",
                   "baseline", "rayleigh_cpd", "std_ppt", "residual_ppt"]
MIN_POINTS = 20


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lc", default=str(RESULTS_DIR / "golden" / "lc.parquet"))
    parser.add_argument("--out",
                        default=str(RESULTS_DIR / "golden" / "descriptores.csv"))
    parser.add_argument("--componentes", default=None,
                        help="csv opcional con una fila por componente extraída")
    parser.add_argument("--tic", type=int, nargs="+", default=None)
    parser.add_argument("--n-max", type=int, default=DEFAULT_MAX_COMPONENTS)
    parser.add_argument("--n-harmonics", type=int, default=DEFAULT_HARMONICS)
    parser.add_argument("--snr-min", type=float, default=DEFAULT_SNR_MIN)
    args = parser.parse_args()

    filters = [("TIC", "in", args.tic)] if args.tic else None
    lightcurves = pd.read_parquet(
        args.lc, filters=filters,
        columns=["TIC", "sector", "Time", "flux", "flux_err"])

    rows = []
    component_rows = []
    groups = list(lightcurves.groupby(["TIC", "sector"], sort=True))
    print(f"{len(groups)} estrella-sector, "
          f"{lightcurves.TIC.nunique()} TIC")
    for position, ((tic, sector), block) in enumerate(groups):
        if len(block) < MIN_POINTS:
            continue
        time, flux, _ = clean_lightcurve(block.Time.to_numpy(),
                                         block.flux.to_numpy(),
                                         block.flux_err.to_numpy())
        if len(time) < MIN_POINTS:
            continue
        relative_flux, _ = to_relative(flux)
        components, residual = extract_components(
            time, relative_flux, n_max=args.n_max,
            n_harmonics=args.n_harmonics, snr_min=args.snr_min)

        baseline = float(time.max() - time.min())
        descriptors = describe_structure(components, relative_flux, residual,
                                         baseline)
        labels = descriptors.pop("labels")
        dominant = descriptors["dominante"]
        row = {"TIC": int(tic), "sector": int(sector),
               "estructura": triage(descriptors),
               "estructura_apinado": triage(descriptors, solo_un_grupo=False),
               "n_points": len(time)}
        row.update(descriptors)
        if components and np.isfinite(dominant):
            leader = components[int(dominant) - 1]
            row["periodo_dominante"] = leader["period"]
            row["amplitud_dominante_ppt"] = leader["amplitude"]
            row["snr_dominante"] = leader["snr"]
        else:
            row["periodo_dominante"] = np.nan
            row["amplitud_dominante_ppt"] = np.nan
            row["snr_dominante"] = np.nan
        rows.append(row)

        if args.componentes:
            for component, label in zip(components, labels):
                component_rows.append({
                    "TIC": int(tic), "sector": int(sector),
                    "index": component["index"],
                    "frequency": component["frequency"],
                    "period": component["period"],
                    "amplitude_ppt": component["amplitude"],
                    "snr": component["snr"],
                    "cycles": component["cycles"],
                    "cluster": label,
                    "harmonic_amplitudes": " ".join(
                        f"{value:.3f}"
                        for value in component["amplitude_harmonics"]),
                })

        if position % 50 == 0:
            print(f"  {position}/{len(groups)}  TIC {tic} s{sector}: "
                  f"{len(components)} componentes, {row['estructura']}")

    table = pd.DataFrame(rows)[DESCRIPTOR_COLS]
    table.to_csv(args.out, index=False)
    print(f"\nescrito {args.out}  ({len(table)} estrella-sector)")

    if args.componentes:
        components_table = pd.DataFrame(component_rows)
        components_table.to_csv(args.componentes, index=False)
        print(f"escrito {args.componentes}  ({len(components_table)} componentes)")

    print("\n=== estructura ===")
    print(table.estructura.value_counts().to_string())
    print("\n=== estructura con la regla de apinamiento (algun grupo >= 3) ===")
    print(table.estructura_apinado.value_counts().to_string())
    print("\n=== descriptores por estructura ===")
    print(table.groupby("estructura").agg(
        n=("TIC", "size"),
        componentes=("n_components", "median"),
        grupos=("n_clusters", "median"),
        coherente=("coherent_fraction", "median"),
        separacion=("separacion_mediana_rayleigh", "median"),
    ).round(3).to_string())


if __name__ == "__main__":
    main()
