#!/usr/bin/env python
"""Sondear el fundamental de TODOS los candidatos, no solo del reportado.

La cadena de picos pregunta si hay potencia EN una frecuencia (el FAP de la LS,
el umbral de Bartlett del ACF). Esta pregunta es otra: si la potencia está en el
FUNDAMENTAL de esa frecuencia o en su primer armónico. Un candidato en 2P tiene
correlación real a ese lag — la función que se repite cada P también se repite
cada 2P — así que ningún umbral de ruido lo puede rechazar; lo que lo delata es
que su fundamental esté vacío y su armónico se lleve toda la amplitud.

Por cada estrella-sector se extraen las componentes una vez y después se sondea
cada período candidato sobre la curva con las OTRAS variaciones removidas. Sale
una fila por pico con `snr` (del fundamental) y `a2_a1` (cuánto domina el primer
armónico), listas para cruzar contra la clase de la CNN.

    PYTHONPATH=src python scripts/probe_peaks.py \
        --clasificacion results/golden/clasificacion_collapsed.csv \
        --lc results/golden/lc.parquet \
        --out results/golden/probe_peaks.csv
"""
import argparse

import numpy as np
import pandas as pd

from msv.cleaning import clean_lightcurve
from msv.config import RESULTS_DIR
from msv.prewhiten import (DEFAULT_HARMONICS, DEFAULT_MAX_COMPONENTS,
                           DEFAULT_SNR_MIN, extract_components, probe,
                           to_relative)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--clasificacion",
                        default=str(RESULTS_DIR / "golden" /
                                    "clasificacion_collapsed.csv"))
    parser.add_argument("--lc",
                        default=str(RESULTS_DIR / "golden" / "lc.parquet"))
    parser.add_argument("--out",
                        default=str(RESULTS_DIR / "golden" / "probe_peaks.csv"))
    parser.add_argument("--only-reported-stars", action="store_true",
                        help="solo las estrella-sector que emiten un reporte")
    parser.add_argument("--filtro-cnn", action="store_true",
                        help="sondear solo los picos que la CNN no llamo Rndm")
    parser.add_argument("--sin-subarmonico", action="store_true",
                        help="una sola pasada: no sondear 0.5/per")
    parser.add_argument("--n-max", type=int, default=DEFAULT_MAX_COMPONENTS)
    parser.add_argument("--n-harmonics", type=int, default=DEFAULT_HARMONICS)
    parser.add_argument("--snr-min", type=float, default=DEFAULT_SNR_MIN)
    args = parser.parse_args()

    peaks = pd.read_csv(args.clasificacion)
    if args.only_reported_stars:
        keys = set(map(tuple, peaks[peaks.reportado][["TIC", "sector"]].values))
        peaks = peaks[[tuple(item) in keys
                       for item in peaks[["TIC", "sector"]].values]]

    if args.filtro_cnn:
        peaks = peaks[peaks.clase != "Rndm"]

    tics = sorted(peaks.TIC.unique().tolist())
    lightcurves = pd.read_parquet(
        args.lc, filters=[("TIC", "in", tics)],
        columns=["TIC", "sector", "Time", "flux", "flux_err"])

    rows = []
    groups = list(peaks.groupby(["TIC", "sector"]))
    for position, ((tic, sector), block) in enumerate(groups):
        curve = lightcurves[(lightcurves.TIC == tic)
                            & (lightcurves.sector == sector)]
        if len(curve) < 20:
            continue
        time, flux, _ = clean_lightcurve(curve.Time.to_numpy(),
                                         curve.flux.to_numpy(),
                                         curve.flux_err.to_numpy())
        relative_flux, _ = to_relative(flux)
        components, _ = extract_components(
            time, relative_flux, n_max=args.n_max,
            n_harmonics=args.n_harmonics, snr_min=args.snr_min)

        for index, peak in block.iterrows():
            if not np.isfinite(peak.per) or peak.per <= 0:
                continue
            result = probe(time, relative_flux, components, 1.0 / peak.per,
                           n_harmonics=args.n_harmonics)
            harmonics = np.asarray(result["amplitude_harmonics"], float)
            fundamental = result["amplitude"]
            # El subarmónico: si el período verdadero fuera el DOBLE del
            # candidato, la señal vive en f/2 y nuestro pico es su primer
            # armónico. `harmonics` solo mira hacia arriba en frecuencia, así
            # que esta dirección hay que sondearla aparte.
            if args.sin_subarmonico:
                half = {"snr": np.nan, "snr_model": np.nan,
                        "amplitude": np.nan}
            else:
                half = probe(time, relative_flux, components, 0.5 / peak.per,
                             n_harmonics=args.n_harmonics)
            rows.append({
                "TIC": tic, "sector": sector, "index": index,
                "per": peak.per, "source": peak.source, "clase": peak.clase,
                "prob": peak.prob, "log_pLPV": peak.log_pLPV,
                "reportado": bool(peak.reportado), "nivel": peak.nivel,
                "power": peak.power, "prominence": peak.prominence,
                "amplitude_ppt": fundamental,
                "snr": result["snr"],
                "snr_breger": result["snr_breger"],
                "snr_model": result["snr_model"],
                "amplitude_model_ppt": result["amplitude_model"],
                "a2_a1": (float(harmonics[0] / fundamental)
                          if len(harmonics) and fundamental > 0 else np.nan),
                "cycles": result["cycles"],
                "snr_half": half["snr"],
                "snr_half_model": half["snr_model"],
                "amplitude_half_ppt": half["amplitude"],
            })
        if position % 25 == 0:
            print(f"  {position}/{len(groups)}  TIC {tic} s{sector}")

    table = pd.DataFrame(rows)
    table.to_csv(args.out, index=False)
    print(f"escrito {args.out}  ({len(table)} picos, "
          f"{table.groupby(['TIC', 'sector']).ngroups} estrella-sector)")

    table["fundamental_vacio"] = table.snr < DEFAULT_SNR_MIN
    print("\n=== picos con el fundamental por debajo de SNR 4, por clase ===")
    summary = table.groupby("clase").agg(
        n=("snr", "size"),
        sin_fundamental=("fundamental_vacio", "mean"),
        snr_mediana=("snr", "median"),
        armonico_domina=("a2_a1", lambda values: (values > 2).mean()),
    )
    print(summary.round(3).to_string())
    print("\n=== lo mismo, solo los picos REPORTADOS ===")
    reported = table[table.reportado]
    print(reported.groupby("clase").agg(
        n=("snr", "size"),
        sin_fundamental=("fundamental_vacio", "mean"),
        snr_mediana=("snr", "median"),
    ).round(3).to_string())
    print("\n=== en qué falla el que no tiene fundamental ===")
    failing = table[table.fundamental_vacio].copy()
    doubled = failing.a2_a1 > 2
    halved = (failing.snr_half >= DEFAULT_SNR_MIN) & ~doubled
    empty = ~doubled & ~halved
    print(f"picos sin fundamental: {len(failing)}")
    print(f"  reportamos el DOBLE del real (armonico domina)   "
          f"{int(doubled.sum()):5d}  {doubled.mean():.1%}")
    print(f"  reportamos la MITAD del real (hay senal en f/2)  "
          f"{int(halved.sum()):5d}  {halved.mean():.1%}")
    print(f"  no hay senal en ninguno de los tres             "
          f"{int(empty.sum()):5d}  {empty.mean():.1%}")
    failing["modo"] = np.where(doubled, "doble", np.where(halved, "mitad", "vacio"))
    print()
    print(pd.crosstab(failing.clase, failing.modo, margins=True).to_string())

    print("\n=== por origen del pico ===")
    print(table.groupby("source").agg(
        n=("snr", "size"),
        sin_fundamental=("fundamental_vacio", "mean"),
        snr_mediana=("snr", "median"),
    ).round(3).to_string())


if __name__ == "__main__":
    main()
