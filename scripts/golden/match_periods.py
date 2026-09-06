#!/usr/bin/env python
"""Etapa 2 (base): comparar nuestros períodos contra los de la golden sample.

Separa dos preguntas que fallan por motivos distintos y que sumadas en un solo
porcentaje no se pueden interpretar:

  Q1  ¿el período del paper está ENTRE nuestros picos?      -> el periodograma
  Q2  ¿la regla de selección lo ELIGE como reportado?       -> la regla

El armónico se etiqueta con tolerancia RELATIVA del 5% sobre el cociente, igual
que `analysis_ell/_common.load_peak_truth`. La diferencia importa: con
tolerancia absoluta de 0.05 sobre un cociente de 2 el test es del 2.5% y pierde
19 de los 63 casos de alias.

    PYTHONPATH=src python scripts/golden/match_periods.py
"""
import argparse

import numpy as np
import pandas as pd

from msv.config import CATALOGS_DIR, RESULTS_DIR

HARMONICS = {"1:1": 1.0, "2P": 2.0, "P/2": 0.5, "3P": 3.0, "P/3": 1.0 / 3.0}
TOLERANCE = 0.05


def harmonic_tag(ratio):
    for name, target in HARMONICS.items():
        if np.abs(ratio / target - 1.0) < TOLERANCE:
            return name
    return "otro"


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--clasificacion",
                        default=str(RESULTS_DIR / "golden" / "clasificacion.csv"))
    parser.add_argument("--truth", default=str(CATALOGS_DIR / "golden_truth.csv"))
    parser.add_argument("--out-dir", default=str(RESULTS_DIR / "golden"))
    args = parser.parse_args()

    peaks = pd.read_csv(args.clasificacion)
    truth = pd.read_csv(args.truth)
    truth = truth[truth.period_usable]

    peaks = peaks.merge(
        truth[["TIC", "class_gold", "period_gold", "period_n",
               "period_references"]],
        on="TIC", how="inner")
    peaks["ratio"] = peaks.per / peaks.period_gold
    peaks["tag"] = [harmonic_tag(value) for value in peaks.ratio]
    peaks["ciclos_gold"] = peaks.baseline / peaks.period_gold

    # --- Q1: recall del período publicado entre todos los picos de la estrella
    rows = []
    for (tic, sector), block in peaks.groupby(["TIC", "sector"]):
        found = block.tag.value_counts()
        rows.append({
            "TIC": tic,
            "sector": sector,
            "class_gold": block.class_gold.iloc[0],
            "period_gold": block.period_gold.iloc[0],
            "n_peaks": len(block),
            "ciclos_gold": block.ciclos_gold.iloc[0],
            "found_1a1": bool(found.get("1:1", 0)),
            "found_2P": bool(found.get("2P", 0)),
            "found_P2": bool(found.get("P/2", 0)),
        })
    recall = pd.DataFrame(rows)

    reported = peaks[peaks.reportado].copy()

    recall.to_csv(f"{args.out_dir}/match_recall.csv", index=False)
    reported.to_csv(f"{args.out_dir}/match_reported.csv", index=False)
    peaks.to_csv(f"{args.out_dir}/match_peaks.csv", index=False)

    print(f"{peaks.TIC.nunique()} TIC con período usable, "
          f"{len(recall)} estrella-sector, {len(peaks)} picos")

    print("\n=== Q1: el período del paper está entre nuestros picos ===")
    table = recall.groupby("class_gold").agg(
        estrella_sector=("TIC", "size"),
        recall_1a1=("found_1a1", "mean"),
        tambien_2P=("found_2P", "mean"),
        tambien_P2=("found_P2", "mean"),
    )
    print(table.round(3).to_string())
    print(f"global: {recall.found_1a1.mean():.3f}")

    print("\n=== Q2: qué elige la regla (pico reportado) ===")
    print(pd.crosstab(reported.class_gold, reported.tag,
                      margins=True).to_string())
    print("\npor nivel de confianza:")
    print(pd.crosstab(reported.nivel, reported.tag, margins=True).to_string())

    print("\n=== Q2 condicionado a que Q1 encontró el período ===")
    key = list(zip(reported.TIC, reported.sector))
    found_map = recall.set_index(["TIC", "sector"]).found_1a1
    reported["q1_ok"] = [bool(found_map.get(item, False)) for item in key]
    recoverable = reported[reported.q1_ok]
    hit = (recoverable.tag == "1:1").mean()
    print(f"de {len(recoverable)} estrella-sector donde el período ERA "
          f"alcanzable, la regla lo eligió en {hit:.1%}")
    print(recoverable.tag.value_counts().to_string())


if __name__ == "__main__":
    main()
