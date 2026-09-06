#!/usr/bin/env python
"""Los tests 1 y 3 de `docs/PLAN_clasificacion_4clases.md`.

    1. Sanidad, por construcción. El % de períodos reportados cuyo fundamental
       tiene SNR >= 4 tiene que pasar de ~49% a ~100%. Si no, el veto está mal
       conectado. No mide calidad: mide que el cable esté puesto.

    3. El test externo que NO depende de ningún período publicado: qué fracción
       de las BE y las SLF de la golden cae en `Irregular`, y qué fracción de
       las PULS y las ECL cae en `Pulsante` y `ECL`. Usa sólo la etiqueta de
       clase del paper, así que no hereda ninguno de los problemas del período
       publicado (§10 del README de golden) y mide justamente la clase nueva.

El test 2 es `score_periods.py` y se corre aparte, porque necesita volver a
sondear el período nuevo contra el publicado.

    PYTHONPATH=src python scripts/golden/validar_4clases.py
"""
import argparse

import pandas as pd

from msv.config import CATALOGS_DIR, RESULTS_DIR

SNR_MIN = 4.0
# Lo que el plan predice para cada clase publicada. BE y SLF son la prueba de
# fuego: son las que no tienen período que reportar.
EXPECTED = {"BE": "Irregular", "SLF": "Irregular", "NOISY": "Irregular",
            "PULS": "Pulsante", "ECL": "ECL", "ELL": "ELL"}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--clase-final",
                        default=str(RESULTS_DIR / "golden" / "clase_final.csv"))
    parser.add_argument("--clase-final-estrella",
                        default=str(RESULTS_DIR / "golden" / "clase_final_estrella.csv"))
    parser.add_argument("--probe-antes",
                        default=str(RESULTS_DIR / "golden" / "probe_peaks.csv"),
                        help="sondas con la columna `reportado` de la cadena "
                             "VIEJA, para el antes del test 1")
    parser.add_argument("--truth", default=str(CATALOGS_DIR / "golden_truth.csv"))
    args = parser.parse_args()

    stars = pd.read_csv(args.clase_final)
    by_star = pd.read_csv(args.clase_final_estrella)
    truth = pd.read_csv(args.truth)

    print("=== TEST 1: sanidad del veto ===")
    before = pd.read_csv(args.probe_antes)
    reported_before = before[before.reportado]
    print(f"cadena vieja (max log_pLPV): "
          f"{(reported_before.snr >= SNR_MIN).mean():6.1%} de "
          f"{len(reported_before)} períodos reportados con fundamental")
    reported = stars[stars.periodo_final.notna()]
    if len(reported):
        print(f"cadena nueva (veto + max SNR): "
              f"{(reported.snr_fundamental >= SNR_MIN).mean():6.1%} de "
              f"{len(reported)} períodos reportados con fundamental")
    print(f"estrella-sector que ya no reportan período: "
          f"{len(stars) - len(reported)}/{len(stars)}")

    print("\n=== TEST 3: la clase contra la etiqueta del paper ===")
    for label, table in [("estrella-sector", stars), ("estrella (TIC)", by_star)]:
        merged = table.merge(truth[["TIC", "class_gold"]], on="TIC", how="inner")
        print(f"\n--- {label}: {len(merged)} filas ---")
        crossed = pd.crosstab(merged.class_gold, merged.clase_final, margins=True)
        print(crossed.to_string())
        print()
        for class_gold, expected in EXPECTED.items():
            block = merged[merged.class_gold == class_gold]
            if not len(block) or expected not in crossed.columns:
                continue
            hit = (block.clase_final == expected).mean()
            print(f"{class_gold:6s} -> {expected:10s} "
                  f"{hit:6.1%}  ({int((block.clase_final == expected).sum())}"
                  f"/{len(block)})")

    print("\n=== períodos reportados por clase publicada (estrella-sector) ===")
    merged = stars.merge(truth[["TIC", "class_gold"]], on="TIC", how="inner")
    print(merged.groupby("class_gold").agg(
        n=("TIC", "size"),
        con_periodo=("periodo_final", "count"),
        fraccion=("periodo_final", lambda values: values.notna().mean()),
        snr_mediano=("snr_fundamental", "median"),
    ).round(3).to_string())


if __name__ == "__main__":
    main()
