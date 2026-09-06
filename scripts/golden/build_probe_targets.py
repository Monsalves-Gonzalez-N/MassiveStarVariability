#!/usr/bin/env python
"""Tabla de sondas para toda la muestra, no solo el caso 2P.

`prewhiten.py` se manejaba con `veredicto_2P.csv`, que es por construcción el
conjunto donde el pipeline ya había fallado. Sondear el fundamental solo ahí no
permite decir nada sobre la muestra completa: condicionar en el fracaso y
después medir es el mismo error que se corrigió en la §7 del README.

Esto emite el mismo schema que el veredicto — TIC, sector, period_gold,
per_nuestro — para TODA estrella-sector con período publicado, de modo que
`prewhiten.py --all --veredicto <esta tabla>` sondee el publicado y el nuestro
en las 316.

    PYTHONPATH=src python scripts/golden/build_probe_targets.py
"""
import argparse

import pandas as pd

from msv.config import CATALOGS_DIR, RESULTS_DIR


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--clasificacion",
                        default=str(RESULTS_DIR / "golden" / "clasificacion_collapsed.csv"))
    parser.add_argument("--truth", default=str(CATALOGS_DIR / "golden_truth.csv"))
    parser.add_argument("--out",
                        default=str(RESULTS_DIR / "golden" / "probe_targets.csv"))
    args = parser.parse_args()

    peaks = pd.read_csv(args.clasificacion)
    truth = pd.read_csv(args.truth)
    truth = truth[truth.period_gold.notna()]

    reported = peaks[peaks.reportado].merge(
        truth[["TIC", "class_gold", "period_gold", "period_is_group",
               "period_usable"]],
        on="TIC", how="inner")
    # Con el veto puesto, el periodo que se reporta es `per_reportado`: el
    # candidato ya adjudicado entre P y 2P. Sin veto esa columna no existe y el
    # reportado es el candidato tal cual.
    if "per_reportado" in reported:
        reported = reported.assign(per=reported.per_reportado)

    targets = reported.rename(columns={"per": "per_nuestro",
                                       "clase": "clase_nuestra"})[
        ["TIC", "sector", "class_gold", "period_gold", "per_nuestro",
         "clase_nuestra", "log_pLPV", "nivel", "period_is_group",
         "period_usable"]
    ].copy()
    targets["multi"] = False
    targets = targets.sort_values(["TIC", "sector"])
    targets.to_csv(args.out, index=False)

    print(f"escrito {args.out}  ({len(targets)} estrella-sector, "
          f"{targets.TIC.nunique()} TIC)")
    print(targets.groupby("class_gold").size().to_string())


if __name__ == "__main__":
    main()
