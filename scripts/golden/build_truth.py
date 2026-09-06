#!/usr/bin/env python
"""Etapa 0: congelar la verdad externa a nivel estrella.

Colapsa `catalogs/golden_sample.csv` (1436 filas, una por TIC y tabla
publicada) a una fila por TIC y le pega el período reportado, quedándose sólo
con los TIC que tienen curva de luz en `catalogs/2_MassiveXTessV8_LC.csv`.

Dos decisiones que quedan grabadas acá y no se rediscuten aguas abajo:

  - La clase se colapsa por prioridad de vocabulario, no por mayoría: entre
    "ECL" y "BE" para la misma estrella gana ECL, porque es la etiqueta
    físicamente más específica. Es el mismo orden que usa
    `scripts/analyse_golden_sample.py`.
  - `period_usable` marca qué estrellas entran al test de período. Quedan
    afuera tres grupos, y el criterio es el mismo en los tres: filas donde
    "acertar el período" no es una pregunta con respuesta, no filas difíciles.

    SLF y NOISY, porque en variabilidad estocástica de baja frecuencia un
    "período" no se puede acertar ni errar: los 24 valores que los papers
    reportan ahí son escalas de tiempo características.

    BE, porque el `period` que la golden les atribuye es `1/fg1` de
    Labadie-Bartz+ 2022, y `fg1` es en el ReadMe de esa tabla "Central
    frequency of g1" — el centro de un GRUPO de frecuencias apiñadas
    (`signals=G`, `Ns` de 2 a 5 grupos por estrella), no una señal coherente.
    El prewhitening lo confirma: en TIC 42889751 s33 cinco de seis componentes
    caen dentro de 1.10-1.33 d, tres de ellas separadas por menos de una
    resolución de Rayleigh. No hay un período que plegar, así que compararlo
    contra el nuestro cuenta como error algo que no es un error. Son 40 de los
    80 TIC usables y 119 de las 316 estrella-sector, o sea que dejarlas adentro
    hacía que casi la mitad del test midiera otra cosa. Quedan marcadas en
    `period_is_group` y se pueden seguir mirando aparte.

    Y las estrellas donde dos referencias no concuerdan (`period_spread`), que
    no pueden arbitrar nada.

`period_spread` es el cociente entre el mayor y el menor período reportado por
referencias distintas para la misma estrella. Cuando pasa de 1.05 los papers no
concuerdan entre ellos y esa estrella no puede arbitrar nada.
"""
import numpy as np
import pandas as pd

from msv.config import CATALOGS_DIR

GOLDEN_SAMPLE = CATALOGS_DIR / "golden_sample.csv"
LIGHT_CURVE_INDEX = CATALOGS_DIR / "2_MassiveXTessV8_LC.csv"
OUTPUT = CATALOGS_DIR / "golden_truth.csv"

VOCABULARY_PRIORITY = ["ECL", "ELL", "BE", "ROT", "PULS", "SLF", "OTHER",
                       "AMBIGUOUS", "NOISY"]
# Clases donde "período" es una cantidad bien definida y por lo tanto
# comparable. SLF es estocástica y NOISY no tiene señal.
CLASSES_WITH_MEANINGFUL_PERIOD = ["ECL", "ELL", "BE", "ROT", "PULS",
                                  "AMBIGUOUS", "OTHER"]
# El "período" publicado es el centro de un grupo de frecuencias, no un
# período: 1/fg1 de Labadie-Bartz+ 2022.
CLASSES_WITH_GROUP_PERIOD = ["BE"]
SPREAD_LIMIT = 1.05


def collapse_class(golden):
    priority = {name: position
                for position, name in enumerate(VOCABULARY_PRIORITY)}
    ranked = golden.copy()
    ranked["priority"] = ranked.class_paper.map(priority).fillna(len(priority))
    ranked = ranked.sort_values(["TIC", "priority"])
    return ranked.drop_duplicates("TIC", keep="first")


def aggregate_periods(golden):
    with_period = golden.dropna(subset=["period"])
    rows = []
    for tic, block in with_period.groupby("TIC"):
        periods = block.period.values
        rows.append({
            "TIC": tic,
            "period_gold": float(np.median(periods)),
            "period_n": len(periods),
            "period_spread": float(periods.max() / periods.min()),
            "period_references": " | ".join(sorted(set(block.reference))),
            "period_methods": " | ".join(sorted(set(block.method.dropna()))),
        })
    return pd.DataFrame(rows)


def main():
    golden = pd.read_csv(GOLDEN_SAMPLE)
    light_curves = pd.read_csv(LIGHT_CURVE_INDEX, low_memory=False)

    sectors = light_curves.groupby("TIC").sector.agg(["size", "nunique"])
    sectors.columns = ["n_products", "n_sectors"]

    truth = collapse_class(golden)[
        ["TIC", "class_paper", "class_raw", "class_is_inferred",
         "reference", "ra_deg", "dec_deg"]
    ].rename(columns={"class_paper": "class_gold",
                      "reference": "class_reference"})

    truth = truth.merge(aggregate_periods(golden), on="TIC", how="left")
    truth = truth.merge(sectors, left_on="TIC", right_index=True, how="left")
    truth["has_light_curve"] = truth.n_sectors.notna()

    truth["period_is_group"] = truth.class_gold.isin(CLASSES_WITH_GROUP_PERIOD)
    truth["period_usable"] = (
        truth.class_gold.isin(CLASSES_WITH_MEANINGFUL_PERIOD)
        & ~truth.period_is_group
        & truth.period_gold.notna()
        & (truth.period_spread <= SPREAD_LIMIT)
    )

    truth = truth.sort_values("TIC")
    truth.to_csv(OUTPUT, index=False)

    print(f"escrito {OUTPUT}  ({len(truth)} TIC)")
    print(f"\ncon curva de luz: {int(truth.has_light_curve.sum())}")
    available = truth[truth.has_light_curve]
    summary = available.groupby("class_gold").agg(
        n_tic=("TIC", "size"),
        con_periodo=("period_gold", "count"),
        usable=("period_usable", "sum"),
        grupo=("period_is_group", "sum"),
        sectores_mediana=("n_sectors", "median"),
    )
    print(summary.to_string())
    print(f"\nperiod_usable con curva: {int(available.period_usable.sum())}")
    disagreeing = truth[truth.period_spread > SPREAD_LIMIT]
    print(f"papers en desacuerdo (spread > {SPREAD_LIMIT}): {len(disagreeing)}"
          f", con curva {int(disagreeing.has_light_curve.sum())}")
    if len(disagreeing):
        print(disagreeing[["TIC", "class_gold", "period_gold", "period_n",
                           "period_spread", "has_light_curve"]]
              .round(4).to_string(index=False))


if __name__ == "__main__":
    main()
