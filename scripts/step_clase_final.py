#!/usr/bin/env python
"""Paso 4/4: la clase final, cuatro clases y nada más.

    ECL          eclipsante
    ELL          elipsoidal
    Pulsante     multiperiódica coherente
    Irregular    sin período: grupos de frecuencias, ruido rojo, Be

Junta las dos preguntas que el resto del pipeline contesta por separado — el
triage de estructura de `step_descriptores.py` (¿hay período? ¿cuántos?) y la
clase que la CNN le da al fold del candidato reportado (¿qué forma tiene?) —
con el veto del fundamental ya aplicado en `step_clasificar_una_red.py`:

    sin candidato que pase el veto                    -> Irregular (sin período)
    estructura irregular (un grupo apiñado)           -> Irregular
    la CNN dice E sobre el candidato reportado        -> ECL
    la CNN dice ELL                                   -> ELL
    la CNN dice Pulsating                             -> Pulsante
    estructura multiperiódica, diga la red lo que diga -> Pulsante
    resto (uniperiódica y la red dice LPV/Rndm)        -> Irregular

Las dos primeras reglas van antes que la red a propósito: una frecuencia que no
existe no tiene forma que clasificar. ECL/ELL tienen prioridad sobre Pulsante
por el mismo criterio de vocabulario que usa `build_truth.py` — es la etiqueta
físicamente más específica, y una eclipsante con pulsaciones sigue siendo una
eclipsante.

La penúltima es la regla del plan tal cual está escrita: "triage =
multiperiódica y la CNN no dice E/ELL -> Pulsante". Importa que sea así y no
"la CNN dice Pulsating": sobre el candidato que el veto elige —el de más SNR,
no el que mejor se ve— la red dice `LPV` en 406 de 810 estrella-sector, y
exigirle la forma manda a `Irregular` casi todo lo que tiene señal. La variante
estricta está en `--cnn-estricta`.

La última regla NO está en el plan, que sólo define el caso multiperiódico. Es
la lectura literal de las definiciones para el uniperiódico: `Pulsante` es
"multiperiódica coherente", así que una uniperiódica cuyo fold la red no lee
como eclipse ni elipsoidal no tiene ninguna de las cuatro clases con período.
Queda marcada en `regla` para poder contarla aparte.

Las Be y la variabilidad estocástica de baja frecuencia caen en `Irregular` por
construcción y no reportan período. Es una decisión de alcance: en variabilidad
estocástica un período no es una cantidad que se pueda acertar o errar.

    PYTHONPATH=src python scripts/step_clase_final.py \
        --clasificacion results/golden/clasificacion_veto.csv \
        --descriptores results/golden/descriptores.csv \
        --out results/golden/clase_final.csv
"""
import argparse

import numpy as np
import pandas as pd

from msv.config import RESULTS_DIR
from msv.structure import IRREGULAR, MULTIPERIODICA, SIN_SENAL

ECL = "ECL"
ELL = "ELL"
PULSANTE = "Pulsante"
IRREGULAR_FINAL = "Irregular"
CLASS_ORDER = [ECL, ELL, PULSANTE, IRREGULAR_FINAL]
# Prioridad de vocabulario para colapsar los sectores de una estrella: la
# etiqueta mas especifica gana, igual que en build_truth.py.
VOCABULARY_PRIORITY = {name: position for position, name in enumerate(CLASS_ORDER)}


def classify(row, structure, cnn_estricta=False):
    """(clase_final, regla) de una estrella-sector."""
    if not row.reportado:
        return IRREGULAR_FINAL, "sin_veto"
    if structure == SIN_SENAL:
        return IRREGULAR_FINAL, "sin_senal"
    if structure == IRREGULAR:
        return IRREGULAR_FINAL, "grupo_apinado"
    if row.clase == "E":
        return ECL, "cnn_E"
    if row.clase == "ELL":
        return ELL, "cnn_ELL"
    if row.clase == "Pulsating":
        return PULSANTE, "cnn_Pulsating"
    # Acá la red dijo LPV o Rndm sobre un fold cuya frecuencia SÍ existe. El
    # plan manda `Pulsante` si la estructura es multiperiódica — la regla es
    # "la CNN no dice E/ELL", no "la CNN dice Pulsating" — porque lo que
    # sostiene la clase es el conteo de frecuencias, no la forma del fold. Con
    # `--cnn-estricta` se exige además que la red diga Pulsating.
    if structure == MULTIPERIODICA and not cnn_estricta:
        return PULSANTE, f"multiperiodica_{row.clase}"
    return IRREGULAR_FINAL, f"cnn_{row.clase}"


def collapse_by_star(sectors):
    """Una fila por TIC: gana la clase mas especifica entre sus sectores."""
    ranked = sectors.copy()
    ranked["priority"] = ranked.clase_final.map(VOCABULARY_PRIORITY)
    ranked = ranked.sort_values(["TIC", "priority", "snr_fundamental"],
                                ascending=[True, True, False])
    return ranked.drop_duplicates("TIC", keep="first").drop(columns="priority")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--clasificacion",
                        default=str(RESULTS_DIR / "golden" / "clasificacion_veto.csv"))
    parser.add_argument("--descriptores",
                        default=str(RESULTS_DIR / "golden" / "descriptores.csv"))
    parser.add_argument("--out",
                        default=str(RESULTS_DIR / "golden" / "clase_final.csv"))
    parser.add_argument("--estructura-col", default="estructura",
                        help="`estructura` es la regla literal del plan (todas "
                             "las componentes en un grupo); "
                             "`estructura_apinado` basta con que algun grupo "
                             "tenga 3 componentes no resueltas")
    parser.add_argument("--cnn-estricta", action="store_true",
                        help="exigir que la red diga Pulsating para llamar "
                             "Pulsante; sin esto basta con que la estructura "
                             "sea multiperiódica, que es la regla del plan")
    parser.add_argument("--out-estrella", default=None,
                        help="csv opcional colapsado a una fila por TIC")
    args = parser.parse_args()

    peaks = pd.read_csv(args.clasificacion)
    descriptors = pd.read_csv(args.descriptores)

    if "reportado" not in peaks or "per_reportado" not in peaks:
        raise SystemExit(f"{args.clasificacion} no tiene las columnas del veto; "
                         f"correr step_clasificar_una_red.py con --probe")

    reported = peaks[peaks.reportado][
        ["TIC", "sector", "source", "per", "per_reportado", "armonico", "clase",
         "prob", "log_pLPV", "snr_fundamental", "a2_a1", "snr_half", "ciclos",
         "irregular", "nivel"]]
    # Una estrella-sector sin ningun candidato que pase el veto no aparece en
    # `reported`, y esa ausencia ES la respuesta: no reporta periodo.
    stars = descriptors.merge(reported, on=["TIC", "sector"], how="left")
    stars["reportado"] = stars.per_reportado.notna()

    if args.estructura_col not in stars:
        raise SystemExit(f"{args.descriptores} no tiene la columna "
                         f"{args.estructura_col}")
    stars["estructura_usada"] = stars[args.estructura_col]
    verdicts = [classify(row, structure, args.cnn_estricta)
                for row, structure in zip(stars.itertuples(index=False),
                                          stars[args.estructura_col])]
    stars["clase_final"] = [verdict for verdict, _ in verdicts]
    stars["regla"] = [rule for _, rule in verdicts]
    stars["periodo_final"] = np.where(stars.clase_final == IRREGULAR_FINAL,
                                      np.nan, stars.per_reportado)

    stars.to_csv(args.out, index=False)
    print(f"escrito {args.out}  ({len(stars)} estrella-sector, "
          f"{stars.TIC.nunique()} TIC)")

    if args.out_estrella:
        by_star = collapse_by_star(stars)
        by_star.to_csv(args.out_estrella, index=False)
        print(f"escrito {args.out_estrella}  ({len(by_star)} TIC)")

    print("\n=== clase final, por estrella-sector ===")
    counts = stars.clase_final.value_counts()
    for name in CLASS_ORDER:
        number = int(counts.get(name, 0))
        print(f"{name:12s} {number:5d}  {number / len(stars):6.1%}")

    print("\n=== por qué regla ===")
    print(stars.regla.value_counts().to_string())

    print(f"\n=== {args.estructura_col} x clase final ===")
    print(pd.crosstab(stars[args.estructura_col], stars.clase_final,
                      margins=True).to_string())

    with_period = stars[stars.periodo_final.notna()]
    print(f"\nperíodos reportados: {len(with_period)}")
    if len(with_period):
        print(f"  con fundamental SNR >= 4: "
              f"{(with_period.snr_fundamental >= 4).mean():.1%}")
        print(f"  adjudicados a 2P: "
              f"{int((with_period.armonico == '2P').sum())}")


if __name__ == "__main__":
    main()
