#!/usr/bin/env python
"""Cuantas de las 27 features hacen falta de verdad.

Se ordenan por importancia del boosting y se evaluan los conjuntos anidados
top-k. Las dos metricas se miden a la vez sobre la MISMA corrida, porque en
esta sesion ya se vio varias veces que se mueven en direcciones opuestas: lo
que sube el acierto del periodo suele bajar la abstencion.

    PYTHONPATH=src python scripts/vsx/reducir_features.py
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
from arbol_abstencion import scores_por_semilla
from arbol_features import (CICLOS, CNN, PERIODOGRAMA, SNR_1PASADA, ajustar,
                            cargar, contexto, familia, pesos_balanceados)

from msv.config import RESULTS_DIR

COLUMNAS = PERIODOGRAMA + CICLOS + CNN + SNR_1PASADA


def metricas(peaks, scores, umbral):
    mejores = peaks.assign(_s=scores).sort_values(
        "_s", ascending=False).groupby("TIC").head(1)
    reporta = mejores._s >= umbral
    es_ok = mejores.verdad == "ok"
    return {
        "acierto top-1": 100 * (mejores.tag == "1:1")[es_ok].mean(),
        "cobertura ok": 100 * (reporta & es_ok).sum() / es_ok.sum(),
        "acierto reportadas": (100 * ((mejores.tag == "1:1") & reporta
                                      & es_ok).sum()
                               / max(int((reporta & es_ok).sum()), 1)),
        "abstiene unconstrained": (100 * (~reporta & ~es_ok).sum()
                                   / max(int((~es_ok).sum()), 1)),
    }


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--probe",
                        default=str(RESULTS_DIR / "vsx" /
                                    "probe_peaks_w001_1pass.csv"))
    parser.add_argument("--repeticiones", type=int, default=5)
    parser.add_argument("--umbral", type=float, default=0.75)
    args = parser.parse_args()

    peaks = contexto(cargar(args.probe, verdicts=("ok", "unconstrained")))
    peaks["verdad"] = peaks.vis_verdict
    peaks["familia"] = familia(peaks)
    entrenables = peaks[peaks.verdad == "ok"]
    fuera = (peaks.verdad != "ok").values

    X = entrenables[COLUMNAS].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    modelo = GradientBoostingClassifier(n_estimators=200, max_depth=3,
                                        random_state=0)
    ajustar(modelo, X.values, entrenables.es_vsx.values.astype(int),
            pesos_balanceados(entrenables))
    orden = pd.Series(modelo.feature_importances_,
                      index=COLUMNAS).sort_values(ascending=False)
    print("importancia (para ordenar los anidados):")
    print(orden.round(3).to_string())

    filas = []
    for k in [3, 5, 6, 8, 10, 12, 15, 20, 27]:
        columnas = list(orden.index[:k])
        valores = [metricas(peaks,
                            scores_por_semilla(peaks, entrenables, fuera,
                                               semilla, columnas),
                            args.umbral)
                   for semilla in range(args.repeticiones)]
        fila = {"k": k}
        for nombre in valores[0]:
            serie = [valor[nombre] for valor in valores]
            fila[nombre] = f"{np.mean(serie):.1f} ± {np.std(serie):.1f}"
        filas.append(fila)
    print(f"\nconjuntos anidados top-k, umbral {args.umbral}, "
          f"{args.repeticiones} semillas")
    print(pd.DataFrame(filas).set_index("k").to_string())


if __name__ == "__main__":
    main()
