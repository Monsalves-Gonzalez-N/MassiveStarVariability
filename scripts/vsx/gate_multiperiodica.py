#!/usr/bin/env python
"""Limpiar primero: sacar las multiperiodicas ANTES de elegir periodo.

Hoy la abstencion es un subproducto — la estrella se descarta porque su mejor
pico saca poca probabilidad en un modelo entrenado para otra cosa. Aca la
pregunta se hace directo: un clasificador binario POR ESTRELLA sobre el
conjunto completo de picos, entrenado para separar `ok` de `unconstrained`.

Si gana, el pipeline se reordena: filtro de Rndm -> filtro de multiperiodicas
-> seleccion de periodo sobre lo que queda, que es un problema mas facil porque
ya no tiene que decidir tambien si la estrella tiene periodo.

La comparacion es contra la curva de abstencion emergente: a igual fraccion de
`ok` conservadas, cuantas `unconstrained` saca cada uno.

    PYTHONPATH=src python scripts/vsx/gate_multiperiodica.py
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import GroupKFold, StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from arbol_features import ajustar, cargar, contexto, familia, pesos_balanceados
from clase_estrella import vector_por_estrella
from selector_periodo import FEATURES, UMBRAL, make_model, matriz

from msv.config import RESULTS_DIR


def scores_gate(tabla, columnas, make_model_gate, repeticiones):
    X = tabla[columnas].replace([np.inf, -np.inf], np.nan).fillna(0.0).values
    y = (tabla.clase_verdadera == "unconstrained").values.astype(int)
    acumulado = np.zeros(len(y))
    for semilla in range(repeticiones):
        scores = np.zeros(len(y))
        particion = StratifiedKFold(5, shuffle=True, random_state=semilla)
        for entrena, prueba in particion.split(X, y):
            modelo = make_model_gate()
            modelo.fit(X[entrena], y[entrena])
            scores[prueba] = modelo.predict_proba(X[prueba])[:, 1]
        acumulado += scores
    return acumulado / repeticiones, y


def scores_selector(peaks, repeticiones):
    entrenables = peaks[peaks.verdad == "ok"]
    X = matriz(FEATURES, entrenables)
    y = entrenables.es_vsx.values.astype(int)
    weights = pesos_balanceados(entrenables)
    tics = entrenables.TIC.values
    indices = np.flatnonzero((peaks.verdad == "ok").values)
    fuera = np.flatnonzero((peaks.verdad != "ok").values)
    acumulado = np.zeros(len(peaks))
    for semilla in range(repeticiones):
        orden = np.random.RandomState(semilla).permutation(np.unique(tics))
        posicion = {tic: index for index, tic in enumerate(orden)}
        grupos = np.array([posicion[tic] for tic in tics])
        scores = np.zeros(len(peaks))
        for entrena, prueba in GroupKFold(5).split(X, y, grupos):
            modelo = ajustar(make_model(), X[entrena], y[entrena],
                             weights[entrena])
            scores[indices[prueba]] = modelo.predict_proba(X[prueba])[:, 1]
        modelo = ajustar(make_model(), X, y, weights)
        scores[fuera] = modelo.predict_proba(
            matriz(FEATURES, peaks.iloc[fuera]))[:, 1]
        acumulado += scores
    return acumulado / repeticiones


def curva(conserva_ok, saca_unconstrained, objetivos):
    """A igual fraccion de ok conservadas, cuantas unconstrained saca."""
    filas = []
    for objetivo in objetivos:
        indice = int(np.argmin(np.abs(conserva_ok - objetivo)))
        filas.append({"conserva ok [%]": round(100 * conserva_ok[indice], 1),
                      "saca unconstrained [%]":
                          round(100 * saca_unconstrained[indice], 1)})
    return pd.DataFrame(filas)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--probe",
                        default=str(RESULTS_DIR / "vsx" /
                                    "probe_peaks_w001_1pass.csv"))
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--repeticiones", type=int, default=5)
    args = parser.parse_args()

    peaks = contexto(cargar(args.probe, verdicts=("ok", "unconstrained")))
    peaks["verdad"] = peaks.vis_verdict
    peaks["familia"] = familia(peaks)
    peaks["clase_verdadera"] = np.where(peaks.vis_verdict == "unconstrained",
                                        "unconstrained", "ok")
    tabla = vector_por_estrella(peaks, args.k, con_per=False)
    columnas = [c for c in tabla.columns
                if c not in ("TIC", "clase_verdadera", "cnn_top")]
    print(f"{len(tabla)} estrellas, "
          f"{int((tabla.clase_verdadera == 'unconstrained').sum())} "
          f"unconstrained, {len(columnas)} features por estrella")

    objetivos = np.arange(0.95, 0.60, -0.05)
    resultados = {}
    for nombre, make_gate in {
        "gate: random forest": lambda: RandomForestClassifier(
            n_estimators=400, min_samples_leaf=2, class_weight="balanced",
            random_state=0, n_jobs=-1),
        "gate: gradient boosting": lambda: GradientBoostingClassifier(
            n_estimators=200, max_depth=3, random_state=0),
    }.items():
        scores, y = scores_gate(tabla, columnas, make_gate, args.repeticiones)
        orden = np.argsort(-scores)
        es_unconstrained = y[orden].astype(bool)
        # recorrer umbrales = ir sacando estrellas de mayor a menor score
        saca = np.cumsum(es_unconstrained) / es_unconstrained.sum()
        conserva = 1 - (np.cumsum(~es_unconstrained)
                        / (~es_unconstrained).sum())
        resultados[nombre] = curva(conserva, saca, objetivos)

    scores = scores_selector(peaks, args.repeticiones)
    mejores = peaks.assign(_s=scores).sort_values(
        "_s", ascending=False).groupby("TIC").head(1)
    orden = np.argsort(mejores._s.values)
    es_unconstrained = (mejores.verdad.values == "unconstrained")[orden]
    saca = np.cumsum(es_unconstrained) / es_unconstrained.sum()
    conserva = 1 - np.cumsum(~es_unconstrained) / (~es_unconstrained).sum()
    resultados["abstencion emergente (actual)"] = curva(conserva, saca,
                                                        objetivos)

    print("\na igual fraccion de estrellas ok conservadas, "
          "que fraccion de multiperiodicas saca cada metodo")
    comparacion = pd.concat(
        {nombre: bloque.set_index("conserva ok [%]")["saca unconstrained [%]"]
         for nombre, bloque in resultados.items()}, axis=1)
    print(comparacion.to_string())


if __name__ == "__main__":
    main()
