#!/usr/bin/env python
"""Pipeline de catalogo: limpiar, elegir periodo, clasificar.

Este es el script de produccion. Todo lo demas en esta carpeta es el registro
de las ablaciones que llevaron a esta configuracion; ver README.md.

El orden importa y no es el obvio. La version anterior elegia periodo y se
abstenia cuando la probabilidad no llegaba a un umbral, pero esa abstencion
emergente conseguia su pureza TIRANDO estrellas buenas: descartaba 59 de 293
`ok` para sacar 64 de 66 multiperiodicas. Separar las dos decisiones —un
clasificador por estrella entrenado para detectar multiperiodicas, y despues un
selector que solo ordena— sube los periodos correctos de 73.4 % a 82.3 % del
total, a cambio de 3.2 % de contaminacion en vez de 0.8 %.

    PYTHONPATH=src python scripts/vsx/selector_periodo.py
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold, StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from arbol_features import (FAMILIAS, ajustar, cargar, contexto, familia,
                            pesos_balanceados)
from clase_estrella import vector_por_estrella

from msv.config import RESULTS_DIR

# --- paso 2: el selector de periodo -----------------------------------------
# 4 del periodograma, 4 de la sonda, 3 de la red. Las 16 que se sacaron o eran
# ruido (importancia < 0.006), o duplicaban a otra por transformacion monotona
# (`p_LPV` contra `log_pLPV`), o eran constantes dentro de la estrella y por lo
# tanto incapaces de elegir entre sus picos (`amplitude`, `irregular`).
FEATURES = ["ciclos", "per_rel", "power", "width",
            "snr", "snr_breger", "a2_a1", "amplitude_ppt",
            "log_pLPV", "p_Rndm", "p_E"]
# Corte del gate. En 0.3 conserva 275/293 ok y saca 57/66 multiperiodicas.
CORTE_GATE = 0.30
TOLERANCIA = 0.05
K_PICOS = 5
# `otro` junta VAR/MISC/ACV/DPV/HB: no es un tipo fisico sino lo que no cae en
# las tres familias, asi que no entra al clasificador ni a su validacion.
CLASES = ["ECL", "ELL", "PULS"]


def make_selector():
    return GradientBoostingClassifier(n_estimators=200, max_depth=3,
                                      random_state=0)


def make_gate():
    return RandomForestClassifier(n_estimators=400, min_samples_leaf=2,
                                  class_weight="balanced", random_state=0,
                                  n_jobs=-1)


def make_clasificador():
    return RandomForestClassifier(n_estimators=400, min_samples_leaf=2,
                                  class_weight="balanced", random_state=0,
                                  n_jobs=-1)


def matriz(columnas, tabla):
    return tabla[columnas].replace([np.inf, -np.inf], np.nan).fillna(0.0).values


def relacion(primero, segundo):
    """Como se relaciona el segundo candidato con el elegido."""
    if not (np.isfinite(primero) and np.isfinite(segundo) and primero > 0):
        return ""
    for factor, etiqueta in [(2.0, "doble"), (0.5, "mitad"),
                             (3.0, "triple"), (1 / 3, "tercio")]:
        if abs(segundo / (factor * primero) - 1.0) < TOLERANCIA:
            return etiqueta
    return "independiente"


def nota_catalogo(fila):
    """La ambiguedad del factor 2, escrita para quien lea el catalogo."""
    if fila.decision != "periodo":
        return "multiperiodica: no se reporta periodo"
    # La direccion del aviso sale del modo de falla medido por familia: las
    # eclipsantes se equivocan reportando P/2 y las pulsantes reportando 2P.
    if fila.relacion_top2 == "mitad":
        return ("ambiguo x2: 2do candidato en P/2; si la estrella es pulsante "
                "el periodo podria ser la mitad del reportado")
    if fila.relacion_top2 == "doble":
        return ("ambiguo x2: 2do candidato en 2P; si es ELL o eclipsante el "
                "periodo podria ser el doble del reportado")
    return ""


def scores_gate_oof(estrellas, columnas, repeticiones):
    X = matriz(columnas, estrellas)
    y = (estrellas.es_multiperiodica).values.astype(int)
    acumulado = np.zeros(len(y))
    for semilla in range(repeticiones):
        scores = np.zeros(len(y))
        for entrena, prueba in StratifiedKFold(
                5, shuffle=True, random_state=semilla).split(X, y):
            modelo = make_gate()
            modelo.fit(X[entrena], y[entrena])
            scores[prueba] = modelo.predict_proba(X[prueba])[:, 1]
        acumulado += scores
    return acumulado / repeticiones


def scores_selector_oof(peaks, repeticiones):
    """Out-of-fold en las `ok`; las multiperiodicas se puntuan con el modelo
    completo, porque nunca aportan etiqueta al fit."""
    entrenables = peaks[~peaks.es_multiperiodica]
    X = matriz(FEATURES, entrenables)
    y = entrenables.es_vsx.values.astype(int)
    weights = pesos_balanceados(entrenables)
    tics = entrenables.TIC.values
    dentro = np.flatnonzero((~peaks.es_multiperiodica).values)
    fuera = np.flatnonzero(peaks.es_multiperiodica.values)
    acumulado = np.zeros(len(peaks))
    for semilla in range(repeticiones):
        orden = np.random.RandomState(semilla).permutation(np.unique(tics))
        posicion = {tic: index for index, tic in enumerate(orden)}
        grupos = np.array([posicion[tic] for tic in tics])
        scores = np.zeros(len(peaks))
        for entrena, prueba in GroupKFold(5).split(X, y, grupos):
            modelo = ajustar(make_selector(), X[entrena], y[entrena],
                             weights[entrena])
            scores[dentro[prueba]] = modelo.predict_proba(X[prueba])[:, 1]
        modelo = ajustar(make_selector(), X, y, weights)
        scores[fuera] = modelo.predict_proba(
            matriz(FEATURES, peaks.iloc[fuera]))[:, 1]
        acumulado += scores
    return acumulado / repeticiones


def clase_oof(estrellas, columnas, repeticiones):
    buenas = estrellas[~estrellas.es_multiperiodica
                       & estrellas.familia.isin(CLASES)]
    X = matriz(columnas, buenas)
    y = buenas.familia.values
    votos = pd.DataFrame(index=buenas.index)
    for semilla in range(repeticiones):
        prediccion = np.empty(len(y), dtype=object)
        for entrena, prueba in StratifiedKFold(
                5, shuffle=True, random_state=semilla).split(X, y):
            modelo = make_clasificador()
            modelo.fit(X[entrena], y[entrena])
            prediccion[prueba] = modelo.predict(X[prueba])
        votos[semilla] = prediccion
    return votos, y


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--probe",
                        default=str(RESULTS_DIR / "vsx" /
                                    "probe_peaks_w001_todos_1pass.csv"))
    parser.add_argument("--peaks",
                        default=str(RESULTS_DIR / "vsx" /
                                    "peaks_con_snr_Number_M.csv"),
                        help="tabla de picos con el bloque de la CNN")
    parser.add_argument("--out", default=str(RESULTS_DIR / "vsx" /
                                             "catalogo_periodos.csv"))
    parser.add_argument("--corte-gate", type=float, default=CORTE_GATE)
    parser.add_argument("--repeticiones", type=int, default=5)
    args = parser.parse_args()

    peaks = contexto(cargar(args.probe, verdicts=("ok", "unconstrained"),
                            peaks_csv=args.peaks))
    peaks["es_multiperiodica"] = peaks.vis_verdict == "unconstrained"
    peaks["familia"] = familia(peaks)
    peaks["clase_verdadera"] = np.where(peaks.es_multiperiodica,
                                        "unconstrained", peaks.familia)
    print(f"{len(peaks)} picos de {peaks.TIC.nunique()} estrellas "
          f"({int(peaks.drop_duplicates('TIC').es_multiperiodica.sum())} "
          "multiperiodicas)")

    estrellas = vector_por_estrella(peaks, K_PICOS, con_per=False)
    por_tic = peaks.drop_duplicates("TIC").set_index("TIC")
    estrellas["es_multiperiodica"] = estrellas.TIC.map(por_tic.es_multiperiodica)
    estrellas["familia"] = estrellas.TIC.map(por_tic.familia)
    columnas = [c for c in estrellas.columns
                if c not in ("TIC", "clase_verdadera", "cnn_top",
                             "es_multiperiodica", "familia")]

    # --- validacion out-of-fold ---------------------------------------------
    gate_oof = scores_gate_oof(estrellas, columnas, args.repeticiones)
    pasa = gate_oof < args.corte_gate
    buenas = ~estrellas.es_multiperiodica.values
    print(f"\n=== paso 1: gate de multiperiodicas (corte {args.corte_gate}) ===")
    print(f"conserva {int((pasa & buenas).sum())}/{int(buenas.sum())} ok  "
          f"({100 * (pasa & buenas).sum() / buenas.sum():.1f} %)")
    print(f"saca {int((~pasa & ~buenas).sum())}/{int((~buenas).sum())} "
          f"multiperiodicas "
          f"({100 * (~pasa & ~buenas).sum() / (~buenas).sum():.1f} %)")

    scores = scores_selector_oof(peaks, args.repeticiones)
    mejores = peaks.assign(_s=scores).sort_values(
        "_s", ascending=False).groupby("TIC").head(1).set_index("TIC")
    mejores["pasa"] = mejores.index.map(pd.Series(pasa, index=estrellas.TIC))
    en_catalogo = mejores[mejores.pasa]
    correctas = en_catalogo[~en_catalogo.es_multiperiodica]
    print(f"\n=== paso 2: seleccion del periodo ===")
    print(f"acierto top-1 sobre las ok del catalogo  "
          f"{100 * (correctas.tag == '1:1').mean():.1f} %")
    print(f"periodos correctos / total de ok         "
          f"{100 * (correctas.tag == '1:1').sum() / buenas.sum():.1f} %")
    print(f"contaminacion del catalogo               "
          f"{100 * en_catalogo.es_multiperiodica.mean():.1f} %")

    votos, verdad = clase_oof(estrellas, columnas, args.repeticiones)
    aciertos = [100 * (votos[semilla].values == verdad).mean()
                for semilla in votos.columns]
    balanceados = [100 * balanced_accuracy_score(verdad, votos[semilla].values)
                   for semilla in votos.columns]
    print(f"\n=== paso 3: clasificacion ({'/'.join(CLASES)}, "
          f"{len(verdad)} estrellas; `otro` excluido) ===")
    print(f"acierto {np.mean(aciertos):.1f} ± {np.std(aciertos):.1f} %   "
          f"balanceado {np.mean(balanceados):.1f} ± {np.std(balanceados):.1f} %")

    # --- modelos de produccion, ajustados con todo ---------------------------
    gate = make_gate()
    gate.fit(matriz(columnas, estrellas),
             estrellas.es_multiperiodica.values.astype(int))
    estrellas["gate_score"] = gate.predict_proba(matriz(columnas, estrellas))[:, 1]

    entrenables = peaks[~peaks.es_multiperiodica]
    selector = ajustar(make_selector(), matriz(FEATURES, entrenables),
                       entrenables.es_vsx.values.astype(int),
                       pesos_balanceados(entrenables))
    peaks["probabilidad"] = selector.predict_proba(matriz(FEATURES, peaks))[:, 1]

    buenas_tabla = estrellas[~estrellas.es_multiperiodica
                             & estrellas.familia.isin(CLASES)]
    clasificador = make_clasificador()
    clasificador.fit(matriz(columnas, buenas_tabla), buenas_tabla.familia.values)
    estrellas["clase_predicha"] = clasificador.predict(matriz(columnas,
                                                              estrellas))

    ordenados = peaks.sort_values("probabilidad", ascending=False)
    elegidos = ordenados.groupby("TIC").head(1).sort_values("TIC").copy()
    segundos = ordenados.groupby("TIC").nth(1)
    segundos = (segundos.set_index("TIC") if "TIC" in segundos.columns
                else segundos)
    elegidos["per_top2"] = elegidos.TIC.map(segundos.per)
    elegidos["prob_top2"] = elegidos.TIC.map(segundos.probabilidad)
    elegidos["relacion_top2"] = [relacion(fila.per, fila.per_top2)
                                 for fila in elegidos.itertuples()]
    indexadas = estrellas.set_index("TIC")
    elegidos["gate_score"] = elegidos.TIC.map(indexadas.gate_score)
    elegidos["clase_predicha"] = elegidos.TIC.map(indexadas.clase_predicha)
    elegidos["decision"] = np.where(elegidos.gate_score < args.corte_gate,
                                    "periodo", "multiperiodica")
    elegidos.loc[elegidos.decision == "multiperiodica",
                 "clase_predicha"] = "unconstrained"
    elegidos["nota"] = [nota_catalogo(fila) for fila in elegidos.itertuples()]

    salida = elegidos[["TIC", "sector", "per", "probabilidad", "gate_score",
                       "decision", "clase_predicha", "nota", "per_top2",
                       "prob_top2", "relacion_top2", "familia", "source",
                       "clase", "snr", "amplitude_ppt", "ciclos",
                       "vis_verdict", "per_vsx"]].rename(
        columns={"per": "per_elegido", "clase": "clase_cnn"})
    salida.to_csv(args.out, index=False)
    print(f"\n-> {args.out}  ({len(salida)} estrellas)")
    print(salida.decision.value_counts().to_string())


if __name__ == "__main__":
    main()
