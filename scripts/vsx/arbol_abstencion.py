#!/usr/bin/env python
"""Un selector que puede decir "no se": `unconstrained` como clase.

Forzar un top-1 por estrella es la pregunta equivocada cuando la estrella no
tiene UN periodo. La revision visual ya lo dice: las `unconstrained` son
multiperiodicas. En esas estrellas cualquier pico que elijamos es una respuesta
inventada, asi que la respuesta correcta es no dar ninguna.

`maybe` y `dudoso` quedan FUERA: son la duda del revisor, no una propiedad de
la estrella, y mezclarlas ensucia justo la clase que queremos medir. La clase
`unconstrained` son las 66 seguras.

El selector se entrena binario por pico sobre las `ok` (las unicas con verdad).
Las `unconstrained` NUNCA entran al fit: solo se mide si el modelo tiene la
decencia de no contestar cuando su mejor pico no llega al umbral. Las features
son exactamente las mismas de la seleccion de picos, sin agregar ninguna.

    PYTHONPATH=src python scripts/vsx/arbol_abstencion.py \
        --probe results/vsx/probe_peaks_w001_1pass.csv --umbral 0.75
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from arbol_features import (CICLOS, CNN, CNN_FINO, PERIODOGRAMA, SNR_1PASADA,
                            agregar_finas, ajustar, cargar, contexto, familia,
                            pesos_balanceados)
from arbol_mc import columnas_mc, resumen_mc

from msv.config import RESULTS_DIR

COLUMNAS = PERIODOGRAMA + CICLOS + CNN + SNR_1PASADA
# Las 11 que sobreviven a la poda por importancia: 4 del periodograma, 4 de la
# sonda, 3 de la red. Ver reducir_features.py.
ONCE = ["ciclos", "per_rel", "power", "width",
        "snr", "snr_breger", "a2_a1", "amplitude_ppt",
        "log_pLPV", "p_Rndm", "p_E"]
GLOBALES = ["amplitude", "irregular"]


def scores_entrenando_todo(peaks, semilla, columnas):
    """Las unconstrained entran al fit con TODOS sus picos en 0.

    Asi la abstencion deja de ser emergente (probabilidad baja por descarte) y
    pasa a ser aprendida, y sus scores salen out-of-fold como los demas.
    """
    X = peaks[columnas].replace([np.inf, -np.inf], np.nan).fillna(0.0).values
    y = np.where(peaks.verdad == "ok", peaks.es_vsx.astype(int), 0)
    pesos = peaks.copy()
    pesos["familia"] = np.where(pesos.verdad == "ok", pesos.familia,
                                "unconstrained")
    pesos["es_vsx"] = y.astype(bool)
    weights = pesos_balanceados(pesos)
    tics = peaks.TIC.values
    orden = np.random.RandomState(semilla).permutation(np.unique(tics))
    posicion = {tic: index for index, tic in enumerate(orden)}
    grupos = np.array([posicion[tic] for tic in tics])
    scores = np.full(len(peaks), np.nan)
    for entrena, prueba in GroupKFold(5).split(X, y, grupos):
        modelo = ajustar(GradientBoostingClassifier(n_estimators=200,
                                                    max_depth=3,
                                                    random_state=0),
                         X[entrena], y[entrena], weights[entrena])
        scores[prueba] = modelo.predict_proba(X[prueba])[:, 1]
    return scores


def scores_por_semilla(peaks, entrenables, fuera, semilla, COLUMNAS=None):
    COLUMNAS = COLUMNAS if COLUMNAS is not None else globals()["COLUMNAS"]
    X_todo = peaks[COLUMNAS].replace([np.inf, -np.inf], np.nan).fillna(0.0).values
    X = entrenables[COLUMNAS].replace([np.inf, -np.inf], np.nan).fillna(0.0).values
    y = entrenables.es_vsx.values.astype(int)
    weights = pesos_balanceados(entrenables)
    tics = entrenables.TIC.values

    def make_model():
        return GradientBoostingClassifier(n_estimators=200, max_depth=3,
                                          random_state=0)

    orden = np.random.RandomState(semilla).permutation(np.unique(tics))
    posicion = {tic: index for index, tic in enumerate(orden)}
    grupos = np.array([posicion[tic] for tic in tics])
    scores = np.full(len(peaks), np.nan)
    indices_ok = np.flatnonzero(~fuera)
    for entrena, prueba in GroupKFold(5).split(X, y, grupos):
        modelo = ajustar(make_model(), X[entrena], y[entrena], weights[entrena])
        scores[indices_ok[prueba]] = modelo.predict_proba(X[prueba])[:, 1]
    modelo = ajustar(make_model(), X, y, weights)
    scores[fuera] = modelo.predict_proba(X_todo[fuera])[:, 1]
    return scores


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--probe",
                        default=str(RESULTS_DIR / "vsx" /
                                    "probe_peaks_w001_1pass.csv"))
    parser.add_argument("--repeticiones", type=int, default=5)
    parser.add_argument("--umbral", type=float, default=0.75)
    parser.add_argument("--entrena-unconstrained", action="store_true",
                        help="meter las unconstrained al fit con todos sus "
                             "picos en 0, en vez de abstener por umbral")
    parser.add_argument("--once", action="store_true",
                        help="usar las 11 features del top-k")
    parser.add_argument("--con-globales", action="store_true",
                        help="sumar amplitude e irregular (cantidades por "
                             "estrella: no eligen pico, pero pueden decidir "
                             "si la estrella tiene periodo)")
    parser.add_argument("--grupos", action="store_true",
                        help="usar los 5 grupos viejos en vez de las 8 finas")
    parser.add_argument("--mc", action="store_true",
                        help="cambiar el bloque CNN por el resumen MC-dropout")
    args = parser.parse_args()

    peaks = contexto(cargar(args.probe, verdicts=("ok", "unconstrained")))
    peaks["verdad"] = peaks.vis_verdict
    peaks = agregar_finas(peaks)
    columnas = COLUMNAS
    if args.grupos:
        columnas = PERIODOGRAMA + CICLOS + CNN + SNR_1PASADA
    if args.once:
        columnas = ONCE
    if args.con_globales:
        columnas = columnas + GLOBALES
    if args.mc:
        mc = resumen_mc(RESULTS_DIR / "vsx" / "cnn_probs_w001_mc20.npz",
                        RESULTS_DIR / "vsx" / "clasificacion_w001.csv")
        peaks = peaks.merge(mc, on=["TIC", "sector", "clave"], how="left")
        peaks[columnas_mc()] = peaks[columnas_mc()].fillna(0.0)
        columnas = PERIODOGRAMA + CICLOS + SNR_1PASADA + columnas_mc()
    peaks["familia"] = familia(peaks)

    entrenables = peaks[peaks.verdad == "ok"]
    fuera = (peaks.verdad != "ok").values
    print(peaks.drop_duplicates("TIC").verdad.value_counts().to_string())
    print(f"\n{len(columnas)} features"
          f"{' (bloque MC-dropout)' if args.mc else ', las mismas de la seleccion de picos'}")

    curvas, matrices, tags = [], [], []
    for semilla in range(args.repeticiones):
        if args.entrena_unconstrained:
            scores = scores_entrenando_todo(peaks, semilla, columnas)
        else:
            scores = scores_por_semilla(peaks, entrenables, fuera, semilla,
                                        columnas)
        mejores = peaks.assign(_s=scores).sort_values(
            "_s", ascending=False).groupby("TIC").head(1)
        for umbral in np.arange(0.0, 0.95, 0.05):
            reporta = mejores._s >= umbral
            es_ok = mejores.verdad == "ok"
            curvas.append({
                "umbral": round(float(umbral), 2),
                "cobertura ok": 100 * (reporta & es_ok).sum() / es_ok.sum(),
                "acierto entre reportadas": (
                    100 * ((mejores.tag == "1:1") & reporta & es_ok).sum()
                    / max(int((reporta & es_ok).sum()), 1)),
                "abstiene unconstrained": (
                    100 * (~reporta & ~es_ok).sum() / max(int((~es_ok).sum()), 1)),
            })
        decision = np.where(mejores._s >= args.umbral, "reporta periodo",
                            "unconstrained")
        matrices.append(pd.crosstab(mejores.verdad, decision))
        reportadas = mejores[(mejores._s >= args.umbral)
                             & (mejores.verdad == "ok")]
        tags.append(pd.crosstab(reportadas.familia, reportadas.tag))

    print(f"\ncurva de abstencion, boosting, {args.repeticiones} semillas")
    print(pd.DataFrame(curvas).groupby("umbral").mean().round(1).to_string())

    matriz = sum(matrices) / len(matrices)
    print(f"\n=== matriz de confusion por ESTRELLA, umbral {args.umbral} "
          f"(promedio de {args.repeticiones} semillas) ===")
    print(matriz.round(1).to_string())
    normalizada = 100 * matriz.div(matriz.sum(axis=1), axis=0)
    print("\nla misma, en % de cada fila:")
    print(normalizada.round(1).to_string())

    tabla_tags = sum(tag.reindex(index=tags[0].index,
                                 columns=tags[0].columns, fill_value=0)
                     for tag in tags) / len(tags)
    print(f"\n=== en que falla el periodo reportado (solo estrellas ok "
          f"reportadas), por familia ===")
    print(tabla_tags.round(1).to_string())
    print("\nla misma, en % de cada fila:")
    print((100 * tabla_tags.div(tabla_tags.sum(axis=1), axis=0))
          .round(1).to_string())


if __name__ == "__main__":
    main()
