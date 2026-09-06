#!/usr/bin/env python
"""Cual de los 7 checkpoints de la CNN sirve mejor al pipeline.

Los pesos vienen del entrenamiento de OGLE y cada uno se guardo optimizando una
clase distinta (`Number_CEP`, `Number_ELL`, ...). El pipeline usa `Number_ELL`
por herencia, no porque se haya medido. Aca se corre el pipeline completo con
cada uno sobre EL MISMO cubo (normalizacion `log`) y la misma sonda, asi que lo
unico que cambia es la red.

Se miden las dos cosas que el catalogo reporta: el periodo y la clase. Un
checkpoint puede ser mejor eligiendo periodo y peor clasificando.

    PYTHONPATH=src python scripts/vsx/comparar_checkpoints.py
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from arbol_features import FAMILIAS, cargar, contexto, familia
from clase_estrella import vector_por_estrella
from selector_periodo import (CLASES, CORTE_GATE, K_PICOS, make_clasificador,
                              make_gate, matriz, scores_selector_oof)

from msv.config import MODELS, RESULTS_DIR

TOLERANCIA = 0.05


def perdidas_por_rndm(clasificacion, estrellas):
    """Estrellas cuyo periodo verdadero no sobrevive al filtro de Rndm."""
    cnn = pd.read_csv(clasificacion, usecols=["TIC", "per", "clase"])
    perdidas = 0
    for fila in estrellas.itertuples():
        bloque = cnn[cnn.TIC == fila.TIC]
        cerca = bloque[np.abs(bloque.per / fila.per_vsx - 1.0) < TOLERANCIA]
        if cerca.empty or not (cerca.clase != "Rndm").any():
            perdidas += 1
    return perdidas


def evaluar(etiqueta, peaks_csv, probe, repeticiones):
    peaks = contexto(cargar(probe, verdicts=("ok", "unconstrained"),
                            peaks_csv=peaks_csv))
    peaks["es_multiperiodica"] = peaks.vis_verdict == "unconstrained"
    peaks["familia"] = familia(peaks)
    peaks["clase_verdadera"] = np.where(peaks.es_multiperiodica,
                                        "unconstrained", peaks.familia)
    peaks["probabilidad"] = scores_selector_oof(peaks, repeticiones)

    estrellas = vector_por_estrella(peaks, K_PICOS, con_per=False)
    por_tic = peaks.drop_duplicates("TIC").set_index("TIC")
    estrellas["es_multiperiodica"] = estrellas.TIC.map(por_tic.es_multiperiodica)
    estrellas["familia"] = estrellas.TIC.map(por_tic.familia)
    columnas = [c for c in estrellas.columns
                if c not in ("TIC", "clase_verdadera", "cnn_top",
                             "es_multiperiodica", "familia")]
    X = matriz(columnas, estrellas)
    y = estrellas.es_multiperiodica.values.astype(int)
    gate = np.zeros(len(y))
    for semilla in range(repeticiones):
        parcial = np.zeros(len(y))
        for entrena, prueba in StratifiedKFold(
                5, shuffle=True, random_state=semilla).split(X, y):
            modelo = make_gate()
            modelo.fit(X[entrena], y[entrena])
            parcial[prueba] = modelo.predict_proba(X[prueba])[:, 1]
        gate += parcial / repeticiones
    pasa = pd.Series(gate < CORTE_GATE, index=estrellas.TIC.values)

    top1 = peaks.sort_values("probabilidad", ascending=False).groupby(
        "TIC").head(1)
    catalogo = top1[top1.TIC.map(pasa)]
    buenas = catalogo[~catalogo.es_multiperiodica]
    n_ok = int((~estrellas.es_multiperiodica).sum())
    n_unc = int(estrellas.es_multiperiodica.sum())

    entrenables = estrellas[~estrellas.es_multiperiodica
                            & estrellas.familia.isin(CLASES)]
    Xc = matriz(columnas, entrenables)
    yc = entrenables.familia.values
    aciertos, balanceados = [], []
    for semilla in range(repeticiones):
        prediccion = np.empty(len(yc), dtype=object)
        for entrena, prueba in StratifiedKFold(
                5, shuffle=True, random_state=semilla).split(Xc, yc):
            modelo = make_clasificador()
            modelo.fit(Xc[entrena], yc[entrena])
            prediccion[prueba] = modelo.predict(Xc[prueba])
        aciertos.append(100 * (prediccion == yc).mean())
        balanceados.append(100 * balanced_accuracy_score(yc, prediccion))

    return {
        "checkpoint": etiqueta,
        "picos": len(peaks),
        "cand/estrella": round(len(peaks) / peaks.TIC.nunique(), 1),
        "saca unc": f"{n_unc - int(catalogo.es_multiperiodica.sum())}/{n_unc}",
        "acierto top-1": round(100 * (buenas.tag == "1:1").mean(), 1),
        "correctos/ok": round(100 * (buenas.tag == "1:1").sum() / n_ok, 1),
        "clase acierto": round(float(np.mean(aciertos)), 1),
        "clase balanceado": round(float(np.mean(balanceados)), 1),
    }


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--probe",
                        default=str(RESULTS_DIR / "vsx" /
                                    "probe_peaks_w001_todos_1pass.csv"))
    parser.add_argument("--repeticiones", type=int, default=3)
    parser.add_argument("--modelos", nargs="*", default=MODELS,
                        help="subconjunto de checkpoints a comparar")
    args = parser.parse_args()

    verdad = pd.read_csv(RESULTS_DIR / "vsx" / "peaks_con_snr.csv",
                         usecols=["TIC", "vis_verdict", "vsx_token",
                                  "per_vsx"])
    estrellas_ok = verdad[verdad.vis_verdict == "ok"].drop_duplicates("TIC")

    filas = []
    for modelo in args.modelos:
        peaks_csv = RESULTS_DIR / "vsx" / f"peaks_con_snr_{modelo}.csv"
        if not peaks_csv.exists():
            print(f"(falta {peaks_csv.name}, se salta)")
            continue
        fila = evaluar(modelo, str(peaks_csv), args.probe, args.repeticiones)
        fila["pierde por Rndm"] = perdidas_por_rndm(
            RESULTS_DIR / "vsx" / f"clasificacion_w001_{modelo}.csv",
            estrellas_ok)
        filas.append(fila)
    tabla = pd.DataFrame(filas).set_index("checkpoint")
    print(f"\npipeline completo por checkpoint (norm log, "
          f"{args.repeticiones} semillas)")
    print(tabla.to_string())


if __name__ == "__main__":
    main()
