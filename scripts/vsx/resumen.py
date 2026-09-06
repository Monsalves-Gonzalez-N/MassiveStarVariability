#!/usr/bin/env python
"""El embudo completo en numeros, para escribir el paper y el catalogo.

Cada paso corta picos y pierde algo de recall. Lo que importa no es el acierto
de un paso aislado sino cuanto queda al final, asi que todo se mide sobre las
mismas estrellas revisadas.

    PYTHONPATH=src python scripts/vsx/resumen.py
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from arbol_features import ajustar, cargar, contexto, familia, pesos_balanceados
from selector_periodo import FEATURES, UMBRAL, make_model, matriz, relacion

from msv.config import RESULTS_DIR


def embudo_del_corte(peaks_todos):
    """Cuanto corta el filtro de Rndm y cuanto recall cuesta."""
    revisadas = peaks_todos[peaks_todos.vis_verdict.isin(["ok",
                                                          "unconstrained"])]
    lineas = []
    for etiqueta, bloque in [("todos los picos", revisadas),
                             ("clase != Rndm", revisadas[revisadas.clase
                                                         != "Rndm"])]:
        ok = bloque[bloque.vis_verdict == "ok"]
        lineas.append({
            "paso": etiqueta,
            "picos": len(bloque),
            "picos/estrella": round(len(bloque) / bloque.TIC.nunique(), 1),
            "estrellas con el 1:1": ok[ok.es_vsx].TIC.nunique(),
        })
    return pd.DataFrame(lineas)


def scores_out_of_fold(peaks, repeticiones):
    entrenables = peaks[peaks.verdad == "ok"]
    X = matriz(FEATURES, entrenables)
    y = entrenables.es_vsx.values.astype(int)
    weights = pesos_balanceados(entrenables)
    tics = entrenables.TIC.values
    acumulado = np.zeros(len(peaks))
    indices = np.flatnonzero((peaks.verdad == "ok").values)
    fuera = np.flatnonzero((peaks.verdad != "ok").values)
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
        scores[fuera] = modelo.predict_proba(matriz(FEATURES,
                                                    peaks.iloc[fuera]))[:, 1]
        acumulado += scores
    return acumulado / repeticiones


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--probe",
                        default=str(RESULTS_DIR / "vsx" /
                                    "probe_peaks_w001_1pass.csv"))
    parser.add_argument("--repeticiones", type=int, default=5)
    args = parser.parse_args()

    todos = pd.read_csv(RESULTS_DIR / "vsx" / "peaks_con_snr.csv")
    print("=== 1. el corte de Rndm ===")
    corte = embudo_del_corte(todos)
    print(corte.to_string(index=False))
    antes, despues = corte.picos.values
    con_1a1 = corte["estrellas con el 1:1"].values
    print(f"corta el {100 * (1 - despues / antes):.0f} % de los picos y "
          f"conserva el 1:1 en {100 * con_1a1[1] / con_1a1[0]:.1f} % de las "
          "estrellas que lo tenian")

    peaks = contexto(cargar(args.probe, verdicts=("ok", "unconstrained")))
    peaks["verdad"] = peaks.vis_verdict
    peaks["familia"] = familia(peaks)
    peaks["score"] = scores_out_of_fold(peaks, args.repeticiones)

    ordenados = peaks.sort_values("score", ascending=False)
    top1 = ordenados.groupby("TIC").head(1).set_index("TIC")
    top2 = ordenados.groupby("TIC").head(2)
    ok = top1[top1.verdad == "ok"]
    reporta = ok.score >= UMBRAL
    acierto_top2 = top2[top2.verdad == "ok"].groupby("TIC").es_vsx.max()

    print(f"\n=== 2. la seleccion del periodo ({len(ok)} estrellas ok) ===")
    print(f"top-1 correcto                 {100 * (ok.tag == '1:1').mean():.1f} %")
    print(f"el 1:1 esta en el top-2        {100 * acierto_top2.mean():.1f} %")
    print(f"con umbral {UMBRAL}: se reporta {100 * reporta.mean():.1f} % de las "
          f"estrellas, y de esas acierta {100 * (ok.tag == '1:1')[reporta].mean():.1f} %")

    print("\n=== 3. donde se pierde, por familia (estrellas reportadas) ===")
    reportadas = ok[reporta].copy()
    reportadas["en_top2"] = reportadas.index.map(acierto_top2)
    resumen = reportadas.groupby("familia").agg(
        estrellas=("tag", "size"),
        top1=("tag", lambda valores: 100 * (valores == "1:1").mean()),
        top2=("en_top2", lambda valores: 100 * valores.mean()),
    )
    resumen["se_gana_mirando_2"] = resumen.top2 - resumen.top1
    print(resumen.round(1).to_string())

    print("\n=== 4. la nota del catalogo: como se relaciona el 2do candidato ===")
    segundos = ordenados.groupby("TIC").nth(1)
    segundos = (segundos.set_index("TIC") if "TIC" in segundos.columns
                else segundos)
    reportadas["relacion"] = [relacion(fila.per, segundos.per.get(fila.Index,
                                                                 np.nan))
                              for fila in reportadas.itertuples()]
    tabla = pd.crosstab(reportadas.familia, reportadas.relacion)
    print(tabla.to_string())
    print("\nen % de cada familia:")
    print((100 * tabla.div(tabla.sum(axis=1), axis=0)).round(1).to_string())

    print("\n=== 5. las unconstrained ===")
    sin_periodo = top1[top1.verdad == "unconstrained"]
    print(f"{len(sin_periodo)} estrellas multiperiodicas; se abstiene en "
          f"{100 * (sin_periodo.score < UMBRAL).mean():.1f} % "
          f"({int((sin_periodo.score < UMBRAL).sum())}/{len(sin_periodo)})")


if __name__ == "__main__":
    main()
