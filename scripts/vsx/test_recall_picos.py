#!/usr/bin/env python
"""Test 1 del benchmark VSX: ¿está el período de VSX entre nuestros picos?

Dos preguntas separadas, sobre las estrellas con veredicto visual utilizable
(`ok`, `maybe`, `unconstrained`) de `catalogs/vsx_visual_review.csv`:

  Q1  recall del periodograma: ¿aparece `per_vsx` entre TODOS los picos de la
      estrella-sector, a tolerancia relativa del 5%? Se cuenta aparte el 1:1 y
      los alias armónicos (2P, P/2, 3P, P/3).

  Q2  qué queda al tirar los picos que la CNN llama Rndm o LPV: cuántos picos
      sobreviven, y cuántas estrellas conservan el pico que reproduce
      `per_vsx`.

    PYTHONPATH=src python scripts/vsx/test_recall_picos.py
"""
import argparse

import numpy as np
import pandas as pd

from msv.config import CATALOGS_DIR, RESULTS_DIR

HARMONICS = {"1:1": 1.0, "2P": 2.0, "P/2": 0.5, "3P": 3.0, "P/3": 1.0 / 3.0}
TOLERANCE = 0.05
DESCARTADAS = ["Rndm"]
VEREDICTOS = ["ok", "maybe", "unconstrained"]


def harmonic_tag(ratio):
    for name, target in HARMONICS.items():
        if np.abs(ratio / target - 1.0) < TOLERANCE:
            return name
    return "otro"


def resumen(tabla, columna, etiqueta):
    total = tabla.groupby(columna).size().rename("estrellas")
    uno = tabla[tabla.found_1a1].groupby(columna).size().rename("1:1")
    alias = tabla[~tabla.found_1a1 & tabla.found_alias].groupby(columna).size().rename("solo alias")
    nada = tabla[~tabla.found_1a1 & ~tabla.found_alias].groupby(columna).size().rename("nada")
    salida = pd.concat([total, uno, alias, nada], axis=1).fillna(0).astype(int)
    salida["recall 1:1"] = (100 * salida["1:1"] / salida.estrellas).round(1)
    salida["recall +alias"] = (100 * (salida["1:1"] + salida["solo alias"])
                               / salida.estrellas).round(1)
    print(f"\n--- {etiqueta} ---")
    print(salida.to_string())
    return salida


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--clasificacion",
                        default=str(RESULTS_DIR / "vsx" / "clasificacion.csv"))
    parser.add_argument("--out-dir", default=str(RESULTS_DIR / "vsx"))
    args = parser.parse_args()

    review = pd.read_csv(CATALOGS_DIR / "vsx_visual_review.csv")
    review = review[review.vis_verdict.isin(VEREDICTOS) & review.per_vsx.notna()]
    review = review[["TIC", "per_vsx", "vsx_token", "expected_class",
                     "vis_verdict", "vis_multi"]]

    peaks = pd.read_csv(args.clasificacion)
    peaks = peaks.merge(review, on="TIC", how="inner")
    peaks["ratio"] = peaks.per / peaks.per_vsx
    peaks["tag"] = [harmonic_tag(value) for value in peaks.ratio]
    peaks["es_vsx"] = peaks.tag == "1:1"
    peaks["es_alias"] = peaks.tag.isin(["2P", "P/2", "3P", "P/3"])
    peaks["sobrevive"] = ~peaks.clase.isin(DESCARTADAS)

    faltan = set(review.TIC) - set(peaks.TIC)
    print(f"{review.TIC.nunique()} estrellas con veredicto {VEREDICTOS} y per_vsx")
    print(f"{peaks.TIC.nunique()} con picos, {len(faltan)} sin ningún pico")
    print(f"{len(peaks)} picos en total "
          f"({peaks.groupby(['TIC', 'sector']).ngroups} estrella-sector)")

    # --- Q1 -----------------------------------------------------------------
    filas = []
    for (tic, sector), block in peaks.groupby(["TIC", "sector"]):
        filas.append({
            "TIC": tic,
            "sector": sector,
            "vis_verdict": block.vis_verdict.iloc[0],
            "vis_multi": block.vis_multi.iloc[0],
            "vsx_token": block.vsx_token.iloc[0],
            "per_vsx": block.per_vsx.iloc[0],
            "n_picos": len(block),
            "found_1a1": bool(block.es_vsx.any()),
            "found_alias": bool(block.es_alias.any()),
            "n_picos_post": int(block.sobrevive.sum()),
            "found_1a1_post": bool((block.es_vsx & block.sobrevive).any()),
            "found_alias_post": bool((block.es_alias & block.sobrevive).any()),
        })
    recall = pd.DataFrame(filas)
    for tic in faltan:
        fila = review[review.TIC == tic].iloc[0]
        recall.loc[len(recall)] = {
            "TIC": tic, "sector": -1, "vis_verdict": fila.vis_verdict,
            "vis_multi": fila.vis_multi, "vsx_token": fila.vsx_token,
            "per_vsx": fila.per_vsx, "n_picos": 0, "found_1a1": False,
            "found_alias": False, "n_picos_post": 0, "found_1a1_post": False,
            "found_alias_post": False}

    print("\n" + "=" * 70)
    print("Q1  ¿está per_vsx entre nuestros picos? (tolerancia relativa 5%)")
    print("=" * 70)
    resumen(recall, "vis_verdict", "por veredicto visual")
    resumen(recall[recall.vis_verdict == "ok"], "vsx_token",
            "por tipo VSX, solo las OK")

    # --- Q2 -----------------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"Q2  descartando los picos con clase {DESCARTADAS}")
    print("=" * 70)
    conteo = peaks.groupby(["vis_verdict", "clase"]).size().unstack(fill_value=0)
    print("\npicos por clase de la CNN:")
    print(conteo.to_string())

    antes = peaks.groupby("vis_verdict").size().rename("picos antes")
    despues = peaks[peaks.sobrevive].groupby("vis_verdict").size().rename("picos después")
    tabla = pd.concat([antes, despues], axis=1).fillna(0).astype(int)
    tabla["% que queda"] = (100 * tabla["picos después"] / tabla["picos antes"]).round(1)
    tabla["picos/estrella antes"] = (
        tabla["picos antes"] / recall.groupby("vis_verdict").size()).round(1)
    tabla["picos/estrella después"] = (
        tabla["picos después"] / recall.groupby("vis_verdict").size()).round(1)
    print("\n" + tabla.to_string())

    print("\nestrellas que se quedan SIN ningún pico tras el corte:")
    print(recall[recall.n_picos_post == 0].groupby("vis_verdict").size().to_string())

    print("\nrecall DESPUÉS del corte (el pico correcto sigue vivo):")
    posterior = recall.rename(columns={"found_1a1": "_a", "found_alias": "_b"})
    posterior = posterior.rename(columns={"found_1a1_post": "found_1a1",
                                          "found_alias_post": "found_alias"})
    resumen(posterior, "vis_verdict", "por veredicto visual, tras el corte")

    salida = f"{args.out_dir}/recall_vsx.csv"
    recall.to_csv(salida, index=False)
    peaks.drop(columns=["hist2d"], errors="ignore").to_csv(
        f"{args.out_dir}/match_peaks_vsx.csv", index=False)
    print(f"\n-> {salida}")


if __name__ == "__main__":
    main()
