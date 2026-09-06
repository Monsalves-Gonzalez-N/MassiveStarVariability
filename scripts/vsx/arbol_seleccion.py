#!/usr/bin/env python
"""¿Existe una regla para elegir el pico bueno? Un árbol sobre las OK de VSX.

La pregunta no es clasificar un pico aislado sino ELEGIR uno por estrella, así
que el árbol se entrena como clasificador binario por pico (`es_vsx`) y se
evalúa como selector: por estrella se toma el pico de mayor probabilidad
predicha y se mira si es el período de VSX. La comparación es contra las reglas
de un solo criterio (`log_pLPV` 68.9%, `power` 63.1%).

Todo se mide FUERA DE MUESTRA con `GroupKFold` agrupando por TIC: los picos de
una estrella no pueden estar a los dos lados de la partición, o el árbol
memoriza la estrella en vez de aprender la regla.

Las features son de tres tipos:

  del pico       power, prominence, width, amplitude, per, ciclos, SNR
  de la CNN      las cinco p_<clase>, log_pLPV, prob
  del CONTEXTO   la posición del pico DENTRO de su estrella: cuántos hay, su
                 rango por power, su power relativo al máximo de su misma
                 fuente, y —lo que resuelve el factor 2— si en el mismo
                 conjunto sobrevive un pico al doble o a la mitad de su período

El contexto es lo que hace comparable un power de LS con uno de ACF: no se usa
el valor absoluto sino su rango dentro de la estrella y de su propia fuente.

    PYTHONPATH=src python scripts/vsx/arbol_seleccion.py
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.tree import DecisionTreeClassifier, export_text

from msv.config import RESULTS_DIR

TOLERANCE = 0.05
PROBE_COLS = ["snr", "snr_model", "snr_breger", "a2_a1", "snr_half",
              "snr_half_model", "amplitude_ppt"]


def contexto(peaks):
    """Features de la posición de cada pico dentro de su propia estrella."""
    peaks = peaks.copy()
    grupo = peaks.groupby("TIC")
    peaks["n_picos"] = grupo.per.transform("size")
    peaks["rango_power"] = grupo.power.rank(ascending=False)
    peaks["rango_plpv"] = grupo.log_pLPV.rank(ascending=False)
    peaks["rango_per"] = grupo.per.rank(ascending=False)
    peaks["es_max_power"] = (peaks.rango_power == 1).astype(int)
    peaks["es_max_plpv"] = (peaks.rango_plpv == 1).astype(int)
    # El power de LS y el del ACF no son la misma cantidad: se normaliza cada
    # uno contra el máximo de SU fuente en ESA estrella.
    por_fuente = peaks.groupby(["TIC", "source"])
    peaks["power_rel"] = peaks.power / por_fuente.power.transform("max")
    peaks["rango_power_fuente"] = por_fuente.power.rank(ascending=False)
    peaks["es_LS"] = (peaks.source == "LS").astype(int)
    peaks["per_rel"] = peaks.per / grupo.per.transform("max")
    peaks["snr_rel"] = peaks.snr / grupo.snr.transform("max")

    # ¿Sobrevive, en el mismo conjunto, un pico al doble o a la mitad? Es la
    # información que ninguna cantidad por-pico contiene y sin la cual el
    # factor 2 no se puede adjudicar.
    tiene_doble, tiene_mitad = [], []
    for _, block in peaks.groupby("TIC"):
        periodos = block.per.values
        for period in periodos:
            tiene_doble.append(int(np.any(np.abs(periodos / (2 * period) - 1)
                                          < TOLERANCE)))
            tiene_mitad.append(int(np.any(np.abs(periodos / (0.5 * period) - 1)
                                          < TOLERANCE)))
    orden = peaks.sort_values("TIC").index
    peaks.loc[orden, "hay_pico_2P"] = tiene_doble
    peaks.loc[orden, "hay_pico_P2"] = tiene_mitad
    return peaks


def seleccionar(peaks, score, nombre):
    """Un pico por estrella, el de mayor `score`; devuelve el reparto de tags."""
    tabla = peaks.assign(_score=score)
    elegidos = tabla.sort_values("_score", ascending=False).groupby("TIC").head(1)
    reparto = elegidos.tag.value_counts(normalize=True) * 100
    return {"regla": nombre, "estrellas": len(elegidos),
            "1:1": round(reparto.get("1:1", 0.0), 1),
            "P/2": round(reparto.get("P/2", 0.0), 1),
            "2P": round(reparto.get("2P", 0.0), 1),
            "otro": round(reparto.get("otro", 0.0), 1)}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--peaks",
                        default=str(RESULTS_DIR / "vsx" / "peaks_con_snr.csv"))
    parser.add_argument("--probe",
                        default=str(RESULTS_DIR / "vsx" /
                                    "probe_peaks_w001_model.csv"))
    parser.add_argument("--max-depth", type=int, nargs="+", default=[2, 3, 4, 5])
    parser.add_argument("--min-leaf", type=int, default=20)
    args = parser.parse_args()

    peaks = pd.read_csv(args.peaks)
    probe = pd.read_csv(args.probe)
    for table in (peaks, probe):
        table["clave"] = [f"{value:.12g}" for value in table.per.values]
    probe = probe.drop_duplicates(subset=["TIC", "sector", "clave"])
    peaks = peaks.drop(columns=[c for c in PROBE_COLS if c in peaks.columns])
    peaks = peaks.merge(
        probe[["TIC", "sector", "clave"] + PROBE_COLS],
        on=["TIC", "sector", "clave"], how="left")
    for column in PROBE_COLS:
        peaks[column] = peaks[column].fillna(0.0)

    peaks = peaks[(peaks.vis_verdict == "ok") & (peaks.clase != "Rndm")]
    peaks = contexto(peaks)
    print(f"{len(peaks)} picos supervivientes de {peaks.TIC.nunique()} "
          f"estrellas OK; {int(peaks.es_vsx.sum())} son el 1:1")

    features = ["power", "prominence", "width", "amplitude", "per", "ciclos",
                "p_ELL", "p_Pulsating", "p_E", "p_LPV", "log_pLPV", "prob",
                "irregular", "snr", "snr_model", "a2_a1", "snr_half",
                "snr_half_model", "amplitude_ppt",
                "n_picos", "rango_power", "rango_plpv", "rango_per",
                "es_max_power", "es_max_plpv", "power_rel",
                "rango_power_fuente", "es_LS", "per_rel", "snr_rel",
                "hay_pico_2P", "hay_pico_P2"]
    X = peaks[features].replace([np.inf, -np.inf], np.nan).fillna(0.0).values
    y = peaks.es_vsx.values.astype(int)
    groups = peaks.TIC.values

    print("\n--- reglas de un solo criterio (referencia) ---")
    referencia = [seleccionar(peaks, peaks.log_pLPV, "log_pLPV (actual)"),
                  seleccionar(peaks, peaks.power, "power"),
                  seleccionar(peaks, peaks.power_rel, "power relativo"),
                  seleccionar(peaks, peaks.snr_model, "SNR del modelo armónico")]
    print(pd.DataFrame(referencia).set_index("regla").to_string())

    print("\n--- árbol, out-of-fold con GroupKFold(5) por TIC ---")
    resultados = []
    partidor = GroupKFold(n_splits=5)
    for depth in args.max_depth:
        scores = np.zeros(len(peaks))
        for entrena, prueba in partidor.split(X, y, groups):
            arbol = DecisionTreeClassifier(max_depth=depth,
                                           min_samples_leaf=args.min_leaf,
                                           class_weight="balanced",
                                           random_state=0)
            arbol.fit(X[entrena], y[entrena])
            scores[prueba] = arbol.predict_proba(X[prueba])[:, 1]
        resultados.append(seleccionar(peaks, scores, f"árbol depth={depth}"))
    print(pd.DataFrame(resultados).set_index("regla").to_string())

    mejor = max(args.max_depth)
    arbol = DecisionTreeClassifier(max_depth=3, min_samples_leaf=args.min_leaf,
                                   class_weight="balanced", random_state=0)
    arbol.fit(X, y)
    importancia = pd.Series(arbol.feature_importances_, index=features)
    print("\n--- importancia (árbol depth=3 sobre TODO, solo para leerlo) ---")
    print(importancia[importancia > 0].sort_values(ascending=False)
          .round(3).to_string())
    print("\n" + export_text(arbol, feature_names=features, decimals=3))


if __name__ == "__main__":
    main()
