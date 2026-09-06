#!/usr/bin/env python
"""Clasificar la ESTRELLA, no el pico: PULS / ECL / ELL / unconstrained.

El pipeline actual pregunta "cual de estos picos es el bueno" y la clase solo
entra como peso. Aca la pregunta cambia: se le da al modelo TODO el conjunto de
picos sobrevivientes de la estrella —cada uno con sus probabilidades de la CNN,
su amplitud, su SNR y su periodo relativo— y la salida es la clase.

Como un modelo sobre conjuntos necesita mas datos de los que hay (359
estrellas), el conjunto se aplana a un vector de largo fijo: los K picos de
mayor SNR ordenados, mas agregados de la estrella. Eso permite usar ML
tradicional y responder si el encuadre de conjunto agrega algo ANTES de invertir
en arquitectura.

La referencia contra la que hay que ganar es el pipeline actual: la clase de la
CNN en el pico de mayor SNR. Si el modelo no le gana a eso, el metodo que ya
tenemos es suficiente.

    PYTHONPATH=src python scripts/vsx/clase_estrella.py --k 5
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

sys.path.insert(0, str(Path(__file__).resolve().parent))
from arbol_features import FAMILIAS, cargar, contexto

from msv.config import RESULTS_DIR

CLASES = ["ECL", "ELL", "PULS", "unconstrained"]
POR_PICO = ["log_per_rel", "power_rel", "log_amp", "log_snr", "a2_a1",
            "p_ELL", "p_E", "p_Pulsating", "p_LPV", "p_Rndm",
            "v_ELL", "v_Pulsating", "es_LS"]
CNN_A_FAMILIA = {"E": "ECL", "ELL": "ELL", "Pulsating": "PULS", "LPV": "PULS"}


def etiqueta(peaks):
    familia = peaks.vsx_token.map(FAMILIAS).fillna("PULS")
    return np.where(peaks.vis_verdict == "unconstrained", "unconstrained",
                    familia)


def vector_por_estrella(peaks, k, con_per, orden="snr"):
    """Aplana el conjunto de picos a un vector de largo fijo por estrella.

    `orden` decide que picos son los K primeros. Por defecto la SNR, que no
    depende de ningun modelo; con `orden="probabilidad"` se usa el ranking del
    selector, que es una eleccion mejor informada pero encadena dos modelos y
    obliga a validar anidado.
    """
    filas = []
    for tic, block in peaks.groupby("TIC", sort=False):
        block = block.sort_values(orden, ascending=False)
        top = block.iloc[0]
        fila = {"TIC": tic, "clase_verdadera": block.clase_verdadera.iloc[0],
                "cnn_top": CNN_A_FAMILIA.get(top.clase, "PULS")}
        fila["n_picos"] = len(block)
        fila["ciclos_top"] = top.ciclos
        fila["frac_LS"] = float((block.source == "LS").mean())
        amplitudes = block.amplitude_ppt.clip(lower=0)
        fila["amp_frac_top"] = float(
            amplitudes.iloc[0] / amplitudes.sum()) if amplitudes.sum() > 0 else 0.0
        fila["rango_snr"] = float(np.log10(max(top.snr, 1e-3)
                                           / max(block.snr.min(), 1e-3)))
        if con_per:
            fila["log_per_top"] = float(np.log10(max(top.per, 1e-6)))
        # medias sobre TODO el conjunto, no solo los K: la cola tambien informa
        for clase in ["ELL", "E", "Pulsating", "LPV", "Rndm"]:
            fila[f"media_p_{clase}"] = float(block[f"p_{clase}"].mean())
            fila[f"max_p_{clase}"] = float(block[f"p_{clase}"].max())
        for posicion in range(k):
            if posicion < len(block):
                peak = block.iloc[posicion]
                valores = {
                    "log_per_rel": float(np.log10(max(peak.per, 1e-6)
                                                  / max(top.per, 1e-6))),
                    "power_rel": peak.power_rel,
                    "log_amp": float(np.log10(max(peak.amplitude_ppt, 1e-4))),
                    "log_snr": float(np.log10(max(peak.snr, 1e-3))),
                    "a2_a1": peak.a2_a1,
                    "p_ELL": peak.p_ELL, "p_E": peak.p_E,
                    "p_Pulsating": peak.p_Pulsating, "p_LPV": peak.p_LPV,
                    "p_Rndm": peak.p_Rndm,
                    "v_ELL": peak.v_ELL, "v_Pulsating": peak.v_Pulsating,
                    "es_LS": float(peak.source == "LS"),
                }
            else:
                valores = {name: 0.0 for name in POR_PICO}
            for name, value in valores.items():
                fila[f"{name}_{posicion}"] = value
        filas.append(fila)
    return pd.DataFrame(filas)


def evaluar(X, y, make_model, repeticiones=5):
    aciertos, balanceados, matrices = [], [], []
    for semilla in range(repeticiones):
        prediccion = np.empty(len(y), dtype=object)
        particion = StratifiedKFold(5, shuffle=True, random_state=semilla)
        for entrena, prueba in particion.split(X, y):
            modelo = make_model()
            modelo.fit(X[entrena], y[entrena])
            prediccion[prueba] = modelo.predict(X[prueba])
        aciertos.append(100 * float((prediccion == y).mean()))
        balanceados.append(100 * balanced_accuracy_score(y, prediccion))
        matrices.append(confusion_matrix(y, prediccion, labels=CLASES))
    return (float(np.mean(aciertos)), float(np.std(aciertos)),
            float(np.mean(balanceados)), float(np.std(balanceados)),
            sum(matrices) / len(matrices))


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--probe",
                        default=str(RESULTS_DIR / "vsx" /
                                    "probe_peaks_w001_1pass.csv"))
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--repeticiones", type=int, default=5)
    parser.add_argument("--sin-per", action="store_true",
                        help="sin el periodo absoluto del pico mas fuerte")
    args = parser.parse_args()

    peaks = contexto(cargar(args.probe, verdicts=("ok", "unconstrained")))
    peaks["clase_verdadera"] = etiqueta(peaks)
    tabla = vector_por_estrella(peaks, args.k, not args.sin_per)
    columnas = [c for c in tabla.columns
                if c not in ("TIC", "clase_verdadera", "cnn_top")]
    X = tabla[columnas].replace([np.inf, -np.inf], np.nan).fillna(0.0).values
    y = tabla.clase_verdadera.values

    print(tabla.clase_verdadera.value_counts().to_string())
    print(f"\n{len(columnas)} features por estrella "
          f"(K={args.k} picos + agregados)"
          f"{', sin per absoluto' if args.sin_per else ''}")

    base = tabla.cnn_top.values
    print("\n--- referencia: la clase de la CNN en el pico de mayor SNR "
          "(el pipeline actual) ---")
    print(f"acierto {100 * (base == y).mean():.1f}%   "
          f"balanceado {100 * balanced_accuracy_score(y, base):.1f}%")
    print(pd.DataFrame(confusion_matrix(y, base, labels=CLASES),
                       index=CLASES, columns=CLASES).to_string())

    modelos = {
        "gradient boosting": lambda: GradientBoostingClassifier(
            n_estimators=200, max_depth=3, random_state=0),
        "random forest": lambda: RandomForestClassifier(
            n_estimators=400, min_samples_leaf=2, class_weight="balanced",
            random_state=0, n_jobs=-1),
        "balanced RF": lambda: BalancedRandomForestClassifier(
            n_estimators=400, min_samples_leaf=2, random_state=0, n_jobs=-1),
        "SVM rbf": lambda: make_pipeline(
            StandardScaler(),
            SVC(kernel="rbf", C=10.0, class_weight="balanced", random_state=0)),
        "PCA(20) + SVM rbf": lambda: make_pipeline(
            StandardScaler(), PCA(n_components=20, random_state=0),
            SVC(kernel="rbf", C=10.0, class_weight="balanced", random_state=0)),
        "PCA(20) + logistica": lambda: make_pipeline(
            StandardScaler(), PCA(n_components=20, random_state=0),
            LogisticRegression(max_iter=2000, class_weight="balanced")),
        "PCA(20) + RF": lambda: make_pipeline(
            StandardScaler(), PCA(n_components=20, random_state=0),
            RandomForestClassifier(n_estimators=400, min_samples_leaf=2,
                                   class_weight="balanced", random_state=0,
                                   n_jobs=-1)),
    }
    filas, guardadas = [], {}
    for nombre, make in modelos.items():
        acierto, sigma, balanceado, sigma_b, matriz = evaluar(
            X, y, make, args.repeticiones)
        filas.append({"modelo": nombre,
                      "acierto": f"{acierto:.1f} ± {sigma:.1f}",
                      "balanceado": f"{balanceado:.1f} ± {sigma_b:.1f}"})
        guardadas[nombre] = (matriz, balanceado)
    print(f"\n--- clasificacion por estrella, {args.repeticiones} semillas "
          "de StratifiedKFold(5) ---")
    print(pd.DataFrame(filas).set_index("modelo").to_string())

    por_acierto = max(guardadas, key=lambda name: guardadas[name][0].trace())
    por_balance = max(guardadas, key=lambda name: guardadas[name][1])
    for etiqueta_modelo in dict.fromkeys([por_acierto, por_balance]):
        matriz = pd.DataFrame(guardadas[etiqueta_modelo][0],
                              index=CLASES, columns=CLASES)
        print(f"\n=== matriz de confusion, {etiqueta_modelo} "
              "(filas = verdad, % de fila) ===")
        print((100 * matriz.div(matriz.sum(axis=1), axis=0)).round(1).to_string())


if __name__ == "__main__":
    main()
