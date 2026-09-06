#!/usr/bin/env python
"""La distribucion de MC-dropout como input, en vez de 5 probabilidades.

Hoy cada pico entra al selector con 5 numeros: las 8 clases de la CNN
colapsadas a 5 grupos, de UNA pasada determinista. Con `--mc-dropout` la red
entrega `(20, N, 8)`: la distribucion completa de la clase bajo dropout.

Las 20 pasadas son INTERCAMBIABLES — la pasada 3 no significa nada distinto de
la 17 — asi que aplanar 20x8=160 y hacerle PCA ajustaria direcciones del indice
de pasada, que es ruido. El resumen tiene que ser invariante al orden: media,
desviacion y dos cuantiles por clase, 8x4 = 32 columnas.

Se comparan el bloque CNN actual, las 32 crudas, y PCA sobre las 32.

    PYTHONPATH=src python scripts/vsx/arbol_mc.py
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.pipeline import make_pipeline

sys.path.insert(0, str(Path(__file__).resolve().parent))
from arbol_features import (CICLOS, CNN, PERIODOGRAMA, SNR_1PASADA, cargar,
                            contexto, familia, out_of_fold, acierto,
                            acierto_por_familia)

from msv.config import CLASS_NAMES, RESULTS_DIR

RESUMEN = ["media", "std", "p10", "p90"]


def columnas_mc():
    return [f"mc_{estadistico}_{clase}"
            for estadistico in RESUMEN for clase in CLASS_NAMES]


def resumen_mc(npz_path, clasificacion_path):
    """Resumen invariante al orden de las pasadas, por pico."""
    p_mc = np.load(npz_path, allow_pickle=True)["p_mc"]
    if p_mc.shape[0] < 2:
        raise SystemExit(f"{npz_path} tiene n_iter={p_mc.shape[0]}: "
                         "hay que correr step_cnn.py con --mc-dropout")
    tabla = pd.read_csv(clasificacion_path,
                        usecols=["TIC", "sector", "per"])
    if len(tabla) != p_mc.shape[1]:
        raise SystemExit("la clasificacion y p_mc no tienen el mismo largo")
    estadisticos = {
        "media": p_mc.mean(axis=0),
        "std": p_mc.std(axis=0),
        "p10": np.percentile(p_mc, 10, axis=0),
        "p90": np.percentile(p_mc, 90, axis=0),
    }
    for estadistico, valores in estadisticos.items():
        for posicion, clase in enumerate(CLASS_NAMES):
            tabla[f"mc_{estadistico}_{clase}"] = valores[:, posicion]
    tabla["clave"] = [f"{value:.12g}" for value in tabla.per.values]
    return tabla.drop(columns=["per"]).drop_duplicates(
        subset=["TIC", "sector", "clave"])


def con_pca(columnas, n_componentes, make_final):
    """PCA solo sobre el bloque MC; el resto pasa sin tocar."""
    indices_mc = [posicion for posicion, name in enumerate(columnas)
                  if name.startswith("mc_")]
    otros = [posicion for posicion, name in enumerate(columnas)
             if not name.startswith("mc_")]
    return make_pipeline(
        ColumnTransformer([("pca", PCA(n_components=n_componentes,
                                       random_state=0), indices_mc),
                           ("resto", "passthrough", otros)]),
        make_final())


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--probe",
                        default=str(RESULTS_DIR / "vsx" /
                                    "probe_peaks_w001_1pass.csv"))
    parser.add_argument("--mc", default=str(RESULTS_DIR / "vsx" /
                                            "cnn_probs_w001_mc20.npz"))
    parser.add_argument("--clasificacion",
                        default=str(RESULTS_DIR / "vsx" /
                                    "clasificacion_w001.csv"))
    parser.add_argument("--repeticiones", type=int, default=5)
    args = parser.parse_args()

    peaks = contexto(cargar(args.probe))
    peaks["familia"] = familia(peaks)
    mc = resumen_mc(args.mc, args.clasificacion)
    antes = len(peaks)
    peaks = peaks.merge(mc, on=["TIC", "sector", "clave"], how="left")
    columnas_faltantes = peaks[columnas_mc()].isna().all(axis=1).sum()
    print(f"{antes} picos, {columnas_faltantes} sin resumen MC")
    peaks[columnas_mc()] = peaks[columnas_mc()].fillna(0.0)

    base = PERIODOGRAMA + CICLOS + SNR_1PASADA
    conjuntos = {
        "CNN actual (5 probs + contexto)": base + CNN,
        "MC 32 crudas": base + columnas_mc(),
        "MC 32 + CNN actual": base + CNN + columnas_mc(),
    }
    modelos = {
        "random forest": lambda: RandomForestClassifier(
            n_estimators=400, min_samples_leaf=5, random_state=0, n_jobs=-1),
        "gradient boosting": lambda: GradientBoostingClassifier(
            n_estimators=200, max_depth=3, random_state=0),
    }
    filas = []
    for nombre_conjunto, columnas in conjuntos.items():
        fila = {"features": nombre_conjunto, "n": len(columnas)}
        for nombre_modelo, make in modelos.items():
            media, sigma = out_of_fold(peaks, columnas, make,
                                       args.repeticiones)
            fila[nombre_modelo] = f"{media:.1f} ± {sigma:.1f}"
        filas.append(fila)

    columnas = base + columnas_mc()
    for n_componentes in (4, 8, 16):
        fila = {"features": f"PCA({n_componentes}) sobre las MC",
                "n": len(base) + n_componentes}
        for nombre_modelo, make in modelos.items():
            media, sigma = out_of_fold(
                peaks, columnas,
                lambda n=n_componentes, m=make: con_pca(columnas, n, m),
                args.repeticiones)
            fila[nombre_modelo] = f"{media:.1f} ± {sigma:.1f}"
        filas.append(fila)

    print(f"\nacierto 1:1 out-of-fold [%], GroupKFold(5) por TIC, "
          f"{args.repeticiones} semillas")
    print(pd.DataFrame(filas).set_index("features").to_string())
    print(f"\nreferencia log_pLPV: {acierto(peaks, peaks.log_pLPV)[0]}%")


if __name__ == "__main__":
    main()
