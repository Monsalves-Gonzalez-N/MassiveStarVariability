#!/usr/bin/env python
"""Cambiar SOLO las columnas de la CNN en `peaks_con_snr.csv`.

Los picos, las etiquetas de VSX y la revision visual no dependen de como se
normaliza el hist2d, asi que para comparar normalizaciones basta con reemplazar
el bloque de la CNN y dejar el resto intacto.

    PYTHONPATH=src python scripts/vsx/build_peaks_norm.py \
        --clasificacion results/vsx/clasificacion_w001_minmax.csv \
        --out results/vsx/peaks_con_snr_minmax.csv
"""
import argparse

import numpy as np
import pandas as pd

from msv.config import RESULTS_DIR

FLOOR = 1e-12
GRUPOS = ["ELL", "Pulsating", "E", "LPV", "Rndm"]
PREFIJOS = ["p_", "s_", "med_", "q1_", "q3_", "lo_", "hi_", "v_"]


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--peaks", default=str(RESULTS_DIR / "vsx" /
                                               "peaks_con_snr.csv"))
    parser.add_argument("--clasificacion", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    peaks = pd.read_csv(args.peaks)
    columnas_cnn = ([f"{prefijo}{grupo}" for prefijo in PREFIJOS
                     for grupo in GRUPOS]
                    + ["clase", "prob", "sigma", "cnn_class", "log_pLPV",
                       "irregular"])
    peaks = peaks.drop(columns=[c for c in columnas_cnn if c in peaks.columns])

    nueva = pd.read_csv(args.clasificacion)
    nueva["clave"] = [f"{v:.12g}" for v in nueva.per.values]
    disponibles = [c for c in columnas_cnn if c in nueva.columns]
    nueva = nueva[["TIC", "sector", "clave"] + disponibles].drop_duplicates(
        subset=["TIC", "sector", "clave"])

    peaks["clave"] = [f"{v:.12g}" for v in peaks.per.values]
    unidas = peaks.merge(nueva, on=["TIC", "sector", "clave"], how="inner")
    if "log_pLPV" not in unidas.columns:
        unidas["log_pLPV"] = -np.log10(np.clip(unidas.p_LPV.values, FLOOR, 1))
    unidas.to_csv(args.out, index=False)
    print(f"-> {args.out}  ({len(unidas)} picos de {len(peaks)}; "
          f"{unidas.TIC.nunique()} estrellas)")


if __name__ == "__main__":
    main()
