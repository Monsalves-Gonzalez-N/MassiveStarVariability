#!/usr/bin/env python
"""Paso 3/3 variante: PROMEDIAR las pasadas de la CNN y correr el BRF UNA vez.

`step_brf.py` corre el BRF sobre cada pasada MC y promedia la salida;
acá se promedia la entrada. Como el BRF es un bosque —escalonado, no lineal—
`media(BRF(x))` no es `BRF(media(x))`, y la diferencia no es de segundo orden:
sobre los 857 picos de review50 el argmax cambia en el 5% de los casos, y la
cuenta de ELL baja de 56 a 36 (200 pasadas) o de ~55 a ~40 (20 pasadas).

El precio es que NO hay distribución de salida: una sola llamada al BRF da un
solo vector de probabilidades. No existen `sigma`, `lo`/`hi`, `vote` ni
`instability`, así que tampoco el gate de estabilidad de ELL — que se apoya en
el voto entre pasadas. El CSV trae solo `p_<grupo>`, y los PDF de revisión
dibujan las barras sin bigote.

Contrato de salida compartido con step_brf.py en lo que la cascada necesita:
`brf_class`, `brf_prob`, `clase`, `prob` y `p_<grupo>`.

    MSV_BRF=~/Dropbox/MassiveStarVariability/models/balanced_random_forest_model.joblib \
      python scripts/step_brf_mean.py results/cnn_input.npz results/cnn_mc20.npz results/clasificacion_mean20.csv
"""
import argparse

import numpy as np
import pandas as pd

from msv import config
from msv.classify_brf import brf_features, group_probs, load_brf
from msv.config import CLASS_NAMES

META_COLS = ["TIC", "sector", "source", "per", "power", "prominence", "width",
             "amplitude"]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("npz_in", help=".npz de step_cnn_export.py")
    parser.add_argument("npz_mc", help=".npz de step_cnn.py (p_mc)")
    parser.add_argument("out", help="CSV de salida")
    parser.add_argument("--brf", default=str(config.BRF_MODEL))
    args = parser.parse_args()

    data = np.load(args.npz_in, allow_pickle=True)
    table = pd.DataFrame({column: data[column] for column in META_COLS})
    probabilities = np.load(args.npz_mc)["p_mc"]           # (n_iter, N, 8)
    per = table["per"].to_numpy(dtype=float)
    amplitude = table["amplitude"].to_numpy(dtype=float)
    number_of_peaks = len(table)

    brf = load_brf(args.brf)
    mean_cnn = probabilities.mean(axis=0)                  # (N, 8)

    valid = np.isfinite(per) & np.isfinite(amplitude)      # amp NaN = flujo <= 0
    fine = np.full((number_of_peaks, len(CLASS_NAMES)), np.nan)
    features = brf_features(mean_cnn, per, amplitude, brf)
    fine[valid] = brf.predict_proba(features[valid])

    fine_winner = np.where(np.isnan(fine), -np.inf, fine).argmax(axis=1)
    rows = np.arange(number_of_peaks)
    table["brf_class"] = [CLASS_NAMES[index] if ok else None
                          for index, ok in zip(fine_winner, valid)]
    table["brf_prob"] = np.where(valid, fine[rows, fine_winner], np.nan)

    grouped, group_names = group_probs(fine)               # (N, n_grupos)
    winner = np.where(np.isnan(grouped), -np.inf, grouped).argmax(axis=1)
    table["clase"] = [group_names[index] if ok else None
                      for index, ok in zip(winner, valid)]
    table["prob"] = np.where(valid, grouped[rows, winner], np.nan)
    for index, name in enumerate(group_names):
        table[f"p_{name}"] = grouped[:, index]

    table.to_csv(args.out, index=False)
    print(f"-> {args.out}   {number_of_peaks} peaks, "
          f"{table.groupby(['TIC', 'sector']).ngroups} pares, "
          f"{probabilities.shape[0]} pasadas promediadas")
    print(table["clase"].value_counts().to_string())


if __name__ == "__main__":
    main()
