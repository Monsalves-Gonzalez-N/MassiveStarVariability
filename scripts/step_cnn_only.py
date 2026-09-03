#!/usr/bin/env python
"""Paso 3/3 alternativo: clasificación con la CNN SOLA, sin pasar por el BRF.

Mismo contrato de salida que `step_brf.py` — `clase`, `prob`, `sigma`, y la
distribución completa `p_<grupo>` / `s_<grupo>` — para que los PDF de revisión
y la cascada funcionen sin cambios.

Por qué existe: el BRF aporta sobre la CNN dos features, `amplitud` y `per`, y
las dos están fuera del rango en que fue entrenado. Barriendo la amplitud con
las probabilidades CNN fijas, su salida es idéntica a tres decimales entre
0.005 y 0.05 mag, y el 86% de los picos de TESS caen por debajo de 0.1 mag. La
constante a la que aterriza no es neutra (ELL 0.51 contra Pulsating 0.23), así
que en este régimen el BRF no discrimina: agrega un corrimiento fijo hacia ELL.
Esta variante mide qué queda si se lo saca.

Las columnas `brf_class` / `brf_prob` se escriben con el argmax FINO de la CNN
(sin agrupar) porque `cascade.path2_cascade` las lee por nombre. Son CNN, no
BRF, pese al nombre.

El estimador central es la MEDIA entre pasadas (las medianas por clase no
suman 1 y σ es la dispersión alrededor de la media); el intervalo va como
percentiles `lo_`/`hi_` (p16/p84). ELL pasa por el mismo gate de estabilidad
que en step_brf.py, aunque acá tiene poco que hacer: sin el BRF la
inestabilidad de ELL ya cae de 0.265 a 0.143.

    python scripts/step_cnn_only.py results/cnn_input.npz results/cnn_mc.npz results/clasificacion_cnn.csv
"""
import argparse

import numpy as np
import pandas as pd

from msv.classify_brf import (apply_vote_stability_gate, group_probs,
                              vote_fractions)
from msv.config import CLASS_NAMES

META_COLS = ["TIC", "sector", "source", "per", "power", "prominence", "width",
             "amplitude"]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("npz_in", help=".npz de step_cnn_export.py")
    parser.add_argument("npz_mc", help=".npz de step_cnn.py (p_mc)")
    parser.add_argument("out", help="CSV de salida")
    parser.add_argument("--ell-vote-min", type=float, default=0.5,
                        help="fracción mínima de pasadas que ELL debe ganar "
                             "para conservar el label; 0 desactiva el gate")
    args = parser.parse_args()

    data = np.load(args.npz_in, allow_pickle=True)
    table = pd.DataFrame({column: data[column] for column in META_COLS})
    probabilities = np.load(args.npz_mc)["p_mc"]          # (n_iter, N, 8)

    mean_probabilities = probabilities.mean(axis=0)
    fine_winner = mean_probabilities.argmax(axis=1)
    rows = np.arange(len(table))
    table["brf_class"] = [CLASS_NAMES[index] for index in fine_winner]
    table["brf_prob"] = mean_probabilities[rows, fine_winner]
    table["sigma_brf"] = probabilities.std(axis=0)[rows, fine_winner]
    table["instability"] = (probabilities.argmax(2) != fine_winner[None, :]).mean(0)

    table["entropy"] = -(mean_probabilities
                         * np.log(mean_probabilities + 1e-12)).sum(axis=1)
    table["sigma_top"] = probabilities.std(axis=0)[rows, fine_winner]

    grouped, group_names = group_probs(probabilities)     # (n_iter, N, n_grupos)
    mean = grouped.mean(axis=0)
    dispersion = grouped.std(axis=0)
    low, high = np.percentile(grouped, [16, 84], axis=0)
    votes = vote_fractions(grouped)
    scores = np.where(np.isnan(mean), -np.inf, mean)
    winner = apply_vote_stability_gate(scores, votes, group_names,
                                       vote_min=args.ell_vote_min)

    table["clase"] = [group_names[index] for index in winner]
    table["prob"] = mean[rows, winner]
    table["sigma"] = dispersion[rows, winner]
    table["lo"] = low[rows, winner]
    table["hi"] = high[rows, winner]
    table["vote"] = votes[rows, winner]
    table["clase_sin_gate"] = [group_names[index]
                               for index in scores.argmax(axis=1)]
    table["instability_clase"] = (grouped.argmax(2) != winner[None, :]).mean(0)
    for index, name in enumerate(group_names):
        table[f"p_{name}"] = mean[:, index]
        table[f"s_{name}"] = dispersion[:, index]
        table[f"lo_{name}"] = low[:, index]
        table[f"hi_{name}"] = high[:, index]
        table[f"v_{name}"] = votes[:, index]

    table.to_csv(args.out, index=False)
    print(f"-> {args.out}   {len(table)} peaks, "
          f"{table.groupby(['TIC', 'sector']).ngroups} pares")
    print(table["clase"].value_counts().to_string())
    gated = table["clase"] != table["clase_sin_gate"]
    if gated.any():
        print(f"gate de estabilidad ELL (voto < {args.ell_vote_min}): "
              f"{gated.sum()} picos reasignados")
        print(table.loc[gated, "clase"].value_counts().to_string())


if __name__ == "__main__":
    main()
