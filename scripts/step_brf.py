#!/usr/bin/env python
"""Paso 3/3 (env CNN_TESS): BRF sobre cada pasada MC -> CSV de clasificación.

La clase sale de la MEDIA de las `n_iter` pasadas, con `sigma` y
`instability` (fracción de pasadas donde cambia la clase ganadora). La media y
no la mediana porque las medianas por clase no suman 1, así que no forman una
distribución sobre la que tomar un argmax.

Para GRAFICAR, en cambio, la media ± σ no sirve: la distribución por pico es
bimodal (el 66% de los picos ELL tiene pasadas por debajo de 0.1 y por encima
de 0.9), así que el centro cae en un valle sin masa y σ se sale de [0, 1]. El
CSV trae por eso tres resúmenes por grupo, y cada uno dice algo distinto:
  p_/s_        media y desviación — para el argmax que decide la clase.
  med_/q1_/q3_ mediana y cuartiles — acotados a [0, 1] y sin suponer simetría.
  v_           fracción de pasadas en que el grupo gana — el único resumen que
               describe de verdad una distribución bimodal, porque es la masa
               de cada modo y no un centro.

ELL pasa además por un gate de estabilidad: pierde el label si no gana el
argmax en al menos `--ell-vote-min` de las pasadas, y cede al subcampeón. El
BRF aporta `amplitud` y `per`, las dos fuera de su rango de entrenamiento, y
ahí aterriza en una constante sesgada hacia ELL — ELL gana el promedio sin
dominar ninguna pasada (media 0.673 / voto 0.715, contra Pulsating 0.546 /
0.850). Con `--ell-vote-min 0` el gate se desactiva.

El agrupamiento `Pulsating` se aplica sumando columnas de la SALIDA del BRF
ANTES del argmax (`classify_brf.group_probs`): tomar el argmax primero perdería
una estrella con M=0.30, DST=0.25, RR=0.10 frente a ELL=0.32. El CSV trae la
distribución completa por grupo, `p_<grupo>` y su dispersión `s_<grupo>`, que
es lo que grafica build_lc_prob_review.py.

    MSV_BRF=~/Dropbox/MassiveStarVariability/models/balanced_random_forest_model.joblib \
      python scripts/step_brf.py results/cnn_input.npz results/cnn_mc.npz results/clasificacion.csv
"""
import argparse

import numpy as np
import pandas as pd

from msv import config
from msv.classify_brf import (aggregate_mc, apply_gate, apply_vote_stability_gate,
                              brf_mc_probs, group_probs, load_brf, vote_fractions)

META_COLS = ["TIC", "sector", "source", "per", "power", "prominence", "width",
             "amplitude"]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("npz_in", help=".npz de step_cnn_export.py")
    parser.add_argument("npz_mc", help=".npz de step_cnn.py (p_mc)")
    parser.add_argument("out", help="CSV de salida")
    parser.add_argument("--brf", default=str(config.BRF_MODEL))
    parser.add_argument("--gate-col", default="instability",
                        choices=["sigma_top", "sigma_brf", "instability", "entropy"])
    parser.add_argument("--sigma-max", type=float, default=config.SIGMA_MAX)
    parser.add_argument("--gate-mode", default="none",
                        choices=["relabel", "drop", "none"])
    parser.add_argument("--ell-vote-min", type=float, default=0.5,
                        help="fracción mínima de pasadas que ELL debe ganar "
                             "para conservar el label; 0 desactiva el gate")
    args = parser.parse_args()

    data = np.load(args.npz_in, allow_pickle=True)
    table = pd.DataFrame({column: data[column] for column in META_COLS})
    probabilities = np.load(args.npz_mc)["p_mc"]
    per = table["per"].to_numpy()
    amplitude = table["amplitude"].to_numpy()

    brf = load_brf(args.brf)
    for column, values in aggregate_mc(probabilities, per, amplitude, brf).items():
        table[column] = values

    # --- distribución por grupo: media, dispersión y voto entre pasadas -----
    per_pass, _ = brf_mc_probs(probabilities, per, amplitude, brf)
    grouped, group_names = group_probs(per_pass)          # (n_iter, N, n_grupos)
    mean = grouped.mean(axis=0)
    dispersion = grouped.std(axis=0)
    low, high = np.percentile(grouped, [16, 84], axis=0)
    median = np.median(grouped, axis=0)
    first_quartile, third_quartile = np.percentile(grouped, [25, 75], axis=0)
    votes = vote_fractions(grouped)
    scores = np.where(np.isnan(mean), -np.inf, mean)
    winner = apply_vote_stability_gate(scores, votes, group_names,
                                       vote_min=args.ell_vote_min)
    rows = np.arange(len(table))

    table["clase"] = [group_names[w] for w in winner]
    table["prob"] = mean[rows, winner]
    table["sigma"] = dispersion[rows, winner]
    table["lo"] = low[rows, winner]
    table["hi"] = high[rows, winner]
    table["vote"] = votes[rows, winner]
    table["mediana"] = median[rows, winner]
    table["q1"] = first_quartile[rows, winner]
    table["q3"] = third_quartile[rows, winner]
    table["clase_sin_gate"] = [group_names[w] for w in scores.argmax(axis=1)]
    table["instability_clase"] = (grouped.argmax(2) != winner[None, :]).mean(0)
    for index, name in enumerate(group_names):
        table[f"p_{name}"] = mean[:, index]
        table[f"s_{name}"] = dispersion[:, index]
        table[f"lo_{name}"] = low[:, index]
        table[f"hi_{name}"] = high[:, index]
        table[f"v_{name}"] = votes[:, index]
        table[f"med_{name}"] = median[:, index]
        table[f"q1_{name}"] = first_quartile[:, index]
        table[f"q3_{name}"] = third_quartile[:, index]

    if args.gate_mode != "none":
        table = apply_gate(table, sigma_max=args.sigma_max, mode=args.gate_mode,
                           sigma_col=args.gate_col)

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
