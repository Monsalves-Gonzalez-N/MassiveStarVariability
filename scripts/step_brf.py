#!/usr/bin/env python
"""Paso 3/3 (env CNN_TESS): BRF sobre cada pasada MC -> CSV de clasificación.

La clase sale de la MEDIANA de las `n_iter` pasadas, con `sigma` y
`instability` (fracción de pasadas donde cambia la clase ganadora).

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
from msv.classify_brf import aggregate_mc, apply_gate, brf_mc_probs, group_probs, load_brf

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
    args = parser.parse_args()

    data = np.load(args.npz_in, allow_pickle=True)
    table = pd.DataFrame({column: data[column] for column in META_COLS})
    probabilities = np.load(args.npz_mc)["p_mc"]
    per = table["per"].to_numpy()
    amplitude = table["amplitude"].to_numpy()

    brf = load_brf(args.brf)
    for column, values in aggregate_mc(probabilities, per, amplitude, brf).items():
        table[column] = values

    # --- distribución por grupo: mediana y dispersión entre pasadas ---------
    per_pass, _ = brf_mc_probs(probabilities, per, amplitude, brf)
    grouped, group_names = group_probs(per_pass)          # (n_iter, N, n_grupos)
    median = np.median(grouped, axis=0)
    dispersion = grouped.std(axis=0)
    winner = np.argmax(np.where(np.isnan(median), -1.0, median), axis=1)
    rows = np.arange(len(table))

    table["clase"] = [group_names[w] for w in winner]
    table["prob"] = median[rows, winner]
    table["sigma"] = dispersion[rows, winner]
    table["instability_clase"] = (grouped.argmax(2) != winner[None, :]).mean(0)
    for index, name in enumerate(group_names):
        table[f"p_{name}"] = median[:, index]
        table[f"s_{name}"] = dispersion[:, index]

    if args.gate_mode != "none":
        table = apply_gate(table, sigma_max=args.sigma_max, mode=args.gate_mode,
                           sigma_col=args.gate_col)

    table.to_csv(args.out, index=False)
    print(f"-> {args.out}   {len(table)} peaks, "
          f"{table.groupby(['TIC', 'sector']).ngroups} pares")
    print(table["clase"].value_counts().to_string())


if __name__ == "__main__":
    main()
