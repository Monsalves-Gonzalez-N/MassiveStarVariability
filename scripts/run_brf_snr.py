#!/usr/bin/env python
"""CNN MC-dropout + BRF + gate de incertidumbre -> CSV por peak.

Por cada peak del parquet de entrada (schema de run_peaks.py, con hist2d y
amplitude): n_iter pasadas de la CNN con Dropout activo, BRF sobre cada
pasada, y métricas de incertidumbre (sigma_top, entropy, sigma_brf,
instability). El gate relabela a 'Rndm' los peaks con incertidumbre alta
antes de la cascada Path-2.

Uso (env con TensorFlow):
  python scripts/run_brf_snr.py --peaks results/peaks.parquet \
      --model Number_DST --n-iter 50 --out results/brf_mc_peaks_Number_DST.csv

  --all-models recorre los 7 checkpoints y escribe un CSV por modelo.
  --gate-col: sigma_top (spec del .md, umbral SIGMA_MAX=0.12) o instability
      (σ del BRF: en el benchmark OGLE separa mejor los FP, AUC~0.86).
"""
import argparse
import sys

import pandas as pd

from msv import config
from msv.classify_brf import apply_gate, classify_peaks, load_brf, load_cnn


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--peaks", default=str(config.RESULTS_DIR / "peaks.parquet"),
                    help="parquet de peaks (salida de run_peaks.py, con hist2d)")
    ap.add_argument("--model", default=config.DEFAULT_MODEL,
                    help=f"nombre del training en WEIGHTS_DIR (default {config.DEFAULT_MODEL})")
    ap.add_argument("--all-models", action="store_true",
                    help=f"correr los 7 modelos: {config.MODELS}")
    ap.add_argument("--brf", default=str(config.BRF_MODEL))
    ap.add_argument("--n-iter", type=int, default=50, help="pasadas MC-dropout")
    ap.add_argument("--sigma-max", type=float, default=config.SIGMA_MAX,
                    help="umbral del gate sobre --gate-col")
    ap.add_argument("--gate-col", default="sigma_top",
                    choices=["sigma_top", "sigma_brf", "instability", "entropy"])
    ap.add_argument("--gate-mode", default="relabel", choices=["relabel", "drop", "none"],
                    help="'none' escribe solo las métricas, sin aplicar gate")
    ap.add_argument("--out", default=None,
                    help="CSV de salida (default results/brf_mc_peaks_<model>.csv)")
    ap.add_argument("--batch-size", type=int, default=512)
    args = ap.parse_args()

    peaks = pd.read_parquet(args.peaks)
    print(f"Peaks: {len(peaks)}  (pares: {peaks.groupby(['TIC','sector']).ngroups})")
    brf = load_brf(args.brf)

    models = config.MODELS if args.all_models else [args.model]
    for name in models:
        print(f"\n=== {name} ===")
        model = load_cnn(name)
        df = classify_peaks(peaks, model, brf, n_iter=args.n_iter,
                            batch_size=args.batch_size)
        if args.gate_mode != "none":
            n_before = (df["brf_class"].isin(config.PERIODIC)).sum()
            df_gated = apply_gate(df, sigma_max=args.sigma_max,
                                  mode=args.gate_mode, sigma_col=args.gate_col)
            n_after = (df_gated["brf_class"].isin(config.PERIODIC)).sum()
            df_gated["brf_class_raw"] = df["brf_class"].reindex(df_gated.index)
            df = df_gated
            print(f"gate {args.gate_col} > {args.sigma_max} ({args.gate_mode}): "
                  f"peaks periódicos {n_before} -> {n_after}")

        out = args.out or str(config.RESULTS_DIR / f"brf_mc_peaks_{name}.csv")
        df.to_csv(out, index=False)
        print(f"-> {out}  ({len(df)} filas)")
        print(df["brf_class"].value_counts().to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
