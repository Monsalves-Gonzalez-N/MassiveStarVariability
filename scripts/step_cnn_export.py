#!/usr/bin/env python
"""Paso 1/3 (env CNN_TESS): peaks parquet -> .npz de entrada para la CNN.

El .npz existe porque la CNN (TensorFlow, env tf_env) y el BRF (sklearn 1.0.2,
env CNN_TESS) no caben en el mismo entorno, y tf_env no tiene pyarrow: numpy es
el único formato que los dos leen.

    python scripts/step_cnn_export.py results/peaks_review50.parquet results/cnn_input.npz
"""
import argparse

import numpy as np
import pandas as pd

META_COLS = ["TIC", "sector", "source", "per", "power", "prominence", "width",
             "amplitude"]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("peaks", help="parquet de run_peaks.py (con hist2d)")
    parser.add_argument("out", help=".npz de salida")
    args = parser.parse_args()

    peaks = pd.read_parquet(args.peaks).reset_index(drop=True)
    cube = np.stack([np.asarray(hist, dtype=np.float32).reshape(32, 32)
                     for hist in peaks["hist2d"].values])[..., np.newaxis]
    np.savez_compressed(
        args.out, X=cube,
        **{column: peaks[column].to_numpy() for column in META_COLS})
    print(f"-> {args.out}   X{cube.shape}  {peaks.groupby(['TIC', 'sector']).ngroups} pares")


if __name__ == "__main__":
    main()
