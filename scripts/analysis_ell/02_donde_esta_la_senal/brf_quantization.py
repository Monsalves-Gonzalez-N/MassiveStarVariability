"""Why the BRF hides p_LPV: a forest vote has no dynamic range below 1/n_trees.

The CNN softmax spans many orders of magnitude; the BRF output is a fraction
of trees, so anything under one tree's worth of vote becomes exactly zero.
That is the whole mechanism behind "p_LPV only shows up under rank": rank was
one of the few normalizations that pushed p_LPV above the forest's floor.
"""
import numpy as np
import pandas as pd

from _common import (brf_grouped, cnn_grouped, load_brf, load_passes,
                     load_peaks)

peaks = load_peaks()
brf = load_brf()
print(f"arboles en el BRF: {brf.n_estimators}   "
      f"resolucion minima de una probabilidad: {1 / brf.n_estimators:.3e}")

rows = []
for normalization in ["log", "min_max", "binary", "rank"]:
    passes = load_passes(normalization)
    cnn_block, names = cnn_grouped(passes)
    brf_block, _ = brf_grouped(passes, peaks, brf)
    LPV, RNDM = names.index("LPV"), names.index("Rndm")
    for label, position in [("p_LPV", LPV), ("p_Rndm", RNDM)]:
        cnn_mean = np.nanmean(cnn_block[..., position], axis=0)
        brf_mean = np.nanmean(brf_block[..., position], axis=0)
        rows.append({
            "norma": normalization, "clase": label,
            "CNN_valores_unicos": len(np.unique(np.round(cnn_mean, 12))),
            "BRF_valores_unicos": len(np.unique(np.round(brf_mean, 12))),
            "CNN_ceros": int((cnn_mean == 0).sum()),
            "BRF_ceros": int((brf_mean == 0).sum()),
            "CNN_min_no_cero": f"{cnn_mean[cnn_mean > 0].min():.1e}",
            "BRF_min_no_cero": f"{brf_mean[brf_mean > 0].min():.1e}",
        })
print("\nrango dinamico sobre los 857 picos:")
print(pd.DataFrame(rows).to_string(index=False))
print("\nCNN_ceros / BRF_ceros: picos cuya probabilidad queda exactamente en 0.")
print("El BRF no puede representar nada por debajo de 1/n_arboles, asi que todo")
print("el rango que la CNN usa entre 1e-18 y 1e-4 se aplasta a cero.")
