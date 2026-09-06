"""Does the BRF destroy the across-pass dispersion the CNN produces?

`prob_snr.py` measured sigma on the grouped BRF output and reported
sigma ~ 0.003 for real and contaminant alike. The same 20 passes are measured
here before and after the BRF, so any difference is the BRF averaging its 500
trees over the CNN excursions.
"""
import numpy as np
import pandas as pd

from _common import (RESULTS, auc, argmax_class, brf_grouped, cnn_grouped,
                     load_brf, load_passes, load_peaks, load_truth)

peaks = load_peaks()
star_truth, has_period, no_period = load_truth(peaks)
brf = load_brf()

dropout20 = np.load(RESULTS / "cnn_mc20.npz")["p_mc"].astype(float)
cnn_passes, names = cnn_grouped(dropout20)
brf_passes, _ = brf_grouped(dropout20, peaks, brf)
ELL, LPV, RNDM = names.index("ELL"), names.index("LPV"), names.index("Rndm")

reference = argmax_class(np.nanmean(brf_grouped(load_passes("log"), peaks, brf)[0], axis=0), names)
is_ell = reference == "ELL"
real, contaminant = is_ell & has_period, is_ell & no_period

print("=" * 76)
print("MISMAS 20 PASADAS DE DROPOUT, ANTES Y DESPUES DEL BRF")
print("=" * 76)
rows = []
for level, block in [("CNN", cnn_passes), ("BRF", brf_passes)]:
    for class_name, position in [("ELL", ELL), ("Rndm", RNDM), ("LPV", LPV)]:
        sigma = np.nanstd(block[..., position], axis=0)
        spread = np.nanmax(block[..., position], axis=0) - np.nanmin(block[..., position], axis=0)
        rows.append({
            "nivel": level, "clase": class_name,
            "sigma_real": round(float(np.median(sigma[real])), 4),
            "sigma_cont": round(float(np.median(sigma[contaminant])), 4),
            "AUC_sigma": round(auc(-sigma, real, contaminant), 3),
            "rango_real": round(float(np.median(spread[real])), 4),
            "rango_cont": round(float(np.median(spread[contaminant])), 4),
            "AUC_rango": round(auc(-spread, real, contaminant), 3),
        })
print(pd.DataFrame(rows).to_string(index=False))
print("\nfactor de compresion del BRF sobre el rango de p_ELL en contaminantes:")
cnn_range = np.nanmax(cnn_passes[..., ELL], axis=0) - np.nanmin(cnn_passes[..., ELL], axis=0)
brf_range = np.nanmax(brf_passes[..., ELL], axis=0) - np.nanmin(brf_passes[..., ELL], axis=0)
print(f"  CNN {np.median(cnn_range[contaminant]):.3f} -> BRF "
      f"{np.median(brf_range[contaminant]):.3f}   "
      f"(x{np.median(cnn_range[contaminant]) / np.median(brf_range[contaminant]):.0f} mas chico)")
