"""The gate that survives both comparisons, and what it costs.

`binary` keeps only which cells of the phase-fold histogram are occupied, so
asking whether any of the 7 checkpoints calls the peak noise under it is a
test on the SHAPE of the folded curve with the density thrown away. That is
the same quantity Gomel's harmonic test measures, read off the network.
"""
import numpy as np
import pandas as pd

from _common import (auc, argmax_class, brf_grouped, cnn_grouped, load_brf,
                     load_passes, load_peak_truth, load_peaks, load_truth)

PERIODIC_CLASSES = {"ELL", "Pulsating", "E"}
PROB_MIN = 0.8

peaks = load_peaks()
star_truth, has_period, no_period = load_truth(peaks)
status, human_class = load_peak_truth(peaks)
brf = load_brf()

brf_log, names = brf_grouped(load_passes("log"), peaks, brf)
ELL, LPV, RNDM = names.index("ELL"), names.index("LPV"), names.index("Rndm")
brf_mean = np.nanmean(brf_log, axis=0)
klass = argmax_class(brf_mean, names)
probability = np.nanmax(np.where(np.isnan(brf_mean), -np.inf, brf_mean), axis=1)

binary_tail = np.nanmax(cnn_grouped(load_passes("binary"))[0][..., RNDM], axis=0)
log_tail = np.nanmax(cnn_grouped(load_passes("log"))[0][..., RNDM], axis=0)
minmax_lpv = np.nanmean(brf_grouped(load_passes("min_max"), peaks, brf)[0], axis=0)[:, LPV]
rank_lpv = np.nanmean(brf_grouped(load_passes("rank"), peaks, brf)[0], axis=0)[:, LPV]

is_ell = klass == "ELL"
real_ell = is_ell & (status == "matched") & (human_class == "ELL")
false_ell = is_ell & (status == "spurious")
matched = status == "matched"

print("max p_Rndm(binary) en los grupos confirmados:")
for label, mask in [("3 ELL reales", real_ell), ("11 E confirmadas", matched & (human_class == "E")),
                    ("43 ELL falsos", false_ell), ("689 spurious", status == "spurious")]:
    values = np.sort(binary_tail[mask])
    print(f"  {label:18s} min={values.min():.2e}  mediana={np.median(values):.2e}  "
          f"max={values.max():.2e}")


def star_level(keep):
    called = pd.Series(keep & np.isin(klass, list(PERIODIC_CLASSES)) & (probability >= PROB_MIN),
                       index=peaks.key.values).groupby(level=0).any()
    aligned = star_truth.index.intersection(called.index)
    predicted, actual = called[aligned], star_truth[aligned]
    return (int((predicted & actual).sum()), int((~predicted & actual).sum()),
            int((predicted & ~actual).sum()), int((~predicted & ~actual).sum()))


gates = {"sin filtro": np.ones(len(peaks), dtype=bool),
         "power >= 0.5 (global)": peaks.power.values >= 0.5}
for threshold in [1e-4, 1e-3, 1e-2, 0.05, 0.1]:
    gates[f"ELL: max p_Rndm(binary) < {threshold}"] = ~is_ell | (binary_tail < threshold)
    gates[f"GLOBAL: max p_Rndm(binary) < {threshold}"] = binary_tail < threshold
gates["ELL: binary<1e-2 Y log<1e-3"] = ~is_ell | ((binary_tail < 1e-2) & (log_tail < 1e-3))
gates["ELL: binary<1e-2 Y p_LPV(min_max)<1e-3"] = ~is_ell | ((binary_tail < 1e-2) & (minmax_lpv < 1e-3))
gates["ELL: binary<1e-2 Y p_LPV(rank)==0"] = ~is_ell | ((binary_tail < 1e-2) & (rank_lpv == 0))

rows = []
for label, keep in gates.items():
    tp, fn, fp, tn = star_level(keep)
    rows.append({"gate": label, "matched_16": int((keep & matched).sum()),
                 "ELL_real_3": int((keep & real_ell).sum()),
                 "ELL_falso_43": int((keep & false_ell).sum()),
                 "TP": tp, "FN": fn, "FP": fp, "TN": tn,
                 "acc": round((tp + tn) / (tp + fn + fp + tn), 3)})
print("\n" + "=" * 100)
print(pd.DataFrame(rows).to_string(index=False))
