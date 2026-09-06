"""Where to put the cut on the p_Rndm tail, chosen where the statistics are.

The threshold is scanned against the 16 confirmed peaks and the 689 spurious
ones, not against the 3 real ELL, so it is not tuned on the quantity it is
meant to protect. The ELL columns are then read off, never fitted.
"""
import numpy as np
import pandas as pd

from _common import (RESULTS, argmax_class, brf_grouped, cnn_grouped, load_brf,
                     load_passes, load_peak_truth, load_peaks, load_truth)

PERIODIC_CLASSES = {"ELL", "Pulsating", "E"}
PROB_MIN = 0.8

peaks = load_peaks()
star_truth, has_period, no_period = load_truth(peaks)
status, human_class = load_peak_truth(peaks)
brf = load_brf()

ensemble = load_passes("log")
brf_ensemble, names = brf_grouped(ensemble, peaks, brf)
cnn_ensemble, _ = cnn_grouped(ensemble)
cnn_dropout, _ = cnn_grouped(np.load(RESULTS / "cnn_mc200.npz")["p_mc"].astype(float))
rank_mean = np.nanmean(brf_grouped(load_passes("rank"), peaks, brf)[0], axis=0)

ELL, LPV, RNDM = names.index("ELL"), names.index("LPV"), names.index("Rndm")
brf_mean = np.nanmean(brf_ensemble, axis=0)
klass = argmax_class(brf_mean, names)
probability = np.nanmax(np.where(np.isnan(brf_mean), -np.inf, brf_mean), axis=1)

is_ell = klass == "ELL"
real_ell = is_ell & (status == "matched") & (human_class == "ELL")
false_ell = is_ell & (status == "spurious")
matched, spurious = status == "matched", status == "spurious"

tails = {
    "max p_Rndm ens7": np.nanmax(cnn_ensemble[..., RNDM], axis=0),
    "2do mayor p_Rndm ens7": np.sort(cnn_ensemble[..., RNDM], axis=0)[-2],
    "q90 p_Rndm drop200": np.quantile(cnn_dropout[..., RNDM], 0.90, axis=0),
}


def star_level(keep):
    call = pd.Series(keep & np.isin(klass, list(PERIODIC_CLASSES)) & (probability >= PROB_MIN),
                     index=peaks.key.values).groupby(level=0).any()
    aligned = star_truth.index.intersection(call.index)
    predicted, actual = call[aligned], star_truth[aligned]
    return (int((predicted & actual).sum()), int((~predicted & actual).sum()),
            int((predicted & ~actual).sum()), int((~predicted & ~actual).sum()))


for label, tail in tails.items():
    print("=" * 96)
    print(f"ESCANEO DE UMBRAL SOBRE {label}")
    print("=" * 96)
    rows = []
    for threshold in [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.05, 0.1, 0.3, 0.5]:
        keep = tail < threshold
        tp, fn, fp, tn = star_level(keep)
        rows.append({
            "umbral": threshold,
            "matched_16": int((keep & matched).sum()),
            "spurious_689": int((keep & spurious).sum()),
            "ELL_real_3": int((keep & real_ell).sum()),
            "ELL_falso_43": int((keep & false_ell).sum()),
            "TP": tp, "FN": fn, "FP": fp, "TN": tn,
            "acc": round((tp + tn) / (tp + fn + fp + tn), 3),
        })
    print(pd.DataFrame(rows).to_string(index=False))
    print()

print("=" * 96)
print("REGLA COMBINADA: cola de Rndm (general) + p_LPV(rank)==0 solo sobre ELL")
print("=" * 96)
tail = tails["max p_Rndm ens7"]
rows = []
for threshold in [1e-4, 1e-3, 1e-2, 0.05]:
    general = tail < threshold
    ell_specific = ~is_ell | (rank_mean[:, LPV] == 0)
    for label, keep in [(f"solo cola < {threshold}", general),
                        (f"cola < {threshold} Y p_LPV(rank)==0 en ELL", general & ell_specific)]:
        tp, fn, fp, tn = star_level(keep)
        rows.append({"regla": label, "matched_16": int((keep & matched).sum()),
                     "ELL_real_3": int((keep & real_ell).sum()),
                     "ELL_falso_43": int((keep & false_ell).sum()),
                     "TP": tp, "FN": fn, "FP": fp, "TN": tn,
                     "acc": round((tp + tn) / (tp + fn + fp + tn), 3)})
baseline = np.ones(len(peaks), dtype=bool)
tp, fn, fp, tn = star_level(baseline)
rows.insert(0, {"regla": "sin filtro", "matched_16": 16, "ELL_real_3": 3,
                "ELL_falso_43": 43, "TP": tp, "FN": fn, "FP": fp, "TN": tn,
                "acc": round((tp + tn) / (tp + fn + fp + tn), 3)})
print(pd.DataFrame(rows).to_string(index=False))
