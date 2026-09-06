"""Which checkpoints to read p_LPV from, under `log`.

Number_ELL alone ranks best (AUC 0.979) but keeps one detection fewer than the
full ensemble. All 127 subsets are scored on the objective that matters: how
many confirmed detections survive the largest threshold that admits no error,
and by what margin.
"""
import itertools

import numpy as np
import pandas as pd

from _common import (auc, argmax_class, brf_grouped, cnn_grouped, load_brf,
                     load_passes, load_peak_truth, load_peaks, load_truth)
from msv.config import MODELS

PERIODIC_CLASSES = ["ELL", "Pulsating", "E"]
PROB_MIN = 0.8

peaks = load_peaks()
star_truth, has_period, no_period = load_truth(peaks)
status, human_class = load_peak_truth(peaks)
brf = load_brf()

ensemble = load_passes("log")
cnn_ensemble, names = cnn_grouped(ensemble)
LPV = names.index("LPV")
brf_mean = np.nanmean(brf_grouped(ensemble, peaks, brf)[0], axis=0)
filled = np.where(np.isnan(brf_mean), -np.inf, brf_mean)
klass = argmax_class(brf_mean, names)
probability = np.nanmax(filled, axis=1)

reported = np.isin(klass, PERIODIC_CLASSES) & (probability >= PROB_MIN)
best_index, keys = [], []
for key in star_truth.index:
    belongs = (peaks.key.values == key) & reported
    if belongs.any():
        best_index.append(int(np.where(belongs)[0][np.argmax(probability[belongs])]))
        keys.append(key)
best_index = np.array(best_index)
star_correct = status[best_index] == "matched"
is_ell = klass == "ELL"
real_ell = is_ell & (status == "matched") & (human_class == "ELL")
false_ell = is_ell & (status == "spurious")
matched, spurious = status == "matched", status == "spurious"


def clean_point(score):
    value = score[best_index]
    first_error = np.min(value[~star_correct])
    kept = int((value[star_correct] < first_error).sum())
    if not kept:
        return 0, np.nan
    ordered = np.sort(value[star_correct])
    return kept, float(first_error / ordered[kept - 1])


rows = []
for size in range(1, len(MODELS) + 1):
    for subset in itertools.combinations(range(len(MODELS)), size):
        for statistic, reduce in [("media", np.nanmean), ("max", np.nanmax)]:
            if size == 1 and statistic == "max":
                continue
            score = reduce(cnn_ensemble[list(subset), :, LPV], axis=0)
            kept, margin = clean_point(score)
            rows.append({
                "modelos": "+".join(MODELS[i].replace("Number_", "")
                                    .replace("batchBalanced_", "bb") for i in subset),
                "n": size, "stat": statistic, "correctas": kept,
                "margen": round(margin, 1) if np.isfinite(margin) else np.nan,
                "AUC_estrella": round(auc(-score[best_index], star_correct, ~star_correct), 3),
                "AUC_16v689": round(auc(-score, matched, spurious), 3)})
frame = pd.DataFrame(rows)

print("=" * 104)
print("MEJORES SUBCONJUNTOS DE CHECKPOINTS (objetivo: correctas sin error, luego margen)")
print("=" * 104)
print(frame.sort_values(["correctas", "margen"], ascending=False).head(15).to_string(index=False))

print("\n" + "=" * 104)
print("EL MEJOR POR TAMANO")
print("=" * 104)
for size in range(1, len(MODELS) + 1):
    block = frame[frame.n == size].sort_values(["correctas", "margen"], ascending=False)
    print(block.head(1).to_string(index=False, header=(size == 1)))

full = frame[(frame.n == 7) & (frame.stat == "max")].iloc[0]
print(f"\nlos 7 con max (la regla actual): {full.correctas}/8 correctas, "
      f"margen x{full.margen}, AUC {full.AUC_estrella}")
