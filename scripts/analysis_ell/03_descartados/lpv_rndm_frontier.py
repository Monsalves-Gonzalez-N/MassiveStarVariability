"""Does p_Rndm add anything to p_LPV, under `log` alone?

The 2D scan reached 6 correct with no errors, but a fine 1D grid has to be
compared against it before crediting the second axis: a frontier that p_LPV
reaches on its own is not evidence that p_Rndm contributes.
"""
import numpy as np
import pandas as pd

from _common import (argmax_class, brf_grouped, cnn_grouped, load_brf,
                     load_passes, load_peak_truth, load_peaks, load_truth)

PERIODIC_CLASSES = ["ELL", "Pulsating", "E"]
PROB_MIN = 0.8

peaks = load_peaks()
star_truth, has_period, no_period = load_truth(peaks)
status, human_class = load_peak_truth(peaks)
brf = load_brf()

passes_log = load_passes("log")
cnn_log, names = cnn_grouped(passes_log)
brf_log, _ = brf_grouped(passes_log, peaks, brf)
LPV, RNDM = names.index("LPV"), names.index("Rndm")
brf_mean = np.nanmean(brf_log, axis=0)
filled = np.where(np.isnan(brf_mean), -np.inf, brf_mean)
klass = argmax_class(brf_mean, names)
probability = np.nanmax(filled, axis=1)

lpv = np.nanmean(cnn_log[..., LPV], axis=0)
rndm = np.nanmean(cnn_log[..., RNDM], axis=0)

reported = np.isin(klass, PERIODIC_CLASSES) & (probability >= PROB_MIN)
best_index, keys = [], []
for key in star_truth.index:
    belongs = (peaks.key.values == key) & reported
    if belongs.any():
        best_index.append(int(np.where(belongs)[0][np.argmax(probability[belongs])]))
        keys.append(key)
best_index = np.array(best_index)
star_correct = status[best_index] == "matched"
lpv_star, rndm_star = lpv[best_index], rndm[best_index]


def frontier_one_axis(score):
    """Fewest errors admitted for each number of correct detections kept."""
    best = {}
    for cut in np.unique(score):
        keep = score < cut * 1.000001
        n_correct = int((keep & star_correct).sum())
        n_wrong = int((keep & ~star_correct).sum())
        if n_correct and (n_correct not in best or n_wrong < best[n_correct]):
            best[n_correct] = n_wrong
    return best


def frontier_two_axes(first, second):
    best = {}
    for cut_first in np.unique(first):
        for cut_second in np.unique(second):
            keep = (first < cut_first * 1.000001) & (second < cut_second * 1.000001)
            n_correct = int((keep & star_correct).sum())
            n_wrong = int((keep & ~star_correct).sum())
            if n_correct and (n_correct not in best or n_wrong < best[n_correct]):
                best[n_correct] = n_wrong
    return best


only_lpv = frontier_one_axis(lpv_star)
only_rndm = frontier_one_axis(rndm_star)
both = frontier_two_axes(lpv_star, rndm_star)

print("=" * 66)
print("FRONTERA: erradas minimas para cada numero de correctas conservadas")
print("=" * 66)
print(f"  {'correctas':>10} {'solo p_LPV':>11} {'solo p_Rndm':>12} {'las dos':>9}")
for n_correct in range(8, 0, -1):
    print(f"  {n_correct:>7}/8 {only_lpv.get(n_correct, '-'):>11} "
          f"{only_rndm.get(n_correct, '-'):>12} {both.get(n_correct, '-'):>9}")

print("\n" + "=" * 66)
print("EL PUNTO 6 CORRECTAS / 0 ERRADAS, CON p_LPV SOLA")
print("=" * 66)
kept = np.sort(lpv_star[star_correct])[:6]
first_error = np.min(lpv_star[~star_correct])
print(f"  ultima CORRECTA conservada   {kept[-1]:.2e}")
print(f"  primer ERROR                 {first_error:.2e}")
print(f"  margen                       x{first_error / kept[-1]:.1f}")
print(f"  umbral sugerido              3e-9  (entre los dos)")
keep = lpv_star < 3e-9
print(f"\n  con 3e-9: {int((keep & star_correct).sum())}/8 correctas, "
      f"{int((keep & ~star_correct).sum())}/18 erradas")
for position in np.argsort(lpv_star):
    mark = "CORRECTA" if star_correct[position] else "ERRADA  "
    inside = "<<" if lpv_star[position] < 3e-9 else "  "
    print(f"  {inside} {mark} {keys[position]:>13} {klass[best_index[position]]:>9} "
          f"p_LPV={lpv_star[position]:.1e}  p_Rndm={rndm_star[position]:.1e}")
    if position == np.argsort(lpv_star)[11]:
        break
