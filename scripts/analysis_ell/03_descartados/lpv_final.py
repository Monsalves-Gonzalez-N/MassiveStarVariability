"""The rule with no periodogram, and how many normalizations it really needs.

Nine normalizations mean nine CNN runs. If a subset of two or three reaches
the same purity the cost drops accordingly, so every subset up to size three
is scored, and the survivors are listed by name.
"""
import itertools

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

from _common import (NORMALIZATIONS, auc, argmax_class, brf_grouped,
                     cnn_grouped, load_brf, load_passes, load_peak_truth,
                     load_peaks, load_truth)

PERIODIC_CLASSES = ["ELL", "Pulsating", "E"]
PROB_MIN = 0.8

peaks = load_peaks()
star_truth, has_period, no_period = load_truth(peaks)
status, human_class = load_peak_truth(peaks)
brf = load_brf()

blocks = {name: cnn_grouped(load_passes(name))[0] for name in NORMALIZATIONS}
brf_log, names = brf_grouped(load_passes("log"), peaks, brf)
LPV = names.index("LPV")
brf_mean = np.nanmean(brf_log, axis=0)
filled = np.where(np.isnan(brf_mean), -np.inf, brf_mean)
klass = argmax_class(brf_mean, names)
probability = np.nanmax(filled, axis=1)
per_norm = {name: np.nanmax(blocks[name][..., LPV], axis=0) for name in NORMALIZATIONS}

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

full = np.stack([per_norm[name] for name in NORMALIZATIONS]).max(axis=0)
clean = np.where(np.isnan(full), np.inf, full)
_, p_star = mannwhitneyu(-clean[best_index][star_correct],
                         -clean[best_index][~star_correct], alternative="greater")
_, p_all = mannwhitneyu(-clean[matched], -clean[spurious], alternative="greater")
print("max p_LPV sobre 9 normalizaciones x 7 pasadas (63 valores por pico)")
print(f"  AUC estrella {auc(-full[best_index], star_correct, ~star_correct):.3f} (p={p_star:.1g})   "
      f"AUC 3v43 {auc(-full, real_ell, false_ell):.3f}   "
      f"AUC 16v689 {auc(-full, matched, spurious):.3f} (p={p_all:.1g})")

print("\n" + "=" * 72)
print("QUE SOBREVIVE A CADA UMBRAL")
print("=" * 72)
for threshold in [1e-4, 1e-3]:
    keep = full[best_index] < threshold
    print(f"\numbral {threshold:.0e}: "
          f"{int((keep & star_correct).sum())}/8 correctas, "
          f"{int((keep & ~star_correct).sum())}/18 erradas")
    for position in np.where(keep)[0]:
        mark = "CORRECTA" if star_correct[position] else "ERRADA  "
        print(f"    {mark} {keys[position]:>13} {klass[best_index[position]]:>9} "
              f"{peaks.period.values[best_index[position]]:7.3f} d  "
              f"max p_LPV={full[best_index[position]]:.1e}")

print("\n" + "=" * 72)
print("CUANTAS NORMALIZACIONES HACEN FALTA")
print("=" * 72)
rows = []
for size in [1, 2, 3]:
    for subset in itertools.combinations(NORMALIZATIONS, size):
        score = np.stack([per_norm[name] for name in subset]).max(axis=0)
        rows.append({"normas": "+".join(subset), "n": size,
                     "AUC_estrella": round(auc(-score[best_index], star_correct,
                                               ~star_correct), 3),
                     "AUC_16v689": round(auc(-score, matched, spurious), 3)})
frame = pd.DataFrame(rows)
frame["minimo"] = frame[["AUC_estrella", "AUC_16v689"]].min(axis=1)
for size in [1, 2, 3]:
    print(f"\nmejores con {size} normalizacion(es):")
    print(frame[frame.n == size].sort_values("minimo", ascending=False)
          .head(5).to_string(index=False))
print(f"\nlas 9 juntas: AUC_estrella {auc(-full[best_index], star_correct, ~star_correct):.3f}  "
      f"AUC_16v689 {auc(-full, matched, spurious):.3f}")
