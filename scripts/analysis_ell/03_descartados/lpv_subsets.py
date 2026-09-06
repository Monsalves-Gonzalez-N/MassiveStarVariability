"""How many normalizations the clean threshold needs, scored on purity.

Ranking subsets by AUC picked `log`, which orders well but has no threshold
that separates: its purity never exceeds 70%. The objective here is the one
that matters - how many correct detections survive at the largest threshold
that lets no error through - plus the margin of that threshold.
"""
import itertools

import numpy as np
import pandas as pd

from _common import (NORMALIZATIONS, auc, argmax_class, brf_grouped,
                     cnn_grouped, load_brf, load_passes, load_peak_truth,
                     load_peaks, load_truth)

peaks = load_peaks()
star_truth, has_period, no_period = load_truth(peaks)
status, human_class = load_peak_truth(peaks)
brf = load_brf()

brf_log, names = brf_grouped(load_passes("log"), peaks, brf)
LPV = names.index("LPV")
per_norm = {name: np.nanmax(cnn_grouped(load_passes(name))[0][..., LPV], axis=0)
            for name in NORMALIZATIONS}
brf_mean = np.nanmean(brf_log, axis=0)
filled = np.where(np.isnan(brf_mean), -np.inf, brf_mean)
klass = argmax_class(brf_mean, names)
probability = np.nanmax(filled, axis=1)

reported = np.isin(klass, ["ELL", "Pulsating", "E"]) & (probability >= 0.8)
best_index, keys = [], []
for key in star_truth.index:
    belongs = (peaks.key.values == key) & reported
    if belongs.any():
        best_index.append(int(np.where(belongs)[0][np.argmax(probability[belongs])]))
        keys.append(key)
best_index = np.array(best_index)
star_correct = status[best_index] == "matched"


def clean_threshold(score):
    """Correct detections kept at the largest threshold that admits no error,
    and the ratio between the first error and the last correct kept."""
    value = score[best_index]
    worst_error = np.min(value[~star_correct])
    kept = value[star_correct] < worst_error
    if not kept.any():
        return 0, np.nan
    margin = worst_error / np.max(value[star_correct][kept])
    return int(kept.sum()), float(margin)


rows = []
for size in range(1, len(NORMALIZATIONS) + 1):
    for subset in itertools.combinations(NORMALIZATIONS, size):
        score = np.stack([per_norm[name] for name in subset]).max(axis=0)
        kept, margin = clean_threshold(score)
        rows.append({"normas": "+".join(subset), "n": size,
                     "correctas_sin_error": kept, "margen": round(margin, 2)
                     if np.isfinite(margin) else np.nan})
frame = pd.DataFrame(rows)
print("=" * 78)
print("CUANTAS CORRECTAS SOBREVIVEN AL MAYOR UMBRAL QUE NO DEJA PASAR NINGUN ERROR")
print("=" * 78)
for size in range(1, 6):
    block = frame[frame.n == size].sort_values(
        ["correctas_sin_error", "margen"], ascending=False)
    print(f"\nmejores con {size}:")
    print(block.head(4).to_string(index=False))
print(f"\nlas 9 juntas: {frame[frame.n == 9].iloc[0].correctas_sin_error} correctas de 8, "
      f"margen x{frame[frame.n == 9].iloc[0].margen}")

winner = frame.sort_values(["correctas_sin_error", "margen"], ascending=False).iloc[0]
subset = winner.normas.split("+")
score = np.stack([per_norm[name] for name in subset]).max(axis=0)
value = score[best_index]
print("\n" + "=" * 78)
print(f"EL MEJOR: max p_LPV sobre {winner.normas}")
print("=" * 78)
order = np.argsort(value)
for position in order:
    mark = "CORRECTA" if star_correct[position] else "ERRADA  "
    print(f"  {mark} {keys[position]:>13} {klass[best_index[position]]:>9} "
          f"{value[position]:.2e}")
