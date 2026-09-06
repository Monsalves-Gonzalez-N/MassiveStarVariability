"""Can a CNN-derived score do what `power` does, so the rule needs no periodogram?

`power >= 0.5` removes 6 of the 7 errors that survive p_LPV_CNN < 1e-5, at the
cost of 2 correct detections. The search here is over the CNN probabilities
only: every class, every normalization, and the statistics over the 7 passes,
scored on the 15 stars that reach that stage.
"""
import itertools

import numpy as np
import pandas as pd

from _common import (NORMALIZATIONS, auc, argmax_class, brf_grouped,
                     cnn_grouped, load_brf, load_passes, load_peak_truth,
                     load_peaks, load_truth)

PERIODIC_CLASSES = ["ELL", "Pulsating", "E"]
PROB_MIN = 0.8
FLOOR = 1e-30

peaks = load_peaks()
star_truth, has_period, no_period = load_truth(peaks)
status, human_class = load_peak_truth(peaks)
brf = load_brf()

brf_log, names = brf_grouped(load_passes("log"), peaks, brf)
brf_mean = np.nanmean(brf_log, axis=0)
filled = np.where(np.isnan(brf_mean), -np.inf, brf_mean)
klass = argmax_class(brf_mean, names)
probability = np.nanmax(filled, axis=1)

blocks = {name: cnn_grouped(load_passes(name))[0] for name in NORMALIZATIONS}
LPV = names.index("LPV")

reported = np.isin(klass, PERIODIC_CLASSES) & (probability >= PROB_MIN)
best_index, keys = [], []
for key in star_truth.index:
    belongs = (peaks.key.values == key) & reported
    if belongs.any():
        best_index.append(int(np.where(belongs)[0][np.argmax(probability[belongs])]))
        keys.append(key)
best_index = np.array(best_index)
star_correct = status[best_index] == "matched"

lpv_cnn = np.nanmean(blocks["log"][..., LPV], axis=0)
stage_one = lpv_cnn[best_index] < 1e-5
print(f"tras p_LPV_CNN<1e-5 quedan {int(stage_one.sum())} estrellas: "
      f"{int((stage_one & star_correct).sum())} correctas, "
      f"{int((stage_one & ~star_correct).sum())} erradas")

# the peak-level control samples, to catch scores that only fit the 15 stars
is_ell = klass == "ELL"
real_ell = is_ell & (status == "matched") & (human_class == "ELL")
false_ell = is_ell & (status == "spurious")
matched, spurious = status == "matched", status == "spurious"

library = {}
for name, block in blocks.items():
    mean = np.nanmean(block, axis=0)
    for position, class_name in enumerate(names):
        column = block[..., position]
        library[f"{name}/media p_{class_name}"] = mean[:, position]
        library[f"{name}/max p_{class_name}"] = np.nanmax(column, axis=0)
        library[f"{name}/min p_{class_name}"] = np.nanmin(column, axis=0)
        library[f"{name}/sigma p_{class_name}"] = np.nanstd(column, axis=0)
    ordered = np.sort(mean, axis=1)
    library[f"{name}/margen top1-top2"] = ordered[:, -1] - ordered[:, -2]
    safe = np.clip(mean, FLOOR, 1)
    library[f"{name}/entropia"] = -(safe * np.log(safe)).sum(axis=1)
    library[f"{name}/desacuerdo pasadas"] = (
        block.argmax(-1) != mean.argmax(-1)[None, :]).mean(axis=0)

for class_name in names:
    stack = np.stack([np.nanmean(blocks[name], axis=0)[:, names.index(class_name)]
                      for name in NORMALIZATIONS])
    library[f"ENTRE-NORMAS/min p_{class_name}"] = stack.min(axis=0)
    library[f"ENTRE-NORMAS/max p_{class_name}"] = stack.max(axis=0)
    library[f"ENTRE-NORMAS/sigma p_{class_name}"] = stack.std(axis=0)

rows = []
for label, score in library.items():
    for direction, signed in [("alto=bueno", score), ("bajo=bueno", -score)]:
        value = signed[best_index][stage_one]
        rows.append({
            "score": label, "sentido": direction,
            "AUC_15": round(auc(value, star_correct[stage_one],
                                ~star_correct[stage_one]), 3),
            "AUC_3v43": round(auc(signed, real_ell, false_ell), 3),
            "AUC_16v689": round(auc(signed, matched, spurious), 3),
        })
frame = pd.DataFrame(rows)
frame["minimo"] = frame[["AUC_15", "AUC_3v43", "AUC_16v689"]].min(axis=1)

print("\n" + "=" * 94)
print("MEJORES SCORES DENTRO DE LAS 15 (y como les va en las muestras grandes)")
print("=" * 94)
print(frame.sort_values("AUC_15", ascending=False).head(18).to_string(index=False))
print("\nreferencia power dentro de las 15: "
      f"AUC {auc(peaks.power.values[best_index][stage_one], star_correct[stage_one], ~star_correct[stage_one]):.3f}")

print("\n" + "=" * 94)
print("LOS QUE AGUANTAN LAS TRES COMPARACIONES A LA VEZ (min de las 3 AUC)")
print("=" * 94)
print(frame.sort_values("minimo", ascending=False).head(12).to_string(index=False))
