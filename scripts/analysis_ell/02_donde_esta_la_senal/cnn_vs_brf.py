"""p_ELL, p_LPV and p_Rndm measured before and after the BRF, side by side.

The recommended flag mixes levels - the class comes from the BRF, the tail
from the CNN - so the question of whether the BRF adds or destroys information
for each of the three probabilities is answered here explicitly, on the same
peaks and the same passes.
"""
import numpy as np
import pandas as pd

from _common import (auc, argmax_class, brf_grouped, cnn_grouped, load_brf,
                     load_passes, load_peak_truth, load_peaks, load_truth)

PERIODIC_CLASSES = ["ELL", "Pulsating", "E"]
PROB_MIN = 0.8
FLOOR = 1e-12

peaks = load_peaks()
star_truth, has_period, no_period = load_truth(peaks)
status, human_class = load_peak_truth(peaks)
brf = load_brf()

reference_mean = np.nanmean(brf_grouped(load_passes("log"), peaks, brf)[0], axis=0)
names = ["ELL", "Pulsating", "E", "LPV", "Rndm"]
ELL, LPV, RNDM = names.index("ELL"), names.index("LPV"), names.index("Rndm")
filled = np.where(np.isnan(reference_mean), -np.inf, reference_mean)
klass = argmax_class(reference_mean, names)
probability = np.nanmax(filled, axis=1)

is_ell = klass == "ELL"
real_ell = is_ell & (status == "matched") & (human_class == "ELL")
false_ell = is_ell & (status == "spurious")
matched, spurious = status == "matched", status == "spurious"

# the star-level sample: the peak the catalogue reports, right or wrong
reported = np.isin(klass, PERIODIC_CLASSES) & (probability >= PROB_MIN)
best_index = []
for key in star_truth.index:
    belongs = (peaks.key.values == key) & reported
    if belongs.any():
        best_index.append(int(np.where(belongs)[0][np.argmax(probability[belongs])]))
best_index = np.array(best_index)
star_correct = status[best_index] == "matched"


def evaluate(score, label, level, normalization):
    star_score = score[best_index]
    return {"norma": normalization, "nivel": level, "score": label,
            "AUC_3v43": round(auc(score, real_ell, false_ell), 3),
            "AUC_16v689": round(auc(score, matched, spurious), 3),
            "AUC_estrella_8v18": round(auc(star_score, star_correct, ~star_correct), 3)}


rows = []
for normalization in ["log", "binary", "min_max"]:
    passes = load_passes(normalization)
    levels = {"CNN": cnn_grouped(passes)[0],
              "BRF": brf_grouped(passes, peaks, brf)[0]}
    for level, block in levels.items():
        mean = np.nanmean(block, axis=0)
        p_ell = np.clip(mean[:, ELL], FLOOR, 1)
        p_lpv = np.clip(mean[:, LPV], FLOOR, 1)
        p_rndm = np.clip(mean[:, RNDM], FLOOR, 1)
        for label, score in [
                ("media p_ELL", p_ell),
                ("media -p_LPV", -p_lpv),
                ("media -p_Rndm", -p_rndm),
                ("log10 p_ELL/p_LPV", np.log10(p_ell / p_lpv)),
                ("log10 p_ELL/p_Rndm", np.log10(p_ell / p_rndm)),
                ("COLA -max p_Rndm", -np.nanmax(block[..., RNDM], axis=0)),
                ("COLA -max p_LPV", -np.nanmax(block[..., LPV], axis=0))]:
            rows.append(evaluate(score, label, level, normalization))
frame = pd.DataFrame(rows)

for normalization in ["log", "binary", "min_max"]:
    print("=" * 92)
    print(f"NORMALIZACION {normalization}")
    print("=" * 92)
    block = frame[frame.norma == normalization]
    pivot = block.pivot(index="score", columns="nivel",
                        values=["AUC_3v43", "AUC_16v689", "AUC_estrella_8v18"])
    print(pivot.to_string())
    print()

print("AUC_3v43           3 ELL confirmadas vs 43 ELL espurias (picos)")
print("AUC_16v689         16 picos con periodo humano vs 689 espurios (todas las clases)")
print("AUC_estrella_8v18  el pico que el catalogo reporta: 8 correctas vs 18 erradas")
