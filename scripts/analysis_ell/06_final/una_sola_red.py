"""Pick one checkpoint that does all three jobs: class, class probability, p_LPV.

Number_ELL won on p_LPV alone, but a single network now has to also choose the
class and supply the probability printed next to it. All three are scored per
checkpoint at CNN level, with no BRF and no ensemble.
"""
import numpy as np
import pandas as pd

from _common import (auc, argmax_class, brf_grouped, cnn_grouped, load_brf,
                     load_passes, load_peak_truth, load_peaks, load_truth)
from msv.config import MODELS

PERIODIC_CLASSES = ["ELL", "Pulsating", "E"]

peaks = load_peaks()
star_truth, has_period, no_period = load_truth(peaks)
status, human_class = load_peak_truth(peaks)
brf = load_brf()

ensemble = load_passes("log")
cnn_ensemble, names = cnn_grouped(ensemble)
LPV, RNDM = names.index("LPV"), names.index("Rndm")
matched, spurious = status == "matched", status == "spurious"

# the reference peak selection stays the production one, so every checkpoint is
# scored on the same stars
brf_mean = np.nanmean(brf_grouped(ensemble, peaks, brf)[0], axis=0)
filled = np.where(np.isnan(brf_mean), -np.inf, brf_mean)
reference_class = argmax_class(brf_mean, names)
reference_prob = np.nanmax(filled, axis=1)
reported = np.isin(reference_class, PERIODIC_CLASSES) & (reference_prob >= 0.8)
best_index, keys = [], []
for key in star_truth.index:
    belongs = (peaks.key.values == key) & reported
    if belongs.any():
        best_index.append(int(np.where(belongs)[0][np.argmax(reference_prob[belongs])]))
        keys.append(key)
best_index = np.array(best_index)
star_correct = status[best_index] == "matched"

rows = []
for position, model in enumerate(MODELS):
    block = cnn_ensemble[position]                               # (N, 5)
    klass = argmax_class(block, names)
    probability = block.max(axis=1)
    lpv = block[:, LPV]
    irregular = np.empty(len(peaks))
    for key in np.unique(peaks.key.values):
        same_star = peaks.key.values == key
        irregular[same_star] = block[same_star, LPV].mean()
    class_hits = int((klass[matched] == human_class[matched]).sum())
    is_ell = klass == "ELL"
    real_ell = is_ell & matched & (human_class == "ELL")
    false_ell = is_ell & (status == "spurious")
    rows.append({
        "checkpoint": model,
        "clase_ok_16": f"{class_hits}/16",
        "prob_AUC": round(auc(probability, matched, spurious), 3),
        "prob_p50": round(float(np.median(probability)), 3),
        "prob_p95": round(float(np.percentile(probability, 95)), 4),
        "pLPV_AUC_estrella": round(auc(-lpv[best_index], star_correct, ~star_correct), 3),
        "pLPV_AUC_16v689": round(auc(-lpv, matched, spurious), 3),
        "irreg_AUC_ELL": round(auc(-irregular, real_ell, false_ell), 3)
        if real_ell.any() and false_ell.any() else np.nan,
        "N_ELL": int(is_ell.sum()),
    })
frame = pd.DataFrame(rows)
print("=" * 108)
print("LOS 7 CHECKPOINTS EN LAS TRES FUNCIONES (CNN sola, normalizacion log)")
print("=" * 108)
print(frame.to_string(index=False))
print("\nclase_ok_16       la clase de esa red contra la humana, en los 16 picos confirmados")
print("prob_AUC/p50/p95  la probabilidad de la clase: separa? y cuanto satura")
print("pLPV_AUC          p_LPV de esa red, a nivel de estrella y de pico")
print("irreg_AUC_ELL     % irregular de esa red separando ELL real de falsa")
print("\nreferencia: ensemble de 7 + BRF -> clase_ok 14/16, "
      f"prob_AUC {auc(reference_prob, matched, spurious):.3f}, "
      f"p50 {np.median(reference_prob):.3f}")
