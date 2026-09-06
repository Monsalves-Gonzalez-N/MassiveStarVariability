"""p_LPV under `log`: per checkpoint, and MC-dropout against the ensemble.

Two things were conflated. The "7 passes" used so far are the 7 CHECKPOINTS of
config.MODELS, one deterministic pass each - not MC-dropout, which exists only
for Number_DST (20 and 200 passes). Both are measured here, plus each
checkpoint on its own, to see whether one network reads LPV better than the
rest.
"""
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

from _common import (RESULTS, auc, argmax_class, brf_grouped, cnn_grouped,
                     load_brf, load_passes, load_peak_truth, load_peaks,
                     load_truth)
from msv.config import MODELS

PERIODIC_CLASSES = ["ELL", "Pulsating", "E"]
PROB_MIN = 0.8

peaks = load_peaks()
star_truth, has_period, no_period = load_truth(peaks)
status, human_class = load_peak_truth(peaks)
brf = load_brf()

ensemble = load_passes("log")                                   # (7, N, 8) checkpoints
dropout20 = np.load(RESULTS / "cnn_mc20.npz")["p_mc"].astype(float)
dropout200 = np.load(RESULTS / "cnn_mc200.npz")["p_mc"].astype(float)
cnn_ensemble, names = cnn_grouped(ensemble)
cnn_dropout20, _ = cnn_grouped(dropout20)
cnn_dropout200, _ = cnn_grouped(dropout200)
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


def describe(score, label):
    """Star-level frontier plus the clean-threshold margin."""
    value = score[best_index]
    frontier = {}
    for cut in np.unique(value):
        keep = value < cut * 1.000001
        n_correct = int((keep & star_correct).sum())
        n_wrong = int((keep & ~star_correct).sum())
        if n_correct and (n_correct not in frontier or n_wrong < frontier[n_correct]):
            frontier[n_correct] = n_wrong
    clean = [n for n in frontier if frontier[n] == 0]
    kept = max(clean) if clean else 0
    if kept:
        ordered = np.sort(value[star_correct])
        margin = np.min(value[~star_correct]) / ordered[kept - 1]
    else:
        margin = np.nan
    signed = np.where(np.isnan(-score), -np.inf, -score)
    _, p_star = mannwhitneyu(signed[best_index][star_correct],
                             signed[best_index][~star_correct], alternative="greater")
    return {"score": label,
            "AUC_estrella": round(auc(-score[best_index], star_correct, ~star_correct), 3),
            "p": f"{p_star:.1g}",
            "correctas_sin_error": f"{kept}/8",
            "margen": round(margin, 1) if np.isfinite(margin) else np.nan,
            "AUC_3v43": round(auc(-score, real_ell, false_ell), 3),
            "AUC_16v689": round(auc(-score, matched, spurious), 3)}


print("=" * 96)
print("CADA CHECKPOINT POR SEPARADO (una pasada determinista, normalizacion log)")
print("=" * 96)
rows = []
for position, model in enumerate(MODELS):
    rows.append(describe(cnn_ensemble[position, :, LPV], model))
print(pd.DataFrame(rows).to_string(index=False))

print("\n" + "=" * 96)
print("AGREGADOS: ensemble de 7 checkpoints contra MC-dropout de Number_DST")
print("=" * 96)
aggregates = {
    "ens7 media": np.nanmean(cnn_ensemble[..., LPV], axis=0),
    "ens7 max": np.nanmax(cnn_ensemble[..., LPV], axis=0),
    "ens7 mediana": np.nanmedian(cnn_ensemble[..., LPV], axis=0),
    "dropout20 media": np.nanmean(cnn_dropout20[..., LPV], axis=0),
    "dropout20 max": np.nanmax(cnn_dropout20[..., LPV], axis=0),
    "dropout200 media": np.nanmean(cnn_dropout200[..., LPV], axis=0),
    "dropout200 max": np.nanmax(cnn_dropout200[..., LPV], axis=0),
    "dropout200 q84": np.quantile(cnn_dropout200[..., LPV], 0.84, axis=0),
}
print(pd.DataFrame([describe(score, label) for label, score in aggregates.items()])
      .to_string(index=False))
print("\ncorrectas_sin_error = detecciones confirmadas al mayor umbral que no")
print("deja pasar ningun error; margen = primer error / ultima correcta conservada")
