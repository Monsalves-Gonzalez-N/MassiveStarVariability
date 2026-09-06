"""Which combination of flags makes the trusted tier, measured on all subsets.

sigma costs TIC 12921082, a confirmed ELL that disperses across passes, so
whether it belongs in the rule is a question about the data, not a matter of
adding every available axis.
"""
import itertools

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

brf_log, names = brf_grouped(load_passes("log"), peaks, brf)
brf_dropout, _ = brf_grouped(np.load("results/cnn_mc20.npz")["p_mc"].astype(float),
                             peaks, brf)
RNDM = names.index("Rndm")
brf_mean = np.nanmean(brf_log, axis=0)
filled = np.where(np.isnan(brf_mean), -np.inf, brf_mean)
klass = argmax_class(brf_mean, names)
winner = filled.argmax(axis=1)
probability = np.nanmax(filled, axis=1)
winner_passes = np.take_along_axis(brf_dropout, winner[None, :, None], axis=2)[..., 0]
sigma = np.nanstd(winner_passes, axis=0)
binary_tail = np.nanmax(cnn_grouped(load_passes("binary"))[0][..., RNDM], axis=0)

reported = np.isin(klass, PERIODIC_CLASSES) & (probability >= PROB_MIN)
records = []
for key in star_truth.index:
    belongs = (peaks.key.values == key) & reported
    if not belongs.any():
        continue
    best = np.where(belongs)[0][np.argmax(probability[belongs])]
    records.append({"key": key, "clase": klass[best],
                    "power": peaks.power.values[best], "sigma": sigma[best],
                    "cola": binary_tail[best], "prob": probability[best],
                    "correcta": status[best] == "matched"})
frame = pd.DataFrame(records)

axes = {
    "power>=0.5": frame.power.values >= 0.5,
    "cola<1e-4": frame.cola.values < 1e-4,
    "sigma<0.05": frame.sigma.values < 0.05,
    "prob>=0.99": frame.prob.values >= 0.99,
}
print("=" * 78)
print("TODAS LAS COMBINACIONES DE EJES (AND), nivel de confianza ALTA")
print("=" * 78)
rows = []
for size in range(1, len(axes) + 1):
    for combination in itertools.combinations(axes, size):
        keep = np.ones(len(frame), dtype=bool)
        for name in combination:
            keep &= axes[name]
        n_correct = int((keep & frame.correcta.values).sum())
        n_wrong = int((keep & ~frame.correcta.values).sum())
        rows.append({
            "ejes": " Y ".join(combination),
            "correctas": f"{n_correct}/8", "erradas": f"{n_wrong}/18",
            "pureza_%": round(100 * n_correct / max(n_correct + n_wrong, 1)),
        })
table = pd.DataFrame(rows).sort_values("pureza_%", ascending=False)
print(table.to_string(index=False))
print(f"\nbaseline sin ejes: 8/8 correctas, 18/18 erradas, pureza 31%")
print("correctas = detecciones confirmadas conservadas; erradas = las que se cuelan")
