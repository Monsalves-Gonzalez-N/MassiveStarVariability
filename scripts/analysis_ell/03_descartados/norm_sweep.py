"""Which normalization to actually read p_LPV and the Rndm tail from.

`rank` won in the previous handoff, but it was scored only on the 3-vs-43
comparison. Both comparisons are shown per normalization: the thin one that
the gate is judged on, and the 16-vs-689 one that carries the statistics.
"""
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

from _common import (NORMALIZATIONS, auc, argmax_class, brf_grouped,
                     cnn_grouped, load_brf, load_passes, load_peak_truth,
                     load_peaks, load_truth)

peaks = load_peaks()
brf = load_brf()
status, human_class = load_peak_truth(peaks)
brf_log, names = brf_grouped(load_passes("log"), peaks, brf)
ELL, LPV, RNDM = names.index("ELL"), names.index("LPV"), names.index("Rndm")
klass = argmax_class(np.nanmean(brf_log, axis=0), names)
is_ell = klass == "ELL"
real = is_ell & (status == "matched") & (human_class == "ELL")
false = is_ell & (status == "spurious")
matched, spurious = status == "matched", status == "spurious"


def report(score, label):
    clean = np.where(np.isnan(score), -np.inf, score)
    _, p_all = mannwhitneyu(clean[matched], clean[spurious], alternative="greater")
    threshold = np.nanmin(score[real])
    surviving = int(((score >= threshold) & false).sum())
    return {"score": label,
            "AUC_3v43": round(auc(score, real, false), 3),
            "AUC_16v689": round(auc(score, matched, spurious), 3),
            "p_16v689": f"{p_all:.1g}",
            "falsos_vivos": f"{surviving}/43"}


rows = []
for name in NORMALIZATIONS:
    passes = load_passes(name)
    brf_mean = np.nanmean(brf_grouped(passes, peaks, brf)[0], axis=0)
    cnn_passes, _ = cnn_grouped(passes)
    rows.append(report(-brf_mean[:, LPV], f"p_LPV BRF ({name})"))
    rows.append(report(-np.nanmax(cnn_passes[..., RNDM], axis=0),
                       f"max p_Rndm CNN ({name})"))
frame = pd.DataFrame(rows)
print("=" * 88)
print("p_LPV Y COLA DE Rndm, POR NORMALIZACION")
print("=" * 88)
print(frame[frame.score.str.startswith("p_LPV")].to_string(index=False))
print()
print(frame[frame.score.str.startswith("max")].to_string(index=False))
print("\nfalsos_vivos = contaminantes que sobreviven al umbral que conserva las 3 ELL reales")
