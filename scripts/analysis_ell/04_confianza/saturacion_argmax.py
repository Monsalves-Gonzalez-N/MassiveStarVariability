"""Is the argmax the problem, or only the number attached to it?

Two separate questions get conflated. Whether the winning class is right is one
thing; whether the probability printed next to it carries information is
another. They are measured separately here, and the log-odds margin is tested
as the replacement for a linear probability that saturates at 1.
"""
import numpy as np
import pandas as pd

from _common import (auc, argmax_class, brf_grouped, cnn_grouped, load_brf,
                     load_passes, load_peak_truth, load_peaks, load_truth)

PERIODIC_CLASSES = ["ELL", "Pulsating", "E"]
PROB_MIN = 0.8
FLOOR = 1e-30

peaks = load_peaks()
star_truth, has_period, no_period = load_truth(peaks)
status, human_class = load_peak_truth(peaks)
brf = load_brf()

ensemble = load_passes("log")
cnn_ensemble, names = cnn_grouped(ensemble)
brf_ensemble, _ = brf_grouped(ensemble, peaks, brf)
cnn_mean = np.nanmean(cnn_ensemble, axis=0)
brf_mean = np.nanmean(brf_ensemble, axis=0)
filled = np.where(np.isnan(brf_mean), -np.inf, brf_mean)
klass = argmax_class(brf_mean, names)
probability = np.nanmax(filled, axis=1)
LPV = names.index("LPV")

matched, spurious = status == "matched", status == "spurious"

print("=" * 78)
print("1. LA SATURACION, EN NUMEROS")
print("=" * 78)
for level, mean in [("CNN", cnn_mean), ("BRF", brf_mean)]:
    top = np.nanmax(np.where(np.isnan(mean), -np.inf, mean), axis=1)
    print(f"  {level}: fraccion de picos con la clase ganadora por encima de")
    for threshold in [0.8, 0.9, 0.99, 0.999]:
        print(f"       {threshold:<6} {100 * (top >= threshold).mean():5.1f}%")

print("\n" + "=" * 78)
print("2. EL ARGMAX ACIERTA? (los 16 picos con periodo humano confirmado)")
print("=" * 78)
print(pd.crosstab(pd.Series(human_class[matched], name="humano"),
                  pd.Series(klass[matched], name="argmax del pipeline")).to_string())
agreement = int((human_class[matched] == klass[matched]).sum())
print(f"\n  acierto del argmax: {agreement}/{int(matched.sum())}")

print("\n" + "=" * 78)
print("3. EL GATE prob >= 0.8 HACE ALGO?")
print("=" * 78)
periodic = np.isin(klass, PERIODIC_CLASSES)
for threshold in [0.0, 0.5, 0.8, 0.9, 0.99]:
    keep = periodic & (probability >= threshold)
    print(f"  prob >= {threshold:<5} picos periodicos {int(keep.sum()):4d}   "
          f"confirmados {int((keep & matched).sum()):2d}/16   "
          f"espurios {int((keep & spurious).sum()):4d}")

print("\n" + "=" * 78)
print("4. LO QUE SI TIENE RANGO: margen en log-odds")
print("=" * 78)
scores = {}
for level, mean in [("CNN", cnn_mean), ("BRF", brf_mean)]:
    safe = np.clip(mean, FLOOR, 1)
    ordered = np.sort(safe, axis=1)
    scores[f"{level} prob lineal (top1)"] = ordered[:, -1]
    scores[f"{level} margen lineal top1-top2"] = ordered[:, -1] - ordered[:, -2]
    scores[f"{level} margen log10 top1/top2"] = np.log10(ordered[:, -1] / ordered[:, -2])
    scores[f"{level} margen log10 top1/top3"] = np.log10(ordered[:, -1] / ordered[:, -3])
scores["CNN -log10 p_LPV"] = -np.log10(np.clip(cnn_mean[:, LPV], FLOOR, 1))
rows = []
for label, score in scores.items():
    ordered = np.sort(score[np.isfinite(score)])
    rows.append({
        "score": label,
        "AUC_16v689": round(auc(score, matched, spurious), 3),
        "p5": f"{ordered[int(0.05 * len(ordered))]:.3g}",
        "p50": f"{np.median(ordered):.3g}",
        "p95": f"{ordered[int(0.95 * len(ordered))]:.3g}",
    })
print(pd.DataFrame(rows).to_string(index=False))
print("\np5/p50/p95 sobre los 857 picos: cuanto rango usa realmente cada score.")
