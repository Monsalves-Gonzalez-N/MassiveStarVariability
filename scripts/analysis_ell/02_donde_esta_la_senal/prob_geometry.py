"""Geometry of the probability vector on the ELL peaks, without argmax.

`norm_matrix.py` compared normalizations through the argmax and concluded the
contaminants and the real ELL behave identically. The argmax keeps one number
out of five; this script keeps the whole vector and asks where the residual
mass (1 - p_ELL) goes, which is the quantity the argmax destroys.
"""
import numpy as np
import pandas as pd

from _common import (auc, argmax_class, brf_grouped, cnn_grouped, load_brf,
                     load_passes, load_peaks, load_truth)

FLOOR = 1e-12

peaks = load_peaks()
star_truth, has_period, no_period = load_truth(peaks)
brf = load_brf()

passes_log = load_passes("log")
cnn_passes, names = cnn_grouped(passes_log)
brf_passes, _ = brf_grouped(passes_log, peaks, brf)
cnn_mean = np.nanmean(cnn_passes, axis=0)
brf_mean = np.nanmean(brf_passes, axis=0)

ELL = names.index("ELL")
LPV = names.index("LPV")
RNDM = names.index("Rndm")

klass = argmax_class(brf_mean, names)
is_ell = klass == "ELL"
real = is_ell & has_period
contaminant = is_ell & no_period

print(f"clases agrupadas: {names}")
print(f"picos ELL bajo log: {int(real.sum())} reales, "
      f"{int(contaminant.sum())} contaminantes, "
      f"{int(is_ell.sum())} en total\n")

# ---- 1. what the argmax throws away ---------------------------------------
print("=" * 78)
print("1. EL VECTOR COMPLETO EN LOS PICOS QUE log LLAMA ELL")
print("=" * 78)
for level, mean in [("CNN", cnn_mean), ("BRF", brf_mean)]:
    rows = []
    for label, mask in [("ELL real", real), ("contaminante", contaminant)]:
        row = {"nivel": level, "grupo": label, "N": int(mask.sum())}
        for position, name in enumerate(names):
            row[name] = round(float(np.median(mean[mask, position])), 4)
        rows.append(row)
    print(pd.DataFrame(rows).to_string(index=False))
print("\n(medianas por clase; el argmax solo veia la columna ELL)")

print("\nsegunda clase (runner-up) en esos mismos picos:")
for level, mean in [("CNN", cnn_mean), ("BRF", brf_mean)]:
    order = np.argsort(-np.where(np.isnan(mean), -np.inf, mean), axis=1)
    runner_up = np.array([names[i] for i in order[:, 1]])
    table = pd.crosstab(
        pd.Series(runner_up[is_ell], name=f"runner-up {level}"),
        pd.Series(np.where(real[is_ell], "REAL", "contaminante"), name=""))
    print(table.to_string())

# ---- 2. p_ELL against p_LPV and p_Rndm ------------------------------------
print("\n" + "=" * 78)
print("2. p_ELL CONTRA p_LPV Y p_Rndm  (AUC real vs contaminante, N=5 vs 41)")
print("=" * 78)
scores = {}
for level, mean in [("CNN", cnn_mean), ("BRF", brf_mean)]:
    p_ell = np.clip(mean[:, ELL], FLOOR, 1)
    p_lpv = np.clip(mean[:, LPV], FLOOR, 1)
    p_rndm = np.clip(mean[:, RNDM], FLOOR, 1)
    residual = np.clip(1 - mean[:, ELL], FLOOR, 1)
    scores[level] = {
        "p_ELL": p_ell,
        "-p_LPV": -p_lpv,
        "-p_Rndm": -p_rndm,
        "-(p_LPV+p_Rndm)": -(p_lpv + p_rndm),
        "log10 p_ELL/p_LPV": np.log10(p_ell / p_lpv),
        "log10 p_ELL/p_Rndm": np.log10(p_ell / p_rndm),
        "-share LPV del residuo": -p_lpv / residual,
        "-share Rndm del residuo": -p_rndm / residual,
    }

rows = []
for label in scores["CNN"]:
    row = {"score": label}
    for level in ("CNN", "BRF"):
        score = scores[level][label]
        row[f"AUC_{level}_ELL"] = round(auc(score, real, contaminant), 3)
        row[f"AUC_{level}_todos"] = round(auc(score, has_period, no_period), 3)
    rows.append(row)
print(pd.DataFrame(rows).to_string(index=False))
print("\nAUC_*_ELL   : separa ELL real de contaminante (5 vs 41)")
print("AUC_*_todos : separa picos de estrella periodica de no periodica "
      f"({int(has_period.sum())} vs {int(no_period.sum())})")
print("referencia: power  AUC_ELL = "
      f"{auc(peaks.power.values, real, contaminant):.3f}   AUC_todos = "
      f"{auc(peaks.power.values, has_period, no_period):.3f}")
