"""Across-pass structure of the probability vector, not the argmax vote.

`ensemble_abstain.py` counted how many of the 7 checkpoints vote ELL and
`prob_snr.py` measured sigma of the top class. Both collapse the pass axis
before looking at the other classes. Here every pass keeps its whole vector,
so the tails (min p_ELL over passes, max p_LPV over passes) and the
cross-class covariance survive.
"""
import numpy as np
import pandas as pd

from _common import (RESULTS, auc, argmax_class, brf_grouped, cnn_grouped,
                     load_brf, load_passes, load_peaks, load_truth)

FLOOR = 1e-12

peaks = load_peaks()
star_truth, has_period, no_period = load_truth(peaks)
brf = load_brf()

ensemble = load_passes("log")                                   # (7, N, 8)
dropout = np.load(RESULTS / "cnn_mc200.npz")["p_mc"].astype(float)   # (200, N, 8)

cnn_ensemble, names = cnn_grouped(ensemble)
cnn_dropout, _ = cnn_grouped(dropout)
brf_ensemble, _ = brf_grouped(ensemble, peaks, brf)

ELL, LPV, RNDM = names.index("ELL"), names.index("LPV"), names.index("Rndm")
klass = argmax_class(np.nanmean(brf_ensemble, axis=0), names)
is_ell = klass == "ELL"
real, contaminant = is_ell & has_period, is_ell & no_period


def across_pass_scores(per_pass, label):
    """Statistics over the pass axis that the mean and the vote both erase."""
    p_ell = np.clip(per_pass[..., ELL], FLOOR, 1)
    p_lpv = np.clip(per_pass[..., LPV], FLOOR, 1)
    p_rndm = np.clip(per_pass[..., RNDM], FLOOR, 1)
    with np.errstate(invalid="ignore"):
        correlation_lpv = _pass_correlation(p_ell, p_lpv)
        correlation_rndm = _pass_correlation(p_ell, p_rndm)
    return {
        f"[{label}] min p_ELL entre pasadas": np.nanmin(p_ell, axis=0),
        f"[{label}] media p_ELL": np.nanmean(p_ell, axis=0),
        f"[{label}] -max p_LPV entre pasadas": -np.nanmax(p_lpv, axis=0),
        f"[{label}] -max p_Rndm entre pasadas": -np.nanmax(p_rndm, axis=0),
        f"[{label}] -max(p_LPV+p_Rndm)": -np.nanmax(p_lpv + p_rndm, axis=0),
        f"[{label}] min log10 p_ELL/p_LPV": np.nanmin(np.log10(p_ell / p_lpv), axis=0),
        f"[{label}] min log10 p_ELL/p_Rndm": np.nanmin(np.log10(p_ell / p_rndm), axis=0),
        f"[{label}] rango p_ELL (max-min)": -(np.nanmax(p_ell, axis=0) - np.nanmin(p_ell, axis=0)),
        f"[{label}] corr(p_ELL,p_LPV) pasadas": correlation_lpv,
        f"[{label}] corr(p_ELL,p_Rndm) pasadas": correlation_rndm,
    }


def _pass_correlation(first, second):
    a = first - np.nanmean(first, axis=0)
    b = second - np.nanmean(second, axis=0)
    numerator = np.nansum(a * b, axis=0)
    denominator = np.sqrt(np.nansum(a * a, axis=0) * np.nansum(b * b, axis=0))
    return np.where(denominator > 0, numerator / np.where(denominator > 0, denominator, 1), np.nan)


all_scores = {}
all_scores.update(across_pass_scores(cnn_ensemble, "CNN ens7"))
all_scores.update(across_pass_scores(cnn_dropout, "CNN drop200"))
all_scores.update(across_pass_scores(brf_ensemble, "BRF ens7"))

rows = []
for label, score in all_scores.items():
    rows.append({
        "score": label,
        "AUC_ELL": round(auc(score, real, contaminant), 3),
        "AUC_todos": round(auc(score, has_period, no_period), 3),
        "med_real": round(float(np.nanmedian(score[real])), 4),
        "med_cont": round(float(np.nanmedian(score[contaminant])), 4),
    })
print("=" * 88)
print("3. ESTADISTICOS ENTRE PASADAS  (AUC_ELL: 5 reales vs 41 contaminantes)")
print("=" * 88)
print(pd.DataFrame(rows).to_string(index=False))
print(f"\nreferencia power: AUC_ELL={auc(peaks.power.values, real, contaminant):.3f} "
      f"AUC_todos={auc(peaks.power.values, has_period, no_period):.3f}")
