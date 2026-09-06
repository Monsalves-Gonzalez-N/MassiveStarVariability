"""Exact Mann-Whitney p-values: how much of this survives N=3.

With 3 real ELL against 43 false ones the smallest reachable p-value is
1/C(46,3) = 6.6e-5, so an AUC of 1.000 is not automatically significant and an
AUC of 0.9 needs the number spelled out. The matched-vs-spurious comparison
(16 vs 689) is the same scores measured where the statistics are not thin.
"""
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

from _common import (RESULTS, auc, argmax_class, brf_grouped, cnn_grouped,
                     load_brf, load_passes, load_peak_truth, load_peaks,
                     load_truth)

peaks = load_peaks()
star_truth, has_period, no_period = load_truth(peaks)
status, human_class = load_peak_truth(peaks)
brf = load_brf()

ensemble = load_passes("log")
brf_ensemble, names = brf_grouped(ensemble, peaks, brf)
cnn_ensemble, _ = cnn_grouped(ensemble)
cnn_dropout, _ = cnn_grouped(np.load(RESULTS / "cnn_mc200.npz")["p_mc"].astype(float))
brf_dropout, _ = brf_grouped(np.load(RESULTS / "cnn_mc20.npz")["p_mc"].astype(float),
                             peaks, brf)
rank_mean = np.nanmean(brf_grouped(load_passes("rank"), peaks, brf)[0], axis=0)

ELL, LPV, RNDM = names.index("ELL"), names.index("LPV"), names.index("Rndm")
brf_mean = np.nanmean(brf_ensemble, axis=0)
klass = argmax_class(brf_mean, names)
winner = np.where(np.isnan(brf_mean), -np.inf, brf_mean).argmax(1)
winner_passes = np.take_along_axis(brf_dropout, winner[None, :, None], axis=2)[..., 0]

is_ell = klass == "ELL"
real_ell = is_ell & (status == "matched") & (human_class == "ELL")
false_ell = is_ell & (status == "spurious")
matched = status == "matched"
spurious = status == "spurious"

scores = {
    "power": peaks.power.values,
    "-p_LPV(rank)": -rank_mean[:, LPV],
    "p16 ganadora (drop20)": np.nanpercentile(winner_passes, 16, axis=0),
    "-max p_Rndm (ens7)": -np.nanmax(cnn_ensemble[..., RNDM], axis=0),
    "-q90 p_Rndm (drop200)": -np.quantile(cnn_dropout[..., RNDM], 0.90, axis=0),
    "-sigma p_ELL (BRF drop20)": -np.nanstd(brf_dropout[..., ELL], axis=0),
}

rows = []
for label, score in scores.items():
    clean = np.where(np.isnan(score), -np.inf, score)
    _, p_ell = mannwhitneyu(clean[real_ell], clean[false_ell], alternative="greater")
    _, p_all = mannwhitneyu(clean[matched], clean[spurious], alternative="greater")
    rows.append({
        "score": label,
        "AUC 3v43": round(auc(score, real_ell, false_ell), 3),
        "p 3v43": f"{p_ell:.2g}",
        "AUC 16v689": round(auc(score, matched, spurious), 3),
        "p 16v689": f"{p_all:.2g}",
    })
print("=" * 84)
print("SIGNIFICANCIA (Mann-Whitney, una cola)")
print("=" * 84)
print(pd.DataFrame(rows).to_string(index=False))
print(f"\np minimo alcanzable con 3 vs 43: {1 / 15180:.1e}")
print("AUC 16v689 = separa picos con periodo humano confirmado de picos espurios,")
print("sobre TODAS las clases. Es la misma cantidad medida donde hay estadistica.")
