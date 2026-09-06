"""How many of the 43 false ELL die at the threshold that keeps all 3 real ones.

The gate is applied only to peaks the pipeline calls ELL, so by construction
it cannot touch the confirmed E or Pulsating peaks. The question is only how
much of the ELL contamination each score removes at zero cost on the real ELL.
"""
import numpy as np
import pandas as pd

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

scores = {
    "power": peaks.power.values,
    "-p_LPV(rank)": -rank_mean[:, LPV],
    "p16 ganadora (drop20)": np.nanpercentile(winner_passes, 16, axis=0),
    "-max p_Rndm (ens7)": -np.nanmax(cnn_ensemble[..., RNDM], axis=0),
    "-q90 p_Rndm (drop200)": -np.quantile(cnn_dropout[..., RNDM], 0.90, axis=0),
    "-sigma p_ELL (BRF drop20)": -np.nanstd(brf_dropout[..., ELL], axis=0),
}

print("=" * 90)
print("UMBRAL QUE CONSERVA LAS 3 ELL REALES: cuantos de los 43 falsos sobreviven")
print("=" * 90)
rows, survivors = [], {}
for label, score in scores.items():
    worst_real = np.nanmin(score[real_ell])
    keep = score >= worst_real
    survivors[label] = keep & false_ell
    rows.append({
        "score": label,
        "umbral": round(float(worst_real), 5),
        "AUC": round(auc(score, real_ell, false_ell), 3),
        "ELL_falsos_vivos": int((keep & false_ell).sum()),
        "de_43": f"{int((keep & false_ell).sum())}/43",
        "eliminados_%": round(100 * (1 - (keep & false_ell).sum() / 43), 1),
    })
print(pd.DataFrame(rows).to_string(index=False))

print("\n" + "=" * 90)
print("SOLAPAMIENTO: los scores matan a los MISMOS contaminantes o a distintos?")
print("=" * 90)
labels = list(survivors)
overlap = pd.DataFrame(index=labels, columns=labels, dtype=int)
for first in labels:
    for second in labels:
        overlap.loc[first, second] = int((survivors[first] & survivors[second]).sum())
print(overlap.to_string())
print("(diagonal = los que sobreviven a ese score; fuera de la diagonal = a los dos)")

print("\ncombinaciones (un pico muere si CUALQUIER score lo mata):")
combinations = [
    ("power + p_LPV(rank)", ["power", "-p_LPV(rank)"]),
    ("power + max p_Rndm(ens7)", ["power", "-max p_Rndm (ens7)"]),
    ("p_LPV(rank) + max p_Rndm(ens7)", ["-p_LPV(rank)", "-max p_Rndm (ens7)"]),
    ("p_LPV(rank) + p16", ["-p_LPV(rank)", "p16 ganadora (drop20)"]),
    ("p_LPV(rank) + p16 + max p_Rndm(ens7)",
     ["-p_LPV(rank)", "p16 ganadora (drop20)", "-max p_Rndm (ens7)"]),
    ("los cuatro sin power",
     ["-p_LPV(rank)", "p16 ganadora (drop20)", "-max p_Rndm (ens7)", "-sigma p_ELL (BRF drop20)"]),
    ("todos", labels),
]
for label, members in combinations:
    alive = np.ones(len(peaks), dtype=bool)
    for member in members:
        alive &= survivors[member] | ~false_ell
    print(f"  {label:40s} sobreviven {int((alive & false_ell).sum()):2d}/43 falsos, "
          f"3/3 reales")

print("\n" + "=" * 90)
print("LOS CONTAMINANTES DUROS: los que sobreviven a TODO")
print("=" * 90)
alive = np.ones(len(peaks), dtype=bool)
for member in labels:
    alive &= survivors[member] | ~false_ell
hard = alive & false_ell
print(pd.DataFrame({
    "TIC": peaks.TIC[hard].values, "sector": peaks.sector[hard].values,
    "per": peaks.period[hard].values.round(3),
    "power": peaks.power[hard].values.round(3),
    "p_ELL": brf_mean[hard, ELL].round(3),
    "p_LPV_rank": rank_mean[hard, LPV].round(4),
    "maxRndm_ens7": np.nanmax(cnn_ensemble[..., RNDM], axis=0)[hard].round(4),
}).to_string(index=False))
