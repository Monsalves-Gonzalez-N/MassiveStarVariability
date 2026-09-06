"""Candidate gates against the corrected peak-level truth.

Every score is a single number per peak, large = "keep". The gate is applied
to ALL peaks, not only to the ELL ones, so the cost on the other signals (the
11 confirmed E peaks) is visible instead of being true by construction.
"""
import numpy as np
import pandas as pd

from _common import (RESULTS, auc, argmax_class, brf_grouped, cnn_grouped,
                     load_brf, load_passes, load_peak_truth, load_peaks,
                     load_truth)

FLOOR = 1e-12
PERIODIC_CLASSES = {"ELL", "Pulsating", "E"}
PROB_MIN = 0.8

peaks = load_peaks()
star_truth, has_period, no_period = load_truth(peaks)
status, human_class = load_peak_truth(peaks)
brf = load_brf()

ensemble = load_passes("log")
brf_ensemble, names = brf_grouped(ensemble, peaks, brf)
cnn_ensemble, _ = cnn_grouped(ensemble)
dropout = np.load(RESULTS / "cnn_mc200.npz")["p_mc"].astype(float)
cnn_dropout, _ = cnn_grouped(dropout)
brf_dropout, _ = brf_grouped(np.load(RESULTS / "cnn_mc20.npz")["p_mc"].astype(float),
                             peaks, brf)
rank_mean = np.nanmean(brf_grouped(load_passes("rank"), peaks, brf)[0], axis=0)

ELL, LPV, RNDM = names.index("ELL"), names.index("LPV"), names.index("Rndm")
brf_mean = np.nanmean(brf_ensemble, axis=0)
klass = argmax_class(brf_mean, names)
filled = np.where(np.isnan(brf_mean), -np.inf, brf_mean)
probability = np.nanmax(filled, axis=1)
winner = filled.argmax(axis=1)
winner_passes = np.take_along_axis(brf_dropout, winner[None, :, None], axis=2)[..., 0]

is_ell = klass == "ELL"
matched = status == "matched"
real_ell = is_ell & matched & (human_class == "ELL")          # 3
false_ell = is_ell & (status == "spurious")                   # 43
matched_e = matched & (human_class == "E") & (klass == "E")   # 11

scores = {
    "power": peaks.power.values,
    "prominence": peaks.prominence.values,
    "-max p_Rndm (CNN ens7)": -np.nanmax(cnn_ensemble[..., RNDM], axis=0),
    "-max p_Rndm (CNN drop200)": -np.nanmax(cnn_dropout[..., RNDM], axis=0),
    "-q90 p_Rndm (CNN drop200)": -np.quantile(cnn_dropout[..., RNDM], 0.90, axis=0),
    "-max p_Rndm (BRF drop20)": -np.nanmax(brf_dropout[..., RNDM], axis=0),
    "-voto Rndm (drop200)": -(cnn_dropout.argmax(-1) == RNDM).mean(axis=0),
    "-max p_LPV (BRF ens7)": -np.nanmax(brf_ensemble[..., LPV], axis=0),
    "-p_LPV rank (BRF)": -rank_mean[:, LPV],
    "p16 clase ganadora (BRF drop20)": np.nanpercentile(winner_passes, 16, axis=0),
    "-sigma p_ELL (BRF drop20)": -np.nanstd(brf_dropout[..., ELL], axis=0),
    "p_ELL - p_Rndm (BRF ens7)": brf_mean[:, ELL] - brf_mean[:, RNDM],
}

print("=" * 96)
print("AUC CON LA VERDAD CORREGIDA A NIVEL DE PICO")
print("=" * 96)
rows = []
for label, score in scores.items():
    rows.append({
        "score": label,
        "ELL real vs falso": round(auc(score, real_ell, false_ell), 3),
        "matched vs spurious": round(auc(score, matched, status == "spurious"), 3),
        "E real vs falso ELL": round(auc(score, matched_e, false_ell), 3),
    })
print(pd.DataFrame(rows).to_string(index=False))
print(f"\nN: ELL real={int(real_ell.sum())}  ELL falso={int(false_ell.sum())}  "
      f"matched={int(matched.sum())}  spurious={int((status == 'spurious').sum())}  "
      f"E confirmadas={int(matched_e.sum())}")


def evaluate(keep, label):
    """Star-level detection plus the peak-level cost on the confirmed signals."""
    call = pd.Series(keep & np.isin(klass, list(PERIODIC_CLASSES)) & (probability >= PROB_MIN),
                     index=peaks.key.values).groupby(level=0).any()
    aligned = star_truth.index.intersection(call.index)
    predicted, actual = call[aligned], star_truth[aligned]
    return {
        "regla": label,
        "TP": int((predicted & actual).sum()), "FN": int((~predicted & actual).sum()),
        "FP": int((predicted & ~actual).sum()), "TN": int((~predicted & ~actual).sum()),
        "ELL_falsos": int((keep & false_ell & (probability >= PROB_MIN)).sum()),
        "ELL_reales_3": int((keep & real_ell & (probability >= PROB_MIN)).sum()),
        "E_conf_11": int((keep & matched_e & (probability >= PROB_MIN)).sum()),
    }


print("\n" + "=" * 96)
print("REGLAS, APLICADAS A TODOS LOS PICOS")
print("=" * 96)
rndm_tail_ensemble = np.nanmax(cnn_ensemble[..., RNDM], axis=0)
rndm_tail_dropout = np.nanmax(cnn_dropout[..., RNDM], axis=0)
everything = np.ones(len(peaks), dtype=bool)
rules = [(everything, "sin filtro (baseline)"),
         (peaks.power.values >= 0.5, "power >= 0.5")]
for threshold in [0.5, 0.2, 0.1, 0.05, 0.01]:
    rules.append((rndm_tail_dropout < threshold, f"max p_Rndm(drop200) < {threshold}"))
for threshold in [0.1, 0.01, 0.001]:
    rules.append((rndm_tail_ensemble < threshold, f"max p_Rndm(ens7) < {threshold}"))
rules += [
    ((peaks.power.values >= 0.5) | (rndm_tail_dropout < 0.05),
     "power>=0.5 O max p_Rndm(drop200)<0.05"),
    ((peaks.power.values >= 0.3) & (rndm_tail_dropout < 0.5),
     "power>=0.3 Y max p_Rndm(drop200)<0.5"),
    ((peaks.power.values >= 0.5) & (rndm_tail_dropout < 0.5),
     "power>=0.5 Y max p_Rndm(drop200)<0.5"),
    ((peaks.power.values >= 0.5) & (rndm_tail_ensemble < 0.01),
     "power>=0.5 Y max p_Rndm(ens7)<0.01"),
]
print(pd.DataFrame([evaluate(keep, label) for keep, label in rules]).to_string(index=False))
print("\nTP/FN sobre 11 estrellas periodicas, FP/TN sobre 28 sin periodo.")
print("ELL_reales_3 y E_conf_11 son los picos confirmados que la regla conserva.")
