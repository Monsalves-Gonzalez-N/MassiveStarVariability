"""Is the p_Rndm tail across MC passes a real signal or a max-of-200 artefact?

The max over passes is the noisiest statistic there is, so it is checked
against the whole quantile ladder, against 20 passes instead of 200, and
against the vote fraction that `bimodal.py` already rejected.
"""
import numpy as np
import pandas as pd

from _common import (RESULTS, auc, argmax_class, brf_grouped, cnn_grouped,
                     load_brf, load_passes, load_peaks, load_truth)

peaks = load_peaks()
star_truth, has_period, no_period = load_truth(peaks)
brf = load_brf()

ensemble = load_passes("log")
brf_ensemble, names = brf_grouped(ensemble, peaks, brf)
ELL, LPV, RNDM = names.index("ELL"), names.index("LPV"), names.index("Rndm")
klass = argmax_class(np.nanmean(brf_ensemble, axis=0), names)
is_ell = klass == "ELL"
real, contaminant = is_ell & has_period, is_ell & no_period

dropout200, _ = cnn_grouped(np.load(RESULTS / "cnn_mc200.npz")["p_mc"].astype(float))
dropout20, _ = cnn_grouped(np.load(RESULTS / "cnn_mc20.npz")["p_mc"].astype(float))

print("=" * 84)
print("A. ESCALERA DE CUANTILES DE p_Rndm ENTRE PASADAS (dropout, CNN, log)")
print("=" * 84)
rows = []
for quantile in [0.50, 0.75, 0.90, 0.95, 0.99, 1.00]:
    row = {"cuantil": quantile}
    for label, block in [("drop200", dropout200), ("drop20", dropout20)]:
        value = np.quantile(block[..., RNDM], quantile, axis=0)
        row[f"AUC_{label}"] = round(auc(-value, real, contaminant), 3)
        row[f"med_real_{label}"] = round(float(np.median(value[real])), 4)
        row[f"med_cont_{label}"] = round(float(np.median(value[contaminant])), 4)
    rows.append(row)
print(pd.DataFrame(rows).to_string(index=False))

print("\n" + "=" * 84)
print("B. LA COLA CONTRA EL VOTO Y CONTRA SIGMA (lo que ya se habia probado)")
print("=" * 84)
rows = []
for label, block in [("drop200", dropout200), ("drop20", dropout20)]:
    argmax_per_pass = block.argmax(axis=-1)
    vote_rndm = (argmax_per_pass == RNDM).mean(axis=0)
    vote_ell = (argmax_per_pass == ELL).mean(axis=0)
    sigma_ell = block[..., ELL].std(axis=0)
    fraction_above = (block[..., RNDM] > 0.5).mean(axis=0)
    for name, score in [("voto Rndm (fraccion argmax)", -vote_rndm),
                        ("voto ELL (fraccion argmax)", vote_ell),
                        ("sigma de p_ELL", -sigma_ell),
                        ("fraccion de pasadas con p_Rndm>0.5", -fraction_above),
                        ("max p_Rndm", -block[..., RNDM].max(axis=0))]:
        rows.append({"pasadas": label, "score": name,
                     "AUC_ELL": round(auc(score, real, contaminant), 3),
                     "med_real": round(float(np.median(-score[real])), 4),
                     "med_cont": round(float(np.median(-score[contaminant])), 4)})
print(pd.DataFrame(rows).to_string(index=False))

print("\n" + "=" * 84)
print("C. LOS 46 PICOS UNO A UNO: q90 y max de p_Rndm, dropout 200")
print("=" * 84)
q90 = np.quantile(dropout200[..., RNDM], 0.90, axis=0)
maximum = dropout200[..., RNDM].max(axis=0)
mean_ell = dropout200[..., ELL].mean(axis=0)
detail = pd.DataFrame({
    "TIC": peaks.TIC[is_ell].values,
    "sector": peaks.sector[is_ell].values,
    "per": peaks.period[is_ell].values.round(3),
    "power": peaks.power[is_ell].values.round(3),
    "tipo": np.where(real[is_ell], "REAL", np.where(contaminant[is_ell], "contam", "-")),
    "media_p_ELL": mean_ell[is_ell].round(4),
    "q90_p_Rndm": q90[is_ell].round(4),
    "max_p_Rndm": maximum[is_ell].round(4),
}).sort_values(["tipo", "max_p_Rndm"])
print(detail.to_string(index=False))
