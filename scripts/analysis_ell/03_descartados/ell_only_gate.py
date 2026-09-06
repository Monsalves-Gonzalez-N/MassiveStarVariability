"""The gate restricted to the ELL peaks: the other classes are untouched.

Applying the tail cut to every peak costs confirmed E and Pulsating
detections. Restricted to ELL it cannot, so the only question left is how much
ELL contamination goes away and what it does to the star-level counts.
"""
import numpy as np
import pandas as pd

from _common import (RESULTS, argmax_class, brf_grouped, cnn_grouped, load_brf,
                     load_passes, load_peak_truth, load_peaks, load_truth)

PERIODIC_CLASSES = {"ELL", "Pulsating", "E"}
PROB_MIN = 0.8

peaks = load_peaks()
star_truth, has_period, no_period = load_truth(peaks)
status, human_class = load_peak_truth(peaks)
brf = load_brf()

ensemble = load_passes("log")
brf_ensemble, names = brf_grouped(ensemble, peaks, brf)
cnn_ensemble, _ = cnn_grouped(ensemble)
rank_mean = np.nanmean(brf_grouped(load_passes("rank"), peaks, brf)[0], axis=0)
minmax_mean = np.nanmean(brf_grouped(load_passes("min_max"), peaks, brf)[0], axis=0)
ELL, LPV, RNDM = names.index("ELL"), names.index("LPV"), names.index("Rndm")

brf_mean = np.nanmean(brf_ensemble, axis=0)
klass = argmax_class(brf_mean, names)
probability = np.nanmax(np.where(np.isnan(brf_mean), -np.inf, brf_mean), axis=1)
rndm_tail = np.nanmax(cnn_ensemble[..., RNDM], axis=0)
p_lpv_rank = rank_mean[:, LPV]
p_lpv_minmax = minmax_mean[:, LPV]

is_ell = klass == "ELL"
real_ell = is_ell & (status == "matched") & (human_class == "ELL")
false_ell = is_ell & (status == "spurious")
matched = status == "matched"

print("de donde vienen los 15 falsos positivos del baseline:")
call = np.isin(klass, list(PERIODIC_CLASSES)) & (probability >= PROB_MIN)
false_star = np.array([bool(star_truth.get(key)) is False and key in star_truth.index
                       for key in peaks.key])
print(pd.Series(klass[call & false_star]).value_counts().to_string())
stars_by_class = {}
for class_name in PERIODIC_CLASSES:
    stars_by_class[class_name] = set(peaks.key[call & false_star & (klass == class_name)])
print("\nestrellas sin periodo llamadas periodicas, por clase que las llama:")
for class_name, keys in stars_by_class.items():
    print(f"  {class_name:10s} {len(keys):2d} estrellas")
print(f"  solo por ELL: "
      f"{len(stars_by_class['ELL'] - stars_by_class['E'] - stars_by_class['Pulsating'])}")


def star_level(keep):
    called = pd.Series(keep & np.isin(klass, list(PERIODIC_CLASSES)) & (probability >= PROB_MIN),
                       index=peaks.key.values).groupby(level=0).any()
    aligned = star_truth.index.intersection(called.index)
    predicted, actual = called[aligned], star_truth[aligned]
    return (int((predicted & actual).sum()), int((~predicted & actual).sum()),
            int((predicted & ~actual).sum()), int((~predicted & ~actual).sum()))


print("\n" + "=" * 100)
print("GATE APLICADO SOLO A LOS PICOS ELL")
print("=" * 100)
gates = {
    "sin filtro": np.ones(len(peaks), dtype=bool),
    "power >= 0.5 (referencia, global)": peaks.power.values >= 0.5,
    "ELL: p_LPV(rank) == 0": ~is_ell | (p_lpv_rank == 0),
    "ELL: max p_Rndm(ens7) < 1e-3": ~is_ell | (rndm_tail < 1e-3),
    "ELL: max p_Rndm(ens7) < 1e-4": ~is_ell | (rndm_tail < 1e-4),
    "ELL: p_LPV(rank)==0 Y max p_Rndm<1e-3": ~is_ell | ((p_lpv_rank == 0) & (rndm_tail < 1e-3)),
    "ELL: p_LPV(rank)==0 Y max p_Rndm<1e-4": ~is_ell | ((p_lpv_rank == 0) & (rndm_tail < 1e-4)),
    "ELL: p_LPV(rank)==0 Y max p_Rndm<1e-5": ~is_ell | ((p_lpv_rank == 0) & (rndm_tail < 1e-5)),
    "ELL: p_LPV(min_max) < 1e-4": ~is_ell | (p_lpv_minmax < 1e-4),
    "ELL: p_LPV(min_max) < 1e-3": ~is_ell | (p_lpv_minmax < 1e-3),
    "ELL: p_LPV(min_max) < 1e-2": ~is_ell | (p_lpv_minmax < 1e-2),
    "ELL: p_LPV(min_max)<1e-3 Y max p_Rndm<1e-3":
        ~is_ell | ((p_lpv_minmax < 1e-3) & (rndm_tail < 1e-3)),
    "GLOBAL: p_LPV(min_max) < 1e-3": p_lpv_minmax < 1e-3,
}
rows = []
for label, keep in gates.items():
    tp, fn, fp, tn = star_level(keep)
    rows.append({
        "gate": label,
        "matched_16": int((keep & matched).sum()),
        "ELL_real_3": int((keep & real_ell).sum()),
        "ELL_falso_43": int((keep & false_ell).sum()),
        "ELL_total_53": int((keep & is_ell).sum()),
        "TP": tp, "FN": fn, "FP": fp, "TN": tn,
        "acc": round((tp + tn) / (tp + fn + fp + tn), 3),
    })
print(pd.DataFrame(rows).to_string(index=False))
print("\nmatched_16 = picos con periodo humano confirmado que sobreviven (todas las clases)")
print("TP/FN sobre 11 estrellas periodicas; FP/TN sobre 28 sin periodo")
