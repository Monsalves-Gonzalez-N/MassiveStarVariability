"""p_LPV against p_Rndm, both from the CNN, both under `log` alone.

No periodogram and no second normalization: the two probabilities the residual
mass falls into, read on the production run. The grid scan reports the whole
purity/recall frontier rather than one hand-picked pair of thresholds.
"""
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

from _common import (auc, argmax_class, brf_grouped, cnn_grouped, load_brf,
                     load_passes, load_peak_truth, load_peaks, load_truth,
                     REPO_ROOT)

PERIODIC_CLASSES = ["ELL", "Pulsating", "E"]
PROB_MIN = 0.8

peaks = load_peaks()
star_truth, has_period, no_period = load_truth(peaks)
status, human_class = load_peak_truth(peaks)
brf = load_brf()

passes_log = load_passes("log")
cnn_log, names = cnn_grouped(passes_log)
brf_log, _ = brf_grouped(passes_log, peaks, brf)
LPV, RNDM = names.index("LPV"), names.index("Rndm")
brf_mean = np.nanmean(brf_log, axis=0)
filled = np.where(np.isnan(brf_mean), -np.inf, brf_mean)
klass = argmax_class(brf_mean, names)
probability = np.nanmax(filled, axis=1)

lpv_mean = np.nanmean(cnn_log[..., LPV], axis=0)
lpv_max = np.nanmax(cnn_log[..., LPV], axis=0)
rndm_mean = np.nanmean(cnn_log[..., RNDM], axis=0)
rndm_max = np.nanmax(cnn_log[..., RNDM], axis=0)

notes = pd.read_csv(REPO_ROOT / "catalogs" / "notas_periodos.csv")
note_by_key = {f"{int(row.TIC)}_{int(row.sector)}": str(row.nota)
               for _, row in notes.iterrows()}
reported = np.isin(klass, PERIODIC_CLASSES) & (probability >= PROB_MIN)
best_index, keys = [], []
for key in star_truth.index:
    belongs = (peaks.key.values == key) & reported
    if belongs.any():
        best_index.append(int(np.where(belongs)[0][np.argmax(probability[belongs])]))
        keys.append(key)
best_index = np.array(best_index)
star_correct = status[best_index] == "matched"

is_ell = klass == "ELL"
real_ell = is_ell & (status == "matched") & (human_class == "ELL")
false_ell = is_ell & (status == "spurious")
matched, spurious = status == "matched", status == "spurious"

print("=" * 96)
print("LAS 26 ESTRELLAS: p_LPV Y p_Rndm DE LA CNN BAJO log")
print("=" * 96)
frame = pd.DataFrame({
    "key": keys, "clase": klass[best_index],
    "per": peaks.period.values[best_index].round(3),
    "p_LPV_media": [f"{value:.1e}" for value in lpv_mean[best_index]],
    "p_LPV_max": [f"{value:.1e}" for value in lpv_max[best_index]],
    "p_Rndm_media": [f"{value:.1e}" for value in rndm_mean[best_index]],
    "p_Rndm_max": [f"{value:.1e}" for value in rndm_max[best_index]],
    "acierto": np.where(star_correct, "CORRECTA", "ERRADA"),
    "nota": [note_by_key.get(key, "") for key in keys],
}).sort_values("acierto")
print(frame.to_string(index=False))

print("\n" + "=" * 78)
print("CADA UNO POR SEPARADO")
print("=" * 78)
singles = {"p_LPV media": lpv_mean, "p_LPV max": lpv_max,
           "p_Rndm media": rndm_mean, "p_Rndm max": rndm_max}
rows = []
for label, score in singles.items():
    clean = np.where(np.isnan(score), np.inf, score)
    _, p_star = mannwhitneyu(-clean[best_index][star_correct],
                             -clean[best_index][~star_correct], alternative="greater")
    rows.append({"score": label,
                 "AUC_estrella": round(auc(-score[best_index], star_correct, ~star_correct), 3),
                 "p_estrella": f"{p_star:.1g}",
                 "AUC_3v43": round(auc(-score, real_ell, false_ell), 3),
                 "AUC_16v689": round(auc(-score, matched, spurious), 3)})
print(pd.DataFrame(rows).to_string(index=False))

print("\n" + "=" * 90)
print("EL PLANO (p_LPV, p_Rndm): mejor pureza por cada numero de correctas conservadas")
print("=" * 90)
grid = np.array([10.0 ** exponent for exponent in np.arange(-9, 0.5, 0.25)])
for lpv_label, lpv_score in [("media", lpv_mean), ("max", lpv_max)]:
    for rndm_label, rndm_score in [("media", rndm_mean), ("max", rndm_max)]:
        frontier = {}
        for lpv_cut in grid:
            for rndm_cut in grid:
                keep = ((lpv_score[best_index] < lpv_cut)
                        & (rndm_score[best_index] < rndm_cut))
                n_correct = int((keep & star_correct).sum())
                n_wrong = int((keep & ~star_correct).sum())
                if n_correct == 0:
                    continue
                previous = frontier.get(n_correct)
                if previous is None or n_wrong < previous[0]:
                    frontier[n_correct] = (n_wrong, lpv_cut, rndm_cut)
        summary = "  ".join(
            f"{n}c/{frontier[n][0]}e" for n in sorted(frontier, reverse=True))
        print(f"  p_LPV {lpv_label:>5} Y p_Rndm {rndm_label:>5}:  {summary}")
print("\n(nc/me = n correctas de 8 conservadas con m erradas de 18 coladas)")
