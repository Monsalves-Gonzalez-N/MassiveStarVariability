"""A star-level variability spectrum: aggregate every peak, not just the reported one.

Each peak carries a CNN probability vector that sums to 1. Averaging those
vectors over all peaks of a star-sector gives another vector that sums to 1,
so it is a genuine distribution over classes at star level and needs no
rescaling. The mass it puts on LPV + Rndm is the "% irregular" of the report,
and the mass on the reported class is how much of the star's variability that
class actually accounts for.
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

ensemble = load_passes("log")
cnn_ensemble, names = cnn_grouped(ensemble)
cnn_mean = np.nanmean(cnn_ensemble, axis=0)               # (N, 5), sums to 1
ELL, PULS, E, LPV, RNDM = range(5)
brf_mean = np.nanmean(brf_grouped(ensemble, peaks, brf)[0], axis=0)
filled = np.where(np.isnan(brf_mean), -np.inf, brf_mean)
klass = argmax_class(brf_mean, names)
probability = np.nanmax(filled, axis=1)
lpv_peak = np.nanmax(cnn_ensemble[..., LPV], axis=0)      # the current rule

notes = pd.read_csv(REPO_ROOT / "catalogs" / "notas_periodos.csv")
note_by_key = {f"{int(row.TIC)}_{int(row.sector)}": str(row.nota)
               for _, row in notes.iterrows()}

reported = np.isin(klass, PERIODIC_CLASSES) & (probability >= PROB_MIN)
records = []
for key in star_truth.index:
    same_star = peaks.key.values == key
    belongs = same_star & reported
    if not belongs.any():
        continue
    best = int(np.where(belongs)[0][np.argmax(probability[belongs])])
    others = same_star.copy()
    others[best] = False
    spectrum = cnn_mean[same_star].mean(axis=0)           # sums to 1
    spectrum_others = cnn_mean[others].mean(axis=0)
    weights = peaks.power.values[same_star]
    weighted = (cnn_mean[same_star] * weights[:, None]).sum(axis=0) / weights.sum()
    records.append({
        "key": key, "clase": klass[best],
        "per": round(float(peaks.period.values[best]), 3),
        "n_picos": int(same_star.sum()),
        "p_LPV_pico": lpv_peak[best],
        "irregular_LPV": spectrum[LPV],
        "irregular_LPV_otros": spectrum_others[LPV],
        "irregular_LPV_Rndm": spectrum[LPV] + spectrum[RNDM],
        "irregular_pesado": weighted[LPV],
        "masa_de_la_clase": spectrum[names.index(klass[best])],
        "correcta": status[best] == "matched",
        "nota": note_by_key.get(key, ""),
    })
frame = pd.DataFrame(records)
correct = frame.correcta.values


def frontier(score, higher_is_better=False):
    value = -score if higher_is_better else score
    first_error = np.min(value[~correct])
    kept = int((value[correct] < first_error).sum())
    if not kept:
        return 0, np.nan
    return kept, float(first_error / np.sort(value[correct])[kept - 1])


print("=" * 92)
print("CANDIDATOS A VALOR GLOBAL DE IRREGULARIDAD, POR ESTRELLA-SECTOR")
print("=" * 92)
rows = []
for label, column, higher in [
        ("p_LPV del pico (regla actual)", "p_LPV_pico", False),
        ("p_LPV medio, TODOS los picos", "irregular_LPV", False),
        ("p_LPV medio, los OTROS picos", "irregular_LPV_otros", False),
        ("p_LPV+p_Rndm medio", "irregular_LPV_Rndm", False),
        ("p_LPV medio pesado por power", "irregular_pesado", False),
        ("masa de la clase reportada", "masa_de_la_clase", True)]:
    score = frame[column].values
    kept, margin = frontier(score, higher)
    signed = score if higher else -score
    _, p_value = mannwhitneyu(signed[correct], signed[~correct], alternative="greater")
    rows.append({"valor": label,
                 "AUC": round(auc(signed, correct, ~correct), 3),
                 "p": f"{p_value:.1g}",
                 "correctas_sin_error": f"{kept}/8",
                 "margen": round(margin, 1) if np.isfinite(margin) else np.nan,
                 "mediana_CORR": f"{np.median(score[correct]):.2e}",
                 "mediana_ERR": f"{np.median(score[~correct]):.2e}"})
print(pd.DataFrame(rows).to_string(index=False))

print("\n" + "=" * 92)
print("EL GLOBAL AGREGA INFORMACION AL PICO? frontera de la combinacion")
print("=" * 92)
peak_score = frame.p_LPV_pico.values
for label, column in [("p_LPV medio TODOS", "irregular_LPV"),
                      ("p_LPV medio OTROS", "irregular_LPV_otros"),
                      ("p_LPV+p_Rndm medio", "irregular_LPV_Rndm")]:
    global_score = frame[column].values
    best_pair = (0, np.nan)
    for peak_cut in np.unique(peak_score):
        for global_cut in np.unique(global_score):
            keep = (peak_score < peak_cut * 1.000001) & (global_score < global_cut * 1.000001)
            if (keep & ~correct).sum() == 0 and (keep & correct).sum() > best_pair[0]:
                best_pair = (int((keep & correct).sum()), (peak_cut, global_cut))
    print(f"  pico Y {label:20s} -> {best_pair[0]}/8 correctas sin error")
print(f"  solo el pico              -> {frontier(peak_score)[0]}/8 "
      f"(margen x{frontier(peak_score)[1]:.1f})")
