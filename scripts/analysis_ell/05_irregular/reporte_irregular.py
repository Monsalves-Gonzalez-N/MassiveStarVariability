"""The two-number report: class from the peak, irregularity from the star.

`p_LPV` averaged over every peak of a star-sector is a share of a distribution
that sums to 1, so it reads directly as a percentage. Rndm is deliberately not
counted as irregular: it is "no signal in this peak" and every star has ~19 of
those, which is why p_LPV+p_Rndm inverts (AUC 0.16).
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
cnn_mean = np.nanmean(cnn_ensemble, axis=0)
LPV, RNDM = names.index("LPV"), names.index("Rndm")
brf_mean = np.nanmean(brf_grouped(ensemble, peaks, brf)[0], axis=0)
filled = np.where(np.isnan(brf_mean), -np.inf, brf_mean)
klass = argmax_class(brf_mean, names)
probability = np.nanmax(filled, axis=1)
lpv_peak = np.nanmax(cnn_ensemble[..., LPV], axis=0)

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
    spectrum = cnn_mean[same_star].mean(axis=0)
    signal = spectrum.sum() - spectrum[RNDM]
    records.append({
        "key": key, "clase": klass[best],
        "per": round(float(peaks.period.values[best]), 3),
        "p_LPV_pico": lpv_peak[best],
        "irregular": spectrum[LPV],
        "irregular_sin_ruido": spectrum[LPV] / signal if signal > 0 else np.nan,
        "correcta": status[best] == "matched",
        "nota": note_by_key.get(key, ""),
    })
frame = pd.DataFrame(records)
correct = frame.correcta.values

print("=" * 74)
print("DOS DEFINICIONES DEL PORCENTAJE IRREGULAR")
print("=" * 74)
for label, column in [("p_LPV medio (masa cruda)", "irregular"),
                      ("p_LPV / (1 - p_Rndm)  (masa con senal)", "irregular_sin_ruido")]:
    score = frame[column].values
    _, p_value = mannwhitneyu(-score[correct], -score[~correct], alternative="greater")
    print(f"{label:40s} AUC {auc(-score, correct, ~correct):.3f}  p {p_value:.1g}   "
          f"mediana {100 * np.median(score[correct]):.1f}% vs "
          f"{100 * np.median(score[~correct]):.1f}%")

peak_score = frame.p_LPV_pico.values
best_pair, best_kept = None, 0
for peak_cut in np.unique(peak_score):
    for global_cut in np.unique(frame.irregular.values):
        keep = ((peak_score < peak_cut * 1.000001)
                & (frame.irregular.values < global_cut * 1.000001))
        if (keep & ~correct).sum() == 0 and (keep & correct).sum() > best_kept:
            best_kept, best_pair = int((keep & correct).sum()), (peak_cut, global_cut)
peak_cut, global_cut = best_pair
peak_margin = np.min(peak_score[~correct & (frame.irregular.values < global_cut * 1.000001)])
print("\n" + "=" * 74)
print("EL PUNTO 7/8 SIN ERRORES")
print("=" * 74)
keep = (peak_score <= peak_cut) & (frame.irregular.values <= global_cut)
print(f"  regla: p_LPV(pico) <= {peak_cut:.1e}  Y  irregular <= {100 * global_cut:.1f}%")
print(f"  conserva {int((keep & correct).sum())}/8 correctas, "
      f"{int((keep & ~correct).sum())}/18 erradas")
inside_global = frame.irregular.values <= global_cut
margin_peak = np.min(peak_score[~correct & inside_global]) / np.max(peak_score[correct & keep])
inside_peak = peak_score <= peak_cut
margin_global = (np.min(frame.irregular.values[~correct & inside_peak])
                 / np.max(frame.irregular.values[correct & keep]))
print(f"  margen en el eje del pico    x{margin_peak:.1f}")
print(f"  margen en el eje irregular   x{margin_global:.1f}")

print("\n" + "=" * 100)
print("EL REPORTE PROPUESTO, LAS 26 ESTRELLAS")
print("=" * 100)
shown = frame.sort_values("irregular").copy()
shown["reporte"] = [f"{row.clase} con {100 * row.irregular:.0f}% irregular"
                    for _, row in shown.iterrows()]
shown["p_LPV_pico"] = shown.p_LPV_pico.map(lambda value: f"{value:.1e}")
shown["acierto"] = np.where(shown.correcta, "CORRECTA", "ERRADA")
print(shown[["key", "per", "p_LPV_pico", "reporte", "acierto", "nota"]].to_string(index=False))
