"""p_LPV read on the CNN, under the production normalization.

The BRF drops this score from 0.958 to 0.740 at star level, so the signal the
handoff called "unexplained, only visible under rank" was being destroyed by
the second stage, not created by the first. `log` needs no extra CNN run.
"""
import itertools

import numpy as np
import pandas as pd

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
brf_log, names = brf_grouped(passes_log, peaks, brf)
cnn_log, _ = cnn_grouped(passes_log)
cnn_binary, _ = cnn_grouped(load_passes("binary"))
ELL, LPV, RNDM = names.index("ELL"), names.index("LPV"), names.index("Rndm")

brf_mean = np.nanmean(brf_log, axis=0)
filled = np.where(np.isnan(brf_mean), -np.inf, brf_mean)
klass = argmax_class(brf_mean, names)
probability = np.nanmax(filled, axis=1)

lpv_cnn = np.nanmean(cnn_log[..., LPV], axis=0)
lpv_cnn_tail = np.nanmax(cnn_log[..., LPV], axis=0)
lpv_brf = brf_mean[:, LPV]
binary_tail = np.nanmax(cnn_binary[..., RNDM], axis=0)

notes = pd.read_csv(REPO_ROOT / "catalogs" / "notas_periodos.csv")
note_by_key = {f"{int(row.TIC)}_{int(row.sector)}": str(row.nota)
               for _, row in notes.iterrows()}

reported = np.isin(klass, PERIODIC_CLASSES) & (probability >= PROB_MIN)
records = []
for key in star_truth.index:
    belongs = (peaks.key.values == key) & reported
    if not belongs.any():
        continue
    best = np.where(belongs)[0][np.argmax(probability[belongs])]
    records.append({
        "key": key, "clase": klass[best],
        "per": round(float(peaks.period[best]), 3),
        "power": float(peaks.power.values[best]),
        "p_LPV_CNN": float(lpv_cnn[best]),
        "p_LPV_BRF": float(lpv_brf[best]),
        "cola_binary": float(binary_tail[best]),
        "correcta": status[best] == "matched",
        "nota": note_by_key.get(key, ""),
    })
frame = pd.DataFrame(records)
correct = frame.correcta.values

print("=" * 100)
print("LAS 26 ESTRELLAS REPORTADAS, ORDENADAS POR p_LPV DE LA CNN (log)")
print("=" * 100)
shown = frame.sort_values("p_LPV_CNN").copy()
for column in ["p_LPV_CNN", "p_LPV_BRF", "cola_binary"]:
    shown[column] = shown[column].map(lambda value: f"{value:.2e}")
shown["power"] = shown.power.round(3)
shown["acierto"] = np.where(shown.correcta, "CORRECTA", "ERRADA")
print(shown[["key", "clase", "per", "power", "p_LPV_CNN", "p_LPV_BRF",
             "cola_binary", "acierto", "nota"]].to_string(index=False))

print("\n" + "=" * 100)
print("COMBINACIONES DE EJES (AND)")
print("=" * 100)
axes = {
    "power>=0.5": frame.power.values >= 0.5,
    "p_LPV_CNN<1e-5": frame.p_LPV_CNN.values < 1e-5,
    "p_LPV_CNN<1e-4": frame.p_LPV_CNN.values < 1e-4,
    "cola_binary<1e-4": frame.cola_binary.values < 1e-4,
}
rows = []
for size in range(1, 4):
    for combination in itertools.combinations(axes, size):
        if "p_LPV_CNN<1e-5" in combination and "p_LPV_CNN<1e-4" in combination:
            continue
        keep = np.ones(len(frame), dtype=bool)
        for name in combination:
            keep &= axes[name]
        n_correct = int((keep & correct).sum())
        n_wrong = int((keep & ~correct).sum())
        rows.append({"ejes": " Y ".join(combination),
                     "correctas": f"{n_correct}/8", "erradas": f"{n_wrong}/18",
                     "pureza_%": round(100 * n_correct / max(n_correct + n_wrong, 1))})
print(pd.DataFrame(rows).sort_values("pureza_%", ascending=False).to_string(index=False))
print(f"\nAUC a nivel estrella: p_LPV CNN {auc(-frame.p_LPV_CNN.values, correct, ~correct):.3f}  "
      f"p_LPV BRF {auc(-frame.p_LPV_BRF.values, correct, ~correct):.3f}  "
      f"cola binary {auc(-frame.cola_binary.values, correct, ~correct):.3f}  "
      f"power {auc(frame.power.values, correct, ~correct):.3f}")
