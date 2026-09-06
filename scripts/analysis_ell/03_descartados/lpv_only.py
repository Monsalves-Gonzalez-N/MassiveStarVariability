"""One number, no periodogram: p_LPV of the CNN, taken over normalizations.

The search for a replacement of `power` returned p_LPV itself, measured more
robustly. So the two-stage rule collapses: instead of p_LPV under `log` AND a
peak-height cut, take the worst p_LPV the CNN gives across normalizations.
"""
import numpy as np
import pandas as pd

from _common import (NORMALIZATIONS, auc, argmax_class, brf_grouped,
                     cnn_grouped, load_brf, load_passes, load_peak_truth,
                     load_peaks, load_truth, REPO_ROOT)

PERIODIC_CLASSES = ["ELL", "Pulsating", "E"]
PROB_MIN = 0.8

peaks = load_peaks()
star_truth, has_period, no_period = load_truth(peaks)
status, human_class = load_peak_truth(peaks)
brf = load_brf()

blocks = {name: cnn_grouped(load_passes(name))[0] for name in NORMALIZATIONS}
brf_log, names = brf_grouped(load_passes("log"), peaks, brf)
LPV = names.index("LPV")
brf_mean = np.nanmean(brf_log, axis=0)
filled = np.where(np.isnan(brf_mean), -np.inf, brf_mean)
klass = argmax_class(brf_mean, names)
probability = np.nanmax(filled, axis=1)

candidates = {
    "p_LPV(log), media pasadas": np.nanmean(blocks["log"][..., LPV], axis=0),
    "p_LPV(log), max pasadas": np.nanmax(blocks["log"][..., LPV], axis=0),
    "p_LPV, max entre 9 normas": np.stack(
        [np.nanmean(blocks[name][..., LPV], axis=0) for name in NORMALIZATIONS]).max(axis=0),
    "p_LPV, max normas Y pasadas": np.stack(
        [np.nanmax(blocks[name][..., LPV], axis=0) for name in NORMALIZATIONS]).max(axis=0),
    "power (referencia)": -peaks.power.values,
}

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

print("=" * 78)
print("CURVA DE PUREZA, UN SOLO EJE (sin power)")
print("=" * 78)
for label, score in candidates.items():
    value = score[best_index]
    print(f"\n{label}   AUC = {auc(-value, star_correct, ~star_correct):.3f}")
    print(f"  {'umbral':>10} {'correctas':>10} {'erradas':>8} {'pureza':>8}")
    for threshold in [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]:
        if label.startswith("power"):
            threshold = -threshold * 0 - {1e-9: 0.9, 1e-8: 0.8, 1e-7: 0.7,
                                          1e-6: 0.6, 1e-5: 0.5, 1e-4: 0.4,
                                          1e-3: 0.3, 1e-2: 0.2}[threshold]
        keep = value < threshold
        n_correct = int((keep & star_correct).sum())
        n_wrong = int((keep & ~star_correct).sum())
        total = n_correct + n_wrong
        purity = 100 * n_correct / total if total else np.nan
        print(f"  {threshold:10.1e} {n_correct:8d}/8 {n_wrong:6d}/18 "
              f"{purity:7.0f}%")

best = candidates["p_LPV, max entre 9 normas"]
print("\n" + "=" * 96)
print("LAS 26 ESTRELLAS POR p_LPV MAXIMO ENTRE NORMAS")
print("=" * 96)
frame = pd.DataFrame({
    "key": keys, "clase": klass[best_index],
    "per": peaks.period.values[best_index].round(3),
    "power": peaks.power.values[best_index].round(3),
    "p_LPV_log": [f"{value:.1e}" for value in
                  candidates["p_LPV(log), media pasadas"][best_index]],
    "p_LPV_max_normas": [f"{value:.1e}" for value in best[best_index]],
    "acierto": np.where(star_correct, "CORRECTA", "ERRADA"),
    "nota": [note_by_key.get(key, "") for key in keys],
}).sort_values("p_LPV_max_normas")
print(frame.to_string(index=False))
