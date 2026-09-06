"""The one star that stays confident and wrong, and the alias error hiding
inside the "ERRADA" column.

Three of the wrong reports sit on genuinely periodic stars at twice the human
period, which is a different failure from a spurious ELL: the class is right,
the period is doubled. `log` was validated against P/2 folding, never against
2P.
"""
import numpy as np
import pandas as pd

from _common import (NORMALIZATIONS, argmax_class, brf_grouped, cnn_grouped,
                     load_brf, load_passes, load_peak_truth, load_peaks,
                     load_truth, REPO_ROOT)

PERIODIC_CLASSES = ["ELL", "Pulsating", "E"]
PROB_MIN = 0.8

peaks = load_peaks()
star_truth, has_period, no_period = load_truth(peaks)
status, human_class = load_peak_truth(peaks)
brf = load_brf()

brf_log, names = brf_grouped(load_passes("log"), peaks, brf)
brf_mean = np.nanmean(brf_log, axis=0)
filled = np.where(np.isnan(brf_mean), -np.inf, brf_mean)
klass = argmax_class(brf_mean, names)
probability = np.nanmax(filled, axis=1)
RNDM = names.index("Rndm")

notes = pd.read_csv(REPO_ROOT / "catalogs" / "notas_periodos.csv")
chosen = {}
for _, row in notes.iterrows():
    if isinstance(row.per_elegido, str) or np.isfinite(row.per_elegido):
        chosen[f"{int(row.TIC)}_{int(row.sector)}"] = [
            float(value) for value in str(row.per_elegido).split(",")]

reported = np.isin(klass, PERIODIC_CLASSES) & (probability >= PROB_MIN)
print("=" * 82)
print("LOS DOS TIPOS DE ERROR EN LAS ESTRELLAS PERIODICAS")
print("=" * 82)
for key in star_truth.index:
    if not star_truth[key]:
        continue
    belongs = (peaks.key.values == key) & reported
    if not belongs.any():
        continue
    best = np.where(belongs)[0][np.argmax(probability[belongs])]
    if status[best] == "matched":
        continue
    period = peaks.period[best]
    ratios = [period / reference for reference in chosen[key]]
    closest = min(ratios, key=lambda value: abs(np.log2(value)))
    print(f"  {key:>13}  reportado {period:7.3f} d   humano "
          f"{', '.join(f'{value:.3f}' for value in chosen[key])}   "
          f"razon {closest:.3f}  -> {'ALIAS 2P' if abs(closest - 2) < 0.1 else 'otro'}")

target_key, target_period = "384805438_58", 2.935
index = int(np.argmin(np.abs(peaks.period.values - target_period)
                      + 1e6 * (peaks.key.values != target_key)))
print("\n" + "=" * 82)
print(f"LA UNICA QUE ESCAPA A TODAS LAS BANDERAS: {target_key} a {target_period} d")
print("=" * 82)
print(f"power={peaks.power[index]:.3f}  prominence={peaks.prominence[index]:.3f}  "
      f"width={peaks.width[index]:.3f}  amplitud={peaks.amplitude[index]:.4f}")
print(f"vector BRF (log): " + "  ".join(
    f"{name}={brf_mean[index, position]:.4f}" for position, name in enumerate(names)))

print("\nque dice cada normalizacion sobre este pico:")
rows = []
for name in NORMALIZATIONS:
    passes = load_passes(name)
    mean = np.nanmean(brf_grouped(passes, peaks, brf)[0], axis=0)
    cnn_passes, _ = cnn_grouped(passes)
    rows.append({
        "norma": name,
        "clase": argmax_class(mean, names)[index],
        "prob": round(float(np.nanmax(mean[index])), 3),
        "p_ELL": round(float(mean[index, names.index("ELL")]), 4),
        "max_p_Rndm_CNN": f"{np.nanmax(cnn_passes[:, index, RNDM]):.2e}",
    })
print(pd.DataFrame(rows).to_string(index=False))

print("\nlos otros picos de esa misma estrella-sector:")
same = peaks.key.values == target_key
detail = pd.DataFrame({
    "per": peaks.period[same].values.round(3),
    "power": peaks.power[same].values.round(3),
    "clase": klass[same], "prob": probability[same].round(3),
}).sort_values("power", ascending=False)
print(detail.to_string(index=False))
