"""Keep the argmax, replace the saturated number next to it.

The argmax gets 14 of the 16 confirmed peaks right, so the class chooser is
not what fails. What fails is `prob >= 0.8`: it is a cut on a quantity that no
longer discriminates (0.995 vs 0.991 at star level), and it costs a confirmed
detection. Two replacements are tested at star level, and the star spectrum is
checked as an alternative class chooser.
"""
import numpy as np
import pandas as pd

from _common import (argmax_class, brf_grouped, cnn_grouped, load_brf,
                     load_passes, load_peak_truth, load_peaks, load_truth)

PERIODIC_CLASSES = ["ELL", "Pulsating", "E"]

peaks = load_peaks()
star_truth, has_period, no_period = load_truth(peaks)
status, human_class = load_peak_truth(peaks)
brf = load_brf()

ensemble = load_passes("log")
cnn_ensemble, names = cnn_grouped(ensemble)
cnn_mean = np.nanmean(cnn_ensemble, axis=0)
brf_mean = np.nanmean(brf_grouped(ensemble, peaks, brf)[0], axis=0)
filled = np.where(np.isnan(brf_mean), -np.inf, brf_mean)
klass = argmax_class(brf_mean, names)
probability = np.nanmax(filled, axis=1)
LPV, RNDM = names.index("LPV"), names.index("Rndm")
lpv_peak = np.nanmax(cnn_ensemble[..., LPV], axis=0)

irregular = np.empty(len(peaks))
for key in np.unique(peaks.key.values):
    same_star = peaks.key.values == key
    irregular[same_star] = cnn_mean[same_star, LPV].mean()

print("=" * 80)
print("EL ESPECTRO DE ESTRELLA SIRVE PARA ELEGIR LA CLASE?")
print("=" * 80)
rows = []
for key in star_truth.index:
    if not star_truth[key]:
        continue
    same_star = peaks.key.values == key
    confirmed = same_star & (status == "matched")
    if not confirmed.any():
        continue
    spectrum = cnn_mean[same_star].mean(axis=0)
    without_noise = spectrum.copy()
    without_noise[RNDM] = -np.inf
    best_peak = int(np.where(confirmed)[0][0])
    rows.append({"key": key, "humano": human_class[best_peak],
                 "argmax del pico": klass[best_peak],
                 "argmax del espectro": names[int(np.argmax(without_noise))]})
frame = pd.DataFrame(rows)
print(frame.to_string(index=False))
print(f"\n  acierto del argmax del pico     "
      f"{int((frame.humano == frame['argmax del pico']).sum())}/{len(frame)}")
print(f"  acierto del argmax del espectro "
      f"{int((frame.humano == frame['argmax del espectro']).sum())}/{len(frame)}")


def star_level(keep, label):
    called = pd.Series(keep & np.isin(klass, PERIODIC_CLASSES),
                       index=peaks.key.values).groupby(level=0).any()
    aligned = star_truth.index.intersection(called.index)
    predicted, actual = called[aligned], star_truth[aligned]
    tp = int((predicted & actual).sum()); fn = int((~predicted & actual).sum())
    fp = int((predicted & ~actual).sum()); tn = int((~predicted & ~actual).sum())
    return {"gate": label, "picos_periodicos": int((keep & np.isin(klass, PERIODIC_CLASSES)).sum()),
            "confirmados_16": int((keep & (status == "matched")).sum()),
            "TP": tp, "FN": fn, "FP": fp, "TN": tn,
            "acc": round((tp + tn) / (tp + fn + fp + tn), 3)}


everything = np.ones(len(peaks), dtype=bool)
is_ell = klass == "ELL"
gates = [
    (everything, "sin gate"),
    (probability >= 0.8, "prob >= 0.8  (el actual, saturado)"),
    (probability >= 0.99, "prob >= 0.99"),
    (lpv_peak < 1.8e-8, "p_LPV(pico) < 1.8e-8"),
    (~is_ell | (irregular < 0.05), "ELL: irregular < 5%"),
    ((probability >= 0.8) & (~is_ell | (irregular < 0.05)),
     "prob>=0.8 Y ELL: irregular<5%"),
    ((lpv_peak < 1.8e-8) & (~is_ell | (irregular < 0.05)),
     "p_LPV(pico)<1.8e-8 Y ELL: irregular<5%"),
]
print("\n" + "=" * 100)
print("REEMPLAZAR EL GATE SATURADO")
print("=" * 100)
print(pd.DataFrame([star_level(keep, label) for keep, label in gates]).to_string(index=False))
print("\nTP/FN sobre 11 estrellas periodicas, FP/TN sobre 28 sin periodo.")
print("confirmados_16 = picos con periodo humano que el gate deja pasar.")
