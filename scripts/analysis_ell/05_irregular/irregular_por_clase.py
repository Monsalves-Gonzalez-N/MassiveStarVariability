"""The % irregular reads differently per class, so measure it per class.

A multimode pulsator with 20% irregular mass is ordinary; an ellipsoidal with
20% is not, because an ELL is a single smooth continuous modulation. The
class-conditional distributions are measured here, plus a relative version
that divides the irregular mass by the class's own mass, which is
class-dependent by construction and still lands in [0,1].
"""
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

ensemble = load_passes("log")
cnn_ensemble, names = cnn_grouped(ensemble)
cnn_mean = np.nanmean(cnn_ensemble, axis=0)
LPV, RNDM = names.index("LPV"), names.index("Rndm")
brf_mean = np.nanmean(brf_grouped(ensemble, peaks, brf)[0], axis=0)
filled = np.where(np.isnan(brf_mean), -np.inf, brf_mean)
klass = argmax_class(brf_mean, names)
probability = np.nanmax(filled, axis=1)

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
    class_name = klass[best]
    class_mass = spectrum[names.index(class_name)]
    irregular = spectrum[LPV]
    records.append({
        "key": key, "clase": class_name,
        "per": round(float(peaks.period.values[best]), 3),
        "irregular": irregular,
        "masa_clase": class_mass,
        "relativo": irregular / (irregular + class_mass) if irregular + class_mass > 0 else np.nan,
        "correcta": status[best] == "matched",
        "nota": note_by_key.get(key, ""),
    })
    for position, name in enumerate(names):
        records[-1][f"esp_{name}"] = spectrum[position]
frame = pd.DataFrame(records)

print("=" * 84)
print("% IRREGULAR CONDICIONADO A LA CLASE REPORTADA")
print("=" * 84)
for class_name in ["E", "ELL", "Pulsating"]:
    block = frame[frame.clase == class_name]
    correct = block.correcta.values
    print(f"\n{class_name}  (N={len(block)}: {int(correct.sum())} correctas, "
          f"{int((~correct).sum())} erradas)")
    for label, column in [("irregular absoluto", "irregular"),
                          ("irregular relativo a la clase", "relativo")]:
        values = block[column].values
        line = (f"  {label:32s} mediana {100 * np.median(values):5.1f}%")
        if correct.any() and (~correct).any():
            line += (f"   CORRECTA {100 * np.median(values[correct]):5.1f}%"
                     f"   ERRADA {100 * np.median(values[~correct]):5.1f}%"
                     f"   AUC {auc(-values, correct, ~correct):.3f}")
        print(line)

print("\n" + "=" * 84)
print("EL MISMO CORTE SIRVE PARA LAS DOS CLASES?")
print("=" * 84)
for column, label in [("irregular", "absoluto"), ("relativo", "relativo")]:
    print(f"\n{label}:")
    for class_name in ["E", "ELL"]:
        block = frame[frame.clase == class_name]
        correct = block.correcta.values
        values = block[column].values
        if not correct.any() or not (~correct).any():
            continue
        limit = np.max(values[correct])
        first_error = np.min(values[~correct])
        print(f"  {class_name:>4}  ultima CORRECTA {100 * limit:5.1f}%   "
              f"primer ERROR {100 * first_error:5.1f}%   "
              f"corte limpio: {'si' if first_error > limit else 'no'}")

print("\n" + "=" * 100)
print("EL ESPECTRO COMPLETO POR ESTRELLA (suma 1)")
print("=" * 100)
shown = frame.sort_values(["clase", "irregular"]).copy()
for name in names:
    shown[name] = (100 * shown[f"esp_{name}"]).round(0).astype(int)
shown["acierto"] = np.where(shown.correcta, "OK ", "ERR")
print(shown[["key", "clase", "per"] + names + ["acierto", "nota"]].to_string(index=False))
