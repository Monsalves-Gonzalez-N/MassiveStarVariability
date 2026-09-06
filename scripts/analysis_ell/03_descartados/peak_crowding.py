"""A fourth axis that does not come from the classifier: is the peak dominant?

384805438_58 escapes every classifier-based flag, but its periodogram has
2.935 d at power 0.549 and 1.459 d at 0.530 - a harmonic ladder where no peak
dominates. A clean periodic star should have one peak well above the rest, so
the contrast between the top peak and the runners-up is measurable without
touching the CNN.
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

brf_log, names = brf_grouped(load_passes("log"), peaks, brf)
brf_dropout, _ = brf_grouped(np.load("results/cnn_mc20.npz")["p_mc"].astype(float),
                             peaks, brf)
RNDM = names.index("Rndm")
brf_mean = np.nanmean(brf_log, axis=0)
filled = np.where(np.isnan(brf_mean), -np.inf, brf_mean)
klass = argmax_class(brf_mean, names)
winner = filled.argmax(axis=1)
probability = np.nanmax(filled, axis=1)
winner_passes = np.take_along_axis(brf_dropout, winner[None, :, None], axis=2)[..., 0]
sigma = np.nanstd(winner_passes, axis=0)
binary_tail = np.nanmax(cnn_grouped(load_passes("binary"))[0][..., RNDM], axis=0)

notes = pd.read_csv(REPO_ROOT / "catalogs" / "notas_periodos.csv")
note_by_key = {f"{int(row.TIC)}_{int(row.sector)}": str(row.nota)
               for _, row in notes.iterrows()}

reported = np.isin(klass, PERIODIC_CLASSES) & (probability >= PROB_MIN)
rows = []
for key in star_truth.index:
    belongs = (peaks.key.values == key) & reported
    if not belongs.any():
        continue
    best = np.where(belongs)[0][np.argmax(probability[belongs])]
    same_star = peaks.key.values == key
    powers = np.sort(peaks.power.values[same_star])[::-1]
    top = powers[0]
    rows.append({
        "key": key, "clase": klass[best],
        "per": round(float(peaks.period[best]), 3),
        "power": round(float(peaks.power[best]), 3),
        "contraste": round(float(top / powers[1]), 3) if len(powers) > 1 else np.inf,
        "n_sobre_70%": int((powers >= 0.7 * top).sum()),
        "sigma": round(float(sigma[best]), 3),
        "cola": float(binary_tail[best]),
        "acierto": "CORRECTA" if status[best] == "matched" else "ERRADA",
        "nota": note_by_key.get(key, ""),
    })
frame = pd.DataFrame(rows)
correct = frame.acierto == "CORRECTA"

print("=" * 92)
print("CONTRASTE DEL PICO (power_1 / power_2) Y NUMERO DE PICOS COMPARABLES")
print("=" * 92)
for label in ["contraste", "n_sobre_70%", "power"]:
    values = frame[label].astype(float).values
    print(f"{label:>12}  CORRECTA mediana {np.median(values[correct]):6.3f}   "
          f"ERRADA mediana {np.median(values[~correct]):6.3f}   "
          f"AUC {auc(values if label != 'n_sobre_70%' else -values, correct.values, ~correct.values):.3f}")

print("\n" + "=" * 92)
print("EL CUARTO EJE AGREGADO A LOS TRES ANTERIORES")
print("=" * 92)
checks = {
    "power>=0.5": frame.power.values >= 0.5,
    "sigma<0.05": frame.sigma.values < 0.05,
    "cola<1e-4": frame.cola.values < 1e-4,
    "contraste>=1.1": frame.contraste.values >= 1.1,
}
passed = np.stack(list(checks.values())).sum(axis=0)
for minimum in [4, 3]:
    tier = passed >= minimum
    n_correct = int((tier & correct.values).sum())
    n_wrong = int((tier & ~correct.values).sum())
    print(f"  pasa >= {minimum} de 4 ejes:  {n_correct} correctas, {n_wrong} erradas, "
          f"pureza {100 * n_correct / max(n_correct + n_wrong, 1):.0f}%")
frame = frame.assign(ejes=passed)
print(f"\nbaseline (todas las reportadas): {int(correct.sum())} correctas, "
      f"{int((~correct).sum())} erradas, pureza {100 * correct.mean():.0f}%")

print("\n" + "=" * 92)
print("LAS ESTRELLAS CON LOS 4 EJES EN VERDE")
print("=" * 92)
columns = ["key", "clase", "per", "power", "contraste", "n_sobre_70%", "sigma",
           "cola", "acierto", "nota"]
print(frame[frame.ejes == 4][columns].to_string(index=False))
print("\nlas ERRADAS que pasan 3 de 4, con el eje que las delata:")
suspects = frame[(frame.ejes == 3) & ~correct]
for _, row in suspects.iterrows():
    failing = [name for name, values in checks.items()
               if not values[frame.index[frame.key == row.key][0]]]
    print(f"  {row.key:>13} {row.clase:>9} {row.per:7.3f} d   falla: {failing[0]:15s} {row.nota}")
