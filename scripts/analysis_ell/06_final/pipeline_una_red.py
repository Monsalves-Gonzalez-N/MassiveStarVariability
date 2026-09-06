"""The whole thing with one network: Number_ELL, normalization log, no BRF.

Selecting the reported peak by the class probability is nearly arbitrary once
that probability saturates (median 0.987), so the two selection rules are
compared: highest class probability, and lowest p_LPV.
"""
import numpy as np
import pandas as pd

from _common import (auc, argmax_class, cnn_grouped, load_passes,
                     load_peak_truth, load_peaks, load_truth, REPO_ROOT)
from msv.config import MODELS

PERIODIC_CLASSES = ["ELL", "Pulsating", "E"]
FLOOR = 1e-30

peaks = load_peaks()
star_truth, has_period, no_period = load_truth(peaks)
status, human_class = load_peak_truth(peaks)
cnn_ensemble, names = cnn_grouped(load_passes("log"))
block = cnn_ensemble[MODELS.index("Number_ELL")]
LPV = names.index("LPV")

klass = argmax_class(block, names)
probability = block.max(axis=1)
lpv = np.clip(block[:, LPV], FLOOR, 1)
score = -np.log10(lpv)
irregular = np.empty(len(peaks))
for key in np.unique(peaks.key.values):
    same_star = peaks.key.values == key
    irregular[same_star] = block[same_star, LPV].mean()

notes = pd.read_csv(REPO_ROOT / "catalogs" / "notas_periodos.csv")
note_by_key = {f"{int(row.TIC)}_{int(row.sector)}": str(row.nota)
               for _, row in notes.iterrows()}
periodic = np.isin(klass, PERIODIC_CLASSES)


def select(rule):
    chosen, labels = [], []
    for key in star_truth.index:
        belongs = (peaks.key.values == key) & periodic
        if not belongs.any():
            continue
        inside = np.where(belongs)[0]
        pick = inside[np.argmax(probability[inside])] if rule == "prob" \
            else inside[np.argmax(score[inside])]
        chosen.append(int(pick))
        labels.append(key)
    return np.array(chosen), labels


print("=" * 76)
print("COMO ELEGIR EL PICO REPORTADO, CON LA PROBABILIDAD SATURADA")
print("=" * 76)
for rule, label in [("prob", "mayor probabilidad de clase"),
                    ("lpv", "menor p_LPV")]:
    chosen, keys = select(rule)
    correct = status[chosen] == "matched"
    print(f"  {label:30s} {int(correct.sum())}/11 estrellas periodicas con "
          f"el pico correcto  (de {len(chosen)} reportadas)")

chosen, keys = select("lpv")
correct = status[chosen] == "matched"
print("\n" + "=" * 76)
print("FRONTERA CON LA RED UNICA, seleccionando por p_LPV")
print("=" * 76)
value = score[chosen]
frontier = {}
for cut in np.unique(value):
    keep = value > cut * 0.999999
    n_correct = int((keep & correct).sum())
    n_wrong = int((keep & ~correct).sum())
    if n_correct and (n_correct not in frontier or n_wrong < frontier[n_correct]):
        frontier[n_correct] = n_wrong
print("  correctas conservadas : erradas coladas")
print("  " + "   ".join(f"{n}:{frontier[n]}" for n in sorted(frontier, reverse=True)))
clean = [n for n in frontier if frontier[n] == 0]
if clean:
    kept = max(clean)
    ordered = np.sort(value[correct])[::-1]
    limit = ordered[kept - 1]
    first_error = np.max(value[~correct])
    print(f"\n  corte limpio: -log10 p_LPV > {(limit + first_error) / 2:.1f}"
          f"   conserva {kept}/{int(correct.sum())} correctas, 0 erradas")
    print(f"  ultima correcta {limit:.1f}   primer error {first_error:.1f}   "
          f"margen {limit - first_error:.1f} decadas")
print(f"\n  AUC de -log10 p_LPV: {auc(value, correct, ~correct):.3f}   "
      f"N={int(correct.sum())} correctas vs {int((~correct).sum())} erradas")

print("\n" + "=" * 104)
print("LA TABLA FINAL")
print("=" * 104)
frame = pd.DataFrame({
    "key": keys, "clase": klass[chosen],
    "periodo": peaks.period.values[chosen].round(4),
    "p_clase": probability[chosen].round(3),
    "irregular_%": (100 * irregular[chosen]).round(0).astype(int),
    "-log10_pLPV": value.round(1),
    "acierto": np.where(correct, "OK ", "ERR"),
    "nota": [note_by_key.get(key, "") for key in keys],
}).sort_values("-log10_pLPV", ascending=False)
print(frame.to_string(index=False))
print(f"\np_clase: mediana {np.median(probability[chosen]):.3f}, "
      f"minimo {probability[chosen].min():.3f}  <- saturada, se reporta pero no se filtra con ella")
