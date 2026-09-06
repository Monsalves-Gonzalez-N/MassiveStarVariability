"""Elegir el pico por p_clase y desempatar con p_LPV, contra usar p_LPV sola.

Probado y descartado. Sobre un solo checkpoint las dos reglas empatan (10/11),
asi que la comparacion se hace promediando sobre los 7: aunque el pipeline
final use una sola red, una diferencia de una estrella sobre 11 no se resuelve
con una sola muestra.

Dos razones para quedarse con p_LPV:
  - acierta mas: 90.9% contra 88.3% de la mejor variante con desempate, y
    80.5% de p_clase sola. El gradiente es monotono en cuanto peso se le da a
    p_LPV.
  - es reproducible: elige el mismo pico en el 79.8% de los pares
    (estrella, checkpoint) contra 55.7% de p_clase.

El mecanismo: no hay UN solo empate exacto en p_clase (0 de 32 estrellas), asi
que el desempate nunca dispara sin una tolerancia; y las diferencias que
decide p_clase son del orden de 0.002 sobre valores de 0.99, o sea ruido de
float cerca de 1. p_LPV recorre 18 ordenes de magnitud.
"""
import numpy as np
import pandas as pd

from _common import (argmax_class, cnn_grouped, load_passes, load_peak_truth,
                     load_peaks, load_truth)
from msv.config import MODELS, REPO_ROOT

CICLOS_MIN = 4.0

peaks = load_peaks()
star_truth, has_period, no_period = load_truth(peaks)
status, human_class = load_peak_truth(peaks)
periodic_stars = {key for key in star_truth.index if bool(star_truth[key])}
cnn, names = cnn_grouped(load_passes("log"))

curves = pd.read_pickle(REPO_ROOT / "results" / "phasefold_curves.pkl")
baseline = np.full(len(peaks), np.nan)
for key, (time, flux) in curves.items():
    good = np.isfinite(time) & np.isfinite(flux)
    same = (peaks.TIC.values == key[0]) & (peaks.sector.values == key[1])
    baseline[same] = time[good].max() - time[good].min()
cycles = baseline / peaks.period.values


def picks(model, rule, tolerance=None):
    block = cnn[MODELS.index(model)]
    klass = argmax_class(block, names)
    probability = block.max(axis=1)
    evidence = -np.log10(np.clip(block[:, names.index("LPV")], 1e-30, 1))
    eligible = (np.isin(klass, ["ELL", "Pulsating", "E"])
                & (~np.isfinite(cycles) | (cycles >= CICLOS_MIN)))
    chosen = {}
    for key in star_truth.index:
        inside = np.where((peaks.key.values == key) & eligible)[0]
        if not len(inside):
            continue
        if rule == "prob":
            chosen[key] = int(inside[np.argmax(probability[inside])])
        elif rule == "lpv":
            chosen[key] = int(inside[np.argmax(evidence[inside])])
        else:
            top = probability[inside].max()
            tied = inside[probability[inside] >= top - tolerance]
            chosen[key] = int(tied[np.argmax(evidence[tied])])
    return chosen


print("=" * 84)
print("EMPATES EXACTOS EN p_clase")
print("=" * 84)
block = cnn[MODELS.index("Number_ELL")]
probability = block.max(axis=1)
klass = argmax_class(block, names)
eligible = (np.isin(klass, ["ELL", "Pulsating", "E"])
            & (~np.isfinite(cycles) | (cycles >= CICLOS_MIN)))
tied_stars = 0
total_stars = 0
for key in star_truth.index:
    inside = np.where((peaks.key.values == key) & eligible)[0]
    if not len(inside):
        continue
    total_stars += 1
    tied_stars += int((probability[inside] == probability[inside].max()).sum() > 1)
print(f"  estrellas con dos picos al mismo p_clase: {tied_stars}/{total_stars}")
print("  el desempate no dispara nunca sin una tolerancia explicita")

print("\n" + "=" * 84)
print("PICO CORRECTO, PROMEDIADO SOBRE LOS 7 CHECKPOINTS")
print("=" * 84)
for rule, tolerance, label in [("prob", None, "max p_clase"),
                               ("mix", 0.01, "p_clase, empate 0.01 -> p_LPV"),
                               ("mix", 0.05, "p_clase, empate 0.05 -> p_LPV"),
                               ("mix", 0.20, "p_clase, empate 0.20 -> p_LPV"),
                               ("lpv", None, "max -log10 p_LPV (adoptada)")]:
    total = correct = 0
    per_model = []
    for model in MODELS:
        chosen = picks(model, rule, tolerance)
        hits = sum(1 for key, index in chosen.items()
                   if key in periodic_stars and status[index] == "matched")
        n_stars = sum(1 for key in chosen if key in periodic_stars)
        per_model.append(f"{hits}/{n_stars}")
        total += n_stars
        correct += hits
    print(f"  {label:32s} {correct:3d}/{total} ({100 * correct / total:4.1f}%)   "
          f"{' '.join(per_model)}")

print("\n" + "=" * 84)
print("REPRODUCIBILIDAD: el mismo pico al cambiar de checkpoint?")
print("=" * 84)
for rule, label in [("prob", "max p_clase"), ("lpv", "max -log10 p_LPV")]:
    reference = picks("Number_ELL", rule)
    total = agree = 0
    for model in MODELS:
        if model == "Number_ELL":
            continue
        other = picks(model, rule)
        shared = set(reference) & set(other)
        total += len(shared)
        agree += sum(1 for key in shared if reference[key] == other[key])
    print(f"  {label:20s} {100 * agree / total:5.1f}%  ({agree}/{total} pares)")
