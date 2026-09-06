"""Is the pipeline's confidence honest? Purity as a function of confidence.

The question is not how many contaminants a cut removes but whether a peak the
model calls ELL at 0.99 is more likely to be right than one it calls at 0.85.
If the purity curve is flat, the probability carries no information and the
number printed in the catalogue is misleading.
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

brf_log, names = brf_grouped(load_passes("log"), peaks, brf)
brf_dropout, _ = brf_grouped(np.load("results/cnn_mc20.npz")["p_mc"].astype(float),
                             peaks, brf)
ELL, LPV, RNDM = names.index("ELL"), names.index("LPV"), names.index("Rndm")
brf_mean = np.nanmean(brf_log, axis=0)
filled = np.where(np.isnan(brf_mean), -np.inf, brf_mean)
klass = argmax_class(brf_mean, names)
winner = filled.argmax(axis=1)
probability = np.nanmax(filled, axis=1)

winner_passes = np.take_along_axis(brf_dropout, winner[None, :, None], axis=2)[..., 0]
binary_tail = np.nanmax(cnn_grouped(load_passes("binary"))[0][..., RNDM], axis=0)
log_tail = np.nanmax(cnn_grouped(load_passes("log"))[0][..., RNDM], axis=0)

called = np.isin(klass, PERIODIC_CLASSES)
labelled = called & (status != "unknown")
matched = labelled & (status == "matched")
spurious = labelled & (status == "spurious")
print(f"picos llamados periodicos y etiquetados: {int(labelled.sum())}  "
      f"({int(matched.sum())} confirmados, {int(spurious.sum())} espurios)")
print(f"pureza global: {100 * matched.sum() / labelled.sum():.1f}%\n")

scores = {
    "prob media (la del catalogo)": probability,
    "p16 entre pasadas": np.nanpercentile(winner_passes, 16, axis=0),
    "1 - max p_Rndm(binary)": 1 - binary_tail,
    "1 - max p_Rndm(log)": 1 - log_tail,
    "prob media x (1 - cola binary)": probability * (1 - binary_tail),
}

print("=" * 84)
print("PUREZA POR CUARTIL DE CONFIANZA (solo picos llamados ELL/E/Pulsating)")
print("=" * 84)
for label, score in scores.items():
    values = score[labelled]
    edges = np.quantile(values, [0, 0.25, 0.50, 0.75, 1.0])
    print(f"\n{label}")
    print(f"  {'cuartil':>10} {'rango':>26} {'conf':>5} {'esp':>5} {'pureza':>8}")
    for position in range(4):
        low, high = edges[position], edges[position + 1]
        if position == 3:
            inside = (values >= low)
        else:
            inside = (values >= low) & (values < high)
        subset = np.where(labelled)[0][inside]
        n_matched = int(matched[subset].sum())
        n_spurious = int(spurious[subset].sum())
        total = n_matched + n_spurious
        purity = 100 * n_matched / total if total else np.nan
        print(f"  {['Q1 (bajo)', 'Q2', 'Q3', 'Q4 (alto)'][position]:>10} "
              f"[{low:10.3e}, {high:10.3e}] {n_matched:5d} {n_spurious:5d} {purity:7.1f}%")
