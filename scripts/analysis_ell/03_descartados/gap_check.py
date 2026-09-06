"""Is the cut a real gap or a numerical degeneracy at exactly zero?

Both winning scores land on a threshold that rounds to 0, which is the sign
the handoff already flagged as suspicious for p_LPV(rank). The raw values are
printed here, plus the same statistic on the 11 confirmed E peaks, which the
gate never sees but which tell whether it is an ELL trick or a general
peak-quality measure.
"""
import numpy as np
import pandas as pd

from _common import (RESULTS, argmax_class, brf_grouped, cnn_grouped, load_brf,
                     load_passes, load_peak_truth, load_peaks, load_truth)

peaks = load_peaks()
star_truth, has_period, no_period = load_truth(peaks)
status, human_class = load_peak_truth(peaks)
brf = load_brf()

ensemble = load_passes("log")
brf_ensemble, names = brf_grouped(ensemble, peaks, brf)
cnn_ensemble, _ = cnn_grouped(ensemble)
rank_mean = np.nanmean(brf_grouped(load_passes("rank"), peaks, brf)[0], axis=0)
ELL, LPV, RNDM = names.index("ELL"), names.index("LPV"), names.index("Rndm")

brf_mean = np.nanmean(brf_ensemble, axis=0)
klass = argmax_class(brf_mean, names)
is_ell = klass == "ELL"
real_ell = is_ell & (status == "matched") & (human_class == "ELL")
false_ell = is_ell & (status == "spurious")
matched_e = (status == "matched") & (human_class == "E")

rndm_tail = np.nanmax(cnn_ensemble[..., RNDM], axis=0)
p_lpv_rank = rank_mean[:, LPV]

print("=" * 78)
print("VALORES CRUDOS DE max p_Rndm ENTRE LOS 7 CHECKPOINTS")
print("=" * 78)
for label, mask in [("3 ELL reales", real_ell), ("11 E confirmadas", matched_e),
                    ("43 ELL falsos", false_ell)]:
    values = np.sort(rndm_tail[mask])
    print(f"{label:18s} N={mask.sum():3d}  min={values.min():.3e}  "
          f"mediana={np.median(values):.3e}  max={values.max():.3e}")
    print(f"{'':18s} los 6 menores: {np.array2string(values[:6], precision=2)}")

print("\ndistribucion de max p_Rndm en los 43 falsos:")
bins = [0, 1e-8, 1e-6, 1e-4, 1e-2, 0.1, 0.5, 1.01]
counts = np.histogram(rndm_tail[false_ell], bins=bins)[0]
for low, high, count in zip(bins[:-1], bins[1:], counts):
    print(f"  [{low:8.1e}, {high:8.1e})  {count:2d}")
print(f"  exactamente 0.0: {int((rndm_tail[false_ell] == 0).sum())} de 43")
print(f"  ELL reales con exactamente 0.0: {int((rndm_tail[real_ell] == 0).sum())} de 3")
print(f"  E confirmadas con exactamente 0.0: {int((rndm_tail[matched_e] == 0).sum())} de 11")

print("\n" + "=" * 78)
print("LO MISMO PARA p_LPV(rank)")
print("=" * 78)
for label, mask in [("3 ELL reales", real_ell), ("11 E confirmadas", matched_e),
                    ("43 ELL falsos", false_ell)]:
    values = np.sort(p_lpv_rank[mask])
    print(f"{label:18s} min={values.min():.3e}  mediana={np.median(values):.3e}  "
          f"max={values.max():.3e}  ceros={int((values == 0).sum())}/{mask.sum()}")

print("\n" + "=" * 78)
print("UN UMBRAL NO AJUSTADO: max p_Rndm(ens7) < 1e-3 SOBRE TODOS LOS PICOS")
print("=" * 78)
keep = rndm_tail < 1e-3
for label, mask in [("ELL reales (3)", real_ell), ("E confirmadas (11)", matched_e),
                    ("ELL falsos (43)", false_ell),
                    ("spurious totales (689)", status == "spurious"),
                    ("matched totales (16)", status == "matched")]:
    print(f"  {label:24s} sobreviven {int((keep & mask).sum()):3d} de {int(mask.sum()):3d}"
          f"   ({100 * (keep & mask).sum() / mask.sum():5.1f}%)")
