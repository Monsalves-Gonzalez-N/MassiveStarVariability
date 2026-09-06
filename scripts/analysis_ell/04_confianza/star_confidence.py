"""Per star: the peak the catalogue reports, and whether its confidence is honest.

Peak-level purity is 7% by construction, because a periodic star still has ~20
spurious peaks. What the user actually inspects is one folded curve per star,
so the unit here is the peak the catalogue would report: the highest-probability
peak among the periodic classes.
"""
import numpy as np
import pandas as pd

from _common import (argmax_class, brf_grouped, cnn_grouped, load_brf,
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
ELL, RNDM = names.index("ELL"), names.index("Rndm")
brf_mean = np.nanmean(brf_log, axis=0)
filled = np.where(np.isnan(brf_mean), -np.inf, brf_mean)
klass = argmax_class(brf_mean, names)
winner = filled.argmax(axis=1)
probability = np.nanmax(filled, axis=1)
winner_passes = np.take_along_axis(brf_dropout, winner[None, :, None], axis=2)[..., 0]

binary_tail = np.nanmax(cnn_grouped(load_passes("binary"))[0][..., RNDM], axis=0)
sigma = np.nanstd(winner_passes, axis=0)
p16 = np.nanpercentile(winner_passes, 16, axis=0)

notes = pd.read_csv(REPO_ROOT / "catalogs" / "notas_periodos.csv")
note_by_key = {f"{int(row.TIC)}_{int(row.sector)}": str(row.nota)
               for _, row in notes.iterrows()}

reported = np.isin(klass, PERIODIC_CLASSES) & (probability >= PROB_MIN)
rows = []
for key in star_truth.index:
    belongs = (peaks.key.values == key) & reported
    if not belongs.any():
        rows.append({"key": key, "periodica": bool(star_truth[key]), "clase": "-",
                     "per": np.nan, "power": np.nan, "prob": np.nan,
                     "sigma": np.nan, "p16": np.nan,
                     "cola_binary": np.nan, "acierto": "no reportada",
                     "nota": note_by_key.get(key, "")})
        continue
    best = np.where(belongs)[0][np.argmax(probability[belongs])]
    rows.append({
        "key": key, "periodica": bool(star_truth[key]), "clase": klass[best],
        "per": round(float(peaks.period[best]), 3),
        "power": round(float(peaks.power[best]), 3),
        "prob": round(float(probability[best]), 3),
        "sigma": round(float(sigma[best]), 3),
        "p16": round(float(p16[best]), 3),
        "cola_binary": float(binary_tail[best]),
        "acierto": "CORRECTA" if status[best] == "matched" else "ERRADA",
        "nota": note_by_key.get(key, ""),
    })
frame = pd.DataFrame(rows)

print("=" * 104)
print("EL PICO QUE EL CATALOGO REPORTA, POR ESTRELLA (39 revisadas)")
print("=" * 104)
shown = frame[frame.clase != "-"].copy()
shown["cola_binary"] = shown.cola_binary.map(lambda value: f"{value:.1e}")
print(shown.sort_values(["acierto", "cola_binary"]).to_string(index=False))
print(f"\nsin reportar (ninguna clase periodica con prob>={PROB_MIN}): "
      f"{int((frame.clase == '-').sum())} estrellas")

print("\n" + "=" * 104)
print("CONFIANZA HONESTA? las ERRADAS deberian tener cola alta o sigma alta")
print("=" * 104)
for label in ["prob", "sigma", "p16", "cola_binary"]:
    correct = shown[shown.acierto == "CORRECTA"][label].astype(float)
    wrong = shown[shown.acierto == "ERRADA"][label].astype(float)
    print(f"{label:>12}  CORRECTA mediana {correct.median():9.3e}  "
          f"ERRADA mediana {wrong.median():9.3e}")

print("\n" + "=" * 104)
print("NIVELES DE CONFIANZA: tres ejes independientes")
print("=" * 104)
tail_value = shown.cola_binary.astype(float).values
checks = np.stack([shown.power.values >= 0.5,
                   shown.sigma.values < 0.05,
                   tail_value < 1e-4])
passed = checks.sum(axis=0)
tier = np.where(passed == 3, "ALTA", np.where(passed == 2, "MEDIA", "BAJA"))
shown = shown.assign(power_ok=checks[0], sigma_ok=checks[1], cola_ok=checks[2],
                     nivel=tier)
summary = pd.crosstab(pd.Series(tier, index=shown.index, name="nivel"), shown.acierto)
summary["pureza_%"] = (100 * summary.get("CORRECTA", 0)
                       / summary.sum(axis=1)).round(1)
print(summary.loc[[level for level in ["ALTA", "MEDIA", "BAJA"] if level in summary.index]]
      .to_string())
print("\nALTA = power>=0.5 Y sigma<0.05 Y cola_binary<1e-4 (los tres)")

print("\n" + "=" * 104)
print("EL DETALLE POR ESTRELLA, ORDENADO POR NIVEL")
print("=" * 104)
columns = ["key", "clase", "per", "power", "prob", "sigma", "cola_binary",
           "nivel", "acierto", "nota"]
print(shown.sort_values(["nivel", "acierto"])[columns].to_string(index=False))
