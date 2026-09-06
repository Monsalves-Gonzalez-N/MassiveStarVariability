"""The % irregular describes the star's variability, so judge it against the
CLASS, not against the period.

Three of the E errors are exact 2P aliases where the class is right and only
the period is doubled, and the ELL contamination of an eclipsing binary at P/2
is the network correctly refusing to call a half-period fold an eclipse. Both
are period-selection failures, not class failures, so a quantity describing the
variability should be scored against class correctness.
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
LPV = names.index("LPV")
brf_mean = np.nanmean(brf_grouped(ensemble, peaks, brf)[0], axis=0)
filled = np.where(np.isnan(brf_mean), -np.inf, brf_mean)
klass = argmax_class(brf_mean, names)
probability = np.nanmax(filled, axis=1)

notes = pd.read_csv(REPO_ROOT / "catalogs" / "notas_periodos.csv")
star_class, star_periods = {}, {}
for _, row in notes.iterrows():
    key = f"{int(row.TIC)}_{int(row.sector)}"
    if isinstance(row.per_elegido, str) or np.isfinite(row.per_elegido):
        star_class[key] = [name.strip() for name in str(row.clase_elegida).split(",")]
        star_periods[key] = [float(value) for value in str(row.per_elegido).split(",")]
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
    reported_class = klass[best]
    period = float(peaks.period.values[best])
    class_ok = reported_class in star_class.get(key, [])
    ratio = np.nan
    if key in star_periods:
        ratios = [period / reference for reference in star_periods[key]]
        ratio = min(ratios, key=lambda value: abs(np.log2(value)))
    records.append({
        "key": key, "clase": reported_class, "per": round(period, 3),
        "humano": "+".join(sorted(set(star_class.get(key, ["sin periodo"])))),
        "razon_per": round(ratio, 2) if np.isfinite(ratio) else np.nan,
        "irregular": float(cnn_mean[same_star, LPV].mean()),
        "periodo_ok": status[best] == "matched",
        "clase_ok": class_ok,
        "nota": note_by_key.get(key, ""),
    })
frame = pd.DataFrame(records)

print("=" * 84)
print("LAS DOS PREGUNTAS, SEPARADAS")
print("=" * 84)
print(pd.crosstab(frame.clase_ok.rename("clase correcta"),
                  frame.periodo_ok.rename("periodo correcto")).to_string())
print("\nlos casos clase-si / periodo-no:")
mixed = frame[frame.clase_ok & ~frame.periodo_ok]
print(mixed[["key", "clase", "per", "humano", "razon_per", "irregular"]]
      .assign(irregular=lambda block: (100 * block.irregular).round(0).astype(int))
      .to_string(index=False))

print("\n" + "=" * 84)
print("% IRREGULAR CONTRA CADA PREGUNTA")
print("=" * 84)
for label, target in [("periodo correcto", frame.periodo_ok.values),
                      ("CLASE correcta", frame.clase_ok.values)]:
    score = frame.irregular.values
    print(f"\n{label}  (N={int(target.sum())} si, {int((~target).sum())} no)")
    print(f"  global          AUC {auc(-score, target, ~target):.3f}   "
          f"mediana si {100 * np.median(score[target]):.1f}%   "
          f"no {100 * np.median(score[~target]):.1f}%")
    for class_name in ["E", "ELL"]:
        inside = frame.clase.values == class_name
        if not (target & inside).any() or not (~target & inside).any():
            continue
        print(f"  solo {class_name:9s} AUC {auc(-score[inside], target[inside], ~target[inside]):.3f}   "
              f"mediana si {100 * np.median(score[inside & target]):.1f}%   "
              f"no {100 * np.median(score[inside & ~target]):.1f}%   "
              f"(N={int((inside & target).sum())} vs {int((inside & ~target).sum())})")

print("\n" + "=" * 84)
print("CORTE LIMPIO POR CLASE, CONTRA LA CLASE CORRECTA")
print("=" * 84)
for class_name in ["E", "ELL"]:
    inside = frame.clase.values == class_name
    target = frame.clase_ok.values
    if not (target & inside).any() or not (~target & inside).any():
        continue
    good = frame.irregular.values[inside & target]
    bad = frame.irregular.values[inside & ~target]
    print(f"  {class_name:>4}  ultima con clase OK {100 * good.max():5.1f}%   "
          f"primer error {100 * bad.min():5.1f}%   "
          f"limpio: {'SI' if bad.min() > good.max() else 'no'}")
