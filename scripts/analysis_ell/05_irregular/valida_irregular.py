"""Does the "% irregular" mean what it says? Check it against the human notes.

The notes separate three regimes the number should order: a confirmed period,
"no se ve el periodo" (noise), and "multiperiodico" (genuinely complex). If the
number is measuring irregular variability rather than peak count or noise
level, the multiperiodic stars should sit at the top.
"""
import numpy as np
import pandas as pd

from _common import (auc, cnn_grouped, load_passes, load_peak_truth, load_peaks,
                     load_truth, REPO_ROOT)

peaks = load_peaks()
star_truth, has_period, no_period = load_truth(peaks)
cnn_ensemble, names = cnn_grouped(load_passes("log"))
cnn_mean = np.nanmean(cnn_ensemble, axis=0)
LPV = names.index("LPV")

notes = pd.read_csv(REPO_ROOT / "catalogs" / "notas_periodos.csv")
records = []
for _, row in notes.iterrows():
    key = f"{int(row.TIC)}_{int(row.sector)}"
    same_star = peaks.key.values == key
    if not same_star.any():
        continue
    note = str(row.nota).lower()
    if isinstance(row.per_elegido, str) or np.isfinite(row.per_elegido):
        regime = "1 periodo confirmado"
    elif "multiperiod" in note.replace(" ", "") or "multi periodico" in note:
        regime = "2 multiperiodico"
    elif "ninguno" in note:
        regime = "3 ninguno periodico"
    else:
        regime = "4 no se ve el periodo"
    records.append({"key": key, "regimen": regime,
                    "irregular": float(cnn_mean[same_star, LPV].mean()),
                    "n_picos": int(same_star.sum()),
                    "power_max": float(peaks.power.values[same_star].max()),
                    "nota": str(row.nota)})
frame = pd.DataFrame(records)

print("=" * 74)
print("% IRREGULAR POR REGIMEN, SEGUN LA NOTA HUMANA (39 estrellas revisadas)")
print("=" * 74)
summary = frame.groupby("regimen").irregular.agg(
    N="size", mediana=lambda values: f"{100 * values.median():.1f}%",
    minimo=lambda values: f"{100 * values.min():.1f}%",
    maximo=lambda values: f"{100 * values.max():.1f}%")
print(summary.to_string())

print("\nlas multiperiodicas, una a una:")
multi = frame[frame.regimen == "2 multiperiodico"].sort_values("irregular")
for _, row in multi.iterrows():
    print(f"  {row.key:>13}  {100 * row.irregular:5.1f}%   {row.nota}")

print("\n" + "=" * 74)
print("NO ES UN PROXY DEL NUMERO DE PICOS NI DEL RUIDO")
print("=" * 74)
for label, column in [("n_picos", "n_picos"), ("power del pico mas alto", "power_max")]:
    correlation = np.corrcoef(frame.irregular.values, frame[column].values)[0, 1]
    print(f"  correlacion con {label:24s} r = {correlation:+.3f}")
confirmed = frame.regimen == "1 periodo confirmado"
print(f"\n  AUC separando confirmadas de no confirmadas: "
      f"{auc(-frame.irregular.values, confirmed.values, ~confirmed.values):.3f}  "
      f"(N={int(confirmed.sum())} vs {int((~confirmed).sum())})")
