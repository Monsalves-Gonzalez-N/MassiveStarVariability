"""Consistencia de la clase entre normalizaciones log y min_max.

Ensemble DETERMINISTICO de los 7 checkpoints en las dos, asi que cualquier
diferencia es de la normalizacion y no del dropout.
"""
import numpy as np, pandas as pd
from msv.classify_brf import brf_mc_probs, group_probs, load_brf

AMBIGUOUS = {16187387, 41903679, 71935430, 116065031, 137113608, 153304135}

data = np.load("results/cnn_input.npz", allow_pickle=True)
tic = data["TIC"].astype(int); sector = data["sector"].astype(int)
per = data["per"].astype(float); amp = data["amplitude"].astype(float)
power = data["power"].astype(float); source = data["source"]
brf = load_brf()

def classify(npz_path):
    probabilities = np.load(npz_path)["p_mc"]
    per_pass, _ = brf_mc_probs(probabilities, per, amp, brf)
    grouped, names = group_probs(per_pass)
    mean = np.nanmean(grouped, 0)
    winner = np.where(np.isnan(mean), -np.inf, mean).argmax(1)
    return (np.array([names[i] for i in winner]),
            mean[np.arange(len(winner)), winner], names)

klass_log, prob_log, gnames = classify("results/cnn_mc.npz")
klass_mm, prob_mm, _ = classify("results/cnn_mc_minmax.npz")

notes = pd.read_csv("catalogs/notas_periodos.csv")
truth = {}
for _, row in notes.iterrows():
    if int(row.TIC) in AMBIGUOUS:
        continue
    truth[(int(row.TIC), int(row.sector))] = bool(
        isinstance(row.per_elegido, str) or np.isfinite(row.per_elegido))
no_period = np.array([(t, s) in truth and not truth[(t, s)] for t, s in zip(tic, sector)])
has_period = np.array([(t, s) in truth and truth[(t, s)] for t, s in zip(tic, sector)])

print(f"acuerdo global de clase entre log y min_max: "
      f"{(klass_log == klass_mm).mean():.3f}  ({(klass_log == klass_mm).sum()}/{len(klass_log)})")
print("\ncontingencia log (filas) vs min_max (columnas):")
print(pd.crosstab(pd.Series(klass_log, name="log"),
                  pd.Series(klass_mm, name="min_max")).to_string())

print("\nESTABILIDAD POR CLASE: de los picos que 'log' llama X, que fraccion "
      "sigue siendo X en min_max")
for name in gnames:
    which = klass_log == name
    if which.sum():
        print(f"  {name:10s} N={which.sum():4d}   estable {(klass_mm[which] == name).mean():.3f}")

print("\n--- LA PREGUNTA: los ELL consistentes, son los buenos? ---")
ell_log = klass_log == "ELL"
consistent = ell_log & (klass_mm == "ELL")
flipped = ell_log & (klass_mm != "ELL")
for label, mask in [("ELL en AMBAS normalizaciones", consistent),
                    ("ELL solo en log (se cae en min_max)", flipped)]:
    real = int((mask & has_period).sum()); false = int((mask & no_period).sum())
    print(f"{label:38s} N={int(mask.sum()):3d}  "
          f"en estrella CON periodo: {real}  SIN periodo: {false}  "
          f"power mediano {np.median(power[mask]):.3f}")
print("\na donde se van los ELL que se caen:",
      pd.Series(klass_mm[flipped]).value_counts().to_dict())

print("\ncontrol positivo, las 2 ELL confirmadas:")
for t, s in [(12921082, 82), (362792232, 41)]:
    which = (tic == t) & (sector == s) & (klass_log == "ELL")
    for index in np.where(which)[0]:
        print(f"  TIC {t} s{s}  P={per[index]:.4f} ({source[index]}) power={power[index]:.3f}"
              f"   log={klass_log[index]} {prob_log[index]:.3f}"
              f"   min_max={klass_mm[index]} {prob_mm[index]:.3f}")

print("\nlos contaminantes ELL de mayor power, en las dos normalizaciones:")
sub = pd.DataFrame({"TIC": tic, "sector": sector, "per": per.round(3),
                    "src": source, "power": power.round(3),
                    "log": klass_log, "p_log": prob_log.round(3),
                    "min_max": klass_mm, "p_mm": prob_mm.round(3)})[ell_log & no_period]
print(sub.sort_values("power", ascending=False).head(12).to_string(index=False))
