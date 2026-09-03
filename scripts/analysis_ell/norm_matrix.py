import numpy as np, pandas as pd
from msv.classify_brf import brf_mc_probs, group_probs, load_brf

AMBIGUOUS = {16187387, 41903679, 71935430, 116065031, 137113608, 153304135}
NORMS = ["log", "min_max", "power_0.5", "power_0.33", "column", "quantized",
         "poisson", "rank", "binary"]

data = np.load("results/cnn_input.npz", allow_pickle=True)
tic = data["TIC"].astype(int); sector = data["sector"].astype(int)
per = data["per"].astype(float); amp = data["amplitude"].astype(float)
power = data["power"].astype(float); source = data["source"]
brf = load_brf()

notes = pd.read_csv("catalogs/notas_periodos.csv")
truth = {}
for _, row in notes.iterrows():
    if int(row.TIC) in AMBIGUOUS:
        continue
    truth[(int(row.TIC), int(row.sector))] = bool(
        isinstance(row.per_elegido, str) or np.isfinite(row.per_elegido))
no_period = np.array([(t, s) in truth and not truth[(t, s)] for t, s in zip(tic, sector)])
has_period = np.array([(t, s) in truth and truth[(t, s)] for t, s in zip(tic, sector)])

classes, probs = {}, {}
for name in NORMS:
    p = np.load(f"results/norm_compare/cnn_mc_{name}.npz")["p_mc"]
    per_pass, _ = brf_mc_probs(p, per, amp, brf)
    grouped, gnames = group_probs(per_pass)
    mean = np.nanmean(grouped, 0)
    w = np.where(np.isnan(mean), -np.inf, mean).argmax(1)
    classes[name] = np.array([gnames[i] for i in w])
    probs[name] = mean[np.arange(len(w)), w]

table = []
for name in NORMS:
    k = classes[name]
    ell = k == "ELL"
    table.append({"norma": name, "ELL": int(ell.sum()),
                  "ELL_falsos": int((ell & no_period).sum()),
                  "ELL_reales_3": int((ell & has_period).sum()),
                  "Puls": int((k == "Pulsating").sum()),
                  "E": int((k == "E").sum()), "LPV": int((k == "LPV").sum()),
                  "Rndm": int((k == "Rndm").sum()),
                  "acuerdo_con_log": round(float((k == classes["log"]).mean()), 3)})
print(pd.DataFrame(table).to_string(index=False))

print("\n--- En cuantas de las 9 normalizaciones cada pico es ELL? ---")
votes_ell = np.sum([classes[n] == "ELL" for n in NORMS], axis=0)
ever = votes_ell > 0
frame = pd.DataFrame({"n_normas_ELL": votes_ell[ever],
                      "tipo": np.where(has_period[ever], "REAL",
                                np.where(no_period[ever], "contaminante", "sin etiqueta")),
                      "power": power[ever]})
print(pd.crosstab(frame.n_normas_ELL, frame.tipo).to_string())
print("\npower mediano por numero de normas que dicen ELL:")
print(frame.groupby("n_normas_ELL")["power"].agg(["size", "median"]).round(3).to_string())

labelled = ever & (has_period | no_period)
print("\nAUC de 'en cuantas normas es ELL' para separar real de contaminante:")
a = votes_ell[ever & has_period]; b = votes_ell[ever & no_period]
print(f"  N real={len(a)}  N contaminante={len(b)}  "
      f"AUC={float((np.subtract.outer(a,b)>0).mean()+0.5*(np.subtract.outer(a,b)==0).mean()):.3f}")

print("\nlas 3 ELL confirmadas y los 6 contaminantes de mayor power, por norma:")
watch = [(12921082, 82, 1.7731), (362792232, 41, 4.4513),
         (8301691, 59, 1.3935), (384805438, 58, 2.9350), (13785212, 41, 3.6459),
         (169640678, 41, 2.0212), (466283988, 10, 7.0632), (239306860, 41, 4.7080)]
rows = []
for t, s, p in watch:
    index = int(np.argmin(np.abs(per - p) + 1e6 * ((tic != t) | (sector != s))))
    label = ("REAL" if has_period[index] else "contam" if no_period[index] else "-")
    row = {"TIC": t, "per": round(per[index], 3), "pw": round(power[index], 2), "tipo": label}
    row.update({n: classes[n][index] for n in NORMS})
    rows.append(row)
print(pd.DataFrame(rows).to_string(index=False))
