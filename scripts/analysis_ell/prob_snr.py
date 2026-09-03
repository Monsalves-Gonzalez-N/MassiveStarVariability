import numpy as np, pandas as pd
from msv.classify_brf import brf_mc_probs, group_probs, load_brf

AMBIGUOUS = {16187387, 41903679, 71935430, 116065031, 137113608, 153304135}
PERIODIC = {"ELL", "Pulsating", "E"}

data = np.load("results/cnn_input.npz", allow_pickle=True)
tic = data["TIC"].astype(int); sector = data["sector"].astype(int)
per = data["per"].astype(float); amp = data["amplitude"].astype(float)
power = data["power"].astype(float)
p_mc = np.load("results/cnn_mc20.npz")["p_mc"]
brf = load_brf()
pp, _ = brf_mc_probs(p_mc, per, amp, brf)
gb, gnames = group_probs(pp)
mean = np.nanmean(gb, 0); sd = np.nanstd(gb, 0)
winner = np.where(np.isnan(mean), -np.inf, mean).argmax(1)
rows = np.arange(len(tic))
klass = np.array([gnames[i] for i in winner])
prob = mean[rows, winner]; sigma = sd[rows, winner]
values = gb[:, rows, winner]
q1 = np.nanpercentile(values, 25, axis=0)
p16 = np.nanpercentile(values, 16, axis=0)
worst = np.nanmin(values, axis=0)

notes = pd.read_csv("catalogs/notas_periodos.csv")
truth = {}
for _, row in notes.iterrows():
    if int(row.TIC) in AMBIGUOUS:
        continue
    truth[(int(row.TIC), int(row.sector))] = bool(
        isinstance(row.per_elegido, str) or np.isfinite(row.per_elegido))
no_period = np.array([(t, s) in truth and not truth[(t, s)] for t, s in zip(tic, sector)])
has_period = np.array([(t, s) in truth and truth[(t, s)] for t, s in zip(tic, sector)])

scores = {
    "prob": prob,
    "prob/sigma": prob / (sigma + 1e-6),
    "(prob-0.5)/sigma": (prob - 0.5) / (sigma + 1e-6),
    "prob*(1-sigma)": prob * (1 - sigma),
    "q1 (cuartil bajo)": q1,
    "p16": p16,
    "min de las 20": worst,
    "power": power,
}

def auc(score, positive, negative):
    a, b = score[positive], score[negative]
    if not len(a) or not len(b):
        return np.nan
    return float((np.subtract.outer(a, b) > 0).mean()
                 + 0.5 * (np.subtract.outer(a, b) == 0).mean())

ell = klass == "ELL"
print(f"AUC dentro de los picos ELL: real (N={(ell & has_period).sum()}) "
      f"vs contaminante (N={(ell & no_period).sum()})")
for name, score in scores.items():
    print(f"  {name:20s} {auc(score, ell & has_period, ell & no_period):.3f}")

print("\nContaminantes ELL que sobreviven cada corte (y ELL reales conservadas de 3):")
print(f"{'score':>20} {'corte':>8} {'contam':>7} {'reales':>7}")
periodic_call = np.isin(klass, list(PERIODIC)) & (prob >= 0.8)
for name in ["prob/sigma", "(prob-0.5)/sigma", "q1 (cuartil bajo)", "min de las 20", "power"]:
    score = scores[name]
    finite = score[np.isfinite(score)]
    for quantile in [50, 75, 90, 95]:
        cut = np.percentile(finite, quantile)
        ok = periodic_call & (score >= cut)
        contaminants = int((ok & no_period & (klass == "ELL")).sum())
        real = int((ok & has_period & (klass == "ELL")).sum())
        print(f"{name:>20} {cut:8.3f} {contaminants:7d} {real:7d}   (percentil {quantile})")
    print()

print("los 3 picos ELL de estrellas CON periodo, y los 4 contaminantes de mayor prob:")
detail = pd.DataFrame({"TIC": tic, "sector": sector, "clase": klass, "per": per.round(3),
                       "prob": prob.round(3), "sigma": sigma.round(3),
                       "prob/sigma": (prob / (sigma + 1e-6)).round(1),
                       "q1": q1.round(3), "min20": worst.round(3),
                       "power": power.round(3),
                       "tipo": np.where(has_period, "REAL", np.where(no_period, "contaminante", "-"))})
sub = detail[ell & (has_period | no_period)].sort_values("prob", ascending=False)
print(sub[sub.tipo == "REAL"].to_string(index=False))
print(sub[sub.tipo == "contaminante"].head(6).to_string(index=False))
