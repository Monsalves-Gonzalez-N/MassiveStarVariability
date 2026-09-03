import numpy as np, pandas as pd
from msv.classify_brf import brf_mc_probs, group_probs, load_brf

AMBIGUOUS = {16187387, 41903679, 71935430, 116065031, 137113608, 153304135}
data = np.load("results/cnn_input.npz", allow_pickle=True)
tic = data["TIC"].astype(int); sector = data["sector"].astype(int)
per = data["per"].astype(float); amp = data["amplitude"].astype(float)
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

def probs(name):
    p = np.load(f"results/norm_compare/cnn_mc_{name}.npz")["p_mc"]
    pp, _ = brf_mc_probs(p, per, amp, brf)
    g, names = group_probs(pp)
    return np.nanmean(g, 0), names

base, gnames = probs("log")
ELL, LPV = gnames.index("ELL"), gnames.index("LPV")
is_ell = np.where(np.isnan(base), -np.inf, base).argmax(1) == ELL
real, cont = is_ell & has_period, is_ell & no_period

def auc(score, positive, negative):
    a, b = score[positive], score[negative]
    return float((np.subtract.outer(a, b) > 0).mean()
                 + 0.5 * (np.subtract.outer(a, b) == 0).mean())

print("EL TEST: sobrevive la senal si los empates se rompen distinto?\n")
print(f"{'variante':>14} | {'ceros reales':>12} {'ceros contam':>12} | "
      f"{'AUC ELL':>8} {'AUC todos':>10}")
for name in ["rank", "rank_random1", "rank_random2", "rank_average"]:
    g, _ = probs(name)
    lpv = g[:, LPV]
    zeros_real = f"{int((lpv[real] == 0).sum())}/{int(real.sum())}"
    zeros_cont = f"{int((lpv[cont] == 0).sum())}/{int(cont.sum())}"
    print(f"{name:>14} | {zeros_real:>12} {zeros_cont:>12} | "
          f"{auc(-lpv, real, cont):8.3f} "
          f"{auc(-lpv, has_period, no_period):10.3f}")
print("\n(AUC ELL: separa ELL real de contaminante, N=5 vs 41)")
print("(AUC todos: separa picos de estrella CON periodo de SIN, N=223 vs 482)")
