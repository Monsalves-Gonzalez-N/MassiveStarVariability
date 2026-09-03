import sys
import numpy as np, pandas as pd
sys.path.insert(0, "scripts")
from msv.classify_brf import brf_mc_probs, group_probs, load_brf

AMBIGUOUS = {16187387, 41903679, 71935430, 116065031, 137113608, 153304135}
PERIODIC = {"ELL", "Pulsating", "E"}
data = np.load("results/cnn_input.npz", allow_pickle=True)
tic = data["TIC"].astype(int); sector = data["sector"].astype(int)
per = data["per"].astype(float); amp = data["amplitude"].astype(float)
power = data["power"].astype(float)
brf = load_brf()

def grouped(name):
    p = np.load(f"results/norm_compare/cnn_mc_{name}.npz")["p_mc"]
    pp, _ = brf_mc_probs(p, per, amp, brf)
    g, names = group_probs(pp)
    return np.nanmean(g, 0), names

log_probs, gnames = grouped("log")
rank_probs, _ = grouped("rank")
LPV, ELL = gnames.index("LPV"), gnames.index("ELL")
klass = np.array([gnames[i] for i in np.where(np.isnan(log_probs), -np.inf, log_probs).argmax(1)])
prob = np.nanmax(np.where(np.isnan(log_probs), -np.inf, log_probs), axis=1)
p_lpv_rank = rank_probs[:, LPV]

notes = pd.read_csv("catalogs/notas_periodos.csv")
truth = {}
for _, row in notes.iterrows():
    if int(row.TIC) in AMBIGUOUS:
        continue
    truth[(int(row.TIC), int(row.sector))] = bool(
        isinstance(row.per_elegido, str) or np.isfinite(row.per_elegido))
keys = np.array([f"{t}_{s}" for t, s in zip(tic, sector)])
star_truth = pd.Series({f"{t}_{s}": v for (t, s), v in truth.items()})
no_period = np.array([(t, s) in truth and not truth[(t, s)] for t, s in zip(tic, sector)])
has_period = np.array([(t, s) in truth and truth[(t, s)] for t, s in zip(tic, sector)])

def evaluate(mask, label):
    call = pd.Series(mask & np.isin(klass, list(PERIODIC)) & (prob >= 0.8),
                     index=keys).groupby(level=0).any()
    aligned = star_truth.index.intersection(call.index)
    predicted, actual = call[aligned], star_truth[aligned]
    tp = int((predicted & actual).sum()); fn = int((~predicted & actual).sum())
    fp = int((predicted & ~actual).sum()); tn = int((~predicted & ~actual).sum())
    ell_false = int((mask & (klass == "ELL") & no_period & (prob >= 0.8)).sum())
    ell_real = int((mask & (klass == "ELL") & has_period & (prob >= 0.8)).sum())
    return {"regla": label, "TP": tp, "FN": fn, "FP": fp, "TN": tn,
            "acc": round((tp + tn) / (tp + fn + fp + tn), 3),
            "ELL_falsos": ell_false, "ELL_reales": ell_real}

everything = np.ones(len(per), dtype=bool)
rows = [
    evaluate(everything, "sin filtro (baseline)"),
    evaluate(power >= 0.5, "power >= 0.5"),
    evaluate(p_lpv_rank <= 0.001, "p_LPV(rank) <= 0.001"),
    evaluate(p_lpv_rank == 0.0, "p_LPV(rank) == 0"),
    evaluate((power >= 0.5) & (p_lpv_rank <= 0.001), "power>=0.5 Y p_LPV(rank)<=0.001"),
    evaluate((power >= 0.3) & (p_lpv_rank <= 0.001), "power>=0.3 Y p_LPV(rank)<=0.001"),
    evaluate((power >= 0.5) | (p_lpv_rank == 0.0), "power>=0.5 O p_LPV(rank)==0"),
]
print(pd.DataFrame(rows).to_string(index=False))

def auc(score, positive, negative):
    a, b = score[positive], score[negative]
    return float((np.subtract.outer(a, b) > 0).mean()
                 + 0.5 * (np.subtract.outer(a, b) == 0).mean())
stars = pd.DataFrame({"key": keys, "power": power, "lpv": p_lpv_rank})
agg = stars.groupby("key").agg(max_power=("power", "max"), min_lpv=("lpv", "min"))
agg = agg.join(star_truth.rename("periodica"), how="inner")
pos = agg.periodica.values.astype(bool)
rank_power = pd.Series(agg.max_power).rank(pct=True).values
rank_lpv = pd.Series(-agg.min_lpv).rank(pct=True).values
print("\nAUC a nivel estrella (N=11 vs 28):")
print(f"  max_power                 {auc(agg.max_power.values, pos, ~pos):.3f}")
print(f"  min p_LPV(rank) invertido {auc(-agg.min_lpv.values, pos, ~pos):.3f}")
print(f"  suma de los dos rangos    {auc(rank_power + rank_lpv, pos, ~pos):.3f}")
