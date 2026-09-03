import numpy as np, pandas as pd
from msv.classify_brf import brf_mc_probs, group_probs, load_brf

AMBIGUOUS = {16187387, 41903679, 71935430, 116065031, 137113608, 153304135}
PERIODIC_GROUPS = {"ELL", "Pulsating", "E"}
REAL_ELL = {(12921082, 82), (362792232, 41)}

data = np.load("results/cnn_input.npz", allow_pickle=True)
tic = data["TIC"].astype(int); sector = data["sector"].astype(int)
per = data["per"].astype(float); amp = data["amplitude"].astype(float)
power = data["power"].astype(float)
p_models = np.load("results/cnn_mc.npz")["p_mc"]           # (7, 857, 8)
brf = load_brf()

notes = pd.read_csv("catalogs/notas_periodos.csv")
truth = {}
for _, row in notes.iterrows():
    if int(row.TIC) in AMBIGUOUS:
        continue
    has_period = isinstance(row.per_elegido, str) or np.isfinite(row.per_elegido)
    truth[(int(row.TIC), int(row.sector))] = bool(has_period)
keys = np.array([f"{t}_{s}" for t, s in zip(tic, sector)])
star_truth = pd.Series({f"{t}_{s}": v for (t, s), v in truth.items()})
no_period = np.array([(t, s) in truth and not truth[(t, s)]
                      for t, s in zip(tic, sector)])

# --- ensemble: BRF sobre cada checkpoint, luego agrupar --------------------
per_pass, _ = brf_mc_probs(p_models, per, amp, brf)         # (7, 857, 8)
grouped, gnames = group_probs(per_pass)                     # (7, 857, 5)
mean = np.nanmean(grouped, 0)
scores = np.where(np.isnan(mean), -np.inf, mean)
winner = scores.argmax(1)
rows_index = np.arange(len(tic))
best_prob = mean[rows_index, winner]
per_model_class = np.where(np.isnan(grouped), -np.inf, grouped).argmax(2)   # (7,857)
agreement = (per_model_class == winner[None, :]).sum(0)     # cuantos de 7 coinciden
klass = np.array([gnames[i] for i in winner])

print("acuerdo entre los 7 checkpoints, por clase ganadora del ensemble:")
frame = pd.DataFrame({"clase": klass, "acuerdo": agreement, "prob": best_prob,
                      "power": power, "no_period": no_period})
print(frame.groupby("clase")["acuerdo"].describe()[["count", "mean", "50%", "min"]].round(2).to_string())

print("\npicos ELL: acuerdo de los falsos vs los de estrellas confirmadas")
ell = frame[frame.clase == "ELL"]
print(f"  falsos (estrella sin periodo)  N={int(ell.no_period.sum()):3d}  "
      f"acuerdo mediano {ell[ell.no_period].acuerdo.median():.1f}")
confirmed = ell[~ell.no_period]
print(f"  resto                          N={len(confirmed):3d}  "
      f"acuerdo mediano {confirmed.acuerdo.median():.1f}")

def evaluate(min_agreement, min_power=None, label=""):
    ok = (agreement >= min_agreement) & (best_prob >= 0.8)
    if min_power is not None:
        ok = ok & (power >= min_power)
    periodic_peak = ok & np.isin(klass, list(PERIODIC_GROUPS))
    called = pd.Series(periodic_peak, index=keys).groupby(level=0).any()
    aligned = star_truth.index.intersection(called.index)
    predicted, actual = called[aligned], star_truth[aligned]
    tp = int((predicted & actual).sum()); fn = int((~predicted & actual).sum())
    fp = int((predicted & ~actual).sum()); tn = int((~predicted & ~actual).sum())
    ell_false = int((periodic_peak & (klass == "ELL") & no_period).sum())
    kept = sum(1 for t, s in REAL_ELL
               if (periodic_peak & (klass == "ELL") & (tic == t) & (sector == s)).any())
    unconstrained = int((~ok).sum())
    return {"regla": label, "acuerdo>=": min_agreement,
            "power>=": min_power if min_power else "-",
            "ELL_falsos": ell_false, "ELL_reales_2": kept,
            "picos_sin_clasificar": unconstrained,
            "TP": tp, "FN": fn, "FP": fp, "TN": tn,
            "acc": round((tp + tn) / (tp + fn + fp + tn), 3)}

results = [evaluate(1, None, "ensemble, sin abstencion")]
for k in (4, 5, 6, 7):
    results.append(evaluate(k, None, f"abstenerse si <{k}/7 coinciden"))
results.append(evaluate(5, 0.5, "acuerdo>=5 + power>=0.5"))
results.append(evaluate(6, 0.5, "acuerdo>=6 + power>=0.5"))
results.append(evaluate(7, 0.5, "acuerdo=7 + power>=0.5"))
print()
print(pd.DataFrame(results).to_string(index=False))
