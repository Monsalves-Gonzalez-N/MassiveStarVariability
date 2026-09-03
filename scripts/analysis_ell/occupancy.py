"""Es p_LPV(rank) un proxy de la OCUPACION del histograma?"""
import sys
import numpy as np, pandas as pd
sys.path.insert(0, "scripts")
from msv.features import _phase_fold_counts
from msv.classify_brf import brf_mc_probs, group_probs, load_brf

AMBIGUOUS = {16187387, 41903679, 71935430, 116065031, 137113608, 153304135}
data = np.load("results/cnn_input.npz", allow_pickle=True)
tic = data["TIC"].astype(int); sector = data["sector"].astype(int)
per = data["per"].astype(float); amp = data["amplitude"].astype(float)
power = data["power"].astype(float)
curves = pd.read_pickle("results/phasefold_curves.pkl")

occupancy = np.empty(len(per)); entropy = np.empty(len(per))
band = np.empty(len(per))
for index, (t, s, period) in enumerate(zip(tic, sector, per)):
    time, flux = curves[(int(t), int(s))]
    good = np.isfinite(time) & np.isfinite(flux)
    counts = _phase_fold_counts(time[good], flux[good], period)
    occupancy[index] = (counts > 0).mean()
    fraction = counts / counts.sum()
    positive = fraction[fraction > 0]
    entropy[index] = -(positive * np.log(positive)).sum() / np.log(counts.size)
    # ancho vertical mediano de la banda ocupada por columna de fase
    per_column = (counts > 0).sum(axis=1)
    band[index] = np.median(per_column[per_column > 0]) / counts.shape[1]

notes = pd.read_csv("catalogs/notas_periodos.csv")
truth = {}
for _, row in notes.iterrows():
    if int(row.TIC) in AMBIGUOUS:
        continue
    truth[(int(row.TIC), int(row.sector))] = bool(
        isinstance(row.per_elegido, str) or np.isfinite(row.per_elegido))
no_period = np.array([(t, s) in truth and not truth[(t, s)] for t, s in zip(tic, sector)])
has_period = np.array([(t, s) in truth and truth[(t, s)] for t, s in zip(tic, sector)])

brf = load_brf()
p = np.load("results/norm_compare/cnn_mc_rank.npz")["p_mc"]
per_pass, _ = brf_mc_probs(p, per, amp, brf)
g, gnames = group_probs(per_pass)
p_lpv_rank = np.nanmean(g, 0)[:, gnames.index("LPV")]

print("correlacion de p_LPV(rank) con cantidades directas del histograma:")
for label, values in [("ocupacion (frac pixeles > 0)", occupancy),
                      ("entropia normalizada", entropy),
                      ("ancho de banda por fase", band),
                      ("power", power)]:
    finite = np.isfinite(values) & np.isfinite(p_lpv_rank)
    print(f"  {label:30s} Pearson r = {np.corrcoef(p_lpv_rank[finite], values[finite])[0,1]:+.3f}")

def auc(score, positive, negative):
    a, b = score[positive], score[negative]
    return float((np.subtract.outer(a, b) > 0).mean()
                 + 0.5 * (np.subtract.outer(a, b) == 0).mean())

labelled_pos, labelled_neg = has_period, no_period
print(f"\nAUC separando picos de estrella CON periodo ({labelled_pos.sum()}) "
      f"de SIN periodo ({labelled_neg.sum()}):")
for label, score in [("p_LPV(rank), invertido", -p_lpv_rank),
                     ("ocupacion, invertida", -occupancy),
                     ("entropia, invertida", -entropy),
                     ("ancho de banda, invertido", -band),
                     ("power", power)]:
    print(f"  {label:28s} AUC = {auc(score, labelled_pos, labelled_neg):.3f}")

print("\nvalores medianos:")
frame = pd.DataFrame({"ocupacion": occupancy, "entropia": entropy,
                      "banda": band, "p_LPV_rank": p_lpv_rank, "power": power,
                      "grupo": np.where(has_period, "CON periodo",
                                 np.where(no_period, "SIN periodo", "sin etiqueta"))})
print(frame[frame.grupo != "sin etiqueta"].groupby("grupo").median().round(4).to_string())

print("\nA NIVEL ESTRELLA (que es donde se decide), min de la ocupacion por estrella:")
stars = pd.DataFrame({"TIC": tic, "sector": sector, "occ": occupancy,
                      "power": power, "p_lpv": p_lpv_rank})
stars["key"] = [f"{t}_{s}" for t, s in zip(tic, sector)]
truth_series = pd.Series({f"{t}_{s}": v for (t, s), v in truth.items()})
agg = stars.groupby("key").agg(min_occ=("occ", "min"), max_power=("power", "max"),
                               min_lpv=("p_lpv", "min"))
agg = agg.join(truth_series.rename("periodica"), how="inner")
pos = agg.periodica.values.astype(bool)
for column in ["min_occ", "max_power", "min_lpv"]:
    sign = -1 if column != "max_power" else 1
    print(f"  {column:10s} AUC = {auc(sign * agg[column].values, pos, ~pos):.3f}")
print(agg.groupby("periodica")[["min_occ", "max_power", "min_lpv"]].describe().round(3).to_string())
