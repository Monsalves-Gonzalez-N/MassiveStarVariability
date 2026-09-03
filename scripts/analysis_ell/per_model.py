import numpy as np, pandas as pd
from msv.classify_brf import brf_mc_probs, group_probs, load_brf
from msv.config import MODELS

AMBIGUOUS = {16187387, 41903679, 71935430, 116065031, 137113608, 153304135}
PERIODIC_GROUPS = {"ELL", "Pulsating", "E"}

data = np.load("results/cnn_input.npz", allow_pickle=True)
tic = data["TIC"].astype(int); sector = data["sector"].astype(int)
per = data["per"].astype(float); amp = data["amplitude"].astype(float)
power = data["power"].astype(float); source = data["source"]
p_models = np.load("results/cnn_mc.npz")["p_mc"]          # (7, 857, 8) = 1 por checkpoint
brf = load_brf()

notes = pd.read_csv("catalogs/notas_periodos.csv")
truth = {}
for _, row in notes.iterrows():
    key = (int(row.TIC), int(row.sector))
    if int(row.TIC) in AMBIGUOUS:
        continue
    truth[key] = bool(isinstance(row.per_elegido, str) or np.isfinite(row.per_elegido))
keys = np.array([f"{t}_{s}" for t, s in zip(tic, sector)])
labelled = np.array([(t, s) in truth for t, s in zip(tic, sector)])
star_truth = pd.Series({f"{t}_{s}": v for (t, s), v in truth.items()})
print(f"{len(truth)} estrellas etiquetadas: {sum(truth.values())} con periodo, "
      f"{len(truth) - sum(truth.values())} sin periodo")

REAL_ELL = {(12921082, 82), (362792232, 41)}
no_period = np.array([(t, s) in truth and not truth[(t, s)]
                      for t, s in zip(tic, sector)])

rows = []
for index, name in enumerate(MODELS + ["ENSEMBLE(7)"]):
    if index < len(MODELS):
        probabilities = p_models[index][None, ...]
    else:
        probabilities = p_models
    grouped_cnn, gnames = group_probs(probabilities.mean(0))
    cnn_class = np.array([gnames[i] for i in grouped_cnn.argmax(1)])

    per_pass, _ = brf_mc_probs(probabilities, per, amp, brf)
    grouped_brf, _ = group_probs(per_pass.mean(0))
    brf_prob = np.nanmax(np.where(np.isnan(grouped_brf), -np.inf, grouped_brf), axis=1)
    brf_class = np.array([gnames[i] for i in
                          np.where(np.isnan(grouped_brf), -np.inf, grouped_brf).argmax(1)])

    ell_all = (cnn_class == "ELL").sum()
    ell_false = ((cnn_class == "ELL") & no_period).sum()
    ell_false_brf = ((brf_class == "ELL") & no_period).sum()
    ell_index = gnames.index("ELL")
    lpv_index = gnames.index("LPV")
    spurious = (cnn_class == "ELL") & no_period
    lpv_in_spurious = (np.median(grouped_cnn[spurious, lpv_index])
                       if spurious.any() else np.nan)

    # nivel estrella: periodica si algun pico tiene clase periodica con prob >= 0.8
    called = pd.Series(
        [(c in PERIODIC_GROUPS) and (p >= 0.8) for c, p in zip(brf_class, brf_prob)],
        index=keys).groupby(level=0).any()
    aligned = star_truth.index.intersection(called.index)
    predicted, actual = called[aligned], star_truth[aligned]
    tp = int((predicted & actual).sum()); fn = int((~predicted & actual).sum())
    fp = int((predicted & ~actual).sum()); tn = int((~predicted & ~actual).sum())

    keeps = sum(1 for t, s in REAL_ELL
                if (cnn_class[(tic == t) & (sector == s)] == "ELL").any())
    rows.append({"checkpoint": name, "ELL_total": ell_all,
                 "ELL_falsos_CNN": ell_false, "ELL_falsos_BRF": ell_false_brf,
                 "medLPV_en_ELLfalso": round(float(lpv_in_spurious), 4),
                 "TP": tp, "FN": fn, "FP": fp, "TN": tn,
                 "acc": round((tp + tn) / (tp + fn + fp + tn), 3),
                 "ELL_reales_2": keeps})
print()
print(pd.DataFrame(rows).to_string(index=False))
