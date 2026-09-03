import numpy as np, pandas as pd
from msv.classify_brf import brf_mc_probs, group_probs, load_brf

AMBIGUOUS = {16187387, 41903679, 71935430, 116065031, 137113608, 153304135}
PERIODIC = {"ELL", "Pulsating", "E"}

data = np.load("results/cnn_input.npz", allow_pickle=True)
tic = data["TIC"].astype(int); sector = data["sector"].astype(int)
per = data["per"].astype(float); amp = data["amplitude"].astype(float)
power = data["power"].astype(float); source = data["source"]
p_mc = np.load("results/cnn_mc20.npz")["p_mc"]
brf = load_brf()
pp, _ = brf_mc_probs(p_mc, per, amp, brf)
gb, gnames = group_probs(pp)
mean = np.nanmean(gb, 0)
winner = np.where(np.isnan(mean), -np.inf, mean).argmax(1)
rows = np.arange(len(tic))
klass = np.array([gnames[i] for i in winner]); prob = mean[rows, winner]

notes = pd.read_csv("catalogs/notas_periodos.csv")
truth = {}
for _, row in notes.iterrows():
    if int(row.TIC) in AMBIGUOUS:
        continue
    truth[(int(row.TIC), int(row.sector))] = bool(
        isinstance(row.per_elegido, str) or np.isfinite(row.per_elegido))
no_period = np.array([(t, s) in truth and not truth[(t, s)] for t, s in zip(tic, sector)])
has_period = np.array([(t, s) in truth and truth[(t, s)] for t, s in zip(tic, sector)])

print("picos que SOBREVIVEN un corte de power, en estrellas SIN periodo (contaminantes):")
print(f"{'power>=':>8} {'ELL':>5} {'Puls':>5} {'E':>4} | {'ELL en estrellas CON periodo':>28}")
for cut in [0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    ok = (power >= cut) & (prob >= 0.8) & np.isin(klass, list(PERIODIC))
    counts = {name: int((ok & no_period & (klass == name)).sum()) for name in PERIODIC}
    good_ell = int((ok & has_period & (klass == "ELL")).sum())
    print(f"{cut:8.1f} {counts['ELL']:5d} {counts['Pulsating']:5d} {counts['E']:4d} | {good_ell:28d}")

print("\ncomposicion de los contaminantes que sobreviven power>=0.5, por clase:")
ok = (power >= 0.5) & (prob >= 0.8) & np.isin(klass, list(PERIODIC)) & no_period
frame = pd.DataFrame({"TIC": tic[ok], "sector": sector[ok], "source": source[ok],
                      "per": per[ok].round(3), "power": power[ok].round(3),
                      "clase": klass[ok], "prob": prob[ok].round(3),
                      "amplitud": amp[ok].round(4)}).sort_values("power", ascending=False)
print(frame.to_string(index=False))
print("\nnotas del humano para esas estrellas:")
for t in frame.TIC.unique():
    note = notes[notes.TIC == t].nota.iloc[0]
    print(f"  {t}: {note}")

print("\nProporcion de ELL entre los contaminantes, antes y despues del corte:")
for cut in [0.0, 0.5, 0.7]:
    ok = (power >= cut) & (prob >= 0.8) & np.isin(klass, list(PERIODIC)) & no_period
    total = int(ok.sum())
    ell = int((ok & (klass == "ELL")).sum())
    share = f"{ell / total:.1%}" if total else "-"
    print(f"  power>={cut}: {ell} de {total} contaminantes son ELL ({share})")
