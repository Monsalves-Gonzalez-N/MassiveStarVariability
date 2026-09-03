"""No el argmax: las PROBABILIDADES por clase en cada normalizacion."""
import numpy as np, pandas as pd
from msv.classify_brf import brf_mc_probs, group_probs, load_brf

AMBIGUOUS = {16187387, 41903679, 71935430, 116065031, 137113608, 153304135}
NORMS = ["log", "min_max", "power_0.5", "power_0.33", "column", "quantized",
         "poisson", "rank", "binary"]

data = np.load("results/cnn_input.npz", allow_pickle=True)
tic = data["TIC"].astype(int); sector = data["sector"].astype(int)
per = data["per"].astype(float); amp = data["amplitude"].astype(float)
power = data["power"].astype(float)
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

grouped_by_norm = {}
for name in NORMS:
    p = np.load(f"results/norm_compare/cnn_mc_{name}.npz")["p_mc"]
    per_pass, _ = brf_mc_probs(p, per, amp, brf)
    grouped, gnames = group_probs(per_pass)
    grouped_by_norm[name] = np.nanmean(grouped, 0)          # (857, 5)
ELL, LPV, RNDM = gnames.index("ELL"), gnames.index("LPV"), gnames.index("Rndm")

# el conjunto de referencia: picos ELL bajo log, separados por etiqueta humana
base = grouped_by_norm["log"]
is_ell_log = np.where(np.isnan(base), -np.inf, base).argmax(1) == ELL
real = is_ell_log & has_period
contaminant = is_ell_log & no_period
print(f"picos ELL en log: {real.sum()} reales, {contaminant.sum()} contaminantes\n")

print("p_ELL y p_LPV MEDIANAS en esos mismos picos, por normalizacion:")
print(f"{'norma':>11} | {'p_ELL real':>10} {'p_ELL cont':>10} | "
      f"{'p_LPV real':>10} {'p_LPV cont':>10} | {'margen ELL-LPV':>14}")
for name in NORMS:
    g = grouped_by_norm[name]
    row = (f"{name:>11} | {np.median(g[real, ELL]):10.3f} {np.median(g[contaminant, ELL]):10.3f} | "
           f"{np.median(g[real, LPV]):10.3f} {np.median(g[contaminant, LPV]):10.3f} | ")
    margin_real = np.median(g[real, ELL] - g[real, LPV])
    margin_cont = np.median(g[contaminant, ELL] - g[contaminant, LPV])
    print(row + f"{margin_real:6.3f} / {margin_cont:6.3f}")

def auc(score, positive, negative):
    a, b = score[positive], score[negative]
    if not len(a) or not len(b):
        return np.nan
    return float((np.subtract.outer(a, b) > 0).mean()
                 + 0.5 * (np.subtract.outer(a, b) == 0).mean())

print("\nAUC para separar ELL REAL de CONTAMINANTE (mas alto = mejor; 0.5 = azar)")
print(f"{'norma':>11} | {'p_ELL':>6} {'p_LPV':>6} {'ELL-LPV':>8} {'p_Rndm':>7}")
for name in NORMS:
    g = grouped_by_norm[name]
    print(f"{name:>11} | {auc(g[:, ELL], real, contaminant):6.3f} "
          f"{auc(-g[:, LPV], real, contaminant):6.3f} "
          f"{auc(g[:, ELL] - g[:, LPV], real, contaminant):8.3f} "
          f"{auc(-g[:, RNDM], real, contaminant):7.3f}")

print("\nCOMBINACIONES entre normalizaciones (no argmax por separado):")
stack_ell = np.stack([grouped_by_norm[n][:, ELL] for n in NORMS])
stack_lpv = np.stack([grouped_by_norm[n][:, LPV] for n in NORMS])
combos = {
    "min p_ELL entre las 9": stack_ell.min(0),
    "media p_ELL entre las 9": stack_ell.mean(0),
    "max p_LPV entre las 9 (invertido)": -stack_lpv.max(0),
    "min (p_ELL - p_LPV)": (stack_ell - stack_lpv).min(0),
    "power (referencia)": power,
}
for label, score in combos.items():
    print(f"  {label:36s} AUC = {auc(score, real, contaminant):.3f}")

print("\nel detalle pico a pico, p_ELL / p_LPV en cada norma:")
watch = [(12921082, 1.7731, "REAL"), (362792232, 4.4513, "REAL"),
         (8301691, 1.3935, "contam"), (384805438, 2.9350, "contam"),
         (13785212, 3.6459, "contam"), (169640678, 2.0212, "contam")]
for t, p, label in watch:
    index = int(np.argmin(np.abs(per - p) + 1e6 * (tic != t)))
    cells = "  ".join(f"{n[:4]}:{grouped_by_norm[n][index, ELL]:.2f}/{grouped_by_norm[n][index, LPV]:.2f}"
                      for n in NORMS)
    print(f"{label:6s} TIC {t:>9} pw={power[index]:.2f}  {cells}")
