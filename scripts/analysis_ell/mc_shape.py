import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from msv.classify_brf import brf_mc_probs, group_probs, load_brf
from msv.viz import GROUP_COLORS

AMBIGUOUS = {16187387, 41903679, 71935430, 116065031, 137113608, 153304135}

data = np.load("results/cnn_input.npz", allow_pickle=True)
tic = data["TIC"].astype(int); sector = data["sector"].astype(int)
per = data["per"].astype(float); amp = data["amplitude"].astype(float)
power = data["power"].astype(float)
p_mc = np.load("results/cnn_mc20.npz")["p_mc"]              # (20, 857, 8)
brf = load_brf()

pp, _ = brf_mc_probs(p_mc, per, amp, brf)
gb, gnames = group_probs(pp)                                 # (20, 857, 5)
mean = np.nanmean(gb, 0)
winner = np.where(np.isnan(mean), -np.inf, mean).argmax(1)
rows = np.arange(len(tic))
values = gb[:, rows, winner]                                 # (20, 857) la ganadora
klass = np.array([gnames[i] for i in winner])

notes = pd.read_csv("catalogs/notas_periodos.csv")
truth = {}
for _, row in notes.iterrows():
    if int(row.TIC) in AMBIGUOUS:
        continue
    truth[(int(row.TIC), int(row.sector))] = bool(
        isinstance(row.per_elegido, str) or np.isfinite(row.per_elegido))
no_period = np.array([(t, s) in truth and not truth[(t, s)]
                      for t, s in zip(tic, sector)])
has_period = np.array([(t, s) in truth and truth[(t, s)]
                       for t, s in zip(tic, sector)])

# --- metricas de FORMA de la distribucion de 20 valores -------------------
low_mass = (values < 0.1).mean(0)
high_mass = (values > 0.9).mean(0)
middle = ((values >= 0.1) & (values <= 0.9)).mean(0)
spread = values.max(0) - values.min(0)
gap = np.array([np.max(np.diff(np.sort(column))) for column in values.T])
shape = pd.DataFrame({"clase": klass, "media": mean[rows, winner],
                      "baja": low_mass, "alta": high_mass, "medio": middle,
                      "rango": spread, "salto_max": gap, "power": power,
                      "no_period": no_period, "has_period": has_period})

with PdfPages("results/figures/mc_shape_20.pdf") as pdf:
    figure, axes = plt.subplots(2, 3, figsize=(15, 8))

    # A: histograma de TODOS los valores por pasada, por clase
    axis = axes[0, 0]
    for name in gnames:
        which = klass == name
        if which.sum():
            axis.hist(values[:, which].ravel(), bins=40, range=(0, 1),
                      histtype="step", lw=1.6, density=True,
                      color=GROUP_COLORS.get(name, "0.5"), label=f"{name} ({which.sum()})")
    axis.set_yscale("log"); axis.legend(fontsize=7)
    axis.set_xlabel("prob de la clase ganadora, valor por pasada")
    axis.set_ylabel("densidad (log)")
    axis.set_title("A. las 20 pasadas de todos los picos\nla masa se apila en 0 y en 1", fontsize=9)

    # B: raster de los picos ELL, 20 puntos por pico, ordenado por media
    axis = axes[0, 1]
    ell = np.where(klass == "ELL")[0]
    ell = ell[np.argsort(mean[ell, winner[ell]])]
    for position, index in enumerate(ell):
        color = ("tab:red" if no_period[index]
                 else "tab:green" if has_period[index] else "0.6")
        axis.plot(values[:, index], np.full(20, position), ".", ms=2.5,
                  color=color, alpha=0.7)
    axis.set_xlabel("prob ELL por pasada"); axis.set_ylabel("picos ELL, ordenados por media")
    axis.set_title("B. un pico por fila, sus 20 pasadas\nrojo = estrella SIN periodo, verde = CON",
                   fontsize=9)

    # C: media contra rango (max-min): el arco de la bimodalidad
    axis = axes[0, 2]
    for name in gnames:
        which = klass == name
        axis.scatter(shape.media[which], shape.rango[which], s=8, alpha=0.5,
                     color=GROUP_COLORS.get(name, "0.5"), label=name)
    axis.set_xlabel("media de las 20"); axis.set_ylabel("rango (max - min)")
    axis.legend(fontsize=7)
    axis.set_title("C. una distribucion unimodal vive abajo;\nel arco superior es bimodal", fontsize=9)

    # D: fraccion en el medio, ELL falsos vs reales
    axis = axes[1, 0]
    for label, mask, color in [("estrella SIN periodo", no_period, "tab:red"),
                               ("estrella CON periodo", has_period, "tab:green")]:
        subset = shape[(shape.clase == "ELL") & mask]
        axis.hist(subset.medio, bins=np.linspace(0, 1, 11), histtype="step",
                  lw=1.8, color=color, label=f"{label} (N={len(subset)})")
    axis.set_xlabel("fraccion de las 20 pasadas con prob en [0.1, 0.9]")
    axis.set_ylabel("picos ELL"); axis.legend(fontsize=7)
    axis.set_title("D. la forma NO separa ELL falso de real", fontsize=9)

    # E: la forma contra power, que si separa
    axis = axes[1, 1]
    for label, mask, color in [("SIN periodo", no_period, "tab:red"),
                               ("CON periodo", has_period, "tab:green")]:
        subset = shape[(shape.clase == "ELL") & mask]
        axis.scatter(subset.power, subset.medio, s=26, alpha=0.75, color=color,
                     label=f"{label} (N={len(subset)})")
    axis.axvline(0.5, color="0.6", ls="--", lw=1)
    axis.set_xlabel("power del pico"); axis.set_ylabel("fraccion en el medio")
    axis.legend(fontsize=7)
    axis.set_title("E. picos ELL: separa el eje x (power),\nno el eje y (forma MC)", fontsize=9)

    # F: los 6 picos ELL de mayor y menor media, sus 20 valores crudos
    axis = axes[1, 2]
    pick = list(ell[:4]) + list(ell[-4:])
    for position, index in enumerate(pick):
        color = "tab:red" if no_period[index] else "tab:green" if has_period[index] else "0.6"
        axis.plot(values[:, index], np.full(20, position), "o", ms=4,
                  color=color, alpha=0.6)
        axis.text(1.02, position, f"TIC {tic[index]}  pw={power[index]:.2f}",
                  fontsize=6.5, va="center")
    axis.set_yticks([]); axis.set_xlim(-0.05, 1.05)
    axis.set_xlabel("prob ELL por pasada")
    axis.set_title("F. 8 picos ELL individuales", fontsize=9)

    figure.tight_layout()
    pdf.savefig(figure, dpi=150)
    plt.close(figure)

print("results/figures/mc_shape_20.pdf")

# --- sirve la forma para predecir? ---------------------------------------
def auc(score, positive, negative):
    a, b = score[positive], score[negative]
    if not len(a) or not len(b):
        return np.nan
    return float((np.subtract.outer(a, b) > 0).mean()
                 + 0.5 * (np.subtract.outer(a, b) == 0).mean())

ell = klass == "ELL"
print("\nAUC a nivel PICO, separar ELL de estrella CON periodo vs SIN periodo:")
for column in ["media", "baja", "alta", "medio", "rango", "salto_max", "power"]:
    score = shape[column].to_numpy()
    print(f"  {column:11s} {auc(score, ell & has_period, ell & no_period):.3f}")
print(f"\n  N: ELL en estrellas con periodo = {(ell & has_period).sum()}, "
      f"sin periodo = {(ell & no_period).sum()}")
