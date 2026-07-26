#!/usr/bin/env python3
"""
Por cada catálogo × sector:
  Fila 1 : imagen del cubo (periodograma/ACF normalizado)
  Fila 2 : curva de luz en fase
Candidatos: prob > PROB_THRESH y clase != Rndm (igual que la última columna de la tabla).
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REPORTS_DIR = os.path.join(REPO_ROOT, "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)
DATA_DIR = os.path.join(REPO_ROOT, "data")

CATALOGS = {
    "LS HistNorm min-max":  ("CNN_RF_prediction_LS_min_max.csv",
                             "Github_MassiveXTessV8_LC_Aperture_Vsx_LS_NO_sigma_cliping_min_max.npy"),
    "LS HistNorm log":      ("CNN_RF_prediction_LS_log.csv",
                             "Github_MassiveXTessV8_LC_Aperture_Vsx_LS_NO_sigma_cliping_log.npy"),
    "ACF HistNorm min-max": ("CNN_RF_prediction_ACF_min_max.csv",
                             "Github_MassiveXTessV8_LC_Aperture_Vsx_ACF_NO_sigma_cliping_min_max.npy"),
    "ACF HistNorm log":     ("CNN_RF_prediction_ACF_log.csv",
                             "Github_MassiveXTessV8_LC_Aperture_Vsx_ACF_NO_sigma_cliping_log.npy"),
}
CLASS_COLS = ["CNN+RF_ELL","CNN+RF_M","CNN+RF_CEP","CNN+RF_DST",
              "CNN+RF_E","CNN+RF_LPV","CNN+RF_RR","CNN+RF_Rndm"]

TIC         = 56630835
SECTORS     = [8, 34, 35]
PROB_THRESH = 0.8
CMAP        = "viridis"
OUTPUT      = os.path.join(REPORTS_DIR, f"phase_grid_{TIC}.pdf")


def load_cat(fname):
    df = pd.read_csv(os.path.join(DATA_DIR, "catalogos", fname))
    avail = [c for c in CLASS_COLS if c in df.columns]
    df["CNN+RF_max_prob"] = df[avail].max(axis=1)
    df["final_class"]    = df[avail].idxmax(axis=1).str.replace("CNN+RF_", "", regex=False)
    return df


def load_lc(tic, sector):
    path = os.path.join(DATA_DIR, "lightcurves", f"{tic}_{sector}.csv")
    return pd.read_csv(path) if os.path.exists(path) else None


def make_page(tic, sector, cat_name, df, cube, pdf):
    """Una página: fila 1 = imágenes cubo, fila 2 = fases. Columnas = candidatos."""
    cands = df.loc[
        (df["TIC"] == tic) &
        (df["sector_list"] == sector) &
        (df["CNN+RF_max_prob"] > PROB_THRESH) &
        (df["final_class"] != "Rndm")
    ].sort_values("CNN+RF_max_prob", ascending=False)

    n = len(cands)
    if n == 0:
        return

    lc = load_lc(tic, sector)

    col_w = 2.4
    fig_w = max(col_w * n, 5)
    fig   = plt.figure(figsize=(fig_w, 5.5))
    fig.suptitle(
        f"TIC {tic}  —  Sector {sector}  —  {cat_name}  —  prob > {PROB_THRESH}, ≠ Rndm",
        fontsize=10, fontweight="bold", y=0.998,
    )

    gs = gridspec.GridSpec(
        2, n, figure=fig,
        height_ratios=[1.0, 1.0],
        hspace=0.5, wspace=0.2,
        left=0.06, right=0.99, top=0.91, bottom=0.10,
    )

    for ci, (idx, row) in enumerate(cands.iterrows()):
        # ── Fila 0: imagen del cubo ───────────────────────────────────────
        ax_img = fig.add_subplot(gs[0, ci])
        img = cube[idx]
        if img.ndim == 3:
            img = img[:, :, 0]
        ax_img.imshow(img, origin="lower", cmap=CMAP, aspect="auto")
        ax_img.set_title(
            f"P={row['period']:.4f} d\n{row['final_class']}  {row['CNN+RF_max_prob']:.3f}",
            fontsize=7, pad=3,
        )
        ax_img.axis("off")

        # ── Fila 1: curva en fase ─────────────────────────────────────────
        ax_ph = fig.add_subplot(gs[1, ci])
        if lc is not None:
            period = row["period"]
            phase  = ((lc["Time"] - lc["Time"].min()) % period) / period
            ax_ph.scatter(phase, lc["flux"], s=0.4, alpha=0.6,
                          c="steelblue", linewidths=0)
            ax_ph.set_xlabel("Fase", fontsize=6.5, labelpad=1)
            ax_ph.tick_params(labelsize=5.5, pad=1)
            ax_ph.yaxis.set_major_locator(plt.MaxNLocator(3))
            ax_ph.xaxis.set_major_locator(plt.MaxNLocator(4))
            if ci == 0:
                ax_ph.set_ylabel("Flujo [e⁻/s]", fontsize=6.5, labelpad=2)
        else:
            ax_ph.text(0.5, 0.5, "Sin LC", ha="center", va="center", fontsize=8)
            ax_ph.axis("off")

    pdf.savefig(fig, dpi=120)
    plt.close(fig)
    print(f"    {n} candidatos")


# ── Main ─────────────────────────────────────────────────────────────────────
print("Cargando catálogos y cubos...")
loaded = {
    name: (load_cat(csv_f),
           np.load(os.path.join(DATA_DIR, "cubos", cube_f), mmap_mode="r"))
    for name, (csv_f, cube_f) in CATALOGS.items()
}

with PdfPages(OUTPUT) as pdf:
    for cat_name, (df, cube) in loaded.items():
        print(f"\n{cat_name}:")
        for sector in SECTORS:
            print(f"  S{sector}:", end="")
            make_page(TIC, sector, cat_name, df, cube, pdf)

print(f"\nGuardado: {OUTPUT}")
