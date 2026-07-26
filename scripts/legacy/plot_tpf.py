#!/usr/bin/env python3
import math, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REPORTS_DIR = os.path.join(REPO_ROOT, "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)
DATA_DIR = os.path.join(REPO_ROOT, "data")

CATALOGS = {
    "LS HistNorm min-max":  "CNN_RF_prediction_LS_min_max.csv",
    "LS HistNorm log":      "CNN_RF_prediction_LS_log.csv",
    "ACF HistNorm min-max": "CNN_RF_prediction_ACF_min_max.csv",
    "ACF HistNorm log":     "CNN_RF_prediction_ACF_log.csv",
}
CUBES = {
    "LS HistNorm min-max":  "Github_MassiveXTessV8_LC_Aperture_Vsx_LS_NO_sigma_cliping_min_max.npy",
    "LS HistNorm log":      "Github_MassiveXTessV8_LC_Aperture_Vsx_LS_NO_sigma_cliping_log.npy",
    "ACF HistNorm min-max": "Github_MassiveXTessV8_LC_Aperture_Vsx_ACF_NO_sigma_cliping_min_max.npy",
    "ACF HistNorm log":     "Github_MassiveXTessV8_LC_Aperture_Vsx_ACF_NO_sigma_cliping_log.npy",
}

CLASS_COLS = ["CNN+RF_ELL","CNN+RF_M","CNN+RF_CEP","CNN+RF_DST",
              "CNN+RF_E","CNN+RF_LPV","CNN+RF_RR","CNN+RF_Rndm"]

TIC     = 56630835
SECTORS = [8, 34, 35]
CMAP    = "viridis"
OUTPUT  = os.path.join(REPORTS_DIR, "tpf_grid_56630835.pdf")


def load_cat(fname):
    path = os.path.join(DATA_DIR, "catalogos", fname)
    df = pd.read_csv(path)
    avail = [c for c in CLASS_COLS if c in df.columns]
    df["CNN+RF_max_prob"] = df[avail].max(axis=1)
    df["final_class"]    = df[avail].idxmax(axis=1).str.replace("CNN+RF_", "", regex=False)
    return df


MAX_PER_PAGE = 64   # máx subplots por página

def plot_tpf_grid(df, cube, tic, sector, cat_name):
    """Devuelve lista de figuras (una por página si hay muchos períodos)."""
    subset = df.loc[(df["TIC"] == tic) & (df["sector_list"] == sector)].copy()
    n = len(subset)
    if n == 0:
        return []

    figs = []
    rows_list = list(subset.iterrows())
    n_pages = math.ceil(n / MAX_PER_PAGE)

    for page in range(n_pages):
        chunk = rows_list[page * MAX_PER_PAGE : (page + 1) * MAX_PER_PAGE]
        nc = math.ceil(math.sqrt(len(chunk)))
        nr = math.ceil(len(chunk) / nc)

        fig, axes = plt.subplots(nr, nc, figsize=(2.0 * nc, 2.0 * nr))
        page_label = f"  ({page+1}/{n_pages})" if n_pages > 1 else ""
        fig.suptitle(
            f"TIC {tic}  —  Sector {sector}  —  {cat_name}{page_label}",
            fontsize=12, fontweight="bold",
        )
        axes_flat = axes.flatten() if len(chunk) > 1 else [axes]

        for i, (idx, row) in enumerate(chunk):
            img = cube[idx]
            if img.ndim == 3:
                img = img[:, :, 0]
            axes_flat[i].imshow(img, origin="lower", cmap=CMAP)
            axes_flat[i].set_title(
                f"P={row['period']:.4f} d\n{row['final_class']}  p={row['CNN+RF_max_prob']:.3f}",
                fontsize=5,
            )
            axes_flat[i].axis("off")

        for j in range(i + 1, len(axes_flat)):
            fig.delaxes(axes_flat[j])

        plt.tight_layout()
        figs.append(fig)

    return figs


print("Cargando catálogos y cubos...")
loaded = {
    name: (load_cat(csv_fname),
           np.load(os.path.join(DATA_DIR, "cubos", CUBES[name]), mmap_mode="r"))
    for name, csv_fname in CATALOGS.items()
}

with PdfPages(OUTPUT) as pdf:
    for sector in SECTORS:
        print(f"\nSector {sector}:")
        for cat_name, (df, cube) in loaded.items():
            subset = df.loc[(df["TIC"] == TIC) & (df["sector_list"] == sector)]
            if len(subset) == 0:
                print(f"  {cat_name}: sin datos")
                continue
            print(f"  {cat_name}: {len(subset)} períodos")
            figs = plot_tpf_grid(df, cube, TIC, sector, cat_name)
            for fig in figs:
                pdf.savefig(fig, dpi=100)
                plt.close(fig)

print(f"\nGuardado: {OUTPUT}")
