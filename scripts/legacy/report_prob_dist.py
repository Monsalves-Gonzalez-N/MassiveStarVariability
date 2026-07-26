#!/usr/bin/env python3
"""
Genera un PDF con la distribución de CNN+RF_max_prob de los catálogos path2.

Página 1 : Distribución global (todas las clases superpuestas, densidad).
Página 2 : Grid de histogramas individuales por clase.
"""

import os, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages

warnings.filterwarnings("ignore")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REPORTS_DIR = os.path.join(REPO_ROOT, "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)
MAIN_CSV = os.path.join(REPO_ROOT, "data/catalogos/catalog_path2_main.csv")
OUTPUT   = os.path.join(REPORTS_DIR, "reporte_prob_dist_path2.pdf")
THRESH   = 0.8

CLASS_COLORS = {
    "ELL":           "#4C72B0",
    "Pulsating":     "#2CA02C",
    "E":             "#8172B2",
    "Unconstrained": "#7F7F7F",
    "Rndm":          "#8C8C8C",
    "Irregular":     "#BCBD22",
    "Mixed":         "#17BECF",
    "No_periodo":    "#AAAAAA",
    "LPV":           "#937860",
    "M":             "#DD8452",
    "CEP":           "#55A868",
    "DST":           "#C44E52",
    "RR":            "#DA8BC3",
}

print("Cargando catálogo principal path2...")
df = pd.read_csv(MAIN_CSV)

# Excluir No_periodo (prob = 0, no hay peak real)
df = df[df["final_class_tic"] != "No_periodo"].copy()

classes_order = (
    df.groupby("final_class_tic")["CNN+RF_max_prob"]
    .median()
    .sort_values(ascending=False)
    .index.tolist()
)

n_cls    = len(classes_order)
bins     = np.linspace(0, 1, 51)
bin_c    = 0.5 * (bins[:-1] + bins[1:])

with PdfPages(OUTPUT) as pdf:

    # ── Página 1: Distribución global superpuesta ─────────────────────────────
    fig1, axes1 = plt.subplots(1, 2, figsize=(14, 6.5),
                               gridspec_kw={"width_ratios": [1.6, 1]})
    fig1.suptitle(
        "Distribución de probabilidad máxima (CNN+RF_max_prob) — Catálogos Path2",
        fontsize=13, fontweight="bold",
    )

    # Panel izquierdo: densidad superpuesta por clase
    ax_ov = axes1[0]
    for cls in classes_order:
        sub  = df[df["final_class_tic"] == cls]["CNN+RF_max_prob"].dropna()
        if len(sub) < 2:
            continue
        color = CLASS_COLORS.get(cls, "gray")
        ax_ov.hist(sub, bins=bins, density=True, alpha=0.55,
                   color=color, label=f"{cls} (n={len(sub)})", linewidth=0)
        ax_ov.hist(sub, bins=bins, density=True, histtype="step",
                   color=color, linewidth=1.1)

    ax_ov.axvline(THRESH, color="red", lw=1.4, ls="--",
                  label=f"umbral = {THRESH}", zorder=5)
    ax_ov.set_xlabel("CNN+RF_max_prob", fontsize=10)
    ax_ov.set_ylabel("Densidad", fontsize=10)
    ax_ov.set_title("Todas las clases superpuestas", fontsize=10)
    ax_ov.set_xlim(0, 1)
    ax_ov.legend(fontsize=8, loc="upper left", framealpha=0.85)
    ax_ov.grid(True, alpha=0.25)

    # Panel derecho: tabla de estadísticas por clase
    ax_tbl = axes1[1]
    ax_tbl.axis("off")

    stats = []
    for cls in classes_order:
        sub = df[df["final_class_tic"] == cls]["CNN+RF_max_prob"].dropna()
        n_above = (sub >= THRESH).sum()
        stats.append([
            cls,
            str(len(sub)),
            f"{sub.median():.3f}",
            f"{sub.mean():.3f}",
            f"{sub.std():.3f}",
            f"{n_above} ({100*n_above/len(sub):.0f}%)" if len(sub) > 0 else "—",
        ])

    col_labels = ["Clase", "N filas", "Mediana", "Media", "Std", f"≥ {THRESH}"]
    tbl = ax_tbl.table(cellText=stats, colLabels=col_labels,
                       cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1, 1.55)

    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#4a6fa5")
            cell.set_text_props(color="white", fontweight="bold")
        else:
            cls_name = stats[r - 1][0]
            base = CLASS_COLORS.get(cls_name, "#cccccc")
            cell.set_facecolor(base + "33")
        cell.set_edgecolor("#cccccc")

    fig1.tight_layout(rect=[0, 0, 1, 0.95])
    pdf.savefig(fig1, dpi=130)
    plt.close(fig1)

    # ── Página 2: Grid individual por clase ───────────────────────────────────
    ncols   = 3
    nrows   = int(np.ceil(n_cls / ncols))
    fig2    = plt.figure(figsize=(14, 3.5 * nrows + 0.8))
    fig2.suptitle(
        "Histogramas individuales por clase — CNN+RF_max_prob",
        fontsize=13, fontweight="bold",
    )

    gs2 = gridspec.GridSpec(nrows, ncols, figure=fig2,
                            hspace=0.55, wspace=0.30,
                            left=0.06, right=0.97,
                            top=0.93, bottom=0.06)

    for idx, cls in enumerate(classes_order):
        r, c = divmod(idx, ncols)
        ax   = fig2.add_subplot(gs2[r, c])
        sub  = df[df["final_class_tic"] == cls]["CNN+RF_max_prob"].dropna()
        color = CLASS_COLORS.get(cls, "steelblue")

        if len(sub) >= 2:
            ax.hist(sub, bins=bins, color=color, alpha=0.75,
                    edgecolor="white", linewidth=0.4)
            ax.axvline(THRESH, color="red", lw=1.2, ls="--", alpha=0.8)

            # Líneas de media y mediana
            ax.axvline(sub.mean(),   color="#333333", lw=1.0, ls="-",
                       label=f"media {sub.mean():.3f}")
            ax.axvline(sub.median(), color="#333333", lw=1.0, ls=":",
                       label=f"mediana {sub.median():.3f}")

            n_above = (sub >= THRESH).sum()
            ax.text(0.97, 0.97,
                    f"n = {len(sub)}\n"
                    f"≥{THRESH}: {n_above} ({100*n_above/len(sub):.0f}%)",
                    ha="right", va="top", fontsize=8,
                    transform=ax.transAxes,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white",
                              ec=color, lw=1.0, alpha=0.9))
            ax.legend(fontsize=7, loc="upper left", framealpha=0.7)
        else:
            ax.text(0.5, 0.5, f"n={len(sub)}\n(sin datos suficientes)",
                    ha="center", va="center", fontsize=9, color="gray",
                    transform=ax.transAxes)

        ax.set_facecolor(CLASS_COLORS.get(cls, "#cccccc") + "18")
        ax.set_xlim(0, 1)
        ax.set_xlabel("CNN+RF_max_prob", fontsize=8)
        ax.set_ylabel("Frecuencia", fontsize=8)
        ax.set_title(cls, fontsize=11, fontweight="bold",
                     color=CLASS_COLORS.get(cls, "black"))
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.25)

    # Apagar subplots vacíos
    for idx in range(n_cls, nrows * ncols):
        r, c = divmod(idx, ncols)
        fig2.add_subplot(gs2[r, c]).axis("off")

    pdf.savefig(fig2, dpi=130)
    plt.close(fig2)

    d = pdf.infodict()
    d["Title"]   = "Distribución prob máxima — Path2"
    d["Subject"] = "CNN+RF_max_prob por clase (final_class_tic)"

print(f"PDF generado: {OUTPUT}")
