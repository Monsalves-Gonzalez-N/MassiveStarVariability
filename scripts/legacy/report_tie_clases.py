#!/usr/bin/env python3
"""
5 ejemplos por clase (ELL, Pulsating, E) × fuente (ACF, LS) donde ocurre
Tie1: ≥2 peaks comparten la probabilidad máxima.
Layout por página: curva de luz  |  fase mayor power  |  fase menor power
Marcadores negros únicamente.
"""

import os, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

warnings.filterwarnings("ignore")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REPORTS_DIR = os.path.join(REPO_ROOT, "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)
ACF_CSV      = os.path.join(REPO_ROOT, "data/catalogos/CNN_RF_prediction_ACF_log.csv")
LS_CSV       = os.path.join(REPO_ROOT, "data/catalogos/CNN_RF_prediction_LS_log.csv")
LC_DIR       = os.path.join(REPO_ROOT, "data/lightcurves")
OUTPUT       = os.path.join(REPORTS_DIR, "reporte_tie_clases.pdf")
N_EACH       = 5
THRESHOLD    = 0.8
NON_PERIODIC = {"No_periodo", "Rndm", "LPV"}
RNG          = np.random.default_rng(seed=42)

CLASSES  = ["ELL", "Pulsating", "E"]
SOURCES  = [("ACF", ACF_CSV), ("LS", LS_CSV)]


def load_lc(tic, sector):
    path = os.path.join(LC_DIR, f"{tic}_{sector}.csv")
    return pd.read_csv(path) if os.path.exists(path) else None


def phase_fold(lc, period):
    t, f, fe = lc["Time"].values, lc["flux"].values, lc["flux_err"].values
    phi = ((t - t[0]) % period) / period
    srt = np.argsort(phi)
    return phi[srt], f[srt], fe[srt]


def find_ties(df, cls):
    """Devuelve lista de casos tie con LC disponible para la clase dada."""
    periodic = df[~df["final_class"].isin(NON_PERIODIC)].copy()
    ties = []
    for (tic, sector), grp in periodic.groupby(["TIC", "sector_list"]):
        max_prob = grp["CNN+RF_max_prob"].max()
        if max_prob < THRESHOLD:
            continue
        candidates = grp[grp["CNN+RF_max_prob"] == max_prob]
        if len(candidates) < 2:
            continue
        row_hi = candidates.loc[candidates["power"].idxmax()]
        row_lo = candidates.loc[candidates["power"].idxmin()]
        if row_hi.name == row_lo.name:
            continue
        if row_hi["final_class"] != cls:
            continue
        lc = load_lc(int(tic), int(sector))
        if lc is None:
            continue
        ties.append({
            "tic": int(tic), "sector": int(sector),
            "prob": max_prob,
            "row_hi": row_hi, "row_lo": row_lo,
            "lc": lc,
        })
    return ties


def draw_page(pdf, sample, cls, src):
    n = len(sample)
    fig, axes = plt.subplots(n, 3, figsize=(14, 2.6 * n + 0.6), squeeze=False)

    fig.suptitle(
        f"{src}  —  Clase: {cls}  —  Tie: misma prob máxima\n"
        "Curva de luz  |  Fase (mayor power)  |  Fase (menor power)",
        fontsize=12, fontweight="bold",
    )

    kw = dict(fmt=".", ms=1.5, lw=0, elinewidth=0.3, color="black", ecolor="silver", alpha=0.7)

    for i, t in enumerate(sample):
        lc    = t["lc"]
        p_hi  = float(t["row_hi"]["period"])
        p_lo  = float(t["row_lo"]["period"])
        pw_hi = float(t["row_hi"]["power"])
        pw_lo = float(t["row_lo"]["power"])
        hdr   = f"TIC {t['tic']}  S{t['sector']}  prob={t['prob']:.3f}"

        # ── col 0: curva de luz ──────────────────────────────────────────
        ax = axes[i, 0]
        ax.errorbar(lc["Time"], lc["flux"], yerr=lc["flux_err"], **kw)
        ax.set_title(hdr, fontsize=8)
        ax.tick_params(labelsize=7)
        ax.yaxis.set_major_locator(plt.MaxNLocator(3))
        ax.grid(True, alpha=0.2, lw=0.5)
        if i == n - 1:
            ax.set_xlabel("Tiempo [BTJD]", fontsize=8)

        # ── col 1: fase mayor power ──────────────────────────────────────
        ax = axes[i, 1]
        phi, f, fe = phase_fold(lc, p_hi)
        phi2 = np.concatenate([phi, phi + 1])
        ax.errorbar(phi2, np.concatenate([f, f]), yerr=np.concatenate([fe, fe]), **kw)
        ax.axvline(1.0, ls="--", color="gray", lw=0.6, alpha=0.5)
        ax.set_xlim(0, 2)
        ax.set_title(f"P = {p_hi:.5f} d   pw = {pw_hi:.4f}", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.yaxis.set_major_locator(plt.MaxNLocator(3))
        ax.grid(True, alpha=0.2, lw=0.5)
        if i == n - 1:
            ax.set_xlabel("Fase", fontsize=8)

        # ── col 2: fase menor power ──────────────────────────────────────
        ax = axes[i, 2]
        phi, f, fe = phase_fold(lc, p_lo)
        phi2 = np.concatenate([phi, phi + 1])
        ax.errorbar(phi2, np.concatenate([f, f]), yerr=np.concatenate([fe, fe]), **kw)
        ax.axvline(1.0, ls="--", color="gray", lw=0.6, alpha=0.5)
        ax.set_xlim(0, 2)
        ax.set_title(f"P = {p_lo:.5f} d   pw = {pw_lo:.4f}", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.yaxis.set_major_locator(plt.MaxNLocator(3))
        ax.grid(True, alpha=0.2, lw=0.5)
        if i == n - 1:
            ax.set_xlabel("Fase", fontsize=8)

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    pdf.savefig(fig, dpi=120)
    plt.close(fig)


# ── Main ─────────────────────────────────────────────────────────────────────
with PdfPages(OUTPUT) as pdf:
    for src, csv_path in SOURCES:
        print(f"\nFuente: {src}")
        df = pd.read_csv(csv_path)
        rng_src = np.random.default_rng(seed=42)
        for cls in CLASSES:
            ties = find_ties(df, cls)
            if len(ties) > N_EACH:
                idx = rng_src.choice(len(ties), N_EACH, replace=False)
                sample = [ties[i] for i in sorted(idx)]
            else:
                sample = ties
            print(f"  {cls}: {len(ties)} ties con LC → mostrando {len(sample)}")
            if sample:
                draw_page(pdf, sample, cls, src)
            else:
                print(f"  Sin casos para {cls} ({src}), omitiendo página.")

    d = pdf.infodict()
    d["Title"] = "Tie1: mayor vs menor power — ELL, Pulsating, E × ACF, LS"

print(f"\nPDF guardado: {OUTPUT}")
