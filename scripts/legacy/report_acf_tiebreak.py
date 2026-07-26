#!/usr/bin/env python3
"""
Genera un PDF comparando los dos candidatos empatados en probabilidad ACF:
  - Columna izquierda : fase con MAYOR power (selección actual del tiebreak)
  - Columna derecha   : fase con MENOR power (alternativa)

Busca grupos (TIC, sector) del catálogo ACF donde ≥2 peaks periódicos
comparten la probabilidad máxima.  Muestrea N_CASES al azar con LC disponible.
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

# ── Configuración ─────────────────────────────────────────────────────────────
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REPORTS_DIR = os.path.join(REPO_ROOT, "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)
ACF_CSV   = os.path.join(REPO_ROOT, "data/catalogos/CNN_RF_prediction_ACF_log.csv")
LC_DIR    = os.path.join(REPO_ROOT, "data/lightcurves")
OUTPUT    = os.path.join(REPORTS_DIR, "reporte_acf_tiebreak.pdf")
N_CASES   = 10
THRESHOLD = 0.8
NON_PERIODIC = {"No_periodo", "Rndm", "LPV"}
RNG       = np.random.default_rng(seed=42)

# ── Cargadores ────────────────────────────────────────────────────────────────
def load_lc(tic, sector):
    path = os.path.join(LC_DIR, f"{tic}_{sector}.csv")
    return pd.read_csv(path) if os.path.exists(path) else None


def phase_fold(lc, period):
    t = lc["Time"].values
    f = lc["flux"].values
    fe = lc["flux_err"].values
    phi = ((t - t[0]) % period) / period
    srt = np.argsort(phi)
    return phi[srt], f[srt], fe[srt]


# ── Buscar empates en ACF ─────────────────────────────────────────────────────
print("Cargando catálogo ACF...")
acf = pd.read_csv(ACF_CSV)
acf["cube_idx"] = acf.index

# Mismo pre-filtro que build_catalog_v2: solo peaks periódicos
periodic = acf[~acf["final_class"].isin(NON_PERIODIC)].copy()

ties = []

for (tic, sector), grp in periodic.groupby(["TIC", "sector_list"]):
    max_prob = grp["CNN+RF_max_prob"].max()
    candidates = grp[grp["CNN+RF_max_prob"] == max_prob]
    if len(candidates) < 2:
        continue  # sin empate

    row_hi = candidates.loc[candidates["power"].idxmax()]  # mayor power (selección actual)
    row_lo = candidates.loc[candidates["power"].idxmin()]  # menor power (alternativa)

    if row_hi.name == row_lo.name:
        continue  # mismo pico (todos tienen el mismo power)

    if row_hi["final_class"] == "ELL":
        continue

    ties.append({
        "tic":      int(tic),
        "sector":   int(sector),
        "prob":     max_prob,
        "cls":      row_hi["final_class"],
        "n_tied":   len(candidates),
        "row_hi":   row_hi,
        "row_lo":   row_lo,
        "above_thresh": max_prob >= THRESHOLD,
    })

print(f"Empates ACF encontrados: {len(ties)}")

# Filtrar a los que tienen LC disponible
ties_with_lc = [t for t in ties if load_lc(t["tic"], t["sector"]) is not None]
print(f"Con LC disponible:        {len(ties_with_lc)}")

if not ties_with_lc:
    raise SystemExit("No hay casos con LC disponible.")

# Muestrear N_CASES
sample = (ties_with_lc if len(ties_with_lc) <= N_CASES
          else [ties_with_lc[i] for i in RNG.choice(len(ties_with_lc), N_CASES, replace=False)])
sample.sort(key=lambda x: (x["tic"], x["sector"]))
print(f"Casos seleccionados:      {len(sample)}")

# ── Generar PDF ───────────────────────────────────────────────────────────────
with PdfPages(OUTPUT) as pdf:

    # ── Portada / índice ──────────────────────────────────────────────────────
    fig0, ax0 = plt.subplots(figsize=(11, 8.5))
    ax0.axis("off")
    title = (
        "Comparación de empates en probabilidad ACF\n"
        "Tiebreak: mayor power (izq.) vs menor power (der.)"
    )
    ax0.text(0.5, 0.80, title, ha="center", va="center",
             fontsize=16, fontweight="bold", transform=ax0.transAxes)

    summary_lines = [
        f"Empates totales en ACF (periódicos):  {len(ties)}",
        f"Con curva de luz disponible:           {len(ties_with_lc)}",
        f"Casos mostrados en este reporte:       {len(sample)}",
        "",
        f"Pre-filtro aplicado: excluye {NON_PERIODIC}",
        f"Umbral probabilidad: {THRESHOLD}",
        "",
        "Caso = (TIC, sector) donde ≥2 peaks ACF comparten la prob máxima.",
        "El tiebreak actual elige el de MAYOR power.",
    ]
    ax0.text(0.5, 0.45, "\n".join(summary_lines), ha="center", va="center",
             fontsize=11, family="monospace", transform=ax0.transAxes,
             bbox=dict(boxstyle="round,pad=0.6", fc="#eef2fb", ec="#aabde0", lw=1.5))

    # Tabla resumen de los casos
    tbl_data = [["#", "TIC", "Sector", "Clase", "Prob", "N tied",
                 "P hi-power [d]", "Power hi",
                 "P lo-power [d]", "Power lo"]]
    for k, t in enumerate(sample, 1):
        rhi, rlo = t["row_hi"], t["row_lo"]
        tbl_data.append([
            str(k),
            str(t["tic"]),
            f"S{t['sector']}",
            t["cls"],
            f"{t['prob']:.3f}",
            str(t["n_tied"]),
            f"{rhi['period']:.4f}",
            f"{rhi['power']:.4f}",
            f"{rlo['period']:.4f}",
            f"{rlo['power']:.4f}",
        ])

    tbl_ax = fig0.add_axes([0.03, 0.02, 0.94, 0.26])
    tbl_ax.axis("off")
    tbl = tbl_ax.table(cellText=tbl_data[1:], colLabels=tbl_data[0],
                       cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.35)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#4a6fa5")
            cell.set_text_props(color="white", fontweight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#f0f4fb")

    pdf.savefig(fig0, dpi=120)
    plt.close(fig0)

    # ── Una página por caso ───────────────────────────────────────────────────
    for k, t in enumerate(sample, 1):
        tic    = t["tic"]
        sector = t["sector"]
        rhi    = t["row_hi"]
        rlo    = t["row_lo"]
        lc     = load_lc(tic, sector)

        p_hi   = float(rhi["period"])
        p_lo   = float(rlo["period"])
        pw_hi  = float(rhi["power"])
        pw_lo  = float(rlo["power"])
        cls    = t["cls"]
        prob   = t["prob"]

        fig = plt.figure(figsize=(14, 9))
        fig.suptitle(
            f"Caso {k}/{len(sample)}  —  TIC {tic}   Sector S{sector}   "
            f"Clase: {cls}   Prob: {prob:.3f}   N empatados: {t['n_tied']}",
            fontsize=13, fontweight="bold", y=0.995,
        )

        gs = gridspec.GridSpec(
            2, 3,
            figure=fig,
            hspace=0.42, wspace=0.30,
            left=0.06, right=0.97,
            top=0.94, bottom=0.07,
            height_ratios=[1, 1.6],
        )

        # Fila 0: Curva de luz cruda (span las 3 columnas)
        ax_lc = fig.add_subplot(gs[0, :])
        if lc is not None:
            ax_lc.errorbar(lc["Time"], lc["flux"], yerr=lc["flux_err"],
                           fmt=".", ms=1.5, lw=0, elinewidth=0.4,
                           color="steelblue", ecolor="lightgray", alpha=0.8)
            ax_lc.set_xlabel("Tiempo [BTJD]", fontsize=9)
            ax_lc.set_ylabel("Flujo [e⁻/s]", fontsize=9)
            ax_lc.set_title(f"Curva de luz  —  TIC {tic}  S{sector}", fontsize=10)
            ax_lc.tick_params(labelsize=8)
            ax_lc.yaxis.set_major_locator(plt.MaxNLocator(5))
            ax_lc.grid(True, alpha=0.25)
        else:
            ax_lc.text(0.5, 0.5, "LC no disponible",
                       ha="center", va="center", fontsize=12, color="gray",
                       transform=ax_lc.transAxes)
            ax_lc.set_axis_off()

        # Fila 1, col 0: fase con MAYOR power (selección actual)
        ax_hi = fig.add_subplot(gs[1, 0])
        ax_hi.set_facecolor("#e8f4e8")

        if lc is not None and p_hi > 0:
            phi, f, fe = phase_fold(lc, p_hi)
            phi2 = np.concatenate([phi, phi + 1])
            f2   = np.concatenate([f, f])
            fe2  = np.concatenate([fe, fe])
            ax_hi.errorbar(phi2, f2, yerr=fe2,
                           fmt=".", ms=1.5, lw=0, elinewidth=0.4,
                           color="#2ca02c", ecolor="lightgray", alpha=0.8)
            ax_hi.set_xlim(0, 2)
            ax_hi.axvline(1.0, ls="--", color="gray", lw=0.7, alpha=0.5)
            ax_hi.yaxis.set_major_locator(plt.MaxNLocator(4))
        else:
            ax_hi.text(0.5, 0.5, "Sin LC", ha="center", va="center",
                       fontsize=10, color="gray", transform=ax_hi.transAxes)

        ax_hi.set_xlabel("Fase", fontsize=9)
        ax_hi.set_ylabel("Flujo [e⁻/s]", fontsize=9)
        ax_hi.set_title(
            f"MAYOR power  ← selección actual\nP = {p_hi:.5f} d   Power = {pw_hi:.4f}",
            fontsize=9, fontweight="bold", color="#1a6e1a",
        )
        ax_hi.tick_params(labelsize=8)
        ax_hi.grid(True, alpha=0.25)

        # Fila 1, col 1: fase con MENOR power (alternativa)
        ax_lo = fig.add_subplot(gs[1, 1])
        ax_lo.set_facecolor("#fdf3e3")

        if lc is not None and p_lo > 0:
            phi, f, fe = phase_fold(lc, p_lo)
            phi2 = np.concatenate([phi, phi + 1])
            f2   = np.concatenate([f, f])
            fe2  = np.concatenate([fe, fe])
            ax_lo.errorbar(phi2, f2, yerr=fe2,
                           fmt=".", ms=1.5, lw=0, elinewidth=0.4,
                           color="#d55e00", ecolor="lightgray", alpha=0.8)
            ax_lo.set_xlim(0, 2)
            ax_lo.axvline(1.0, ls="--", color="gray", lw=0.7, alpha=0.5)
            ax_lo.yaxis.set_major_locator(plt.MaxNLocator(4))
        else:
            ax_lo.text(0.5, 0.5, "Sin LC", ha="center", va="center",
                       fontsize=10, color="gray", transform=ax_lo.transAxes)

        ax_lo.set_xlabel("Fase", fontsize=9)
        ax_lo.set_ylabel("Flujo [e⁻/s]", fontsize=9)
        ax_lo.set_title(
            f"MENOR power  ← alternativa\nP = {p_lo:.5f} d   Power = {pw_lo:.4f}",
            fontsize=9, fontweight="bold", color="#a04000",
        )
        ax_lo.tick_params(labelsize=8)
        ax_lo.grid(True, alpha=0.25)

        # Fila 1, col 2: info del empate
        ax_info = fig.add_subplot(gs[1, 2])
        ax_info.axis("off")

        info_lines = [
            f"TIC:       {tic}",
            f"Sector:    S{sector}",
            f"Clase:     {cls}",
            f"Prob:      {prob:.4f}",
            f"N empatados: {t['n_tied']}",
            "",
            "MAYOR power (actual):",
            f"  P  = {p_hi:.5f} d",
            f"  pw = {pw_hi:.4f}",
            "",
            "MENOR power (alternativa):",
            f"  P  = {p_lo:.5f} d",
            f"  pw = {pw_lo:.4f}",
            "",
            f"Δ Periodo = {abs(p_hi - p_lo):.5f} d",
            f"Δ Power   = {abs(pw_hi - pw_lo):.4f}",
            "",
            ("★ Sobre umbral (0.8)" if t["above_thresh"]
             else "  Bajo umbral (Unconstrained)"),
        ]

        ax_info.text(0.05, 0.95, "\n".join(info_lines),
                     ha="left", va="top", fontsize=9,
                     family="monospace", linespacing=1.5,
                     transform=ax_info.transAxes,
                     bbox=dict(boxstyle="round,pad=0.5",
                               fc="#f0f4fb", ec="#aabde0", lw=1.2))

        pdf.savefig(fig, dpi=120)
        plt.close(fig)

    d = pdf.infodict()
    d["Title"]   = "Comparación tiebreak ACF — mayor vs menor power"
    d["Subject"] = f"N={len(sample)} casos muestreados de {len(ties)} empates"

print(f"\nPDF generado: {OUTPUT}")
