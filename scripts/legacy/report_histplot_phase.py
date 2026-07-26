#!/usr/bin/env python3
"""
Genera un PDF con histogramas (cubos .npy) y curvas de luz plegadas en fase,
para todos los períodos con probabilidad > PROB_THRESH (default 0.8) y clase != Rndm.

Layout por página (una estrella por página)
───────────────────────────────────────────
  Fila 0 : imagen del histograma del cubo para cada normalización
  Fila 1 : curva de luz plegada en fase con el mejor período de cada catálogo

  Columnas: LS min-max | LS log | ACF min-max | ACF log

Uso:
    python report_histplot_phase.py <subcatalogo.csv> [-o reporte.pdf] [--prob 0.8]
"""

import sys, os, argparse, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages

warnings.filterwarnings("ignore")

# ── Rutas ──────────────────────────────────────────────────────────────────────
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(REPO_ROOT, "data")
LC_DIR   = os.path.join(DATA_DIR, "lightcurves")

CATALOGS = {
    "LS  |  HistNorm min-max": {
        "csv":  os.path.join(DATA_DIR, "catalogos", "CNN_RF_prediction_LS_min_max.csv"),
        "cube": os.path.join(DATA_DIR, "cubos",
                             "Github_MassiveXTessV8_LC_Aperture_Vsx_LS_NO_sigma_cliping_min_max.npy"),
        "kind": "LS",
    },
    "LS  |  HistNorm log": {
        "csv":  os.path.join(DATA_DIR, "catalogos", "CNN_RF_prediction_LS_log.csv"),
        "cube": os.path.join(DATA_DIR, "cubos",
                             "Github_MassiveXTessV8_LC_Aperture_Vsx_LS_NO_sigma_cliping_log.npy"),
        "kind": "LS",
    },
    "ACF |  HistNorm min-max": {
        "csv":  os.path.join(DATA_DIR, "catalogos", "CNN_RF_prediction_ACF_min_max.csv"),
        "cube": os.path.join(DATA_DIR, "cubos",
                             "Github_MassiveXTessV8_LC_Aperture_Vsx_ACF_NO_sigma_cliping_min_max.npy"),
        "kind": "ACF",
    },
    "ACF |  HistNorm log": {
        "csv":  os.path.join(DATA_DIR, "catalogos", "CNN_RF_prediction_ACF_log.csv"),
        "cube": os.path.join(DATA_DIR, "cubos",
                             "Github_MassiveXTessV8_LC_Aperture_Vsx_ACF_NO_sigma_cliping_log.npy"),
        "kind": "ACF",
    },
}

CLASS_COLS   = ["CNN+RF_ELL", "CNN+RF_M", "CNN+RF_CEP", "CNN+RF_DST",
                "CNN+RF_E",   "CNN+RF_LPV", "CNN+RF_RR", "CNN+RF_Rndm"]
CLASS_COLORS = {
    "ELL": "#4C72B0", "M": "#DD8452", "CEP": "#55A868", "DST": "#C44E52",
    "E":   "#8172B2", "LPV": "#937860", "RR": "#DA8BC3", "Rndm": "#8C8C8C",
}

PROB_THRESH = 0.8


# ── Cargadores ─────────────────────────────────────────────────────────────────
def load_catalog(csv_path):
    if not os.path.exists(csv_path):
        return None
    df = pd.read_csv(csv_path)
    avail = [c for c in CLASS_COLS if c in df.columns]
    if avail:
        df["CNN+RF_max_prob"] = df[avail].max(axis=1)
        df["final_class"]     = df[avail].idxmax(axis=1).str.replace(
            "CNN+RF_", "", regex=False)
    return df


def load_cube(cube_path):
    if not os.path.exists(cube_path):
        return None
    return np.load(cube_path, mmap_mode="r")


def load_lc(tic, sector):
    path = os.path.join(LC_DIR, f"{tic}_{sector}.csv")
    return pd.read_csv(path) if os.path.exists(path) else None


def build_loaded(tic_list):
    """Carga catálogos y cubos; filtra cada catálogo a los TICs de interés."""
    loaded = {}
    for name, cfg in CATALOGS.items():
        cat  = load_catalog(cfg["csv"])
        cube = load_cube(cfg["cube"])
        if cat is not None:
            cat = cat[cat["TIC"].isin(tic_list)].copy()
        loaded[name] = {"cat": cat, "cube": cube, "kind": cfg["kind"]}
        print(f"  {name}: {'OK' if cat is not None else 'no encontrado'}")
    return loaded


# ── Helpers de plot ────────────────────────────────────────────────────────────
def _get_sector(row):
    for col in ("sector_list", "sector"):
        if col in row.index and pd.notna(row[col]):
            try:
                return int(float(row[col]))
            except (ValueError, TypeError):
                pass
    return None


def _plot_hist(ax, cube, idx, period, cls, prob, norm_lbl):
    ax.set_title(
        f"{norm_lbl}\nP={period:.4f} d  [{cls}]  ({prob:.3f})",
        fontsize=7, linespacing=1.4,
    )
    if cube is not None and idx < len(cube):
        img = cube[idx]
        ax.imshow(np.rot90(img, k=2), origin="lower", aspect="auto",
                  cmap="viridis", interpolation="nearest")
        ax.set_xlabel("Período (bins)", fontsize=7)
        ax.set_ylabel("Potencia (bins)", fontsize=7)
        ax.tick_params(labelsize=6)
    else:
        ax.text(0.5, 0.5, "Cubo no disponible",
                ha="center", va="center", fontsize=8, color="gray",
                transform=ax.transAxes)
        ax.set_xticks([]); ax.set_yticks([])


def _plot_phase(ax, tic, sector, period, cls):
    sec_lbl = f"S{sector}" if sector is not None else "?"
    ax.set_title(f"Fase  ({sec_lbl})", fontsize=7)
    lc = load_lc(tic, sector) if sector is not None else None
    if lc is not None and period > 0:
        t   = lc["Time"].values
        f   = lc["flux"].values
        fe  = lc["flux_err"].values
        phi = ((t - t[0]) % period) / period
        srt = np.argsort(phi)
        ax.errorbar(phi[srt], f[srt], yerr=fe[srt],
                    fmt=".", ms=2.0, lw=0, elinewidth=0.5,
                    color=CLASS_COLORS.get(cls, "steelblue"),
                    ecolor="lightgray", alpha=0.8)
        ax.set_xlabel("Fase", fontsize=7)
        ax.set_ylabel("Flujo [e⁻/s]", fontsize=7)
        ax.set_xlim(0, 1)
        ax.tick_params(labelsize=6)
        ax.yaxis.set_major_locator(plt.MaxNLocator(4))
    else:
        ax.text(0.5, 0.5, "Sin LC",
                ha="center", va="center", fontsize=8, color="gray",
                transform=ax.transAxes)
        ax.set_xticks([]); ax.set_yticks([])


# ── Función principal de páginas ───────────────────────────────────────────────
def pages_histplot_phase(pdf, tic_list, loaded, prob_thresh=0.8):
    """
    Una página por estrella.

    Layout
    ──────
    Agrupado por catálogo.  Para cada catálogo con candidatos:
      • Fila header  : nombre del catálogo/normalización (banda de color)
      • Fila por cand: [hist]  [fase]  — ordenados mayor → menor prob

    Candidatos: prob > prob_thresh, clase != Rndm.
    """
    CAT_NAMES  = list(CATALOGS.keys())
    COL_W      = 4.2   # ancho por columna (hist | fase)
    ROW_H      = 3.4   # alto por fila de candidato
    HDR_H      = 0.32  # alto relativo del header de catálogo (fracción de ROW_H)
    CAT_COLORS = ["#dde3ef", "#e8f4e8", "#fdf3e3", "#f5e6ef"]  # uno por catálogo

    for tic in tic_list:
        # ── Todos los candidatos por catálogo ──────────────────────────────
        all_cands = {}
        for name in CAT_NAMES:
            data = loaded[name]
            cat  = data["cat"]
            cube = data["cube"]
            if cat is None:
                all_cands[name] = []
                continue
            sdf   = cat[cat["TIC"] == tic].copy()
            cands = sdf[
                (sdf["CNN+RF_max_prob"] > prob_thresh) &
                (sdf["final_class"] != "Rndm")
            ].sort_values("CNN+RF_max_prob", ascending=False)
            all_cands[name] = [
                {"row": row, "cube": cube, "cube_idx": int(row.name)}
                for _, row in cands.iterrows()
            ]

        # Catálogos con al menos un candidato
        active = [(name, all_cands[name])
                  for name in CAT_NAMES if len(all_cands[name]) > 0]
        if not active:
            continue

        # ── Construir estructura de filas ──────────────────────────────────
        # row_struct: lista de ("header", cat_name, color) | ("cand", cat_name, cand)
        row_struct = []
        for name, cands in active:
            row_struct.append(("header", name, None))
            for cand in cands:
                row_struct.append(("cand", name, cand))

        n_rows  = len(row_struct)
        h_ratios = [HDR_H if r[0] == "header" else 1.0 for r in row_struct]
        fig_h   = sum(h_ratios) * ROW_H + 0.5

        fig = plt.figure(figsize=(COL_W * 2, fig_h))
        fig.suptitle(
            f"TIC {tic}  —  prob > {prob_thresh},  clase ≠ Rndm",
            fontsize=11, fontweight="bold", y=0.998,
        )

        gs = gridspec.GridSpec(
            n_rows, 2,
            figure=fig,
            height_ratios=h_ratios,
            hspace=0.45, wspace=0.25,
            left=0.07, right=0.97,
            top=0.97, bottom=0.02,
        )

        cat_color = {name: CAT_COLORS[i % len(CAT_COLORS)]
                     for i, (name, _) in enumerate(active)}

        for ri, entry in enumerate(row_struct):
            kind = entry[0]
            name = entry[1]
            cand = entry[2]
            color = cat_color[name]

            if kind == "header":
                ax = fig.add_subplot(gs[ri, :])   # span ambas columnas
                ax.set_facecolor(color)
                for sp in ax.spines.values():
                    sp.set_visible(False)
                ax.set_xticks([]); ax.set_yticks([])
                ax.text(0.5, 0.5, name.strip(),
                        ha="center", va="center",
                        fontsize=9, fontweight="bold",
                        transform=ax.transAxes)

            else:  # "cand"
                ax_hist  = fig.add_subplot(gs[ri, 0])
                ax_phase = fig.add_subplot(gs[ri, 1])

                row    = cand["row"]
                cube   = cand["cube"]
                idx    = cand["cube_idx"]
                period = float(row["period"])
                cls    = row["final_class"]
                prob   = float(row["CNN+RF_max_prob"])
                sector = _get_sector(row)

                _plot_hist(ax_hist, cube, idx, period, cls, prob, norm_lbl="")
                _plot_phase(ax_phase, tic, sector, period, cls)

                # Fondo sutil de catálogo en las celdas
                for ax in (ax_hist, ax_phase):
                    ax.set_facecolor(color + "55" if len(color) == 7 else color)

        pdf.savefig(fig, dpi=120)
        plt.close(fig)


# ── CLI ────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Genera PDF con histogramas y curvas en fase (prob > umbral, ≠ Rndm)."
    )
    parser.add_argument("subcatalog",
                        help="CSV de subcatálogo con columna 'TIC'")
    parser.add_argument("-o", "--output", default=os.path.join(REPO_ROOT, "reports", "reporte_histplot_phase.pdf"),
                        help="PDF de salida (default: reporte_histplot_phase.pdf)")
    parser.add_argument("--prob", type=float, default=0.8,
                        help="Umbral de probabilidad (default: 0.8)")
    args = parser.parse_args()

    global PROB_THRESH
    PROB_THRESH = args.prob

    if not os.path.exists(args.subcatalog):
        sys.exit(f"Error: no se encuentra '{args.subcatalog}'")
    sub = pd.read_csv(args.subcatalog)
    if "TIC" not in sub.columns:
        sys.exit("Error: el CSV debe tener una columna 'TIC'")
    tic_list = sub["TIC"].dropna().astype(int).unique().tolist()
    print(f"TICs a procesar: {len(tic_list)}")

    print("Cargando catálogos y cubos...")
    loaded = build_loaded(tic_list)

    with PdfPages(args.output) as pdf:
        print("Generando páginas histograma + fase...")
        pages_histplot_phase(pdf, tic_list, loaded, prob_thresh=args.prob)

        d = pdf.infodict()
        d["Title"]   = "Histogramas y Fase — Variabilidad Estrellas Masivas"
        d["Subject"] = f"Subcatalogo: {args.subcatalog}  |  prob > {args.prob}"

    print(f"\nPDF generado: {args.output}")


if __name__ == "__main__":
    main()
