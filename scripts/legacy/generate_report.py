#!/usr/bin/env python3
"""
Genera un PDF de reporte de variabilidad estelar.

Estructura del PDF:
  Página 1      : Tabla resumen global de todas las estrellas del subcatálogo.
  Páginas 2..N  : Una página por estrella (LC arriba + fase abajo).
                  Si hay muchos candidatos, se agregan páginas extra para esa estrella.

Uso:
    python generate_report.py <subcatalogo.csv> [-o reporte.pdf] [--prob 0.9]
"""

import sys, os, glob, argparse, warnings
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
        "cube": os.path.join(DATA_DIR, "cubos", "Github_MassiveXTessV8_LC_Aperture_Vsx_LS_NO_sigma_cliping_min_max.npy"),
        "kind": "LS",
    },
    "LS  |  HistNorm log": {
        "csv":  os.path.join(DATA_DIR, "catalogos", "CNN_RF_prediction_LS_log.csv"),
        "cube": os.path.join(DATA_DIR, "cubos", "Github_MassiveXTessV8_LC_Aperture_Vsx_LS_NO_sigma_cliping_log.npy"),
        "kind": "LS",
    },
    "ACF |  HistNorm min-max": {
        "csv":  os.path.join(DATA_DIR, "catalogos", "CNN_RF_prediction_ACF_min_max.csv"),
        "cube": os.path.join(DATA_DIR, "cubos", "Github_MassiveXTessV8_LC_Aperture_Vsx_ACF_NO_sigma_cliping_min_max.npy"),
        "kind": "ACF",
    },
    "ACF |  HistNorm log": {
        "csv":  os.path.join(DATA_DIR, "catalogos", "CNN_RF_prediction_ACF_log.csv"),
        "cube": os.path.join(DATA_DIR, "cubos", "Github_MassiveXTessV8_LC_Aperture_Vsx_ACF_NO_sigma_cliping_log.npy"),
        "kind": "ACF",
    },
}

CLASS_COLS   = ["CNN+RF_ELL", "CNN+RF_M", "CNN+RF_CEP", "CNN+RF_DST",
                "CNN+RF_E",   "CNN+RF_LPV", "CNN+RF_RR", "CNN+RF_Rndm"]
CLASS_LABELS = ["ELL", "M", "CEP", "DST", "E", "LPV", "RR", "Rndm"]
CLASS_COLORS = {
    "ELL": "#4C72B0", "M": "#DD8452", "CEP": "#55A868", "DST": "#C44E52",
    "E":   "#8172B2", "LPV": "#937860", "RR": "#DA8BC3", "Rndm": "#8C8C8C",
}

PROB_THRESH = 0.9   # sobreescrito por --prob
MAX_CAND    = 20    # máx. candidatos (fase) por TIC; sobreescrito por --max-cand


# ── Cargadores ─────────────────────────────────────────────────────────────────
def load_catalog(csv_path):
    if not os.path.exists(csv_path):
        return None
    df = pd.read_csv(csv_path)
    avail = [c for c in CLASS_COLS if c in df.columns]
    if avail:
        df["CNN+RF_max_prob"] = df[avail].max(axis=1)
        df["final_class"]     = df[avail].idxmax(axis=1).str.replace("CNN+RF_", "", regex=False)
    return df


def load_cube(cube_path):
    if not os.path.exists(cube_path):
        return None
    return np.load(cube_path, mmap_mode="r")


def available_sectors(tic):
    files = glob.glob(os.path.join(LC_DIR, f"{tic}_*.csv"))
    return sorted(int(os.path.basename(f).split("_")[1].replace(".csv", "")) for f in files)


def load_lc(tic, sector):
    path = os.path.join(LC_DIR, f"{tic}_{sector}.csv")
    return pd.read_csv(path) if os.path.exists(path) else None


def build_cat_data(tic, loaded):
    """Devuelve lista de dicts con los datos de cada catálogo filtrados al TIC."""
    result = []
    for name, data in loaded.items():
        cat  = data["cat"]
        cube = data["cube"]
        sdf  = None
        if cat is not None:
            tmp = cat[cat["TIC"] == tic].copy()
            if len(tmp) > 0:
                sdf = tmp
        result.append({"name": name, "kind": data["kind"], "star_df": sdf, "cube": cube})
    return result


def get_candidates(cat_data):
    """Todos los períodos con prob > PROB_THRESH y clase != Rndm, ordenados."""
    entries = []
    for entry in cat_data:
        sdf  = entry["star_df"]
        if sdf is None:
            continue
        cands = sdf[
            (sdf["CNN+RF_max_prob"] > PROB_THRESH) &
            (sdf["final_class"] != "Rndm")
        ].sort_values("CNN+RF_max_prob", ascending=False)
        for _, row in cands.iterrows():
            entries.append({
                "name": entry["name"], "kind": entry["kind"],
                "row": row, "cube": entry["cube"], "cube_idx": int(row.name),
            })
    return entries


# ══════════════════════════════════════════════════════════════════════════════
# PÁGINA 1: Tabla global  –  formato paper
#
# Jerarquía de filas:  TIC  →  Sector  →  Catálogo (6 por sector)
# Columnas: TIC | Sector | Periodograma + hist inline | N | N>thresh | Mejor período
# ══════════════════════════════════════════════════════════════════════════════

def _flatten_rows(all_stars):
    """
    Lista plana de filas para la tabla (solo type='cat').
    Cada fila incluye lc_sectors con la lista de sectores del TIC,
    para que el ax abarcador de col-0 pueda dibujar las LCs.
    """
    rows = []
    for star in all_stars:
        tic      = star["tic"]
        sectors  = star["sectors"]
        cat_data = star["cat_data"]
        if not sectors:
            cat_secs = set()
            for e in cat_data:
                if e["star_df"] is not None:
                    cat_secs.update(
                        e["star_df"]["sector_list"].dropna().astype(int).unique()
                    )
            sectors = sorted(cat_secs) or [None]
        for si, sec in enumerate(sectors):
            for ci, entry in enumerate(cat_data):
                sdf_full = entry["star_df"]
                if sdf_full is not None and sec is not None:
                    sdf = sdf_full[sdf_full["sector_list"] == sec].copy()
                    sdf = sdf if len(sdf) > 0 else None
                else:
                    sdf = sdf_full
                rows.append(dict(
                    type        = "cat",
                    tic         = tic,
                    sector      = sec,
                    cat_name    = entry["name"],
                    sdf         = sdf,
                    lc_sectors  = sectors,       # sectores del TIC (para dibujar LC)
                    is_first_tic= (si == 0 and ci == 0),
                    is_first_sec= (ci == 0),
                ))
    return rows


def _style_ax(ax, fc="white"):
    """Axes de celda: sin ejes, fondo configurable."""
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_facecolor(fc)


def _border(ax, top_lw=0, top_col="black", bot_lw=0, bot_col="#cccccc",
            left_lw=0, left_col="#888888", right_lw=0, right_col="#888888"):
    """Dibuja bordes de celda usando ax.plot en coordenadas de datos (xlim=ylim=[0,1])."""
    kw = dict(clip_on=False, solid_capstyle="butt")
    if top_lw:
        ax.plot([0, 1], [1, 1], color=top_col,   lw=top_lw,   **kw)
    if bot_lw:
        ax.plot([0, 1], [0, 0], color=bot_col,   lw=bot_lw,   **kw)
    if left_lw:
        ax.plot([0, 0], [0, 1], color=left_col,  lw=left_lw,  **kw)
    if right_lw:
        ax.plot([1, 1], [0, 1], color=right_col, lw=right_lw, **kw)


def pages_global_table(pdf, all_stars):
    """
    Página(s) 1: tabla estilo paper (GridSpec).
    Columnas: TIC (con LC incrustada) | Sector | Periodograma | N | N>thresh | Mejor período
    Col 0 abarca todas las filas del TIC y muestra el número arriba + LCs abajo.
    """
    ROWS_PER_PAGE = 24   # filas de catálogo por página
    W      = [4.0, 0.6, 1.1, 0.7, 0.7, 2.0]
    PAGE_W = 17.0
    ROW_H  = 0.72
    HEAD_H = 0.55
    NCOLS  = len(W)

    HDR = [
        "TIC", "Sector",
        "Periodograma /\nNormalización",
        "N\nperíodos", f"N\nprob>{PROB_THRESH}",
        f"Período  [clase]  (prob)   —   prob>{PROB_THRESH},  ≠ Rndm",
    ]

    flat = _flatten_rows(all_stars)
    if not flat:
        return

    shown_tics = set()   # TICs cuyo número ya fue escrito en alguna página

    for page_start in range(0, len(flat), ROWS_PER_PAGE):
        batch = flat[page_start : page_start + ROWS_PER_PAGE]
        n     = len(batch)

        hr_data = [1.0] * n
        fig_h   = 0.40 + HEAD_H + sum(h * ROW_H for h in hr_data) + 0.10
        fig     = plt.figure(figsize=(PAGE_W, fig_h))

        fig.suptitle(
            f"Tabla resumen  —  prob > {PROB_THRESH}",
            fontsize=11, fontweight="bold", y=0.998,
        )

        hr = [HEAD_H / ROW_H] + hr_data
        gs = gridspec.GridSpec(
            n + 1, NCOLS, figure=fig,
            left=0.01, right=0.99,
            top=0.965, bottom=0.005,
            hspace=0, wspace=0,
            width_ratios=W, height_ratios=hr,
        )

        # ── Cabecera ─────────────────────────────────────────────────────────
        for j, label in enumerate(HDR):
            ax = fig.add_subplot(gs[0, j])
            _style_ax(ax, fc="#dde3ef")
            ax.text(0.5, 0.5, label, ha="center", va="center",
                    fontsize=8, fontweight="bold", multialignment="center")
            _border(ax,
                    top_lw=1.5, top_col="black",
                    bot_lw=1.3, bot_col="black",
                    left_lw=(1.3 if j == 0 else 0.5), left_col="black" if j == 0 else "#888888",
                    right_lw=(1.3 if j == NCOLS-1 else 0.5), right_col="black" if j == NCOLS-1 else "#888888")

        # ── Identificar grupos de TIC en este batch ───────────────────────────
        tic_order  = []   # TICs en orden de aparición
        tic_ranges = {}   # tic -> [first_idx, last_idx] en batch
        for i, rd in enumerate(batch):
            tic = rd["tic"]
            if tic not in tic_ranges:
                tic_order.append(tic)
                tic_ranges[tic] = [i, i]
            else:
                tic_ranges[tic][1] = i

        tic_parity = {}
        par = 0
        for tic in tic_order:
            tic_parity[tic] = par
            par = 1 - par

        # ── Col 0: un ax por (TIC, sector) — alineado con cols 1–5 ─────────
        # Agrupamos por (tic, sector) manteniendo el orden de aparición
        tic_sec_seen  = {}   # (tic,sec) -> [first_idx, last_idx]
        tic_first_sec = {}   # tic -> primer sector visto
        for i, rd in enumerate(batch):
            tic, sec = rd["tic"], rd["sector"]
            key = (tic, sec)
            if key not in tic_sec_seen:
                tic_sec_seen[key] = [i, i]
                if tic not in tic_first_sec:
                    tic_first_sec[tic] = sec
            else:
                tic_sec_seen[key][1] = i

        for (tic, sec), (ri_start, ri_end) in tic_sec_seen.items():
            gs_r0 = ri_start + 1
            gs_r1 = ri_end + 1
            # Mostrar número TIC solo la primera vez que aparece en todo el PDF
            is_first_sec_of_tic = tic not in shown_tics
            if is_first_sec_of_tic:
                shown_tics.add(tic)

            ax_s = fig.add_subplot(gs[gs_r0 : gs_r1 + 1, 0])
            fc   = "#f3f5fb" if tic_parity[tic] else "white"
            _style_ax(ax_s, fc=fc)

            # Borde: mismo grosor que cols 1–5 para alinear con separadores
            if is_first_sec_of_tic:
                top_lw  = 1.5 if ri_start == 0 else 1.2
                top_col = "black"
            else:
                top_lw, top_col = 0.7, "#444444"
            is_last_row = (ri_end == n - 1)
            _border(ax_s,
                    top_lw=top_lw,   top_col=top_col,
                    bot_lw=(1.3 if is_last_row else 0), bot_col="black",
                    left_lw=1.3,     left_col="black",
                    right_lw=0.4,    right_col="#aaaaaa")

            # TIC number solo en el primer sector, arriba a la izquierda
            if is_first_sec_of_tic:
                ax_s.text(0.5, 0.97, str(tic), ha="center", va="top",
                          fontsize=8, fontweight="bold")
                lc_top = 0.80   # deja 20% para el número TIC
            else:
                lc_top = 0.96

            # LC incrustada — márgenes amplios para que los tick labels no desborden
            lc    = load_lc(tic, sec) if sec is not None else None
            inset = ax_s.inset_axes([0.22, 0.14, 0.74, lc_top - 0.18])
            if lc is not None:
                inset.errorbar(
                    lc["Time"], lc["flux"], yerr=lc["flux_err"],
                    fmt=".", ms=1.0, lw=0, elinewidth=0.3,
                    color="steelblue", ecolor="lightgray", alpha=0.8,
                )
                inset.set_xlabel("Días [BTJD]", fontsize=6.5)
                inset.set_ylabel("Flujo [e⁻/s]", fontsize=6.5)
                inset.tick_params(labelsize=6, pad=2)
                inset.yaxis.set_major_locator(plt.MaxNLocator(4))
                inset.xaxis.set_major_locator(plt.MaxNLocator(5))
                plt.setp(inset.get_xticklabels(), rotation=20, ha="right")
            else:
                inset.text(0.5, 0.5, "Sin datos", ha="center", va="center",
                           fontsize=7, color="gray")
                inset.set_axis_off()

        # ── Filas de datos: cols 1–5 ──────────────────────────────────────────
        for i, rd in enumerate(batch):
            ri      = i + 1
            is_last = (i == n - 1)
            tic     = rd["tic"]
            fc      = "#f3f5fb" if tic_parity[tic] else "white"

            # Fila de catálogo
            sec      = rd["sector"]
            cat_name = rd["cat_name"]
            sdf      = rd["sdf"]
            is_ftic  = rd["is_first_tic"]
            is_fsec  = rd["is_first_sec"]

            if is_ftic and i > 0:
                top_lw, top_col = 1.2, "black"
            elif is_fsec and i > 0:
                top_lw, top_col = 0.7, "#444444"
            else:
                top_lw, top_col = 0.3, "#cccccc"

            n_total = len(sdf) if sdf is not None else 0
            if sdf is not None and n_total > 0:
                high   = sdf[sdf["CNN+RF_max_prob"] > PROB_THRESH]
                n_high = len(high)
                cands  = high[high["final_class"] != "Rndm"].sort_values(
                    "CNN+RF_max_prob", ascending=False)
            else:
                n_high, cands = 0, pd.DataFrame()

            def make_cell(col, _fc=fc, _top_lw=top_lw, _top_col=top_col, _last=is_last):
                ax = fig.add_subplot(gs[ri, col])
                _style_ax(ax, fc=_fc)
                _border(ax,
                        top_lw=_top_lw, top_col=_top_col,
                        bot_lw=(1.3 if _last else 0), bot_col="black",
                        left_lw=0.4,  left_col="#aaaaaa",
                        right_lw=(1.3 if col == NCOLS - 1 else 0.4),
                        right_col="black" if col == NCOLS - 1 else "#aaaaaa")
                return ax

            # Col 1: Sector
            ax1 = make_cell(1)
            if is_fsec and sec is not None:
                ax1.text(0.5, 0.5, f"S{sec}", ha="center", va="center", fontsize=9)

            # Col 2: Nombre catálogo
            ax2 = make_cell(2)
            ax2.text(0.05, 0.5, cat_name, ha="left", va="center",
                     fontsize=7.5, fontweight="bold")

            # Col 3: N total
            ax3 = make_cell(3)
            ax3.text(0.5, 0.5, str(n_total) if n_total > 0 else "—",
                     ha="center", va="center", fontsize=10,
                     fontweight="bold" if n_total > 0 else "normal",
                     color="black" if n_total > 0 else "gray")

            # Col 4: N > umbral
            ax4 = make_cell(4)
            ax4.text(0.5, 0.5, str(n_high) if n_total > 0 else "—",
                     ha="center", va="center", fontsize=10,
                     fontweight="bold" if n_high > 0 else "normal",
                     color="#2ca02c" if n_high > 0 else "gray")

            # Col 5: Mejores períodos
            ax5 = make_cell(5)
            if len(cands) > 0:
                lines = [
                    f"P={r['period']:.4f} d  [{r['final_class']}]  ({r['CNN+RF_max_prob']:.3f})"
                    for _, r in cands.head(3).iterrows()
                ]
                if len(cands) > 3:
                    lines.append(f"  +{len(cands)-3} más")
                ax5.text(0.03, 0.5, "\n".join(lines), ha="left", va="center",
                         fontsize=7.5, color="#1a6e1a",
                         family="monospace", linespacing=1.35)
            else:
                ax5.text(0.5, 0.5, "—", ha="center", va="center",
                         fontsize=10, color="gray")

        pdf.savefig(fig, dpi=120)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Genera PDF de reporte de variabilidad estelar."
    )
    parser.add_argument("subcatalog",
                        help="CSV de subcatálogo con columna 'TIC'")
    parser.add_argument("-o", "--output", default=os.path.join(REPO_ROOT, "reports", "reporte_variabilidad.pdf"),
                        help="PDF de salida (default: reporte_variabilidad.pdf)")
    parser.add_argument("--prob", type=float, default=0.9,
                        help="Umbral de probabilidad (default: 0.9)")
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
    loaded = {}
    for name, cfg in CATALOGS.items():
        cat  = load_catalog(cfg["csv"])
        cube = load_cube(cfg["cube"])
        loaded[name] = {"cat": cat, "cube": cube, "kind": cfg["kind"]}
        print(f"  {name}: {'OK' if cat is not None else 'no encontrado'}")

    # Pre-computar datos por estrella
    all_stars = []
    for tic in tic_list:
        sectors  = available_sectors(tic)
        cat_data = build_cat_data(tic, loaded)
        all_stars.append({"tic": tic, "sectors": sectors, "cat_data": cat_data})

    with PdfPages(args.output) as pdf:
        print("Generando tabla resumen...")
        pages_global_table(pdf, all_stars)

        d = pdf.infodict()
        d["Title"]   = "Reporte Variabilidad Estrellas Masivas"
        d["Subject"] = f"Subcatalogo: {args.subcatalog}  |  prob > {PROB_THRESH}"

    print(f"\nPDF generado: {args.output}")


if __name__ == "__main__":
    main()
