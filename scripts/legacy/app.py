import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os

st.set_page_config(page_title="Massive Star Variability Explorer", layout="wide")

# ── Paths ──────────────────────────────────────────────────────────────────────
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(REPO_ROOT, "data")
LC_DIR   = os.path.join(DATA_DIR, "lightcurves")

_UNIFIED_CUBES = {
    "cube_ls":  os.path.join(DATA_DIR, "cubos", "Github_MassiveXTessV8_LC_Aperture_Vsx_LS_NO_sigma_cliping_log.npy"),
    "cube_acf": os.path.join(DATA_DIR, "cubos", "Github_MassiveXTessV8_LC_Aperture_Vsx_ACF_NO_sigma_cliping_log.npy"),
    "base_csv": os.path.join(DATA_DIR, "catalogos", "Github_MassiveXTessV8_LC_Aperture_Vsx_LS.csv"),
}

CATALOG_OPTIONS = {
    "Path2 — Completo (3499)": {
        "csv": os.path.join(DATA_DIR, "catalogos", "catalog_path2_main.csv"),
        **_UNIFIED_CUBES,
    },
    "Path2 — ELL (1849)": {
        "csv": os.path.join(DATA_DIR, "catalogos", "catalog_path2_ELL.csv"),
        **_UNIFIED_CUBES,
    },
    "Path2 — Pulsating (581)": {
        "csv": os.path.join(DATA_DIR, "catalogos", "catalog_path2_Pulsating.csv"),
        **_UNIFIED_CUBES,
    },
    "Path2 — E (281)": {
        "csv": os.path.join(DATA_DIR, "catalogos", "catalog_path2_E.csv"),
        **_UNIFIED_CUBES,
    },
    "Path2 — Rndm (85)": {
        "csv": os.path.join(DATA_DIR, "catalogos", "catalog_path2_Rndm.csv"),
        **_UNIFIED_CUBES,
    },
    "Path2 — Unconstrained (477)": {
        "csv": os.path.join(DATA_DIR, "catalogos", "catalog_path2_Unconstrained.csv"),
        **_UNIFIED_CUBES,
    },
    "Path2 — Irregular (18)": {
        "csv": os.path.join(DATA_DIR, "catalogos", "catalog_path2_Irregular.csv"),
        **_UNIFIED_CUBES,
    },
    "Path2 — No_periodo (8)": {
        "csv": os.path.join(DATA_DIR, "catalogos", "catalog_path2_No_periodo.csv"),
        **_UNIFIED_CUBES,
    },
    "Path2 — Mixed (200)": {
        "csv": os.path.join(DATA_DIR, "catalogos", "catalog_path2_mixed.csv"),
        **_UNIFIED_CUBES,
    },
}

CLASS_COLS   = ["CNN+RF_ELL", "CNN+RF_M", "CNN+RF_CEP", "CNN+RF_DST",
                "CNN+RF_E",   "CNN+RF_LPV", "CNN+RF_RR", "CNN+RF_Rndm",
                "CNN+RF_Pulsating"]
CLASS_LABELS = ["ELL", "M", "CEP", "DST", "E", "LPV", "RR", "Rndm", "Pulsating"]

PLOT_COLS   = CLASS_COLS
PLOT_LABELS = CLASS_LABELS
CLASS_COLORS = {
    "ELL": "#4C72B0", "M": "#DD8452", "CEP": "#55A868", "DST": "#C44E52",
    "E": "#8172B2",   "LPV": "#937860", "RR": "#DA8BC3", "Rndm": "#8C8C8C",
    "Pulsating": "#2CA02C",
}

# Columnas espectrales a mostrar, en orden de prioridad
SP_COLS = ["sp_IACOB", "sp_GOSS", "sp_Chini2012", "sp_ALS", "sp_simbad", "sp_Monsalves+2025"]
SP_LABELS = {"sp_IACOB": "IACOB", "sp_GOSS": "GOSS", "sp_Chini2012": "Chini+2012",
             "sp_ALS": "ALS", "sp_simbad": "SIMBAD", "sp_Monsalves+2025": "Monsalves+2025"}

# ── Loaders ────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Cargando catálogo...")
def load_catalog(csv_path, _mtime=None):
    df = pd.read_csv(csv_path)
    df.index.name = "cube_idx"
    return df

@st.cache_data(show_spinner="Cargando catálogo base...")
def load_base_spectral(base_csv_path):
    """Carga sólo las columnas espectrales del catálogo base, alineadas por índice de fila."""
    cols_to_load = ["TIC", "sector_list"] + [c for c in SP_COLS]
    try:
        return pd.read_csv(base_csv_path, usecols=lambda c: c in cols_to_load)
    except Exception:
        return None

@st.cache_resource(show_spinner="Mapeando cubo (mmap)...")
def load_cube_mmap(cube_path):
    return np.load(cube_path, mmap_mode="r")

@st.cache_data(show_spinner="Cargando curva de luz...")
def load_lc(tic, sector):
    path = os.path.join(LC_DIR, f"{tic}_{sector}.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)

def available_sectors(tic):
    files = glob.glob(os.path.join(LC_DIR, f"{tic}_*.csv"))
    return sorted(int(os.path.basename(f).split("_")[1].replace(".csv", "")) for f in files)

def get_spectral_info(base_df, tic):
    """Devuelve dict {label: sp_type} con todos los tipos espectrales disponibles para un TIC."""
    if base_df is None:
        return {}
    rows = base_df[base_df["TIC"] == tic]
    if rows.empty:
        return {}
    row = rows.iloc[0]
    result = {}
    for col in SP_COLS:
        if col in row.index:
            val = row[col]
            if pd.notna(val) and str(val).strip() not in ("", "nan"):
                result[SP_LABELS[col]] = str(val).strip()
    return result

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    catalog_key = st.selectbox("Catálogo / normalización", list(CATALOG_OPTIONS.keys()))
    if "prev_catalog" not in st.session_state:
        st.session_state["prev_catalog"] = catalog_key
    if st.session_state["prev_catalog"] != catalog_key:
        st.session_state["prev_catalog"] = catalog_key
        st.rerun()
    cfg      = CATALOG_OPTIONS[catalog_key]
    df       = load_catalog(cfg["csv"], _mtime=os.path.getmtime(cfg["csv"])).copy()
    cube_ls  = load_cube_mmap(cfg["cube_ls"])
    cube_acf = load_cube_mmap(cfg["cube_acf"])
    base_sp  = load_base_spectral(cfg["base_csv"])

    if "final_class_tic" in df.columns:
        df["final_class_row"] = df["final_class"]   # row-level class (LS or ACF source)
        df["final_class"] = df["final_class_tic"]

    all_classes = sorted(df["final_class"].dropna().unique().tolist())
    if len(all_classes) > 1:
        class_filter = st.multiselect("Filtrar por clase", all_classes, default=all_classes,
                                       help="Selecciona las clases a mostrar")
        if class_filter:
            df = df[df["final_class"].isin(class_filter)]

    tic_list = sorted(df["TIC"].unique().tolist())
    tic_str_list = [str(t) for t in tic_list]
    tic_widget_key = f"tic_select_{catalog_key}"
    tic_str  = st.selectbox("TIC", options=tic_str_list, key=tic_widget_key,
                             help=f"{len(tic_list)} TICs disponibles")
    tic = int(tic_str)

    st.divider()
    show_raw_cube = st.checkbox("Mostrar imagen cubo (32×32)", value=True)

    st.divider()
    st.caption("Explorador de Variabilidad en Estrellas Masivas")

# ── Datos de la estrella ───────────────────────────────────────────────────────
star_df  = df[df["TIC"] == tic].copy()
sectors  = available_sectors(tic)

_sector_key = f"lc_sector_{tic}"
_lc_sector_preview = st.session_state.get(_sector_key, sectors[0] if sectors else None)

st.header(f"TIC {tic}")
col_info1, col_info2, col_info3 = st.columns(3)
_sector_df_preview = star_df[star_df["sector_list"] == _lc_sector_preview] if _lc_sector_preview else star_df
col_info1.metric("Periodos candidatos", len(_sector_df_preview))
col_info2.metric("Sectores en catálogo", star_df["sector_list"].nunique())
col_info3.metric("Sectores con LC local", len(sectors))

# ── Información espectral ─────────────────────────────────────────────────────
sp_info = get_spectral_info(base_sp, tic)
if sp_info:
    with st.expander("Información espectral", expanded=True):
        sp_cols_disp = st.columns(len(sp_info))
        for i, (label, val) in enumerate(sp_info.items()):
            sp_cols_disp[i].metric(label, val)

st.divider()

# ══ Sección 1: LC + Power spectrum lado a lado ════════════════════════════════
col_lc, col_ps = st.columns([1.6, 1])

with col_lc:
    st.subheader("Curvas de luz por sector")
    if not sectors:
        st.info("No se encontraron archivos de LC locales para este TIC.")
        lc_sector_sel = None
    else:
        cat_sectors = sorted(star_df["sector_list"].unique().tolist())
        default_idx = 0
        for cs in cat_sectors:
            if cs in sectors:
                default_idx = sectors.index(cs)
                break
        lc_sector_sel = st.selectbox("Sector", sectors, index=default_idx,
                                     format_func=lambda s: f"S{s}",
                                     key=f"lc_sector_{tic}")
        lc = load_lc(tic, lc_sector_sel)
        if lc is None:
            st.warning(f"Archivo no encontrado: {tic}_{lc_sector_sel}.csv")
        else:
            fig, ax = plt.subplots(figsize=(7, 2.8))
            ax.errorbar(lc["Time"], lc["flux"], yerr=lc["flux_err"],
                        fmt=".", ms=2, lw=0.5, color="steelblue", ecolor="lightgray",
                        alpha=0.8)
            ax.set_xlabel("Tiempo [BTJD]")
            ax.set_ylabel("Flujo [e⁻/s]")
            ax.set_title(f"TIC {tic}  –  Sector {lc_sector_sel}")
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

with col_ps:
    st.subheader("Power spectrum")
    _ps_placeholder = st.empty()

st.divider()

# ══ Sección 2: Tabla de periodos ══════════════════════════════════════════════
st.subheader("Periodos candidatos y clasificaciones")

star_df_sorted = star_df.sort_values("CNN+RF_max_prob", ascending=False)

_clase_col = "final_class_row" if "final_class_row" in star_df_sorted.columns else "final_class"
display_df = star_df_sorted[["sector_list", "period", "power", "amplitud_TESS",
                              _clase_col, "CNN+RF_max_prob", "source"]].copy()
display_df.columns = ["Sector", "Periodo [d]", "Power", "Amplitud", "Clase (CNN)", "Prob. max", "Fuente"]
display_df = display_df.reset_index(drop=True)

table_sel = st.dataframe(
    display_df.style.background_gradient(subset=["Power"], cmap="YlOrRd")
                    .format({"Periodo [d]": "{:.4f}", "Power": "{:.4f}",
                             "Amplitud": "{:.4f}", "Prob. max": "{:.3f}"}),
    use_container_width=True,
    height=250,
    on_select="rerun",
    selection_mode="single-row",
    key=f"period_table_{tic}_{lc_sector_sel}_{catalog_key}",
)

st.divider()

# ══ Sección 3: Imagen · Curva en fase · Probabilidades ════════════════════════
selected_rows = table_sel.selection.get("rows", [])
sel     = selected_rows[0] if selected_rows else 0
sel_row = star_df_sorted.iloc[sel]
period  = float(sel_row["period"])
sector  = int(sel_row["sector_list"])

cidx       = int(sel_row["cube_idx"])
source     = sel_row["source"]
img        = (cube_ls if source == "LS" else cube_acf)[cidx, :, :, 0]
kind_label = source

st.subheader(f"Imagen {kind_label} · Curva en fase · Probabilidades")
_row_class = sel_row.get("final_class_row", sel_row["final_class"])
st.caption(f"Fila {sel + 1}  ·  P={period:.4f} d  ·  S{sector}  ·  {_row_class} [{source}]  ({sel_row['CNN+RF_max_prob']:.2f})")

col_img, col_phase, col_bar = st.columns([1, 1.6, 1.4])

with col_img:
    if show_raw_cube:
        fig3, ax3 = plt.subplots(figsize=(4, 4))
        im = ax3.imshow(np.rot90(img, k=2), origin="lower", cmap="viridis", aspect="auto")
        ax3.set_title(f"Imagen {kind_label}\nP={period:.4f} d", fontsize=9)
        ax3.set_xlabel("Bin frecuencia", fontsize=8)
        ax3.set_ylabel("Bin amplitud", fontsize=8)
        plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
        fig3.tight_layout()
        st.pyplot(fig3, use_container_width=True)
        plt.close(fig3)

with col_phase:
    lc_phase = load_lc(tic, sector)
    if lc_phase is None:
        st.info(f"LC no disponible para S{sector}")
    else:
        phase  = ((lc_phase["Time"] - lc_phase["Time"].iloc[0]) / period) % 1.0
        phase2 = np.concatenate([phase.values, phase.values + 1.0])
        flux2  = np.concatenate([lc_phase["flux"].values, lc_phase["flux"].values])
        err2   = np.concatenate([lc_phase["flux_err"].values, lc_phase["flux_err"].values])

        fig5, ax5 = plt.subplots(figsize=(5, 3.8))
        ax5.errorbar(phase2, flux2, yerr=err2,
                     fmt=".", ms=1.5, lw=0, elinewidth=0.3,
                     color="steelblue", ecolor="lightgray", alpha=0.7)
        ax5.set_xlabel("Fase")
        ax5.set_ylabel("Flujo [e⁻/s]")
        ax5.set_title(f"Curva en fase – P={period:.4f} d  S{sector}", fontsize=9)
        ax5.set_xlim(0, 2)
        ax5.axvline(1.0, ls="--", color="gray", lw=0.7, alpha=0.5)
        fig5.tight_layout()
        st.pyplot(fig5, use_container_width=True)
        plt.close(fig5)

with col_bar:
    probs  = sel_row[PLOT_COLS].values.astype(float)
    colors = [CLASS_COLORS.get(lbl, "gray") for lbl in PLOT_LABELS]

    fig4, ax4 = plt.subplots(figsize=(5, 3.8))
    bars = ax4.barh(PLOT_LABELS, probs, color=colors, edgecolor="white")
    ax4.set_xlim(0, 1)
    ax4.set_xlabel("Probabilidad CNN+RF")
    ax4.set_title(f"Clase [{source}]: {_row_class}  ({sel_row['CNN+RF_max_prob']:.2f})", fontsize=9)
    ax4.axvline(0.5, ls="--", color="gray", lw=0.8)
    for bar, prob in zip(bars, probs):
        ax4.text(min(prob + 0.01, 0.97), bar.get_y() + bar.get_height() / 2,
                 f"{prob:.3f}", va="center", fontsize=8)
    fig4.tight_layout()
    st.pyplot(fig4, use_container_width=True)
    plt.close(fig4)

# ── Power spectrum (rellena el placeholder de arriba, con periodo seleccionado) ─
with _ps_placeholder.container():
    fig2, ax2 = plt.subplots(figsize=(5, 2.8))
    for cls in star_df["final_class"].unique():
        sub = star_df[star_df["final_class"] == cls]
        ax2.scatter(sub["period"], sub["power"],
                    label=cls, color=CLASS_COLORS.get(cls, "gray"),
                    s=30, zorder=3)
    ax2.axvline(period, ls="--", color="red", lw=1, alpha=0.7, label=f"P={period:.4f}")
    ax2.set_xlabel("Periodo [d]")
    ax2.set_ylabel("Power")
    ax2.legend(fontsize=7, loc="upper right")
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    st.pyplot(fig2, use_container_width=True)
    plt.close(fig2)
