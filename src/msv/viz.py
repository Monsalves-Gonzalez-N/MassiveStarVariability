"""Helpers de visualización para revisar el pipeline estrella por estrella.

Solo matplotlib + pandas — sin TensorFlow: la parte CNN/BRF se lee de los CSV
precalculados de scripts/run_brf_snr.py. Los ports de plot_periodograms /
plot_hist2d vienen del notebook 2_Github_TESS_variability_MassiveStarG12.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .cleaning import clean_lightcurve
from .config import (CLASS_NAMES, LC_PARQUET_MASSIVE, PEAKS_PARQUET,
                     PERIODIC, PERIODOGRAMS_ACF, PERIODOGRAMS_LS, SIGMA_MAX)
from .peaks import select_peaks_acf, select_peaks_ls

# Un color fijo por clase BRF (no se recicla ni depende del orden de aparición)
CLASS_COLORS = {
    "ELL": "#4477AA", "M": "#EE6677", "CEP": "#228833", "DST": "#CCBB44",
    "E": "#66CCEE", "LPV": "#AA3377", "RR": "#BBBBBB", "Rndm": "#555555",
}
# Las pulsantes se reportan juntas (config.PULSATING): un color propio para el
# grupo, el resto hereda el de su clase.
GROUP_COLORS = dict(CLASS_COLORS, Pulsating="#EE6677")


def _flt(TIC, sector, source=None):
    flt = [("TIC", "=", int(TIC)), ("sector", "=", int(sector))]
    if source is not None:
        flt.append(("source", "=", source))
    return flt


def load_lc(TIC, sector, lc_parquet=LC_PARQUET_MASSIVE):
    """LC cruda de (TIC, sector) desde el parquet agregado (pushdown filters)."""
    df = pd.read_parquet(lc_parquet, columns=["Time", "flux", "flux_err"],
                         filters=_flt(TIC, sector))
    if df.empty:
        raise ValueError(f"Sin LC para TIC={TIC}, sector={sector} en {lc_parquet}")
    return df.sort_values("Time").reset_index(drop=True)


def load_peaks(TIC, sector, source=None, path=PEAKS_PARQUET):
    """Picos de (TIC, sector). source: 'LS', 'ACF' o None (ambos)."""
    df = pd.read_parquet(path, filters=_flt(TIC, sector, source))
    return df.sort_values(["source", "prominence"],
                          ascending=[True, False]).reset_index(drop=True)


def random_pairs(n=1, path=PEAKS_PARQUET, rng=None):
    """n pares (TIC, sector) al azar entre los que tienen picos.

    Sin rng explícito usa default_rng() SIN seed: cada llamada da estrellas
    distintas (a propósito, para auditar el pipeline en runs sucesivos).
    """
    rng = rng or np.random.default_rng()
    pairs = pd.read_parquet(path, columns=["TIC", "sector"]).drop_duplicates()
    idx = rng.choice(len(pairs), size=min(n, len(pairs)), replace=False)
    return [(int(t), int(s)) for t, s in pairs.to_numpy()[idx]]


def plot_cleaning_curve(time, flux, ax=None, title=None, gap_days=2.0,
                        show_gaps=True, **clean_kw):
    """Curva cruda con los puntos que se lleva el sigma-clip marcados.

    Toma arrays, no un par (TIC, sector): es la que usan los PDF de revisión,
    que leen los FITS directamente. `plot_cleaning` es la versión por par.
    """
    t, f, _, keep = clean_lightcurve(time, flux, return_mask=True, **clean_kw)
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 3))
    if show_gaps:
        for edge in t[:-1][np.diff(t) >= gap_days]:
            ax.axvline(edge, color="tab:blue", ls=":", lw=0.8)
    ax.plot(t[keep], f[keep], ".", color="k", ms=1.2, alpha=0.55,
            label=f"{keep.sum()} conservados")
    if (~keep).any():
        ax.plot(t[~keep], f[~keep], "x", color="tab:red", ms=5, mew=1.2,
                label=f"{(~keep).sum()} sigma-clip (gaps/bordes)")
    ax.set_xlabel("Time [BTJD]", fontsize=9)
    ax.set_ylabel("PDCSAP flux", fontsize=8)
    if title:
        ax.set_title(title, fontsize=9)
    ax.legend(loc="best", fontsize=7)
    ax.grid(alpha=0.25)
    return ax, (t[keep], f[keep])


def plot_cleaning(TIC, sector, lc_parquet=LC_PARQUET_MASSIVE, ax=None, **clean_kw):
    """LC cruda con los puntos descartados por clean_lightcurve marcados."""
    lc = load_lc(TIC, sector, lc_parquet)
    # return_mask=True devuelve la curva ya ordenada y sin no-finitos + keep
    t, f, e, keep = clean_lightcurve(lc["Time"].to_numpy(), lc["flux"].to_numpy(),
                                     lc["flux_err"].to_numpy(),
                                     return_mask=True, **clean_kw)
    ax, _ = plot_cleaning_curve(lc["Time"].to_numpy(), lc["flux"].to_numpy(), ax=ax,
                                title=f"TIC {int(TIC)} — sector {int(sector)}: limpieza",
                                **clean_kw)
    return ax, (t[keep], f[keep], e[keep])


def plot_phase_fold(time, flux, period, ax=None, title=None, phase_bins=None,
                    color="k", ms=2):
    """LC plegada a `period` (dos ciclos para ver la continuidad en fase 1).

    Con `phase_bins` se superpone la mediana por bin de fase: es la que hace
    visible la ambigüedad P vs P/2 de una eclipsante (en P se ven dos
    profundidades distintas, en P/2 una sola).
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 3))
    phase = (np.asarray(time) / period) % 1.0
    f = np.asarray(flux)
    for shift in (0.0, 1.0):
        ax.plot(phase + shift, f, ".", color=color, ms=ms, alpha=0.6)
    if phase_bins:
        edges = np.linspace(0.0, 1.0, phase_bins + 1)
        which = np.digitize(phase, edges) - 1
        binned = np.array([np.median(f[which == b]) if (which == b).any() else np.nan
                           for b in range(phase_bins)])
        centres = 0.5 * (edges[:-1] + edges[1:])
        for shift in (0.0, 1.0):
            ax.plot(centres + shift, binned, "-", color="tab:red", lw=1.4)
    ax.axvline(1.0, color="0.8", lw=0.8)
    ax.set_xlim(0, 2)
    ax.set_xlabel("Phase")
    ax.set_ylabel("Flux")
    ax.set_title(title or f"P = {period:.4f} d", fontsize=10)
    ax.grid(alpha=0.3)
    return ax


def compute_periodograms(time, flux, err=None):
    """LS + ACF en vivo (curva ya limpia), con el mismo schema que los
    parquets precalculados (columna `fap` incluida)."""
    from .periodograms import acf_periodogram, ls_periodogram
    df_ls = ls_periodogram(time, flux, err)
    df_ls["fap"] = df_ls.attrs["fap_level"]
    df_acf = acf_periodogram(time, flux, err)
    return df_ls, df_acf


def plot_periodograms_pair(TIC, sector, ls_path=PERIODOGRAMS_LS,
                           acf_path=PERIODOGRAMS_ACF, dfs=None, figsize=(14, 8),
                           top_n=None, min_peak_sep_days=None,
                           prominence_frac=None, xscale="log", savepath=None):
    """LS + ACF desde los parquets precalculados, con picos de msv.peaks.

    `dfs=(df_ls, df_acf)` salta la lectura de parquet (periodogramas en vivo
    de compute_periodograms). `top_n=None` -> grafica TODOS los picos > FAP.
    Devuelve (df_ls, df_acf, top_ls, top_acf).
    """
    if dfs is not None:
        df_ls, df_acf = dfs
    else:
        flt = _flt(TIC, sector)
        df_ls = pd.read_parquet(ls_path, filters=flt).sort_values("per").reset_index(drop=True)
        df_acf = pd.read_parquet(acf_path, filters=flt).sort_values("per").reset_index(drop=True)
    if df_ls.empty or df_acf.empty:
        raise ValueError(f"Sin periodogramas para TIC={TIC}, sector={sector}")

    fap_ls = float(df_ls["fap"].iloc[0])
    # grillas: LS uniforme en frecuencia, ACF uniforme en período
    freq_step = float(np.median(np.abs(np.diff(1.0 / df_ls["per"].to_numpy()))))
    cadence = float(np.median(np.diff(df_acf["per"].to_numpy())))

    kw_ls = {} if prominence_frac is None else {"prominence_frac": prominence_frac}
    top_ls = select_peaks_ls(
        df_ls["per"].to_numpy(), df_ls["power"].to_numpy(), fap_ls,
        window=df_ls["window"].to_numpy() if "window" in df_ls.columns else None,
        freq_step=freq_step, top_n=top_n, **kw_ls,
    )
    kw_acf = {}
    if min_peak_sep_days is not None:
        kw_acf["min_peak_sep_days"] = min_peak_sep_days
    if prominence_frac is not None:
        kw_acf["prominence_frac"] = prominence_frac
    top_acf = select_peaks_acf(
        df_acf["per"].to_numpy(), df_acf["power"].to_numpy(),
        df_acf["fap"].to_numpy(), cadence, top_n=top_n, **kw_acf,
    )

    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)

    ax = axes[0]
    if "window" in df_ls.columns:
        ax.plot(df_ls["per"], df_ls["window"], color="0.6", lw=0.6,
                label="window (LS de flux=1)")
    ax.plot(df_ls["per"], df_ls["power"], "k-", lw=0.7, label="LS power")
    ax.axhline(fap_ls, color="r", ls="--", lw=1.0,
               label=f"FAP Baluev = {fap_ls:.3f}")
    if not top_ls.empty:
        ax.plot(top_ls["per"], top_ls["power"], "o", color="tab:orange",
                ms=7, mfc="none", mew=1.5, label=f"{len(top_ls)} peaks > FAP")
        for _, r in top_ls.iterrows():
            ax.annotate(f"{r['per']:.3f} d", (r["per"], r["power"]),
                        xytext=(4, 4), textcoords="offset points", fontsize=9)
    ax.set_xscale(xscale)
    ax.set_ylabel("LS power", fontsize=12)
    ax.set_title(f"TIC {TIC} — sector {sector}", fontsize=13)
    ax.legend(loc="best", fontsize=10)
    ax.grid(alpha=0.3, which="both")

    ax = axes[1]
    ax.plot(df_acf["per"], df_acf["power"], "k-", lw=0.7, label="ACF")
    ax.plot(df_acf["per"], df_acf["fap"], "r--", lw=1.0,
            label="±FAP bartlett-statsmodels")
    ax.plot(df_acf["per"], -df_acf["fap"], "r--", lw=1.0)
    if not top_acf.empty:
        ax.plot(top_acf["per"], top_acf["power"], "o", color="tab:orange",
                ms=7, mfc="none", mew=1.5, label=f"{len(top_acf)} peaks > FAP")
        for _, r in top_acf.iterrows():
            ax.annotate(f"{r['per']:.3f} d", (r["per"], r["power"]),
                        xytext=(4, 4), textcoords="offset points", fontsize=9)
    ax.axhline(0, color="0.85", lw=0.5)
    ax.set_xscale(xscale)
    ax.set_xlabel("Period / Lag [d]", fontsize=12)
    ax.set_ylabel("ACF", fontsize=12)
    ax.legend(loc="best", fontsize=10)
    ax.grid(alpha=0.3, which="both")

    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=150, bbox_inches="tight")
    return df_ls, df_acf, top_ls, top_acf


def plot_hist2d_grid(TIC, sector, source=None, path=PEAKS_PARQUET, lc=None,
                     ncols=4, panel_size=(3.0, 2.6), cmap="viridis"):
    """Todos los hist2d (32x32) de (TIC, sector) en una grilla de subplots.

    Si el parquet no trae la columna `hist2d` (runs viejos), pasar
    `lc=(time, flux)` limpios para computarlos en vivo.
    """
    df = pd.read_parquet(path, filters=_flt(TIC, sector, source))
    if df.empty:
        raise ValueError(f"No hay picos para TIC={TIC}, sector={sector}, source={source}")
    if "hist2d" not in df.columns:
        if lc is None:
            raise ValueError(f"{path} no trae hist2d; pasar lc=(time, flux) "
                             "para computarlos en vivo")
        from .features import phase_fold_hist2d_log
        t, f = lc
        df["hist2d"] = [phase_fold_hist2d_log(t, f, p).ravel() for p in df["per"]]

    df = df.sort_values("prominence", ascending=False).reset_index(drop=True)
    n = len(df)
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(panel_size[0] * ncols, panel_size[1] * nrows),
        squeeze=False, layout="constrained",
    )
    im = None
    for i, row in df.iterrows():
        ax = axes[i // ncols, i % ncols]
        H = np.asarray(row["hist2d"]).reshape(32, 32)
        im = ax.imshow(H, origin="upper", aspect="auto",
                       extent=[0.0, 1.0, 0.0, 1.0], cmap=cmap, vmin=0, vmax=1)
        ax.set_title(f"{row['source']}  P={row['per']:.4f} d\n"
                     f"prom={row['prominence']:.3f}  width={row['width']:.1f}",
                     fontsize=9)
        ax.tick_params(labelsize=7)
    for j in range(n, nrows * ncols):
        axes[j // ncols, j % ncols].axis("off")

    fig.suptitle(f"TIC {int(TIC)}  sector {int(sector)}  —  {n} picos"
                 + (f"  [{source}]" if source else ""), fontsize=11)
    fig.supxlabel("Phase", fontsize=10)
    fig.supylabel("Flux bin (normalized)", fontsize=10)
    fig.colorbar(im, ax=axes, shrink=0.7, label="log1p(h)/log1p(max)",
                 location="right")
    return fig, axes


def load_brf(TIC, sector, brf_csv):
    """Filas de brf_mc_peaks_<model>.csv para (TIC, sector)."""
    df = pd.read_csv(brf_csv)
    sel = df[(df["TIC"] == int(TIC)) & (df["sector"] == int(sector))]
    return sel.sort_values("prominence", ascending=False).reset_index(drop=True)


def plot_brf_summary(TIC, sector, brf_csv, sigma_max=SIGMA_MAX,
                     sigma_col="instability", ax=None):
    """Por peak: prob del BRF coloreada por clase + métrica de incertidumbre.

    Si el CSV trae `brf_class_raw` (gate aplicado), los peaks relabelados a
    Rndm se marcan con hatch — es lo que el gate mató.
    """
    df = load_brf(TIC, sector, brf_csv)
    if df.empty:
        raise ValueError(f"No hay predicciones para TIC={TIC}, sector={sector} en {brf_csv}")
    if ax is None:
        _, ax = plt.subplots(figsize=(max(6, 1.1 * len(df)), 3.2))

    gated = ("brf_class_raw" in df.columns
             and (df["brf_class_raw"] != df["brf_class"]).to_numpy())
    if not isinstance(gated, np.ndarray):
        gated = np.zeros(len(df), bool)

    x = np.arange(len(df))
    colors = [CLASS_COLORS.get(c, "0.5") for c in df["brf_class"]]
    ax.bar(x, df["brf_prob"], width=0.6, color=colors,
           edgecolor="white", linewidth=1,
           hatch=["///" if g else "" for g in gated])
    for i, r in df.iterrows():
        lab = r["brf_class"] + (f" (era {r['brf_class_raw']})" if gated[i] else "")
        ax.annotate(lab, (i, r["brf_prob"]), ha="center", va="bottom", fontsize=8)

    ax2 = ax.twinx()
    ax2.plot(x, df[sigma_col], "o", color="k", ms=5, label=sigma_col)
    ax2.axhline(sigma_max, color="tab:red", ls="--", lw=1,
                label=f"gate = {sigma_max}")
    ax2.set_ylabel(sigma_col, fontsize=10)
    ax2.legend(loc="upper right", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{r['source']}\n{r['per']:.3f} d"
                        for _, r in df.iterrows()], fontsize=8)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("BRF prob", fontsize=10)
    ax.set_title(f"TIC {int(TIC)} sector {int(sector)} — BRF por peak "
                 f"(hatch = relabelado por gate)", fontsize=11)
    return df


def show_sample(TIC, sector, lc_parquet=LC_PARQUET_MASSIVE,
                ls_path=PERIODOGRAMS_LS, acf_path=PERIODOGRAMS_ACF,
                peaks_path=PEAKS_PARQUET, brf_csv=None,
                sigma_max=SIGMA_MAX, sigma_col="instability"):
    """Revisión completa de una estrella: limpieza -> LS/ACF -> hist2d ->
    fase plegada al mejor per -> (opcional) BRF + gate desde brf_csv.

    Devuelve dict con los DataFrames intermedios para inspección manual.
    """
    out = {}
    fig, ax = plt.subplots(figsize=(12, 3))
    _, (t, f, e) = plot_cleaning(TIC, sector, lc_parquet, ax=ax)
    plt.show()

    try:
        out["ls"], out["acf"], out["top_ls"], out["top_acf"] = plot_periodograms_pair(
            TIC, sector, ls_path=ls_path, acf_path=acf_path)
    except (ValueError, FileNotFoundError, OSError):
        # el par no está en los parquets precalculados -> computar en vivo
        print("(periodogramas no precalculados: computando en vivo)")
        dfs = compute_periodograms(t, f, e)
        out["ls"], out["acf"], out["top_ls"], out["top_acf"] = plot_periodograms_pair(
            TIC, sector, dfs=dfs)
    plt.show()

    plot_hist2d_grid(TIC, sector, path=peaks_path, lc=(t, f))
    plt.show()

    peaks = load_peaks(TIC, sector, path=peaks_path)
    out["peaks"] = peaks
    if not peaks.empty:
        best = peaks.loc[peaks["prominence"].idxmax()]
        plot_phase_fold(t, f, best["per"],
                        title=f"mejor peak: {best['source']} P={best['per']:.4f} d")
        plt.show()

    if brf_csv is not None:
        out["brf"] = plot_brf_summary(TIC, sector, brf_csv,
                                      sigma_max=sigma_max, sigma_col=sigma_col)
        plt.show()
        cls = out["brf"]["brf_class"]
        n_per = cls.isin(PERIODIC).sum()
        print(f"BRF: {cls.value_counts().to_dict()}  ({n_per} peaks periódicos"
              f" tras gate {sigma_col}<{sigma_max})")
    return out
