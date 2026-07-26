"""Limpieza de saltos de telemetría de TESS.

Dos etapas, aplicadas en este orden por `clean_lightcurve()`:

1. `clean_ramps_binned`  — rampas de scattered light en los bordes de cada
   segmento entre gaps: se binea el segmento, los bins interiores dan la
   referencia y se descarta la racha de bins "rampy" desde cada borde.
2. `sigma_clip_gap_edges` — sigma-clipping global (mediana/std calculadas
   EXCLUYENDO las zonas vecinas a gaps/bordes) aplicado SOLO a los puntos
   dentro de esas zonas; el resto de la curva queda intacto.
"""
import numpy as np
import pandas as pd


def clean_ramps_binned(time, flux, gap_days=2.0, n_bins=8, ratio=3.0,
                       use_median=True, min_seg=None, return_mask=False,
                       plot=False, title=""):
    """Limpia rampas por bins, vectorizado con pandas. Asume `time` ordenado.

    Por cada segmento entre gaps (diff(t) >= gap_days):
      1. lo parte en `n_bins` bins equiespaciados en tiempo,
      2. referencia = bins INTERIORES (sin 1o ni ultimo, que tocan gaps),
      3. marca un bin "rampy" si su std o su offset de mediana superan
         ratio*ref_std, y descarta la RACHA de bins rampy desde cada borde
         de gap (cummin), parando en el primer bin sano.
    Segmentos con menos de `min_seg` puntos (default 3*n_bins) no se tocan.
    """
    if min_seg is None:
        min_seg = 3 * n_bins
    df = pd.DataFrame({"t": np.asarray(time, float), "f": np.asarray(flux, float)})
    df["seg"] = (df["t"].diff() >= gap_days).cumsum()                  # segmento por gaps
    seg_t = df.groupby("seg")["t"]
    df["seg_n"] = seg_t.transform("size")
    span = (seg_t.transform("max") - seg_t.transform("min")).replace(0, np.nan)
    frac = ((df["t"] - seg_t.transform("min")) / span).fillna(0.0)
    df["bin"] = np.clip((frac * n_bins).astype(int), 0, n_bins - 1)    # bin equi-ancho en t
    b = df.groupby(["seg", "bin"]).agg(t0=("t", "min"), t1=("t", "max"),
                                       std=("f", "std"), med=("f", "median")).reset_index()
    inner = b["bin"].between(1, n_bins - 2)                            # bins interiores = referencia
    ref = b[inner].groupby("seg").agg(ref_std=("std", "median"), ref_med=("med", "median"))
    b = b.merge(ref, on="seg", how="left")
    b["rampy"] = (b["std"] > ratio * b["ref_std"]).fillna(False)
    if use_median:
        b["rampy"] |= ((b["med"] - b["ref_med"]).abs() > ratio * b["ref_std"]).fillna(False)
    b["rampy"] = b["rampy"].astype(int)
    b = b.sort_values(["seg", "bin"])
    lead = b.groupby("seg")["rampy"].cummin()                          # racha desde borde izq
    trail = b.sort_values(["seg", "bin"], ascending=[True, False]).groupby("seg")["rampy"].cummin()
    b["drop"] = (lead.astype(bool) | trail.astype(bool))               # racha desde borde der
    df = df.merge(b[["seg", "bin", "drop"]], on=["seg", "bin"], how="left")
    df["drop"] = df["drop"].fillna(False) & (df["seg_n"] >= min_seg)
    keep = (~df["drop"]).to_numpy()
    t = df["t"].to_numpy(); f = df["f"].to_numpy()

    if plot:
        _plot_ramps_diagnostic(t, f, keep, b, df, n_bins, min_seg, ratio, gap_days, title)

    return (t, f, keep) if return_mask else (t[keep], f[keep])


def _plot_ramps_diagnostic(t, f, keep, b, df, n_bins, min_seg, ratio, gap_days, title):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    b = b.merge(df.groupby("seg").size().rename("seg_n2"), on="seg", how="left")
    b["drop_eff"] = b["drop"].astype(bool) & (b["seg_n2"] >= min_seg)
    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(15, 4))
    for g in np.where(np.diff(t) >= gap_days)[0]:                  # gaps sombreados
        ax.axvspan(t[g], t[g + 1], color="0.85", alpha=0.6, zorder=0)
    for bin_id in range(n_bins):                                   # puntos por bin (color)
        sel = (df["bin"] == bin_id).to_numpy()
        ax.plot(t[sel], f[sel], ".", ms=2.5, color=cmap(bin_id % 10), alpha=0.6, zorder=1)
    for _, bb in b.iterrows():                                     # mediana +/- std por bin
        c = "red" if bb["drop_eff"] else ("orange" if bb["rampy"] else "k")
        ax.hlines(bb["med"], bb["t0"], bb["t1"], color=c, lw=2.4, zorder=4)
        if np.isfinite(bb["std"]):
            ax.add_patch(plt.Rectangle((bb["t0"], bb["med"] - bb["std"]),
                                       bb["t1"] - bb["t0"], 2 * bb["std"],
                                       color=c, alpha=0.12, zorder=2))
    for _, sb in b.groupby("seg"):                                 # referencia por segmento
        rm, rs = sb["ref_med"].iloc[0], sb["ref_std"].iloc[0]
        if np.isfinite(rm) and np.isfinite(rs):
            x0, x1 = sb["t0"].min(), sb["t1"].max()
            ax.hlines(rm, x0, x1, color="navy", ls=":", lw=1.3, zorder=3)
            ax.fill_between([x0, x1], rm - ratio * rs, rm + ratio * rs,
                            color="navy", alpha=0.07, zorder=0)
    ax.plot(t[~keep], f[~keep], "x", ms=6, color="red", mew=1.3, zorder=5)  # descartados
    ax.set_title(f"{title} | n={len(t)}->{int(keep.sum())} "
                 f"({100 * (~keep).mean():.1f}% descartado)", fontsize=9)
    ax.set_xlabel("Time [BTJD]"); ax.set_ylabel("flux")
    ax.legend(handles=[
        Line2D([0], [0], color="k", lw=2, label="mediana bin sano"),
        Line2D([0], [0], color="orange", lw=2, label="bin rampy (no en racha de borde)"),
        Line2D([0], [0], color="red", lw=2, label="bin descartado"),
        Line2D([0], [0], color="navy", ls=":", lw=1.3, label=f"ref_med +/- {ratio:g}*ref_std")],
        fontsize=7, ncol=4, loc="best")
    plt.tight_layout(); plt.show()


def sigma_clip_gap_edges(time, flux, seg_days=0.5, gap_thresh=1.0, sigma=3.0,
                         n_edge_blocks=2, min_ref_points=20, return_mask=False):
    """Sigma-clipping global aplicado SOLO cerca de gaps/bordes.

    La mediana/std se calculan con la curva completa EXCLUYENDO las "zonas
    vecinas" (primeros/últimos `n_edge_blocks` bloques de `seg_days` días y
    los bloques adyacentes a cada gap > `gap_thresh` días) — una referencia
    global limpia. El rechazo a `sigma` desviaciones se aplica únicamente a
    los puntos DENTRO de esas zonas; el resto queda intacto.
    """
    t = np.asarray(time, float)
    f = np.asarray(flux, float)
    keep = np.ones(len(t), dtype=bool)
    if len(t) < min_ref_points:
        return (t, f, keep) if return_mask else (t, f)

    block = np.floor((t - t.min()) / seg_days).astype(int)
    n_blocks = block.max() + 1

    gap_pos = np.where(np.diff(t) > gap_thresh)[0]
    near_blocks = set(range(n_edge_blocks)) | set(range(n_blocks - n_edge_blocks, n_blocks))
    for g in gap_pos:
        near_blocks.add(int(block[g]))       # último punto antes del gap
        near_blocks.add(int(block[g + 1]))   # primer punto después del gap
    in_near_zone = np.isin(block, list(near_blocks))

    ref = f[~in_near_zone]
    if len(ref) < min_ref_points:            # curva demasiado corta: no tocar
        return (t, f, keep) if return_mask else (t, f)

    med, std = np.median(ref), np.std(ref)
    drop = (np.abs(f - med) > sigma * std) & in_near_zone
    keep = ~drop
    return (t, f, keep) if return_mask else (t[keep], f[keep])


def clean_lightcurve(time, flux, err=None,
                     gap_days=2.0, n_bins=8, ratio=3.0, use_median=True, min_seg=None,
                     seg_days=0.5, gap_thresh=1.0, sigma=3.0,
                     return_mask=False):
    """Entry point de producción: rampas + sigma-clip en gaps/bordes.

    Ordena por tiempo, descarta no-finitos, aplica `clean_ramps_binned` y
    luego `sigma_clip_gap_edges`. Devuelve (time, flux) o (time, flux, err),
    o (time, flux, err, keep) sobre la curva ordenada si `return_mask=True`.
    """
    t = np.asarray(time, float)
    f = np.asarray(flux, float)
    e = np.asarray(err, float) if err is not None else None

    m = np.isfinite(t) & np.isfinite(f)
    if e is not None:
        m &= np.isfinite(e)
    order = np.argsort(t[m])
    t, f = t[m][order], f[m][order]
    if e is not None:
        e = e[m][order]

    _, _, keep1 = clean_ramps_binned(t, f, gap_days=gap_days, n_bins=n_bins,
                                     ratio=ratio, use_median=use_median,
                                     min_seg=min_seg, return_mask=True)
    keep = keep1.copy()
    _, _, keep2 = sigma_clip_gap_edges(t[keep1], f[keep1], seg_days=seg_days,
                                       gap_thresh=gap_thresh, sigma=sigma,
                                       return_mask=True)
    keep[np.where(keep1)[0][~keep2]] = False

    if return_mask:
        return t, f, e, keep
    if e is not None:
        return t[keep], f[keep], e[keep]
    return t[keep], f[keep]
