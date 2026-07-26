"""Features de entrada de la CNN: hist2d 32x32 phase-folded + amplitud."""
import numpy as np

N_EDGES = 33                     # 33 edges -> 32 bins por eje
HIST_SIZE = (N_EDGES - 1) ** 2   # 1024


def phase_fold_hist2d_log(time, flux, period, n_edges=N_EDGES):
    """Phase-fold + histograma 2D con normalización logarítmica (32x32)."""
    fase = np.mod(time, period) / period
    bins_x = np.linspace(0.0, 1.0, n_edges)
    bins_y = np.linspace(flux.min(), flux.max(), n_edges)
    h, _, _ = np.histogram2d(fase, flux, bins=(bins_x, bins_y))
    hmax = h.max()
    if hmax > 0:
        h = np.log1p(h) / np.log1p(hmax)
    return h.T[::-1].astype(np.float32)


def amplitude_of(flux):
    """|−2.5·log10(fmax/fmin)|; NaN si hay flujo <= 0 (error de cleaning)."""
    f = np.asarray(flux)
    if f.min() > 0:
        return float(np.absolute(-2.5 * np.log10(f.max() / f.min())))
    return float("nan")


def build_cube(peaks_df, lc_groups, min_points=20):
    """Cubo (N,32,32,1) + amplitud por peak. Independiente del modelo CNN.

    `lc_groups`: dict {(TIC, sector): DataFrame con Time y flux}.
    Peaks sin curva válida quedan con hist2d en cero y amplitud NaN.
    """
    N = len(peaks_df)
    X = np.zeros((N, 32, 32, 1), dtype=np.float32)
    amp = np.full(N, np.nan, dtype=np.float64)
    amp_cache, n_skip = {}, 0
    for i, row in enumerate(peaks_df.itertuples(index=False)):
        key = (row.TIC, row.sector)
        lc = lc_groups.get(key)
        if lc is None:
            n_skip += 1
            continue
        t = lc["Time"].to_numpy()
        f = lc["flux"].to_numpy()
        m = np.isfinite(t) & np.isfinite(f)
        t, f = t[m], f[m]
        if len(t) < min_points:
            n_skip += 1
            continue
        if key not in amp_cache:
            amp_cache[key] = amplitude_of(f)
        amp[i] = amp_cache[key]
        X[i, ..., 0] = phase_fold_hist2d_log(t, f, float(row.per))
    if n_skip:
        print(f"build_cube: {n_skip} peaks sin curva válida")
    return X, amp
