"""Features de entrada de la CNN: hist2d 32x32 phase-folded + amplitud.

Dos normalizaciones, seleccionables por `config.HIST_NORM`:

- `log`   log1p(h)/log1p(max h). NO es invariante al número de puntos: si
          h -> a*h la imagen se satura, así que una curva densa produce una
          imagen más plana que una rala. Es el default por lo que ese defecto
          implica — ver el comentario de HIST_NORM en config.py: al doblar una
          eclipsante en P/2 el histograma se llena más y la imagen se degrada,
          de modo que la CNN solo dice `E` en el período verdadero.
- `min_max` (h - min)/(max - min). Es la normalización con la que se entrenaron
          los pesos y es exactamente invariante a h -> a*h (el caso p=1 de la
          familia h^p/max(h^p); log es la única de la familia que no lo es).
          Da probabilidades más altas y estables, pero dice `E` tanto en P como
          en P/2.

Los cubos históricos vienen en las dos variantes (`*_log.npy`, `*_min_max.npy`).
"""
import numpy as np

from .config import HIST_NORM

N_EDGES = 33                     # 33 edges -> 32 bins por eje
HIST_SIZE = (N_EDGES - 1) ** 2   # 1024


def _phase_fold_counts(time, flux, period, n_edges=N_EDGES):
    fase = np.mod(time, period) / period
    bins_x = np.linspace(0.0, 1.0, n_edges)
    bins_y = np.linspace(flux.min(), flux.max(), n_edges)
    counts, _, _ = np.histogram2d(fase, flux, bins=(bins_x, bins_y))
    return counts


def phase_fold_hist2d_minmax(time, flux, period, n_edges=N_EDGES):
    """Phase-fold + histograma 2D normalizado min-max (32x32)."""
    counts = _phase_fold_counts(time, flux, period, n_edges)
    span = counts.max() - counts.min()
    if span > 0:
        counts = (counts - counts.min()) / span
    return counts.T[::-1].astype(np.float32)


def phase_fold_hist2d_log(time, flux, period, n_edges=N_EDGES):
    """Phase-fold + histograma 2D con normalización logarítmica (32x32)."""
    counts = _phase_fold_counts(time, flux, period, n_edges)
    hmax = counts.max()
    if hmax > 0:
        counts = np.log1p(counts) / np.log1p(hmax)
    return counts.T[::-1].astype(np.float32)


def phase_fold_hist2d(time, flux, period, n_edges=N_EDGES, norm=None):
    """Despacha según `config.HIST_NORM` ('min_max' o 'log')."""
    norm = norm or HIST_NORM
    if norm == "min_max":
        return phase_fold_hist2d_minmax(time, flux, period, n_edges)
    if norm == "log":
        return phase_fold_hist2d_log(time, flux, period, n_edges)
    raise ValueError(f"norm='{norm}' no soportada (usar 'min_max' o 'log')")


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
        X[i, ..., 0] = phase_fold_hist2d(t, f, float(row.per))
    if n_skip:
        print(f"build_cube: {n_skip} peaks sin curva válida")
    return X, amp
