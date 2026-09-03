#!/usr/bin/env python
"""Comparación de normalizaciones del histograma 2D que entra a la CNN.

Los pesos de la CNN se entrenaron con min-max sobre curvas OGLE de unos pocos
cientos de puntos. `log1p` NO es invariante a h -> alpha*h (número de puntos),
así que una curva TESS de 2 min satura la imagen; cualquier h^p / max(h^p) sí
lo es exactamente. Este script construye el cubo de entrada bajo cada variante
para poder elegir mirando el resultado, sin tocar `msv.features`.

Subcomandos (env CNN_TESS, PYTHONPATH=src):

    python scripts/compare_hist_norms.py cubes       results/norm_compare
    python scripts/compare_hist_norms.py invariance  results/norm_compare
"""
import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
N_EDGES = 33
QUANTIZED_REFERENCE_POINTS = 300
POISSON_CLIP = 5.0
META_COLS = ["TIC", "sector", "source", "per", "power", "prominence", "width",
             "amplitude"]

NORMALIZATION_NAMES = ["log", "min_max", "power_0.5", "power_0.33", "column",
                       "quantized", "poisson"]

SUBSAMPLE_SIZES = [2000, 1000, 500, 200, 100]
DENSE_MIN_POINTS = 5000
N_DENSE_STARS = 8
N_INVARIANCE_PEAKS = 3
REQUIRED_DENSE_PAIRS = [(12675729, 82)]
RANDOM_SEED = 20260903


def _min_max(image):
    span = image.max() - image.min()
    if span > 0:
        return (image - image.min()) / span
    return np.zeros_like(image)


def _power_norm(counts, exponent):
    scaled = np.power(counts, exponent)
    maximum = scaled.max()
    if maximum > 0:
        return scaled / maximum
    return np.zeros_like(scaled)


def normalize_counts(counts, normalization, n_points=None):
    """Normaliza el histograma crudo de cuentas (fase x flujo)."""
    counts = np.asarray(counts, dtype=np.float64)

    if normalization == "log":
        maximum = counts.max()
        if maximum > 0:
            return np.log1p(counts) / np.log1p(maximum)
        return np.zeros_like(counts)

    if normalization == "min_max":
        return _min_max(counts)

    if normalization == "power_0.5":
        return _power_norm(counts, 0.5)

    if normalization == "power_0.33":
        return _power_norm(counts, 1.0 / 3.0)

    if normalization == "column":
        column_maximum = counts.max(axis=1, keepdims=True)
        safe = np.where(column_maximum > 0, column_maximum, 1.0)
        return counts / safe

    if normalization == "quantized":
        if n_points is None or n_points <= 0:
            raise ValueError("'quantized' necesita n_points")
        rescaled = np.rint(counts * (QUANTIZED_REFERENCE_POINTS / float(n_points)))
        return _min_max(rescaled)

    if normalization == "poisson":
        total = counts.sum()
        if total <= 0:
            return np.zeros_like(counts)
        expected = np.outer(counts.sum(axis=1), counts.sum(axis=0)) / total
        safe_expected = np.where(expected > 0, expected, np.nan)
        residual = (counts - expected) / np.sqrt(safe_expected)
        residual = np.nan_to_num(residual, nan=0.0, posinf=0.0, neginf=0.0)
        return _min_max(np.clip(residual, 0.0, POISSON_CLIP))

    raise ValueError(f"normalización desconocida: {normalization}")


def phase_fold_counts(time, flux, period, flux_range=None, n_edges=N_EDGES):
    phase = np.mod(time, period) / period
    if flux_range is None:
        flux_low, flux_high = flux.min(), flux.max()
    else:
        flux_low, flux_high = flux_range
    bins_phase = np.linspace(0.0, 1.0, n_edges)
    bins_flux = np.linspace(flux_low, flux_high, n_edges)
    counts, _, _ = np.histogram2d(phase, flux, bins=(bins_phase, bins_flux))
    return counts


def phase_fold_image(time, flux, period, normalization, flux_range=None):
    """Imagen 32x32 lista para la CNN, con la orientación de msv.features."""
    counts = phase_fold_counts(time, flux, period, flux_range=flux_range)
    normalized = normalize_counts(counts, normalization, n_points=len(time))
    return normalized.T[::-1].astype(np.float32)


def load_inputs():
    curves_path = REPO_ROOT / "results" / "phasefold_curves.pkl"
    peaks_path = REPO_ROOT / "results" / "peaks_review50.parquet"
    with open(curves_path, "rb") as handle:
        curves = pickle.load(handle)
    peaks = pd.read_parquet(peaks_path).reset_index(drop=True)
    return curves, peaks


def cadence_table(curves):
    rows = []
    for (tic, sector), (time, flux) in curves.items():
        cadence_minutes = float(np.median(np.diff(np.sort(time))) * 24.0 * 60.0)
        if cadence_minutes < 6.0:
            group = "2 min"
        elif cadence_minutes < 20.0:
            group = "10 min"
        else:
            group = "30 min"
        rows.append({"TIC": int(tic), "sector": int(sector),
                     "n_points": int(len(time)),
                     "cadence_minutes": cadence_minutes,
                     "cadence_group": group})
    return pd.DataFrame(rows)


def build_cubes(out_dir):
    curves, peaks = load_inputs()
    out_dir.mkdir(parents=True, exist_ok=True)

    for normalization in NORMALIZATION_NAMES:
        cube = np.zeros((len(peaks), 32, 32, 1), dtype=np.float32)
        n_missing = 0
        for position, row in enumerate(peaks.itertuples(index=False)):
            curve = curves.get((int(row.TIC), int(row.sector)))
            if curve is None:
                n_missing += 1
                continue
            time, flux = curve
            cube[position, ..., 0] = phase_fold_image(
                np.asarray(time), np.asarray(flux), float(row.per), normalization)
        target = out_dir / f"input_{normalization}.npz"
        np.savez_compressed(
            target, X=cube,
            **{column: peaks[column].to_numpy() for column in META_COLS})
        print(f"-> {target}  X{cube.shape}  sin curva: {n_missing}")

    cadence_table(curves).to_csv(out_dir / "cadence.csv", index=False)


def select_dense_peaks(curves, peaks):
    cadence = cadence_table(curves)
    dense = cadence[cadence["n_points"] > DENSE_MIN_POINTS]
    dense = dense.sort_values(["TIC", "sector"]).reset_index(drop=True)

    selected_pairs = [pair for pair in REQUIRED_DENSE_PAIRS
                      if pair in set(zip(dense["TIC"], dense["sector"]))]
    for tic, sector in zip(dense["TIC"], dense["sector"]):
        if len(selected_pairs) >= N_DENSE_STARS:
            break
        if (tic, sector) not in selected_pairs:
            selected_pairs.append((int(tic), int(sector)))

    chosen = []
    for tic, sector in selected_pairs:
        subset = peaks[(peaks["TIC"] == tic) & (peaks["sector"] == sector)]
        subset = subset.sort_values("prominence", ascending=False).head(N_INVARIANCE_PEAKS)
        chosen.append(subset)
    return pd.concat(chosen, ignore_index=True)


def build_invariance(out_dir):
    """Sonda de invariancia: sólo cambia el número de puntos.

    Los bordes de flujo y la amplitud se congelan en los de la curva completa
    para que la única diferencia entre tamaños sea la escala de las cuentas.
    """
    curves, peaks = load_inputs()
    out_dir.mkdir(parents=True, exist_ok=True)
    chosen = select_dense_peaks(curves, peaks)
    generator = np.random.default_rng(RANDOM_SEED)

    images = []
    meta_rows = []
    for row in chosen.itertuples(index=False):
        time_full, flux_full = curves[(int(row.TIC), int(row.sector))]
        time_full = np.asarray(time_full)
        flux_full = np.asarray(flux_full)
        flux_range = (flux_full.min(), flux_full.max())
        n_full = len(time_full)

        sizes = [n_full] + [size for size in SUBSAMPLE_SIZES if size < n_full]
        for size in sizes:
            if size == n_full:
                time, flux = time_full, flux_full
            else:
                selection = generator.choice(n_full, size=size, replace=False)
                selection.sort()
                time, flux = time_full[selection], flux_full[selection]
            for normalization in NORMALIZATION_NAMES:
                images.append(phase_fold_image(time, flux, float(row.per),
                                               normalization, flux_range=flux_range))
                meta_rows.append({
                    "TIC": int(row.TIC), "sector": int(row.sector),
                    "source": row.source, "per": float(row.per),
                    "power": float(row.power), "prominence": float(row.prominence),
                    "width": float(row.width), "amplitude": float(row.amplitude),
                    "normalization": normalization, "n_points": int(size),
                    "is_full": bool(size == n_full)})

    meta = pd.DataFrame(meta_rows)
    cube = np.stack(images)[..., np.newaxis]
    np.savez_compressed(
        out_dir / "invariance_input.npz", X=cube,
        **{column: meta[column].to_numpy() for column in META_COLS})
    meta.to_csv(out_dir / "invariance_meta.csv", index=False)
    print(f"-> {out_dir / 'invariance_input.npz'}  X{cube.shape}  "
          f"{chosen.groupby(['TIC', 'sector']).ngroups} pares densos")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=["cubes", "invariance"])
    parser.add_argument("out_dir", nargs="?", default=str(REPO_ROOT / "results" / "norm_compare"))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    if args.step == "cubes":
        build_cubes(out_dir)
    else:
        build_invariance(out_dir)


if __name__ == "__main__":
    main()
