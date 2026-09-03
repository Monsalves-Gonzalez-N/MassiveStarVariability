#!/usr/bin/env python
"""Mide la selección de picos contra dos sets con verdad conocida.

No mide "se ve mejor": mide dos números comparables entre criterios.

  FP   `config.FP_PAIRS` — 24 falsos positivos conocidos del corte Number_DST
       (OGLE). No tienen período real: cada candidato emitido es un espurio.
  VSX  cruce masivas × VSX con `Period < 13 d` (label=1, `msv.review`). Se
       cuenta recovery si alguno de los top-N candidatos matchea `per_vsx`
       exacto o en un armónico de `review.HARMONICS`, con tolerancia `config.TOL`.

Las curvas se leen SIEMPRE con `msv.io.open_lc` (quality_bitmask="hardest"), se
limpian con `clean_lightcurve` y los periodogramas se calculan en vivo: los
parquets precalculados y los cubos vienen de la selección vieja y no sirven
para comparar criterios.

    python scripts/validate_peak_selection.py --set fp   --fits-dir ogle_download
    python scripts/validate_peak_selection.py --set vsx  --fits-dir download_paralell --top 3
"""
import argparse
import ast
import json

import numpy as np
import pandas as pd

from msv import config
from msv.cleaning import clean_lightcurve
from msv.io import fits_path, read_lc_fits
from msv.peaks import candidate_periods, select_peaks
from msv.review import HARMONICS, load_vsx_xmatch
from msv.viz import compute_periodograms


def matches(period, reference, tol=config.TOL):
    """True si `period` es `reference` o uno de sus armónicos, dentro de `tol`."""
    if not np.isfinite(period) or not np.isfinite(reference) or reference <= 0:
        return False
    ratios = period / (reference * np.array(HARMONICS))
    return bool(np.min(np.abs(ratios - 1.0)) <= tol)


def candidates_for(time, flux, err=None, top=3):
    """Los `top` candidatos de período de una curva ya leída, más prominentes
    primero: es exactamente lo que el pipeline le pasaría a la CNN."""
    time, flux, err = clean_lightcurve(time, flux, err)
    df_ls, df_acf = compute_periodograms(time, flux, err)
    frames = []
    for source, df in (("LS", df_ls), ("ACF", df_acf)):
        peaks = select_peaks(df, source)
        if peaks.empty:
            continue
        frames.append(candidate_periods(peaks).assign(source=source))
    if not frames:
        return pd.DataFrame(columns=["period", "prominence", "source", "kind"])
    everything = pd.concat(frames).sort_values("prominence", ascending=False)
    return everything.head(top).reset_index(drop=True)


def load_pair(tic, sector, fits_dir):
    path = fits_path(tic, sector, fits_dir)
    if not path.exists():
        return None, f"falta {path.name}"
    if path.stat().st_size == 0:
        return None, f"placeholder 0 B (Dropbox sin hidratar): {path.name}"
    frame = read_lc_fits(path)
    return frame, None


def fp_pairs():
    return [(tic, sector, np.nan) for tic, sector in config.FP_PAIRS]


def vsx_pairs():
    """(TIC, sector, per_vsx) de las label=1, un sector por TIC-sector listado."""
    xmatch = load_vsx_xmatch()
    xmatch = xmatch[xmatch["label"] == 1]
    catalog = pd.read_csv(config.CATALOGS_DIR / "3_MassiveXTessV8_LC_DeltaMagnitude_Clean.csv",
                          usecols=["TIC", "sector_list"])
    catalog["sector_list"] = catalog["sector_list"].apply(
        lambda value: ast.literal_eval(value) if isinstance(value, str) else value)
    sectors = catalog.explode("sector_list").dropna(subset=["sector_list"])
    merged = xmatch[["TIC", "per_vsx"]].merge(sectors, on="TIC", how="inner")
    return [(int(row.TIC), int(row.sector_list), float(row.per_vsx))
            for row in merged.itertuples(index=False)]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--set", choices=["fp", "vsx"], default="fp")
    parser.add_argument("--fits-dir", default=None,
                        help="directorio con los FITS (default: el de config "
                             "según el set)")
    parser.add_argument("--top", type=int, default=3, help="candidatos por curva")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", default=None, help="CSV con el detalle por par")
    args = parser.parse_args()

    if args.set == "fp":
        pairs = fp_pairs()
        default_dir = config.OGLE_DOWNLOAD_DIR
    else:
        pairs = vsx_pairs()
        default_dir = config.TESS_DOWNLOAD_DIR
    fits_dir = default_dir if args.fits_dir is None else (
        config.RAW_DIR / args.fits_dir)
    if args.limit:
        pairs = pairs[:args.limit]

    rows, skipped = [], []
    for tic, sector, reference in pairs:
        frame, problem = load_pair(tic, sector, fits_dir)
        if problem is not None:
            skipped.append((tic, sector, problem))
            continue
        found = candidates_for(frame["Time"].to_numpy(), frame["flux"].to_numpy(),
                               frame["flux_err"].to_numpy(), top=args.top)
        periods = found["period"].astype(float).tolist()
        rows.append({
            "TIC": tic,
            "sector": sector,
            "per_ref": reference,
            "n_candidates": len(periods),
            "periods": json.dumps([round(period, 5) for period in periods]),
            "recovered": any(matches(period, reference) for period in periods),
        })

    table = pd.DataFrame(rows)
    print(f"set={args.set}  dir={fits_dir}")
    print(f"pares pedidos {len(pairs)}   medidos {len(table)}   sin datos {len(skipped)}")
    for tic, sector, problem in skipped[:5]:
        print(f"  - TIC {tic} s{sector}: {problem}")
    if skipped[5:]:
        print(f"  ... y {len(skipped) - 5} más")
    if table.empty:
        raise SystemExit("sin curvas legibles: hidratar los FITS antes de medir")

    if args.set == "fp":
        spurious = int((table["n_candidates"] > 0).sum())
        print(f"\nespurios: {spurious}/{len(table)} pares emiten al menos un "
              f"candidato  (objetivo 0)")
        print(f"candidatos totales: {int(table['n_candidates'].sum())}")
    else:
        recovered = int(table["recovered"].sum())
        print(f"\nrecovery top-{args.top} (exacto + armónico, tol={config.TOL}): "
              f"{recovered}/{len(table)} = {100 * recovered / len(table):.1f}%")

    if args.out:
        table.to_csv(args.out, index=False)
        print(f"detalle -> {args.out}")


if __name__ == "__main__":
    main()
