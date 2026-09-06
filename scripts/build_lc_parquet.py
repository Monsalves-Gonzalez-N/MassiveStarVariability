#!/usr/bin/env python
"""Directorio de FITS SPOC -> parquet de curvas de luz (Time, flux, flux_err).

Es la versión ejecutable de la celda de ray del notebook 1: la misma lectura
canónica (`msv.io.read_lc_fits`) sobre un directorio, para poder correr el
pipeline completo sobre un subconjunto sin tocar el parquet grande.

    python scripts/build_lc_parquet.py --dir review_50 --out results/lc_review50.parquet
"""
import argparse
from pathlib import Path

import pandas as pd

from msv import config
from msv.io import parse_fits_name, read_lc_fits


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dir", default="review_50",
                        help="subdirectorio de RAW_DIR con los FITS")
    parser.add_argument("--out", default=str(config.RESULTS_DIR / "lc_review50.parquet"))
    args = parser.parse_args()

    paths = sorted((config.RAW_DIR / args.dir).glob("*_lc.fits"))
    if not paths:
        raise SystemExit(f"sin FITS en {config.RAW_DIR / args.dir}")

    frames, errors = [], []
    for path in paths:
        tic, sector = parse_fits_name(path)
        try:
            frame = read_lc_fits(path)
        except Exception as problem:
            errors.append((tic, sector, repr(problem)))
            continue
        frame["TIC"] = tic
        frame["sector"] = sector
        frames.append(frame)

    table = pd.concat(frames, ignore_index=True)
    # Un nombre suelto va a RESULTS_DIR; cualquier cosa con separador se toma
    # tal cual, relativa al directorio de trabajo. Antes se prefijaba SIEMPRE
    # con RESULTS_DIR y `--out results/golden/lc.parquet` terminaba en
    # `results/results/golden/`.
    out = Path(args.out)
    if not out.is_absolute() and len(out.parts) == 1:
        out = config.RESULTS_DIR / out
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(out, index=False)
    print(f"-> {out}")
    print(f"   {table.groupby(['TIC', 'sector']).ngroups} pares, {len(table)} puntos, "
          f"{len(errors)} errores")
    for row in errors[:5]:
        print("   ", row)


if __name__ == "__main__":
    main()
