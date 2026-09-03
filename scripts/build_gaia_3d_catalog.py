#!/usr/bin/env python
"""Merge the two subsets of J/A+A/704/A155 into one table for the 3D map.

    python scripts/build_gaia_3d_catalog.py
    python scripts/build_gaia_3d_catalog.py --max-label 12 --output catalogs/0_MassiveG12_3D.csv

The Vizier catalog ships three tables:
  candms  — 9089 BRF candidates, excluded from the training labels
  g12par  — 99387 parent stars: training `label`, astrometry and photometry
  auxcode — same 99387 stars with coordinates and coded Skiff spectral type

`label` is the Skiff spectral type code (O0 = 0 ... M = 60s, -9999 = no type),
so the binary "massive" training class of the paper is `label <= 12` (B2 or
earlier) — that is the parent subset plotted in 3D.

`auxcode` supplies the coordinates of the parent subset, which used to come
from `Masivas_untilG12_Summary.csv`, so the table is built from Vizier alone.
No Gaia DR3 query is needed: g12par already carries plx, e_plx, RPlx, the
proper motions and RUWE.

The tables are read from the CDS .dat.gz files rather than through
astroquery.Vizier: the web service returns only its default column subset
(which drops e_plx, RPlx, the proper motions and RUWE), and asking it for
`columns=["**"]` on 99387 rows times out.
"""
import argparse
import gzip
import shutil
import urllib.request
from pathlib import Path

import astropy.units as u
import numpy as np
import pandas as pd
from astropy.coordinates import ICRS, Galactic, Galactocentric
from astropy.table import Table

REPO_ROOT = Path(__file__).resolve().parents[1]
CDS_URL = "https://cdsarc.cds.unistra.fr/ftp/J/A+A/704/A155"
CDS_FILES = ["ReadMe", "candms.dat.gz", "g12par.dat.gz", "auxcode.dat.gz"]
MISSING_INTEGER = -9999
MISSING_FLOAT = 1e20


def download_cds_files(cache_dir):
    cache_dir.mkdir(parents=True, exist_ok=True)
    for name in CDS_FILES:
        destination = cache_dir / name
        if destination.exists():
            continue
        print(f"  downloading {name}")
        urllib.request.urlretrieve(f"{CDS_URL}/{name}", destination)
    return cache_dir


def load_cds_tables(cache_dir):
    readme = cache_dir / "ReadMe"
    tables = []
    for name in ["candms.dat", "g12par.dat", "auxcode.dat"]:
        # ascii.cds matches the data file against its entry in the ReadMe by
        # basename, so the .gz has to be expanded to its declared name.
        plain = cache_dir / name
        if not plain.exists():
            with gzip.open(cache_dir / f"{name}.gz", "rb") as source, \
                    open(plain, "wb") as destination:
                shutil.copyfileobj(source, destination)
        table = Table.read(str(plain), format="ascii.cds", readme=str(readme))
        tables.append(table.to_pandas())
    return tables


def clean_sentinels(catalog):
    for column in ["label", "sk-type-mean"]:
        if column in catalog:
            catalog[column] = catalog[column].replace(MISSING_INTEGER, np.nan)
    for column in ["sk-type-sd", "sk-lum-mean", "sk-lum-sd",
                   "sb-type-code", "sb-lum-code"]:
        if column in catalog:
            catalog[column] = catalog[column].replace(MISSING_FLOAT, np.nan)
    return catalog


def build_merged_catalog(candidates, parameters, auxiliary, max_label):
    auxiliary = auxiliary.rename(columns={"RAdeg": "ra", "DEdeg": "dec",
                                          "RA_ICRS": "ra", "DE_ICRS": "dec"})
    coordinates = auxiliary[["GaiaDR3", "ra", "dec", "sk-type-mean",
                             "sk-type-sd", "sk-lum-mean", "sk-lum-sd"]]

    labelled = parameters.loc[parameters["label"] <= max_label].copy()
    labelled["is_labelled"] = True

    # The candidates are a subset of g12par, so their astrometry and photometry
    # come from the same merge rather than from a second query.
    candidates = candidates.rename(columns={"RAdeg": "ra", "DEdeg": "dec",
                                            "RA_ICRS": "ra", "DE_ICRS": "dec"})
    # candms and g12par both carry B2-cut-prob with the same values; dropping
    # the duplicate here keeps one column instead of a _x/_y pair that then
    # fails to line up with the labelled rows on the concat below.
    duplicated = [column for column in candidates.columns
                  if column in parameters.columns and column != "GaiaDR3"]
    candidates = candidates.drop(columns=["ra", "dec"] + duplicated).merge(
        parameters.drop(columns=["label"]), on="GaiaDR3", how="left")
    candidates["is_candidate"] = True

    merged = pd.concat([labelled, candidates], ignore_index=True)
    merged = merged.groupby("GaiaDR3", as_index=False).first()
    merged = merged.merge(coordinates, on="GaiaDR3", how="left")

    merged["is_labelled"] = merged["is_labelled"].fillna(False).astype(bool)
    merged["is_candidate"] = merged["is_candidate"].fillna(False).astype(bool)
    return clean_sentinels(merged)


def add_galactocentric_cartesian(catalog):
    """X/Y/Z with the Galactic centre at the origin, plus galactic l/b.

    astropy's Galactocentric carries the current IAU-endorsed R0 = 8.122 kpc
    (GRAVITY 2018) and z_sun = 20.8 pc (Bennett & Bovy 2019), and applies the
    small roll that puts the Sun above the plane — doing it by hand with
    X = d*cos(b)*cos(l) - R0 silently drops that tilt.
    """
    parallax = catalog["plx"].astype(float)
    # A negative parallax has no 1/plx distance at all; those stars keep NaN
    # coordinates instead of being silently mirrored to the other side of the Sun.
    usable = parallax > 0
    catalog["distance_kpc"] = np.where(usable, 1.0 / parallax, np.nan)  # mas -> kpc

    equatorial = ICRS(
        ra=catalog.loc[usable, "ra"].to_numpy() * u.deg,
        dec=catalog.loc[usable, "dec"].to_numpy() * u.deg,
        distance=catalog.loc[usable, "distance_kpc"].to_numpy() * u.kpc,
    )
    galactic = equatorial.transform_to(Galactic())
    galactocentric = equatorial.transform_to(Galactocentric())

    for column in ["l", "b", "X_kpc", "Y_kpc", "Z_kpc"]:
        catalog[column] = np.nan
    catalog.loc[usable, "l"] = galactic.l.to_value(u.deg)
    catalog.loc[usable, "b"] = galactic.b.to_value(u.deg)
    catalog.loc[usable, "X_kpc"] = galactocentric.x.to_value(u.kpc)
    catalog.loc[usable, "Y_kpc"] = galactocentric.y.to_value(u.kpc)
    catalog.loc[usable, "Z_kpc"] = galactocentric.z.to_value(u.kpc)

    catalog["R_galactocentric_kpc"] = np.hypot(catalog["X_kpc"], catalog["Y_kpc"])
    return catalog


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-label", type=int, default=12,
                        help="Skiff type code of the massive class (12 = B2).")
    parser.add_argument("--output", type=Path,
                        default=REPO_ROOT / "catalogs" / "0_MassiveG12_3D.csv")
    parser.add_argument("--cache-dir", type=Path,
                        default=REPO_ROOT / "catalogs" / "cds_A155",
                        help="Where the CDS .dat.gz files are kept.")
    arguments = parser.parse_args()

    print(f"CDS {CDS_URL}")
    cache_dir = download_cds_files(arguments.cache_dir)
    candidates, parameters, auxiliary = load_cds_tables(cache_dir)
    print(f"  candms {len(candidates)}x{len(candidates.columns)}  "
          f"g12par {len(parameters)}x{len(parameters.columns)}  "
          f"auxcode {len(auxiliary)}x{len(auxiliary.columns)}")

    merged = build_merged_catalog(candidates, parameters, auxiliary,
                                  arguments.max_label)
    print(f"Merged: {len(merged)} unique stars "
          f"({merged['is_labelled'].sum()} labelled massive (label <= "
          f"{arguments.max_label}), {merged['is_candidate'].sum()} candidates, "
          f"{(merged['is_labelled'] & merged['is_candidate']).sum()} in both)")

    catalog = add_galactocentric_cartesian(merged)
    print(f"  usable parallax: {(catalog['plx'] > 0).sum()}/{len(catalog)}")

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    catalog.to_csv(arguments.output, index=False)
    print(f"Written {arguments.output} ({len(catalog)} rows, {len(catalog.columns)} columns)")


if __name__ == "__main__":
    main()
