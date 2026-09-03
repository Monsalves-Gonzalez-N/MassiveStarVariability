#!/usr/bin/env python
"""Junta picos clasificados + `candidate_periods` -> entrada del PDF de folds.

El CSV de `step_brf.py` es por PICO (`per`), y el phase-fold hay que hacerlo
con el período que sale de colapsar el peine de armónicos del ACF
(`peaks.candidate_periods`: `period` + `kind`, con el subarmónico P0/2 cuando
el peine tiene miembros impares). Este script hace ese cruce y, de paso,
guarda las curvas YA LIMPIAS en un pickle para que el fold no se dibuje sobre
la curva cruda.

    python scripts/build_phasefold_input.py results/clasificacion_review50.csv \
        results/peaks_review50.parquet results/lc_review50.parquet --outdir results
"""
import argparse
import pickle
from pathlib import Path

import pandas as pd

from msv import config
from msv.cleaning import clean_lightcurve
from msv.peaks import candidate_periods

MERGE_KEYS = ["TIC", "sector", "source", "per"]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("classification", help="CSV de step_brf.py")
    parser.add_argument("peaks", help="parquet de run_peaks.py")
    parser.add_argument("lightcurves", help="parquet de curvas (Time, flux, TIC, sector)")
    parser.add_argument("--outdir", default=str(config.RESULTS_DIR))
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    peaks = pd.read_parquet(args.peaks, columns=[c for c in MERGE_KEYS + ["power",
                            "prominence", "width"]])
    candidates = []
    for (tic, sector, source), group in peaks.groupby(["TIC", "sector", "source"]):
        table = candidate_periods(group.reset_index(drop=True))
        candidates.append(table.assign(TIC=tic, sector=sector, source=source))
    candidates = pd.concat(candidates, ignore_index=True)

    classified = pd.read_csv(args.classification)
    merged = candidates.merge(classified, on=MERGE_KEYS, how="left",
                              suffixes=("", "_clf"))
    merged = merged.rename(columns={"brf_prob": "prob", "sigma_brf": "sigma"})
    merged = merged.dropna(subset=["period", "clase"])

    out_csv = outdir / "phasefold_input.csv"
    merged.to_csv(out_csv, index=False)

    lightcurves = pd.read_parquet(args.lightcurves,
                                  columns=["TIC", "sector", "Time", "flux", "flux_err"])
    curves = {}
    for (tic, sector), group in lightcurves.groupby(["TIC", "sector"]):
        time, flux, _ = clean_lightcurve(group["Time"].to_numpy(),
                                         group["flux"].to_numpy(),
                                         group["flux_err"].to_numpy())
        curves[(int(tic), int(sector))] = (time, flux)
    out_pickle = outdir / "phasefold_curves.pkl"
    with open(out_pickle, "wb") as handle:
        pickle.dump(curves, handle)

    print(f"-> {out_csv}   {len(merged)} candidatos, "
          f"{merged.groupby(['TIC', 'sector']).ngroups} pares")
    print(merged["kind"].value_counts().to_string())
    print(f"-> {out_pickle}   {len(curves)} curvas limpias")


if __name__ == "__main__":
    main()
