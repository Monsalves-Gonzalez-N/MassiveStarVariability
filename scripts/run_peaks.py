#!/usr/bin/env python
"""Pipeline LS + ACF -> peaks parquet (con hist2d + amplitude por peak).

Por cada par (TIC, sector) del parquet de curvas de luz:
  1. clean_lightcurve()  — rampas de telemetría + sigma-clip en gaps/bordes
  2. ls_periodogram + acf_periodogram
  3. select_peaks_ls / select_peaks_acf  — find_peaks sobre power crudo,
     ventana temporal mínima (--min-peak-sep-days) en vez de smoothing
  4. phase_fold_hist2d_log + amplitude -> filas del parquet de salida

Uso (env base, sin TF):
  python scripts/run_peaks.py --lc lightcurves_all_OGLE.parquet \
      --out results/peaks.parquet --min-peak-sep-days 0.5 [--ray --num-cpus 8]
"""
import argparse
import sys

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from msv import config
from msv.cleaning import clean_lightcurve
from msv.features import HIST_SIZE, amplitude_of, phase_fold_hist2d_log
from msv.peaks import select_peaks
from msv.periodograms import acf_periodogram, ls_periodogram

PEAK_COLS = ["TIC", "sector", "source", "per", "power", "power_effective",
             "prominence", "width", "amplitude", "hist2d"]


def process_pair(tic, sector, lc_path, args_dict):
    """Procesa un par (TIC, sector). Devuelve DataFrame de peaks, None si no
    hay picos/datos, o ('ERR', tic, sector, msg)."""
    try:
        lc = pd.read_parquet(
            lc_path,
            filters=[("TIC", "=", int(tic)), ("sector", "=", int(sector))],
            columns=["Time", "flux", "flux_err"],
        )
        if len(lc) < 20:
            return None
        t = lc["Time"].to_numpy()
        f = lc["flux"].to_numpy()
        e = lc["flux_err"].to_numpy()

        if not args_dict["no_clean"]:
            t, f, e = clean_lightcurve(t, f, e)
            if len(t) < 20:
                return None

        parts = []
        if "ls" in args_dict["sources"]:
            df_ls = ls_periodogram(t, f, e, oversample=args_dict["oversample"])
            pk = select_peaks(df_ls, "LS", top_n=args_dict["top_n"])
            if not pk.empty:
                # power_effective LS: power - window en la posición del pico
                win = df_ls["window"].to_numpy()
                pos = np.abs(df_ls["per"].to_numpy()[None, :]
                             - pk["per"].to_numpy()[:, None]).argmin(axis=1)
                pk["power_effective"] = pk["power"].to_numpy() - win[pos]
                pk["source"] = "LS"
                parts.append(pk)

        if "acf" in args_dict["sources"]:
            df_acf = acf_periodogram(t, f, e, fill_gaps=args_dict["fill_gaps"])
            pk = select_peaks(df_acf, "ACF",
                              min_peak_sep_days=args_dict["min_peak_sep_days"],
                              top_n=args_dict["top_n"])
            if not pk.empty:
                pk["power_effective"] = pk["power"]
                pk["source"] = "ACF"
                parts.append(pk)

        if not parts:
            return None
        res = pd.concat(parts, ignore_index=True)

        amplitude = amplitude_of(f)
        res["hist2d"] = [phase_fold_hist2d_log(t, f, float(p)).ravel()
                         for p in res["per"].to_numpy()]
        res["amplitude"] = amplitude
        res["TIC"] = int(tic)
        res["sector"] = int(sector)
        return res[PEAK_COLS]
    except Exception as ex:
        return ("ERR", int(tic), int(sector), repr(ex))


def df_to_table(df):
    """DataFrame -> Arrow Table con hist2d como FixedSizeList<float32,1024>."""
    flat = np.concatenate([np.asarray(h, dtype=np.float32) for h in df["hist2d"].values])
    hist_arr = pa.FixedSizeListArray.from_arrays(pa.array(flat, type=pa.float32()), HIST_SIZE)
    return pa.table({
        "TIC": pa.array(df["TIC"].to_numpy(), type=pa.int64()),
        "sector": pa.array(df["sector"].to_numpy(), type=pa.int64()),
        "source": pa.array(df["source"].to_numpy()),
        "per": pa.array(df["per"].to_numpy(), type=pa.float64()),
        "power": pa.array(df["power"].to_numpy(), type=pa.float64()),
        "power_effective": pa.array(df["power_effective"].to_numpy(), type=pa.float64()),
        "prominence": pa.array(df["prominence"].to_numpy(), type=pa.float64()),
        "width": pa.array(df["width"].to_numpy(), type=pa.float64()),
        "amplitude": pa.array(df["amplitude"].to_numpy(), type=pa.float64()),
        "hist2d": hist_arr,
    })


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lc", default=str(config.LC_PARQUET_OGLE),
                    help="parquet de curvas de luz (Time, flux, flux_err, TIC, sector)")
    ap.add_argument("--out", default=str(config.RESULTS_DIR / "peaks.parquet"))
    ap.add_argument("--min-peak-sep-days", type=float, default=config.MIN_PEAK_SEP_DAYS,
                    help="ventana temporal mínima entre picos del ACF [días]")
    ap.add_argument("--sources", nargs="+", default=["ls", "acf"], choices=["ls", "acf"])
    ap.add_argument("--top-n", type=int, default=None, help="máx. picos por periodograma")
    ap.add_argument("--oversample", type=int, default=5)
    ap.add_argument("--fill-gaps", default=False,
                    help="relleno de gaps del ACF vía astrobase (p.ej. 'noiselevel')")
    ap.add_argument("--no-clean", action="store_true",
                    help="omite clean_lightcurve (rampas + sigma-clip)")
    ap.add_argument("--no-exclude-bad", action="store_true",
                    help="NO excluir pares con flux <= 0")
    ap.add_argument("--ray", action="store_true", help="paralelizar con Ray")
    ap.add_argument("--num-cpus", type=int, default=8)
    args = ap.parse_args()

    args_dict = dict(
        sources=[s.lower() for s in args.sources],
        min_peak_sep_days=args.min_peak_sep_days,
        top_n=args.top_n, oversample=args.oversample,
        fill_gaps=args.fill_gaps, no_clean=args.no_clean,
    )

    pairs = pd.read_parquet(args.lc, columns=["TIC", "sector"]).drop_duplicates()
    if not args.no_exclude_bad:
        bad = config.find_bad_flux_pairs(args.lc)
        n0 = len(pairs)
        pairs = pairs[~pairs.apply(lambda r: (int(r.TIC), int(r.sector)) in bad, axis=1)]
        print(f"Excluidos {n0 - len(pairs)} pares con flux <= 0")
    print(f"Pares (TIC, sector) a procesar: {len(pairs)}")

    writer = None
    n_ok = n_none = n_err = 0
    errors = []

    def handle(res):
        nonlocal writer, n_ok, n_none, n_err
        if res is None:
            n_none += 1
            return
        if isinstance(res, tuple) and res[0] == "ERR":
            n_err += 1
            errors.append(res[1:])
            return
        tbl = df_to_table(res)
        if writer is None:
            writer = pq.ParquetWriter(args.out, tbl.schema, compression="snappy")
        writer.write_table(tbl)
        n_ok += 1

    if args.ray:
        import ray
        if not ray.is_initialized():
            ray.init(num_cpus=args.num_cpus, ignore_reinit_error=True,
                     include_dashboard=False)
        remote_fn = ray.remote(process_pair)
        futures = [remote_fn.remote(int(r.TIC), int(r.sector), args.lc, args_dict)
                   for r in pairs.itertuples(index=False)]
        remaining = futures
        while remaining:
            done, remaining = ray.wait(remaining, num_returns=min(64, len(remaining)))
            for fut in done:
                handle(ray.get(fut))
            print(f"  con_picos={n_ok} sin_picos={n_none} err={n_err} "
                  f"pendientes={len(remaining)}")
    else:
        for i, r in enumerate(pairs.itertuples(index=False)):
            handle(process_pair(int(r.TIC), int(r.sector), args.lc, args_dict))
            if (i + 1) % 20 == 0:
                print(f"  {i + 1}/{len(pairs)} | con_picos={n_ok} err={n_err}")

    if writer is not None:
        writer.close()

    print(f"\n-> {args.out}")
    print(f"   pares con picos: {n_ok} | sin picos: {n_none} | errores: {n_err}")
    if errors[:5]:
        print("   primeros errores:", errors[:5])
    return 0 if n_err == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
