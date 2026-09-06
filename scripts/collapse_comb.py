#!/usr/bin/env python
"""Colapsa el peine armónico del ACF en sus períodos, antes de la red.

El ACF de una señal periódica tiene un pico en CADA múltiplo del período, así
que sus 2P y 3P entran a la lista de candidatos como si fueran períodos
independientes. Medido sobre la golden: 55 de los 59 picos reportados en 2P
vienen del ACF, contra 4 del LS. `msv.peaks.candidate_periods` colapsa la serie
en su fundamental y ya existía; lo que faltaba era meterla en el camino y
protegerla del caso eclipsante.

**El caso eclipsante.** La literatura reporta habitualmente el período
FOTOMÉTRICO de una binaria, que es la mitad del orbital. El ACF de una
eclipsante de eclipses desiguales tiene el peine completo — correlaciona
primario-primario en P_orb y primario-secundario en P_orb/2 — y
`label_harmonics` elige el fundamental maximizando la racha 1,2,3..., así que
se queda con P_orb/2 y el orbital deja de ser candidato. En TIC 220197273 s6 el
colapso a secas emite solo 0.6080 d y destruye el 1.2083 d, que tiene 8.2 ppt
de amplitud en su fundamental.

Por eso cada serie declarada se somete a un TEST DE DOBLADO antes de colapsar:
se ajusta una sinusoide en 2*P0 y se mide el SNR de su fundamental contra el
ruido local del espectro. Si una eclipsante tiene eclipses desiguales, su
fundamental en P_orb tiene potencia real — es justamente la asimetría entre
eclipses la que la produce. Si el 2P es un subarmónico inventado por el peine,
no hay nada ahí:

    TIC 337886863 s58  ECL   SNR(2*P0) = 9.06   -> se emite 2*P0
    TIC 220197273 s6   ECL   SNR(2*P0) = 5.19   -> se emite 2*P0
    TIC 42889751  s6   Be    SNR(2*P0) = 1.17   -> solo P0
    TIC 48217508  s26  ROT   SNR(2*P0) = 0.92   -> solo P0
    TIC 238381092 s62  ROT   SNR(2*P0) = 0.02   -> solo P0

El mismo test se hace hacia abajo, en P0/2, porque el error va en las dos
direcciones: el peine se lleva puesto tanto el orbital de la eclipsante como el
período fotométrico que la literatura reporta.

O sea que se colapsa siempre, pero nunca se borra un período que tenga potencia
propia. Un armónico de más lo descarta la red mirando el phase-fold; un
fundamental borrado por un colapso dudoso no lo recupera nadie.

    PYTHONPATH=src python scripts/collapse_comb.py --peaks results/golden/peaks.parquet \\
        --lc results/golden/lc.parquet --out results/golden/peaks_collapsed.parquet
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from msv.cleaning import clean_lightcurve
from msv.config import RESULTS_DIR
from msv.features import HIST_SIZE, amplitude_of, phase_fold_hist2d
from msv.peaks import candidate_periods
from msv.prewhiten import DEFAULT_SNR_MIN, probe, to_relative

PEAK_COLS = ["TIC", "sector", "source", "per", "power", "power_effective",
             "prominence", "width", "amplitude", "hist2d"]
# Dos candidatos que difieren menos que esto son el mismo período: emitirlos
# por separado solo le da a la regla de selección dos formas de la misma cosa
# entre las que elegir al azar.
DUPLICATE_TOLERANCE = 0.02


def deduplicate(candidates):
    """Un candidato por período, quedándose con el de mayor prominencia."""
    kept = []
    for candidate in sorted(candidates, key=lambda item: -item["prominence"]):
        if any(abs(candidate["period"] / other["period"] - 1.0) < DUPLICATE_TOLERANCE
               for other in kept):
            continue
        kept.append(candidate)
    return kept


def candidates_for_pair(peaks, time, relative_flux, snr_min):
    """Candidatos de una estrella-sector, con el rescate por potencia aplicado.

    El colapso se hace siempre, pero después se le pregunta a la curva por cada
    período que el peine se llevó puesto: se ajusta una sinusoide exactamente
    ahí y se mide el SNR de su FUNDAMENTAL contra el ruido local del espectro.
    El que tiene potencia propia vuelve como candidato.

    Es la regla en una línea: nunca borrar un período que exista. Un armónico
    de más lo descarta la red mirando el phase-fold; un fundamental borrado por
    un colapso dudoso no lo recupera nadie.

    Se prueban dos conjuntos. Los picos ABSORBIDOS por la serie — que es donde
    vive el orbital de una eclipsante de eclipses desiguales, porque su
    asimetría entre eclipses es justamente lo que le da potencia al fundamental
    en P_orb. Y P0/2 y 2*P0 aunque no sean picos: la literatura reporta
    habitualmente el período fotométrico, que es la mitad del orbital, y puede
    no haber quedado como pico propio si el eclipse secundario es poco
    profundo.
    """
    candidates = []
    diagnostics = []
    for source, block in peaks.groupby("source"):
        block = block.reset_index(drop=True)
        collapsed = candidate_periods(block)
        for row in collapsed.itertuples(index=False):
            candidates.append({
                "period": float(row.period), "kind": row.kind, "source": source,
                "power": float(row.power), "prominence": float(row.prominence),
                "width": float(row.width),
            })

        emitted = [float(row.period) for row in collapsed.itertuples(index=False)]
        series = [float(row.period) for row in collapsed.itertuples(index=False)
                  if row.kind == "fundamental"]
        if not series:
            continue

        absorbed = [row for row in block.itertuples(index=False)
                    if not any(abs(row.per / value - 1.0) < DUPLICATE_TOLERANCE
                               for value in emitted)]
        probes = [(float(row.per), "absorbido", float(row.power),
                   float(row.prominence), float(row.width)) for row in absorbed]
        reference = block.nlargest(1, "prominence").iloc[0]
        for factor, kind in ((2.0, "orbital"), (0.5, "fotometrico")):
            period = factor * series[0]
            if any(abs(period / value - 1.0) < DUPLICATE_TOLERANCE
                   for value in emitted + [item[0] for item in probes]):
                continue
            probes.append((period, kind, float(reference.power),
                           float(reference.prominence), float(reference.width)))

        for period, kind, power, prominence, width in probes:
            tested = probe(time, relative_flux, [], 1.0 / period)
            diagnostics.append({
                "source": source, "period_series": series[0], "period": period,
                "kind": kind, "snr": tested["snr"],
                "amplitude": tested["amplitude"],
                "rescued": bool(tested["snr"] >= snr_min),
            })
            if tested["snr"] >= snr_min:
                candidates.append({
                    "period": period, "kind": kind, "source": source,
                    "power": power, "prominence": prominence, "width": width,
                })
    return deduplicate(candidates), diagnostics


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--peaks", default=str(RESULTS_DIR / "golden" / "peaks.parquet"))
    parser.add_argument("--lc", default=str(RESULTS_DIR / "golden" / "lc.parquet"))
    parser.add_argument("--out", default=str(RESULTS_DIR / "golden" / "peaks_collapsed.parquet"))
    parser.add_argument("--snr-min", type=float, default=DEFAULT_SNR_MIN,
                        help="SNR del fundamental en 2*P0 para emitirlo como "
                             "candidato orbital")
    args = parser.parse_args()

    peaks = pd.read_parquet(args.peaks, columns=["TIC", "sector", "source", "per",
                                                 "power", "prominence", "width"])
    pairs = peaks[["TIC", "sector"]].drop_duplicates()
    print(f"{len(peaks)} picos en {len(pairs)} estrella-sector")

    lightcurves = pd.read_parquet(args.lc, columns=["TIC", "sector", "Time",
                                                    "flux", "flux_err"])
    grouped = {key: block for key, block in lightcurves.groupby(["TIC", "sector"])}

    rows = []
    diagnostic_rows = []
    n_missing = 0
    for (tic, sector), block in peaks.groupby(["TIC", "sector"]):
        curve = grouped.get((tic, sector))
        if curve is None or len(curve) < 20:
            n_missing += 1
            continue
        time, flux, error = clean_lightcurve(curve.Time.to_numpy(),
                                             curve.flux.to_numpy(),
                                             curve.flux_err.to_numpy())
        if len(time) < 20:
            n_missing += 1
            continue
        relative_flux, _ = to_relative(flux)
        candidates, diagnostics = candidates_for_pair(
            block.reset_index(drop=True), time, relative_flux, args.snr_min)
        curve_amplitude = amplitude_of(flux)
        for candidate in candidates:
            rows.append({
                "TIC": int(tic), "sector": int(sector),
                "source": f"{candidate['source']}-{candidate['kind']}",
                "per": candidate["period"], "power": candidate["power"],
                "power_effective": candidate["power"],
                "prominence": candidate["prominence"], "width": candidate["width"],
                "amplitude": curve_amplitude,
                "hist2d": phase_fold_hist2d(time, flux, candidate["period"]).ravel(),
            })
        for entry in diagnostics:
            diagnostic_rows.append({"TIC": int(tic), "sector": int(sector), **entry})
        if len(rows) % 2000 < len(candidates):
            print(f"  {len(rows)} candidatos...")

    table = pd.DataFrame(rows)
    flat = np.concatenate([np.asarray(hist, dtype=np.float32)
                           for hist in table.hist2d.values])
    arrow = pa.table({
        "TIC": pa.array(table.TIC.to_numpy(), type=pa.int64()),
        "sector": pa.array(table.sector.to_numpy(), type=pa.int64()),
        "source": pa.array(table.source.to_numpy()),
        "per": pa.array(table.per.to_numpy(), type=pa.float64()),
        "power": pa.array(table.power.to_numpy(), type=pa.float64()),
        "power_effective": pa.array(table.power_effective.to_numpy(), type=pa.float64()),
        "prominence": pa.array(table.prominence.to_numpy(), type=pa.float64()),
        "width": pa.array(table.width.to_numpy(), type=pa.float64()),
        "amplitude": pa.array(table.amplitude.to_numpy(), type=pa.float64()),
        "hist2d": pa.FixedSizeListArray.from_arrays(
            pa.array(flat, type=pa.float32()), HIST_SIZE),
    })
    output = Path(args.out)
    pq.write_table(arrow, output, compression="snappy")
    diagnostics_table = pd.DataFrame(diagnostic_rows)
    diagnostics_path = output.with_name(output.stem + "_rescate.csv")
    diagnostics_table.to_csv(diagnostics_path, index=False)

    print(f"\n-> {output}   {len(table)} candidatos "
          f"(de {len(peaks)} picos, {len(table) / len(peaks):.0%})")
    print(f"-> {diagnostics_path}   {len(diagnostics_table)} series declaradas")
    if n_missing:
        print(f"   {n_missing} estrella-sector sin curva utilizable")
    print("\ncandidatos por tipo:")
    print(table.source.value_counts().to_string())
    if not diagnostics_table.empty:
        print("\nrescate por potencia propia (SNR del fundamental >= "
              f"{args.snr_min:.0f}):")
        print(diagnostics_table.groupby("kind").rescued.agg(
            probados="size", rescatados="sum").to_string())


if __name__ == "__main__":
    main()
