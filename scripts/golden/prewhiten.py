#!/usr/bin/env python
"""Etapa 1b: separar las variaciones de una estrella multiperiódica.

La red clasifica UN fold, y se entrenó con estrellas cuya curva es esa única
variación. En una multiperiódica el fold en cualquier período lleva las otras
variaciones encima como dispersión, así que la imagen que recibe no es del tipo
que vio entrenando: no es que se equivoque, es que se le está preguntando algo
fuera de su dominio. Este script quita las otras componentes antes de plegar y
deja que la red opine sobre una variación a la vez.

Por cada estrella-sector:

  1. prewhitening iterativo (`msv.prewhiten.extract_components`): se extrae la
     frecuencia más fuerte con sus armónicos, se resta, y se repite sobre el
     residuo hasta que el pico deja de llegar a SNR 4.
  2. se sondean además tres frecuencias IMPUESTAS, que son las que están en
     disputa: el período publicado, el nuestro, y — para las Be de
     Labadie-Bartz+ 2022 — los centros fg1 y fg2 de sus grupos de frecuencias.
     Sondear es ajustar una sinusoide exactamente ahí y leer su amplitud: dice
     si el período está en los datos, que es distinto de si se ve en el fold.
  3. cada sonda se pliega dos veces — sobre la curva cruda (`:raw`) y sobre la
     curva con las OTRAS variaciones removidas (`:iso`) — y las dos imágenes
     salen en el parquet con el schema de picos, para que la cadena
     step_cnn_export -> step_cnn -> step_clasificar_una_red corra sin tocarla.

Una componente conmensurable con la sonda NO se remueve: el primer armónico de
una elipsoidal es lo que le da sus dos máximos por órbita, y quitarlo borraría
justamente la forma que el fold tiene que mostrar.

    PYTHONPATH=src python scripts/golden/prewhiten.py

`irregular`, `reportado` y `nivel` de step_clasificar_una_red.py NO significan
nada sobre esta salida: mezclan las filas :raw con las :iso de la misma
estrella. Lo que se lee acá es `clase`, `prob` y `log_pLPV` por fila.
"""
import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from msv.cleaning import clean_lightcurve
from msv.config import CATALOGS_DIR, RESULTS_DIR
from msv.features import HIST_SIZE, amplitude_of, phase_fold_hist2d
from msv.prewhiten import (DEFAULT_HARMONICS, DEFAULT_MAX_COMPONENTS,
                           DEFAULT_SNR_MIN, cluster_components,
                           extract_components, fit_harmonic_model,
                           from_relative, isolate_single, probe, to_relative)

LABADIE_TABLE = CATALOGS_DIR / "golden" / "v_J_AJ_163_226_table2.tsv"
COMPONENT_COLS = ["TIC", "sector", "kind", "source", "frequency", "period",
                  "amplitude_ppt", "snr", "snr_sequential", "cycles",
                  "harmonic_amplitudes", "extracted", "cluster", "cluster_size",
                  "ratio_gold", "ratio_ours", "n_points", "baseline",
                  "std_ppt", "residual_ppt", "coherent_fraction",
                  "n_components", "rayleigh_cpd"]
# Las cuatro imágenes que se le dan a la red por sonda: la curva cruda; la
# curva con las otras variaciones removidas pero conservando las conmensurables;
# la componente completamente sola, sin ni siquiera su grupo armónico; y el
# modelo. `alone` solo existe para las componentes extraídas — en una frecuencia
# impuesta no hay ninguna componente que dejar, y cae de vuelta en `iso`. La
# última es una ilustración de la FORMA y nada más: una curva sin ruido está tan
# fuera de la distribución de entrenamiento como una multiperiódica.
VARIANTS = ["raw", "iso", "alone", "model"]


def load_frequency_groups():
    """fg1/fg2 de Labadie-Bartz+ 2022: centros de sus grupos de frecuencias.

    El `period` que la golden atribuye a estas estrellas es 1/fg1, o sea el
    centro de un GRUPO — un apiñamiento de frecuencias no resueltas, no una
    señal coherente. Es el motivo por el que su fold no cierra.
    """
    if not LABADIE_TABLE.exists():
        return {}
    table = pd.read_csv(LABADIE_TABLE, sep="\t", comment="#", skiprows=[1, 2],
                        skipinitialspace=True)
    table = table[pd.to_numeric(table.TIC, errors="coerce").notna()].copy()
    table["TIC"] = table.TIC.astype(int)
    groups = {}
    for _, row in table.iterrows():
        first = pd.to_numeric(row.fg1, errors="coerce")
        second = pd.to_numeric(row.fg2, errors="coerce")
        if np.isfinite(first):
            groups[int(row.TIC)] = (float(first),
                                    float(second) if np.isfinite(second) else None)
    return groups


def build_targets(veredicto, only_multi, tics):
    if tics:
        selected = veredicto[veredicto.TIC.isin(tics)]
    elif only_multi:
        selected = veredicto[veredicto.multi.fillna(False).astype(bool)]
    else:
        selected = veredicto
    return selected.sort_values(["TIC", "sector"]).reset_index(drop=True)


def ratio_or_nan(value, reference):
    if reference is None or not np.isfinite(reference) or reference == 0:
        return np.nan
    return float(value / reference)


def probes_for_star(time, relative_flux, components, period_gold, period_ours,
                    groups, n_harmonics):
    """Sondas de una estrella-sector: las extraídas más las impuestas.

    Las extraídas se vuelven a sondear en vez de reportar el SNR con el que
    salieron: el de la extracción se mide contra un residuo que todavía
    contiene las componentes que faltaban extraer, así que PW1 y PW6 no serían
    comparables entre sí ni contra el período publicado. `snr_sequential`
    conserva el valor con el que se decidió parar.
    """
    frequencies = [("PW%d" % component["index"], component["frequency"], True,
                    component["snr"]) for component in components]

    if period_gold and np.isfinite(period_gold):
        frequencies.append(("GOLD", 1.0 / float(period_gold), False, np.nan))
    if period_ours and np.isfinite(period_ours):
        frequencies.append(("OURS", 1.0 / float(period_ours), False, np.nan))
    if groups is not None:
        first, second = groups
        frequencies.append(("FG1", first, False, np.nan))
        if second is not None:
            frequencies.append(("FG2", second, False, np.nan))

    rows = []
    for kind, frequency, extracted, snr_sequential in frequencies:
        result = probe(time, relative_flux, components, frequency,
                       n_harmonics=n_harmonics)
        rows.append({
            "kind": kind,
            "extracted": extracted,
            "snr_sequential": snr_sequential,
            "frequency": result["frequency"],
            "amplitude": result["amplitude"],
            "harmonics": result["amplitude_harmonics"],
            "snr": result["snr"],
            "cycles": result["cycles"],
            "isolated": result["isolated"],
        })
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lc", default=str(RESULTS_DIR / "golden" / "lc.parquet"))
    parser.add_argument("--veredicto",
                        default=str(RESULTS_DIR / "golden" / "veredicto_2P.csv"))
    parser.add_argument("--out-dir", default=str(RESULTS_DIR / "golden"))
    parser.add_argument("--tic", type=int, nargs="+", default=None,
                        help="TIC a procesar; default los marcados multi")
    parser.add_argument("--all", action="store_true",
                        help="todas las estrella-sector del veredicto, no solo "
                             "las multi")
    parser.add_argument("--n-max", type=int, default=DEFAULT_MAX_COMPONENTS)
    parser.add_argument("--n-harmonics", type=int, default=DEFAULT_HARMONICS)
    parser.add_argument("--snr-min", type=float, default=DEFAULT_SNR_MIN)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    veredicto = pd.read_csv(args.veredicto)
    targets = build_targets(veredicto, not args.all, args.tic)
    groups_by_tic = load_frequency_groups()
    print(f"{len(targets)} estrella-sector, {targets.TIC.nunique()} TIC")

    lightcurves = pd.read_parquet(
        args.lc, filters=[("TIC", "in", sorted(targets.TIC.unique().tolist()))],
        columns=["TIC", "sector", "Time", "flux", "flux_err"])

    component_rows = []
    parquet_rows = []
    curves = {}
    for row in targets.itertuples(index=False):
        tic, sector = int(row.TIC), int(row.sector)
        block = lightcurves[(lightcurves.TIC == tic) & (lightcurves.sector == sector)]
        if len(block) < 20:
            print(f"  TIC {tic} s{sector}: sin curva")
            continue
        time, flux, _ = clean_lightcurve(block.Time.to_numpy(),
                                         block.flux.to_numpy(),
                                         block.flux_err.to_numpy())
        relative_flux, mean_flux = to_relative(flux)
        components, residual = extract_components(
            time, relative_flux, n_max=args.n_max,
            n_harmonics=args.n_harmonics, snr_min=args.snr_min)

        probes = probes_for_star(time, relative_flux, components,
                                 row.period_gold, row.per_nuestro,
                                 groups_by_tic.get(tic), args.n_harmonics)
        baseline = float(time.max() - time.min())
        rayleigh = 1.0 / baseline
        clusters = cluster_components(components, rayleigh)
        cluster_of = {"PW%d" % component["index"]: label
                      for component, label in zip(components, clusters)}
        cluster_size = {label: clusters.count(label) for label in set(clusters)}
        raw_amplitude = amplitude_of(flux)

        for entry in probes:
            source = entry["kind"]
            label = cluster_of.get(source)
            component_rows.append({
                "TIC": tic, "sector": sector, "kind": source, "source": source,
                "frequency": entry["frequency"], "period": 1.0 / entry["frequency"],
                "amplitude_ppt": entry["amplitude"], "snr": entry["snr"],
                "snr_sequential": entry["snr_sequential"],
                "cycles": entry["cycles"],
                "harmonic_amplitudes": " ".join(f"{value:.3f}"
                                                for value in entry["harmonics"]),
                "extracted": entry["extracted"],
                "cluster": label if label is not None else np.nan,
                "cluster_size": cluster_size.get(label, np.nan),
                "ratio_gold": ratio_or_nan(1.0 / entry["frequency"], row.period_gold),
                "ratio_ours": ratio_or_nan(1.0 / entry["frequency"], row.per_nuestro),
                "n_points": len(time), "baseline": baseline,
                "std_ppt": float(relative_flux.std()),
                "residual_ppt": float(residual.std()),
                "coherent_fraction": float(1.0 - residual.var() / relative_flux.var()),
                "n_components": len(components),
                "rayleigh_cpd": rayleigh,
            })

            period = 1.0 / entry["frequency"]
            model = entry["isolated"] - np.mean(entry["isolated"])
            model_only, _ = fit_harmonic_model(time, model, entry["frequency"],
                                               args.n_harmonics)
            alone = (isolate_single(relative_flux, components,
                                    int(source[2:]) - 1)
                     if entry["extracted"] else entry["isolated"])
            by_variant = {"raw": relative_flux, "iso": entry["isolated"],
                          "alone": alone, "model": model_only}
            for variant in VARIANTS:
                in_flux_units = from_relative(by_variant[variant], mean_flux)
                parquet_rows.append({
                    "TIC": tic, "sector": sector, "source": f"{source}:{variant}",
                    "per": period, "power": entry["amplitude"],
                    "power_effective": entry["amplitude"],
                    "prominence": entry["snr"], "width": float(entry["cycles"]),
                    "amplitude": raw_amplitude if variant == "raw"
                                 else amplitude_of(in_flux_units),
                    "hist2d": phase_fold_hist2d(time, in_flux_units, period).ravel(),
                })

        curves[(tic, sector)] = {
            "time": time, "relative_flux": relative_flux, "mean_flux": mean_flux,
            "residual": residual,
            "components": [{key: value for key, value in component.items()}
                           for component in components],
        }
        print(f"  TIC {tic} s{sector}: {len(components)} componentes, "
              f"{relative_flux.std():.2f} -> {residual.std():.2f} ppt")

    components_table = pd.DataFrame(component_rows)[COMPONENT_COLS]
    components_path = out_dir / "prewhiten_components.csv"
    components_table.to_csv(components_path, index=False)

    peaks = pd.DataFrame(parquet_rows)
    flat = np.concatenate([np.asarray(hist, dtype=np.float32)
                           for hist in peaks.hist2d.values])
    table = pa.table({
        "TIC": pa.array(peaks.TIC.to_numpy(), type=pa.int64()),
        "sector": pa.array(peaks.sector.to_numpy(), type=pa.int64()),
        "source": pa.array(peaks.source.to_numpy()),
        "per": pa.array(peaks.per.to_numpy(), type=pa.float64()),
        "power": pa.array(peaks.power.to_numpy(), type=pa.float64()),
        "power_effective": pa.array(peaks.power_effective.to_numpy(), type=pa.float64()),
        "prominence": pa.array(peaks.prominence.to_numpy(), type=pa.float64()),
        "width": pa.array(peaks.width.to_numpy(), type=pa.float64()),
        "amplitude": pa.array(peaks.amplitude.to_numpy(), type=pa.float64()),
        "hist2d": pa.FixedSizeListArray.from_arrays(
            pa.array(flat, type=pa.float32()), HIST_SIZE),
    })
    peaks_path = out_dir / "prewhiten_peaks.parquet"
    pq.write_table(table, peaks_path, compression="snappy")

    curves_path = out_dir / "prewhiten_curves.pkl"
    with open(curves_path, "wb") as handle:
        pickle.dump(curves, handle)

    print(f"\n-> {components_path}   {len(components_table)} sondas")
    variants = ", ".join(VARIANTS)
    print(f"-> {peaks_path}   {len(peaks)} folds ({variants})")
    print(f"-> {curves_path}   {len(curves)} estrella-sector")


if __name__ == "__main__":
    main()
