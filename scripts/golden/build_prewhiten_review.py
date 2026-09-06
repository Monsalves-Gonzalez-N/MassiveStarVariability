#!/usr/bin/env python
"""Etapa 1b: el PDF de la separación de variaciones, una página por estrella-sector.

Contesta las dos preguntas de la revisión con la misma figura:

  ¿por qué el paper reporta un período que en el fold no se ve?
      El espectro de amplitud con la frecuencia publicada marcada. Si ahí hay
      amplitud pero el fold no cierra, el período está en los datos y lo que
      falla es el fold — porque encima hay otras variaciones, o porque lo que
      el paper reporta es el centro de un GRUPO de frecuencias y no una señal
      coherente (las Be de Labadie-Bartz+ 2022: `fg1`). La columna `grupo` de
      la tabla dice cuántas componentes no resueltas cayeron en el mismo lugar.

  ¿se puede darle a la red una variación a la vez?
      La fila de folds del medio: el mismo período plegado sobre la curva cruda
      y sobre la curva con las otras variaciones removidas, con la clase y el
      `log_pLPV` que la red devuelve en cada caso.

La segunda fila es la descomposición apilada sobre los primeros días: cada
componente por separado, que es la forma de ver a ojo las dos variaciones que
en la curva cruda están sumadas.

Se corre después de `prewhiten.py` y de la cadena de la CNN sobre su parquet.

    PYTHONPATH=src python scripts/golden/build_prewhiten_review.py
"""
import argparse
import pickle
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _preview import open_in_preview  # noqa: E402

from msv.config import RESULTS_DIR
from msv.prewhiten import amplitude_spectrum, frequency_grid, isolate
from msv.viz import plot_phase_fold

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RASTER_DPI = 130
PAGE_SIZE = (14.0, 16.5)
ZOOM_DAYS = 8.0
PALETTE = ["tab:red", "tab:blue", "tab:orange", "tab:green", "tab:purple",
           "tab:brown"]
IMPOSED_COLORS = {"GOLD": "tab:green", "OURS": "tab:red",
                  "FG1": "0.45", "FG2": "0.45"}
FOLD_BINS = 40
# Fila 3: el período publicado, crudo y aislado. Fila 4: el nuestro igual, y
# si la estrella es una Be de Labadie-Bartz, los centros de sus dos grupos.
FOLD_ROWS = [["GOLD:raw", "GOLD:iso"],
             ["OURS:raw", "OURS:iso", "FG1:iso", "FG2:iso"]]


def rasterize_points(axis):
    for line in axis.get_lines():
        if line.get_linestyle() == "None":
            line.set_rasterized(True)


def draw_light_curve(axis, time, flux, total_model, title):
    axis.plot(time, flux, ".k", ms=1.2, alpha=0.45, rasterized=True)
    if total_model is not None:
        axis.plot(time, total_model, "-", color="tab:red", lw=0.9, alpha=0.85)
    axis.set_xlabel("BTJD")
    axis.set_ylabel("Flujo relativo [ppt]")
    axis.set_title(title, fontsize=9.5, loc="left")
    axis.grid(alpha=0.3)


def draw_decomposition(axis, time, flux, components, residual):
    """Las componentes apiladas y separadas en vertical, sobre unos días.

    La curva cruda es la suma de todo y por eso no se le ve la estructura;
    apiladas, cada variación se lee sola y se ve cuál es rápida y cuál lenta.
    """
    window = time <= time.min() + ZOOM_DAYS
    traces = [("curva", flux, "k")]
    for position, component in enumerate(components):
        traces.append((f"PW{component['index']} P={component['period']:.3f} d "
                       f"A={component['amplitude']:.2f}",
                       component["model"], PALETTE[position % len(PALETTE)]))
    traces.append(("residuo", residual, "0.55"))

    spans = [np.ptp(values[window]) for _, values, _ in traces]
    spacing = 1.25 * max(spans) if spans else 1.0
    for position, (label, values, color) in enumerate(traces):
        offset = -position * spacing
        centred = values[window] - np.mean(values[window]) + offset
        style = "." if position == 0 else "-"
        axis.plot(time[window], centred, style, color=color, ms=1.2, lw=0.9,
                  alpha=0.85 if position else 0.45, rasterized=(position == 0))
        axis.annotate(label, (time[window].min(), offset), fontsize=7,
                      color=color, va="bottom", ha="left",
                      xytext=(2, 0.20 * spacing), textcoords="offset points",
                      bbox=dict(facecolor="white", edgecolor="none", alpha=0.7,
                                pad=0.8))
    axis.set_ylim(-(len(traces) - 0.35) * spacing, 0.75 * spacing)
    axis.set_yticks([])
    axis.set_xlabel("BTJD")
    axis.set_title(f"Descomposicion, primeros {ZOOM_DAYS:.0f} dias "
                   f"(misma escala vertical en todas)", fontsize=9, loc="left")
    axis.grid(alpha=0.25, axis="x")


def draw_spectrum(axis, time, flux, probes):
    grid = frequency_grid(time)
    spectrum = amplitude_spectrum(time, flux - np.mean(flux), grid)
    axis.plot(grid, spectrum, "-k", lw=0.7)
    top = spectrum.max() * 1.15
    axis.set_ylim(0, top)
    for row in probes[probes.extracted].itertuples(index=False):
        axis.plot([row.frequency], [row.amplitude_ppt], "v", color="tab:blue", ms=5)
        axis.annotate(row.kind, (row.frequency, row.amplitude_ppt),
                      textcoords="offset points", xytext=(0, 6), fontsize=6.5,
                      ha="center", color="tab:blue")
    imposed = probes[~probes.extracted]
    for position, row in enumerate(imposed.itertuples(index=False)):
        axis.axvline(row.frequency, color=IMPOSED_COLORS[row.kind], lw=1.1,
                     ls="--" if row.kind.startswith("FG") else "-", alpha=0.85)
        axis.annotate(row.kind, (row.frequency, top * (0.96 - 0.08 * position)),
                      textcoords="offset points", xytext=(3, 0), fontsize=7.5,
                      ha="left", va="top", color=IMPOSED_COLORS[row.kind])
    axis.set_xlabel("Frecuencia [1/d]")
    axis.set_ylabel("Amplitud [ppt]")
    axis.set_xlim(0, min(grid.max(), 1.3 * probes.frequency.max()))
    axis.grid(alpha=0.3)
    axis.set_title("Espectro de amplitud de la curva cruda", fontsize=9, loc="left")


def component_table(probes, head):
    lines = [f"{int(head.n_components)} componentes   "
             f"Rayleigh {head.rayleigh_cpd:.4f}/d   "
             f"coherente {head.coherent_fraction:.0%}", "",
             f"{'':5s} {'P [d]':>8s} {'f [1/d]':>8s} {'A [ppt]':>8s} "
             f"{'SNR':>6s} {'A2/A1':>6s} {'P/Pgold':>8s} {'grupo':>6s}"]
    for row in probes.itertuples(index=False):
        harmonics = [float(value) for value in row.harmonic_amplitudes.split()]
        ratio = harmonics[0] / row.amplitude_ppt if row.amplitude_ppt > 0 else np.nan
        group = "" if not np.isfinite(row.cluster) else f"g{int(row.cluster)}"
        if np.isfinite(row.cluster_size) and row.cluster_size > 1:
            group += f"x{int(row.cluster_size)}"
        lines.append(f"{row.kind:5s} {row.period:8.4f} {row.frequency:8.4f} "
                     f"{row.amplitude_ppt:8.2f} {row.snr:6.1f} {ratio:6.1f} "
                     f"{row.ratio_gold:8.3f} {group:>6s}")
    return "\n".join(lines)


def draw_fold(axis, time, flux, row, variant, classification):
    entry = classification.get((row.TIC, row.sector, f"{row.kind}:{variant}"))
    verdict = "" if entry is None else f"  ->  {entry[0]} {entry[1]:.1f}"
    plot_phase_fold(time, flux, row.period, ax=axis, phase_bins=FOLD_BINS,
                    title=f"{row.kind}:{variant}   P={row.period:.4f} d\n"
                          f"A={row.amplitude_ppt:.2f} ppt   "
                          f"SNR={row.snr:.1f}{verdict}")
    axis.set_ylabel("ppt")
    axis.title.set_fontsize(8)
    rasterize_points(axis)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    golden = RESULTS_DIR / "golden"
    parser.add_argument("--components", default=str(golden / "prewhiten_components.csv"))
    parser.add_argument("--curves", default=str(golden / "prewhiten_curves.pkl"))
    parser.add_argument("--clasificacion",
                        default=str(golden / "prewhiten_clasificacion.csv"))
    parser.add_argument("--veredicto", default=str(golden / "veredicto_2P.csv"))
    parser.add_argument("--out", default=str(RESULTS_DIR / "figures" /
                                             "golden_prewhiten.pdf"))
    parser.add_argument("--no-open", action="store_true")
    args = parser.parse_args()

    components = pd.read_csv(args.components)
    with open(args.curves, "rb") as handle:
        curves = pickle.load(handle)
    veredicto = pd.read_csv(args.veredicto).set_index(["TIC", "sector"])

    classification = {}
    if Path(args.clasificacion).exists():
        table = pd.read_csv(args.clasificacion)
        for row in table.itertuples(index=False):
            classification[(row.TIC, row.sector, row.source)] = (row.clase,
                                                                 row.log_pLPV)

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)

    # Una fila por sonda con las dos clases al lado: es la tabla sobre la que
    # se leen los resultados, y sale de acá porque es el único paso que tiene
    # las componentes y la clasificación abiertas a la vez.
    summary = pd.read_csv(args.clasificacion) if classification else None
    if summary is not None:
        summary[["kind", "variant"]] = summary.source.str.split(":", expand=True)
        wide = summary.pivot_table(index=["TIC", "sector", "kind"],
                                   columns="variant",
                                   values=["clase", "log_pLPV"], aggfunc="first")
        wide.columns = [f"{value}_{variant}" for value, variant in wide.columns]
        merged = components.set_index(["TIC", "sector", "kind"]).join(wide)
        summary_path = Path(args.components).parent / "prewhiten_resumen.csv"
        merged.reset_index().to_csv(summary_path, index=False)
        print(f"-> {summary_path}")

    with PdfPages(output) as pdf:
        for (tic, sector), block in components.groupby(["TIC", "sector"]):
            curve = curves.get((tic, sector))
            if curve is None:
                continue
            time = curve["time"]
            flux = curve["relative_flux"]
            head = block.iloc[0]
            by_kind = block.set_index("kind")
            info = (veredicto.loc[(tic, sector)]
                    if (tic, sector) in veredicto.index else None)

            figure = plt.figure(figsize=PAGE_SIZE)
            grid = figure.add_gridspec(5, 4, height_ratios=[0.85, 1.5, 1.2, 1.1, 1.1],
                                       hspace=0.62, wspace=0.30,
                                       left=0.055, right=0.985, top=0.965,
                                       bottom=0.04)

            title = (f"TIC {tic}   sector {sector}   "
                     f"{head.n_points} puntos, {head.baseline:.1f} d   "
                     f"sigma={head.std_ppt:.2f} ppt")
            if info is not None:
                title += (f"   |   paper {info.class_gold}: P={info.period_gold:.4f} d"
                          f"   |   pipeline {info.clase_nuestra}: "
                          f"P={info.per_nuestro:.4f} d")
            total_model = flux - np.mean(flux) - curve["residual"]
            draw_light_curve(figure.add_subplot(grid[0, :]), time, flux,
                             total_model + np.mean(flux), title)
            draw_decomposition(figure.add_subplot(grid[1, :2]), time, flux,
                               curve["components"], curve["residual"])
            draw_spectrum(figure.add_subplot(grid[1, 2:]), time, flux, block)

            text_axis = figure.add_subplot(grid[2, 2:])
            text_axis.set_axis_off()
            text_axis.text(0.0, 1.0, component_table(block, head), fontsize=7,
                           family="monospace", va="top", ha="left",
                           transform=text_axis.transAxes)

            for row_index, names in enumerate(FOLD_ROWS):
                for position, name in enumerate(names):
                    kind, variant = name.split(":")
                    if kind not in by_kind.index:
                        continue
                    row = by_kind.loc[kind].copy()
                    row["TIC"], row["sector"], row["kind"] = tic, sector, kind
                    values = (flux if variant == "raw"
                              else isolate(flux, curve["components"], row.frequency))
                    draw_fold(figure.add_subplot(grid[2 + row_index, position]),
                              time, values, row, variant, classification)

            extracted = block[block.extracted].sort_values("amplitude_ppt",
                                                           ascending=False)
            for position, (_, row) in enumerate(extracted.head(4).iterrows()):
                row = row.copy()
                row["TIC"], row["sector"] = tic, sector
                values = isolate(flux, curve["components"], row.frequency)
                draw_fold(figure.add_subplot(grid[4, position]), time, values,
                          row, "iso", classification)

            pdf.savefig(figure, dpi=RASTER_DPI)
            plt.close(figure)
            print(f"  TIC {tic} s{sector}")

    print(f"\n-> {output}")
    if not args.no_open:
        open_in_preview(output)


if __name__ == "__main__":
    main()
