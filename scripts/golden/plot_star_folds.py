#!/usr/bin/env python
"""Los folds independientes de UNA estrella, una página por sector.

Cada componente extraída se pliega en su propio período sobre la curva con las
otras removidas: un panel por variación, que es lo que la red ve cuando se le
pregunta de a una. Al final, una página que compara los sectores entre sí.

Esa última página es el discriminante entre rotación y pulsación, que en el
espectro de un solo sector no se puede separar: una señal rotacional vuelve en
la MISMA frecuencia con la misma amplitud en cada sector, porque la estrella
sigue girando igual. Un grupo de frecuencias de una Be no: vuelve la envolvente
y no sus miembros, y la amplitud del grupo cambia entre épocas.

    PYTHONPATH=src python scripts/golden/plot_star_folds.py --tic 42889751
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

from msv.config import CATALOGS_DIR, RESULTS_DIR
from msv.prewhiten import (amplitude_spectrum, frequency_grid, isolate,
                           isolate_single, probe)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

LABADIE_TABLE = CATALOGS_DIR / "golden" / "v_J_AJ_163_226_table2.tsv"
RASTER_DPI = 140
PAGE_SIZE = (14.0, 16.0)
FOLDS_PER_ROW = 4
FOLD_BINS = 40
IMPOSED_COLORS = {"GOLD": "tab:green", "OURS": "tab:red",
                  "FG1": "0.4", "FG2": "0.4"}
SECTOR_COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:purple"]


def star_header(tic):
    """Fila de Labadie-Bartz+ 2022 de la estrella, si está."""
    if not LABADIE_TABLE.exists():
        return None
    table = pd.read_csv(LABADIE_TABLE, sep="\t", comment="#", skiprows=[1, 2],
                        skipinitialspace=True)
    table = table[pd.to_numeric(table.TIC, errors="coerce").notna()].copy()
    table["TIC"] = table.TIC.astype(int)
    rows = table[table.TIC == tic]
    return None if rows.empty else rows.iloc[0]


def rasterize_points(axis):
    for line in axis.get_lines():
        if line.get_linestyle() == "None":
            line.set_rasterized(True)


def draw_spectrum(axis, time, flux, probes, limit=None):
    grid = frequency_grid(time)
    spectrum = amplitude_spectrum(time, flux - np.mean(flux), grid)
    axis.plot(grid, spectrum, "-k", lw=0.8)
    top = spectrum.max() * 1.18
    axis.set_ylim(0, top)
    for row in probes[probes.extracted].itertuples(index=False):
        axis.plot([row.frequency], [row.amplitude_ppt], "v", color="tab:blue", ms=6)
        axis.annotate(row.kind, (row.frequency, row.amplitude_ppt),
                      textcoords="offset points", xytext=(0, 7), fontsize=7.5,
                      ha="center", color="tab:blue")
    for position, row in enumerate(probes[~probes.extracted].itertuples(index=False)):
        axis.axvline(row.frequency, color=IMPOSED_COLORS[row.kind], lw=1.2,
                     ls="--" if row.kind.startswith("FG") else "-", alpha=0.85)
        axis.annotate(row.kind, (row.frequency, top * (0.97 - 0.07 * position)),
                      textcoords="offset points", xytext=(3, 0), fontsize=8,
                      ha="left", va="top", color=IMPOSED_COLORS[row.kind])
    axis.set_xlim(0, limit or min(grid.max(), 1.4 * probes.frequency.max()))
    axis.set_xlabel("Frecuencia [1/d]")
    axis.set_ylabel("Amplitud [ppt]")
    axis.grid(alpha=0.3)
    return spectrum


def draw_fold(axis, time, values, model, row, classes, show_points=True):
    """Una sola variabilidad: la curva del modelo, y detrás lo que queda de dato.

    Los puntos NO son otra variabilidad: son el residuo — lo que ninguna
    componente coherente explica — más el ruido. Por eso la línea roja es el
    protagonista acá y los puntos van en gris: la variación es la línea, y la
    nube dice cuánta dispersión tiene alrededor.
    """
    period = row.period
    phase = (np.asarray(time) / period) % 1.0
    centred = np.asarray(values) - np.mean(values)
    if show_points:
        for shift in (0.0, 1.0):
            axis.plot(phase + shift, centred, ".", color="0.70", ms=1.6,
                      alpha=0.45, rasterized=True)
        span = np.percentile(np.abs(centred), 99.5)
        axis.set_ylim(-1.15 * span, 1.15 * span)
    order = np.argsort(phase)
    for shift in (0.0, 1.0):
        axis.plot(phase[order] + shift, np.asarray(model)[order], "-",
                  color="tab:red", lw=2.0)
    axis.axvline(1.0, color="0.85", lw=0.8)
    axis.set_xlim(0, 2)
    axis.set_xlabel("Phase")
    axis.set_ylabel("ppt")
    axis.grid(alpha=0.3)

    group = ""
    if np.isfinite(row.cluster):
        group = f"   g{int(row.cluster)}"
        if row.cluster_size > 1:
            group += f" (x{int(row.cluster_size)})"
    verdicts = "   ".join(f"{label} -> {clase} {value:.1f}"
                          for label, clase, value in classes)
    axis.set_title(f"{row.kind}   P={period:.4f} d   f={row.frequency:.4f}/d{group}\n"
                   f"A={row.amplitude_ppt:.2f} ppt   SNR={row.snr:.1f}\n{verdicts}",
                   fontsize=8.5)


def fold_classes(classification, tic, sector, kind, variant):
    """(etiqueta, clase, log_pLPV) para la variante mostrada y para el modelo."""
    out = []
    for label, name in (("sola" if variant == "alone" else "aislada", variant),
                        ("modelo", "model")):
        entry = classification.get((tic, sector, f"{kind}:{name}"))
        if entry is not None:
            out.append((label, entry[0], entry[1]))
    return out


def sector_page(pdf, tic, sector, curve, block, classification, header, info,
                show_points=True):
    time = curve["time"]
    flux = curve["relative_flux"]
    components = curve["components"]
    extracted = block[block.extracted].sort_values("frequency")
    imposed = block[~block.extracted]
    head = block.iloc[0]

    n_folds = len(extracted) + len(imposed)
    n_rows = 2 + int(np.ceil(n_folds / FOLDS_PER_ROW))
    figure = plt.figure(figsize=(PAGE_SIZE[0], 3.0 + 2.6 * (n_rows - 1)))
    grid = figure.add_gridspec(n_rows, FOLDS_PER_ROW,
                               height_ratios=[0.9, 1.15] + [1.15] * (n_rows - 2),
                               hspace=0.68, wspace=0.30,
                               left=0.055, right=0.985, top=0.955, bottom=0.045)

    title = f"TIC {tic}   sector {sector}"
    if header is not None:
        title += (f"   {str(header.OName).strip()}   {str(header.SpT).strip()}   "
                  f"signals={str(header.signals).strip()}  Ns={header.Ns}  "
                  f"fg1={header.fg1}  fg2={header.fg2}  /d")
    subtitle = (f"{head.n_points} puntos, {head.baseline:.1f} d   "
                f"sigma={head.std_ppt:.2f} ppt   coherente {head.coherent_fraction:.0%}   "
                f"Rayleigh {head.rayleigh_cpd:.4f}/d")
    if info is not None:
        subtitle += (f"   |   paper {info.class_gold} P={info.period_gold:.4f} d"
                     f"   |   pipeline {info.clase_nuestra} P={info.per_nuestro:.4f} d")

    curve_axis = figure.add_subplot(grid[0, :])
    curve_axis.plot(time, flux, ".k", ms=1.3, alpha=0.5, rasterized=True)
    curve_axis.plot(time, flux - np.mean(flux) - curve["residual"] + np.mean(flux),
                    "-", color="tab:red", lw=0.9, alpha=0.85)
    curve_axis.set_xlabel("BTJD")
    curve_axis.set_ylabel("Flujo relativo [ppt]")
    curve_axis.set_title(f"{title}\n{subtitle}", fontsize=9.5, loc="left")
    curve_axis.grid(alpha=0.3)

    spectrum_axis = figure.add_subplot(grid[1, :])
    draw_spectrum(spectrum_axis, time, flux, block)
    spectrum_axis.set_title(
        "Espectro de amplitud de la curva cruda.  PW = componente extraida, "
        "GOLD = periodo publicado, OURS = el del pipeline\n"
        "Abajo, un panel por variacion: la LINEA ROJA es esa variacion sola. "
        "Los puntos grises son el residuo — lo que ninguna componente explica — "
        "y no otra variacion",
        fontsize=9, loc="left")

    # Las componentes se muestran SOLAS (`alone`): sin las otras y sin su grupo
    # armónico, que es lo que quiere decir fold independiente. Las frecuencias
    # impuestas van con `iso`, conservando lo conmensurable: en OURS el armónico
    # es justo lo que hay que ver, porque es de donde sale la forma de ELL.
    ordered = []
    for row in extracted.itertuples(index=False):
        index = int(row.kind[2:]) - 1
        ordered.append((row, "alone", isolate_single(flux, components, index),
                        components[index]["model"]))
    for row in imposed.itertuples(index=False):
        result = probe(time, flux, components, row.frequency)
        ordered.append((row, "iso", result["isolated"], result["model"]))

    for position, (row, variant, values, model) in enumerate(ordered):
        axis = figure.add_subplot(grid[2 + position // FOLDS_PER_ROW,
                                       position % FOLDS_PER_ROW])
        draw_fold(axis, time, values, model, row,
                  fold_classes(classification, tic, sector, row.kind, variant),
                  show_points=show_points)

    pdf.savefig(figure, dpi=RASTER_DPI)
    plt.close(figure)


def comparison_page(pdf, tic, sectors, curves, components, header):
    """Los sectores enfrentados: qué vuelve igual y qué no."""
    figure = plt.figure(figsize=(PAGE_SIZE[0], 10.5))
    grid = figure.add_gridspec(3, 2, height_ratios=[1.2, 1.2, 1.0],
                               hspace=0.5, wspace=0.25,
                               left=0.06, right=0.98, top=0.93, bottom=0.06)

    spectra_axis = figure.add_subplot(grid[0, :])
    for position, sector in enumerate(sectors):
        curve = curves[(tic, sector)]
        time, flux = curve["time"], curve["relative_flux"]
        frequencies = frequency_grid(time)
        spectrum = amplitude_spectrum(time, flux - np.mean(flux), frequencies)
        spectra_axis.plot(frequencies, spectrum, "-", lw=0.9,
                          color=SECTOR_COLORS[position % len(SECTOR_COLORS)],
                          label=f"sector {sector}  (sigma={flux.std():.1f} ppt)")
    if header is not None:
        for label, value in (("fg1", header.fg1), ("fg2", header.fg2)):
            if np.isfinite(pd.to_numeric(value, errors="coerce")):
                spectra_axis.axvline(float(value), color="0.4", ls="--", lw=1.1)
                spectra_axis.annotate(label, (float(value), spectra_axis.get_ylim()[1]),
                                      textcoords="offset points", xytext=(3, -10),
                                      fontsize=8, color="0.35")
    block = components[components.TIC == tic]
    spectra_axis.set_xlim(0, 1.4 * block[block.extracted].frequency.max())
    spectra_axis.set_xlabel("Frecuencia [1/d]")
    spectra_axis.set_ylabel("Amplitud [ppt]")
    spectra_axis.legend(fontsize=8)
    spectra_axis.grid(alpha=0.3)
    spectra_axis.set_title("Los dos sectores superpuestos: vuelve la envolvente, "
                           "no la frecuencia", fontsize=10, loc="left")

    zoom_axis = figure.add_subplot(grid[1, :])
    for position, sector in enumerate(sectors):
        curve = curves[(tic, sector)]
        time, flux = curve["time"], curve["relative_flux"]
        frequencies = frequency_grid(time)
        spectrum = amplitude_spectrum(time, flux - np.mean(flux), frequencies)
        color = SECTOR_COLORS[position % len(SECTOR_COLORS)]
        zoom_axis.plot(frequencies, spectrum / spectrum.max(), "-", lw=1.0, color=color)
        for row in block[(block.sector == sector) & block.extracted].itertuples(index=False):
            height = np.interp(row.frequency, frequencies, spectrum) / spectrum.max()
            zoom_axis.plot([row.frequency], [height], "v", color=color, ms=6)
    if header is not None and np.isfinite(pd.to_numeric(header.fg1, errors="coerce")):
        centre = float(header.fg1)
        zoom_axis.axvline(centre, color="0.4", ls="--", lw=1.1)
        zoom_axis.set_xlim(centre * 0.55, centre * 1.45)
    zoom_axis.set_xlabel("Frecuencia [1/d]")
    zoom_axis.set_ylabel("Amplitud normalizada")
    zoom_axis.grid(alpha=0.3)
    zoom_axis.set_title("Zoom al grupo g1, cada espectro normalizado a su propio "
                        "maximo", fontsize=10, loc="left")

    text_axis = figure.add_subplot(grid[2, :])
    text_axis.set_axis_off()
    lines = ["Componentes extraidas por sector", ""]
    for sector in sectors:
        rows = block[(block.sector == sector) & block.extracted].sort_values("frequency")
        lines.append(f"  sector {sector}:  " + "   ".join(
            f"{row.frequency:.4f} ({row.amplitude_ppt:.1f})" for row in rows.itertuples(index=False)))
    lines += ["", "  f [1/d] (amplitud en ppt).  Una senal rotacional volveria en la",
              "  misma frecuencia en los dos sectores; un grupo de frecuencias no.", ""]
    for sector in sectors:
        rows = block[(block.sector == sector) & block.extracted].sort_values("frequency")
        spacing = np.diff(rows.frequency.to_numpy())
        rayleigh = rows.rayleigh_cpd.iloc[0]
        inside = spacing[spacing < 3 * rayleigh]
        if inside.size:
            lines.append(f"  sector {sector}: separacion entre componentes vecinas "
                         f"{np.median(inside) / rayleigh:.2f} Rayleigh (mediana)")
    lines += ["  Separadas por ~1 Rayleigh = el prewhitening esta cortando en pedazos",
              "  una estructura ancha. Marcan donde esta la joroba, no son modos", ""]
    for sector in sectors:
        rows = block[(block.sector == sector) & (block.kind == "GOLD")]
        if not rows.empty:
            row = rows.iloc[0]
            lines.append(f"  sector {sector}: en el periodo publicado "
                         f"A={row.amplitude_ppt:.2f} ppt, SNR={row.snr:.1f}")
        rows = block[(block.sector == sector) & (block.kind == "OURS")]
        if not rows.empty:
            row = rows.iloc[0]
            lines.append(f"  sector {sector}: en el periodo del pipeline "
                         f"A={row.amplitude_ppt:.2f} ppt, SNR={row.snr:.1f}")
    text_axis.text(0.0, 1.0, "\n".join(lines), fontsize=9, family="monospace",
                   va="top", ha="left", transform=text_axis.transAxes)

    pdf.savefig(figure, dpi=RASTER_DPI)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    golden = RESULTS_DIR / "golden"
    parser.add_argument("--tic", type=int, default=42889751)
    parser.add_argument("--components", default=str(golden / "prewhiten_components.csv"))
    parser.add_argument("--curves", default=str(golden / "prewhiten_curves.pkl"))
    parser.add_argument("--clasificacion",
                        default=str(golden / "prewhiten_clasificacion.csv"))
    parser.add_argument("--veredicto", default=str(golden / "veredicto_2P.csv"))
    parser.add_argument("--out", default=None)
    parser.add_argument("--only-model", action="store_true",
                        help="solo la curva de la variacion, sin la nube de puntos")
    parser.add_argument("--no-open", action="store_true")
    args = parser.parse_args()

    components = pd.read_csv(args.components)
    components = components[components.TIC == args.tic]
    if components.empty:
        raise SystemExit(f"TIC {args.tic} no esta en {args.components}")
    with open(args.curves, "rb") as handle:
        curves = pickle.load(handle)
    veredicto = pd.read_csv(args.veredicto).set_index(["TIC", "sector"])

    classification = {}
    if Path(args.clasificacion).exists():
        table = pd.read_csv(args.clasificacion)
        for row in table.itertuples(index=False):
            classification[(row.TIC, row.sector, row.source)] = (row.clase, row.log_pLPV)

    header = star_header(args.tic)
    output = Path(args.out) if args.out else (RESULTS_DIR / "figures" /
                                              f"folds_TIC{args.tic}.pdf")
    output.parent.mkdir(parents=True, exist_ok=True)

    sectors = sorted(components.sector.unique())
    with PdfPages(output) as pdf:
        for sector in sectors:
            curve = curves.get((args.tic, int(sector)))
            if curve is None:
                continue
            info = (veredicto.loc[(args.tic, int(sector))]
                    if (args.tic, int(sector)) in veredicto.index else None)
            sector_page(pdf, args.tic, int(sector), curve,
                        components[components.sector == sector], classification,
                        header, info, show_points=not args.only_model)
            print(f"  sector {sector}")
        if len(sectors) > 1:
            comparison_page(pdf, args.tic, [int(value) for value in sectors],
                            curves, components, header)
            print("  comparacion entre sectores")

    print(f"\n-> {output}")
    if not args.no_open:
        open_in_preview(output)


if __name__ == "__main__":
    main()
