#!/usr/bin/env python
"""PDF de revisión: curva de luz + clase de cada período candidato.

Una ESTRELLA por fila. A la izquierda la curva limpia en tiempo (BTJD) vs
flujo; a la derecha una barra horizontal por período candidato, ordenados por
prominencia: el largo es la probabilidad de SU clase ganadora, la barra de
error la dispersión entre las pasadas MC-dropout, y el color y la etiqueta
dicen qué clase es.

El argmax de cada período se toma **excluyendo `Rndm`**: `Rndm` no es una
clase, es la ausencia de señal en ESE fold, y dejarlo competir tapa la única
información útil de ese candidato (cuál de las clases reales se parece más).
Así cada período reporta su mejor clase entre ELL / Pulsating / E / LPV con la
probabilidad y la dispersión de esa clase.

La clase de la estrella es la de `cascade.path2_cascade`, NO el candidato de
mayor probabilidad: las probabilidades están condicionadas al período con que
se dobló, así que un `Rndm 0.99` ("doblando a este período no hay nada") no
compite con un `E 0.97` de otro período. La cascada declara `Rndm` solo si
TODOS los candidatos lo son, y si hay alguno periódico vota únicamente entre
esos. El candidato que ganó el voto va marcado con ★.

El título reporta además el período MÁS LARGO entre los candidatos que no son
`Rndm`, marcado con ◄ y siempre graficado aunque no esté entre los más
prominentes: el período largo es el que arriesga quedar fuera —- su resolución
`±P²/T` es la peor y la prominencia del ACF cae con el lag— y es justamente el
que importa cuando la curva varía en escala comparable al sector.

    python scripts/build_lc_prob_review.py results/clasificacion_review50.csv \
        --curves results/phasefold_curves.pkl
"""
import argparse
import pickle
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _preview import open_in_preview  # noqa: E402

from msv import config
from msv.cascade import path2_cascade
from msv.viz import GROUP_COLORS

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def group_columns(table):
    """Nombres de grupo presentes como pares `p_<grupo>` / `s_<grupo>`."""
    return [column[2:] for column in table.columns
            if column.startswith("p_") and f"s_{column[2:]}" in table.columns]


def draw_lightcurve(axis, time, flux, gap_days=2.0):
    for edge in time[:-1][np.diff(time) >= gap_days]:
        axis.axvline(edge, color="tab:blue", ls=":", lw=0.8)
    axis.plot(time, flux, ".", color="k", ms=1.2, alpha=0.55)
    axis.set_xlabel("Time [BTJD]", fontsize=8)
    axis.set_ylabel("flux", fontsize=8)
    axis.grid(alpha=0.25)


def best_non_random(group, names):
    """Por período: argmax de las clases reales, ignorando la columna Rndm.

    Devuelve (clase, prob, sigma) alineados con `group`.
    """
    usable = [name for name in names if name != "Rndm"]
    probabilities = group[[f"p_{name}" for name in usable]].to_numpy()
    dispersions = group[[f"s_{name}" for name in usable]].to_numpy()
    winner = probabilities.argmax(axis=1)
    rows = np.arange(len(group))
    return ([usable[index] for index in winner],
            probabilities[rows, winner], dispersions[rows, winner])


def draw_period_classes(axis, group, period_col, max_periods, names,
                        voted=frozenset(), longest=None):
    """Una barra por período: su mejor clase NO-Rndm, con prob y dispersión."""
    shown = group.nlargest(max_periods, "prominence")
    if longest is not None and longest not in shown.index:
        shown = pd.concat([shown, group.loc[[longest]]])
    group = shown.iloc[::-1]

    classes, probabilities, dispersions = best_non_random(group, names)
    positions = np.arange(len(group))
    axis.barh(positions, probabilities, xerr=dispersions, height=0.7,
              color=[GROUP_COLORS.get(name, "#888888") for name in classes],
              edgecolor="k", linewidth=0.4,
              error_kw={"ecolor": "k", "elinewidth": 0.9, "capsize": 2.5})
    for position, index, name, probability in zip(positions, group.index,
                                                  classes, probabilities):
        mark = " ★" if index in voted else ""
        if index == longest:
            mark += " ◄"
        axis.annotate(f" {name}{mark}", (probability, position), va="center",
                      fontsize=7.5, fontweight="bold" if mark else "normal")
    axis.set_yticks(positions)
    axis.set_yticklabels([f"{row[period_col]:.4f} d  ({row['source']})"
                          for _, row in group.iterrows()], fontsize=7.5)
    axis.set_xlim(0, 1.25)
    axis.axvline(1.0, color="0.85", lw=0.8)
    axis.set_xlabel(f"prob de la mejor clase no-Rndm (mediana ± σ de "
                    f"{config.MC_ITER} pasadas)", fontsize=8)
    axis.grid(axis="x", alpha=0.25)
    return dict(zip(group.index, zip(classes, probabilities, dispersions)))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("classification", help="CSV de step_brf.py (con p_/s_ por clase)")
    parser.add_argument("--curves", default=None,
                        help="pickle {(TIC, sector): (time, flux)} de curvas limpias")
    parser.add_argument("--lightcurves", default=None,
                        help="parquet de curvas, alternativa al pickle")
    parser.add_argument("--per-page", type=int, default=4)
    parser.add_argument("--max-periods", type=int, default=10,
                        help="candidatos por estrella, los más prominentes")
    parser.add_argument("--threshold", type=float, default=0.8,
                        help="umbral de brf_prob para votar en la cascada")
    parser.add_argument("--outdir", default=str(config.RESULTS_DIR / "figures"))
    parser.add_argument("--no-open", action="store_true")
    args = parser.parse_args()

    table = pd.read_csv(args.classification)
    names = group_columns(table)
    if not names:
        raise SystemExit("el CSV no trae columnas p_<clase>/s_<clase>: "
                         "regenerarlo con la versión actual de step_brf.py")

    if args.curves:
        with open(args.curves, "rb") as handle:
            curves = pickle.load(handle)
    elif args.lightcurves:
        from msv.cleaning import clean_lightcurve
        frame = pd.read_parquet(args.lightcurves,
                                columns=["TIC", "sector", "Time", "flux", "flux_err"])
        curves = {}
        for (tic, sector), group in frame.groupby(["TIC", "sector"]):
            time, flux, _ = clean_lightcurve(group["Time"].to_numpy(),
                                             group["flux"].to_numpy(),
                                             group["flux_err"].to_numpy())
            curves[(int(tic), int(sector))] = (time, flux)
    else:
        raise SystemExit("pasar --curves o --lightcurves")

    period_col = "period" if "period" in table.columns else "per"
    voters, tic_class = path2_cascade(table, threshold=args.threshold)
    voted = set(voters["cube_idx"]) if "cube_idx" in voters.columns else set()
    table = table.reset_index(drop=True)
    table["cube_idx"] = np.arange(len(table))
    pairs = sorted(table.groupby(["TIC", "sector"]).groups)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / "lc_prob_review.pdf"

    with PdfPages(outpath) as pdf:
        for start in range(0, len(pairs), args.per_page):
            page = pairs[start:start + args.per_page]
            figure, axes = plt.subplots(len(page), 2, squeeze=False,
                                        figsize=(14, 3.0 * len(page)),
                                        gridspec_kw={"width_ratios": [3, 1.6]})
            for (left, right), (tic, sector) in zip(axes, page):
                group = table[(table["TIC"] == tic) & (table["sector"] == sector)]
                time, flux = curves[(int(tic), int(sector))]
                draw_lightcurve(left, time, flux)
                voted_here = {index for index in group.index
                              if group.loc[index, "cube_idx"] in voted}
                longest = group[period_col].idxmax()
                drawn = draw_period_classes(right, group, period_col,
                                            args.max_periods, names,
                                            voted=voted_here, longest=longest)
                name, probability, dispersion = drawn[longest]
                headline = (f"◄ P max = {group.loc[longest, period_col]:.4f} d "
                            f"({name} {probability:.2f} ± {dispersion:.2f})")
                left.set_title(f"TIC {tic} — sector {sector}", fontsize=10,
                               fontweight="bold")
                right.set_title(f"{tic_class.get(int(tic), '?')}   |   {headline}",
                                fontsize=9)
            figure.tight_layout()
            pdf.savefig(figure, dpi=140)
            plt.close(figure)

    print(f"{outpath}  ({len(pairs)} estrellas, {len(table)} candidatos)")
    print(tic_class.value_counts().to_string())
    if not args.no_open:
        open_in_preview(outpath)


if __name__ == "__main__":
    main()
