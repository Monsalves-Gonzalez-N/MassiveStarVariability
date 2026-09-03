#!/usr/bin/env python
"""PDF de revisión: curva de luz + clase de cada período candidato.

Una ESTRELLA por fila. A la izquierda la curva limpia en tiempo (BTJD) vs
flujo; a la derecha una barra horizontal por período candidato, ordenados por
prominencia: el largo es la probabilidad de SU clase ganadora (media entre las
pasadas MC-dropout), la barra de error +/- su desviación estándar, y el color y
la etiqueta dicen qué clase es.

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

El encabezado del panel reporta el candidato de MAYOR probabilidad, marcado con
◄ y siempre graficado aunque no esté entre los más prominentes. Es el mismo
período que dobla build_phasefold_review.py, para poder leer los dos PDF juntos.

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
    """Nombres de grupo presentes como columnas `p_<grupo>`."""
    return [column[2:] for column in table.columns if column.startswith("p_")]


# Cada estadístico: (prefijo del centro, prefijos del intervalo, etiqueta).
STATS = {"mean": ("p_", ("s_",), "barra = media, bigote = media +/- sigma"),
         "median": ("med_", ("q1_", "q3_"), "barra = mediana, caja = Q1-Q3")}


def has_dispersion(table, names, stat):
    """True si el CSV trae las columnas del intervalo de `stat`.

    `step_brf_mean.py` promedia las pasadas ANTES del BRF, así que su salida es
    una sola llamada al bosque y no tiene dispersión que graficar.
    """
    _, spread, _ = STATS[stat]
    return all(f"{prefix}{name}" in table.columns
               for prefix in spread for name in names)


def draw_lightcurve(axis, time, flux, gap_days=2.0):
    for edge in time[:-1][np.diff(time) >= gap_days]:
        axis.axvline(edge, color="tab:blue", ls=":", lw=0.8)
    axis.plot(time, flux, ".", color="k", ms=1.2, alpha=0.55)
    axis.set_xlabel("Time [BTJD]", fontsize=8)
    axis.set_ylabel("flux", fontsize=8)
    axis.grid(alpha=0.25)


def best_non_random(group, names, stat="mean"):
    """Por período: argmax de las clases reales, ignorando la columna Rndm.

    Devuelve (clase, centro, borde_bajo, borde_alto), o (clase, centro, None,
    None) si el CSV no trae el intervalo de `stat`.

    Con `stat="mean"` el intervalo es media ± σ SIN recortar a [0, 1]: σ es la
    dispersión alrededor de la media, y la distribución MC por pico es bimodal,
    así que se pasa de 1 en la mayoría de los picos. Recortarlo escondería
    justamente eso.

    Con `stat="median"` es la mediana con la caja Q1-Q3, que sí queda dentro de
    [0, 1] y no supone simetría. Sobre una distribución bimodal la caja no
    reporta un ancho de error: se estira de 0 a 1 cuando los dos modos tienen
    masa, y ese estiramiento ES el diagnóstico de bimodalidad. El resumen que
    la describe de verdad sigue siendo `v_<grupo>` (la masa de cada modo), que
    se anota junto a la etiqueta de la clase.

    El argmax se toma siempre sobre `p_` (la media) aunque el dibujo sea la
    mediana: las medianas por clase no suman 1, así que el argmax entre ellas
    no compara cantidades de la misma escala.
    """
    usable = [name for name in names if name != "Rndm"]
    means = group[[f"p_{name}" for name in usable]].to_numpy()
    winner = means.argmax(axis=1)
    rows = np.arange(len(group))
    classes = [usable[index] for index in winner]

    center_prefix, spread, _ = STATS[stat]
    center = group[[f"{center_prefix}{name}" for name in usable]].to_numpy()[rows, winner]
    if not all(f"{prefix}{usable[0]}" in group.columns for prefix in spread):
        return classes, center, None, None
    if len(spread) == 1:
        dispersions = group[[f"s_{name}" for name in usable]].to_numpy()[rows, winner]
        return classes, center, center - dispersions, center + dispersions
    low = group[[f"q1_{name}" for name in usable]].to_numpy()[rows, winner]
    high = group[[f"q3_{name}" for name in usable]].to_numpy()[rows, winner]
    return classes, center, low, high


def winning_votes(group, names):
    """Voto (fracción de pasadas ganadas) de la clase que reporta cada período."""
    usable = [name for name in names if name != "Rndm"]
    if f"v_{usable[0]}" not in group.columns:
        return None
    means = group[[f"p_{name}" for name in usable]].to_numpy()
    winner = means.argmax(axis=1)
    votes = group[[f"v_{name}" for name in usable]].to_numpy()
    return votes[np.arange(len(group)), winner]


def best_candidate(group, names):
    """Índice del candidato con la mayor probabilidad, ignorando `Rndm`.

    Es el período que se dobla en build_phasefold_review.py: el más largo
    seleccionaba armónicos altos del peine, donde la mejor clase real tiene
    probabilidad ~0.
    """
    _, probabilities, _, _ = best_non_random(group, names)
    return group.index[int(np.argmax(probabilities))]


def draw_period_classes(axis, group, period_col, max_periods, names,
                        voted=frozenset(), longest=None, stat="mean"):
    """Una barra por período: su mejor clase NO-Rndm, con centro e intervalo."""
    shown = group.nlargest(max_periods, "prominence")
    if longest is not None and longest not in shown.index:
        shown = pd.concat([shown, group.loc[[longest]]])
    group = shown.iloc[::-1]

    classes, probabilities, low, high = best_non_random(group, names, stat)
    votes = winning_votes(group, names)
    positions = np.arange(len(group))
    axis.barh(positions, probabilities, height=0.7,
              color=[GROUP_COLORS.get(name, "#888888") for name in classes],
              edgecolor="k", linewidth=0.4)
    if low is not None:
        axis.hlines(positions, low, high, color="k", linewidth=0.9)
        for edge in (low, high):
            axis.scatter(edge, positions, marker="|", s=22, color="k",
                         linewidths=0.9)
    for offset, (position, index, name, probability) in enumerate(
            zip(positions, group.index, classes, probabilities)):
        mark = " ★" if index in voted else ""
        if index == longest:
            mark += " ◄"
        vote = "" if votes is None else f" (v={votes[offset]:.2f})"
        axis.annotate(f" {name}{vote}{mark}", (probability, position),
                      va="center", fontsize=7.5,
                      fontweight="bold" if mark else "normal")
    axis.set_yticks(positions)
    axis.set_yticklabels([f"{row[period_col]:.4f} d  ({row['source']})"
                          for _, row in group.iterrows()], fontsize=7.5)
    if low is None:
        axis.set_xlim(0, 1.45)
        axis.set_xlabel("prob de la mejor clase no-Rndm: BRF sobre la media "
                        "de las pasadas (sin dispersion)", fontsize=8)
    else:
        axis.set_xlim(-0.2 if stat == "mean" else 0, 1.7)
        axis.set_xlabel(f"prob de la mejor clase no-Rndm: {STATS[stat][2]} "
                        f"de {config.MC_ITER} pasadas; v = fraccion de pasadas "
                        f"que la clase gana", fontsize=8)
    axis.axvline(1.0, color="0.85", lw=0.8)
    axis.grid(axis="x", alpha=0.25)
    nothing = [None] * len(group)
    return dict(zip(group.index, zip(classes, probabilities,
                                     nothing if low is None else low,
                                     nothing if high is None else high)))


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
    parser.add_argument("--outname", default="lc_prob_review.pdf",
                        help="nombre del PDF dentro de --outdir")
    parser.add_argument("--stat", default="mean", choices=list(STATS),
                        help="resumen graficado: media +/- sigma o mediana Q1-Q3")
    parser.add_argument("--no-cascade", action="store_true",
                        help="no correr path2_cascade: sin ★ y sin clase por "
                             "TIC en el título, cada período se lee solo")
    parser.add_argument("--no-open", action="store_true")
    args = parser.parse_args()

    table = pd.read_csv(args.classification)
    names = group_columns(table)
    if not names:
        raise SystemExit("el CSV no trae columnas p_<clase>: "
                         "regenerarlo con la versión actual de step_brf.py")
    if not has_dispersion(table, names, args.stat):
        print(f"CSV sin las columnas de --stat {args.stat}: barras sin intervalo")

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
    if args.no_cascade:
        tic_class, voted = None, set()
    else:
        voters, tic_class = path2_cascade(table, threshold=args.threshold)
        voted = set(voters["cube_idx"]) if "cube_idx" in voters.columns else set()
    table = table.reset_index(drop=True)
    table["cube_idx"] = np.arange(len(table))
    pairs = sorted(table.groupby(["TIC", "sector"]).groups)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / args.outname

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
                best = best_candidate(group, names)
                drawn = draw_period_classes(right, group, period_col,
                                            args.max_periods, names,
                                            voted=voted_here, longest=best,
                                            stat=args.stat)
                name, probability, low, high = drawn[best]
                if low is None:
                    spread = ""
                elif args.stat == "median":
                    spread = f", Q1-Q3 {low:.2f}-{high:.2f}"
                else:
                    spread = f", sigma {(high - low) / 2:.2f}"
                headline = (f"◄ P = {group.loc[best, period_col]:.4f} d "
                            f"({name} {probability:.2f}{spread})")
                left.set_title(f"TIC {tic} — sector {sector}", fontsize=10,
                               fontweight="bold")
                prefix = ("" if tic_class is None
                          else f"{tic_class.get(int(tic), '?')}   |   ")
                right.set_title(f"{prefix}{headline}", fontsize=9)
            figure.tight_layout()
            pdf.savefig(figure, dpi=140)
            plt.close(figure)

    print(f"{outpath}  ({len(pairs)} estrellas, {len(table)} candidatos)")
    if tic_class is not None:
        print(tic_class.value_counts().to_string())
    if not args.no_open:
        open_in_preview(outpath)


if __name__ == "__main__":
    main()
