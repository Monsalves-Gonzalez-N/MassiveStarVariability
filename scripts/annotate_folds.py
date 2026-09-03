#!/usr/bin/env python
"""Revisión interactiva: abre los N mejores folds de cada estrella y anota.

Una ventana por estrella con sus candidatos de período doblados lado a lado; en
la terminal se escribe la nota y se elige cuál es el período bueno. Cada
respuesta se guarda en el CSV APENAS se escribe, así que cortar con Ctrl-C no
pierde nada y volver a correr el script retoma donde quedó.

En el prompt:
  a / b / c ...   marca ese panel como el período correcto; se pueden dar
                  varios ("a y b", "c,d"), que es el caso normal cuando el
                  mismo período aparece como pico de LS y de ACF
  n / ninguno     ninguno de los candidatos sirve
  ?               no se puede decidir con lo que se ve
  <texto libre>   la nota; puede ir sola o después de la letra ("b armonico 2x")
  <enter>         saltea la estrella sin anotarla
  back            vuelve a la estrella anterior
  q               guarda y sale

El CSV de salida tiene una fila por ESTRELLA (TIC, sector, elegida, per_elegido,
nota) y no por panel: la pregunta que se está respondiendo es cuál de los
candidatos es el período bueno, que es una sola decisión por estrella.

    python scripts/annotate_folds.py results/clasificacion_review20.csv \
        --curves results/phasefold_curves.pkl --out catalogs/notas_periodos.csv
"""
import argparse
import re
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_lc_prob_review import (STATS, best_non_random, group_columns,  # noqa: E402
                                  winning_votes)
from build_top3_folds import panel_label, rank_candidates  # noqa: E402

from msv.viz import GROUP_COLORS, plot_phase_fold  # noqa: E402

# DESPUÉS de los imports: build_lc_prob_review fija "Agg" al importarse, que es
# lo correcto para los scripts que escriben PDF y lo contrario de lo que hace
# falta acá. `force=True` porque el backend ya quedó elegido.
for _backend in ("macosx", "TkAgg"):
    try:
        matplotlib.use(_backend, force=True)
        break
    except ImportError:
        continue

import matplotlib.pyplot as plt  # noqa: E402

OUT_COLS = ["TIC", "sector", "elegida", "per_elegido", "clase_elegida", "nota"]


def grid_shape(how_many, columns):
    columns = min(columns, how_many)
    return int(np.ceil(how_many / columns)), columns


def draw_star(figure, tic, sector, chosen, classes, centers, low, high, votes,
              period_col, curves, phase_bins, columns):
    figure.clf()
    rows, columns = grid_shape(len(chosen), columns)
    axes = figure.subplots(rows, columns, squeeze=False).ravel()
    for axis in axes[len(chosen):]:
        axis.set_visible(False)
    time, flux = curves[(tic, sector)]
    good = np.isfinite(time) & np.isfinite(flux)
    time, flux = time[good], flux[good]
    for column, (axis, (_, candidate)) in enumerate(zip(axes, chosen.iterrows())):
        period = candidate[period_col]
        plot_phase_fold(time, flux, period, ax=axis, phase_bins=phase_bins,
                        color="0.6", ms=1.2)
        spread = "" if low is None else f"  [{low[column]:.2f}, {high[column]:.2f}]"
        vote = "" if votes is None else f"  v={votes[column]:.2f}"
        axis.set_title(f"({panel_label(column)})  P = {period:.4f} d "
                       f"({candidate['source']}, prom {candidate['prominence']:.2f})\n"
                       f"{classes[column]} {centers[column]:.3f}{spread}{vote}",
                       fontsize=9, color=GROUP_COLORS.get(classes[column], "k"))
        axis.set_xlabel("phase", fontsize=8)
        axis.set_ylabel("flux" if column % columns == 0 else "", fontsize=8)
        axis.tick_params(labelsize=7)
    figure.suptitle(f"TIC {tic} — sector {sector}   ({len(chosen)} candidatos)",
                    fontsize=11, fontweight="bold")
    figure.tight_layout()
    figure.canvas.draw_idle()
    plt.pause(0.05)


# Palabras que unen dos etiquetas ("a y b bien") y no cortan la lista, y
# sinónimos de "ningún candidato sirve" que se escriben solos.
#
# OJO con "y": es a la vez el conector en castellano y la etiqueta del panel 25.
# Se resuelve por contexto — vale como conector solo si ya hay una etiqueta
# elegida Y el token siguiente es otra etiqueta — así que "a y b" son los
# paneles a y b, mientras que "y" sola o "a y" siguen pudiendo señalar al
# panel 25.
CONNECTORS = {"y", "and", "+", "&"}
NONE_WORDS = {"ninguno", "ninguna", "ningun", "ningún", "none", "nada"}


def parse_answer(text, how_many):
    """-> (elegidas, nota). `elegidas` es una lista de etiquetas, ['n'], ['?'] o [].

    Acepta varias etiquetas seguidas porque en la práctica hay estrellas donde
    dos candidatos son el mismo período (el pico de LS y el del ACF, que caen a
    0.003 d de distancia) y marcar solo uno perdería la mitad del dato.
    """
    tokens = [token for token in re.split(r"[,\s]+", text.strip()) if token]
    if not tokens:
        return [], ""
    valid = {panel_label(index) for index in range(how_many)} | {"n", "?"}

    def is_label(position):
        return position < len(tokens) and tokens[position].lower() in valid

    picked, position = [], 0
    while position < len(tokens):
        token = tokens[position].lower()
        if token in CONNECTORS and picked and is_label(position + 1):
            pass
        elif token in valid:
            picked.append(token)
        elif token in NONE_WORDS and not picked:
            picked.append("n")
        else:
            break
        position += 1
    return picked, " ".join(tokens[position:]).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("classification", help="CSV de step_brf.py")
    parser.add_argument("--curves", required=True,
                        help="pickle {(TIC, sector): (time, flux)}")
    parser.add_argument("--out", default="catalogs/notas_periodos.csv")
    parser.add_argument("--top", type=int, default=3,
                        help="candidatos por estrella; 0 = todos los del CSV, "
                             "que es lo que muestran los PDF de revisión")
    parser.add_argument("--cols", type=int, default=5,
                        help="paneles por fila en la ventana")
    parser.add_argument("--rank", default="prob", choices=["prob", "prominence"])
    parser.add_argument("--stat", default="median", choices=list(STATS))
    parser.add_argument("--phase-bins", type=int, default=50)
    parser.add_argument("--redo", action="store_true",
                        help="volver a preguntar por las estrellas ya anotadas")
    args = parser.parse_args()

    table = pd.read_csv(args.classification)
    names = group_columns(table)
    if not names:
        raise SystemExit("el CSV no trae columnas p_<clase>")
    curves = pd.read_pickle(args.curves)
    period_col = "period" if "period" in table.columns else "per"

    stars = []
    for (tic, sector), group in table.groupby(["TIC", "sector"], sort=True):
        chosen = rank_candidates(group, names, args.rank, args.top)
        classes, centers, low, high = best_non_random(chosen, names, args.stat)
        stars.append((int(tic), int(sector), chosen, classes, centers, low,
                      high, winning_votes(chosen, names)))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        notes = pd.read_csv(out_path).set_index(["TIC", "sector"])
        notes = notes.to_dict("index")
    else:
        notes = {}

    def flush():
        frame = pd.DataFrame([{**{"TIC": tic, "sector": sector}, **values}
                              for (tic, sector), values in sorted(notes.items())])
        frame.reindex(columns=OUT_COLS).to_csv(out_path, index=False)

    print(f"{len(stars)} estrellas, {len(notes)} ya anotadas -> {out_path}")
    print("letra = periodo correcto | n = ninguno | ? = no decido | "
          "enter = saltear | back | q = salir\n")

    widest = max(len(chosen) for _, _, chosen, *_ in stars)
    rows, columns = grid_shape(widest, args.cols)
    figure = plt.figure(figsize=(min(4.6 * columns, 19.0),
                                 min(3.4 * rows, 11.0)))
    plt.show(block=False)
    index = 0
    while 0 <= index < len(stars):
        tic, sector, chosen, classes, centers, low, high, votes = stars[index]
        if (tic, sector) in notes and not args.redo:
            index += 1
            continue
        draw_star(figure, tic, sector, chosen, classes, centers, low, high,
                  votes, period_col, curves, args.phase_bins, args.cols)
        header = "  ".join(
            f"({panel_label(column)}) {candidate[period_col]:.4f} {classes[column]}"
            for column, (_, candidate) in enumerate(chosen.iterrows()))
        print(f"[{index + 1}/{len(stars)}] TIC {tic} s{sector}\n    {header}")
        try:
            answer = input("  nota> ")
        except (EOFError, KeyboardInterrupt):
            print("\ncortado")
            break
        if answer.strip().lower() == "q":
            break
        if answer.strip().lower() == "back":
            index = max(0, index - 1)
            continue
        picked, note = parse_answer(answer, len(chosen))
        if not picked and not note:
            index += 1
            continue
        labels = [panel_label(column) for column in range(len(chosen))]
        columns = [labels.index(label) for label in picked if label in labels]
        notes[(tic, sector)] = {
            "elegida": ",".join(picked),
            "per_elegido": ",".join(f"{chosen.iloc[column][period_col]:.6f}"
                                    for column in columns),
            "clase_elegida": ",".join(classes[column] for column in columns),
            "nota": note}
        flush()
        index += 1

    flush()
    plt.close(figure)
    print(f"\n{len(notes)} estrellas anotadas -> {out_path}")


if __name__ == "__main__":
    main()
