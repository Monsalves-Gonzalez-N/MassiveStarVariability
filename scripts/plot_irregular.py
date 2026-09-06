#!/usr/bin/env python
"""La masa irregular de cada estrella, con su nombre y su clase reportada.

`irregular` es la masa media en LPV sobre TODOS los picos — no la del pico
elegido. Mide cuánta de la probabilidad total se va a la clase de las curvas sin
forma, así que es una propiedad de la curva y no de un período candidato.

`step_clasificar_una_red.py` la calcula por estrella-SECTOR. Acá `--nivel star`
la recalcula sobre los picos de todos los sectores de la estrella juntos, que es
lo que hay que mirar cuando la pregunta es sobre la estrella y no sobre una
época: son promedios sobre picos, así que el de la estrella no es el promedio de
los de sus sectores salvo que todos tengan el mismo número de picos.

Las barras van ordenadas y etiquetadas con el TIC; el color es NUESTRA clase
reportada y el borde negro marca `nivel alta`. Con `--truth` el color pasa a ser
la clase publicada, para la golden sample.

Ojo con la lectura fácil: `irregular` bajo NO quiere decir "un solo período",
quiere decir "mucha señal coherente", y una pulsante multiperiódica cae ahí
(§8 de scripts/golden/README.md). Lo que sí trackea es estocasticidad.

    PYTHONPATH=src python scripts/plot_irregular.py
    PYTHONPATH=src python scripts/plot_irregular.py --nivel star-sector \
        --clasificacion results/golden/clasificacion_collapsed.csv \
        --truth catalogs/golden_truth.csv --out results/figures/golden_irregular.pdf
"""
import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _preview import open_in_preview  # noqa: E402

from msv.config import REPO_ROOT, RESULTS_DIR

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUR_COLORS = {"E": "tab:blue", "ELL": "tab:cyan", "Pulsating": "tab:orange",
              "LPV": "tab:red", "Rndm": "0.6"}
GOLD_COLORS = {"ECL": "tab:blue", "ELL": "tab:cyan", "ROT": "tab:green",
               "PULS": "tab:orange", "BE": "tab:red", "SLF": "tab:purple",
               "AMBIGUOUS": "0.55", "OTHER": "0.75", "NOISY": "0.35"}
GOLD_ORDER = ["ECL", "ELL", "ROT", "PULS", "BE", "SLF", "AMBIGUOUS", "OTHER",
              "NOISY"]
OUR_ORDER = ["E", "ELL", "Pulsating", "LPV", "Rndm"]
# Con más barras que esto las etiquetas por TIC dejan de leerse y el gráfico
# pasa a ser una rampa anónima.
MAX_LABELS = 80


def collapse(peaks, keys):
    """Una fila por estrella (o estrella-sector) con su masa irregular.

    Se recalcula desde `p_LPV` en vez de leer la columna `irregular`: esa está
    promediada por estrella-sector, y promediar promedios sobre grupos de
    distinto tamaño no da el promedio del conjunto.
    """
    rows = []
    for key, block in peaks.groupby(keys):
        reported = block[block.reportado] if "reportado" in block else block[:0]
        if len(reported):
            best = reported.loc[reported.log_pLPV.idxmax()]
        else:
            best = block.loc[block.log_pLPV.idxmax()]
        entry = dict(zip(keys, key if isinstance(key, tuple) else (key,)))
        entry.update({
            "irregular": float(block.p_LPV.mean()),
            "n_peaks": len(block),
            "n_sectors": block.sector.nunique(),
            "clase": best.clase,
            "per": float(best.per),
            "log_pLPV": float(best.log_pLPV),
            "nivel": best.nivel if "nivel" in block else "baja",
            "reportado": bool(len(reported)),
        })
        rows.append(entry)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--clasificacion",
                        default=str(RESULTS_DIR / "clasificacion_una_red.csv"))
    parser.add_argument("--nivel", choices=["star", "star-sector"],
                        default="star")
    parser.add_argument("--truth", default=None,
                        help="golden_truth.csv: colorea por clase publicada")
    parser.add_argument("--out",
                        default=str(REPO_ROOT / "results" / "figures" /
                                    "irregular.pdf"))
    parser.add_argument("--no-preview", action="store_true")
    args = parser.parse_args()

    peaks = pd.read_csv(args.clasificacion)
    keys = ["TIC"] if args.nivel == "star" else ["TIC", "sector"]
    stars = collapse(peaks, keys)

    if args.truth:
        truth = pd.read_csv(args.truth)
        stars = stars.merge(truth[["TIC", "class_gold"]], on="TIC", how="left")
        label_column, palette, order = "class_gold", GOLD_COLORS, GOLD_ORDER
    else:
        label_column, palette, order = "clase", OUR_COLORS, OUR_ORDER

    stars = stars.sort_values("irregular").reset_index(drop=True)
    colors = [palette.get(name, "0.5") for name in stars[label_column]]
    edges = ["k" if level == "alta" else "none" for level in stars.nivel]

    labelled = len(stars) <= MAX_LABELS
    width = max(9.0, 0.22 * len(stars)) if labelled else 13.0
    figure, axes = plt.subplots(figsize=(width, 7.0))

    positions = np.arange(len(stars))
    axes.bar(positions, stars.irregular.values, width=0.85 if labelled else 1.0,
             color=colors, edgecolor=edges, linewidth=1.2)
    axes.set_ylim(0, 1)
    axes.set_ylabel("masa irregular (p_LPV media de todos los picos)")
    axes.axhline(np.median(stars.irregular), color="k", linestyle=":",
                 linewidth=1.0)
    axes.text(0.002, np.median(stars.irregular) + 0.015,
              f"mediana {np.median(stars.irregular):.2f}",
              transform=axes.get_yaxis_transform(), fontsize=9)

    if labelled:
        if args.nivel == "star":
            names = [f"{int(tic)}" for tic in stars.TIC]
        else:
            names = [f"{int(tic)} s{int(sector)}"
                     for tic, sector in zip(stars.TIC, stars.sector)]
        axes.set_xticks(positions)
        axes.set_xticklabels(names, rotation=90, fontsize=7)
        axes.set_xlim(-0.7, len(stars) - 0.3)
    else:
        axes.set_xlabel(f"{'estrella' if args.nivel == 'star' else 'estrella-sector'}"
                        f", ordenada  (n = {len(stars)})")
        axes.set_xlim(-0.5, len(stars) - 0.5)

    present = [name for name in order if name in set(stars[label_column].dropna())]
    handles = [plt.Rectangle((0, 0), 1, 1, color=palette[name])
               for name in present]
    handles.append(plt.Rectangle((0, 0), 1, 1, facecolor="w", edgecolor="k",
                                 linewidth=1.2))
    axes.legend(handles, present + ["nivel alta"], ncol=len(present) + 1,
                fontsize=9, loc="upper left", frameon=False)
    axes.set_title(f"{len(stars)} {'estrellas' if args.nivel == 'star' else 'estrella-sector'}"
                   f", {int(stars.n_peaks.sum())} picos"
                   f"  —  {Path(args.clasificacion).name}", fontsize=10)

    figure.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.out)
    plt.close(figure)
    print(f"escrito {args.out}")

    summary = stars.groupby(label_column).agg(
        n=("irregular", "size"),
        mediana=("irregular", "median"),
        q1=("irregular", lambda values: values.quantile(0.25)),
        q3=("irregular", lambda values: values.quantile(0.75)),
        picos_mediana=("n_peaks", "median"),
    )
    print(summary.round(3).to_string())

    if not args.no_preview:
        open_in_preview(args.out)


if __name__ == "__main__":
    main()
