#!/usr/bin/env python
"""PDF de decisión sobre la normalización del histograma 2D de entrada.

Consume lo que dejó `compare_hist_norms.py` + step_cnn/step_brf en
`results/norm_compare/` y arma un único PDF con las tres vistas que deciden:

  1. conteo de clases, clase por TIC de la cascada y — el criterio duro — la
     fracción de picos `Rndm` SEPARADA por cadencia: una normalización
     invariante al número de puntos no puede hacerla depender de la cadencia.
  2. sonda de invariancia: probabilidad de la clase ganadora vs número de
     puntos (curva plana = normalización invariante).
  3. las 7 imágenes 32x32 lado a lado para 6 estrellas, con la clase y
     probabilidad que da cada una.

    PYTHONPATH=src python scripts/build_norm_review.py
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

from msv.cascade import path2_cascade

from compare_hist_norms import NORMALIZATION_NAMES, SUBSAMPLE_SIZES

REPO_ROOT = Path(__file__).resolve().parents[1]
NORM_DIR = REPO_ROOT / "results" / "norm_compare"
CADENCE_ORDER = ["2 min", "10 min", "30 min"]

# Un pico en P_orb/2 sigue siendo una detección correcta de la eclipsante:
# el espaciado eclipse-a-eclipse es medio período y es lo que ve Lomb-Scargle.
KNOWN_TRUTH = {
    (12675729, 82): {"clase": "E", "periodos": [2.8810, 1.4405],
                     "nota": "eclipsante, 18 eclipses, P orbital 2.8810 d (excéntrica)"},
    (339568213, 12): {"clase": "E", "periodos": [5.7084, 2.8542],
                      "nota": "eclipsante clara, P 5.7084 d"},
}
N_EXAMPLES = 6


def load_all():
    cadence = pd.read_csv(NORM_DIR / "cadence.csv")
    tables = {}
    for normalization in NORMALIZATION_NAMES:
        table = pd.read_csv(NORM_DIR / f"class_{normalization}.csv")
        table["row"] = np.arange(len(table))
        tables[normalization] = table.merge(
            cadence[["TIC", "sector", "cadence_group", "n_points"]],
            on=["TIC", "sector"], how="left")
    cubes = {normalization: np.load(NORM_DIR / f"input_{normalization}.npz")["X"]
             for normalization in NORMALIZATION_NAMES}
    return cadence, tables, cubes


def summary_frames(tables):
    class_counts = pd.DataFrame(
        {normalization: table["clase"].value_counts()
         for normalization, table in tables.items()}).fillna(0).astype(int)

    cascade_counts = {}
    for normalization, table in tables.items():
        _, tic_class = path2_cascade(table, threshold=0.8)
        cascade_counts[normalization] = tic_class.value_counts()
    cascade_counts = pd.DataFrame(cascade_counts).fillna(0).astype(int)

    rndm_rows = {}
    for normalization, table in tables.items():
        fractions = {}
        for group in CADENCE_ORDER:
            subset = table[table["cadence_group"] == group]
            if len(subset) == 0:
                fractions[group] = np.nan
            else:
                fractions[group] = float((subset["clase"] == "Rndm").mean())
        fractions["spread"] = (np.nanmax(list(fractions.values()))
                               - np.nanmin(list(fractions.values())))
        rndm_rows[normalization] = fractions
    rndm = pd.DataFrame(rndm_rows).T[CADENCE_ORDER + ["spread"]]
    return class_counts, cascade_counts, rndm


def _table_axes(axes, frame, title, float_format=None):
    axes.axis("off")
    axes.set_title(title, fontsize=10, loc="left", fontweight="bold")
    if float_format is None:
        cell_text = frame.astype(str).values
    else:
        cell_text = np.vectorize(float_format)(frame.values.astype(float))
    table = axes.table(cellText=cell_text, rowLabels=frame.index,
                       colLabels=frame.columns, loc="upper center",
                       cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(7.5)
    table.scale(1.0, 1.25)


def page_summary(pdf, class_counts, cascade_counts, rndm):
    figure, axes_list = plt.subplots(3, 1, figsize=(11.7, 8.3))
    figure.suptitle("Normalización del histograma 2D — resumen sobre 50 estrellas / 839 picos",
                    fontsize=12, fontweight="bold")
    _table_axes(axes_list[0], class_counts, "Clases por pico (n = 839)")
    _table_axes(axes_list[1], cascade_counts, "Clase por TIC (cascada Path-2, umbral 0.8)")
    _table_axes(axes_list[2], rndm,
                "Fracción de picos Rndm por cadencia — CRITERIO PRINCIPAL: "
                "no debe depender de la cadencia (spread pequeño)",
                float_format=lambda value: f"{value:.3f}")
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    pdf.savefig(figure)
    plt.close(figure)


def page_invariance(pdf):
    meta = pd.read_csv(NORM_DIR / "invariance_meta.csv")
    predictions = pd.read_csv(NORM_DIR / "invariance_class.csv")
    meta = meta.reset_index(drop=True)
    meta["clase"] = predictions["clase"].values
    meta["prob"] = predictions["prob"].values

    figure, axes_list = plt.subplots(2, 4, figsize=(16.5, 8.3), sharex=True, sharey=True)
    figure.suptitle("Sonda de invariancia al número de puntos "
                    "(bordes de flujo y amplitud congelados; sólo cambia N).  "
                    "Plana = normalización invariante", fontsize=12, fontweight="bold")
    axes_flat = axes_list.ravel()

    for position, normalization in enumerate(NORMALIZATION_NAMES):
        axes = axes_flat[position]
        subset = meta[meta["normalization"] == normalization]
        drifts = []
        for (tic, sector, period), group in subset.groupby(["TIC", "sector", "per"]):
            group = group.sort_values("n_points")
            reference = group[group["is_full"]]
            if len(reference) == 0:
                continue
            reference_class = reference["clase"].iloc[0]
            probability_of_reference_class = []
            for _, row in group.iterrows():
                if row["clase"] == reference_class:
                    probability_of_reference_class.append(row["prob"])
                else:
                    probability_of_reference_class.append(np.nan)
            values = np.array(probability_of_reference_class, dtype=float)
            axes.plot(group["n_points"], values, marker="o", markersize=3,
                      linewidth=0.9, alpha=0.8)
            finite = values[np.isfinite(values)]
            full_value = values[-1]
            if np.isfinite(full_value) and len(finite) > 0:
                drifts.append(np.nanmax(np.absolute(values - full_value)))
        axes.set_xscale("log")
        axes.set_ylim(-0.02, 1.02)
        median_drift = np.nanmedian(drifts) if drifts else np.nan
        axes.set_title(f"{normalization}\nderiva mediana {median_drift:.2f}", fontsize=9)
        axes.grid(alpha=0.25)
        if position >= 3:
            axes.set_xlabel("número de puntos")
        if position % 4 == 0:
            axes.set_ylabel("prob. de la clase de la curva completa")

    axes_flat[-1].axis("off")
    axes_flat[-1].text(0.02, 0.95,
                       "Hueco en la curva = la clase ganadora cambió\n"
                       "respecto de la curva completa.\n\n"
                       "8 pares (TIC, sector) densos x 3 picos\n"
                       "más prominentes; N = completa, 2000,\n"
                       "1000, 500, 200, 100 (semilla fija).",
                       fontsize=9, va="top",
                       transform=axes_flat[-1].transAxes)
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    pdf.savefig(figure)
    plt.close(figure)
    return meta


def pick_examples(tables):
    reference = tables["min_max"]
    chosen = []
    for tic, sector in KNOWN_TRUTH:
        subset = reference[(reference["TIC"] == tic) & (reference["sector"] == sector)]
        if len(subset) == 0:
            continue
        chosen.append((int(tic), int(sector)))

    remaining = reference[~reference.set_index(["TIC", "sector"]).index.isin(chosen)]
    for group in CADENCE_ORDER:
        subset = remaining[(remaining["cadence_group"] == group)
                           & (remaining["clase"] != "Rndm")]
        if len(subset) == 0:
            continue
        ranked = subset.sort_values("prob", ascending=False)
        for _, row in ranked.iterrows():
            pair = (int(row["TIC"]), int(row["sector"]))
            if pair not in chosen:
                chosen.append(pair)
                break
    for _, row in reference.sort_values("prominence", ascending=False).iterrows():
        if len(chosen) >= N_EXAMPLES:
            break
        pair = (int(row["TIC"]), int(row["sector"]))
        if pair not in chosen:
            chosen.append(pair)
    return chosen[:N_EXAMPLES]


def pick_row(table, tic, sector):
    """Pico más prominente no-Rndm del par; si todos son Rndm, el más prominente."""
    subset = table[(table["TIC"] == tic) & (table["sector"] == sector)]
    non_random = subset[subset["clase"] != "Rndm"]
    pool = non_random if len(non_random) else subset
    return pool.sort_values("prominence", ascending=False).iloc[0]


def page_examples(pdf, tables, cubes):
    pairs = pick_examples(tables)
    figure, axes_list = plt.subplots(len(pairs), len(NORMALIZATION_NAMES),
                                     figsize=(16.5, 2.35 * len(pairs)))
    figure.suptitle("Imagen 32x32 que ve la CNN, misma estrella y mismo período, "
                    "una columna por normalización", fontsize=12, fontweight="bold")

    for line, (tic, sector) in enumerate(pairs):
        selected = pick_row(tables["min_max"], tic, sector)
        row_index = int(selected["row"])
        period = float(selected["per"])
        truth = KNOWN_TRUTH.get((tic, sector))

        for column, normalization in enumerate(NORMALIZATION_NAMES):
            axes = axes_list[line, column]
            axes.imshow(cubes[normalization][row_index, ..., 0], cmap="gray",
                        interpolation="nearest", aspect="auto")
            prediction = tables[normalization].iloc[row_index]
            correct = truth is not None and prediction["clase"] == truth["clase"]
            axes.set_title(f"{normalization}\n{prediction['clase']} "
                           f"{prediction['prob']:.2f}", fontsize=8,
                           color=("green" if correct else
                                  "red" if truth is not None else "black"))
            axes.set_xticks([])
            axes.set_yticks([])
            if column == 0:
                label = (f"TIC {tic}  s{sector}\n"
                         f"{selected['cadence_group']}, "
                         f"{int(selected['n_points'])} pts\n"
                         f"P = {period:.4f} d  ({selected['source']})")
                if truth is not None:
                    label += (f"\nVERDAD: {truth['clase']}, "
                              f"P {truth['periodos'][0]:.4f} d")
                axes.set_ylabel(label, fontsize=7.5, rotation=0, ha="right", va="center",
                                labelpad=6)

    figure.tight_layout(rect=(0.07, 0, 1, 0.94))
    pdf.savefig(figure)
    plt.close(figure)
    return pairs


def page_known_truth(pdf, tables):
    """Las dos eclipsantes conocidas, evaluadas en su período verdadero y en su mitad."""
    rows = []
    for (tic, sector), truth in KNOWN_TRUTH.items():
        for normalization in NORMALIZATION_NAMES:
            table = tables[normalization]
            subset = table[(table["TIC"] == tic) & (table["sector"] == sector)].copy()
            if len(subset) == 0:
                continue
            entry = {"estrella": f"TIC {tic} s{sector}", "norm": normalization}
            for label, period in zip(["P_orb", "P_orb/2"], truth["periodos"]):
                subset["period_distance"] = (subset["per"] - period).abs()
                nearest = subset.sort_values("period_distance").iloc[0]
                entry[f"{label} pico"] = f"{nearest['per']:.4f}"
                entry[f"{label} clase"] = f"{nearest['clase']} {nearest['prob']:.2f}"
            eclipsing = subset[(subset["clase"] == truth["clase"])
                               & (subset["prob"] >= 0.8)]
            entry["algún pico E>=0.8"] = ("SI " + ", ".join(
                f"{value:.3f} d" for value in sorted(eclipsing["per"])[:4])
                if len(eclipsing) else "no")
            rows.append(entry)
    frame = pd.DataFrame(rows).set_index(["estrella", "norm"])

    figure, axes = plt.subplots(figsize=(14.0, 8.3))
    axes.axis("off")
    axes.set_title("Verdad conocida: dos eclipsantes.  P_orb y P_orb/2 "
                   "(el espaciado eclipse-a-eclipse es lo que ve LS)",
                   fontsize=12, fontweight="bold", loc="left")
    table = axes.table(cellText=frame.values,
                       rowLabels=[" / ".join(index) for index in frame.index],
                       colLabels=frame.columns, loc="upper center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.35)
    pdf.savefig(figure)
    plt.close(figure)
    return frame


def drift_versus_size(invariance):
    """Cambio de clase y deriva de probabilidad respecto de la curva completa."""
    sizes = ["full"] + [size for size in SUBSAMPLE_SIZES]
    flip_rows, drift_rows = {}, {}
    for normalization in NORMALIZATION_NAMES:
        subset = invariance[invariance["normalization"] == normalization]
        flips, drifts = {}, {}
        for _, group in subset.groupby(["TIC", "sector", "per"]):
            reference = group[group["is_full"]].iloc[0]
            for _, row in group.iterrows():
                key = "full" if row["is_full"] else int(row["n_points"])
                same = row["clase"] == reference["clase"]
                flips.setdefault(key, []).append(0.0 if same else 1.0)
                drifts.setdefault(key, []).append(
                    abs(row["prob"] - reference["prob"]) if same else 1.0)
        flip_rows[normalization] = {key: float(np.mean(value))
                                    for key, value in flips.items()}
        drift_rows[normalization] = {key: float(np.mean(value))
                                     for key, value in drifts.items()}
    flip = pd.DataFrame(flip_rows).T.reindex(columns=sizes)
    drift = pd.DataFrame(drift_rows).T.reindex(columns=sizes)
    return flip, drift


def page_invariance_tables(pdf, flip, drift):
    figure, axes_list = plt.subplots(2, 1, figsize=(11.7, 8.3))
    figure.suptitle("Sonda de invariancia — resumen numérico", fontsize=12,
                    fontweight="bold")
    _table_axes(axes_list[0], flip,
                "Fracción de picos cuya CLASE cambia respecto de la curva completa",
                float_format=lambda value: f"{value:.2f}")
    _table_axes(axes_list[1], drift,
                "Deriva media de la probabilidad (un cambio de clase cuenta como 1)",
                float_format=lambda value: f"{value:.2f}")
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    pdf.savefig(figure)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=str(REPO_ROOT / "results" / "figures"
                                             / "normalization_review.pdf"))
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cadence, tables, cubes = load_all()
    class_counts, cascade_counts, rndm = summary_frames(tables)

    summary = rndm.copy()
    summary["n_Rndm_total"] = [int((tables[name]["clase"] == "Rndm").sum())
                               for name in summary.index]
    summary["n_E_total"] = [int((tables[name]["clase"] == "E").sum())
                            for name in summary.index]
    summary.to_csv(NORM_DIR / "summary.csv")

    with PdfPages(out_path) as pdf:
        page_summary(pdf, class_counts, cascade_counts, rndm)
        invariance = page_invariance(pdf)
        flip, drift = drift_versus_size(invariance)
        page_invariance_tables(pdf, flip, drift)
        truth_frame = page_known_truth(pdf, tables)
        page_examples(pdf, tables, cubes)

    flip.to_csv(NORM_DIR / "invariance_class_flip.csv")
    drift.to_csv(NORM_DIR / "invariance_drift.csv")
    truth_frame.to_csv(NORM_DIR / "known_truth.csv")

    print(f"-> {out_path}")
    print("\n--- fracción de Rndm por cadencia ---")
    print(rndm.round(3).to_string())
    print("\n--- clases por pico ---")
    print(class_counts.to_string())

    print("\n--- fracción de cambio de clase vs N ---")
    print(flip.round(2).to_string())
    print("\n--- deriva media de la probabilidad vs N ---")
    print(drift.round(2).to_string())
    print("\n--- verdad conocida ---")
    print(truth_frame.to_string())


if __name__ == "__main__":
    main()
