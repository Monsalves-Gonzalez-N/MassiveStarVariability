#!/usr/bin/env python
"""Las estrellas donde el periodo elegido no es el de VSX, dobladas a los dos.

Un panel por estrella con la curva doblada a NUESTRO periodo y al de VSX, para
poder mirar cual de los dos describe mejor la curva. La mayoria de los fallos
son factores de 2 exactos, y en las binarias eso no es un error: VSX reporta el
periodo ORBITAL y la curva de luz se repite en la mitad.

    PYTHONPATH=src python scripts/vsx/plot_misclasificadas.py
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from arbol_features import cargar, contexto, familia
from msv.cleaning import clean_lightcurve
from msv.prewhiten import to_relative
from clase_estrella import vector_por_estrella
from selector_periodo import (CORTE_GATE, K_PICOS, make_gate, matriz,
                              scores_selector_oof)

from msv.config import RESULTS_DIR

POR_PAGINA = 4


def doblar(time, flux, period, n_bins=60):
    """Fase mas el promedio por bin, que es lo que hace legible el fold."""
    fase = np.mod(time - np.nanmin(time), period) / period
    bordes = np.linspace(0, 1, n_bins + 1)
    indices = np.clip(np.digitize(fase, bordes) - 1, 0, n_bins - 1)
    medias = np.array([np.nanmedian(flux[indices == posicion])
                       if np.any(indices == posicion) else np.nan
                       for posicion in range(n_bins)])
    return fase, 0.5 * (bordes[:-1] + bordes[1:]), medias


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--probe",
                        default=str(RESULTS_DIR / "vsx" /
                                    "probe_peaks_w001_1pass.csv"))
    # OJO: results/vsx/curves.pkl tiene el flujo en CEROS. La fuente buena es
    # el parquet, que es lo que consume probe_peaks.py.
    parser.add_argument("--lc", default=str(RESULTS_DIR / "vsx" /
                                            "lc.parquet"))
    parser.add_argument("--out", default=str(RESULTS_DIR / "vsx" /
                                             "misclasificadas.pdf"))
    args = parser.parse_args()

    peaks = contexto(cargar(args.probe, verdicts=("ok", "unconstrained")))
    peaks["es_multiperiodica"] = peaks.vis_verdict == "unconstrained"
    peaks["familia"] = familia(peaks)
    peaks["clase_verdadera"] = np.where(peaks.es_multiperiodica,
                                        "unconstrained", peaks.familia)
    peaks["probabilidad"] = scores_selector_oof(peaks, 5)

    estrellas = vector_por_estrella(peaks, K_PICOS, con_per=False)
    por_tic = peaks.drop_duplicates("TIC").set_index("TIC")
    estrellas["es_multiperiodica"] = estrellas.TIC.map(por_tic.es_multiperiodica)
    columnas = [c for c in estrellas.columns
                if c not in ("TIC", "clase_verdadera", "cnn_top",
                             "es_multiperiodica")]
    X = matriz(columnas, estrellas)
    y = estrellas.es_multiperiodica.values.astype(int)
    gate = np.zeros(len(y))
    for semilla in range(5):
        parcial = np.zeros(len(y))
        for entrena, prueba in StratifiedKFold(
                5, shuffle=True, random_state=semilla).split(X, y):
            modelo = make_gate()
            modelo.fit(X[entrena], y[entrena])
            parcial[prueba] = modelo.predict_proba(X[prueba])[:, 1]
        gate += parcial / 5
    pasa = pd.Series(gate < CORTE_GATE, index=estrellas.TIC.values)

    top1 = peaks.sort_values("probabilidad", ascending=False).groupby(
        "TIC").head(1)
    catalogo = top1[top1.TIC.map(pasa) & ~top1.es_multiperiodica]
    fallan = catalogo[catalogo.tag != "1:1"].copy()
    review = pd.read_csv(Path(__file__).resolve().parents[2] / "catalogs" /
                         "vsx_visual_review.csv")
    fallan = fallan.merge(review[["TIC", "Type", "vis_notes"]], on="TIC",
                          how="left")
    fallan["razon"] = fallan.per / fallan.per_vsx
    fallan = fallan.sort_values(["familia", "tag", "razon"])
    print(f"{len(fallan)} estrellas fallan el 1:1 sobre "
          f"{len(catalogo)} del catalogo")

    lightcurves = pd.read_parquet(
        args.lc, filters=[("TIC", "in", sorted(fallan.TIC.unique().tolist()))],
        columns=["TIC", "sector", "Time", "flux", "flux_err"])
    curvas = {}
    for (tic, sector), bloque in lightcurves.groupby(["TIC", "sector"]):
        time, flux, _ = clean_lightcurve(bloque.Time.to_numpy(),
                                         bloque.flux.to_numpy(),
                                         bloque.flux_err.to_numpy())
        relativo, _ = to_relative(flux)
        curvas[(tic, sector)] = (time, relativo)
    with PdfPages(args.out) as pdf:
        figura, ejes = None, None
        for posicion, fila in enumerate(fallan.itertuples()):
            if posicion % POR_PAGINA == 0:
                figura, ejes = plt.subplots(POR_PAGINA, 2,
                                            figsize=(8.3, 11.7))
                figura.subplots_adjust(hspace=0.55, wspace=0.2,
                                       top=0.94, bottom=0.05)
                figura.suptitle("Estrellas donde el periodo elegido no es "
                                "el de VSX", fontsize=11)
            curva = curvas.get((fila.TIC, fila.sector))
            fila_ejes = ejes[posicion % POR_PAGINA]
            if curva is None:
                for eje in fila_ejes:
                    eje.set_axis_off()
                continue
            time, flux = curva
            limites = np.nanpercentile(flux, [0.5, 99.5])
            margen = 0.1 * (limites[1] - limites[0])
            for eje, (period, etiqueta) in zip(
                    fila_ejes, [(fila.per, "nuestro"),
                                (fila.per_vsx, "VSX")]):
                fase, centros, medias = doblar(time, flux, period)
                eje.plot(fase, flux, ".", markersize=1.0, color="0.7")
                eje.plot(centros, medias, "-", linewidth=1.2, color="black")
                eje.set_xlabel(f"fase ({etiqueta}: {period:.4f} d)",
                               fontsize=7)
                eje.set_ylim(limites[0] - margen, limites[1] + margen)
                eje.tick_params(labelsize=6)
            fila_ejes[0].set_ylabel("flujo rel. [ppt]", fontsize=6)
            nota = (f"  |  {fila.vis_notes}"
                    if isinstance(fila.vis_notes, str) else "")
            fila_ejes[0].set_title(
                f"TIC {fila.TIC}  ({fila.familia}, {fila.Type})  "
                f"tag {fila.tag}  razon {fila.razon:.3f}  "
                f"prob {fila.probabilidad:.2f}{nota}",
                fontsize=7, loc="left")
            if posicion % POR_PAGINA == POR_PAGINA - 1:
                pdf.savefig(figura)
                plt.close(figura)
                figura = None
        if figura is not None:
            for resto in range(len(fallan) % POR_PAGINA, POR_PAGINA):
                for eje in ejes[resto]:
                    eje.set_axis_off()
            pdf.savefig(figura)
            plt.close(figura)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
