#!/usr/bin/env python
"""Corte de SNR del fundamental sobre el benchmark VSX, solo y junto al de Rndm.

`probe_peaks.py` mide, por candidato, la amplitud de SU fundamental sobre la
curva con las otras variaciones removidas, contra el ruido local del espectro.
Ese número no depende de la CNN, así que se puede aplicar ANTES de clasificar.

Se cruzan los dos cortes:

    SNR      >= umbral   ¿la frecuencia existe?     (no mira la clase)
    clase    != Rndm     ¿el fold tiene forma?      (no mira la potencia)

Como los dos son máscaras por pico, la intersección no depende del orden; lo
que sí cambia es el COSTO MARGINAL de cada uno, que es lo que se tabula: qué
agrega el segundo corte sobre el primero.

    PYTHONPATH=src python scripts/vsx/barrido_snr.py
"""
import argparse

import numpy as np
import pandas as pd

from msv.config import CATALOGS_DIR, RESULTS_DIR

TOLERANCE = 0.05
HARMONICS = {"1:1": 1.0, "2P": 2.0, "P/2": 0.5, "3P": 3.0, "P/3": 1.0 / 3.0}
VEREDICTOS = ["ok", "maybe", "unconstrained"]
UMBRALES = [3.0, 4.0, 5.0, 6.0]


def harmonic_tag(ratio):
    for name, target in HARMONICS.items():
        if np.abs(ratio / target - 1.0) < TOLERANCE:
            return name
    return "otro"


def evaluar(peaks, mascara, etiqueta, estrellas_por_grupo):
    """Picos que sobreviven y recall del 1:1, por veredicto visual."""
    vivos = peaks[mascara]
    filas = {}
    for verdict in VEREDICTOS:
        bloque = vivos[vivos.vis_verdict == verdict]
        total = peaks[peaks.vis_verdict == verdict]
        n_estrellas = estrellas_por_grupo[verdict]
        con_periodo = bloque[bloque.es_vsx].TIC.nunique()
        con_alias = bloque[bloque.es_vsx | bloque.es_alias].TIC.nunique()
        filas[verdict] = {
            "picos": len(bloque),
            "% del total": round(100 * len(bloque) / len(total), 1),
            "picos/estrella": round(len(bloque) / n_estrellas, 1),
            "recall 1:1": round(100 * con_periodo / n_estrellas, 1),
            "recall +alias": round(100 * con_alias / n_estrellas, 1),
            "sin picos": int(n_estrellas - bloque.TIC.nunique()),
        }
    return pd.concat({verdict: pd.Series(valores)
                      for verdict, valores in filas.items()},
                     names=["veredicto"]).rename(etiqueta)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--clasificacion",
                        default=str(RESULTS_DIR / "vsx" / "clasificacion_w001.csv"))
    parser.add_argument("--probe",
                        default=str(RESULTS_DIR / "vsx" / "probe_peaks_w001.csv"))
    args = parser.parse_args()

    review = pd.read_csv(CATALOGS_DIR / "vsx_visual_review.csv")
    review = review[review.vis_verdict.isin(VEREDICTOS) & review.per_vsx.notna()]
    review = review[["TIC", "per_vsx", "vsx_token", "vis_verdict"]]

    peaks = pd.read_csv(args.clasificacion).merge(review, on="TIC", how="inner")
    # step_clasificar_una_red escribe las columnas de la sonda vacías cuando
    # corre sin --probe; si se quedan, el merge las duplica con sufijos.
    peaks = peaks.drop(columns=["snr", "snr_breger", "a2_a1", "snr_half",
                                "amplitude_ppt", "snr_fundamental"],
                       errors="ignore")
    probe = pd.read_csv(args.probe)
    # El período viaja por dos caminos que difieren en el último bit (ver
    # step_clasificar_una_red.period_key): la clave se redondea a 12 cifras.
    for table in (peaks, probe):
        table["clave"] = [f"{value:.12g}" for value in table.per.values]
    probe = probe.drop_duplicates(subset=["TIC", "sector", "clave"])
    peaks = peaks.merge(probe[["TIC", "sector", "clave", "snr", "snr_breger",
                               "a2_a1", "snr_half", "amplitude_ppt"]],
                        on=["TIC", "sector", "clave"], how="left")

    peaks["tag"] = [harmonic_tag(value) for value in peaks.per / peaks.per_vsx]
    peaks["es_vsx"] = peaks.tag == "1:1"
    peaks["es_alias"] = peaks.tag.isin(["2P", "P/2", "3P", "P/3"])

    sin_sonda = peaks.snr.isna()
    if sin_sonda.any():
        print(f"AVISO: {sin_sonda.mean():.1%} de los picos sin sonda; "
              f"se tratan como SNR = 0")
    peaks["snr"] = peaks.snr.fillna(0.0)

    estrellas = review.groupby("vis_verdict").TIC.nunique().to_dict()
    print(f"{len(peaks)} picos, {peaks.TIC.nunique()} estrellas")
    print({verdict: estrellas[verdict] for verdict in VEREDICTOS})

    todo = np.ones(len(peaks), dtype=bool)
    no_rndm = (peaks.clase != "Rndm").values

    print("\n" + "=" * 78)
    print("A. cada corte por separado (el de SNR no usa la CNN)")
    print("=" * 78)
    columnas = [evaluar(peaks, todo, "sin corte", estrellas)]
    for umbral in UMBRALES:
        columnas.append(evaluar(peaks, (peaks.snr >= umbral).values,
                                f"SNR>={umbral:.0f}", estrellas))
    columnas.append(evaluar(peaks, no_rndm, "sin Rndm", estrellas))
    tabla = pd.concat(columnas, axis=1)
    print(tabla.unstack(level=0).to_string())

    print("\n" + "=" * 78)
    print("B. los dos juntos: SNR sobre el conjunto ya sin Rndm")
    print("=" * 78)
    columnas = [evaluar(peaks, no_rndm, "sin Rndm", estrellas)]
    for umbral in UMBRALES:
        columnas.append(evaluar(peaks, no_rndm & (peaks.snr >= umbral).values,
                                f"sin Rndm + SNR>={umbral:.0f}", estrellas))
    tabla = pd.concat(columnas, axis=1)
    print(tabla.unstack(level=0).to_string())

    print("\n" + "=" * 78)
    print("C. qué queda tras 'sin Rndm + SNR>=5': composición por clase")
    print("=" * 78)
    vivos = peaks[no_rndm & (peaks.snr >= 5.0)]
    print("\npicos por clase y veredicto:")
    print(vivos.groupby(["vis_verdict", "clase"]).size()
          .unstack(fill_value=0).to_string())
    print("\nde esos picos, cuántos son el 1:1 de VSX (pureza por clase, "
          "solo las OK):")
    ok = vivos[vivos.vis_verdict == "ok"]
    pureza = ok.groupby("clase").agg(picos=("es_vsx", "size"),
                                     es_vsx=("es_vsx", "sum"),
                                     es_alias=("es_alias", "sum"))
    pureza["% 1:1"] = (100 * pureza.es_vsx / pureza.picos).round(1)
    print(pureza.to_string())

    print("\ntag armónico de los picos supervivientes (solo las OK):")
    print(ok.tag.value_counts().to_string())

    print("\n" + "=" * 78)
    print("D. candidatos donde el ARMÓNICO domina (a2_a1 > 2): el alias 2P")
    print("=" * 78)
    ok_todo = peaks[(peaks.vis_verdict == "ok")]
    for nombre, mascara in [("todos", np.ones(len(ok_todo), bool)),
                            ("sin Rndm + SNR>=5",
                             (ok_todo.clase != "Rndm").values
                             & (ok_todo.snr >= 5.0).values)]:
        bloque = ok_todo[mascara]
        domina = bloque.a2_a1 > 2
        print(f"\n{nombre}: {len(bloque)} picos, {int(domina.sum())} con "
              f"a2/a1 > 2 ({100 * domina.mean():.1f}%)")
        print("  de los que dominan, su tag:")
        print(bloque[domina].tag.value_counts().to_string())
        print("  de los que NO dominan, su tag:")
        print(bloque[~domina].tag.value_counts().to_string())

    peaks.to_csv(RESULTS_DIR / "vsx" / "peaks_con_snr.csv", index=False)


if __name__ == "__main__":
    main()
