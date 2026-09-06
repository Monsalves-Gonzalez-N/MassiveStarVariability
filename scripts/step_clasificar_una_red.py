#!/usr/bin/env python
"""Paso 3/3: clasificación con UN checkpoint de la CNN, sin BRF ni ensemble.

Reemplaza a `step_brf.py` y a `step_cnn_only.py`. Ver
`docs/HANDOFF_contaminacion_ELL.md` §3 para los números que justifican cada
decisión; en resumen:

  - un solo checkpoint, `Number_ELL`: acierta la clase en 15 de los 16 picos
    con período humano confirmado, contra 14 del ensemble de 7 más el BRF
  - sin BRF: sus 500 árboles no pueden representar nada por debajo de 2.9e-4 y
    aplastan a cero el rango 1e-18..1e-4 de la CNN, que es donde `p_LPV` separa
  - el pico reportado por estrella se elige por MENOR `p_LPV`, no por mayor
    probabilidad de clase: esa probabilidad tiene mediana 0.996 y ordenar por
    ella es casi arbitrario. Elegir por `p_LPV` sube de 8/11 a 10/11 las
    estrellas con el período correcto, y corrige dos de los tres alias 2P.
  - con `--probe` (la salida de `probe_peaks.py`) se aplica el VETO DEL
    FUNDAMENTAL: un candidato cuyo fundamental no llega a SNR 4 no se reporta,
    y de los que sobreviven se reporta el de mayor SNR — no el de mayor
    `log_pLPV`, que es la apariencia de una imagen, y no filtrado por clase.
    Si no sobrevive ninguno la estrella NO reporta período, que es una
    respuesta que antes no existía. Medido sobre la golden, los reportes ELL
    caen de 141 a 42 (-70%) bajo una regla que nunca mira la clase.
  - sobre el candidato reportado se adjudica el armónico: si el fundamental en
    2P también llega a SNR 4 se reporta 2P (eclipses desiguales: ese es el
    orbital), y si no se reporta P por parsimonia. Una doble onda SIMÉTRICA es
    fotométricamente idéntica en P y en 2P, así que la regla elige el período
    parsimonioso cuando los datos no distinguen; no prueba que 2P esté mal.
  - con `--curves`, un piso de `CICLOS_MIN` ciclos (baseline/P) descarta los
    candidatos submuestreados. No es un umbral ajustado: con 2 ciclos no se
    establece un período, y la CNN se entrenó con OGLE, cuyos baselines son de
    años, así que un fold de 3 ciclos está fuera de distribución. Cuesta 0 de
    los 16 picos con período confirmado y saca 3 de las 35 estrellas
    reportadas, todas espurias. La clase ELL es la que más lo necesita: se
    lleva el 19% de los picos de 3-4 ciclos contra el 0.6% de los de más de
    20 (factor 32).

Columnas de salida por pico: la clase (argmax), su probabilidad `prob`, las
cinco probabilidades agrupadas `p_<grupo>`, `log_pLPV` = -log10(p_LPV) como
confianza, y dos cantidades por ESTRELLA repetidas en sus picos: `irregular`
(masa media en LPV sobre todos los picos de la estrella-sector) y `reportado`.

`prob` se escribe pero NO se filtra con ella: con 26 estrellas revisadas lo
único medible es que las 5 con `prob < 0.7` son las 5 erradas. Ver §5.

Mantiene el contrato de columnas de `step_brf.py` (`sigma`, `s_<grupo>`,
`med_`/`q1_`/`q3_`, `v_<grupo>`) para que los PDF de revisión y la cascada
sigan funcionando, pero **con una sola pasada determinista esas columnas son
degeneradas**: sigma vale 0, la mediana y los cuartiles valen la media, y el
voto es 0 o 1. No es que la dispersión sea chica: no existe. El número que
mide confianza acá es `log_pLPV`.

    python scripts/step_clasificar_una_red.py results/cnn_input.npz \
        results/cnn_mc.npz results/clasificacion_una_red.csv
"""
import argparse

import numpy as np
import pandas as pd

from msv.classify_brf import group_probs
from msv.config import CLASS_NAMES, MODELS

META_COLS = ["TIC", "sector", "source", "per", "power", "prominence", "width",
             "amplitude"]
PERIODIC_CLASSES = ["ELL", "Pulsating", "E"]
# Debajo de este p_LPV el pico es fiable: 7 de 10 detecciones confirmadas
# sobreviven sin dejar pasar ningun error, con 1.5 decadas de margen.
LOG_PLPV_MIN = 12.0
# Ciclos = baseline / P. Debajo de esto el fold no cubre suficientes ciclos
# para que "periodo" signifique algo, y es donde se acumulan las ELL espurias.
CICLOS_MIN = 4.0
# En ELL una masa irregular alta invalida la clase (una elipsoidal es UNA
# modulacion continua). En E no: un eclipse sobrevive superpuesto a otra cosa.
IRREGULAR_MAX_ELL = 0.05
# Breger+ 1993 con la mediana en lugar de la media (ver msv.prewhiten.
# local_noise): por debajo de esto el fundamental del candidato esta vacio y
# la frecuencia no existe, diga lo que diga la forma del fold.
SNR_MIN = 4.0
FLOOR = 1e-30
PROBE_COLS = ["snr", "snr_breger", "a2_a1", "snr_half", "amplitude_ppt"]


def period_key(periods):
    return [f"{value:.12g}" for value in np.asarray(periods, float)]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("npz_in", help=".npz con X y metadatos (step_cnn_export.py)")
    parser.add_argument("npz_probs", help=".npz con p_mc (n_pasadas, N, 8)")
    parser.add_argument("csv_out")
    parser.add_argument("--model", default="Number_ELL",
                        help="checkpoint a usar si npz_probs tiene varias pasadas")
    parser.add_argument("--probe", default=None,
                        help="csv de probe_peaks.py: SNR del fundamental por "
                             "candidato; sin esto no se aplica el veto y el "
                             "pico se elige por mayor log_pLPV, como antes")
    parser.add_argument("--snr-min", type=float, default=SNR_MIN)
    parser.add_argument("--curves", default=None,
                        help="pickle de curvas para el baseline; sin esto no se "
                             "aplica el piso de ciclos")
    args = parser.parse_args()

    data = np.load(args.npz_in, allow_pickle=True)
    table = pd.DataFrame({column: data[column] for column in META_COLS})
    probabilities = np.load(args.npz_probs)["p_mc"].astype(float)

    if probabilities.shape[0] == 1:
        fine = probabilities[0]
    else:
        if args.model not in MODELS:
            raise SystemExit(f"--model debe ser uno de {MODELS}")
        if probabilities.shape[0] != len(MODELS):
            raise SystemExit(
                f"{args.npz_probs} tiene {probabilities.shape[0]} pasadas; para "
                f"elegir un checkpoint por nombre debe tener {len(MODELS)}, una "
                f"por modelo de config.MODELS")
        fine = probabilities[MODELS.index(args.model)]

    grouped, group_names = group_probs(fine)
    winner = grouped.argmax(axis=1)
    table["clase"] = [group_names[index] for index in winner]
    table["prob"] = grouped.max(axis=1)
    table["sigma"] = 0.0
    for position, name in enumerate(group_names):
        table[f"p_{name}"] = grouped[:, position]
        table[f"s_{name}"] = 0.0
        table[f"med_{name}"] = grouped[:, position]
        table[f"q1_{name}"] = grouped[:, position]
        table[f"q3_{name}"] = grouped[:, position]
        table[f"lo_{name}"] = grouped[:, position]
        table[f"hi_{name}"] = grouped[:, position]
        table[f"v_{name}"] = (winner == position).astype(float)
    # El argmax FINO, sin agrupar: cascade.path2_cascade lo lee por nombre.
    table["cnn_class"] = [CLASS_NAMES[index] for index in fine.argmax(axis=1)]
    table["log_pLPV"] = -np.log10(np.clip(table.p_LPV.values, FLOOR, 1))

    key = table.TIC.astype(str) + "_" + table.sector.astype(str)
    table["irregular"] = key.map(table.groupby(key).p_LPV.mean())

    table["baseline"] = np.nan
    if args.curves:
        curves = pd.read_pickle(args.curves)
        for (tic, sector), (time, flux) in curves.items():
            good = np.isfinite(time) & np.isfinite(flux)
            if not good.any():
                continue
            same = (table.TIC.values == tic) & (table.sector.values == sector)
            table.loc[same, "baseline"] = time[good].max() - time[good].min()
    table["ciclos"] = table.baseline / table.per
    # Sin curvas no hay baseline: el piso no filtra nada en vez de filtrar todo.
    muestreado = ~np.isfinite(table.ciclos) | (table.ciclos >= CICLOS_MIN)

    for column in PROBE_COLS:
        table[column] = np.nan
    if args.probe:
        # El periodo viaja por el .npz de un lado y por un .csv del otro, y los
        # dos caminos difieren en el ultimo bit: cruzar por el float crudo
        # pierde 1424 de 9651 picos. La clave se redondea a 12 cifras
        # significativas, muy por encima de la precision de un periodo y muy
        # por debajo de la separacion entre dos candidatos distintos.
        probes = pd.read_csv(args.probe)
        probes["clave"] = period_key(probes.per)
        probes = probes.drop_duplicates(subset=["TIC", "sector", "clave"],
                                        keep="first")
        left = table[["TIC", "sector"]].assign(clave=period_key(table.per))
        merged = left.merge(probes[["TIC", "sector", "clave"] + PROBE_COLS],
                            on=["TIC", "sector", "clave"], how="left")
        for column in PROBE_COLS:
            table[column] = merged[column].values
        sin_sonda = table.snr.isna().mean()
        if sin_sonda > 0:
            print(f"AVISO: {sin_sonda:.1%} de los picos no tienen sonda en "
                  f"{args.probe}; esos no pasan el veto")

    table["snr_fundamental"] = table.snr
    table["pasa_veto"] = table.snr_fundamental >= args.snr_min

    table["reportado"] = False
    if args.probe:
        # El veto no mira la clase: el candidato se elige por potencia real en
        # su fundamental, y la red opina despues sobre la forma del que gano.
        elegible = table.pasa_veto & muestreado
        criterio = table.snr_fundamental
    else:
        elegible = table.clase.isin(PERIODIC_CLASSES) & muestreado
        criterio = table.log_pLPV
    for star, block in criterio[elegible].groupby(key[elegible]):
        table.loc[block.idxmax(), "reportado"] = True

    # Adjudicacion P vs 2P sobre el candidato reportado (msv.prewhiten.probe
    # sondea f/2 aparte porque los armonicos solo miran hacia arriba).
    doblar = table.reportado & (table.snr_half >= args.snr_min)
    table["armonico"] = np.where(doblar, "2P", "P")
    table["per_reportado"] = np.where(table.reportado,
                                      np.where(doblar, 2.0 * table.per, table.per),
                                      np.nan)

    fiable = table.log_pLPV >= LOG_PLPV_MIN
    irregular_ok = (table.clase != "ELL") | (table.irregular < IRREGULAR_MAX_ELL)
    con_senal = table.pasa_veto if args.probe else True
    table["nivel"] = np.where(fiable & irregular_ok & muestreado & con_senal,
                              "alta", "baja")

    table.to_csv(args.csv_out, index=False)
    reported = table[table.reportado]
    n_stars = key.nunique()
    print(f"{len(table)} picos, {n_stars} estrella-sector, "
          f"{len(reported)} con periodo reportado")
    print(reported.groupby(["clase", "nivel"]).size().to_string())
    print(f"\nnivel alta: {int((reported.nivel == 'alta').sum())} estrellas")
    if args.probe:
        print(f"\ncandidatos que pasan el veto (SNR >= {args.snr_min}): "
              f"{int(table.pasa_veto.sum())}/{len(table)} "
              f"({table.pasa_veto.mean():.1%})")
        print(f"estrella-sector sin ningun candidato con fundamental: "
              f"{n_stars - len(reported)}/{n_stars}")
        print(f"reportados con fundamental SNR >= {args.snr_min}: "
              f"{reported.pasa_veto.mean():.1%}")
        print(f"adjudicados a 2P: {int((reported.armonico == '2P').sum())}"
              f"/{len(reported)}")


if __name__ == "__main__":
    main()
