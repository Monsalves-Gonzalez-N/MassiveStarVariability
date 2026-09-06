#!/usr/bin/env python
"""¿Cuánto gana el árbol de selección al agregar features de período y amplitud?

Dos advertencias que ordenan qué vale la pena agregar:

  - `amplitude` (la de `run_peaks`) es de la ESTRELLA, no del pico: es idéntica
    en los 377 grupos del benchmark. Elegir un pico dentro de una estrella con
    una cantidad constante en esa estrella es imposible. La amplitud por pico es
    `amplitude_ppt`, la del fundamental que ajusta la sonda.
  - un árbol es invariante a transformaciones monótonas de UNA feature, así que
    `log(per)` no agrega nada sobre `per`. Lo que sí agrega información son los
    COCIENTES: la amplitud del candidato contra la de la estrella, su rango
    dentro de la estrella, y las cantidades del PAR (candidato, su mitad/doble),
    que es donde vive la adjudicación del factor 2.

Se comparan tres conjuntos de features con la misma partición out-of-fold
(`GroupKFold` por TIC, promediado sobre 5 permutaciones de estrellas).

    PYTHONPATH=src python scripts/vsx/arbol_features.py
"""
import argparse

import numpy as np
import pandas as pd
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, export_text

from msv.config import CLASS_NAMES, RESULTS_DIR

TOLERANCE = 0.05
PROBE_COLS = ["snr", "snr_model", "snr_breger", "a2_a1", "snr_half",
              "snr_half_model", "amplitude_ppt"]

BASE = ["power", "prominence", "width", "amplitude", "per", "ciclos",
        "p_ELL", "p_Pulsating", "p_E", "p_LPV", "p_Rndm", "log_pLPV", "prob",
        "irregular", "snr", "snr_model", "a2_a1", "snr_half",
        "snr_half_model", "amplitude_ppt",
        "n_picos", "rango_power", "rango_plpv", "rango_per",
        "es_max_power", "es_max_plpv", "power_rel",
        "rango_power_fuente", "es_LS", "per_rel", "snr_rel"]
# Lo que se puede calcular SIN la sonda: el periodograma y la salida de la red.
# `ciclos` = baseline / per, y el contexto es reordenamiento de esas mismas
# cantidades dentro de la estrella, asi que ninguna toca la SNR.
# `per` en crudo NO entra: los candidatos ya salen del periodograma, asi que un
# corte en dias absolutos aprende la distribucion de periodos publicados por
# VSX, no una regla de seleccion. Lo que queda son cantidades RELATIVAS dentro
# de la estrella. `ciclos` = baseline/per es monotona en `per` dentro de un
# sector, asi que se aisla aparte para poder medir cuanto de eso reintroduce.
PERIODOGRAMA = ["power", "prominence", "width",
                "n_picos", "rango_power", "rango_per", "es_max_power",
                "power_rel", "rango_power_fuente", "es_LS", "per_rel"]
PERIODO_ABSOLUTO = ["per"]
CICLOS = ["ciclos"]
CNN = ["p_ELL", "p_Pulsating", "p_E", "p_LPV", "p_Rndm", "log_pLPV", "prob",
       "rango_plpv", "es_max_plpv"]

# La sonda de UNA pasada: todo lo que sale de sondear el fundamental del
# candidato. `snr_half` exige una segunda llamada a `probe` sobre 0.5/per y
# queda fuera a proposito.
SNR_1PASADA = ["snr", "snr_model", "snr_breger", "a2_a1", "amplitude_ppt",
               "snr_rel"]

# Las 8 clases FINAS de la CNN, tal como salen de la red. El pipeline las
# colapsaba a 5 grupos (M/CEP/RR/DST -> Pulsating) y perdia como se reparte la
# probabilidad entre las pulsantes. Salen de la misma pasada determinista que ya
# se corre, asi que no cuestan nada.
FINAS = [f"fina_{name}" for name in CLASS_NAMES]
CNN_FINO = FINAS + ["log_pLPV", "prob", "rango_plpv", "es_max_plpv"]

CONTEXTO_2P = ["hay_pico_2P", "hay_pico_P2"]
NUEVAS = ["amp_frac_estrella", "amp_rel", "rango_amp", "amp_por_ciclo",
          "per_dias_log_rel", "densidad_periodos",
          "power_ratio_P2", "amp_ratio_P2", "clase_P2_es_puls",
          "power_ratio_2P", "amp_ratio_2P", "clase_2P_es_ell",
          "snr_ratio_P2", "snr_ratio_2P"]


def cargar(probe_csv, verdicts=("ok",), peaks_csv=None):
    peaks = pd.read_csv(peaks_csv
                        or RESULTS_DIR / "vsx" / "peaks_con_snr.csv")
    probe = pd.read_csv(probe_csv)
    for table in (peaks, probe):
        table["clave"] = [f"{value:.12g}" for value in table.per.values]
    probe = probe.drop_duplicates(subset=["TIC", "sector", "clave"])
    peaks = peaks.drop(columns=[c for c in PROBE_COLS if c in peaks.columns])
    peaks = peaks.merge(probe[["TIC", "sector", "clave"] + PROBE_COLS],
                        on=["TIC", "sector", "clave"], how="left")
    for column in PROBE_COLS:
        peaks[column] = peaks[column].fillna(0.0)
    return peaks[peaks.vis_verdict.isin(verdicts)
                 & (peaks.clase != "Rndm")].reset_index(drop=True)


def agregar_finas(peaks, npz_path=None, clasificacion_path=None):
    """Las 8 probabilidades sin agrupar, cruzadas por (TIC, sector, periodo)."""
    npz_path = npz_path or RESULTS_DIR / "vsx" / "cnn_probs_w001.npz"
    clasificacion_path = (clasificacion_path
                          or RESULTS_DIR / "vsx" / "clasificacion_w001.csv")
    finas = np.load(npz_path, allow_pickle=True)["p_mc"][0]
    tabla = pd.read_csv(clasificacion_path, usecols=["TIC", "sector", "per"])
    for position, name in enumerate(CLASS_NAMES):
        tabla[f"fina_{name}"] = finas[:, position]
    tabla["clave"] = [f"{value:.12g}" for value in tabla.per.values]
    tabla = tabla.drop(columns=["per"]).drop_duplicates(
        subset=["TIC", "sector", "clave"])
    peaks = peaks.merge(tabla, on=["TIC", "sector", "clave"], how="left")
    peaks[FINAS] = peaks[FINAS].fillna(0.0)
    return peaks


def contexto(peaks):
    peaks = peaks.copy()
    grupo = peaks.groupby("TIC")
    peaks["n_picos"] = grupo.per.transform("size")
    peaks["rango_power"] = grupo.power.rank(ascending=False)
    peaks["rango_plpv"] = grupo.log_pLPV.rank(ascending=False)
    peaks["rango_per"] = grupo.per.rank(ascending=False)
    peaks["es_max_power"] = (peaks.rango_power == 1).astype(int)
    peaks["es_max_plpv"] = (peaks.rango_plpv == 1).astype(int)
    por_fuente = peaks.groupby(["TIC", "source"])
    peaks["power_rel"] = peaks.power / por_fuente.power.transform("max")
    peaks["rango_power_fuente"] = por_fuente.power.rank(ascending=False)
    peaks["es_LS"] = (peaks.source == "LS").astype(int)
    peaks["per_rel"] = peaks.per / grupo.per.transform("max")
    peaks["snr_rel"] = peaks.snr / grupo.snr.transform("max").replace(0, np.nan)

    # --- amplitud, la única versión por pico -------------------------------
    # `amplitude` es de la estrella; la fracción de ESA amplitud que el
    # candidato explica sí distingue entre sus picos.
    peaks["amp_frac_estrella"] = peaks.amplitude_ppt / (1e3 * peaks.amplitude)
    peaks["amp_rel"] = (peaks.amplitude_ppt
                        / grupo.amplitude_ppt.transform("max").replace(0, np.nan))
    peaks["rango_amp"] = grupo.amplitude_ppt.rank(ascending=False)
    # Amplitud por ciclo observado: una señal de amplitud alta vista 3 veces no
    # está establecida igual que la misma vista 40 veces.
    peaks["amp_por_ciclo"] = peaks.amplitude_ppt / peaks.ciclos.replace(0, np.nan)

    # --- período, en forma NO monótona -------------------------------------
    # log(per) no aporta a un árbol; el período relativo a la MEDIANA de la
    # estrella sí, porque reordena los picos según dónde cae la estrella.
    peaks["per_dias_log_rel"] = np.log10(
        peaks.per / grupo.per.transform("median"))
    # Cuántos candidatos hay por década de período: mide si la estrella tiene
    # un peine denso o dos frecuencias sueltas.
    rango = (grupo.per.transform("max") / grupo.per.transform("min")).clip(1.01)
    peaks["densidad_periodos"] = peaks.n_picos / np.log10(rango)

    # --- el PAR (candidato, su mitad y su doble) ---------------------------
    for factor, sufijo in [(0.5, "P2"), (2.0, "2P")]:
        ratios_power, ratios_amp, ratios_snr, clase_par = [], [], [], []
        for _, block in peaks.groupby("TIC", sort=False):
            for _, row in block.iterrows():
                cerca = block[np.abs(block.per / (factor * row.per) - 1)
                              < TOLERANCE]
                if cerca.empty:
                    ratios_power.append(0.0)
                    ratios_amp.append(0.0)
                    ratios_snr.append(0.0)
                    clase_par.append("")
                    continue
                par = cerca.sort_values("power", ascending=False).iloc[0]
                ratios_power.append(float(par.power / row.power)
                                    if row.power else 0.0)
                ratios_amp.append(float(par.amplitude_ppt / row.amplitude_ppt)
                                  if row.amplitude_ppt else 0.0)
                ratios_snr.append(float(par.snr / row.snr) if row.snr else 0.0)
                clase_par.append(par.clase)
        peaks[f"power_ratio_{sufijo}"] = ratios_power
        peaks[f"amp_ratio_{sufijo}"] = ratios_amp
        peaks[f"snr_ratio_{sufijo}"] = ratios_snr
        peaks[f"clase_par_{sufijo}"] = clase_par
        peaks[f"hay_pico_{sufijo}"] = (np.asarray(ratios_power) > 0).astype(int)
    peaks["clase_P2_es_puls"] = (peaks.clase_par_P2 == "Pulsating").astype(int)
    peaks["clase_2P_es_ell"] = (peaks.clase_par_2P == "ELL").astype(int)
    return peaks


# La firma de la clase esta en el fold aunque no sepamos la clase a priori, y
# las familias de VSX estan muy desbalanceadas (los eclipsantes son la mitad de
# la muestra). Sin pesos, el selector optimiza la regla de las ECL y las ELL —
# que son justo donde vive la ambiguedad del factor 2 — no pesan nada.
FAMILIAS = {
    "E": "ECL", "EA": "ECL", "EB": "ECL", "EC": "ECL", "ED": "ECL",
    "ESD": "ECL", "EW": "ECL",
    "ELL": "ELL",
    "BCEP": "PULS", "DSCT": "PULS", "SPB": "PULS", "SXARI": "PULS",
    "PULS": "PULS",
    # las Cefeidas son pulsantes: misma familia, no una aparte
    "CEP": "PULS", "CWA": "PULS", "CWB": "PULS", "DCEP": "PULS",
    "DCEPS": "PULS",
}


def familia(peaks):
    return peaks.vsx_token.map(FAMILIAS).fillna("otro")


def pesos_balanceados(peaks):
    """Cada celda (familia, es_vsx) aporta la misma masa total."""
    celda = peaks.familia + "|" + peaks.es_vsx.astype(int).astype(str)
    conteo = celda.value_counts()
    weights = (len(peaks) / (len(conteo) * celda.map(conteo))).values
    return weights.astype(float)


def ajustar(modelo, X, y, weights):
    if hasattr(modelo, "steps"):
        paso = modelo.steps[-1][0]
        modelo.fit(X, y, **{f"{paso}__sample_weight": weights})
    else:
        modelo.fit(X, y, sample_weight=weights)
    return modelo


def acierto(peaks, score):
    elegidos = peaks.assign(_s=score).sort_values(
        "_s", ascending=False).groupby("TIC").head(1)
    reparto = elegidos.tag.value_counts(normalize=True) * 100
    return (round(reparto.get("1:1", 0.0), 1), round(reparto.get("P/2", 0.0), 1),
            round(reparto.get("2P", 0.0), 1))


def acierto_por_familia(peaks, score):
    elegidos = peaks.assign(_s=score).sort_values(
        "_s", ascending=False).groupby("TIC").head(1)
    return elegidos.groupby("familia").apply(
        lambda block: 100 * (block.tag == "1:1").mean())


def scores_out_of_fold(peaks, columnas, make_model, semilla, weights):
    X = peaks[columnas].replace([np.inf, -np.inf], np.nan).fillna(0.0).values
    y = peaks.es_vsx.values.astype(int)
    tics = peaks.TIC.values
    orden = np.random.RandomState(semilla).permutation(np.unique(tics))
    posicion = {tic: index for index, tic in enumerate(orden)}
    grupos = np.array([posicion[tic] for tic in tics])
    scores = np.zeros(len(peaks))
    for entrena, prueba in GroupKFold(5).split(X, y, grupos):
        modelo = ajustar(make_model(), X[entrena], y[entrena],
                         weights[entrena])
        scores[prueba] = modelo.predict_proba(X[prueba])[:, 1]
    return scores


def out_of_fold(peaks, columnas, make_model, repeticiones=5):
    X = peaks[columnas].replace([np.inf, -np.inf], np.nan).fillna(0.0).values
    y = peaks.es_vsx.values.astype(int)
    tics = peaks.TIC.values
    weights = pesos_balanceados(peaks)
    aciertos = []
    for semilla in range(repeticiones):
        orden = np.random.RandomState(semilla).permutation(np.unique(tics))
        posicion = {tic: index for index, tic in enumerate(orden)}
        grupos = np.array([posicion[tic] for tic in tics])
        scores = np.zeros(len(peaks))
        for entrena, prueba in GroupKFold(5).split(X, y, grupos):
            modelo = ajustar(make_model(), X[entrena], y[entrena],
                             weights[entrena])
            scores[prueba] = modelo.predict_proba(X[prueba])[:, 1]
        aciertos.append(acierto(peaks, scores)[0])
    return float(np.mean(aciertos)), float(np.std(aciertos))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repeticiones", type=int, default=5)
    parser.add_argument("--probe",
                        default=str(RESULTS_DIR / "vsx" /
                                    "probe_peaks_w001_model.csv"),
                        help="tabla de la sonda a cruzar contra los picos")
    args = parser.parse_args()

    peaks = contexto(cargar(args.probe))
    peaks["familia"] = familia(peaks)
    print(f"{len(peaks)} picos de {peaks.TIC.nunique()} estrellas OK, "
          f"{int(peaks.es_vsx.sum())} son el 1:1")
    estrellas = peaks.drop_duplicates("TIC")
    print("estrellas por familia:",
          estrellas.familia.value_counts().to_dict())
    print(f"referencia log_pLPV: {acierto(peaks, peaks.log_pLPV)[0]}%")
    print("  por familia:",
          acierto_por_familia(peaks, peaks.log_pLPV).round(1).to_dict(), "\n")

    conjuntos = {
        "sin per ni ciclos": PERIODOGRAMA + CNN + SNR_1PASADA,
        "+ ciclos": PERIODOGRAMA + CICLOS + CNN + SNR_1PASADA,
        "+ ciclos + per absoluto": (PERIODOGRAMA + CICLOS + PERIODO_ABSOLUTO
                                    + CNN + SNR_1PASADA),
    }
    modelos = {
        "arbol depth=6": lambda: DecisionTreeClassifier(
            max_depth=6, min_samples_leaf=20, random_state=0),
        "random forest": lambda: RandomForestClassifier(
            n_estimators=400, min_samples_leaf=5, random_state=0, n_jobs=-1),
        "balanced RF": lambda: BalancedRandomForestClassifier(
            n_estimators=400, min_samples_leaf=5, random_state=0, n_jobs=-1),
        "SVM rbf": lambda: make_pipeline(
            StandardScaler(),
            SVC(kernel="rbf", C=10.0, gamma="scale", probability=True,
                random_state=0)),
        "gradient boosting": lambda: GradientBoostingClassifier(
            n_estimators=200, max_depth=3, random_state=0),
    }
    filas = []
    for nombre_conjunto, columnas in conjuntos.items():
        fila = {"features": nombre_conjunto, "n": len(columnas)}
        for nombre_modelo, make in modelos.items():
            media, dispersion = out_of_fold(peaks, columnas, make,
                                            args.repeticiones)
            fila[nombre_modelo] = f"{media:.1f} ± {dispersion:.1f}"
        filas.append(fila)
    print("acierto 1:1 out-of-fold [%], GroupKFold(5) por TIC")
    print(pd.DataFrame(filas).set_index("features").to_string())

    columnas = PERIODOGRAMA + CICLOS + CNN + SNR_1PASADA
    weights = pesos_balanceados(peaks)
    print("\nacierto 1:1 por familia [%], boosting sobre '+ ciclos' "
          "(out-of-fold, 5 semillas)")
    por_familia = pd.DataFrame([
        acierto_por_familia(
            peaks,
            scores_out_of_fold(peaks, columnas,
                               lambda: GradientBoostingClassifier(
                                   n_estimators=200, max_depth=3,
                                   random_state=0), semilla, weights))
        for semilla in range(args.repeticiones)])
    resumen = pd.DataFrame({
        "estrellas": peaks.drop_duplicates("TIC").familia.value_counts(),
        "acierto": por_familia.mean().round(1),
        "sigma": por_familia.std().round(1),
    })
    print(resumen.to_string())

    X = peaks[columnas].replace([np.inf, -np.inf], np.nan).fillna(0.0).values
    modelo = GradientBoostingClassifier(n_estimators=200, max_depth=3,
                                        random_state=0)
    ajustar(modelo, X, peaks.es_vsx.values.astype(int), weights)
    importancia = pd.Series(modelo.feature_importances_, index=columnas)
    print("\nimportancia (boosting sobre todo, solo para leerlo):")
    print(importancia.sort_values(ascending=False).head(15).round(3).to_string())

    arbol = DecisionTreeClassifier(max_depth=3, min_samples_leaf=20,
                                   random_state=0)
    ajustar(arbol, X, peaks.es_vsx.values.astype(int), weights)
    print("\n" + export_text(arbol, feature_names=columnas, decimals=2))


if __name__ == "__main__":
    main()
