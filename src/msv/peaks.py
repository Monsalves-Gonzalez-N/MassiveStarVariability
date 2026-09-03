"""Selección de picos en periodogramas LS/ACF con scipy.signal.find_peaks.

Reemplaza el suavizado gaussiano del ACF (gaussian_filter1d, sigma=50 pts)
que degradaba la resolución de la grilla de períodos: los picos operan
SIEMPRE sobre el power crudo, con FAP + prominencia + una separación mínima
entre picos.

- ACF: separación adaptativa de Rayleigh (default): dos picos a menos de
  k·P²/T (T = baseline) no son distinguibles físicamente. Como `distance`
  de scipy es un escalar, se aplica como post-filtro greedy por prominencia
  (`_rayleigh_nms`). Alternativa legacy: ventana temporal FIJA
  (`min_peak_sep_days`) vía distance = round(sep/cadence) — ojo: una fija
  de 0.5 d suprimía el fundamental de estrellas con per < 0.5 d y dejaba
  el armónico 2x.
- LS: la grilla es uniforme en FRECUENCIA; una separación fija en período no
  es constante en frecuencia. Se convierte con |Δf| ≈ ΔP/P² evaluado en el
  período más largo buscado (conversión conservadora: garantiza al menos esa
  separación en todo el rango). Por defecto queda desactivada para LS: sus
  picos son naturalmente angostos y basta FAP + prominencia.
"""
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, peak_prominences, peak_widths

from .config import (ACF_PROMINENCE_FRAC, ACF_PROMINENCE_K_FAP,
                     ACF_WIDTH_FRAC, ACF_WIDTH_MAX_SAMPLES,
                     ACF_RAYLEIGH_K, HARMONIC_FIT_TOL, HARMONIC_MAX_ORDER,
                     HARMONIC_MIN_RUN, HARMONIC_TOL, MIN_PEAK_SEP_DAYS)

_EMPTY = ["per", "power", "prominence", "width"]


def _find(power, height=None, distance=None, prominence=None, width=None):
    """find_peaks + prominencias/anchos garantizados (aunque no se pidan)."""
    idx, props = find_peaks(power, height=height, distance=distance,
                            prominence=prominence, width=width)
    if idx.size == 0:
        return idx, np.array([]), np.array([])
    prom = props.get("prominences")
    if prom is None:
        prom = peak_prominences(power, idx)[0]
    wid = props.get("widths")
    if wid is None:
        wid = peak_widths(power, idx)[0]
    return idx, prom, wid


def _build(per, power, idx, prom, wid, sort_by, top_n):
    if idx.size == 0:
        return pd.DataFrame(columns=_EMPTY)
    out = pd.DataFrame({
        "per": np.asarray(per)[idx],
        "power": np.asarray(power)[idx],
        "prominence": prom,
        "width": wid,
    }).sort_values(sort_by, ascending=False).reset_index(drop=True)
    if top_n is not None:
        out = out.head(top_n).reset_index(drop=True)
    return out


def _rayleigh_nms(per_peaks, prom, min_freq_sep):
    """Supresión greedy por prominencia con separación mínima en FRECUENCIA.

    Acepta picos en orden de prominencia descendente; descarta los que caen
    a menos de `min_freq_sep` [ciclos/día] de uno ya aceptado.

    El criterio de Rayleigh vive en frecuencia (|Δf| < 1/T -> indistinguibles)
    y ahí es una constante, así que no hay ventana por-pico ni asimetría que
    resolver. Su forma en período, ΔP > k·P²/T, es solo la linealización
    local: como radio de exclusión alrededor de un pico crece con P² y a
    lags largos supera el rango completo, borrando el fundamental y dejando
    su armónico n-ésimo (TIC 316947536 s83: el pico de 10.06 d excluía todo
    hasta 12.18 d y se comía el fundamental de 2.06 d).

    Devuelve máscara booleana sobre los picos de entrada.
    """
    freq_peaks = 1.0 / np.asarray(per_peaks, dtype=float)
    order = np.argsort(prom)[::-1]
    keep = np.zeros(freq_peaks.size, dtype=bool)
    kept_freq = []
    for j in order:
        f = freq_peaks[j]
        if all(abs(f - g) >= min_freq_sep for g in kept_freq):
            keep[j] = True
            kept_freq.append(f)
    return keep


def select_peaks_acf(per, power, fap, cadence=None, *,
                     rayleigh_k=ACF_RAYLEIGH_K,
                     min_peak_sep_days=MIN_PEAK_SEP_DAYS,
                     prominence_k_fap=ACF_PROMINENCE_K_FAP,
                     prominence_frac=ACF_PROMINENCE_FRAC,
                     width_frac=ACF_WIDTH_FRAC,
                     width_max_samples=ACF_WIDTH_MAX_SAMPLES, width=None,
                     sort_by="prominence", top_n=None):
    """Picos del ACF sobre power CRUDO con separación mínima adaptativa.

    - `height = fap`: array de Bartlett dependiente del lag — solo picos
      significativos.
    - `prominence_k_fap`: prominencia mínima como múltiplo de `fap[lag]`,
      es decir un umbral POR LAG con la misma escala de ruido que ya usa
      `height`. Suprime los wiggles sobre el decaimiento de red noise del
      ACF (lo que antes hacía el smoothing gaussiano, pero sin perder
      resolución de grilla).
    - `width_frac`: ancho mínimo `width_frac * P / cadencia` muestras, acotado
      por `width_max_samples`, como array a `width` de find_peaks. Un pico real del ACF es ancho porque la
      correlación persiste; un spike de ruido dura 1-2 muestras. Es el único
      de los tres criterios que no está correlacionado con los otros dos en
      ruido blanco denso (ver ACF_WIDTH_FRAC en config).
    - `width`: piso ABSOLUTO en muestras, alternativo. Si se pasa, reemplaza
      a `width_frac`.
    - `prominence_frac`: legacy, fracción del máximo del power. Si se pasa,
      reemplaza a `prominence_k_fap`. Se normalizaba por la SEÑAL: una
      estrella sin periodicidad tiene max(ACF) pequeño y por lo tanto un
      umbral más permisivo, así que el ruido pasaba el filtro.
    - `rayleigh_k`: separación mínima k/T en FRECUENCIA entre picos
      (post-filtro greedy por prominencia). T se toma como 2·max(per) —
      asume el default del pipeline maxlag = baseline/2. Dos picos más
      cercanos que el límite de Rayleigh no son distinguibles físicamente.
    - `min_peak_sep_days`: alternativa legacy — ventana temporal FIJA vía
      `distance` de find_peaks (requiere `cadence`). Si se pasa, reemplaza
      a la de Rayleigh. Ojo: 0.5 d suprimía el fundamental de per < 0.5 d.

    Validación 2026-07-26 (67 pares label=1, 24 FP_PAIRS, top-3
    exact+harmonic TOL=0.05): Rayleigh k=3 + prominence 0.20 → recovery
    77.6% / 0 espurios (fija 0.5 d + prominence 0.05 daba 68.7% / 1).
    Con todos los peaks (top_n=None): ACF 80.6%, ACF∪LS 98.5%.
    """
    per = np.asarray(per)
    power = np.asarray(power)
    fap = np.asarray(fap)

    distance = None
    if min_peak_sep_days is not None:
        if cadence is None or not np.isfinite(cadence) or cadence <= 0:
            raise ValueError("select_peaks_acf: se requiere `cadence` > 0 "
                             "para convertir min_peak_sep_days a muestras")
        distance = max(1, int(round(min_peak_sep_days / cadence)))

    prominence = None
    if prominence_frac is not None:
        prominence = float(prominence_frac * np.nanmax(power))
    elif prominence_k_fap is not None:
        # find_peaks acepta un umbral por muestra siempre que tenga el largo
        # de `power`; un array de 2 elementos lo leería como (min, max).
        prominence = prominence_k_fap * np.broadcast_to(fap, power.shape)

    if width is None and width_frac is not None:
        # La grilla del ACF es uniforme en período por construcción
        # (periodograms.acf_periodogram), así que la cadencia es el paso.
        step = cadence if cadence else float(np.median(np.diff(per)))
        width = width_frac * per / step
        if width_max_samples is not None:
            width = np.minimum(width, width_max_samples)

    idx, prom, wid = _find(power, height=fap, distance=distance,
                           prominence=prominence, width=width)

    if idx.size and distance is None and rayleigh_k is not None:
        # La serie armónica se identifica ANTES del NMS y queda exenta. El NMS
        # responde "¿son dos frecuencias distinguibles?", y para armónicos esa
        # pregunta es irrelevante: ya sabemos que son la misma señal. Aplicarlo
        # los borraba, porque la separación entre consecutivos se achica como
        # f0/(n(n+1)) y cae bajo 1/T desde n~5 (TIC 384805438 s58), dejando
        # sobrevivir uno suelto por el hueco que abre el greedy.
        raw = _build(per, power, idx, prom, wid, "prominence", None)
        # Exento TODO el peine (paso P0/2), no solo los múltiplos enteros: los
        # impares son la misma señal y sin exención el NMS los adelgaza a lag
        # largo igual que hacía con los armónicos (TIC 169640678 s41).
        in_series = label_harmonics(raw)["comb_order"].notna().to_numpy()
        order = np.argsort(np.argsort(-prom))          # posición de cada pico en `raw`
        exempt = in_series[order]

        T = 2.0 * float(np.nanmax(per))
        keep = np.ones(idx.size, dtype=bool)
        keep[~exempt] = _rayleigh_nms(per[idx][~exempt], prom[~exempt],
                                      rayleigh_k / T)
        idx, prom, wid = idx[keep], prom[keep], wid[keep]

    return _build(per, power, idx, prom, wid, sort_by, top_n)


def select_peaks_ls(per, power, fap, *, window=None, freq_step=None,
                    min_peak_sep_days=None,
                    prominence_frac=0.01, width=1,
                    sort_by="prominence", top_n=None):
    """Picos del LS sobre power crudo.

    - Significancia: si `window` (LS de flux=1) está disponible se exige
      `power - window > fap` (piso local de la ventana de observación);
      si no, `power > fap`. `fap` es el nivel escalar de Baluev.
    - `prominence = prominence_frac * max(power)`: cuánto debe destacar el
      pico de su entorno local (independiente del FAP).
    - `min_peak_sep_days` (opcional, requiere `freq_step` en ciclos/día):
      separación mínima en período convertida a muestras de frecuencia con
      |Δf| ≈ ΔP/P² evaluado en per_max = max(per). Desactivada por defecto.
    """
    per = np.asarray(per)
    power = np.asarray(power)

    distance = None
    if min_peak_sep_days is not None:
        if freq_step is None or freq_step <= 0:
            raise ValueError("select_peaks_ls: se requiere `freq_step` > 0 "
                             "para convertir min_peak_sep_days a muestras")
        per_max = float(np.nanmax(per))
        dfreq_min = min_peak_sep_days / per_max ** 2
        distance = max(1, int(round(dfreq_min / freq_step)))

    prominence = None
    if prominence_frac is not None:
        prominence = float(prominence_frac * np.nanmax(power))

    idx, prom, wid = _find(power, distance=distance,
                           prominence=prominence, width=width)
    if idx.size and fap is not None:
        fap_val = float(np.asarray(fap))
        if window is not None:
            valid = (power[idx] - np.asarray(window)[idx]) > fap_val
        else:
            valid = power[idx] > fap_val
        idx, prom, wid = idx[valid], prom[valid], wid[valid]

    return _build(per, power, idx, prom, wid, sort_by, top_n)


def select_peaks(df_periodogram, source, **kwargs):
    """Wrapper por DataFrame: despacha según `source` ('LS' o 'ACF').

    Usa las columnas `per`, `power` (+ `fap`/`window` si existen) y los attrs
    del periodograma (`cadence_days`, `step_cpd`, `fap_level`).
    """
    df = df_periodogram
    if source.upper() == "ACF":
        return select_peaks_acf(
            df["per"].to_numpy(), df["power"].to_numpy(),
            df["fap"].to_numpy(),
            cadence=df.attrs.get("cadence_days"),
            **kwargs,
        )
    if source.upper() == "LS":
        window = df["window"].to_numpy() if "window" in df.columns else None
        return select_peaks_ls(
            df["per"].to_numpy(), df["power"].to_numpy(),
            df.attrs.get("fap_level"),
            window=window,
            freq_step=df.attrs.get("step_cpd"),
            **kwargs,
        )
    raise ValueError(f"source='{source}' no soportado (usar 'LS' o 'ACF')")


def label_harmonics(peaks, tol=HARMONIC_TOL, max_order=HARMONIC_MAX_ORDER,
                    min_run=HARMONIC_MIN_RUN, fit_tol=HARMONIC_FIT_TOL):
    """Marca qué picos son armónicos de un mismo período fundamental.

    El ACF de una señal periódica repite el pico en cada múltiplo del período,
    así que una estrella con un solo período genera una serie completa. Esos
    picos están perfectamente RESUELTOS (la separación en frecuencia entre
    armónicos consecutivos es f0/(n(n+1)), no cero), de modo que ningún
    criterio de separación mínima los puede colapsar: hay que identificarlos
    por su relación aritmética.

    Como fundamental se prueba cada pico detectado. Probar solo el más
    prominente no basta: cuando el ACF decae lento el armónico n-ésimo puede
    ser el más alto (TIC 316947536 s83, donde el máximo es 5*P0).

    El candidato se puntúa por la longitud de la RACHA contigua 1, 2, 3, ...
    que sí está presente, no por el total de múltiplos que explica: contando
    el total, cualquier pico puede reclamar un múltiplo lejano y se arman
    series como {1x, 11x} sobre puro ruido (TIC 194120571 s82) o con el 2x
    faltando (TIC 313513149 s65).

    La serie se declara SOLO si es inequívoca, porque declararla es lo que
    autoriza aguas abajo a colapsarla en un único candidato de período:

      - `min_run` armónicos consecutivos desde el fundamental,
      - residuo relativo mediano <= `fit_tol` contra el período AJUSTADO,
        bastante por dentro de la tolerancia con que se emparejó (una serie
        real ajusta al ~0.5%, una coincidencia aritmética queda pegada al
        borde del 5%), y
      - el pico MÁS PROMINENTE dentro de la serie: si el más alto del ACF
        quedó afuera, lo que se identificó no es la señal dominante
        (TIC 436272344 s65, con el pico de 0.664 d fuera de la serie).

    Si alguna falla, `harmonic_order` queda todo <NA>: ningún pico se pliega
    y todos siguen vivos como candidatos independientes. Un armónico de más
    lo descarta la red al mirar el phase-folded; un fundamental borrado por
    un plegado dudoso no lo recupera nadie.

    El período de la serie NO es el del pico n=1 sino la pendiente de
    lag vs n por mínimos cuadrados por el origen, P = Σ(n·P_n)/Σ(n²): usa
    todos los armónicos y no propaga el error del primer pico multiplicado
    por n. En TIC 13785212 s41 el espaciado es 1.771/1.778/1.770 pero el
    pico n=1 cae en 1.875, así que anclar ahí daba residuos del 3.6% sobre
    una serie que ajusta al 0.7%.

    Devuelve una copia de `peaks` con tres columnas nuevas:
      `harmonic_order`     n tal que per ≈ n * P0 (1 = fundamental), <NA> si
                           el pico no pertenece a la serie. El orden máximo
                           sale del rango de lags si `max_order` es None
      `is_fundamental`     True solo en el pico de orden 1
      `period_series`      período ajustado de la serie, igual en todas sus
                           filas; <NA> fuera de la serie
      `comb_order`         m tal que per ≈ m * P0/2. El peine del ACF tiene
                           paso P0/2, no P0: una eclipsante con eclipses de
                           profundidad distinta correlaciona primario-primario
                           en P0 (m par) y primario-secundario en P0/2 (m
                           impar), más débil. Los m impares son la MISMA señal,
                           no candidatos aparte, pero hay que reconocerlos para
                           no tratarlos como picos sueltos.
    """
    out = peaks.copy()
    if out.empty:
        out["harmonic_order"] = pd.Series(dtype="Int64")
        out["is_fundamental"] = pd.Series(dtype=bool)
        out["period_series"] = pd.Series(dtype="Float64")
        out["comb_order"] = pd.Series(dtype="Int64")
        return out

    per = out["per"].to_numpy(dtype=float)
    prom = out["prominence"].to_numpy(dtype=float)

    lag_max = float(np.nanmax(per))

    def orders_for(fundamental):
        ratio = per / fundamental
        order = np.rint(ratio)
        top = max_order if max_order is not None else np.ceil(lag_max / fundamental)
        valid = (order >= 1) & (order <= top)
        valid &= np.abs(per - order * fundamental) <= tol * per
        return np.where(valid, order, np.nan)

    def run_length(orders):
        present = set(orders[np.isfinite(orders)].astype(int))
        run = 0
        while run + 1 in present:
            run += 1
        return run

    best_orders, best_key = None, (-1, -1, -np.inf)
    for candidate, candidate_prom in zip(per, prom):
        orders = orders_for(candidate)
        key = (run_length(orders), int(np.isfinite(orders).sum()), candidate_prom)
        if key > best_key:
            best_orders, best_key = orders, key

    # Un múltiplo suelto más allá del corte de la racha no es evidencia de
    # nada: 8x sin 5x, 6x ni 7x es aritmética, no una serie armónica.
    best_orders = np.where(best_orders <= best_key[0], best_orders, np.nan)
    in_series = np.isfinite(best_orders)

    orders_in = best_orders[in_series]
    if orders_in.size:
        period_series = float(np.sum(orders_in * per[in_series])
                              / np.sum(orders_in ** 2))
        residual = np.abs(per[in_series] - orders_in * period_series)
        fit = float(np.median(residual / per[in_series]))
    else:
        period_series, fit = np.nan, np.inf

    unambiguous = (best_key[0] >= min_run and fit <= fit_tol
                   and bool(in_series[np.argmax(prom)]))
    if not unambiguous:
        best_orders = np.full(per.size, np.nan)
        period_series = np.nan

    if np.isfinite(period_series):
        half = period_series / 2.0
        comb = np.rint(per / half)
        valid_comb = ((comb >= 1) & (comb <= np.ceil(lag_max / half))
                      & (np.abs(per - comb * half) <= tol * per))
        comb = np.where(valid_comb, comb, np.nan)
    else:
        comb = np.full(per.size, np.nan)

    out["harmonic_order"] = pd.array(best_orders, dtype="Float64").astype("Int64")
    out["is_fundamental"] = out["harmonic_order"].eq(1).fillna(False)
    out["period_series"] = pd.array(
        np.where(np.isfinite(best_orders), period_series, np.nan), dtype="Float64")
    out["comb_order"] = pd.array(comb, dtype="Float64").astype("Int64")
    return out


def candidate_periods(peaks):
    """Períodos a presentar a la red, un phase-fold por cada uno.

    Colapsa el peine en sus períodos SOLO si `label_harmonics` declaró la serie
    (ver ahí los criterios); todo pico fuera del peine sobrevive como candidato
    propio. Sin serie declarada devuelve los picos tal cual.

    Si el peine tiene miembros IMPARES se emiten dos candidatos, `P0` y `P0/2`,
    en vez de decidir nosotros: esa es justamente la ambigüedad de medio período
    de una eclipsante, y la resuelve el phase-fold de un vistazo. Usar la
    alternancia de amplitud para elegir apuesta a que la asimetría entre
    eclipses siempre se ve, y en profundidades casi iguales no se ve.

    Agrega la columna `period`, que es el valor con el que hay que doblar:
    `period_series` para el fundamental (ajustado sobre todos los armónicos,
    no el `per` del pico 1x), la mitad para el subarmónico, y el `per` del pico
    para los aislados. Ordenado por prominencia descendente.
    """
    if peaks.empty:
        return peaks.assign(period=pd.Series(dtype="Float64"),
                            kind=pd.Series(dtype=object))
    labelled = peaks if "comb_order" in peaks else label_harmonics(peaks)

    on_comb = labelled["comb_order"].notna()
    parts = []

    fundamental = labelled[labelled["is_fundamental"]]
    if not fundamental.empty:
        period_0 = float(fundamental["period_series"].iloc[0])
        parts.append(fundamental.assign(period=period_0, kind="fundamental"))

        odd = labelled[on_comb & labelled["comb_order"].mod(2).eq(1)
                       & ~labelled["is_fundamental"]]
        if not odd.empty:
            best = odd.nlargest(1, "prominence")
            parts.append(best.assign(period=period_0 / 2.0, kind="subarmonico"))

    isolated = labelled[~on_comb]
    if not isolated.empty:
        parts.append(isolated.assign(period=isolated["per"], kind="aislado"))

    if not parts:
        return labelled.assign(period=labelled["per"], kind="aislado")
    return (pd.concat(parts).sort_values("prominence", ascending=False)
            .reset_index(drop=True))
