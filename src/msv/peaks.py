"""Selección de picos en periodogramas LS/ACF con scipy.signal.find_peaks.

Reemplaza el suavizado gaussiano del ACF (gaussian_filter1d, sigma=50 pts)
que degradaba la resolución de la grilla de períodos: en su lugar, los picos
espurios de ruido se suprimen imponiendo una separación temporal mínima entre
picos (`min_peak_sep_days`) vía el parámetro `distance` de find_peaks,
operando SIEMPRE sobre el power crudo.

- ACF: la grilla es uniforme en período (per = lag * cadence), así que
  distance = round(min_peak_sep_days / cadence) muestras.
- LS: la grilla es uniforme en FRECUENCIA; una separación fija en período no
  es constante en frecuencia. Se convierte con |Δf| ≈ ΔP/P² evaluado en el
  período más largo buscado (conversión conservadora: garantiza al menos esa
  separación en todo el rango). Por defecto queda desactivada para LS: sus
  picos son naturalmente angostos y basta FAP + prominencia.
"""
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, peak_prominences, peak_widths

from .config import ACF_PROMINENCE_FRAC, MIN_PEAK_SEP_DAYS

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


def select_peaks_acf(per, power, fap, cadence, *,
                     min_peak_sep_days=MIN_PEAK_SEP_DAYS,
                     prominence_frac=ACF_PROMINENCE_FRAC, width=None,
                     sort_by="prominence", top_n=None):
    """Picos del ACF sobre power CRUDO con ventana temporal mínima.

    - `distance = round(min_peak_sep_days / cadence)`: dos picos no pueden
      estar a menos de esa ventana temporal (reemplaza al smoothing).
    - `height = fap`: array de Bartlett dependiente del lag — solo picos
      significativos.
    - `prominence_frac`: prominencia mínima como fracción del máximo del
      power. Suprime los wiggles de ruido sobre el decaimiento de red noise
      del ACF (lo que antes hacía el smoothing gaussiano, pero sin perder
      resolución de grilla). Default 0.05: en los 24 FP_PAIRS elimina todos
      los picos espurios y en 68 pares label=1 no pierde ningún match con
      per_ogle (el smoothing viejo recuperaba 0%).
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

    idx, prom, wid = _find(power, height=fap, distance=distance,
                           prominence=prominence, width=width)
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
